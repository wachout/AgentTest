#include "gbdt.h"
#include "tree.h"
#include "io.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#define NUM_PREDICT_THREADS 4

// Structs for Parallel Prediction
typedef struct {
    const GBDTModel* model;
    const Dataset* test_data;
    PredictionResult* result;
    int start_sample;
    int end_sample;
} PredictThreadArgs;


// --- Helper Functions ---

static void softmax(double* scores, int num_classes) {
    double max_score = scores[0];
    for (int i = 1; i < num_classes; i++) {
        if (scores[i] > max_score) max_score = scores[i];
    }
    double sum_exp = 0.0;
    for (int i = 0; i < num_classes; i++) {
        scores[i] = exp(scores[i] - max_score);
        sum_exp += scores[i];
    }
    for (int i = 0; i < num_classes; i++) {
        scores[i] /= sum_exp;
    }
}

// Calculates multiclass log loss (cross-entropy)
static double calculate_log_loss(const GBDTModel* model, const Dataset* dataset) {
    if (!dataset || dataset->num_samples == 0) return 0.0;

    double total_loss = 0.0;
    PredictionResult* result = predict_gbdt(model, dataset);

    for (int i = 0; i < dataset->num_samples; i++) {
        int true_label = dataset->labels[i];
        // Add a small epsilon to prevent log(0)
        double prob = result->probabilities[i][true_label] + 1e-15;
        total_loss += -log(prob);
    }

    free_prediction_result(result);
    return total_loss / dataset->num_samples;
}

// --- Training Function (with Early Stopping) ---

GBDTModel* train_gbdt(const Dataset* train_data, const GBDTParams* params, const Dataset* valid_data) {
    GBDTModel* model = (GBDTModel*)malloc(sizeof(GBDTModel));
    if (!model) { perror("Failed to allocate GBDT model"); return NULL; }

    model->params = *params;
    model->trees = (DecisionTree***)malloc(params->num_classes * sizeof(DecisionTree**));
    model->initial_prediction = (double*)calloc(params->num_classes, sizeof(double));
    if (!model->trees || !model->initial_prediction) {
        perror("Failed to allocate model components");
        free_gbdt_model(model);
        return NULL;
    }
    for (int k = 0; k < params->num_classes; k++) {
        // Allocate for the max number of trees, we will track the actual number
        model->trees[k] = (DecisionTree**)calloc(params->num_trees, sizeof(DecisionTree*));
        if (!model->trees[k]) {
            perror("Failed to allocate trees for a class");
            free_gbdt_model(model);
            return NULL;
        }
    }

    double** F = (double**)malloc(train_data->num_samples * sizeof(double*));
    for (int i = 0; i < train_data->num_samples; i++) {
        F[i] = (double*)calloc(params->num_classes, sizeof(double));
    }

    double* gradients = (double*)malloc(train_data->num_samples * sizeof(double));
    int* all_indices = (int*)malloc(train_data->num_samples * sizeof(int));
    for(int i=0; i<train_data->num_samples; ++i) all_indices[i] = i;

    int actual_trees = 0;
    for (int m = 0; m < params->num_trees; m++) {
        printf("Building tree %d/%d...\n", m + 1, params->num_trees);
        for (int k = 0; k < params->num_classes; k++) {
            for (int i = 0; i < train_data->num_samples; i++) {
                double probs[params->num_classes];
                memcpy(probs, F[i], params->num_classes * sizeof(double));
                softmax(probs, params->num_classes);

                int target = (train_data->labels[i] == k);
                gradients[i] = target - probs[k];
            }

            model->trees[k][m] = build_decision_tree(train_data, gradients, all_indices, train_data->num_samples, params);

            for (int i = 0; i < train_data->num_samples; i++) {
                double prediction = predict_tree(model->trees[k][m], train_data->features[i]);
                F[i][k] += params->learning_rate * prediction;
            }
        }
        actual_trees++;

        // Early stopping check
        if (valid_data && params->early_stopping_loss > 0) {
            // Temporarily update model's tree count for accurate loss calculation
            model->params.num_trees = actual_trees;
            double current_loss = calculate_log_loss(model, valid_data);
            printf("  Validation loss after tree %d: %f\n", actual_trees, current_loss);
            if (current_loss < params->early_stopping_loss) {
                printf("  Early stopping triggered at tree %d. Loss %f < %f\n", actual_trees, current_loss, params->early_stopping_loss);
                break;
            }
        }
    }

    // Set the final number of trees in the model
    model->params.num_trees = actual_trees;

    free(gradients);
    free(all_indices);
    for (int i = 0; i < train_data->num_samples; i++) free(F[i]);
    free(F);

    return model;
}


// --- Prediction Functions (Parallel) ---

void* predict_worker(void* args) {
    PredictThreadArgs* thread_args = (PredictThreadArgs*)args;
    const GBDTModel* model = thread_args->model;
    const Dataset* test_data = thread_args->test_data;
    PredictionResult* result = thread_args->result;

    for (int i = thread_args->start_sample; i < thread_args->end_sample; i++) {
        double scores[model->params.num_classes];
        memcpy(scores, model->initial_prediction, model->params.num_classes * sizeof(double));

        for (int k = 0; k < model->params.num_classes; k++) {
            for (int m = 0; m < model->params.num_trees; m++) {
                // Important: Check if tree exists, for early-stopped models
                if (model->trees[k][m]) {
                    scores[k] += model->params.learning_rate * predict_tree(model->trees[k][m], test_data->features[i]);
                }
            }
        }

        softmax(scores, model->params.num_classes);
        memcpy(result->probabilities[i], scores, model->params.num_classes * sizeof(double));

        int max_class = 0;
        double max_prob = scores[0];
        for (int k = 1; k < model->params.num_classes; k++) {
            if (scores[k] > max_prob) {
                max_prob = scores[k];
                max_class = k;
            }
        }
        result->labels[i] = max_class;
    }
    pthread_exit(NULL);
}

PredictionResult* predict_gbdt(const GBDTModel* model, const Dataset* test_data) {
    PredictionResult* result = (PredictionResult*)malloc(sizeof(PredictionResult));
    if (!result) {
        perror("Failed to allocate memory for PredictionResult");
        return NULL;
    }
    result->num_samples = test_data->num_samples;
    result->labels = (int*)malloc(test_data->num_samples * sizeof(int));
    result->probabilities = (double**)malloc(test_data->num_samples * sizeof(double*));
    if (!result->labels || !result->probabilities) {
        free(result->labels); free(result->probabilities); free(result);
        return NULL;
    }
    for(int i=0; i<test_data->num_samples; ++i) {
        result->probabilities[i] = (double*)malloc(model->params.num_classes * sizeof(double));
    }

    pthread_t threads[NUM_PREDICT_THREADS];
    PredictThreadArgs thread_args[NUM_PREDICT_THREADS];
    int samples_per_thread = test_data->num_samples / NUM_PREDICT_THREADS;
    if (test_data->num_samples < NUM_PREDICT_THREADS) samples_per_thread = test_data->num_samples;


    for (int i = 0; i < NUM_PREDICT_THREADS; i++) {
        thread_args[i].model = model;
        thread_args[i].test_data = test_data;
        thread_args[i].result = result;
        thread_args[i].start_sample = i * samples_per_thread;
        if (i == NUM_PREDICT_THREADS - 1) {
            thread_args[i].end_sample = test_data->num_samples;
        } else {
            thread_args[i].end_sample = (i + 1) * samples_per_thread;
        }
        if (thread_args[i].start_sample >= test_data->num_samples) {
            thread_args[i].start_sample = thread_args[i].end_sample;
        }

        pthread_create(&threads[i], NULL, predict_worker, &thread_args[i]);
    }

    for (int i = 0; i < NUM_PREDICT_THREADS; i++) {
        if (thread_args[i].start_sample < thread_args[i].end_sample) {
            pthread_join(threads[i], NULL);
        }
    }

    return result;
}

PredictionResult* predict_gbdt_single(const GBDTModel* model, const double* features) {
    PredictionResult* result = (PredictionResult*)malloc(sizeof(PredictionResult));
    if (!result) {
        perror("Failed to allocate memory for PredictionResult");
        return NULL;
    }
    result->num_samples = 1;
    result->labels = (int*)malloc(sizeof(int));
    result->probabilities = (double**)malloc(sizeof(double*));
    if (!result->labels || !result->probabilities) {
        free(result->labels); free(result->probabilities); free(result);
        return NULL;
    }
    result->probabilities[0] = (double*)malloc(model->params.num_classes * sizeof(double));
    if (!result->probabilities[0]) {
        free(result->labels); free(result->probabilities); free(result);
        return NULL;
    }

    double scores[model->params.num_classes];
    memcpy(scores, model->initial_prediction, model->params.num_classes * sizeof(double));

    for (int k = 0; k < model->params.num_classes; k++) {
        for (int m = 0; m < model->params.num_trees; m++) {
            if (model->trees[k][m]) {
                scores[k] += model->params.learning_rate * predict_tree(model->trees[k][m], features);
            }
        }
    }

    softmax(scores, model->params.num_classes);
    memcpy(result->probabilities[0], scores, model->params.num_classes * sizeof(double));

    int max_class = 0;
    double max_prob = scores[0];
    for (int k = 1; k < model->params.num_classes; k++) {
        if (scores[k] > max_prob) {
            max_prob = scores[k];
            max_class = k;
        }
    }
    result->labels[0] = max_class;

    return result;
}

void free_prediction_result(PredictionResult* result) {
    if (result) {
        free(result->labels);
        if (result->probabilities) {
            for (int i = 0; i < result->num_samples; i++) {
                free(result->probabilities[i]);
            }
            free(result->probabilities);
        }
        free(result);
    }
}


// --- Model Freeing Function ---

void free_gbdt_model(GBDTModel* model) {
    if (model) {
        if (model->trees) {
            for (int k = 0; k < model->params.num_classes; k++) {
                if (model->trees[k]) {
                    for (int m = 0; m < model->params.num_trees; m++) {
                        // Check if tree exists before freeing
                        if (model->trees[k][m]) {
                            free_decision_tree(model->trees[k][m]);
                        }
                    }
                    free(model->trees[k]);
                }
            }
            free(model->trees);
        }
        free(model->initial_prediction);
        free(model);
    }
}
