#include "gbdt.h"
#include "tree.h"
#include "io.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#define NUM_PREDICT_THREADS 4

// --- Structs for Parallel Prediction ---
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

// --- Model Creation and Training ---

GBDTModel* create_gbdt_model(const GBDTParams* params) {
    GBDTModel* model = (GBDTModel*)calloc(1, sizeof(GBDTModel));
    if (!model) {
        perror("Failed to allocate GBDT model");
        return NULL;
    }
    model->params = *params;
    model->params.num_trees = 0; // Start with 0 trees
    model->initial_prediction = (double*)calloc(params->num_classes, sizeof(double));

    // The 'trees' array will be allocated by train_gbdt as needed
    model->trees = (DecisionTree***)calloc(params->num_classes, sizeof(DecisionTree**));

    if (!model->initial_prediction || !model->trees) {
        perror("Failed to allocate model components");
        free(model->initial_prediction);
        free(model->trees);
        free(model);
        return NULL;
    }
    return model;
}

void train_gbdt(GBDTModel* model, const Dataset* train_data, int additional_trees, int save_interval, const char* save_path) {
    if (!model || !train_data || additional_trees <= 0) {
        return;
    }

    // --- State Initialization ---
    double** F = (double**)malloc(train_data->num_samples * sizeof(double*));
    for (int i = 0; i < train_data->num_samples; i++) {
        F[i] = (double*)malloc(model->params.num_classes * sizeof(double));
        // Initialize F with scores from existing trees
        memcpy(F[i], model->initial_prediction, model->params.num_classes * sizeof(double));
        for (int k = 0; k < model->params.num_classes; k++) {
            for (int m = 0; m < model->params.num_trees; m++) {
                F[i][k] += model->params.learning_rate * predict_tree(model->trees[k][m], train_data->features[i]);
            }
        }
    }

    double* gradients = (double*)malloc(train_data->num_samples * sizeof(double));
    int* all_indices = (int*)malloc(train_data->num_samples * sizeof(int));
    for(int i = 0; i < train_data->num_samples; ++i) all_indices[i] = i;

    // --- Iterative Training Loop ---
    int starting_trees = model->params.num_trees;
    int new_total_trees = starting_trees + additional_trees;

    // Reallocate tree arrays to hold new trees
    for (int k = 0; k < model->params.num_classes; k++) {
        model->trees[k] = (DecisionTree**)realloc(model->trees[k], new_total_trees * sizeof(DecisionTree*));
    }

    for (int m = starting_trees; m < new_total_trees; m++) {
        printf("Building tree %d...\n", m + 1);
        for (int k = 0; k < model->params.num_classes; k++) {
            // Calculate gradients based on current F scores
            for (int i = 0; i < train_data->num_samples; i++) {
                double probs[model->params.num_classes];
                memcpy(probs, F[i], model->params.num_classes * sizeof(double));
                softmax(probs, model->params.num_classes);
                int target = (train_data->labels[i] == k);
                gradients[i] = target - probs[k];
            }

            // Fit a new tree
            DecisionTree* new_tree = build_decision_tree(train_data, gradients, all_indices, train_data->num_samples, &model->params);
            model->trees[k][m] = new_tree;

            // Update F scores with the new tree's predictions
            for (int i = 0; i < train_data->num_samples; i++) {
                F[i][k] += model->params.learning_rate * predict_tree(new_tree, train_data->features[i]);
            }
        }

        // Update the model's tree count after each full iteration
        model->params.num_trees++;

        // Periodically save the model
        if (save_interval > 0 && (model->params.num_trees % save_interval == 0)) {
            printf("Saving model at iteration %d...\n", model->params.num_trees);
            save_gbdt_model(model, save_path);
        }
    }

    // Final save
    printf("Finalizing training and saving model...\n");
    save_gbdt_model(model, save_path);

    // --- Cleanup ---
    free(gradients);
    free(all_indices);
    for (int i = 0; i < train_data->num_samples; i++) free(F[i]);
    free(F);
}


// --- Prediction Functions ---

void* predict_worker(void* args) {
    // ... (This part remains the same)
    PredictThreadArgs* thread_args = (PredictThreadArgs*)args;
    const GBDTModel* model = thread_args->model;
    const Dataset* test_data = thread_args->test_data;
    PredictionResult* result = thread_args->result;

    for (int i = thread_args->start_sample; i < thread_args->end_sample; i++) {
        double scores[model->params.num_classes];
        memcpy(scores, model->initial_prediction, model->params.num_classes * sizeof(double));

        for (int k = 0; k < model->params.num_classes; k++) {
            for (int m = 0; m < model->params.num_trees; m++) {
                scores[k] += model->params.learning_rate * predict_tree(model->trees[k][m], test_data->features[i]);
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
    // ... (This part remains the same)
    PredictionResult* result = (PredictionResult*)malloc(sizeof(PredictionResult));
    if (!result) {
        perror("Failed to allocate memory for PredictionResult");
        return NULL;
    }
    result->num_samples = test_data->num_samples;
    result->labels = (int*)malloc(test_data->num_samples * sizeof(int));
    result->probabilities = (double**)malloc(test_data->num_samples * sizeof(double*));
    if (!result->labels || !result->probabilities) {
        free(result->labels);
        free(result->probabilities);
        free(result);
        return NULL;
    }
    for(int i=0; i<test_data->num_samples; ++i) {
        result->probabilities[i] = (double*)malloc(model->params.num_classes * sizeof(double));
    }

    pthread_t threads[NUM_PREDICT_THREADS];
    PredictThreadArgs thread_args[NUM_PREDICT_THREADS];
    int samples_per_thread = test_data->num_samples / NUM_PREDICT_THREADS;

    for (int i = 0; i < NUM_PREDICT_THREADS; i++) {
        thread_args[i].model = model;
        thread_args[i].test_data = test_data;
        thread_args[i].result = result;
        thread_args[i].start_sample = i * samples_per_thread;
        thread_args[i].end_sample = (i == NUM_PREDICT_THREADS - 1) ? test_data->num_samples : (i + 1) * samples_per_thread;

        pthread_create(&threads[i], NULL, predict_worker, &thread_args[i]);
    }

    for (int i = 0; i < NUM_PREDICT_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

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
                    // Use the actual number of trees stored in the model
                    for (int m = 0; m < model->params.num_trees; m++) {
                        free_decision_tree(model->trees[k][m]);
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
