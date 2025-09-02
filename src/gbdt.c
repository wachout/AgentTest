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
    int* predictions;
    int start_sample;
    int end_sample;
} PredictThreadArgs;


// --- Helper Functions ---

// Helper to calculate softmax probabilities
static void softmax(double* scores, int num_classes) {
    double max_score = scores[0];
    for (int i = 1; i < num_classes; i++) {
        if (scores[i] > max_score) {
            max_score = scores[i];
        }
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

// --- Training Function ---

GBDTModel* train_gbdt(const Dataset* train_data, const GBDTParams* params) {
    // Allocate model
    GBDTModel* model = (GBDTModel*)malloc(sizeof(GBDTModel));
    if (!model) {
        perror("Failed to allocate GBDT model");
        return NULL;
    }
    model->params = *params;
    model->trees = (DecisionTree***)malloc(params->num_classes * sizeof(DecisionTree**));
    model->initial_prediction = (double*)calloc(params->num_classes, sizeof(double)); // Init to 0
    if (!model->trees || !model->initial_prediction) {
        perror("Failed to allocate model components");
        free(model->trees); free(model);
        return NULL;
    }
    for (int k = 0; k < params->num_classes; k++) {
        model->trees[k] = (DecisionTree**)malloc(params->num_trees * sizeof(DecisionTree*));
        if (!model->trees[k]) {
            perror("Failed to allocate trees for a class");
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

    // Main training loop
    for (int m = 0; m < params->num_trees; m++) {
        printf("Building tree %d...\n", m + 1);
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
    }

    free(gradients);
    free(all_indices);
    for (int i = 0; i < train_data->num_samples; i++) {
        free(F[i]);
    }
    free(F);

    return model;
}


// --- Prediction Functions ---

void* predict_worker(void* args) {
    PredictThreadArgs* thread_args = (PredictThreadArgs*)args;
    const GBDTModel* model = thread_args->model;
    const Dataset* test_data = thread_args->test_data;

    for (int i = thread_args->start_sample; i < thread_args->end_sample; i++) {
        double scores[model->params.num_classes];
        memcpy(scores, model->initial_prediction, model->params.num_classes * sizeof(double));

        for (int k = 0; k < model->params.num_classes; k++) {
            for (int m = 0; m < model->params.num_trees; m++) {
                scores[k] += model->params.learning_rate * predict_tree(model->trees[k][m], test_data->features[i]);
            }
        }

        int max_class = 0;
        double max_score = scores[0];
        for (int k = 1; k < model->params.num_classes; k++) {
            if (scores[k] > max_score) {
                max_score = scores[k];
                max_class = k;
            }
        }
        thread_args->predictions[i] = max_class;
    }
    pthread_exit(NULL);
}

int* predict_gbdt(const GBDTModel* model, const Dataset* test_data) {
    int* predictions = (int*)malloc(test_data->num_samples * sizeof(int));
    if (!predictions) {
        perror("Failed to allocate memory for predictions");
        return NULL;
    }

    pthread_t threads[NUM_PREDICT_THREADS];
    PredictThreadArgs thread_args[NUM_PREDICT_THREADS];
    int samples_per_thread = test_data->num_samples / NUM_PREDICT_THREADS;

    for (int i = 0; i < NUM_PREDICT_THREADS; i++) {
        thread_args[i].model = model;
        thread_args[i].test_data = test_data;
        thread_args[i].predictions = predictions;
        thread_args[i].start_sample = i * samples_per_thread;
        thread_args[i].end_sample = (i == NUM_PREDICT_THREADS - 1) ? test_data->num_samples : (i + 1) * samples_per_thread;

        pthread_create(&threads[i], NULL, predict_worker, &thread_args[i]);
    }

    for (int i = 0; i < NUM_PREDICT_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    return predictions;
}


// --- Model Freeing Function ---

void free_gbdt_model(GBDTModel* model) {
    if (model) {
        if (model->trees) {
            for (int k = 0; k < model->params.num_classes; k++) {
                if (model->trees[k]) {
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
