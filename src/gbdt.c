#include "gbdt.h"
#include "tree.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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
            // Complex cleanup needed here
            perror("Failed to allocate trees for a class");
            return NULL;
        }
    }

    // Initialize predictions
    double** F = (double**)malloc(train_data->num_samples * sizeof(double*));
    for (int i = 0; i < train_data->num_samples; i++) {
        F[i] = (double*)calloc(params->num_classes, sizeof(double));
    }

    // Gradients
    double* gradients = (double*)malloc(train_data->num_samples * sizeof(double));

    // All sample indices
    int* all_indices = (int*)malloc(train_data->num_samples * sizeof(int));
    for(int i=0; i<train_data->num_samples; ++i) all_indices[i] = i;

    // Main training loop
    for (int m = 0; m < params->num_trees; m++) {
        printf("Building tree %d...\n", m + 1);
        for (int k = 0; k < params->num_classes; k++) {
            // Calculate current probabilities and gradients
            for (int i = 0; i < train_data->num_samples; i++) {
                double probs[params->num_classes];
                memcpy(probs, F[i], params->num_classes * sizeof(double));
                softmax(probs, params->num_classes);

                int target = (train_data->labels[i] == k);
                gradients[i] = target - probs[k];
            }

            // Fit a tree
            model->trees[k][m] = build_decision_tree(train_data, gradients, all_indices, train_data->num_samples, params);

            // Update predictions
            for (int i = 0; i < train_data->num_samples; i++) {
                double prediction = predict_tree(model->trees[k][m], train_data->features[i]);
                F[i][k] += params->learning_rate * prediction;
            }
        }
    }

    // Cleanup
    free(gradients);
    free(all_indices);
    for (int i = 0; i < train_data->num_samples; i++) {
        free(F[i]);
    }
    free(F);

    return model;
}

int* predict_gbdt(const GBDTModel* model, const Dataset* test_data) {
    int* predictions = (int*)malloc(test_data->num_samples * sizeof(int));
    if (!predictions) {
        perror("Failed to allocate memory for predictions");
        return NULL;
    }

    for (int i = 0; i < test_data->num_samples; i++) {
        double scores[model->params.num_classes];
        memcpy(scores, model->initial_prediction, model->params.num_classes * sizeof(double));

        for (int k = 0; k < model->params.num_classes; k++) {
            for (int m = 0; m < model->params.num_trees; m++) {
                scores[k] += model->params.learning_rate * predict_tree(model->trees[k][m], test_data->features[i]);
            }
        }

        // Find class with max score
        int max_class = 0;
        double max_score = scores[0];
        for (int k = 1; k < model->params.num_classes; k++) {
            if (scores[k] > max_score) {
                max_score = scores[k];
                max_class = k;
            }
        }
        predictions[i] = max_class;
    }

    return predictions;
}

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
