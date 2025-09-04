#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include "gbdt.h"
#include "data.h"
#include "io.h"

// A simple function to evaluate and return accuracy
double get_accuracy(const PredictionResult* result, const Dataset* dataset) {
    if (!result || !dataset) return 0.0;
    int correct = 0;
    for (int i = 0; i < result->num_samples; i++) {
        if (result->labels[i] == dataset->labels[i]) {
            correct++;
        }
    }
    return (double)correct / result->num_samples;
}


int main(int argc, char* argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <train_features> <train_labels> <test_features> <test_labels>\n", argv[0]);
        fprintf(stderr, "Note: Hyperparameters for grid search are hardcoded in main.c\n");
        return 1;
    }

    srand(time(NULL));

    const char* train_features_file = argv[1];
    const char* train_labels_file = argv[2];
    const char* test_features_file = argv[3];
    const char* test_labels_file = argv[4];
    const char* model_file = "best_gbdt_model.json";

    // --- Grid Search Hyperparameters ---
    double learning_rates[] = {0.1, 0.2};
    int max_depths[] = {3, 4};
    int n_estimators = 50; // Max estimators for each run
    double early_stopping_loss = 0.4; // Stop if loss on test set is < this. Set to 0 to disable.

    int num_lr = sizeof(learning_rates) / sizeof(double);
    int num_md = sizeof(max_depths) / sizeof(int);

    printf("--- Starting Grid Search ---\n");
    printf("Learning rates to test: %d\n", num_lr);
    printf("Max depths to test: %d\n", num_md);
    printf("Max estimators per run: %d\n", n_estimators);
    printf("Early stopping loss threshold: %f\n", early_stopping_loss);
    printf("Total models to train: %d\n", num_lr * num_md);

    // --- Load Data Once ---
    printf("\nLoading data...\n");
    int num_classes;
    Dataset* train_data = load_dataset(train_features_file, train_labels_file, &num_classes);
    Dataset* test_data = load_dataset(test_features_file, test_labels_file, &num_classes);
    if (!train_data || !test_data) {
        fprintf(stderr, "Failed to load data.\n");
        return 1;
    }
    printf("Training data: %d samples. Test data: %d samples.\n", train_data->num_samples, test_data->num_samples);

    // --- Grid Search Loop ---
    GBDTModel* best_model = NULL;
    GBDTParams best_params;
    double best_accuracy = -1.0;

    for (int i = 0; i < num_lr; i++) {
        for (int j = 0; j < num_md; j++) {
            GBDTParams current_params;
            current_params.learning_rate = learning_rates[i];
            current_params.max_depth = max_depths[j];
            current_params.num_trees = n_estimators;
            current_params.min_samples_split = 2;
            current_params.subsample = 0.8;
            current_params.num_classes = num_classes;
            current_params.early_stopping_loss = early_stopping_loss;

            printf("\n--- Training with lr=%.2f, max_depth=%d ---\n", current_params.learning_rate, current_params.max_depth);

            GBDTModel* current_model = train_gbdt(train_data, &current_params, test_data);

            printf("Training finished. Model has %d trees.\n", current_model->params.num_trees);

            PredictionResult* predictions = predict_gbdt(current_model, test_data);
            double current_accuracy = get_accuracy(predictions, test_data);
            printf("Accuracy for this model: %.2f%%\n", current_accuracy * 100.0);

            if (current_accuracy > best_accuracy) {
                printf("!!! New best model found.\n");
                best_accuracy = current_accuracy;
                best_params = current_model->params;

                // Free the old best model if it exists
                if (best_model) {
                    free_gbdt_model(best_model);
                }
                // This becomes the new best model
                best_model = current_model;

            } else {
                // This model is not the best, so we can free it immediately
                free_gbdt_model(current_model);
            }
            free_prediction_result(predictions);
        }
    }

    // --- Results ---
    printf("\n--- Grid Search Complete ---\n");
    if (best_model) {
        printf("Best Accuracy: %.2f%%\n", best_accuracy * 100.0);
        printf("Best Parameters:\n");
        printf("  - learning_rate: %f\n", best_params.learning_rate);
        printf("  - max_depth: %d\n", best_params.max_depth);
        printf("  - trees_built: %d (stopped early? %s)\n", best_params.num_trees, best_params.num_trees < n_estimators ? "yes" : "no");

        printf("\nSaving best model to %s...\n", model_file);
        save_gbdt_model(best_model, model_file);
        printf("Model saved.\n");
    } else {
        printf("Grid search did not produce a valid model.\n");
    }

    // --- Clean up ---
    printf("\nCleaning up...\n");
    free_dataset(train_data);
    free_dataset(test_data);
    free_gbdt_model(best_model);
    printf("Done.\n");

    return 0;
}
