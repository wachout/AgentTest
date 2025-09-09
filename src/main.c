#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data.h"
#include "xgboost.h"

void print_accuracy(XGBoostModel *model, Dataset *dataset) {
    int *predictions = (int*)malloc(dataset->num_rows * sizeof(int));
    if (!predictions) {
        perror("Failed to allocate memory for predictions");
        return;
    }
    xgboost_predict(model, dataset, predictions);

    int correct_predictions = 0;
    for (int i = 0; i < dataset->num_rows; i++) {
        if (predictions[i] == dataset->labels[i]) {
            correct_predictions++;
        }
    }

    float accuracy = (float)correct_predictions / dataset->num_rows;
    printf("Model Accuracy: %.2f%% (Total trees: %d)\n", accuracy * 100, model->params.num_trees);

    free(predictions);
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    const char* train_file = "data/train_mc.txt";
    const char* label_file = "data/label_mc.txt";
    const char* model_file = "model.json";

    printf("Loading data from %s and %s...\n", train_file, label_file);
    Dataset *dataset = load_data(train_file, label_file);
    if (!dataset) {
        fprintf(stderr, "Failed to load data.\n");
        return 1;
    }
    printf("Dataset loaded: %d samples, %d features\n\n", dataset->num_rows, dataset->num_cols);

    // --- Initial Training ---
    XGBoostParameter params;
    params.max_depth = 4;
    params.learning_rate = 0.1f;
    params.min_split_gain = 0.0f;
    params.lambda = 1.0f;
    params.gamma = 0.0f;
    params.num_classes = 3;
    params.subsample = 0.8f;
    params.colsample_bytree = 0.8f;
    params.num_trees = 10; // Train for 10 rounds initially

    printf("--- 1. Initial Training (10 rounds) ---\n");
    XGBoostModel *model = xgboost_train(dataset, params, NULL);
    if (!model) {
        fprintf(stderr, "Initial training failed.\n");
        free_dataset(dataset);
        return 1;
    }
    print_accuracy(model, dataset);

    // --- Iterative Training ---
    printf("\n--- 2. Iterative Training (10 more rounds) ---\n");
    params.num_trees = 10; // Train for 10 *additional* rounds
    model = xgboost_train(dataset, params, model);
    if (!model) {
        fprintf(stderr, "Iterative training failed.\n");
        free_dataset(dataset);
        return 1;
    }
    print_accuracy(model, dataset);

    // --- Save/Load Demonstration ---
    printf("\n--- 3. Testing Save/Load ---\n");
    printf("Saving model to %s...\n", model_file);
    save_model_json(model, model_file);
    printf("Model saved.\n");

    // Free the original model before loading it back
    free_xgboost_model(model);

    printf("Loading model from %s...\n", model_file);
    XGBoostModel *loaded_model = load_model_json(model_file);
    if (!loaded_model) {
        fprintf(stderr, "Failed to load model.\n");
        free_dataset(dataset);
        return 1;
    }
    printf("Model loaded.\n");

    printf("Verifying loaded model accuracy...\n");
    print_accuracy(loaded_model, dataset);

    // --- Cleanup ---
    free_xgboost_model(loaded_model);
    free_dataset(dataset);
    printf("\nCleanup complete. Exiting.\n");

    return 0;
}
