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
    printf("\nPredictions vs Actual Labels:\n");
    for (int i = 0; i < dataset->num_rows; i++) {
        printf("Sample %d: Predicted: %d, Actual: %d\n", i, predictions[i], dataset->labels[i]);
        if (predictions[i] == dataset->labels[i]) {
            correct_predictions++;
        }
    }

    float accuracy = (float)correct_predictions / dataset->num_rows;
    printf("Accuracy: %.2f%%\n", accuracy * 100);

    free(predictions);
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    // Using default file paths, but can be overridden by command line arguments
    const char* train_file = "data/train_mc.txt";
    const char* label_file = "data/label_mc.txt";
    const char* model_file = "model.json";

    printf("Loading data from %s and %s...\n", train_file, label_file);
    Dataset *dataset = load_data(train_file, label_file);
    if (!dataset) {
        fprintf(stderr, "Failed to load data.\n");
        return 1;
    }
    printf("Dataset loaded: %d samples, %d features\n", dataset->num_rows, dataset->num_cols);

    XGBoostParameter params;
    params.num_trees = 20;
    params.max_depth = 4;
    params.learning_rate = 0.1f;
    params.min_split_gain = 0.0f;
    params.lambda = 1.0f;
    params.gamma = 0.0f;
    params.num_classes = 3; // Should be determined from data, but hardcoded for now
    params.subsample = 0.8f;
    params.colsample_bytree = 0.8f;

    printf("\nStarting XGBoost training...\n");
    XGBoostModel *model = xgboost_train(dataset, params, NULL);
    if (!model) {
        fprintf(stderr, "Training failed.\n");
        free_dataset(dataset);
        return 1;
    }
    printf("Training complete.\n");

    printf("\n--- Original Model Accuracy ---\n");
    print_accuracy(model, dataset);

    printf("\nSaving model to %s...\n", model_file);
    save_model_json(model, model_file);
    printf("Model saved.\n");
    free_xgboost_model(model);

    printf("\nLoading model from %s...\n", model_file);
    XGBoostModel *loaded_model = load_model_json(model_file);
    if (!loaded_model) {
        fprintf(stderr, "Failed to load model.\n");
        free_dataset(dataset);
        return 1;
    }
    printf("Model loaded.\n");

    printf("\n--- Loaded Model Accuracy ---\n");
    print_accuracy(loaded_model, dataset);

    free_xgboost_model(loaded_model);
    free_dataset(dataset);

    return 0;
}
