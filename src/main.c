#include <stdio.h>
#include <stdlib.h>
#include "gbdt.h"
#include "data.h"
#include "io.h"

// A simple function to evaluate and print accuracy
void evaluate(const int* predictions, const Dataset* dataset) {
    if (!predictions || !dataset) return;
    int correct = 0;
    for (int i = 0; i < dataset->num_samples; i++) {
        if (predictions[i] == dataset->labels[i]) {
            correct++;
        }
        printf("Sample %d: Predicted=%d, Actual=%d\n", i, predictions[i], dataset->labels[i]);
    }
    double accuracy = (double)correct / dataset->num_samples;
    printf("Accuracy: %.2f%% (%d/%d)\n", accuracy * 100.0, correct, dataset->num_samples);
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <train_features> <train_labels> <test_features> <test_labels>\n", argv[0]);
        return 1;
    }

    const char* train_features_file = argv[1];
    const char* train_labels_file = argv[2];
    const char* test_features_file = argv[3];
    const char* test_labels_file = argv[4];
    const char* model_file = "gbdt_model.json";

    // 1. Set GBDT parameters
    GBDTParams params;
    params.num_trees = 20;
    params.max_depth = 3;
    params.learning_rate = 0.1;
    params.min_samples_split = 2;
    params.subsample = 1.0; // Using all samples for now
    // num_classes will be determined from the data

    // 2. Load training data
    printf("Loading training data...\n");
    int num_classes;
    Dataset* train_data = load_dataset(train_features_file, train_labels_file, &num_classes);
    if (!train_data) {
        fprintf(stderr, "Failed to load training data.\n");
        return 1;
    }
    params.num_classes = num_classes;
    printf("Training data loaded: %d samples, %d features, %d classes.\n", train_data->num_samples, train_data->num_features, num_classes);

    // 3. Train the model
    printf("\nTraining GBDT model...\n");
    GBDTModel* model = train_gbdt(train_data, &params);
    if (!model) {
        fprintf(stderr, "Failed to train GBDT model.\n");
        free_dataset(train_data);
        return 1;
    }
    printf("Model training complete.\n");

    // 4. Save the model
    printf("\nSaving model to %s...\n", model_file);
    save_gbdt_model(model, model_file);
    printf("Model saved.\n");

    // Free the original model to ensure we test loading
    free_gbdt_model(model);
    model = NULL;

    // 5. Load the model from file
    printf("\nLoading model from %s...\n", model_file);
    GBDTModel* loaded_model = load_gbdt_model(model_file);
    if (!loaded_model) {
        fprintf(stderr, "Failed to load GBDT model.\n");
        free_dataset(train_data);
        return 1;
    }
    printf("Model loaded.\n");

    // 6. Load test data
    printf("\nLoading test data...\n");
    Dataset* test_data = load_dataset(test_features_file, test_labels_file, &num_classes);
    if (!test_data) {
        fprintf(stderr, "Failed to load test data.\n");
        free_dataset(train_data);
        free_gbdt_model(loaded_model);
        return 1;
    }
    printf("Test data loaded: %d samples, %d features.\n", test_data->num_samples, test_data->num_features);

    // 7. Make predictions
    printf("\nMaking predictions on test data...\n");
    int* predictions = predict_gbdt(loaded_model, test_data);

    // 8. Evaluate predictions
    if (predictions) {
        evaluate(predictions, test_data);
    }

    // 9. Clean up
    printf("\nCleaning up...\n");
    free(predictions);
    free_dataset(train_data);
    free_dataset(test_data);
    free_gbdt_model(loaded_model);
    printf("Done.\n");

    return 0;
}
