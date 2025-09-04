#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gbdt.h"
#include "data.h"
#include "io.h"

void evaluate(const PredictionResult* result, const Dataset* dataset, const GBDTModel* model) {
    if (!result || !dataset || !model) return;
    int correct = 0;
    int num_classes = model->params.num_classes;

    printf("\n--- Predictions & Probabilities ---\n");
    for (int i = 0; i < result->num_samples; i++) {
        if (result->labels[i] == dataset->labels[i]) {
            correct++;
        }
        printf("Sample %d: Predicted=%d, Actual=%d, Probs=[", i, result->labels[i], dataset->labels[i]);
        for (int k = 0; k < num_classes; k++) {
            printf("%.4f%s", result->probabilities[i][k], k == num_classes - 1 ? "" : ", ");
        }
        printf("]\n");
    }
    double accuracy = (double)correct / result->num_samples;
    printf("\nAccuracy: %.2f%% (%d/%d)\n", accuracy * 100.0, correct, result->num_samples);
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <train_features> <train_labels> <test_features> <test_labels> <iterations>\n", argv[0]);
        return 1;
    }

    srand(time(NULL));

    const char* train_features_file = argv[1];
    const char* train_labels_file = argv[2];
    const char* test_features_file = argv[3];
    const char* test_labels_file = argv[4];
    int iterations_to_add = atoi(argv[5]);
    const char* model_file = "gbdt_model.json";
    const int save_interval = 10;

    // --- Load or Create Model ---
    GBDTModel* model = load_gbdt_model(model_file);
    if (model) {
        printf("Loaded existing model with %d trees. Preparing to add %d more.\n", model->params.num_trees, iterations_to_add);
    } else {
        printf("No existing model found. Creating a new one.\n");
        GBDTParams params;
        params.max_depth = 3;
        params.learning_rate = 0.1;
        params.min_samples_split = 2;
        params.subsample = 0.8;
        // num_classes will be set after loading data
        model = create_gbdt_model(&params);
        if (!model) {
            fprintf(stderr, "Failed to create new GBDT model.\n");
            return 1;
        }
    }

    // --- Load Data ---
    printf("Loading training data...\n");
    int num_classes;
    Dataset* train_data = load_dataset(train_features_file, train_labels_file, &num_classes);
    if (!train_data) {
        fprintf(stderr, "Failed to load training data.\n");
        free_gbdt_model(model);
        return 1;
    }
    // Set num_classes for a new model, or verify it for a loaded model
    if (model->params.num_trees == 0) {
        model->params.num_classes = num_classes;
    } else if (model->params.num_classes != num_classes) {
        fprintf(stderr, "Error: Mismatch in number of classes between loaded model (%d) and new data (%d).\n", model->params.num_classes, num_classes);
        free_dataset(train_data);
        free_gbdt_model(model);
        return 1;
    }
    printf("Training data loaded: %d samples, %d features, %d classes.\n", train_data->num_samples, train_data->num_features, num_classes);

    // --- Train Model ---
    printf("\nStarting training...\n");
    train_gbdt(model, train_data, iterations_to_add, save_interval, model_file);
    printf("Training finished. Final model has %d trees.\n", model->params.num_trees);

    // --- Test Model ---
    printf("\nLoading test data for evaluation...\n");
    Dataset* test_data = load_dataset(test_features_file, test_labels_file, &num_classes);
    if (!test_data) {
        fprintf(stderr, "Failed to load test data.\n");
        free_dataset(train_data);
        free_gbdt_model(model);
        return 1;
    }
    printf("Test data loaded: %d samples, %d features.\n", test_data->num_samples, test_data->num_features);

    printf("\nMaking predictions on test data...\n");
    PredictionResult* predictions = predict_gbdt(model, test_data);

    if (predictions) {
        evaluate(predictions, test_data, model);
    }

    // --- Clean up ---
    printf("\nCleaning up...\n");
    free_prediction_result(predictions);
    free_dataset(train_data);
    free_dataset(test_data);
    free_gbdt_model(model);
    printf("Done.\n");

    return 0;
}
