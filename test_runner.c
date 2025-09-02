#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data.h"
#include "xgboost.h"
#include "test_runner.h"

Dataset* generate_dummy_data(int num_samples, int num_features, int num_classes) {
    Dataset *dataset = (Dataset*)malloc(sizeof(Dataset));
    dataset->num_rows = num_samples;
    dataset->num_cols = num_features;
    dataset->data = (float**)malloc(num_samples * sizeof(float*));
    dataset->labels = (int*)malloc(num_samples * sizeof(int));

    for (int i = 0; i < num_samples; i++) {
        dataset->data[i] = (float*)malloc(num_features * sizeof(float));
        for (int j = 0; j < num_features; j++) {
            dataset->data[i][j] = (float)rand() / RAND_MAX;
        }
        dataset->labels[i] = rand() % num_classes;
    }
    return dataset;
}

void print_accuracy(XGBoostModel *model, Dataset *dataset) {
    int *predictions = (int*)malloc(dataset->num_rows * sizeof(int));
    xgboost_predict(model, dataset, predictions);

    int correct_predictions = 0;
    for (int i = 0; i < dataset->num_rows; i++) {
        if (predictions[i] == dataset->labels[i]) {
            correct_predictions++;
        }
    }

    float accuracy = (float)correct_predictions / dataset->num_rows;
    printf("Accuracy: %.2f%%\n", accuracy * 100);

    free(predictions);
}

void run_tests() {
    int num_samples = 1000;
    int num_features = 10000;
    int num_classes = 5;

    printf("Generating in-memory dataset with %d samples and %d features...\n", num_samples, num_features);
    Dataset *dataset = generate_dummy_data(num_samples, num_features, num_classes);
    printf("Dataset generated.\n");

    XGBoostParameter params;
    params.num_trees = 10;
    params.max_depth = 5;
    params.learning_rate = 0.1f;
    params.min_split_gain = 0.0f;
    params.lambda = 1.0f;
    params.gamma = 0.0f;
    params.num_classes = num_classes;

    printf("Starting XGBoost training...\n");
    clock_t start = clock();
    XGBoostModel *model = xgboost_train(dataset, params);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Training time: %f seconds\n", time_spent);

    if (model) {
        printf("Training complete.\n");
        print_accuracy(model, dataset);
        free_xgboost_model(model);
    } else {
        fprintf(stderr, "Training failed.\n");
    }

    free_dataset(dataset);
}
