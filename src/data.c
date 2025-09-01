#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 10240 // Adjust as needed

// Helper function to count lines and features
static void get_data_dimensions(const char* filepath, int* num_samples, int* num_features) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    *num_samples = 0;
    *num_features = 0;
    char line[MAX_LINE_LENGTH];

    if (fgets(line, sizeof(line), file)) {
        (*num_samples)++;
        char* token = strtok(line, ",");
        while (token) {
            (*num_features)++;
            token = strtok(NULL, ",");
        }
    }

    while (fgets(line, sizeof(line), file)) {
        (*num_samples)++;
    }

    fclose(file);
}

// Helper function to count unique classes
static int count_classes(const char* label_filepath) {
    FILE* file = fopen(label_filepath, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", label_filepath);
        exit(EXIT_FAILURE);
    }

    int max_label = -1;
    int label;
    while (fscanf(file, "%d", &label) == 1) {
        if (label > max_label) {
            max_label = label;
        }
    }
    fclose(file);
    return max_label + 1;
}


Dataset* load_dataset(const char* feature_filepath, const char* label_filepath, int* num_classes) {
    int num_samples, num_features;
    get_data_dimensions(feature_filepath, &num_samples, &num_features);
    *num_classes = count_classes(label_filepath);

    // Allocate Dataset structure
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    if (!dataset) {
        fprintf(stderr, "Memory allocation failed for Dataset.\n");
        exit(EXIT_FAILURE);
    }
    dataset->num_samples = num_samples;
    dataset->num_features = num_features;

    // Allocate memory for features and labels
    dataset->features = (double**)malloc(num_samples * sizeof(double*));
    dataset->labels = (int*)malloc(num_samples * sizeof(int));
    if (!dataset->features || !dataset->labels) {
        fprintf(stderr, "Memory allocation failed for features/labels.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_samples; i++) {
        dataset->features[i] = (double*)malloc(num_features * sizeof(double));
        if (!dataset->features[i]) {
            fprintf(stderr, "Memory allocation failed for feature row.\n");
            exit(EXIT_FAILURE);
        }
    }

    // Load features
    FILE* feature_file = fopen(feature_filepath, "r");
    if (!feature_file) {
        fprintf(stderr, "Error opening file: %s\n", feature_filepath);
        exit(EXIT_FAILURE);
    }

    char line[MAX_LINE_LENGTH];
    int i = 0;
    while (fgets(line, sizeof(line), feature_file) && i < num_samples) {
        char* token = strtok(line, ",");
        int j = 0;
        while (token && j < num_features) {
            dataset->features[i][j] = atof(token);
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }
    fclose(feature_file);

    // Load labels
    FILE* label_file = fopen(label_filepath, "r");
    if (!label_file) {
        fprintf(stderr, "Error opening file: %s\n", label_filepath);
        exit(EXIT_FAILURE);
    }
    i = 0;
    while(fscanf(label_file, "%d", &dataset->labels[i]) == 1 && i < num_samples) {
        i++;
    }
    fclose(label_file);

    return dataset;
}

void free_dataset(Dataset* dataset) {
    if (dataset) {
        if (dataset->features) {
            for (int i = 0; i < dataset->num_samples; i++) {
                free(dataset->features[i]);
            }
            free(dataset->features);
        }
        if (dataset->labels) {
            free(dataset->labels);
        }
        free(dataset);
    }
}
