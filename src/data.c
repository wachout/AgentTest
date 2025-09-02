#define _GNU_SOURCE // For getline
#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper function to count lines and features robustly
static void get_data_dimensions(const char* filepath, int* num_samples, int* num_features) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    *num_samples = 0;
    *num_features = 1; // Start with 1 feature
    int c;
    int first_line = 1;

    while ((c = fgetc(file)) != EOF) {
        if (first_line && c == ',') {
            (*num_features)++;
        }
        if (c == '\n') {
            (*num_samples)++;
            first_line = 0;
        }
    }
    // If the file does not end with a newline, count the last line
    fseek(file, -1, SEEK_END);
    if (fgetc(file) != '\n') {
        (*num_samples)++;
    }

    fclose(file);
}

// Helper function to read a line of arbitrary length
static char* read_dynamic_line(FILE *fp) {
    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    read = getline(&line, &len, fp);

    if (read == -1) {
        free(line);
        return NULL;
    }

    // Strip newline character if present
    if (read > 0 && line[read - 1] == '\n') {
        line[read - 1] = '\0';
    }

    return line;
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

    if (num_samples == 0 || num_features == 0) {
        fprintf(stderr, "Error: No data found in %s\n", feature_filepath);
        exit(EXIT_FAILURE);
    }

    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    dataset->num_samples = num_samples;
    dataset->num_features = num_features;

    dataset->features = (double**)malloc(num_samples * sizeof(double*));
    for (int i = 0; i < num_samples; i++) {
        dataset->features[i] = (double*)malloc(num_features * sizeof(double));
    }
    dataset->labels = (int*)malloc(num_samples * sizeof(int));

    // Load features using dynamic line reading
    FILE* feature_file = fopen(feature_filepath, "r");
    if (!feature_file) {
        fprintf(stderr, "Error opening file: %s\n", feature_filepath);
        exit(EXIT_FAILURE);
    }

    char *line;
    int row = 0;
    while ((line = read_dynamic_line(feature_file)) != NULL && row < num_samples) {
        char *token = strtok(line, ",");
        int col = 0;
        while (token != NULL && col < num_features) {
            dataset->features[row][col++] = atof(token);
            token = strtok(NULL, ",");
        }
        free(line);
        row++;
    }
    fclose(feature_file);

    // Load labels
    FILE* label_file = fopen(label_filepath, "r");
    if (!label_file) {
        fprintf(stderr, "Error opening file: %s\n", label_filepath);
        exit(EXIT_FAILURE);
    }
    int sample_idx = 0;
    while(fscanf(label_file, "%d", &dataset->labels[sample_idx]) == 1 && sample_idx < num_samples) {
        sample_idx++;
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
