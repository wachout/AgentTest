#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data.h"

Dataset* load_data(const char *train_file, const char *label_file) {
    FILE *train_fp = fopen(train_file, "r");
    if (!train_fp) {
        perror("Could not open train file");
        return NULL;
    }

    FILE *label_fp = fopen(label_file, "r");
    if (!label_fp) {
        perror("Could not open label file");
        fclose(train_fp);
        return NULL;
    }

    // Determine number of columns from the first line of the training file
    char line[1024 * 100]; // Increased buffer size for large number of features
    if (fgets(line, sizeof(line), train_fp) == NULL) {
        fprintf(stderr, "Training file is empty.\n");
        fclose(train_fp);
        fclose(label_fp);
        return NULL;
    }
    int num_cols = 0;
    char *line_copy_for_count = strdup(line);
    char *token = strtok(line_copy_for_count, ",");
    while (token) {
        num_cols++;
        token = strtok(NULL, ",");
    }
    free(line_copy_for_count);

    // Count number of rows
    int num_rows = 1;
    while (fgets(line, sizeof(line), train_fp)) {
        num_rows++;
    }

    // Rewind file pointers
    rewind(train_fp);

    // Allocate dataset
    Dataset *dataset = (Dataset*)malloc(sizeof(Dataset));
    dataset->num_rows = num_rows;
    dataset->num_cols = num_cols;
    dataset->data = (float**)malloc(num_rows * sizeof(float*));
    dataset->labels = (int*)malloc(num_rows * sizeof(int));

    for (int i = 0; i < num_rows; i++) {
        dataset->data[i] = (float*)malloc(num_cols * sizeof(float));
    }

    // Read training data
    int row = 0;
    while (fgets(line, sizeof(line), train_fp) && row < num_rows) {
        char *line_ptr = line;
        line_ptr[strcspn(line_ptr, "\r\n")] = 0;
        char *line_copy = strdup(line_ptr);
        char *to_free = line_copy;
        int col = 0;
        while ((token = strsep(&line_copy, ",")) != NULL && col < num_cols) {
            dataset->data[row][col] = atof(token);
            col++;
        }
        free(to_free);
        row++;
    }

    // Read label data
    row = 0;
    while (fgets(line, sizeof(line), label_fp) && row < num_rows) {
        dataset->labels[row] = atoi(line);
        row++;
    }

    fclose(train_fp);
    fclose(label_fp);

    return dataset;
}

void free_dataset(Dataset *dataset) {
    if (!dataset) {
        return;
    }
    if (dataset->data) {
        for (int i = 0; i < dataset->num_rows; i++) {
            if (dataset->data[i]) {
                free(dataset->data[i]);
            }
        }
        free(dataset->data);
    }
    if (dataset->labels) {
        free(dataset->labels);
    }
    free(dataset);
}
