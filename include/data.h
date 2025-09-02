#ifndef DATA_H
#define DATA_H

#include <stdio.h>

typedef struct {
    float **data;
    int *labels;
    int num_rows;
    int num_cols;
} Dataset;

Dataset* load_data(const char *train_file, const char *label_file);
void free_dataset(Dataset *dataset);

#endif
