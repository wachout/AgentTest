#ifndef DATA_H
#define DATA_H

#include "gbdt.h"

Dataset* load_dataset(const char* feature_filepath, const char* label_filepath, int* num_classes);
void free_dataset(Dataset* dataset);

#endif // DATA_H
