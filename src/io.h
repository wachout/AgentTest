#ifndef IO_H
#define IO_H

#include "gbdt.h"

// Save a GBDT model to a file
void save_gbdt_model(const GBDTModel* model, const char* filepath);

// Load a GBDT model from a file
GBDTModel* load_gbdt_model(const char* filepath);

#endif // IO_H
