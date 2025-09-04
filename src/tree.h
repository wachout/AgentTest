#ifndef TREE_H
#define TREE_H

#include "gbdt.h"

// Note: The 'dataset' parameter in these functions is the full dataset.
// 'sample_indices' is an array of indices indicating which samples from the dataset
// are to be used for building or prediction at the current node.

// Build a decision tree
DecisionTree* build_decision_tree(const Dataset* dataset, const double* gradients, const int* sample_indices, int num_samples, const GBDTParams* params);

// Free a decision tree
void free_decision_tree(DecisionTree* tree);

// Predict a value using a decision tree
double predict_tree(const DecisionTree* tree, const double* features);

#endif // TREE_H
