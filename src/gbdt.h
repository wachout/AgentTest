#ifndef GBDT_H
#define GBDT_H

#include <stdio.h>
#include <stdlib.h>

// GBDT Hyperparameters
typedef struct {
    int num_trees;
    int max_depth;
    double learning_rate;
    int min_samples_split;
    double subsample;
    int num_classes;
} GBDTParams;

// Dataset structure
typedef struct {
    double** features;
    int* labels;
    int num_samples;
    int num_features;
} Dataset;

// Decision Tree Node
typedef struct DecisionTreeNode {
    int feature_index;
    double threshold;
    double leaf_value;
    struct DecisionTreeNode* left_child;
    struct DecisionTreeNode* right_child;
    int is_leaf;
} DecisionTreeNode;

// Decision Tree
typedef struct {
    DecisionTreeNode* root;
    int max_depth;
} DecisionTree;

// GBDT Model
typedef struct {
    DecisionTree*** trees; // A 2D array of trees: num_classes x num_trees
    GBDTParams params;
    double* initial_prediction; // Initial prediction for each class
} GBDTModel;

// GBDT training function
GBDTModel* train_gbdt(const Dataset* train_data, const GBDTParams* params);

// GBDT prediction function
int* predict_gbdt(const GBDTModel* model, const Dataset* test_data);

// Function to free the GBDT model
void free_gbdt_model(GBDTModel* model);


#endif // GBDT_H
