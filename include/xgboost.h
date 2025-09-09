#ifndef XGBOOST_H
#define XGBOOST_H

#include "data.h"
#include "cJSON.h"

// XGBoost Parameters
typedef struct {
    int num_trees;          // Number of boosting rounds
    int max_depth;          // Maximum depth of a tree
    float learning_rate;    // Step size shrinkage
    float min_split_gain;   // Minimum loss reduction required to make a further partition
    float lambda;           // L2 regularization term on weights
    float gamma;            // Minimum sum of instance weight (hessian) needed in a child
    int num_classes;        // Number of classes for multi-class classification
    float subsample;        // Fraction of samples to be used for growing trees
    float colsample_bytree; // Fraction of features to be used for growing trees
} XGBoostParameter;

// Decision Tree Node
typedef struct DecisionTreeNode {
    int is_leaf;
    float value; // value for leaf node
    int split_feature;
    float split_value;
    struct DecisionTreeNode *left;
    struct DecisionTreeNode *right;
} DecisionTreeNode;

// Decision Tree
typedef struct {
    DecisionTreeNode *root;
} DecisionTree;

// XGBoost Model
typedef struct {
    DecisionTree **trees; // Total number of trees will be num_trees * num_classes
    XGBoostParameter params;
} XGBoostModel;

// Function declarations for the XGBoost algorithm
XGBoostModel* xgboost_train(Dataset *dataset, XGBoostParameter params, XGBoostModel *model);
void xgboost_predict(XGBoostModel *model, Dataset *dataset, int *predictions);
void free_xgboost_model(XGBoostModel *model);

// Functions for saving and loading the model
void save_model_json(XGBoostModel *model, const char *filename);
XGBoostModel* load_model_json(const char *filename);

// Function for single prediction
int xgboost_predict_single(XGBoostModel *model, float *features);

#endif
