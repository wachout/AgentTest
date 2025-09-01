#include "tree.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Forward declarations for helper functions
static DecisionTreeNode* build_node(const Dataset* dataset, const double* gradients, const int* sample_indices, int num_samples, const GBDTParams* params, int depth);
static void find_best_split(const Dataset* dataset, const double* gradients, const int* sample_indices, int num_samples, const GBDTParams* params, int* best_feature, double* best_threshold);
static double calculate_mse(const double* values, int num_samples);
static double calculate_leaf_value(const double* gradients, const int* sample_indices, int num_samples);
static void free_node(DecisionTreeNode* node);

// --- Public Functions ---

DecisionTree* build_decision_tree(const Dataset* dataset, const double* gradients, const int* sample_indices, int num_samples, const GBDTParams* params) {
    DecisionTree* tree = (DecisionTree*)malloc(sizeof(DecisionTree));
    if (!tree) {
        perror("Failed to allocate memory for DecisionTree");
        return NULL;
    }
    tree->max_depth = params->max_depth;
    tree->root = build_node(dataset, gradients, sample_indices, num_samples, params, 0);
    return tree;
}

void free_decision_tree(DecisionTree* tree) {
    if (tree) {
        free_node(tree->root);
        free(tree);
    }
}

double predict_tree(const DecisionTree* tree, const double* features) {
    if (!tree || !tree->root) {
        return 0.0;
    }
    DecisionTreeNode* current = tree->root;
    while (!current->is_leaf) {
        if (features[current->feature_index] <= current->threshold) {
            current = current->left_child;
        } else {
            current = current->right_child;
        }
    }
    return current->leaf_value;
}

// --- Helper Functions ---

static void free_node(DecisionTreeNode* node) {
    if (node) {
        free_node(node->left_child);
        free_node(node->right_child);
        free(node);
    }
}

static double calculate_leaf_value(const double* gradients, const int* sample_indices, int num_samples) {
    if (num_samples == 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < num_samples; i++) {
        sum += gradients[sample_indices[i]];
    }
    return sum / num_samples;
}

static DecisionTreeNode* create_leaf_node(const double* gradients, const int* sample_indices, int num_samples) {
    DecisionTreeNode* node = (DecisionTreeNode*)calloc(1, sizeof(DecisionTreeNode));
    if (!node) {
        perror("Failed to allocate memory for a leaf node");
        return NULL;
    }
    node->is_leaf = 1;
    node->leaf_value = calculate_leaf_value(gradients, sample_indices, num_samples);
    return node;
}

static DecisionTreeNode* build_node(const Dataset* dataset, const double* gradients, const int* sample_indices, int num_samples, const GBDTParams* params, int depth) {
    // Check stopping conditions
    if (depth >= params->max_depth || num_samples < params->min_samples_split) {
        return create_leaf_node(gradients, sample_indices, num_samples);
    }

    int best_feature = -1;
    double best_threshold = 0.0;
    find_best_split(dataset, gradients, sample_indices, num_samples, params, &best_feature, &best_threshold);

    if (best_feature == -1) {
        return create_leaf_node(gradients, sample_indices, num_samples);
    }

    // Create an internal node
    DecisionTreeNode* node = (DecisionTreeNode*)calloc(1, sizeof(DecisionTreeNode));
    if (!node) {
        perror("Failed to allocate memory for an internal node");
        return NULL;
    }
    node->is_leaf = 0;
    node->feature_index = best_feature;
    node->threshold = best_threshold;

    // Split samples into left and right children
    int* left_indices = (int*)malloc(num_samples * sizeof(int));
    int* right_indices = (int*)malloc(num_samples * sizeof(int));
    if (!left_indices || !right_indices) {
        perror("Failed to allocate memory for indices");
        free(node); free(left_indices); free(right_indices);
        return NULL;
    }
    int n_left = 0, n_right = 0;
    for (int i = 0; i < num_samples; i++) {
        int sample_idx = sample_indices[i];
        if (dataset->features[sample_idx][best_feature] <= best_threshold) {
            left_indices[n_left++] = sample_idx;
        } else {
            right_indices[n_right++] = sample_idx;
        }
    }

    // If split is not possible, create a leaf
    if (n_left == 0 || n_right == 0) {
        free(left_indices);
        free(right_indices);
        free(node); // Free the allocated internal node
        return create_leaf_node(gradients, sample_indices, num_samples);
    }

    node->left_child = build_node(dataset, gradients, left_indices, n_left, params, depth + 1);
    node->right_child = build_node(dataset, gradients, right_indices, n_right, params, depth + 1);

    free(left_indices);
    free(right_indices);

    return node;
}

static void find_best_split(const Dataset* dataset, const double* gradients, const int* sample_indices, int num_samples, const GBDTParams* params, int* best_feature, double* best_threshold) {
    double best_mse_reduction = -1.0;
    *best_feature = -1;

    double parent_mse = calculate_mse(gradients, num_samples);

    for (int f_idx = 0; f_idx < dataset->num_features; f_idx++) {
        for (int s_idx = 0; s_idx < num_samples; s_idx++) {
            int sample_idx = sample_indices[s_idx];
            double threshold = dataset->features[sample_idx][f_idx];

            double* left_gradients = (double*)malloc(num_samples * sizeof(double));
            double* right_gradients = (double*)malloc(num_samples * sizeof(double));
            if (!left_gradients || !right_gradients) {
                perror("Failed to allocate memory for gradients split");
                free(left_gradients); free(right_gradients);
                continue;
            }
            int n_left = 0, n_right = 0;

            for (int i = 0; i < num_samples; i++) {
                int current_sample_idx = sample_indices[i];
                if (dataset->features[current_sample_idx][f_idx] <= threshold) {
                    left_gradients[n_left++] = gradients[current_sample_idx];
                } else {
                    right_gradients[n_right++] = gradients[current_sample_idx];
                }
            }

            if (n_left == 0 || n_right == 0) {
                free(left_gradients);
                free(right_gradients);
                continue;
            }

            double mse_left = calculate_mse(left_gradients, n_left);
            double mse_right = calculate_mse(right_gradients, n_right);
            double weighted_mse = ((double)n_left / num_samples) * mse_left + ((double)n_right / num_samples) * mse_right;
            double mse_reduction = parent_mse - weighted_mse;

            if (mse_reduction > best_mse_reduction) {
                best_mse_reduction = mse_reduction;
                *best_feature = f_idx;
                *best_threshold = threshold;
            }
            free(left_gradients);
            free(right_gradients);
        }
    }
}

static double calculate_mse(const double* values, int num_samples) {
    if (num_samples < 2) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < num_samples; i++) {
        sum += values[i];
    }
    double mean = sum / num_samples;
    double sse = 0.0;
    for (int i = 0; i < num_samples; i++) {
        sse += (values[i] - mean) * (values[i] - mean);
    }
    return sse / num_samples;
}
