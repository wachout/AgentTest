#include "tree.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Struct to hold feature value and gradient for sorting
typedef struct {
    double feature_value;
    double gradient;
} FeatureGradientPair;

// qsort comparison function for the struct
static int compare_pairs(const void* a, const void* b) {
    double val1 = ((FeatureGradientPair*)a)->feature_value;
    double val2 = ((FeatureGradientPair*)b)->feature_value;
    if (val1 < val2) return -1;
    if (val1 > val2) return 1;
    return 0;
}

// Fisher-Yates shuffle for an array of integers
static void shuffle_indices(int* array, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

// Forward declarations for helper functions
static DecisionTreeNode* build_node(const Dataset* dataset, const double* gradients, const int* sample_indices, int num_samples, const GBDTParams* params, int depth);
static void find_best_split_sorted(const Dataset* dataset, const double* gradients, const int* sample_indices, int num_samples, const GBDTParams* params, int* best_feature, double* best_threshold);
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
    if (depth >= params->max_depth || num_samples < params->min_samples_split) {
        return create_leaf_node(gradients, sample_indices, num_samples);
    }

    int best_feature = -1;
    double best_threshold = 0.0;
    find_best_split_sorted(dataset, gradients, sample_indices, num_samples, params, &best_feature, &best_threshold);

    if (best_feature == -1) {
        return create_leaf_node(gradients, sample_indices, num_samples);
    }

    DecisionTreeNode* node = (DecisionTreeNode*)calloc(1, sizeof(DecisionTreeNode));
    if (!node) {
        perror("Failed to allocate memory for an internal node");
        return NULL;
    }
    node->is_leaf = 0;
    node->feature_index = best_feature;
    node->threshold = best_threshold;

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

    if (n_left == 0 || n_right == 0) {
        free(left_indices);
        free(right_indices);
        free(node);
        return create_leaf_node(gradients, sample_indices, num_samples);
    }

    node->left_child = build_node(dataset, gradients, left_indices, n_left, params, depth + 1);
    node->right_child = build_node(dataset, gradients, right_indices, n_right, params, depth + 1);

    free(left_indices);
    free(right_indices);

    return node;
}

static void find_best_split_sorted(const Dataset* dataset, const double* gradients, const int* sample_indices, int num_samples, const GBDTParams* params, int* best_feature, double* best_threshold) {
    double best_mse_reduction = -DBL_MAX;
    *best_feature = -1;

    double total_grad_sum = 0.0;
    double total_grad_sq_sum = 0.0;
    for (int i = 0; i < num_samples; i++) {
        double grad = gradients[sample_indices[i]];
        total_grad_sum += grad;
        total_grad_sq_sum += grad * grad;
    }
    double parent_mse = total_grad_sq_sum / num_samples - (total_grad_sum / num_samples) * (total_grad_sum / num_samples);

    FeatureGradientPair* pairs = (FeatureGradientPair*)malloc(num_samples * sizeof(FeatureGradientPair));
    if (!pairs) {
        perror("Failed to allocate memory for feature-gradient pairs");
        return;
    }

    // --- Feature Sub-sampling Logic ---
    int num_features_to_sample = dataset->num_features;
    if (params->subsample < 1.0 && params->subsample > 0.0) {
        num_features_to_sample = (int)(dataset->num_features * params->subsample);
        if (num_features_to_sample < 1) num_features_to_sample = 1;
    }

    int* feature_indices = (int*)malloc(dataset->num_features * sizeof(int));
    if (!feature_indices) { free(pairs); return; }
    for (int i = 0; i < dataset->num_features; i++) feature_indices[i] = i;

    if (num_features_to_sample < dataset->num_features) {
        shuffle_indices(feature_indices, dataset->num_features);
    }
    // --- End Sub-sampling Logic ---

    for (int i = 0; i < num_features_to_sample; i++) {
        int f_idx = feature_indices[i]; // Use the sampled feature index

        for (int j = 0; j < num_samples; j++) {
            int sample_idx = sample_indices[j];
            pairs[j].feature_value = dataset->features[sample_idx][f_idx];
            pairs[j].gradient = gradients[sample_idx];
        }

        qsort(pairs, num_samples, sizeof(FeatureGradientPair), compare_pairs);

        double left_grad_sum = 0.0;
        double left_grad_sq_sum = 0.0;

        for (int j = 0; j < num_samples - 1; j++) {
            left_grad_sum += pairs[j].gradient;
            left_grad_sq_sum += pairs[j].gradient * pairs[j].gradient;
            int n_left = j + 1;

            if (pairs[j].feature_value == pairs[j+1].feature_value) continue;

            double right_grad_sum = total_grad_sum - left_grad_sum;
            double right_grad_sq_sum = total_grad_sq_sum - left_grad_sq_sum;
            int n_right = num_samples - n_left;

            if (n_left < params->min_samples_split || n_right < params->min_samples_split) continue;

            double left_mse = left_grad_sq_sum / n_left - (left_grad_sum / n_left) * (left_grad_sum / n_left);
            double right_mse = right_grad_sq_sum / n_right - (right_grad_sum / n_right) * (right_grad_sum / n_right);

            double weighted_mse = ((double)n_left / num_samples) * left_mse + ((double)n_right / num_samples) * right_mse;
            double mse_reduction = parent_mse - weighted_mse;

            if (mse_reduction > best_mse_reduction) {
                best_mse_reduction = mse_reduction;
                *best_feature = f_idx;
                *best_threshold = pairs[j].feature_value;
            }
        }
    }
    free(pairs);
    free(feature_indices);
}
