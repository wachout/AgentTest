#include "tree.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>

#define NUM_THREADS 4

// Struct to hold feature value and gradient for sorting
typedef struct {
    double feature_value;
    double gradient;
} FeatureGradientPair;

// Struct for thread arguments
typedef struct {
    // Input to thread
    const Dataset* dataset;
    const double* gradients;
    const int* sample_indices;
    int num_samples;
    const GBDTParams* params;
    double parent_mse;
    int start_feature;
    int end_feature;

    // Output from thread
    double best_mse_reduction;
    int best_feature;
    double best_threshold;
} ThreadArgs;

// qsort comparison function for the struct
static int compare_pairs(const void* a, const void* b) {
    double val1 = ((FeatureGradientPair*)a)->feature_value;
    double val2 = ((FeatureGradientPair*)b)->feature_value;
    if (val1 < val2) return -1;
    if (val1 > val2) return 1;
    return 0;
}

// Forward declarations for helper functions
static DecisionTreeNode* build_node(const Dataset* dataset, const double* gradients, const int* sample_indices, int num_samples, const GBDTParams* params, int depth);
static void find_best_split_parallel(const Dataset* dataset, const double* gradients, const int* sample_indices, int num_samples, const GBDTParams* params, int* best_feature, double* best_threshold);
static void* find_best_split_worker(void* args);
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
    find_best_split_parallel(dataset, gradients, sample_indices, num_samples, params, &best_feature, &best_threshold);

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

// Worker thread function
void* find_best_split_worker(void* args) {
    ThreadArgs* thread_args = (ThreadArgs*)args;

    double total_grad_sum = 0.0;
    double total_grad_sq_sum = 0.0;
    for (int i = 0; i < thread_args->num_samples; i++) {
        double grad = thread_args->gradients[thread_args->sample_indices[i]];
        total_grad_sum += grad;
        total_grad_sq_sum += grad * grad;
    }

    FeatureGradientPair* pairs = (FeatureGradientPair*)malloc(thread_args->num_samples * sizeof(FeatureGradientPair));
    if (!pairs) {
        perror("Failed to allocate memory for pairs in worker");
        pthread_exit(NULL);
    }

    for (int f_idx = thread_args->start_feature; f_idx < thread_args->end_feature; f_idx++) {
        for (int i = 0; i < thread_args->num_samples; i++) {
            int sample_idx = thread_args->sample_indices[i];
            pairs[i].feature_value = thread_args->dataset->features[sample_idx][f_idx];
            pairs[i].gradient = thread_args->gradients[sample_idx];
        }

        qsort(pairs, thread_args->num_samples, sizeof(FeatureGradientPair), compare_pairs);

        double left_grad_sum = 0.0;
        double left_grad_sq_sum = 0.0;

        for (int i = 0; i < thread_args->num_samples - 1; i++) {
            left_grad_sum += pairs[i].gradient;
            left_grad_sq_sum += pairs[i].gradient * pairs[i].gradient;
            int n_left = i + 1;

            if (pairs[i].feature_value == pairs[i+1].feature_value) continue;

            double right_grad_sum = total_grad_sum - left_grad_sum;
            double right_grad_sq_sum = total_grad_sq_sum - left_grad_sq_sum;
            int n_right = thread_args->num_samples - n_left;

            if (n_left < thread_args->params->min_samples_split || n_right < thread_args->params->min_samples_split) continue;

            double left_mse = left_grad_sq_sum / n_left - (left_grad_sum / n_left) * (left_grad_sum / n_left);
            double right_mse = right_grad_sq_sum / n_right - (right_grad_sum / n_right) * (right_grad_sum / n_right);

            double weighted_mse = ((double)n_left / thread_args->num_samples) * left_mse + ((double)n_right / thread_args->num_samples) * right_mse;
            double mse_reduction = thread_args->parent_mse - weighted_mse;

            if (mse_reduction > thread_args->best_mse_reduction) {
                thread_args->best_mse_reduction = mse_reduction;
                thread_args->best_feature = f_idx;
                thread_args->best_threshold = pairs[i].feature_value;
            }
        }
    }
    free(pairs);
    pthread_exit(NULL);
}


// Master function to orchestrate parallel split finding
static void find_best_split_parallel(const Dataset* dataset, const double* gradients, const int* sample_indices, int num_samples, const GBDTParams* params, int* best_feature, double* best_threshold) {
    *best_feature = -1;
    double best_mse_reduction = -DBL_MAX;

    // Calculate parent MSE once
    double total_grad_sum = 0.0;
    double total_grad_sq_sum = 0.0;
    for (int i = 0; i < num_samples; i++) {
        double grad = gradients[sample_indices[i]];
        total_grad_sum += grad;
        total_grad_sq_sum += grad * grad;
    }
    double parent_mse = (num_samples > 0) ? (total_grad_sq_sum / num_samples - (total_grad_sum / num_samples) * (total_grad_sum / num_samples)) : 0.0;

    pthread_t threads[NUM_THREADS];
    ThreadArgs thread_args[NUM_THREADS];
    int features_per_thread = dataset->num_features / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_args[i].dataset = dataset;
        thread_args[i].gradients = gradients;
        thread_args[i].sample_indices = sample_indices;
        thread_args[i].num_samples = num_samples;
        thread_args[i].params = params;
        thread_args[i].parent_mse = parent_mse;
        thread_args[i].start_feature = i * features_per_thread;
        thread_args[i].end_feature = (i == NUM_THREADS - 1) ? dataset->num_features : (i + 1) * features_per_thread;
        thread_args[i].best_mse_reduction = -DBL_MAX;
        thread_args[i].best_feature = -1;
        thread_args[i].best_threshold = 0.0;

        pthread_create(&threads[i], NULL, find_best_split_worker, &thread_args[i]);
    }

    // Join threads and collect results
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        if (thread_args[i].best_mse_reduction > best_mse_reduction) {
            best_mse_reduction = thread_args[i].best_mse_reduction;
            *best_feature = thread_args[i].best_feature;
            *best_threshold = thread_args[i].best_threshold;
        }
    }
}
