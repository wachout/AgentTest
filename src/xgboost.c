#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include "xgboost.h"

// Forward declarations for helper functions
static float predict_tree(DecisionTreeNode *node, float *features);
static DecisionTreeNode* build_tree(Dataset *dataset, int *sample_indices, int num_samples, float *gradients, float *hessians, XGBoostParameter params, int depth);
static void find_best_split(Dataset *dataset, int *sample_indices, int num_samples, float *gradients, float *hessians, XGBoostParameter params, int *best_feature, float *best_value, float *best_gain);
static float calculate_leaf_value(float *gradients, float *hessians, int *sample_indices, int num_samples, float lambda);
static void free_tree(DecisionTreeNode *node);

XGBoostModel* xgboost_train(Dataset *dataset, XGBoostParameter params) {
    int num_total_trees = params.num_trees * params.num_classes;

    XGBoostModel *model = (XGBoostModel*)malloc(sizeof(XGBoostModel));
    model->params = params;
    model->trees = (DecisionTree**)malloc(num_total_trees * sizeof(DecisionTree*));

    float **predictions = (float**)malloc(dataset->num_rows * sizeof(float*));
    for (int i = 0; i < dataset->num_rows; i++) {
        predictions[i] = (float*)calloc(params.num_classes, sizeof(float));
    }

    float *gradients = (float*)malloc(dataset->num_rows * sizeof(float));
    float *hessians = (float*)malloc(dataset->num_rows * sizeof(float));

    for (int t = 0; t < params.num_trees; t++) {
        float **probabilities = (float**)malloc(dataset->num_rows * sizeof(float*));
        for (int i = 0; i < dataset->num_rows; i++) {
            probabilities[i] = (float*)malloc(params.num_classes * sizeof(float));
            float sum_exp = 0.0f;
            for (int c = 0; c < params.num_classes; c++) {
                sum_exp += expf(predictions[i][c]);
            }
            for (int c = 0; c < params.num_classes; c++) {
                probabilities[i][c] = expf(predictions[i][c]) / sum_exp;
            }
        }

        for (int c = 0; c < params.num_classes; c++) {
            for (int i = 0; i < dataset->num_rows; i++) {
                int target = (dataset->labels[i] == c) ? 1 : 0;
                gradients[i] = probabilities[i][c] - target;
                hessians[i] = probabilities[i][c] * (1.0f - probabilities[i][c]);
            }

            int *sample_indices = (int*)malloc(dataset->num_rows * sizeof(int));
            for(int i = 0; i < dataset->num_rows; i++) {
                sample_indices[i] = i;
            }

            DecisionTreeNode *root = build_tree(dataset, sample_indices, dataset->num_rows, gradients, hessians, params, 0);

            int tree_index = t * params.num_classes + c;
            model->trees[tree_index] = (DecisionTree*)malloc(sizeof(DecisionTree));
            model->trees[tree_index]->root = root;

            for (int i = 0; i < dataset->num_rows; i++) {
                predictions[i][c] += params.learning_rate * predict_tree(root, dataset->data[i]);
            }

            free(sample_indices);
        }

        for (int i = 0; i < dataset->num_rows; i++) {
            free(probabilities[i]);
        }
        free(probabilities);
    }

    for (int i = 0; i < dataset->num_rows; i++) {
        free(predictions[i]);
    }
    free(predictions);
    free(gradients);
    free(hessians);

    return model;
}

static float predict_tree(DecisionTreeNode *node, float *features) {
    if (node->is_leaf) {
        return node->value;
    }
    if (features[node->split_feature] < node->split_value) {
        return predict_tree(node->left, features);
    } else {
        return predict_tree(node->right, features);
    }
}

void xgboost_predict(XGBoostModel *model, Dataset *dataset, int *predictions) {
    int num_total_trees = model->params.num_trees * model->params.num_classes;

    for (int i = 0; i < dataset->num_rows; i++) {
        float *scores = (float*)calloc(model->params.num_classes, sizeof(float));
        for (int t = 0; t < num_total_trees; t++) {
            int class_idx = t % model->params.num_classes;
            scores[class_idx] += model->params.learning_rate * predict_tree(model->trees[t]->root, dataset->data[i]);
        }

        int max_class = 0;
        float max_score = scores[0];
        for (int c = 1; c < model->params.num_classes; c++) {
            if (scores[c] > max_score) {
                max_score = scores[c];
                max_class = c;
            }
        }
        predictions[i] = max_class;
        free(scores);
    }
}

void free_xgboost_model(XGBoostModel *model) {
    if (model) {
        if (model->trees) {
            int num_total_trees = model->params.num_trees * model->params.num_classes;
            for (int i = 0; i < num_total_trees; i++) {
                if (model->trees[i]) {
                    free_tree(model->trees[i]->root);
                    free(model->trees[i]);
                }
            }
            free(model->trees);
        }
        free(model);
    }
}

static DecisionTreeNode* build_tree(Dataset *dataset, int *sample_indices, int num_samples, float *gradients, float *hessians, XGBoostParameter params, int depth) {
    DecisionTreeNode *node = (DecisionTreeNode*)calloc(1, sizeof(DecisionTreeNode));

    if (depth >= params.max_depth || num_samples <= 1) {
        node->is_leaf = 1;
        node->value = calculate_leaf_value(gradients, hessians, sample_indices, num_samples, params.lambda);
        return node;
    }

    int best_feature = -1;
    float best_value = 0.0f;
    float best_gain = 0.0f;

    find_best_split(dataset, sample_indices, num_samples, gradients, hessians, params, &best_feature, &best_value, &best_gain);

    if (best_gain <= params.min_split_gain) {
        node->is_leaf = 1;
        node->value = calculate_leaf_value(gradients, hessians, sample_indices, num_samples, params.lambda);
        return node;
    }

    node->is_leaf = 0;
    node->split_feature = best_feature;
    node->split_value = best_value;

    int *left_indices = (int*)malloc(num_samples * sizeof(int));
    int *right_indices = (int*)malloc(num_samples * sizeof(int));
    int left_count = 0;
    int right_count = 0;

    for (int i = 0; i < num_samples; i++) {
        int sample_idx = sample_indices[i];
        if (dataset->data[sample_idx][best_feature] < best_value) {
            left_indices[left_count++] = sample_idx;
        } else {
            right_indices[right_count++] = sample_idx;
        }
    }

    node->left = build_tree(dataset, left_indices, left_count, gradients, hessians, params, depth + 1);
    node->right = build_tree(dataset, right_indices, right_count, gradients, hessians, params, depth + 1);

    free(left_indices);
    free(right_indices);

    return node;
}

typedef struct {
    float value;
    int index;
} FeatureValue;

int compare_feature_values(const void *a, const void *b) {
    float val_a = ((FeatureValue*)a)->value;
    float val_b = ((FeatureValue*)b)->value;
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    return 0;
}

static void find_best_split(Dataset *dataset, int *sample_indices, int num_samples, float *gradients, float *hessians, XGBoostParameter params, int *best_feature, float *best_value, float *best_gain) {
    float G_total = 0.0f, H_total = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        int idx = sample_indices[i];
        G_total += gradients[idx];
        H_total += hessians[idx];
    }

    float base_score = (G_total * G_total) / (H_total + params.lambda);
    *best_gain = 0.0f;
    *best_feature = -1;

    float thread_best_gain = 0.0f;
    int thread_best_feature = -1;
    float thread_best_value = 0.0f;

    #pragma omp parallel private(thread_best_gain, thread_best_feature, thread_best_value)
    {
        thread_best_gain = 0.0f;
        thread_best_feature = -1;
        thread_best_value = 0.0f;

        #pragma omp for
        for (int j = 0; j < dataset->num_cols; j++) {
            FeatureValue *feature_values = (FeatureValue*)malloc(num_samples * sizeof(FeatureValue));
            for (int i = 0; i < num_samples; i++) {
                int sample_idx = sample_indices[i];
                feature_values[i].value = dataset->data[sample_idx][j];
                feature_values[i].index = sample_idx;
            }

            qsort(feature_values, num_samples, sizeof(FeatureValue), compare_feature_values);

            float G_left = 0.0f, H_left = 0.0f;
            for (int i = 0; i < num_samples - 1; i++) {
                int sample_idx = feature_values[i].index;
                G_left += gradients[sample_idx];
                H_left += hessians[sample_idx];

                if (feature_values[i].value == feature_values[i+1].value) {
                    continue;
                }

                float G_right = G_total - G_left;
                float H_right = H_total - H_left;

                if (H_left < params.gamma || H_right < params.gamma) {
                    continue;
                }

                float gain = (G_left * G_left) / (H_left + params.lambda) + (G_right * G_right) / (H_right + params.lambda) - base_score;

                if (gain > thread_best_gain) {
                    thread_best_gain = gain;
                    thread_best_feature = j;
                    thread_best_value = (feature_values[i].value + feature_values[i+1].value) / 2.0f;
                }
            }
            free(feature_values);
        }

        #pragma omp critical
        {
            if (thread_best_gain > *best_gain) {
                *best_gain = thread_best_gain;
                *best_feature = thread_best_feature;
                *best_value = thread_best_value;
            }
        }
    }
}

static float calculate_leaf_value(float *gradients, float *hessians, int *sample_indices, int num_samples, float lambda) {
    float G_sum = 0.0f, H_sum = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        int idx = sample_indices[i];
        G_sum += gradients[idx];
        H_sum += hessians[idx];
    }
    return -G_sum / (H_sum + lambda);
}

static void free_tree(DecisionTreeNode *node) {
    if (node) {
        if (!node->is_leaf) {
            free_tree(node->left);
            free_tree(node->right);
        }
        free(node);
    }
}

static cJSON* serialize_tree_node(DecisionTreeNode *node);
static DecisionTreeNode* deserialize_tree_node(cJSON *json_node);

void save_model_json(XGBoostModel *model, const char *filename) {
    cJSON *root = cJSON_CreateObject();

    cJSON *params_json = cJSON_CreateObject();
    cJSON_AddNumberToObject(params_json, "num_trees", model->params.num_trees);
    cJSON_AddNumberToObject(params_json, "max_depth", model->params.max_depth);
    cJSON_AddNumberToObject(params_json, "learning_rate", model->params.learning_rate);
    cJSON_AddNumberToObject(params_json, "min_split_gain", model->params.min_split_gain);
    cJSON_AddNumberToObject(params_json, "lambda", model->params.lambda);
    cJSON_AddNumberToObject(params_json, "gamma", model->params.gamma);
    cJSON_AddNumberToObject(params_json, "num_classes", model->params.num_classes);
    cJSON_AddItemToObject(root, "params", params_json);

    cJSON *trees_json = cJSON_CreateArray();
    int num_total_trees = model->params.num_trees * model->params.num_classes;
    for (int i = 0; i < num_total_trees; i++) {
        cJSON *tree_json = cJSON_CreateObject();
        cJSON_AddItemToObject(tree_json, "root", serialize_tree_node(model->trees[i]->root));
        cJSON_AddItemToArray(trees_json, tree_json);
    }
    cJSON_AddItemToObject(root, "trees", trees_json);

    char *json_string = cJSON_Print(root);
    FILE *fp = fopen(filename, "w");
    if (fp) {
        fprintf(fp, "%s", json_string);
        fclose(fp);
    }
    free(json_string);
    cJSON_Delete(root);
}

XGBoostModel* load_model_json(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return NULL;

    fseek(fp, 0, SEEK_END);
    long length = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buffer = (char*)malloc(length + 1);
    fread(buffer, 1, length, fp);
    fclose(fp);
    buffer[length] = '\0';

    cJSON *root = cJSON_Parse(buffer);
    free(buffer);

    if (!root) return NULL;

    XGBoostModel *model = (XGBoostModel*)malloc(sizeof(XGBoostModel));

    cJSON *params_json = cJSON_GetObjectItem(root, "params");
    model->params.num_trees = cJSON_GetObjectItem(params_json, "num_trees")->valueint;
    model->params.max_depth = cJSON_GetObjectItem(params_json, "max_depth")->valueint;
    model->params.learning_rate = cJSON_GetObjectItem(params_json, "learning_rate")->valuedouble;
    model->params.min_split_gain = cJSON_GetObjectItem(params_json, "min_split_gain")->valuedouble;
    model->params.lambda = cJSON_GetObjectItem(params_json, "lambda")->valuedouble;
    model->params.gamma = cJSON_GetObjectItem(params_json, "gamma")->valuedouble;
    model->params.num_classes = cJSON_GetObjectItem(params_json, "num_classes")->valueint;

    cJSON *trees_json = cJSON_GetObjectItem(root, "trees");
    int num_trees = cJSON_GetArraySize(trees_json);
    model->trees = (DecisionTree**)malloc(num_trees * sizeof(DecisionTree*));
    for (int i = 0; i < num_trees; i++) {
        cJSON *tree_json = cJSON_GetArrayItem(trees_json, i);
        cJSON *root_json = cJSON_GetObjectItem(tree_json, "root");
        model->trees[i] = (DecisionTree*)malloc(sizeof(DecisionTree));
        model->trees[i]->root = deserialize_tree_node(root_json);
    }

    cJSON_Delete(root);
    return model;
}

static cJSON* serialize_tree_node(DecisionTreeNode *node) {
    if (!node) return cJSON_CreateNull();
    cJSON *json_node = cJSON_CreateObject();
    cJSON_AddNumberToObject(json_node, "is_leaf", node->is_leaf);
    if (node->is_leaf) {
        cJSON_AddNumberToObject(json_node, "value", node->value);
    } else {
        cJSON_AddNumberToObject(json_node, "split_feature", node->split_feature);
        cJSON_AddNumberToObject(json_node, "split_value", node->split_value);
        cJSON_AddItemToObject(json_node, "left", serialize_tree_node(node->left));
        cJSON_AddItemToObject(json_node, "right", serialize_tree_node(node->right));
    }
    return json_node;
}

static DecisionTreeNode* deserialize_tree_node(cJSON *json_node) {
    if (!json_node || cJSON_IsNull(json_node)) return NULL;

    DecisionTreeNode *node = (DecisionTreeNode*)calloc(1, sizeof(DecisionTreeNode));
    if (!node) {
        perror("Failed to allocate memory for tree node");
        return NULL;
    }

    cJSON *is_leaf_json = cJSON_GetObjectItem(json_node, "is_leaf");
    if (!is_leaf_json) { free(node); return NULL; }
    node->is_leaf = is_leaf_json->valueint;

    if (node->is_leaf) {
        cJSON *value_json = cJSON_GetObjectItem(json_node, "value");
        if (!value_json) { free(node); return NULL; }
        node->value = value_json->valuedouble;
    } else {
        cJSON *split_feature_json = cJSON_GetObjectItem(json_node, "split_feature");
        if (!split_feature_json) { free(node); return NULL; }
        node->split_feature = split_feature_json->valueint;

        cJSON *split_value_json = cJSON_GetObjectItem(json_node, "split_value");
        if (!split_value_json) { free(node); return NULL; }
        node->split_value = split_value_json->valuedouble;

        node->left = deserialize_tree_node(cJSON_GetObjectItem(json_node, "left"));
        node->right = deserialize_tree_node(cJSON_GetObjectItem(json_node, "right"));

        if (node->left == NULL || node->right == NULL) {
            free_tree(node->left);
            free_tree(node->right);
            free(node);
            return NULL;
        }
    }
    return node;
}
