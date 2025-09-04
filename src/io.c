#include "io.h"
#include "tree.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// --- Forward Declarations ---
static void save_node_json(const DecisionTreeNode* node, FILE* fp, int indent);
static DecisionTreeNode* load_node_json(FILE* fp);

// --- JSON Serialization (Save) ---

static void print_indent(FILE* fp, int indent) {
    for(int i = 0; i < indent; ++i) fprintf(fp, " ");
}

static void save_node_json(const DecisionTreeNode* node, FILE* fp, int indent) {
    if (!node) {
        fprintf(fp, "null");
        return;
    }

    fprintf(fp, "{\n");

    print_indent(fp, indent + 2);
    fprintf(fp, "\"is_leaf\": %s", node->is_leaf ? "true" : "false");

    if (node->is_leaf) {
        fprintf(fp, ",\n");
        print_indent(fp, indent + 2);
        fprintf(fp, "\"leaf_value\": %f\n", node->leaf_value);
    } else {
        fprintf(fp, ",\n");
        print_indent(fp, indent + 2);
        fprintf(fp, "\"feature_index\": %d,\n", node->feature_index);
        print_indent(fp, indent + 2);
        fprintf(fp, "\"threshold\": %f,\n", node->threshold);
        print_indent(fp, indent + 2);
        fprintf(fp, "\"left_child\": ");
        save_node_json(node->left_child, fp, indent + 4);
        fprintf(fp, ",\n");
        print_indent(fp, indent + 2);
        fprintf(fp, "\"right_child\": ");
        save_node_json(node->right_child, fp, indent + 4);
        fprintf(fp, "\n");
    }

    print_indent(fp, indent);
    fprintf(fp, "}");
}

void save_gbdt_model(const GBDTModel* model, const char* filepath) {
    FILE* fp = fopen(filepath, "w");
    if (!fp) {
        perror("Failed to open file for writing");
        return;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"params\": {\n");
    fprintf(fp, "    \"num_trees\": %d,\n", model->params.num_trees);
    fprintf(fp, "    \"max_depth\": %d,\n", model->params.max_depth);
    fprintf(fp, "    \"learning_rate\": %f,\n", model->params.learning_rate);
    fprintf(fp, "    \"min_samples_split\": %d,\n", model->params.min_samples_split);
    fprintf(fp, "    \"subsample\": %f,\n", model->params.subsample);
    fprintf(fp, "    \"num_classes\": %d\n", model->params.num_classes);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"initial_prediction\": [");
    for (int i = 0; i < model->params.num_classes; i++) {
        fprintf(fp, "%f%s", model->initial_prediction[i], (i == model->params.num_classes - 1) ? "" : ", ");
    }
    fprintf(fp, "],\n");

    fprintf(fp, "  \"trees\": [\n");
    for (int k = 0; k < model->params.num_classes; k++) {
        fprintf(fp, "    [");
        for (int m = 0; m < model->params.num_trees; m++) {
            fprintf(fp, "\n");
            print_indent(fp, 6);
            save_node_json(model->trees[k][m]->root, fp, 6);
            if (m < model->params.num_trees - 1) fprintf(fp, ",");
        }
        fprintf(fp, "\n    ]%s\n", (k < model->params.num_classes - 1) ? "," : "");
    }
    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");
    fclose(fp);
}


// --- JSON Deserialization (Load) ---

static int next_char_skip_space(FILE* fp) {
    int c;
    while (isspace(c = fgetc(fp)));
    return c;
}

static int expect_char(FILE* fp, char expected) {
    int c = next_char_skip_space(fp);
    if (c == expected) return 1;
    fprintf(stderr, "Parser Error: Expected '%c', but got '%c' (EOF? %d)\n", expected, c, feof(fp));
    return 0;
}

static int parse_key(FILE* fp, const char* key) {
    if (!expect_char(fp, '"')) return 0;
    char buffer[100];
    int i = 0;
    char c;
    while((c = fgetc(fp)) != '"' && i < 99) {
        if (c == EOF) return 0;
        buffer[i++] = c;
    }
    buffer[i] = '\0';
    if (strcmp(buffer, key) != 0) {
        fprintf(stderr, "Parser Error: Expected key '\"%s\"', got '\"%s\"'\n", key, buffer);
        return 0;
    }
    return expect_char(fp, ':');
}

static int read_value_as_string(FILE* fp, char* buffer, int max_len) {
    int i = 0;
    int c = next_char_skip_space(fp);
    if (c == EOF) return 0;

    buffer[i++] = c;
    while((c = fgetc(fp)) != ',' && c != '}' && c != ']' && !isspace(c) && i < max_len -1) {
        if(c == EOF) break;
        buffer[i++] = c;
    }
    buffer[i] = '\0';
    ungetc(c, fp); // Put back the delimiter
    return 1;
}

static DecisionTreeNode* load_node_json(FILE* fp) {
    int c = next_char_skip_space(fp);
    if (c == 'n') { // "null"
        if (fgetc(fp) == 'u' && fgetc(fp) == 'l' && fgetc(fp) == 'l') return NULL;
        fprintf(stderr, "Parser Error: expected 'null'\n"); return NULL;
    }
    ungetc(c, fp);
    if (!expect_char(fp, '{')) return NULL;

    DecisionTreeNode* node = (DecisionTreeNode*)calloc(1, sizeof(DecisionTreeNode));
    char buffer[32];

    if (!parse_key(fp, "is_leaf")) { free(node); return NULL; }
    read_value_as_string(fp, buffer, 32);
    node->is_leaf = (strcmp(buffer, "true") == 0);

    if (!expect_char(fp, ',')) { free(node); return NULL; }

    if (node->is_leaf) {
        if (!parse_key(fp, "leaf_value")) { free(node); return NULL; }
        read_value_as_string(fp, buffer, 32);
        node->leaf_value = atof(buffer);
    } else {
        if (!parse_key(fp, "feature_index")) { free(node); return NULL; }
        read_value_as_string(fp, buffer, 32);
        node->feature_index = atoi(buffer);
        if (!expect_char(fp, ',')) { free(node); return NULL; }

        if (!parse_key(fp, "threshold")) { free(node); return NULL; }
        read_value_as_string(fp, buffer, 32);
        node->threshold = atof(buffer);
        if (!expect_char(fp, ',')) { free(node); return NULL; }

        if (!parse_key(fp, "left_child")) { free(node); return NULL; }
        node->left_child = load_node_json(fp);
        if (!expect_char(fp, ',')) { free(node); return NULL; }

        if (!parse_key(fp, "right_child")) { free(node); return NULL; }
        node->right_child = load_node_json(fp);
    }

    if (!expect_char(fp, '}')) { free_decision_tree( (DecisionTree*) node); return NULL; }
    return node;
}

GBDTModel* load_gbdt_model(const char* filepath) {
    FILE* fp = fopen(filepath, "r");
    if (!fp) { perror("Failed to open file for reading"); return NULL; }

    GBDTModel* model = (GBDTModel*)calloc(1, sizeof(GBDTModel));
    if (!model) { fclose(fp); return NULL; }
    char buffer[32];

    if (!expect_char(fp, '{')) goto error;

    // Params
    if (!parse_key(fp, "params")) goto error;
    if (!expect_char(fp, '{')) goto error;
    if (!parse_key(fp, "num_trees")) goto error;
    read_value_as_string(fp, buffer, 32); model->params.num_trees = atoi(buffer);
    if (!expect_char(fp, ',')) goto error;
    if (!parse_key(fp, "max_depth")) goto error;
    read_value_as_string(fp, buffer, 32); model->params.max_depth = atoi(buffer);
    if (!expect_char(fp, ',')) goto error;
    if (!parse_key(fp, "learning_rate")) goto error;
    read_value_as_string(fp, buffer, 32); model->params.learning_rate = atof(buffer);
    if (!expect_char(fp, ',')) goto error;
    if (!parse_key(fp, "min_samples_split")) goto error;
    read_value_as_string(fp, buffer, 32); model->params.min_samples_split = atoi(buffer);
    if (!expect_char(fp, ',')) goto error;
    if (!parse_key(fp, "subsample")) goto error;
    read_value_as_string(fp, buffer, 32); model->params.subsample = atof(buffer);
    if (!expect_char(fp, ',')) goto error;
    if (!parse_key(fp, "num_classes")) goto error;
    read_value_as_string(fp, buffer, 32); model->params.num_classes = atoi(buffer);
    if (!expect_char(fp, '}')) goto error;
    if (!expect_char(fp, ',')) goto error;

    // Initial prediction
    if (!parse_key(fp, "initial_prediction")) goto error;
    if (!expect_char(fp, '[')) goto error;
    model->initial_prediction = (double*)malloc(model->params.num_classes * sizeof(double));
    for (int i = 0; i < model->params.num_classes; i++) {
        read_value_as_string(fp, buffer, 32);
        model->initial_prediction[i] = atof(buffer);
        if (i < model->params.num_classes - 1) {
            if (!expect_char(fp, ',')) goto error;
        }
    }
    if (!expect_char(fp, ']')) goto error;
    if (!expect_char(fp, ',')) goto error;

    // Trees
    if (!parse_key(fp, "trees")) goto error;
    if (!expect_char(fp, '[')) goto error;
    model->trees = (DecisionTree***)malloc(model->params.num_classes * sizeof(DecisionTree**));
    for (int k = 0; k < model->params.num_classes; k++) {
        if (!expect_char(fp, '[')) goto error;
        model->trees[k] = (DecisionTree**)malloc(model->params.num_trees * sizeof(DecisionTree*));
        for (int m = 0; m < model->params.num_trees; m++) {
            model->trees[k][m] = (DecisionTree*)malloc(sizeof(DecisionTree));
            model->trees[k][m]->root = load_node_json(fp);
            if (m < model->params.num_trees - 1) {
                if (!expect_char(fp, ',')) goto error;
            }
        }
        if (!expect_char(fp, ']')) goto error;
        if (k < model->params.num_classes - 1) {
            if (!expect_char(fp, ',')) goto error;
        }
    }
    if (!expect_char(fp, ']')) goto error;
    if (!expect_char(fp, '}')) goto error;

    fclose(fp);
    return model;

error:
    fprintf(stderr, "Failed to parse model file.\n");
    fclose(fp);
    free_gbdt_model(model);
    return NULL;
}
