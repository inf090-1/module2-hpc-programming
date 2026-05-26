#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

typedef struct Node {
    int value;
    struct Node* left;
    struct Node* right;
} Node;

#ifndef PER_NODE_WORK
#define PER_NODE_WORK 10000
#endif

static int do_work(int val) {
    int s = val;
    for (int i = 0; i < PER_NODE_WORK; i++) {
        s += (i * val) & 0xFF;
    }
    return s;
}

// Task-based tree sum (solution).
static int tree_sum(Node* root) {
    if (root == NULL) return 0;

    int left_sum = 0;
    int right_sum = 0;

    #pragma omp taskgroup
    {
        if (root->left) {
            #pragma omp task shared(left_sum) firstprivate(root)
            left_sum = tree_sum(root->left);
        }

        if (root->right) {
            #pragma omp task shared(right_sum) firstprivate(root)
            right_sum = tree_sum(root->right);
        }
    }

    return do_work(root->value) + left_sum + right_sum;
}

static int tree_sum_serial(Node* root) {
    if (root == NULL) return 0;
    return do_work(root->value) + tree_sum_serial(root->left) + tree_sum_serial(root->right);
}

static int tree_count_nodes(Node* root) {
    if (root == NULL) return 0;
    return 1 + tree_count_nodes(root->left) + tree_count_nodes(root->right);
}

static void tree_collect_values(Node* root, int* out, int* idx) {
    if (!root) return;
    out[(*idx)++] = root->value;
    tree_collect_values(root->left, out, idx);
    tree_collect_values(root->right, out, idx);
}

static int tree_sum_omp_parallel_for(Node* root) {
    const int count = tree_count_nodes(root);
    int* vals = (int*)malloc((size_t)count * sizeof(int));
    if (!vals) {
        fprintf(stderr, "Allocation failed\n");
        exit(1);
    }

    int idx = 0;
    tree_collect_values(root, vals, &idx);

    int sum = 0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < count; ++i) {
        sum += do_work(vals[i]);
    }

    free(vals);
    return sum;
}

#ifndef TREE_DEPTH
#define TREE_DEPTH 12
#endif

static Node* build_perfect_tree(int value, int depth) {
    Node* node = malloc(sizeof(Node));
    node->value = value;
    if (depth > 1) {
        node->left = build_perfect_tree(value * 2, depth - 1);
        node->right = build_perfect_tree(value * 2 + 1, depth - 1);
    } else {
        node->left = NULL;
        node->right = NULL;
    }
    return node;
}

static void free_tree(Node* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main(void) {
    Node* root = build_perfect_tree(1, TREE_DEPTH);
    const int num_nodes = (1 << TREE_DEPTH) - 1;

    // Baseline 1: CPU serial.
    double t_serial0 = omp_get_wtime();
    int serial_res = tree_sum_serial(root);
    double t_serial1 = omp_get_wtime();
    double t_serial = t_serial1 - t_serial0;

    // Baseline 2: CPU OpenMP parallel-for.
    double t_omp0 = omp_get_wtime();
    int omp_pf_res = tree_sum_omp_parallel_for(root);
    double t_omp1 = omp_get_wtime();
    double t_omp = t_omp1 - t_omp0;

    // Solution: tree sum using tasks.
    int result = 0;
    double t_task0 = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            result = tree_sum(root);
        }
    }

    double t_task1 = omp_get_wtime();
    double t_task = t_task1 - t_task0;

    const int ok = (result == serial_res);

    printf("[tree-sum] depth=%d nodes=%d\n", TREE_DEPTH, num_nodes);
    printf("[tree-sum] cpu_serial=%d time=%.6f s\n", serial_res, t_serial);
    printf("[tree-sum] cpu_omp_parallel_for=%d time=%.6f s\n", omp_pf_res, t_omp);
    printf("[tree-sum] solution_tasks=%d time=%.6f s | %s\n",
           result, t_task, ok ? "PASS" : "FAIL");
    if (t_task > 0.0) {
        printf("[tree-sum] speedup_serial_vs_tasks=%.2fx\n", t_serial / t_task);
        printf("[tree-sum] speedup_omp_vs_tasks=%.2fx\n", t_omp / t_task);
    }

    free_tree(root);
    return ok ? 0 : 1;
}
