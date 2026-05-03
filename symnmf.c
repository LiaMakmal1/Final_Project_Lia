#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"

#define EPS 0.0001
#define MAX_ITER 300
#define BETA 0.5
#define LINE_BUFFER 8192

static void error_exit(void) {
    printf("An Error Has Occurred\n");
    exit(1);
}

double** alloc_matrix(int rows, int cols) {
    double** mat;
    double* data;
    int i;

    mat = (double**)malloc(rows * sizeof(double*));
    if (mat == NULL) {
        error_exit();
    }
    data = (double*)calloc(rows * cols, sizeof(double));
    if (data == NULL) {
        free(mat);
        error_exit();
    }
    for (i = 0; i < rows; i++) {
        mat[i] = data + (i * cols);
    }
    return mat;
}

void free_matrix(double** mat, int rows) {
    (void)rows;
    if (mat != NULL) {
        free(mat[0]);
        free(mat);
    }
}

void print_matrix(double** mat, int rows, int cols) {
    int i, j;
    double val;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            val = mat[i][j];
            if (val > -0.00005 && val < 0.00005) {
                val = 0.0;
            }
            if (j < cols - 1) {
                printf("%.4f,", val);
            } else {
                printf("%.4f", val);
            }
        }
        printf("\n");
    }
}

/* count_dimensions_in_line: count comma-separated values in one line */
static int count_dimensions_in_line(const char* line) {
    int count, i;

    if (line == NULL || line[0] == '\0' || line[0] == '\n') {
        return 0;
    }
    count = 1;
    for (i = 0; line[i] != '\0' && line[i] != '\n'; i++) {
        if (line[i] == ',') {
            count++;
        }
    }
    return count;
}

/* count_rows_and_dims: scan fp to determine n (row count) and d (dimension) */
static void count_rows_and_dims(FILE* fp, int* out_n, int* out_d) {
    char line[LINE_BUFFER];

    *out_n = 0;
    *out_d = 0;
    while (fgets(line, LINE_BUFFER, fp) != NULL) {
        if (line[0] == '\n' || line[0] == '\0') {
            continue;
        }
        if (*out_d == 0) {
            *out_d = count_dimensions_in_line(line);
        }
        (*out_n)++;
    }
}

/* fill_points_matrix: parse lines from fp into pre-allocated X (n x d) */
static void fill_points_matrix(FILE* fp, double** X, int n, int d) {
    char line[LINE_BUFFER];
    char* token;
    int row, col;
    (void)n;

    row = 0;
    while (fgets(line, LINE_BUFFER, fp) != NULL) {
        if (line[0] == '\n' || line[0] == '\0') {
            continue;
        }
        col = 0;
        token = strtok(line, ",\n");
        while (token != NULL) {
            if (col >= d) {
                error_exit();
            }
            X[row][col] = strtod(token, NULL);
            col++;
            token = strtok(NULL, ",\n");
        }
        if (col != d) {
            error_exit();
        }
        row++;
    }
}

/* read_points_file: open filename, parse all points, return n x d matrix */
double** read_points_file(const char* filename, int* out_n, int* out_d) {
    FILE* fp;
    double** X;

    fp = fopen(filename, "r");
    if (fp == NULL) {
        error_exit();
    }
    count_rows_and_dims(fp, out_n, out_d);
    if (*out_n <= 0 || *out_d <= 0) {
        fclose(fp);
        error_exit();
    }
    X = alloc_matrix(*out_n, *out_d);
    rewind(fp);
    fill_points_matrix(fp, X, *out_n, *out_d);
    fclose(fp);
    return X;
}

/* squared_distance: return squared Euclidean distance between vectors a and b */
static double squared_distance(double* a, double* b, int d) {
    double sum, diff;
    int i;

    sum = 0.0;
    for (i = 0; i < d; i++) {
        diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/* compute_similarity_matrix: build n x n similarity matrix A from data X */
double** compute_similarity_matrix(double** X, int n, int d) {
    double** A;
    double dist;
    int i, j;

    A = alloc_matrix(n, n);
    for (i = 0; i < n; i++) {
        A[i][i] = 0.0;
        for (j = i + 1; j < n; j++) {
            dist = squared_distance(X[i], X[j], d);
            A[i][j] = exp(-(dist / 2.0));
            A[j][i] = A[i][j];
        }
    }
    return A;
}

/* compute_diagonal_degree_matrix: build diagonal degree matrix D from A */
double** compute_diagonal_degree_matrix(double** A, int n) {
    double** D;
    double sum;
    int i, j;

    D = alloc_matrix(n, n);
    for (i = 0; i < n; i++) {
        sum = 0.0;
        for (j = 0; j < n; j++) {
            sum += A[i][j];
        }
        D[i][i] = sum;
    }
    return D;
}

/* compute_normalized_similarity_matrix: return W = D^{-1/2} A D^{-1/2} */
double** compute_normalized_similarity_matrix(double** X, int n, int d) {
    double** A;
    double** D;
    double** W;
    double di, dj;
    int i, j;

    A = compute_similarity_matrix(X, n, d);
    D = compute_diagonal_degree_matrix(A, n);
    W = alloc_matrix(n, n);

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            di = D[i][i];
            dj = D[j][j];
            if (di == 0.0 || dj == 0.0) {
                W[i][j] = 0.0;
            } else {
                W[i][j] = A[i][j] / (sqrt(di) * sqrt(dj));
            }
        }
    }
    free_matrix(A, n);
    free_matrix(D, n);
    return W;
}

/* frobenius_diff_squared: return ||A - B||^2_F */
double frobenius_diff_squared(double** A, double** B, int rows, int cols) {
    double sum, diff;
    int i, j;

    sum = 0.0;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            diff = A[i][j] - B[i][j];
            sum += diff * diff;
        }
    }
    return sum;
}

/* multiply_matrices: return C = A (a_rows x a_cols) * B (a_cols x b_cols) */
static double** multiply_matrices(double** A, int a_rows, int a_cols,
    double** B, int b_cols) {
    double** C;
    double sum;
    int i, j, k;

    C = alloc_matrix(a_rows, b_cols);
    for (i = 0; i < a_rows; i++) {
        for (j = 0; j < b_cols; j++) {
            sum = 0.0;
            for (k = 0; k < a_cols; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}

/* transpose_matrix: return T = A^T where A is (rows x cols) */
static double** transpose_matrix(double** A, int rows, int cols) {
    double** T;
    int i, j;

    T = alloc_matrix(cols, rows);
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

/* copy_matrix_values: copy all entries from src into dst (rows x cols) */
static void copy_matrix_values(double** src, double** dst, int rows, int cols) {
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            dst[i][j] = src[i][j];
        }
    }
}

/* compute_h_next: apply one multiplicative NMF update step to H_curr */
static void compute_h_next(double** W, double** H_curr, double** H_next,
    int n, int k, double beta) {
    double** H_t;
    double** WH;
    double** HtH;
    double** denom;
    double ratio;
    int i, j;

    H_t   = transpose_matrix(H_curr, n, k);
    WH    = multiply_matrices(W, n, n, H_curr, k);
    HtH   = multiply_matrices(H_t, k, n, H_curr, k);
    denom = multiply_matrices(H_curr, n, k, HtH, k);

    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            if (denom[i][j] == 0.0) {
                ratio = 0.0;
            } else {
                ratio = WH[i][j] / denom[i][j];
            }
            H_next[i][j] = H_curr[i][j] * (1.0 - beta + beta * ratio);
        }
    }
    free_matrix(H_t, k);
    free_matrix(WH, n);
    free_matrix(HtH, k);
    free_matrix(denom, n);
}

/* symnmf_optimize: run multiplicative updates on H until convergence */
double** symnmf_optimize(double** H, double** W, int n, int k,
    int max_iter, double eps, double beta) {
    double** H_curr;
    double** H_next;
    int iter;

    H_curr = alloc_matrix(n, k);
    H_next = alloc_matrix(n, k);
    copy_matrix_values(H, H_curr, n, k);

    for (iter = 0; iter < max_iter; iter++) {
        compute_h_next(W, H_curr, H_next, n, k, beta);
        if (frobenius_diff_squared(H_next, H_curr, n, k) < eps) {
            copy_matrix_values(H_next, H_curr, n, k);
            break;
        }
        copy_matrix_values(H_next, H_curr, n, k);
    }
    free_matrix(H_next, n);
    return H_curr;
}

#ifndef SYMNMF_PYTHON_MODULE

/* run_goal: dispatch goal string to the correct computation and print result */
static void run_goal(const char* goal, double** X, int n, int d) {
    double** A;
    double** D;
    double** W;

    if (strcmp(goal, "sym") == 0) {
        A = compute_similarity_matrix(X, n, d);
        print_matrix(A, n, n);
        free_matrix(A, n);
    } else if (strcmp(goal, "ddg") == 0) {
        A = compute_similarity_matrix(X, n, d);
        D = compute_diagonal_degree_matrix(A, n);
        print_matrix(D, n, n);
        free_matrix(A, n);
        free_matrix(D, n);
    } else if (strcmp(goal, "norm") == 0) {
        W = compute_normalized_similarity_matrix(X, n, d);
        print_matrix(W, n, n);
        free_matrix(W, n);
    } else {
        error_exit();
    }
}

int main(int argc, char* argv[]) {
    double** X;
    int n, d;

    if (argc != 3) {
        error_exit();
    }
    X = read_points_file(argv[2], &n, &d);
    run_goal(argv[1], X, n, d);
    free_matrix(X, n);
    return 0;
}

#endif