#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"

#define LINE_BUFFER 8192

/* Print error message and terminate.
   input: none
   output: none */
void error_exit(void) {
    printf("An Error Has Occurred\n");
    exit(1);
}

/* Allocate a contiguous rows x cols zero-initialized matrix.
   input: rows, cols - matrix dimensions
   output: allocated matrix pointer, or NULL on failure */
double** alloc_matrix(int rows, int cols) {
    double** mat;
    double*  data;
    int i;

    mat = (double**)malloc(rows * sizeof(double*));
    if (mat == NULL) return NULL;
    data = (double*)calloc(rows * cols, sizeof(double));
    if (data == NULL) { free(mat); return NULL; }
    for (i = 0; i < rows; i++)
        mat[i] = data + i * cols;
    return mat;
}

/* Free a matrix allocated by alloc_matrix.
   input: mat - matrix to free, rows - unused
   output: none */
void free_matrix(double** mat, int rows) {
    (void)rows;
    if (mat != NULL) { free(mat[0]); free(mat); }
}

/* Print a matrix to stdout, comma-separated per row, 4 decimal places.
   input: mat - matrix to print, rows, cols - dimensions
   output: none */
void print_matrix(double** mat, int rows, int cols) {
    int i, j;
    double v;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            v = mat[i][j];
            if (v > -0.00005 && v < 0.00005) v = 0.0;
            if (j < cols - 1) printf("%.4f,", v);
            else               printf("%.4f",  v);
        }
        printf("\n");
    }
}

/* Count the number of comma-separated values in a line.
   input: line - null-terminated string
   output: column count, or 0 for empty/null lines */
static int count_cols(const char* line) {
    int c = 1, i;
    if (line == NULL || line[0] == '\0' || line[0] == '\n') return 0;
    for (i = 0; line[i] != '\0' && line[i] != '\n'; i++)
        if (line[i] == ',') c++;
    return c;
}

/* Scan a file to determine the number of data rows and columns.
   input: fp - open file pointer, n, d - output pointers for dimensions
   output: none (results written to *n and *d) */
static void scan_dims(FILE* fp, int* n, int* d) {
    char line[LINE_BUFFER];
    *n = 0; *d = 0;
    while (fgets(line, LINE_BUFFER, fp) != NULL) {
        if (line[0] == '\n' || line[0] == '\0') continue;
        if (*d == 0) *d = count_cols(line);
        (*n)++;
    }
}

/* Parse an open file into a pre-allocated matrix; file is assumed valid.
   input: fp - open file pointer rewound to start, X - pre-allocated matrix
   output: none */
static void fill_matrix(FILE* fp, double** X) {
    char line[LINE_BUFFER];
    char* tok;
    int row = 0, col;

    while (fgets(line, LINE_BUFFER, fp) != NULL) {
        if (line[0] == '\n' || line[0] == '\0') continue;
        col = 0;
        for (tok = strtok(line, ",\n"); tok != NULL; tok = strtok(NULL, ",\n"))
            X[row][col++] = strtod(tok, NULL);
        row++;
    }
}

/* Read data points from a file into a newly allocated matrix.
   input: filename - path to input file, out_n, out_d - output dimension pointers
   output: allocated n x d matrix, exits on error */
double** read_points_file(const char* filename, int* out_n, int* out_d) {
    FILE* fp;
    double** X;

    fp = fopen(filename, "r");
    if (fp == NULL) error_exit();
    scan_dims(fp, out_n, out_d);
    if (*out_n <= 0 || *out_d <= 0) { fclose(fp); error_exit(); }
    X = alloc_matrix(*out_n, *out_d);
    if (X == NULL) { fclose(fp); error_exit(); }
    rewind(fp);
    fill_matrix(fp, X);
    fclose(fp);
    return X;
}

/* Compute the squared Euclidean distance between two vectors.
   input: a, b - d-dimensional vectors, d - dimension
   output: squared distance as double */
static double sq_dist(double* a, double* b, int d) {
    double s = 0.0, diff;
    int i;
    for (i = 0; i < d; i++) { diff = a[i] - b[i]; s += diff * diff; }
    return s;
}

/* Compute similarity matrix: A[i][j] = exp(-||xi-xj||^2/2), diagonal 0.
   input: X - n x d data matrix, n, d - dimensions
   output: allocated n x n similarity matrix, or NULL on failure */
double** compute_similarity_matrix(double** X, int n, int d) {
    double** A;
    int i, j;

    A = alloc_matrix(n, n);
    if (A == NULL) return NULL;
    for (i = 0; i < n; i++)
        for (j = i + 1; j < n; j++) {
            A[i][j] = exp(-sq_dist(X[i], X[j], d) / 2.0);
            A[j][i] = A[i][j];
        }
    return A;
}

/* Compute diagonal degree matrix: D[i][i] = sum of row i of A.
   input: A - n x n similarity matrix, n - dimension
   output: allocated n x n diagonal matrix, or NULL on failure */
double** compute_diagonal_degree_matrix(double** A, int n) {
    double** D;
    double s;
    int i, j;

    D = alloc_matrix(n, n);
    if (D == NULL) return NULL;
    for (i = 0; i < n; i++) {
        s = 0.0;
        for (j = 0; j < n; j++) s += A[i][j];
        D[i][i] = s;
    }
    return D;
}

/* Compute normalized similarity W = D^{-1/2} A D^{-1/2}; frees A and D.
   input: X - n x d data matrix, n, d - dimensions
   output: allocated n x n normalized matrix, or NULL on failure */
double** compute_normalized_similarity_matrix(double** X, int n, int d) {
    double** A;
    double** D;
    double** W;
    double di, dj;
    int i, j;

    A = compute_similarity_matrix(X, n, d);
    if (A == NULL) return NULL;
    D = compute_diagonal_degree_matrix(A, n);
    if (D == NULL) { free_matrix(A, n); return NULL; }
    W = alloc_matrix(n, n);
    if (W == NULL) { free_matrix(A, n); free_matrix(D, n); return NULL; }
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            di = D[i][i]; dj = D[j][j];
            W[i][j] = (di == 0.0 || dj == 0.0) ? 0.0 : A[i][j] / (sqrt(di) * sqrt(dj));
        }
    free_matrix(A, n);
    free_matrix(D, n);
    return W;
}

/* Compute the squared Frobenius norm of (A - B).
   input: A, B - matrices of shape rows x cols
   output: sum of squared element-wise differences */
double frobenius_diff_squared(double** A, double** B, int rows, int cols) {
    double s = 0.0, diff;
    int i, j;
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++) { diff = A[i][j] - B[i][j]; s += diff * diff; }
    return s;
}

/* Multiply two matrices: C = A (ar x ac) * B (ac x bc).
   input: A - ar x ac matrix, B - ac x bc matrix, dimensions
   output: allocated ar x bc result matrix, or NULL on failure */
static double** mat_mul(double** A, int ar, int ac, double** B, int bc) {
    double** C;
    double s;
    int i, j, k;

    C = alloc_matrix(ar, bc);
    if (C == NULL) return NULL;
    for (i = 0; i < ar; i++)
        for (j = 0; j < bc; j++) {
            s = 0.0;
            for (k = 0; k < ac; k++) s += A[i][k] * B[k][j];
            C[i][j] = s;
        }
    return C;
}

/* Transpose a matrix.
   input: A - rows x cols matrix
   output: allocated cols x rows transposed matrix, or NULL on failure */
static double** mat_transpose(double** A, int rows, int cols) {
    double** T;
    int i, j;

    T = alloc_matrix(cols, rows);
    if (T == NULL) return NULL;
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
            T[j][i] = A[i][j];
    return T;
}

/* Copy src into dst element-wise.
   input: src, dst - matrices of shape rows x cols
   output: none */
static void mat_copy(double** src, double** dst, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
            dst[i][j] = src[i][j];
}

/* Apply one multiplicative H update: H_new = H*(1-b + b*WH/(H*Ht*H+1e-6)).
   input: W - n x n weight matrix, H, H_new - n x k matrices, n, k, beta
   output: 1 on success, 0 on allocation failure */
static int update_h(double** W, double** H, double** H_new, int n, int k, double beta) {
    double** Ht;
    double** WH;
    double** HtH;
    double** denom;
    int i, j;

    Ht    = mat_transpose(H, n, k);
    if (Ht == NULL) return 0;
    WH    = mat_mul(W, n, n, H, k);
    if (WH == NULL) { free_matrix(Ht, k); return 0; }
    HtH   = mat_mul(Ht, k, n, H, k);
    if (HtH == NULL) { free_matrix(Ht, k); free_matrix(WH, n); return 0; }
    denom = mat_mul(H, n, k, HtH, k);
    if (denom == NULL) {
        free_matrix(Ht, k); free_matrix(WH, n); free_matrix(HtH, k); return 0;
    }
    for (i = 0; i < n; i++)
        for (j = 0; j < k; j++)
            H_new[i][j] = H[i][j] * (1.0 - beta + beta * WH[i][j] / (denom[i][j] + 1e-6));
    free_matrix(Ht, k); free_matrix(WH, n);
    free_matrix(HtH, k); free_matrix(denom, n);
    return 1;
}

/* Run SymNMF optimization until convergence or max_iter steps.
   input: H - initial n x k matrix, W - n x n weight matrix, n, k, max_iter, eps, beta
   output: final H matrix (caller must free), or NULL on failure */
double** symnmf_optimize(double** H, double** W, int n, int k,
    int max_iter, double eps, double beta) {
    double** H_curr;
    double** H_next;
    int iter;

    H_curr = alloc_matrix(n, k);
    if (H_curr == NULL) return NULL;
    H_next = alloc_matrix(n, k);
    if (H_next == NULL) { free_matrix(H_curr, n); return NULL; }
    mat_copy(H, H_curr, n, k);
    for (iter = 0; iter < max_iter; iter++) {
        if (!update_h(W, H_curr, H_next, n, k, beta)) {
            free_matrix(H_curr, n); free_matrix(H_next, n); return NULL;
        }
        if (frobenius_diff_squared(H_next, H_curr, n, k) < eps) {
            mat_copy(H_next, H_curr, n, k);
            break;
        }
        mat_copy(H_next, H_curr, n, k);
    }
    free_matrix(H_next, n);
    return H_curr;
}

#ifndef SYMNMF_PYTHON_MODULE

/* Compute and print the matrix for a given goal; frees X on error.
   input: goal - "sym", "ddg", or "norm"; X - n x d data matrix, n, d
   output: none */
static void run_goal(const char* goal, double** X, int n, int d) {
    double** A;
    double** D;
    double** W;

    if (strcmp(goal, "sym") == 0) {
        A = compute_similarity_matrix(X, n, d);
        if (A == NULL) { free_matrix(X, n); error_exit(); }
        print_matrix(A, n, n);
        free_matrix(A, n);
    } else if (strcmp(goal, "ddg") == 0) {
        A = compute_similarity_matrix(X, n, d);
        if (A == NULL) { free_matrix(X, n); error_exit(); }
        D = compute_diagonal_degree_matrix(A, n);
        if (D == NULL) { free_matrix(A, n); free_matrix(X, n); error_exit(); }
        print_matrix(D, n, n);
        free_matrix(A, n);
        free_matrix(D, n);
    } else if (strcmp(goal, "norm") == 0) {
        W = compute_normalized_similarity_matrix(X, n, d);
        if (W == NULL) { free_matrix(X, n); error_exit(); }
        print_matrix(W, n, n);
        free_matrix(W, n);
    }
}

/* Entry point: read goal and filename from argv, compute and print the result.
   input: argc - argument count, argv - [program, goal, filename]
   output: 0 on success */
int main(int argc, char* argv[]) {
    double** X;
    int n, d;

    if (argc != 3) error_exit();
    if (strcmp(argv[1], "sym") != 0 &&
        strcmp(argv[1], "ddg") != 0 &&
        strcmp(argv[1], "norm") != 0) error_exit();
    X = read_points_file(argv[2], &n, &d);
    run_goal(argv[1], X, n, d);
    free_matrix(X, n);
    return 0;
}

#endif
