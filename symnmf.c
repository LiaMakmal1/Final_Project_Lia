#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"

#define LINE_BUFFER 8192

void error_exit(void) {
    printf("An Error Has Occurred\n");
    exit(1);
}

/* allocate a contiguous rows x cols zero matrix; return NULL on failure */
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

void free_matrix(double** mat, int rows) {
    (void)rows;
    if (mat != NULL) { free(mat[0]); free(mat); }
}

/* print matrix to stdout with 4 decimal places, comma-separated per row */
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

/* count comma-separated values in one line */
static int count_cols(const char* line) {
    int c = 1, i;
    if (line == NULL || line[0] == '\0' || line[0] == '\n') return 0;
    for (i = 0; line[i] != '\0' && line[i] != '\n'; i++)
        if (line[i] == ',') c++;
    return c;
}

/* scan fp to find number of data rows (n) and dimensions (d) */
static void scan_dims(FILE* fp, int* n, int* d) {
    char line[LINE_BUFFER];
    *n = 0; *d = 0;
    while (fgets(line, LINE_BUFFER, fp) != NULL) {
        if (line[0] == '\n' || line[0] == '\0') continue;
        if (*d == 0) *d = count_cols(line);
        (*n)++;
    }
}

/* parse fp into pre-allocated X; input file is assumed valid */
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

static double sq_dist(double* a, double* b, int d) {
    double s = 0.0, diff;
    int i;
    for (i = 0; i < d; i++) { diff = a[i] - b[i]; s += diff * diff; }
    return s;
}

/* A[i][j] = exp(-||xi - xj||^2 / 2), diagonal is 0 */
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

/* D[i][i] = sum of row i of A */
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

/* W = D^{-1/2} A D^{-1/2}; frees A and D before returning */
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

double frobenius_diff_squared(double** A, double** B, int rows, int cols) {
    double s = 0.0, diff;
    int i, j;
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++) { diff = A[i][j] - B[i][j]; s += diff * diff; }
    return s;
}

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

static void mat_copy(double** src, double** dst, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
            dst[i][j] = src[i][j];
}

/* one multiplicative update: H_new = H * (1-b + b*WH / (H*Ht*H + 1e-6)) */
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

/* iterate H until ||H_new - H||^2 < eps or max_iter steps; caller must free result */
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
