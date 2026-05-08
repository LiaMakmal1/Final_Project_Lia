#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"

#define EPS         0.0001
#define MAX_ITER    300
#define BETA        0.5
#define LINE_BUFFER 8192

/* Prints the standard error message and exits.
   input: none
   output: none */
static void error_exit(void) {
    printf("An Error Has Occurred\n");
    exit(1);
}

/* Allocates a rows x cols matrix as one contiguous block.
   input: rows, cols
   output: zero-initialized matrix */
double** alloc_matrix(int rows, int cols) {
    double** mat;
    double* data;
    int i;

    mat = (double**)malloc(rows * sizeof(double*));
    if (mat == NULL) {
        error_exit();
    }
    /* single block; free_matrix only needs to free mat[0] and mat */
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

/* Frees a matrix allocated by alloc_matrix.
   input: matrix, rows (unused, kept for a consistent interface)
   output: none */
void free_matrix(double** mat, int rows) {
    (void)rows;
    if (mat != NULL) {
        free(mat[0]);
        free(mat);
    }
}

/* Prints the matrix to stdout, comma-separated, 4 decimal places per value.
   input: matrix (rows x cols), rows, cols
   output: none */
void print_matrix(double** mat, int rows, int cols) {
    int i, j;
    double val;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            val = mat[i][j];
            /* avoid -0.0000 in output */
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

/* Counts the number of comma-separated values in a single line.
   input: one line of text
   output: number of values */
static int count_dims(const char* line) {
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

/* Scans the file to determine the number of rows and columns.
   input: open file pointer, pointers for n and d
   output: fills *out_n with row count, *out_d with column count */
static void scan_dims(FILE* fp, int* out_n, int* out_d) {
    char line[LINE_BUFFER];

    *out_n = 0;
    *out_d = 0;
    while (fgets(line, LINE_BUFFER, fp) != NULL) {
        if (line[0] == '\n' || line[0] == '\0') {
            continue;
        }
        if (*out_d == 0) {
            *out_d = count_dims(line);
        }
        (*out_n)++;
    }
}

/* Parses values from fp into the pre-allocated matrix X.
   input: open file pointer, matrix X (n x d), n, d
   output: 1 on success, 0 on malformed input */
static int fill_matrix(FILE* fp, double** X, int n, int d) {
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
                return 0;
            }
            X[row][col] = strtod(token, NULL);
            col++;
            token = strtok(NULL, ",\n");
        }
        if (col != d) {
            return 0;
        }
        row++;
    }
    return 1;
}

/* Reads a .txt data file and returns its contents as a matrix.
   input: path to file, pointers for n and d
   output: data matrix X (n x d) */
double** read_data(const char* filename, int* out_n, int* out_d) {
    FILE* fp;
    double** X;

    fp = fopen(filename, "r");
    if (fp == NULL) {
        error_exit();
    }
    scan_dims(fp, out_n, out_d);
    if (*out_n <= 0 || *out_d <= 0) {
        fclose(fp);
        error_exit();
    }
    X = alloc_matrix(*out_n, *out_d);
    rewind(fp);
    if (!fill_matrix(fp, X, *out_n, *out_d)) {
        free_matrix(X, *out_n);
        fclose(fp);
        error_exit();
    }
    fclose(fp);
    return X;
}

/* Returns the squared Euclidean distance between two vectors.
   input: vectors a and b of length d
   output: ||a - b||^2 */
static double sq_dist(double* a, double* b, int d) {
    double sum, diff;
    int i;

    sum = 0.0;
    for (i = 0; i < d; i++) {
        diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/* Builds the similarity matrix A from data X.
   input: data matrix X (n x d), n, d
   output: similarity matrix A (n x n) */
double** sim_mat(double** X, int n, int d) {
    double** A;
    double dist;
    int i, j;

    A = alloc_matrix(n, n);
    for (i = 0; i < n; i++) {
        A[i][i] = 0.0;
        /* fill upper triangle, mirror to lower for symmetry */
        for (j = i + 1; j < n; j++) {
            dist = sq_dist(X[i], X[j], d);
            A[i][j] = exp(-(dist / 2.0));
            A[j][i] = A[i][j];
        }
    }
    return A;
}

/* Builds the diagonal degree matrix D from similarity matrix A.
   input: similarity matrix A (n x n), n
   output: diagonal matrix D (n x n) */
double** ddg_mat(double** A, int n) {
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

/* Builds the normalized similarity matrix W = D^{-1/2} A D^{-1/2}.
   input: data matrix X (n x d), n, d
   output: normalized similarity matrix W (n x n) */
double** norm_mat(double** X, int n, int d) {
    double** A;
    double** D;
    double** W;
    double di, dj;
    int i, j;

    A = sim_mat(X, n, d);
    D = ddg_mat(A, n);
    W = alloc_matrix(n, n);

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            di = D[i][i];
            dj = D[j][j];
            /* W[i][j] = A[i][j] / sqrt(d_i * d_j) */
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

/* Returns the squared Frobenius norm ||A - B||^2_F.
   input: matrices A and B (rows x cols)
   output: squared Frobenius norm */
double frob_sq(double** A, double** B, int rows, int cols) {
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

/* Multiplies two matrices and returns the result.
   input: A (a_rows x a_cols), B (a_cols x b_cols)
   output: C = A * B (a_rows x b_cols) */
static double** mat_mul(double** A, int a_rows, int a_cols,
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

/* Returns the transpose of A.
   input: A (rows x cols)
   output: A^T (cols x rows) */
static double** transpose(double** A, int rows, int cols) {
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

/* Copies all values from src into dst.
   input: src matrix, dst matrix (rows x cols)
   output: none */
static void mat_copy(double** src, double** dst, int rows, int cols) {
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            dst[i][j] = src[i][j];
        }
    }
}

/* Applies one SymNMF update step, writing the result into H_next.
   input: W (n x n), H_curr (n x k), H_next (n x k), n, k, beta
   output: none */
static void update_h(double** W, double** H_curr, double** H_next,
    int n, int k, double beta) {
    double** H_t;
    double** WH;
    double** HtH;
    double** denom;
    int i, j;

    H_t = transpose(H_curr, n, k);          /* k x n */
    WH = mat_mul(W, n, n, H_curr, k);       /* n x k, numerator */
    /* H*(H^T*H) = (H*H^T)*H by associativity; H^T*H is k x k */
    HtH = mat_mul(H_t, k, n, H_curr, k);   /* k x k */
    denom = mat_mul(H_curr, n, k, HtH, k); /* n x k, denominator */

    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            /* 1e-6 prevents division by zero */
            H_next[i][j] = H_curr[i][j] * (1.0 - beta + beta * WH[i][j] / (denom[i][j] + 1e-6));
        }
    }
    free_matrix(H_t, k);
    free_matrix(WH, n);
    free_matrix(HtH, k);
    free_matrix(denom, n);
}

/* Iteratively updates H until convergence or max_iter is reached.
   input: initial H (n x k), W (n x n), n, k, max_iter, eps, beta
   output: converged H (n x k) - caller must free */
double** optimize_h(double** H, double** W, int n, int k,
    int max_iter, double eps, double beta) {
    double** H_curr;
    double** H_next;
    int iter;

    H_curr = alloc_matrix(n, k);
    H_next = alloc_matrix(n, k);
    mat_copy(H, H_curr, n, k);

    for (iter = 0; iter < max_iter; iter++) {
        update_h(W, H_curr, H_next, n, k, beta);
        /* convergence check */
        if (frob_sq(H_next, H_curr, n, k) < eps) {
            mat_copy(H_next, H_curr, n, k);
            break;
        }
        mat_copy(H_next, H_curr, n, k);
    }
    free_matrix(H_next, n);
    return H_curr;
}

#ifndef SYMNMF_PYTHON_MODULE

/* Computes and prints the matrix requested by goal.
   input: goal string, data X (n x d), n, d
   output: none */
static void run_goal(const char* goal, double** X, int n, int d) {
    double** A;
    double** D;
    double** W;

    if (strcmp(goal, "sym") == 0) {
        A = sim_mat(X, n, d);
        print_matrix(A, n, n);
        free_matrix(A, n);
    } else if (strcmp(goal, "ddg") == 0) {
        A = sim_mat(X, n, d);
        D = ddg_mat(A, n);
        print_matrix(D, n, n);
        free_matrix(A, n);
        free_matrix(D, n);
    } else if (strcmp(goal, "norm") == 0) {
        W = norm_mat(X, n, d);
        print_matrix(W, n, n);
        free_matrix(W, n);
    }
}

/* Reads arguments, loads the data file, and runs the requested goal.
   input: argv[1] = goal, argv[2] = file path
   output: none */
int main(int argc, char* argv[]) {
    double** X;
    int n, d;

    if (argc != 3) {
        error_exit();
    }
    /* validate goal before allocating X so no memory is leaked on early exit */
    if (strcmp(argv[1], "sym") != 0 &&
        strcmp(argv[1], "ddg") != 0 &&
        strcmp(argv[1], "norm") != 0) {
        error_exit();
    }
    X = read_data(argv[2], &n, &d);
    run_goal(argv[1], X, n, d);
    free_matrix(X, n);
    return 0;
}

#endif
