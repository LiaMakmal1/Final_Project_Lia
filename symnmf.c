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
            }
            else {
                printf("%.4f", val);
            }
        }
        printf("\n");
    }
}

static int count_dimensions_in_line(const char* line) {
    int count;
    int i;

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

double** read_points_file(const char* filename, int* out_n, int* out_d) {
    FILE* fp;
    char line[LINE_BUFFER];
    int n, d, row, col;
    double** X;
    char* token;

    fp = fopen(filename, "r");
    if (fp == NULL) {
        error_exit();
    }

    n = 0;
    d = 0;
    while (fgets(line, LINE_BUFFER, fp) != NULL) {
        if (line[0] == '\n' || line[0] == '\0') {
            continue;
        }
        if (d == 0) {
            d = count_dimensions_in_line(line);
        }
        n++;
    }

    if (n <= 0 || d <= 0) {
        fclose(fp);
        error_exit();
    }

    X = alloc_matrix(n, d);

    rewind(fp);
    row = 0;
    while (fgets(line, LINE_BUFFER, fp) != NULL) {
        if (line[0] == '\n' || line[0] == '\0') {
            continue;
        }

        col = 0;
        token = strtok(line, ",\n");
        while (token != NULL) {
            if (col >= d) {
                fclose(fp);
                free_matrix(X, n);
                error_exit();
            }
            X[row][col] = strtod(token, NULL);
            col++;
            token = strtok(NULL, ",\n");
        }

        if (col != d) {
            fclose(fp);
            free_matrix(X, n);
            error_exit();
        }

        row++;
    }

    fclose(fp);

    *out_n = n;
    *out_d = d;
    return X;
}

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
            }
            else {
                W[i][j] = A[i][j] / (sqrt(di) * sqrt(dj));
            }
        }
    }

    free_matrix(A, n);
    free_matrix(D, n);
    return W;
}

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

double** symnmf_optimize(double** H, double** W, int n, int k,
    int max_iter, double eps, double beta) {
    double** H_curr;
    double** H_next;
    double** H_t;
    double** WH;
    double** HtH;
    double** denom;
    double ratio;
    int iter, i, j;

    H_curr = alloc_matrix(n, k);
    H_next = alloc_matrix(n, k);

    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            H_curr[i][j] = H[i][j];
        }
    }

    for (iter = 0; iter < max_iter; iter++) {
        H_t = transpose_matrix(H_curr, n, k);
        WH = multiply_matrices(W, n, n, H_curr, k);
        HtH = multiply_matrices(H_t, k, n, H_curr, k);
        denom = multiply_matrices(H_curr, n, k, HtH, k);

        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                if (denom[i][j] == 0.0) {
                    ratio = 0.0;
                }
                else {
                    ratio = WH[i][j] / denom[i][j];
                }
                H_next[i][j] = H_curr[i][j] * (1.0 - beta + beta * ratio);
            }
        }

        free_matrix(H_t, k);
        free_matrix(WH, n);
        free_matrix(HtH, k);
        free_matrix(denom, n);

        if (frobenius_diff_squared(H_next, H_curr, n, k) < eps) {
            for (i = 0; i < n; i++) {
                for (j = 0; j < k; j++) {
                    H_curr[i][j] = H_next[i][j];
                }
            }
            break;
        }

        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                H_curr[i][j] = H_next[i][j];
            }
        }
    }

    free_matrix(H_next, n);
    return H_curr;
}

#ifndef SYMNMF_PYTHON_MODULE

int main(int argc, char* argv[]) {
    char* goal;
    char* filename;
    double** X;
    double** A;
    double** D;
    double** W;
    int n, d;

    if (argc != 3) {
        error_exit();
    }

    goal = argv[1];
    filename = argv[2];

    X = read_points_file(filename, &n, &d);

    if (strcmp(goal, "sym") == 0) {
        A = compute_similarity_matrix(X, n, d);
        print_matrix(A, n, n);
        free_matrix(A, n);
    }
    else if (strcmp(goal, "ddg") == 0) {
        A = compute_similarity_matrix(X, n, d);
        D = compute_diagonal_degree_matrix(A, n);
        print_matrix(D, n, n);
        free_matrix(A, n);
        free_matrix(D, n);
    }
    else if (strcmp(goal, "norm") == 0) {
        W = compute_normalized_similarity_matrix(X, n, d);
        print_matrix(W, n, n);
        free_matrix(W, n);
    }
    else {
        free_matrix(X, n);
        error_exit();
    }

    free_matrix(X, n);
    return 0;
}

#endif