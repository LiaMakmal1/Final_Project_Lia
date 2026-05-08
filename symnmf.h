#ifndef SYMNMF_H
#define SYMNMF_H

/* matrix utilities */
double** alloc_matrix(int rows, int cols);
void     free_matrix(double** mat, int rows);
void     print_matrix(double** mat, int rows, int cols);

/* file I/O */
double** read_points_file(const char* filename, int* out_n, int* out_d);

/* core SymNMF computations */
double** compute_similarity_matrix(double** X, int n, int d);
double** compute_diagonal_degree_matrix(double** A, int n);
double** compute_normalized_similarity_matrix(double** X, int n, int d);
double** symnmf_optimize(double** H, double** W, int n, int k,
    int max_iter, double eps, double beta);

/* helper used by the optimization loop */
double frobenius_diff_squared(double** A, double** B, int rows, int cols);

#endif
