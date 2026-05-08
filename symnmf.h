/*
 * symnmf.h  --  Public interface for the SymNMF C implementation.
 *
 * Functions declared here are implemented in symnmf.c and used by both
 * the standalone executable (main in symnmf.c) and the Python extension
 * (symnmfmodule.c).
 */

#ifndef SYMNMF_H
#define SYMNMF_H

/* --- Matrix memory management ------------------------------------------- */

/* alloc_matrix: allocate a zero-initialised rows x cols matrix. */
double** alloc_matrix(int rows, int cols);

/* free_matrix: release a matrix allocated by alloc_matrix. */
void free_matrix(double** mat, int rows);

/* print_matrix: print every entry formatted to 4 decimal places. */
void print_matrix(double** mat, int rows, int cols);

/* --- File I/O ------------------------------------------------------------ */

/* read_points_file: parse a comma-separated .txt file; set *out_n, *out_d. */
double** read_points_file(const char* filename, int* out_n, int* out_d);

/* --- SymNMF math --------------------------------------------------------- */

/* compute_similarity_matrix: build A where A[i][j] = exp(-||xi-xj||^2/2). */
double** compute_similarity_matrix(double** X, int n, int d);

/* compute_diagonal_degree_matrix: build D where D[i][i] = sum of row i of A. */
double** compute_diagonal_degree_matrix(double** A, int n);

/* compute_normalized_similarity_matrix: return W = D^{-1/2} A D^{-1/2}. */
double** compute_normalized_similarity_matrix(double** X, int n, int d);

/* symnmf_optimize: iterate multiplicative updates on H until convergence. */
double** symnmf_optimize(double** H, double** W, int n, int k,
    int max_iter, double eps, double beta);

/* --- Utility ------------------------------------------------------------ */

/* frobenius_diff_squared: return ||A - B||^2_F. */
double frobenius_diff_squared(double** A, double** B, int rows, int cols);

#endif /* SYMNMF_H */
