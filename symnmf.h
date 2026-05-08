#ifndef SYMNMF_H
#define SYMNMF_H

/* matrix utilities */
double** alloc_matrix(int rows, int cols);
void     free_matrix(double** mat, int rows);
void     print_matrix(double** mat, int rows, int cols);

/* file I/O */
double** read_data(const char* filename, int* out_n, int* out_d);

/* core SymNMF computations */
double** sim_mat(double** X, int n, int d);
double** ddg_mat(double** A, int n);
double** norm_mat(double** X, int n, int d);
double** optimize_h(double** H, double** W, int n, int k,
    int max_iter, double eps, double beta);

/* helper used by the optimization loop */
double frob_sq(double** A, double** B, int rows, int cols);

#endif
