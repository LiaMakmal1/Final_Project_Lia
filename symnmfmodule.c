#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/* convert Python list-of-lists to a C double matrix */
static double** to_c_mat(PyObject* py_mat, int* rows, int* cols) {
    double** mat;
    int i, j;

    *rows = (int)PyList_Size(py_mat);
    *cols = (int)PyList_Size(PyList_GetItem(py_mat, 0));
    mat = alloc_matrix(*rows, *cols);
    if (mat == NULL) return NULL;
    for (i = 0; i < *rows; i++) {
        PyObject* row = PyList_GetItem(py_mat, i);
        for (j = 0; j < *cols; j++)
            mat[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
    }
    return mat;
}

/* convert C matrix to Python list-of-lists */
static PyObject* to_py_mat(double** mat, int rows, int cols) {
    PyObject* outer = PyList_New(rows);
    int i, j;

    for (i = 0; i < rows; i++) {
        PyObject* row = PyList_New(cols);
        for (j = 0; j < cols; j++)
            PyList_SET_ITEM(row, j, PyFloat_FromDouble(mat[i][j]));
        PyList_SET_ITEM(outer, i, row);
    }
    return outer;
}

static PyObject* py_sym(PyObject* self, PyObject* args) {
    PyObject* py_x;
    double** X;
    double** A;
    PyObject* result;
    int n, d;

    (void)self;
    if (!PyArg_ParseTuple(args, "O", &py_x)) return NULL;
    X = to_c_mat(py_x, &n, &d);
    if (X == NULL) error_exit();
    A = compute_similarity_matrix(X, n, d);
    free_matrix(X, n);
    if (A == NULL) error_exit();
    result = to_py_mat(A, n, n);
    free_matrix(A, n);
    return result;
}

static PyObject* py_ddg(PyObject* self, PyObject* args) {
    PyObject* py_x;
    double** X;
    double** A;
    double** D;
    PyObject* result;
    int n, d;

    (void)self;
    if (!PyArg_ParseTuple(args, "O", &py_x)) return NULL;
    X = to_c_mat(py_x, &n, &d);
    if (X == NULL) error_exit();
    A = compute_similarity_matrix(X, n, d);
    if (A == NULL) { free_matrix(X, n); error_exit(); }
    D = compute_diagonal_degree_matrix(A, n);
    free_matrix(X, n); free_matrix(A, n);
    if (D == NULL) error_exit();
    result = to_py_mat(D, n, n);
    free_matrix(D, n);
    return result;
}

static PyObject* py_norm(PyObject* self, PyObject* args) {
    PyObject* py_x;
    double** X;
    double** W;
    PyObject* result;
    int n, d;

    (void)self;
    if (!PyArg_ParseTuple(args, "O", &py_x)) return NULL;
    X = to_c_mat(py_x, &n, &d);
    if (X == NULL) error_exit();
    W = compute_normalized_similarity_matrix(X, n, d);
    free_matrix(X, n);
    if (W == NULL) error_exit();
    result = to_py_mat(W, n, n);
    free_matrix(W, n);
    return result;
}

static PyObject* py_symnmf(PyObject* self, PyObject* args) {
    PyObject *py_h, *py_w;
    double **H, **W, **H_final;
    PyObject* result;
    int n, k, nw, dw, max_iter;
    double eps, beta;

    (void)self;
    if (!PyArg_ParseTuple(args, "OOidd", &py_h, &py_w, &max_iter, &eps, &beta))
        return NULL;
    H = to_c_mat(py_h, &n, &k);
    if (H == NULL) error_exit();
    W = to_c_mat(py_w, &nw, &dw);
    if (W == NULL) { free_matrix(H, n); error_exit(); }
    H_final = symnmf_optimize(H, W, n, k, max_iter, eps, beta);
    free_matrix(H, n); free_matrix(W, nw);
    if (H_final == NULL) error_exit();
    result = to_py_mat(H_final, n, k);
    free_matrix(H_final, n);
    return result;
}

static PyMethodDef SymNMFMethods[] = {
    {"sym",    (PyCFunction)py_sym,    METH_VARARGS, "Similarity matrix"},
    {"ddg",    (PyCFunction)py_ddg,    METH_VARARGS, "Diagonal degree matrix"},
    {"norm",   (PyCFunction)py_norm,   METH_VARARGS, "Normalized similarity matrix"},
    {"symnmf", (PyCFunction)py_symnmf, METH_VARARGS, "SymNMF optimization"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT, "symnmfmodule", NULL, -1, SymNMFMethods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_symnmfmodule(void) {
    return PyModule_Create(&symnmfmodule);
}
