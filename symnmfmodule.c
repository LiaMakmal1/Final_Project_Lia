#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/* Converts a Python list-of-lists to a C matrix.
   Returns NULL on allocation failure so the caller can clean up.
   input: Python list-of-lists, pointers for rows and cols
   output: C matrix (caller must free), or NULL on failure */
static double** to_c_mat(PyObject* py_mat, int* out_rows, int* out_cols) {
    double** mat;
    Py_ssize_t i, j, rows, cols;
    PyObject* row_obj;

    rows = PyList_Size(py_mat);
    cols = PyList_Size(PyList_GetItem(py_mat, 0));
    *out_rows = (int)rows;
    *out_cols = (int)cols;

    mat = alloc_matrix(*out_rows, *out_cols);
    if (mat == NULL) {
        return NULL;
    }
    for (i = 0; i < rows; i++) {
        row_obj = PyList_GetItem(py_mat, i);
        for (j = 0; j < cols; j++) {
            mat[i][j] = PyFloat_AsDouble(PyList_GetItem(row_obj, j));
        }
    }
    return mat;
}

/* Converts a C matrix to a Python list-of-lists of floats.
   input: C matrix (rows x cols)
   output: Python list-of-lists */
static PyObject* to_py_mat(double** mat, int rows, int cols) {
    PyObject* outer;
    PyObject* row;
    int i, j;

    outer = PyList_New(rows);
    for (i = 0; i < rows; i++) {
        row = PyList_New(cols);
        for (j = 0; j < cols; j++) {
            PyList_SET_ITEM(row, j, PyFloat_FromDouble(mat[i][j]));
        }
        PyList_SET_ITEM(outer, i, row);
    }
    return outer;
}

/* Computes the similarity matrix from data points X.
   input: Python list-of-lists X (n x d)
   output: Python list-of-lists A (n x n) */
static PyObject* py_sym(PyObject* self, PyObject* args) {
    PyObject* py_x;
    double** X;
    double** A;
    PyObject* result;
    int n, d;

    (void)self;
    if (!PyArg_ParseTuple(args, "O", &py_x)) {
        return NULL;
    }
    X = to_c_mat(py_x, &n, &d);
    if (X == NULL) { error_exit(); }
    A = sim_mat(X, n, d);
    if (A == NULL) { free_matrix(X, n); error_exit(); }
    free_matrix(X, n);
    result = to_py_mat(A, n, n);
    free_matrix(A, n);
    return result;
}

/* Computes the diagonal degree matrix from data points X.
   input: Python list-of-lists X (n x d)
   output: Python list-of-lists D (n x n) */
static PyObject* py_ddg(PyObject* self, PyObject* args) {
    PyObject* py_x;
    double** X;
    double** A;
    double** D;
    PyObject* result;
    int n, d;

    (void)self;
    if (!PyArg_ParseTuple(args, "O", &py_x)) {
        return NULL;
    }
    X = to_c_mat(py_x, &n, &d);
    if (X == NULL) { error_exit(); }
    A = sim_mat(X, n, d);
    if (A == NULL) { free_matrix(X, n); error_exit(); }
    D = ddg_mat(A, n);
    if (D == NULL) { free_matrix(X, n); free_matrix(A, n); error_exit(); }
    free_matrix(X, n);
    free_matrix(A, n);
    result = to_py_mat(D, n, n);
    free_matrix(D, n);
    return result;
}

/* Computes the normalized similarity matrix from data points X.
   input: Python list-of-lists X (n x d)
   output: Python list-of-lists W (n x n) */
static PyObject* py_norm(PyObject* self, PyObject* args) {
    PyObject* py_x;
    double** X;
    double** W;
    PyObject* result;
    int n, d;

    (void)self;
    if (!PyArg_ParseTuple(args, "O", &py_x)) {
        return NULL;
    }
    X = to_c_mat(py_x, &n, &d);
    if (X == NULL) { error_exit(); }
    W = norm_mat(X, n, d);
    if (W == NULL) { free_matrix(X, n); error_exit(); }
    free_matrix(X, n);
    result = to_py_mat(W, n, n);
    free_matrix(W, n);
    return result;
}

/* Extracts and validates H and W from Python objects.
   Frees *H_out if *W_out allocation fails, so the caller has nothing to free on 0.
   input: Python H and W, output pointers for C matrices and dimensions
   output: 1 on success, 0 if allocation fails or shapes are inconsistent */
static int parse_hw(PyObject* py_h, PyObject* py_w,
    double*** H_out, double*** W_out,
    int* n_h, int* k, int* n_w, int* d_w) {
    *H_out = to_c_mat(py_h, n_h, k);
    if (*H_out == NULL) { return 0; }
    *W_out = to_c_mat(py_w, n_w, d_w);
    if (*W_out == NULL) { free_matrix(*H_out, *n_h); return 0; }
    /* W must be square and match the row count of H */
    if (*n_w != *d_w || *n_h != *n_w) {
        free_matrix(*H_out, *n_h);
        free_matrix(*W_out, *n_w);
        return 0;
    }
    return 1;
}

/* Runs the SymNMF optimization from an initial H and W.
   input: Python H (n x k), W (n x n), max_iter, eps, beta
   output: Python list-of-lists final H (n x k) */
static PyObject* py_symnmf(PyObject* self, PyObject* args) {
    PyObject* py_h;
    PyObject* py_w;
    double** H = NULL;
    double** W = NULL;
    double** H_final;
    PyObject* result;
    int n_h = 0, k = 0, n_w = 0, d_w = 0, max_iter;
    double eps, beta;

    (void)self;
    if (!PyArg_ParseTuple(args, "OOidd", &py_h, &py_w, &max_iter, &eps, &beta)) {
        return NULL;
    }
    if (!parse_hw(py_h, py_w, &H, &W, &n_h, &k, &n_w, &d_w)) {
        error_exit();
    }
    H_final = optimize_h(H, W, n_h, k, max_iter, eps, beta);
    if (H_final == NULL) {
        free_matrix(H, n_h);
        free_matrix(W, n_w);
        error_exit();
    }
    free_matrix(H, n_h);
    free_matrix(W, n_w);
    result = to_py_mat(H_final, n_h, k);
    free_matrix(H_final, n_h);
    return result;
}

static PyMethodDef SymNMFMethods[] = {
    {"sym",    (PyCFunction)py_sym,    METH_VARARGS,
     PyDoc_STR("sym(X) -> similarity matrix A")},
    {"ddg",    (PyCFunction)py_ddg,    METH_VARARGS,
     PyDoc_STR("ddg(X) -> diagonal degree matrix D")},
    {"norm",   (PyCFunction)py_norm,   METH_VARARGS,
     PyDoc_STR("norm(X) -> normalized similarity matrix W")},
    {"symnmf", (PyCFunction)py_symnmf, METH_VARARGS,
     PyDoc_STR("symnmf(H, W, max_iter, eps, beta) -> optimized H")},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    NULL,
    -1,
    SymNMFMethods,
    NULL, NULL, NULL, NULL
};

/* Initializes the symnmfmodule Python extension.
   input: none (called by Python on import)
   output: module object */
PyMODINIT_FUNC PyInit_symnmfmodule(void) {
    return PyModule_Create(&symnmfmodule);
}
