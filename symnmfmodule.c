/*
 * symnmfmodule.c  --  Python C extension that exposes the SymNMF operations.
 *
 * Registered functions (callable from Python as symnmfmodule.<name>):
 *   sym    (X)               -> similarity matrix A
 *   ddg    (X)               -> diagonal degree matrix D
 *   norm   (X)               -> normalised similarity matrix W
 *   symnmf (H, W, max_iter,
 *           eps, beta)       -> optimised factor matrix H
 *
 * All functions accept and return Python lists-of-lists of floats.
 * On any error a RuntimeError is raised with the message
 * "An Error Has Occurred".
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/* free_partial_matrix: thin wrapper so call-sites read naturally. */
static void free_partial_matrix(double** mat, int rows_allocated) {
    free_matrix(mat, rows_allocated);
}

/*
 * get_matrix_dims: read row and column counts from a Python list-of-lists.
 * Returns 1 on success, 0 if the object is not a non-empty 2-D list.
 */
static int get_matrix_dims(PyObject* py_mat, int* out_rows, int* out_cols) {
    Py_ssize_t rows_ssize, cols_ssize;
    PyObject* first_row;

    if (!PyList_Check(py_mat)) {
        return 0;
    }
    rows_ssize = PyList_Size(py_mat);
    if (rows_ssize <= 0) {
        return 0;
    }
    first_row = PyList_GetItem(py_mat, 0);
    if (first_row == NULL || !PyList_Check(first_row)) {
        return 0;
    }
    cols_ssize = PyList_Size(first_row);
    if (cols_ssize <= 0) {
        return 0;
    }
    *out_rows = (int)rows_ssize;
    *out_cols = (int)cols_ssize;
    return 1;
}

/*
 * fill_c_matrix_row: copy one Python list row into a C matrix row.
 * Returns 1 on success, 0 if any element cannot be read as a double.
 */
static int fill_c_matrix_row(double** mat, PyObject* row_obj,
    int row, Py_ssize_t cols) {
    Py_ssize_t j;
    PyObject* item;
    double value;

    for (j = 0; j < cols; j++) {
        item = PyList_GetItem(row_obj, j);
        if (item == NULL) {
            return 0;
        }
        value = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            return 0;
        }
        mat[row][j] = value;
    }
    return 1;
}

/*
 * py_to_c_matrix: convert a Python list-of-lists to a heap-allocated C matrix.
 *
 * Returns NULL if the input is malformed or if memory allocation fails.
 * On success the caller is responsible for calling free_matrix on the result.
 */
static double** py_to_c_matrix(PyObject* py_mat, int* out_rows, int* out_cols) {
    double** mat;
    Py_ssize_t i, rows_ssize, cols_ssize;
    PyObject* row_obj;

    if (!get_matrix_dims(py_mat, out_rows, out_cols)) {
        return NULL;
    }
    rows_ssize = (Py_ssize_t)(*out_rows);
    cols_ssize = (Py_ssize_t)(*out_cols);

    mat = alloc_matrix(*out_rows, *out_cols);
    if (mat == NULL) {
        return NULL;
    }
    for (i = 0; i < rows_ssize; i++) {
        row_obj = PyList_GetItem(py_mat, i);
        if (row_obj == NULL || !PyList_Check(row_obj)
                || PyList_Size(row_obj) != cols_ssize) {
            free_partial_matrix(mat, *out_rows);
            return NULL;
        }
        if (!fill_c_matrix_row(mat, row_obj, (int)i, cols_ssize)) {
            free_partial_matrix(mat, *out_rows);
            return NULL;
        }
    }
    return mat;
}

/*
 * c_matrix_to_py: convert a C matrix to a Python list-of-lists of floats.
 *
 * Returns NULL and sets a Python exception if any allocation fails.
 * Reference counts are cleaned up on partial failure so no leak occurs.
 */
static PyObject* c_matrix_to_py(double** mat, int rows, int cols) {
    PyObject* outer;
    PyObject* row;
    PyObject* value;
    int i, j;

    outer = PyList_New(rows);
    if (outer == NULL) {
        return NULL;
    }
    for (i = 0; i < rows; i++) {
        row = PyList_New(cols);
        if (row == NULL) {
            Py_DECREF(outer);
            return NULL;
        }
        for (j = 0; j < cols; j++) {
            value = PyFloat_FromDouble(mat[i][j]);
            if (value == NULL) {
                Py_DECREF(row);
                Py_DECREF(outer);
                return NULL;
            }
            PyList_SET_ITEM(row, j, value);
        }
        PyList_SET_ITEM(outer, i, row);
    }
    return outer;
}

/* py_sym: compute and return the similarity matrix A from data points X. */
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
    X = py_to_c_matrix(py_x, &n, &d);
    if (X == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    A = compute_similarity_matrix(X, n, d);
    free_matrix(X, n);
    if (A == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    result = c_matrix_to_py(A, n, n);
    free_matrix(A, n);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    return result;
}

/* compute_ddg_matrix: helper — compute A then derive D; frees A internally. */
static double** compute_ddg_matrix(double** X, int n, int d) {
    double** A;
    double** D;

    A = compute_similarity_matrix(X, n, d);
    if (A == NULL) {
        return NULL;
    }
    D = compute_diagonal_degree_matrix(A, n);
    free_matrix(A, n);
    return D;
}

/* py_ddg: compute and return the diagonal degree matrix D from data points X. */
static PyObject* py_ddg(PyObject* self, PyObject* args) {
    PyObject* py_x;
    double** X;
    double** D;
    PyObject* result;
    int n, d;

    (void)self;
    if (!PyArg_ParseTuple(args, "O", &py_x)) {
        return NULL;
    }
    X = py_to_c_matrix(py_x, &n, &d);
    if (X == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    D = compute_ddg_matrix(X, n, d);
    free_matrix(X, n);
    if (D == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    result = c_matrix_to_py(D, n, n);
    free_matrix(D, n);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    return result;
}

/* py_norm: compute and return the normalised similarity matrix W from X. */
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
    X = py_to_c_matrix(py_x, &n, &d);
    if (X == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    W = compute_normalized_similarity_matrix(X, n, d);
    free_matrix(X, n);
    if (W == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    result = c_matrix_to_py(W, n, n);
    free_matrix(W, n);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    return result;
}

/*
 * parse_h_and_w: convert Python H and W to C matrices and validate shapes.
 *
 * W must be square (n x n) and H must have exactly n rows.
 * Returns 1 on success; on failure both matrices are freed and 0 is returned.
 */
static int parse_h_and_w(PyObject* py_h, PyObject* py_w,
    double*** H_out, double*** W_out,
    int* n_h, int* k, int* n_w, int* d_w) {
    *H_out = py_to_c_matrix(py_h, n_h, k);
    if (*H_out == NULL) {
        return 0;
    }
    *W_out = py_to_c_matrix(py_w, n_w, d_w);
    if (*W_out == NULL) {
        free_matrix(*H_out, *n_h);
        return 0;
    }
    /* W must be square and H must have the same row count as W */
    if (*n_w != *d_w || *n_h != *n_w) {
        free_matrix(*H_out, *n_h);
        free_matrix(*W_out, *n_w);
        return 0;
    }
    return 1;
}

/* py_symnmf: run the full SymNMF optimisation and return the final H matrix. */
static PyObject* py_symnmf(PyObject* self, PyObject* args) {
    PyObject* py_h;
    PyObject* py_w;
    double** H;
    double** W;
    double** H_final;
    PyObject* result;
    int n_h, k, n_w, d_w, max_iter;
    double eps, beta;

    (void)self;
    if (!PyArg_ParseTuple(args, "OOidd", &py_h, &py_w, &max_iter, &eps, &beta)) {
        return NULL;
    }
    if (!parse_h_and_w(py_h, py_w, &H, &W, &n_h, &k, &n_w, &d_w)) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    H_final = symnmf_optimize(H, W, n_h, k, max_iter, eps, beta);
    free_matrix(H, n_h);
    free_matrix(W, n_w);
    if (H_final == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    result = c_matrix_to_py(H_final, n_h, k);
    free_matrix(H_final, n_h);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    return result;
}

static PyMethodDef SymNMFMethods[] = {
    {"sym",    (PyCFunction)py_sym,    METH_VARARGS,
     PyDoc_STR("sym(X) -> similarity matrix A")},
    {"ddg",    (PyCFunction)py_ddg,    METH_VARARGS,
     PyDoc_STR("ddg(X) -> diagonal degree matrix D")},
    {"norm",   (PyCFunction)py_norm,   METH_VARARGS,
     PyDoc_STR("norm(X) -> normalised similarity matrix W")},
    {"symnmf", (PyCFunction)py_symnmf, METH_VARARGS,
     PyDoc_STR("symnmf(H, W, max_iter, eps, beta) -> optimised H")},
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

PyMODINIT_FUNC PyInit_symnmfmodule(void) {
    return PyModule_Create(&symnmfmodule);
}
