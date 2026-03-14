#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

static void free_partial_matrix(double** mat, int rows_allocated) {
    free_matrix(mat, rows_allocated);
}

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

static double** py_to_c_matrix(PyObject* py_mat, int* out_rows, int* out_cols) {
    double** mat;
    Py_ssize_t i, j;
    Py_ssize_t rows_ssize, cols_ssize;
    PyObject* row_obj;
    PyObject* item;
    double value;

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
        if (row_obj == NULL || !PyList_Check(row_obj) || PyList_Size(row_obj) != cols_ssize) {
            free_partial_matrix(mat, *out_rows);
            return NULL;
        }

        for (j = 0; j < cols_ssize; j++) {
            item = PyList_GetItem(row_obj, j);
            if (item == NULL) {
                free_partial_matrix(mat, *out_rows);
                return NULL;
            }

            value = PyFloat_AsDouble(item);
            if (PyErr_Occurred()) {
                free_partial_matrix(mat, *out_rows);
                return NULL;
            }

            mat[i][j] = value;
        }
    }

    return mat;
}

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

    D = compute_diagonal_degree_matrix(A, n);
    free_matrix(A, n);

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

static PyObject* py_symnmf(PyObject* self, PyObject* args) {
    PyObject* py_h;
    PyObject* py_w;
    double** H;
    double** W;
    double** H_final;
    PyObject* result;
    int n_h, k, n_w, d_w;
    int max_iter;
    double eps;
    double beta;

    (void)self;

    if (!PyArg_ParseTuple(args, "OOidd", &py_h, &py_w, &max_iter, &eps, &beta)) {
        return NULL;
    }

    H = py_to_c_matrix(py_h, &n_h, &k);
    if (H == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }

    W = py_to_c_matrix(py_w, &n_w, &d_w);
    if (W == NULL) {
        free_matrix(H, n_h);
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }

    if (n_w != d_w || n_h != n_w) {
        free_matrix(H, n_h);
        free_matrix(W, n_w);
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
    {"sym", (PyCFunction)py_sym, METH_VARARGS, PyDoc_STR("Compute similarity matrix")},
    {"ddg", (PyCFunction)py_ddg, METH_VARARGS, PyDoc_STR("Compute diagonal degree matrix")},
    {"norm", (PyCFunction)py_norm, METH_VARARGS, PyDoc_STR("Compute normalized similarity matrix")},
    {"symnmf", (PyCFunction)py_symnmf, METH_VARARGS, PyDoc_STR("Optimize H for SymNMF")},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    NULL,
    -1,
    SymNMFMethods
};

PyMODINIT_FUNC PyInit_symnmfmodule(void) {
    return PyModule_Create(&symnmfmodule);
}