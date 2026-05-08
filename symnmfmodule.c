#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/* input: matrix pointer, number of allocated rows
   output: none - frees the matrix */
static void free_mat(double** mat, int rows) {
    free_matrix(mat, rows);
}

/* input: Python list-of-lists
   output: 1 on success and fills *out_rows and *out_cols, 0 if invalid */
static int mat_dims(PyObject* py_mat, int* out_rows, int* out_cols) {
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

/* input: C matrix, Python row object, row index, number of cols
   output: 1 on success, 0 if any element can't be read as a double */
static int fill_row(double** mat, PyObject* row_obj,
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

/* input: Python list-of-lists
   output: C matrix (caller must free), sets *out_rows and *out_cols */
static double** to_c_mat(PyObject* py_mat, int* out_rows, int* out_cols) {
    double** mat;
    Py_ssize_t i, rows_ssize, cols_ssize;
    PyObject* row_obj;

    if (!mat_dims(py_mat, out_rows, out_cols)) {
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
            free_mat(mat, *out_rows);
            return NULL;
        }
        if (!fill_row(mat, row_obj, (int)i, cols_ssize)) {
            free_mat(mat, *out_rows);
            return NULL;
        }
    }
    return mat;
}

/* input: C matrix (rows x cols)
   output: Python list-of-lists of floats */
static PyObject* to_py_mat(double** mat, int rows, int cols) {
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
                /* clean up both lists before bailing out */
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

/* input: Python list-of-lists X (n x d)
   output: Python list-of-lists similarity matrix A (n x n) */
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
    if (X == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    A = sim_mat(X, n, d);
    free_matrix(X, n);
    if (A == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    result = to_py_mat(A, n, n);
    free_matrix(A, n);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    return result;
}

/* input: data matrix X (n x d), n, d
   output: diagonal degree matrix D (n x n) */
static double** ddg_helper(double** X, int n, int d) {
    double** A;
    double** D;

    A = sim_mat(X, n, d);
    if (A == NULL) {
        return NULL;
    }
    D = ddg_mat(A, n);
    free_matrix(A, n);
    return D;
}

/* input: Python list-of-lists X (n x d)
   output: Python list-of-lists diagonal degree matrix D (n x n) */
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
    X = to_c_mat(py_x, &n, &d);
    if (X == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    D = ddg_helper(X, n, d);
    free_matrix(X, n);
    if (D == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    result = to_py_mat(D, n, n);
    free_matrix(D, n);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    return result;
}

/* input: Python list-of-lists X (n x d)
   output: Python list-of-lists normalized similarity matrix W (n x n) */
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
    if (X == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    W = norm_mat(X, n, d);
    free_matrix(X, n);
    if (W == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    result = to_py_mat(W, n, n);
    free_matrix(W, n);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    return result;
}

/* input: Python H and W, output pointers for C matrices, dimension outputs
   output: 1 on success, 0 if shapes are wrong or allocation fails */
static int parse_hw(PyObject* py_h, PyObject* py_w,
    double*** H_out, double*** W_out,
    int* n_h, int* k, int* n_w, int* d_w) {
    *H_out = to_c_mat(py_h, n_h, k);
    if (*H_out == NULL) {
        return 0;
    }
    *W_out = to_c_mat(py_w, n_w, d_w);
    if (*W_out == NULL) {
        free_matrix(*H_out, *n_h);
        return 0;
    }
    /* W must be square and H must have the same number of rows */
    if (*n_w != *d_w || *n_h != *n_w) {
        free_matrix(*H_out, *n_h);
        free_matrix(*W_out, *n_w);
        return 0;
    }
    return 1;
}

/* input: Python H (n x k), W (n x n), max_iter, eps, beta
   output: Python list-of-lists final H (n x k) */
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
    if (!parse_hw(py_h, py_w, &H, &W, &n_h, &k, &n_w, &d_w)) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    H_final = optimize_h(H, W, n_h, k, max_iter, eps, beta);
    free_matrix(H, n_h);
    free_matrix(W, n_w);
    if (H_final == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    result = to_py_mat(H_final, n_h, k);
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

PyMODINIT_FUNC PyInit_symnmfmodule(void) {
    return PyModule_Create(&symnmfmodule);
}
