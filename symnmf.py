import sys
import numpy as np

import symnmfmodule as symnmf_c

EPS = 1e-4
MAX_ITER = 300
BETA = 0.5

np.random.seed(1234)


def error_exit():
    """Prints the error message and terminates the program.
    input: none
    output: none"""
    print("An Error Has Occurred")
    sys.exit(1)


def parse_args():
    """Parses and validates command-line arguments.
    input: sys.argv
    output: (k, goal, file_name) tuple"""
    if len(sys.argv) != 4:
        error_exit()
    try:
        k = int(sys.argv[1])
    except Exception:
        error_exit()
    goal = sys.argv[2]
    file_name = sys.argv[3]
    if goal not in {"symnmf", "sym", "ddg", "norm"}:
        error_exit()
    return k, goal, file_name


def read_input_file(file_name):
    """Reads a comma-separated .txt file into a numpy matrix.
    input: path to .txt file
    output: data matrix as float64 numpy array (n x d)"""
    data = []
    try:
        with open(file_name, "r") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                data.append([float(x) for x in line.split(",")])
    except Exception:
        error_exit()
    if not data:
        error_exit()
    try:
        vectors = np.array(data, dtype=np.float64)
    except Exception:
        error_exit()
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        error_exit()
    return vectors


def format_number(x):
    """Formats a float to 4 decimal places, avoiding -0.0000.
    input: float x
    output: formatted string"""
    s = f"{x:.4f}"
    return "0.0000" if s == "-0.0000" else s


def print_matrix(mat):
    """Prints a 2D array to stdout, one comma-separated row per line.
    input: 2D array
    output: none"""
    for row in mat:
        print(",".join(format_number(val) for val in row))


def initialize_h(w, k):
    """Randomly initializes H from [0, 2*sqrt(mean(W)/k)].
    input: W matrix (n x n), number of clusters k
    output: H matrix (n x k)"""
    n = w.shape[0]
    m = float(np.mean(w))
    upper = 2.0 * np.sqrt(m / k)
    return np.random.uniform(0.0, upper, size=(n, k)).astype(np.float64)


def main():
    """Runs the requested goal and prints the result matrix.
    input: command-line arguments (k, goal, file_name)
    output: none"""
    k, goal, file_name = parse_args()
    vectors = read_input_file(file_name)

    n = vectors.shape[0]
    if k <= 0 or k >= n:
        error_exit()

    try:
        if goal == "sym":
            result = np.array(symnmf_c.sym(vectors.tolist()), dtype=np.float64)

        elif goal == "ddg":
            result = np.array(symnmf_c.ddg(vectors.tolist()), dtype=np.float64)

        elif goal == "norm":
            result = np.array(symnmf_c.norm(vectors.tolist()), dtype=np.float64)

        else:
            w = np.array(symnmf_c.norm(vectors.tolist()), dtype=np.float64)
            h_init = initialize_h(w, k)
            result = np.array(
                symnmf_c.symnmf(h_init.tolist(), w.tolist(), MAX_ITER, EPS, BETA),
                dtype=np.float64
            )

        print_matrix(result)

    except Exception:
        error_exit()


if __name__ == "__main__":
    main()
