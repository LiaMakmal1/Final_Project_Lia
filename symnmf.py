import sys
import numpy as np

import symnmfmodule as symnmf_c

EPS      = 1e-4
MAX_ITER = 300
BETA     = 0.5

# set seed once at startup so H initialization is reproducible
np.random.seed(1234)


def error_exit():
    """input: none
    output: none - prints error and exits"""
    print("An Error Has Occurred")
    sys.exit(1)


def parse_args():
    """input: sys.argv
    output: (k, goal, file_name) tuple"""
    if len(sys.argv) != 4:
        error_exit()
    try:
        k = int(sys.argv[1])
    except Exception:
        error_exit()
    goal      = sys.argv[2]
    file_name = sys.argv[3]
    if goal not in {"symnmf", "sym", "ddg", "norm"}:
        error_exit()
    return k, goal, file_name


def read_input_file(file_name):
    """input: path to a .txt file
    output: data matrix as a float64 numpy array (n x d)"""
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
    """input: a float value
    output: string with exactly 4 decimal places"""
    rounded = float(f"{x:.4f}")
    # avoid printing -0.0000
    if rounded == -0.0:
        rounded = 0.0
    return f"{rounded:.4f}"


def print_matrix(mat):
    """input: 2D array
    output: none - prints comma-separated rows to stdout"""
    for row in mat:
        print(",".join(format_number(val) for val in row))


def initialize_h(w, k):
    """input: W matrix (n x n), number of clusters k
    output: randomly initialized H matrix (n x k)"""
    n     = w.shape[0]
    m     = float(np.mean(w))
    # upper bound from the formula: 2 * sqrt(mean(W) / k)
    upper = 2.0 * np.sqrt(m / k)
    h     = np.random.uniform(0.0, upper, size=(n, k)).astype(np.float64)
    return h


def main():
    """input: command-line arguments (k, goal, file_name)
    output: none - prints the result matrix"""
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
            # compute W first, then initialize H and run the optimization
            w      = np.array(symnmf_c.norm(vectors.tolist()), dtype=np.float64)
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
