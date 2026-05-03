import sys
import numpy as np

import symnmfmodule as symnmf_c

EPS = 1e-4
MAX_ITER = 300
BETA = 0.5

np.random.seed(1234)


def error_exit():
    """Print error message and terminate with exit code 1."""
    print("An Error Has Occurred")
    sys.exit(1)


def parse_args():
    """Parse and validate command-line arguments; return (k, goal, file_name)."""
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
    """Read a comma-separated .txt file and return a 2-D float32 NumPy array."""
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
        vectors = np.array(data, dtype=np.float32)
    except Exception:
        error_exit()
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        error_exit()
    return vectors


def format_number(x):
    """Format a float to 4 decimal places, converting -0.0 to 0.0."""
    rounded = float(f"{x:.4f}")
    if rounded == -0.0:
        rounded = 0.0
    return f"{rounded:.4f}"


def print_matrix(mat):
    """Print a 2-D array as comma-separated rows with 4 decimal places."""
    for row in mat:
        print(",".join(format_number(val) for val in row))


def initialize_h(w, k):
    """Randomly initialize H from [0, 2*sqrt(mean(W)/k)] using the global seed."""
    n = w.shape[0]
    m = float(np.mean(w))
    upper = 2.0 * np.sqrt(m / k)
    h = np.random.uniform(0.0, upper, size=(n, k)).astype(np.float32)
    return h


def main():
    """Parse arguments, run the requested goal via the C extension, and print."""
    k, goal, file_name = parse_args()
    vectors = read_input_file(file_name)
    n = vectors.shape[0]
    if k <= 0 or k >= n:
        error_exit()
    try:
        if goal == "sym":
            result = np.array(symnmf_c.sym(vectors.tolist()), dtype=np.float32)
        elif goal == "ddg":
            result = np.array(symnmf_c.ddg(vectors.tolist()), dtype=np.float32)
        elif goal == "norm":
            result = np.array(symnmf_c.norm(vectors.tolist()), dtype=np.float32)
        else:
            w = np.array(symnmf_c.norm(vectors.tolist()), dtype=np.float32)
            h_init = initialize_h(w, k)
            result = np.array(
                symnmf_c.symnmf(h_init.tolist(), w.tolist(), MAX_ITER, EPS, BETA),
                dtype=np.float32
            )
        print_matrix(result)
    except Exception:
        error_exit()


if __name__ == "__main__":
    main()