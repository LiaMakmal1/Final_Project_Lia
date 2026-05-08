"""
symnmf.py -- Python interface for the SymNMF clustering algorithm.

Usage:
    python3 symnmf.py <k> <goal> <file_name>

Arguments:
    k          Number of clusters (integer, 1 <= k < N)
    goal       One of: symnmf | sym | ddg | norm
    file_name  Path to a comma-separated .txt file with N data points

Goals:
    symnmf  Run the full SymNMF optimisation and print the H matrix.
    sym     Print the similarity matrix A.
    ddg     Print the diagonal degree matrix D.
    norm    Print the normalised similarity matrix W.
"""

import sys
import numpy as np

import symnmfmodule as symnmf_c

# Convergence parameters (shared with the C layer).
EPS      = 1e-4
MAX_ITER = 300
BETA     = 0.5

# Fix the random seed for reproducible H initialisation (PDF §1.4.1).
np.random.seed(1234)


def error_exit():
    """Print the required error message and terminate with exit code 1."""
    print("An Error Has Occurred")
    sys.exit(1)


def parse_args():
    """Parse and validate command-line arguments.

    Returns:
        (k, goal, file_name) -- validated tuple ready for use in main().
    """
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
    """Read a comma-separated .txt file and return a 2-D float32 NumPy array.

    Blank lines are silently skipped.  Any I/O or parse error terminates
    the program via error_exit().
    """
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
    """Format a single float to exactly 4 decimal places.

    Converts -0.0 to 0.0 so the output never contains a bare '-0.0000'.
    """
    rounded = float(f"{x:.4f}")
    if rounded == -0.0:
        rounded = 0.0
    return f"{rounded:.4f}"


def print_matrix(mat):
    """Print a 2-D array as comma-separated rows, each on its own line."""
    for row in mat:
        print(",".join(format_number(val) for val in row))


def initialize_h(w, k):
    """Randomly initialise H with values drawn from U(0, 2*sqrt(mean(W)/k)).

    The upper bound follows the formula in PDF §1.4.1.  The global seed
    set at module load time (np.random.seed(1234)) guarantees reproducibility
    across runs.

    Args:
        w: Normalised similarity matrix W as a NumPy array (n x n).
        k: Number of clusters / columns in H.

    Returns:
        H0: Initial H matrix of shape (n, k), dtype float32.
    """
    n     = w.shape[0]
    m     = float(np.mean(w))               # average entry of W
    upper = 2.0 * np.sqrt(m / k)            # upper bound of the uniform range
    h     = np.random.uniform(0.0, upper, size=(n, k)).astype(np.float32)
    return h


def main():
    """Parse arguments, delegate to the C extension, and print the result."""
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
            # goal == "symnmf": compute W first, initialise H, then optimise.
            w      = np.array(symnmf_c.norm(vectors.tolist()), dtype=np.float32)
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
