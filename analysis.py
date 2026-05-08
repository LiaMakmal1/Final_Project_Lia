import sys
import numpy as np
from sklearn.metrics import silhouette_score

import symnmfmodule
from kmeans import kmeans_labels

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


def read_input_file(file_name):
    """Reads a comma-separated .txt file into a numpy matrix.
    input: path to .txt file
    output: data matrix as float64 numpy array (n x d)"""
    data = []
    try:
        with open(file_name, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append([float(x) for x in line.split(",")])
    except Exception:
        error_exit()

    if not data:
        error_exit()

    try:
        x = np.array(data, dtype=np.float64)
    except Exception:
        error_exit()

    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        error_exit()

    return x


def initialize_h(w, k):
    """Randomly initializes H from [0, 2*sqrt(mean(W)/k)].
    input: W matrix (n x n), number of clusters k
    output: H matrix (n x k)"""
    m = float(np.mean(w))
    upper = 2.0 * np.sqrt(m / k)
    return np.random.uniform(0.0, upper, size=(w.shape[0], k)).astype(np.float64)


def symnmf_labels(x, k):
    """Runs SymNMF and assigns each point to its highest-scoring cluster.
    input: data matrix x (n x d), number of clusters k
    output: cluster label array of length n"""
    w = np.array(symnmfmodule.norm(x.tolist()), dtype=np.float64)
    h_init = initialize_h(w, k)
    h_final = np.array(
        symnmfmodule.symnmf(h_init.tolist(), w.tolist(), MAX_ITER, EPS, BETA),
        dtype=np.float64,
    )
    return np.argmax(h_final, axis=1)


def main():
    """Computes and prints silhouette scores for SymNMF and K-means.
    input: sys.argv - k and file_name
    output: none"""
    if len(sys.argv) != 3:
        error_exit()

    try:
        k = int(sys.argv[1])
    except Exception:
        error_exit()

    file_name = sys.argv[2]
    x = read_input_file(file_name)

    n = x.shape[0]
    if k <= 0 or k >= n:
        error_exit()

    try:
        nmf_assignments = symnmf_labels(x, k)
        kmeans_assignments = kmeans_labels(x, k, MAX_ITER, EPS)

        nmf_score = silhouette_score(x, nmf_assignments)
        kmeans_score = silhouette_score(x, kmeans_assignments)

        print(f"nmf: {nmf_score:.4f}")
        print(f"kmeans: {kmeans_score:.4f}")

    except Exception:
        error_exit()


if __name__ == "__main__":
    main()
