import sys
import numpy as np
from sklearn.metrics import silhouette_score

import symnmfmodule
from kmeans import kmeans_labels
from symnmf import error_exit, read_input_file, initialize_h, EPS, MAX_ITER, BETA


def symnmf_labels(x, k):
    """Run SymNMF and assign each point to its highest-value cluster.
    input: x - n x d data matrix, k - number of clusters
    output: integer array of cluster labels of length n"""
    w = np.array(symnmfmodule.norm(x.tolist()), dtype=np.float64)
    h_init = initialize_h(w, k)
    h_final = np.array(
        symnmfmodule.symnmf(h_init.tolist(), w.tolist(), MAX_ITER, EPS, BETA),
        dtype=np.float64,
    )
    return np.argmax(h_final, axis=1)


def main():
    """Compare SymNMF and K-Means silhouette scores and print the results.
    input: none
    output: none"""
    np.random.seed(1234)
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
        nmf_score = silhouette_score(x, symnmf_labels(x, k))
        kmeans_score = silhouette_score(x, kmeans_labels(x, k, MAX_ITER, EPS))
        print(f"nmf: {nmf_score:.4f}")
        print(f"kmeans: {kmeans_score:.4f}")
    except Exception:
        error_exit()


if __name__ == "__main__":
    main()
