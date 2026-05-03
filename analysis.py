import sys
import numpy as np
from sklearn.metrics import silhouette_score

import symnmfmodule

EPS = 1e-4
MAX_ITER = 300
BETA = 0.5


def error_exit():
    """Print error message and terminate."""
    print("An Error Has Occurred")
    sys.exit(1)


def read_input_file(file_name):
    """Parse a CSV-formatted .txt file into a 2-D float64 NumPy array."""
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
    """Randomly initialize H in [0, 2*sqrt(mean(W)/k)] with seed 1234."""
    np.random.seed(1234)
    m = float(np.mean(w))
    upper = 2.0 * np.sqrt(m / k)
    return np.random.uniform(0.0, upper, size=(w.shape[0], k)).astype(np.float64)


def symnmf_labels(x, k):
    """Run symNMF via the C extension and return hard cluster assignments."""
    w = np.array(symnmfmodule.norm(x.tolist()), dtype=np.float64)
    h_init = initialize_h(w, k)
    h_final = np.array(
        symnmfmodule.symnmf(h_init.tolist(), w.tolist(), MAX_ITER, EPS, BETA),
        dtype=np.float64,
    )
    return np.argmax(h_final, axis=1)


# ---------------------------------------------------------------------------
# K-Means helpers
# ---------------------------------------------------------------------------

def assign_labels(x, centroids):
    """Return nearest-centroid index for every data point.

    Vectorised via NumPy broadcasting: O(n*k*d) in compiled C, not Python.
    x : (n, d),  centroids : (k, d)  →  labels : (n,)
    """
    diffs = x[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    sq_dists = np.einsum("ijk,ijk->ij", diffs, diffs)
    return np.argmin(sq_dists, axis=1)


def update_centroids(x, labels, k, old_centroids):
    """Recompute centroid positions; keeps old position for empty clusters.

    x : (n, d),  labels : (n,),  old_centroids : (k, d)  →  (k, d)
    """
    new_centroids = old_centroids.copy()
    for j in range(k):
        members = x[labels == j]
        if len(members) > 0:
            new_centroids[j] = members.mean(axis=0)
    return new_centroids


def has_converged(centroids, new_centroids, eps):
    """Return True when every centroid moved by less than *eps*."""
    shifts = np.sqrt(np.sum((centroids - new_centroids) ** 2, axis=1))
    return bool(np.all(shifts < eps))


def kmeans_labels(x, k, max_iter=MAX_ITER, eps=EPS):
    """Run K-Means and return cluster-label array.

    Centroids are seeded from the first k data points.
    All distance computation is vectorised for large-dataset performance.

    Parameters
    ----------
    x        : ndarray (n, d)
    k        : int – number of clusters
    max_iter : int – iteration cap (default MAX_ITER = 300)
    eps      : float – convergence threshold (default EPS = 1e-4)
    """
    centroids = x[:k].copy()

    for _ in range(max_iter):
        labels = assign_labels(x, centroids)
        new_centroids = update_centroids(x, labels, k, centroids)

        if has_converged(centroids, new_centroids, eps):
            centroids = new_centroids
            break

        centroids = new_centroids

    return assign_labels(x, centroids)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Parse CLI args, run both clustering methods, and print scores."""
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