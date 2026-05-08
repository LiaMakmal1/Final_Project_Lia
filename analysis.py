"""
analysis.py -- Compare SymNMF and K-Means clustering via silhouette score.

Usage:
    python3 analysis.py <k> <file_name>

Arguments:
    k          Number of clusters (integer, 1 <= k < N)
    file_name  Path to a comma-separated .txt file with N data points

Output (one line each, 4 decimal places):
    nmf: <score>
    kmeans: <score>

The silhouette score is computed with sklearn and measures cluster quality:
a higher value (closer to 1) means better-separated, well-defined clusters.
"""

import sys
import numpy as np
from sklearn.metrics import silhouette_score

import symnmfmodule

# Convergence parameters -- must match the values used in symnmf.py.
EPS      = 1e-4
MAX_ITER = 300
BETA     = 0.5


def error_exit():
    """Print the required error message and terminate with exit code 1."""
    print("An Error Has Occurred")
    sys.exit(1)


def read_input_file(file_name):
    """Parse a comma-separated .txt file into a 2-D float64 NumPy array.

    Blank lines are silently skipped.  Any I/O or parse error terminates
    the program via error_exit().
    """
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
    """Randomly initialise H with values drawn from U(0, 2*sqrt(mean(W)/k)).

    The seed is reset here so that analysis runs are reproducible regardless
    of any prior numpy random state.

    Args:
        w: Normalised similarity matrix W (n x n float64 array).
        k: Number of clusters / columns in H.

    Returns:
        H0: Initial matrix of shape (n, k), dtype float64.
    """
    np.random.seed(1234)
    m     = float(np.mean(w))
    upper = 2.0 * np.sqrt(m / k)
    return np.random.uniform(0.0, upper, size=(w.shape[0], k)).astype(np.float64)


def symnmf_labels(x, k):
    """Run SymNMF on x and return a hard cluster-label array.

    The C extension computes W and iterates the multiplicative update rule.
    Hard assignments are derived by taking the column with the highest
    association score in each row of the converged H matrix (argmax).

    Args:
        x: Data matrix (n x d float64 array).
        k: Number of clusters.

    Returns:
        labels: Integer array of length n with values in [0, k-1].
    """
    w       = np.array(symnmfmodule.norm(x.tolist()), dtype=np.float64)
    h_init  = initialize_h(w, k)
    h_final = np.array(
        symnmfmodule.symnmf(h_init.tolist(), w.tolist(), MAX_ITER, EPS, BETA),
        dtype=np.float64,
    )
    # Each row of H is an association vector; the highest entry picks the cluster.
    return np.argmax(h_final, axis=1)


# ---------------------------------------------------------------------------
# K-Means helpers
# ---------------------------------------------------------------------------

def assign_labels(x, centroids):
    """Return the index of the nearest centroid for every data point.

    Uses vectorised NumPy broadcasting to avoid a Python-level loop over
    points, making it practical for large datasets.

    Args:
        x:         Data matrix (n x d).
        centroids: Current centroid positions (k x d).

    Returns:
        labels: Integer array of length n in [0, k-1].
    """
    diffs    = x[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    sq_dists = np.einsum("ijk,ijk->ij", diffs, diffs)
    return np.argmin(sq_dists, axis=1)


def update_centroids(x, labels, k, old_centroids):
    """Recompute each centroid as the mean of its assigned points.

    If a cluster becomes empty its centroid is left unchanged, which keeps
    the number of clusters fixed at k throughout the run.

    Args:
        x:            Data matrix (n x d).
        labels:       Current cluster assignments (length n).
        k:            Number of clusters.
        old_centroids: Previous centroid positions (k x d), used as fallback.

    Returns:
        new_centroids: Updated centroid positions (k x d).
    """
    new_centroids = old_centroids.copy()
    for j in range(k):
        members = x[labels == j]
        if len(members) > 0:
            new_centroids[j] = members.mean(axis=0)
    return new_centroids


def has_converged(centroids, new_centroids, eps):
    """Return True when every centroid moved by less than eps in Euclidean distance."""
    shifts = np.sqrt(np.sum((centroids - new_centroids) ** 2, axis=1))
    return bool(np.all(shifts < eps))


def kmeans_labels(x, k, max_iter=MAX_ITER, eps=EPS):
    """Run K-Means and return a cluster-label array.

    Centroids are seeded from the first k data points (deterministic,
    matching the HW1 specification).  Each iteration reassigns points and
    recomputes centroids; the loop stops when centroids stop moving or
    max_iter is reached.

    Args:
        x:        Data matrix (n x d).
        k:        Number of clusters.
        max_iter: Maximum number of iterations (default 300).
        eps:      Convergence threshold for centroid shift (default 1e-4).

    Returns:
        labels: Integer array of length n with values in [0, k-1].
    """
    centroids = x[:k].copy()

    for _ in range(max_iter):
        labels        = assign_labels(x, centroids)
        new_centroids = update_centroids(x, labels, k, centroids)

        if has_converged(centroids, new_centroids, eps):
            centroids = new_centroids
            break

        centroids = new_centroids

    # Final assignment uses the last converged centroid positions.
    return assign_labels(x, centroids)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Parse CLI arguments, run both methods, and print silhouette scores."""
    if len(sys.argv) != 3:
        error_exit()

    try:
        k = int(sys.argv[1])
    except Exception:
        error_exit()

    file_name = sys.argv[2]
    x         = read_input_file(file_name)

    n = x.shape[0]
    if k <= 0 or k >= n:
        error_exit()

    try:
        nmf_assignments    = symnmf_labels(x, k)
        kmeans_assignments = kmeans_labels(x, k, MAX_ITER, EPS)

        nmf_score    = silhouette_score(x, nmf_assignments)
        kmeans_score = silhouette_score(x, kmeans_assignments)

        print(f"nmf: {nmf_score:.4f}")
        print(f"kmeans: {kmeans_score:.4f}")

    except Exception:
        error_exit()


if __name__ == "__main__":
    main()
