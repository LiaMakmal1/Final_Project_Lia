import sys
import numpy as np
from sklearn.metrics import silhouette_score

import symnmfmodule

EPS      = 1e-4
MAX_ITER = 300
BETA     = 0.5


def error_exit():
    """input: none
    output: none - prints error and exits"""
    print("An Error Has Occurred")
    sys.exit(1)


def read_data(file_name):
    """input: path to a .txt file
    output: data matrix as a float64 numpy array (n x d)"""
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


def init_h(w, k):
    """input: W matrix (n x n), number of clusters k
    output: randomly initialized H matrix (n x k)"""
    np.random.seed(1234)
    m     = float(np.mean(w))
    upper = 2.0 * np.sqrt(m / k)
    return np.random.uniform(0.0, upper, size=(w.shape[0], k)).astype(np.float64)


def nmf_labels(x, k):
    """input: data matrix x (n x d), number of clusters k
    output: cluster label array of length n"""
    w       = np.array(symnmfmodule.norm(x.tolist()), dtype=np.float64)
    h_init  = init_h(w, k)
    h_final = np.array(
        symnmfmodule.symnmf(h_init.tolist(), w.tolist(), MAX_ITER, EPS, BETA),
        dtype=np.float64,
    )
    # each row of H is an association vector - pick the column with the highest score
    return np.argmax(h_final, axis=1)


def assign(x, centroids):
    """input: data matrix x (n x d), centroids (k x d)
    output: nearest centroid index for each point, length n"""
    diffs    = x[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    sq_dists = np.einsum("ijk,ijk->ij", diffs, diffs)
    return np.argmin(sq_dists, axis=1)


def update_cents(x, labels, k, old_centroids):
    """input: data matrix x, current labels, k, previous centroids
    output: updated centroids (k x d)"""
    new_centroids = old_centroids.copy()
    for j in range(k):
        members = x[labels == j]
        # keep old centroid if cluster is empty
        if len(members) > 0:
            new_centroids[j] = members.mean(axis=0)
    return new_centroids


def converged(centroids, new_centroids, eps):
    """input: old and new centroid arrays, threshold eps
    output: True if every centroid moved less than eps"""
    shifts = np.sqrt(np.sum((centroids - new_centroids) ** 2, axis=1))
    return bool(np.all(shifts < eps))


def km_labels(x, k, max_iter=MAX_ITER, eps=EPS):
    """input: data matrix x (n x d), k, max_iter, eps
    output: cluster label array of length n"""
    # seed from the first k points to match HW1 spec
    centroids = x[:k].copy()

    for _ in range(max_iter):
        labels        = assign(x, centroids)
        new_centroids = update_cents(x, labels, k, centroids)

        if converged(centroids, new_centroids, eps):
            centroids = new_centroids
            break

        centroids = new_centroids

    return assign(x, centroids)


def main():
    """input: sys.argv - k and file_name
    output: none - prints nmf and kmeans silhouette scores"""
    if len(sys.argv) != 3:
        error_exit()

    try:
        k = int(sys.argv[1])
    except Exception:
        error_exit()

    file_name = sys.argv[2]
    x         = read_data(file_name)

    n = x.shape[0]
    if k <= 0 or k >= n:
        error_exit()

    try:
        nmf_assignments    = nmf_labels(x, k)
        kmeans_assignments = km_labels(x, k, MAX_ITER, EPS)

        nmf_score    = silhouette_score(x, nmf_assignments)
        kmeans_score = silhouette_score(x, kmeans_assignments)

        print(f"nmf: {nmf_score:.4f}")
        print(f"kmeans: {kmeans_score:.4f}")

    except Exception:
        error_exit()


if __name__ == "__main__":
    main()
