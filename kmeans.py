import sys
import numpy as np

MAX_ITER = 300
EPS      = 1e-4


def error_exit():
    """input: none
    output: none - prints error and exits"""
    print("An Error Has Occurred")
    sys.exit(1)


def assign_labels(x, centroids):
    """input: data matrix x (n x d), centroids (k x d)
    output: nearest centroid index for each point, length n"""
    diffs    = x[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    sq_dists = np.einsum("ijk,ijk->ij", diffs, diffs)
    return np.argmin(sq_dists, axis=1)


def update_centroids(x, labels, k, old_centroids):
    """input: data matrix x, current labels, k, previous centroids
    output: updated centroids (k x d)"""
    new_centroids = old_centroids.copy()
    for j in range(k):
        members = x[labels == j]
        # keep old centroid if cluster is empty
        if len(members) > 0:
            new_centroids[j] = members.mean(axis=0)
    return new_centroids


def has_converged(centroids, new_centroids, eps):
    """input: old and new centroid arrays, threshold eps
    output: True if every centroid moved less than eps"""
    shifts = np.sqrt(np.sum((centroids - new_centroids) ** 2, axis=1))
    return bool(np.all(shifts < eps))


def kmeans_labels(x, k, max_iter=MAX_ITER, eps=EPS):
    """input: data matrix x (n x d), k, max_iter, eps
    output: cluster label array of length n"""
    # seed from the first k points to match HW1 spec
    centroids = x[:k].copy()

    for _ in range(max_iter):
        labels        = assign_labels(x, centroids)
        new_centroids = update_centroids(x, labels, k, centroids)

        if has_converged(centroids, new_centroids, eps):
            centroids = new_centroids
            break

        centroids = new_centroids

    return assign_labels(x, centroids)
