import sys
import math

MAX_ITER = 300
EPS      = 1e-4


def error_exit():
    """Print error message and exit.
    input: none
    output: none"""
    print("An Error Has Occurred")
    sys.exit(1)


def calc_euclidean_distance(u, v):
    """Compute the Euclidean distance between two vectors.
    input: u, v - equal-length lists
    output: Euclidean distance as float"""
    d = len(u)
    squared_sum = 0.0
    for i in range(d):
        squared_sum += (u[i] - v[i]) ** 2
    return math.sqrt(squared_sum)


def find_cluster(vector, centroids):
    """Find the nearest centroid to a given vector.
    input: vector - a data point, centroids - list of centroid vectors
    output: index of the nearest centroid"""
    min_idx = 0
    min_dist = calc_euclidean_distance(vector, centroids[0])
    for i in range(1, len(centroids)):
        dist = calc_euclidean_distance(vector, centroids[i])
        if dist < min_dist:
            min_dist = dist
            min_idx = i
    return min_idx


def kmeans_labels(x, k, max_iter=MAX_ITER, eps=EPS):
    """Run K-Means and return a cluster label for each data point.
    input: x - n x d numpy array, k, max_iter, eps
    output: list of cluster labels of length n"""
    vectors = x.tolist()
    n = len(vectors)
    d = len(vectors[0])

    centroids = [list(vectors[i]) for i in range(k)]

    for _ in range(max_iter):
        sum_cluster  = [[0.0] * d for _ in range(k)]
        cluster_size = [0] * k

        for j in range(n):
            c = find_cluster(vectors[j], centroids)
            for m in range(d):
                sum_cluster[c][m] += vectors[j][m]
            cluster_size[c] += 1

        new_centroids = []
        for c in range(k):
            if cluster_size[c] == 0:
                new_centroids.append(list(vectors[0]))
            else:
                new_centroids.append(
                    [sum_cluster[c][m] / cluster_size[c] for m in range(d)]
                )

        converged = all(
            calc_euclidean_distance(centroids[c], new_centroids[c]) < eps
            for c in range(k)
        )
        centroids = new_centroids
        if converged:
            break

    return [find_cluster(vectors[j], centroids) for j in range(n)]
