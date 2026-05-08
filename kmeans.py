import sys
import math

MAX_ITER = 300
EPS      = 1e-4


def error_exit():
    """input: none
    output: none - prints error and exits"""
    print("An Error Has Occurred")
    sys.exit(1)


def calc_euclidean_distance(u, v):
    """input: two equal-length lists u, v
    output: Euclidean distance between them"""
    d = len(u)
    squared_sum = 0.0
    for i in range(d):
        squared_sum += (u[i] - v[i]) ** 2
    return math.sqrt(squared_sum)


def find_cluster(vector, centroids):
    """input: a single data point vector, list of centroid vectors
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
    """input: data matrix x (n x d numpy array), k, max_iter, eps
    output: cluster label array of length n"""
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
