import sys
import numpy as np
from sklearn.metrics import silhouette_score

import symnmfmodule

EPS = 1e-4
MAX_ITER = 300
BETA = 0.5


def error_exit():
    print("An Error Has Occurred")
    sys.exit(1)


def read_input_file(file_name):
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
        x = np.array(data, dtype=np.float32)
    except Exception:
        error_exit()

    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        error_exit()

    return x


def initialize_h(w, k):
    np.random.seed(1234)
    m = float(np.mean(w))
    upper = 2.0 * np.sqrt(m / k)
    return np.random.uniform(0.0, upper, size=(w.shape[0], k)).astype(np.float32)


def symnmf_labels(x, k):
    w = np.array(symnmfmodule.norm(x.tolist()), dtype=np.float32)
    h_init = initialize_h(w, k)
    h_final = np.array(
        symnmfmodule.symnmf(h_init.tolist(), w.tolist(), MAX_ITER, EPS, BETA),
        dtype=np.float32
    )
    return np.argmax(h_final, axis=1)


def euclidean_distance(u, v):
    diff = u - v
    return float(np.sqrt(np.sum(diff * diff)))


def kmeans_labels(x, k, max_iter=MAX_ITER, eps=EPS):
    n = x.shape[0]
    centroids = x[:k].copy()

    for _ in range(max_iter):
        labels = np.empty(n, dtype=int)

        for i in range(n):
            best_cluster = 0
            best_dist = euclidean_distance(x[i], centroids[0])

            for j in range(1, k):
                curr_dist = euclidean_distance(x[i], centroids[j])
                if curr_dist < best_dist:
                    best_dist = curr_dist
                    best_cluster = j

            labels[i] = best_cluster

        new_centroids = centroids.copy()

        for j in range(k):
            cluster_points = x[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = np.mean(cluster_points, axis=0)

        stop = True
        for j in range(k):
            if euclidean_distance(centroids[j], new_centroids[j]) >= eps:
                stop = False
                break

        centroids = new_centroids

        if stop:
            break

    final_labels = np.empty(n, dtype=int)
    for i in range(n):
        best_cluster = 0
        best_dist = euclidean_distance(x[i], centroids[0])

        for j in range(1, k):
            curr_dist = euclidean_distance(x[i], centroids[j])
            if curr_dist < best_dist:
                best_dist = curr_dist
                best_cluster = j

        final_labels[i] = best_cluster

    return final_labels


def main():
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