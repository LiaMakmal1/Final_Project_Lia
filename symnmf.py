import sys
import numpy as np

k = 0
file_name = ""
vectors = None
d = 0
N = 0

def compute_similarity_matrix(vectors):
    sq_norms = np.sum(vectors ** 2, axis=1)
    sq_dist = sq_norms[:, None] + sq_norms[None, :] - 2 * vectors @ vectors.T
    A = np.exp(-sq_dist / 2)
    np.fill_diagonal(A, 0.0)
    return A

def parse_input_from_file(file_name):
    global vectors, d, N
    try:
        data = []
        with open(file_name, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                row = [float(x) for x in line.split(',')]
                data.append(row)

        if not data:
            return 0, 0, np.array([])

        vectors = np.array(data, dtype=np.float64)
        # Validate consistent dimensions
        d = vectors.shape[1]
        N = vectors.shape[0]
        return N, d, vectors
    
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)


def get_args():
    global k, file_name

    if len(sys.argv) != 3:
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

    file_name = sys.argv[2]
    return k, file_name


def main():
    global k, file_name, N, d, vectors

    k, file_name = get_args()
    N, d, vectors = parse_input_from_file(file_name)
    A = compute_similarity_matrix(vectors)

if __name__ == "__main__":
    main()
