import sys
import pandas as pd
import numpy as np
import mykmeanssp



k = 0
file_name = ""
vectors = []
d = 0
N = 0


def parse_input_from_file(file_name):
    global vectors, d, N
    vectors = []
    try:
        with open(file_name, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                row = [float(x) for x in line.split(',')]
                vectors.append(row)
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)
    if not vectors:
        return 0, 0, []
    d = len(vectors[0])
    N = len(vectors)
    return N, d, vectors

def get_args():
    global k, file_name
    if len(sys.argv) != 2:
        print("An Error Has Occurred")
        sys.exit(1)

    k = int(sys.argv[1])    
    file_name = sys.argv[2]
    return k, file_name

def main():
    global k, file_name, N, d, vectors
    try:
        k, file_name = get_args()
        N, d, vectors = parse_input_from_file(file_name)
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

if __name__ == "__main__":
    main()
