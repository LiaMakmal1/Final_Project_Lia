# SymNMF clustering: a C extension for Python

Clustering with **Symmetric Non-negative Matrix Factorization (SymNMF)**, implemented in C and exposed to Python through the Python C API, then benchmarked against K-means using the silhouette score.

Pair project for the Software Project course at Tel Aviv University.

## How it works

Given $n$ points $x_1, \dots, x_n \in \mathbb{R}^d$ and a number of clusters $k$:

1. **Similarity matrix** $A \in \mathbb{R}^{n \times n}$ with a Gaussian kernel:

$$a_{ij} = \exp\left(-\frac{\lVert x_i - x_j \rVert^2}{2}\right) \text{ for } i \neq j, \qquad a_{ii} = 0$$

2. **Degree matrix** $D = \mathrm{diag}(d_1, \dots, d_n)$ with $d_i = \sum_j a_{ij}$.

3. **Normalized similarity**:

$$W = D^{-1/2} A D^{-1/2}$$

4. **Factorization.** Find a non-negative $H \in \mathbb{R}^{n \times k}$ that minimizes $\lVert W - HH^\top \rVert_F^2$, using the multiplicative update rule with $\beta = 0.5$:

$$H \leftarrow H \odot \left(1 - \beta + \beta \cdot \frac{WH}{HH^\top H}\right)$$

The update repeats until $\lVert H_{\text{new}} - H \rVert_F^2 < 10^{-4}$ or for at most 300 iterations. $H$ starts from uniform random values in $[0,\ 2\sqrt{\mathrm{mean}(W)/k}]$ with a fixed seed, and a small $10^{-6}$ is added to the denominator for numerical stability.

5. **Assignment.** Point $i$ goes to cluster $\arg\max_j H_{ij}$.

The heavy matrix work (steps 1 to 4) runs in C. Python handles input, initialization of $H$ and the analysis.

## Results

Silhouette scores (higher is better) for SymNMF and K-means on the three sample datasets:

| Dataset | Points × dims | k | SymNMF | K-means |
|---|---|---|---|---|
| `input_1.txt` | 800 × 3 | 2 | 0.8856 | 0.8856 |
| `input_1.txt` | 800 × 3 | 3 | 0.5957 | 0.5957 |
| `input_1.txt` | 800 × 3 | 4 | 0.4944 | 0.5154 |
| `input_2.txt` | 430 × 11 | 6 | 0.6272 | 0.6820 |
| `input_2.txt` | 430 × 11 | 8 | **0.6637** | 0.6475 |
| `input_3.txt` | 5000 × 5 | 4 | 0.1439 | 0.4091 |

The full grid is in [`results.csv`](results.csv) and can be regenerated with `generate_results.py`.

**Takeaways**
- On the clearly separated structure of `input_1` with k = 2 and k = 3, both methods find the same clustering.
- Across the 21 settings in `results.csv`, K-means scores higher in 18. SymNMF comes out ahead in one setting (`input_2`, k = 8).
- The silhouette score rewards compact, well-separated clusters in Euclidean space, which is close to what K-means optimizes, so it tends to favor K-means. SymNMF clusters from a similarity graph instead, so this comparison measures one kind of cluster quality rather than every kind.

## Project structure

| File | Purpose |
|---|---|
| `symnmf.c`, `symnmf.h` | C implementation: matrix helpers, `sym` / `ddg` / `norm`, SymNMF optimization, and a standalone CLI |
| `symnmfmodule.c` | Python C API wrapper exposing `sym`, `ddg`, `norm` and `symnmf` |
| `symnmf.py` | Python CLI: reads data, initializes $H$, calls the C extension |
| `kmeans.py` | K-means baseline |
| `analysis.py` | Silhouette comparison of SymNMF and K-means |
| `generate_results.py` | Runs the analysis over several k values and datasets and writes `results.csv` |
| `test_all.py`, `test_symnmf.py` | Test suites (134 checks in total): argument errors, output format, mathematical properties of the matrices, edge cases, the C binary and regression against `check.csv` |
| `setup.py`, `Makefile`, `Dockerfile` | Build files |

## Build and run

**Requirements:** Python 3.9+, gcc, NumPy and scikit-learn.

```bash
pip install numpy scikit-learn
python3 setup.py build_ext --inplace   # builds the symnmfmodule extension
make                                   # builds the standalone C binary
```

Input files have one point per line, with comma-separated values. Outputs are printed with 4 decimal places.

```bash
# Final H matrix (n x k) for k = 2
python3 symnmf.py 2 symnmf input_1.txt

# Intermediate matrices: sym (A), ddg (D) or norm (W)
python3 symnmf.py 2 norm input_1.txt
./symnmf norm input_1.txt

# Silhouette scores: SymNMF vs K-means
python3 analysis.py 4 input_1.txt

# Tests
python3 test_all.py
```
