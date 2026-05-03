#!/usr/bin/env python3
"""
============================================================
  COMPREHENSIVE TEST SUITE — symNMF Project
  Tests analysis.py, symnmf.py, C extension
  Based strictly on PDF specifications
============================================================
"""

import subprocess
import tempfile
import os
import sys
import math
import csv
import numpy as np

# ── Terminal colours ────────────────────────────────────────
G  = "\033[92m"   # green
R  = "\033[91m"   # red
Y  = "\033[93m"   # yellow
B  = "\033[1m"    # bold
D  = "\033[2m"    # dim
RS = "\033[0m"    # reset

PASS = f"{G}[PASS]{RS}"
FAIL = f"{R}[FAIL]{RS}"
HEAD = f"{B}{{title}}{RS}"

results = []


# ── Helpers ─────────────────────────────────────────────────

def run(cmd, timeout=60):
    """Run a subprocess and return (stdout, stderr, returncode)."""
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.stdout, r.stderr, r.returncode


def make_file(content, suffix=".txt"):
    """Write content to a temp file; caller must delete it."""
    f = tempfile.NamedTemporaryFile(mode='w', suffix=suffix,
                                    delete=False, dir='.')
    f.write(content)
    f.close()
    return f.name


def record(name, passed, note=""):
    results.append((name, passed, note))
    icon = PASS if passed else FAIL
    note_str = f"  {D}{note}{RS}" if note else ""
    print(f"  {icon} {name}{note_str}")


def section(title):
    bar = "─" * 56
    print(f"\n{B}{bar}{RS}")
    print(f"{B}  {title}{RS}")
    print(f"{B}{bar}{RS}")


# ═══════════════════════════════════════════════════════════
#  GROUP 1 — symnmf.py  (PDF §2.1)
# ═══════════════════════════════════════════════════════════
section("GROUP 1 · symnmf.py — argument parsing")

# 1.1 Too few args
o, e, rc = run(["python3", "symnmf.py"])
record("1.1 no args → error message + exit 1",
       rc == 1 and "An Error Has Occurred" in o)

# 1.2 Too many args
o, e, rc = run(["python3", "symnmf.py", "2", "sym", "f.txt", "extra"])
record("1.2 too many args → error",
       rc == 1 and "An Error Has Occurred" in o)

# 1.3 k not integer
f = make_file("1.0,2.0\n3.0,4.0\n5.0,6.0\n")
o, e, rc = run(["python3", "symnmf.py", "abc", "sym", f])
record("1.3 k=abc (not int) → error", rc == 1 and "An Error Has Occurred" in o)
os.unlink(f)

# 1.4 invalid goal
f = make_file("1.0,2.0\n3.0,4.0\n5.0,6.0\n")
o, e, rc = run(["python3", "symnmf.py", "2", "bad_goal", f])
record("1.4 invalid goal → error", rc == 1 and "An Error Has Occurred" in o)
os.unlink(f)

# 1.5 k=0 boundary
f = make_file("1.0,2.0\n3.0,4.0\n5.0,6.0\n")
o, e, rc = run(["python3", "symnmf.py", "0", "sym", f])
record("1.5 k=0 → error", rc == 1 and "An Error Has Occurred" in o)
os.unlink(f)

# 1.6 k >= n boundary (k==n)
f = make_file("1.0,2.0\n3.0,4.0\n5.0,6.0\n")
o, e, rc = run(["python3", "symnmf.py", "3", "sym", f])
record("1.6 k=n → error (PDF: k < N)", rc == 1 and "An Error Has Occurred" in o)
os.unlink(f)

# 1.7 file does not exist
o, e, rc = run(["python3", "symnmf.py", "2", "sym", "no_such_file.txt"])
record("1.7 missing file → error", rc == 1 and "An Error Has Occurred" in o)

# 1.8 empty file
f = make_file("")
o, e, rc = run(["python3", "symnmf.py", "2", "sym", f])
record("1.8 empty file → error", rc == 1 and "An Error Has Occurred" in o)
os.unlink(f)

# ═══════════════════════════════════════════════════════════
#  GROUP 2 — symnmf.py output format  (PDF §2.9)
# ═══════════════════════════════════════════════════════════
section("GROUP 2 · symnmf.py — output format (PDF §2.9)")

SIMPLE = "1.0,0.0\n0.0,1.0\n0.5,0.5\n"

# 2.1 sym produces n×n output
f = make_file(SIMPLE)
o, e, rc = run(["python3", "symnmf.py", "2", "sym", f])
lines = [l for l in o.strip().split("\n") if l]
record("2.1 sym: n×n matrix (3 rows)", rc == 0 and len(lines) == 3)
record("2.2 sym: each row has n values",
       rc == 0 and all(len(l.split(",")) == 3 for l in lines))
os.unlink(f)

# 2.3 diagonal of sym is 0.0000 (PDF §1.1: a_ii = 0)
f = make_file(SIMPLE)
o, e, rc = run(["python3", "symnmf.py", "2", "sym", f])
rows = [l.split(",") for l in o.strip().split("\n") if l]
diag_zero = all(rows[i][i].strip() == "0.0000" for i in range(len(rows)))
record("2.3 sym: diagonal entries = 0.0000", rc == 0 and diag_zero)
os.unlink(f)

# 2.4 exactly 4 decimal places
f = make_file(SIMPLE)
o, e, rc = run(["python3", "symnmf.py", "2", "sym", f])
import re
vals = re.findall(r'[\d\-]+\.\d+', o)
four_dp = all(len(v.split(".")[1]) == 4 for v in vals)
record("2.4 all values have exactly 4 decimal places", rc == 0 and four_dp)
os.unlink(f)

# 2.5 sym is symmetric (A[i][j] == A[j][i])
f = make_file(SIMPLE)
o, e, rc = run(["python3", "symnmf.py", "2", "sym", f])
mat = [[float(x) for x in l.split(",")] for l in o.strip().split("\n") if l]
symm = all(abs(mat[i][j] - mat[j][i]) < 1e-9
           for i in range(len(mat)) for j in range(len(mat)))
record("2.5 sym: matrix is symmetric A[i][j]==A[j][i]", rc == 0 and symm)
os.unlink(f)

# 2.6 ddg diagonal only (off-diag = 0)
f = make_file(SIMPLE)
o, e, rc = run(["python3", "symnmf.py", "2", "ddg", f])
mat = [[float(x) for x in l.split(",")] for l in o.strip().split("\n") if l]
n = len(mat)
off_zero = all(abs(mat[i][j]) < 1e-9 for i in range(n)
               for j in range(n) if i != j)
record("2.6 ddg: off-diagonal entries = 0", rc == 0 and off_zero)
os.unlink(f)

# 2.7 ddg diagonal = row-sum of similarity matrix
f = make_file(SIMPLE)
o_ddg, _, rc_ddg = run(["python3", "symnmf.py", "2", "ddg", f])
o_sym, _, rc_sym = run(["python3", "symnmf.py", "2", "sym", f])
ddg_mat = [[float(x) for x in l.split(",")] for l in o_ddg.strip().split("\n") if l]
sym_mat = [[float(x) for x in l.split(",")] for l in o_sym.strip().split("\n") if l]
diag_ok = all(abs(ddg_mat[i][i] - sum(sym_mat[i])) < 1e-3 for i in range(len(ddg_mat)))
record("2.7 ddg diagonal = row-sum of sim matrix", rc_ddg == 0 and diag_ok)
os.unlink(f)

# 2.8 norm values in [0, 1]
f = make_file(SIMPLE)
o, e, rc = run(["python3", "symnmf.py", "2", "norm", f])
vals = [float(x) for l in o.strip().split("\n") if l for x in l.split(",")]
in_range = all(-1e-9 <= v <= 1.0 + 1e-9 for v in vals)
record("2.8 norm: all values in [0, 1]", rc == 0 and in_range)
os.unlink(f)

# 2.9 norm diagonal = 0
f = make_file(SIMPLE)
o, e, rc = run(["python3", "symnmf.py", "2", "norm", f])
mat = [[float(x) for x in l.split(",")] for l in o.strip().split("\n") if l]
diag_zero = all(abs(mat[i][i]) < 1e-9 for i in range(len(mat)))
record("2.9 norm: diagonal = 0", rc == 0 and diag_zero)
os.unlink(f)

# 2.10 -0.0000 must not appear (PDF §2.9)
f = make_file(SIMPLE)
o, e, rc = run(["python3", "symnmf.py", "2", "sym", f])
record("2.10 no '-0.0000' in output", rc == 0 and "-0.0000" not in o)
os.unlink(f)

# ═══════════════════════════════════════════════════════════
#  GROUP 3 — symnmf.py H initialization  (PDF §1.4.1)
# ═══════════════════════════════════════════════════════════
section("GROUP 3 · symnmf.py — H initialization (PDF §1.4.1)")

DATA3 = "1.0,2.0\n3.0,4.0\n5.0,6.0\n7.0,8.0\n9.0,10.0\n"

# 3.1 symnmf output shape: n rows, k cols
f = make_file(DATA3)
o, e, rc = run(["python3", "symnmf.py", "2", "symnmf", f])
rows = [l for l in o.strip().split("\n") if l]
shape_ok = (len(rows) == 5 and all(len(r.split(",")) == 2 for r in rows))
record("3.1 symnmf output shape = (n=5, k=2)", rc == 0 and shape_ok)
os.unlink(f)

# 3.2 H values non-negative (PDF §1.4: H >= 0)
f = make_file(DATA3)
o, e, rc = run(["python3", "symnmf.py", "2", "symnmf", f])
vals = [float(x) for l in o.strip().split("\n") if l for x in l.split(",")]
record("3.2 H values non-negative", rc == 0 and all(v >= -1e-9 for v in vals))
os.unlink(f)

# 3.3 reproducibility — same seed → same output
f = make_file(DATA3)
o1, _, _ = run(["python3", "symnmf.py", "2", "symnmf", f])
o2, _, _ = run(["python3", "symnmf.py", "2", "symnmf", f])
record("3.3 seed=1234 → reproducible output", o1 == o2)
os.unlink(f)

# ═══════════════════════════════════════════════════════════
#  GROUP 4 — analysis.py  (PDF §2.5)
# ═══════════════════════════════════════════════════════════
section("GROUP 4 · analysis.py — interface and output")

# 4.1 no args
o, e, rc = run(["python3", "analysis.py"])
record("4.1 no args → error", rc == 1 and "An Error Has Occurred" in o)

# 4.2 too many args
f = make_file(SIMPLE)
o, e, rc = run(["python3", "analysis.py", "2", f, "extra"])
record("4.2 too many args → error", rc == 1 and "An Error Has Occurred" in o)
os.unlink(f)

# 4.3 k not integer
f = make_file(SIMPLE)
o, e, rc = run(["python3", "analysis.py", "two", f])
record("4.3 k not integer → error", rc == 1 and "An Error Has Occurred" in o)
os.unlink(f)

# 4.4 k=0
f = make_file(SIMPLE)
o, e, rc = run(["python3", "analysis.py", "0", f])
record("4.4 k=0 → error", rc == 1 and "An Error Has Occurred" in o)
os.unlink(f)

# 4.5 k >= n
f = make_file(SIMPLE)
o, e, rc = run(["python3", "analysis.py", "3", f])
record("4.5 k=n → error", rc == 1 and "An Error Has Occurred" in o)
os.unlink(f)

# 4.6 k negative
f = make_file(SIMPLE)
o, e, rc = run(["python3", "analysis.py", "-1", f])
record("4.6 k negative → error", rc == 1 and "An Error Has Occurred" in o)
os.unlink(f)

# 4.7 missing file
o, e, rc = run(["python3", "analysis.py", "2", "ghost.txt"])
record("4.7 missing file → error", rc == 1 and "An Error Has Occurred" in o)

# 4.8 output format: starts with 'nmf:' and 'kmeans:'
f = make_file(DATA3)
o, e, rc = run(["python3", "analysis.py", "2", f], timeout=120)
lines = o.strip().split("\n")
record("4.8 output has exactly 2 lines", rc == 0 and len(lines) == 2)
record("4.9 first line starts with 'nmf:'",
       rc == 0 and lines[0].startswith("nmf:"))
record("4.10 second line starts with 'kmeans:'",
       rc == 0 and lines[1].startswith("kmeans:"))
os.unlink(f)

# 4.11 scores are parseable floats
f = make_file(DATA3)
o, e, rc = run(["python3", "analysis.py", "2", f], timeout=120)
try:
    nmf_val   = float(o.strip().split("\n")[0].split(":")[1])
    km_val    = float(o.strip().split("\n")[1].split(":")[1])
    parseable = True
except Exception:
    parseable = False
record("4.11 scores are valid floats", rc == 0 and parseable)
os.unlink(f)

# 4.12 scores in [-1, 1] (valid silhouette range)
f = make_file(DATA3)
o, e, rc = run(["python3", "analysis.py", "2", f], timeout=120)
try:
    nmf_val = float(o.strip().split("\n")[0].split(":")[1])
    km_val  = float(o.strip().split("\n")[1].split(":")[1])
    in_range = (-1.0 <= nmf_val <= 1.0) and (-1.0 <= km_val <= 1.0)
except Exception:
    in_range = False
record("4.12 silhouette scores in [-1, 1]", rc == 0 and in_range)
os.unlink(f)

# 4.13 4 decimal places in output
f = make_file(DATA3)
o, e, rc = run(["python3", "analysis.py", "2", f], timeout=120)
lines = o.strip().split("\n")
try:
    dp_ok = all(len(l.split(":")[1].strip().split(".")[1]) == 4 for l in lines)
except Exception:
    dp_ok = False
record("4.13 scores formatted to 4 decimal places", rc == 0 and dp_ok)
os.unlink(f)

# ═══════════════════════════════════════════════════════════
#  GROUP 5 — known regression values (Student's results.csv)
# ═══════════════════════════════════════════════════════════
section("GROUP 5 · regression — exact match to student results.csv")

EXPECTED = {
    ("input_1.txt", "2"):  (0.8856, 0.8856),
    ("input_1.txt", "3"):  (0.5957, 0.5957),
    ("input_1.txt", "4"):  (0.4944, 0.5154),
    ("input_2.txt", "3"):  (0.2714, 0.4789),
    ("input_2.txt", "6"):  (0.6272, 0.6820),
    ("input_3.txt", "2"):  (0.0878, 0.3128),
    ("input_3.txt", "3"):  (0.0715, 0.3395),
    ("input_3.txt", "4"):  (0.1439, 0.4091),
}

for (fname, k), (exp_nmf, exp_km) in EXPECTED.items():
    if not os.path.exists(fname):
        record(f"5.x {fname} k={k}", False, "input file missing")
        continue
    o, e, rc = run(["python3", "analysis.py", k, fname], timeout=120)
    try:
        got_nmf = float(o.strip().split("\n")[0].split(":")[1])
        got_km  = float(o.strip().split("\n")[1].split(":")[1])
        match = (got_nmf == exp_nmf and got_km == exp_km)
    except Exception:
        match = False
    record(f"5.x {fname} k={k}: nmf={exp_nmf} kmeans={exp_km}",
           match, f"got nmf={got_nmf:.4f} kmeans={got_km:.4f}" if 'got_nmf' in dir() else "parse error")

# ═══════════════════════════════════════════════════════════
#  GROUP 6 — Mathematical properties  (PDF §1.1–1.3)
# ═══════════════════════════════════════════════════════════
section("GROUP 6 · mathematical properties (PDF §1.1–1.3)")

MATH_DATA = "\n".join([
    "1.0,0.0", "0.0,1.0", "2.0,0.0", "0.0,2.0", "1.5,1.5"
]) + "\n"

f = make_file(MATH_DATA)

# 6.1 sim values in [0, 1) (exp of negative number)
o, e, rc = run(["python3", "symnmf.py", "2", "sym", f])
vals = [float(x) for l in o.strip().split("\n") if l for x in l.split(",")]
record("6.1 sim values in [0, 1)", rc == 0 and all(0.0 <= v < 1.0 + 1e-9 for v in vals))

# 6.2 sim is symmetric
mat = [[float(x) for x in l.split(",")] for l in o.strip().split("\n") if l]
symm = all(abs(mat[i][j] - mat[j][i]) < 1e-6
           for i in range(len(mat)) for j in range(len(mat)))
record("6.2 sim symmetric A[i][j] == A[j][i]", rc == 0 and symm)

# 6.3 identical points → similarity = exp(0) = 1
SAME = "1.0,1.0\n1.0,1.0\n2.0,2.0\n"
f2 = make_file(SAME)
o2, e2, rc2 = run(["python3", "symnmf.py", "2", "sym", f2])
mat2 = [[float(x) for x in l.split(",")] for l in o2.strip().split("\n") if l]
record("6.3 identical points → sim = 1.0",
       rc2 == 0 and abs(mat2[0][1] - 1.0) < 1e-3)
os.unlink(f2)

# 6.4 ddg diagonal = sum of sim row
o_sym, _, _ = run(["python3", "symnmf.py", "2", "sym", f])
o_ddg, _, _ = run(["python3", "symnmf.py", "2", "ddg", f])
sym_m = [[float(x) for x in l.split(",")] for l in o_sym.strip().split("\n") if l]
ddg_m = [[float(x) for x in l.split(",")] for l in o_ddg.strip().split("\n") if l]
ddg_ok = all(abs(ddg_m[i][i] - sum(sym_m[i])) < 1e-3 for i in range(len(ddg_m)))
record("6.4 ddg[i][i] = sum of sim row i", ddg_ok)

# 6.5 norm symmetric
o_norm, _, rc_norm = run(["python3", "symnmf.py", "2", "norm", f])
norm_m = [[float(x) for x in l.split(",")] for l in o_norm.strip().split("\n") if l]
norm_symm = all(abs(norm_m[i][j] - norm_m[j][i]) < 1e-5
                for i in range(len(norm_m)) for j in range(len(norm_m)))
record("6.5 norm symmetric W[i][j] == W[j][i]", rc_norm == 0 and norm_symm)

os.unlink(f)

# ═══════════════════════════════════════════════════════════
#  GROUP 7 — edge case inputs
# ═══════════════════════════════════════════════════════════
section("GROUP 7 · edge case inputs")

# 7.1 single dimension (d=1)
f = make_file("1.0\n2.0\n3.0\n4.0\n")
o, e, rc = run(["python3", "symnmf.py", "2", "sym", f])
record("7.1 d=1 (single dimension) works", rc == 0 and len(o.strip().split("\n")) == 4)
os.unlink(f)

# 7.2 many dimensions (d=10)
row = ",".join(["1.0"] * 10)
content = "\n".join([row] * 5) + "\n"
f = make_file(content)
o, e, rc = run(["python3", "symnmf.py", "2", "sym", f])
record("7.2 d=10 (many dimensions) works", rc == 0)
os.unlink(f)

# 7.3 minimal valid dataset (n=3, k=2)
f = make_file("1.0,2.0\n3.0,4.0\n5.0,6.0\n")
o, e, rc = run(["python3", "analysis.py", "2", f], timeout=60)
record("7.3 minimal dataset n=3, k=2 works", rc == 0)
os.unlink(f)

# 7.4 k=1 boundary (k < n satisfied, k=1 valid)
f = make_file(DATA3)
o, e, rc = run(["python3", "symnmf.py", "1", "sym", f])
record("7.4 k=1 accepted by symnmf.py (k<n)", rc == 0)
os.unlink(f)

# 7.5 file with blank lines (should be skipped)
f = make_file("1.0,2.0\n\n3.0,4.0\n\n5.0,6.0\n")
o, e, rc = run(["python3", "symnmf.py", "2", "sym", f])
record("7.5 blank lines in file → skipped, n=3", rc == 0 and len(o.strip().split("\n")) == 3)
os.unlink(f)

# 7.6 large values (no overflow)
f = make_file("1000.0,2000.0\n3000.0,4000.0\n5000.0,6000.0\n")
o, e, rc = run(["python3", "symnmf.py", "2", "sym", f])
record("7.6 large coordinate values → no crash", rc == 0)
os.unlink(f)

# 7.7 negative coordinates
f = make_file("-1.0,-2.0\n-3.0,-4.0\n-5.0,-6.0\n")
o, e, rc = run(["python3", "symnmf.py", "2", "sym", f])
record("7.7 negative coordinates → no crash", rc == 0)
os.unlink(f)

# 7.8 float k (e.g. "2.5") → error
f = make_file(SIMPLE)
o, e, rc = run(["python3", "symnmf.py", "2.5", "sym", f])
record("7.8 k=2.5 (float string) → error", rc == 1 and "An Error Has Occurred" in o)
os.unlink(f)

# 7.9 k > n
f = make_file("1.0,2.0\n3.0,4.0\n5.0,6.0\n")
o, e, rc = run(["python3", "symnmf.py", "99", "sym", f])
record("7.9 k=99 >> n → error", rc == 1 and "An Error Has Occurred" in o)
os.unlink(f)

# ═══════════════════════════════════════════════════════════
#  GROUP 8 — C executable via Makefile  (PDF §2.2, §2.7)
# ═══════════════════════════════════════════════════════════
section("GROUP 8 · C executable (make + ./symnmf)")

# 8.1 make builds without errors
o, e, rc = run(["make"], timeout=30)
record("8.1 make completes without error", rc == 0)

# 8.2 executable exists
record("8.2 ./symnmf binary exists", os.path.exists("./symnmf"))

# 8.3 C: sym output format
if os.path.exists("./symnmf") and os.path.exists("input_1.txt"):
    o, e, rc = run(["./symnmf", "sym", "input_1.txt"], timeout=30)
    lines = [l for l in o.strip().split("\n") if l]
    record("8.3 C sym: produces output", rc == 0 and len(lines) > 0)
    vals = re.findall(r'[\d\-]+\.\d+', o)
    four_dp = all(len(v.split(".")[1]) == 4 for v in vals)
    record("8.4 C sym: 4 decimal places", rc == 0 and four_dp)
else:
    record("8.3 C sym: produces output", False, "binary or input_1.txt missing")
    record("8.4 C sym: 4 decimal places", False, "binary or input_1.txt missing")

# 8.5 C: bad goal → error
if os.path.exists("./symnmf"):
    f = make_file(SIMPLE)
    o, e, rc = run(["./symnmf", "bad", f])
    record("8.5 C: bad goal → error exit", rc != 0 and "An Error Has Occurred" in o)
    os.unlink(f)
else:
    record("8.5 C: bad goal → error exit", False, "binary missing")

# 8.6 C: wrong arg count
if os.path.exists("./symnmf"):
    o, e, rc = run(["./symnmf", "sym"])
    record("8.6 C: wrong arg count → error", rc != 0 and "An Error Has Occurred" in o)
else:
    record("8.6 C: wrong arg count → error", False, "binary missing")

# 8.7 C: missing file → error
if os.path.exists("./symnmf"):
    o, e, rc = run(["./symnmf", "sym", "no_such_file.txt"])
    record("8.7 C: missing file → error", rc != 0 and "An Error Has Occurred" in o)
else:
    record("8.7 C: missing file → error", False, "binary missing")

# ═══════════════════════════════════════════════════════════
#  FINAL REPORT
# ═══════════════════════════════════════════════════════════
passed = sum(1 for _, p, _ in results if p)
failed = sum(1 for _, p, _ in results if not p)
total  = len(results)
pct    = int(100 * passed / total) if total else 0

bar = "═" * 58
print(f"\n{B}{bar}{RS}")
print(f"{B}  RESULTS{RS}")
print(f"{B}{bar}{RS}")
print(f"  {G}PASSED : {passed}/{total}{RS}")
if failed:
    print(f"  {R}FAILED : {failed}/{total}{RS}")
    print(f"\n{B}  Failed tests:{RS}")
    for name, p, note in results:
        if not p:
            print(f"    {R}✖{RS} {name}  {D}{note}{RS}")
else:
    print(f"  {G}{B}ALL TESTS PASSED ✔{RS}")
print(f"  Coverage : {pct}%")
print(f"{B}{bar}{RS}\n")
