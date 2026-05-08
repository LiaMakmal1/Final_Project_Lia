import subprocess, os, re, csv, math, tempfile

G="\033[92m"; R="\033[91m"; B="\033[1m"; RS="\033[0m"
results=[]

def run(cmd, timeout=60):
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
    return r.stdout,r.stderr,r.returncode

def tmp(content):
    f=tempfile.NamedTemporaryFile(mode='w',suffix='.txt',delete=False,dir='.')
    f.write(content); f.close(); return f.name

def check(name, passed, note=""):
    results.append((name,passed,note))
    print(f"  {G}[PASS]{RS} {name}" if passed else f"  {R}[FAIL]{RS} {name}  {note}")

def section(t):
    print(f"\n{B}{'─'*60}{RS}\n{B}  {t}{RS}\n{B}{'─'*60}{RS}")

SIMPLE="1.0,0.0\n0.0,1.0\n0.5,0.5\n"
DATA5="1.0,2.0\n3.0,4.0\n5.0,6.0\n7.0,8.0\n9.0,10.0\n"

# ── 1. symnmf.py argument errors ────────────────────────────
section("1 · symnmf.py — argument errors")
o,e,rc=run(["python3","symnmf.py"])
check("no args → error+exit1", rc==1 and "An Error Has Occurred" in o)
o,e,rc=run(["python3","symnmf.py","2","sym","f.txt","extra"])
check("too many args → error", rc==1 and "An Error Has Occurred" in o)
f=tmp(SIMPLE)
o,e,rc=run(["python3","symnmf.py","abc","sym",f])
check("k=abc → error", rc==1 and "An Error Has Occurred" in o)
o,e,rc=run(["python3","symnmf.py","2","bad_goal",f])
check("invalid goal → error", rc==1 and "An Error Has Occurred" in o)
o,e,rc=run(["python3","symnmf.py","0","sym",f])
check("k=0 → error", rc==1 and "An Error Has Occurred" in o)
o,e,rc=run(["python3","symnmf.py","3","sym",f])
check("k=n → error (k must be < N)", rc==1 and "An Error Has Occurred" in o)
o,e,rc=run(["python3","symnmf.py","99","sym",f])
check("k>n → error", rc==1 and "An Error Has Occurred" in o)
o,e,rc=run(["python3","symnmf.py","2.5","sym",f])
check("k=2.5 (float) → error", rc==1 and "An Error Has Occurred" in o)
os.unlink(f)
o,e,rc=run(["python3","symnmf.py","2","sym","no_file.txt"])
check("missing file → error", rc==1 and "An Error Has Occurred" in o)
f=tmp("")
o,e,rc=run(["python3","symnmf.py","2","sym",f])
check("empty file → error", rc==1 and "An Error Has Occurred" in o)
os.unlink(f)

# ── 2. output format ─────────────────────────────────────────
section("2 · symnmf.py — output format")
f=tmp(SIMPLE)
o,e,rc=run(["python3","symnmf.py","2","sym",f])
lines=[l for l in o.strip().split("\n") if l]
check("sym: 3 rows output", rc==0 and len(lines)==3)
check("sym: 3 values per row", rc==0 and all(len(l.split(","))==3 for l in lines))
vals=re.findall(r'-?\d+\.\d+',o)
check("all values: exactly 4 decimal places", rc==0 and all(len(v.split(".")[1])==4 for v in vals))
check("no '-0.0000' in output", rc==0 and "-0.0000" not in o)
mat=[[float(x) for x in l.split(",")] for l in lines]
check("sym diagonal = 0.0000", all(abs(mat[i][i])<1e-9 for i in range(len(mat))))
check("sym is symmetric A[i][j]==A[j][i]", all(abs(mat[i][j]-mat[j][i])<1e-9 for i in range(3) for j in range(3)))
os.unlink(f)

f=tmp(SIMPLE)
o,e,rc=run(["python3","symnmf.py","2","ddg",f])
mat=[[float(x) for x in l.split(",")] for l in o.strip().split("\n") if l]
check("ddg: off-diagonal = 0", all(abs(mat[i][j])<1e-9 for i in range(3) for j in range(3) if i!=j))
check("ddg: diagonal > 0", all(mat[i][i]>0 for i in range(3)))
os.unlink(f)

f=tmp(SIMPLE)
o,e,rc=run(["python3","symnmf.py","2","norm",f])
vals=[float(x) for l in o.strip().split("\n") if l for x in l.split(",")]
check("norm: all values in [0,1]", rc==0 and all(-1e-9<=v<=1+1e-9 for v in vals))
mat=[[float(x) for x in l.split(",")] for l in o.strip().split("\n") if l]
check("norm diagonal = 0", all(abs(mat[i][i])<1e-9 for i in range(3)))
check("norm is symmetric", all(abs(mat[i][j]-mat[j][i])<1e-5 for i in range(3) for j in range(3)))
os.unlink(f)

# ── 3. symnmf H output ───────────────────────────────────────
section("3 · symnmf.py — symnmf goal")
f=tmp(DATA5)
o,e,rc=run(["python3","symnmf.py","2","symnmf",f])
rows=[l for l in o.strip().split("\n") if l]
check("symnmf: shape = (5 rows, 2 cols)", rc==0 and len(rows)==5 and all(len(r.split(","))==2 for r in rows))
vals=[float(x) for r in rows for x in r.split(",")]
check("symnmf: H values >= 0", rc==0 and all(v>=-1e-9 for v in vals))
o2,_,_=run(["python3","symnmf.py","2","symnmf",f])
check("symnmf: reproducible (seed=1234)", o==o2)
os.unlink(f)

# ── 4. mathematical properties ───────────────────────────────
section("4 · mathematical properties")
SAME="1.0,1.0\n1.0,1.0\n2.0,2.0\n"
f=tmp(SAME)
o,e,rc=run(["python3","symnmf.py","2","sym",f])
mat=[[float(x) for x in l.split(",")] for l in o.strip().split("\n") if l]
check("identical points → sim=1.0", rc==0 and abs(mat[0][1]-1.0)<1e-3)
os.unlink(f)

f=tmp(SIMPLE)
o_s,_,_=run(["python3","symnmf.py","2","sym",f])
o_d,_,_=run(["python3","symnmf.py","2","ddg",f])
sm=[[float(x) for x in l.split(",")] for l in o_s.strip().split("\n") if l]
dm=[[float(x) for x in l.split(",")] for l in o_d.strip().split("\n") if l]
check("ddg[i][i] = row-sum of sim", all(abs(dm[i][i]-sum(sm[i]))<1e-3 for i in range(3)))
os.unlink(f)

f=tmp("1.0,0.0\n0.0,1.0\n2.0,0.0\n0.0,2.0\n1.5,1.5\n")
o,e,rc=run(["python3","symnmf.py","2","sym",f])
vals=[float(x) for l in o.strip().split("\n") if l for x in l.split(",")]
check("sim values in [0,1)", rc==0 and all(0<=v<1+1e-9 for v in vals))
os.unlink(f)

# ── 5. edge cases ────────────────────────────────────────────
section("5 · edge cases")
f=tmp("1.0\n2.0\n3.0\n4.0\n")
o,e,rc=run(["python3","symnmf.py","2","sym",f])
check("d=1 (single dimension)", rc==0 and len(o.strip().split("\n"))==4)
os.unlink(f)

f=tmp("\n".join([",".join(["1.0"]*10)]*5)+"\n")
o,e,rc=run(["python3","symnmf.py","2","sym",f])
check("d=10 (many dimensions)", rc==0)
os.unlink(f)

f=tmp("1.0,2.0\n\n3.0,4.0\n\n5.0,6.0\n")
o,e,rc=run(["python3","symnmf.py","2","sym",f])
check("blank lines in file skipped → n=3", rc==0 and len([l for l in o.strip().split("\n") if l])==3)
os.unlink(f)

f=tmp("1000.0,2000.0\n3000.0,4000.0\n5000.0,6000.0\n")
o,e,rc=run(["python3","symnmf.py","2","sym",f])
check("large coordinates → no crash", rc==0)
os.unlink(f)

f=tmp("-1.0,-2.0\n-3.0,-4.0\n-5.0,-6.0\n")
o,e,rc=run(["python3","symnmf.py","2","sym",f])
check("negative coordinates → no crash", rc==0)
os.unlink(f)

# ── 6. analysis.py ───────────────────────────────────────────
section("6 · analysis.py")
o,e,rc=run(["python3","analysis.py"])
check("no args → error", rc==1 and "An Error Has Occurred" in o)
f=tmp(SIMPLE)
o,e,rc=run(["python3","analysis.py","0",f])
check("k=0 → error", rc==1 and "An Error Has Occurred" in o)
o,e,rc=run(["python3","analysis.py","3",f])
check("k=n → error", rc==1 and "An Error Has Occurred" in o)
o,e,rc=run(["python3","analysis.py","-1",f])
check("k negative → error", rc==1 and "An Error Has Occurred" in o)
os.unlink(f)
o,e,rc=run(["python3","analysis.py","2","ghost.txt"])
check("missing file → error", rc==1 and "An Error Has Occurred" in o)
f=tmp(DATA5)
o,e,rc=run(["python3","analysis.py","2",f],timeout=120)
lines=o.strip().split("\n")
check("output has 2 lines", rc==0 and len(lines)==2)
check("first line starts 'nmf:'", rc==0 and lines[0].startswith("nmf:"))
check("second line starts 'kmeans:'", rc==0 and lines[1].startswith("kmeans:"))
try:
    nv=float(lines[0].split(":")[1]); kv=float(lines[1].split(":")[1])
    check("scores are valid floats in [-1,1]", rc==0 and -1<=nv<=1 and -1<=kv<=1)
    check("scores have 4 decimal places", rc==0 and len(lines[0].split(":")[1].strip().split(".")[1])==4)
except: check("scores parseable", False)
os.unlink(f)

# ── 7. C executable ──────────────────────────────────────────
section("7 · C executable (make + ./symnmf)")
o,e,rc=run(["make"],timeout=30)
check("make: no errors", rc==0)
check("./symnmf binary exists", os.path.exists("./symnmf"))
if os.path.exists("./symnmf"):
    f=tmp(SIMPLE)
    o,e,rc=run(["./symnmf","sym",f])
    lines=[l for l in o.strip().split("\n") if l]
    check("C sym: 3 rows", rc==0 and len(lines)==3)
    vals=re.findall(r'-?\d+\.\d+',o)
    check("C sym: 4 decimal places", rc==0 and all(len(v.split(".")[1])==4 for v in vals))
    check("C sym: no '-0.0000'", rc==0 and "-0.0000" not in o)
    os.unlink(f)
    f=tmp(SIMPLE)
    o,e,rc=run(["./symnmf","bad",f])
    check("C: bad goal → error", rc!=0 and "An Error Has Occurred" in o)
    os.unlink(f)
    o,e,rc=run(["./symnmf","sym"])
    check("C: wrong arg count → error", rc!=0 and "An Error Has Occurred" in o)
    o,e,rc=run(["./symnmf","sym","no_file.txt"])
    check("C: missing file → error", rc!=0 and "An Error Has Occurred" in o)

# ── 8. regression vs check.csv ───────────────────────────────
section("8 · regression vs check.csv")
if not os.path.exists("check.csv"):
    print("  (skip — check.csv not found)")
else:
    expected={}
    with open("check.csv") as f:
        for r in csv.DictReader(f):
            expected[(r['input_file'],r['k'])]=(round(float(r['nmf_score']),4),round(float(r['kmeans_score']),4))
    for (fname,k),(exp_nmf,exp_km) in expected.items():
        if not os.path.exists(fname):
            check(f"{fname} k={k}",False,"input file missing"); continue
        o,e,rc=run(["python3","analysis.py",k,fname],timeout=120)
        try:
            got_nmf=round(float(o.strip().split("\n")[0].split(":")[1]),4)
            got_km =round(float(o.strip().split("\n")[1].split(":")[1]),4)
            match=(got_nmf==exp_nmf and got_km==exp_km)
        except: match=False; got_nmf=got_km=None
        note=f"got nmf={got_nmf} kmeans={got_km}" if got_nmf is not None else "parse error"
        check(f"{fname} k={k}: nmf={exp_nmf} kmeans={exp_km}", match, note)

# ── final report ─────────────────────────────────────────────
passed=sum(1 for _,p,_ in results if p)
failed=sum(1 for _,p,_ in results if not p)
total=len(results)
print(f"\n{B}{'═'*60}{RS}")
print(f"{B}  RESULTS{RS}")
print(f"{B}{'═'*60}{RS}")
print(f"  {G}PASSED: {passed}/{total}{RS}")
if failed:
    print(f"  {R}FAILED: {failed}/{total}{RS}")
    print(f"\n{B}  Failed tests:{RS}")
    for name,p,note in results:
        if not p: print(f"    {R}✖{RS} {name}  {note}")
else:
    print(f"  {G}{B}ALL TESTS PASSED ✔{RS}")
print(f"{B}{'═'*60}{RS}\n")
