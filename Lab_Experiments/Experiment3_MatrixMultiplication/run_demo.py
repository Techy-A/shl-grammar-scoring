#!/usr/bin/env python3
"""Run both matrix multiplications step by step on any OS:
(i) x = M v and (ii) C = A B, each as input -> map -> shuffle & sort -> reduce -> result.
Writes output/mv_result.txt and output/mm_result.txt.

Usage:  python run_demo.py              (readable trace)
        python run_demo.py --markdown   (trace as Markdown, used by build_docx.py)
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
MD = "--markdown" in sys.argv


def run(script, lines, env=None):
    proc = subprocess.run([sys.executable, os.path.join(ROOT, script)], input="".join(l + "\n" for l in lines),
                          capture_output=True, text=True, check=True, env={**os.environ, **(env or {})})
    return proc.stdout.splitlines()


def show(title, lines, command, limit=None):
    shown = lines if limit is None or len(lines) <= limit + 2 else lines[:limit]
    body = [f"$ {command}"] + [l.replace("\t", "    ") for l in shown]
    if len(shown) < len(lines):
        body.append(f"... ({len(lines) - len(shown)} more lines)")
    if MD:
        print(f"**{title}** ({len(lines)} lines)\n\n```text\n" + "\n".join(body) + "\n```\n")
    else:
        print(f"\n--- {title}  [{len(lines)} lines]\n" + "\n".join(body))


def heading(text, level=2):
    print(f"\n{'#' * level} {text}\n" if MD else "\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


def matrix(title, rows):
    """Print a dense matrix as a table (Markdown) or as aligned text."""
    fmt = lambda x: f"{x:g}"
    if MD:
        cols = len(rows[0])
        print(f"**{title}**\n")
        print("| |" + "".join(f" col {k} |" for k in range(cols)))
        print("|---|" + "---|" * cols)
        for i, r in enumerate(rows):
            print(f"| **row {i}** |" + "".join(f" {fmt(x)} |" for x in r))
        print()
    else:
        print(f"\n{title}")
        for r in rows:
            print("  [" + "  ".join(f"{fmt(x):>4}" for x in r) + " ]")


def read(name):
    with open(os.path.join(DATA, name)) as f:
        return f.read().splitlines()


def dense(lines, shape):
    out = [[0.0] * shape[1] for _ in range(shape[0])]
    for line in lines:
        _, i, j, x = line.split(",")
        out[int(i)][int(j)] = float(x)
    return out


def save(name, lines):
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    with open(os.path.join(ROOT, "output", name), "w") as f:
        f.writelines(l + "\n" for l in lines)


# ---------------- (i) Matrix-vector ----------------
heading("Part (i): Matrix-vector multiplication x = M v")
M_lines, v_lines = read("matrix_M.txt"), read("vector_v.txt")
M = dense(M_lines, (3, 4))
v = [float(l.split(",")[2]) for l in v_lines]
matrix("Matrix M (3 x 4)", M)
matrix("Vector v (4 x 1)", [[x] for x in v])
show("Input file data/matrix_M.txt (M,i,j,value; zeros omitted)", M_lines, "cat data/matrix_M.txt")
show("Side file data/vector_v.txt (shipped to every mapper)", v_lines, "cat data/vector_v.txt")
env = {"VECTOR_FILE": os.path.join(DATA, "vector_v.txt")}
mapped = run("mv_mapper.py", M_lines, env)
show("Mapper output (i TAB m_ij x v_j)", mapped, "python mv_mapper.py < data/matrix_M.txt")
shuffled = sorted(mapped, key=lambda l: int(l.split("\t")[0]))
show("Shuffle and Sort (grouped by row i)", shuffled, "... | sort -n")
x_lines = run("mv_reducer.py", shuffled)
show("Reducer output (i TAB x_i)", x_lines, "... | sort -n | python mv_reducer.py")
save("mv_result.txt", x_lines)
x = [float(l.split("\t")[1]) for l in x_lines]
matrix("Result x = M v (3 x 1)", [[e] for e in x])
expected_x = [sum(M[i][j] * v[j] for j in range(4)) for i in range(3)]

# ---------------- (ii) Matrix-matrix ----------------
heading("Part (ii): Matrix-matrix multiplication C = A B")
A_lines, B_lines = read("matrix_A.txt"), read("matrix_B.txt")
A, B = dense(A_lines, (3, 3)), dense(B_lines, (3, 2))
matrix("Matrix A (3 x 3)", A)
matrix("Matrix B (3 x 2)", B)
show("Input files data/matrix_A.txt and data/matrix_B.txt", A_lines + B_lines,
     "cat data/matrix_A.txt data/matrix_B.txt")
env = {"ROWS_A": "3", "COLS_B": "2"}
mapped = run("mm_mapper.py", A_lines + B_lines, env)
show("Mapper output (i,k TAB matrix,j,value)", mapped,
     "ROWS_A=3 COLS_B=2 python mm_mapper.py < A and B", 10)
shuffled = sorted(mapped, key=lambda l: tuple(map(int, l.split("\t")[0].split(","))))
show("Shuffle and Sort: the group for result cell C[0][0]", [l for l in shuffled if l.startswith("0,0\t")],
     "... | sort | grep '^0,0'")
show("Shuffle and Sort (all groups)", shuffled, "... | sort", 10)
c_lines = run("mm_reducer.py", shuffled)
show("Reducer output (i,k TAB C[i][k])", c_lines, "... | sort | python mm_reducer.py")
save("mm_result.txt", c_lines)
C = [[0.0] * 2 for _ in range(3)]
for line in c_lines:
    key, value = line.split("\t")
    i, k = map(int, key.split(","))
    C[i][k] = float(value)
matrix("Result C = A B (3 x 2)", C)
expected_C = [[sum(A[i][j] * B[j][k] for j in range(3)) for k in range(2)] for i in range(3)]

# Worked example of one cell so the reducer's join is visible.
c00 = " + ".join(f"{A[0][j]:g} x {B[j][0]:g}" for j in range(3))
ok = x == expected_x and C == expected_C
summary = [f"C[0][0] = sum over j of A[0][j] x B[j][0] = {c00} = {C[0][0]:g}",
           f"Matrix-vector: {len(M_lines)} non-zero entries of M -> {len(x_lines)} result rows",
           f"Matrix-matrix: {len(A_lines) + len(B_lines)} input entries -> {len(mapped)} mapper records"
           f" -> {len(c_lines)} result cells",
           f"Both results match plain Python multiplication: {'YES' if ok else 'NO'}"]
show("Summary and verification", summary, "python run_demo.py")
sys.exit(0 if ok else 1)
