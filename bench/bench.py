"""Timing comparison on simulated alignments. Run: python bench/bench.py"""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from snpdist.core import distance_matrix, naive_distance_matrix


def make(n, length, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.choice(np.frombuffer(b"ACGT", dtype=np.uint8), size=length)
    arr = np.tile(base, (n, 1))
    mut = rng.random((n, length)) < 0.02
    arr[mut] = rng.choice(np.frombuffer(b"ACGTN-", dtype=np.uint8), size=int(mut.sum()))
    return [bytes(r) for r in arr]


def timed(f, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t)
    return best


NCPU = os.cpu_count() or 1
print(f"{'case':<28}{'naive python':>14}{'numpy (this)':>14}{'C -j 1':>14}{f'C -j {NCPU}':>14}")


def fmt(x):
    return f"{x:.3f}s" if x is not None else "-"


for n, length, with_naive in [(40, 5_000, True), (200, 50_000, False), (500, 100_000, False)]:
    seqs = make(n, length)
    t_np = timed(lambda s=seqs: distance_matrix(s))
    t_naive = timed(lambda s=seqs: naive_distance_matrix(s), repeat=1) if with_naive else None
    t_c = t_cj = None
    if shutil.which("snp-dists"):
        with tempfile.TemporaryDirectory() as d:
            aln = Path(d) / "a.fa"
            aln.write_bytes(b"".join(b">s%d\n%s\n" % (i, s) for i, s in enumerate(seqs)))

            def run(j, path=aln):
                return subprocess.run(
                    ["snp-dists", "-q", "-j", str(j), str(path)], capture_output=True, check=True
                )

            reps = 1 if n >= 500 else 3
            t_c = timed(lambda: run(1), repeat=reps)
            t_cj = timed(lambda: run(NCPU), repeat=reps)
    row = f"{n} seqs x {length:>7,} bp".ljust(28)
    print(row + f"{fmt(t_naive):>14}{fmt(t_np):>14}{fmt(t_c):>14}{fmt(t_cj):>14}")
