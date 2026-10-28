"""Timing comparison on simulated alignments. Run: python bench/bench.py"""

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


import os
NCPU = os.cpu_count() or 1
print(f"{'case':<28}{'naive python':>14}{'numpy (this)':>14}{'C -j 1':>14}{f'C -j {NCPU}':>14}")
for n, length, with_naive in [(40, 5_000, True), (200, 50_000, False), (500, 100_000, False)]:
    seqs = make(n, length)
    t_np = timed(lambda: distance_matrix(seqs))
    t_naive = timed(lambda: naive_distance_matrix(seqs), repeat=1) if with_naive else None
    t_c = t_cj = None
    if shutil.which("snp-dists"):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.fa"
            p.write_bytes(b"".join(b">s%d\n%s\n" % (i, s) for i, s in enumerate(seqs)))
            run = lambda j: subprocess.run(  # noqa: E731
                ["snp-dists", "-q", "-j", str(j), str(p)], capture_output=True, check=True
            )
            t_c = timed(lambda: run(1), repeat=1 if n >= 500 else 3)
            t_cj = timed(lambda: run(NCPU), repeat=1 if n >= 500 else 3)
    fmt = lambda x: f"{x:.3f}s" if x is not None else "-"  # noqa: E731
    print(
        f"{n} seqs x {length:>7,} bp".ljust(28) + f"{fmt(t_naive):>14}{fmt(t_np):>14}{fmt(t_c):>14}{fmt(t_cj):>14}"
    )
