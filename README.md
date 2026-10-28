# snpdist

Pairwise SNP distance matrix from a FASTA alignment, in Python and NumPy. A re-implementation of the idea
behind Torsten Seemann's [snp-dists](https://github.com/tseemann/snp-dists), written as a learning and
portfolio project, and **tested against the real C binary** so correctness is checked, not asserted.

```bash
micromamba create -y -f environment.yml    # python, numpy, pytest, ruff, and snp-dists (oracle only)
micromamba run -n snpdist pip install -e .
snpdist alignment.fa > matrix.tsv          # also: -a  -k  -m  -c  -b  (same meaning as snp-dists)
micromamba run -n snpdist python -m pytest -q
micromamba run -n snpdist python bench/bench.py
```

## How it works
The distance between two sequences is the number of columns where both bases are A/C/G/T and differ.
A double loop is O(n²L) in Python. Instead, for each chunk of columns:

`distance = (V Vᵀ) − Σ_b (B_b B_bᵀ)`, where V marks valid (ACGT) cells and B_b is the one-hot matrix for base b.

Both terms are matrix multiplications, so the work runs in BLAS. Counts are accumulated in float64 and each
float32 partial sum is bounded by the chunk size (20,000), so results are exact integers.

## Evidence it is correct
- 17 tests (`tests/test_snpdist.py`), all passing, none skipped:
  hand-computed cases; symmetry, zero diagonal and the triangle inequality; equality with a slow naive reference
  on messy input (lower case, N, gaps, IUPAC codes); chunk-size independence; error handling for unequal lengths,
  empty and headerless files; CRLF, multi-line and gzip input; the command-line output.
- **Four tests compare the matrix with the C `snp-dists` binary** (default, `-a`, `-k`, `-a -k`) on random 25 × 2,000 bp
  alignments containing lower case, `N`, `-`, `R`, `Y`. The matrices are identical.
- `ruff` clean.

## Speed (indicative only)
`python bench/bench.py`, simulated alignments, 8-core Apple-silicon laptop:

| case | naive Python | this (NumPy) | C, 1 thread | C, 8 threads |
|---|---|---|---|---|
| 40 × 5 kb | 0.07 s | 0.002–0.004 s | 0.015 s | 0.010 s |
| 200 × 50 kb | n/a | 0.15–0.21 s | 1.7–1.9 s | 0.45–0.51 s |
| 500 × 100 kb | n/a | 1.3–6.1 s | 22–32 s | 5.5 s |

Read this carefully: the NumPy version is faster than the C program at one thread because BLAS uses all cores and
SIMD, and it is **comparable to the C program at eight threads** on the largest case, not clearly faster. The
timings were noisy: another process on the machine kept the load average at about 11 on 8 cores, and the
NumPy 500 × 100 kb case varied from 1.3 s to 6.1 s between runs. The result depends on the BLAS library, so
it will differ on other machines.

## Limitations
No `-j`, `-x` or `-L` options. Memory use is n² floats plus a chunk of one-hot matrices (about 80 MB for
500 sequences). Distances are computed on the whole alignment; there is no streaming for very large inputs.
The C tool is the right choice for those.

## Provenance
Written with the assistance of Claude Code (Anthropic); I ran and checked the tests and benchmark. The C
reference is GPL-3.0 and is used only as an external test oracle, not copied.
