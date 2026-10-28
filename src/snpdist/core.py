"""Core algorithm.

For n sequences of length L the distance between i and j is the number of alignment columns where
both bases are A/C/G/T and differ. Instead of an O(n^2 * L) Python loop, count the columns where
both are valid and subtract the columns where they are equal:

    dist = V V^T - sum_b (B_b B_b^T)

with V the 0/1 "is ACGT" matrix and B_b the one-hot matrix of base b. Each product is a matrix
multiplication, so the heavy work runs in BLAS. Columns are processed in chunks to bound memory.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

_ACGT = np.frombuffer(b"ACGT", dtype=np.uint8)


class AlignmentError(ValueError):
    """The input is not a valid alignment."""


def read_fasta(path: str | Path) -> tuple[list[str], list[bytes]]:
    """Read a (optionally gzipped) FASTA file. Returns names and raw sequences (bytes)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    names: list[str] = []
    seqs: list[list[bytes]] = []
    with opener(path, "rb") as fh:
        for raw in fh:
            line = raw.rstrip(b"\r\n")
            if not line:
                continue
            if line.startswith(b">"):
                names.append(line[1:].split(None, 1)[0].decode() if len(line) > 1 else "")
                seqs.append([])
            else:
                if not seqs:
                    raise AlignmentError("sequence data before the first '>' header")
                seqs[-1].append(line.strip())
    if not names:
        raise AlignmentError("no sequences found")
    return names, [b"".join(s) for s in seqs]


def _to_array(seqs: list[bytes]) -> np.ndarray:
    length = len(seqs[0])
    if length == 0:
        raise AlignmentError("sequences are empty")
    for i, s in enumerate(seqs):
        if len(s) != length:
            raise AlignmentError(
                f"sequence {i + 1} has length {len(s)} but the first has length {length}: "
                "not an alignment"
            )
    return np.frombuffer(b"".join(seqs), dtype=np.uint8).reshape(len(seqs), length)


def distance_matrix(
    seqs: list[bytes], *, count_all: bool = False, keep_case: bool = False, chunk: int = 20_000
) -> np.ndarray:
    """Pairwise SNP distances as an (n, n) int64 array.

    count_all: count every column where the characters differ (including N, gaps and IUPAC codes),
        like `snp-dists -a`. Otherwise only columns where both bases are A/C/G/T are compared.
    keep_case: do not upper-case first, so lower-case bases are not A/C/G/T (like `-k`).
    """
    arr = _to_array(seqs)
    if not keep_case:
        arr = np.where((arr >= 97) & (arr <= 122), arr - 32, arr).astype(np.uint8)
    n, length = arr.shape
    out = np.zeros((n, n), dtype=np.float64)
    symbols = np.unique(arr) if count_all else _ACGT
    # count_all: dist = L - (columns where characters are equal);  default: both-valid - equal-valid
    both_valid = np.zeros((n, n), dtype=np.float64)
    for start in range(0, length, chunk):
        block = arr[:, start : start + chunk]
        if not count_all:
            valid = np.isin(block, _ACGT).astype(np.float32)
            both_valid += valid @ valid.T
        for sym in symbols:
            hot = (block == sym).astype(np.float32)
            out += hot @ hot.T
    dist = (length - out) if count_all else (both_valid - out)
    return np.rint(dist).astype(np.int64)


def naive_distance_matrix(
    seqs: list[bytes], *, count_all: bool = False, keep_case: bool = False
) -> np.ndarray:
    """Reference implementation: a plain double loop. Slow; used by the tests and the benchmark."""
    if not keep_case:
        seqs = [s.upper() for s in seqs]
    _to_array(seqs)
    n = len(seqs)
    d = np.zeros((n, n), dtype=np.int64)
    valid = set(b"ACGT")
    for i in range(n):
        for j in range(i + 1, n):
            c = 0
            for a, b in zip(seqs[i], seqs[j], strict=True):
                if a != b and (count_all or (a in valid and b in valid)):
                    c += 1
            d[i, j] = d[j, i] = c
    return d
