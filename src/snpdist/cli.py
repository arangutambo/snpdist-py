"""Command-line interface; output format follows snp-dists."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__
from .core import AlignmentError, distance_matrix, read_fasta


def render(names: list[str], d: np.ndarray, *, sep: str, molten: bool, corner: str) -> str:
    lines: list[str] = []
    if molten:
        for i, a in enumerate(names):
            lines.extend(f"{a}{sep}{b}{sep}{d[i, j]}" for j, b in enumerate(names))
    else:
        lines.append(sep.join([corner, *names]))
        lines.extend(sep.join([a, *map(str, d[i])]) for i, a in enumerate(names))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="snpdist", description="Pairwise SNP distance matrix from a FASTA alignment"
    )
    p.add_argument("alignment", help="aligned FASTA, optionally .gz")
    p.add_argument(
        "-a", "--all", action="store_true", help="count all differences, not just [ACGT]"
    )
    p.add_argument(
        "-k",
        "--keep-case",
        action="store_true",
        help="do not upper-case; lower-case bases are ignored",
    )
    p.add_argument("-m", "--molten", action="store_true", help="long (molten) output")
    p.add_argument("-c", "--csv", action="store_true", help="comma instead of tab")
    p.add_argument("-b", "--blank-corner", action="store_true", help="blank the top-left cell")
    p.add_argument("-v", "--version", action="version", version=f"snpdist {__version__}")
    args = p.parse_args(argv)
    try:
        names, seqs = read_fasta(args.alignment)
        d = distance_matrix(seqs, count_all=args.all, keep_case=args.keep_case)
    except (AlignmentError, OSError) as exc:
        print(f"snpdist: error: {exc}", file=sys.stderr)
        return 1
    corner = "" if args.blank_corner else f"snpdist {__version__}"
    sys.stdout.write(
        render(names, d, sep="," if args.csv else "\t", molten=args.molten, corner=corner)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
