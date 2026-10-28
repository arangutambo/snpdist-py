import gzip
import shutil
import subprocess

import numpy as np
import pytest

from snpdist.cli import main
from snpdist.core import AlignmentError, distance_matrix, naive_distance_matrix, read_fasta

TINY = b">a\nACGTACGTNN\n>b\nACGAACGTAC\n>c\nacgtacgt--\n"


def write(tmp_path, data, name="aln.fa"):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def rand_alignment(rng, n, length, alphabet=b"ACGT", extra=b"", p_extra=0.0):
    pool = np.frombuffer(alphabet, dtype=np.uint8)
    arr = rng.choice(pool, size=(n, length))
    if extra:
        ex = np.frombuffer(extra, dtype=np.uint8)
        mask = rng.random((n, length)) < p_extra
        arr[mask] = rng.choice(ex, size=int(mask.sum()))
    return [bytes(r) for r in arr]


def test_hand_computed_default():
    _, seqs = read_fasta_bytes(TINY)
    assert distance_matrix(seqs).tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]


def test_hand_computed_count_all():
    _, seqs = read_fasta_bytes(TINY)
    assert distance_matrix(seqs, count_all=True).tolist() == [[0, 3, 2], [3, 0, 3], [2, 3, 0]]


def test_keep_case_ignores_lowercase():
    _, seqs = read_fasta_bytes(TINY)
    assert distance_matrix(seqs, keep_case=True).tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 0]]


def read_fasta_bytes(data):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.fa"
        p.write_bytes(data)
        return read_fasta(p)


def test_properties_symmetry_diagonal_triangle():
    rng = np.random.default_rng(0)
    seqs = rand_alignment(rng, 30, 400, extra=b"N-", p_extra=0.02)
    d = distance_matrix(
        seqs, count_all=True
    )  # count_all is a true metric; the default ignores columns pairwise
    assert (d == d.T).all()
    assert (np.diag(d) == 0).all()
    for i in range(0, 30, 3):
        for j in range(0, 30, 3):
            for k in range(0, 30, 3):
                assert d[i, k] <= d[i, j] + d[j, k]


@pytest.mark.parametrize("count_all", [False, True])
@pytest.mark.parametrize("keep_case", [False, True])
def test_matches_naive_on_messy_input(count_all, keep_case):
    rng = np.random.default_rng(1)
    seqs = rand_alignment(rng, 12, 300, alphabet=b"ACGTacgt", extra=b"N-RYKM", p_extra=0.05)
    fast = distance_matrix(seqs, count_all=count_all, keep_case=keep_case)
    slow = naive_distance_matrix(seqs, count_all=count_all, keep_case=keep_case)
    assert (fast == slow).all()


def test_chunking_does_not_change_result():
    rng = np.random.default_rng(2)
    seqs = rand_alignment(rng, 10, 1001, extra=b"N", p_extra=0.03)
    assert (distance_matrix(seqs, chunk=1001) == distance_matrix(seqs, chunk=97)).all()


def test_unequal_lengths_is_an_error():
    with pytest.raises(AlignmentError, match="not an alignment"):
        distance_matrix([b"ACGT", b"ACG"])


def test_empty_and_headerless_inputs(tmp_path):
    with pytest.raises(AlignmentError):
        read_fasta(write(tmp_path, b""))
    with pytest.raises(AlignmentError, match="before the first"):
        read_fasta(write(tmp_path, b"ACGT\n>a\nACGT\n"))


def test_multiline_crlf_and_gzip(tmp_path):
    data = b">s1 description\r\nAC\r\nGT\r\n>s2\r\nACGA\r\n"
    names, seqs = read_fasta(write(tmp_path, data))
    assert names == ["s1", "s2"] and seqs == [b"ACGT", b"ACGA"]
    gz = tmp_path / "aln.fa.gz"
    gz.write_bytes(gzip.compress(data))
    assert read_fasta(gz)[1] == seqs


def test_cli_outputs(tmp_path, capsys):
    p = write(tmp_path, TINY)
    assert main([str(p), "-b"]) == 0
    first = capsys.readouterr().out.splitlines()
    assert first[0] == "\ta\tb\tc" and first[1] == "a\t0\t1\t0"
    assert main([str(p), "-m", "-c"]) == 0
    assert capsys.readouterr().out.splitlines()[1] == "a,b,1"
    assert main([str(tmp_path / "missing.fa")]) == 1


@pytest.mark.skipif(
    shutil.which("snp-dists") is None, reason="reference snp-dists binary not installed"
)
@pytest.mark.parametrize("flags", [[], ["-a"], ["-k"], ["-a", "-k"]])
def test_agrees_with_reference_binary(tmp_path, flags):
    rng = np.random.default_rng(3)
    seqs = rand_alignment(rng, 25, 2000, alphabet=b"ACGTacgt", extra=b"N-RY", p_extra=0.04)
    aln = tmp_path / "r.fa"
    aln.write_bytes(b"".join(b">s%d\n%s\n" % (i, s) for i, s in enumerate(seqs)))
    ref = subprocess.run(
        ["snp-dists", "-q", *flags, str(aln)], capture_output=True, text=True, check=True
    ).stdout
    ref_m = np.array([[int(x) for x in row.split("\t")[1:]] for row in ref.splitlines()[1:]])
    ours = distance_matrix(seqs, count_all="-a" in flags, keep_case="-k" in flags)
    assert (ours == ref_m).all()
