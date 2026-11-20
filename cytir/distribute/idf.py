"""cytir.distribute.idf — the pinned BM25 IDF snapshot of a full dump (task 687, S8;
specs/dump-format.md §1 `sparse.idf_snapshot`, §6a).

WHY. Qdrant scores a sparse vector under `modifier: idf` as
`stored_weight * ln((n + 1) / (df + 0.5))`, where `n` is the number of vectors in the
sparse index and `df(t)` the length of term `t`'s posting list (`cytir/index/layout.py`).
The stored weights travel in the dump; `n` and `df` are properties of the corpus the
index was built over. A consumer who cannot see them gets scores that look plausible
and are wrong (release-doctrine #5). So a full dump ships the integers, not a float
table: `n` in the manifest block, and `df` per term-hash in a sorted sidecar. The IDF is
the formula above, applied to those integers by whoever reads them.

WHERE IT COMES FROM. From the dump's own rows, in the same pass that writes the shards:
the snapshot describes exactly the corpus the shards hold, and a fresh `cyt apply` of
those shards reproduces it (the rehearsal proves this by scoring a query on the applied
collection and comparing with `stored_weight * idf` taken from the snapshot). It is NOT
read from the live twin's index header, which carries stale postings (task 627).

A DELTA PACK ships no snapshot: its rows are a change set, not a corpus. The snapshot of
the full dump it is based on stays the reference.

The sidecar is `bm25-idf.tsv.gz`: one `term_hash<TAB>df` line per term, ascending, gzip
with a zero mtime so the same corpus gives the same bytes. The manifest block is covered
by the manifest signature, so a swapped sidecar fails the sha256 check.
"""
from __future__ import annotations

import gzip
from cytir.acquire.gzip_pinned import deterministic_gzip_open
import math
import pathlib
from collections import Counter
from typing import Iterator

SCHEMA = "aw.bm25-idf-snapshot.v1"
FILENAME = "bm25-idf.tsv.gz"
FORMULA = "ln((n + 1) / (df + 0.5))"


class IdfSnapshotRefusal(ValueError):
    pass


def idf(n: int, df: int) -> float:
    """The one formula (matches Qdrant v1.18 `modifier: idf`; `cytir/index/layout.py`)."""
    return math.log((n + 1) / (df + 0.5))


class IdfAccumulator:
    """Counts `n` (rows that ship a sparse vector) and `df` (rows containing a term)."""

    def __init__(self) -> None:
        self.n = 0
        self.df: Counter = Counter()

    def add(self, row: dict) -> None:
        indices = row.get("sparse_indices")
        if indices is None:
            return
        self.n += 1
        self.df.update(set(indices))

    def write(self, out_dir: pathlib.Path) -> dict | None:
        """Writes the sidecar; returns the manifest's `idf_snapshot` block, or None when
        no row shipped a sparse vector."""
        if not self.n:
            return None
        from cytir.distribute import pack

        path = out_dir / FILENAME
        with open(path, "wb") as raw, deterministic_gzip_open(
                raw, compresslevel=6) as gz:
            for term in sorted(self.df):
                gz.write(f"{term}\t{self.df[term]}\n".encode("ascii"))
        return {"schema": SCHEMA, "path": FILENAME, "formula": FORMULA,
                "n": self.n, "vocabulary": len(self.df),
                "sha256": pack.sha256_file(str(path)),
                "bytes": path.stat().st_size}


def iter_snapshot(path: pathlib.Path) -> Iterator[tuple[int, int]]:
    """Yields `(term_hash, df)` in file order; refuses an unsorted or malformed line."""
    last = None
    with gzip.open(path, "rt", encoding="ascii") as fh:
        for line in fh:
            term, _, df = line.rstrip("\n").partition("\t")
            if not (term.lstrip("-").isdigit() and df.isdigit()):
                raise IdfSnapshotRefusal(f"malformed line in {path}: {line[:60]!r}")
            term, df = int(term), int(df)
            if last is not None and term <= last:
                raise IdfSnapshotRefusal(f"{path} is not strictly ascending at {term}")
            last = term
            yield term, df


def load_snapshot(manifest: dict, manifest_dir: pathlib.Path) -> dict[int, float]:
    """Verifies the declared snapshot and returns `{term_hash: idf}`. Memory grows with the
    vocabulary; a consumer that cannot hold it streams `iter_snapshot` instead."""
    block = verify_snapshot(manifest, manifest_dir)
    if block is None:
        raise IdfSnapshotRefusal("this manifest declares no BM25 IDF snapshot")
    return {t: idf(block["n"], df) for t, df in iter_snapshot(manifest_dir / block["path"])}


def verify_snapshot(manifest: dict, manifest_dir: pathlib.Path) -> dict | None:
    """The check `cyt apply` runs before it touches Qdrant. Returns the block, or None when
    the manifest declares none. A declared snapshot that is missing, altered, miscounted
    or in another schema/formula is a refusal, never a skip."""
    from cytir.distribute import pack

    sparse = manifest.get("sparse")
    block = (sparse or {}).get("idf_snapshot")
    if not block:
        return None
    if block.get("schema") != SCHEMA or block.get("formula") != FORMULA:
        raise IdfSnapshotRefusal(
            f"IDF snapshot is {block.get('schema')!r}/{block.get('formula')!r}; this build "
            f"reads {SCHEMA!r}/{FORMULA!r}, and scores would not match")
    path = manifest_dir / block["path"]
    if not path.is_file():
        raise IdfSnapshotRefusal(f"declared IDF snapshot {block['path']} is missing")
    got = pack.sha256_file(str(path))
    if got != block["sha256"]:
        raise IdfSnapshotRefusal(
            f"IDF snapshot sha256 mismatch: got {got}, manifest says {block['sha256']}")
    vocabulary = 0
    for _term, df in iter_snapshot(path):
        vocabulary += 1
        if not 1 <= df <= block["n"]:
            raise IdfSnapshotRefusal(f"df {df} outside 1..n={block['n']} in {path}")
    if vocabulary != block["vocabulary"]:
        raise IdfSnapshotRefusal(
            f"IDF snapshot has {vocabulary} terms, manifest says {block['vocabulary']}")
    if block["n"] > manifest["counts"]["points"]:
        raise IdfSnapshotRefusal(
            f"snapshot n={block['n']} exceeds the dump's {manifest['counts']['points']} points")
    return block
