#!/usr/bin/env python3
"""Cut the committed `openalex_academic_dump` sample from one part of the OpenAlex works snapshot.

    python tests/fixture_tools/cut_openalex.py PART.parquet OUT_DIR

Deterministic: the works whose id is in `SELECTED`, `RECOVERY` or `REJECTED` are copied, whole and
with every column, in the part's own order, into `OUT_DIR/updated_date=2026-02-09/part_0046.parquet`
(the part's own relative path in the snapshot, so a stranger's directory looks like the snapshot's).
No value is edited, shortened or removed; the schema is the part's schema. Prints the source part's
size and sha256, the cut file's size and sha256 and, for each work, the sha256 of the JSON that the
extractor reads (DuckDB `to_json` of the whole row), so `SOURCE.md` can be checked against a fresh cut.
A parquet writer stamps its own version in the file footer: another pyarrow version writes other file
bytes for the same rows. The per-work JSON hashes are the stable identity of the rows.

Needs `pyarrow` and `duckdb` (the extractor's own dependency); nothing here is a runtime dependency of
the family. Nothing is fetched: the part is a local file the caller already holds.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

RELATIVE_PATH = "updated_date=2026-02-09/part_0046.parquet"
#: Ten works the pinned policy selects (languages en, fr, de, it and null; with and without a DOI;
#: one and several authors; an abstract of 105 characters and one of 2,700).
SELECTED = (
    "https://openalex.org/W1043719763", "https://openalex.org/W114559381",
    "https://openalex.org/W1433235132", "https://openalex.org/W1551380949",
    "https://openalex.org/W2019609318", "https://openalex.org/W2062740319",
    "https://openalex.org/W2266349331", "https://openalex.org/W2520232370",
    "https://openalex.org/W2625258660", "https://openalex.org/W2905023637",
)
#: A work the selection admits and the renderer refuses (an abstract under 40 characters once the
#: title is gone): the old script writes it to its recovery stage, the family excludes it with its reason.
RECOVERY = "https://openalex.org/W3122717005"
#: Works the selection refuses: a medical work, a work with no abstract, a retracted work, an
#: uncited work of a non-core venue.
REJECTED = (
    "https://openalex.org/W153375954", "https://openalex.org/W127391412",
    "https://openalex.org/W6959737561", "https://openalex.org/W1060887079",
)
WANTED = frozenset((*SELECTED, RECOVERY, *REJECTED))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cut_table(source: Path, ids=WANTED):
    """The rows of `source` whose id is in `ids`, as one pyarrow table with the source's schema.
    Streams by row group, so a part of any size is never held whole."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(source))
    pieces = []
    for i in range(pf.num_row_groups):
        group = pf.read_row_group(i)
        pieces.append(group.filter(pc.is_in(group["id"], value_set=pa.array(sorted(ids)))))
    return pa.concat_tables(pieces) if pieces else pf.schema_arrow.empty_table()


def work_hashes(path: Path) -> list[tuple[str, str]]:
    """`(id, sha256 of DuckDB to_json of the whole row)`, in id order."""
    import duckdb

    con = duckdb.connect()
    try:
        con.execute("SET TimeZone='UTC'")       # `to_json` writes created_date in the session's zone
        rows = con.execute("SELECT id, to_json(t) FROM read_parquet(?) AS t ORDER BY id",
                           [str(path)]).fetchall()
    finally:
        con.close()
    return [(i, hashlib.sha256(j.encode()).hexdigest()) for i, j in rows]


def cut(source: Path, out: Path) -> list[tuple[str, str]]:
    import pyarrow.parquet as pq

    table = cut_table(source)
    missing = WANTED - set(table["id"].to_pylist())
    if missing:
        raise SystemExit(f"works not found in {source.name}: {sorted(missing)}")
    target = out / RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(target), compression="zstd", row_group_size=len(table))
    return work_hashes(target)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    source, out = Path(argv[1]), Path(argv[2])
    print(f"part {source.stat().st_size} {sha256_file(source)}")
    hashes = cut(source, out)
    target = out / RELATIVE_PATH
    print(f"{RELATIVE_PATH} {target.stat().st_size} {sha256_file(target)}")
    for work_id, digest in hashes:
        print(f"work {work_id} {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
