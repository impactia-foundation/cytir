#!/usr/bin/env python3
"""Cut the committed `service_public_fiche` sample from the publisher's dump archive.

    python tests/fixture_tools/cut_service_public_fiche.py ARCHIVE.zip OUT_DIR

Deterministic: the members named in `MEMBERS` are copied byte for byte (no re-encoding, no
trimming) from the archive to `OUT_DIR/<name>`, in that order. Prints the archive's sha256 and each
copied file's size and sha256, so the provenance file can be checked against a fresh cut.
Nothing is fetched: the archive is a local file the caller already holds.
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

MEMBERS = ("F1613.xml", "F2725.xml", "F13476.xml", "F24415.xml", "F31417.xml", "F36491.xml")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cut(archive: Path, out: Path) -> list[tuple[str, int, str]]:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(archive) as zf:
        for name in MEMBERS:
            data = zf.read(name)
            (out / name).write_bytes(data)
            rows.append((name, len(data), hashlib.sha256(data).hexdigest()))
    return rows


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    archive, out = Path(argv[1]), Path(argv[2])
    print(f"archive sha256 {sha256_file(archive)}")
    for name, size, digest in cut(archive, out):
        print(f"{name} {size} {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
