#!/usr/bin/env python3
"""Cut the committed `stackexchange_nonso_qa_p0` sample from a Stack Exchange data-dump archive.

    python tests/fixture_tools/cut_stackexchange_nonso.py ARCHIVE.7z OUT_DIR

Deterministic: the `<row>` lines of `Posts.xml` whose post id is in `POST_IDS` are copied byte for
byte (no re-encoding, no trimming, no attribute removed) to `OUT_DIR/Posts.xml`, in the archive's
own order, between the archive file's own first two lines (byte-order mark, XML declaration and
`<posts>`) and a closing `</posts>`. `Users.xml` is NOT cut: it holds the profiles of natural
persons. Needs the `7z` program. Prints the archive's sha256, the output's size and sha256 and each
copied row's id and sha256, so the provenance file can be checked against a fresh cut. Nothing is
fetched: the archive is a local file the caller already holds.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

#: Six questions of one small site and their answers that score 3 or more (the accepted answer
#: where it is one of them). Picked because every row's author is a placeholder: the question
#: rows carry `OwnerDisplayName="userNNN"` and no `OwnerUserId`; the answer rows carry
#: `OwnerDisplayName="userNNN"` and no `OwnerUserId` (the dump's own names for deleted accounts).
POST_IDS = ("16941", "16942", "17138",       # question, two answers (both "Top answer")
            "24832", "24833",                # question, accepted answer
            "63305", "63309",                # question, accepted answer
            "143964", "144186",              # question, accepted answer (dated 2016; row says 4.0)
            "189582", "189589",              # question, top answer (2018-07, after the 4.0 cut)
            "266446", "266456")              # question, top answer (2022)
MEMBER = "Posts.xml"
_ROW = re.compile(rb'^\s*<row Id="(\d+)" ')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cut_lines(lines, ids=POST_IDS):
    """`(header lines, [(id, row line)])` from the byte lines of a `Posts.xml`."""
    want = set(ids)
    header, rows = [], []
    for i, line in enumerate(lines):
        if i < 2:
            header.append(line)
            continue
        m = _ROW.match(line)
        if m and m.group(1).decode() in want:
            rows.append((m.group(1).decode(), line))
    return header, rows


def cut(archive: Path, out: Path) -> list[tuple[str, str]]:
    out.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(["7z", "e", "-so", str(archive), MEMBER], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    try:
        header, rows = cut_lines(p.stdout)
    finally:
        p.stdout.close()
        p.wait()
    missing = set(POST_IDS) - {i for i, _ in rows}
    if missing:
        raise SystemExit(f"posts not found in {MEMBER}: {sorted(missing)}")
    with (out / MEMBER).open("wb") as fh:
        fh.writelines(header)
        fh.writelines(line for _i, line in rows)
        fh.write(b"</posts>\n")
    return [(i, hashlib.sha256(line).hexdigest()) for i, line in rows]


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    archive, out = Path(argv[1]), Path(argv[2])
    print(f"archive sha256 {sha256_file(archive)}")
    rows = cut(archive, out)
    target = out / MEMBER
    print(f"{MEMBER} {target.stat().st_size} {sha256_file(target)}")
    for pid, digest in rows:
        print(f"row {pid} {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
