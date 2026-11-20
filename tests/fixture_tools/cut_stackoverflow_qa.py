#!/usr/bin/env python3
"""Cut the Stack Overflow sample (`data/samples/stackoverflow-qa/Posts.xml`) out of the data dump.

    cut_stackoverflow_qa.py stackoverflow.com-Posts.7z OUT_DIR       (needs the `7z` program)
    cut_stackoverflow_qa.py --stdin OUT_DIR < Posts.xml               (an extracted Posts.xml)

Streams `Posts.xml` once and copies the rows named below, byte for byte, in the dump's own order,
between the dump's own first two lines and its last line. Deterministic: the same dump gives the same
file. Prints the sha256 of every row and of the file, and refuses when a named row is not found.

The ten pairs are the ones of the live Stack Overflow tranches whose posts name their owners as the
dump names a deleted account (`OwnerDisplayName="userNNN"`, no `OwnerUserId`), and each is one passage
under the declared chunk policy without the embed-time tokenizer. The four extra rows are the
negatives the selection must refuse (specs in `data/samples/stackoverflow-qa/SOURCE.md`).
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import subprocess
import sys

#: (question id, accepted answer id)
PAIRS = [
    (1130479, 1136143), (3283306, 3283326), (9560723, 9815605), (10269701, 10269828),
    (30676720, 30803311), (40897069, 40917149), (54220859, 54220863), (51081754, 51102564),
    (42363405, 42363636), (35666177, 35666946),
]
#: score below 10 with an accepted answer (38784, 10595); no accepted answer (1311);
#: an answer that is not accepted (38834)
NEGATIVES = [38784, 10595, 1311, 38834]
POST_IDS = sorted({i for pair in PAIRS for i in pair} | set(NEGATIVES))
ROW = re.compile(rb'\s*<row Id="(\d+)"')


def cut(stream) -> tuple[bytes, dict[int, bytes]]:
    head, tail, found, order = [], b"", {}, {}
    wanted = set(POST_IDS)
    for n, line in enumerate(stream):
        m = ROW.match(line)
        if not m:
            if not found and len(head) < 2:
                head.append(line)
            tail = line
            continue
        i = int(m.group(1))
        if i in wanted and i not in found:
            found[i], order[i] = line, n
    missing = sorted(wanted - set(found))
    if missing:
        raise SystemExit(f"rows not found in the dump: {missing}")
    body = b"".join(found[i] for i in sorted(found, key=order.__getitem__))
    return b"".join(head) + body + tail, found


def main() -> int:
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "--stdin":
        data, found = cut(sys.stdin.buffer)
        out = pathlib.Path(args[1])
    elif len(args) == 2:
        proc = subprocess.Popen(["7z", "x", "-so", args[0], "Posts.xml"], stdout=subprocess.PIPE)
        data, found = cut(proc.stdout)
        proc.stdout.close()
        if proc.wait() != 0:
            raise SystemExit("7z failed")
        out = pathlib.Path(args[1])
    else:
        raise SystemExit(__doc__)
    out.mkdir(parents=True, exist_ok=True)
    (out / "Posts.xml").write_bytes(data)
    for i in POST_IDS:
        print(i, hashlib.sha256(found[i]).hexdigest())
    print("Posts.xml", len(data), hashlib.sha256(data).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
