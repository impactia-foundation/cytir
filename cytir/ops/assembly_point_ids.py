#!/usr/bin/env python3
"""The ONE reader for a pinned assembly point-ID artifact.

The assembly's producers publish a point-ID set in two shapes, and both are
canonical:

  * `*.jsonl.gz` -- one id per line, as a bare string, `{"point_id": …}`, or
    the task-427 stage's `{"old_point_id": …, "url": …}`.
  * a bucket-witness `RECEIPT.json` -- format
    `uuid128-be+first-byte-buckets+sorted-unique+same-domain-residual`: 256
    raw-binary bucket files of 16-byte big-endian UUIDs, each pinned by its own
    sha256 in the receipt's `artifacts`, bucketed by leading byte so the
    concatenation in bucket order is already globally sorted. `gesetze` and
    `wikipedia_413a` publish their claims this way, and the receipt also lists
    non-bucket artifacts, so the bucket files are selected by URI rather than
    by taking every artifact.

Until task 513b the fold understood only the first shape, so a fragment for
either of those two sets quarantined on a `BadGzipFile`. The reader for the
second shape already existed -- in a task script. Copying it into the fold would
have made two implementations of "what does this set claim?", which is the
mirror-contract defect this project has been bitten by before, so it lives here
once and both sides import it.

Every read is refused rather than degraded: digest drift, a bucket whose size is
not a multiple of 16, a total that disagrees with the receipt's own `records`,
buckets that are not actually sorted-unique as the format declares, a duplicate
id, or a count that disagrees with the binding.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import uuid
from pathlib import Path

BUCKET_MARKER = "buckets/ids-"
UUID_BYTES = 16


class PointIdArtifactDefect(Exception):
    """A pinned point-ID artifact is missing, drifted, or malformed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(uri: str, root: Path) -> Path:
    return root / str(uri).removeprefix("artifact://")


def _read_jsonl_gz(path: Path, label: str) -> set[bytes]:
    out: set[bytes] = set()
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                value = row.get("point_id") or row.get("old_point_id") or row.get("id")
                if value is None:
                    raise PointIdArtifactDefect(f"{label}: no point ID in a row")
            else:
                value = row
            key = uuid.UUID(str(value)).bytes
            if key in out:
                raise PointIdArtifactDefect(f"{label}: duplicate point ID in {path}")
            out.add(key)
    return out


def _read_bucket_witness(path: Path, root: Path, label: str) -> set[bytes]:
    receipt = json.loads(path.read_text())
    out: set[bytes] = set()
    ordered: list[bytes] = []
    buckets = 0
    for art in receipt.get("artifacts") or []:
        if BUCKET_MARKER not in art.get("uri", ""):
            continue
        p = resolve(art["uri"], root)
        if not p.is_file():
            raise PointIdArtifactDefect(f"{label}: missing bucket {p}")
        data = p.read_bytes()
        got = hashlib.sha256(data).hexdigest()
        if got != art.get("sha256"):
            raise PointIdArtifactDefect(
                f"{label}: bucket digest drift {p} -- file is {got}, receipt "
                f"pins {art.get('sha256')}")
        if len(data) % UUID_BYTES:
            raise PointIdArtifactDefect(
                f"{label}: bucket {p} is {len(data)} bytes, not a multiple of "
                f"{UUID_BYTES}")
        buckets += 1
        for i in range(0, len(data), UUID_BYTES):
            key = data[i:i + UUID_BYTES]
            if key in out:
                raise PointIdArtifactDefect(
                    f"{label}: duplicate point ID across buckets, but the format "
                    "declares sorted-unique")
            out.add(key)
            ordered.append(key)
    if not buckets:
        raise PointIdArtifactDefect(
            f"{label}: {path} names no `{BUCKET_MARKER}` artifact -- it is not a "
            "bucket-witness receipt")
    declared = receipt.get("records")
    if declared is not None and len(out) != int(declared):
        raise PointIdArtifactDefect(
            f"{label}: {len(out):,} ids across {buckets} bucket(s), receipt "
            f"declares {int(declared):,} records")
    # the format's own claim, checked rather than trusted
    if ordered != sorted(ordered):
        raise PointIdArtifactDefect(
            f"{label}: buckets are not globally sorted as the format declares")
    return out


def read_point_ids(root: Path, binding: dict, label: str) -> set[bytes]:
    """Read a pinned point-ID artifact in either canonical shape.

    `binding` is the claim-manifest / fragment binding: `uri`, optional
    `sha256` (of the named file itself), optional `count`.
    """
    path = resolve(binding["uri"], root)
    if not path.is_file():
        raise PointIdArtifactDefect(f"{label}: missing point-ID artifact {path}")
    if binding.get("sha256"):
        actual = sha256_file(path)
        if actual != binding["sha256"]:
            raise PointIdArtifactDefect(
                f"{label}: digest drift -- file is {actual} "
                f"({path.stat().st_size:,} bytes), the producer published "
                f"{binding['sha256']} ({int(binding.get('bytes') or 0):,} bytes)")
    if "".join(path.suffixes[-2:]) == ".jsonl.gz" or path.suffix == ".gz":
        out = _read_jsonl_gz(path, label)
    elif path.suffix == ".json":
        out = _read_bucket_witness(path, root, label)
    else:
        raise PointIdArtifactDefect(
            f"{label}: {path} has an unrecognised shape (expected *.jsonl.gz or "
            "a bucket-witness RECEIPT.json)")
    if binding.get("count") is not None and len(out) != int(binding["count"]):
        raise PointIdArtifactDefect(
            f"{label}: {len(out):,} IDs read, the binding declares "
            f"{int(binding['count']):,}")
    return out
