#!/usr/bin/env python3
"""The one point-ID reader, and the shape that used to quarantine a whole set.

`gesetze` and `wikipedia_413a` publish their claims as a bucket-witness
`RECEIPT.json` -- 256 raw-binary files of 16-byte big-endian UUIDs, each pinned
by its own sha256, bucketed by leading byte so bucket order IS sorted order.
The fold understood only `*.jsonl.gz`, so a fragment for either set quarantined
on a `BadGzipFile`; the reader for the other shape existed only inside a task
script. It lives in one module now, and these are the refusals that matter.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import sys
import uuid

import pytest

from cytir.ops.assembly_point_ids import (  # noqa: E402
    PointIdArtifactDefect, read_point_ids)


def _ids(n, offset=0):
    return [uuid.UUID(int=i + offset) for i in range(n)]


def _write_jsonl_gz(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
            for i in ids:
                gz.write((json.dumps({"point_id": str(i)}) + "\n").encode())
    b = path.read_bytes()
    return {"uri": path.name, "sha256": hashlib.sha256(b).hexdigest(),
            "bytes": len(b), "count": len(ids)}


def _write_bucket_witness(root, ids, *, corrupt_sha=False, unsorted=False,
                          bad_size=False, wrong_records=False):
    """A miniature of the real shape: bucket by leading byte, sorted within."""
    d = root / "_work" / "witness"
    (d / "buckets").mkdir(parents=True, exist_ok=True)
    by_bucket = {}
    for i in ids:
        by_bucket.setdefault(i.bytes[0], []).append(i.bytes)
    artifacts = []
    for key in sorted(by_bucket):
        blob = b"".join(sorted(by_bucket[key]) if not unsorted
                        else sorted(by_bucket[key], reverse=True))
        if bad_size:
            blob += b"\x00"
        p = d / "buckets" / f"ids-{key:02x}.bin"
        p.write_bytes(blob)
        sha = hashlib.sha256(blob).hexdigest()
        artifacts.append({"uri": f"artifact://_work/witness/buckets/ids-{key:02x}.bin",
                          "sha256": "0" * 64 if corrupt_sha else sha,
                          "bytes": len(blob)})
    # the real receipts also carry non-bucket artifacts; include one so the
    # reader has to select by URI rather than take everything
    artifacts.append({"uri": "artifact://_work/witness/RUN.json",
                      "sha256": "0" * 64, "bytes": 1})
    receipt = {"schema": "aw.point-witness.v1",
               "format": "uuid128-be+first-byte-buckets+sorted-unique",
               "records": (len(ids) + 1) if wrong_records else len(ids),
               "artifacts": artifacts}
    rp = d / "RECEIPT.json"
    rp.write_text(json.dumps(receipt))
    b = rp.read_bytes()
    return {"uri": "artifact://_work/witness/RECEIPT.json",
            "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b),
            "count": len(ids)}


def test_reads_a_jsonl_gz_id_list(tmp_path):
    ids = _ids(5)
    b = _write_jsonl_gz(tmp_path / "x.jsonl.gz", ids)
    got = read_point_ids(tmp_path, b, "x")
    assert got == {i.bytes for i in ids}


def test_reads_a_bucket_witness_receipt(tmp_path):
    ids = _ids(300)
    b = _write_bucket_witness(tmp_path, ids)
    got = read_point_ids(tmp_path, b, "gesetze-like")
    assert got == {i.bytes for i in ids}


def test_bucket_witness_selects_bucket_artifacts_not_every_artifact(tmp_path):
    """The receipt lists non-bucket artifacts too; taking them all would fail."""
    ids = _ids(40)
    b = _write_bucket_witness(tmp_path, ids)
    receipt = json.loads((tmp_path / "_work/witness/RECEIPT.json").read_text())
    assert any("buckets/ids-" not in a["uri"] for a in receipt["artifacts"])
    assert read_point_ids(tmp_path, b, "x") == {i.bytes for i in ids}


def test_refuses_a_bucket_whose_digest_drifted(tmp_path):
    b = _write_bucket_witness(tmp_path, _ids(20), corrupt_sha=True)
    with pytest.raises(PointIdArtifactDefect, match="bucket digest drift"):
        read_point_ids(tmp_path, b, "x")


def test_refuses_a_bucket_that_is_not_whole_uuids(tmp_path):
    b = _write_bucket_witness(tmp_path, _ids(20), bad_size=True)
    with pytest.raises(PointIdArtifactDefect, match="not a multiple of 16"):
        read_point_ids(tmp_path, b, "x")


def test_refuses_a_record_count_that_disagrees_with_the_buckets(tmp_path):
    b = _write_bucket_witness(tmp_path, _ids(20), wrong_records=True)
    b.pop("count")
    with pytest.raises(PointIdArtifactDefect, match="receipt declares"):
        read_point_ids(tmp_path, b, "x")


def test_refuses_buckets_that_are_not_sorted_as_the_format_declares(tmp_path):
    # needs >1 id in some bucket for the order to be observable
    ids = [uuid.UUID(int=(1 << 120) + i) for i in range(6)]
    b = _write_bucket_witness(tmp_path, ids, unsorted=True)
    with pytest.raises(PointIdArtifactDefect, match="not globally sorted"):
        read_point_ids(tmp_path, b, "x")


def test_refuses_a_json_that_is_not_a_bucket_witness(tmp_path):
    p = tmp_path / "plain.json"
    p.write_text(json.dumps({"artifacts": [{"uri": "artifact://x", "sha256": "0"}]}))
    b = {"uri": "plain.json"}
    with pytest.raises(PointIdArtifactDefect, match="not a bucket-witness"):
        read_point_ids(tmp_path, b, "x")


def test_refuses_an_unrecognised_shape(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("nope")
    with pytest.raises(PointIdArtifactDefect, match="unrecognised shape"):
        read_point_ids(tmp_path, {"uri": "x.txt"}, "x")


def test_refuses_digest_drift_on_the_named_file(tmp_path):
    b = _write_jsonl_gz(tmp_path / "x.jsonl.gz", _ids(3))
    b["sha256"] = "0" * 64
    with pytest.raises(PointIdArtifactDefect, match="digest drift"):
        read_point_ids(tmp_path, b, "x")


def test_refuses_a_count_that_disagrees(tmp_path):
    b = _write_jsonl_gz(tmp_path / "x.jsonl.gz", _ids(3))
    b["count"] = 4
    with pytest.raises(PointIdArtifactDefect, match="declares 4"):
        read_point_ids(tmp_path, b, "x")
