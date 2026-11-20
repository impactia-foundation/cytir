import hashlib
import json
import os
from pathlib import Path

import pytest

import cytir.acquire.stage_receipt as subject


def test_seal_inventory_is_portable_exhaustive_and_resumable(tmp_path: Path):
    root = tmp_path / "artifacts"
    data = root / "source" / "outlinks" / "empty.jsonl"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"")
    receipt = root / "source" / "outlinks" / "receipt.json"
    logical = hashlib.sha256(b"").hexdigest()
    first = subject.seal(
        artifact_root=root, artifact_paths=[data], receipt_path=receipt,
        records=0, fmt="canonical-jsonl", logical_sha256=logical,
        producer_id="example-producer/test.py", producer_version="v1")
    second = subject.seal(
        artifact_root=root, artifact_paths=[data], receipt_path=receipt,
        records=0, fmt="canonical-jsonl", logical_sha256=logical,
        producer_id="example-producer/test.py", producer_version="v1")
    assert first == second
    value = json.loads(receipt.read_text())
    assert value["schema"] == "aw.stage-receipt.v1"
    assert value["deterministic_encoding"] is True
    assert value["artifacts"] == [{
        "uri": "artifact://source/outlinks/empty.jsonl", "bytes": 0,
        "sha256": logical,
    }]
    assert first["receipt"]["uri"] == "artifact://source/outlinks/receipt.json"


def test_rejects_local_escape_and_missing_data_artifact(tmp_path: Path):
    with pytest.raises(subject.StageSealError, match="at least one"):
        subject.seal(
            artifact_root=tmp_path / "root", artifact_paths=[],
            receipt_path=tmp_path / "root/receipt.json", records=0,
            fmt="jsonl", logical_sha256="0" * 64,
            producer_id="x", producer_version="v1")


def test_staged_atomic_publish_uris_bind_final_paths(tmp_path: Path):
    root = tmp_path / "artifacts"
    partial = root / "source.partial.123"
    final = root / "source"
    data = partial / "markdown/rows.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text("{}\n")
    result = subject.seal(
        artifact_root=root, artifact_paths=[data],
        receipt_path=partial / "markdown/receipt.json", records=1,
        fmt="canonical-jsonl", logical_sha256=hashlib.sha256(b"{}\n").hexdigest(),
        producer_id="example-producer/test.py", producer_version="v1",
        publish_from=partial, publish_to=final)
    receipt = json.loads((partial / "markdown/receipt.json").read_text())
    assert receipt["artifacts"][0]["uri"] == (
        "artifact://source/markdown/rows.jsonl")
    assert result["receipt"]["uri"] == "artifact://source/markdown/receipt.json"
    partial.rename(final)
    assert (root / receipt["artifacts"][0]["uri"].removeprefix(
        "artifact://")).read_text() == "{}\n"


def test_composite_stage_can_bind_existing_published_artifacts(tmp_path: Path):
    root = tmp_path / "artifacts"
    existing = root / "original/archive.tar.gz"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"archive")
    partial = root / "combined.partial.1"
    final = root / "combined"
    partial.mkdir()
    result = subject.seal(
        artifact_root=root, artifact_paths=[existing],
        receipt_path=partial / "receipt.json", records=1, fmt="archive",
        logical_sha256=hashlib.sha256(existing.read_bytes()).hexdigest(),
        producer_id="example-producer/test.py", producer_version="v1",
        publish_from=partial, publish_to=final,
        allow_existing_published=True)
    receipt = json.loads((partial / "receipt.json").read_text())
    assert receipt["artifacts"][0]["uri"] == "artifact://original/archive.tar.gz"
    assert result["receipt"]["uri"] == "artifact://combined/receipt.json"
    outside = tmp_path / "outside"
    outside.write_bytes(b"x")
    with pytest.raises(subject.StageSealError, match="outside portable root"):
        subject.seal(
            artifact_root=tmp_path / "root", artifact_paths=[outside],
            receipt_path=tmp_path / "root/receipt.json", records=1,
            fmt="jsonl", logical_sha256="0" * 64,
            producer_id="x", producer_version="v1")


def test_prevalidated_inventory_streams_without_rehashing(tmp_path: Path,
                                                          monkeypatch):
    root = tmp_path / "artifacts"
    first = root / "data/a"; second = root / "data/b"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"a"); second.write_bytes(b"bb")
    rows = [{"uri": f"artifact://data/{path.name}",
             "bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "_stat": [path.stat().st_dev, path.stat().st_ino,
                       path.stat().st_size, path.stat().st_mtime_ns,
                       path.stat().st_ctime_ns]}
            for path in (first, second)]
    original = subject.sha256_file
    hashed = []

    def observed(path):
        hashed.append(Path(path))
        return original(path)

    monkeypatch.setattr(subject, "sha256_file", observed)
    result = subject.seal_prevalidated(
        artifact_root=root, artifact_inventory=iter(rows),
        receipt_path=root / "stage/receipt.json", records=2, fmt="test",
        logical_sha256="0" * 64, producer_id="test", producer_version="v2")
    assert result["bytes"] == 3
    assert first not in hashed and second not in hashed
    receipt = json.loads((root / "stage/receipt.json").read_text())
    assert receipt["artifacts"] == [
        {key: row[key] for key in ("uri", "bytes", "sha256")} for row in rows]


def test_prevalidated_inventory_rejects_unsorted_or_drifted_rows(tmp_path: Path):
    root = tmp_path / "artifacts"; data = root / "data/a"
    data.parent.mkdir(parents=True); data.write_bytes(b"a")
    stat = data.stat()
    row = {"uri": "artifact://data/a", "bytes": 1, "sha256": "0" * 64,
           "_stat": [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
                     stat.st_ctime_ns]}
    with pytest.raises(subject.StageSealError, match="unique and URI-sorted"):
        subject.seal_prevalidated(
            artifact_root=root, artifact_inventory=[row, row],
            receipt_path=root / "stage/receipt.json", records=1, fmt="test",
            logical_sha256="0" * 64, producer_id="test", producer_version="v2")
    # Drift must be made VISIBLE, not hoped for. `seal_prevalidated` compares
    # [st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns], and a same-size
    # rewrite of the same inode changes only the two timestamps -- which the
    # filesystem very often does not advance between two back-to-back writes.
    # Measured on this box: an identical fingerprint after `write_bytes(b"a")`
    # then `write_bytes(b"b")` in 194 of 200 trials, so this assertion used to
    # pass on luck and failed in CI on 2026-09-21 when the luck ran out. Setting
    # the mtime explicitly is what a real drift looks like on any filesystem
    # whose timestamps work, and makes the check deterministic without weakening
    # it: the size stays 1, so the size comparison still cannot be what fires.
    data.write_bytes(b"b")
    drifted_ns = stat.st_mtime_ns + 1_000_000_000
    os.utime(data, ns=(drifted_ns, drifted_ns))
    assert data.stat().st_mtime_ns != stat.st_mtime_ns, (
        "the filesystem ignored os.utime, so this test cannot observe drift")
    assert data.stat().st_size == stat.st_size, (
        "the size must stay equal or the size check fires instead of drift")
    with pytest.raises(subject.StageSealError, match="identity drift"):
        subject.seal_prevalidated(
            artifact_root=root, artifact_inventory=[row],
            receipt_path=root / "stage/receipt.json", records=1, fmt="test",
            logical_sha256="0" * 64, producer_id="test", producer_version="v2")
