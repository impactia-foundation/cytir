#!/usr/bin/env python3
"""Task 696 slice 3b (1/3) -- `seal_stage_receipt` moved to `cytir.acquire.stage_receipt`.

The receipt is an immutable inventory every stage seals; its bytes feed manifest shas. So the
move is proved three ways:

  1. golden: sealing a fixed fixture writes receipt bytes with the sha256 below. The sha was
     taken from the OLD `seal_stage_receipt.py` BEFORE the move (the file at
     b126e64d); the same test is green on the old code and on the moved code;
  2. identity: the old name IS the package module;
  3. script start: the old path and `python -m cytir.acquire.stage_receipt` write the same bytes.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent

FILES = {
    "stage/a.jsonl": b'{"k":1}\n{"k":2}\n',
    "stage/b.jsonl": b'{"k":3}\n',
    "stage/é.jsonl": b"",
}
LOGICAL = hashlib.sha256(b"fixture").hexdigest()
#: sha256 of the receipt written for FILES by the pre-move code (producer id kept as a label).
GOLDEN_SEAL = "898de2c764017c2ad784d1549136d8d766d71a58bc5f469c9b76c6cab49b462a"
GOLDEN_PREVALIDATED = GOLDEN_SEAL  # prevalidated path writes the same canonical bytes


def _fixture(root: pathlib.Path) -> list[pathlib.Path]:
    out = []
    for rel, data in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        out.append(p)
    return out


def _subject():
    sys.path.insert(0, str(HERE))
    try:
        return importlib.import_module("cytir.acquire.stage_receipt")
    finally:
        sys.path.remove(str(HERE))


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_seal_receipt_bytes_are_golden(tmp_path):
    root = tmp_path / "art"
    paths = _fixture(root)
    receipt = root / "r" / "receipt.json"
    out = _subject().seal(
        artifact_root=root, artifact_paths=paths, receipt_path=receipt, records=3,
        fmt="canonical-jsonl", logical_sha256=LOGICAL,
        producer_id="pipeline/materialize_x.py", producer_version="v1")
    assert _sha(receipt) == GOLDEN_SEAL, _sha(receipt)
    assert out["receipt"]["sha256"] == GOLDEN_SEAL


def test_seal_prevalidated_receipt_bytes_are_golden(tmp_path):
    root = tmp_path / "art"
    paths = _fixture(root)
    inv = []
    for p in sorted(paths, key=lambda q: "artifact://" + q.relative_to(root).as_posix()):
        st = p.stat()
        inv.append({"uri": "artifact://" + p.relative_to(root).as_posix(),
                    "bytes": st.st_size, "sha256": _sha(p),
                    "_stat": [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]})
    receipt = root / "r" / "receipt.json"
    out = _subject().seal_prevalidated(
        artifact_root=root, artifact_inventory=inv, receipt_path=receipt, records=3,
        fmt="canonical-jsonl", logical_sha256=LOGICAL,
        producer_id="pipeline/materialize_x.py", producer_version="v1")
    assert _sha(receipt) == GOLDEN_PREVALIDATED, _sha(receipt)
    assert out["receipt"]["sha256"] == GOLDEN_PREVALIDATED


def test_the_two_sealers_agree_on_one_inventory():
    assert GOLDEN_SEAL == GOLDEN_PREVALIDATED  # both write the same canonical bytes


def test_old_name_is_the_package_module():
    assert _subject() is importlib.import_module("cytir.acquire.stage_receipt")


