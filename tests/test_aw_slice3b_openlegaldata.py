#!/usr/bin/env python3
"""Task 696 slice 3b (2/3) -- the OpenLegalData materializer moved to `cytir.extract.openlegaldata`.

What the move may not change: the sealed output. The golden values below were taken from the
OLD `materialize_openlegaldata.py` on the real fixture rows (the same 25 + 3 real
rows `test_materialize_openlegaldata.py` uses) BEFORE the move; the test is green on the old
code and on the moved code. They pin the data a later stage hashes: the emitted markdown rows
(decompressed bytes), the stage's `logical_sha256`, and the `producer` label written into the
receipt (still the old path string: it is a provenance label, never opened).
"""
from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
FIXTURE = HERE / "fixtures" / "openlegaldata" / "task633_openlegaldata_sample_rows.json"

GOLDEN_MARKDOWN_ROWS_SHA256 = "3eb39e7389652cf853b3f1486066aae9960d5ca37336e3965277501559496d55"
GOLDEN_MARKDOWN_LOGICAL_SHA256 = "3eb39e7389652cf853b3f1486066aae9960d5ca37336e3965277501559496d55"
GOLDEN_RECORDS = 25


def _subject():
    sys.path.insert(0, str(HERE))
    try:
        return importlib.import_module("cytir.extract.openlegaldata")
    finally:
        sys.path.remove(str(HERE))


def _acquisition(tmp_path: pathlib.Path, subject):
    import duckdb
    from cytir.acquire.artifact_io import atomic_publish, sha256_file

    root = tmp_path / "acquisition"
    pq = root / "original" / "parquet" / "dump-20260520"
    pq.mkdir(parents=True)
    shard = pq / "train-00000-of-00054.parquet"
    con = duckdb.connect()
    con.execute(f"COPY (SELECT * FROM read_json_auto('{FIXTURE}')) TO '{shard}' (FORMAT PARQUET)")
    con.close()
    receipt = {
        "schema": "aw.stage-receipt.v1", "records": 1, "bytes": shard.stat().st_size,
        "format": "exact-openlegaldata-hf-parquet+dataset-info+HF-HEAD+range-transport",
        "logical_sha256": sha256_file(shard), "deterministic_encoding": True,
        "producer": {"id": "pipeline/acquire_openlegaldata_hf_dump.py",
                     "version": "openlegaldata-hf-dump-acquisition.v1"},
        "artifacts": [{"uri": "artifact://original/parquet/dump-20260520/"
                              "train-00000-of-00054.parquet",
                       "bytes": shard.stat().st_size, "sha256": sha256_file(shard)}],
    }
    rp = root / "stages" / "original" / "receipt.json"
    atomic_publish(rp, (json.dumps(receipt) + "\n").encode())
    stage = {"bytes": receipt["bytes"], "deterministic_encoding": True,
             "format": receipt["format"], "logical_sha256": receipt["logical_sha256"],
             "producer": receipt["producer"],
             "receipt": {"uri": "artifact://stages/original/receipt.json",
                         "bytes": rp.stat().st_size, "sha256": sha256_file(rp)},
             "records": 1}
    run = root / "RUN.json"
    atomic_publish(run, (json.dumps({"schema": subject.ACQUISITION_SCHEMA, "status": "complete",
                                     "stages": {"original": stage}}) + "\n").encode())
    return root, run


def _materialize(tmp_path, subject):
    acq, run = _acquisition(tmp_path, subject)
    art = tmp_path / "corpus"
    out = art / "stages" / "openlegaldata-test"
    result = subject.materialize(artifact_root=art, acquisition_root=acq,
                                 acquisition_run_path=run, shard_indices=[0], output_root=out)
    return art, result


def test_the_sealed_markdown_rows_are_golden(tmp_path):
    subject = _subject()
    art, result = _materialize(tmp_path, subject)
    md = result["stages"]["markdown"]
    receipt = json.loads((art / md["receipt"]["uri"].removeprefix("artifact://")).read_text())
    assert receipt["producer"]["id"] == "pipeline/materialize_openlegaldata.py"
    assert receipt["records"] == GOLDEN_RECORDS
    rows = b""
    for a in receipt["artifacts"]:
        with gzip.open(art / a["uri"].removeprefix("artifact://"), "rb") as fh:
            rows += fh.read()
    assert hashlib.sha256(rows).hexdigest() == GOLDEN_MARKDOWN_ROWS_SHA256
    assert receipt["logical_sha256"] == GOLDEN_MARKDOWN_LOGICAL_SHA256


def test_producer_label_is_the_old_path_string():
    assert _subject().PRODUCER == "pipeline/materialize_openlegaldata.py"


def test_package_module_starts_as_a_script():
    r = subprocess.run([sys.executable, "-m", "cytir.extract.openlegaldata", "--help"], cwd=REPO,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "usage:" in r.stdout
