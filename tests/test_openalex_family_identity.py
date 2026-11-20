"""Characterization: the family's extractor makes the bytes the old OpenAlex materializer makes.

`cytir/extract/openalex_academic.py` is the logic of `materialize_openalex_academic_pilot.py` moved into the
package. This test runs the OLD script's own functions (selection SQL, abstract rebuild, `render_record`)
on the committed sample (`data/samples/openalex-academic/`, fifteen real works of one snapshot part) and
compares the sha256 of its output with the pins of `tests/test_aw_family_openalex_academic_dump.py`, which
the new extractor meets on the same part. It goes when the old script goes (task 643 S9).
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("duckdb")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
TESTS = REPO / "tests" / "test_aw_family_openalex_academic_dump.py"
PART = REPO / "data" / "samples" / "openalex-academic" / "updated_date=2026-02-09" / "part_0046.parquet"


def _pins():
    spec = importlib.util.spec_from_file_location("openalex_family_pins", TESTS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_old_script_makes_the_same_bytes_as_the_family_extractor(word_only_chunking_allowed):
    pins = _pins()
    code = f"""
import hashlib, json, pathlib, sys
sys.path.insert(0, {str(REPO)!r}); sys.path.insert(0, {str(HERE)!r})
import cytir.sources.openalex.materialize_pilot as old
from cytir.acquire.artifact_io import canonical_json
rows = old.query_selected_records(pathlib.Path({str(PART)!r}))
sha = hashlib.sha256(pathlib.Path({str(PART)!r}).read_bytes()).hexdigest()
out, recovery = [], []
for raw in rows:
    try:
        out.append(old.render_record(raw, relative_path={pins.REL!r}, parent_sha256=sha)[0])
    except Exception:
        recovery.append(raw["id"])
raw_sha = hashlib.sha256("".join(canonical_json(r) + "\\n" for r in rows).encode()).hexdigest()
md_sha = hashlib.sha256("".join(canonical_json(r) + "\\n" for r in out).encode()).hexdigest()
# the embedded surfaces: the old stage script's own chunking call on the old Markdown
from cytir.core.common import explicit_structured_chunker_selection
from cytir.chunk.generic import chunk_generic_records
from cytir.chunk import chunker as _ck
_ck._ALLOW_WORD_ONLY_FOR_TESTS = True  # test subprocess opt-in (task 2289)
sel = explicit_structured_chunker_selection(
    {str(REPO / "data" / "chunk-policy-struct-md-768w-o0-adjacent-sections-tail-1152-tok2560.v1.json")!r},
    "7dcb54c497e1855ad91ddafe80ff34b1da14f3d23b3af26dc7c7b53ab70ce6bf",
    "ac6544c8a46e00af67e330e85a9028c66b8cfd9a")
dense, sparse = [], []
for row in out:
    for n, u in enumerate(chunk_generic_records(row["markdown"], selection=sel)):
        for acc, v in ((dense, u.dense_text or u.text), (sparse, u.sparse_text or u.text)):
            acc.append(canonical_json({{"url": row["url"], "ordinal": n, "text": v}}) + "\\n")
surf = [hashlib.sha256("".join(a).encode()).hexdigest() for a in (dense, sparse)]
print(json.dumps([raw_sha, md_sha, recovery, surf]))
"""
    # the old script reads the machine's time zone (`created_date` in the raw record JSON); pin it
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                         env={**os.environ, "TZ": "UTC"})
    assert res.returncode == 0, res.stderr[-2000:]
    assert json.loads(res.stdout.splitlines()[-1]) == [
        pins.SELECTED_RAW_SHA256, pins.RENDERED_SHA256, [],
        [pins.DENSE_SURFACE_SHA256, pins.SPARSE_SURFACE_SHA256]]
