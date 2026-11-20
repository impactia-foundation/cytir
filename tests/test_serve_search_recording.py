"""S10 step 8b (task 1125): the per-stage test for stage 2, the levers and the response.

`testdata/search_recordings.json.gz` holds whole `retrieval.search()` calls recorded on the BASE tree of the
step (the code before the move; see `search_recording.py`), on the build-box twin: the arguments, the
embedding, every Qdrant call, the constants it ran under, the returned list and the deep-tier side
channels (per-candidate score debug, component ledger, query-plan trace). Each replays offline against the
CURRENT code: the client calls must be the same calls, in the same order, and the result and the side
channels must equal the recording byte for byte. The arms cover the default serving flags and the lever
flags (name gate, ref-canon, name density and coverage, required terms, primacy, trust floor, hub gate) under
`gated`, `rank_scaled` and `multiplicative` trust (the default is `additive`), and the page cap.

NOT covered, by design: the tokenizer (it answers from the recording: a decompounder regression is not
caught here) and the levers that need trained artifacts (rank heads, country map): they are inert in this
fixture, as in a checkout without the artifacts. The parity gate on the twin covers both.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import search_recording as sr  # noqa: E402
from cytir.serve import (fusion, geo, heads, legs, levers, models, plan, qdrant_io, respond,  # noqa: E402
                       retrieval, schema, tables, trace)

FIXTURE = HERE / "testdata" / "search_recordings.json.gz"
MODS = {"retrieval": retrieval, "legs": legs, "fusion": fusion, "levers": levers, "respond": respond,
        "trace": trace, "plan": plan, "heads": heads, "geo": geo, "tables": tables, "schema": schema,
        "qdrant_io": qdrant_io}
DOC = json.loads(gzip.decompress(FIXTURE.read_bytes()))
RECS = DOC["records"]


def test_the_recordings_cover_the_stage_2_paths():
    assert len(RECS) >= 20
    arms = {r["arm"] for r in RECS}
    assert {"served-defaults", "levers-armed-gated", "levers-armed-rank-scaled",
            "levers-armed-multiplicative", "pagecap-agg"} <= arms
    modes = {r["constants"]["scalars"]["_TRUST_MODE"] for r in RECS}
    assert modes == {"additive", "gated", "rank_scaled", "multiplicative"}
    assert any(r["constants"]["scalars"]["_NAMEGATE"] for r in RECS)
    assert any(r["constants"]["scalars"]["_PRIMACY"] for r in RECS)
    # the coverage probe is the one stage-2 client call: some recording must contain a second batch call
    assert any(sum(c["method"] == "query_batch_points" for c in r["calls"]) >= 2 for r in RECS)
    assert all(r["result_text"] != "[]" for r in RECS)


@pytest.mark.parametrize("i", range(len(RECS)))
def test_search_replays_byte_identical(i, monkeypatch):
    rec = RECS[i]
    text, side, client = sr.replay(rec, monkeypatch, MODS, models, schema)
    assert client.exhausted(), "search() made fewer client calls than recorded"
    assert text == rec["result_text"]
    assert sr.mask_timing(side) == sr.mask_timing(rec["side"])
