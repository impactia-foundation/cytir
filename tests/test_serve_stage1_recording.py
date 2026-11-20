"""S10 step 8a (task 1116): the per-stage test on deep-tier recordings.

`testdata/stage1_recordings.json.gz` holds recorded `legs.stage1` calls: requests the parity gate
sends (chooser / held-out / probe sets), taken on the build-box twin with the gate's served
environment (`recorded_on` names the collection and the tree). Each replays offline: the Qdrant
calls are answered from the recording and must be the SAME calls (method and arguments), and the
result must equal the recorded result byte for byte. The recordings that carry a flag arm (batched
legs, authority leg, page cap) replay under that arm's constants. Task 1125 added one empty-pool
recording (`lang="zz"` with the card exemption off: no leg returns a point, `stage1` answers None and
`search()` answers []).

NOT covered, by design: the tokenizer. The query tokens (and so the decompounder, `schema._decompound`,
whose output depends on the assets of the machine) answer FROM THE RECORDING, so a regression in the
decompounder or in the IDF table is not caught here; `schema` has its own tests, and the parity gate
runs the real tokenizer on both trees.
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

import stage1_recording as sr  # noqa: E402
from cytir.serve import fusion, legs, plan, qdrant_io, trace  # noqa: E402

FIXTURE = HERE / "testdata" / "stage1_recordings.json.gz"
MODS = {"legs": legs, "fusion": fusion, "plan": plan, "trace": trace, "qdrant_io": qdrant_io}


def _load():
    return json.loads(gzip.decompress(FIXTURE.read_bytes()))


DOC = _load()
RECS = DOC["records"]


def test_the_recordings_cover_the_stage_paths():
    assert len(RECS) >= 20
    paths = set()
    for r in RECS:
        c = r["constants"]["legs"]
        methods = [x["method"] for x in r["calls"]]
        paths.add(("batch" if "query_batch_points" in methods[:1] else "seq"))
        paths.add("two_stage" if c["_TWO_STAGE"] else "legacy")
        paths.add("empty" if r["result_text"] == "null" else "pool")
    assert {"seq", "pool", "two_stage", "empty"} <= paths
    assert sum(r["result_text"] == "null" for r in RECS) >= 1


@pytest.mark.parametrize("i", range(len(RECS)))
def test_stage1_replays_byte_identical(i, monkeypatch):
    rec = RECS[i]
    text, client = sr.replay(rec, monkeypatch, MODS)
    assert client.exhausted(), "stage 1 made fewer client calls than recorded"
    assert text == rec["result_text"]
