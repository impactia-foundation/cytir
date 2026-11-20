"""The flip gate reads the leg-read shapes of both sides (task 926, B6; follow-up 3 of task 873).

`cytir.gates.prod_legs` (incumbent, schema aw.prod-leg-read.v2) and `cytir.gates.scratch_legs` (candidate,
aw.scratch-leg-read.v1) write rows; the gate scores them the same way, and takes two `reread` null
pairs per component leg per instrument.
"""
from __future__ import annotations

import json
import pathlib

import pytest

import cytir

from cytir.gates import flip_gate as FG

CACHE = "bd5e59e3abc4b2d324c0ee94079484b72ea6b676b345db8ead3d404902f1c417"


def _prod(ranks):
    rows = [{"id": f"authoritative|g{i}", "cls": "authoritative", "status": "OK", "gold_rank": r}
            for i, r in enumerate(ranks)]
    return {"schema": "aw.prod-leg-read.v2", "k": 10, "rows": rows, "abort": None, "invalid": [],
            "embedder": {"qvec_cache": "c.json", "sha256": CACHE}, "gate_set_version": "v4",
            "target_before": {"collection": "aw_e9"},
            "effective_config": {"fusion_weights": {"sha256": "ab"}}}


def _scratch(ranks):
    doc = _prod(ranks)
    doc.update(schema="aw.scratch-leg-read.v1", collection="aw_e10")
    doc["rows"] = [{"id": r["id"], "cls": r["cls"], "status": "OK", "rank": r["gold_rank"]}
                   for r in doc["rows"]]
    return doc


def test_both_shapes_give_the_same_arm_numbers_and_per_gold_ids_without_the_class_prefix():
    a = FG.normalise_arm(_prod([1, 3, None, 12]), instrument="1418", leg="sparse-only", side="incumbent")
    b = FG.normalise_arm(_scratch([1, 3, None, 12]), instrument="1418", leg="sparse-only", side="candidate")
    for arm in (a, b):
        assert arm["classes"]["authoritative"] == {"hit1": 1, "hitk": 2, "mrr": round((1 + 1 / 3 + 1 / 12) / 4, 4),
                                                   "n_gold": 4}
        assert arm["instrument_digest"] == CACHE and arm["gate_set_version"] == "v4"
        assert arm["per_gold"]["authoritative"]["g1"] == 3
        assert arm["n_errors"] == 0 and arm["weights_sha256"] == "ab"
    assert (a["collection"], b["collection"]) == ("aw_e9", "aw_e10")


def test_an_errored_row_counts_and_an_aborted_read_is_no_arm():
    doc = _prod([1, 2])
    doc["rows"][1]["status"] = "REFUSED"
    assert FG.normalise_arm(doc, instrument="1418", leg="dense-only", side="incumbent")["n_errors"] == 1
    doc["abort"] = "p95 over the limit"
    with pytest.raises(FG.FlipGateError):
        FG.normalise_arm(doc, instrument="1418", leg="dense-only", side="incumbent")


def test_two_reread_null_pairs_per_leg_and_instrument_are_both_used(tmp_path):
    paths = []
    for i, ranks in enumerate(([1, 2], [1, 2], [1, 3], [1, 2])):
        p = tmp_path / f"r{i}.json"
        p.write_text(json.dumps(_scratch(ranks)))
        paths.append(str(p))
    pairs = [FG.load_null_pair(f"reread-{n}:reread:1418:sparse-only={a},{b}")
             for n, (a, b) in (("inc", paths[:2]), ("cand", paths[2:]))]
    contract = json.loads((pathlib.Path(cytir.__file__).resolve().parents[1] / "data" /
                           "flip-gate-contract.v1.json").read_text())
    readings, notes = FG._null_pair_readings(pairs, contract, {"dense-only", "sparse-only"})
    assert [n for n, _k, _d in readings[("1418", "sparse-only")]] == ["reread-inc", "reread-cand"], notes
