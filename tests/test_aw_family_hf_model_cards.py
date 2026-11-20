"""`scratch_ai_hf_model_cards_exact` on the Protocol: rows identical to the legacy driver's (task 1101).

`data/samples/hf-model-cards/` is a sealed five-model list with its fetch cache (a major-org card
with a body, a gated card, a re-quant, a card too short for a body, a model with no cache).
`expected_rows.jsonl` is what the legacy `ingest_hf_model_cards.py parse` wrote for it before the
card shape moved into the package. No network.
"""
from __future__ import annotations

import json
import pathlib

from cytir.sources import contract, run_family

REPO = pathlib.Path(__file__).resolve().parent.parent
SAMPLE = REPO / "data" / "samples" / "hf-model-cards"
OK = {"bucket": "ok", "search_ok": True, "train_ok": True, "status": "ok"}


FAMILY = "scratch_ai_hf_model_cards_exact"


def _declared_contracts(tmp_path):
    """A copy of the shipped contract with the row's live per-origin basis declared. These tests exercise
    the card shape, not the declaration. The shipped file stays untouched."""
    doc = json.loads((REPO / contract.DEFAULT_PATH).read_text(encoding="utf-8"))
    row = next(f for f in doc["families"] if f["family_id"] == FAMILY)
    row["consent"].update(basis="declared: take the host's verdict", resolution="live-per-origin")
    row["unknowns"] = sorted(contract.actual_unknowns(row))
    path = tmp_path / "contracts.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _run(tmp_path, consent=lambda h, d: {**OK, "host": h}):
    return run_family.run_family(
        "scratch_ai_hf_model_cards_exact", out_dir=tmp_path / "o", today="2026-10-23",
        log=lambda m: None, consent=consent, options={"root": str(SAMPLE)},
        contracts_path=_declared_contracts(tmp_path))


def test_rows_are_byte_identical_to_the_legacy_driver(tmp_path):
    _run(tmp_path)
    want = (SAMPLE / "expected_rows.jsonl").read_text(encoding="utf-8").splitlines()
    _, recs = run_family.load_pack(tmp_path / "o" / "pack")
    recs = [{k: v for k, v in r.items() if k != 'family_id'} for r in recs]   # ADR-124: the door's stamp is not the driver's
    assert len(recs) == len(want) == 6
    # the pack orders rows by point id; the driver wrote list order
    assert sorted(json.dumps(r, ensure_ascii=False) for r in recs) == sorted(
        json.dumps(json.loads(x), ensure_ascii=False) for x in want)


def test_each_dropped_model_has_the_drivers_own_reason(tmp_path):
    t = _run(tmp_path)["terminals"]
    assert t["units_in"] == 5 == t["units_admitted"] + t["units_excluded"] + t["units_quarantined"]
    assert t["units_admitted"] == 3
    assert t["excluded_reasons"] == {"no_detail": 1, "quant_spam_skipped": 1}


def test_a_declined_host_leaves_every_model_excluded(tmp_path):
    t = _run(tmp_path, consent=lambda h, d: {"host": h, "bucket": "declined"})["terminals"]
    assert t["units_admitted"] == 0
    assert t["excluded_reasons"] == {"consent_declined:huggingface.co": 5}


def test_a_row_without_a_consent_basis_refuses(tmp_path):
    # a row without a consent basis MUST refuse (task 1406). Task 2262 declared the shipped row's
    # basis, so the refusal is shown on a copy whose basis is put back to `unknown`.
    import pytest
    doc = json.loads((REPO / contract.DEFAULT_PATH).read_text(encoding="utf-8"))
    row = next(f for f in doc["families"] if f["family_id"] == FAMILY)
    row["consent"].update(basis="unknown", resolution="unknown")
    row["unknowns"] = sorted(contract.actual_unknowns(row))
    path = tmp_path / "undeclared.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(run_family.RunError, match="declares no consent basis"):
        run_family.run_family(FAMILY, out_dir=tmp_path / "o", today="2026-10-23",
                              log=lambda m: None, options={"root": str(SAMPLE)},
                              consent=lambda h, d: {**OK, "host": h}, contracts_path=path)
