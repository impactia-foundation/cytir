"""Task 801: a prod-only resume keeps the twin-first receipt shape; the delta reader and the backfill verb.

Defect: tranche 14's receipt, rewritten by the sealed prod-only resume, held only
`{schema, pack_id, checked_at, manifest_sha, family, status, prod, resumed_prod_only}`, so
`cytir.ops.cc_tranche_delta` (reads `mechanical_plan.rows.distinct_urls`) exited 8 for shape alone.
"""
import gzip
import json

import pytest

from cytir.index import promote_queue as PQ
from cytir.ops import cc_tranche_delta as D
from cytir.ops import landing_check as LC
from cytir.ops import receipt_backfill as RB

ROWS = {"vectors": 10, "projected": 10, "skipped_by_projection": 0, "excluded_undated": 0,
        "excluded_out_of_scope_language": 0, "already_present": 0, "distinct_urls": 4,
        "already_served_urls": None}
PROD = {"rows": dict(ROWS), "identity_check": {"mode": "pack-declared"}, "written": 10,
        "resume": {"present_before": 0}, "resumed_partial": False}


def _resumed(prior):
    """What `drain` writes after a resume: the fresh record + the contract fields."""
    record = {"schema": "aw.promotion-queue-drain.v1", "pack_id": "p", "status": "LANDED", "manifest_sha": "m1",
              "prod": PROD, "resumed_prod_only": True}
    record.update(PQ.resume_receipt_contract(prior, record["prod"], record))
    return record


def test_a_resume_with_no_prior_twin_record_derives_mechanical_plan_from_the_prod_plan():
    rec = _resumed(None)
    assert rec["mechanical_plan"]["rows"]["distinct_urls"] == 4
    assert "derived_from" in rec["mechanical_plan"]
    # kept: prod, resumed_prod_only and the #720 resume fields
    assert rec["resumed_prod_only"] is True and rec["prod"]["resume"] == {"present_before": 0}
    assert rec["status"] == "LANDED"
    # every reader finds what it reads
    assert D._rows_served(rec) and D._distinct_urls(rec) == 4
    acc = LC.accounting(rec["prod"], rec.get("mechanical_content") or {})
    assert acc["vectors"] == 10 and acc["written"] == 10 and acc["balanced"]
    assert LC.accounting(rec.get("twin") or {}, {})["balanced"]      # a missing twin side stays readable


def test_a_resume_merges_the_twin_leg_record_the_prior_receipt_kept():
    prior = {"status": "ABORTED", "manifest_sha": "m1", "mechanical_plan": {"rows": {**ROWS, "distinct_urls": 4}, "identity_check": {}},
             "twin": {"rows": ROWS, "written": 9}, "gate": {"profile": "mechanical"},
             "mechanical_content": {"verdict": "PASS"}, "german_parts": {"verdict": "PASS"}}
    rec = _resumed(prior)
    assert rec["twin"]["written"] == 9 and rec["gate"] == prior["gate"]
    assert rec["mechanical_plan"] is prior["mechanical_plan"] and "derived_from" not in rec["mechanical_plan"]
    assert rec["mechanical_content"] == {"verdict": "PASS"}
    assert rec["status"] == "LANDED" and rec["prod"] is PROD      # the prior status never wins


def test_a_prior_receipt_of_another_manifest_carries_nothing():
    """Review 802 M2: a reseal keeps an ABORTED receipt of the OLD manifest; its plan and counts are not this pack's."""
    old = {"status": "ABORTED", "manifest_sha": "OLD", "mechanical_plan": {"rows": {**ROWS, "distinct_urls": 999}},
           "twin": {"written": 1}, "mechanical_content": {"word_floor_dropped": 77}, "gate": {"profile": "x"}}
    rec = _resumed(old)
    assert rec["mechanical_plan"]["rows"]["distinct_urls"] == 4 and "derived_from" in rec["mechanical_plan"]
    assert not {"twin", "mechanical_content", "gate"} & set(rec)


def test_an_oversized_or_broken_prior_receipt_reads_as_none(tmp_path, monkeypatch):
    (tmp_path / "receipt.json").write_text('{"a": 1}')
    assert PQ._read_prior_receipt(tmp_path) == {"a": 1}
    monkeypatch.setattr(PQ, "MAX_PRIOR_RECEIPT_BYTES", 3)
    assert PQ._read_prior_receipt(tmp_path) is None
    (tmp_path / "receipt.json").write_text("{broken")
    monkeypatch.setattr(PQ, "MAX_PRIOR_RECEIPT_BYTES", 10**6)
    assert PQ._read_prior_receipt(tmp_path) is None


def _write_rec(tmp_path, rec):
    (tmp_path / "pack").mkdir()
    (tmp_path / "pack" / "receipt.json").write_text(json.dumps(rec))
    return tmp_path


@pytest.mark.parametrize("rec,expected", [
    ({"mechanical_plan": {"rows": {"distinct_urls": 7}}, "prod": {"rows": {"distinct_urls": 1}}}, 7),
    ({"distinct_urls": 5}, None),                      # nothing writes a top-level value: not read
    ({"distinct_urls": 5, "prod": {"rows": {"distinct_urls": 9}}}, 9),
    ({"twin": {"rows": {"distinct_urls": 3}}, "prod": {"rows": {"distinct_urls": 1}}}, 3),
    ({"prod": {"rows": {"distinct_urls": 257715}}}, 257715),
    ({"prod": {"rows": {"vectors": 1}}}, None),
    ({"prod": None, "mechanical_plan": {"rows": {"distinct_urls": None}}}, None),
    ({}, None)])
def test_distinct_urls_is_read_from_the_first_present_place(rec, expected):
    assert D._distinct_urls({"status": "LANDED", **rec}) == expected


def test_an_already_landed_resumed_pack_is_measurable_and_a_genuinely_absent_value_still_blocks(tmp_path):
    q = tmp_path
    for pack, rec in (("cc-authority-frontier-w7-tranche14", {"status": "LANDED", "prod": {"rows": ROWS}}),
                      ("cc-authority-frontier-w7-tranche15", {"status": "LANDED", "prod": {"rows": {"vectors": 1}}})):
        (q / pack).mkdir()
        (q / pack / "receipt.json").write_text(json.dumps(rec))
    ok = D.first_pack_state(q, 14)
    assert ok["state"] == "served" and ok["expected_urls"] == 4
    assert D.first_pack_state(q, 15)["expected_urls"] is None        # run_delta -> unmeasurable -> exit 8


def _chunks(tmp_path, urls):
    d = tmp_path / "chunks"
    d.mkdir()
    with gzip.open(d / "a.texts.jsonl.gz", "wt") as fh:
        for u in urls:
            fh.write(json.dumps({"url": u, "point_id": u}) + "\n")
    return d


def test_backfill_adds_the_value_atomically_with_a_note_and_changes_nothing_else(tmp_path):
    chunks = _chunks(tmp_path, ["u1", "u1", "u2", "u3", "u4"])
    old = {"status": "LANDED", "pack_id": "p", "resumed_prod_only": True,
           "prod": {"rows": {**ROWS, "vectors": 5, "projected": 5}, "chunks_dir": str(chunks)}}
    root = _write_rec(tmp_path, old)
    out = RB.backfill(root / "pack", now=lambda: 0, log=lambda *_: None)
    assert out == {"pack_id": "p", "status": "backfilled", "distinct_urls": 4}
    rec = json.loads((root / "pack" / "receipt.json").read_text())
    assert rec["mechanical_plan"]["rows"]["distinct_urls"] == 4
    assert rec["backfilled_by"]["chunks_distinct_urls"] == 4 and rec["backfilled_by"]["at"] == "1970-01-01T00:00:00Z"
    assert {k: v for k, v in rec.items() if k not in ("mechanical_plan", "backfilled_by")} == old
    assert D._distinct_urls(rec) == 4
    assert [p.name for p in (root / "pack").iterdir()] == ["receipt.json"], "no temp file left"
    assert ((root / "pack" / "receipt.json").stat().st_mode & 0o777) == 0o644
    assert RB.backfill(root / "pack", log=lambda *_: None)["status"] == "already_present"


def test_backfill_refuses_a_count_the_chunks_do_not_confirm_and_a_pack_that_is_not_landed(tmp_path):
    chunks = _chunks(tmp_path, ["u1", "u2"])
    rec = {"status": "LANDED", "pack_id": "p", "prod": {"rows": ROWS, "chunks_dir": str(chunks)}}
    root = _write_rec(tmp_path, rec)
    with pytest.raises(RB.BackfillError, match="holds 2"):
        RB.backfill(root / "pack", log=lambda *_: None)
    assert json.loads((root / "pack" / "receipt.json").read_text()) == rec
    (root / "pack" / "receipt.json").write_text(json.dumps({**rec, "status": "REFUSED"}))
    with pytest.raises(RB.BackfillError, match="not LANDED"):
        RB.backfill(root / "pack", log=lambda *_: None)
    assert RB.main([str(root / "pack")]) == 2


def test_the_cli_routes_the_verb(tmp_path, capsys):
    from cytir.cli import main
    assert main(["ops", "receipt", "backfill-distinct-urls", str(tmp_path / "nope")]) == 2
    err = capsys.readouterr().err
    assert "REFUSED: cannot read" in err and "usage: cyt ops cut" not in err    # the verb's own message
