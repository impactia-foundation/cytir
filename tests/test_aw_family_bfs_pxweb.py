"""`named_slice:bfs-pxweb` on the family Protocol (task 1412).

No network. The fixture is a hand-written three-response catalog in the sealed L0 layout (a German
and a French table, and a response with no variables). Pins: registered and conformant;
discover/acquire/extract on the sealed responses; a `plan` dry run through the framework; consent
and licence read from the row; and PARITY with the pre-move driver, whose output on the fixture
(`build_passage` and `structured_record` for every response) was digested before the move
(driver blob 03dfef4b99ac279e7308602272483aaf74e36d11, identical at origin/main and at 9b23bc508).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import shutil

import pytest

from cytir.extract import bfs_pxweb as B
from cytir.sources import contract as contract_mod
from cytir.sources import registry
from cytir.sources.families.bfs_pxweb import BfsPxwebFamily
from cytir.sources.family import RunContext, SourceFamily, UnitExcluded
from cytir.sources.run_family import run_family

FIXTURE = pathlib.Path(__file__).resolve().parent / "data" / "bfs_pxweb"
#: sha256 of json.dumps(rows, sort_keys=True, ensure_ascii=False), rows = [[asdict(build_passage),
#: structured_record] per catalog entry], computed by the PRE-MOVE driver over FIXTURE
PRE_MOVE_DIGEST = "8add74e9d8dc87e7483a57422ad88f912fd89440acadfd4af666ea9f7374c49b"


def _row() -> dict:
    return contract_mod.by_id(contract_mod.load())["named_slice:bfs-pxweb"]


def _ctx(root, tmp_path, contract=None) -> RunContext:
    return RunContext(out_dir=tmp_path / "out", today="2026-10-30", skip_consent=True,
                      options={"root": str(root)}, contract=contract or _row(), log=lambda m: None)


def _catalog():
    return [json.loads(line) for line in (FIXTURE / "catalog.jsonl").read_text().splitlines()]


def test_registered_and_conformant():
    assert "named_slice:bfs-pxweb" in registry.registered()
    adapter, row, spec = registry.admit("named_slice:bfs-pxweb")
    assert isinstance(adapter, SourceFamily)
    assert row["family_id"] == "named_slice:bfs-pxweb"
    assert not getattr(adapter, "PARTIAL", ())
    assert spec.declared == "chunk-policy-atomic-record.v1"


def test_parser_output_matches_the_pre_move_driver():
    out = []
    for entry in _catalog():
        lang, dbid = entry["url"].split("/")[-3:-1]
        ref = entry["response"]
        meta = json.loads((FIXTURE / ref["uri"].removeprefix("artifact://")).read_bytes())
        passage = B.build_passage(dbid, lang, meta)
        out.append([dataclasses.asdict(passage) if passage else None,
                    B.structured_record(dbid, lang, meta, response_uri=ref["uri"],
                                        response_sha256=ref["sha256"],
                                        retrieved_at=entry["retrieved_at"])])
    assert out[2] == [None, None]          # no variables: no passage, no record
    digest = hashlib.sha256(json.dumps(out, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert digest == PRE_MOVE_DIGEST


def test_discover_acquire_extract_the_responses(tmp_path):
    fam, ctx = BfsPxwebFamily(), _ctx(FIXTURE, tmp_path)
    units = fam.discover(ctx)
    assert [u.key for u in units] == ["de:px-x-0102010000_101", "fr:px-x-0102010000_101",
                                      "en:px-x-9999999999_999"]
    assert ctx.notes["responses"] == 3
    (cap,) = fam.acquire(units[0], ctx)
    assert cap.meta["sha256"] == hashlib.sha256(cap.body).hexdigest()
    (doc,) = fam.extract(cap, ctx)
    assert doc.url == "https://www.pxweb.bfs.admin.ch/api/v1/de/px-x-0102010000_101/px-x-0102010000_101.px"
    assert doc.lang == "de" and doc.text.startswith("# Ständige Wohnbevölkerung")
    row = doc.extra["row"]
    assert row["parent_response"]["sha256"] == cap.meta["sha256"]
    assert row["source_record_id"] == "de:px-x-0102010000_101"


def test_a_response_without_variables_is_excluded(tmp_path):
    fam, ctx = BfsPxwebFamily(), _ctx(FIXTURE, tmp_path)
    (cap,) = fam.acquire(fam.discover(ctx)[2], ctx)
    with pytest.raises(UnitExcluded):
        fam.extract(cap, ctx)


def test_a_response_that_differs_from_its_catalog_entry_is_refused(tmp_path):
    root = tmp_path / "root"
    shutil.copytree(FIXTURE, root)
    victim = root / "responses" / "fr-px-x-0102010000_101.json"
    victim.write_bytes(victim.read_bytes() + b" ")
    fam, ctx = BfsPxwebFamily(), _ctx(root, tmp_path)
    units = fam.discover(ctx)
    with pytest.raises(ValueError, match="differs"):
        fam.acquire(units[1], ctx)


def test_terms_come_from_the_contract_row(tmp_path):
    fam, ctx = BfsPxwebFamily(), _ctx(FIXTURE, tmp_path)
    fam.discover(ctx)
    t, row = ctx.notes["contract_terms"], _row()
    assert t["search_ok"] is row["consent"]["search_ok"]
    assert t["consent_basis"] == row["consent"]["basis"]
    assert t["licence_basis"] == row["licence"]["basis"]
    with pytest.raises(ValueError, match="contract row"):
        BfsPxwebFamily().discover(_ctx(FIXTURE, tmp_path, {"family_id": "named_slice:bfs-pxweb"}))


def test_a_root_without_a_catalog_is_a_wrong_root(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(RuntimeError, match="catalog.jsonl"):
        BfsPxwebFamily().discover(_ctx(tmp_path / "empty", tmp_path))


def test_plan_dry_run_on_the_sealed_fixture(tmp_path):
    out = tmp_path / "plan"
    plan = run_family("named_slice:bfs-pxweb", out_dir=out, today="2026-10-30", dry_run=True,
                      skip_consent=True, options={"root": str(FIXTURE)}, log=lambda m: None)
    assert plan is not None
    assert not (out / "pack").exists()


def test_atomic_policy_is_declared_and_chunk_returns_the_record(tmp_path):
    # task 2260: the row declares chunk-policy-atomic-record.v1 (one record = one passage). The row's
    # consent basis is still `unknown`, so the stage door refuses a whole run (a gate, untouched
    # here); the adapter's own stages on the committed fixture are driven directly instead.
    adapter, row, spec = registry.admit("named_slice:bfs-pxweb")
    assert row["chunk_policy"]["policy_id"] == "chunk-policy-atomic-record.v1"
    ctx = _ctx(FIXTURE, tmp_path)
    recs = []
    for unit in adapter.discover(ctx):
        for cap in adapter.acquire(unit, ctx):
            try:
                docs = adapter.extract(cap, ctx)
            except UnitExcluded:
                continue
            for doc in docs:
                recs += adapter.chunk(doc, ctx)
    assert recs and all(r.get("url") and r.get("text") for r in recs)
    assert len({r["url"] for r in recs}) == len(recs)
