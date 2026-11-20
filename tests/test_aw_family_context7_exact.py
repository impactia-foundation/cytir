"""`context7_exact` on the family Protocol (task 1411): FULL (chunk under the declared 7dcb54c4@4 policy), delegating to the moved materializer.

No network. The fixture is synthetic and built here, in the shape the acquirer seals (one terminal receipt with
one complete document and one quarantined path). Pins: registry conformance; the plan over the receipts; the
extract equals the materializer's own rows; a recovery-only receipt is EXCLUDED; and the moved materializer
writes the same stage bytes as the pre-move script (pinned digests).
"""
from __future__ import annotations

import gzip
import hashlib
import json

import pytest

from cytir.acquire import preserved_stage_io as PS
from cytir.sources import contract as contract_mod
from cytir.sources import registry
from cytir.sources.context7 import materialize as M
from cytir.sources.families.context7_exact import Context7ExactFamily
from cytir.sources.family import RunContext, SourceFamily, UnitExcluded

RAW = b"# Guide\n\nSee [API](https://example.test/api).\n"


def _build(tmp_path):
    art = tmp_path / "aw-corpus"
    src = art / "source"
    src.mkdir(parents=True)
    body = PS.store_artifact(src, "original/blobs", "raw", RAW)
    transport = PS.store_artifact(src, "transport", "json", b"{}\n")
    uri = lambda a: "artifact://source/" + a["uri"].removeprefix("artifact://")   # noqa: E731
    complete = {"source_path": "docs/a.md", "status": "complete", "git_blob_sha1": "b" * 40,
                "artifacts": {"response_body": {k: (uri(body) if k == "uri" else body[k]) for k in
                                                ("uri", "bytes", "sha256", "body_bytes", "body_sha256", "compression")},
                              "transport": {"uri": uri(transport), "bytes": transport["bytes"],
                                            "sha256": transport["sha256"]}}}
    quarantine = {"source_path": "docs/missing.md", "status": "quarantine",
                  "reason": "path_absent_from_preserved_tree", "artifacts": {}}
    (src / "receipts").mkdir()
    for name, docs in (("o--r", [complete, quarantine]), ("p--q", [quarantine])):
        owner_repo = name.replace("--", "/")
        (src / "receipts" / f"{name}.json").write_text(json.dumps({
            "schema": "aw.context7-exact-path-repository-receipt.v1", "status": "terminal",
            "owner_repo": owner_repo, "requested_ref": "main", "resolved_revision": "a" * 40,
            "documents": docs}))
    return art, src


def _ctx(src, art, tmp_path) -> RunContext:
    return RunContext(out_dir=tmp_path / "out", today="2026-10-30", skip_consent=True,
                      options={"root": str(src), "artifact_root": str(art)},
                      contract=contract_mod.by_id(contract_mod.load())["context7_exact"], log=lambda m: None)


def test_registered_conformant_and_full():
    assert "context7_exact" in registry.registered()
    adapter, row, spec = registry.admit("context7_exact")
    assert isinstance(adapter, SourceFamily)
    assert not getattr(adapter, "PARTIAL", ())
    assert row["chunk_policy"]["sha256"].startswith("7dcb54c4") and row["chunk_policy"]["epoch"] == 4
    assert row["consent"]["search_ok"] is None and row["licence"]["basis"] == "unknown"   # read from the row, none invented


def test_plan_over_the_sealed_receipts(tmp_path):
    art, src = _build(tmp_path)
    fam, ctx = Context7ExactFamily(), _ctx(src, art, tmp_path)
    units = list(fam.discover(ctx))
    assert [u.key for u in units] == ["o--r", "p--q"] and ctx.notes["receipts"] == 2
    assert all(u.url.startswith("https://github.com/") for u in units)
    (cap,) = fam.acquire(units[0], ctx)
    assert cap.body == (src / "receipts" / "o--r.json").read_bytes()
    (doc,) = fam.extract(cap, ctx)
    md, ol, rec = M.receipt_rows(src / "receipts" / "o--r.json", art)
    assert doc.text == md[0]["markdown"] and doc.title == "Guide" and doc.url == md[0]["url"]
    assert doc.extra["row"] == md[0] and doc.extra["outlinks"] == ol and len(rec) == 1
    assert ol[0]["target_url"] == "https://example.test/api"


def test_a_recovery_only_receipt_is_excluded_not_empty(tmp_path):
    art, src = _build(tmp_path)
    fam, ctx = Context7ExactFamily(), _ctx(src, art, tmp_path)
    units = list(fam.discover(ctx))
    (cap,) = fam.acquire(units[1], ctx)
    with pytest.raises(UnitExcluded, match="recovery_required:1"):
        fam.extract(cap, ctx)


def test_chunk_runs_under_the_declared_policy_with_the_stage_anchor(tmp_path, word_only_chunking_allowed):
    art, src = _build(tmp_path)
    fam, ctx = Context7ExactFamily(), _ctx(src, art, tmp_path)
    with pytest.raises(RuntimeError, match="before preflight"):
        fam.chunk(None, ctx)
    fam.preflight(ctx)
    (unit, _) = list(fam.discover(ctx))
    (cap,) = fam.acquire(unit, ctx)
    (doc,) = fam.extract(cap, ctx)
    recs = fam.chunk(doc, ctx)
    assert recs and [r["chunk_ordinal"] for r in recs] == list(range(len(recs)))
    assert all(r["anchor"] == f"document:{doc.extra['row']['document_id']}:chunk-{r['chunk_ordinal']}" for r in recs)
    assert all(r["stamps"]["chunk"] == 4 for r in recs)
    assert not any(k in r for r in recs for k in ("tier", "source_type", "trust_tier"))
    assert "See [API](https://example.test/api)." in recs[0]["text"]



def test_missing_options_refuse(tmp_path):
    art, src = _build(tmp_path)
    ctx = _ctx(src, art, tmp_path)
    ctx.options.pop("artifact_root")
    with pytest.raises(ValueError, match="artifact_root"):
        Context7ExactFamily().discover(ctx)


def _run_materialize(mod, art, src, out, monkeypatch):
    run = src / "RUN.json"
    run.write_text(json.dumps({"schema": "aw.context7-exact-path-acquisition-run.v1", "status": "complete",
                               "counts": {"tranche_repository_ref_pairs": 2,
                                          "document_statuses": {"complete": 1, "quarantine": 2}},
                               "stages": {"original": {"receipt": {"uri": "artifact://fake"}}}}))
    monkeypatch.setattr(mod, "_validate_stage", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_upstream_evidence", lambda **kw: (json.loads(run.read_text()), [run]))
    monkeypatch.setattr(mod.SS, "seal", lambda **kw: {"records": kw["records"], "logical_sha256": kw["logical_sha256"]})
    return mod.materialize(source_root=src, source_run=run, output_root=out, artifact_root=art,
                           expected_complete=1, expected_recovery=2)


def _stage_bytes(out):
    return {name: gzip.decompress(next((out / "stages" / name / "data").rglob("*.gz")).read_bytes())
            for name in ("markdown", "outlinks", "recovery")}


# reference digests from the pre-move script at 9b23bc508, produced once from a scratch copy on 2026-10-30
STAGE_SHA256 = {
    "markdown": "6b14a1aeb1bd605b3dcfe550b0d5a6be1eba31c94a07359fd7bdf61897f6aacd",
    "outlinks": "f46bfcb6a1193182db2b800d8cef2a5a0951e9256d5f4b729d18a66628d79b6f",
    "recovery": "b3afe722ae2735dfda4e2491db3797ab5ed8f0172518ce304b705eea543e5e30",
}
# receipt_rows has no pre-move twin; pinned from the moved module (its output feeds the stages above)
RECEIPT_ROWS_SHA256 = "f5dc673fad74978653065af4ca1a2e28a9e11cf69f141c5aac4d96917dfe0043"


def test_moved_materializer_writes_the_pinned_stage_bytes(tmp_path, monkeypatch):
    art, src = _build(tmp_path)
    got = _run_materialize(M, art, src, art / "derived", monkeypatch)
    assert got["counts"]["markdown_documents"] == 1
    digests = {k: hashlib.sha256(v).hexdigest() for k, v in _stage_bytes(art / "derived").items()}
    assert digests == STAGE_SHA256
    rows = M.receipt_rows(src / "receipts" / "o--r.json", art)
    assert hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest() == RECEIPT_ROWS_SHA256
