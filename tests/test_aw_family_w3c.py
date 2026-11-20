"""`w3c_exact` on the family Protocol (task 1456, S7 L2): parity with `materialize`.

No network. A SYNTHETIC sealed artifact root: four Route-C route rows (a /TR/ HTML report, a
non-/TR/ W3C object, a quarantined direct response, a /TR/ PDF). The PINS below were produced ONCE
by the unchanged extractor `cytir.extract.routec.w3c_technical_reports.materialize` on this fixture;
the adapter must reproduce them. Pins: registered and chunk on the declared policy, seal PARTIAL; every unit reaches exactly one
terminal with the extractor's own reason; the parent markdown, document id and parent digest equal
the sealed row; the outlinks equal the sealed outlink rows; consent and licence come from the row;
a tampered body is refused.
"""
from __future__ import annotations

import gzip
import hashlib
import json

import pytest

from cytir.acquire import preserved_stage_io as PS
from cytir.extract.routec import common as ROUTEC
from cytir.extract.routec import w3c_technical_reports as W3C
from cytir.sources.stage_out import OFFLINE_ROUTE
from cytir.sources import contract as contract_mod
from cytir.sources import registry
from cytir.sources.family import Doc, RunContext, SourceFamily, UnitExcluded
from cytir.sources.families.w3c import W3cExactFamily

TR = "https://www.w3.org/TR/2017/WD-html-aria-20170909"
BLOG = "https://www.w3.org/blog/2017/spec"
BAD = "https://www.w3.org/TR/quarantined-report"
PDF = "https://www.w3.org/TR/report.pdf"
HTML = ("<html lang='en'><head><title>ARIA in HTML</title>"
        "<link rel='canonical' href='/TR/html-aria/'></head><body><main><h1>ARIA in HTML</h1>"
        "<p>This specification defines the use of ARIA attributes in HTML documents.</p>"
        "<p>Authors must follow the rules in <a href='https://www.w3.org/TR/wai-aria/'>WAI-ARIA</a> "
        "and <a href='/TR/html/' rel='nofollow'>HTML</a>.</p></main></body></html>").encode()

# produced once from the unchanged extractor, run on this synthetic fixture
PIN_DOCUMENT_ID = "246f560663b0568ed1ee9cfabc3f43da380488924123fc316af0e454ff1e2723"
PIN_MARKDOWN_SHA = "7a1af01c2469e8c576929139505623d9482cdce8be6f90667de028ffc8d6a41b"
PIN_BODY_SHA = hashlib.sha256(HTML).hexdigest()
PIN_MARKDOWN_STREAM = "66bb045694e3d56a7cb6b3a0d509bc36c3527067c294518e10a4dcf74f05e60c"


def _row(fid="w3c_exact") -> dict:
    return contract_mod.by_id(contract_mod.load())[fid]


def _artifacts(root, tag, body, ctype):
    arts = {"response_body": PS.store_artifact(root, "w3c", f"{tag}.body", body),
            "response_headers": PS.store_artifact(
                root, "w3c", f"{tag}.headers", json.dumps([["Content-Type", ctype]]).encode()),
            "transport": PS.store_artifact(root, "w3c", f"{tag}.transport", b'{"http_status":200}'),
            "consent": PS.store_artifact(root, "w3c", f"{tag}.consent", b'{"search_ok":true}')}
    receipt = root / f"terminal/{tag}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({"tag": tag}))
    return arts, ROUTEC.binding(receipt, root)


@pytest.fixture
def sealed(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    rows = []
    for url, tag, body, ctype, status, pts in (
            (TR, "tr", HTML, "text/html; charset=utf-8", "complete", 3),
            (BLOG, "blog", HTML, "text/html", "complete", 2),
            (BAD, "bad", HTML, "text/html", "quarantined", 1),
            (PDF, "pdf", b"%PDF-1.4", "application/pdf", "complete", 4)):
        arts, terminal = _artifacts(root, tag, body, ctype)
        rows.append({"route_id": W3C.ROUTE_ID, "url": url, "production_points": pts,
                     "direct_status": status, "final_url": url + "/",
                     "original_artifacts": arts, "terminal_receipt": terminal})
    rows.append({"route_id": "other-route", "url": "https://example.org/x", "production_points": 9})
    routes = root / "routes.jsonl.gz"
    with gzip.open(routes, "wt", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    return root, routes


def _ctx(sealed, tmp_path, contract=None) -> RunContext:
    root, routes = sealed
    return RunContext(out_dir=tmp_path / "out", today="2026-10-30", skip_consent=True,
                      options={"artifact_root": str(root), "routes": str(routes)},
                      contract=contract or _row(), log=lambda m: None)


def _lines(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(x) for x in fh]


def _terminals(fam, ctx):
    out = {}
    for unit in fam.discover(ctx):
        try:
            (cap,) = fam.acquire(unit, ctx)
            out[unit.url] = fam.extract(cap, ctx)
        except UnitExcluded as exc:
            out[unit.url] = exc.reason
    return out


def test_registered_conformant_and_partial():
    assert "w3c_exact" in registry.registered()
    adapter, row, spec = registry.admit("w3c_exact")
    assert isinstance(adapter, SourceFamily)
    assert row["family_id"] == "w3c_exact" and adapter.route == OFFLINE_ROUTE
    assert row["chunk_policy"]["sha256"].startswith("7dcb54c4") and row["chunk_policy"]["epoch"] == 4
    assert not getattr(adapter, "PARTIAL", ())


def test_every_unit_reaches_one_terminal_with_the_extractors_reason(sealed, tmp_path):
    ctx = _ctx(sealed, tmp_path)
    got = _terminals(W3cExactFamily(), ctx)
    assert ctx.notes["route_urls"] == 4 and ctx.notes["route_reported_production_points"] == 10
    assert got[BLOG] == "non_TR_W3C_object_requires_separate_projection"
    assert got[BAD] == "terminal_source_response_quarantined"
    assert got[PDF] == "non_HTML_TR_response_requires_media_specific_projection"
    assert isinstance(got[TR], list) and len(got[TR]) == 1


def test_parity_with_materialize(sealed, tmp_path):
    root, routes = sealed
    out = root / "w3c-run"
    run = W3C.materialize(artifact_root=root, routes_path=routes, output_root=out)
    (parent,) = _lines(out / "markdown/w3c-technical-reports.jsonl.gz")
    links = _lines(out / "outlinks/w3c-technical-report-links.jsonl.gz")
    assert run["counts"]["residual_urls"] == 3
    ctx = _ctx(sealed, tmp_path)
    (doc,) = _terminals(W3cExactFamily(), ctx)[TR]
    assert doc.text == parent["markdown"] and doc.title == parent["title"]
    assert doc.lang == parent["lang"] and doc.source_text == doc.text
    x = doc.extra
    assert x["document_id"] == parent["document_id"] == PIN_DOCUMENT_ID
    assert x["markdown_sha256"] == parent["markdown_sha256"] == PIN_MARKDOWN_SHA
    assert x["parent_source_sha256"] == parent["parent_source_sha256"] == PIN_BODY_SHA
    assert x["parent_identity"] == parent["parent_identity"]
    assert x["source_version"] == parent["source_version"]
    assert x["source_version"]["declared_canonical_url"] == "https://www.w3.org/TR/html-aria/"
    assert x["extraction_mode"] == parent["extraction"]["mode"]
    assert x["source_id"] == parent["source_id"] and x["source_profile"] == parent["source_profile"]
    assert x["reported_production_points"] == parent["reported_production_points"] == 3
    assert [(l["url"], l["nofollow"], l["anchor_or_context"]) for l in x["outlinks"]] == \
           [(l["target_url"], l["nofollow"], l["anchor_or_context"]) for l in links]
    assert run["stages"]["markdown"]["logical_sha256"] == PIN_MARKDOWN_STREAM


def test_terms_come_from_the_contract_row(sealed, tmp_path):
    ctx = _ctx(sealed, tmp_path)
    W3cExactFamily().discover(ctx)
    t, row = ctx.notes["contract_terms"], _row()
    assert t["search_ok"] is row["consent"]["search_ok"]
    assert t["licence_basis"] == row["licence"]["basis"]
    with pytest.raises(ValueError, match="contract row"):
        W3cExactFamily().discover(_ctx(sealed, tmp_path, {"family_id": "w3c_exact"}))


def test_a_tampered_body_is_refused_not_excluded(sealed, tmp_path):
    root, _ = sealed
    ctx = _ctx(sealed, tmp_path)
    fam = W3cExactFamily()
    unit = next(u for u in fam.discover(ctx) if u.url == TR)
    body = PS.artifact_path(root, unit.meta["original_artifacts"]["response_body"])
    body.write_bytes(body.read_bytes() + b"x")
    with pytest.raises(Exception) as err:
        fam.acquire(unit, ctx)
    assert not isinstance(err.value, UnitExcluded)


def test_chunk_runs_under_the_declared_policy_with_the_stage_anchor(sealed, tmp_path, word_only_chunking_allowed):
    ctx = _ctx(sealed, tmp_path)
    fam = W3cExactFamily()
    with pytest.raises(RuntimeError, match="before preflight"):
        fam.chunk(None, ctx)
    fam.preflight(ctx)
    text = "# Title\n\n" + "\n\n".join(f"## Section {i}\n\nBody of section {i}. " * 1 for i in range(4)) + "\n"
    doc = Doc(url=TR, text=text, title="T", lang="en", source_text=text, extra={"document_id": "d" * 64})
    recs = fam.chunk(doc, ctx)
    assert recs and [r["chunk_ordinal"] for r in recs] == list(range(len(recs)))
    assert all(r["anchor"] == f"document:{'d' * 64}:chunk-{r['chunk_ordinal']}" for r in recs)
    assert all(r["stamps"]["chunk"] == 4 for r in recs)
    assert not any(k in r for r in recs for k in ("tier", "source_type", "trust_tier"))


def test_seal_streams_delegates_to_materialize(sealed, tmp_path):
    root, routes = sealed
    ref = W3C.materialize(artifact_root=root, routes_path=routes, output_root=root / "ref-run")
    ctx = _ctx(sealed, tmp_path)
    ctx.options["output_root"] = str(root / "adapter-run")
    run = W3cExactFamily().seal_streams(ctx)
    assert run["stages"]["markdown"]["logical_sha256"] == PIN_MARKDOWN_STREAM
    assert run["counts"] == ref["counts"]
    with pytest.raises(ValueError):
        W3cExactFamily().seal_streams(_ctx(sealed, tmp_path))
