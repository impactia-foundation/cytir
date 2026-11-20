"""Route-C generic preserved-HTML materializers and the `routec_generic_complete_exact` adapter (task 1432, S7).

Parity (PR #1152 standard): the digests below were produced ONCE by running the pre-move materializers
(blobs 4fd220e2 and 592a5e06 at origin/main b49ae3318, task 1432) on THIS fixture; this test runs only the
moved code and asserts it still writes the same bytes. The fixture is synthetic: a few preserved HTML
responses, a reviewed-route registry stub, a stub principal-content extractor (so the digests do not
move with the extraction library). The seal receipts hold no clock, so the run is deterministic.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import types
from pathlib import Path

import pytest

from cytir.acquire import preserved_stage_io as PS
from cytir.acquire import stage_receipt as SS

DIRECT_VERSION = "exact-url-direct-preserved.v4-consent-gated"

PINNED_CC = {
    "routes": "441518df8e7bab4b91545ee5af450f616f5b2084ebaf9c715aa3e6f18ba4221f",
    "capture_outlinks": "e81c291908e2f825fea43ae0a9aaa38a512cf4250e32d05e8f0a8c45b5b7baab",
    "markdown": "52745fe97579ade6d454cc60c416e3e74ff6638fda17fe78896aa404925693c9",
    "outlinks": "13a2c725cf3860c3241d0ff5fe75b1674accefd375c3e64cfe725f9476e4d8a0",
    "residual": "268c2f7b3abe793b59443fe8393fb93b0d306dc343313676398cf9ba0d295a28",
    "quarantine": "74258bb030a1a73c77af1ef5cdb23472bc8747fa6a9cba16993ba00a694923e6",
    "RUN": "91fb7db27704127b7e99fb132fa2ced13e3852cdd32b6bd2ed719043953ef27c",
}
PINNED_DIRECT = {
    "routes": "330f64294aa924f177679745645a904ade67d346a57bad2b9d32f17137cb961b",
    "markdown": "ef1bcc4348b47313b96a4724c19153e102700acd92abf3e19096d6c8b7c470aa",
    "outlinks": "7efe7574c9b8e09417a8ccf3eb94fa1fe3c3b1713bce10743c5a1c71b9fde55c",
    "residual": "d5c069d9184fcc777c464909c56f9d1c6108d315b144ef7f34a21d494eee32ae",
    "RUN": "9179c30d16c55c15510c4eee574a7da63f0ddb02a8782ae2bfecadb03fcfc06a",
}

GENERIC = ("<html lang='en'><title>Generic title</title><body><main><h1>Guide</h1>"
           "<p>Complete useful content.</p><a href='https://outside.example/evidence'>Evidence</a>"
           "<a href='/same-site' rel='nofollow'>Same</a></main></body></html>")
NOLINKS = "<html><title>Plain</title><body><main><p>No links here at all.</p></main></body></html>"
EMPTY = "<html><title>Hollow</title><body>EMPTY</body></html>"
LAW = "<html><title>Law</title><body>Official act text</body></html>"


def stub_extract(html, url):
    """Deterministic principal-content stand-in: empty for the EMPTY page."""
    if "EMPTY" in html:
        return "", "fixture"
    return f"# {url}\n\nlength {len(html)}\n", "fixture"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, list):
        path.write_text("".join(json.dumps(row) + "\n" for row in value))
    else:
        path.write_text(json.dumps(value) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


REGISTRY = {
    "schema": "aw.routec-source-route-registry.v1",
    "identity_basis": "fixture exact URL routes",
    "routes": [{"route_id": "law", "hosts": ["laws.example"], "path_prefixes": ["/"],
                "source_profile": "anchored_law", "projection": "source-specific-pending",
                "source_route": "official-law"}],
    "fallback": {"route_id": "generic", "source_profile": "web_html",
                 "projection": "generic-web-materialize", "source_route": "cc-html"},
}

ROWS = [
    {"url": "https://laws.example/act/1", "retained_points": 5, "wave": "w1"},
    {"url": "https://generic.example/page", "retained_points": 2, "wave": "w1"},
    {"url": "https://generic.example/plain", "retained_points": 3, "wave": "w1"},
    {"url": "https://generic.example/hollow", "retained_points": 4, "wave": "w2"},
    {"url": "https://generic.example/missing", "retained_points": 1, "wave": "w2"},
]


def _frontier(root: Path, rows: list[dict]) -> tuple[Path, Path]:
    frontier = root / "frontier.jsonl"
    _write(frontier, rows)
    receipt = root / "frontier-receipt.json"
    _write(receipt, {"schema": "aw.exact-url-frontier-receipt.v1",
                     "result": {"frontier": {"urls": len(rows), "sha256": sha256_file(frontier)}}})
    return frontier, receipt


def _registry(root: Path) -> Path:
    path = root / "routes.json"
    _write(path, REGISTRY)
    return path


def build_cc(tmp: Path) -> dict:
    """A sealed signed-S3 Common Crawl run over ROWS (the last URL is a CC miss)."""
    artifact_root = tmp / "artifacts"
    root = artifact_root / "lane"
    cc_root = root / "cc"
    frontier, frontier_receipt = _frontier(root, ROWS)
    htmls = [LAW, GENERIC, NOLINKS, EMPTY, None]
    plan = cc_root / "plan/captures.jsonl"
    _write(plan, [{**row, "state": "MISS" if html is None else "CAPTURE"}
                  for row, html in zip(ROWS, htmls)])
    plan_sha = sha256_file(plan)
    terminals = []
    for row, html in zip(ROWS, htmls):
        path = cc_root / "receipts" / (hashlib.sha256(row["url"].encode()).hexdigest() + ".json")
        if html is None:
            value = {"schema": "aw.exact-url-cc-capture-receipt.v1", "status": "cc_miss",
                     "source_url": row["url"], "plan_sha256": plan_sha, "artifacts": {}}
        else:
            family = cc_root.relative_to(artifact_root).as_posix()
            value = {"schema": "aw.exact-url-cc-capture-receipt.v1", "status": "complete",
                     "source_url": row["url"], "plan_sha256": plan_sha, "artifacts": {
                         "decoded_html": PS.store_artifact(artifact_root, family + "/original/decoded-html",
                                                           "html", html.encode()),
                         "raw_warc": PS.store_artifact(artifact_root, family + "/original/warc",
                                                       "warc.gz", b"warc:" + row["url"].encode()),
                         "transport": PS.store_artifact(artifact_root, family + "/original/transport",
                                                        "json", b"{}\n")}}
        _write(path, value)
        terminals.append(path)
    paths = [plan, *terminals]
    for terminal in terminals:
        paths += [PS.artifact_path(artifact_root, a)
                  for a in json.loads(terminal.read_text())["artifacts"].values()]
    original = SS.seal(artifact_root=artifact_root, artifact_paths=sorted(set(paths)),
                       receipt_path=cc_root / "stages/original/receipt.json", records=4,
                       fmt="fixture-original", logical_sha256="a" * 64,
                       producer_id="fixture", producer_version="v1")
    run = cc_root / "RUN.json"
    _write(run, {"schema": "aw.exact-url-cc-preserved-run.v1", "status": "complete",
                 "counts": {"frontier": 5, "cc_captures": 4, "cc_misses": 1},
                 "stages": {"original": original}})
    return {"artifact_root": artifact_root, "frontier": frontier,
            "frontier_receipt": frontier_receipt, "cc_root": cc_root, "cc_run_path": run,
            "route_registry_path": _registry(root), "output": artifact_root / "materialized"}


DIRECT_CASES = [
    # (row, status, content type, body, consent)
    (ROWS[1], "complete", "text/html; charset=utf-8", GENERIC, {"status": "ok", "search_ok": True}),
    (ROWS[2], "complete", "text/html", NOLINKS, {"status": "ok", "search_ok": True}),
    (ROWS[3], "complete", "text/html", EMPTY, {"status": "ok", "search_ok": True}),
    (ROWS[0], "complete", "text/html", LAW, {"status": "ok", "search_ok": True}),
    (ROWS[4], "complete", "application/pdf", "%PDF", {"status": "ok", "search_ok": True}),
    ({"url": "https://declined.example/x", "retained_points": 6, "wave": "w3"}, "quarantine",
     "", "", {"status": "ok", "search_ok": False}),
]


def build_direct(tmp: Path) -> dict:
    """A sealed consent-gated direct run: HTML, empty page, source-specific URL, a PDF, a decline."""
    artifact_root = tmp / "artifacts"
    root = artifact_root / "lane"
    direct_root = root / "direct"
    rows = [case[0] for case in DIRECT_CASES]
    frontier, frontier_receipt = _frontier(root, rows)
    family = direct_root.relative_to(artifact_root).as_posix()
    receipts, paths, counts = [], [], {"complete": 0, "quarantine": 0}
    for row, status, ctype, body, consent in DIRECT_CASES:
        url = row["url"]
        arts = {"consent": PS.store_artifact(artifact_root, family + "/original/consent", "json",
                                             json.dumps(consent).encode())}
        if status == "complete":
            arts["response_body"] = PS.store_artifact(artifact_root, family + "/original/body", "html",
                                                      body.encode())
            arts["response_headers"] = PS.store_artifact(
                artifact_root, family + "/original/headers", "json",
                json.dumps([["Content-Type", ctype]]).encode())
            arts["transport"] = PS.store_artifact(artifact_root, family + "/original/transport",
                                                  "json", b"{}\n")
        value = {"schema": "aw.exact-url-direct-receipt.v1", "version": DIRECT_VERSION,
                 "status": status, "source_url": url, "final_url": url, "artifacts": arts}
        if status == "quarantine":
            value["quarantine_reason"] = "consent_declined"
        path = direct_root / "receipts" / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        _write(path, value)
        counts[status] += 1
        receipts.append(path)
        paths += [PS.artifact_path(artifact_root, a) for a in arts.values()]
    qfile = direct_root / "quarantine/q.json"
    _write(qfile, {"declined": 1})
    original = SS.seal(artifact_root=artifact_root, artifact_paths=sorted(set(paths + receipts)),
                       receipt_path=direct_root / "stages/original/receipt.json",
                       records=len(rows), fmt="fixture-direct-original", logical_sha256="b" * 64,
                       producer_id="fixture", producer_version="v1")
    quarantine = SS.seal(artifact_root=artifact_root, artifact_paths=[qfile],
                         receipt_path=direct_root / "stages/quarantine/receipt.json", records=1,
                         fmt="fixture-direct-quarantine", logical_sha256="c" * 64,
                         producer_id="fixture", producer_version="v1")
    _write(direct_root / "RUN.json", {
        "schema": "aw.exact-url-direct-preserved-run.v1", "version": DIRECT_VERSION,
        "status": "complete", "counts": {"frontier": len(rows), **counts},
        "stages": {"original": original, "quarantine": quarantine}})
    return {"artifact_root": artifact_root, "frontier": frontier,
            "frontier_receipt": frontier_receipt, "direct_root": direct_root,
            "route_registry_path": _registry(root), "output": artifact_root / "materialized"}


CC_STREAMS = {"routes": "routes/all.jsonl.gz", "capture_outlinks": "outlinks/capture-context.jsonl.gz",
              "markdown": "markdown/generic-web.jsonl.gz",
              "outlinks": "outlinks/generic-web-parent-bound.jsonl.gz",
              "residual": "residual/cc-misses.jsonl.gz", "quarantine": "quarantine/generic-web.jsonl.gz"}
DIRECT_STREAMS = {"routes": "routes/all.jsonl.gz", "markdown": "markdown/generic-web.jsonl.gz",
                  "outlinks": "outlinks/generic-web-parent-bound.jsonl.gz",
                  "residual": "residual/non-generic-or-unresolved.jsonl.gz"}


def digests(output: Path, streams: dict[str, str]) -> dict[str, str]:
    """sha256 of each decompressed row stream and of RUN.json."""
    out = {name: hashlib.sha256(gzip.decompress((output / rel).read_bytes())).hexdigest()
           for name, rel in streams.items()}
    out["RUN"] = sha256_file(output / "RUN.json")
    return out


def stand_in_direct():
    """Stand-in for the direct acquirer's validators (same two entry points, same consent rule)."""
    def receipt(rec, artifact_root):
        if rec.get("status") not in {"complete", "quarantine"}:
            raise ValueError("direct receipt is not current consent-gated evidence")
        art = (rec.get("artifacts") or {}).get("consent")
        PS.validate_artifact(artifact_root, art)
        consent = json.loads(gzip.decompress(PS.artifact_path(artifact_root, art).read_bytes()))
        if rec["status"] == "complete" and not (consent.get("status") == "ok"
                                                and consent.get("search_ok") is True):
            raise ValueError("complete direct receipt lacks affirmative search consent")
        return consent

    def run(root, artifact_root, *, deep=True):
        value = json.loads((root / "RUN.json").read_text())
        if value.get("schema") != "aw.exact-url-direct-preserved-run.v1":
            raise ValueError("direct RUN is not current consent-gated evidence")
        return value
    return types.SimpleNamespace(validate_consent_gated_receipt=receipt, validate_consent_gated_run=run)


@pytest.fixture
def subject(monkeypatch):
    from cytir.sources.routec import direct_preserved_html as dm
    from cytir.sources.routec import preserved_html as cm
    monkeypatch.setattr(cm, "extract_content_markdown", stub_extract)
    monkeypatch.setattr(dm, "DIRECT", stand_in_direct())
    return cm, dm


def test_cc_materializer_writes_the_pinned_rows(tmp_path, subject):
    cm, _ = subject
    kw = build_cc(tmp_path)
    result = cm.materialize(**kw)
    assert result["counts"] == {
        "frontier_urls": 5, "production_points": 15, "cc_captures": 4, "cc_misses": 1,
        "generic_markdown_parents": 2, "generic_quarantine": 1, "capture_outlink_rows": 5,
        "outlink_rows": 3}
    assert digests(kw["output"], CC_STREAMS) == PINNED_CC


def test_direct_materializer_writes_the_pinned_rows(tmp_path, subject):
    _, dm = subject
    kw = build_direct(tmp_path)
    result = dm.materialize(**kw)
    assert result["counts"]["generic_markdown_parents"] == 2
    assert result["counts"]["residual_urls"] == 4
    assert digests(kw["output"], DIRECT_STREAMS) == PINNED_DIRECT


def test_the_digests_are_stable_across_two_runs(tmp_path, subject):
    cm, _ = subject
    one, two = build_cc(tmp_path / "a"), build_cc(tmp_path / "b")
    cm.materialize(**one)
    cm.materialize(**two)
    assert digests(one["output"], CC_STREAMS) == digests(two["output"], CC_STREAMS)


# ---------------------------------------------------------------- the adapter (family door)

FAMILY = "routec_generic_complete_exact"


def _ctx(tmp_path, **options):
    from cytir.sources import registry
    from cytir.sources.family import RunContext
    adapter, row, spec = registry.admit(FAMILY)
    return adapter, RunContext(out_dir=tmp_path / "out", today="2026-10-30", options=options,
                               chunk_spec=spec, contract=row, log=lambda m: None)


def _terminals(adapter, ctx):
    """discover, acquire, extract: url -> ('doc', Doc) | ('excluded', reason) | ('quarantined', msg)."""
    from cytir.sources.family import UnitExcluded
    out = {}
    for unit in adapter.discover(ctx):
        try:
            caps = list(adapter.acquire(unit, ctx))
            docs = [d for c in caps for d in adapter.extract(c, ctx)]
            out[unit.url] = ("doc", docs[0])
        except UnitExcluded as exc:
            out[unit.url] = ("excluded", str(exc))
        except (ValueError, RuntimeError) as exc:
            out[unit.url] = ("quarantined", str(exc))
    return out


def test_the_family_is_registered_full_and_declared(tmp_path):
    from cytir.sources import registry
    registry.load_modules()
    assert FAMILY in registry.registered()
    adapter, _ = _ctx(tmp_path)
    assert not getattr(adapter, "PARTIAL", ())
    assert adapter.direct_crawl is False


def _materialize_cc(tmp, cm):
    kw = build_cc(tmp)
    cm.materialize(**kw)
    return kw["output"] / "markdown/generic-web.jsonl.gz"


def test_cc_route_terminals(tmp_path, monkeypatch):
    from cytir.sources.routec import preserved_html as cm
    monkeypatch.setattr(cm, "extract_content_markdown", stub_extract)
    kw = build_cc(tmp_path)
    adapter, ctx = _ctx(tmp_path, artifact_root=kw["artifact_root"], frontier=kw["frontier"],
                        frontier_receipt=kw["frontier_receipt"], route_registry=kw["route_registry_path"],
                        cc_root=kw["cc_root"], cc_run=kw["cc_run_path"])
    got = _terminals(adapter, ctx)
    assert got["https://laws.example/act/1"] == ("excluded", "source-specific-projection-pending")
    assert got["https://generic.example/missing"] == ("excluded", "cc_miss")
    assert got["https://generic.example/hollow"] == ("excluded", "principal-content-extraction-empty")
    kind, doc = got["https://generic.example/page"]
    assert kind == "doc" and doc.lang == "en" and doc.title == "Generic title"
    assert doc.extra["source_id"] == "routec-phase2-all-common-crawl"
    # the adapter and the materializer write the same parent: same document id, same markdown
    with gzip.open(_materialize_cc(tmp_path / "m", cm), "rt") as fh:
        parents = {r["url"]: r for r in map(json.loads, fh)}
    parent = parents["https://generic.example/page"]
    assert (doc.extra["document_id"], doc.text) == (parent["document_id"], parent["markdown"])
    assert ctx.notes["contract_terms"]["search_ok"] is None        # unknown stays unknown


def test_direct_route_terminals(tmp_path, monkeypatch):
    from cytir.sources.routec import direct_preserved_html as dm
    from cytir.sources.routec import preserved_html as cm
    monkeypatch.setattr(cm, "extract_content_markdown", stub_extract)
    monkeypatch.setattr(dm, "DIRECT", stand_in_direct())
    kw = build_direct(tmp_path)
    adapter, ctx = _ctx(tmp_path, artifact_root=kw["artifact_root"], frontier=kw["frontier"],
                        frontier_receipt=kw["frontier_receipt"], route_registry=kw["route_registry_path"],
                        direct_root=kw["direct_root"])
    got = _terminals(adapter, ctx)
    assert got["https://laws.example/act/1"] == ("excluded", "source-specific-projection-pending")
    assert got["https://generic.example/missing"] == ("excluded", "generic-fallback-media-type-is-not-web-html")
    assert got["https://generic.example/hollow"] == ("excluded", "principal-content-extraction-empty")
    assert got["https://declined.example/x"] == ("excluded", "consent_declined")
    kind, doc = got["https://generic.example/page"]
    assert kind == "doc" and doc.extra["source_id"] == "routec-phase2-all-direct-after-cc-miss"
    assert doc.extra["baseline_production_points"] == 2


def test_a_complete_direct_receipt_without_search_consent_quarantines(tmp_path, monkeypatch):
    """The consent boundary: a `complete` receipt whose preserved consent is not affirmative yields no doc."""
    from cytir.sources.routec import direct_preserved_html as dm
    monkeypatch.setattr(dm, "DIRECT", stand_in_direct())
    kw = build_direct(tmp_path)
    url = "https://generic.example/page"
    path = kw["direct_root"] / "receipts" / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    receipt = json.loads(path.read_text())
    receipt["artifacts"]["consent"] = PS.store_artifact(
        kw["artifact_root"], "lane/direct/original/consent", "json",
        json.dumps({"status": "ok", "search_ok": False}).encode())
    path.write_text(json.dumps(receipt) + "\n")
    adapter, ctx = _ctx(tmp_path, artifact_root=kw["artifact_root"], frontier=kw["frontier"],
                        frontier_receipt=kw["frontier_receipt"], route_registry=kw["route_registry_path"],
                        direct_root=kw["direct_root"])
    unit = next(u for u in adapter.discover(ctx) if u.url == url)
    with pytest.raises(ValueError, match="affirmative search consent"):
        list(adapter.acquire(unit, ctx))


def test_the_moved_modules_read_no_identity_and_fetch_nothing():
    """No crawler identity at import, no HTTP client, no path hack."""
    import ast
    import cytir.sources.routec.direct_preserved_html as dm
    import cytir.sources.routec.preserved_html as cm
    for mod in (dm, cm):
        text = Path(mod.__file__).read_text()
        names = {n.id for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Name)}
        assert not names & {"httpx", "requests", "user_agent"}, mod.__name__
        assert "sys.path" not in text and "AW_" not in text


def test_chunk_runs_the_declared_policy_with_the_stage_anchor_and_no_invented_values(tmp_path):
    from cytir.sources import contract as contract_mod
    from cytir.sources.family import Doc, RunContext
    adapter, row, _ = __import__("cytir.sources.registry", fromlist=["admit"]).admit(FAMILY)
    assert not getattr(adapter, "PARTIAL", ())
    assert row["chunk_policy"]["sha256"].startswith("7dcb54c4") and row["chunk_policy"]["epoch"] == 4
    ctx = RunContext(out_dir=tmp_path / "out", today="2026-10-31", skip_consent=True, options={},
                     contract=contract_mod.by_id(contract_mod.load())[FAMILY], log=lambda m: None)
    doc = Doc(url="https://example.test/x", title="T", lang="en", source_text="", extra={"document_id": "d" * 64},
              text="# T\n\n## A\n\nalpha beta gamma.\n\n## B\n\ndelta epsilon.\n")
    with pytest.raises(RuntimeError, match="before preflight"):
        adapter.chunk(doc, ctx)
    adapter.preflight(ctx)
    recs = adapter.chunk(doc, ctx)
    assert recs and [r["chunk_ordinal"] for r in recs] == list(range(len(recs)))
    assert all(r["anchor"] == f"document:{'d' * 64}:chunk-{r['chunk_ordinal']}" for r in recs)
    assert all(r["stamps"]["chunk"] == 4 for r in recs)
    assert not any(k in r for r in recs for k in ("tier", "source_type", "trust_tier"))
