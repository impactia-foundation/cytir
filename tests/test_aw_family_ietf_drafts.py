"""`ietf_drafts_exact` on the family door (task 1455, S7).

Parity: the PINNED digests below were produced ONCE by running the unchanged extractor
(`cytir.extract.routec.ietf_internet_drafts.materialize`) on THIS synthetic fixture. The committed test
re-runs the extractor and the adapter and asserts the adapter emits the same bytes (markdown, document id)
and the same terminals. Seal receipts hold no clock, so the run is deterministic.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from cytir.extract.routec import ietf_internet_drafts as IETF

FAMILY = "ietf_drafts_exact"
D1 = "https://datatracker.ietf.org/doc/draft-example"
D2 = "https://datatracker.ietf.org/doc/draft-other"
OLD = "https://datatracker.ietf.org/doc/draft-old"
PAGE1 = ("<html><head><title>draft-example-02 - Example</title></head><body><main>"
         "<h1>Example Protocol draft-example-02</h1><p>Abstract</p><p>Complete abstract.</p>"
         "<a href='https://www.ietf.org/archive/id/draft-example-02.txt'>txt</a></main></body></html>")
PAGE2 = ("<html><head><title>draft-other-00 - Other</title></head><body><main>"
         "<h1>Other draft-other-00</h1><dl><dt>State</dt><dd>Active</dd></dl></main></body></html>")
RFC = "<html><head><title>RFC 9852</title></head><body><main><h1>RFC 9852</h1></main></body></html>"
ROWS = [(D1, D1 + "/", PAGE1, 2), (D2, D2 + "/", PAGE2, 3),
        (OLD, "https://datatracker.ietf.org/doc/rfc9852/", RFC, 4)]

# sha256 of the markdown each draft record renders to, and of the sealed markdown stage, from the
# extractor's own run (pinned once)
PINNED = {
    D1: "8a4e212987791512e61635a45e381270257ae3acc3880da48916ade8ec44e20f",
    D2: "bdd8508b449576ad71ea9ca2b38452e386eefb0c00cd57a5f9630c08d62ee47e",
    "markdown_stage_logical": "294bac31bb5aed32747d722381224543d490f9f82b198955137c67df0406f8cf",
}


def _gz(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as target:
        target.write(data)
    return path


def _artifact(root: Path, path: Path, body: bytes) -> dict:
    blob = path.read_bytes()
    return {"uri": "artifact://" + path.relative_to(root).as_posix(), "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(), "compression": "gzip",
            "body_bytes": len(body), "body_sha256": hashlib.sha256(body).hexdigest()}


def build(tmp: Path) -> dict:
    root = tmp / "artifacts"
    lane = root / "lane"
    routes, resolutions = [], []
    for n, (url, final, html, points) in enumerate(ROWS):
        arts = {}
        for kind, body in (("response_body", html.encode()), ("response_headers", b"content-type: text/html\n"),
                           ("transport", b"{}"), ("consent", b'{"status": "ok"}')):
            arts[kind] = _artifact(root, _gz(lane / f"{n}-{kind}.gz", body), body)
        terminal_path = lane / f"{n}-terminal.json"
        terminal_path.write_text(json.dumps({"url": url}))
        blob = terminal_path.read_bytes()
        terminal = {"uri": "artifact://" + terminal_path.relative_to(root).as_posix(),
                    "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
        routes.append({"url": url, "direct_status": "complete", "production_points": points,
                       "final_url": final, "terminal_receipt": terminal, "original_artifacts": arts})
        resolutions.append({"url": url, "disposition": IETF.DISPOSITION, "reported_production_points": points})
    routes_path = _gz(lane / "routes.jsonl.gz", "".join(json.dumps(r) + "\n" for r in routes).encode())
    resolution_path = _gz(lane / "resolution.jsonl.gz",
                          "".join(json.dumps(r) + "\n" for r in resolutions).encode())
    declaration = lane / "declaration.json"
    declaration.write_text(json.dumps({
        "schema": "aw.routec-pack-declaration.v1", "source_id": IETF.SOURCE_ID,
        "counts": {"route_urls": 3, "route_reported_production_points": 9, "source_parents": 2,
                   "source_parent_reported_production_points": 5, "redirected_current_rfc_records": 1,
                   "redirected_current_rfc_reported_production_points": 4}}))
    output = lane / "run"
    result = IETF.materialize(artifact_root=root, routes_path=routes_path, resolution_path=resolution_path,
                              output_root=output, declaration=declaration)
    return {"root": root, "run": output, "result": result}


def _ctx(built, **extra):
    from cytir.sources import registry
    from cytir.sources.family import RunContext
    adapter, row, spec = registry.admit(FAMILY)
    return adapter, RunContext(out_dir=built["root"] / "out", today="2026-10-30", chunk_spec=spec, contract=row,
                               options={"artifact_root": built["root"], "run_root": built["run"], **extra},
                               log=lambda m: None)


def _terminals(adapter, ctx):
    from cytir.sources.family import UnitExcluded
    out = {}
    for unit in adapter.discover(ctx):
        try:
            docs = [d for c in adapter.acquire(unit, ctx) for d in adapter.extract(c, ctx)]
            out[unit.url] = ("doc", docs[0])
        except UnitExcluded as exc:
            out[unit.url] = ("excluded", exc.reason)
    return out


def test_the_family_is_registered_full_and_declared(tmp_path):
    from cytir.sources import registry
    registry.load_modules()
    assert FAMILY in registry.registered()
    adapter, _ = _ctx({"root": tmp_path, "run": tmp_path})
    assert not getattr(adapter, "PARTIAL", ()) and adapter.direct_crawl is False


def test_adapter_emits_the_extractors_records_byte_for_byte(tmp_path):
    built = build(tmp_path)
    adapter, ctx = _ctx(built)
    got = _terminals(adapter, ctx)
    assert got[OLD] == ("excluded", "current-response-is-RFC-record-not-draft-record")
    with gzip.open(built["run"] / "markdown/ietf-draft-records.jsonl.gz", "rt") as fh:
        parents = {r["url"]: r for r in map(json.loads, fh)}
    assert set(parents) == {D1, D2}
    for url in (D1, D2):
        kind, doc = got[url]
        assert kind == "doc" and doc.lang == "en"
        assert (doc.extra["document_id"], doc.text) == (parents[url]["document_id"], parents[url]["markdown"])
        assert doc.extra["parent_source_sha256"] == parents[url]["parent_source_sha256"]
        assert hashlib.sha256(doc.text.encode()).hexdigest() == PINNED[url]
    assert got[D1][1].title == "draft-example-02 - Example"
    assert ctx.notes["contract_terms"]["search_ok"] is None        # unknown stays unknown
    assert ctx.notes["rfc_redirects"] == 1


def test_the_sealed_markdown_stage_is_pinned(tmp_path):
    built = build(tmp_path)
    assert built["result"]["stages"]["markdown"]["logical_sha256"] == PINNED["markdown_stage_logical"]


def test_a_tampered_body_digest_is_refused_not_excluded(tmp_path):
    from cytir.sources.family import Unit
    built = build(tmp_path)
    adapter, ctx = _ctx(built)
    unit = {u.url: u for u in adapter.discover(ctx)}[D1]
    bad = Unit(key=unit.key, url=unit.url, meta=dict(unit.meta, body_sha256="f" * 64))
    with pytest.raises(ValueError, match="body digest drift"):
        adapter.acquire(bad, ctx)


def test_a_run_root_without_a_sealed_run_is_refused(tmp_path):
    built = build(tmp_path)
    (built["run"] / "RUN.json").write_text(json.dumps({"schema": "x", "status": "complete"}))
    adapter, ctx = _ctx(built)
    with pytest.raises(RuntimeError, match="complete sealed Internet-Draft run"):
        adapter.discover(ctx)


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
