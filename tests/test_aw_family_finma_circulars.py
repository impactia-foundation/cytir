"""`finma_circulars` on the family Protocol (task 1396, S7).

No network, no Docling, no language model. The document-center API and the PDF host are a
`httpx.MockTransport`; the API answers are synthetic and shaped like the widget's JSON
(`Items` with Title / Description / Link / Extension / OtherLanguageLinks, and `Count`). Pins: the
adapter is registered and conformant; discover de-duplicates and keeps the language variants;
acquire/extract/chunk make records under the provenance `finma_circulars`; a body-less document
keeps ONE metadata record; consent and licence come from the contract row (a row without them
refuses); the Docling-less gate path and the adapter share one function.
"""
from __future__ import annotations

import json

import httpx
import pytest

from cytir.sources import contract as contract_mod
from cytir.sources import registry
from cytir.sources.family import RunContext, SourceFamily
from cytir.sources.families.finma_circulars import FinmaCircularsFamily
from cytir.sources.finma import circulars as C

BODY = ("Die Banken halten ausreichende Liquidität. " * 40).strip()
PDF = b"%PDF-1.7 synthetic"


def _api(ds_items):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/de/api/search/getresult":
            ds = request.content.decode()
            items = ds_items if "3009DAA1" in ds else []
            return httpx.Response(200, json={"Items": items, "Count": len(items)})
        if request.url.path.endswith("/missing.pdf"):
            return httpx.Response(404, content=b"nope")
        return httpx.Response(200, content=PDF, headers={"content-type": "application/pdf"})
    return handler


ITEMS = [
    {"Title": "FINMA-Rundschreiben 2015/02", "Description": "Liquiditätsrisiken Banken",
     "Extension": "pdf", "Date": "2015-11-20",
     "OtherLanguageLinks": [{"Url": "/de/~/media/a/rs-2015-02.pdf?sc_lang=de", "Name": "Deutsch"},
                            {"Url": "/fr/~/media/a/rs-2015-02.pdf?sc_lang=fr", "Name": "Français"}]},
    {"Title": "Ohne Inhalt", "Description": "Nur Metadaten", "Extension": "pdf",
     "OtherLanguageLinks": [{"Url": "/de/~/media/b/missing.pdf?sc_lang=de", "Name": "Deutsch"}]},
    {"Title": "Not a PDF", "Extension": "docx", "Link": "/de/~/media/c/x.docx"},
]


def _row() -> dict:
    return contract_mod.by_id(contract_mod.load())["finma_circulars"]


def _ctx(tmp_path, contract=None, **opt) -> RunContext:
    client = httpx.Client(transport=httpx.MockTransport(_api(ITEMS)), base_url="https://www.finma.ch")
    return RunContext(out_dir=tmp_path / "out", today="2026-10-30", skip_consent=True, http=client,
                      options={"delay": "0", **opt}, contract=contract or _row(), log=lambda m: None)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(C, "pdf_to_markdown", lambda conv, body, name: BODY)
    monkeypatch.setattr(C, "make_converter", lambda: object())
    from cytir.acquire import fetch_process as fp
    monkeypatch.setattr(fp, "detect_lang", lambda text, url, default: (default, False))


def test_registered_and_conformant():
    assert "finma_circulars" in registry.registered()
    adapter, row, spec = registry.admit("finma_circulars")
    assert isinstance(adapter, SourceFamily)
    assert adapter.direct_crawl is True and not getattr(adapter, "PARTIAL", ())
    assert spec.declared == row["chunk_policy"]["policy_id"]


def test_discover_keeps_language_variants_and_skips_non_pdf(tmp_path):
    fam, ctx = FinmaCircularsFamily(), _ctx(tmp_path, only="rundschreiben")
    units = list(fam.discover(ctx))
    assert len(units) == 3 and ctx.notes["pdf_targets"] == 3
    assert {u.meta["lang"] for u in units} == {"de", "fr"}
    assert all(u.url.startswith("https://www.finma.ch/") for u in units)
    assert ctx.notes["contract_terms"]["search_ok"] is True
    assert ctx.notes["contract_terms"]["licence_basis"] == _row()["licence"]["basis"]


def test_acquire_extract_chunk_make_records(tmp_path):
    fam, ctx = FinmaCircularsFamily(), _ctx(tmp_path)
    units = list(fam.discover(ctx))
    good = next(u for u in units if "rs-2015-02" in u.url and u.meta["lang"] == "de")
    (cap,) = fam.acquire(good, ctx)
    assert cap.body == PDF and cap.meta["status"] == 200
    (doc,) = fam.extract(cap, ctx)
    assert doc.title == "FINMA-Rundschreiben 2015/02" and doc.extra["doc"]["ok"] is True
    recs = list(fam.chunk(doc, ctx))
    assert recs and all(r["provenance"] == "finma_circulars" for r in recs)
    assert "content_digest" not in recs[0]
    assert "Rundschreiben 2015/02" in recs[0]["text"]
    assert recs[0]["url"] == good.url


def test_a_missing_pdf_keeps_one_metadata_record(tmp_path):
    fam, ctx = FinmaCircularsFamily(), _ctx(tmp_path)
    bad = next(u for u in fam.discover(ctx) if u.url.endswith("missing.pdf?sc_lang=de"))
    (cap,) = fam.acquire(bad, ctx)
    assert cap.meta["status"] == 404
    (doc,) = fam.extract(cap, ctx)
    assert doc.extra["doc"]["ok"] is False
    (rec,) = list(fam.chunk(doc, ctx))
    assert rec["text"].startswith("Ohne Inhalt")


def test_a_row_without_search_ok_refuses(tmp_path):
    row = json.loads(json.dumps(_row()))
    row["consent"]["search_ok"] = None
    with pytest.raises(ValueError, match="search_ok"):
        list(FinmaCircularsFamily().discover(_ctx(tmp_path, contract=row)))
    with pytest.raises(ValueError, match="consent and licence"):
        list(FinmaCircularsFamily().discover(_ctx(tmp_path, contract={"family_id": "x"})))


def test_gate_and_adapter_share_one_function():
    from cytir.sources.families import finma_circulars as mod
    assert mod.C.doc_passages is C.doc_passages
    stats = C.new_stats()
    doc = {"url": "https://www.finma.ch/de/~/media/z.pdf", "lang": "de", "title": "T",
           "description": "D", "source": "rundschreiben", "ok": False, "md": ""}
    (p,) = C.doc_passages(doc, stats, 50, last_seen="2026-10-30")
    assert stats["meta_fallback"] == 1 and p.lang == "de"


def test_finma_exact_shares_the_driver_and_marks_chunk_partial(tmp_path):
    """task 2171: `finma_exact` is the same delegated discover/acquire/extract; its chunk is PARTIAL."""
    from cytir.sources.families.finma_circulars import FinmaExactFamily
    assert "finma_exact" in registry.registered()
    fam = FinmaExactFamily()
    assert fam.PARTIAL == ("chunk",) and fam.family_id == "finma_exact"
    # the row's consent is `unknown` (backfilled): the run refuses; nothing is inferred
    row = contract_mod.by_id(contract_mod.load())["finma_exact"]
    assert row["consent"]["search_ok"] is not True
    with pytest.raises(ValueError, match="finma_exact: the contract does not declare search_ok"):
        list(fam.discover(_ctx(tmp_path, contract=row, only="rundschreiben")))
    # with a row that declares consent (synthetic, test only) the delegated stages run
    ok = json.loads(json.dumps(_row()))
    ok["family_id"] = "finma_exact"
    units = list(fam.discover(_ctx(tmp_path, contract=ok, only="rundschreiben")))
    assert len(units) == 3
    (cap,) = fam.acquire(next(u for u in units if "rs-2015-02" in u.url), _ctx(tmp_path, contract=ok))
    (doc,) = fam.extract(cap, _ctx(tmp_path, contract=ok))
    with pytest.raises(NotImplementedError, match="PARTIAL"):
        fam.chunk(doc, _ctx(tmp_path, contract=ok))
