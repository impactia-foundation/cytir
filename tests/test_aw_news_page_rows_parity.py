"""task 2783: the news per-page extract+chunk+row-finishing block is ONE function.

`golden_rows.json` was computed once with the pre-change inline block of
`sitemap_ingest.process_host`. Both new paths (the function `process_host` calls, and the
`news_daily` Protocol `extract`/`chunk`) must reproduce it field for field.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from cytir.acquire import sitemap_ingest as SI
from cytir.extract.html_extract import process_html
from cytir.news.page_rows import finish_page_rows, page_rows
from cytir.sources.family import Captured, RunContext, Unit
from cytir.sources.families.news_daily import NewsDailyFamily

FX = pathlib.Path(__file__).parent / "fixtures" / "news_page_rows"
URL = "https://www.example-news.ch/politik/verkehr-neu"
COMMON = dict(domain="example-news.ch", tier="C", trust_tier=0.9, source_type="news",
              last_seen="2026-11-11")
CONSENT = {"search_ok": True, "train_ok": False, "checked_at": "2026-11-11T06:00:00Z"}
RECEIPT = {"object_id": "obj1", "requested_url": URL, "final_url": URL, "http_status": 200,
           "fetched_at": "2026-11-11T06:01:00Z", "body_sha256": "b" * 64,
           "artifacts": {"transport": {"body_sha256": "t" * 64}}}


@pytest.fixture(autouse=True)
def _lang(monkeypatch):
    monkeypatch.setenv("AW_LANGGATE_OPTIONAL", "1")


def _golden():
    return json.loads((FX / "golden_rows.json").read_text(encoding="utf-8"))


def _html():
    return (FX / "page.html").read_text(encoding="utf-8")


def _host_meta():
    return {"passage_common": dict(COMMON), "cls": "news", "lang_default": "de",
            "freshness": True, "consent_stamp": SI._consent_stamp(CONSENT)}


def _fetch_meta():
    return {"sitemap_published": "2026-11-10", "lastmod": "2026-11-10",
            "page_receipt": RECEIPT, "document_id": SI.news_document_id(URL)}


def _norm(rows):
    return json.loads(json.dumps(rows, ensure_ascii=False, sort_keys=True))


def test_process_host_path_matches_golden():
    html = _html()
    emeta = {"content": True, "links": True, "passage_common": dict(COMMON), "lang_default": "de",
             "digest": None, "outlinks_column": False, "sitemap_lastmod": "2026-11-10",
             "now_ref": None, "published_htmldate": True}
    passages, _links = process_html(URL, html, emeta)
    rows, page_published = finish_page_rows(
        passages, extract_input=html, url=URL, cls="news",
        consent_stamp=SI._consent_stamp(CONSENT), now_ref=SI.page_date_clamp_day(None),
        freshness=True, sitemap_published="2026-11-10", page_receipt=RECEIPT,
        document_id=SI.news_document_id(URL))
    assert page_published
    assert _norm(rows) == _golden()


def test_page_rows_matches_golden():
    assert _norm(page_rows(_html(), URL, _fetch_meta(), _host_meta())) == _golden()


def test_protocol_stages_match_golden(tmp_path):
    fam = NewsDailyFamily()
    ctx = RunContext(out_dir=tmp_path, today="2026-11-11", skip_consent=True, options={},
                     contract=None, log=lambda m: None)
    unit = Unit(key="day-2026-11-10@example-news.ch", url=URL, consent_hosts=("example-news.ch",),
                meta={})
    cap = Captured(unit, _html().encode("utf-8"),
                   {"status": 200, "content_type": "text/html; charset=utf-8",
                    "host_meta": _host_meta(), "fetch_meta": _fetch_meta()})
    (doc,) = fam.extract(cap, ctx)
    assert _norm(list(fam.chunk(doc, ctx))) == _golden()


def test_counts_capture_is_not_extractable(tmp_path):
    fam = NewsDailyFamily()
    ctx = RunContext(out_dir=tmp_path, today="2026-11-11", skip_consent=True, options={},
                     contract=None, log=lambda m: None)
    unit = Unit(key="k", url="https://x/", consent_hosts=("x",), meta={})
    with pytest.raises(ValueError):
        fam.extract(Captured(unit, b"{}", {"status": 200, "content_type": "application/json"}), ctx)
