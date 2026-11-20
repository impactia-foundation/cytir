"""Task 450: `sitemap_ingest --preserve-dir` seals before it reads.

Collected only where `sitemap_ingest` is importable (it pulls duckdb through
`cc_columnar`), i.e. on the Heimdall worker, exactly like
`test_sitemap_ingest_freshness.py`.
"""
from __future__ import annotations

import json

import pytest

from cytir.acquire import preserved_page_fetch as PF
from cytir.acquire import sitemap_ingest as subject
from test_preserved_page_fetch import UA, Response


ALLOWED = {"publisher.test", "www.publisher.test"}
URL = "https://publisher.test/article/one"
PAGE = ("<html lang='de'><head><title>Bundesrat entscheidet</title></head>"
        "<body><article><p>" + ("Der Bundesrat hat entschieden. " * 40) +
        "</p><a href='https://ref.test/law'>Bundesgesetz</a></article>"
        "</body></html>").encode("utf-8")


class Bucket:
    def wait(self):
        return None


class Client:
    """One-shot fake httpx client that counts the GETs it actually serves."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.gets = 0

    def get(self, url, **_kwargs):
        self.gets += 1
        if not self._responses:
            raise AssertionError(f"unexpected extra GET for {url}")
        value = self._responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.fixture(autouse=True)
def _pin_crawler_identity(monkeypatch):
    monkeypatch.setitem(subject.UA, "User-Agent", UA)


def fetch(root, client, url=URL, allowed=None):
    return subject.fetch_page(client, url, Bucket(), Bucket(),
                              allowed if allowed is not None else ALLOWED,
                              preserve_root=root)


def test_a_preserved_fetch_returns_text_read_back_from_sealed_bytes(tmp_path):
    client = Client([Response(content=PAGE)])
    final_url, status, html, note, receipt = fetch(tmp_path, client)

    assert (final_url, status, note) == (URL, 200, None)
    assert "Bundesrat" in html
    assert receipt["schema"] == PF.RECEIPT_SCHEMA
    assert PF.load_body(tmp_path, receipt) == PAGE
    assert PF.receipt_path(tmp_path, URL).is_file()


def test_an_unpreservable_fetch_yields_no_text_at_all(tmp_path):
    # Wrong crawler identity on the wire -> not sealable -> not admitted.
    client = Client([Response(content=PAGE,
                              request_headers=[("user-agent", "Mozilla/5.0")])])
    final_url, status, html, note, receipt = fetch(tmp_path, client)

    assert html is None and receipt is None
    assert note.startswith("unpreserved:")
    assert not PF.receipt_path(tmp_path, URL).exists()
    assert not list(tmp_path.rglob("*.html.gz"))


def test_error_and_non_html_responses_are_sealed_but_produce_no_content(tmp_path):
    error = Client([Response(content=b"gone", status_code=404)])
    _f, status, html, note, receipt = fetch(tmp_path, error)
    assert (status, html, note) == (404, None, "http 404")
    assert PF.load_body(tmp_path, receipt) == b"gone"

    feed = Client([Response(content=b"{}", content_type="application/json",
                            url=URL + "/j", requested_url=URL + "/j")])
    _f, _s, html, note, receipt = fetch(tmp_path, feed, url=URL + "/j")
    assert (html, note) == (None, "non-html")
    assert PF.load_body(tmp_path, receipt) == b"{}"


def test_an_off_host_redirect_is_preserved_and_still_refused(tmp_path):
    client = Client([Response(content=PAGE, url="https://elsewhere.test/a")])
    _f, status, html, note, receipt = fetch(tmp_path, client)
    assert (status, html) == (200, None)
    assert note.startswith("off-host redirect")
    assert PF.load_body(tmp_path, receipt) == PAGE


def test_a_resumed_run_reuses_the_sealed_receipt_instead_of_refetching(tmp_path):
    client = Client([Response(content=PAGE)])
    first = fetch(tmp_path, client)[4]
    assert client.gets == 1

    resumed = Client([])          # a second GET would raise
    again = fetch(tmp_path, resumed)
    assert resumed.gets == 0
    assert again[4] == first
    assert again[2] is not None


def test_transport_failures_never_invent_a_receipt(tmp_path):
    client = Client([RuntimeError("conn reset")] * 3)
    final_url, status, html, note, receipt = fetch(tmp_path, client)
    assert (status, html, receipt) == (0, None, None)
    assert note.startswith("conn:")
    assert not PF.receipt_path(tmp_path, URL).exists()


def test_unpreserved_mode_is_byte_identical_to_the_old_contract(tmp_path):
    client = Client([Response(content=PAGE)])
    final_url, status, html, note, receipt = subject.fetch_page(
        client, URL, Bucket(), Bucket(), ALLOWED)
    assert (final_url, status, note, receipt) == (URL, 200, None, None)
    assert html == PAGE.decode("utf-8")
    assert not list(tmp_path.rglob("*"))


# --- the news_article document record ---------------------------------------

def _receipt(tmp_path):
    return PF.seal_page_fetch(tmp_path, URL, Response(content=PAGE),
                              "2026-08-28T05:00:00Z", UA)


def test_a_document_record_carries_the_profile_semantics_and_lineage(tmp_path):
    receipt = _receipt(tmp_path)
    markdown = "Der Bundesrat hat entschieden.\n\nMehr Text."
    record = subject._news_document_record(
        url=URL, document_id=subject.news_document_id(URL), html=PAGE.decode(),
        markdown=markdown, receipt=receipt, publisher="publisher.test",
        published="2026-08-27",
        outlinks=[{"url": "https://ref.test/law", "nofollow": False,
                   "anchor_or_context": "Bundesgesetz"}],
        point_ids=["11111111-1111-1111-1111-111111111111"])

    assert record["document_id"] == subject.news_document_id(URL)
    assert record["publisher"] == "publisher.test"
    assert record["publication_time"] == "2026-08-27"
    assert record["publication_time_status"] == "source_declared"
    assert (record["byline_status"] == "source_declared") == bool(record["byline"])
    text = record["markdown"]
    assert text.startswith(f"# {record['headline']}\n\nByline: ")
    assert (f"\n\nPublisher: publisher.test\n\nFetched: "
            f"{receipt['fetched_at']}\n\n") in text
    assert f"\n\nSource: {URL}\n\n" in text
    assert text.split(f"\n\nSource: {URL}\n\n", 1)[1].strip() == markdown
    assert record["lineage"]["original_sha256"] == receipt["body_sha256"]
    assert record["lineage"]["fetched_at"] == receipt["fetched_at"]
    assert record["outlinks"] == [
        {"ordinal": 0, "target_url": "https://ref.test/law",
         "anchor_or_context": "Bundesgesetz", "nofollow": False}]
    assert record["chunk_point_ids"] == [
        "11111111-1111-1111-1111-111111111111"]
    # The record must round-trip as one JSONL line.
    assert json.loads(json.dumps(record, ensure_ascii=False)) == record


def test_an_undeclared_byline_or_date_stays_explicitly_absent(tmp_path):
    record = subject._news_document_record(
        url=URL, document_id=subject.news_document_id(URL),
        html="<html><body>no metadata at all</body></html>",
        markdown="body", receipt=_receipt(tmp_path),
        publisher="publisher.test", published=None, outlinks=[], point_ids=[])
    assert record["byline"] is None
    assert record["byline_status"] == "not_declared_by_publisher"
    assert record["publication_time"] is None
    assert record["publication_time_status"] == "not_declared_by_publisher"
    assert subject.NEWS_BYLINE_ABSENT in record["markdown"]
    assert record["outlinks"] == []


def test_document_identity_matches_the_admitted_news_materializers():
    import hashlib
    assert subject.news_document_id(URL) == hashlib.sha256(
        f"news-document\0{URL}".encode()).hexdigest()


def test_docs_out_without_a_preservation_root_is_refused(tmp_path):
    class Args:
        preserve_dir = ""
        docs_out = str(tmp_path / "docs.jsonl")

    with pytest.raises(SystemExit, match="requires --preserve-dir"):
        subject._preservation_outputs(Args())

    class Both:
        preserve_dir = str(tmp_path / "pages")
        docs_out = str(tmp_path / "docs.jsonl")

    preserve_root, docs_out = subject._preservation_outputs(Both())
    assert preserve_root.name == "pages" and docs_out.name == "docs.jsonl"

    class Neither:
        preserve_dir = ""
        docs_out = ""

    assert subject._preservation_outputs(Neither()) == (None, None)
