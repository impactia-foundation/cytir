import gzip
import json
import re
from pathlib import Path

import httpx
import pytest

import cytir.sources.hn.freshness_ingest as subject


class Response:
    def __init__(self, value, status_code=200):
        self.value = value
        self.status_code = status_code
        self.text = str(value)

    def json(self):
        return self.value


class FirebaseClient:
    def __init__(self):
        self.calls = []
        self.values = {
            "topstories.json": [1, 2],
            "beststories.json": [2, 3],
            "item/1.json": {"type": "story", "url": "https://one.test",
                             "title": "one", "score": 120, "time": 950},
            "item/2.json": {"type": "story", "url": "https://two.test",
                             "title": "two", "score": 99, "time": 950},
            "item/3.json": {"type": "story", "url": "https://old.test",
                             "title": "old", "score": 200, "time": -90000},
        }

    def get(self, url, **_kwargs):
        self.calls.append(url)
        return Response(self.values[url.rsplit("/", 1)[-1] if "/item/" not in url
                                    else "item/" + url.rsplit("/", 1)[-1]])


class PartitioningAlgoliaClient:
    """Synthetic Algolia: the broad window is capped, its two children are not."""
    def __init__(self):
        self.calls = []

    def get(self, _url, *, params, **_kwargs):
        filt = params["numericFilters"]
        match = re.fullmatch(r"created_at_i>=(\d+),created_at_i<(\d+)", filt)
        assert match
        lo, hi = map(int, match.groups())
        self.calls.append((lo, hi))
        if (lo, hi) == (0, 100):
            # This is exactly the legacy failure: server says one page while
            # retaining only its result-window maximum.
            return Response({"nbHits": 1500, "nbPages": 1,
                             "hits": [{"objectID": str(i), "created_at_i": i}
                                      for i in range(1000)]})
        if (lo, hi) == (0, 50):
            return Response({"nbHits": 2, "nbPages": 1, "hits": [
                {"objectID": "early", "created_at_i": 10},
                {"objectID": "same", "created_at_i": 20},
            ]})
        if (lo, hi) == (50, 100):
            return Response({"nbHits": 2, "nbPages": 1, "hits": [
                {"objectID": "late", "created_at_i": 90},
                {"objectID": "same", "created_at_i": 60},
            ]})
        raise AssertionError(f"unexpected slice {(lo, hi)}")


def test_firebase_fallback_is_bounded_deduped_and_explicitly_incomplete():
    client = FirebaseClient()
    stories, receipt = subject.source_stories(
        client, since_days=1, min_points=100, now_i=1000, tags="story",
        source="firebase", firebase_item_cap=10)

    assert [story["objectID"] for story in stories] == ["1"]
    assert receipt == {
        "transport": "firebase", "completeness": "best-effort-current-lists",
        "feeds": ["topstories", "beststories"], "item_cap": 10,
    }
    assert client.calls.count(subject.FIREBASE + "/item/2.json") == 1


def test_auto_fallback_never_labels_firebase_as_a_complete_algolia_window(monkeypatch):
    client = FirebaseClient()

    def unavailable(*_args, **_kwargs):
        raise subject.HNSourceError("Algolia unreachable")

    monkeypatch.setattr(subject, "algolia_window", unavailable)
    stories, receipt = subject.source_stories(
        client, since_days=1, min_points=100, now_i=1000, tags="story",
        source="auto", firebase_item_cap=10)
    assert [story["objectID"] for story in stories] == ["1"]
    assert receipt["transport"] == "firebase"
    assert receipt["completeness"] == "best-effort-current-lists"


def test_algolia_recursively_partitions_a_capped_time_window_and_records_proof():
    client = PartitioningAlgoliaClient()
    hits, proof = subject.algolia_window(client, 0, 100, "story")

    assert client.calls == [(0, 100), (0, 50), (50, 100)]
    assert [hit["objectID"] for hit in hits] == ["late", "same", "early"]
    assert proof == {
        "algolia_result_cap": 1000,
        "algolia_time_partition_count": 2,
        "algolia_time_partitions": [
            {"start_epoch": 0, "end_epoch": 50, "nb_hits": 2, "returned_hits": 2},
            {"start_epoch": 50, "end_epoch": 100, "nb_hits": 2, "returned_hits": 2},
        ],
        "algolia_raw_hits": 4,
        "algolia_unique_object_ids": 3,
    }


def test_algolia_fails_closed_when_a_one_second_slice_is_still_capped(monkeypatch):
    monkeypatch.setattr(subject, "_algolia_slice", lambda *_args: {
        "nbHits": 1001, "nbPages": 1, "hits": [{}] * 1000})
    with pytest.raises(subject.HNSourceError, match="unresolved"):
        subject.algolia_window(object(), 9, 10, "story")


def test_low_score_admission_is_data_scoped_to_reviewed_primary_owner(tmp_path):
    root = tmp_path / "release"
    pipeline = root / "pipeline"; pipeline.mkdir(parents=True)
    data = root / "data"; data.mkdir()
    owner_policy = data / "owners.json"
    owner_policy.write_text(json.dumps({"publishers": ["deepseek-ai"]}))
    host_meta = pipeline / "host-meta.json"
    host_meta.write_text(json.dumps({"huggingface.co": {
        "hn_low_score_admission": {
            "kind": "path-owner", "min_points": 1, "owner_segment": 0,
            "owner_policy": {"path": "../data/owners.json", "list_key": "publishers",
                             "sha256": subject._sha256_file(owner_policy)},
        }}}))
    raw = [
        {"objectID": "official-low", "url": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731",
         "points": 2, "created_at_i": 1},
        {"objectID": "official-stronger-repeat", "url": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731",
         "points": 16, "created_at_i": 2},
        {"objectID": "unreviewed-low", "url": "https://huggingface.co/random-user/spam-model",
         "points": 2, "created_at_i": 3},
        {"objectID": "other-host-low", "url": "https://spam.example/test",
         "points": 2, "created_at_i": 4},
    ]
    rules = subject.load_authoritative_hn_admission(host_meta)
    subject.mark_authoritative_primary(raw, rules)
    # The strongest real DeepSeek card submission was 16 points.  At the
    # community threshold of 40 it remains only because the generic, reviewed
    # path-owner rule admits it; the unreviewed low-score URLs still do not.
    selected = subject.qualifying_stories(raw, min_points=40)

    assert [story["objectID"] for story in selected] == ["official-stronger-repeat"]
    assert selected[0]["hn_admission"] == "authoritative-primary"
    assert (selected[0]["points"] or 0) < subject.DEFAULT_COMMUNITY_MIN_POINTS


def test_low_score_admission_fails_closed_on_owner_policy_hash_drift(tmp_path):
    root = tmp_path / "release"
    pipeline = root / "pipeline"; pipeline.mkdir(parents=True)
    data = root / "data"; data.mkdir()
    owner_policy = data / "owners.json"; owner_policy.write_text('{"publishers": ["x"]}')
    host_meta = pipeline / "host-meta.json"
    host_meta.write_text(json.dumps({"example.test": {"hn_low_score_admission": {
        "kind": "path-owner", "min_points": 1, "owner_segment": 0,
        "owner_policy": {"path": "../data/owners.json", "list_key": "publishers",
                         "sha256": "0" * 64}}}}))
    with pytest.raises(subject.HNSourceError, match="hash drift"):
        subject.load_authoritative_hn_admission(host_meta)


def test_row_keeps_primary_provenance_but_makes_fallback_auditable():
    story = {"objectID": "9", "url": "https://example.test/a", "title": "A",
             "points": 120, "created_at_i": 1_700_000_000}
    primary = subject.build_row(story, None, "en", subject.dt.date(2026, 8, 4),
                                {"transport": "algolia", "completeness": "time-window-complete"})
    fallback = subject.build_row(story, None, "en", subject.dt.date(2026, 8, 4),
                                 {"transport": "firebase", "completeness": "best-effort-current-lists"})
    assert primary["provenance"] == "hn-freshness"
    assert fallback["provenance"] == "hn-freshness-firebase-best-effort"
    assert fallback["hn_source_receipt"]["completeness"] == "best-effort-current-lists"
    assert primary["hn_strong_endorsement"] is True


def test_cli_writes_source_receipt_for_an_empty_window(tmp_path, monkeypatch):
    receipt = tmp_path / "receipt.json"
    output = tmp_path / "rows.jsonl"
    monkeypatch.setattr(subject, "source_stories", lambda *_args: ([], {
        "transport": "algolia", "completeness": "time-window-complete"}))
    monkeypatch.setattr("sys.argv", ["hn", "--out", str(output), "--receipt", str(receipt),
                                      "--no-fetch"])
    assert subject.main() == 0
    assert output.read_text() == ""
    assert json.loads(receipt.read_text())["completeness"] == "time-window-complete"
    value = json.loads(receipt.read_text())
    assert value["qualifying_stories"] == 0
    assert len(value["qualifying_object_ids_sha256"]) == 64


# --------------------------------------------------------------------------- #
# Task 445 — the linked publisher page is an ACQUISITION, so its original bytes,
# transport headers and request identity must survive the fetch.  The old
# `fetch_article` GETted the page, kept a snippet in memory and dropped the
# response, which is what made every daily HN slice non-authorizing under the
# `news_article` preservation profile.
# --------------------------------------------------------------------------- #
PAGE = (b'<!DOCTYPE html><html lang="de"><head><title>T</title></head><body>'
        b'<p>' + b'Ein sehr ausfuehrlicher Artikeltext. ' * 12 + b'</p></body></html>')


def _response(url, *, status=200, content=PAGE,
              content_type="text/html; charset=utf-8", ua=None):
    request = httpx.Request("GET", url, headers={
        "User-Agent": subject.UA["User-Agent"] if ua is None else ua,
        "Accept": "text/html"})
    return httpx.Response(status, request=request,
                          headers={"content-type": content_type},
                          content=content)


class LinkedPageClient:
    """Fake linked-page transport; `failures` names URLs that cannot be fetched."""

    def __init__(self, responses, failures=()):
        self.responses = responses
        self.failures = set(failures)
        self.calls = []

    def get(self, url, **_kwargs):
        self.calls.append(url)
        if url in self.failures:
            raise httpx.ConnectError("linked host unreachable")
        return self.responses[url]


def _sealed(root: Path, receipt: dict, name: str) -> bytes:
    return gzip.decompress(
        subject.PS.artifact_path(root, receipt["artifacts"][name]).read_bytes())


def test_linked_fetch_preserves_bytes_transport_headers_and_request_identity(tmp_path):
    url = "https://publisher.example/story"
    client = LinkedPageClient({url: _response(url)})
    html, lang, receipt = subject.fetch_article(client, url, tmp_path, "77")

    assert lang == "de" and "Artikeltext" in html
    # The extraction input is the PRESERVED bytes read back, not a live buffer.
    assert _sealed(tmp_path, receipt, "body") == PAGE
    assert receipt["artifacts"]["body"]["body_sha256"] == subject.hashlib.sha256(
        PAGE).hexdigest()

    transport = json.loads(_sealed(tmp_path, receipt, "transport"))
    assert transport["schema"] == "aw.http-api-response.v2"
    assert transport["requested_url"] == url
    assert transport["request_method"] == "GET"
    assert transport["request_url"] == url
    assert transport["final_url"] == url
    assert transport["http_status"] == 200
    assert transport["body"] == {"bytes": len(PAGE),
                                 "sha256": subject.hashlib.sha256(PAGE).hexdigest()}
    assert dict(transport["response_headers"])["content-type"] == \
        "text/html; charset=utf-8"
    request_headers = dict(transport["request_headers"])
    assert request_headers["user-agent"] == subject.UA["User-Agent"]
    assert request_headers["user-agent"].startswith("authoritative-web/")
    assert request_headers["accept"] == "text/html"

    on_disk = json.loads((tmp_path / "receipts/77.json").read_text())
    assert on_disk == receipt
    assert on_disk["schema"] == "aw.hn-linked-page-receipt.v1"


def test_a_row_derived_from_a_linked_page_carries_its_preservation_receipt(tmp_path):
    url = "https://publisher.example/story"
    client = LinkedPageClient({url: _response(url)})
    html, lang, receipt = subject.fetch_article(client, url, tmp_path, "77")
    row = subject.build_row(
        {"objectID": "77", "url": url, "title": "T", "points": 90,
         "created_at_i": 1_700_000_000},
        html, lang, subject.dt.date(2026, 8, 27),
        {"transport": "algolia", "completeness": "time-window-complete"}, receipt)

    assert row["hn_content_mode"] == "linked_article_derived"
    assert row["hn_linked_fetch"] == receipt
    assert row["text"] != row["title"]


def test_an_unpreservable_fetch_produces_no_linked_derived_row(tmp_path):
    url = "https://publisher.example/dead"
    client = LinkedPageClient({}, failures=[url])
    html, lang, receipt = subject.fetch_article(client, url, tmp_path, "78")

    assert (html, lang, receipt) == (None, None, None)
    assert not (tmp_path / "receipts").exists()
    row = subject.build_row(
        {"objectID": "78", "url": url, "title": "T", "points": 90,
         "created_at_i": 1_700_000_000},
        html, lang, subject.dt.date(2026, 8, 27),
        {"transport": "algolia", "completeness": "time-window-complete"}, receipt)
    # The URL is still the grounding value; the CONTENT falls back to the title.
    assert row["hn_content_mode"] == "title_only"
    assert "hn_linked_fetch" not in row
    assert row["text"] == row["title"]


def test_a_fetch_that_cannot_bind_our_crawler_identity_is_not_sealed(tmp_path):
    """Consent evidence must name the crawler that actually made the request."""
    url = "https://publisher.example/story"
    client = LinkedPageClient({url: _response(url, ua="Mozilla/5.0")})
    assert subject.fetch_article(client, url, tmp_path, "79") == (None, None, None)
    assert list(tmp_path.rglob("*.html.gz")) == []
    assert not (tmp_path / "receipts/79.json").exists()


def test_a_non_html_or_error_response_is_preserved_but_not_linked_derived(tmp_path):
    url = "https://publisher.example/gone"
    client = LinkedPageClient({url: _response(url, status=404, content=b"nope",
                                              content_type="text/plain")})
    html, lang, receipt = subject.fetch_article(client, url, tmp_path, "80")

    assert html is None and lang is None
    # The attempt is honest evidence, but it cannot produce article content.
    assert receipt["http_status"] == 404
    assert _sealed(tmp_path, receipt, "body") == b"nope"


def test_a_resumed_run_reuses_the_sealed_receipt_instead_of_refetching(tmp_path):
    url = "https://publisher.example/story"
    client = LinkedPageClient({url: _response(url)})
    first = subject.fetch_article(client, url, tmp_path, "77")[2]
    second_html, _, second = subject.fetch_article(client, url, tmp_path, "77")

    assert client.calls == [url]          # one GET, one immutable receipt
    assert second == first
    assert "Artikeltext" in second_html


def test_fetching_without_a_preservation_root_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["hn", "--out", str(tmp_path / "rows.jsonl")])
    with pytest.raises(SystemExit) as raised:
        subject.main()
    assert raised.value.code == 2


def test_cli_seals_each_linked_row_and_falls_back_when_a_fetch_cannot_be_sealed(
        tmp_path, monkeypatch):
    good, bad = "https://ok.example/a", "https://dead.example/b"
    stories = [
        {"objectID": "1", "url": good, "title": "Good", "points": 90,
         "created_at_i": 1_700_000_000},
        {"objectID": "2", "url": bad, "title": "Bad", "points": 91,
         "created_at_i": 1_700_000_001},
    ]
    client = LinkedPageClient({good: _response(good)}, failures=[bad])
    monkeypatch.setattr(subject, "source_stories", lambda *_args: (stories, {
        "transport": "algolia", "completeness": "time-window-complete"}))
    monkeypatch.setattr(subject, "consent_ok", lambda _url: True)
    monkeypatch.setattr(subject.httpx, "Client", lambda *a, **k: client)
    out = tmp_path / "rows.jsonl"
    preserve = tmp_path / "hn_linked"
    monkeypatch.setattr("sys.argv", ["hn", "--out", str(out), "--now", "2026-08-27",
                                     "--preserve-dir", str(preserve)])
    assert subject.main() == 0

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [row["hn_content_mode"] for row in rows] == [
        "linked_article_derived", "title_only"]
    assert _sealed(preserve, rows[0]["hn_linked_fetch"], "body") == PAGE
    assert "hn_linked_fetch" not in rows[1]
    assert rows[1]["text"] == rows[1]["title"]


def test_no_fetch_keeps_the_title_only_lane_without_a_preservation_root(
        tmp_path, monkeypatch):
    story = {"objectID": "1", "url": "https://ok.example/a", "title": "Good",
             "points": 90, "created_at_i": 1_700_000_000}
    monkeypatch.setattr(subject, "source_stories", lambda *_args: ([story], {
        "transport": "algolia", "completeness": "time-window-complete"}))
    out = tmp_path / "rows.jsonl"
    monkeypatch.setattr("sys.argv", ["hn", "--out", str(out), "--no-fetch"])
    assert subject.main() == 0
    row = json.loads(out.read_text().strip())
    assert row["hn_content_mode"] == "title_only"
    assert "hn_linked_fetch" not in row


def test_a_non_canonical_object_id_is_a_source_failure_not_a_receipt_name(tmp_path):
    url = "https://publisher.example/story"
    client = LinkedPageClient({url: _response(url)})
    with pytest.raises(subject.HNSourceError, match="non-canonical"):
        subject.fetch_article(client, url, tmp_path, "../escape")
    assert client.calls == []


def test_robots_and_algolia_requests_send_exactly_one_user_agent(monkeypatch):
    """Review 1956: the robots client must not add a second User-Agent line."""
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, json={"hits": []})

    real = subject.make_client
    monkeypatch.setattr(subject, "make_client", lambda spec=None, **kw: real(
        spec, transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(subject, "_robots", {})
    assert subject.consent_ok("https://example.org/page") is True
    with subject.make_client(subject.HttpSpec(timeout=5.0)) as client:
        subject._algolia_slice(client, 0, 10, "story")
    paths = [r.url.path for r in seen]
    assert "/robots.txt" in paths and any("search_by_date" in p for p in paths)
    for r in seen:
        assert len(r.headers.get_list("user-agent")) == 1
