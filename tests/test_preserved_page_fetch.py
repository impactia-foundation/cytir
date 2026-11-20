"""Task 450: the publisher-page fetch must preserve before anything reads it."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from cytir.acquire import preserved_page_fetch as subject
from cytir.acquire import preserved_stage_io as PS


UA = "authoritative-web/0.1 (+https://example.test; contact@example.test)"
URL = "https://publisher.test/article/one"


class Headers:
    """The bits of an httpx header mapping this module actually uses."""

    def __init__(self, pairs):
        self._pairs = [(str(k), str(v)) for k, v in pairs]

    def multi_items(self):
        return list(self._pairs)

    def get(self, key, default=None):
        for name, value in self._pairs:
            if name.casefold() == key.casefold():
                return value
        return default


class Request:
    def __init__(self, url, headers, method="GET"):
        self.method = method
        self.url = url
        self.headers = Headers(headers)


class Response:
    def __init__(self, *, content, url=URL, status_code=200,
                 content_type="text/html; charset=utf-8",
                 request_headers=None, requested_url=URL):
        self.content = content
        self.url = url
        self.status_code = status_code
        self.headers = Headers([("content-type", content_type),
                                ("server", "nginx"),
                                ("etag", '"abc"')])
        self.request = Request(
            requested_url,
            request_headers if request_headers is not None
            else [("user-agent", UA), ("accept", "*/*")])

    @property
    def text(self):
        """httpx's decoded view — the very thing the old lane kept instead of
        the bytes. Present only so the unpreserved legacy path stays testable."""
        return self.content.decode("utf-8", "replace")


def seal(root: Path, response: Response, url: str = URL,
         fetched_at: str = "2026-08-28T05:00:00Z", ua: str = UA) -> dict:
    receipt = subject.seal_page_fetch(root, url, response, fetched_at, ua)
    subject.publish_receipt(root, url, receipt)
    return receipt


def test_a_sealed_fetch_keeps_bytes_transport_and_request_identity(tmp_path):
    body = "<html lang='de'><body><p>Bundesrat</p></body></html>".encode("utf-8")
    receipt = seal(tmp_path, Response(content=body))

    assert receipt["schema"] == subject.RECEIPT_SCHEMA
    assert subject.load_body(tmp_path, receipt) == body
    assert receipt["body_bytes"] == len(body)

    transport = json.loads(gzip.decompress(
        (tmp_path / receipt["artifacts"]["transport"]["uri"].removeprefix(
            "artifact://")).read_bytes()))
    assert transport["schema"] == subject.TRANSPORT_SCHEMA
    assert transport["request_method"] == "GET"
    assert transport["request_url"] == URL
    assert transport["requested_url"] == URL
    assert transport["final_url"] == URL
    assert transport["http_status"] == 200
    assert transport["fetched_at"] == "2026-08-28T05:00:00Z"
    assert ["user-agent", UA] in transport["request_headers"]
    assert ["server", "nginx"] in transport["response_headers"]
    assert transport["body"]["bytes"] == len(body)
    assert transport["body"]["sha256"] == receipt["body_sha256"]
    # The receipt itself is on disk, immutable, named after the requested URL.
    path = subject.receipt_path(tmp_path, URL)
    assert json.loads(path.read_text()) == receipt


def test_no_response_body_is_ever_truncated(tmp_path):
    body = b"x" * (3 * 1024 * 1024)
    receipt = seal(tmp_path, Response(content=body))
    assert subject.load_body(tmp_path, receipt) == body


def test_a_fetch_that_cannot_bind_our_crawler_identity_is_not_sealable(tmp_path):
    for headers in ([("accept", "*/*")],
                    [("user-agent", "Mozilla/5.0"), ("accept", "*/*")],
                    [("user-agent", UA), ("user-agent", UA)]):
        with pytest.raises(subject.PagePreservationError, match="unbound"):
            subject.seal_page_fetch(
                tmp_path, URL,
                Response(content=b"<html></html>", request_headers=headers),
                "2026-08-28T05:00:00Z", UA)
    # Nothing at all is published for an unattributable fetch.
    assert not subject.receipt_path(tmp_path, URL).exists()


def test_an_immutable_receipt_is_never_replaced(tmp_path):
    seal(tmp_path, Response(content=b"<html>a</html>"))
    with pytest.raises(subject.PagePreservationError, match="collision"):
        seal(tmp_path, Response(content=b"<html>b</html>"))


def test_a_valid_receipt_is_reused_and_a_broken_one_is_not_evidence(tmp_path):
    receipt = seal(tmp_path, Response(content=b"<html>a</html>"))
    path = subject.receipt_path(tmp_path, URL)
    assert subject.valid_receipt(tmp_path, path, URL) == receipt
    # Same store, different URL -> not this page's evidence.
    assert subject.valid_receipt(tmp_path, path, URL + "?x=1") is None
    # Body artifact removed -> the receipt no longer validates.
    body_path = tmp_path / receipt["artifacts"]["body"]["uri"].removeprefix(
        "artifact://")
    body_path.unlink()
    assert subject.valid_receipt(tmp_path, path, URL) is None


def test_a_tampered_body_artifact_refuses_to_load(tmp_path):
    receipt = seal(tmp_path, Response(content=b"<html>a</html>"))
    body_path = tmp_path / receipt["artifacts"]["body"]["uri"].removeprefix(
        "artifact://")
    body_path.write_bytes(PS.deterministic_gzip(b"<html>tampered</html>"))
    with pytest.raises(PS.PreservedArtifactError):
        subject.load_body(tmp_path, receipt)


def test_error_and_non_html_responses_are_preserved_but_not_admitted(tmp_path):
    error = seal(tmp_path, Response(content=b"nope", status_code=404),
                 url=URL + "/404")
    assert subject.load_body(tmp_path, error) == b"nope"
    assert subject.is_admissible_article(error) is False

    feed = seal(tmp_path, Response(content=b"{}",
                                   content_type="application/json"),
                url=URL + "/json")
    assert subject.load_body(tmp_path, feed) == b"{}"
    assert subject.is_admissible_article(feed) is False

    ok = seal(tmp_path, Response(content=b"<html></html>"))
    assert subject.is_admissible_article(ok) is True


def test_identical_bodies_share_one_content_addressed_object(tmp_path):
    first = seal(tmp_path, Response(content=b"<html>same</html>"), url=URL)
    second = seal(tmp_path, Response(content=b"<html>same</html>",
                                     url=URL + "/b", requested_url=URL + "/b"),
                  url=URL + "/b")
    assert (first["artifacts"]["body"]["uri"] ==
            second["artifacts"]["body"]["uri"])
    assert first["object_id"] != second["object_id"]


def test_body_decoding_prefers_the_declared_charset(tmp_path):
    body = "Zürich".encode("latin-1")
    assert subject.decode_body(body, "text/html; charset=latin-1") == "Zürich"
    # Undecodable under the declared charset falls back rather than raising.
    assert subject.decode_body(body, "text/html; charset=utf-8")
    assert subject.decode_body("Zürich".encode("utf-8"), None) == "Zürich"


def test_object_id_is_the_requested_url_not_the_redirect_target():
    assert subject.object_id(URL) != subject.object_id(URL + "/redirected")
    assert subject.object_id(URL) == subject.object_id(URL)
