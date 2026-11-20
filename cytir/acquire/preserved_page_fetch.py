"""Preserve a publisher page response BEFORE anything reads it (task 450).

THE DEFECT THIS REMOVES
-----------------------
``sitemap_ingest.py --raw-dir`` stored, per fetched page,
``{url, final_url, http_status, fetched_at, lastmod, sha1, ua, html}``:

  * ``html`` is *decoded text*, not the response bytes — the transfer encoding,
    the declared charset and any byte the decoder replaced are already gone;
  * it is ``html[:_HTML_CAP]`` — a silent prefix cap;
  * there are **no response headers** at all; and
  * there is **no record of the request we sent** — one flattened ``ua`` string
    is not the outbound header set, and nothing binds the method or the exact
    URL requested.

``news_article.original`` (``data/source-preservation-profiles.v1.json``)
requires the *exact article/feed response*, its fetch time, and publisher
identity, and ``preservation_contract._validate_content_fidelity`` requires
``source_text_truncation_applied is False``.  A row derived from that raw store
can satisfy neither, which is why every daily news pack was non-authorizing
(task 449 §6.2).

THE FIX
-------
This module is the publisher-lane twin of the HN linked-page preservation that
task 445 added to ``hn_freshness_ingest.py``, and it seals the *same* evidence
in the *same* content-addressed store (``preserved_stage_io``), using the same
transport schema the HN Firebase leg seals (``aw.http-api-response.v2``):

  ``original/publisher_pages/<sha>.html.gz``      exact response bytes
  ``original/publisher_transport/<sha>.json.gz``  method, request URL, outbound
                                                  headers (incl. the crawler
                                                  UA), final URL, status, fetch
                                                  time, response headers, and
                                                  the body digest
  ``receipts/<sha256(requested_url)>.json``       ``aw.news-publisher-page-receipt.v1``

Callers then read the extraction input back through
``preserved_stage_io.load_validated_artifact``, so a publisher-derived row
cannot exist without preserved bytes.  Request identity is *checked*, not just
recorded: exactly one ``user-agent`` request header, equal to the crawler
identity the lane is configured to send.  An unbound identity is not sealable.

Fail-closed per page: a fetch that cannot be sealed yields no row at all.  It is
never downgraded to "we have the text but not the evidence" — that is precisely
the state this module exists to make impossible.

Resume-safe: a valid existing receipt for the same URL is reused instead of
re-GETting, so a same-day ``AW_NEWS_RESUME=1`` retry neither double-fetches nor
collides with immutable evidence.

The legacy ``--raw-dir`` per-host jsonl is deliberately left exactly as it was,
in a different directory, so every existing ``glob("*.raw.jsonl.gz")`` reader
keeps its current meaning (the same separation task 445 kept for HN).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cytir.acquire.artifact_io import atomic_publish, canonical_json
from cytir.acquire import preserved_stage_io as PS


RECEIPT_SCHEMA = "aw.news-publisher-page-receipt.v1"
TRANSPORT_SCHEMA = "aw.http-api-response.v2"
BODY_FAMILY = "original/publisher_pages"
TRANSPORT_FAMILY = "original/publisher_transport"


class PagePreservationError(RuntimeError):
    """A publisher page response could not be preserved as evidence."""


def object_id(url: str) -> str:
    """Stable receipt name for one requested URL.

    The receipt file is named after the URL we *asked for*, not the URL we were
    redirected to, so a resumed run can find the evidence for a queued URL
    without replaying the redirect chain.
    """
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def receipt_path(root: Path, url: str) -> Path:
    return Path(root) / "receipts" / f"{object_id(url)}.json"


def _header_pairs(headers) -> list[list[str]]:
    """httpx multi_items() -> a sorted, JSON-canonical header inventory."""
    items = (headers.multi_items() if hasattr(headers, "multi_items")
             else list(headers.items()))
    return sorted([str(key), str(value)] for key, value in items)


def seal_page_fetch(root: Path, url: str, response, fetched_at: str,
                    expected_user_agent: str) -> dict[str, Any]:
    """Seal one publisher response as bytes + transport + request identity."""
    root = Path(root)
    body = bytes(response.content)
    request = response.request
    transport = {
        "schema": TRANSPORT_SCHEMA,
        "requested_url": url,
        "request_method": str(request.method),
        "request_url": str(request.url),
        "request_headers": _header_pairs(request.headers),
        "final_url": str(response.url),
        "http_status": int(response.status_code),
        "fetched_at": fetched_at,
        "response_headers": _header_pairs(response.headers),
        "body": {"bytes": len(body),
                 "sha256": hashlib.sha256(body).hexdigest()},
    }
    agents = [value for key, value in transport["request_headers"]
              if key.casefold() == "user-agent"]
    # Consent evidence is only worth something if it names the crawler that
    # actually made the request (task 432).  An unbound identity is not
    # sealable — and a page we cannot honestly attribute is not admitted.
    if len(agents) != 1 or agents[0] != expected_user_agent:
        raise PagePreservationError(
            f"publisher-page request identity is unbound for {url}")
    return {
        "schema": RECEIPT_SCHEMA,
        "object_id": object_id(url),
        "requested_url": url,
        "final_url": transport["final_url"],
        "http_status": transport["http_status"],
        "content_type": str(response.headers.get("content-type", "")),
        "fetched_at": fetched_at,
        "body_sha256": transport["body"]["sha256"],
        "body_bytes": transport["body"]["bytes"],
        "artifacts": {
            "body": PS.store_artifact(root, BODY_FAMILY, "html", body),
            "transport": PS.store_artifact(
                root, TRANSPORT_FAMILY, "json",
                (canonical_json(transport) + "\n").encode("utf-8")),
        },
    }


def valid_receipt(root: Path, path: Path, url: str) -> dict[str, Any] | None:
    """Reuse an already-sealed fetch instead of re-GETting it.

    A receipt whose artifacts do not validate is not evidence, so it is treated
    as absent rather than trusted.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (value.get("schema") != RECEIPT_SCHEMA or
                value.get("requested_url") != url or
                value.get("object_id") != object_id(url)):
            return None
        for artifact in value["artifacts"].values():
            PS.validate_artifact(Path(root), artifact)
        return value
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError,
            PS.PreservedArtifactError):
        return None


def publish_receipt(root: Path, url: str, receipt: dict[str, Any]) -> Path:
    """Publish an immutable receipt; never replace an existing path."""
    path = receipt_path(root, url)
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        raise PagePreservationError(f"publisher-page receipt collision: {path}")
    atomic_publish(path, encoded)
    return path


def load_body(root: Path, receipt: dict[str, Any]) -> bytes:
    """Read the sealed response bytes back, validated, for extraction."""
    return PS.load_validated_artifact(Path(root), receipt["artifacts"]["body"])


def decode_body(body: bytes, content_type: str | None) -> str:
    """Decode preserved bytes to text using the declared charset when usable.

    Mirrors ``hn_freshness_ingest._decode_html``: the declared charset first,
    then UTF-8, then a UTF-8 replacement decode.  The bytes stay the evidence;
    this is only the reading of them.
    """
    declared = ""
    for part in (content_type or "").split(";"):
        part = part.strip()
        if part.casefold().startswith("charset="):
            declared = part.split("=", 1)[1].strip().strip('"\'')
            break
    for encoding in (declared, "utf-8"):
        if encoding:
            try:
                return body.decode(encoding)
            except (LookupError, UnicodeDecodeError):
                continue
    return body.decode("utf-8", "replace")


def is_admissible_article(receipt: dict[str, Any]) -> bool:
    """A 200 HTML response is the only shape that can carry article content.

    A non-200 or non-HTML response is still preserved as honest evidence of
    what the publisher actually returned; it simply produces no content row.
    """
    if receipt.get("http_status") != 200:
        return False
    content_type = receipt.get("content_type") or "text/html"
    return "html" in content_type.casefold()
