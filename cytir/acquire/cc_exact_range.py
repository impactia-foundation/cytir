"""Exact-URL Common Crawl capture helpers shared by the CC recovery drivers (task 1410, S7).

Moved verbatim from the news truncated-CC recovery driver, which now imports them from
here: the URL-variant rule, the newest-complete capture choice, the signed-S3 / HTTPS WARC range
fetch and the WARC response parse. One home for the code (architecture-doctrine: one way per verb);
`cytir.sources.exact_url.cc_preserved` and the news recovery driver both call it.
"""
from __future__ import annotations

import hashlib
import importlib
import time
from urllib.parse import urlsplit, urlunsplit

import httpx

WARC_BASE = "https://data.commoncrawl.org/"


def exact_url_variants(url: str) -> list[str]:
    parsed = urlsplit(url.split("#", 1)[0])
    if not parsed.hostname:
        return []
    hosts = [parsed.hostname.casefold()]
    hosts.append(hosts[0][4:] if hosts[0].startswith("www.") else "www." + hosts[0])
    return list(dict.fromkeys(urlunsplit((scheme, host, parsed.path or "/",
                                         parsed.query, ""))
                              for scheme in (parsed.scheme, "https", "http")
                              for host in hosts))


def choose_exact(records: dict[str, dict], variants: list[str]) -> dict | None:
    candidates = [(record, -variants.index(key)) for key, record in records.items()
                  if key in variants]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (
        str(item[0].get("fetch_time") or ""), int(item[0].get("length") or 0),
        str(item[0].get("warc_filename") or ""),
        -int(item[0].get("offset") or 0), item[1]))[0]


def _default_fetch(client: httpx.Client, capture: dict,
                   transport: str = "signed-s3") -> tuple[bytes, dict]:
    if transport not in {"signed-s3", "https-mirror"}:
        raise ValueError(f"unsupported Common Crawl WARC transport: {transport}")
    key = capture["warc_filename"]
    url = WARC_BASE + key
    start, length = int(capture["offset"]), int(capture["length"])
    headers = {"Range": f"bytes={start}-{start + length - 1}"}
    last = None
    for attempt in range(1, 9):
        try:
            if transport == "signed-s3":
                COLUMNAR = importlib.import_module("cytir.acquire.cc_columnar")
                response = COLUMNAR._s3_client().get_object(
                    Bucket=COLUMNAR.S3_BUCKET, Key=key,
                    Range=headers["Range"])
                content = response["Body"].read()
                status = int((response.get("ResponseMetadata") or {}).get(
                    "HTTPStatusCode", 0))
                response_headers = sorted(
                    (str(k), str(v)) for k, v in
                    ((response.get("ResponseMetadata") or {}).get(
                        "HTTPHeaders") or {}).items())
                requested = final = f"s3://{COLUMNAR.S3_BUCKET}/{key}"
            else:
                response = client.get(url, headers=headers)
                content = bytes(response.content)
                status = response.status_code
                response_headers = sorted((str(k), str(v))
                                          for k, v in response.headers.multi_items())
                requested, final = url, str(response.url)
            if status != 206 or len(content) != length:
                raise ValueError(f"WARC range response {status}/{len(content)}")
            return content, {
                "schema": "aw.common-crawl-range-transport.v1",
                "route": ("common-crawl-s3-warc-range" if transport == "signed-s3"
                          else "common-crawl-https-warc-range"),
                "requested_url": requested, "final_url": final,
                "http_status": 206, "request_range": headers["Range"],
                "retrieved_at": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc).isoformat().replace("+00:00", "Z"),
                "response_headers": response_headers,
                "raw_warc": {"bytes": len(content),
                             "sha256": hashlib.sha256(content).hexdigest()},
                "attempt": attempt,
            }
        except Exception as exc:  # network SDK boundary; bounded retries below
            last = exc
            time.sleep(min(attempt, 5))
    raise RuntimeError(f"WARC range exhausted retries: {last}")


def _default_parse(raw_warc: bytes) -> tuple[bytes, bytes, str]:
    from cytir.acquire import discovery_ingest as DI
    parts = DI._warc_response_parts(raw_warc)
    if parts is None:
        raise ValueError("WARC response record cannot be parsed")
    http_headers, body = parts
    return http_headers, body, DI._decode_http_body(http_headers, body)
