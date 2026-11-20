"""Replay sealed HTTP replies with no network (task 683, S7).

A `tape` is a list of recorded replies (`{"method", "url", "form", "range", "status",
"content_type", "body" | "body_b64"}`; `form` is the POST body, `range` the `Range` request
header, both null when absent). `Tape.client()` returns an `httpx.Client` over
`httpx.MockTransport` that answers from it; a request the tape does not hold gets a 599 and is
listed in `Tape.unexpected`, so an offline run can assert it asked for nothing else.

This is what makes a source family reproducible from a clean clone: a stranger with the repo
and a sealed tape gets the same bytes the recorder saw, with no crawler identity, no consent
probe, no egress. The fedlex (#442) and sitemap tapes use the text form; a Common Crawl tape
uses `body_b64` (WARC members are gzip).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path


class Tape:
    def __init__(self, entries: list[dict]):
        self._replies = {(e["method"], e["url"], e.get("form"), e.get("range")): e
                         for e in entries}
        self.unexpected: list[str] = []
        self.served: list[tuple] = []

    @classmethod
    def load(cls, path_or_doc) -> "Tape":
        doc = path_or_doc
        if not isinstance(doc, (dict, list)):
            doc = json.loads(Path(path_or_doc).read_text(encoding="utf-8"))
        return cls(doc["tape"] if isinstance(doc, dict) else doc)

    def client(self):
        import httpx

        def handler(request: "httpx.Request") -> "httpx.Response":
            form = request.content.decode() if request.method == "POST" else None
            key = (request.method, str(request.url), form, request.headers.get("range"))
            hit = self._replies.get(key)
            if hit is None:
                self.unexpected.append(f"{key[0]} {key[1]} range={key[3]!r}"[:240])
                return httpx.Response(599, text="not on the tape")
            self.served.append(key)
            if "body_b64" in hit:
                content = base64.b64decode(hit["body_b64"])
            else:
                content = hit["body"].encode("utf-8")
            headers = {"content-type": hit["content_type"]} if hit.get("content_type") else {}
            return httpx.Response(hit["status"], content=content, headers=headers)

        return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
