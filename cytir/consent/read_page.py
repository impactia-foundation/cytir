"""`cyt consent read-page URL... --out DIR`: sealed licence/terms-page reads (task 2804).

One bounded GET per URL through the resolver's own fetch primitive (`resolver._get`: same
UA, same timeouts, redirect bound, curl fallback; TLS verification is never touched here).
The record, not the exit code, is the result: an unreadable page is exit 0 with
``blocked`` set. Declaration lanes cite the record file and quote with ``--quote``.

``--quote`` prints the first matching sentence-or-line fragment. The split is
``(?<=[.!?])\\s+|\\n+``, so it also breaks after "Art." or "e.g."; the quote may be a fragment.
The body is capped at ``resolver.BODY_BYTE_CAP`` (the curl fallback has no cap of its own);
``truncated`` records the cut, and ``bytes``/``sha256`` describe the stored (capped) body.

Coupling: this verb depends on the resolver's private ``_get`` and ``_client`` names and on
``_get`` returning ``(response | None, error | None)``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import socket
import sys
from datetime import datetime, timezone

from cytir.consent import resolver
from cytir.reader.extract import to_text

EXCERPT_CHARS = 4000
MIN_TEXT_CHARS = 500
CHALLENGE_MAX_CHARS = 1500
CHALLENGE_RX = re.compile(
    r"verify that requests are coming from real users|just a moment|"
    r"enable javascript and cookies|checking your browser|verification required|"
    r"attention required! \| cloudflare", re.I)


def classify(status: int, text: str, body: str) -> str:
    """'' when readable, else a short reason (the record keeps the reason)."""
    if status == 0:
        return "no_response"
    if status >= 400:
        return f"http_{status}"
    if len(text) < MIN_TEXT_CHARS and "<noscript" in body.lower():
        return "js_shell"
    probe = text if text.strip() else body
    if status == 200 and len(probe) < CHALLENGE_MAX_CHARS and CHALLENGE_RX.search(probe):
        return "challenge"
    if not text.strip():
        return "empty_text"
    return ""


def url_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def first_sentence(text: str, pattern: str) -> str | None:
    rx = re.compile(pattern, re.I)
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        if rx.search(sent):
            return sent.strip()
    return None


def read_page(client, url: str, out_dir: pathlib.Path) -> dict:
    resp, err = resolver._get(client, url)
    fetched = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if resp is None:
        body_b, status, ctype, final = b"", 0, "", url
    else:
        status = int(resp.status_code)
        content = getattr(resp, "content", None)
        body_b = bytes(content) if content is not None else resp.text.encode("utf-8", "replace")
        headers = getattr(resp, "headers", None) or {}
        ctype = headers.get("content-type", "") if hasattr(headers, "get") else ""
        final = str(getattr(resp, "url", "") or url)
    truncated = len(body_b) > resolver.BODY_BYTE_CAP
    body_b = body_b[:resolver.BODY_BYTE_CAP]
    html = body_b.decode("utf-8", "replace")
    plain = ctype.lower().startswith("text/plain") or "<" not in html[:256]
    text = (html if plain else to_text(html)) if body_b else ""
    key = url_key(url)
    rec = {
        "url": url, "final_url": final, "fetched_at": fetched,
        "vantage": socket.gethostname().lower(), "http_status": status,
        "content_type": ctype, "bytes": len(body_b),
        "sha256": hashlib.sha256(body_b).hexdigest(),
        "text_excerpt": text[:EXCERPT_CHARS], "text_source": "plain" if plain else "html",
        "blocked": classify(status, text, html),
        "truncated": truncated, "error": err or "", "body_file": f"{key}.body",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{key}.body").write_bytes(body_b)
    (out_dir / f"{key}.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    rec["_text"] = text
    return rec


def main(argv: list[str], prog: str = "cyt consent read-page") -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--out", required=True, help="directory for <sha256(url)[:16]>.json + .body")
    ap.add_argument("--quote", metavar="REGEX", help="print the first matching sentence")
    args = ap.parse_args(argv)
    if args.quote:
        try:
            re.compile(args.quote)
        except re.error as exc:
            print(f"{prog}: invalid --quote regex: {exc}", file=sys.stderr)
            return 2
    out = pathlib.Path(args.out)
    with resolver._client() as client:
        for url in args.urls:
            rec = read_page(client, url, out)
            state = f"unreadable ({rec['blocked']})" if rec["blocked"] else "readable"
            print(f"{rec['http_status']}\t{rec['bytes']}\t{state}\t{url}")
            if args.quote:
                hit = first_sentence(rec["_text"], args.quote)
                print(f"  quote: {hit}" if hit else "  quote: (no match)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
