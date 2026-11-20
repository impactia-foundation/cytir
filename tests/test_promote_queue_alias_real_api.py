#!/usr/bin/env python3
"""`resolve_prod_alias` against the REAL `PSP._api` (no fake of `_api`).

Bug found live 2026-09-29, right after the aw_e9 flip: `_api` returns
`payload.get("result")`, and `resolve_prod_alias` read `result` a second
time, so it never found an alias and refused every drain and finalize. Every
test faked `_api` with the WRAPPED shape, so all stayed green. This module
serves qdrant's exact body from a local HTTP server and runs the real
transport, so a fake can no longer disagree with the wire.
"""
from __future__ import annotations

import http.server
import json
import os
import pathlib
import sys
import threading

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

@pytest.fixture(autouse=True)
def _operator_identity(monkeypatch):
    """Operator identity for each test, restored after it. A module-level setdefault leaked it into every later test."""
    monkeypatch.setenv("AW_OPERATOR_URL", "https://websear.ch")
    monkeypatch.setenv("AW_OPERATOR_CONTACT", "bender@impactia.org")

from cytir.index import promote_queue as Q  # noqa: E402

# Verbatim from `curl 127.0.0.1:6334/aliases` on the build box, 2026-09-29.
QDRANT_ALIASES_BODY = (
    '{"result":{"aliases":[{"alias_name":"aw_prod",'
    '"collection_name":"aw_e9"}]},"status":"ok"}')
QDRANT_NO_ALIAS_BODY = '{"result":{"aliases":[]},"status":"ok"}'


@pytest.fixture
def qdrant_stub():
    served = {"body": QDRANT_ALIASES_BODY, "paths": []}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            served["paths"].append(self.path)
            if self.path != "/aliases":
                self.send_response(404)
                self.end_headers()
                return
            raw = served["body"].encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", served
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_resolve_prod_alias_reads_qdrants_real_aliases_body(qdrant_stub):
    url, served = qdrant_stub
    assert Q.resolve_prod_alias(url) == "aw_e9"
    assert served["paths"] == ["/aliases"]


def test_resolve_prod_alias_raises_when_alias_absent(qdrant_stub):
    url, served = qdrant_stub
    served["body"] = QDRANT_NO_ALIAS_BODY
    with pytest.raises(Q.QueueError, match="alias 'aw_prod' not found"):
        Q.resolve_prod_alias(url)


def test_resolve_prod_alias_raises_when_other_alias_only(qdrant_stub):
    url, served = qdrant_stub
    served["body"] = json.dumps({"result": {"aliases": [
        {"alias_name": "aw_old", "collection_name": "aw_e8"}]},
        "status": "ok"})
    with pytest.raises(Q.QueueError, match="not found"):
        Q.resolve_prod_alias(url)
