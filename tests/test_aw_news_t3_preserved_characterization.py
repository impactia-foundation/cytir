#!/usr/bin/env python3
"""Task 683 S7 (news port, tranche 3): the preserved-page store modules, pinned through their
OLD `pipeline/` paths before and after they move into `cytir/acquire/`. Passes unchanged on the shims."""
from __future__ import annotations

import hashlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


def test_preserved_stage_io_store_artifact_is_content_addressed_and_byte_stable(tmp_path):
    sys.path.insert(0, str(HERE))
    from cytir.acquire import preserved_stage_io as ps
    art = ps.store_artifact(tmp_path, "original/x", "html", b"<p>hallo</p>")
    body_sha = hashlib.sha256(b"<p>hallo</p>").hexdigest()
    assert art["uri"] == f"artifact://original/x/{body_sha[:2]}/{body_sha}.html.gz"
    assert (art["compression"], art["body_bytes"], art["body_sha256"]) == ("gzip", 12, body_sha)
    assert art["sha256"] == hashlib.sha256(ps.deterministic_gzip(b"<p>hallo</p>")).hexdigest()
    assert ps.load_validated_artifact(tmp_path, art) == b"<p>hallo</p>"
    assert ps.store_artifact(tmp_path, "original/x", "html", b"<p>hallo</p>") == art   # idempotent
    for bad in ({"uri": "http://x"}, {"uri": "artifact://../escape"}):
        try:
            ps.artifact_path(tmp_path, bad)
        except ps.PreservedArtifactError:
            continue
        raise AssertionError(bad)
    ps.artifact_path(tmp_path, art).write_bytes(b"corrupt")
    try:
        ps.validate_artifact(tmp_path, art)
    except ps.PreservedArtifactError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("a corrupt artifact must be refused")


def test_artifact_io_immutable_publish_and_canonical_json(tmp_path):
    sys.path.insert(0, str(HERE))
    from cytir.acquire import artifact_io as aio
    assert aio.canonical_json({"b": 1, "a": "ü"}) == '{"a":"ü","b":1}'
    path = tmp_path / "sub" / "rows.jsonl"
    digest, count = aio.publish_jsonl_or_verify(path, [{"b": 1, "a": 2}, {"a": 3}])
    assert (count, path.read_text()) == (2, '{"a":2,"b":1}\n{"a":3}\n')
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert aio.publish_jsonl_or_verify(path, [{"b": 1, "a": 2}, {"a": 3}]) == (digest, 2)
    try:
        aio.publish_jsonl_or_verify(path, [{"other": 1}])
    except ValueError as exc:
        assert "immutable artifact collision" in str(exc)
    else:
        raise AssertionError("a content collision on an immutable path must be refused")
    try:
        aio.atomic_publish(path, b"x")
    except FileExistsError:
        pass
    else:
        raise AssertionError("atomic_publish never replaces")
    assert aio.uncompressed_info(path)["bytes"] == len(path.read_bytes())


class _H(dict):
    def multi_items(self):
        return list(self.items())


class _Resp:
    def __init__(self, ua="AW-Bot/1"):
        self.content = b"<html>x</html>"
        self.status_code = 200
        self.url = "https://www.rts.ch/a"
        self.headers = _H({"content-type": "text/html; charset=utf-8"})
        self.request = type("R", (), {"method": "GET", "url": "https://www.rts.ch/a",
                                      "headers": _H({"user-agent": ua})})()


def test_preserved_page_fetch_seals_bytes_and_refuses_an_unbound_identity(tmp_path):
    sys.path.insert(0, str(HERE))
    from cytir.acquire import preserved_page_fetch as ppf
    url = "https://www.rts.ch/a"
    receipt = ppf.seal_page_fetch(tmp_path, url, _Resp(), "2026-10-03T00:00:00Z", "AW-Bot/1")
    assert (receipt["schema"], receipt["object_id"], receipt["http_status"]) == (
        "aw.news-publisher-page-receipt.v1", hashlib.sha256(url.encode()).hexdigest(), 200)
    assert receipt["artifacts"]["body"]["uri"].startswith("artifact://original/publisher_pages/")
    assert receipt["artifacts"]["transport"]["uri"].startswith("artifact://original/publisher_transport/")
    path = ppf.publish_receipt(tmp_path, url, receipt)
    assert path == tmp_path / "receipts" / f"{receipt['object_id']}.json"
    assert ppf.valid_receipt(tmp_path, path, url) == receipt
    assert ppf.load_body(tmp_path, receipt) == b"<html>x</html>"
    assert ppf.is_admissible_article(receipt)
    assert ppf.decode_body(b"<html>x</html>", "text/html") == "<html>x</html>"
    try:
        ppf.publish_receipt(tmp_path, url, receipt)
    except ppf.PagePreservationError:
        pass
    else:
        raise AssertionError("a receipt is immutable")
    try:
        ppf.seal_page_fetch(tmp_path, url, _Resp(ua="other"), "t", "AW-Bot/1")
    except ppf.PagePreservationError as exc:
        assert "identity is unbound" in str(exc)
    else:
        raise AssertionError("an unbound crawler identity is not sealable")
