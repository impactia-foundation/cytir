import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from cytir.news import truncated_cc_recovery as subject
from cytir.index.preservation_contract import _validate_stage


def test_default_warc_fetch_uses_signed_s3_and_preserves_transport(monkeypatch):
    content = b"exact compressed WARC range"

    class Body:
        def read(self):
            return content

    class Client:
        def get_object(self, **kwargs):
            assert kwargs == {"Bucket": "commoncrawl", "Key": "crawl/a.warc.gz",
                              "Range": f"bytes=10-{10 + len(content) - 1}"}
            return {"Body": Body(), "ResponseMetadata": {
                "HTTPStatusCode": 206,
                "HTTPHeaders": {"content-range": "bytes fixture"},
            }}

    monkeypatch.setitem(sys.modules, "cytir.acquire.cc_columnar", SimpleNamespace(
        S3_BUCKET="commoncrawl", _s3_client=lambda: Client()))
    raw, transport = subject._default_fetch(None, {
        "warc_filename": "crawl/a.warc.gz", "offset": 10,
        "length": len(content)}, transport="signed-s3")
    assert raw == content
    assert transport["route"] == "common-crawl-s3-warc-range"
    assert transport["requested_url"] == "s3://commoncrawl/crawl/a.warc.gz"
    assert transport["raw_warc"]["sha256"] == hashlib.sha256(content).hexdigest()


def test_exact_cc_capture_preserves_warc_http_body_html_and_transport(tmp_path: Path):
    source_root = tmp_path / "source"
    raw_path = source_root / "raw/example.raw.jsonl.gz"
    raw_path.parent.mkdir(parents=True)
    stored_prefix = "<html>prefix"
    row = {"url": "https://www.example.org/a/",
           "final_url": "https://www.example.org/a/", "http_status": 200,
           "html": stored_prefix, "sha1": "f" * 40}
    raw_path.write_bytes(gzip.compress((json.dumps(row) + "\n").encode(), mtime=0))
    (source_root / "FILE-SHA256SUMS").write_text("fixture\n")
    capture = {"url": row["url"], "warc_filename": "crawl/a.warc.gz",
               "offset": 10, "length": 8, "digest": "sha1:FIXTURE",
               "fetch_time": "2026-08-01T00:00:00Z",
               "content_truncated": False, "truncation_reason": None}

    result = subject.acquire(
        source_root=source_root, root=tmp_path / "out", crawl="CC-MAIN-X",
        expected_frontier=1, rate=4,
        resolver=lambda _urls: {row["url"]: capture},
        fetcher=lambda _capture: (b"raw-warc", {
            "schema": "aw.common-crawl-range-transport.v1",
            "requested_url": "https://data.commoncrawl.org/crawl/a.warc.gz",
            "final_url": "https://data.commoncrawl.org/crawl/a.warc.gz",
            "http_status": 206, "request_range": "bytes=10-17",
            "response_headers": [], "raw_warc": {"bytes": 8,
            "sha256": hashlib.sha256(b"raw-warc").hexdigest()}}),
        parser=lambda _warc: (b"HTTP/1.1 200 OK\r\n\r\n", b"full-body",
                              "<html>full body</html>"))
    assert result["counts"] == {"frontier": 1, "cc_captures": 1, "cc_misses": 0}
    checked = _validate_stage(result["stages"]["original"], "original",
                              artifact_root=tmp_path / "out", deep=True)
    assert checked["records"] == 1
    receipt = next((tmp_path / "out/receipts").glob("*.json"))
    artifacts = json.loads(receipt.read_text())["artifacts"]
    assert set(artifacts) == {"raw_warc", "http_headers", "response_body",
                              "decoded_html", "transport"}


def test_driver_binds_time_module_for_https_mirror_sleep():
    # task 1410 review: the https-mirror branch calls time.sleep; a dropped `import time` is a NameError.
    import time

    assert subject.time is time
    assert callable(subject.time.sleep)
