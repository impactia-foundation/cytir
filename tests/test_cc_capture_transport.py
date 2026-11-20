#!/usr/bin/env python3
"""Offline contract tests for deterministic CC capture selection and WARC bytes."""
from __future__ import annotations

import gzip
import io
import os
import sys
import types
from unittest import mock

import duckdb


from cytir.acquire import cc_columnar as ccx  # noqa: E402
# WARC transport tests do not exercise extraction.  Avoid importing the heavy
# trafilatura dependency that discovery_ingest registers for its worker pool.
# task 530: the stub MUST be removed again after this import — `setdefault`
# with no teardown leaked an attribute-less fetch_process into sys.modules
# for the rest of the pytest process, poisoning every later `import
# fetch_process` (chunk/detect_lang/_LID_MODEL_PATH all AttributeError) in
# whatever test file happened to run next alphabetically.
_fp_stub_inserted = "fetch_process" not in sys.modules
if _fp_stub_inserted:
    sys.modules["fetch_process"] = types.ModuleType("fetch_process")
try:
    from cytir.acquire import discovery_ingest as di  # noqa: E402
finally:
    if _fp_stub_inserted:
        del sys.modules["fetch_process"]


def _write_index(path):
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE captures (
          url_surtkey VARCHAR, url VARCHAR, warc_filename VARCHAR,
          warc_record_offset BIGINT, warc_record_length BIGINT,
          content_digest VARCHAR, fetch_time TIMESTAMP,
          content_truncated VARCHAR, fetch_status INTEGER,
          content_mime_type VARCHAR, url_path VARCHAR
        )
    """)
    url = "https://example.com/"
    key = ccx.url_surtkey(url)
    con.executemany(
        "INSERT INTO captures VALUES (?, ?, ?, ?, ?, ?, ?, ?, 200, 'text/html', '/')",
        [
            # Newest wins even though it is shorter.
            (key, url, "old-large.warc.gz", 10, 900, "sha1:OLD",
             "2026-07-01 00:00:00", None),
            (key, url, "new-small.warc.gz", 20, 300, "sha1:NEW",
             "2026-07-02 00:00:00", ""),
            # Same time: length is the second selection key.
            (key, url, "new-large.warc.gz", 30, 400, "sha1:WIN",
             "2026-07-02 00:00:00", "length"),
            # A newer status-200 text/html row with an empty payload is not a
            # usable capture and must not displace the older real body.
            (key, url, "newest-empty.warc.gz", 40, 500,
             ccx.CC_DIGEST.EMPTY_SHA1_BASE32,
             "2026-07-03 00:00:00", ""),
        ],
    )
    con.execute("COPY captures TO ? (FORMAT PARQUET)", [str(path)])
    con.close()
    return url, key


def test_exact_url_resolver_emits_capture_metadata_and_pins_selection(tmp_path):
    parquet = tmp_path / "captures.parquet"
    url, key = _write_index(parquet)
    con = duckdb.connect()
    with mock.patch.object(ccx, "file_bounds", return_value=[
        (str(parquet), key, key),
    ]):
        got = ccx.resolve("offline", [url], con=con)[url]
    con.close()

    assert got == {
        "url": url,
        "warc_filename": "new-large.warc.gz",
        "offset": 30,
        "length": 400,
        "digest": "sha1:WIN",
        "fetch_time": "2026-07-02T00:00:00Z",
        "content_truncated": True,
        "truncation_reason": "length",
    }


def test_exact_file_resolver_skips_newer_empty_payload_capture(tmp_path):
    parquet = tmp_path / "captures.parquet"
    url, key = _write_index(parquet)
    con = duckdb.connect()
    got = ccx.resolve_exact_file(con, str(parquet), [(key, url)])
    con.close()

    assert got["any"][url]["warc_filename"] == "new-large.warc.gz"
    assert got["complete"][url]["warc_filename"] == "new-small.warc.gz"


def test_exact_url_resolver_can_prefer_older_complete_capture(tmp_path):
    parquet = tmp_path / "captures.parquet"
    url, key = _write_index(parquet)
    con = duckdb.connect()
    with mock.patch.object(ccx, "file_bounds", return_value=[
        (str(parquet), key, key),
    ]):
        got = ccx.resolve(
            "offline", [url], con=con, require_complete=True
        )[url]
    con.close()

    # The newest/largest row is truncated. Of the complete rows, the newest
    # one wins even though an older one has a larger WARC record.
    assert got["warc_filename"] == "new-small.warc.gz"
    assert got["fetch_time"] == "2026-07-02T00:00:00Z"
    assert got["content_truncated"] is False
    assert got["truncation_reason"] is None


def test_domain_resolver_uses_same_capture_selection_contract(tmp_path):
    parquet = tmp_path / "captures.parquet"
    url, key = _write_index(parquet)
    con = duckdb.connect()
    with mock.patch.object(ccx, "file_bounds", return_value=[
        (str(parquet), key, key),
    ]):
        got = ccx.resolve_domains("offline", ["example.com"], con=con)
    con.close()

    assert len(got["example.com"]) == 1
    assert got["example.com"][0]["warc_filename"] == "new-large.warc.gz"
    assert got["example.com"][0]["fetch_time"] == "2026-07-02T00:00:00Z"
    assert got["example.com"][0]["content_truncated"] is True
    assert got["example.com"][0]["truncation_reason"] == "length"


def test_domain_resolver_can_require_complete_capture(tmp_path):
    parquet = tmp_path / "captures.parquet"
    _url, key = _write_index(parquet)
    con = duckdb.connect()
    with mock.patch.object(ccx, "file_bounds", return_value=[
        (str(parquet), key, key),
    ]):
        got = ccx.resolve_domains(
            "offline", ["example.com"], con=con, require_complete=True
        )
    con.close()

    assert got["example.com"][0]["warc_filename"] == "new-small.warc.gz"
    assert got["example.com"][0]["content_truncated"] is False


def test_resolver_error_is_not_collapsed_to_empty_result():
    con = mock.Mock()
    with mock.patch.object(ccx, "file_bounds", side_effect=RuntimeError("offline")):
        try:
            ccx.resolve("offline", ["https://example.com/"], con=con)
        except RuntimeError as exc:
            assert str(exc) == "offline"
        else:  # pragma: no cover - makes the contract failure explicit
            raise AssertionError("resolver transport error was swallowed as empty")


def test_genuine_empty_inputs_remain_empty_without_querying():
    con = mock.Mock()
    assert ccx.resolve("offline", [], con=con) == {}
    assert ccx.resolve_domains("offline", [], con=con) == {}
    con.execute.assert_not_called()


def test_truncation_string_normalisation_distinguishes_complete_and_reason():
    assert ccx._truncation_metadata(None) == (False, None)
    assert ccx._truncation_metadata("") == (False, None)
    assert ccx._truncation_metadata("  ") == (False, None)
    assert ccx._truncation_metadata("length") == (True, "length")


def _warc_range(body: bytes, charset: bytes = b"windows-1252") -> bytes:
    payload = (
        b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset="
        + charset
        + b"\r\n\r\n"
        + body
    )
    raw = (b"WARC/1.0\r\nWARC-Type: response\r\nContent-Length: "
           + str(len(payload)).encode() + b"\r\n\r\n" + payload
           + b"\r\n\r\n")
    return gzip.compress(raw)


class _MirrorResponse:
    status_code = 206

    def __init__(self, content: bytes):
        self.content = content


class _MirrorClient:
    def __init__(self, content: bytes):
        self.content = content
        self.calls = []

    def get(self, url, headers):
        self.calls.append((url, headers))
        return _MirrorResponse(self.content)


def test_mirror_exact_body_bytes_and_legacy_decoded_text_are_both_available():
    body = b"prix: 10 \x80"
    record = _warc_range(body)
    rec = {"warc_filename": "crawl-data/x.warc.gz", "offset": 7,
           "length": len(record)}
    client = _MirrorClient(record)

    with mock.patch.object(ccx, "use_s3", return_value=False), \
         mock.patch.object(di, "_throttle"):
        assert di.fetch_warc_response_body(client, rec) == body
        assert di.fetch_warc(client, rec) == "prix: 10 €"

    assert client.calls[0] == (
        di.WARC + "crawl-data/x.warc.gz",
        {**di.UA, "Range": f"bytes=7-{7 + len(record) - 1}"},
    )


def test_signed_s3_exact_body_bytes_use_the_same_columnar_record():
    body = b"\x00exact\xffbody"
    record = _warc_range(body, charset=b"iso-8859-1")
    stream = io.BytesIO(record)
    s3 = mock.Mock()
    s3.get_object.return_value = {"Body": stream}
    rec = {"warc_filename": "crawl-data/x.warc.gz", "offset": 11,
           "length": len(record)}

    with mock.patch.object(ccx, "use_s3", return_value=True), \
         mock.patch.object(di, "_warc_s3_client", return_value=s3):
        assert di.fetch_warc_bytes(None, rec) == body

    s3.get_object.assert_called_once_with(
        Bucket=ccx.S3_BUCKET,
        Key="crawl-data/x.warc.gz",
        Range=f"bytes=11-{11 + len(record) - 1}",
    )


def test_signed_s3_failure_is_single_attempt_and_keeps_range_identity():
    s3 = mock.Mock()
    s3.get_object.side_effect = TimeoutError("fixture timeout")
    rec = {"warc_filename": "crawl-data/x.warc.gz", "offset": 11,
           "length": 17}
    with mock.patch.object(di, "_warc_s3_client", return_value=s3):
        try:
            di._fetch_warc_record_s3(rec)
        except di.WarcRangeFetchError as exc:
            message = str(exc)
            assert "crawl-data/x.warc.gz" in message
            assert "bytes=11-27" in message
            assert "TimeoutError" in message
        else:
            raise AssertionError("signed-S3 failure was swallowed")
    assert s3.get_object.call_count == 1


def test_empty_archived_body_is_not_a_fetch_error():
    record = _warc_range(b"")
    client = _MirrorClient(record)
    with mock.patch.object(ccx, "use_s3", return_value=False), \
         mock.patch.object(di, "_throttle"):
        assert di.fetch_warc_response_body(
            client, {"filename": "x", "offset": 0, "length": len(record)}
        ) == b""


def test_malformed_warc_is_error_not_empty():
    record = gzip.compress(b"not a response record")
    client = _MirrorClient(record)
    with mock.patch.object(ccx, "use_s3", return_value=False), \
         mock.patch.object(di, "_throttle"):
        assert di.fetch_warc_response_body(
            client, {"filename": "x", "offset": 0, "length": len(record)}
        ) is None


def test_warc_parser_rejects_declared_block_length_drift():
    raw = (b"WARC/1.0\r\nWARC-Type: response\r\nContent-Length: 999\r\n\r\n"
           b"HTTP/1.1 200 OK\r\n\r\nbody\r\n\r\n")
    assert di._warc_response_parts(gzip.compress(raw)) is None
