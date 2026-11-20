"""`cytir.acquire.warc` against ONE real Common Crawl record (task 683, S7 part 2).

The fixture `fixtures/cc_real_record/member.gz` is the exact gzip member a single range read of
`data.commoncrawl.org` returned on 2026-09-30 (HTTP 206, 61,031 bytes). `record.json` pins its
coordinates and the sha256 of the L0 body that the chmedia583 lane had stored on the private build box two
months earlier. The synthetic tranche in `test_aw_family_cc.py` proves the plumbing; this proves
the parser against bytes a real crawler wrote. No network.
"""
import base64
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
FIX = HERE / "fixtures" / "cc_real_record"

from cytir.acquire import warc  # noqa: E402

REC = json.loads((FIX / "record.json").read_text())
MEMBER = (FIX / "member.gz").read_bytes()


def test_real_member_parses_to_the_stored_l0_body():
    assert len(MEMBER) == REC["length"]
    rec = warc.parse_warc_record(MEMBER)
    assert rec.status == 200
    assert rec.warc_headers["warc-type"] == "response"
    assert rec.warc_headers["warc-target-uri"] == REC["url"]
    assert rec.warc_headers["warc-date"] == REC["warc_date"]
    assert rec.http_headers["content-type"].startswith("text/html")
    # The L0 body the build box stored (the old pipeline copy) kept the record's 4-byte CRLFCRLF terminator;
    # the package parser cuts the block at WARC Content-Length, so it is that body minus 4 bytes.
    assert len(rec.body) == REC["body_bytes"] - 4 == int(rec.http_headers["content-length"])
    assert hashlib.sha256(rec.body + b"\r\n\r\n").hexdigest() == REC["sha256"]


def test_real_body_matches_the_crawlers_own_payload_digest():
    rec = warc.parse_warc_record(MEMBER)
    sha1_b32 = base64.b32encode(hashlib.sha1(rec.body).digest()).decode()
    assert rec.warc_headers["warc-payload-digest"] == "sha1:" + sha1_b32
    assert sha1_b32 == REC["digest"]


def test_range_get_asks_for_exactly_the_record_and_returns_its_bytes():
    seen = []

    class Resp:
        status_code = 206
        content = MEMBER

    class Client:
        def get(self, url, headers=None):
            seen.append((url, headers))
            return Resp()

    got = warc.range_get(Client(), REC, limiter=warc.RateLimiter(5.0, sleep=lambda s: None))
    assert got == MEMBER
    end = REC["offset"] + REC["length"] - 1
    assert seen == [(warc.WARC_BASE + REC["filename"], {"Range": f"bytes={REC['offset']}-{end}"})]


def test_a_truncated_real_member_is_an_error_not_a_body():
    import pytest
    with pytest.raises(warc.WarcReadError):
        warc.parse_warc_record(MEMBER[: len(MEMBER) // 2])
