#!/usr/bin/env python3
"""Offline unit tests for the CC transport-selection logic (task 124).

The signed-S3 fast path must be picked ONLY when it can actually work, and must
otherwise fall back to the HTTPS mirror byte-for-byte as before. The auto rule
is what keeps the same pipeline correct on every box:

  * creds absent            -> mirror  (can't sign an S3 GET → anonymous = 403)
  * creds present, live probe FAIL -> mirror (transport failure fallback)
  * creds present, probe OK  -> S3
  * AW_CC_S3=0               -> mirror  (forced, no probe, even if S3 works)
  * AW_CC_S3=1               -> S3 iff creds  (forced, skip probe; no creds→mirror)

and the base-prefix / manifest-key derivation must be transport-correct so the
S3 key is the mirror path suffix verbatim.

Run: python3 -m pytest tests/test_cc_s3_selection.py
 (or: python3 tests/test_cc_s3_selection.py)
"""
from __future__ import annotations

import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cytir.acquire import cc_columnar as ccx  # noqa: E402


def _decide(env, creds, probe):
    """Evaluate use_s3() under a controlled env + stubbed creds/probe."""
    ccx._reset_s3_decision()
    envpatch = {k: v for k, v in env.items() if v is not None}
    # ensure a stale AW_CC_S3 doesn't leak in from the caller's shell
    with mock.patch.dict(os.environ, envpatch, clear=False):
        if "AW_CC_S3" not in envpatch:
            os.environ.pop("AW_CC_S3", None)
        with mock.patch.object(ccx, "_creds_available", return_value=creds), \
             mock.patch.object(ccx, "_probe_s3", return_value=probe) as p:
            got = ccx.use_s3()
            return got, p


def test_no_creds_falls_back_to_mirror():
    # auto mode, no credential in the chain -> mirror, and we must NOT even probe
    got, probe = _decide({}, creds=False, probe=True)
    assert got is False
    probe.assert_not_called()  # short-circuit: no creds → don't waste a probe


def test_creds_but_probe_fails_falls_back_to_mirror():
    # build-box case: creds exist but the AWS route is blackholed → probe fails
    got, probe = _decide({}, creds=True, probe=False)
    assert got is False
    probe.assert_called_once()


def test_creds_and_probe_ok_selects_s3():
    got, _ = _decide({}, creds=True, probe=True)
    assert got is True


def test_env_force_off_uses_mirror_even_when_s3_works():
    got, probe = _decide({"AW_CC_S3": "0"}, creds=True, probe=True)
    assert got is False
    probe.assert_not_called()  # forced off → never probes


def test_env_force_on_selects_s3_without_probing():
    got, probe = _decide({"AW_CC_S3": "1"}, creds=True, probe=False)
    assert got is True         # forced on ignores a failing probe...
    probe.assert_not_called()  # ...and skips it entirely


def test_env_force_on_without_creds_still_falls_back():
    # can't sign without a key → forced-on is honoured only if creds exist
    got, _ = _decide({"AW_CC_S3": "1"}, creds=False, probe=True)
    assert got is False


def test_decision_is_cached_per_process():
    ccx._reset_s3_decision()
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("AW_CC_S3", None)
        with mock.patch.object(ccx, "_creds_available", return_value=True) as cr, \
             mock.patch.object(ccx, "_probe_s3", return_value=True):
            assert ccx.use_s3() is True
            assert ccx.use_s3() is True
            cr.assert_called_once()   # second call served from cache


def test_s3_client_is_connection_pooled_per_process():
    ccx._s3_client.cache_clear()
    sentinel = object()
    with mock.patch("boto3.client", return_value=sentinel) as create:
        assert ccx._s3_client() is sentinel
        assert ccx._s3_client() is sentinel
    create.assert_called_once()
    ccx._s3_client.cache_clear()


# --- transport-correct path derivation --------------------------------------
def test_manifest_key_is_transport_independent():
    assert ccx._manifest_key("CC-MAIN-2026-25") == \
        "crawl-data/CC-MAIN-2026-25/cc-index-table.paths.gz"


def test_crawl_paths_prefixes_s3_when_selected():
    lines = [
        "cc-index/table/cc-main/warc/crawl=CC-MAIN-2026-25/subset=warc/part-0.parquet",
        "cc-index/table/cc-main/robotstxt/crawl=CC-MAIN-2026-25/subset=robotstxt/x.parquet",
    ]
    ccx._PATHS_CACHE.pop("CC-MAIN-2026-25", None)
    ccx._reset_s3_decision()
    with mock.patch.object(ccx, "use_s3", return_value=True), \
         mock.patch.object(ccx, "_read_manifest_lines", return_value=lines):
        paths = ccx.crawl_paths("CC-MAIN-2026-25")
    assert paths == [ccx.S3_PREFIX +
                     "cc-index/table/cc-main/warc/crawl=CC-MAIN-2026-25/"
                     "subset=warc/part-0.parquet"]          # only warc subset
    assert all(p.startswith("s3://commoncrawl/") for p in paths)


def test_crawl_paths_prefixes_mirror_when_not_s3():
    lines = ["cc-index/table/cc-main/warc/crawl=CC-MAIN-2026-25/subset=warc/p.parquet"]
    ccx._PATHS_CACHE.pop("CC-MAIN-2026-25", None)
    with mock.patch.object(ccx, "use_s3", return_value=False), \
         mock.patch.object(ccx, "_read_manifest_lines", return_value=lines):
        paths = ccx.crawl_paths("CC-MAIN-2026-25")
    assert paths == [ccx.MIRROR +
                     "cc-index/table/cc-main/warc/crawl=CC-MAIN-2026-25/"
                     "subset=warc/p.parquet"]
    assert all(p.startswith("https://data.commoncrawl.org/") for p in paths)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
