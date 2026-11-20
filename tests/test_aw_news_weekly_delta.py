#!/usr/bin/env python3
"""Tests for the weekly-delta v1 change-detection + config-guard (task 169).

Covers the two pure cores that gate whether prod gets written:

  check_signal(entry, prev, get)  — native-signal classification, HTTP injected:
    * http_etag:  304 -> unchanged (skip);  new ETag != stored -> changed;
                  no stored ETag -> unknown (seed+embed once);  net error -> unknown
    * http_last_modified:  Last-Modified newer -> changed;  older/equal -> unchanged;
                  first sight -> unknown;  304 -> unchanged
    * sitemap_lastmod:  <lastmod> advanced -> changed;  same -> unchanged
    * none / absent:  -> none  (WAF-blocked snippets keep always-re-embed)

  config_guard(live, pinned)  — refuse to run on embedder/dim/quant/sparse drift.

  classify(...)  — end-to-end status assignment + cursor advance over a mixed set.

Run: python3 -m pytest tests/test_aw_news_weekly_delta.py -v
 (or: python3 tests/test_aw_news_weekly_delta.py)
"""
from __future__ import annotations

import sys

from cytir.news import weekly_delta as wd  # noqa: E402
from cytir.chunk.policy import DEFAULT_POLICY_PATH, load_chunk_policy  # noqa: E402


def _resp(status, headers=None, body=b""):
    return wd.HttpResp(status, headers or {}, body)


def _getter(resp):
    """A get() stub returning a fixed response and recording the request."""
    calls = []

    def get(url, headers=None, timeout=20):
        calls.append((url, dict(headers or {})))
        return resp

    get.calls = calls
    return get


# ------------------------------ http_etag ------------------------------ #

def test_etag_304_is_unchanged_and_sends_conditional_header():
    e = {"url": "https://x.org/v", "change_signal": {"type": "http_etag"}}
    get = _getter(_resp(304))
    status, cur = wd.check_signal(e, {"etag": "abc"}, get)
    assert status == "unchanged"
    assert cur == {}
    assert get.calls[0][1].get("If-None-Match") == "abc"  # conditional GET made


def test_etag_changed_when_new_differs_from_stored():
    e = {"url": "https://x.org/v", "change_signal": {"type": "http_etag"}}
    get = _getter(_resp(200, {"ETag": "NEW"}))
    status, cur = wd.check_signal(e, {"etag": "OLD"}, get)
    assert status == "changed"
    assert cur == {"etag": "NEW"}


def test_etag_unchanged_when_200_but_same_etag():
    e = {"url": "https://x.org/v", "change_signal": {"type": "http_etag"}}
    get = _getter(_resp(200, {"etag": "same"}))  # header case-insensitive
    status, cur = wd.check_signal(e, {"etag": "same"}, get)
    assert status == "unchanged"


def test_etag_first_sight_is_unknown_and_seeds_cursor():
    e = {"url": "https://x.org/v", "change_signal": {"type": "http_etag"}}
    get = _getter(_resp(200, {"ETag": "first"}))
    status, cur = wd.check_signal(e, {}, get)
    assert status == "unknown"            # embed once to seed
    assert cur == {"etag": "first"}


def test_etag_network_error_is_unknown():
    e = {"url": "https://x.org/v", "change_signal": {"type": "http_etag"}}
    get = _getter(_resp(0))  # DNS/timeout
    status, cur = wd.check_signal(e, {"etag": "abc"}, get)
    assert status == "unknown"


# --------------------------- http_last_modified --------------------------- #

def test_last_modified_newer_is_changed():
    e = {"url": "https://x.org/p", "change_signal": {"type": "http_last_modified"}}
    get = _getter(_resp(200, {"Last-Modified": "Wed, 09 Jul 2026 12:00:00 GMT"}))
    status, cur = wd.check_signal(
        e, {"last_modified": "Mon, 07 Jul 2026 12:00:00 GMT"}, get)
    assert status == "changed"
    assert cur == {"last_modified": "Wed, 09 Jul 2026 12:00:00 GMT"}


def test_last_modified_older_or_equal_is_unchanged():
    e = {"url": "https://x.org/p", "change_signal": {"type": "http_last_modified"}}
    get = _getter(_resp(200, {"Last-Modified": "Mon, 07 Jul 2026 12:00:00 GMT"}))
    status, _ = wd.check_signal(
        e, {"last_modified": "Wed, 09 Jul 2026 12:00:00 GMT"}, get)
    assert status == "unchanged"


def test_last_modified_304_is_unchanged():
    e = {"url": "https://x.org/p", "change_signal": {"type": "http_last_modified"}}
    get = _getter(_resp(304))
    status, _ = wd.check_signal(
        e, {"last_modified": "Mon, 07 Jul 2026 12:00:00 GMT"}, get)
    assert status == "unchanged"
    assert get.calls[0][1].get("If-Modified-Since") == "Mon, 07 Jul 2026 12:00:00 GMT"


def test_last_modified_first_sight_is_unknown():
    e = {"url": "https://x.org/p", "change_signal": {"type": "http_last_modified"}}
    get = _getter(_resp(200, {"Last-Modified": "Mon, 07 Jul 2026 12:00:00 GMT"}))
    status, cur = wd.check_signal(e, {}, get)
    assert status == "unknown"
    assert cur == {"last_modified": "Mon, 07 Jul 2026 12:00:00 GMT"}


# ----------------------------- sitemap_lastmod ---------------------------- #

_SITEMAP = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://x.org/a</loc><lastmod>2026-07-05</lastmod></url>
  <url><loc>https://x.org/p</loc><lastmod>2026-07-10</lastmod></url>
</urlset>"""


def test_sitemap_lastmod_advanced_is_changed():
    e = {"url": "https://x.org/p",
         "change_signal": {"type": "sitemap_lastmod", "url": "https://x.org/sitemap.xml"}}
    get = _getter(_resp(200, {}, _SITEMAP))
    status, cur = wd.check_signal(e, {"lastmod": "2026-07-01"}, get)
    assert status == "changed"
    assert cur == {"lastmod": "2026-07-10"}


def test_sitemap_lastmod_same_is_unchanged():
    e = {"url": "https://x.org/p",
         "change_signal": {"type": "sitemap_lastmod", "url": "https://x.org/sitemap.xml"}}
    get = _getter(_resp(200, {}, _SITEMAP))
    status, _ = wd.check_signal(e, {"lastmod": "2026-07-10"}, get)
    assert status == "unchanged"


def test_sitemap_url_not_in_sitemap_is_unknown():
    e = {"url": "https://x.org/missing",
         "change_signal": {"type": "sitemap_lastmod", "url": "https://x.org/sitemap.xml"}}
    get = _getter(_resp(200, {}, _SITEMAP))
    status, _ = wd.check_signal(e, {"lastmod": "2026-07-01"}, get)
    assert status == "unknown"


# -------------------------------- none / absent --------------------------- #

def test_no_signal_is_none_and_makes_no_http_call():
    e = {"url": "https://waf.example/x", "blocked": True}  # no change_signal
    get = _getter(_resp(200, {"ETag": "should-not-be-read"}))
    status, cur = wd.check_signal(e, {}, get)
    assert status == "none"
    assert cur == {}
    assert get.calls == []  # signal-less entries are not probed


def test_explicit_none_type_is_none():
    e = {"url": "https://waf.example/x", "change_signal": {"type": "none"}}
    status, _ = wd.check_signal(e, {}, _getter(_resp(0)))
    assert status == "none"


# ------------------------------ config_guard ------------------------------ #

_PINNED = {
    "embed_model": "Snowflake/snowflake-arctic-embed-l-v2.0", "mrl_dim": 128,
    "dense_size": 128, "dense_distance": "Cosine", "dense_on_disk": True,
    "sparse_bm25_modifier": "idf", "sparse_bm25_on_disk": True, "quant_type": "int8",
    "chunker_mode": "legacy", "chunk_epoch": 1,
    "chunk_rule_legend": "legacy-180w-o30@1",
}


def test_config_guard_passes_on_exact_match():
    ok, diffs = wd.config_guard(dict(_PINNED), _PINNED)
    assert ok and diffs == []


def test_weekly_delta_fails_closed_after_structured_policy_flip():
    import pytest
    policy = load_chunk_policy(DEFAULT_POLICY_PATH)
    with pytest.raises(SystemExit, match="chunk@1 only"):
        wd.require_supported_chunk_epoch(
            dict(_PINNED, chunker_mode="structured",
                 chunk_epoch=policy.epoch,
                 chunk_rule_legend=policy.rule_legend,
                 chunk_policy_path=str(policy.path),
                 chunk_policy_sha256=policy.sha256))


def test_legacy_url_writer_is_blocked_before_any_real_write():
    import pytest
    from cytir.news import freshness_ingest as fi
    with pytest.raises(SystemExit, match="retired url-addressed freshness writer"):
        fi.require_principal_source_replay(dry_run=False)
    # Signal and consent diagnostics remain explicitly read-only.
    fi.require_principal_source_replay(dry_run=True)


def test_config_guard_refuses_on_dim_change():
    live = dict(_PINNED, dense_size=256)  # a dim change = fresh build, NOT a delta
    ok, diffs = wd.config_guard(live, _PINNED)
    assert not ok
    assert any("dense_size" in d for d in diffs)


def test_config_guard_refuses_on_model_or_quant_change():
    ok, diffs = wd.config_guard(dict(_PINNED, quant_type="float16"), _PINNED)
    assert not ok and any("quant_type" in d for d in diffs)
    ok2, diffs2 = wd.config_guard(dict(_PINNED, embed_model="bge-m3"), _PINNED)
    assert not ok2 and any("embed_model" in d for d in diffs2)


def test_normalize_live_config_flattens_qdrant_shape():
    coll = {"config": {"params": {
        "vectors": {"dense": {"size": 128, "distance": "Cosine", "on_disk": True}},
        "sparse_vectors": {"bm25": {"modifier": "idf", "index": {"on_disk": True}}},
    }, "quantization_config": {"scalar": {"type": "int8", "always_ram": False}}}}
    live = wd.normalize_live_config(coll, "Snowflake/snowflake-arctic-embed-l-v2.0", 128)
    ok, diffs = wd.config_guard(live, _PINNED)
    assert ok, diffs


# --------------------------------- classify -------------------------------- #

def test_classify_advances_cursors_and_assigns_actions():
    sources = [
        {"url": "https://a/etag-unchanged", "source_type": "official-docs",
         "change_signal": {"type": "http_etag"}},
        {"url": "https://a/etag-changed", "source_type": "official-docs",
         "change_signal": {"type": "http_etag"}},
        {"url": "https://a/waf", "source_type": "gov-stats"},  # none
    ]
    state = {"https://a/etag-unchanged": {"etag": "same"},
             "https://a/etag-changed": {"etag": "old"}}

    def get(url, headers=None, timeout=20):
        if url.endswith("unchanged"):
            return _resp(304)
        return _resp(200, {"ETag": "new"})

    rows = wd.classify(sources, state, get=get, now_iso="2026-07-11T00:00:00Z")
    statuses = [r["status"] for r in rows]
    assert statuses == ["unchanged", "changed", "none"]
    # cursor advanced only for the changed one; last_checked stamped on all
    assert state["https://a/etag-changed"]["etag"] == "new"
    assert all("last_checked" in state[s["url"]] for s in sources[:2])

    embed = [r["entry"] for r in rows if r["status"] in wd.EMBED_STATUSES]
    touch = [r["entry"] for r in rows if r["status"] in wd.TOUCH_STATUSES]
    assert len(embed) == 2  # changed + none
    assert len(touch) == 1  # unchanged


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
