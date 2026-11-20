#!/usr/bin/env python3
"""Offline unit tests for the shared discovery fetcher's intra-host heartbeat
(task 151).

The bug this pins: `process_domain` fetched a big-cap host (e.g. a 19k-page
insee) SILENTLY between the host's start and the outer loop's
`[i/N] <host>: <status>` completion line. For large caps that silence runs
past the task 100 supervisors' 20-min log-mtime stall guard, which then
kills+resumes the run in a loop. Tasks 111 and the 145 FR+CA sibling each
patched a PRIVATE copy; this test locks the fix into the SHARED pipeline.

Contract pinned here:
  * a large-host fetch emits >=1 flushed `  [hb] <host>: <p>/<budget> pages`
    line BEFORE the host finishes (so the stall guard's mtime keeps moving),
  * heartbeats land exactly every --heartbeat-pages fetched pages and are
    labelled by host (readable under --workers>1),
  * --heartbeat-pages 0 disables them (opt-out, no behaviour change),
  * the heartbeat does NOT change the fetch result (pages/rows/status) — it
    is pure stdout.

The WARC fetch and the extraction pool are mocked, so no network / no process
pool is touched.

Run: python3 -m pytest tests/test_discovery_ingest.py
 (or: python3 tests/test_discovery_ingest.py)
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
import tempfile
import time
from contextlib import redirect_stdout
from unittest import mock

import pytest

# Offline: no lid.176 model here. Same opt-in, same setdefault (never a
# force-override -- other modules' hard-fail langgate tests must still see
# their own explicit unset), as the process-html test.
os.environ.setdefault("AW_LANGGATE_OPTIONAL", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cytir.acquire import discovery_ingest as di  # noqa: E402


def _preresolved(n):
    """n unique WARC records (shape the columnar resolver hands process_domain)
    with distinct URLs so dedup_rank keeps all n."""
    return [{"url": f"https://big.example/page{i}", "offset": 0, "length": 10,
             "warc_filename": "f.warc.gz", "digest": None} for i in range(n)]


def _run_host(n_records, heartbeat_pages, max_pages=100000):
    """Run process_domain over a mocked large host, capturing stdout.

    fetch_warc is stubbed to return a byte of HTML (never touches the network);
    _extract_in_pool is stubbed to return no passages/links (never spawns the
    extraction process pool). Returns (result_tuple, [captured stdout lines])."""
    item = {"domain": "big.example", "class": "gov"}
    buf = io.StringIO()
    with mock.patch.object(di, "fetch_warc", return_value="<html>x</html>"), \
            mock.patch.object(di, "_extract_in_pool",
                              return_value=([], set(), {}, False)), \
            redirect_stdout(buf):
        res = di.process_domain(None, item, di.DEFAULT_CRAWL, max_pages,
                                preresolved=_preresolved(n_records),
                                heartbeat_pages=heartbeat_pages)
    return res, buf.getvalue().splitlines()


def _hb_lines(lines):
    return [ln for ln in lines if "[hb]" in ln]


# --- tests --------------------------------------------------------------------
def test_large_host_emits_heartbeat_before_completion():
    """>=1 heartbeat line for a host far bigger than the heartbeat interval.

    The completion `[i/N]` line is printed by the OUTER cmd_fetch loop AFTER
    process_domain returns, so any line emitted from inside process_domain (the
    only place that prints during the fetch) is by construction before it."""
    (dom, rows, links, n_pages, status, rej), lines = _run_host(
        450, heartbeat_pages=200)
    assert status == "ok" and n_pages == 450
    hbs = _hb_lines(lines)
    assert len(hbs) >= 1                       # not silent for a big host
    # exactly one per full interval: 200, 400 -> 2
    assert len(hbs) == 450 // 200 == 2


def test_heartbeat_line_shape_and_host_label():
    """Line is host-labelled `<host>: <p>/<budget> pages, <r> rows` and flushed
    (readable when --workers>1 interleaves several hosts' output)."""
    _, lines = _run_host(250, heartbeat_pages=200)
    hbs = _hb_lines(lines)
    assert hbs == ["  [hb] big.example: 200/250 pages, 0 rows"]


def test_heartbeat_disabled_is_silent():
    """--heartbeat-pages 0 -> no heartbeat lines (opt-out, unchanged behaviour)."""
    (_, _, _, n_pages, status, _), lines = _run_host(450, heartbeat_pages=0)
    assert status == "ok" and n_pages == 450   # identical fetch result
    assert _hb_lines(lines) == []


def test_small_host_under_interval_stays_silent():
    """A host with fewer than one interval of pages emits no heartbeat (the
    default cadence only fires for the big hosts that need it)."""
    _, lines = _run_host(120, heartbeat_pages=200)
    assert _hb_lines(lines) == []


def test_heartbeat_does_not_change_fetch_result():
    """Heartbeat is pure stdout: pages/rows/status identical with it on vs off."""
    on = _run_host(400, heartbeat_pages=200)[0]
    off = _run_host(400, heartbeat_pages=0)[0]
    assert on == off                           # same (dom, rows, links, n, status, rej)


def test_cli_exposes_heartbeat_pages():
    """The REAL fetch subcommand exposes --heartbeat-pages (default 200,
    overridable). Drives di.main()'s actual parser; cmd_fetch is mocked so the
    parsed args are captured without running a fetch."""
    seen = {}

    def _capture(args):
        seen["hb"] = args.heartbeat_pages

    with mock.patch.object(di, "cmd_fetch", _capture):
        with mock.patch.object(sys, "argv",
                               ["di", "fetch", "--queue", "q", "--out", "o"]):
            di.main()
        assert seen["hb"] == 200               # default
        with mock.patch.object(sys, "argv",
                               ["di", "fetch", "--queue", "q", "--out", "o",
                                "--heartbeat-pages", "0"]):
            di.main()
        assert seen["hb"] == 0                  # overridable / opt-out


# --- class_basis/n_endorser_regs must never reach Passage() (task 634) ----
# The regression this pins: task 440 stamped class_basis/n_endorser_regs
# into `common`, which process_domain spreads as **fields straight into
# aw_common.Passage(**fields) (via passage_from_chunk). Passage has no such
# constructor arguments -- EVERY real extraction since task 440 merged
# (2026-08-20) raised TypeError inside the extraction pool worker, silently
# swallowed and mislabeled "extract-timeout/skip" by _extract_in_pool's
# broad except (needed to also catch a genuine timeout). Zero passages were
# ever produced by a real discovery_ingest.py fetch run in that window; task
# 440's own test suite mocks below the Passage-construction layer so it
# never caught this. Confirmed by task 634 against real CC-fetched pages
# (nature.com's scitable article: 4,936 chars via bare trafilatura.extract,
# 0 chars via the real fetch -- TypeError, not a timeout).

def test_class_basis_never_reaches_the_extraction_meta():
    """common/passage_common (what actually reaches Passage(**fields)) must
    not carry class_basis or n_endorser_regs -- those are ROW-level stamps,
    added after Passage construction, not constructor arguments."""
    captured_meta = {}

    def _capture(url, html, meta):
        captured_meta.update(meta)
        return [], set(), {}, False

    item = {"domain": "big.example", "class": "gov",
            "class_basis": "endorsed", "n_endorser_regs": 3}
    with mock.patch.object(di, "fetch_warc", return_value="<html>x</html>"), \
            mock.patch.object(di, "_extract_in_pool", side_effect=_capture):
        di.process_domain(None, item, di.DEFAULT_CRAWL, 1,
                          preresolved=_preresolved(1), heartbeat_pages=0)

    common = captured_meta.get("passage_common", {})
    assert "class_basis" not in common, (
        "class_basis leaked into passage_common -- this crashes "
        "Passage(**fields) in real aw_common.process_html")
    assert "n_endorser_regs" not in common


def test_class_basis_and_n_endorser_regs_land_on_the_row_instead():
    """The two fields ARE preserved -- on the returned ROW, exactly like
    disc_class/provenance, never through the Passage constructor."""
    class _FakePassage:
        def payload(self):
            return {"url": "https://big.example/page0", "title": "t"}
        text = "body text"
        point_id = "p1"

    item = {"domain": "big.example", "class": "gov",
            "class_basis": "endorsed", "n_endorser_regs": 5}
    with mock.patch.object(di, "fetch_warc", return_value="<html>x</html>"), \
            mock.patch.object(di, "_extract_in_pool",
                              return_value=([_FakePassage()], set(), {}, False)):
        dom, rows, links, n_pages, status, rej = di.process_domain(
            None, item, di.DEFAULT_CRAWL, 1,
            preresolved=_preresolved(1), heartbeat_pages=0)

    assert rows[0]["class_basis"] == "endorsed"
    assert rows[0]["n_endorser_regs"] == 5
    assert rows[0]["disc_class"] == "gov"                # unchanged sibling stamp


def test_real_passage_construction_does_not_raise_with_process_domains_common():
    """The exact `common` dict process_domain builds, spread into the REAL
    aw_common.Passage constructor (not a mock), must not raise. This is the
    end-to-end guard the mocked tests above cannot provide by themselves."""
    from cytir.core.common import Passage
    common = dict(domain="big.example", tier="C", trust_tier=0.7,
                  source_type="docs", last_seen="2026-09-22")
    # Must succeed -- this is what raised TypeError before the fix.
    Passage(url="https://big.example/x", title="t", snippet="s", text="body",
            lang="en", **common)


# --- one domain's uncaught failure must not crash the whole run (task 634) --
# Observed live: a signed-S3 WarcRangeFetchError (SlowDown, retries
# exhausted) propagated out of process_domain, through the worker future,
# and crashed cmd_fetch's main loop entirely -- losing every OTHER domain's
# already-durable progress in a multi-hour run. process_domain already has
# an "error" status for a resolve failure a few lines above this loop; the
# fix extends the same status to an uncaught exception from fut.result().

def test_one_domain_uncaught_exception_does_not_abort_the_run(tmp_path):
    import argparse
    from cytir.acquire import discovery_ingest as di

    queue = [{"domain": "good.example", "class": "gov"},
             {"domain": "bad.example", "class": "gov"}]
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(queue))
    out = tmp_path / "passages.jsonl"
    links_out = tmp_path / "outlinks.jsonl"
    state_path = tmp_path / "state.json"

    args = argparse.Namespace(
        queue=str(queue_path), out=str(out), links_out=str(links_out),
        state=str(state_path), only="", limit_domains=0, rate=4.0,
        workers=2, max_per_domain=60, resolver="cdx", columnar_threads=2,
        columnar_chunk=200, crawl=di.DEFAULT_CRAWL, resume=True,
        min_free_gb=0.0, heartbeat_pages=0)

    def fake_process_domain(client, item, crawl, max_pages, preresolved=None,
                            heartbeat_pages=0):
        if item["domain"] == "bad.example":
            raise di.WarcRangeFetchError("simulated: SlowDown, retries exhausted")
        return (item["domain"], [{"url": "https://good.example/x",
                                   "title": "t", "text": "body"}],
                [{"url": "https://good.example/x", "domain": "good.example",
                  "links": []}], 1, "ok", {})

    with mock.patch.object(di, "process_domain", fake_process_domain), \
            mock.patch.object(di.httpx, "Client"), \
            redirect_stdout(io.StringIO()):
        di.cmd_fetch(args)   # must not raise

    rows = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["url"] == "https://good.example/x"
    state = json.loads(state_path.read_text())
    assert "good.example" in state["done"]
    assert "bad.example" not in state["done"]   # error -> retryable on resume


# --- startup smoke check (task 634) ---------------------------------------
# The guard the coordinator asked for: cmd_fetch refuses to START a
# (potentially hours-long) run if the real pipeline cannot turn a known-good
# page into >=1 passage. Every failure mode -- an uncaught exception deep in
# process_html/Passage, or a clean-looking but empty result -- must surface
# as SmokeCheckFailed, never propagate raw and never pass silently.

def test_smoke_check_passes_on_the_real_current_pipeline():
    di.smoke_check()   # must not raise, on the code as it ships today


def test_a_real_non_timeout_exception_is_reraised_not_masked_as_a_timeout():
    """The CI-red this task actually shipped: a genuine exception raised
    inside the extraction pool worker (LangGateInactive on that runner,
    re-raised by AsyncResult.get() with its original type -- the real
    multiprocessing contract this pins) used to be caught by the SAME
    broad except as mp.TimeoutError, printed as the identical
    "extract-timeout/skip", and swallowed into a clean-looking ('ok', 0
    rows) result. It must now propagate as the REAL exception. Mocks at
    the AsyncResult boundary (what .get() raises in the PARENT process) --
    mocking the worker function itself would not affect an already-spawned
    'spawn'-context child, which re-imports fresh and never sees an
    in-memory parent-process patch."""
    di.reset_extraction_pool()
    fake_async_result = mock.Mock()
    fake_async_result.get.side_effect = ValueError("simulated worker defect")
    di._XPOOL = di._mp.get_context("spawn").Pool(1)
    try:
        with mock.patch.object(di._XPOOL, "apply_async",
                               return_value=fake_async_result):
            with pytest.raises(ValueError, match="simulated worker defect"):
                di._extract_in_pool("https://x.example/", "<html></html>", {})
    finally:
        di.reset_extraction_pool()


def test_reset_extraction_pool_clears_the_singleton():
    """The mechanical contract smoke_check() relies on: after
    reset_extraction_pool(), the next _extract_in_pool call spawns a FRESH
    pool (picking up the current os.environ) rather than reusing one
    created earlier in this process under a different environment -- the
    exact regression reproduced live in the task 634 w7 frontier note (a pool
    spawned before AW_LANGGATE_OPTIONAL was set kept raising
    LangGateInactive in its stale workers long after the env var was
    corrected in the same process)."""
    di._XPOOL = di._mp.get_context("spawn").Pool(1)
    stale_pool = di._XPOOL
    di.reset_extraction_pool()
    assert di._XPOOL is None
    assert di._XPOOL is not stale_pool


def test_smoke_check_translates_an_uncaught_exception():
    # smoke_check() calls aw_common.process_html directly (no pool -- see its
    # docstring); patch that same name.
    with mock.patch("cytir.extract.html_extract.process_html",
                    side_effect=TypeError("simulated Passage() TypeError")):
        with pytest.raises(di.SmokeCheckFailed, match="TypeError"):
            di.smoke_check()


def test_smoke_check_fails_on_a_clean_but_empty_result():
    with mock.patch("cytir.extract.html_extract.process_html",
                    return_value=([], set(), {})):
        with pytest.raises(di.SmokeCheckFailed, match="0 passages"):
            di.smoke_check()


def test_cmd_fetch_refuses_to_start_when_the_smoke_check_fails(tmp_path):
    import argparse
    args = argparse.Namespace(queue=str(tmp_path / "q.json"))
    (tmp_path / "q.json").write_text("[]")
    with mock.patch.object(di, "smoke_check",
                           side_effect=di.SmokeCheckFailed("boom")):
        with pytest.raises(SystemExit, match="REFUSING to start"):
            di.cmd_fetch(args)


def test_cmd_fetch_skip_smoke_check_bypasses_it(tmp_path):
    import argparse
    queue_path = tmp_path / "q.json"
    queue_path.write_text("[]")
    args = argparse.Namespace(
        queue=str(queue_path), skip_smoke_check=True, rate=4.0, only="",
        limit_domains=0, state=str(tmp_path / "state.json"), resume=True,
        crawl=di.DEFAULT_CRAWL, workers=2, max_per_domain=60)
    with mock.patch.object(di, "smoke_check",
                           return_value=None) as m:
        di.cmd_fetch(args)   # empty queue -> returns immediately; must not raise
        m.assert_not_called()


# --- circuit breaker (task 634) ------------------------------------------
# The bug this pins: apps.who.int's per-chemical JECFA database pages each
# hit the full _EXTRACT_TIMEOUT (90s default); with nothing bounding
# CONSECUTIVE timeouts, one domain could burn up to max-per-domain x
# _EXTRACT_TIMEOUT (here: 60 x 90s = 90 minutes) before process_domain moved
# on, stalling an entire multi-domain run behind a single extraction-hostile
# host. _extract_in_pool now returns a 4th value (`timed_out`); process_domain
# stops issuing further pages after _DOMAIN_TIMEOUT_BREAKER CONSECUTIVE
# timeouts, and a single non-timing-out page resets the counter.

def _run_host_with_extract(n_records, extract_side_effect, max_pages=100000):
    """Like _run_host, but the caller controls _extract_in_pool's per-call
    behaviour directly (a list of return tuples or a callable)."""
    item = {"domain": "big.example", "class": "gov"}
    buf = io.StringIO()
    with mock.patch.object(di, "fetch_warc", return_value="<html>x</html>"), \
            mock.patch.object(di, "_extract_in_pool",
                              side_effect=extract_side_effect), \
            redirect_stdout(buf):
        res = di.process_domain(None, item, di.DEFAULT_CRAWL, max_pages,
                                preresolved=_preresolved(n_records),
                                heartbeat_pages=0)
    return res, buf.getvalue().splitlines()


def test_circuit_breaker_stops_after_consecutive_timeouts():
    """50 candidate pages, every single one a timeout -> stops at exactly
    _DOMAIN_TIMEOUT_BREAKER pages attempted, not all 50."""
    always_timeout = ([], set(), {}, True)
    (dom, rows, links, n_pages, status, rej), lines = _run_host_with_extract(
        50, lambda *a, **k: always_timeout)
    assert n_pages == di._DOMAIN_TIMEOUT_BREAKER
    assert status == "ok"                       # pages WERE fetched, just cut short
    assert rej["circuit_breaker_early_stop"] == 50 - di._DOMAIN_TIMEOUT_BREAKER
    assert any("consecutive extract-timeouts" in ln for ln in lines)


def test_circuit_breaker_resets_on_a_good_page():
    """Timeouts that never reach the threshold consecutively (a good page
    lands every 3rd attempt) do NOT trip the breaker -- all candidates run."""
    calls = {"n": 0}

    def side_effect(*_a, **_k):
        calls["n"] += 1
        timed_out = calls["n"] % 3 != 0          # 2 timeouts, 1 ok, repeat
        return ([], set(), {}, timed_out)

    (dom, rows, links, n_pages, status, rej), _ = _run_host_with_extract(
        30, side_effect)
    assert n_pages == 30                          # never cut short
    assert "circuit_breaker_early_stop" not in rej


def test_circuit_breaker_threshold_is_overridable():
    """A caller with a tighter budget can lower the threshold."""
    always_timeout = ([], set(), {}, True)
    with mock.patch.object(di, "_DOMAIN_TIMEOUT_BREAKER", 2):
        (dom, rows, links, n_pages, status, rej), _ = _run_host_with_extract(
            50, lambda *a, **k: always_timeout)
    assert n_pages == 2


def test_extract_in_pool_return_shape_has_four_values():
    """Pin the contract the breaker depends on: (passages, links, rejects,
    timed_out) -- a 3-tuple caller would silently stop seeing the flag."""
    import inspect
    src = inspect.getsource(di._extract_in_pool)
    assert "return passages, links, stats.get(\"extraction_rejects\", {}), False" in src
    assert "return [], set(), {}, True" in src


# --- config-B promotability fence (task 634) ------------------------------
# The bug this pins: tranche 1's 204,737-row sealed pack was embedded through
# `discovery_ingest.py embed` (config B: int8/MRL-128) and then scrolled
# straight into an admission-ledger pack -- a shape PR #279 could only enter
# with `admission: false`, because config B's vectors are not the durable
# 1024-d fp16 artifact task 413 requires. `--intent` makes every future
# caller say which one a run is; `admission-pack` refuses at the very top,
# before the consent-stamp check and before index_bakeoff.py is ever
# spawned. A REAL gated row (the shape consent_gate.py actually stamps --
# search_ok/train_ok alongside url/domain/text/title) is used throughout,
# not a hand-trimmed fixture, per the project's own "real records" doctrine.

_REAL_GATED_ROW = {
    "point_id": "1f2ddf1e-524b-5884-aaec-aeb3d951ab94",
    "url": "https://www.bger.ch/ext/eurospider/live/de/php/aza/http/index.php",
    "domain": "bger.ch",
    "title": "Bundesgericht",
    "site_name": "bger.ch",
    "text": "Das Schweizerische Bundesgericht ist die hoechste richterliche"
            " Behoerde des Bundes.",
    "snippet": "Das Schweizerische Bundesgericht ist die hoechste ...",
    "lang": "de",
    "source_type": "discovery",
    "trust_tier": 0.70,
    "search_ok": True,
    "train_ok": True,
    "consent_checked_at": "2026-09-22T18:55:00Z",
}


def _embed_args(tmp_path, intent, **overrides):
    import argparse
    gated = tmp_path / "passages_ok.jsonl"
    gated.write_text(json.dumps(_REAL_GATED_ROW) + "\n")
    ns = argparse.Namespace(
        inp=str(gated), collection="aw_task634_tranche1_ch_law_courts",
        qdrant_url="http://localhost:6457", batch=64,
        allow_ungated=False, intent=intent)
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_cmd_embed_refuses_admission_pack_intent(tmp_path):
    args = _embed_args(tmp_path, "admission-pack")
    with mock.patch.object(di.subprocess, "call") as spawned:
        with pytest.raises(SystemExit, match="NOT a promotable artifact"):
            di.cmd_embed(args)
    spawned.assert_not_called()  # refused before any subprocess is spawned


def test_cmd_embed_refusal_names_the_1024_fp16_chain(tmp_path):
    args = _embed_args(tmp_path, "admission-pack")
    with mock.patch.object(di.subprocess, "call"):
        with pytest.raises(SystemExit) as exc:
            di.cmd_embed(args)
    assert "embed_shards.py" in str(exc.value)
    assert "1024" in str(exc.value)


def test_cmd_embed_measurement_only_still_runs_config_b(tmp_path):
    """The fence is opt-in per run, not a blanket disable: a real
    measurement-only staging collection (e.g. a funnel/ranking check) still
    reaches index_bakeoff.py with the config-B flags, unchanged."""
    args = _embed_args(tmp_path, "measurement-only")
    with mock.patch.object(di.subprocess, "call", return_value=0) as spawned:
        with pytest.raises(SystemExit, match="^0$"):
            di.cmd_embed(args)
    spawned.assert_called_once()
    cmd = spawned.call_args[0][0]
    assert "--quant" in cmd and "int8" in cmd
    assert "--mrl-dim" in cmd and "128" in cmd


def test_fetch_columnar_threads_default_is_8_not_2():
    """task 634 tranche 2: resolve measured ~20 domains/min at threads=2 (the
    run's own bottleneck). Raised to 8 for the RESOLVE query only -- the
    anonymous-HTTPS-mirror throttle caution (cc_columnar._con's docstring)
    does not bind signed-S3 metadata reads."""
    old_argv = sys.argv
    sys.argv = ["discovery_ingest.py", "fetch", "--queue", "q.json",
                "--out", "out.jsonl"]
    try:
        parser_args = None
        with mock.patch.object(di, "cmd_fetch") as fake:
            di.main()
            parser_args = fake.call_args[0][0]
    finally:
        sys.argv = old_argv
    assert parser_args.columnar_threads == 8


# --- --require-local-index: a columnar resolve error must NEVER fall
# through to a live CDX call (task634, 2026-09-23) --------------------------
# The bug this pins: process_domain's own contract says preresolved=None
# means "re-queue, don't treat as empty" for a COLUMNAR error, but its body
# can't tell that apart from "--resolver cdx, do a live lookup" -- both look
# like preresolved is None. cmd_fetch's work() used to pass an errored
# domain's None straight through, silently substituting a CDX HTTP call for
# what should have been a zero-network deferral. This matters doubly once
# --require-local-index makes cc_columnar refuse a remote parquet read
# (RemoteIndexReadBlocked): if the CALLER then quietly falls back to CDX,
# the "refuse the read outright" guarantee is fiction.

def _fetch_args(tmp_path, **overrides):
    import argparse
    base = dict(
        queue=str(tmp_path / "q.json"), out=str(tmp_path / "out.jsonl"),
        links_out=str(tmp_path / "links.jsonl"), state=str(tmp_path / "state.json"),
        crawl=di.DEFAULT_CRAWL, max_per_domain=60, rate=4.0, workers=2,
        heartbeat_pages=0, resolver="columnar", columnar_threads=2,
        columnar_chunk=1, resume=True, limit_domains=0, min_free_gb=0.0,
        only="", skip_smoke_check=True, egress_mbps=0, egress_state="",
        require_local_index="")
    base.update(overrides)
    return argparse.Namespace(**base)


def test_a_columnar_resolve_error_never_falls_back_to_a_live_cdx_call(tmp_path):
    queue = [{"domain": "ok.example", "class": "gov"},
             {"domain": "errored.example", "class": "gov"}]
    (tmp_path / "q.json").write_text(json.dumps(queue))
    args = _fetch_args(tmp_path)

    def fake_resolve(crawl, chunk, con=None, per_domain=None):
        if chunk == ["ok.example"]:
            return {"ok.example": []}   # genuinely empty, real resolve success
        raise di.ccx.RemoteIndexReadBlocked("needs a not-yet-mirrored file")

    with mock.patch.object(di.ccx, "_con", return_value=mock.MagicMock()), \
         mock.patch.object(di.ccx, "resolve_domains", side_effect=fake_resolve), \
         mock.patch.object(di, "cdx_query") as fake_cdx, \
         mock.patch.object(di, "fetch_warc") as fake_warc:
        di.cmd_fetch(args)

    fake_cdx.assert_not_called()    # the whole point: zero network for the error
    fake_warc.assert_not_called()   # nothing to fetch -- resolve never produced records
    state = json.loads((tmp_path / "state.json").read_text())
    assert "ok.example" in state["done"]           # resolved fine, recorded
    assert "errored.example" not in state["done"]  # NOT marked done -> retried on --resume


def test_require_local_index_sets_the_env_vars_before_resolve_starts(tmp_path):
    (tmp_path / "q.json").write_text("[]")
    args = _fetch_args(tmp_path, require_local_index="/mirror/CC-MAIN-2026-39")
    with mock.patch.dict(os.environ, {}, clear=False):
        di.cmd_fetch(args)   # empty queue -> returns right after setting env vars
        assert os.environ.get("AW_CC_LOCAL_INDEX_DIR") == "/mirror/CC-MAIN-2026-39"
        assert os.environ.get("AW_CC_REQUIRE_LOCAL_INDEX") == "1"


def test_require_local_index_unset_clears_a_stale_env_var(tmp_path):
    """A later run in the SAME interpreter (tests, a long-lived worker) with
    --require-local-index omitted must not silently inherit an earlier run's
    flag -- the cross-run isolation bug class task634's own egress env-var
    setup was written to avoid (see cmd_fetch's AW_EGRESS_MBPS comment)."""
    (tmp_path / "q.json").write_text("[]")
    with mock.patch.dict(os.environ, {"AW_CC_REQUIRE_LOCAL_INDEX": "1"}):
        args = _fetch_args(tmp_path, require_local_index="")
        di.cmd_fetch(args)
        assert "AW_CC_REQUIRE_LOCAL_INDEX" not in os.environ


def test_cmd_embed_intent_is_required_on_the_cli():
    """No default: argparse refuses a bare `embed` invocation missing
    --intent, so a caller cannot silently fall through either branch."""
    old_argv = sys.argv
    sys.argv = ["discovery_ingest.py", "embed", "--in", "x.jsonl"]
    try:
        with pytest.raises(SystemExit):
            di.main()
    finally:
        sys.argv = old_argv


# --- shared egress byte-rate limiter (task 634, 2026-09-23) -----------------
# The bug this pins: raising --workers alone does not bound bytes/sec (only
# concurrency), so two concurrent tranches drove the build box's aggregate inbound
# to 26 MB/s against a founder-ruled <=8-10 MB/s ZyWALL cap. The fix is a
# cross-process token bucket keyed on a shared, flock-guarded state file --
# these tests exercise the bucket directly (no network, no subprocess), and
# pin the SHARED (not per-instance) property: two separate limiter objects
# pointed at the same path draw from one pool.

def test_egress_limiter_disabled_at_rate_zero_never_blocks(tmp_path):
    lim = di.SharedByteRateLimiter(str(tmp_path / "bucket.json"), 0.0)
    t0 = time.monotonic()
    lim.acquire(10_000_000)  # would take >1s at any real rate if not a no-op
    # 2s, not the original 0.05s (weekly goal, 2026-09-25): ci.yml's own
    # comments document up to 2 concurrent PR jobs sharing the build box's cores,
    # and this suite ran TWICE per job until scripts/check_roster_freshness.py
    # stopped re-running the full battery -- both raise real scheduler
    # jitter on a wall-clock bound this tight. The bug this guards against
    # (the "disabled" limiter accidentally rate-limiting for real) would
    # show MULTIPLE SECONDS for 10 MB, not tens of milliseconds, so 2s stays
    # a real guard while tolerating contention.
    assert time.monotonic() - t0 < 2.0


def test_egress_limiter_paces_a_single_caller_to_the_configured_rate():
    """The scheduling design (task 634 rewrite) never delays a lone FIRST
    request — the very first acquire() always proceeds immediately, it is
    the NEXT reservation that inherits the cost. So the invariant to pin is
    on the SECOND call: it must wait for the first one's reserved slot to
    elapse."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bucket.json")
        rate = 1_000_000.0  # 1 MB/s
        lim = di.SharedByteRateLimiter(path, rate, burst_seconds=0.0)
        t0 = time.monotonic()
        lim.acquire(1_000_000)   # first call: immediate (no prior reservation)
        first_elapsed = time.monotonic() - t0
        # Upper bounds widened (weekly goal, 2026-09-25 -- CI-runner
        # scheduler jitter under concurrent jobs, same reasoning as
        # test_egress_limiter_disabled_at_rate_zero_never_blocks above): the
        # LOWER bound (0.7s) is the actual invariant under test -- did the
        # second call really wait for the reservation -- and stays tight.
        assert first_elapsed < 1.0, f"first acquire should be immediate, got {first_elapsed:.2f}s"
        lim.acquire(1_000_000)   # second call: must wait ~1s for the first's slot
        total_elapsed = time.monotonic() - t0
        assert 0.7 < total_elapsed < 3.0, f"expected ~1s total for 2 MB at 1 MB/s, got {total_elapsed:.2f}s"


def test_egress_limiter_holds_the_aggregate_rate_under_real_thread_concurrency():
    """Pins the actual production bug (task 634, 2026-09-23): the FIRST
    version of this limiter computed each concurrent caller's wait
    independently against an observed token count, so N threads arriving
    together all saw ~the same deficit and all proceeded together -- a
    10-thread stress test at a 1 MB/s target measured ~11 MB/s actual (a
    10-worker production run at a 6 MB/s target measured ~24 MB/s, the same
    failure shape). This test drives REAL concurrent threads (not a single
    caller) and asserts the aggregate stays within a generous tolerance of
    the configured rate."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bucket.json")
        rate = 1_000_000.0  # 1 MB/s target
        lim = di.SharedByteRateLimiter(path, rate, burst_seconds=1.0)
        total = [0]
        totals_lock = threading.Lock()

        def worker():
            for _ in range(10):
                n = 200_000  # 200 KB per acquire, WARC-record-sized
                lim.acquire(n)
                with totals_lock:
                    total[0] += n

        threads = [threading.Thread(target=worker) for _ in range(10)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - t0
        effective_mbps = (total[0] / elapsed) / 1e6
        assert total[0] == 20_000_000
        # generous tolerance (the burst allowance and thread-wake jitter both
        # push it above the exact target) -- the bug produced ~11x, not ~2x
        assert effective_mbps < 2.5, (
            f"aggregate rate {effective_mbps:.2f} MB/s far exceeds the "
            f"{rate/1e6:.1f} MB/s target under concurrent load")


def test_egress_limiter_shares_the_bucket_across_instances_same_path():
    """Two SharedByteRateLimiter objects (standing in for two OS processes,
    since the real sharing mechanism is the flock'd file, not the Python
    object) pointed at the SAME path draw from ONE pool -- the second
    caller pays for what the first already spent."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bucket.json")
        rate = 1_000_000.0
        lim_a = di.SharedByteRateLimiter(path, rate, burst_seconds=1.0)
        lim_b = di.SharedByteRateLimiter(path, rate, burst_seconds=1.0)
        t0 = time.monotonic()
        lim_a.acquire(1_000_000)   # drains the whole 1s burst allowance
        lim_b.acquire(500_000)     # must now wait on lim_a's own bucket
        elapsed = time.monotonic() - t0
        assert elapsed > 0.3, (
            f"expected lim_b to wait for lim_a's draw (shared bucket), "
            f"got {elapsed:.2f}s total — looks like two independent buckets")


def test_egress_limiter_self_heals_from_a_corrupted_bucket_file(tmp_path):
    """Task 664, 2026-09-26: found live -- four CC tranche fetches sharing
    one bucket file drove a fetch's success rate to ~0%, every acquire()
    raising `json.JSONDecodeError: Extra data` on a file that read back as a
    complete valid object immediately followed by stray trailing bytes
    (e.g. `...}0}`). This file holds nothing but a disposable rate-limit
    cursor: a malformed read must never propagate as an uncaught error to
    the caller (who has no idea this file even exists), it must be treated
    exactly like a missing file -- start the bucket fresh from `now`."""
    path = tmp_path / "bucket.json"
    path.write_text('{"next_avail": 1.0, "rate_bytes_per_sec": 1000.0}0}')
    lim = di.SharedByteRateLimiter(str(path), 1_000_000.0, burst_seconds=1.0)
    lim.acquire(1_000)  # must not raise
    json.loads(path.read_text())  # and must leave the file valid afterward


def test_egress_limiter_flushes_and_fsyncs_before_releasing_the_lock(tmp_path, monkeypatch):
    """The actual root cause (task 664, 2026-09-26): `acquire()` released its
    flock BEFORE its own buffered `json.dump` write was durable on the
    shared file -- a second process could open the same path, acquire the
    now-free lock, and read/truncate/write while the first process's write
    was still sitting in its own in-memory buffer, so that write landed
    LATER (at `close()` time) and appended onto whatever the second process
    had since written -- exactly how one complete JSON object ends up
    followed by a second writer's stray trailing bytes (`...}0}`, the real
    shape found live). A real multi-process stress test cannot reliably PIN
    this: reproducing it depends on OS scheduling luck, so it can pass by
    luck on the buggy code too. This pins the fix directly and
    deterministically instead: `os.fsync` (proof the write reached the
    file, not just this process's buffer) must happen strictly BEFORE
    `fcntl.flock(..., LOCK_UN)`."""
    import fcntl
    path = tmp_path / "bucket.json"
    path.write_text('{"next_avail": 0.0, "rate_bytes_per_sec": 100.0}')
    events: list[str] = []
    real_flock = fcntl.flock

    def spy_flock(fd, op):
        if op == fcntl.LOCK_UN:
            events.append("unlock")
        return real_flock(fd, op)

    real_fsync = os.fsync

    def spy_fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    monkeypatch.setattr(fcntl, "flock", spy_flock)
    monkeypatch.setattr(os, "fsync", spy_fsync)
    lim = di.SharedByteRateLimiter(str(path), 100.0, burst_seconds=0.0)
    lim.acquire(10)
    assert events == ["fsync", "unlock"], (
        f"the write must be fsync'd before the lock is released, got {events}")


def test_egress_limiter_is_wired_into_the_s3_warc_fetch_path():
    """The record's length is known before the GET; acquire() must be called
    with that length BEFORE the S3 call, not after (metering the download
    after it already happened doesn't bound the burst)."""
    calls = []

    class FakeLimiter:
        def acquire(self, n):
            calls.append(n)

    class FakeBody:
        def read(self):
            return b"x" * 42

    class FakeS3:
        def get_object(self, **kw):
            assert calls == [42], (
                "acquire() must run BEFORE the S3 get_object call")
            return {"Body": FakeBody()}

    with mock.patch.object(di, "_egress_limiter", return_value=FakeLimiter()), \
         mock.patch.object(di, "_warc_s3_client", return_value=FakeS3()), \
         mock.patch.object(di, "ccx") as fake_ccx:
        fake_ccx.S3_BUCKET = "commoncrawl"
        di._fetch_warc_record_s3({"filename": "f.warc.gz", "offset": 0, "length": 42})
    assert calls == [42]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok   {fn.__name__}")
    print(f"all {len(fns)} discovery_ingest tests passed")


# --- a page with one malformed href still yields its passages (task 707) -----
# Real page, L0 raw-HTML archive on the build box (read 2026-10-05): the tranche-4 fetch logged
# `ValueError: Invalid IPv6 URL` for it and recorded its domain as ERRORED at every run. The bug
# was ours: `outbound_link_anchors` called urljoin on `<a href="http://Mongolian 2014 final report
# [pdf]">` with no guard (its two siblings had one). The fixture is the page's <article> (the
# public institutional page of a university alumni association: expedition names and report links,
# no personal data) in a minimal shell.
#   url      https://rcsa.org.uk/rcsa-trust/rcsa-trust-and-exploration-board/
#   captured 2026-09-26T09:21:01Z (L0 sha256 of the full 65,566-byte page:
#            f4e1a87cbc0124d89d9f9be3782ea8f62b265a21fd27c898e722dc2e6f5eba44)
#   fixture  sha256 6fbcd173d78cd8ab8730f4b62dfcd3807bf2346f8bde86096a481f5256ef637e (6,779 bytes)
_IPV6_URL = "https://rcsa.org.uk/rcsa-trust/rcsa-trust-and-exploration-board/"
_IPV6_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                             "invalid_ipv6_href", "rcsa_trust_exploration_board.html")


def test_a_real_page_with_a_malformed_href_yields_passages():
    import hashlib
    from cytir.core.common import process_html
    raw = open(_IPV6_FIXTURE, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == \
        "6fbcd173d78cd8ab8730f4b62dfcd3807bf2346f8bde86096a481f5256ef637e"
    assert b"http://Mongolian 2014 final report [pdf]" in raw     # the href that raised
    common = dict(domain="rcsa.org.uk", tier="C", trust_tier=di.class_trust_for("docs", "self-signal", 0),
                  source_type=di.CLASS_SOURCE.get("docs", "docs"), last_seen="2026-10-05")
    meta = {"content": True, "links": True, "want_stats": True, "passage_common": common,
            "lang_default": "en", "digest": "ipv6-fixture"}
    passages, links, _stats = process_html(_IPV6_URL, raw.decode("utf-8"), meta)   # was: ValueError
    assert len(passages) >= 1
    assert not any("Mongolian 2014 final report" in str(l) for l in links)         # the bad link is skipped
