"""task 661 F5 follow-up (2026-09-28): the refresh-cadence lane's first-
refresh / weekly-cadence composer for the newly-admitted GitHub-releases /
HF-org-models / vendor-blog packs (#383/#387).

Real-record tests for the pure/mockable glue (row-shape, dispatch,
subprocess sequencing); the subprocess targets themselves
(consent_gate.py, stage_pack_b1.py, materialize_news_pack.py) are each
already tested by their own modules and are not re-tested here.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess

os.environ.setdefault("AW_LANGGATE_OPTIONAL", "1")

import sys

import pytest

from cytir.core import paths as PATHS
from cytir.sources.refresh_direct import direct_compose as RC


@pytest.fixture(autouse=True)
def _neutral_box(tmp_path, monkeypatch):
    """A stranger's box: the lane's work root, release and lang-gate interpreter are named settings,
    configured here to scratch values (the suite carries no operator overlay)."""
    monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("AW_NEWS_RELEASE_DIR", str(tmp_path / "release"))
    monkeypatch.setenv("AW_NEWS_LANGGATE_PY", sys.executable)
    PATHS.reset_settings_cache()
    yield
    PATHS.reset_settings_cache()


REAL_ARTICLE_HTML = """
<html><head><title>Example Release Notes — Example Project</title></head>
<body>
<main>
  <h1>Example Release Notes</h1>
  <p>This release adds a reasonably long paragraph of real prose describing a
     new feature, so that the extractor's minimum length and quality gates
     are cleared the same way a genuine GitHub release note, Hugging Face
     model card, or vendor blog post would clear them in production.</p>
  <p>A second paragraph carries further genuine content so the chunker has
     enough real substance to keep and split into at least one passage for
     the index, well past the floor the pipeline applies downstream.</p>
</main>
</body></html>
"""


def test_pack_meta_covers_every_ratified_pack():
    # the HN-linked-pages route is registered but switched off (AW_HN_OUTLINKS), so the
    # unattended default is exactly the three ratified packs
    assert set(RC.default_pack_ids({})) == {
        "us-github-releases-tracked",
        "us-huggingface-orgs-tracked",
        "us-vendor-ai-blogs-tracked",
    }
    # task 972: the two docs delta packs are registered and switched off (AW_DOCS_DELTA) too
    assert set(RC.PACK_META) - set(RC.default_pack_ids({})) == {
        "hn-linked-pages", "us-opencode-ai-docs", "us-github-repo-docs-tracked"}
    for pack_id, meta in RC.PACK_META.items():
        assert meta["tier"] in ("A", "B", "C", "D")
        assert 0.0 <= meta["trust_tier"] <= 1.0
        assert meta["label"], pack_id


def test_docs_delta_per_pack_switches():
    # task 2622: opencode armed alone -> 4 packs; repo-docs stays off and is refused by name
    op, gh = "us-opencode-ai-docs", "us-github-repo-docs-tracked"
    ids = RC.default_pack_ids({"AW_DOCS_DELTA_OPENCODE": "1"})
    assert len(ids) == 4 and op in ids and gh not in ids
    assert RC.pack_armed(op, {"AW_DOCS_DELTA_OPENCODE": "1"})
    assert not RC.pack_armed(gh, {"AW_DOCS_DELTA_OPENCODE": "1"})
    assert RC.pack_armed(gh, {"AW_DOCS_DELTA_GH_REPO_DOCS": "1"})
    assert not RC.pack_armed(op, {"AW_DOCS_DELTA_GH_REPO_DOCS": "1"})
    both = RC.default_pack_ids({"AW_DOCS_DELTA": "1"})
    assert op in both and gh in both and "hn-linked-pages" not in both
    assert not RC.pack_armed("hn-linked-pages", {"AW_DOCS_DELTA": "1"})
    assert not RC.pack_armed(op, {"AW_DOCS_DELTA_OPENCODE": "0"})


def test_cadence_row_finds_the_real_committed_row():
    from cytir.core.paths import data_path
    cadence_path = data_path("refresh-cadence.v1.json")
    if not cadence_path.is_file():
        pytest.skip("data/refresh-cadence.v1.json is withheld from the public cut")
    row = RC.cadence_row("us-github-releases-tracked", cadence_path)
    assert row["id"] == "us-github-releases-tracked"
    assert row["adapter"] == "api"


def test_cadence_row_refuses_for_an_unknown_pack(tmp_path):
    cadence_path = tmp_path / "cadence.json"
    cadence_path.write_text(json.dumps({"sources": []}))
    try:
        RC.cadence_row("does-not-exist", cadence_path)
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert "does-not-exist" in str(exc)


def test_discover_candidates_dispatches_through_the_same_registry_refresh_lane_new_refs_uses(
        monkeypatch):
    """Never a second discovery path -- this must reuse
    `new_refs.ADAPTER_REGISTRY`, the exact object
    `new_refs.run()` looks up (family key first, then pack
    id -- identical order, task 661 F5)."""
    calls = []

    def fake_adapter(row):
        calls.append(row["id"])
        return {"candidates": [{"url": "https://example.com/a", "title": "A"}]}

    from cytir.sources.refresh_direct import new_refs as RLN
    monkeypatch.setitem(RLN.ADAPTER_REGISTRY, "us-github-releases-tracked",
                        fake_adapter)
    try:
        row = {"id": "us-github-releases-tracked", "adapter": "api"}
        discovery = RC.discover_candidates("us-github-releases-tracked", row)
    finally:
        RLN.ADAPTER_REGISTRY.pop("us-github-releases-tracked", None)
        # restore the real one, since it may have been popped by an
        # earlier failed run of this same test
        from cytir.sources.refresh_direct import github_releases as _gh
        RLN.ADAPTER_REGISTRY.setdefault(
            "us-github-releases-tracked", _gh.discover)

    assert calls == ["us-github-releases-tracked"]
    assert discovery["candidates"] == [{"url": "https://example.com/a", "title": "A"}]


def test_discover_candidates_refuses_when_no_adapter_is_registered():
    row = {"id": "nonexistent-pack", "adapter": "nonexistent-family"}
    try:
        RC.discover_candidates("nonexistent-pack", row)
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert "no adapter registered" in str(exc)


def test_build_rows_for_candidate_extracts_real_prose_into_the_b1_native_shape():
    meta = {"tier": "B", "source_type": "reference", "trust_tier": 0.85,
           "published": "2026-09-20"}
    rows = RC.build_rows_for_candidate(
        "https://github.com/qdrant/qdrant/releases/tag/v1.99.0",
        REAL_ARTICLE_HTML, meta=meta)
    assert rows, "expected at least one extracted passage from real prose"
    for row in rows:
        assert row["url"].startswith("https://github.com/")
        assert row["text"], "text must survive into the b1-native row"
        assert row["domain"] == "github.com"
        assert row["tier"] == "B"
        assert row["source_type"] == "reference"
        assert row["trust_tier"] == 0.85
        # published falls back to the candidate's own date when the page
        # itself carries none
        assert row.get("published") in (None, "2026-09-20")


def test_build_rows_for_candidate_returns_empty_for_too_short_html():
    meta = {"tier": "B", "source_type": "reference", "trust_tier": 0.85}
    rows = RC.build_rows_for_candidate(
        "https://example.com/x", "<html></html>", meta=meta)
    assert rows == []


def test_run_stage_writes_errors_for_a_failed_candidate_without_dropping_others(
        tmp_path, monkeypatch):
    """One candidate's fetch failing must not take the whole pack's stage
    step down -- named in `errors`, the rest still processed."""
    monkeypatch.setattr(RC, "W", tmp_path)

    def fake_discover(pack_id, row):
        return {"candidates": [
            {"url": "https://github.com/good/repo/releases/tag/v1", "published": None},
            {"url": "https://github.com/bad/repo/releases/tag/v1", "published": None}]}

    def fake_fetch(url):
        if "bad" in url:
            raise RuntimeError("HTTP 500")
        return REAL_ARTICLE_HTML

    def fake_cadence_row(pack_id, cadence_path):
        return {"id": pack_id, "adapter": "api"}

    monkeypatch.setattr(RC, "discover_candidates", fake_discover)
    monkeypatch.setattr(RC, "cadence_row", fake_cadence_row)

    gate_calls = []

    def fake_run(cmd, **kwargs):
        gate_calls.append(cmd)
        # simulate cytir.consent.gate writing passages_ok.jsonl with everything
        if "gate" in cmd[:4]:
            out_dir = tmp_path / f"gated-us-github-releases-tracked"
            out_dir.mkdir(parents=True, exist_ok=True)
            passages_in = tmp_path / "passages-us-github-releases-tracked.jsonl"
            (out_dir / "passages_ok.jsonl").write_text(passages_in.read_text())
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)

    class FakeCollInfo:
        points_count = 1

    class FakeClient:
        def get_collection(self, name):
            return FakeCollInfo()

    import cytir.core.qdrant as AQ
    monkeypatch.setattr(AQ, "qdrant_client", lambda *a, **k: FakeClient())

    out_log = tmp_path / "stage.out"
    result = RC.run_stage("us-github-releases-tracked",
                          cadence_path=tmp_path / "cadence.json",
                          out_log=out_log, fetch=fake_fetch)
    assert result["candidates"] == 2
    assert "https://github.com/bad/repo/releases/tag/v1" in result["errors"]
    assert "500" in result["errors"]["https://github.com/bad/repo/releases/tag/v1"]
    assert result["staged"] == 1


def test_run_stage_never_stages_a_previous_runs_gated_output(tmp_path, monkeypatch):
    """Live 2026-10-03 (gh-releases): every candidate older than the window,
    so 0 rows were extracted, but `gated-<pack>/passages_ok.jsonl` of the
    2026-09-28 run was still on disk and was staged again (54 rows, same
    bytes as the 10-02 pack). A quiet run must stage 0."""
    monkeypatch.setattr(RC, "W", tmp_path)
    stale = tmp_path / "gated-us-github-releases-tracked"
    stale.mkdir()
    (stale / "passages_ok.jsonl").write_text('{"url": "https://github.com/old/x"}\n' * 54)
    monkeypatch.setattr(RC, "discover_candidates", lambda pack_id, row: {"candidates": [
        {"url": "https://github.com/old/repo/releases/tag/v1",
         "published": "2025-07-07T12:33:34+00:00"}]})
    monkeypatch.setattr(RC, "cadence_row", lambda pack_id, cadence_path: {"id": pack_id})
    calls = []

    def fake_run(cmd, **kwargs):          # the gate with no input writes nothing
        calls.append(cmd)

        class R:
            returncode = 0
        return R()
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = RC.run_stage("us-github-releases-tracked",
                          cadence_path=tmp_path / "cadence.json",
                          out_log=tmp_path / "stage.out", since_days=3)
    assert result["candidates"] == 0
    assert result["staged"] == 0
    assert not (stale / "passages_ok.jsonl").exists()
    assert all("stage_pack_b1.py" not in " ".join(map(str, c)) for c in calls)


def test_run_one_reports_a_quiet_run_without_raising_when_nothing_stages(
        tmp_path, monkeypatch):
    monkeypatch.setattr(RC, "W", tmp_path)

    def fake_run_stage(pack_id, **kwargs):
        return {"staged": 0, "collection": "irrelevant",
               "candidates": 3, "candidates_discovered": 3, "errors": {},
               "discovery_errors": {}}

    monkeypatch.setattr(RC, "run_stage", fake_run_stage)

    class Args:
        cadence_table = str(tmp_path / "cadence.json")
        collection = "aw_e8"
        twin_url = "http://127.0.0.1:6457"
        staging_url = "http://localhost:6398"
        mrl_dim = 256
        since_days = 14
        no_tombstones = True
        egress_iface = "lo"

    result = RC.run_one("us-github-releases-tracked", Args())
    assert result == {"pack_id": "us-github-releases-tracked", "staged": 0,
                      "packed": 0, "queued": False, "candidates": 3,
                      "candidates_discovered": 3, "fetch_errors": 0,
                      "derivatives_skipped": 0, "discovery_errors": {}}


def test_parse_published_handles_all_three_adapters_date_shapes():
    # github_releases_adapter: Atom <updated>, ISO 8601
    d1 = RC.parse_published("2026-09-25T19:00:00Z")
    assert d1.year == 2026 and d1.month == 9 and d1.day == 25
    # hf_org_models_adapter: HF Hub API lastModified/createdAt, ISO 8601+ms
    d2 = RC.parse_published("2026-07-13T04:10:11.000Z")
    assert d2.year == 2026 and d2.month == 7 and d2.day == 13
    # vendor_blog_rss_adapter: RSS 2.0 <pubDate>, RFC 822
    d3 = RC.parse_published("Fri, 25 Sep 2026 19:00:00 GMT")
    assert d3.year == 2026 and d3.month == 9 and d3.day == 25


def test_parse_published_returns_none_for_unparseable_or_missing():
    assert RC.parse_published(None) is None
    assert RC.parse_published("") is None
    assert RC.parse_published("not a date") is None


def test_filter_recent_candidates_drops_only_confidently_old_items():
    """Root-caused live, task 661 (2026-09-28): a first run against
    openai.com's RSS feed (1230 historical items) tried to fetch the
    ENTIRE archive, not just what changed since the last run -- 1230 of
    1250 total candidates errored (mostly ancient posts), and the ones
    that succeeded took 30-60s each once the Cloudflare-challenge
    fallback kicked in. This is the fix: cap to a recency window."""
    now = _dt.datetime(2026, 9, 28, tzinfo=_dt.timezone.utc)
    candidates = [
        {"url": "https://x/recent", "published": "2026-09-25T19:00:00Z"},
        {"url": "https://x/old", "published": "2020-01-01T00:00:00Z"},
        {"url": "https://x/undated", "published": None},
    ]
    result = RC.filter_recent_candidates(candidates, since_days=14, now=now)
    kept_urls = {c["url"] for c in result["kept"]}
    assert kept_urls == {"https://x/recent", "https://x/undated"}
    assert result["dropped_old"] == 1


def test_plain_get_returns_text_on_success():
    import httpx

    class FakeHttpx:
        @staticmethod
        def get(url, **kwargs):
            req = httpx.Request("GET", url)
            return httpx.Response(200, request=req, text="<html>ok</html>")

    html = RC.plain_get("https://example.com/a", ua="test-ua", httpx_module=FakeHttpx)
    assert html == "<html>ok</html>"


def test_plain_get_raises_challenged_error_on_403_never_retries(monkeypatch):
    """Coordinator ruling, task 661 (2026-09-28), reversing a same-day
    headless-browser-render fallback: a 403 challenge is the site's own
    access control (consent-probe-vantage) -- record as unreadable,
    never circumvent. ONE plain GET, no subprocess, no retry."""
    import httpx

    class FakeHttpx:
        @staticmethod
        def get(url, **kwargs):
            req = httpx.Request("GET", url)
            return httpx.Response(403, request=req, text="blocked")

    def fake_run(cmd, **kwargs):
        raise AssertionError("must never shell out to render/bypass a challenge")

    monkeypatch.setattr(subprocess, "run", fake_run)
    try:
        RC.plain_get("https://openai.com/index/x", ua="test-ua", httpx_module=FakeHttpx)
        assert False, "expected ChallengedError"
    except RC.ChallengedError:
        pass


def test_plain_get_raises_a_plain_error_for_a_non_403_failure():
    import httpx

    class FakeHttpx:
        @staticmethod
        def get(url, **kwargs):
            req = httpx.Request("GET", url)
            return httpx.Response(404, request=req, text="not found")

    try:
        RC.plain_get("https://example.com/gone", ua="test-ua", httpx_module=FakeHttpx)
        assert False, "expected an HTTPStatusError"
    except httpx.HTTPStatusError as exc:
        assert "404" in str(exc)


def test_build_feed_item_row_uses_only_the_feeds_own_metadata():
    meta = {"tier": "B", "source_type": "reference", "trust_tier": 0.8}
    cand = {"url": "https://openai.com/index/x", "title": "Some Post",
           "description": "A one-sentence summary from the feed itself.",
           "published": "Wed, 23 Sep 2026 12:00:00 GMT"}
    row = RC.build_feed_item_row(cand, meta=meta)
    assert row["url"] == "https://openai.com/index/x"
    assert row["title"] == "Some Post"
    assert "one-sentence summary" in row["text"]
    assert row["domain"] == "openai.com"
    assert row["lang"], "lang must be set -- stage_pack_b1.py refuses a " \
        "pack whose lang coverage falls below 99% (found live, task 661)"
    assert row["source_extract"] == "feed_item"
    assert row["consent_state"] == "unreadable (challenge)"
    assert row["published"] == "Wed, 23 Sep 2026 12:00:00 GMT"
    # never the page body -- there is none here, by construction
    assert "cf-mitigated" not in row["text"]


def test_build_feed_item_row_returns_none_with_no_title_or_description():
    meta = {"tier": "B", "source_type": "reference", "trust_tier": 0.8}
    cand = {"url": "https://openai.com/index/x", "title": "", "description": ""}
    assert RC.build_feed_item_row(cand, meta=meta) is None


def test_run_stage_falls_back_to_feed_item_for_a_challenged_candidate(
        tmp_path, monkeypatch):
    """End-to-end: a candidate that 403s gets indexed from its own feed
    metadata, not dropped and not fetched via any other path."""
    monkeypatch.setattr(RC, "W", tmp_path)

    def fake_discover(pack_id, row):
        return {"candidates": [{"url": "https://openai.com/index/challenged",
                 "title": "Challenged Post",
                 "description": "Feed-published summary sentence.",
                 "published": None}]}

    def fake_fetch(url):
        raise RC.ChallengedError(f"403 for {url}")

    def fake_cadence_row(pack_id, cadence_path):
        return {"id": pack_id, "adapter": "api"}

    monkeypatch.setattr(RC, "discover_candidates", fake_discover)
    monkeypatch.setattr(RC, "cadence_row", fake_cadence_row)

    def fake_run(cmd, **kwargs):
        if "gate" in cmd[:4]:
            out_dir = tmp_path / "gated-us-vendor-ai-blogs-tracked"
            out_dir.mkdir(parents=True, exist_ok=True)
            passages_in = tmp_path / "passages-us-vendor-ai-blogs-tracked.jsonl"
            (out_dir / "passages_ok.jsonl").write_text(passages_in.read_text())
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)

    class FakeCollInfo:
        points_count = 1

    class FakeClient:
        def get_collection(self, name):
            return FakeCollInfo()

    import cytir.core.qdrant as AQ
    monkeypatch.setattr(AQ, "qdrant_client", lambda *a, **k: FakeClient())

    out_log = tmp_path / "stage.out"
    result = RC.run_stage("us-vendor-ai-blogs-tracked",
                          cadence_path=tmp_path / "cadence.json",
                          out_log=out_log, fetch=fake_fetch)
    assert result["errors"] == {}
    assert result["challenged"] == 1
    passages = (tmp_path / "passages-us-vendor-ai-blogs-tracked.jsonl").read_text()
    row = json.loads(passages.strip())
    assert row["source_extract"] == "feed_item"
    assert row["consent_state"] == "unreadable (challenge)"
    assert row["url"] == "https://openai.com/index/challenged"


def test_main_refuses_when_langgate_optional_is_set(monkeypatch):
    """Same defended-against class as hn_discussion_stage.sh/
    news_refresh_stage.sh -- this module runs fastText language detection
    in-process (via aw_common.process_html) and must never silently ship
    degraded (URL-heuristic-only) language detection into a real index."""
    monkeypatch.setenv("AW_LANGGATE_OPTIONAL", "1")
    try:
        RC.main(["--pack-id", "us-github-releases-tracked"])
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert "AW_LANGGATE_OPTIONAL=1 is forbidden" in str(exc)


def test_discovery_refusal_is_an_error_not_an_empty():
    """Task 693: every org erroring and 0 candidates is a dead channel
    (refused); the same 0 with no errors is a genuine quiet day; a partial
    failure with survivors is not refused (main() exits non-zero instead)."""
    dead = {"orgs_checked": 3, "orgs_errored": 3, "candidates": [],
            "errors": {"zai-org": "HTTPStatusError: 429"}}
    assert "an error is not an empty" in RC.discovery_refusal("p", dead)
    quiet = {"orgs_checked": 3, "orgs_errored": 0, "candidates": [], "errors": {}}
    assert RC.discovery_refusal("p", quiet) is None
    partial = {"orgs_checked": 3, "orgs_errored": 1, "errors": {"x": "e"},
               "candidates": [{"url": "https://huggingface.co/a/b"}]}
    assert RC.discovery_refusal("p", partial) is None
    github_like = {"candidates": []}  # adapters without per-item errors
    assert RC.discovery_refusal("p", github_like) is None


def test_run_stage_refuses_when_discovery_is_dead(tmp_path, monkeypatch):
    monkeypatch.setattr(RC, "W", tmp_path)
    monkeypatch.setattr(RC, "cadence_row", lambda pid, cp: {"id": pid, "adapter": "api"})
    monkeypatch.setattr(RC, "discover_candidates", lambda pid, row: {
        "orgs_checked": 2, "orgs_errored": 2, "candidates": [],
        "errors": {"a": "ConnectError", "b": "ConnectError"}})
    try:
        RC.run_stage("us-huggingface-orgs-tracked", cadence_path=tmp_path / "c.json",
                     out_log=tmp_path / "o.out", fetch=lambda u: "")
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert "REFUSED" in str(exc.code)


def test_run_stage_refuses_when_every_fetch_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(RC, "W", tmp_path)
    monkeypatch.setattr(RC, "FETCH_INTERVAL_S", 0)
    monkeypatch.setattr(RC, "cadence_row", lambda pid, cp: {"id": pid, "adapter": "api"})
    monkeypatch.setattr(RC, "discover_candidates", lambda pid, row: {
        "candidates": [{"url": "https://huggingface.co/a/b", "published": None}]})

    def boom(url):
        raise RuntimeError("HTTP 429")

    try:
        RC.run_stage("us-huggingface-orgs-tracked", cadence_path=tmp_path / "c.json",
                     out_log=tmp_path / "o.out", fetch=boom)
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert "dead channel" in str(exc.code)


def test_write_state_advances_last_success_only_on_a_clean_run(tmp_path):
    import datetime as dt
    sp, np_ = tmp_path / "state.json", tmp_path / "NEXT.md"
    t1 = dt.datetime(2026, 10, 2, 11, 30, tzinfo=dt.timezone.utc)
    t2 = dt.datetime(2026, 10, 3, 11, 30, tzinfo=dt.timezone.utc)
    ok = {"pack_id": "p", "queued": True, "queue_pack_id": "refresh-direct-x-1",
          "packed": 5, "candidates": 5, "candidates_discovered": 6,
          "fetch_errors": 0, "discovery_errors": {}}
    RC.write_state([ok], state_path=sp, next_path=np_, now=t1)
    bad = {"pack_id": "p", "failed": "REFUSED: discovery FAILED"}
    state = RC.write_state([bad], state_path=sp, next_path=np_, now=t2)
    e = state["packs"]["p"]
    assert e["last_run"] == "2026-10-03T11:30:00Z"
    assert e["last_success"] == "2026-10-02T11:30:00Z"
    assert e["last_queued"]["pack"] == "refresh-direct-x-1"
    assert "REFUSED" in np_.read_text()


def test_plain_get_backs_off_on_429_then_succeeds():
    calls, slept = [], []

    class Resp:
        def __init__(self, code, headers=None, text=""):
            self.status_code, self.headers, self.text = code, headers or {}, text

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    seq = [Resp(429, {"retry-after": "7"}), Resp(429), Resp(200, text="ok")]

    class FakeHttpx:
        @staticmethod
        def get(url, **kw):
            calls.append(url)
            return seq[len(calls) - 1]

    assert RC.plain_get("https://huggingface.co/a/b", ua="u",
                        httpx_module=FakeHttpx, sleep=slept.append) == "ok"
    assert len(calls) == 3 and slept == [7.0, 15.0]


def test_plain_get_gives_up_after_the_last_429():
    class Resp:
        status_code, headers, text = 429, {}, ""

        def raise_for_status(self):
            raise RuntimeError("HTTP 429")

    class FakeHttpx:
        @staticmethod
        def get(url, **kw):
            return Resp()

    try:
        RC.plain_get("https://huggingface.co/a/b", ua="u",
                     httpx_module=FakeHttpx, sleep=lambda s: None)
        assert False, "expected an error"
    except RuntimeError as exc:
        assert "429" in str(exc)


def test_partial_fetch_failure_marks_the_run_not_clean(tmp_path):
    import datetime as dt
    assert RC.partial_fetch_failure({"candidates": 244, "fetch_errors": 144})
    assert not RC.partial_fetch_failure({"candidates": 244, "fetch_errors": 2})
    assert not RC.partial_fetch_failure({"candidates": 5, "fetch_errors": 2})
    state = RC.write_state(
        [{"pack_id": "p", "queued": True, "packed": 1, "candidates": 244,
          "fetch_errors": 144, "discovery_errors": {}}],
        state_path=tmp_path / "s.json", next_path=tmp_path / "N.md",
        now=dt.datetime(2026, 10, 2, tzinfo=dt.timezone.utc))
    assert "last_success" not in state["packs"]["p"]


def _detail(mid, **kw):
    d = {"id": mid, "tags": ["text-generation"], "pipeline_tag": "text-generation",
         "lastModified": "2026-09-04T06:41:23.000Z", "createdAt": "2026-08-25T00:00:00.000Z",
         "cardData": {"license": "mit"}}
    d.update(kw)
    return d


def test_derivative_base_is_owner_based_not_org_special_cased():
    import cytir.sources.hf_model_cards.ingest as H
    # quantization by another org: derivative (via cardData and via the Hub tag)
    assert H.derivative_base("nvidia/GLM-5.3-NVFP4", _detail(
        "x", cardData={"base_model": "zai-org/GLM-5.3"})) == "zai-org/GLM-5.3"
    assert H.derivative_base("nvidia/GLM-5.3-NVFP4", _detail(
        "x", tags=["base_model:quantized:zai-org/GLM-5.3"])) == "zai-org/GLM-5.3"
    # same-org variant, no base, and a case-only owner difference: not derivatives
    assert H.derivative_base("zai-org/GLM-5.3-BF16", _detail(
        "x", cardData={"base_model": "zai-org/GLM-5.3"})) is None
    assert H.derivative_base("Qwen/Qwen3-8B", _detail("x")) is None
    assert H.derivative_base("qwen/Qwen3-8B", _detail(
        "x", cardData={"base_model": ["Qwen/Qwen3-8B-Base"]})) is None


def test_hf_card_rows_skips_derivatives_and_builds_task120_shape():
    readme = "# Model\n\n" + ("This model is a large language model for coding and agents. " * 40)

    def make_get(detail):
        def get(url, accept_json=True, tries=6):
            if "/resolve/main/README.md" in url:
                return 200, readme, {}
            return 200, json.dumps(detail), {}
        return get

    cand = {"model_id": "zai-org/GLM-5.3", "org": "zai-org", "published": None}
    kind, rows = RC.hf_card_rows(cand, get=make_get(_detail("zai-org/GLM-5.3")))
    assert kind == "rows" and len(rows) >= 2
    assert {r["title"] for r in rows} == {"zai-org/GLM-5.3"}  # clean repo-id title
    assert all(r["url"] == "https://huggingface.co/zai-org/GLM-5.3" for r in rows)
    assert all(r["published"] == "2026-09-04T06:41:23.000Z" for r in rows)
    assert all(r["provenance"] == "ai_hf_model_cards" for r in rows)
    assert "Hugging Face –" not in json.dumps(rows)
    dcand = {"model_id": "nvidia/GLM-5.3-NVFP4", "org": "nvidia"}
    dd = _detail("nvidia/GLM-5.3-NVFP4", cardData={"base_model": "zai-org/GLM-5.3"})
    assert RC.hf_card_rows(dcand, get=make_get(dd)) == ("derivative", "zai-org/GLM-5.3")


def test_hf_card_rows_raises_on_a_failed_detail_fetch():
    try:
        RC.hf_card_rows({"model_id": "a/b", "org": "a"},
                        get=lambda url, accept_json=True, tries=6: (429, None, {}))
        assert False, "expected an error"
    except RuntimeError as exc:
        assert "429" in str(exc)


def test_hf_card_rows_on_real_saved_bytes():
    """Real Hub API + README bytes saved 2026-10-03 (fixtures/hf_card_real):
    the original card gets the repo-id title (never the page-route
    '<name> · Hugging Face – Huggingface'); nvidia's NVFP4 quantization of it
    is a cross-org derivative and is skipped."""
    import pathlib
    fx = pathlib.Path(__file__).parent / "fixtures" / "hf_card_real"

    def get(url, accept_json=True, tries=6):
        if "/resolve/main/README.md" in url:
            return 200, (fx / "GLM-5.3.README.md").read_text(), {}
        name = "GLM-5.3-NVFP4" if "NVFP4" in url else "GLM-5.3"
        return 200, (fx / f"{name}.api.json").read_text(), {}

    kind, rows = RC.hf_card_rows(
        {"model_id": "zai-org/GLM-5.3", "org": "zai-org"}, get=get)
    assert kind == "rows" and rows
    assert {r["title"] for r in rows} == {"zai-org/GLM-5.3"}
    assert not any("Hugging Face" in r["title"] or "·" in r["title"] for r in rows)
    assert {r["url"] for r in rows} == {"https://huggingface.co/zai-org/GLM-5.3"}
    assert {r["provenance"] for r in rows} == {"ai_hf_model_cards"}
    assert RC.hf_card_rows(
        {"model_id": "nvidia/GLM-5.3-NVFP4", "org": "nvidia"},
        get=get) == ("derivative", "zai-org/GLM-5.3")


def test_since_days_is_a_floor_grown_by_the_last_success():
    now = _dt.datetime(2026, 10, 4, 11, 30, tzinfo=_dt.timezone.utc)
    st = {"packs": {"p": {"last_success": "2026-10-02T11:32:00Z"}}}
    f = RC.effective_since_days
    assert f("p", 3, state=st, now=now) == 3          # 1.99 d gap -> 3 days
    st["packs"]["p"]["last_success"] = "2026-09-29T11:32:00Z"
    assert f("p", 3, state=st, now=now) == 6          # 4.99 d gap + margin
    st["packs"]["p"]["last_success"] = "2026-06-01T00:00:00Z"
    assert f("p", 3, state=st, now=now) == RC.SINCE_CAP_DAYS
    assert f("other", 3, state=st, now=now) == 3      # no state: the floor
    st["packs"]["p"]["last_success"] = "garbage"
    assert f("p", 3, state=st, now=now) == 3


def test_same_day_rerun_never_reuses_a_queue_folder(tmp_path):
    """10-02: a re-run reused an id whose folder had DONE; the drain skips
    DONE, so its rows were lost. A re-run now mints -r2, -r3 (any state)."""
    pid, day = "us-huggingface-orgs-tracked", "2026-10-03"
    base = "refresh-direct-hf-models"
    f = RC.free_pack_suffix
    assert f(pid, "", day, tmp_path) == ""
    (tmp_path / f"{base}-20261003").mkdir()
    (tmp_path / f"{base}-20261003" / "DONE").write_text("x")
    assert f(pid, "", day, tmp_path) == "-r2"
    (tmp_path / f"{base}-r2-20261003").mkdir()
    assert f(pid, "", day, tmp_path) == "-r3"
    # an explicit suffix that is taken moves on too; a free one is kept
    assert f(pid, "-r2", day, tmp_path) == "-r3"
    assert f(pid, "-r9", day, tmp_path) == "-r9"
    # another day is independent
    assert f(pid, "", "2026-10-04", tmp_path) == ""


# -- task 976, RULING 39: docs rows are cut under the pinned epoch-4 generic policy --------------
def _long_doc(words: int = 2700) -> str:
    sents = ["The configuration file lists every provider and its default model for the agent.",
             "Run the migration command once and the old settings are copied to the new location.",
             "Plugins that used the earlier hook names must be renamed before the next release.",
             "Each session now stores its history in a separate file that you can inspect by hand.",
             "If a key is missing the tool falls back to the value from the project directory."]
    out, n, i = [], 0, 0
    while n < words:
        if i % 6 == 0:
            out.append(f"## Section {i // 6}\n")
        if i % 6 == 2:
            out.append("| key | default |\n| --- | --- |\n| model | small |\n| retries | 3 |\n")
        if i % 6 == 4:
            out.append("```json\n{\"model\": \"small\", \"retries\": 3}\n```\n")
        para = " ".join(sents[(i + k) % len(sents)] for k in range(4))
        out.append(para + "\n")
        n += len(para.split())
        i += 1
    return "# Migrating from version one\n\n" + "\n".join(out)


def test_docs_rows_on_the_markdown_path_are_cut_at_generic_epoch_4(word_only_chunking_allowed):
    md = _long_doc()
    rows = RC.build_rows_for_markdown(
        "https://example.org/docs/migrate-v1", md, title="Migrate",
        meta={"tier": "A", "source_type": "docs", "trust_tier": 0.75})
    assert 0 < len(rows) <= 6   # the legacy 180w window gave ~17 rows for this size
    assert {r["stamps"]["chunk"] for r in rows} == {4}


def test_docs_rows_on_the_html_path_are_cut_at_generic_epoch_4(word_only_chunking_allowed):
    body = "".join(f"<p>{p}</p>" if not p.startswith("#") else f"<h2>{p.lstrip('# ')}</h2>"
                   for p in _long_doc().split("\n") if p.strip() and not p.startswith(("|", "`", "{", "-")))
    html = f"<html><head><title>Migrate</title></head><body><main><h1>Migrate</h1>{body}</main></body></html>"
    rows = RC.build_rows_for_candidate(
        "https://example.org/docs/migrate-v1", html,
        meta={"tier": "A", "source_type": "docs", "trust_tier": 0.75})
    assert 0 < len(rows) <= 6
    assert {r["stamps"]["chunk"] for r in rows} == {4}


def test_docs_rows_carry_rule_epochs_through_staging_and_news_rows_do_not():
    meta = {"tier": "A", "source_type": "docs", "trust_tier": 0.75}
    row = RC.stamp_docs_row({"text": "x"}, meta=meta, published=None, license_="MIT")
    assert row["rule_epochs"] == {"chunk": 4}
    news = RC.stamp_docs_row({"text": "x"}, meta={**meta, "source_type": "news"}, published=None, license_=None)
    assert "rule_epochs" not in news


def test_a_news_row_on_the_same_path_stays_at_the_legacy_epoch():
    rows = RC.build_rows_for_markdown(
        "https://example.org/news/story", _long_doc(900), title="Story",
        meta={"tier": "B", "source_type": "news", "trust_tier": 0.7})
    assert rows
    assert {r["stamps"]["chunk"] for r in rows} == {1}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_github_family_pack_meta_equals_the_lanes_and_stamp_takes_the_lane_clock():
    """Task 1169: the family adapter's `DOCS_MARKDOWN_PACK_META` is the lane's pack row minus labels and gate;
    the lane's `stamp_docs_row` supplies the clock the package's does not read."""
    import datetime as dt  # noqa: PLC0415
    from cytir.sources.families.github_agent_resources import DOCS_MARKDOWN_PACK_META as FM  # noqa: PLC0415
    lane = RC.PACK_META["us-github-repo-docs-tracked"]
    assert FM == {k: lane[k] for k in ("tier", "source_type", "trust_tier", "route")}
    row = RC.stamp_docs_row({"text": "x"}, meta=lane, published=None, license_=None)
    assert row["last_seen"] == dt.date.today().isoformat()


def test_the_source_verb_runs_the_driver_under_its_own_prog(capsys):
    """`cyt source refresh-direct-compose` is the module with its prog name (task 2116)."""
    from cytir import cli

    with pytest.raises(SystemExit) as stop:
        cli.main(["source", "refresh-direct-compose", "--help"])
    assert stop.value.code == 0
    assert "cyt source refresh-direct-compose" in capsys.readouterr().out
