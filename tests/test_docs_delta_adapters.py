"""task 972: the docs delta adapters of the refresh lane (`docs_sitemap`, `github_repo_docs`).

Fixtures are real responses read from opencode.ai on 2026-10-17: `testdata/opencode-sitemap.xml`
is 14 of the 684 real <url> blocks (bytes unchanged), `testdata/opencode-v2-docs-landing.html` is
the whole real landing page of the v2 docs. The pack YAMLs are the committed ones.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import subprocess

os.environ.setdefault("AW_LANGGATE_OPTIONAL", "1")

import pytest

from tests.helpers.repo_files import repo_file_or_skip
import cytir.sources.refresh_direct.docs_sitemap as DSA
from cytir.sources.refresh_direct import github_repo_docs as GRA
from cytir.sources.refresh_direct import direct_compose as RC
from cytir.sources.refresh_direct import new_refs as RLN
from cytir.core import ids

TESTDATA = pathlib.Path(__file__).parent / "testdata"
SITEMAP = (TESTDATA / "opencode-sitemap.xml").read_bytes()


@pytest.fixture
def withheld_data():
    repo_file_or_skip("data/refresh-cadence.v1.json")
    repo_file_or_skip("data/github-repo-docs-phase1.v1.jsonl")
LANDING = (TESTDATA / "opencode-v2-docs-landing.html").read_bytes()
DOCS_PACK = "sources/us/opencode-ai-docs.yaml"
GH_PACK = "sources/us/github-repo-docs-tracked.yaml"


class _Free:
    def wait(self):
        return None


def _fetch(robots: bytes = b""):
    def fetch(url):
        if url.endswith("/robots.txt"):
            return (200, robots) if robots else (404, b"")
        if url == "https://opencode.ai/sitemap.xml":
            return 200, SITEMAP
        if url == "https://opencode.ai/v2/docs/":
            return 200, LANDING
        return 404, b""
    return fetch


def _discover(robots=b""):
    return DSA.discover({"pack_path": DOCS_PACK}, fetch_bytes=_fetch(robots), bucket=_Free())


def _urls(result):
    return [c["url"] for c in result["candidates"]]


def test_sitemap_candidates_exclude_every_locale_mirror_and_keep_the_english_pages():
    res = _discover()
    assert res["orgs_errored"] == 0
    sitemap_urls = {c["url"] for c in res["candidates"] if c["route"] == DSA.SITEMAP_ROUTE}
    assert sitemap_urls == {
        "https://opencode.ai/", "https://opencode.ai/enterprise", "https://opencode.ai/go",
        "https://opencode.ai/docs/config", "https://opencode.ai/docs/tui",
        "https://opencode.ai/docs/cli"}
    # /go is a two-letter path that is NOT a locale; zh-cn / pt-br / zh-tw are locales a
    # "two letters" rule would have kept
    assert "https://opencode.ai/docs/zh-cn/cli" not in sitemap_urls


def test_v2_docs_pages_come_from_the_landing_page_links_one_level_deep():
    urls = _urls(_discover())
    assert "https://opencode.ai/v2/docs/migrate-v1" in urls or "https://opencode.ai/v2/docs/migrate-v1/" in urls
    v2 = [u for u in urls if "/v2/" in u]
    assert len(v2) == len({u.rstrip("/") for u in v2}), "a /x and /x/ twin was kept twice"
    assert all(u.startswith("https://opencode.ai/v2/docs/") for u in v2)
    assert not any(u.endswith((".css", ".js", ".svg")) for u in v2)


def test_robots_disallow_removes_a_url_before_it_is_fetched():
    res = _discover(robots=b"User-agent: *\nDisallow: /docs/tui\nDisallow: /v2/\n")
    urls = _urls(res)
    assert "https://opencode.ai/docs/tui" not in urls
    assert "https://opencode.ai/docs/cli" in urls
    assert not any("/v2/" in u for u in urls)


def test_a_failed_route_is_named_not_a_quiet_zero():
    def fetch(url):
        if url.endswith("/robots.txt"):
            return 404, b""
        return 500, b""
    res = DSA.discover({"pack_path": DOCS_PACK}, fetch_bytes=fetch, bucket=_Free())
    assert res["candidates"] == [] and res["orgs_errored"] >= 1 and res["errors"]
    assert RC.discovery_refusal("us-opencode-ai-docs", res) is not None


def test_cap_limits_the_candidates(monkeypatch):
    plans = DSA.load_plans(DOCS_PACK)
    import dataclasses
    plans = [dataclasses.replace(plans[0], cap=2)]
    res = DSA.discover({}, fetch_bytes=_fetch(), plans=plans, bucket=_Free())
    assert len(res["candidates"]) == 2


ROWS_TEXT = ("OpenCode reads its configuration from opencode.json in the project root. "
             "Each key is documented below with its type and default value. ") * 6


def test_unchanged_page_is_one_whose_chunk_digests_equal_the_twins():
    rows = [{"text": ROWS_TEXT}, {"text": ROWS_TEXT + " second chunk"}]
    twin = {ids.content_digest(r["text"]) for r in rows}
    assert DSA.unchanged_against_twin(rows, twin) is True
    assert DSA.unchanged_against_twin(rows, {d[:32] for d in twin}) is True   # the coded twin's 32-hex key
    assert DSA.unchanged_against_twin(rows[:1], twin) is False          # a chunk disappeared
    assert DSA.unchanged_against_twin([{"text": ROWS_TEXT + " edited"}] + rows[1:], twin) is False
    assert DSA.unchanged_against_twin(rows, {"8b5acec3b72ccea3c9830550260bb3de"}) is False  # a different digest
    assert DSA.unchanged_against_twin(rows, set()) is False             # new page
    assert DSA.unchanged_against_twin([], twin) is False


class _TwinClient:
    def __init__(self, by_url):
        self.by_url, self.asked = by_url, []

    def scroll(self, collection, scroll_filter=None, limit=0, offset=None, with_payload=None,
               with_vectors=False):
        url = scroll_filter.must[0].match.value
        self.asked.append(url)
        class P:
            def __init__(self, d):
                self.payload = {"content_digest": d}
        return [P(d) for d in self.by_url.get(url, [])], None


ARTICLE = """<html><head><title>Config - OpenCode</title></head><body><main>
<h1>Config</h1>
<p>%s</p><p>A second paragraph carries more genuine documentation prose about the configuration
file, its keys, the defaults that apply when a key is missing, and the order in which the
global and project files are merged, so the chunker keeps a real passage.</p>
</main></body></html>""" % ROWS_TEXT


def test_twin_digests_reads_by_exact_url():
    c = _TwinClient({"https://opencode.ai/docs/config": ["a", "b"]})
    assert DSA.twin_digests(c, "coll", "https://opencode.ai/docs/config") == {"a", "b"}
    assert c.asked == ["https://opencode.ai/docs/config"]


def _stage(tmp_path, monkeypatch, pack_id, cands, fetch, twin):
    monkeypatch.setattr(RC, "W", tmp_path)
    monkeypatch.setattr(RC, "FETCH_INTERVAL_S", 0)
    monkeypatch.setattr(RC, "discover_candidates", lambda p, row: {"candidates": cands})
    monkeypatch.setattr(RC, "cadence_row", lambda p, c: {"id": p, "adapter": "sitemap"})

    def fake_run(cmd, **kw):
        if cmd[2] == "gate":
            out = tmp_path / f"gated-{pack_id}"
            out.mkdir(parents=True, exist_ok=True)
            (out / "passages_ok.jsonl").write_text((tmp_path / f"passages-{pack_id}.jsonl").read_text())
        return type("R", (), {"returncode": 0})()
    monkeypatch.setattr(subprocess, "run", fake_run)
    import cytir.core.qdrant as AQ
    monkeypatch.setattr(AQ, "qdrant_client", lambda *a, **k: type("C", (), {
        "get_collection": lambda self, n: type("I", (), {"points_count": 1})()})())
    return RC.run_stage(pack_id, cadence_path=tmp_path / "c.json", out_log=tmp_path / "s.out",
                        fetch=fetch, twin_client=twin)


def test_run_stage_rewrites_only_changed_pages_and_stamps_the_docs_fields(tmp_path, monkeypatch, word_only_chunking_allowed):
    meta = RC.PACK_META["us-opencode-ai-docs"]
    old = RC.build_rows_for_candidate("https://opencode.ai/docs/config", ARTICLE, meta=meta)
    assert old, "fixture article must extract"
    twin = _TwinClient({"https://opencode.ai/docs/config":
                        [ids.content_digest(r["text"]) for r in old]})
    cands = [{"url": "https://opencode.ai/docs/config", "lastmod": None, "delay_s": 0},
             {"url": "https://opencode.ai/docs/tui", "lastmod": "2026-10-10", "delay_s": 0}]
    res = _stage(tmp_path, monkeypatch, "us-opencode-ai-docs", cands,
                 lambda u: ARTICLE.replace("Config", "Terminal UI"), twin)
    assert res["unchanged"] == 0       # the fetched body differs from the twin's for /config too
    rows = [json.loads(l) for l in (tmp_path / "passages-us-opencode-ai-docs.jsonl").read_text().splitlines()]
    by_url = {r["url"] for r in rows}
    assert "https://opencode.ai/docs/tui" in by_url
    tui = [r for r in rows if r["url"].endswith("/tui")]
    assert {r["published"] for r in tui} == {"2026-10-10"}
    assert {r["license"] for r in tui} == {"docs-index-only-unstated"}
    assert {r["last_seen"] for r in tui} == {_dt.date.today().isoformat()}
    assert {r["source_type"] for r in tui} == {"docs"}


def test_run_stage_skips_a_page_identical_to_the_twin(tmp_path, monkeypatch, word_only_chunking_allowed):
    meta = RC.PACK_META["us-opencode-ai-docs"]
    old = RC.build_rows_for_candidate("https://opencode.ai/docs/config", ARTICLE, meta=meta)
    twin = _TwinClient({"https://opencode.ai/docs/config":
                        [ids.content_digest(r["text"]) for r in old]})
    cands = [{"url": "https://opencode.ai/docs/config", "lastmod": None, "delay_s": 0}]
    res = _stage(tmp_path, monkeypatch, "us-opencode-ai-docs", cands, lambda u: ARTICLE, twin)
    assert res["unchanged"] == 1 and res["staged"] == 0


def test_docs_packs_are_off_unless_armed():
    assert "us-opencode-ai-docs" not in RC.default_pack_ids(env={})
    assert "us-github-repo-docs-tracked" not in RC.default_pack_ids(env={})
    assert "us-opencode-ai-docs" in RC.default_pack_ids(env={RC.DOCS_DELTA_SWITCH: "1"})


@pytest.mark.usefixtures("withheld_data")
def test_registry_resolves_both_adapters():
    repo_file_or_skip("data/refresh-cadence.v1.json")      # withheld from the public cut
    assert RLN.ADAPTER_REGISTRY["us-opencode-ai-docs"] is DSA.discover
    assert RLN.ADAPTER_REGISTRY["us-github-repo-docs-tracked"] is GRA.discover
    rows = {r["id"]: r for r in RLN.load_cadence_rows(RLN.REPO_ROOT / "data" / "refresh-cadence.v1.json")}
    for pid in ("us-opencode-ai-docs", "us-github-repo-docs-tracked"):
        assert rows[pid]["layer"] == "tech-docs"


# ---------------------------------------------------------------- github_repo_docs
MD = ("# Config\n\n" + ("OpenCode reads its configuration from opencode.json in the project root "
      "and merges it over the global file. Every key has a default value. " * 8) + "\n")


def _fake_acquire(calls, license_="MIT", status="complete"):
    from cytir.acquire import preserved_stage_io as P
    from cytir.sources.github_repo_docs.acquire import receipt_name

    def acquire(queue, root, max_files):
        q = json.loads(pathlib.Path(queue).read_text())
        calls.append((q, max_files))
        docs = []
        for path in ["packages/web/src/content/docs/config.mdx", "README.md"][:max_files]:
            art = P.store_artifact(pathlib.Path(root), "markdown", "md", MD.encode())
            docs.append({"source_path": path, "title": "Config",
                         "source_url": f"https://github.com/{q['owner_repo']}/blob/abc123/{path}",
                         "markdown": art})
        rec = {"status": status, "revision": "abc123", "license": license_,
               "acquired_at": "2026-10-17T03:00:00Z", "documents": docs}
        d = pathlib.Path(root) / "receipts"
        d.mkdir(parents=True, exist_ok=True)
        (d / receipt_name(q["owner_repo"])).write_text(json.dumps(rec))
    return acquire


@pytest.mark.usefixtures("withheld_data")
def test_github_pack_lists_opencode_with_a_declared_licence():
    repo_file_or_skip("data/github-repo-docs-phase1.v1.jsonl")      # withheld from the public cut
    repos = GRA.load_repos(GH_PACK)
    assert repos[0]["owner_repo"] == "anomalyco/opencode"
    assert (repos[0]["branch"], repos[0]["license"], repos[0]["max_files"]) == ("dev", "MIT", 40)
    import re
    rx = re.compile(repos[0]["exclude_regex"])
    assert rx.search("packages/web/src/content/docs/ar/cli.mdx")
    assert rx.search("packages/web/src/content/docs/zh-cn/cli.mdx")
    assert rx.search("packages/web/src/content/docs/bs/cli.mdx")   # a locale the website does not have
    assert not rx.search("packages/web/src/content/docs/cli.mdx")


def test_acquirer_drops_excluded_paths_before_it_applies_the_file_cap():
    from cytir.sources.github_repo_docs.acquire import select_documents
    d = "packages/web/src/content/docs/"
    nodes = [{"type": "blob", "sha": "x", "path": p} for p in
             (d + "acp.mdx", d + "ar/cli.mdx", d + "cli.mdx", d + "de/cli.mdx", d + "tui.mdx")]
    got = [n["path"] for n in select_documents(nodes, True, 3, r"/content/docs/[^/]+/")]
    assert got == [d + "acp.mdx", d + "cli.mdx", d + "tui.mdx"]


@pytest.mark.usefixtures("withheld_data")
def test_github_candidates_use_the_commit_blob_url_and_the_file_cap(tmp_path):
    repo_file_or_skip("data/github-repo-docs-phase1.v1.jsonl")      # withheld from the public cut
    calls = []
    repos = [{**GRA.load_repos(GH_PACK)[0], "max_files": 1}]
    res = GRA.discover({}, repos=repos, acquire=_fake_acquire(calls), root=tmp_path)
    assert res["orgs_errored"] == 0 and len(res["candidates"]) == 1
    c = res["candidates"][0]
    assert c["url"] == ("https://github.com/anomalyco/opencode/blob/abc123/"
                        "packages/web/src/content/docs/config.mdx")
    assert (c["license"], c["revision"]) == ("MIT", "abc123")
    assert calls[0][0] == {"owner_repo": "anomalyco/opencode", "branch": "dev", "license": "MIT",
                           "mode": "deep",
                           "exclude_regex": repos[0]["exclude_regex"]} and calls[0][1] == 1
    assert pathlib.Path(res["preserve_root"]).parent == tmp_path    # a per-run root


@pytest.mark.usefixtures("withheld_data")
def test_github_licence_mismatch_and_missing_licence_are_refused(tmp_path):
    repo_file_or_skip("data/github-repo-docs-phase1.v1.jsonl")      # withheld from the public cut
    repo = GRA.load_repos(GH_PACK)[0]
    res = GRA.discover({}, repos=[repo], acquire=_fake_acquire([], license_="GPL-3.0"), root=tmp_path)
    assert res["orgs_errored"] == 1 and not res["candidates"]
    with pytest.raises(ValueError):
        GRA.discover({}, repos=[{**repo, "license": ""}], acquire=_fake_acquire([]), root=tmp_path)


@pytest.mark.usefixtures("withheld_data")
def test_markdown_candidate_becomes_docs_rows_and_goes_through_the_stage(tmp_path, monkeypatch, word_only_chunking_allowed):
    repo_file_or_skip("data/github-repo-docs-phase1.v1.jsonl")      # withheld from the public cut
    repo = GRA.load_repos(GH_PACK)[0]
    res = GRA.discover({}, repos=[repo], acquire=_fake_acquire([]), root=tmp_path)
    twin = _TwinClient({})
    out = _stage(tmp_path, monkeypatch, "us-github-repo-docs-tracked", res["candidates"],
                 None, twin)
    rows = [json.loads(l) for l in
            (tmp_path / "passages-us-github-repo-docs-tracked.jsonl").read_text().splitlines()]
    assert rows and out["unchanged"] == 0
    assert all(r["url"].startswith("https://github.com/anomalyco/opencode/blob/abc123/") for r in rows)
    assert {r["license"] for r in rows} == {"MIT"}
    assert all("published" not in r for r in rows)   # the acquire date is a capture date, never `published`
    assert {r["source_type"] for r in rows} == {"docs"}


def test_docs_row_without_lastmod_has_no_published_and_survives_merge():
    """Ruling (task 973): `published` is never a capture date; a docs row with no lastmod keeps no
    `published` key and is NOT dropped by the composer's merge (the news-lane undated drop)."""
    from cytir.news import compose as NC
    meta = RC.PACK_META["us-opencode-ai-docs"]
    row = RC.stamp_docs_row({"url": "https://opencode.ai/docs/x", "text": "t"}, meta=meta,
                            published=None, license_="docs-index-only-unstated")
    assert "published" not in row
    dated = RC.stamp_docs_row({"url": "https://opencode.ai/docs/y"}, meta=meta,
                              published="2026-10-10", license_=None)
    assert dated["published"] == "2026-10-10"
    packed = [{"id": "1", "payload": row}]
    assert NC.merge_rows(packed)["undated"] == packed          # news default unchanged
    got = NC.merge_rows(packed, require_published=False)
    assert got["kept"] == packed and got["undated"] == []


def test_docs_recut_declares_exact_chunk_words_through_the_real_path(tmp_path, word_only_chunking_allowed):
    """Task 984, review 989: build_rows_for_candidate -> stage projection -> write_pack. The manifest's
    `content_words_by_url` holds the exact chunk word count of a url with a superseded tombstone; the
    pack rows do not carry `chunk_n_words` (ADR-039)."""
    import argparse
    from cytir.index import stage_pack_b1 as SPB
    from cytir.core.canon import document_url_key
    from cytir.news import compose as C
    url = "https://opencode.ai/docs/config"
    meta = RC.PACK_META["us-opencode-ai-docs"]
    rows = RC.build_rows_for_candidate(url, ARTICLE, meta=meta)
    assert rows and all(r["chunk_n_words"] == len(r["text"].split()) > 0 for r in rows)
    args = argparse.Namespace(attribution=None, license_version=None, preserve_rule_epochs=True)
    consent = {"consent_state": "resolved", "consent_basis": "robots-live-resolve",
               "consent_source": "t", "consent_checked_at": "2026-10-19"}
    kept = []
    for r in rows:
        r = RC.stamp_docs_row(dict(r), meta=meta, published="2026-10-01", license_="MIT")
        r.update(search_ok=True, train_ok=True)
        projected = SPB.project(r, args, consent)
        assert projected is not None
        pid, _dense, _sparse, payload = projected
        assert payload["chunk_n_words"] == r["chunk_n_words"]       # survives staging
        kept.append({"id": pid, "payload": payload, "vector": {}})
    tombs = [{"id": "old1", "url": url, "reason": "superseded"}]
    man = C.write_pack(str(tmp_path / "pack"), kept=kept, tombstones=tombs, pack_id="p-984",
                       family="docs", collection="c", days=["2026-10-19"])
    expected = sum(len(r["text"].split()) for r in rows)
    assert man["content_words_by_url"] == {document_url_key(url): expected}
    assert all("chunk_n_words" not in p["payload"] for p in C.read_pack_points(str(tmp_path / "pack")))


def test_markdown_rows_carry_their_exact_chunk_words(word_only_chunking_allowed):
    meta = RC.PACK_META["us-opencode-ai-docs"]
    md = "# Config\n\n" + ROWS_TEXT + "\n\nA second paragraph about the configuration file and its keys.\n"
    rows = RC.build_rows_for_markdown("https://opencode.ai/docs/config", md, title="Config", meta=meta)
    assert rows and all(r["chunk_n_words"] == len(r["text"].split()) > 0 for r in rows)


# ---------------------------------------------------------------- task 996: repo list + batching
def _list_repo(n, sha=None, in_index=False, license_="MIT"):
    return {"owner_repo": f"o{n}/r{n}", "branch": "main", "sha": sha, "max_files": None,
            "exclude_regex": None, "tree": "docs", "license": license_, "in_index": in_index,
            "trust_tier": 0.75}


def _multi_acquire(calls, revision="abc123"):
    """Sealed double of the acquirer: one receipt per queue row; records (repo, workers)."""
    from cytir.acquire import preserved_stage_io as P
    from cytir.sources.github_repo_docs.acquire import SCHEMA, receipt_name

    def acquire(queue, root, max_files, tree="capped", workers=1):
        for line in pathlib.Path(queue).read_text().splitlines():
            q = json.loads(line)
            calls.append((q["owner_repo"], max_files, tree, workers))
            art = P.store_artifact(pathlib.Path(root), "markdown", "md", MD.encode())
            st = lambda: P.store_artifact(pathlib.Path(root), "original", "txt", b"x")  # noqa: E731
            doc = {"source_path": "README.md", "title": "T", "markdown": art,
                   "original": st(), "outlinks": st(),
                   "source_url": f"https://github.com/{q['owner_repo']}/blob/{revision}/README.md"}
            rec = {"status": "complete", "revision": revision, "license": q["license"],
                   "acquired_at": "2026-10-19T03:00:00Z", "documents": [doc],
                   "schema": SCHEMA, "repository": q["owner_repo"],
                   "tree": {"commit_response": st(), "tree_response": st()},
                   "counts": {"original_documents": 1, "markdown_documents": 1,
                              "outlink_documents": 1}}
            d = pathlib.Path(root) / "receipts"
            d.mkdir(parents=True, exist_ok=True)
            (d / receipt_name(q["owner_repo"])).write_text(json.dumps(rec))
    return acquire


@pytest.mark.usefixtures("withheld_data")
def test_pack_merges_routes_with_the_repo_list_and_the_list_licence_is_per_row():
    repo_file_or_skip("data/github-repo-docs-phase1.v1.jsonl")      # withheld from the public cut
    repos = GRA.load_repos(GH_PACK)
    assert repos[0]["owner_repo"] == "anomalyco/opencode"          # routes first, then list order
    listed = [r for r in repos if r.get("in_index") is not None]
    assert len(listed) == 1746      # the opencode ROUTE wins over its list row
    assert all(r["tree"] == "docs" and r["max_files"] is None for r in listed)
    assert all(r["license"] and r["sha"] for r in listed)
    assert {r["trust_tier"] for r in listed} == {repos[0]["trust_tier"]}
    with pytest.raises(ValueError):                                  # unknown licence refused
        GRA.discover({}, repos=[_list_repo(1, license_="NOASSERTION")], acquire=_multi_acquire([]),
                     root=pathlib.Path("/nonexistent"))


def test_windows_are_stable_and_the_in_index_filter_runs_before_offset(tmp_path):
    repos = [_list_repo(i, in_index=(i % 2 == 0)) for i in range(10)]
    a = GRA.window(repos, offset=2, limit=3)
    assert [r["owner_repo"] for r in a] == ["o2/r2", "o3/r3", "o4/r4"]
    assert GRA.window(repos, offset=2, limit=3) == a                 # deterministic
    new = GRA.window(repos, offset=1, limit=2, in_index=False)
    assert [r["owner_repo"] for r in new] == ["o3/r3", "o5/r5"]
    assert len(GRA.window(repos, offset=8, limit=5)) == 2 and GRA.window(repos, offset=20) == []
    calls = []
    res = GRA.discover({"adapter_opts": {"repo_offset": 1, "repo_limit": 2, "repo_in_index": False,
                                         "workers": 2}},
                       repos=repos, acquire=_multi_acquire(calls), root=tmp_path)
    assert [c[0] for c in calls] == ["o3/r3", "o5/r5"] and {c[3] for c in calls} == {2}
    assert res["orgs_checked"] == 2 and res["candidates_found"] == 2
    with pytest.raises(ValueError):                                  # an empty window is loud
        GRA.discover({"adapter_opts": {"repo_offset": 99}}, repos=repos,
                     acquire=_multi_acquire([]), root=tmp_path)


def test_preserve_root_reuse_skips_a_repo_present_at_the_pin_and_fetches_the_rest(tmp_path):
    pre = tmp_path / "keep"
    first = []
    opts = {"preserve_root": str(pre)}
    r0 = _list_repo(0, sha="abc123")
    GRA.discover({"adapter_opts": opts}, repos=[r0], acquire=_multi_acquire(first), root=tmp_path)
    assert [c[0] for c in first] == ["o0/r0"]
    second = []
    res = GRA.discover({"adapter_opts": opts}, repos=[r0, _list_repo(1, sha="abc123")],
                       acquire=_multi_acquire(second), root=tmp_path)
    assert [c[0] for c in second] == ["o1/r1"]                       # o0 not fetched again
    assert res["orgs_reused"] == 1 and res["candidates_found"] == 2
    assert res["preserve_root"] == str(pre)
    third = []                                                       # pin moved -> fetched again
    GRA.discover({"adapter_opts": opts}, repos=[_list_repo(0, sha="d" * 40)],
                 acquire=_multi_acquire(third, revision="d" * 40), root=tmp_path)
    assert [c[0] for c in third] == ["o0/r0"]


def test_composer_passes_batch_flags_to_the_adapter_only_when_set():
    import argparse
    ns = argparse.Namespace(repo_offset=0, repo_limit=None, repo_in_index="any",
                            preserve_root=None, workers=1)
    assert RC.adapter_opts_from_args(ns) is None
    ns = argparse.Namespace(repo_offset=100, repo_limit=100, repo_in_index="false",
                            preserve_root="/x", workers=2)
    assert RC.adapter_opts_from_args(ns) == {"repo_offset": 100, "repo_limit": 100,
                                             "repo_in_index": False, "preserve_root": "/x",
                                             "workers": 2}
