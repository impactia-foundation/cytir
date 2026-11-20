"""task 2630: the head-sha skip of the github_repo_docs refresh adapter. No network: the acquirer and
the commit lookup are doubles; the composer's stage/pack/queue steps are faked around `run_one`."""
from __future__ import annotations

import os
import pathlib

os.environ.setdefault("AW_LANGGATE_OPTIONAL", "1")

from cytir.news import compose as C
from cytir.sources.refresh_direct import direct_compose as RC
from cytir.sources.refresh_direct import github_repo_docs as G
from tests.test_docs_delta_adapters import _list_repo, _multi_acquire

PACK = "us-github-repo-docs-tracked"
SHA_A, SHA_B = "a" * 40, "b" * 40


def _repos(n):
    return [_list_repo(i, sha=None) for i in range(n)]


def _lookup(heads_now, seen):
    def lookup(repo):
        seen.append(repo["owner_repo"])
        return {"head_sha": heads_now[repo["owner_repo"]], "tree_sha": "t" * 40}
    return lookup


def test_first_run_with_no_heads_file_fetches_everything_and_makes_no_lookup(tmp_path):
    calls, seen = [], []
    opts = {"heads_path": str(tmp_path / "heads.json")}
    res = G.discover({"adapter_opts": opts}, repos=_repos(3), acquire=_multi_acquire(calls, revision=SHA_A),
                     root=tmp_path, head_lookup=_lookup({}, seen))
    assert [c[0] for c in calls] == ["o0/r0", "o1/r1", "o2/r2"] and seen == []
    assert res["orgs_unchanged"] == 0 and res["candidates_found"] == 3
    assert set(res["repo_heads"]) == {"o0/r0", "o1/r1", "o2/r2"}
    assert not (tmp_path / "heads.json").exists()          # discover itself never writes the heads


def test_an_unreadable_heads_file_means_no_skips(tmp_path):
    hp = tmp_path / "heads.json"
    hp.write_text("{not json")
    assert G.load_heads(hp) == {} and G.load_heads(tmp_path / "missing.json") == {}


def test_second_run_with_identical_heads_skips_every_repo_with_zero_fetches_and_no_tombstones(tmp_path):
    hp = tmp_path / "heads.json"
    G.record_heads(hp, {f"o{i}/r{i}": {"head_sha": SHA_A} for i in range(3)}, pack_id="p-1")
    calls, seen = [], []
    now = {f"o{i}/r{i}": SHA_A for i in range(3)}
    res = G.discover({"adapter_opts": {"heads_path": str(hp)}}, repos=_repos(3),
                     acquire=_multi_acquire(calls), root=tmp_path, head_lookup=_lookup(now, seen))
    assert calls == []                                      # no tree listing, no raw GETs
    assert len(seen) == 3                                   # one commit lookup per repo
    assert res["orgs_unchanged"] == 3 and res["candidates"] == [] and res["repo_heads"] == {}

    # a skipped repo yields no kept rows, so the supersede pass observes nothing and tombstones nothing
    class NoCalls:
        def __getattr__(self, name):
            raise AssertionError(f"the supersede pass touched the twin: {name}")
    assert C.compute_superseded_tombstones(NoCalls(), "aw_x", []) == []


def test_one_moved_repo_is_fetched_alone_and_only_its_head_advances(tmp_path):
    hp = tmp_path / "heads.json"
    G.record_heads(hp, {f"o{i}/r{i}": {"head_sha": SHA_A} for i in range(3)}, pack_id="p-1")
    calls, seen = [], []
    now = {"o0/r0": SHA_A, "o1/r1": SHA_B, "o2/r2": SHA_A}
    res = G.discover({"adapter_opts": {"heads_path": str(hp)}}, repos=_repos(3),
                     acquire=_multi_acquire(calls, revision=SHA_B), root=tmp_path,
                     head_lookup=_lookup(now, seen))
    assert [c[0] for c in calls] == ["o1/r1"]
    assert res["orgs_unchanged"] == 2 and res["candidates_found"] == 1
    assert res["repo_heads"] == {"o1/r1": {"head_sha": SHA_B, "tree_sha": "t" * 40}}
    G.record_heads(hp, res["repo_heads"], pack_id="p-2")
    heads = G.load_heads(hp)
    assert {k: v["head_sha"] for k, v in heads.items()} == {"o0/r0": SHA_A, "o1/r1": SHA_B, "o2/r2": SHA_A}
    assert heads["o1/r1"]["pack_id"] == "p-2" and heads["o0/r0"]["pack_id"] == "p-1"


def test_a_failed_lookup_forfeits_the_skip_not_the_repo(tmp_path):
    hp = tmp_path / "heads.json"
    G.record_heads(hp, {"o0/r0": {"head_sha": SHA_A}}, pack_id="p-1")
    calls = []
    res = G.discover({"adapter_opts": {"heads_path": str(hp)}}, repos=_repos(1),
                     acquire=_multi_acquire(calls, revision=SHA_A), root=tmp_path,
                     head_lookup=lambda repo: None)
    assert [c[0] for c in calls] == ["o0/r0"] and res["orgs_unchanged"] == 0


def test_a_failed_lookup_is_counted_in_heads_lookup_failed_and_the_repo_is_still_fetched(tmp_path):
    hp = tmp_path / "heads.json"
    G.record_heads(hp, {"o0/r0": {"head_sha": SHA_A}}, pack_id="p-1")
    calls = []
    res = G.discover({"adapter_opts": {"heads_path": str(hp)}}, repos=_repos(1),
                     acquire=_multi_acquire(calls, revision=SHA_A), root=tmp_path,
                     head_lookup=lambda repo: None)
    assert [c[0] for c in calls] == ["o0/r0"]
    assert res["heads_lookup_failed"] == 1


def _run_one(tmp_path, monkeypatch, *, stage_only, repo_heads, errors=None, heads_is_dir=False):
    monkeypatch.setattr(RC, "W", tmp_path / "w")
    monkeypatch.setattr(RC, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(RC, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(RC, "run_stage", lambda pack_id, **kw: {
        "staged": 5, "collection": "c", "candidates": 5, "candidates_discovered": 5,
        "errors": errors or {}, "discovery_errors": {}, "challenged": 0, "unchanged": 0,
        "unchanged_repos": 2, "repo_heads": repo_heads})
    monkeypatch.setattr(RC, "run_pack", lambda *a, **k: {"rows": 5, "artifact": tmp_path / "a" / "x"})
    monkeypatch.setattr(C, "read_pack_points", lambda d: [])
    monkeypatch.setattr(C, "merge_rows", lambda rows, **k: {"kept": [], "undated": []})

    def write_pack(out_dir, **kw):
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(out_dir) / "READY").write_text("")
        return {"pack_id": kw["pack_id"]}
    monkeypatch.setattr(C, "write_pack", write_pack)
    hp = tmp_path / "heads.json"
    if heads_is_dir:
        hp.mkdir()
    so = str(tmp_path / "staged") if stage_only else None

    class Args:
        cadence_table = str(tmp_path / "cadence.json")
        collection = "aw_x"
        twin_url = "http://127.0.0.1:1"
        staging_url = "http://127.0.0.1:2"
        mrl_dim = 256
        since_days = 3
        no_tombstones = True
        egress_iface = "lo"
        pack_suffix = ""
        stage_only = so
        preserve_root = None
    monkeypatch.setattr(RC, "adapter_opts_from_args", lambda a: {"heads_path": str(hp)})
    return RC.run_one(PACK, Args()), hp


def test_a_queued_run_writes_the_heads_after_queueing_and_reports_unchanged_repos(tmp_path, monkeypatch):
    res, hp = _run_one(tmp_path, monkeypatch, stage_only=False,
                       repo_heads={"o0/r0": {"head_sha": SHA_A, "tree_sha": None},
                                   "o1/r1": {"head_sha": SHA_B, "tree_sha": None}},
                       errors={"https://github.com/o1/r1/blob/" + SHA_B + "/README.md": "boom"})
    assert res["queued"] is True and res["unchanged_repos"] == 2
    heads = G.load_heads(hp)
    assert set(heads) == {"o0/r0"}                          # o1/r1 had a file error: retried next run
    assert heads["o0/r0"]["head_sha"] == SHA_A and heads["o0/r0"]["pack_id"].startswith("refresh-direct-")


def test_a_stage_only_run_does_not_write_the_heads(tmp_path, monkeypatch):
    res, hp = _run_one(tmp_path, monkeypatch, stage_only=True,
                       repo_heads={"o0/r0": {"head_sha": SHA_A, "tree_sha": None}})
    assert res["queued"] is False and not hp.exists()


def test_an_unwritable_heads_path_leaves_the_queued_result_intact(tmp_path, monkeypatch, capsys):
    res, hp = _run_one(tmp_path, monkeypatch, stage_only=False, heads_is_dir=True,
                       repo_heads={"o0/r0": {"head_sha": SHA_A, "tree_sha": None}})
    assert res["queued"] is True
    assert "[refresh-direct] heads not recorded:" in capsys.readouterr().err
