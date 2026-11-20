"""Slice 3 of the producer restore (task 1088): the general-eval pair. (The mechanism-caller census has its own freshness test.)

The two general-eval producers need a served tree or an API key for their full run. These tests drive the pure
parts (the pool build end to end on a mini data dir; the judge stage end to end with the API call stubbed;
the retrieve stage is NOT exercised: it needs a served tree, a twin and a trained head)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cytir import cli
from cytir.ops import eval_general_run as egr, general_pool_build as gpb

APP = Path(__file__).resolve().parents[1]


def test_pool_build_dedupes_filters_and_writes_a_sidecar(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "labels_search_arena.jsonl").write_text("\n".join([
        json.dumps({"query": "March Madness  tournament teams", "lang": "en"}),   # dup of a flood seed after norm
        json.dumps({"query": "site:example.com", "lang": "en"}),                  # operator: dropped
        json.dumps({"query": "wie funktioniert die ahv", "lang": "de"}),
        "not json",
    ]) + "\n")
    (data / "eval_agentic.yaml").write_text("- {query: 'latest python release', lang: en, vertical: tech}\n")
    out = tmp_path / "pool.jsonl"
    assert gpb.main([str(out), "--data", str(data)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    qs = [r["query"] for r in rows]
    assert len(qs) == len(set(gpb.norm(q) for q in qs))
    assert "wie funktioniert die ahv" in qs and "latest python release" in qs
    assert not any(q.startswith("site:") for q in qs)
    assert sum(1 for r in rows if gpb.norm(r["query"]) == "march madness tournament teams") == 1
    assert {r["src"] for r in rows} == {"flood_seed", "search_arena", "agentic"}
    side = json.loads((tmp_path / "pool.jsonl.provenance.json").read_text())
    assert side["producer"] == "aw ops general-pool-build"


def test_judge_stage_scores_a_results_file_with_the_api_stubbed(tmp_path, monkeypatch):
    rows = [
        {"id": "a", "query": "q1", "flood_prone": True, "p_general_205": 0.9, "p_general_206": 0.8,
         "results": [{"rank": 1, "url": "https://www.canada.ca/x", "title": "t", "trust_tier": 0.95},
                     {"rank": 2, "url": "https://blog.example/y", "title": "u", "trust_tier": 0.3}]},
        {"id": "b", "query": "q2", "flood_prone": False, "results": []},
    ]
    res, out = tmp_path / "results.jsonl", tmp_path / "scored.json"
    res.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(egr, "_call", lambda model, user, key, retries=4: 'x {"flood_ranks":[1],"rel_at_1":false}')
    assert egr.host("https://www.canada.ca/x") == "canada.ca"
    egr.main(["judge", "--results", str(res), "--out", str(out), "--workers", "1"])
    s = json.loads(out.read_text())["summary"]
    assert s["n"] == 2 and s["n_flood_prone"] == 1
    assert s["FLOOD_mean"] == 0.25 and s["FLOOD_mean_flood_prone"] == 0.5   # (1/2 + empty 0) / 2 ; 1/2
    assert s["routing_gated_floodprone_206"] == 1.0


def test_judge_stage_refuses_without_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        egr.main(["judge", "--results", str(tmp_path / "r"), "--out", str(tmp_path / "o")])


def test_new_verbs_are_dispatched_once_each():
    for verb in ("eval-general-run", "general-pool-build"):
        assert list(cli._PRODUCER_VERBS).count(verb) == 1
        with pytest.raises(SystemExit) as e:
            cli._ops_main([verb, "--help"])
        assert e.value.code == 0
