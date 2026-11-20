"""cytir.retune (task 724, ADR-099 amendment 3): the retune EMITS a candidate and a report.

What these tests pin:
  1. FIDELITY. The tuner's top-1 equals the server's top-1 on the golden sample (60 queries, the
     task-711 audit sample, where the old simulator agreed on 49/60). The arm asks the server's own
     `GET /search`; the test fails by NAMING the rows that differ.
  2. NO SECOND COPY. No scoring helper of the server is imported or restated under cytir/retune/
     (the private tree's test also pins that the old simulator and the deploy script are gone).
  3. NO DEPLOY. Deploy-shaped flags, serving-path outputs and non-loopback qdrants are refused.
  4. THE JUDGE. ADR-043 ranking bar, null band, the chooser split by class.
  5. AN ERROR IS NOT A MISS. A non-200 raises.
"""
from __future__ import annotations

import ast
import hashlib
import json
import pathlib

import pytest
import yaml

from cytir.retune import arm, emit, grid, guard, judge
from cytir.retune.score import aggregate, hit_rank

REPO = pathlib.Path(__file__).resolve().parents[1]
SAMPLE = REPO / "tests" / "fixtures" / "retune_fidelity_sample.yaml"


# ------------------------------------------------------------------ 1. fidelity --

def _fake_search(q, k=10, lang=None, country=None, **_):
    """A deterministic stand-in for retrieval.search(): the ranking is a function of the query only."""
    h = hashlib.sha256(q.encode()).hexdigest()
    return [{"url": f"https://example.org/{h[i:i + 6]}/", "score": 1.0 - i / 100} for i in range(0, 40, 4)][:k]


def test_tuner_top1_equals_server_top1_on_the_golden_sample(monkeypatch):
    rows = yaml.safe_load(SAMPLE.read_text())["eval"]
    assert len(rows) == 60
    monkeypatch.syspath_prepend(str(REPO / "server"))
    from cytir.serve import retrieval
    monkeypatch.setattr(retrieval, "search", _fake_search)
    # the target of every row is its server #1, so a re-ranked arm shows up as a rank != 1
    server_top1 = {r["id"]: _fake_search(r["query"], k=10)[0]["url"] for r in rows}
    tuned = arm.run_chooser([{**r, "target_url": server_top1[r["id"]]} for r in rows],
                            arm._testclient_get(), k=10)
    differ = [(i, v["urls"][:1], server_top1[i]) for i, v in tuned["per_q"].items()
              if v["urls"][:1] != [server_top1[i]]]
    assert not differ, f"the tuner's top-1 differs from the server's on {len(differ)}/60 rows: {differ}"
    assert tuned["h1"] == 60


def test_the_arm_passes_exactly_q_and_k_like_run_eval(monkeypatch):
    """The audit sample's rows carry a `lang` and a `type`; neither may reach /search (run_eval
    and task 533's arms send q and k only)."""
    monkeypatch.syspath_prepend(str(REPO / "server"))
    from cytir.serve import retrieval
    seen = []
    monkeypatch.setattr(retrieval, "search", lambda q, k=10, lang=None, country=None, **kw: seen.append((lang, country, kw)) or [])
    arm.run_chooser([{"id": "x", "query": "q", "target_url": "https://a.b/", "lang": "de", "type": "legal"}],
                    arm._testclient_get(), k=10)
    assert seen == [(None, None, {})]


# --------------------------------------------------------------- 2. no second copy --

SERVER_SCORING = {"_minmax", "_exact", "_exact_v", "_recency", "_title_centrality", "_title_centrality_v",
                  "_content_terms", "_query_terms", "_academic_penalty", "_machine_url", "_parasite",
                  "_trust_tier", "_canonical_key", "candidate_pool", "fuse_margin", "_DENSE_SP", "WEIGHTS"}


@pytest.mark.parametrize("name", ["arm", "emit", "judge", "grid", "score", "preflight"])
def test_no_server_scoring_helper_is_imported_or_restated(name):
    tree = ast.parse((REPO / "cytir" / "retune" / f"{name}.py").read_text())
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
           {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
           {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert not (used | defined) & SERVER_SCORING, sorted((used | defined) & SERVER_SCORING)


# ------------------------------------------------------------------ 3. no deploy --

@pytest.mark.parametrize("flag", ["--deploy", "--promote", "--apply", "--restart", "--stage", "--flip",
                                  "--live-weights=/x", "--promote-target", "--weights-out", "--incumbent-weights=/opt/aw/x.json"])
def test_deploy_shaped_flags_exit_nonzero_before_anything_runs(flag):
    with pytest.raises(SystemExit) as e:
        emit.main([flag, "--collection", "c", "--qdrant-url", "http://localhost:1", "--heldout-cache", "/nope"])
    assert e.value.code == emit.EXIT_REFUSED != 0
    assert "deploy-shaped" in str(e.value)


def test_old_mode_env_is_refused(monkeypatch):
    monkeypatch.setenv("AW_RETUNE_MODE", "prod")
    with pytest.raises(SystemExit) as e:
        emit.main(["--collection", "c", "--qdrant-url", "http://localhost:1", "--heldout-cache", "/nope"])
    assert e.value.code == emit.EXIT_REFUSED


def _url(port, host="localhost"):
    return f"http://{host}:{port}"          # built, not spelled: no literal endpoint in a test


TWIN_PORT, TUNNEL_PORT = 6457, 6334
SS_TUNNEL = ('LISTEN 0 128 127.0.0.1:6334 0.0.0.0:* users:(("ssh",pid=393969,fd=5))\n'
             'LISTEN 0 4096 127.0.0.1:6457 0.0.0.0:* users:(("docker-proxy",pid=4132,fd=4))\n')
ALIASES = {"aw_prod": "aw_e9"}


@pytest.mark.parametrize("out", [
    "/opt/aw/cytir/serve", str(REPO / "cytir" / "serve" / "w"), str(REPO / "deploy" / "x"),
    "/app/aw/serve/x",  # rename-codemod: keep (the live image and the boxes keep the old name)
    "/cytir/serve/x", "/app/cytir/serve/x"])
def test_serving_paths_are_not_output_directories(out):
    with pytest.raises(SystemExit) as e:
        guard.refuse_serving_path(out, "--out-dir")
    assert e.value.code == emit.EXIT_REFUSED


def test_work_dir_and_arm_out_are_checked_too(tmp_path):
    for bad in ("/opt/aw/x", "/app/cytir/serve/x"):
        with pytest.raises(SystemExit) as e:
            emit.main(["--collection", "aw_e8", "--qdrant-url", _url(TWIN_PORT), "--heldout-cache", "/nope",
                       "--work-dir", bad])
        assert e.value.code == emit.EXIT_REFUSED
        with pytest.raises(SystemExit) as e:
            arm.main(["--weights", "/w.json", "--out", bad + "/arm.json"])
        assert e.value.code == emit.EXIT_REFUSED


def test_the_twin_passes_the_guard():
    guard.refuse_non_twin(_url(TWIN_PORT), "aw_e8", aliases=ALIASES, ss_text=SS_TUNNEL)
    guard.refuse_non_twin(_url(TWIN_PORT, "127.0.0.1"), "aw_e8", aliases={}, ss_text=SS_TUNNEL)


def test_the_exact_env_pair_the_review_found_is_refused():
    """AW_RETUNE_TWIN_QDRANT=<the tunnel port> AW_RETUNE_COLLECTION=aw_prod passed every old refusal."""
    with pytest.raises(SystemExit) as e:
        guard.refuse_non_twin(_url(TUNNEL_PORT), "aw_prod", aliases=ALIASES, ss_text=SS_TUNNEL)
    assert e.value.code == emit.EXIT_REFUSED and "not a declared twin endpoint" in str(e.value)


@pytest.mark.parametrize("url,coll,why", [
    (_url(TUNNEL_PORT), "aw_e8", "declared twin endpoint"),                  # the tunnel with a twin collection name
    (_url(TWIN_PORT), "aw_prod", "declared twin collection"),
    (_url(TWIN_PORT), "aw_e9", "declared twin collection"),                  # the alias TARGET
    ("https://prod.example.org", "aw_e8", "declared twin endpoint"),
    (_url(TWIN_PORT, "10.0.0.5"), "aw_e8", "declared twin endpoint"),
])
def test_every_non_twin_pair_is_refused(url, coll, why):
    with pytest.raises(SystemExit) as e:
        guard.refuse_non_twin(url, coll, aliases=ALIASES, ss_text=SS_TUNNEL)
    assert e.value.code == emit.EXIT_REFUSED and why in str(e.value)


def test_a_twin_named_collection_that_is_an_alias_of_prod_is_refused():
    with pytest.raises(SystemExit) as e:
        guard.refuse_non_twin(_url(TWIN_PORT), "aw_e8", aliases={"aw_prod": "aw_e8"}, ss_text=SS_TUNNEL)
    assert "that is prod" in str(e.value)
    with pytest.raises(SystemExit) as e:        # an alias NAME that resolves to prod's target
        guard.refuse_non_twin(_url(TWIN_PORT), "aw_e8", aliases={"aw_prod": "x", "aw_e8": "x"}, ss_text=SS_TUNNEL)
    assert "that is prod" in str(e.value)


def test_unreadable_aliases_refuse_because_we_cannot_tell(monkeypatch):
    monkeypatch.setattr(guard, "fetch_aliases", lambda url: None)
    with pytest.raises(SystemExit) as e:
        guard.refuse_non_twin(_url(TWIN_PORT), "aw_e8", ss_text=SS_TUNNEL)
    assert "cannot tell" in str(e.value)


def test_an_ssh_listener_on_the_port_is_refused_and_an_unreadable_listener_is_loud(monkeypatch):
    assert guard.listener_is_ssh(TUNNEL_PORT, SS_TUNNEL) is True
    assert guard.listener_is_ssh(TWIN_PORT, SS_TUNNEL) is False
    ssh_on_twin_port = SS_TUNNEL.replace("docker-proxy", "ssh")
    with pytest.raises(SystemExit) as e:
        guard.refuse_non_twin(_url(TWIN_PORT), "aw_e8", aliases=ALIASES, ss_text=ssh_on_twin_port)
    assert "ssh process" in str(e.value)
    monkeypatch.setattr(guard.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("ss")))
    said = []
    guard.refuse_non_twin(_url(TWIN_PORT), "aw_e8", aliases=ALIASES, say=said.append)
    assert said and "WARNING" in said[0]


def test_the_arm_cli_refuses_a_non_twin_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AW_QDRANT_URL", _url(TUNNEL_PORT))
    monkeypatch.setenv("AW_COLLECTION", "aw_prod")
    with pytest.raises(SystemExit) as e:
        arm.worker(type("A", (), {})())
    assert e.value.code == emit.EXIT_REFUSED


def test_the_twin_endpoint_and_collection_come_from_the_declared_homes():
    from cytir.core import twin
    assert guard._hostport(twin.configured_url()) in guard.twin_endpoints()
    assert "aw_e8" in guard.twin_collections() and "aw_e9" not in guard.twin_collections()


def test_the_instrument_pins_come_from_the_flip_gate_contract():
    chooser = REPO / "data" / "eval_tasks.v2.yaml"
    if not chooser.is_file():
        pytest.skip("the public cut does not ship the chooser set")
    pins = emit.contract_pins()
    assert pins["chooser"] == hashlib.sha256((REPO / "data" / "eval_tasks.v2.yaml").read_bytes()).hexdigest()


def test_a_moved_chooser_is_refused(tmp_path):
    moved = tmp_path / "c.yaml"
    moved.write_text("eval: []\n")
    cache = tmp_path / "v.json"
    cache.write_text("{}")
    with pytest.raises(SystemExit) as e:
        emit.check_instruments(str(moved), str(cache))
    assert e.value.code == emit.EXIT_REFUSED


def test_the_incumbent_is_the_file_the_serving_env_names():
    inc = emit.incumbent_path()
    assert pathlib.Path(inc).name == "fusion_weights.candidate533.json"
    assert pathlib.Path(inc).is_file()


def test_arm_env_drops_inherited_aw_knobs_and_points_at_the_twin():
    env = arm.build_env("/w.json", "aw_e8", _url(TWIN_PORT), _url(8899),
                        base={"AW_AUTHORITY": "0", "AW_RANKHEAD": "0", "PATH": "/bin"})
    assert env["AW_AUTHORITY"] == "1" and env["AW_RANKHEAD"] == "1"      # prod's values, not the caller's
    assert env["AW_COLLECTION"] == "aw_e8" and env["AW_QDRANT_URL"] == _url(TWIN_PORT)
    assert env["AW_FUSION_WEIGHTS"] == "/w.json" and env["PATH"] == "/bin"
    assert env["AW_MRL_DIM"] == "256"


def test_every_arm_runs_with_one_pinned_hash_seed():
    """Task 752 D6: exact score ties follow set order, which follows the per-process hash salt.
    A null pair of two unpinned arms moves; the seed is pinned, and a caller's own seed is not
    inherited."""
    env = arm.build_env("/w.json", "aw_e8", _url(TWIN_PORT), _url(8899),
                        base={"PYTHONHASHSEED": "random", "PATH": "/bin"})
    assert env["PYTHONHASHSEED"] == arm.HASH_SEED == "0"


# ----------------------------------------------------------------- 5. error != miss --

def test_a_non_200_is_an_error_not_a_miss():
    rows = [{"id": "a", "query": "q", "target_url": "https://a.b/"}]
    with pytest.raises(RuntimeError, match="503"):
        arm.run_chooser(rows, lambda q, k: (503, {"error": "x"}))


def test_run_chooser_scores_rank_and_type():
    rows = [{"id": "a", "query": "q", "target_url": "https://a.b/p", "type": "legal"}]
    out = arm.run_chooser(rows, lambda q, k: (200, {"results": [{"url": "https://x.y/"}, {"url": "https://a.b/p/"}]}))
    assert out["per_q"]["a"] == {"urls": ["https://x.y/", "https://a.b/p/"], "rank": 2, "type": "legal"}
    assert (out["h1"], out["h10"], out["mrr"]) == (0, 1, 0.5)


def test_hit_rank_rule_matches_run_eval():
    assert hit_rank("https://a.b/c", ["https://x/", "http://A.B/c/"]) == 2
    assert hit_rank("https://a.b/c", ["https://a.b/c/d"]) == 1       # prefix, as run_eval.py
    assert hit_rank("https://a.b/c", ["https://z.z/"]) is None
    assert aggregate({"a": {"rank": 1}, "b": {"rank": None}})["mrr"] == 0.5


# ------------------------------------------------------------------- 4. the judge --

def _arm(ranks, held=None):
    sets = {"chooser": {"per_q": {f"q{i}": {"urls": [f"u{r}"], "rank": r, "type": "legal" if i % 2 else "coding"}
                                  for i, r in enumerate(ranks)}}}
    if held is not None:
        sets["authoritative"] = {"n": len(held), "n_gold": len(held),
                                 "h1": sum(1 for r in held if r == 1), "h10": sum(1 for r in held if r),
                                 "mrr": 0.5, "per_q": {f"h{i}": {"urls": [f"v{r}"], "rank": r} for i, r in enumerate(held)}}
    return {"sets": sets}


def test_chooser_is_split_by_type():
    assert sorted(judge.classes(_arm([1, 2, 3, 4]))) == ["chooser:coding", "chooser:legal"]


def test_pass_needs_a_gain_above_the_band_and_no_regression():
    inc, same = _arm([2, 3, None, 5], [1, 2]), _arm([2, 3, None, 5], [1, 2])
    cand = _arm([1, 3, 4, 5], [1, 2])
    cmp, null = judge.compare(inc, cand), judge.compare(inc, same)
    assert null["moved"] == 0
    bar = judge.bar(cmp, null)
    assert bar["verdict"] == "PASS", bar
    assert cmp["chooser_net"] > 0


def test_one_lost_gold_fails_the_strict_ranking_bar():
    inc = _arm([1, 3, None, 5])
    cand = _arm([1, 4, 2, 5])                       # q1: 3 -> 4 is a regression even though q2 gained
    bar = judge.bar(judge.compare(inc, cand), judge.compare(inc, inc))
    assert bar["verdict"] == "FAIL" and "lost rank" in bar["reasons"][0]
    assert judge.compare(inc, cand)["regressed"] == [{"class": "chooser:legal", "id": "q1", "from": 3, "to": 4}]


def test_heldout_regression_alone_fails():
    inc, cand = _arm([2, 3], [1, 2]), _arm([1, 3], [1, 3])
    assert judge.bar(judge.compare(inc, cand), judge.compare(inc, inc))["verdict"] == "FAIL"


def test_a_loss_inside_the_null_band_is_inconclusive_and_never_a_pass():
    inc, noisy = _arm([1, 2, 5, 6]), _arm([1, 3, 5, 6])          # the null pair moved one gold
    cand = _arm([1, 3, 1, 1])                                    # q1 lost rank (2 -> 3); three gained
    bar = judge.bar(judge.compare(inc, cand), judge.compare(inc, noisy))
    assert bar["verdict"] == "INCONCLUSIVE" and bar["band"] == 1


def test_a_loss_larger_than_the_null_band_fails():
    inc, noisy = _arm([1, 2, 5, 6]), _arm([1, 3, 5, 6])
    cand = _arm([2, 3, 6, 7])                                    # four golds lost rank, band 1
    assert judge.bar(judge.compare(inc, cand), judge.compare(inc, noisy))["verdict"] == "FAIL"


def test_a_gain_inside_the_band_is_not_better_even_with_no_loss():
    inc, noisy = _arm([2, 3]), _arm([2, 4])
    cand = _arm([1, 3])                                          # +1 h@1 and +1 h@10? net 1 <= band 1
    assert judge.bar(judge.compare(inc, cand), judge.compare(inc, noisy))["verdict"] == "NOT_BETTER"


def test_no_regression_but_no_gain_is_not_better():
    inc = _arm([2, 3])
    assert judge.bar(judge.compare(inc, inc), judge.compare(inc, inc))["verdict"] == "NOT_BETTER"


def test_arms_that_scored_different_queries_cannot_be_compared():
    with pytest.raises(ValueError):
        judge.compare(_arm([1, 2]), _arm([1, 2, 3]))


def test_pick_winner_prefers_fewest_regressions_then_largest_gain():
    inc = _arm([2, 3, 4, 5])
    arms = {"loud": _arm([1, 1, 1, 6]), "clean": _arm([1, 3, 4, 5]), "flat": _arm([2, 3, 4, 5])}
    winner, scored = judge.pick_winner(arms, inc)
    assert winner == "clean" and scored["loud"]["regressed"] and not scored["flat"]["chooser_net"]
    assert judge.pick_winner({"flat": _arm([2, 3, 4, 5])}, inc)[0] is None


def test_grid_moves_one_key_and_never_mutates():
    inc = {"academic": -2.0, "trust": 3.0, "title": 2.5, "bm25": 1.0, "dense": 1.0}
    before = json.dumps(inc)
    props = grid.proposals(inc)
    assert json.dumps(inc) == before
    for name, w in props:
        assert sum(1 for k in inc if w[k] != inc[k]) == 1, name
    assert all("dense" not in n for n, _ in props) and "academic=-2" not in [n for n, _ in props]
    assert grid.proposals(inc, ["machine"]) == []          # absent from the incumbent: not proposed


def test_report_puts_the_decision_first():
    c = {"emitted_at": "2026-10-08T05:00:00Z", "bar": {"verdict": "FAIL", "band": 0, "reasons": ["2 gold(s) lost rank"]},
         "candidate": {"change": "trust=3.5"}, "index": {"collection": "aw_e8", "points": 61003110},
         "incumbent": {"name": "fusion_weights.candidate533.json", "sha256": "ab" * 32}, "null": {"moved": 0},
         "chooser_summary": {"d_h1": 1, "d_h10": 0, "d_mrr": 0.001}, "heldout_summary": {"regressed": 2, "improved": 1},
         "arms": {"trust=3.5": {"chooser_net": 1, "regressed": 2}}, "wall_seconds": 600, "classes": [],
         "regressed": [], "instruments": {"gate_set_version": "v4"}}
    md = emit.render_report(c)
    assert md.splitlines()[0].endswith("FAIL")
    assert md.splitlines()[2].startswith("**Decision: do not promote `trust=3.5`")
    assert "fusion_weights.candidate533.json" in md and "FLIP_FUSION_WEIGHTS_FILE" in md and "cytir/core/serving_config.py" in md


def test_report_renders_for_no_candidate_and_for_a_pass():
    base = {"emitted_at": "2026-10-08T05:00:00Z", "index": {"collection": "aw_e8", "points": 61003110, "prod_points": 61003110},
            "incumbent": {"name": "w.json", "sha256": "cd" * 32}, "null": {"moved": 0}, "arms": {}, "wall_seconds": 61,
            "classes": [{"class": "chooser:legal", "n_gold": 2, "a": {"h1": 1, "h10": 2}, "b": {"h1": 2, "h10": 2}, "d_mrr": 0.1, "reordered": 1}],
            "regressed": [], "instruments": {"gate_set_version": "v4"}}
    none = emit.render_report({**base, "bar": {"verdict": "NO_CANDIDATE", "band": 0, "reasons": []}, "candidate": None,
                               "chooser_summary": {"d_h1": 0, "d_h10": 0, "d_mrr": 0.0}})
    assert "**Decision: nothing to promote.**" in none and "(prod 61,003,110)" in none
    ok = emit.render_report({**base, "bar": {"verdict": "PASS", "band": 0, "reasons": []}, "candidate": {"change": "title=3"},
                             "chooser_summary": {"d_h1": 1, "d_h10": 0, "d_mrr": 0.1}, "heldout_summary": {"regressed": 0, "improved": 2}})
    assert "**Decision: you may promote `title=3`.**" in ok


def test_a_missing_instrument_file_is_a_refusal_not_a_traceback(tmp_path):
    with pytest.raises(SystemExit) as e:
        emit.check_instruments(str(REPO / "data" / "eval_tasks.v2.yaml"), str(tmp_path / "nope.json"))
    assert e.value.code == emit.EXIT_REFUSED and "does not exist" in str(e.value)


def test_perquery_csv_lists_only_queries_with_a_signal():
    inc, nul, cand = _arm([1, None, 3]), _arm([1, None, 3]), _arm([2, None, 3])
    rows = judge.perquery_csv(inc, nul, cand).splitlines()
    assert rows[0] == "class,id,live,null,candidate"
    assert sorted(rows[1:]) == ["chooser:coding,q0,1,1,2", "chooser:coding,q2,3,3,3"]


# ---- M2: an error row is not a miss; reused arms are verified -------------------------------

def _doc_with_error():
    return {"sets": {"authoritative": {"per_q": {"a": {"urls": ["https://x/"], "rank": 1},
                                                 "b": {"urls": ["__ERROR__:ReadTimeout"], "rank": None},
                                                 "c": {"urls": [], "rank": None}}}}}


def test_an_arm_with_an_error_row_is_refused_but_an_empty_result_is_not():
    assert judge.error_rows(_doc_with_error()) == [("authoritative", "b")]
    with pytest.raises(SystemExit) as e:
        judge.assert_no_errors(_doc_with_error(), "incumbent")
    assert e.value.code == 4 and "error is not a miss" in str(e.value)
    clean = _doc_with_error()
    del clean["sets"]["authoritative"]["per_q"]["b"]
    judge.assert_no_errors(clean, "ok")


def test_the_incumbent_error_that_would_hide_a_candidate_loss_is_caught_before_the_judge():
    """The false-PASS route: the incumbent errored on a gold (rank None), so a candidate that loses it
    cannot 'regress'. The judge alone says PASS; the integrity check refuses the arm first."""
    inc = {"sets": {"authoritative": {"n": 1, "n_gold": 1, "h1": 0, "h10": 0, "mrr": 0.0,
                                      "per_q": {"g": {"urls": ["__ERROR__:X"], "rank": None}}}}}
    cand = {"sets": {"authoritative": {"n": 1, "n_gold": 1, "h1": 1, "h10": 1, "mrr": 1.0,
                                       "per_q": {"g": {"urls": ["u"], "rank": 1}}}}}
    assert judge.compare(inc, cand)["regressed"] == []
    with pytest.raises(SystemExit):
        judge.assert_no_errors(inc, "incumbent")


def test_a_reused_arm_must_match_weights_collection_and_code(tmp_path):
    w = tmp_path / "w.json"
    w.write_text("{}")
    sha = arm.sha256_file(w)
    job = {"name": "incumbent", "weights": str(w)}
    good = {"weights_sha256": sha, "collection": "aw_e8", "code": "abc"}
    emit.verify_reused(good, job, "abc", "aw_e8")
    for bad in ({**good, "code": None}, {**good, "code": "zzz"}, {**good, "collection": "x"}, {**good, "weights_sha256": "0" * 64}):
        with pytest.raises(SystemExit):
            emit.verify_reused(bad, job, "abc", "aw_e8")


def test_an_existing_arm_file_is_refused_without_resume(tmp_path):
    out = tmp_path / "arm.json"
    out.write_text("{}")
    with pytest.raises(SystemExit, match="--resume"):
        emit._run_arms([{"name": "x", "weights": "/w", "out": str(out)}], 1, stamps={}, points_fn=lambda: 1,
                       code="c", resume=False, collection="aw_e8")


# ------------------------------------------------------------------ the prod echo names the OLD or the renamed image
def test_preflight_reads_the_old_and_the_renamed_image_prefix_as_the_same_serve_file():
    from cytir.retune import preflight
    for pkg in ("aw", "cytir"):  # rename-codemod: keep (the old name is the live image)
        served = {"hubfloor_table": f"/app/{pkg}/serve/hub_hosts.aw_b1.json"}
        local = {"hubfloor_table": str(REPO / "cytir" / "serve" / "hub_hosts.aw_b1.json")}
        assert preflight.config_parity_problems(served, local) == [], pkg


def test_preflight_refuses_an_image_path_under_no_known_prefix_by_naming_the_key():
    from cytir.retune import preflight
    problems = preflight.config_parity_problems({"hubfloor_table": "/app/other/x.json"},
                                                {"hubfloor_table": "/app/other/x.json"})
    assert any("hubfloor_table" in p and "/app/other/x.json" in p and "refused" in p for p in problems), problems
