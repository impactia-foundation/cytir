"""Serve parity records `ranked_equal` and `envelope_paths` per request, and `check
--envelope-change REF --envelope-fields ...` accepts an envelope-only FAIL report (task 2679).

Reports here are synthetic: a real PASS report between two output-neutral trees (so the pair and
the tree digests are genuine) with its DIFF rows written by hand."""
from __future__ import annotations

import copy
import json
import os
import subprocess

import pytest

import test_aw_serve_parity as TP
from cytir import cli
from cytir.gates import serve_parity as sp

BASE_SHA = "a" * 40
FIELDS = "boosts.country,filters.note,notes[]"


def _seq(base_body, head_body, base_deep=None, head_deep=None):
    seq = {"base": {"status": 200, "body": base_body}, "head": {"status": 200, "body": head_body},
           "null": {"status": 200, "body": base_body}}
    if base_deep is not None:
        seq["base_deep"] = {"status": 200, "body": base_body, "deep": base_deep}
        seq["null_deep"] = {"status": 200, "body": base_body, "deep": base_deep}
        seq["head_deep"] = {"status": 200, "body": head_body, "deep": head_deep}
    return seq


RES = [{"id": "p1", "url": "u1", "score": 0.9}, {"id": "p2", "url": "u2", "score": 0.5}]


def test_classify_records_the_ranked_list_apart_from_the_envelope():
    a = {"results": RES, "q": "x"}
    b = {"results": copy.deepcopy(RES), "q": "x", "boosts": {"country": "FR"}, "filters": {"note": "n"}}
    deep = {"score_debug": [1.0], "trace": {"k": 1}, "rankhead_v2_dump": []}
    c = sp.classify(_seq(a, b, deep, copy.deepcopy(deep)), 1e-6)
    assert c["state"] == "DIFF" and c["tier"] == "served"
    assert c["ranked_equal"] == {"served": True, "deep": True}
    assert c["envelope_paths"] == ["boosts", "filters"]
    # one level into a key present on both sides
    a2 = dict(a, filters={})
    c = sp.classify(_seq(a2, b), 1e-6)
    assert c["envelope_paths"] == ["boosts", "filters.note"]
    assert c["ranked_equal"] == {"served": True, "deep": None}


def test_classify_sees_a_ranked_change_hidden_behind_a_key_set_difference():
    a = {"results": RES}
    b = {"results": list(reversed(RES)), "boosts": {}}
    c = sp.classify(_seq(a, b), 1e-6)
    assert c["state"] == "DIFF" and c["ranked_equal"]["served"] is False
    assert c["envelope_paths"] == ["boosts"]


def test_the_deep_tier_is_compared_even_when_the_served_tier_differs():
    a = {"results": RES}
    b = {"results": RES, "boosts": {}}
    d1 = {"score_debug": [1.0], "trace": {}, "rankhead_v2_dump": []}
    d2 = {"score_debug": [2.0], "trace": {}, "rankhead_v2_dump": []}
    c = sp.classify(_seq(a, b, d1, d2), 1e-6)
    assert c["state"] == "DIFF" and c["tier"] == "served"
    assert c["ranked_equal"] == {"served": True, "deep": False}


def test_an_equal_request_is_still_equal_and_ranked_equal():
    a = {"results": RES}
    c = sp.classify(_seq(a, copy.deepcopy(a)), 1e-6)
    assert c["state"] == "EQUAL" and c["ranked_equal"]["served"] is True and c["envelope_paths"] == []


# ------------------------------------------------------------------ the check mode

def _git(repo, *a):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    t = tmp_path_factory.mktemp("env")
    base = TP.make_tree(t, "base")
    head = TP.make_tree(t, "noop", tweak="noop")
    rep = sp.run_parity(TP.requests(), base_tree=base, head_tree=head, collection="twin_x",
                        qdrant_url=TP.free_url(), vec_cache=TP.write_cache(t / "v.json"),
                        out_dir=t / "out", shards=1, deep=True, seed_probe=False,
                        guard=lambda url, coll: None, declared_env={"AW_FAKE_KNOB": "1"},
                        base_sha=BASE_SHA, head_sha="b" * 40, say=lambda m: None)
    assert rep["verdict"] == "PASS"
    return {"base": base, "head": head, "report": rep}


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    (r / "plan.md").write_text("# plan\n- T2679 envelope release\n")
    _git(r, "init", "-q")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "release")
    return r, _git(r, "rev-parse", "HEAD")


def _row(i, *, served=True, deep=True, paths=("boosts.country",)):
    return {"id": f"heldout:{i}", "params": {"q": "x"}, "attempts": 1, "first_attempt_unstable": False,
            "final": {"state": "DIFF", "tier": "served", "diffs": ["body: keys ['boosts'] differ"],
                      "max_abs_delta": 0.0, "ranked_equal": {"served": served, "deep": deep},
                      "envelope_paths": list(paths)}}


def _report(world, sha, rows):
    r = copy.deepcopy(world["report"])
    r["release"] = {"base_sha": BASE_SHA, "head_sha": sha}
    r["verdict"] = "FAIL"
    n = r["requests"]
    r["counts"] = {"EQUAL": n - len(rows), "DIFF": len(rows), "UNSTABLE": 0, "ERROR": 0}
    r["not_equal"] = rows
    return r


def _run(tmp_path, world, report, sha, repo_dir, extra):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(report))
    return cli.main(["gate", "serve-parity", "check", "--report", str(p), "--base-sha", BASE_SHA,
                     "--head-sha", sha, "--base-server-dir", str(world["base"] / "cytir" / "serve"),
                     "--head-server-dir", str(world["head"] / "cytir" / "serve"),
                     "--repo-dir", str(repo_dir), *extra])


ENV = ["--envelope-change", "T2679", "--envelope-fields", FIELDS]


def test_accepted_when_only_declared_fields_differ_and_ranking_is_equal_in_both_tiers(
        world, repo, tmp_path, capsys):
    rd, sha = repo
    rep = _report(world, sha, [_row(1), _row(2, paths=("filters.note", "notes[]"))])
    assert _run(tmp_path, world, rep, sha, rd, ENV) == 0
    out = capsys.readouterr().out
    assert "ENVELOPE CHANGE T2679 accepted" in out and "heldout:1 | True/True | boosts.country" in out
    # the plain mode still refuses the same report
    assert _run(tmp_path, world, rep, sha, rd, []) == 1


def test_refused_for_an_undeclared_path(world, repo, tmp_path, capsys):
    rd, sha = repo
    rep = _report(world, sha, [_row(1, paths=("boosts.country", "boosts.region"))])
    assert _run(tmp_path, world, rep, sha, rd, ENV) == 1
    assert "undeclared envelope path(s) ['boosts.region']" in capsys.readouterr().err


@pytest.mark.parametrize("kw", [{"served": False}, {"deep": False}, {"deep": None}])
def test_refused_for_a_ranked_change_under_a_declared_field(world, repo, tmp_path, capsys, kw):
    rd, sha = repo
    rep = _report(world, sha, [_row(1, **kw)])
    assert _run(tmp_path, world, rep, sha, rd, ENV) == 1
    assert "ranked" in capsys.readouterr().err


def test_refused_for_a_report_that_predates_the_schema(world, repo, tmp_path, capsys):
    rd, sha = repo
    rep = _report(world, sha, [_row(1)])
    del rep["not_equal"][0]["final"]["ranked_equal"]
    del rep["not_equal"][0]["final"]["envelope_paths"]
    assert _run(tmp_path, world, rep, sha, rd, ENV) == 1
    assert "predates the ranked_equal schema" in capsys.readouterr().err


def test_refused_when_the_ref_is_not_in_plan_or_decisions_at_the_sha(world, repo, tmp_path, capsys):
    rd, sha = repo
    rep = _report(world, sha, [_row(1)])
    assert _run(tmp_path, world, rep, sha, rd, ["--envelope-change", "T9999",
                                                "--envelope-fields", FIELDS]) == 1
    assert "'T9999' is in neither plan.md nor decisions.md" in capsys.readouterr().err


def test_refused_for_unstable_error_or_a_dirty_null(world, repo, tmp_path):
    rd, sha = repo
    for key in ("UNSTABLE", "ERROR"):
        rep = _report(world, sha, [_row(1)])
        rep["counts"][key] = 1
        rep["counts"]["EQUAL"] -= 1
        assert _run(tmp_path, world, rep, sha, rd, ENV) == 1
    rep = _report(world, sha, [_row(1)])
    rep["null_pair"]["unstable_after_retries"] = 1
    assert _run(tmp_path, world, rep, sha, rd, ENV) == 1


def test_both_modes_together_are_refused(world, repo, tmp_path, capsys):
    rd, sha = repo
    rep = _report(world, sha, [_row(1)])
    both = ENV + ["--ranking-change", "T2679", "--max-diff", "5", "--null-report", str(tmp_path / "x.json")]
    assert _run(tmp_path, world, rep, sha, rd, both) == 2
    assert "two modes" in capsys.readouterr().err


def test_fields_are_required_with_the_ref(world, repo, tmp_path):
    rd, sha = repo
    rep = _report(world, sha, [_row(1)])
    assert _run(tmp_path, world, rep, sha, rd, ["--envelope-change", "T2679"]) == 2


# ------------------------------------------------------------------ review 2680 F1 / F3

_DEEP = {"score_debug": [{"id": "1", "s": 0.9}], "rankhead_v2_dump": [],
         "trace": {"final_ranked": [{"id": "1"}], "candidate_urls": {"1": "a", "2": "b"}, "calls": 3}}


def test_a_key_present_in_one_deep_trace_only_is_a_ranking_difference():
    body = {"results": RES}
    grown = copy.deepcopy(_DEEP)
    grown["trace"]["candidate_urls"]["3"] = "c"
    c = sp.classify(_seq(body, body, _DEEP, grown), 1e-6)
    assert c["state"] == "DIFF" and c["ranked_equal"]["deep"] is False and c["envelope_paths"] == []
    dropped = copy.deepcopy(_DEEP)
    del dropped["trace"]["final_ranked"]
    c = sp.classify(_seq(body, body, _DEEP, dropped), 1e-6)
    assert c["state"] == "DIFF" and c["ranked_equal"]["deep"] is False


def test_declared_fields_starting_deep_are_refused(world, repo, tmp_path, capsys):
    rd, sha = repo
    rep = _report(world, sha, [_row(1)])
    assert _run(tmp_path, world, rep, sha, rd, ["--envelope-change", "T2679",
                                                "--envelope-fields", "boosts.country,deep.trace.x"]) == 2
    assert "deep." in capsys.readouterr().err


def test_fields_match_exact_paths_a_bare_parent_does_not_cover_its_children(world, repo, tmp_path,
                                                                           capsys):
    rd, sha = repo
    rep = _report(world, sha, [_row(1, paths=("filters.note",))])
    assert _run(tmp_path, world, rep, sha, rd, ["--envelope-change", "T2679",
                                                "--envelope-fields", "filters"]) == 1
    assert "undeclared envelope path(s) ['filters.note']" in capsys.readouterr().err


# ------------------------------------------------------------------ task 2712: results[].<field>

def _ten():
    return [{"id": f"p{i}", "url": f"u{i}", "score": round(1 - i / 20, 4), "title": f"t{i}"}
            for i in range(10)]


def _with_read_url(rows, idx=(2, 7)):
    out = copy.deepcopy(rows)
    for i in idx:
        out[i]["read_url"] = f"https://r/{i}"
    return out


def _row_of(final):
    return {"id": "h:1", "params": {}, "attempts": 1, "first_attempt_unstable": False, "final": final}


def test_a_declared_per_result_field_is_stripped_and_counted():
    a = {"results": _ten(), "q": "x"}
    b = {"results": _with_read_url(_ten()), "q": "x"}
    c = sp.classify(_seq(a, b), 1e-6)
    assert c["state"] == "DIFF"
    assert c["ranked_equal"]["served"] is False and c["ranked_equal"]["served_stripped"] is True
    assert c["envelope_paths"] == ["results[].read_url"]
    assert c["result_field_counts"] == {"read_url": {"base": 0, "head": 2}}


def test_check_accepts_the_declared_per_result_field(world, repo, tmp_path):
    rd, sha = repo
    final = sp.classify(_seq({"results": _ten()}, {"results": _with_read_url(_ten())},
                             _DEEP, copy.deepcopy(_DEEP)), 1e-6)
    rep = _report(world, sha, [_row_of(final)])
    args = ["--envelope-change", "T2679", "--envelope-fields", "results[].read_url"]
    assert _run(tmp_path, world, rep, sha, rd, args) == 0


def test_undeclared_per_result_field_stays_not_equal(world, repo, tmp_path, capsys):
    final = sp.classify(_seq({"results": _ten()}, {"results": _with_read_url(_ten())}), 1e-6)
    assert final["state"] == "DIFF" and final["ranked_equal"]["served"] is False
    rd, sha = repo
    rep = _report(world, sha, [_row_of(final)])
    assert _run(tmp_path, world, rep, sha, rd, ENV) == 1
    assert "undeclared envelope path(s) ['results[].read_url']" in capsys.readouterr().err


def test_a_changed_url_beside_the_declared_field_is_refused(world, repo, tmp_path, capsys):
    rows = _with_read_url(_ten())
    rows[4]["url"] = "other"
    final = sp.classify(_seq({"results": _ten()}, {"results": rows}, _DEEP, copy.deepcopy(_DEEP)), 1e-6)
    assert final["ranked_equal"]["served_stripped"] is False
    rd, sha = repo
    rep = _report(world, sha, [_row_of(final)])
    args = ["--envelope-change", "T2679", "--envelope-fields", "results[].read_url"]
    assert _run(tmp_path, world, rep, sha, rd, args) == 1
    assert "the served ranked list differs" in capsys.readouterr().err


def test_a_reordered_or_shortened_list_is_not_stripped_equal():
    a = {"results": _ten()}
    c = sp.classify(_seq(a, {"results": list(reversed(_with_read_url(_ten())))}), 1e-6)
    assert c["ranked_equal"]["served_stripped"] is False
    c = sp.classify(_seq(a, {"results": _with_read_url(_ten())[:9]}), 1e-6)
    assert c["ranked_equal"]["served_stripped"] is False


@pytest.mark.parametrize("field,msg", [
    ("results[].url", "ranking-bearing"), ("results[].score", "ranking-bearing"),
    ("results[].a.b", "one level"), ("results[].*", "one level"), ("results[].", "one level"),
    ("results", "declared only as"), ("results.url", "declared only as")])
def test_per_result_paths_refused_at_parsing(world, repo, tmp_path, capsys, field, msg):
    rd, sha = repo
    rep = _report(world, sha, [_row(1)])
    args = ["--envelope-change", "T2679", "--envelope-fields", f"results[].read_url,{field}"]
    assert _run(tmp_path, world, rep, sha, rd, args) == 2
    assert msg in capsys.readouterr().err


# ------------------------------------------------------------------ task 2712 review round

READ_URL = ["--envelope-change", "T2679", "--envelope-fields", "results[].read_url"]


def _check_final(world, repo, tmp_path, base_rows, head_rows, args=READ_URL):
    final = sp.classify(_seq({"results": base_rows}, {"results": head_rows}, _DEEP, copy.deepcopy(_DEEP)),
                        1e-6)
    rd, sha = repo
    return final, _run(tmp_path, world, _report(world, sha, [_row_of(final)]), sha, rd, args)


@pytest.mark.parametrize("key", sorted(sp.RANKING_RESULT_KEYS))
def test_every_key_respond_emits_today_cannot_be_declared(world, repo, tmp_path, capsys, key):
    rd, sha = repo
    args = ["--envelope-change", "T2679", "--envelope-fields", f"results[].{key}"]
    assert _run(tmp_path, world, _report(world, sha, [_row(1)]), sha, rd, args) == 2
    assert "ranking-bearing" in capsys.readouterr().err


@pytest.mark.parametrize("key,val", [("score", 0.123), ("anchor", "x"), ("snippet", "x"),
                                     ("title", "x"), ("trust_tier", "Z")])
def test_a_changed_existing_key_beside_the_declared_field_is_refused(world, repo, tmp_path, capsys,
                                                                    key, val):
    rows = _with_read_url(_ten())
    rows[3][key] = val
    _, rc = _check_final(world, repo, tmp_path, _ten(), rows)
    assert rc == 1 and "the served ranked list differs" in capsys.readouterr().err


def test_a_nested_change_in_an_existing_key_is_refused(world, repo, tmp_path):
    a = _ten()
    a[1]["confidence"] = {"low": False}
    b = _with_read_url(copy.deepcopy(a))
    b[1]["confidence"]["low"] = True
    assert _check_final(world, repo, tmp_path, a, b)[1] == 1


def test_an_undeclared_extra_key_on_a_result_is_refused(world, repo, tmp_path, capsys):
    rows = _with_read_url(_ten())
    rows[0]["extra"] = 1
    assert _check_final(world, repo, tmp_path, _ten(), rows)[1] == 1
    assert "undeclared envelope path(s) ['results[].extra']" in capsys.readouterr().err


def test_removal_of_the_declared_field_is_refused(world, repo, tmp_path, capsys):
    base = _with_read_url(_ten(), (0, 1, 2))
    final, rc = _check_final(world, repo, tmp_path, base, _ten())
    assert final["result_field_counts"] == {"read_url": {"base": 3, "head": 0}}
    assert rc == 1 and "is additive but 3 base result(s) carry it and only 0 head" in capsys.readouterr().err


def test_a_value_change_with_both_arms_carrying_the_field_is_allowed(world, repo, tmp_path):
    base = _with_read_url(_ten())
    head = copy.deepcopy(base)
    head[2]["read_url"] = "https://r/other"
    assert _check_final(world, repo, tmp_path, base, head)[1] == 0


def test_a_pre_change_report_row_gets_the_hint(world, repo, tmp_path, capsys):
    rd, sha = repo
    row = _row(1, served=False, paths=())
    assert _run(tmp_path, world, _report(world, sha, [row]), sha, rd, ENV) == 1
    assert "report predates per-result envelope support" in capsys.readouterr().err
