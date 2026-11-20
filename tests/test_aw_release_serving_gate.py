"""`cyt gate serve-parity check` and its callers `deploy/scripts/release-serving.sh` and
`deploy/scripts/rebuild-serving-image.sh` (task 792, refactor slice 8 step 5; fix round 799 after the
review of task 796; specs/s10-serve-split.md section 3: "a check the operator must remember to run
is not a refusal").

Three layers, all on real code:

1. The verb. A REAL `run_parity` on fake server trees (the harness of test_aw_serve_parity) writes
   `parity.json` with the (prod sha, new sha) pair; `check` is run on it through the real `aw`
   dispatcher and must exit 0 for the PASS and 1 for each way the report can fail to license the
   release. BOTH trees are bound by digest (the release's and the one the box serves now).
2. The ranking-change mode (`--ranking-change REF --max-diff N --null-report R2`): accepts a FAIL
   report only when every one of its conditions holds; each condition is flipped alone below.
3. The callers. The REAL scripts run in a throwaway git repo. `ssh` is a stub that runs the command
   locally (the "box" is a temp dir named by AW_REMOTE_ROOT), `curl` a stub that answers the
   parity query, `compose.sh` a stub that only records that it was reached. So the scripts' own
   preflight, staging, change set, gate, history and render are the real ones; only the network
   and Docker are replaced.
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

import test_aw_serve_parity as TP
from cytir import cli
from cytir.gates import serve_parity as sp

APP = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = APP / "deploy" / "scripts"
SCRIPT = SCRIPTS / "release-serving.sh"
REBUILD = SCRIPTS / "rebuild-serving-image.sh"
BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
#: `pipeline/` is the private build tree and never ships (data/public-cut.v1.json "never"): its
#: absence is the cut marker. In the private tree a missing script is a failure, not a skip.
IN_PUBLIC_CUT = not (APP / "pipeline").is_dir()


def _srv(tree):
    return tree / "cytir" / "serve"


def _report(tmp_path, base, head, *, base_sha=BASE_SHA, head_sha=HEAD_SHA, deep=True,
            seed_probe=False, out="out", declared_env=None):
    rep = sp.run_parity(TP.requests(), base_tree=base, head_tree=head, collection="twin_x",
                        qdrant_url=TP.free_url(), vec_cache=TP.write_cache(tmp_path / f"{out}-v.json"),
                        out_dir=tmp_path / out, shards=2, deep=deep, seed_probe=seed_probe,
                        guard=lambda url, coll: None,
                        declared_env=declared_env or {"AW_FAKE_KNOB": "1"},
                        base_sha=base_sha, head_sha=head_sha, say=lambda m: None)
    return tmp_path / out / "parity.json", rep


def _check(report, *, base=BASE_SHA, head=HEAD_SHA, base_dir, head_dir, extra=()):
    argv = ["gate", "serve-parity", "check", "--report", str(report),
            "--base-sha", base, "--head-sha", head,
            "--base-server-dir", str(base_dir), "--head-server-dir", str(head_dir), *extra]
    return cli.main(argv)


@pytest.fixture(scope="module")
def world_trees(tmp_path_factory):
    """base, an output-neutral head ("noop": different source, same answers), a head that swaps
    one answer ("swap"), and the real reports between them."""
    t = tmp_path_factory.mktemp("trees")
    base = TP.make_tree(t, "base")
    noop = TP.make_tree(t, "noop", tweak="noop")
    swap = TP.make_tree(t, "swap", tweak="swap")
    _, passed = _report(t, base, noop, out="pass")
    _, ranked = _report(t, base, swap, out="ranked", deep=False)
    _, r2 = _report(t, swap, swap, base_sha=HEAD_SHA, head_sha=HEAD_SHA, out="r2", deep=False,
                    seed_probe=True)
    _, selfself = _report(t, noop, noop, out="selfself", deep=False)
    assert passed["verdict"] == "PASS" and passed["trees"]["same_server_code"] is False
    assert ranked["verdict"] == "FAIL" and ranked["counts"]["DIFF"] >= 1
    assert r2["verdict"] == "PASS" and r2["seed_probe"]["sensitive"] == 0
    assert selfself["verdict"] == "PASS" and selfself["trees"]["same_server_code"] is True
    return {"tmp": t, "base": base, "noop": noop, "swap": swap, "pass": passed, "ranked": ranked,
            "r2": r2, "selfself": selfself}


def _w(tmp_path, name, rep):
    p = tmp_path / name
    p.write_text(json.dumps(rep))
    return p


# ------------------------------------------------------------------ the verb: plain mode

def test_run_records_the_sha_pair_and_check_accepts_that_pass(world_trees, tmp_path):
    w = world_trees
    assert w["pass"]["release"] == {"base_sha": BASE_SHA, "head_sha": HEAD_SHA}
    assert _check(_w(tmp_path, "r.json", w["pass"]), base_dir=_srv(w["base"]),
                  head_dir=_srv(w["noop"])) == 0


def test_check_refuses_a_fail_verdict(world_trees, tmp_path, capsys):
    w = world_trees
    assert _check(_w(tmp_path, "r.json", w["ranked"]), base_dir=_srv(w["base"]),
                  head_dir=_srv(w["swap"])) == 1
    err = capsys.readouterr().err
    assert "verdict is 'FAIL'" in err and "not every request is EQUAL" in err


@pytest.mark.parametrize("kw", [{"base": "c" * 40}, {"head": "d" * 40}], ids=["other-base", "other-head"])
def test_check_refuses_a_report_for_another_sha_pair(world_trees, tmp_path, kw):
    w = world_trees
    assert _check(_w(tmp_path, "r.json", w["pass"]), base_dir=_srv(w["base"]),
                  head_dir=_srv(w["noop"]), **kw) == 1


def test_check_refuses_a_report_without_a_sha_pair(world_trees, tmp_path):
    w = world_trees
    rep = copy.deepcopy(w["pass"])
    rep["release"] = {"base_sha": None, "head_sha": None}
    assert _check(_w(tmp_path, "r.json", rep), base_dir=_srv(w["base"]), head_dir=_srv(w["noop"])) == 1


def test_check_refuses_a_head_tree_that_is_not_the_release(world_trees, tmp_path):
    w = world_trees
    assert _check(_w(tmp_path, "r.json", w["pass"]), base_dir=_srv(w["base"]),
                  head_dir=_srv(w["swap"])) == 1


def test_check_refuses_a_base_tree_that_is_not_the_one_the_box_serves(world_trees, tmp_path, capsys):
    """M2 (review 796): the base arm must have run on the code the box serves NOW."""
    w = world_trees
    assert _check(_w(tmp_path, "r.json", w["pass"]), base_dir=_srv(w["swap"]),
                  head_dir=_srv(w["noop"])) == 1
    assert "not the serving package the box serves now" in capsys.readouterr().err


def test_check_refuses_a_report_that_compares_the_release_with_itself(world_trees, tmp_path, capsys):
    """M2 (repro R2): base = head = the release tree, labelled with the prod sha. The digests of
    BOTH real trees are given (base = the release itself, as the labelled-prod arm), so only the
    `same_server_code` binding can refuse it."""
    w = world_trees
    rep = w["selfself"]
    assert _check(_w(tmp_path, "r.json", rep), base_dir=_srv(w["noop"]), head_dir=_srv(w["noop"])) == 1
    assert "compares a tree with itself" in capsys.readouterr().err


def test_digest_covers_the_json_tables_not_only_the_python(tmp_path):
    """S2 (review 796): fusion_weights.json, qhead.json, host lists are rendered with the code."""
    a = TP.make_tree(tmp_path, "a")
    d0 = sp._source_digest(_srv(a))
    (_srv(a) / "fusion_weights.json").write_text('{"w": 1}')
    d1 = sp._source_digest(_srv(a))
    (_srv(a) / "fusion_weights.json").write_text('{"w": 2}')
    d2 = sp._source_digest(_srv(a))
    assert len({d0, d1, d2}) == 3
    (_srv(a) / "__pycache__").mkdir()
    (_srv(a) / "__pycache__" / "x.json").write_text("{}")
    assert sp._source_digest(_srv(a)) == d2, "__pycache__ is not part of the release"


def test_check_refuses_a_head_whose_json_table_differs(world_trees, tmp_path):
    w = world_trees
    other = shutil.copytree(_srv(w["noop"]), tmp_path / "other")
    (other / "fusion_weights.json").write_text('{"w": 9}')
    assert _check(_w(tmp_path, "r.json", w["pass"]), base_dir=_srv(w["base"]), head_dir=other) == 1


def test_check_refuses_an_unreadable_or_foreign_report(world_trees, tmp_path):
    w = world_trees
    dirs = dict(base_dir=_srv(w["base"]), head_dir=_srv(w["noop"]))
    assert _check(tmp_path / "missing.json", **dirs) == 2
    foreign = tmp_path / "x.json"
    foreign.write_text(json.dumps({"schema": "other", "verdict": "PASS"}))
    assert _check(foreign, **dirs) == 1


def test_run_records_a_data_file_difference_instead_of_exiting(tmp_path):
    """M3: a ranking change that ships a changed served table must still get a report."""
    env = {"AW_FAKE_KNOB": "1", "AW_FAKE_TABLE": "/app/cytir/serve/table.json"}
    base = TP.make_tree(tmp_path, "base")
    head = TP.make_tree(tmp_path, "head")
    (_srv(base) / "table.json").write_text('{"w": 1}')
    (_srv(head) / "table.json").write_text('{"w": 2}')
    path, rep = _report(tmp_path, base, head, deep=False, declared_env=env)
    assert rep["verdict"] == "FAIL"
    assert set(rep["data_file_diffs"]) == {"AW_FAKE_TABLE"}
    assert rep["data_file_diffs"]["AW_FAKE_TABLE"]["base"] != rep["data_file_diffs"]["AW_FAKE_TABLE"]["head"]
    assert any("AW_FAKE_TABLE" in d for d in rep["fingerprint_diffs"])
    assert json.loads(path.read_text())["data_file_diffs"] == rep["data_file_diffs"]


# ------------------------------------------------------------------ the verb: ranking-change mode

def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True,
                          env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}).stdout.strip()


@pytest.fixture
def ruled_repo(tmp_path):
    """A repo whose plan.md at HEAD names the ruling T11 (committed), whose working tree names T77
    (uncommitted), and whose decisions.md at HEAD names ADR-777."""
    r = tmp_path / "repo"
    r.mkdir()
    (r / "plan.md").write_text("# plan\n- T11 tie-break release, ruled by the board\n")
    (r / "decisions.md").write_text("# decisions\n## ADR-777 something\n")
    _git(r, "init", "-q")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "release")
    sha = _git(r, "rev-parse", "HEAD")
    (r / "plan.md").write_text("# plan\n- T11 tie-break release, ruled by the board\n- T77 only in the checkout\n")
    return r, sha


def _good_ranking(world_trees, repo, sha):
    w = world_trees
    main = copy.deepcopy(w["ranked"])
    main["release"] = {"base_sha": BASE_SHA, "head_sha": sha}
    r2 = copy.deepcopy(w["r2"])
    r2["release"] = {"base_sha": sha, "head_sha": sha}
    return {"main": main, "r2": r2, "base": BASE_SHA, "head": sha, "base_dir": _srv(w["base"]),
            "head_dir": _srv(w["swap"]), "ref": "T11", "max_diff": main["counts"]["DIFF"],
            "repo_dir": repo}


def _run_ranking(tmp_path, a):
    return _check(_w(tmp_path, "main.json", a["main"]), base=a["base"], head=a["head"],
                  base_dir=a["base_dir"], head_dir=a["head_dir"],
                  extra=["--ranking-change", a["ref"], "--max-diff", str(a["max_diff"]),
                         "--null-report", str(_w(tmp_path, "r2.json", a["r2"])),
                         "--repo-dir", str(a["repo_dir"])])


def test_ranking_change_accepts_a_fail_report_when_every_condition_holds(world_trees, ruled_repo,
                                                                         tmp_path, capsys):
    repo, sha = ruled_repo
    a = _good_ranking(world_trees, repo, sha)
    assert a["main"]["verdict"] == "FAIL"
    assert _run_ranking(tmp_path, a) == 0
    out = capsys.readouterr().out
    assert "RANKING CHANGE T11 accepted" in out and "differing requests" in out
    # and the same report is refused without the declared mode
    assert _check(_w(tmp_path, "m.json", a["main"]), base=a["base"], head=a["head"],
                  base_dir=a["base_dir"], head_dir=a["head_dir"]) == 1


def test_ranking_change_finds_the_ruling_in_decisions_md_at_the_release(world_trees, ruled_repo, tmp_path):
    repo, sha = ruled_repo
    a = _good_ranking(world_trees, repo, sha)
    a["ref"] = "ADR-777"
    assert _run_ranking(tmp_path, a) == 0


def _m_error(a):
    a["main"]["counts"].update(EQUAL=a["main"]["counts"]["EQUAL"] - 1, ERROR=1)


def _m_unstable(a):
    a["main"]["counts"].update(EQUAL=a["main"]["counts"]["EQUAL"] - 1, UNSTABLE=1)


def _m_config(a):
    a["main"]["fingerprint_diffs"] = ["effective_config.AW_X: 1 != 2"]


def _m_data_file(a):
    a["main"]["data_file_diffs"] = {"AW_FUSION_WEIGHTS": {"base": "1", "head": "2"}}


def _m_null(a):
    a["main"]["null_pair"]["unstable_after_retries"] = 1


def _m_too_many_diffs(a):
    a["max_diff"] = a["main"]["counts"]["DIFF"] - 1


def _m_base_digest(a):
    a["base_dir"] = a["head_dir"]


def _m_head_digest(a):
    a["head_dir"] = a["base_dir"]


def _m_same_code(a):
    a["main"]["trees"]["same_server_code"] = True


def _m_r2_not_pass(a):
    a["r2"]["verdict"] = "FAIL"


def _m_r2_sensitive(a):
    a["r2"]["seed_probe"]["sensitive"] = 1


def _m_r2_no_probe(a):
    a["r2"]["seed_probe"] = None


def _m_r2_not_head_vs_head(a):
    a["r2"]["release"]["base_sha"] = BASE_SHA


def _m_r2_other_tree(a):
    for arm in ("base", "head"):
        a["r2"]["fingerprints"][arm]["server_source_sha256"] = "0" * 64


def _m_r2_missing(a):
    a["r2"] = {"schema": "other"}


def _m_ref_not_in_the_tree(a):
    a["ref"] = "T99"


def _m_ref_only_in_the_checkout(a):
    a["ref"] = "T77"


@pytest.mark.parametrize("flip", [
    _m_error, _m_unstable, _m_config, _m_data_file, _m_null, _m_too_many_diffs, _m_base_digest,
    _m_head_digest, _m_same_code, _m_r2_not_pass, _m_r2_sensitive, _m_r2_no_probe,
    _m_r2_not_head_vs_head, _m_r2_other_tree, _m_r2_missing, _m_ref_not_in_the_tree,
    _m_ref_only_in_the_checkout], ids=lambda f: f.__name__[3:])
def test_ranking_change_refuses_when_one_condition_is_flipped(world_trees, ruled_repo, tmp_path, flip):
    repo, sha = ruled_repo
    a = _good_ranking(world_trees, repo, sha)
    assert _run_ranking(tmp_path, a) == 0, "the unflipped inputs must be accepted"
    flip(a)
    assert _run_ranking(tmp_path, a) == 1


def test_ranking_change_flags_need_all_three(world_trees, ruled_repo, tmp_path):
    repo, sha = ruled_repo
    a = _good_ranking(world_trees, repo, sha)
    rep = _w(tmp_path, "m.json", a["main"])
    for extra in (["--ranking-change", "T11"], ["--ranking-change", "T11", "--max-diff", "5"],
                  ["--max-diff", "5", "--null-report", str(rep)]):
        assert _check(rep, base=a["base"], head=a["head"], base_dir=a["base_dir"],
                      head_dir=a["head_dir"], extra=extra) == 2


def test_the_758_tiebreak_report_passes_the_ranking_mode_and_fails_without_it(world_trees, ruled_repo,
                                                                              tmp_path, capsys):
    """The real prod-vs-head report of task 758 (2,080 EQUAL / 192 DIFF / 0 UNSTABLE / 0 ERROR,
    verdict FAIL by design): refused as a plain release, accepted as a declared ranking change
    with N = 200 and a head-vs-head R2. Its tree digests are replaced by the fake trees' (the
    digest binding is proven above); everything else is verbatim."""
    repo, sha = ruled_repo
    a = _good_ranking(world_trees, repo, sha)
    real = json.loads((APP / "tests" / "testdata" / "task758_serve_parity_main_vs_head.json").read_text())
    assert real["verdict"] == "FAIL" and real["counts"] == {"EQUAL": 2080, "DIFF": 192, "UNSTABLE": 0, "ERROR": 0}
    real["release"] = {"base_sha": BASE_SHA, "head_sha": sha}
    real["collection"] = a["r2"]["collection"]
    real["fingerprints"]["base"]["server_source_sha256"] = sp._source_digest(a["base_dir"])
    real["fingerprints"]["head"]["server_source_sha256"] = sp._source_digest(a["head_dir"])
    a["main"], a["max_diff"] = real, 200
    assert _run_ranking(tmp_path, a) == 0
    assert "192 DIFF of 2272" in capsys.readouterr().out
    assert _check(_w(tmp_path, "m.json", real), base=a["base"], head=a["head"],
                  base_dir=a["base_dir"], head_dir=a["head_dir"]) == 1
    a["max_diff"] = 191
    assert _run_ranking(tmp_path, a) == 1


# ------------------------------------------------------------------ the scripts

def _stub(bin_dir: pathlib.Path, name: str, body: str) -> None:
    p = bin_dir / name
    p.write_text("#!/usr/bin/env bash\n" + body)
    p.chmod(0o755)


def _age(box):
    """rsync skips a same-size file with the same mtime (two commits in one second): age the box."""
    for f in box.rglob("*"):
        if f.is_file():
            os.utime(f, (1, 1))


@pytest.fixture
def world(tmp_path):
    """A throwaway repo with the real scripts, a 'box' dir, and stubbed ssh/curl/compose."""
    if IN_PUBLIC_CUT:
        pytest.skip("deploy/scripts/* is operator tooling: not in the public cut")
    assert SCRIPT.is_file(), "release-serving.sh is missing from the private tree"
    repo = tmp_path / "repo"
    app = repo / "app"
    (app / "deploy" / "scripts").mkdir(parents=True)
    for f in (SCRIPT, REBUILD, SCRIPTS / "serving-gate.sh", SCRIPTS / "render-serving-tree.sh"):
        shutil.copy(f, app / "deploy" / "scripts" / f.name)
    (app / "cytir").mkdir()                      # the verb under test, from the real tree; `serve` is the fixture's own
    for child in (APP / "cytir").iterdir():
        if child.name not in ("serve", "__pycache__"):
            (app / "cytir" / child.name).symlink_to(child)
    (app / "deploy" / "scripts" / "compose.sh").write_text(
        '#!/usr/bin/env bash\necho "compose.sh $*" >> "$COMPOSE_LOG"\nexit 7\n')
    (app / "deploy" / "scripts" / "compose.sh").chmod(0o755)
    srv = app / "cytir" / "serve"
    srv.mkdir(parents=True)
    (app / "webapp").mkdir()
    (srv / "app.py").write_text("V = 1\n")
    (app / "webapp" / "index.html").write_text("<html>1</html>")
    (app / "webapp" / "privacy.html").write_text("<html>p1</html>")
    # The scripts run python from this tree; bytecode written under `cytir/serve` must never be committed
    # (a later rewrite of it is a local change that blocks `git checkout <earlier sha>`).
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "prod")
    prod_sha = _git(repo, "rev-parse", "HEAD")
    box = tmp_path / "box"
    for sub in ("cytir/serve", "webapp"):
        shutil.copytree(app / sub, box / sub)
    (box / ".serving-release").write_text(prod_sha + "\n")
    _age(box)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir, "ssh", 'if [ "$1" = "-p" ]; then shift 2; fi\nshift\nexec bash -c "$*"\n')
    _stub(bin_dir, "scp", 'echo "scp $*" >> "$COMPOSE_LOG"\n')
    _stub(bin_dir, "curl", 'echo \'{"results":[{"url":"https://x/1"}]}\'\n')
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "AW_REMOTE_ROOT": str(box),
           "AW_WRITER_LOCK": str(tmp_path / "no-lock"), "COMPOSE_LOG": str(tmp_path / "compose.log"),
           "AW_SERVE_PARITY_PYTHON": sys.executable, "AW_DEPLOY_HOST": "box",
           "AW_HEALTH_HOST": "https://stub.invalid"}
    for k in ("AW_PARITY_REPORT", "AW_PARITY_NULL_REPORT", "AW_RANKING_CHANGE", "AW_RANKING_MAX_DIFF",
              "AW_ENVELOPE_CHANGE", "AW_ENVELOPE_FIELDS"):
        env.pop(k, None)
    return repo, app, box, env, prod_sha, tmp_path


def _release(repo, app, file, text, *, also=()) -> str:
    (app / file).write_text(text)
    for f, t in also:
        (app / f).write_text(t)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", f"change {file}")
    return _git(repo, "rev-parse", "HEAD")


def _run_script(app, sha, env, *flags, cwd=None, **extra):
    return subprocess.run(["bash", str(app / "deploy" / "scripts" / "release-serving.sh"), sha, *flags],
                          capture_output=True, text=True, cwd=cwd,
                          env={**env, **extra, "AW_EXPECT_CHANGES": "*"}, timeout=120)


def _bound(rep, *, box, app, base_sha, head_sha):
    """Bind a fake-tree report to the box's live code (base) and the release (head)."""
    rep = copy.deepcopy(rep)
    rep["release"] = {"base_sha": base_sha, "head_sha": head_sha}
    rep["fingerprints"]["base"]["server_source_sha256"] = sp._source_digest(box / "cytir" / "serve")
    rep["fingerprints"]["head"]["server_source_sha256"] = sp._source_digest(app / "cytir" / "serve")
    return rep


def _pass_report(world_trees, w):
    repo, app, box, env, prod_sha, tmp = w
    return lambda sha, name="parity.json", base_sha=None: _w(
        tmp, name, _bound(world_trees["pass"], box=box, app=app, base_sha=base_sha or prod_sha,
                          head_sha=sha))


def test_script_refuses_a_serving_code_release_without_a_report_and_renders_nothing(world):
    repo, app, box, env, prod_sha, tmp = world
    sha = _release(repo, app, "cytir/serve/app.py", "V = 2\n")
    r = _run_script(app, sha, env)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "AW_PARITY_REPORT is not set" in r.stderr
    assert (box / "cytir/serve/app.py").read_text() == "V = 1\n", "refusal must precede the render"
    assert not (tmp / "compose.log").exists()


def test_script_refuses_a_report_that_does_not_cover_the_pair(world, world_trees):
    repo, app, box, env, prod_sha, tmp = world
    sha = _release(repo, app, "cytir/serve/app.py", "V = 2\n")
    path = _w(tmp, "p.json", _bound(world_trees["pass"], box=box, app=app, base_sha=prod_sha,
                                    head_sha="e" * 40))   # wrong head sha
    r = _run_script(app, sha, env, AW_PARITY_REPORT=str(path))
    assert r.returncode == 1, r.stdout + r.stderr
    assert "NOT satisfied" in r.stderr and "parity gate is not satisfied" in r.stderr
    assert (box / "cytir/serve/app.py").read_text() == "V = 1\n"
    assert not (tmp / "compose.log").exists()


def test_script_with_a_pass_report_for_exactly_this_release_reaches_render_and_restart(world, world_trees):
    repo, app, box, env, prod_sha, tmp = world
    sha = _release(repo, app, "cytir/serve/app.py", "V = 2\n")
    path = _pass_report(world_trees, world)(sha)
    r = _run_script(app, sha, env, AW_PARITY_REPORT=str(path))
    out = r.stdout + r.stderr
    assert "serve-parity check: PASS report covers" in out, out
    assert (box / "cytir/serve/app.py").read_text() == "V = 2\n", out
    assert "compose.sh restart" in (tmp / "compose.log").read_text()
    assert r.returncode == 1 and "did not come back healthy" in out    # the compose stub fails on purpose
    # S4: the rollback it prints is the real one, in the real argument order
    assert "<previous SHA> --reuse" in out
    # M1: the box recorded what it served and what it serves now
    assert (box / ".serving-release.history").read_text().split() == [prod_sha, sha]


def test_script_binds_the_head_tree_of_the_report_to_the_release(world, world_trees):
    """S1 (review 796): dropping `--head-server-dir` from the script must be caught."""
    repo, app, box, env, prod_sha, tmp = world
    sha = _release(repo, app, "cytir/serve/app.py", "V = 2\n")
    rep = _bound(world_trees["pass"], box=box, app=app, base_sha=prod_sha, head_sha=sha)
    rep["fingerprints"]["head"]["server_source_sha256"] = "f" * 64      # a different head tree
    r = _run_script(app, sha, env, AW_PARITY_REPORT=str(_w(tmp, "p.json", rep)))
    assert r.returncode == 1 and "head tree is not the release" in r.stderr, r.stdout + r.stderr
    assert (box / "cytir/serve/app.py").read_text() == "V = 1\n"
    assert not (tmp / "compose.log").exists()


def test_script_binds_the_base_tree_of_the_report_to_the_code_the_box_serves(world, world_trees):
    """M2: the report's base arm must be the live code, so a drifted box is refused too (S9)."""
    repo, app, box, env, prod_sha, tmp = world
    sha = _release(repo, app, "cytir/serve/app.py", "V = 2\n")
    path = _pass_report(world_trees, world)(sha)
    (box / "cytir/serve/app.py").write_text("V = 'hand edit'\n")      # the box drifted
    r = _run_script(app, sha, env, AW_PARITY_REPORT=str(path))
    assert r.returncode == 1 and "not the serving package the box serves now" in r.stderr, r.stdout + r.stderr
    assert not (tmp / "compose.log").exists()


def test_script_refuses_a_self_vs_self_report_labelled_with_the_prod_sha(world, world_trees):
    """M2 (repro R2): the release compared with itself, passing as (prod -> release)."""
    repo, app, box, env, prod_sha, tmp = world
    sha = _release(repo, app, "cytir/serve/app.py", "V = 2\n")
    rep = _bound(world_trees["selfself"], box=box, app=app, base_sha=prod_sha, head_sha=sha)
    # even with the base digest forged to the box's, the arms are one tree
    assert rep["verdict"] == "PASS" and rep["trees"]["same_server_code"] is True
    r = _run_script(app, sha, env, AW_PARITY_REPORT=str(_w(tmp, "p.json", rep)))
    assert r.returncode == 1 and "compares a tree with itself" in r.stderr, r.stdout + r.stderr
    assert (box / "cytir/serve/app.py").read_text() == "V = 1\n"


def test_script_needs_no_report_for_a_webapp_only_release(world):
    repo, app, box, env, prod_sha, tmp = world
    sha = _release(repo, app, "webapp/privacy.html", "<html>p2</html>")
    r = _run_script(app, sha, env)
    out = r.stdout + r.stderr
    assert "no serving-code change, no report needed" in out, out
    assert (box / "webapp/privacy.html").read_text() == "<html>p2</html>"


def test_script_resolves_a_relative_report_against_the_callers_directory(world, world_trees):
    """S6 (review 796)."""
    repo, app, box, env, prod_sha, tmp = world
    sha = _release(repo, app, "cytir/serve/app.py", "V = 2\n")
    _pass_report(world_trees, world)(sha, "rel-parity.json")
    r = _run_script(app, sha, env, cwd=tmp, AW_PARITY_REPORT="rel-parity.json")
    assert "serve-parity check: PASS report covers" in r.stdout + r.stderr, r.stdout + r.stderr


# ----------------------------------------------- --reuse is a rollback, not a bypass (M1)

def test_script_reuse_refuses_a_sha_the_gate_just_refused(world):
    """M1 (repro R1): a release refused for want of a report is STAGED; `--reuse` must not render it."""
    repo, app, box, env, prod_sha, tmp = world
    sha = _release(repo, app, "cytir/serve/app.py", "V = 2\n")
    first = _run_script(app, sha, env)
    assert first.returncode == 1 and "AW_PARITY_REPORT is not set" in first.stderr
    assert (box / "releases" / sha).is_dir()                    # staged before the gate
    r = _run_script(app, sha, env, "--reuse")
    assert r.returncode == 1 and "never served" in r.stderr, r.stdout + r.stderr
    assert (box / "cytir/serve/app.py").read_text() == "V = 1\n"
    assert (box / ".serving-release").read_text().strip() == prod_sha
    assert not (tmp / "compose.log").exists()


def test_script_reuse_rolls_back_to_a_sha_the_box_served_and_says_so(world, world_trees):
    repo, app, box, env, prod_sha, tmp = world
    sha1 = _release(repo, app, "cytir/serve/app.py", "V = 2\n")
    r1 = _run_script(app, sha1, env, AW_PARITY_REPORT=str(_pass_report(world_trees, world)(sha1)))
    assert (box / "cytir/serve/app.py").read_text() == "V = 2\n", r1.stdout + r1.stderr
    _age(box)
    sha2 = _release(repo, app, "cytir/serve/app.py", "V = 3\n")
    r2 = _run_script(app, sha2, env, AW_PARITY_REPORT=str(_pass_report(world_trees, world)(sha2, "p2.json", sha1)))
    assert (box / "cytir/serve/app.py").read_text() == "V = 3\n", r2.stdout + r2.stderr
    assert (box / ".serving-release.history").read_text().split() == [prod_sha, sha1, sha2]
    _git(repo, "checkout", "-q", sha1)      # the script packages from a clean checkout of the sha
    back = _run_script(app, sha1, env, "--reuse")                 # no report: it is a rollback
    assert "parity gate NOT applied" in back.stdout, back.stdout + back.stderr
    assert (box / "cytir/serve/app.py").read_text() == "V = 2\n"
    assert (box / ".serving-release").read_text().strip() == sha1


def test_script_reuse_accepts_a_sha_the_operator_seeded_into_the_history(world):
    repo, app, box, env, prod_sha, tmp = world
    sha = _release(repo, app, "cytir/serve/app.py", "V = 2\n")
    _run_script(app, sha, env)                                   # stages it, refused
    (box / ".serving-release.history").write_text(sha + "\n")     # the documented one-off seed
    r = _run_script(app, sha, env, "--reuse")
    assert "parity gate NOT applied" in r.stdout, r.stdout + r.stderr
    assert (box / "cytir/serve/app.py").read_text() == "V = 2\n"


# ----------------------------------------------- the declared ranking change, end to end (M3)

def test_script_ranking_change_accepts_the_fail_report_and_refuses_it_undeclared(world, world_trees):
    repo, app, box, env, prod_sha, tmp = world
    plan = "# plan\n- T11 tie-break release ruled by the board\n"
    sha = _release(repo, app, "cytir/serve/app.py", "V = 2\n", also=[("plan.md", plan)])
    main = _bound(world_trees["ranked"], box=box, app=app, base_sha=prod_sha, head_sha=sha)
    r2 = copy.deepcopy(world_trees["r2"])
    r2["release"] = {"base_sha": sha, "head_sha": sha}
    r2["fingerprints"]["base"]["server_source_sha256"] = sp._source_digest(app / "cytir" / "serve")
    r2["fingerprints"]["head"]["server_source_sha256"] = sp._source_digest(app / "cytir" / "serve")
    rp, r2p = _w(tmp, "main.json", main), _w(tmp, "r2.json", r2)
    decl = dict(AW_PARITY_REPORT=str(rp), AW_PARITY_NULL_REPORT=str(r2p), AW_RANKING_CHANGE="T11",
                AW_RANKING_MAX_DIFF=str(main["counts"]["DIFF"]))
    undeclared = _run_script(app, sha, env, AW_PARITY_REPORT=str(rp))
    assert undeclared.returncode == 1 and "verdict is 'FAIL'" in undeclared.stderr
    assert (box / "cytir/serve/app.py").read_text() == "V = 1\n"
    no_n = _run_script(app, sha, env, **{**decl, "AW_RANKING_MAX_DIFF": ""})
    assert no_n.returncode == 1 and "AW_RANKING_MAX_DIFF" in no_n.stderr
    over = _run_script(app, sha, env, **{**decl, "AW_RANKING_CHANGE": "T99"})   # ruling not in plan.md at sha
    assert over.returncode == 1 and "neither plan.md nor decisions.md" in over.stderr
    assert (box / "cytir/serve/app.py").read_text() == "V = 1\n"
    ok = _run_script(app, sha, env, **decl)
    out = ok.stdout + ok.stderr
    assert "RANKING CHANGE T11 accepted" in out, out
    assert (box / "cytir/serve/app.py").read_text() == "V = 2\n"
    assert "ranking change declared: T11" in out


# ----------------------------------------------- the second door (S3)

def test_rebuild_script_applies_the_same_gate_before_it_builds_or_ships(world):
    repo, app, box, env, prod_sha, tmp = world
    sha = _release(repo, app, "cytir/serve/app.py", "V = 2\n")
    r = subprocess.run(["bash", str(app / "deploy" / "scripts" / "rebuild-serving-image.sh"), sha],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "AW_PARITY_REPORT is not set" in r.stderr and "nothing was built, shipped or rendered" in r.stderr
    assert not (tmp / "compose.log").exists(), "nothing may be shipped (scp) or restarted before the gate"
    assert (box / "cytir/serve/app.py").read_text() == "V = 1\n"


def test_rebuild_script_needs_no_report_when_serving_code_is_unchanged(world):
    repo, app, box, env, prod_sha, tmp = world
    sha = _release(repo, app, "webapp/privacy.html", "<html>p2</html>")
    r = subprocess.run(["bash", str(app / "deploy" / "scripts" / "rebuild-serving-image.sh"), sha],
                       capture_output=True, text=True, env=env, timeout=120)
    out = r.stdout + r.stderr
    assert "no serving-code change, no report needed" in out, out
    assert "nothing was built" not in out        # it went on past the gate (and stopped at the stubbed build host)


# ----------------------------------------------- the envelope-change door (task 2679, review 2680 F2)

def _envelope_report(world_trees, w, sha, paths=("boosts",)):
    repo, app, box, env, prod_sha, tmp = w
    rep = _bound(world_trees["pass"], box=box, app=app, base_sha=prod_sha, head_sha=sha)
    rep["verdict"] = "FAIL"
    rep["counts"] = {"EQUAL": rep["requests"] - 1, "DIFF": 1, "UNSTABLE": 0, "ERROR": 0}
    rep["not_equal"] = [{"id": "x:1", "params": {"q": "x"}, "attempts": 1,
                         "first_attempt_unstable": False,
                         "final": {"state": "DIFF", "tier": "served", "diffs": ["body: keys differ"],
                                   "max_abs_delta": 0.0, "ranked_equal": {"served": True, "deep": True},
                                   "envelope_paths": list(paths)}}]
    return _w(tmp, "env.json", rep)


def test_script_envelope_change_accepts_declared_and_refuses_the_rest(world, world_trees):
    repo, app, box, env, prod_sha, tmp = world
    plan = "# plan\n- T2679 envelope release\n"
    sha = _release(repo, app, "cytir/serve/app.py", "V = 2\n", also=[("plan.md", plan)])
    rp = _envelope_report(world_trees, world, sha)
    decl = dict(AW_PARITY_REPORT=str(rp), AW_ENVELOPE_CHANGE="T2679", AW_ENVELOPE_FIELDS="boosts")
    undeclared = _run_script(app, sha, env, AW_PARITY_REPORT=str(rp))
    assert undeclared.returncode == 1 and "verdict is 'FAIL'" in undeclared.stderr
    no_f = _run_script(app, sha, env, **{**decl, "AW_ENVELOPE_FIELDS": ""})
    assert no_f.returncode == 1 and "AW_ENVELOPE_FIELDS" in no_f.stderr
    both = _run_script(app, sha, env, **{**decl, "AW_RANKING_CHANGE": "T2679"})
    assert both.returncode == 1 and "two modes" in both.stderr
    wrong = _run_script(app, sha, env, **{**decl, "AW_ENVELOPE_FIELDS": "filters.note"})
    assert wrong.returncode == 1 and "undeclared envelope path" in wrong.stderr
    over = _run_script(app, sha, env, **{**decl, "AW_ENVELOPE_CHANGE": "T9999"})
    assert over.returncode == 1 and "neither plan.md nor decisions.md" in over.stderr
    assert (box / "cytir/serve/app.py").read_text() == "V = 1\n"
    ok = _run_script(app, sha, env, **decl)
    out = ok.stdout + ok.stderr
    assert "ENVELOPE CHANGE T2679 accepted" in out and "envelope change declared: T2679" in out, out
    assert (box / "cytir/serve/app.py").read_text() == "V = 2\n"


def test_rebuild_script_envelope_change_goes_through_the_same_gate(world, world_trees):
    repo, app, box, env, prod_sha, tmp = world
    sha = _release(repo, app, "cytir/serve/app.py", "V = 2\n", also=[("plan.md", "# plan\n- T2679 x\n")])
    rp = _envelope_report(world_trees, world, sha, paths=("boosts", "hits"))
    r = subprocess.run(["bash", str(app / "deploy" / "scripts" / "rebuild-serving-image.sh"), sha],
                       capture_output=True, text=True, timeout=120,
                       env={**env, "AW_PARITY_REPORT": str(rp), "AW_ENVELOPE_CHANGE": "T2679",
                            "AW_ENVELOPE_FIELDS": "boosts"})
    assert r.returncode == 1 and "undeclared envelope path(s) ['hits']" in r.stderr, r.stdout + r.stderr
    assert not (tmp / "compose.log").exists()
    assert (box / "cytir/serve/app.py").read_text() == "V = 1\n"
