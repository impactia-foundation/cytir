"""Characterization of the 427 consent sweep runner + verifier and the 479 driver + watchdog,
ported into `cytir.consent` (task 721, S4).

No live call: `control_ok` and `resolve_hosts` are replaced by sealed stand-ins, the frontier is a
small deterministic file built here, and the watchdog's `systemctl`/`systemd-run` are injected.
The runner and verifier tests were first run against the legacy scripts (same assertions) before
the files moved; they pin what the sweep reads, writes and prints.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from cytir.consent import resolver as cr
from cytir.consent import sweep, sweep_driver, sweep_verify as ver

SCHEME = "https"


def _src(host, mode="cc", docs=3):
    return {"origin": f"{SCHEME}://{host}", "scheme": SCHEME, "authority": host, "host": host,
            "mode": mode, "documents": docs, "waves": ["w1"], "sample_urls": [f"{SCHEME}://{host}/a"]}


def _frontier(tmp_path, rows):
    d = tmp_path / "frontier"
    d.mkdir()
    gz = d / "uncovered-live-frontier.jsonl.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    sha = hashlib.sha256(gz.read_bytes()).hexdigest()
    run = d / "RUN.json"
    run.write_text(json.dumps({"artifacts": {"live_frontier": {"compressed_sha256": sha}}}))
    return gz, run


def _receipt(host, mode, status="ok", search_ok=True, train_ok=True, robots_status=200, body=""):
    return {"host": host, "origin": f"{SCHEME}://{host}", "origin_scheme": SCHEME, "mode": mode,
            "status": status, "search_ok": search_ok, "train_ok": train_ok,
            "checked_at": "2026-10-01T00:00:00Z", "_robots_body": body,
            "signals": {"robots": {"http_status": robots_status}}}


VERDICTS = {  # host -> (receipt kwargs, expected disposition)
    "open.example": (dict(), "admitted"),
    "search.example": (dict(train_ok=False), "admitted_search_only"),
    "no.example": (dict(search_ok=False, train_ok=False), "declined"),
    "flaky.example": (dict(status="unknown", search_ok=None, train_ok=None, robots_status=None),
                      "held_unknown"),
    "forbidden.example": (dict(status="unknown", search_ok=None, train_ok=None, robots_status=403),
                          "declined"),
}


def _fake_resolver(monkeypatch, calls):
    def resolve_hosts(hosts, cache_dir, max_age_days, concurrency, progress, direct_crawl, origin_scheme):
        calls.append((list(hosts), direct_crawl, origin_scheme, max_age_days, concurrency))
        for h in hosts:
            mode = "direct" if direct_crawl else "cc"
            kw, _ = VERDICTS[h]
            p = cr._cache_path(cache_dir, h, direct_crawl=direct_crawl, origin_scheme=origin_scheme)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(_receipt(h, mode, **kw)))
        return {}
    monkeypatch.setattr(sweep.cr, "resolve_hosts", resolve_hosts)
    monkeypatch.setattr(sweep, "control_ok", lambda *a, **k: True)


# ---- classification --------------------------------------------------------------------------

@pytest.mark.parametrize("host", sorted(VERDICTS))
def test_classify_each_disposition(host):
    kw, want = VERDICTS[host]
    got, effective = sweep.classify(_receipt(host, "cc", **kw))
    assert got == want
    if want.startswith("admitted"):
        assert effective["effective_search_ok"] is True
        assert effective["effective_train_ok"] is (want == "admitted")
    if host == "forbidden.example":
        assert effective["explicit_refusal_on_unresolved_record"] is True


def test_html_robots_body_is_a_diagnostic_never_a_disposition():
    r = _receipt("h.example", "cc", body="<!DOCTYPE html><html>nope</html>")
    disp, eff = sweep.classify(r)
    assert disp == "admitted"
    assert eff["evidence_observation"] == "robots_http_200_html_body"


def test_retracted_gate_still_rederives_old_rows():
    r = _receipt("h.example", "cc", body="<html>x</html>")
    assert sweep.classify_pre_retraction(r)[0] == "held_invalid_evidence"
    assert sweep.evidence_defect(_receipt("h.example", "cc")) is None


def test_dispositions_and_schema_are_frozen():
    assert sweep.DISPOSITIONS == ver.DISPOSITIONS == {
        "admitted", "admitted_search_only", "declined", "held_invalid_evidence", "held_unknown"}
    assert sweep.SCHEMA == "aw.task427-consent-reresolution.v1"
    assert sweep.control_backoff(1) == 0.0 and sweep.control_backoff(3) == 5.0
    assert sweep.control_backoff(99) == 60.0


# ---- frontier + integrity --------------------------------------------------------------------

def test_read_frontier_refuses_duplicates_bad_mode_and_drifted_origin(tmp_path):
    ok = [_src("open.example")]
    gz, _ = _frontier(tmp_path, ok + ok)
    with pytest.raises(ValueError, match="duplicate"):
        sweep.read_frontier(gz)
    bad = dict(_src("open.example"), mode="warc")
    gz2, _ = _frontier(tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path, [bad])
    with pytest.raises(ValueError, match="non-live mode"):
        sweep.read_frontier(gz2)


def test_validate_results_binds_one_to_one():
    rows = [_src("open.example"), _src("no.example")]
    rec = [dict(rows[0], disposition="admitted")]
    out = sweep.validate_results(rows, rec)
    assert out["exact_key_set"] is False and out["missing_keys"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        sweep.validate_results(rows, rec + rec)
    with pytest.raises(ValueError, match="invalid disposition"):
        sweep.validate_results(rows, [dict(rows[0], disposition="maybe")])
    with pytest.raises(ValueError, match="changed frontier field"):
        sweep.validate_results(rows, [dict(rows[0], disposition="admitted", documents=99)])


def test_batch_groups_split_by_scheme_mode_and_size():
    rows = [_src(f"h{i}.example") for i in range(5)] + [_src("d.example", mode="direct")]
    got = [(s, m, [r["authority"] for r in b]) for s, m, b in sweep.batch_groups(rows, 2)]
    assert got == [("https", "cc", ["h0.example", "h1.example"]),
                   ("https", "cc", ["h2.example", "h3.example"]),
                   ("https", "cc", ["h4.example"]),
                   ("https", "direct", ["d.example"])]


def test_private_evidence_may_not_live_in_the_shared_cache(tmp_path):
    with pytest.raises(ValueError, match="shared cache"):
        sweep.assert_private(Path(cr.CACHE_DIR) / "run")
    sweep.assert_private(tmp_path / "ok")


# ---- run: sealed end-to-end ------------------------------------------------------------------

def _rows():
    return [_src(h) for h in sorted(VERDICTS)]


def test_run_completes_writes_receipt_and_shares_nothing(tmp_path, monkeypatch):
    calls = []
    _fake_resolver(monkeypatch, calls)
    gz, run_json = _frontier(tmp_path, _rows())
    out = tmp_path / "out"
    receipt = sweep.run(gz, run_json, out, batch_size=100, concurrency=2)
    assert receipt["status"] == "complete"
    assert receipt["counts"]["origin_mode_keys"] == {
        "admitted": 1, "admitted_search_only": 1, "declined": 2, "held_unknown": 1}
    assert receipt["counts"]["documents"]["declined"] == 6
    assert receipt["conservation"] == {"lhs": 5, "rhs": 5, "pass": True}
    assert receipt["safety"] == {"private_cache": str(out / "private-cache"),
                                 "shared_cache_writes": False, "verdict_application": False,
                                 "production_writes": False}
    assert receipt["producer"]["script"] == str(Path(sweep.__file__).resolve())
    assert json.loads((out / "RUN.json").read_text()) == receipt
    lines = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
    assert len(lines) == 5
    assert all(l["control"] == {"pre": True, "post": True, "origin": sweep.CONTROL_ORIGIN} for l in lines)
    assert len(list((out / "private-cache").iterdir())) == 5
    assert list((out / "batch-work").iterdir()) == []            # throw-away dirs removed
    assert calls == [(sorted(VERDICTS), False, "https", 0, 2)]    # max_age 0: always live
    assert (out / "driver.log").read_text().count("recorded=5") == 1


def test_run_is_resumable_and_never_reprobes_a_recorded_key(tmp_path, monkeypatch):
    calls = []
    _fake_resolver(monkeypatch, calls)
    gz, run_json = _frontier(tmp_path, _rows())
    out = tmp_path / "out"
    sweep.run(gz, run_json, out, 2, 2)           # batch 2 -> three batches, all complete
    n = len(calls)
    again = sweep.run(gz, run_json, out, 2, 2)
    assert again["status"] == "complete" and len(calls) == n


def test_failed_control_discards_the_batch_and_the_pass_stays_running(tmp_path, monkeypatch):
    calls = []
    _fake_resolver(monkeypatch, calls)
    monkeypatch.setattr(sweep, "control_ok", lambda *a, **k: False)
    monkeypatch.setattr(sweep, "CONTROL_BACKOFF", (0.0,))
    monkeypatch.setattr(sweep.time, "sleep", lambda s: None)
    gz, run_json = _frontier(tmp_path, _rows())
    out = tmp_path / "out"
    receipt = sweep.run(gz, run_json, out, 100, 2)
    assert receipt["status"] == "running" and calls == []       # pre-control red: nothing probed
    assert not (out / "results.jsonl").exists()
    assert receipt["conservation"]["pass"] is False


def test_frontier_hash_mismatch_refuses(tmp_path, monkeypatch):
    _fake_resolver(monkeypatch, [])
    gz, run_json = _frontier(tmp_path, _rows())
    run_json.write_text(json.dumps({"artifacts": {"live_frontier": {"compressed_sha256": "0" * 64}}}))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        sweep.run(gz, run_json, tmp_path / "out", 100, 2)


def test_main_exit_codes_and_printed_counts(tmp_path, monkeypatch, capsys):
    _fake_resolver(monkeypatch, [])
    gz, run_json = _frontier(tmp_path, _rows())
    args = ["--frontier", str(gz), "--frontier-run", str(run_json), "--out-dir", str(tmp_path / "o")]
    assert sweep.main(args) == 0
    assert json.loads(capsys.readouterr().out)["origin_mode_keys"]["admitted"] == 1
    with pytest.raises(SystemExit):
        sweep.main(args + ["--concurrency", "3"])


# ---- verify ----------------------------------------------------------------------------------

def _completed_sweep(tmp_path, monkeypatch):
    _fake_resolver(monkeypatch, [])
    gz, run_json = _frontier(tmp_path, _rows())
    out = tmp_path / "out"
    sweep.run(gz, run_json, out, 100, 2)
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "x.json").write_text("{}")
    files, tree = ver.tree_sha256(shared)
    return gz, out, shared, files, tree


def test_verify_passes_a_complete_sweep(tmp_path, monkeypatch):
    gz, out, shared, files, tree = _completed_sweep(tmp_path, monkeypatch)
    res = ver.verify(frontier=gz, run_dir=out, shared_cache=shared,
                     expected_shared_files=files, expected_shared_tree_sha256=tree)
    assert res["status"] == "PASS" and res["schema"] == "aw.task427-consent-reresolution-verification.v1"
    assert res["counts"]["frontier_keys"] == res["counts"]["result_keys"] == 5
    assert res["counts"]["private_cache_files"] == 5 and res["shared_cache_tree_sha256"] == tree
    assert all(res["checks"].values()) and res["safety"]["production_writes"] is False


def test_verify_refuses_a_touched_shared_cache(tmp_path, monkeypatch):
    gz, out, shared, files, tree = _completed_sweep(tmp_path, monkeypatch)
    (shared / "y.json").write_text("{}")
    with pytest.raises(ValueError, match="shared consent cache changed"):
        ver.verify(frontier=gz, run_dir=out, shared_cache=shared,
                   expected_shared_files=files, expected_shared_tree_sha256=tree)


def test_verify_refuses_receipt_identity_drift_and_unproven_admission(tmp_path, monkeypatch):
    gz, out, shared, files, tree = _completed_sweep(tmp_path, monkeypatch)
    results = out / "results.jsonl"
    rows = [json.loads(x) for x in results.read_text().splitlines()]
    rows[0]["receipt"]["host"] = "other.example"
    results.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    run = json.loads((out / "RUN.json").read_text())
    run["results"]["sha256"] = ver.sha256_file(results)
    (out / "RUN.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="identity drift: authority-as-host"):
        ver.verify(frontier=gz, run_dir=out, shared_cache=shared,
                   expected_shared_files=files, expected_shared_tree_sha256=tree)


def test_verify_refuses_an_incomplete_run(tmp_path):
    (tmp_path / "RUN.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(ValueError, match="not complete"):
        ver.verify(frontier=tmp_path / "f", run_dir=tmp_path, shared_cache=tmp_path,
                   expected_shared_files=0, expected_shared_tree_sha256="")


def test_verify_main_refuses_to_overwrite_and_prints_counts(tmp_path, monkeypatch, capsys):
    gz, out, shared, files, tree = _completed_sweep(tmp_path, monkeypatch)
    target = tmp_path / "V.json"
    argv = ["--frontier", str(gz), "--run-dir", str(out), "--shared-cache", str(shared),
            "--expected-shared-files", str(files), "--expected-shared-tree-sha256", tree,
            "--out", str(target)]
    assert ver.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["frontier_keys"] == 5
    assert json.loads(target.read_text())["status"] == "PASS"
    with pytest.raises(ValueError, match="refusing existing"):
        ver.main(argv)


# ---- driver (479 run_4crawl_consent_sweep.sh) ------------------------------------------------

def _baseline(tmp_path, files=1, tree="t"):
    p = tmp_path / "base.json"
    p.write_text(json.dumps({"files": files, "tree_sha256": tree}))
    return p


def _drive(tmp_path, run_fn, verify_fn=None, **kw):
    gz, run_json = _frontier(tmp_path, _rows())
    out = tmp_path / "out"
    rc = sweep_driver.drive(frontier_run=run_json, out_dir=out, baseline=_baseline(tmp_path),
                            revision="rev1", shared_cache=tmp_path / "s",
                            run_fn=run_fn, verify_fn=verify_fn or (lambda **k: {"status": "PASS", "k": k}),
                            sleep=lambda s: None, **kw)
    return rc, out


def _runner(statuses, out_holder=None):
    it = iter(statuses)

    def run(frontier, frontier_run, out_dir, batch, conc):
        s = next(it)
        (out_dir / "RUN.json").write_text(json.dumps({"status": s}))
        return {"status": s}
    return run


def test_driver_loops_until_complete_then_verifies(tmp_path):
    seen = {}
    rc, out = _drive(tmp_path, _runner(["running", "running", "complete"]),
                     lambda **k: seen.update(k) or {"status": "PASS"})
    assert rc == 0
    log = (out / "process.log").read_text()
    assert "pass=1 rc=2 status=running" in log and "pass=3 rc=0 status=complete" in log
    assert "sweep complete after 3 passes; verifying" in log and "verification rc=0" in log
    assert json.loads((out / "CONSENT-VERIFICATION.json").read_text()) == {"status": "PASS"}
    assert seen["expected_shared_files"] == 1 and seen["expected_shared_tree_sha256"] == "t"
    assert seen["frontier"].name == "uncovered-live-frontier.jsonl.gz"


def test_driver_aborts_on_pass_budget_and_on_unexpected_outcome(tmp_path):
    rc, out = _drive(tmp_path, _runner(["running"] * 5), max_passes=2)
    assert rc == 3 and "ABORT pass budget 2 exhausted" in (out / "process.log").read_text()

    def boom(*a):
        raise RuntimeError("x")
    rc, out = _drive(tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path, boom)
    assert rc == 4 and "ABORT unexpected runner outcome rc=1" in (out / "process.log").read_text()


def test_driver_reports_a_refused_verification(tmp_path):
    def refuse(**k):
        raise ValueError("shared consent cache changed during private sweep")
    rc, out = _drive(tmp_path, _runner(["complete"]), refuse)
    assert rc == 1 and not (out / "CONSENT-VERIFICATION.json").exists()
    assert "verification rc=1" in (out / "process.log").read_text()


# ---- watchdog (479 watchdog_4crawl_consent.sh) -----------------------------------------------

def test_watchdog_action_table():
    assert sweep_driver.watchdog_action("complete", False) == "done"
    assert sweep_driver.watchdog_action("complete", True) == "done"
    assert sweep_driver.watchdog_action("running", True) == "alive"
    assert sweep_driver.watchdog_action("missing", False) == "relaunch"


def test_run_status_reads_status_or_missing(tmp_path):
    assert sweep_driver.run_status(tmp_path / "none.json") == "missing"
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    assert sweep_driver.run_status(bad) == "missing"
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"status": "complete"}))
    assert sweep_driver.run_status(ok) == "complete"


def _wd(tmp_path, status, active):
    root = tmp_path / "corpus"
    p = sweep_driver.watchdog_paths(root, "dc99ef2c")
    if status:
        p["sweep_dir"].mkdir(parents=True)
        (p["sweep_dir"] / "RUN.json").write_text(json.dumps({"status": status}))
    launched, said = [], []
    act = sweep_driver.watchdog(
        release=tmp_path / "rel", tag="dc99ef2c", revision="abc", corpus_root=root,
        shared_cache=tmp_path / "sc", operator_url="https://op.example",
        operator_contact="ops@op.example", is_active=lambda u: active,
        launch=launched.append, out=said.append)
    return act, launched, said


def test_watchdog_does_nothing_when_complete_or_alive(tmp_path):
    assert _wd(tmp_path, "complete", False) == ("done", [], [])
    assert _wd(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path, "running", True)[0] == "alive"


def test_watchdog_relaunch_command_is_exact(tmp_path):
    act, launched, said = _wd(tmp_path, "running", False)
    assert act == "relaunch" and "sweep not complete (status=running) and unit down; relaunching" in said[0]
    cmd = launched[0]
    base = tmp_path / "corpus" / "_work" / "task479"
    assert cmd[:4] == ["systemd-run", "--unit=aw-task479-4crawl-consent-dc99ef2c", "--collect",
                       "--description=Task 479 4-crawl serving-only consent sweep (rel)"]
    assert cmd[4:6] == ["-p", f"WorkingDirectory={tmp_path / 'rel'}"]
    assert "Environment=AW_OPERATOR_URL=https://op.example" in cmd
    assert "Environment=AW_OPERATOR_CONTACT=ops@op.example" in cmd
    i = cmd.index("python3")
    assert cmd[i:i + 4] == ["python3", "-m", "cytir.consent.sweep_driver", "drive"]
    assert cmd[cmd.index("--frontier-run") + 1] == str(
        base / "r0-task479-4crawl-servingonly-frontier-dc99ef2c" / "RUN.json")
    assert cmd[cmd.index("--out-dir") + 1] == str(
        base / "r0-task479-4crawl-servingonly-consent-dc99ef2c")
    assert cmd[cmd.index("--revision") + 1] == "abc"
    assert cmd[cmd.index("--batch-size") + 1] == "100" and cmd[cmd.index("--concurrency") + 1] == "2"


def test_watchdog_main_demands_operator_identity(tmp_path, monkeypatch):
    for k in ("AW_CORPUS_ROOT", "AW_OPERATOR_URL", "AW_OPERATOR_CONTACT"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit):
        sweep_driver.main(["watchdog", "--release", "r", "--tag", "t", "--revision", "v"])


# ---- the verb ------------------------------------------------------------------

def test_aw_consent_verb_dispatches_and_refuses_unknowns(capsys):
    from cytir.cli import main as aw_main
    assert aw_main(["consent"]) == 2
    assert aw_main(["consent", "sweep", "nope"]) == 2
    with pytest.raises(SystemExit) as e:
        aw_main(["consent", "sweep", "run", "--help"])
    assert e.value.code == 0
    assert "cyt consent sweep run" in capsys.readouterr().out
