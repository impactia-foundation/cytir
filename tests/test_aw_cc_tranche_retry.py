"""Tests for cytir.ops.cc_tranche_retry (task 760): retry named domains of a sealed CC tranche. No network, no fetch:
the runner is a fake that writes what a fetch would, into a fixture tranche dir."""
import fcntl
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from cytir.ops import cc_tranche_retry as rt

REPO = Path(__file__).resolve().parents[1]
ROSTER = ["alpha.example", "beta.example", "gamma.example", "delta.example"]


@pytest.fixture
def tranche(tmp_path):
    """A sealed tranche 7: 4 roster rows, `alpha` done, `beta` errored (absent from done)."""
    root = tmp_path / "task634"
    (root / "tranches" / "t7_out").mkdir(parents=True)
    (root / "tranches" / "t7_gpu").mkdir()
    (root / "tranches" / "t7_gpu" / ".sealed").write_text("2026-10-04T00:00:00Z\n")
    (root / "tranches" / "tranche7_never_queued.json").write_text(
        json.dumps([{"domain": d, "class": "edu"} for d in ROSTER]))
    (root / "tranches" / "t7_out" / "state.json").write_text(
        json.dumps({"done": {"alpha.example": {"status": "ok", "pages": 3, "rows": 9}}, "_egress": {}}))
    (root / "tranches" / "t7_out" / "passages.jsonl").write_text("")
    return root


def run(tranche, domains, *, dry=False, busy=(), runner=None):
    calls = []

    def fake(cmd, **kw):
        calls.append((cmd, kw))
        if runner:
            runner(cmd, kw)
        return SimpleNamespace(returncode=0)
    rc = rt.retry(7, domains, data_root=tranche, python="/py", hdd_root=Path("/hdd"), dry_run=dry,
                  probe=lambda: list(busy), runner=fake, now=lambda: "2026-10-09T00:00:00Z")
    return rc, calls


def fetch_adds(tranche, **done):
    def go(cmd, kw):
        st = tranche / "tranches" / "t7_out" / "state.json"
        d = json.loads(st.read_text())
        for dom, rows in done.items():
            d["done"][dom.replace("_", ".")] = {"status": "ok", "pages": 2, "rows": rows}
        st.write_text(json.dumps(d))
    return go


def test_parse_domains_file_shapes(tmp_path):
    j = tmp_path / "a.json"
    j.write_text(json.dumps({"7": ["A.example", "b.example", "a.example"], "8": ["x.example"]}))
    assert rt.parse_domains(str(j), 7) == ["a.example", "b.example"]
    l = tmp_path / "b.json"
    l.write_text(json.dumps(["c.example"]))
    assert rt.parse_domains(str(l), 7) == ["c.example"]
    t = tmp_path / "c.txt"
    t.write_text("d.example  # why\n\ne.example\n")
    assert rt.parse_domains(str(t), 7) == ["d.example", "e.example"]
    assert rt.parse_domains("f.example, g.example", 7) == ["f.example", "g.example"]
    with pytest.raises(SystemExit, match="no domain for tranche 9"):
        rt.parse_domains(str(j), 9)


def test_refuses_a_domain_outside_the_roster(tranche):
    with pytest.raises(SystemExit, match="not in the roster of tranche 7.*stranger.example"):
        run(tranche, ["beta.example", "stranger.example"])


def test_refuses_an_unsealed_tranche(tranche):
    (tranche / "tranches" / "t7_gpu" / ".sealed").unlink()
    with pytest.raises(SystemExit, match="not sealed"):
        run(tranche, ["beta.example"])


def test_dry_run_prints_the_plan_and_runs_nothing(tranche, capsys):
    rc, calls = run(tranche, ["alpha.example", "beta.example", "gamma.example"], dry=True)
    out = capsys.readouterr().out
    assert rc == 0 and calls == []
    plan = json.loads(out[:out.index("DRY RUN")])
    assert plan["to_fetch"] == ["beta.example", "gamma.example"] and plan["already_done"] == ["alpha.example"]
    assert "--only beta.example,gamma.example" in plan["command"] and "DRY RUN: ready" in out


def test_dry_run_names_a_block_but_does_not_fail(tranche, capsys):
    rc, calls = run(tranche, ["beta.example"], dry=True, busy=["1 python ... tranche14_never_queued.json"])
    assert rc == 0 and calls == [] and "BLOCKED now" in capsys.readouterr().out


def test_refuses_while_a_never_queued_fetch_runs(tranche):
    with pytest.raises(SystemExit, match="never-queued CC fetch is running"):
        run(tranche, ["beta.example"], busy=["1 python -m cytir.acquire.discovery_ingest fetch --queue tranches/tranche14_never_queued.json"])


def test_refuses_while_the_watchdog_pass_holds_its_lock(tranche):
    with open(tranche / ".watchdog.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(SystemExit, match="watchdog's pass holds"):
            run(tranche, ["beta.example"])


def test_nothing_to_do_when_every_domain_is_done(tranche, capsys):
    rc, calls = run(tranche, ["alpha.example"])
    assert rc == 0 and calls == [] and "nothing to do" in capsys.readouterr().out


def test_run_logs_per_domain_and_exits_4_on_a_domain_still_not_done(tranche, capsys):
    rc, calls = run(tranche, ["alpha.example", "beta.example", "gamma.example"],
                    runner=fetch_adds(tranche, beta_example=12))
    assert rc == rt.EXIT_NOT_DONE and len(calls) == 1
    cmd, kw = calls[0]
    assert kw["cwd"] == str(tranche) and str(rt.REPO_ROOT) in kw["env"]["PYTHONPATH"]
    log = (tranche / "logs" / "tranche7-retry-domains.log").read_text().splitlines()
    rows = {r["domain"]: r for r in map(json.loads, log)}
    assert rows["alpha.example"]["status"] == "already-done"
    assert rows["beta.example"]["status"] == "done" and rows["beta.example"]["rows"] == 12
    assert rows["gamma.example"]["status"] == "NOT-DONE"


def test_run_exit_0_when_all_done(tranche):
    rc, _ = run(tranche, ["beta.example", "gamma.example"], runner=fetch_adds(tranche, beta_example=1, gamma_example=2))
    assert rc == 0


def test_command_writes_where_the_original_fetch_did(tranche):
    """Same out dir, state, queue name as watchdog_cc_tranches.sh: the seal watchdog's late-growth pass reads them unchanged."""
    _, calls = run(tranche, ["beta.example"], runner=fetch_adds(tranche, beta_example=1))
    cmd = calls[0][0]
    a = lambda flag: cmd[cmd.index(flag) + 1]
    assert a("--out") == "tranches/t7_out/passages.jsonl" and a("--state") == "tranches/t7_out/state.json"
    assert a("--links-out") == "tranches/t7_out/outlinks.jsonl" and a("--queue") == "tranches/tranche7_never_queued.json"
    assert a("--require-local-index") == "/hdd/aw-work/cc-index-mirror/" + rt.CRAWL
    assert "--no-resume" not in cmd
    assert re.search(rt.NEW_FETCH_RE, " ".join(cmd)), "the watchdog must see this as tranche 7's live fetch"


def test_flags_match_the_watchdog_recipe():
    """Drift guard: the retry uses the watchdog's rate, depth, workers, egress and crawl."""
    script = REPO / "deploy" / "scripts" / "watchdog_cc_tranches.sh"
    if not script.is_file():
        pytest.skip("the watchdog script is not part of the public cut")
    text = script.read_text()
    block = text[text.index("-m cytir.acquire.discovery_ingest fetch"):]
    block = block[:block.index("--require-local-index")]
    for flag, val in (("--rate", "4"), ("--workers", "10"), ("--heartbeat-pages", "20"), ("--columnar-threads", "3"),
                      ("--egress-mbps", "4.5"), ("--crawl", rt.CRAWL), ("--resolver", "columnar")):
        assert re.search(rf"{re.escape(flag)} {re.escape(val)}\b", block), flag
        assert rt.FETCH_FLAGS[rt.FETCH_FLAGS.index(flag) + 1] == val, flag
    assert "--max-per-domain" in block and rt.FETCH_FLAGS[rt.FETCH_FLAGS.index("--max-per-domain") + 1] == "60"
    assert "taskset -c 12-23 nice -n15 ionice -c3" in text and " ".join(rt.NICE) == "taskset -c 12-23 nice -n15 ionice -c3"
    assert "tranches/tranche[0-9]+_never_queued" in text and r"discovery_ingest(\.py)? fetch --queue" in text
    assert rt.NEW_FETCH_RE == r"discovery_ingest(\.py)? fetch --queue tranches/tranche[0-9]+_never_queued"
    assert f"cc-index-mirror/{rt.CRAWL}" in text


def test_aw_ops_dispatches_the_verb(tranche, capsys):
    from cytir.cli import main
    rc = main(["ops", "cc-tranche", "retry-domains", "--tranche", "7", "--domains", "beta.example", "--dry-run",
               "--data-root", str(tranche), "--python", "/py", "--hdd-root", "/hdd"])
    assert rc == 0 and "DRY RUN" in capsys.readouterr().out


def test_a_non_never_queued_tranche_number_is_refused(tranche):
    with pytest.raises(SystemExit, match="never-queued tranche"):
        rt.main(["--tranche", "3", "--domains", "x.example", "--data-root", str(tranche), "--python", "/py",
                 "--hdd-root", "/hdd"])
