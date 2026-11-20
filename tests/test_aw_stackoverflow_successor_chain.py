"""Stack Overflow successor chain: step order, byte-identical argv, exit propagation, parent gate.

The fixture holds the argv the former shell chain produced for one representative tranche (recorded by
running that script with a recording interpreter and nice/ionice shims); `<REL>` stands for the release dir.
"""
import json
from pathlib import Path

import pytest

from cytir.sources.stackoverflow import successor_chain as sc

FIX = json.loads((Path(__file__).parent / "fixtures/stackoverflow_successor_chain_argv.v1.json").read_text())
REL = Path("/rel")


def _cfg(**over):
    c = dict(FIX["cfg"])
    base = {k: (Path(v) if k in ("artifact_root", "parent_root", "parent_outcome", "pack_root",
                                 "audit_root", "source_adapter") else v) for k, v in c.items()}
    base["parent_outcome"] = Path(c["outcome"])
    base.pop("outcome")
    base["source_adapter"] = Path(base.pop("adapter"))
    base["python_bin"] = "PY"
    base.update(over)
    return sc.SuccessorCfg(**base)


def _expected(cfg_over=None):
    return [[*s["prefix"], "PY", *s["argv"]] for s in FIX["steps"]]


def _unrel(argv):
    return [a.replace("<REL>", str(REL)) for a in argv]


def test_argv_is_byte_identical_to_the_shell_chain():
    got = sc.build_steps(_cfg(), release_dir=REL)
    assert got == [_unrel(a) for a in _expected()]


def test_nice_ionice_on_steps_one_and_two_only():
    steps = sc.build_steps(_cfg(), release_dir=REL)
    assert [s[:5] for s in steps[:2]] == [sc.NICE, sc.NICE]
    assert steps[2][0] == steps[3][0] == "PY"
    assert [s[s.index("-m") + 1] for s in steps] == [
        "cytir.corpus.stage_structured_markdown", "cytir.sources.stackoverflow.audit_profile",
        "cytir.index.seal_content_fidelity_audit", "cytir.index.seal_source_profile_compliance"]
    assert "--source-profile" in steps[0] and steps[0][steps[0].index("--source-profile") + 1] == "community_qa"


def test_expected_cell_omitted_when_empty():
    steps = sc.build_steps(_cfg(expected_cell=""), release_dir=REL)
    assert "--expected-cell" not in steps[1]
    assert "--expected-cell" in sc.build_steps(_cfg(), release_dir=REL)[1]


def _write_parent(tmp_path, *, outcome=None, run=None):
    parent = tmp_path / "parent"
    parent.mkdir()
    out = tmp_path / "OUTCOME.json"
    if outcome is not None:
        out.write_text(json.dumps(outcome))
    if run is not None:
        (parent / "RUN.json").write_text(json.dumps(run))
    return parent, out


GOOD_RUN = {"schema": FIX["cfg"]["expected_schema"], "status": FIX["cfg"]["expected_status"]}


@pytest.mark.parametrize("outcome,run,msg", [
    (None, GOOD_RUN, "OUTCOME is absent"),
    ({"status": "failed", "rc": 1}, GOOD_RUN, "did not succeed"),
    ({"status": "succeeded", "rc": 2}, GOOD_RUN, "did not succeed"),
    ({"status": "succeeded", "rc": 0}, None, "RUN is absent"),
    ({"status": "succeeded", "rc": 0}, {**GOOD_RUN, "schema": "other"}, "nonterminal"),
    ({"status": "succeeded", "rc": 0}, {**GOOD_RUN, "status": "running"}, "nonterminal"),
])
def test_parent_gate_rejects(tmp_path, outcome, run, msg):
    parent, out = _write_parent(tmp_path, outcome=outcome, run=run)
    with pytest.raises(sc.SuccessorChainError, match=msg):
        sc.check_parent_terminal(out, parent / "RUN.json", GOOD_RUN["schema"], GOOD_RUN["status"])


class _Runner:
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = [], fail_at

    def __call__(self, argv, env=None):
        self.calls.append((argv, env))
        rc = 1 if self.fail_at == len(self.calls) else 0
        return type("R", (), {"returncode": rc})()


def _ready(tmp_path):
    parent, out = _write_parent(tmp_path, outcome={"status": "succeeded", "rc": 0}, run=GOOD_RUN)
    return _cfg(parent_root=parent, parent_outcome=out)


def test_runs_all_four_in_order_with_release_pythonpath_first(tmp_path):
    cfg, r = _ready(tmp_path), _Runner()
    sc.run_successor(cfg, env={"PYTHONPATH": "/drv/repo:/drv/repo/pipeline"}, runner=r, log=lambda m: None,
                     release_dir=REL)
    assert [c[0] for c in r.calls] == sc.build_steps(cfg, release_dir=REL)
    assert all(c[1]["PYTHONPATH"] == "/rel:/drv/repo:/drv/repo/pipeline" for c in r.calls)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_failure_at_step_k_stops_the_chain(tmp_path, k):
    r = _Runner(fail_at=k)
    with pytest.raises(sc.SuccessorChainError, match="rc=1"):
        sc.run_successor(_ready(tmp_path), env={}, runner=r, log=lambda m: None, release_dir=REL)
    assert len(r.calls) == k


def test_absent_outcome_runs_nothing(tmp_path):
    parent, out = _write_parent(tmp_path, run=GOOD_RUN)
    r = _Runner()
    with pytest.raises(sc.SuccessorChainError, match="OUTCOME is absent"):
        sc.run_successor(_cfg(parent_root=parent, parent_outcome=out), env={}, runner=r,
                         log=lambda m: None, release_dir=REL)
    assert r.calls == []
