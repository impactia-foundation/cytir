"""The assembly binder's command line: hold first, defaults unchanged, exit code passed through."""
from __future__ import annotations

import pytest

from cytir.core.paths import REPO_ROOT
from cytir.index import assemble_cli as CLI
from cytir.index import twin_write_hold as TWH


def _spy(monkeypatch, order, rc=0):
    monkeypatch.setattr(TWH, "refuse_cli_write", lambda args, tool: order.append(("hold", tool)))

    def fn(args):
        order.append(("fn", args.cmd))
        return rc
    for name in ("cmd_guards", "cmd_plan", "cmd_stage", "cmd_load", "cmd_verify"):
        monkeypatch.setattr(CLI, name, fn)


@pytest.mark.parametrize("argv", [["guards"], ["plan", "--work", "w"], ["stage", "--work", "w"],
                                  ["load", "--work", "w", "--config", "256-pq8", "--collection", "c"], ["verify", "--work", "w"]])
def test_the_hold_is_called_before_every_verb_dispatches(monkeypatch, argv):
    order = []
    _spy(monkeypatch, order)
    assert CLI.main(argv) == 0
    assert [o[0] for o in order] == ["hold", "fn"]
    assert order[0][1] == "cyt index assemble"


def test_a_held_collection_refuses_before_the_verb_runs(monkeypatch):
    order = []
    _spy(monkeypatch, order)

    def refuse(args, tool):
        raise SystemExit("REFUSED: held")
    monkeypatch.setattr(TWH, "refuse_cli_write", refuse)
    with pytest.raises(SystemExit):
        CLI.main(["verify", "--work", "w"])
    assert order == []


def test_the_verbs_exit_code_is_returned(monkeypatch):
    order = []
    _spy(monkeypatch, order, rc=1)
    assert CLI.main(["verify", "--work", "w"]) == 1


def test_the_default_plan_is_the_current_assembly_plan(monkeypatch):
    seen = {}
    monkeypatch.setattr(TWH, "refuse_cli_write", lambda args, tool: None)
    monkeypatch.setattr(CLI, "cmd_guards", lambda args: seen.setdefault("plan", args.plan) and 0)
    CLI.main(["guards"])
    assert seen["plan"] == str(REPO_ROOT / "data" / "assembly" / "assembly-plan.current.v1.json")
