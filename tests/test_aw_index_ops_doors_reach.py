"""Task 2361 (S5 W5): the five operator tools answer behind `cyt index` / `cyt ops` with the door convention."""
import importlib

import pytest

from cytir import cli

DOORS = [
    ("index", "apply-delta", "cytir.index.apply_delta"),
    ("index", "verify-supersede", "cytir.index.verify_supersede_by_id"),
    ("index", "layout-check", "cytir.index.layout_check"),
    ("ops", "landing-check", "cytir.ops.landing_check"),
    ("ops", "restore-pack", "cytir.ops.restore_pack"),
    ("ops", "qdrant-hop-verify", "cytir.ops.qdrant_hop_verify"),
    ("ops", "http-drift", "cytir.ops.http_drift"),
    ("ops", "prod-env-run", "cytir.ops.prod_env_run"),
    ("ops", "guards", "cytir.ops.guards"),
    ("ops", "registry-census", "cytir.ops.registry_census"),
    ("ops", "github-docs-gold", "cytir.ops.github_docs_gold"),
    ("ops", "experiments-ledger", "cytir.ops.experiments_ledger"),
    ("ops", "agent-probe", "cytir.ops.agent_probe"),
]


def test_tables_name_the_modules():
    assert cli._INDEX_TOOL_VERBS == {v: m for g, v, m in DOORS if g == "index"}
    assert cli._OPS_TOOL_VERBS == {v: m for g, v, m in DOORS if g == "ops"}


@pytest.mark.parametrize("group,verb,mod", DOORS)
def test_help_first_line_names_the_door(group, verb, mod, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main([group, verb, "--help"])
    assert e.value.code == 0
    assert capsys.readouterr().out.splitlines()[0].startswith(f"usage: cyt {group} {verb}")


def test_apply_delta_help_says_it_writes(capsys):
    with pytest.raises(SystemExit):
        cli.main(["index", "apply-delta", "--help"])
    assert "WRITES" in capsys.readouterr().out


@pytest.mark.parametrize("group,verb,mod", DOORS)
def test_exit_code_propagates(group, verb, mod, monkeypatch):
    m = importlib.import_module(mod)
    seen = {}

    def fake(argv=None, prog=None):
        seen.update(argv=argv, prog=prog)
        return 7

    monkeypatch.setattr(m, "main", fake)
    assert cli.main([group, verb, "--x", "1"]) == 7
    assert seen == {"argv": ["--x", "1"], "prog": f"cyt {group} {verb}"}


def test_bench_scope_door_routes(capsys):
    """Task 2546: `cyt bench scope` forwards argv to cytir.bench.scope.main; a bad subcommand fails."""
    with pytest.raises(SystemExit) as e:
        cli.main(["bench", "scope", "report", "--help"])
    assert e.value.code == 0
    assert "--dir1024" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        cli.main(["bench", "scope", "nonsense"])
    assert e.value.code != 0
    assert cli.main(["bench", "nonsense"]) == 2
