"""`cyt ops cc-tranche <subverb>` reaches every cc_tranche module through the verb table (task 2355)."""
import importlib
from pathlib import Path

from cytir import cli
from cytir.ops import cc_tranche_door as door

# cc_tranche_* modules that are deliberately NOT a subverb, each with a reason.
EXCLUDED: dict[str, str] = {"cytir.ops.cc_tranche_door": "the dispatcher itself"}


def test_every_subverb_resolves_to_a_module_with_main():
    for name, (module, _, why) in cli.CC_TRANCHE_VERBS.items():
        assert callable(importlib.import_module(module).main), name
        assert why


def test_verb_table_reaches_each_subverb(monkeypatch):
    seen = []
    for name, (module, prefix, _) in cli.CC_TRANCHE_VERBS.items():
        mod = importlib.import_module(module)
        monkeypatch.setattr(mod, "main", lambda *a, n=module, **k: seen.append((n, list(a[0]) if a else [])) or 0)
    for name in cli.CC_TRANCHE_VERBS:
        assert cli._ops_main(["cc-tranche", name, "--x"]) == 0
    assert seen == [(m, [*pre, "--x"]) for m, pre, _ in cli.CC_TRANCHE_VERBS.values()]


# depth-fetch refuses before argparse (old behaviour: it hands argv to discovery_ingest), so it has no usage line
NO_ARGPARSE_HELP = {"depth-fetch"}


def test_each_subverb_help_names_the_door(capsys):
    import pytest
    for name in cli.CC_TRANCHE_VERBS:
        if name in NO_ARGPARSE_HELP:
            continue
        with pytest.raises(SystemExit) as e:
            cli._ops_main(["cc-tranche", name, "--help"])
        assert e.value.code == 0, name
        out = capsys.readouterr().out
        assert out.startswith(f"usage: cyt ops cc-tranche {name}"), (name, out[:80])


def test_unknown_subverb_is_refused(capsys):
    assert cli._ops_main(["cc-tranche", "nope"]) == 2
    assert "unknown cc-tranche subverb" in capsys.readouterr().err


def test_help_lists_subverbs(capsys):
    assert cli._ops_main(["cc-tranche", "--help"]) == 0
    out = capsys.readouterr().out
    assert all(n in out for n in cli.CC_TRANCHE_VERBS)


def test_every_cc_tranche_module_is_registered_or_excluded():
    root = Path(door.__file__).resolve().parent
    found = {f"cytir.ops.{p.stem}" for p in root.glob("cc_tranche_*.py")}
    registered = {m for m, _, _ in cli.CC_TRANCHE_VERBS.values()}
    assert all(EXCLUDED.values()), "every exclusion needs a reason"
    assert found - registered - set(EXCLUDED) == set()
    assert not (set(EXCLUDED) - found)


def test_exit_code_of_the_module_main_is_returned(monkeypatch):
    mod = importlib.import_module("cytir.ops.cc_tranche_delta")
    monkeypatch.setattr(mod, "main", lambda argv=None, prog=None: 3)
    assert door.main(cli.CC_TRANCHE_VERBS, ["delta"]) == 3
