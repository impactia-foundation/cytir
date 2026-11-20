"""`cyt source audit <family>` reaches every audit module through the verb table (task 2352)."""
import importlib
import sys
from pathlib import Path

import pytest

from cytir import cli
from cytir.sources import audit_door

EXPECTED = {"hn", "rfc-editor", "service-public", "routec", "stackoverflow", "context7", "finma-partitions",
            "tech-product"}


def test_every_family_resolves_to_a_module_with_main():
    assert set(audit_door.FAMILIES) == EXPECTED
    for family, module in audit_door.FAMILIES.items():
        assert callable(importlib.import_module(module).main), family


def test_verb_table_reaches_each_family(monkeypatch):
    seen = []
    for family, module in audit_door.FAMILIES.items():
        mod = importlib.import_module(module)
        monkeypatch.setattr(mod, "main", lambda argv, prog=None, f=family: seen.append((f, argv, prog)))
    for family in audit_door.FAMILIES:
        assert cli._source_main(["audit", family, "--x", "1"]) == 0
    assert seen == [(f, ["--x", "1"], f"cyt source audit {f}") for f in audit_door.FAMILIES]


def test_every_family_help_prints_its_usage_line(capsys):
    for family in audit_door.FAMILIES:
        with pytest.raises(SystemExit) as exc:
            audit_door.main([family, "--help"])
        assert exc.value.code == 0, family
        out = capsys.readouterr().out
        assert out.startswith(f"usage: cyt source audit {family}"), (family, out[:80])


def test_dispatch_does_not_rebind_sys_argv(monkeypatch):
    before = list(sys.argv)
    seen = []
    mod = importlib.import_module(audit_door.FAMILIES["hn"])
    monkeypatch.setattr(mod, "main", lambda argv, prog=None: seen.append(list(sys.argv)) or 0)
    assert audit_door.main(["hn", "--x"]) == 0
    assert seen == [before]
    assert sys.argv == before


def test_unknown_family_is_refused(capsys):
    assert cli._source_main(["audit", "nope"]) == 2
    assert "unknown audit family" in capsys.readouterr().err


def test_help_lists_families(capsys):
    assert cli._source_main(["audit", "--help"]) == 0
    out = capsys.readouterr().out
    assert all(f in out for f in EXPECTED)


# audit_profile.py files that are deliberately NOT a family, each with a reason.
EXCLUDED: dict[str, str] = {}


def test_every_audit_profile_module_is_registered_or_excluded():
    root = Path(audit_door.__file__).resolve().parents[1]
    found = {".".join(p.relative_to(root.parent).with_suffix("").parts)
             for p in root.rglob("audit_profile.py")}
    registered = set(audit_door.FAMILIES.values())
    assert all(EXCLUDED.values()), "every exclusion needs a reason"
    assert found - registered - set(EXCLUDED) == set()
    assert not (set(EXCLUDED) - found)


def test_exit_code_of_the_family_main_is_returned(monkeypatch):
    mod = importlib.import_module(audit_door.FAMILIES["hn"])
    monkeypatch.setattr(mod, "main", lambda argv, prog=None: 3)
    assert audit_door.main(["hn"]) == 3
