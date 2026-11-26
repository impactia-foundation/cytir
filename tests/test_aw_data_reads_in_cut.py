"""Guard (task 2277): `cytir/` code must not name a `data/` file that the public cut neither ships nor declares.

A stranger's copy holds the allowlisted tree only. A reader that opens `data/x.json` outside it fails there
with a missing-file error that master never shows. The one table `OPTIONAL_DATA_READS` in `cytir/ops/public_cut.py`
is the only place a file may be named without shipping, and every row carries its reason.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from cytir.ops import public_cut as pc

ROOT = pc.APP_ROOT


# Task 2903: files the sharper scanner flagged that are neither declared nor made to degrade yet. SHRINK-ONLY:
# a row leaves this tuple when it is declared in OPTIONAL_DATA_READS (or shipped); no row may be added.
PINNED_UNDECLARED = ()


def _problem_path(problem: str) -> str:
    return problem.split(" names ", 1)[1].split(",", 1)[0]


def test_every_data_file_aw_names_ships_or_is_declared():
    problems, stale = pc.unshipped_data_reads(ROOT)
    paths = {_problem_path(p) for p in problems}
    new = [p for p in problems if _problem_path(p) not in PINNED_UNDECLARED]
    assert not new, "cytir/ names unshipped data files:\n  " + "\n  ".join(new[:20])
    # The cut ships none of the pinned files, so compare against the ones present in this tree.
    expected = {p for p in PINNED_UNDECLARED if (ROOT / p).exists()}
    assert paths == expected, (
        "PINNED_UNDECLARED is shrink-only and must equal what is flagged; drop the rows now declared: "
        f"{sorted(expected - paths)}")
    assert not stale, "stale OPTIONAL_DATA_READS rows:\n  " + "\n  ".join(stale)


def test_table_rows_are_well_formed():
    paths = [r[0] for r in pc.OPTIONAL_DATA_READS]
    assert len(paths) == len(set(paths)), "a path is declared twice"
    for path, state, reason in pc.OPTIONAL_DATA_READS:
        assert path.startswith("data/")
        assert state in ("degrades", "baseline"), (path, state)
        assert len(reason) > 20, f"{path}: a row needs its reason"


def test_guard_catches_an_undeclared_reader(tmp_path):
    """The scanner itself: one literal, one joined path, one f-string, a docstring that must not count."""
    (tmp_path / "cytir").mkdir()
    (tmp_path / "data").mkdir()
    for name in ("a", "b", "c", "d"):
        (tmp_path / "data" / f"{name}.json").write_text("{}")
    (tmp_path / "data" / "public-cut.v1.json").write_text(
        json.dumps({"schema": pc.SCHEMA, "rules": [{"id": "r", "why": "w", "paths": ["cytir/**"]}],
                    "never": [], "scan_exempt": []}))
    (tmp_path / "cytir" / "m.py").write_text(
        '"""Mentions data/d.json in prose only."""\n'
        'import os\nDATA = "x"\n'
        'A = "data/a.json"\n'
        'B = os.path.join("root", "data", "b.json")\n'
        'C = f"{DATA}/c.json"\n')
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    assert sorted(pc.data_literals(tmp_path)) == ["data/a.json", "data/b.json", "data/c.json"]
    problems, _ = pc.unshipped_data_reads(tmp_path)
    assert len(problems) == 3 and all("cytir/m.py" in p for p in problems)


def test_scanner_sees_every_path_join_shape(tmp_path):
    """Task 2903: `APP / "data" / "x.json"` hid because ast.walk is breadth-first (right operand first)."""
    (tmp_path / "cytir").mkdir()
    (tmp_path / "cytir" / "m.py").write_text(
        'import os, pathlib\n'
        'APP = pathlib.Path(".")\n'
        'A = APP / "data" / "slash.json"\n'
        'B = os.path.join(str(APP), "data", "osjoin.json")\n'
        'C = APP.joinpath("data", "joinpath.json")\n'
        'D = pathlib.Path("data", "ctor.json")\n'
        'E = pathlib.Path("data") / "sub" / "deep.jsonl"\n'
        'F = APP / "other" / "nodata.json"\n')
    assert sorted(pc.data_literals(tmp_path)) == [
        "data/ctor.json", "data/joinpath.json", "data/osjoin.json", "data/slash.json", "data/sub/deep.jsonl"]


def test_scanner_sees_a_path_built_through_a_data_directory_alias(tmp_path):
    """Task 3013: `DATA = APP / "data"` then `DATA / "x.json"` has no "data" literal at the use site."""
    (tmp_path / "cytir").mkdir()
    (tmp_path / "cytir" / "m.py").write_text(
        'import os, pathlib\n'
        'APP = pathlib.Path(".")\n'
        'DATA = os.path.join(APP, "data")\n'
        'PROFILES = APP / "data" / "serving-profiles"\n'
        'A = pathlib.Path(DATA) / "alias_slash.yaml"\n'
        'B = os.path.join(DATA, "alias_osjoin.json")\n'
        'C = PROFILES / "alias_sub.config.v1.json"\n'
        'D = f"{PROFILES}/alias_fstr.json"\n'
        'E = APP / "elsewhere" / "other.json"\n'
        'F = PROFILES.glob("*.json")\n')
    assert sorted(pc.data_literals(tmp_path)) == [
        "data/alias_osjoin.json", "data/alias_slash.yaml",
        "data/serving-profiles/alias_fstr.json", "data/serving-profiles/alias_sub.config.v1.json"]


def test_scan_manifest_names_the_missing_operator_manifest(tmp_path, capsys):
    """data/refresh_manifest.json is OPTIONAL: absent -> one named reason and rc 2; present -> it is scanned."""
    from cytir.corpus import authored_text_guard as g

    missing = tmp_path / "refresh_manifest.json"
    assert g.main(["scan-manifest", "--manifest", str(missing)]) == 2
    err = capsys.readouterr().err
    assert "manifest not found" in err and str(missing) in err

    missing.write_text(json.dumps({"sources": []}))
    assert g.main(["scan-manifest", "--manifest", str(missing)]) != 2


@pytest.mark.parametrize("mod", ["cytir.ops.experiments_ledger", "cytir.ops.lessons_ledger"])
def test_operator_ledger_verb_degrades_when_the_default_file_is_not_shipped(mod, tmp_path, monkeypatch, capsys):
    """Task 2903: in a stranger's cut the default ledger is absent; the verb says so and returns 0, no traceback."""
    import importlib

    m = importlib.import_module(mod)
    monkeypatch.setattr(m, "LEDGER_PATH", tmp_path / "absent.jsonl")
    assert m.main([]) == 0
    assert "ledger not shipped in this tree" in capsys.readouterr().out
