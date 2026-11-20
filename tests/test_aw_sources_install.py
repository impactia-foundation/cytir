"""Task 735 -- `cyt source ...` works from a non-editable install (the validator lives in the package).

Holds: the package never reads `sources/_schema/` at import; the shipped schema copy equals the policy
original byte for byte; `sources_dir()` finds the packs (or refuses by name); a pack reader says so
when the packs are absent instead of failing in a loader; the validator accepts every shipped pack.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

APP = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))


def test_shipped_schema_equals_the_policy_original_byte_for_byte():
    original = APP / "sources" / "_schema" / "pack.schema.json"
    shipped = APP / "data" / "pack.schema.json"
    assert shipped.read_bytes() == original.read_bytes(), (
        "data/pack.schema.json drifted from sources/_schema/pack.schema.json: "
        "`cp sources/_schema/pack.schema.json data/pack.schema.json`")


def test_the_package_never_loads_code_from_sources():
    """No `spec_from_file_location` on a `validate.py`, no `sys.path` entry for `sources/`."""
    offenders = []
    for p in (APP / "cytir").rglob("*.py"):
        for i, line in enumerate(p.read_text("utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if "validate.py" in code and "spec_from_file_location" in code:
                offenders.append(f"{p.relative_to(APP)}:{i}")
    assert not offenders, offenders
    assert not (APP / "cytir" / "sources" / "pack_loader.py").read_text("utf-8").count("SCHEMA_DIR")


def test_importing_the_pack_loader_needs_no_sources_tree(monkeypatch, tmp_path):
    """A family that reads no pack must import from a bare install: a clean interpreter, no
    checkout beside the code, AW_SOURCES_DIR unset."""
    code = ("import cytir.sources.pack_loader as PL; "
            "from cytir.core.paths import SourcesNotFound, sources_dir; "
            "from cytir.core import paths; paths.REPO_ROOT = __import__('pathlib').Path('/nonexistent'); "
            "import sys\n"
            "try: sources_dir(); sys.exit(1)\n"
            "except SourcesNotFound as e: print(e)")
    env = {k: v for k, v in __import__("os").environ.items()
           if k not in ("AW_SOURCES_DIR", "AW_REPO_ROOT", "CYTIR_SOURCES_DIR", "CYTIR_REPO_ROOT", "_CYTIR_ENV_MIRRORED")}
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env={**env, "PYTHONPATH": str(APP)},
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "CYTIR_SOURCES_DIR" in r.stdout and "public repository" in r.stdout


def test_sources_dir_order(monkeypatch, tmp_path):
    from cytir.core import paths
    monkeypatch.delenv("AW_REPO_ROOT", raising=False)
    monkeypatch.delenv("AW_SOURCES_DIR", raising=False)
    assert paths.sources_dir() == paths.REPO_ROOT / "sources"          # a checkout
    (tmp_path / "sources").mkdir()
    monkeypatch.setenv("AW_REPO_ROOT", str(tmp_path))
    assert paths.sources_dir() == tmp_path / "sources"
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setenv("AW_SOURCES_DIR", str(other))
    assert paths.sources_dir() == other                                 # explicit wins
    monkeypatch.setenv("AW_SOURCES_DIR", str(tmp_path / "missing"))
    with pytest.raises(paths.SourcesNotFound, match="not a directory"):
        paths.sources_dir()


def test_find_pack_without_packs_is_a_named_refusal(monkeypatch, tmp_path):
    from cytir.sources import pack_loader as PL
    monkeypatch.setenv("AW_SOURCES_DIR", str(tmp_path / "nope"))
    with pytest.raises(PL.PackError, match="CYTIR_SOURCES_DIR"):
        PL.find_pack("anything")


@pytest.mark.skipif(not (APP / "pipeline").is_dir(),
                    reason="packs name extractors under pipeline/, which the public cut does not ship")
def test_every_shipped_pack_validates_through_the_package(monkeypatch):
    from cytir.sources import schema
    monkeypatch.delenv("AW_SOURCES_DIR", raising=False)
    assert schema.main([]) == 0


def test_the_pack_path_follows_AW_SOURCES_DIR(monkeypatch, tmp_path):
    """A copy of the packs elsewhere is found and validated; the copy's parent is the tree root."""
    import shutil
    from cytir.sources import pack_loader as PL
    tree = tmp_path / "clone"
    shutil.copytree(APP / "sources", tree / "sources", ignore=shutil.ignore_patterns("__pycache__"))
    (tree / "cytir" / "extract").mkdir(parents=True)
    monkeypatch.setenv("AW_SOURCES_DIR", str(tree / "sources"))
    pid = next(__import__("yaml").safe_load(p.read_text())["id"]
               for p in sorted((tree / "sources").rglob("*.yaml")) if "_schema" not in p.parts)
    assert PL.find_pack(pid).is_relative_to(tree)


def test_the_pack_cli_refuses_by_name_when_no_packs_are_in_reach(monkeypatch, tmp_path, capsys):
    from cytir.sources import runner
    monkeypatch.setenv("AW_SOURCES_DIR", str(tmp_path / "nope"))
    assert runner.main(["ch-media", "--plan"]) == 1
    assert "REFUSED" in capsys.readouterr().err
