"""Guard (task 1140): the serving image's install layer must have what `pip install .` reads.

Task 733 made the package read `README.md` (pyproject `readme`), `setup.py` and `data/`
(`package-dir "cytir._data" = "data"`), but the Dockerfile kept copying only `pyproject.toml` and
`cytir/`. The first image rebuild after that failed at `pip install .` (1138). No docker daemon in CI,
so this test stages EXACTLY the sources of the install stage's `COPY` lines into a temp dir and
builds a non-editable wheel there. A source the Dockerfile forgets to COPY fails the build here.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import zipfile

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
DOCKERFILE = REPO / "server" / "Dockerfile"


def install_stage_copies() -> list[str]:
    """Sources of the `COPY` lines from the `pyproject.toml` copy to the `pip install ... .` layer."""
    out, active = [], False
    for line in DOCKERFILE.read_text().splitlines():
        s = line.strip()
        if s.startswith("COPY ") and "pyproject.toml" in s.split():
            active = True
        if active and s.startswith("COPY "):
            parts = [p for p in s.split()[1:] if not p.startswith("--")]
            out += parts[:-1]
        if active and s.startswith("RUN") and re.search(r"pip install .* \.\s*$", s):
            return out
    raise AssertionError("server/Dockerfile: no install-stage `COPY pyproject.toml ... RUN pip install .` found")


def test_install_stage_copies_are_found():
    assert "pyproject.toml" in install_stage_copies()


def test_install_stage_sources_build_a_wheel(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    for src in install_stage_copies():
        p = REPO / src
        assert p.exists(), f"Dockerfile COPYs {src!r}, which is not in the repo"
        dst = stage / src
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(p, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")) if p.is_dir() else shutil.copy2(p, dst)
    # what .dockerignore removes from the context must not be needed (and must not be in the image)
    for pat in _dockerignore_globs():
        for f in stage.glob(pat):
            f.unlink()
    # The venv's own setuptools builds it when it is new enough to read [project] (pyproject asks
    # for >=68; an older one names the wheel "UNKNOWN"). Otherwise let pip fetch the declared
    # build requirements, as the image build does.
    cmd = [sys.executable, "-m", "pip", "wheel", "--no-deps", "-q", "-w", str(tmp_path / "dist"), str(stage)]
    if _setuptools_major() >= 68:
        cmd.insert(5, "--no-build-isolation")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    out = r.stdout + r.stderr
    if r.returncode and "--no-build-isolation" not in cmd and re.search(
            r"No matching distribution|Could not find a version|Connection|Temporary failure|Max retries", out):
        pytest.skip("no setuptools>=68 in this venv and no network to fetch one")
    assert r.returncode == 0, (
        "`pip wheel .` fails on the files server/Dockerfile copies into the install stage "
        "(a file pyproject.toml/setup.py reads is not COPYed):\n" + out[-1500:]
    )
    wheel = next((tmp_path / "dist").glob("cytir-*.whl"))
    names = zipfile.ZipFile(wheel).namelist()
    assert any(n.startswith("cytir/_data/") for n in names), "wheel carries no cytir/_data/ (data/ not copied)"
    assert not any("held-pending-anonymization" in n for n in names)


def _dockerignore_globs() -> list[str]:
    f = REPO / ".dockerignore"
    if not f.exists():
        return []
    return [l.strip() for l in f.read_text().splitlines() if l.strip() and not l.startswith("#")]


def _setuptools_major() -> int:
    try:
        import setuptools
        return int(setuptools.__version__.split(".")[0])
    except Exception:
        return 0
