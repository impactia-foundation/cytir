#!/usr/bin/env python3
"""The server image is built from exact pins (deployment-doctrine 1, 6).

PR #628's deploy was rolled back because unpinned requirements let every image
build resolve whatever PyPI had that day (18 silent bumps). Three properties:
every direct requirement is `==` or a pinned URL; every constraint is `==`;
and the Dockerfile installs from those two files, not from anything floating.
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "server"   # requirements.txt, constraints.txt, Dockerfile


def _lines(name: str) -> list[str]:
    out = []
    for raw in (HERE / name).read_text().splitlines():
        line = raw.split(" #")[0].strip() if not raw.lstrip().startswith("#") else ""
        if line:
            out.append(line)
    return out


def test_every_requirement_is_pinned():
    lines = _lines("requirements.txt") + _lines("requirements-embed-local.txt")
    assert lines
    for line in lines:
        pinned = re.match(r"^[A-Za-z0-9_.\-]+(\[[a-z,]+\])?==[^=<>~ ]+$", line)
        url = re.match(r"^[A-Za-z0-9_.\-]+ @ https://\S+/[0-9a-f]{40}\.tar\.gz$", line)
        assert pinned or url, f"unpinned requirement: {line!r}"


def test_every_constraint_is_exact():
    lines = _lines("constraints.txt")
    assert lines
    for line in lines:
        assert re.match(r"^[A-Za-z0-9_.\-]+==[^=<>~ ]+$", line), f"inexact constraint: {line!r}"


def test_every_direct_requirement_is_constrained_consistently():
    cons = {l.split("==")[0].lower().replace("_", "-"): l.split("==")[1] for l in _lines("constraints.txt")}
    for line in _lines("requirements.txt") + _lines("requirements-embed-local.txt"):
        m = re.match(r"^([A-Za-z0-9_.\-]+)(?:\[[a-z,]+\])?==(\S+)$", line)
        if m:
            name = m.group(1).lower().replace("_", "-")
            assert cons.get(name, m.group(2)) == m.group(2), f"{name}: requirements vs constraints disagree"


def test_dockerfile_installs_from_those_files():
    d = (HERE / "Dockerfile").read_text()
    assert "COPY server/requirements.txt server/requirements-embed-local.txt server/constraints.txt" in d
    assert "pip install --no-cache-dir -r requirements.txt -r requirements-embed-local.txt -c constraints.txt" in d
    assert "-c constraints.txt --index-url" in d  # the CPU torch install is constrained too
    for line in d.splitlines():
        if "pip install" in line and not line.lstrip().startswith("#"):
            assert "-c " in line, f"unconstrained pip install: {line!r}"


def test_serve_requirements_carry_no_torch_and_the_embed_local_file_does():
    """Task 2281: the serve-only file is torch-free; the torch carriers live in the embed-local file."""
    serve = " ".join(_lines("requirements.txt")).lower()
    for pkg in ("flagembedding", "sentence-transformers", "torch", "transformers"):
        assert pkg not in serve, pkg
    local = " ".join(_lines("requirements-embed-local.txt")).lower()
    assert "flagembedding" in local and "sentence-transformers" in local
