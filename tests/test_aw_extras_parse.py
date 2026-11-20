"""The extras in pyproject.toml stay parseable and reproducible (task 908).

Resolve cost of `.[pipeline]` is download volume, not backtracking, so the
guard is: every requirement parses, names are unique within an extra, and a
direct URL requirement is pinned to an immutable commit.
"""
import re
from pathlib import Path

from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:  # python < 3.11; tomli is a pytest dependency there
    import tomli as tomllib

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
EXTRAS = tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]


def test_expected_extras_exist():
    assert {"dev", "pipeline", "langgate", "pdf", "serve"} <= set(EXTRAS)


def test_every_requirement_parses_and_names_are_unique():
    for extra, reqs in EXTRAS.items():
        names = [Requirement(r).name.lower().replace("_", "-") for r in reqs]
        assert len(names) == len(set(names)), f"duplicate requirement in extra {extra}"


def test_direct_url_requirements_pin_a_commit():
    for extra, reqs in EXTRAS.items():
        for r in reqs:
            req = Requirement(r)
            if req.url:
                assert re.search(r"/archive/[0-9a-f]{40}\.", req.url), f"{extra}: {r} not commit-pinned"


def test_dev_extra_is_fully_pinned():
    for r in EXTRAS["dev"]:
        req = Requirement(r)
        if req.url or req.name == "pip-licenses":
            continue
        assert any(s.operator == "==" for s in req.specifier), f"dev: {r} not == pinned"
