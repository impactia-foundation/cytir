"""The extras smoke map (cytir.ops.extras_smoke) is held equal to pyproject.toml (task 2153)."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from cytir.ops import extras_smoke

ROOT = Path(__file__).resolve().parent.parent


def _extras() -> set[str]:
    # No tomllib on the CI runner's python3.10: read the table's top-level keys textually.
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = text.split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]
    return set(re.findall(r"^([a-z0-9_-]+) = \[", block, flags=re.M))


def test_every_extra_but_dev_has_a_map_entry() -> None:
    assert set(extras_smoke.EXTRA_MODULES) == _extras() - {"dev"}


def test_every_aw_module_in_the_map_exists() -> None:
    for extra, mods in extras_smoke.EXTRA_MODULES.items():
        assert mods, extra
        for mod in mods:
            if mod.startswith("cytir."):
                assert importlib.util.find_spec(mod) is not None, (extra, mod)


def test_every_extra_names_at_least_one_third_party_module() -> None:
    for extra, mods in extras_smoke.EXTRA_MODULES.items():
        assert any(not m.startswith("cytir.") for m in mods), extra


def test_unknown_extra_is_a_usage_error() -> None:
    assert extras_smoke.main(["nope"]) == 2
