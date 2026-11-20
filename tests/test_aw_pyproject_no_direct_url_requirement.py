"""PyPI refuses an upload whose metadata has a direct-URL requirement, also inside an extra (task 2989).

`400 Can't have direct dependency: charsplit @ https://... ; extra == "dev"` blocked the 0.1.0 upload.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_DIRECT = re.compile(r"@\s*(https?|git\+|file|ssh)|^\s*(git\+|https?://|file:)", re.I)


def _requirements() -> list[tuple[str, str]]:
    try:
        import tomllib
    except ModuleNotFoundError:  # python < 3.11
        import tomli as tomllib
    proj = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    out = [("dependencies", r) for r in proj.get("dependencies", [])]
    for extra, reqs in proj.get("optional-dependencies", {}).items():
        out += [(extra, r) for r in reqs]
    return out


def test_no_requirement_is_a_direct_url() -> None:
    bad = [(where, r) for where, r in _requirements() if _DIRECT.search(r)]
    assert not bad, f"direct-URL requirement(s) PyPI refuses: {bad}; install them by hand (see README)"


def test_negative_control_the_pattern_catches_the_refused_line() -> None:
    assert _DIRECT.search("charsplit @ https://github.com/x/y/archive/abc.tar.gz")
    assert not _DIRECT.search("charsplit==1.0")
