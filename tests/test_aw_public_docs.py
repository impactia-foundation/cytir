"""The three public docs (README, QUICKSTART, ADDING-A-SOURCE) describe only what the public cut
holds (task 998, S11 slice 12).

A stranger holds the cut, not the private tree. So in each doc: every relative markdown link, every
`pip install -r FILE` and every `python3 FILE.py` names a file the cut ships, and no fenced command
block names a path in the two private trees. The private tree names are built from parts so this
module does not trip the leak scan itself.
"""
from __future__ import annotations

import functools
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
DOCS = ("README.md", "QUICKSTART.md", "ADDING-A-SOURCE.md")
if not all((REPO / d).is_file() for d in DOCS):
    pytest.skip("public docs not all present in this tree", allow_module_level=True)

_PRIVATE = ("pipe" + "line", "tas" + "ks")
_PRIVATE_PATH = re.compile(r"(?<![\w.-])(?:%s)/[\w.-]+" % "|".join(_PRIVATE))


def _text(doc: str) -> str:
    return (REPO / doc).read_text(encoding="utf-8")


def _fenced(doc: str) -> list[str]:
    return re.findall(r"```[a-z]*\n(.*?)```", _text(doc), flags=re.S)


@functools.lru_cache(maxsize=None)
def _shipped() -> frozenset[str]:
    """The shipped-path set, from the allowlist + tracked files only. `public_cut.plan` also
    runs the whole-tree leak scan (minutes); these tests need the listing, not the leaks."""
    from cytir.ops import public_cut

    try:
        doc = public_cut.load_allowlist(REPO)
        return frozenset(public_cut.select(doc, public_cut.tracked_files(REPO))[0])
    except OSError:
        pytest.skip("no public-cut allowlist in this tree")
    except public_cut.CutRefusal as e:
        pytest.fail(f"public cut refused: {e}")


@pytest.mark.parametrize("doc", DOCS)
def test_doc_names_only_files_the_public_cut_ships(doc):
    text = _text(doc)
    refs = set(re.findall(r"pip install (?:-q )?-r (\S+)", text))
    refs |= set(re.findall(r"python3 ([\w./-]+\.py)\b", "\n".join(_fenced(doc))))
    refs |= {m for m in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text)
             if not re.match(r"[a-z]+:", m)}
    missing = sorted(r for r in refs if r not in _shipped())
    assert not missing, f"{doc} names files the public cut does not ship: {missing}"


@pytest.mark.parametrize("doc", DOCS)
def test_no_fenced_command_names_a_private_tree_path(doc):
    bad = [m for block in _fenced(doc) for m in _PRIVATE_PATH.findall(block)]
    assert not bad, f"{doc}: a fenced block names a private-tree path: {bad}"


@pytest.mark.parametrize("doc", DOCS)
def test_no_prose_names_a_private_tree_script(doc):
    """Prose may say 'private operator step'; it may not point a stranger at a script path."""
    bad = _PRIVATE_PATH.findall(_text(doc))
    assert not bad, f"{doc} names a private-tree path: {bad}"


def test_readme_says_the_production_rebuild_is_an_operator_step():
    text = _text("README.md")
    assert "private tooling" in text and "Swiss slice" in text
    assert "rebuild_from_manifest" not in text and "production-rebuild-lock" not in text
