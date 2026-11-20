#!/usr/bin/env python3
"""The Fedlex identity contract, against REAL gold URLs.

`fedlex_contract.py` is a mirror contract in the strict sense: three
independent callers (`fedlex_dump.py` and two task scripts) derive the same
ELI/language/anchor identity from a URL, and `fedlex_xml_url` reconstructs
the filestore URL the AKN XML is fetched from. It had no test at all — and it
is exactly the shape the verification doctrine reserves unit tests for: a pure
function where a regression is silent and catastrophic, because a changed
identity means a whole family re-mints under different point ids and every
guard downstream reasons about the wrong set (ADR-101 Ruling 2).

The inputs are real: every URL below is a gold target in
`data/eval_authoritative.yaml`, so a change that breaks them breaks the
instrument.

Run: python3 -m pytest tests/test_fedlex_contract.py
"""
from __future__ import annotations

import pathlib
import re
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from cytir.sources.families import fedlex_contract as FC  # noqa: E402

REPO = HERE.parent

#: Gold URLs, copied verbatim out of data/eval_authoritative.yaml — four
#: languages and both ELI shapes the corpus actually carries (a dated
#: `1999/404` and a triple `1966/57_57_57`).
GOLD = [
    ("https://www.fedlex.admin.ch/eli/cc/1966/57_57_57/de#art_9",
     ("cc/1966/57_57_57", "de", "art_9")),
    ("https://www.fedlex.admin.ch/eli/cc/1991/1184_1184_1184/it#art_36",
     ("cc/1991/1184_1184_1184", "it", "art_36")),
    ("https://www.fedlex.admin.ch/eli/cc/1995/1328_1328_1328/fr#art_64",
     ("cc/1995/1328_1328_1328", "fr", "art_64")),
    ("https://www.fedlex.admin.ch/eli/cc/1999/404/de#art_8",
     ("cc/1999/404", "de", "art_8")),
    ("https://www.fedlex.admin.ch/eli/cc/2009/615/en#art_10",
     ("cc/2009/615", "en", "art_10")),
    ("https://www.fedlex.admin.ch/eli/cc/24/233_245_233/de#art_471",
     ("cc/24/233_245_233", "de", "art_471")),
]


@pytest.mark.parametrize("url,want", GOLD)
def test_source_native_key_on_real_gold_urls(url, want):
    assert FC.source_native_key(url) == want


def test_the_gold_urls_are_still_in_the_instrument():
    """If the eval file drops these, this test is measuring nothing."""
    eval_file = REPO / "data" / "eval_authoritative.yaml"
    if not eval_file.is_file():
        pytest.skip("the eval instrument is not shipped in the public cut")
    text = eval_file.read_text()
    missing = [u for u, _ in GOLD if u not in text]
    assert missing == [], (
        f"{len(missing)} URL(s) no longer appear in eval_authoritative.yaml: "
        f"{missing}. Re-derive this fixture from the instrument rather than "
        "keeping a copy that has drifted.")


def test_a_window_suffix_collapses_to_the_base_anchor():
    """`#art_9-c2` and `#art_9` are the same article, one windowed chunk of it.

    Two identities for one article is the defect: the guards would see a
    replacement that claims points it does not cover.
    """
    base = FC.source_native_key(
        "https://www.fedlex.admin.ch/eli/cc/1999/404/de#art_8")
    windowed = FC.source_native_key(
        "https://www.fedlex.admin.ch/eli/cc/1999/404/de#art_8-c2")
    assert base == windowed == ("cc/1999/404", "de", "art_8")


def test_a_multi_digit_window_suffix_also_collapses():
    assert FC.source_native_key(
        "https://www.fedlex.admin.ch/eli/cc/1999/404/de#art_8-c17"
    ) == ("cc/1999/404", "de", "art_8")


def test_an_anchor_that_merely_ends_in_a_letter_c_and_digits_is_not_a_window():
    """`-c2` is a window; `_c2` and a bare `c2` are part of the anchor."""
    assert FC.source_native_key(
        "https://www.fedlex.admin.ch/eli/cc/1999/404/de#art_8_c2"
    ) == ("cc/1999/404", "de", "art_8_c2")


def test_a_url_with_no_anchor_has_no_native_key():
    """Returns None rather than a partial identity — fail loud, not silent."""
    assert FC.source_native_key(
        "https://www.fedlex.admin.ch/eli/cc/1999/404/de") is None


def test_a_non_fedlex_url_has_no_native_key():
    assert FC.source_native_key("https://example.org/eli/cc/1999/404/de") is None


def test_an_unsupported_language_is_not_silently_accepted():
    """Romansh was dropped from scope on 2026-06-21; `rm` must not parse as a
    language, or a family would mint identities for a language we do not
    index."""
    assert FC.source_native_key(
        "https://www.fedlex.admin.ch/eli/cc/1999/404/rm#art_8") is None


@pytest.mark.parametrize("url,want", GOLD)
def test_fedlex_xml_url_round_trips_the_native_key(url, want):
    """The filestore URL the AKN XML is fetched from is a pure function of the
    identity — so it must be reconstructible from the identity alone."""
    eli, lang, _anchor = want
    got = FC.fedlex_xml_url(eli, lang, lang)
    assert got.startswith("https://fedlex.data.admin.ch/filestore/")
    assert eli.replace("/", "-") in got


def test_fedlex_xml_url_is_exactly_the_documented_shape():
    got = FC.fedlex_xml_url("cc/1999/404", "20260101", "de")
    assert got == (
        "https://fedlex.data.admin.ch/filestore/fedlex.data.admin.ch/eli/"
        "cc/1999/404/20260101/de/xml/"
        "fedlex-data-admin-ch-eli-cc-1999-404-20260101-de-xml.xml")


def test_the_filename_flattens_every_dot_and_slash():
    """A single unflattened separator produces a 404 on the filestore, which
    the acquisition lane records as a missing document rather than a bad URL."""
    got = FC.fedlex_xml_url("cc/1999/404", "20260101", "de")
    filename = got.rsplit("/", 1)[1]
    assert "/" not in filename
    assert re.fullmatch(r"[a-z0-9\-]+\.xml", filename), filename


def test_only_this_module_defines_the_fedlex_identity():
    """Engineering doctrine 1: copying it is forking a contract.

    Every caller must import it; a second `ELI_ANCHOR_RE` anywhere under
    the private pipeline tree is the defect, whatever its comment says.
    """
    offenders = []
    private = REPO / "pipeline"
    if not private.is_dir():
        pytest.skip("no private pipeline tree in this checkout")
    for p in sorted(private.glob("*.py")):
        if p.name in ("fedlex_contract.py", "test_fedlex_contract.py"):
            continue
        src = p.read_text()
        if "/eli/(" in src or "ELI_ANCHOR_RE = " in src:
            offenders.append(p.name)
    assert offenders == [], (
        f"{offenders} define their own Fedlex ELI pattern instead of "
        "importing cytir.sources.families.fedlex_contract.")
