"""AW_FOLD_ACCENTS parity tests for the BM25 tokenizer (task 290 fold, task 301
rebuild-prep). Verifies the accent-fold added to BOTH bm25_tokens paths:

  - write path : cytir/core/common.py::bm25_tokens
  - query path : cytir/serve/schema.py::bm25_tokens

They MUST match exactly or query terms won't hit index terms. Checks:
  1. OFF (default) is a byte-identical no-op vs the raw-token hash on ASCII
     exact-lookup tokens (art/336/co/che/fedlex/cochrane) — legal citations,
     UIDs and signatures cannot regress.
  2. ON collapses accented variants to the same token as the un-accented ASCII
     form (Genève→geneve, café→cafe, psychédélique→psychedelique), while those
     same ASCII controls are byte-identical folded vs unfolded.
  3. Both paths produce identical tokens for the same input, OFF and ON
     (the write/query mirror contract).

Run: python3 -m pytest tests/test_bm25_fold.py
"""
import hashlib
import importlib.util
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent          # .../authoritative-web-app/tests
_APP = _HERE.parent                                       # .../authoritative-web-app


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass annotation resolution (which looks the
    # module up in sys.modules by __module__) works under the custom name.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load both tokenizers as independent modules straight from their files so the
# test is agnostic to sys.path / package layout and can compare them head-to-head.
aw_common = _load(_APP / "cytir" / "core" / "bm25_tokenizer.py", "aw_common_fold_test")  # task 2333: the tokenizer moved out of common.py
schema = _load(_APP / "cytir" / "serve" / "schema.py", "aw_schema_fold_test")

PATHS = [("write", aw_common), ("query", schema)]


def _set_fold(on: bool):
    aw_common._FOLD_ACCENTS = on
    schema._FOLD_ACCENTS = on


def _raw_hash(tok: str) -> int:
    # Independent reference for the pre-fold behaviour: md5 of the lowercased
    # token, first 8 hex → int (identical formula in both modules).
    return int(hashlib.md5(tok.lower().encode("utf-8")).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# 1. OFF is a byte-identical no-op on ASCII exact-lookup tokens.
# ---------------------------------------------------------------------------

ASCII_CONTROLS = ["art", "336", "co", "che", "fedlex", "cochrane"]


def test_off_is_raw_hash_noop_ascii():
    _set_fold(False)
    for name, mod in PATHS:
        for tok in ASCII_CONTROLS:
            # lang='en' isolates the token (no German decompound noise); none of
            # these are >=12 chars anyway.
            out = mod.bm25_tokens(tok, lang="en")
            assert out == {_raw_hash(tok): 1.0}, (name, tok, out)


def test_off_phrase_is_raw_hash_noop():
    _set_fold(False)
    phrase = "art 336 CO che fedlex cochrane"
    expected = {}
    for tok in ["art", "336", "co", "che", "fedlex", "cochrane"]:
        expected[_raw_hash(tok)] = expected.get(_raw_hash(tok), 0.0) + 1.0
    for name, mod in PATHS:
        assert mod.bm25_tokens(phrase, lang="en") == expected, name


# ---------------------------------------------------------------------------
# 2. ON: ASCII controls unchanged (folded == unfolded); accents collapse.
# ---------------------------------------------------------------------------

def test_ascii_controls_identical_folded_vs_unfolded():
    for name, mod in PATHS:
        for tok in ASCII_CONTROLS:
            _set_fold(False)
            off = mod.bm25_tokens(tok, lang="en")
            _set_fold(True)
            on = mod.bm25_tokens(tok, lang="en")
            assert off == on == {_raw_hash(tok): 1.0}, (name, tok, off, on)


# (accented form, un-accented ASCII form)
ACCENT_PAIRS = [
    ("Genève", "geneve"),
    ("genève", "geneve"),
    ("café", "cafe"),
    ("psychédélique", "psychedelique"),
    ("médicament", "medicament"),
    ("société", "societe"),
    ("Zürich", "zurich"),
    ("français", "francais"),
]


def test_accents_collapse_to_ascii_when_on():
    _set_fold(True)
    for name, mod in PATHS:
        for accented, ascii_form in ACCENT_PAIRS:
            a = mod.bm25_tokens(accented, lang="fr")
            b = mod.bm25_tokens(ascii_form, lang="fr")
            assert a == b, (name, accented, ascii_form, a, b)
            # and the collapsed token is exactly the ASCII hash
            assert a == {_raw_hash(ascii_form): 1.0}, (name, accented, a)


def test_accents_do_NOT_collapse_when_off():
    _set_fold(False)
    for name, mod in PATHS:
        # sanity: with folding OFF the accented and ASCII forms are distinct
        # tokens (proves the ON collapse is the fold's doing, not a coincidence).
        a = mod.bm25_tokens("genève", lang="fr")
        b = mod.bm25_tokens("geneve", lang="fr")
        assert a != b, name


# ---------------------------------------------------------------------------
# 3. Write path and query path are byte-identical for the same input.
# ---------------------------------------------------------------------------

CROSS_INPUTS = [
    ("art 336 CO", "fr"),
    ("art 336 CO", "de"),
    ("tribunal fédéral genève médicament", "fr"),
    ("Bundesverwaltungsgericht Schutzdauer Urheberrecht", "de"),
    ("O'Hear d'Angelo société anonyme", "fr"),
    ("psychédélique substances contrôlées", None),
    ("café Zürich français", "de"),
]


def test_write_and_query_paths_identical_off():
    _set_fold(False)
    for text, lang in CROSS_INPUTS:
        assert aw_common.bm25_tokens(text, lang) == schema.bm25_tokens(text, lang), (text, lang)


def test_write_and_query_paths_identical_on():
    _set_fold(True)
    for text, lang in CROSS_INPUTS:
        assert aw_common.bm25_tokens(text, lang) == schema.bm25_tokens(text, lang), (text, lang)


# ---------------------------------------------------------------------------
# 4. _fold helper + env parsing are byte-identical across the two modules.
# ---------------------------------------------------------------------------

def test_fold_helper_identical_across_paths():
    for s in ["Genève", "psychédélique", "café", "ART336", "naïve", "Œuvre", "Straße"]:
        assert aw_common._fold(s) == schema._fold(s), s


def test_env_truthy_parsing():
    from cytir.serve import settings as st
    for fn in (aw_common._env_truthy, st._truthy):
        for v in ("1", "true", "True", "YES", "on", " on "):
            assert fn(v) is True, (fn, v)
        for v in ("0", "", None, "false", "no", "off", "2x"):
            assert fn(v) is False, (fn, v)
    assert not hasattr(schema, "_env_truthy") and not hasattr(st, "env_truthy")


def test_writing_the_fold_flag_on_common_raises():
    """task 2333: the flag lives in the tokenizer module; a write on `cytir.core.common` must refuse."""
    import pytest
    from cytir.core import common
    with pytest.raises(RuntimeError, match="bm25_tokenizer"):
        common._FOLD_ACCENTS = True
    with pytest.raises(AttributeError):
        common._FOLD_ACCENTS
