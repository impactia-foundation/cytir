"""Task 411 unit tests — the pure functions whose regression would be silent.

Per verification-doctrine 1, unit tests here are TRIAGE, not coverage: the
ranking effect is measured by the gate. What is tested here is exactly what a
gate cannot see:

  1. `bm25_token_groups` is a partition of `bm25_tokens` (the mirror contract —
     if the group union ever drifts from the emitted token set, required-term
     matching silently tests the wrong tokens);
  2. the two tokenization artefacts the design exists to handle are handled:
     an apostrophe name's canonical token is the JOINED form, and a German
     compound is ONE requirement, not three;
  3. every task-411 flag is inert at its default.

Run: python3 -m pytest tests/test_task411_levers.py
"""
import importlib.util
import pathlib
import sys

_APP = pathlib.Path(__file__).resolve().parent.parent


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


schema = _load(_APP / "cytir" / "serve" / "schema.py", "aw_schema_411_test")

CORPUS = [
    "Tim O'Hear",
    "Marc Maret",
    "who is Ursula von der Leyen",
    "Kuendigungsfrist Arbeitsvertrag BGB",
    "Mietvertrag Kuendigung Frist BGB",
    "Laura Tocmacov",
    "accenture vector8",
    "what is the notice period under Art. 336 CO",
    "Genève fiduciaire comptable",
    "d'Angelo l'école",
    "NVIDIA DGX Spark worth it",
    "",
    "a",
    "3M web3 terminal42",
]
LANGS = [None, "de", "fr", "en", "it"]


def test_groups_partition_bm25_tokens():
    """Union of the groups' hashes == the token set bm25_tokens emits."""
    for text in CORPUS:
        for lang in LANGS:
            flat = set(schema.bm25_tokens(text, lang))
            grouped = set()
            for _w, _p, hs in schema.bm25_token_groups(text, lang):
                grouped |= set(hs)
            assert grouped == flat, (text, lang, grouped ^ flat)


def test_group_primary_is_in_its_own_group():
    for text in CORPUS:
        for lang in LANGS:
            for _w, p, hs in schema.bm25_token_groups(text, lang):
                assert p in hs


def test_apostrophe_name_primary_is_the_joined_form():
    """`O'Hear` emits BOTH `hear` (junk) and `ohear`; the canonical token a NAME
    must match is the joined one, else an all-must-match rule is unsatisfiable
    on every apostrophe name."""
    groups = dict((w, (p, hs)) for w, p, hs in
                  schema.bm25_token_groups("Tim O'Hear", "fr"))
    p, hs = groups["o'hear"]
    assert p == schema._hash("ohear")
    assert schema._hash("hear") in hs      # emitted, but not the primary
    assert len(hs) == 2


def test_german_compound_is_one_group_not_three():
    """`Arbeitsvertrag` de-compounds to three TOKENS but is ONE required WORD —
    the unit that makes `all terms match` definable (task 406 §3.2 measured this
    query at 5 `q_content` terms, over AW_NAMEGATE_MAXTERMS=4)."""
    groups = schema.bm25_token_groups("Kuendigungsfrist Arbeitsvertrag BGB", "de")
    assert [w for w, _p, _h in groups] == [
        "kuendigungsfrist", "arbeitsvertrag", "bgb"]
    by_word = {w: hs for w, _p, hs in groups}
    # the compound's own hash is always first; any split parts ride along
    assert by_word["arbeitsvertrag"][0] == schema._hash("arbeitsvertrag")
    assert all(len(hs) >= 1 for hs in by_word.values())


def test_defaults_are_inert():
    """Every 411 flag defaults to today's shipped behaviour."""
    import os
    for k in ("AW_NAMECOV_SURFACE", "AW_REQTERM_W", "AW_REQTERM_MODE",
              "AW_REQTERM_DFMAX", "AW_REQTERM_MAXQ"):
        os.environ.pop(k, None)
    sys.path.insert(0, str(_APP / "server"))
    for m in [m for m in list(sys.modules) if m.startswith("cytir.serve")]:
        del sys.modules[m]
    from cytir.serve import retrieval as R
    assert R._NAMECOV_SURFACE == "display"
    assert R._REQTERM_W == 0.0
    assert R._REQTERM_DFMAX == 0
    assert R._REQTERM_MAXQ == 0


def test_bad_surface_fails_loud():
    """A typo'd surface must crash, not silently degrade to today's behaviour
    (engineering-doctrine 3)."""
    import os
    os.environ["AW_NAMECOV_SURFACE"] = "sparce"
    sys.path.insert(0, str(_APP / "server"))
    for m in [m for m in list(sys.modules) if m.startswith("cytir.serve")]:
        del sys.modules[m]
    try:
        from cytir.serve import retrieval  # noqa: F401
        raised = False
    except ValueError:
        raised = True
    finally:
        os.environ.pop("AW_NAMECOV_SURFACE", None)
        for m in [m for m in list(sys.modules) if m.startswith("cytir.serve")]:
            del sys.modules[m]
    assert raised


def test_required_word_groups_drops_non_distinctive_words():
    import os
    os.environ.pop("AW_NAMECOV_SURFACE", None)
    sys.path.insert(0, str(_APP / "server"))
    for m in [m for m in list(sys.modules) if m.startswith("cytir.serve")]:
        del sys.modules[m]
    from cytir.serve import retrieval as R
    q = "who is Laura Tocmacov"
    groups = schema.bm25_token_groups(q, None)
    hs = {w: set(h) for w, _p, h in groups}
    # only `tocmacov` is distinctive => exactly one required word, and the
    # stopwords are gone even before distinctiveness is consulted.
    req = R._required_word_groups(q, None, hs["tocmacov"], 4)
    assert [w for w, _ in req] == ["tocmacov"]
    # both name words distinctive => two required words
    req2 = R._required_word_groups(q, None, hs["tocmacov"] | hs["laura"], 4)
    assert [w for w, _ in req2] == ["laura", "tocmacov"]
    # nothing distinctive => no requirement at all (verbose-query recall guard)
    assert R._required_word_groups(q, None, set(), 4) == []
