"""Task 404 — focused tests for the two epoch-3 displacement levers.

(a) AW_NAMEGATE  — person-query wrong-person reference-bio damping (names/n01)
(b) AW_REFCANON  — reference-above-canonical demotion on authority-lookup
                   intent (403 §8.4 eviction-by-worse class)

The pure functions are exec'd out of ``cytir/serve/retrieval.py`` without
importing the Qdrant-serving module (pattern: test_learned_only_scope.py).
Env is set BEFORE the chunks are exec'd so the module-level flag constants
resolve to the tested configuration.

Run: ``python3 tests/test_task404_levers.py`` (or via pytest).
"""
import os
import pathlib
import re




def _load(env=None, decompound=None):
    """Import the serving modules under `env` and return the `retrieval` namespace (S10 step 3,
    task 1083: the lever code lives in `query.py`/`levers.py` now, so the tests import it instead
    of slicing source). Settings are read at import, so the `aw_server` modules are purged first
    and again after, and a fresh import parses the env this test just set. `decompound` replaces
    the German splitter in `textmatch` (the real one needs an optional package)."""
    import importlib
    import sys
    root = pathlib.Path(__file__).resolve().parent.parent
    for p in (str(root), str(root / "server")):
        if p not in sys.path:
            sys.path.insert(0, p)

    def _purge():
        for m in [m for m in sys.modules if m == "cytir.serve" or m.startswith("cytir.serve.")]:
            del sys.modules[m]

    saved = {}
    for k, v in (env or {}).items():
        saved[k] = os.environ.get(k)
        os.environ[k] = v
    _purge()
    try:
        r = importlib.import_module("cytir.serve.retrieval")
        sys.modules["cytir.serve.textmatch"]._decompound = decompound or (lambda t: [])
        return vars(r)
    finally:
        _purge()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


NS = _load({"AW_NAMEGATE": "1", "AW_NAMEGATE_W": "-2.5",
            "AW_REFCANON": "1", "AW_REFCANON_W": "-2.0"})


def _q(ns, q, lang=None):
    return ns["_content_terms"](q, lang)


# ---------------------------------------------------------------- (a) shape --

def test_person_shape_positives():
    for q in ("Tim O'Hear", "Timothy O'Hear", "who was Ada Lovelace",
              "Marie Curie", "Laura Tocmacov"):
        assert NS["_person_name_query"](q, _q(NS, q)), q


def test_person_shape_negatives():
    for q in ("fondation impactIA",                      # lowercase content
              "newest Mistral AI model announcement",    # lowercase content
              "was regelt Artikel 257 des Schweizer Obligationenrechts zur Miete",
              "zurich insurance group versicherung schweiz sitz zürich",
              "R2 D2 unit 42",                           # digits
              "Einstein",                                # single term
              "what is the CAP theorem in distributed systems"):
        assert not NS["_person_name_query"](q, _q(NS, q)), q


def _nt(ns, q):
    return ns["_person_name_tokens"](q, _q(ns, q))


def test_person_name_tokens_are_the_name_only():
    assert _nt(NS, "Tim O'Hear") == {"tim", "ohear"}
    # interrogative filler is NOT part of the name (coverage basis)
    assert _nt(NS, "who was Ada Lovelace") == {"ada", "lovelace"}
    assert _nt(NS, "newest Mistral AI model announcement") is None


def test_name_token_coverage_full_vs_partial():
    qt = _nt(NS, "Tim O'Hear")             # {tim, ohear}
    cov = NS["_name_token_coverage"]
    assert cov(qt, {"title": "Tim O'Hear", "snippet": ""}) == 1.0
    # the six n01 flood shapes: exactly one name part each
    assert cov(qt, {"title": "Anthony O'Hear", "snippet": ""}) == 0.5
    assert cov(qt, {"title": "Tim O'Connor", "snippet": ""}) == 0.5
    assert cov(qt, {"title": "Tim Duquette", "snippet": ""}) == 0.5
    # gold shape: name in the SNIPPET, not the title
    assert cov(qt, {"title": "Conseil de fondation - impactIA",
                    "snippet": "… Tim O'Hear, président …"}) == 1.0
    # token equality, not substring: 'estimation' must not satisfy 'tim'
    assert cov(qt, {"title": "estimation ohear", "snippet": ""}) == 0.5
    assert cov(qt, {"title": "", "snippet": ""}) == 0.0
    # "who was Ada Lovelace" — the RIGHT bio fully covered => exempt band
    qt2 = _nt(NS, "who was Ada Lovelace")  # {ada, lovelace}
    assert cov(qt2, {"title": "Ada Lovelace", "snippet": ""}) == 1.0
    assert cov(qt2, {"title": "Lovelace (film)", "snippet": ""}) == 0.5


def test_apply_namegate_n01_shape():
    # pool: gold (full name in snippet), 3 half-match floods, 1 zero-coverage
    apply = NS["_apply_namegate"]
    qt = _nt(NS, "Tim O'Hear")
    cov = NS["_name_token_coverage"]
    rows = [["gold", {"title": "Conseil de fondation - impactIA",
                      "snippet": "Tim O'Hear, président"}, 4.0, 0.0, 1.0],
            ["f1", {"title": "Anthony O'Hear", "snippet": ""}, 5.0, 0.0, 0.7],
            ["f2", {"title": "Tim O'Connor", "snippet": ""}, 4.5, 0.0, 0.7],
            ["f3", {"title": "Tim Duquette", "snippet": ""}, 4.4, 0.0, 0.7],
            ["z", {"title": "Ohne Namen", "snippet": ""}, 3.0, 0.0, 0.9]]
    name_cov = {r[0]: cov(qt, r[1]) for r in rows}
    demoted = apply(rows, name_cov, True)
    # v3: zero-coverage is demoted too (the measured aw_e3 reservoir of
    # near-name bios re-filled the top-10 through the v2 zero exemption)
    assert demoted == ["f1", "f2", "f3", "z"]
    assert [r[3] for r in rows] == [0.0, -2.5, -2.5, -2.5, -2.5]


def test_apply_namegate_inert_without_full_coverage_candidate():
    apply = NS["_apply_namegate"]
    rows = [["f1", {"title": "Anthony O'Hear"}, 5.0, 0.0, 0.7],
            ["f2", {"title": "Tim O'Connor"}, 4.5, 0.0, 0.7]]
    name_cov = {"f1": 0.5, "f2": 0.5}
    assert apply(rows, name_cov, True) == []
    assert [r[3] for r in rows] == [0.0, 0.0]


def test_apply_namegate_inert_when_inactive():
    apply = NS["_apply_namegate"]
    rows = [["gold", {"title": "Tim O'Hear"}, 4.0, 0.0, 1.0],
            ["f1", {"title": "Anthony O'Hear"}, 5.0, 0.0, 0.7]]
    assert apply(rows, {"gold": 1.0, "f1": 0.5}, False) == []
    assert [r[3] for r in rows] == [0.0, 0.0]


# ------------------------------------------------------------- (b) canonical --

def test_refcanon_canonical_classes():
    can = NS["_refcanon_canonical"]
    for st in ("official", "gov", "gov-law", "gov-court", "official-docs",
               "case_law", "central-bank", "registry_record"):
        assert can(st), st
    for st in ("reference", "docs", "news", "blog", "entity_card",
               "community-qa", None, ""):
        assert not can(st), st


def test_refcanon_extra_classes_env():
    ns = _load({"AW_REFCANON": "1", "AW_REFCANON_W": "-2.0",
                "AW_REFCANON_EXTRA_CLASSES": "docs, model-hub"})
    assert ns["_refcanon_canonical"]("docs")
    assert ns["_refcanon_canonical"]("model-hub")
    assert not ns["_refcanon_canonical"]("news")


# ---------------------------------------------------------- (b) application --

def _rows(*specs):
    """[(source_type, rel_core, title, names)] -> mutable rows list."""
    return [[f"id{i}", {"source_type": st, "title": t, "names": nm},
             rel, 0.0, 0.9]
            for i, (st, rel, t, nm) in enumerate(specs)]


def _pen(rows):
    return [r[3] for r in rows]


def test_refcanon_fires_on_authority_intent_with_canonical_in_pool():
    # the ca-law-005 shape: wiki (reference) rel-strong, SCC (official) competitive
    rows = _rows(("reference", 5.0, "R v Oakes", ""),
                 ("official", 4.2, "R. v. Oakes, [1986] 1 SCR 103", ""),
                 ("news", 3.0, "Oakes test explained", ""))
    demoted = NS["_apply_refcanon"](rows, _q(NS, "R v Oakes decision"), None,
                                    head_on=True, rh_beta=0.05,
                                    official_pref_fired=False)
    assert demoted == ["id0"]
    assert _pen(rows) == [-2.0, 0.0, 0.0]


def test_refcanon_inert_on_general_intent():
    rows = _rows(("reference", 5.0, "Photosynthesis", ""),
                 ("official", 4.5, "Some ministry page", ""))
    n = NS["_apply_refcanon"](rows, _q(NS, "what is photosynthesis"), None,
                              head_on=True, rh_beta=0.6,
                              official_pref_fired=False)
    assert n == [] and _pen(rows) == [0.0, 0.0]


def test_refcanon_inert_when_head_off():
    # head off => rh_beta==0 would LOOK like authority intent; must be inert.
    rows = _rows(("reference", 5.0, "R v Oakes", ""),
                 ("official", 4.2, "R. v. Oakes", ""))
    n = NS["_apply_refcanon"](rows, _q(NS, "R v Oakes"), None,
                              head_on=False, rh_beta=0.0,
                              official_pref_fired=False)
    assert n == [] and _pen(rows) == [0.0, 0.0]


def test_refcanon_inert_without_competitive_canonical():
    rows = _rows(("reference", 5.0, "Direct democracy", ""),
                 ("official", 1.0, "Weakly related page", ""))  # < 0.7 * 5.0
    n = NS["_apply_refcanon"](rows, _q(NS, "direct democracy definition"), None,
                              head_on=True, rh_beta=0.05,
                              official_pref_fired=False)
    assert n == [] and _pen(rows) == [0.0, 0.0]


def test_refcanon_exempt_on_card_named_entity_query():
    # the dom_entity shape: wiki gold at 1, official homepage right behind,
    # a name-matched entity card in-pool => the card lane owns it, lever inert.
    q = _q(NS, "roche holding pharma unternehmen sitz basel", "de")
    rows = _rows(("reference", 5.0, "Roche Holding", ""),
                 ("official", 4.6, "Roche — Standorte Basel", ""),
                 ("entity_card", 4.0, "Roche Holding AG", "Roche|Hoffmann-La Roche"))
    n = NS["_apply_refcanon"](rows, q, "de", head_on=True, rh_beta=0.05,
                              official_pref_fired=False)
    assert n == [] and _pen(rows) == [0.0, 0.0, 0.0]


def test_card_alias_containment():
    f = NS["_card_alias_in_query"]
    q = _q(NS, "roche holding pharma unternehmen sitz basel", "de")
    assert f(q, "Roche Holding AG", "Roche|Hoffmann-La Roche", "de")  # "Roche"
    q2 = _q(NS, "Supreme Court of Canada R v Oakes decision section 1 Charter")
    assert not f(q2, "Swisscom AG", "Swisscom", None)
    # partial alias overlap is NOT containment
    q3 = _q(NS, "sandoz history basel")
    assert not f(q3, "Novartis AG", "Novartis|Novartis International", None)


def test_refcanon_defers_to_official_pref():
    rows = _rows(("reference", 5.0, "BGB 536", ""),
                 ("official", 4.8, "Bürgerliches Gesetzbuch — § 536", ""))
    n = NS["_apply_refcanon"](rows, _q(NS, "Miete mindern BGB", "de"), "de",
                              head_on=True, rh_beta=0.05,
                              official_pref_fired=True)
    assert n == [] and _pen(rows) == [0.0, 0.0]


def test_refcanon_inert_at_default_zero_weight():
    ns = _load({"AW_REFCANON": "1"})   # W stays 0.0 default
    rows = _rows(("reference", 5.0, "R v Oakes", ""),
                 ("official", 4.2, "R. v. Oakes", ""))
    n = ns["_apply_refcanon"](rows, _q(ns, "R v Oakes"), None,
                              head_on=True, rh_beta=0.05,
                              official_pref_fired=False)
    assert n == [] and _pen(rows) == [0.0, 0.0]


def test_namegate_flags_inert_by_default():
    ns = _load({})  # both flags off
    assert ns["_NAMEGATE"] is False and ns["_REFCANON"] is False
    assert ns["_NAMEGATE_W"] == 0.0 and ns["_REFCANON_W"] == 0.0


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    raise SystemExit(1 if fails else 0)
