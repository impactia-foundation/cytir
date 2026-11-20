"""Task 406 — focused tests for the intent-conditioned person-name legs.

Tim's design ruling (2026-08-13): "our 'intent' head either at search time or
re-ranking reduces the relevance of dense and increases the importance of all
terms being in the sparse."

  H1  AW_NAMEDENSE / _SCALE      dense leg scaled down under person-name intent
  H2  AW_NAMECOV_W               rel_core bonus for FULL name-token coverage
  H3  AW_NAMEGATE_TRUSTKEEP      full-coverage keeps its trust floor
      AW_TRUSTFLOOR_UNDEMOTED    ratio exemption vs best NON-DEMOTED rel_core
  H5  AW_NAMEGATE_SHAPE_V2       German-noun-capitalisation FP cure

The pure functions are exec'd out of ``cytir/serve/retrieval.py`` without
importing the Qdrant-serving module (same pattern as test_task404_levers.py).
Env is set BEFORE the chunks are exec'd so module-level flag constants resolve.

Run: ``python3 tests/test_task406_levers.py`` (or via pytest).
"""
import os
import pathlib
import re



# The German de-compounder used by shape-v2 rule 2. The real one is the ML
# `char_split` splitter behind an lru_cache (schema.py) and only fires at
# len >= 12; the stub reproduces its DECISION for the tokens under test so the
# unit test does not depend on the optional `compound_split` package being
# installed. The live behaviour is measured on the gate, not asserted here.
_DECOMPOUNDABLE = {"schwarzenegger": ("schwarzen", "egger"),
                   "kuendigungsfrist": ("kuendigung", "frist"),
                   "arbeitsvertrag": ("arbeit", "vertrag"),
                   "volkswirtschaft": ("volk", "wirtschaft")}


def _load(env=None, decompound=True):
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
        sys.modules["cytir.serve.textmatch"]._decompound = ((lambda t: _DECOMPOUNDABLE.get(t, ())) if decompound else (lambda t: ()))
        return vars(r)
    finally:
        _purge()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# The gate arm: 404 penalty + both 406 halves + the H3/H5 riders.
ON = {"AW_NAMEGATE": "1", "AW_NAMEGATE_W": "-2.5",
      "AW_NAMEDENSE": "1", "AW_NAMEDENSE_SCALE": "0.4",
      "AW_NAMECOV_W": "0.5", "AW_NAMEGATE_TRUSTKEEP": "1",
      "AW_TRUSTFLOOR_UNDEMOTED": "1", "AW_NAMEGATE_SHAPE_V2": "1"}
NS_V2 = _load(ON)
NS_V1 = _load({"AW_NAMEGATE": "1", "AW_NAMEGATE_W": "-2.5"})   # 404 shape v1


def _q(ns, q, lang=None):
    return ns["_content_terms"](q, lang)


def _nt(ns, q):
    return ns["_person_name_tokens"](q, _q(ns, q))


# ------------------------------------------------------------- inertness ----

def test_all_406_flags_inert_by_default():
    ns = _load({})
    assert ns["_NAMEDENSE"] is False
    assert ns["_NAMEDENSE_SCALE"] == 1.0     # multiplicative identity
    assert ns["_NAMECOV_W"] == 0.0
    assert ns["_NAMEGATE_TRUSTKEEP"] is False
    assert ns["_TRUSTFLOOR_UNDEMOTED"] is False
    assert ns["_NAMEGATE_SHAPE_V2"] is False
    # nothing arms the shape detector when every lever is off
    assert ns["_NAME_LEVERS_ARMED"] is False


def test_406_halves_arm_the_detector_without_the_404_penalty():
    """A 406 half must run the detector even with AW_NAMEGATE off — otherwise
    the halves could only ever be measured on top of the 404 penalty."""
    for half in ({"AW_NAMEDENSE": "1", "AW_NAMEDENSE_SCALE": "0.4"},
                 {"AW_NAMECOV_W": "0.5"},
                 {"AW_NAMEGATE_TRUSTKEEP": "1"}):
        ns = _load(half)
        assert ns["_NAME_LEVERS_ARMED"] is True, half
        assert ns["_NAMEGATE"] is False, half   # the 404 penalty stays off


def test_namedense_scale_1_is_not_armed():
    """AW_NAMEDENSE=1 with the identity scale is inert, not armed."""
    ns = _load({"AW_NAMEDENSE": "1", "AW_NAMEDENSE_SCALE": "1.0"})
    assert ns["_NAME_LEVERS_ARMED"] is False


# ---------------------------------------------------- H5 shape v2 (FP cure) --

def test_shape_v1_reproduces_the_measured_german_false_positive():
    """The defect being cured (404 §5.2): bb-008 reads as a person name under
    v1, which is why it went 4->8 under AW_NAMEGATE."""
    assert _nt(NS_V1, "was ist Inflation in der Volkswirtschaft") is not None


def test_dom_de_01_was_never_a_detector_false_positive():
    """404 §5.2 REFUTED (the task-406 shape-census artifact, run with the live
    `compound_split` splitter): 'Kuendigungsfrist Arbeitsvertrag BGB'
    de-compounds to 5 content terms, over AW_NAMEGATE_MAXTERMS=4, so v1 already
    returns None and the namegate never fired on it. Its 5->6 has another
    cause. Pinned so the wrong attribution cannot be re-derived from prose."""
    q = "Kuendigungsfrist Arbeitsvertrag BGB"
    # with the real decompounder's decision reproduced by the stub
    assert len(_q(NS_V1, q)) > 4
    assert _nt(NS_V1, q) is None
    assert _nt(NS_V2, q) is None


def test_shape_v2_rejects_the_german_false_positive():
    # bb-008: capitalised tokens separated by "in der" -> not a contiguous name
    # ("in" is not a nobiliary particle)
    assert _nt(NS_V2, "was ist Inflation in der Volkswirtschaft") is None


def test_shape_v2_rejection_rules_are_independently_sufficient():
    # contiguity alone (no acronym)
    assert _nt(NS_V2, "Steuer bei der Rente") is None
    # acronym alone (measured drops: ABB Schweiz / ETH Zürich / AHV … / … BGB)
    assert _nt(NS_V2, "Datenschutz DSGVO") is None
    assert _nt(NS_V2, "ABB Schweiz") is None
    assert _nt(NS_V2, "ETH Zürich") is None


def test_shape_v2_keeps_nobiliary_particle_names():
    """The contiguity rule must not cost a true name: the ONLY permitted infix
    is a nobiliary particle. zero_signal/gd-14 is the measured case."""
    assert _nt(NS_V2, "who is Ursula von der Leyen") == {"ursula", "leyen"}
    assert _nt(NS_V2, "Ludwig van Beethoven") == {"ludwig", "beethoven"}
    assert _nt(NS_V2, "Charles de Gaulle") == {"charles", "gaulle"}


def test_decompound_rule_was_dropped_on_the_measurement():
    """A candidate third rule (reject de-compoundable tokens) is NOT in the
    detector: it drops zero extra gate queries and would disarm the lever on
    long true surnames, since `_decompound('schwarzenegger')` splits."""
    assert "schwarzenegger" in _DECOMPOUNDABLE   # the real splitter does split it
    assert _nt(NS_V2, "Arnold Schwarzenegger") == {"arnold", "schwarzenegger"}


def test_shape_v2_keeps_every_person_positive():
    """The FP cure must not cost a single true person-name query — including
    the encyclopedic-person class the guard depends on."""
    for q in ("Tim O'Hear", "Timothy O'Hear", "Laura Tocmacov",
              "who was Ada Lovelace", "Marie Curie", "Alan Turing",
              "Albert Einstein", "Rosalind Franklin", "Niklaus Wirth"):
        assert _nt(NS_V2, q) is not None, q
    assert _nt(NS_V2, "Tim O'Hear") == {"tim", "ohear"}
    assert _nt(NS_V2, "who was Ada Lovelace") == {"ada", "lovelace"}


def test_shape_v2_keeps_the_v1_negatives_negative():
    for q in ("fondation impactIA", "newest Mistral AI model announcement",
              "R2 D2 unit 42", "Einstein",
              "what is the CAP theorem in distributed systems"):
        assert _nt(NS_V2, q) is None, q


# --------------------------------------------- the encyclopedic-person guard --

def test_encyclopedic_person_wiki_bio_has_full_coverage():
    """The property the whole design leans on: on 'Marie Curie' / 'Alan Turing'
    the encyclopedic page NAMES the person, so it takes the H2 bonus and is
    exempt from the 404 penalty, while the dense down-weight strips only the
    half-match noise floor around it."""
    cov = NS_V2["_name_token_coverage"]
    qt = _nt(NS_V2, "Marie Curie")
    assert cov(qt, {"title": "Marie Curie", "snippet": "Polish-French physicist"}) == 1.0
    # the noise floor the dense leg floats on a bare person query
    assert cov(qt, {"title": "Pierre Curie", "snippet": ""}) == 0.5
    assert cov(qt, {"title": "Marie Antoinette", "snippet": ""}) == 0.5
    assert cov(qt, {"title": "Curie temperature", "snippet": ""}) == 0.5
    qt = _nt(NS_V2, "Alan Turing")
    assert cov(qt, {"title": "Alan Turing", "snippet": ""}) == 1.0
    assert cov(qt, {"title": "Turing machine", "snippet": ""}) == 0.5


# ------------------------------------- H4 refcanon trigger distinctiveness ---

def _rc_rows(*specs):
    """(source_type, rel_core, title) -> the [pid, payload, rel, pen, trust]
    row shape _apply_refcanon mutates."""
    return [[i, {"source_type": st, "title": ti, "names": "", "url": ""},
             rel, 0.0, 0.0] for i, (st, rel, ti) in enumerate(specs)]


def _pens(rows):
    return [r[3] for r in rows]


def test_refcanon_mincent_blocks_an_offtopic_canonical_trigger():
    """The MEASURED 404 collateral (gate-e3-after-404rc.json): on dom_entity/
    en01 the trigger that demoted the Nestlé wiki gold was an unrelated
    cantonal statute, bgs.so.ch/texts_of_law/624.512, which cleared β=0.35 on
    relevance alone. With MINCENT the trigger must also own the query's tokens."""
    ns = _load({"AW_REFCANON": "1", "AW_REFCANON_W": "-2.0",
                "AW_REFCANON_BETA": "0.35", "AW_REFCANON_MINCENT": "0.10"})
    q = ns["_content_terms"]("Nestlé Konzern Schweiz", "de")
    rows = _rc_rows(("reference", 5.0, "Nestlé"),
                    ("official", 2.0, "Gesetz über die Staatsverwaltung 624.512"))
    assert ns["_apply_refcanon"](rows, q, "de", head_on=True, rh_beta=0.05,
                                 official_pref_fired=False) == []
    assert _pens(rows) == [0.0, 0.0]


def test_refcanon_mincent_still_fires_on_an_ontopic_canonical():
    """The lever's REASON for existing must survive its own fix: a canonical
    page whose title owns the query still demotes the encyclopedia."""
    ns = _load({"AW_REFCANON": "1", "AW_REFCANON_W": "-2.0",
                "AW_REFCANON_BETA": "0.35", "AW_REFCANON_MINCENT": "0.10"})
    q = ns["_content_terms"]("R v Oakes", None)
    rows = _rc_rows(("reference", 5.0, "R v Oakes"),
                    ("case_law", 2.0, "R. v. Oakes - SCC Decisions"))
    assert ns["_apply_refcanon"](rows, q, None, head_on=True, rh_beta=0.05,
                                 official_pref_fired=False) == [0]
    assert _pens(rows) == [-2.0, 0.0]


def test_refcanon_mincent_zero_is_the_404_behaviour():
    ns = _load({"AW_REFCANON": "1", "AW_REFCANON_W": "-2.0",
                "AW_REFCANON_BETA": "0.35"})
    assert ns["_REFCANON_MINCENT"] == 0.0
    q = ns["_content_terms"]("Nestlé Konzern Schweiz", "de")
    rows = _rc_rows(("reference", 5.0, "Nestlé"),
                    ("official", 2.0, "Gesetz über die Staatsverwaltung 624.512"))
    assert ns["_apply_refcanon"](rows, q, "de", head_on=True, rh_beta=0.05,
                                 official_pref_fired=False) == [0]


# ------------------------------------------------- H1/H2 design arithmetic ---
#
# Reproduces the MEASURED n01 no-lang breakdown (405 diag receipt) through the
# same arithmetic the serving loop uses, so the design's claimed mechanism is
# pinned as a number rather than as prose. It is a model of the fusion step, NOT
# a test of the deployed system — the real attribution is the gate.
#
#   Tim_Tim (Austrian reggae band)  rel 2.307  pen −2.50  trust +1.75  cov 0.50
#   gold impactia/conseil-de-fondation  rel 1.289  pen 0  trust 0  cov 1.00
# The gold reaches the pool via BM25 only (dense_norm 0); Tim_Tim scores both.

TIMTIM = {"rel": 2.307, "pen": -2.50, "trust_full": 1.75, "cov": 0.5,
          "dense_norm": 0.90}
GOLD = {"rel": 1.289, "pen": 0.0, "trust_full": 1.75, "cov": 1.0,
        "dense_norm": 0.0}
W_DENSE = 1.0          # fusion_weights.json "dense"
REL_BETA = 0.7         # AW_TRUSTFLOOR_REL_BETA


def _fused(row, dense_scale, cov_w, trustkeep, best_rel, undemoted_best):
    rel = row["rel"] - (1.0 - dense_scale) * W_DENSE * row["dense_norm"]
    if row["cov"] >= 0.999:
        rel += cov_w
    # trustfloor: a dense-absent row keeps its trust only via the ratio
    # exemption (or, with H3(i), via full coverage)
    ratio_base = undemoted_best if undemoted_best else best_rel
    exempt = (trustkeep and row["cov"] >= 0.999) or rel >= REL_BETA * ratio_base
    trust = row["trust_full"] if (exempt or row["dense_norm"] > 0) else 0.0
    return rel + row["pen"] + trust, rel


def _pair(dense_scale=1.0, cov_w=0.0, trustkeep=False, undemoted=False):
    _, rel_t = _fused(TIMTIM, dense_scale, cov_w, trustkeep, 1.0, 0.0)
    _, rel_g = _fused(GOLD, dense_scale, cov_w, trustkeep, 1.0, 0.0)
    best = max(rel_t, rel_g)
    # every sub-full-coverage row is namegate-demoted, so with H3(ii) the only
    # undemoted row is the full-coverage gold
    ub = rel_g if undemoted else best
    return (_fused(TIMTIM, dense_scale, cov_w, trustkeep, best, ub)[0],
            _fused(GOLD, dense_scale, cov_w, trustkeep, best, ub)[0])


def test_baseline_reproduces_the_measured_loss():
    """Levers off: the receipt's f=1.557 vs f=1.289, gold loses by 0.268."""
    t, g = _pair()
    assert abs(t - 1.557) < 0.002 and abs(g - 1.289) < 0.002
    assert t > g and abs((t - g) - 0.268) < 0.002


def test_h1_alone_flips_it_and_the_threshold_is_where_predicted():
    """Tim's prediction: dense down-weighting lowers the two-leg flooder's
    relevance and leaves the BM25-only gold's untouched, so the gold wins once
    it comes within the flooder's −0.75 non-relevance contribution."""
    assert _pair(dense_scale=1.0)[0] > _pair(dense_scale=1.0)[1]   # no flip
    t, g = _pair(dense_scale=0.4)
    assert g > t, (t, g)
    # the flip needs (1-s)*W*dense_norm > 0.268 -> s < ~0.702
    t, g = _pair(dense_scale=0.75)
    assert t > g, "s=0.75 removes too little dense to flip — bound holds"


def test_h2_alone_flips_it_via_the_trustfloor_exemption():
    """The coverage bonus lands in rel_core, so it also lifts the gold's rel
    ratio over the 0.7 trustfloor exemption and restores the trust the 405
    defect stripped — which is why H2 is a rel_core term, not a penalty."""
    t, g = _pair(cov_w=0.5)
    assert g > t, (t, g)


def test_h3_undemoted_best_alone_restores_the_gold_trust():
    """405 fix candidate (ii): the demoted wiki row must not set the bar."""
    t, g = _pair(undemoted=True)
    assert g > t, (t, g)


def test_h3_trustkeep_alone_restores_the_gold_trust():
    """405 fix candidate (i): the page proven to name the person keeps its
    floor."""
    t, g = _pair(trustkeep=True)
    assert g > t, (t, g)


def test_halves_compose_without_overshooting():
    t, g = _pair(dense_scale=0.4, cov_w=0.5, trustkeep=True, undemoted=True)
    assert g > t
    # the flooder is demoted, never negatively amplified beyond its own legs
    assert t > TIMTIM["pen"]


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
