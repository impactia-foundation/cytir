"""ADR-062 Phase-2 (country knob) — SERVING off-country classifier tests.

The Phase-2 off-country penalty MAGNITUDE knob (spec §1 knob 12) may only demote a
candidate the serving off-country *decision* (`_off_country_hit` / `_country_is_binding`
in cytir/serve/geo.py, over `cytir.core.country.is_location_bound`) says is an off-country hit.
ADR-053's standing rule — "never gate a payload rule on an ENUMERATION of observed
source_type values; use a prefix/predicate that fails safe" — is what keeps that
decision correct across prod's OPEN 25-value source_type vocabulary. task 259 already
made the pipeline stamp (`aw_country.is_location_bound`) prefix-based and
pinned it with `test_aw_country.py`; this file extends the SAME discipline to the three
SERVING functions, exercised over ALL 25 prod source_type classes plus fail-safe
future/edge values, so the knob-12 penalty can never silently touch (or silently miss)
a class nobody enumerated.

The functions live in `geo.py` (a light module: no qdrant_client, no collection), so this
test imports them.

Run: python3 tests/test_offcountry_serving.py   (or via pytest)
"""
import pathlib


def _settings_ns() -> dict:
    """The typed settings the sliced code reads (S10 step 2, task 1071), loaded NOW: a fresh
    `settings.py` (a leaf module) parses the env this test just set, as an import would."""
    import importlib.util
    import sys as _sys
    path = pathlib.Path(__file__).resolve().parent.parent / "cytir" / "serve" / "settings.py"
    spec = importlib.util.spec_from_file_location("_aw_settings_for_slice", path)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return {"_S": mod.SETTINGS, "_settings": mod}

# The full source_type vocabulary observed on prod `aw_prod` AND twin `aw_v4_sat`
# (25; the vocabularies converged 2026-07-19 after the twin rebuild-from-prod).
# Counts (twin facet 2026-07-19) in the comment show the low-count tail ADR-053
# warns about: gov-stat 175713 vs gov-stats 51; official 2.86M vs official-docs 63 /
# official-news 2; gov-health 7; gov-law 3; gov-court 1; model-hub 2.
PROD_SOURCE_TYPES = [
    "reference", "official", "gov-geo", "docs", "entity_card", "gov-stat", "blog",
    "community-qa", "news", "central-bank", "tech", "health", "hn-discussion", "gov",
    "gov-tax", "case_law", "registry_record", "edu",
    "official-docs", "gov-stats", "gov-health", "gov-law", "model-hub",
    "official-news", "gov-court",
]

# Location-bound families (per policy): official* / gov* / central-bank / case_law /
# registry_record. Everything else is general (reference/docs/blog/news/tech/health/
# community-qa/hn-discussion/edu/entity_card/model-hub) and is NOT location-bound.
EXPECT_LOCBOUND = {
    "reference": False, "official": True, "gov-geo": True, "docs": False,
    "entity_card": False, "gov-stat": True, "blog": False, "community-qa": False,
    "news": False, "central-bank": True, "tech": False, "health": False,
    "hn-discussion": False, "gov": True, "gov-tax": True, "case_law": True,
    "registry_record": True, "edu": False,
    "official-docs": True, "gov-stats": True, "gov-health": True, "gov-law": True,
    "model-hub": False, "official-news": True, "gov-court": True,
}


def _load_serving_classifier():
    """The serving classifier: the off-country gate and its binding test live in
    `aw_server.geo` (S10 step 4, task 1095), the location-bound test in `cytir.core.country`
    (D7). `geo` is light (settings, signals, the core module), so import it, not a source slice."""
    import sys
    root = pathlib.Path(__file__).resolve().parent.parent
    for d in (root, root / "server"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    from cytir.core.country import is_location_bound
    from cytir.serve import geo
    return {"_is_location_bound": is_location_bound,
            "_country_is_binding": geo._country_is_binding,
            "_off_country_hit": geo._off_country_hit}


NS = _load_serving_classifier()


def test_is_location_bound_all_25():
    """Serving `_is_location_bound` classifies every prod source_type per policy —
    prefix/family match, not an enumeration (ADR-053)."""
    f = NS["_is_location_bound"]
    bad = [(st, f(st), EXPECT_LOCBOUND[st]) for st in PROD_SOURCE_TYPES
           if f(st) != EXPECT_LOCBOUND[st]]
    assert not bad, f"location-bound mismatches: {bad}"
    # the low-count tail ADR-053 flags explicitly, by prefix (not enumerated):
    for st in ("gov-stats", "gov-health", "gov-law", "gov-court", "official-docs",
               "official-news"):
        assert f(st) is True, st
    assert f("model-hub") is False           # not gov-/official- -> general
    # fail-safe: an unseen gov-*/official-* class is location-bound by default.
    for st in ("gov-transport", "official-register", "gov-energy-v2"):
        assert f(st) is True, st
    # edges
    assert f(None) is False and f("") is False and f("   ") is False
    assert f("  GOV-LAW  ") is True          # normalised (strip+lower)
    assert f("government") is False and f("officialdom") is False


def test_country_is_binding_all_25():
    """`_country_is_binding` == location-bound OR a REGISTERED entity_card. A
    register-LESS card is not binding; a registered one is (task-240 shoptiques)."""
    f = NS["_country_is_binding"]
    for st in PROD_SOURCE_TYPES:
        if st == "entity_card":
            assert f(st, has_register=False) is False, st
            assert f(st, has_register=True) is True, st
        else:
            assert f(st, has_register=False) is EXPECT_LOCBOUND[st], st
            # has_register only ever matters for entity_card
            assert f(st, has_register=True) is EXPECT_LOCBOUND[st], st


def test_off_country_hit_over_25():
    """`_off_country_hit` (the gate the magnitude knob multiplies) over all 25, on a
    CH-scoped query with a FR-stamped candidate. Invariants:
      * unscoped (q_country falsy)         -> never fires (0.0)
      * source_type == "official"          -> 0.0 (that lane is foreign_gov's, not
                                              the off-country knob) — EXACT match
      * exempt / in-country / unstamped cty -> 0.0
      * otherwise a demotable foreign scope -> 1.0 (the knob-12 penalty applies)
    Default flags (LOCBOUND_ONLY off): every NON-official class with a demotable
    conflicting stamp is a hit — location-bound or not — which is exactly today's
    live behaviour (ADR-056 exempts only `reference` via a SEPARATE graded weight,
    not this function).
    """
    f = NS["_off_country_hit"]
    # unscoped: never fires, whatever the candidate is
    for st in PROD_SOURCE_TYPES:
        assert f(None, "FR", st) == 0.0, st
        assert f("", "FR", st) == 0.0, st
    # CH-scoped, FR-stamped candidate:
    for st in PROD_SOURCE_TYPES:
        hit = f("CH", "FR", st)
        if st == "official":
            assert hit == 0.0, f"official must ride foreign_gov, not off_country ({st})"
        else:
            assert hit == 1.0, f"{st}: expected off-country hit on FR!=CH"
    # in-country / exempt / unstamped candidate never fires (non-official)
    for cty in ("CH", "INTL", "EU", "", None):
        assert f("CH", cty, "gov-stat") == 0.0, cty
    # a non-scope-country stamp (IE) is NOT demotable under default flags
    assert f("CH", "IE", "gov") == 0.0
    # eu-supra content is exempt when the scope is an EU member (task 164)
    assert f("FR", "DE", "gov", is_eu_supra=True) == 0.0
    # ...but NOT exempt when the scope is a non-member (CH): DE stays demotable
    assert f("CH", "DE", "gov", is_eu_supra=True) == 1.0


def test_official_family_lane_split_is_documented():
    """DELIBERATE asymmetry the knob-12 gate must be aware of: the foreign_gov lane
    exclusion in `_off_country_hit` is an EXACT `== "official"` test, while
    location-bindingness is a `official-*` PREFIX family. So `official-docs` /
    `official-news` (65 records total) are location-bound BUT are NOT excluded from
    the off-country lane — they ride the knob-12 penalty (-2.0), not foreign_gov
    (-1.0). This is intentional (they are foreign location-bound content and the
    stronger demotion is appropriate) and is pinned here so a future 'simplify the
    lane split' edit is a conscious decision, not a silent behaviour change."""
    f = NS["_off_country_hit"]
    assert f("CH", "FR", "official") == 0.0          # exact 'official' -> foreign_gov
    assert f("CH", "FR", "official-docs") == 1.0     # official-* -> off_country lane
    assert f("CH", "FR", "official-news") == 1.0


if __name__ == "__main__":
    test_is_location_bound_all_25()
    test_country_is_binding_all_25()
    test_off_country_hit_over_25()
    test_official_family_lane_split_is_documented()
    print("all serving off-country classifier tests passed (25 source_types)")
