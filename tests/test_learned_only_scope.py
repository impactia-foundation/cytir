"""Focused tests for the inert learned-only free-text country-scope arm.

The pure policy functions are executed from ``retrieval.py`` without importing
the Qdrant-serving module.  This pins the experiment's two load-bearing rules:
keyword/gazetteer output cannot control free-text scope in learned-only mode,
and the learned conflict signal covers official records while preserving the
EU/INTL/unstamped exemptions.

Run: ``python3 tests/test_learned_only_scope.py`` (or via pytest).
"""
import os
import pathlib


RETRIEVAL = (pathlib.Path(__file__).resolve().parent.parent
             / "cytir" / "serve" / "retrieval.py")


def _heads_namespace(**overrides) -> dict:
    """A copy of `aw_server.heads`' globals with `overrides` set, and the scope functions rebuilt
    against it (S10 step 5, task 1099: the heads moved out of `retrieval.py`, so the tests import
    them instead of slicing source). The real module is never mutated."""
    import sys
    import types
    root = pathlib.Path(__file__).resolve().parent.parent
    for d in (root, root / "server"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    from cytir.serve import heads
    ns = dict(vars(heads))
    ns.update(overrides)
    for name in ("_parse_country_scope_mode", "_scope_head_predict", "_resolve_scope",
                 "_effective_free_text_country_scope", "_country_scope_for_ranking"):
        f = getattr(heads, name)
        ns[name] = types.FunctionType(f.__code__, ns, name, f.__defaults__, f.__closure__)
    for name in overrides:
        ns[name] = overrides[name]
    return ns


def _load_resolver():
    # the pre-Task374 default for the mode flags reproduces `legacy`; each test sets what it needs
    return _heads_namespace(_POLICY_ONLY_SCOPE=False, _LEARNED_ONLY_SCOPE=False, _SCOPE_ACTIVE=False)


def _load_conflict_classifier():
    """`aw_server.geo` (S10 step 4, task 1095): the classifier moved out of `retrieval.py`."""
    import sys
    root = pathlib.Path(__file__).resolve().parent.parent
    for d in (root, root / "server"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    from cytir.serve import geo
    return {"_off_country_hit": geo._off_country_hit, "geo": geo}


def test_scope_mode_rejects_typos():
    ns = _heads_namespace()
    parse = ns["_parse_country_scope_mode"]
    assert parse(None) == "legacy"
    assert parse(" LEARNED_ONLY ") == "learned_only"
    try:
        parse("lerned_only")
    except ValueError as exc:
        assert "AW_COUNTRY_SCOPE_MODE" in str(exc)
    else:
        raise AssertionError("invalid country-scope mode silently accepted")


def test_learned_only_ignores_keyword_and_gazetteer():
    ns = _load_resolver()
    calls = []
    ns["_gazetteer_scope"] = lambda _q: calls.append("gazetteer") or ("CH", False)
    ns["_scope_head_predict"] = lambda _vec: ("DE", 0.68)

    # Even a conflicting keyword and gazetteer result cannot become deterministic
    # scope in the learned-only arm.
    ns["_LEARNED_ONLY_SCOPE"] = True
    ns["_SCOPE_ACTIVE"] = True
    assert ns["_effective_free_text_country_scope"](
        "German-law query", "FR", [0.0]
    ) == (None, "DE", 0.68)
    assert calls == []


def test_learned_only_head_failure_abstains_without_keyword_fallback():
    ns = _load_resolver()
    ns["_gazetteer_scope"] = lambda _q: ("CH", False)
    ns["_scope_head_predict"] = lambda _vec: (None, 0.0)

    ns["_LEARNED_ONLY_SCOPE"] = True
    ns["_SCOPE_ACTIVE"] = True
    assert ns["_effective_free_text_country_scope"](
        "query naming France", "FR", None
    ) == (None, None, 0.0)


def test_legacy_resolution_is_unchanged():
    ns = _load_resolver()
    ns["_gazetteer_scope"] = lambda _q: ("CA", False)
    ns["_scope_head_predict"] = lambda _vec: ("DE", 0.75)

    ns["_LEARNED_ONLY_SCOPE"] = False
    ns["_SCOPE_ACTIVE"] = True
    assert ns["_effective_free_text_country_scope"](
        "France", "FR", [0.0]) == ("FR", None, 1.0)
    assert ns["_effective_free_text_country_scope"](
        "Ontario", None, [0.0]) == ("CA", None, 1.0)


def test_soft_conflict_includes_official_but_preserves_scope_exemptions():
    hit = _load_conflict_classifier()["_off_country_hit"]

    # Legacy lane split is unchanged unless the experiment opts official in.
    assert hit("DE", "CH", "official") == 0.0
    assert hit("DE", "CH", "official", include_official=True) == 1.0

    # Cross-jurisdiction and unknown stamps never take a learned-scope penalty.
    for country in ("DE", "INTL", "EU", "", None):
        assert hit("DE", country, "official", include_official=True) == 0.0

    # EU institutional material remains authoritative for an EU-member query.
    assert hit("DE", "FR", "official", is_eu_supra=True,
               include_official=True) == 0.0
    # The exemption is not incorrectly extended to a non-member query.
    assert hit("CH", "FR", "official", is_eu_supra=True,
               include_official=True) == 1.0


def test_same_calibrated_conflict_helper_is_wired_in_both_stages():
    src = (RETRIEVAL.parent / "levers.py").read_text()          # stage 2 moved there (task 1125)
    legs_src = (RETRIEVAL.parent / "legs.py").read_text()          # stage 1 moved there (task 1116)
    helper = "_learned_scope_conflict_penalty("
    # Stage-1 pool scoring (legs.stage1_meta) + stage-2 final scoring (the definition lives in `geo`).
    assert legs_src.count(helper) == 1 and src.count(helper) == 1
    assert RETRIEVAL.read_text().count(helper) == 0
    stage1 = legs_src[legs_src.index("def stage1_meta("):legs_src.index("def run_legs(")]
    s2 = src.index("        # task 240: off-country demotion (geo-scoped country gate)")
    stage2 = src[s2:s2 + 3000]
    assert helper in stage1
    assert helper in stage2

    # Exercise the helper itself with the real conflict classifier but light
    # stubs for payload plumbing: confidence is applied exactly once.
    from unittest import mock
    geo = _load_conflict_classifier()["geo"]
    with mock.patch.multiple(
            geo, _effective_country=lambda pl: pl.get("country"),
            _is_eu_supra=lambda _value: False, _OFF_COUNTRY_NONBOUND_W=None,
            _OFF_COUNTRY_NONBOUND_CLASSES=frozenset(),
            _nonbound_class=lambda *_args: "general"):
        penalty = geo._learned_scope_conflict_penalty
        assert penalty("DE", 0.68, {
            "country": "CH", "source_type": "official", "domain": "fedlex.admin.ch"
        }, -2.0) == -1.36
        assert penalty("DE", 0.68, {
            "country": "INTL", "source_type": "official", "domain": "example.int"
        }, -2.0) == 0.0


if __name__ == "__main__":
    test_scope_mode_rejects_typos()
    test_learned_only_ignores_keyword_and_gazetteer()
    test_learned_only_head_failure_abstains_without_keyword_fallback()
    test_legacy_resolution_is_unchanged()
    test_soft_conflict_includes_official_but_preserves_scope_exemptions()
    test_same_calibrated_conflict_helper_is_wired_in_both_stages()
    print("all learned-only country-scope tests passed")
