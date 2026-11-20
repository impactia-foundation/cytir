"""Task374 scope-mode isolation tests; no Qdrant, model, or collection needed.

The country ranking scope has one entry boundary:
``_effective_free_text_country_scope``.  Pinning that boundary ensures the
Task374 candidate-policy experiment cannot accidentally re-enable the older
keyword/gazetteer/classifier ranking machinery.
"""
import pathlib


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


def _mode_namespace(*, policy_only, learned_only=False, scope_active=False):
    calls = []

    def gazetteer(query):
        calls.append(("gazetteer", query))
        return "CH", False

    def scope_head(vector):
        calls.append(("scope_head", vector))
        return "FR", 0.93

    ns = _heads_namespace(
        _POLICY_ONLY_SCOPE=policy_only,
        _LEARNED_ONLY_SCOPE=learned_only,
        _SCOPE_ACTIVE=scope_active,
        _gazetteer_scope=gazetteer,
        _scope_head_predict=scope_head,
    )
    return ns, calls


def test_scope_mode_parser_preserves_legacy_and_accepts_policy_only():
    parse, _ = _mode_namespace(policy_only=False)
    assert parse["_parse_country_scope_mode"](None) == "legacy"
    assert parse["_parse_country_scope_mode"]("learned_only") == "learned_only"
    assert parse["_parse_country_scope_mode"]("policy_only") == "policy_only"
    try:
        parse["_parse_country_scope_mode"]("country_legs")
    except ValueError as exc:
        assert "policy_only" in str(exc)
    else:
        raise AssertionError("unknown scope mode must fail at startup")


def test_policy_only_blocks_keyword_gazetteer_legacy_head_and_ranking_scope():
    ns, calls = _mode_namespace(policy_only=True, learned_only=False, scope_active=True)
    # A keyword country is deliberately supplied.  Policy-only must not even
    # consult the fallbacks: all inferred country action belongs solely to the
    # typed candidate policy and is parked when its country-leg flag is off.
    got = ns["_country_scope_for_ranking"](None, "Swiss law", "CH", object())
    assert got == (None, None, 1.0)
    assert calls == []
    # The typed plan can nominate CH for candidate legs, but its country is not
    # an input to the ranking resolver.  The only possible explicit input is
    # the caller-owned argument in position one.
    typed_plan_country = "CH"
    assert typed_plan_country == "CH"
    assert ns["_country_scope_for_ranking"](None, "Swiss law", None, object()) == (
        None, None, 1.0)


def test_policy_only_preserves_explicit_country_ranking_override():
    ns, calls = _mode_namespace(policy_only=True, learned_only=False, scope_active=True)
    assert ns["_country_scope_for_ranking"]("DE", "Swiss law", "CH", object()) == (
        "DE", None, 1.0)
    # Explicit ownership means neither free-text inference source competes.
    assert calls == []


def test_legacy_and_learned_only_free_text_contracts_are_unchanged():
    legacy, legacy_calls = _mode_namespace(
        policy_only=False, learned_only=False, scope_active=False)
    assert legacy["_effective_free_text_country_scope"]("Swiss law", "CH", object()) == (
        "CH", None, 1.0)
    assert legacy_calls == []

    learned, learned_calls = _mode_namespace(
        policy_only=False, learned_only=True, scope_active=True)
    assert learned["_effective_free_text_country_scope"]("Swiss law", "CH", "qvec") == (
        None, "FR", 0.93)
    assert learned_calls == [("scope_head", "qvec")]


if __name__ == "__main__":
    test_scope_mode_parser_preserves_legacy_and_accepts_policy_only()
    test_policy_only_blocks_keyword_gazetteer_legacy_head_and_ranking_scope()
    test_policy_only_preserves_explicit_country_ranking_override()
    test_legacy_and_learned_only_free_text_contracts_are_unchanged()
    print("country scope modes: PASS")
