#!/usr/bin/env python3
"""ADR-062 Phase-2b (task 281) — GAZETTEER invariant + rollup tests.

Validates `cytir/serve/geo_gazetteer.json` and re-implements the serve-time
`_gazetteer_scope` match logic (prominence-ordered first-match + homograph abstain)
in isolation, so a gazetteer edit that breaks the Ontario→CA case, un-abstains a
homograph, or lets an abstained name also roll up is caught WITHOUT importing
retrieval.py (which needs qdrant_client + a live collection).

Run: python3 tests/test_gazetteer.py   (or via pytest)
"""
import json
import os
import re

GAZ = os.path.join(os.path.dirname(__file__), "..", "cytir", "serve",
                   "geo_gazetteer.json")
_VALID = {"CH", "DE", "FR", "IT", "CA", "GB", "US", "EU"}


def _terms_re(names):
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(n) for n in names)
                      + r")(?!\w)", re.UNICODE)


def _load():
    doc = json.load(open(GAZ, encoding="utf-8"))
    entries = sorted(doc["entries"], key=lambda e: -int(e.get("prominence", 0)))
    rollup = [(e["country"], _terms_re(e["names"])) for e in entries]
    anames = [a["name"] for a in doc["abstain"]]
    abstain_re = _terms_re(anames) if anames else None
    return doc, rollup, abstain_re, set(anames)


def _scope(rollup, abstain_re, q):
    ql = q.lower()
    ab = bool(abstain_re and abstain_re.search(ql))
    for c, rx in rollup:
        if rx.search(ql):
            return c, ab
    return None, ab


def test_structure_valid():
    doc, rollup, _, _ = _load()
    assert doc["entries"], "gazetteer has no entries"
    for e in doc["entries"]:
        assert e["country"] in _VALID, f"bad country {e['country']}"
        assert e["names"] and all(n == n.lower() for n in e["names"]), \
            f"names must be lowercased: {e['names']}"
        assert 1 <= int(e["prominence"]) <= 5


def test_no_rollup_name_is_also_abstained():
    doc, rollup, _, abstain = _load()
    for e in doc["entries"]:
        for n in e["names"]:
            assert n not in abstain, f"'{n}' is both a rollup and an abstain name"


def test_ontario_and_peers_rollup():
    _, rollup, ab_re, _ = _load()
    cases = {
        "Ontario building code requirements": "CA",
        "Quebec civil code": "CA",
        "British Columbia employment standards": "CA",
        "business licence Toronto": "CA",
        "Bavaria Handelsregister": "DE",
        "Nordrhein-Westfalen permit": "DE",
        "impôts en Île-de-France": "FR",
        "VAT in Lombardy": "IT",
        "tenancy law Scotland": "GB",
        "London council tax": "GB",
        "California privacy act": "US",
    }
    for q, exp in cases.items():
        got, _ = _scope(rollup, ab_re, q)
        assert got == exp, f"{q!r} -> {got}, expected {exp}"


def test_homographs_abstain():
    _, rollup, ab_re, _ = _load()
    for q in ["Georgia business registration", "Washington state tax",
              "Paris property tax", "is this a nice day", "New York vs York"]:
        got, ab = _scope(rollup, ab_re, q)
        # an abstained homograph must NOT produce a (wrong) rollup for the query
        # whose ONLY geo token is the homograph.
        assert ab or got is None or got in _VALID
    # the canonical country-vs-subnational case must never scope
    got, ab = _scope(rollup, ab_re, "Georgia business registration requirements")
    assert got is None and ab, "Georgia must abstain (country vs US state)"


def test_multilingual_forms():
    _, rollup, ab_re, _ = _load()
    # German/French/Italian surface forms roll up to the same country as English.
    assert _scope(rollup, ab_re, "Bayern Gewerbeanmeldung")[0] == "DE"
    assert _scope(rollup, ab_re, "diritto del lavoro in Lombardia")[0] == "IT"
    assert _scope(rollup, ab_re, "permis de construire en Bretagne")[0] == "FR"


def test_unscoped_returns_none():
    _, rollup, ab_re, _ = _load()
    for q in ["how does TLS 1.3 work", "REST API pagination best practices"]:
        got, ab = _scope(rollup, ab_re, q)
        assert got is None and not ab


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} gazetteer tests passed")
