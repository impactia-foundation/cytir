#!/usr/bin/env python3
"""Task 583 -- the per-component debug ledger (`_LAST_COMPONENTS`).

`rel_core` and `penalties` are BUCKETS. A dozen named levers add into them and
the sealed `AW_DEBUG_SCORES` record could only ever say "penalties = -1.4",
never WHICH lever. Task 583 has to answer "what does production do differently
to rank this gold" per TERM, so the ledger names them.

Two properties, and the second is the one that makes the decomposition worth
reading:

  1. DEFAULT OFF IS BYTE-IDENTICAL, AND SILENT. With `AW_DEBUG_SCORES` unset
     the scores and the ordering must be exactly what shipped, and
     `_LAST_COMPONENTS` must stay empty. The ledger is a diagnostic, not a
     lever.

  2. THE COMPONENTS RECONCILE, WITH NO RESIDUAL. For every scored row:
         rel_core      == SUM(components["rel.*"])
         penalties     == SUM(components["pen.*"])
         trust_contrib == components["trust.realised"] + SUM(components["bonus.*"])
     This is a guard that can fail (verification-doctrine 12): a NEW lever
     added to either bucket without a matching `_dbgc` call leaves a residual
     and turns this test red. Without it, an unnamed lever would quietly land
     in "OTHER" in every attribution table built on this surface, and the
     table would look complete while hiding the term that mattered.

  3. AN ABSENT KEY MEANS ZERO, NEVER "NOT MEASURED". A lever that did not fire
     writes no key, which is only honest if the reconciliation in (2) holds --
     which is why (2) is asserted on the same rows.

The reconciliation is asserted against the ADDITIVE trust mode, which is what
production and the candidate both serve; the test states that rather than
assuming it, and refuses if the mode is anything else.

Run: python3 tests/test_serve_task583_components.py   (or pytest tests/test_serve_*.py)
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def _child_env() -> dict:
    """The child runs this file by path, so `aw` must be importable without `pip install .`."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HERE.parent), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])])
    return env

QUERY = "annual inflation statistics report for the canton of geneva 2026"

# A pool shaped so that several DIFFERENT levers fire at once -- otherwise a
# reconciliation over one lever proves nothing about the other eleven.
GOV, WIKI, CARD, JUNK = 1, 2, 3, 4
PAYLOAD = {
    # url, title, trust, source_type, canton, country
    GOV:  ("https://bfs.admin.ch/inflation-2026", "Inflation statistics 2026",
           0.95, "gov-stat", None, "CH"),
    WIKI: ("https://en.wikipedia.org/wiki/Inflation", "Inflation",
           0.70, "reference", None, None),
    CARD: ("https://example.ch/firm", "Geneva Statistics AG",
           0.60, "entity_card", "GE", "CH"),
    # a canton-stamped page for a canton the query does NOT name -> canton pen.
    JUNK: ("https://zh.ch/steuern.json", "Steuern Zurich",
           0.80, "official", "ZH", "CH"),
}
BM25 = {GOV: 41.0, WIKI: 33.0, JUNK: 21.0, CARD: 9.0}
DENSE = {GOV: 0.61, WIKI: 0.55, CARD: 0.34}      # JUNK is dense-absent


def _arm(debug: str) -> dict:
    """Run the real search() under a stub client. Returns the result list and,
    when debug is on, the raw `_LAST_SCORE_DEBUG` rows."""
    os.environ.update({
        "AW_COLLECTION": "test_task583", "AW_TWO_STAGE": "1",
        "AW_BATCH_LEGS": "0", "AW_BM25_PER_TOKEN": "0", "AW_RANKHEAD": "0",
        "AW_QHEAD": "0", "AW_AUTHLEG": "0", "AW_COMMERCIAL": "0",
        "AW_DEDUP_V2": "0", "AW_RELFLOOR": "0", "AW_HUBGATE": "0",
        "AW_RERANK": "0", "AW_NAMEGATE": "0", "AW_EMBEDDER": "arctic",
        "AW_FOLD_ACCENTS": "0", "AW_BM25_QSTOP": "0",
        # levers deliberately ARMED so the ledger has something to reconcile
        "AW_TRUSTFLOOR": "1", "AW_TRUST_RELGATE": "1",
        "AW_TRUST_RELGATE_BETA": "0.5",
        "AW_AUTHORITY": "0", "AW_HUBFLOOR": "0",
        "AW_DEBUG_SCORES": debug,
    })
    sys.path.insert(0, str(HERE))
    from cytir.serve import models, retrieval

    if retrieval._TRUST_MODE != "additive":
        raise SystemExit(
            f"[583] REFUSE: trust mode is {retrieval._TRUST_MODE!r}; the "
            "reconciliation identity this test asserts is the additive one "
            "production serves. An unchecked mode is not a pass.")

    class P:
        def __init__(self, pid, score=0.0, payload=None):
            self.id, self.score, self.payload = pid, score, payload

    class Resp:
        def __init__(self, points):
            self.points = points

    def _pl(pid):
        url, title, trust, st, canton, cty = PAYLOAD[pid]
        return {"url": url, "domain": url.split("/")[2], "title": title,
                "snippet": None, "lang": "en", "source_type": st,
                "trust_tier": trust, "last_seen": "2026-06-01",
                "canton": canton, "country": cty, "names": ""}

    class Stub:
        def query_points(self, collection_name, query, using, limit,
                         query_filter=None, search_params=None, with_payload=True):
            src = DENSE if using == "dense" else BM25
            pts = [P(i, score=s) for i, s in
                   sorted(src.items(), key=lambda kv: -kv[1])]
            if with_payload is True:
                for p in pts:
                    p.payload = _pl(p.id)
            return Resp(pts)

        def query_batch_points(self, collection_name, requests):
            return [Resp([]) for _ in requests]

        def retrieve(self, collection_name, ids, with_payload=True):
            return [P(i, payload=_pl(i)) for i in ids]

    retrieval.qdrant_io.client = lambda: Stub()
    models.embed_query = lambda q: ([0.1] * 128, None)
    models.last_full_query_vec = lambda: None
    hits = retrieval.search(QUERY, k=4, lang="en", country="CH")
    return {
        "hits": [{"url": h["url"], "score": round(float(h["score"]), 9)}
                 for h in hits],
        "debug": [
            {k: d[k] for k in ("url", "rel_core", "penalties", "trust_contrib",
                               "fused", "components") if k in d}
            for d in retrieval._LAST_SCORE_DEBUG],
        "components_empty": not retrieval._LAST_COMPONENTS,
    }


def _run(debug: str) -> dict:
    r = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()),
                        "--arm", debug], capture_output=True, text=True,
                       env=_child_env())
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    assert line, f"arm {debug} produced no result:\n{r.stdout}\n{r.stderr}"
    return json.loads(line[-1])


def test_1_default_off_is_byte_identical_and_silent():
    off, on = _run("0"), _run("1")
    assert off["hits"] == on["hits"], (
        "AW_DEBUG_SCORES changed the served result:\n"
        f"  off: {off['hits']}\n  on:  {on['hits']}")
    assert off["debug"] == [], "the debug list must stay empty with the flag off"
    assert off["components_empty"] is True, (
        "_LAST_COMPONENTS must stay empty with the flag off -- a diagnostic "
        "that allocates on the serving path is not inert")
    assert on["debug"], "the flag ON produced no debug rows (an error is not an empty)"
    print(f"  [1] flag off == shipped ordering ({len(off['hits'])} hits, "
          f"identical scores), ledger silent  OK")


def test_2_components_reconcile_with_no_residual():
    on = _run("1")
    # The debug record rounds rel_core/penalties/trust_contrib/fused to 4dp and
    # each component to 6dp, so an exact identity is not available at this
    # surface: the arithmetic bound is 0.5e-4 per rounded bucket plus ~1e-5 over
    # the components. 2e-4 is that bound, not a fudge -- every real lever in
    # WEIGHTS is O(0.1)-O(3.0), three orders of magnitude above it.
    tol = 2e-4
    checked = 0
    for d in on["debug"]:
        c = d["components"]
        rel = sum(v for k, v in c.items() if k.startswith("rel."))
        pen = sum(v for k, v in c.items() if k.startswith("pen."))
        bon = sum(v for k, v in c.items() if k.startswith("bonus."))
        trust = c.get("trust.realised", 0.0)
        assert abs(rel - d["rel_core"]) < tol, (
            f"{d['url']}: rel.* sums to {rel}, rel_core is {d['rel_core']} "
            f"(residual {d['rel_core'] - rel:+.6f}) -- a relevance term is "
            f"unnamed in the ledger")
        assert abs(pen - d["penalties"]) < tol, (
            f"{d['url']}: pen.* sums to {pen}, penalties is {d['penalties']} "
            f"(residual {d['penalties'] - pen:+.6f}) -- a penalty/boost lever "
            f"is unnamed in the ledger")
        assert abs((trust + bon) - d["trust_contrib"]) < tol, (
            f"{d['url']}: trust.realised+bonus.* is {trust + bon}, "
            f"trust_contrib is {d['trust_contrib']} "
            f"(residual {d['trust_contrib'] - trust - bon:+.6f})")
        assert abs(d["rel_core"] + d["penalties"] + d["trust_contrib"]
                   - d["fused"]) < tol
        checked += 1
    assert checked >= 4, f"only {checked} rows reconciled; the pool is too thin"
    print(f"  [2] {checked} rows reconcile exactly: rel_core = SUM(rel.*), "
          f"penalties = SUM(pen.*), trust_contrib = trust.realised + "
          f"SUM(bonus.*)  OK")


def test_3_a_lever_that_did_not_fire_writes_no_key():
    on = _run("1")
    by_url = {d["url"]: d["components"] for d in on["debug"]}
    # the canton lever fires on the ZH page under a GE-naming query and on
    # nothing else -- so the key is present on exactly one row.
    fired = [u for u, c in by_url.items() if "pen.canton" in c]
    assert fired == [PAYLOAD[JUNK][0]], (
        f"pen.canton should be present on exactly the off-canton row, got {fired}")
    assert all(c["pen.canton"] < 0 for u, c in by_url.items() if u in fired)
    # and no row carries a key whose value is 0.0 -- absent IS the zero
    for u, c in by_url.items():
        zeros = [k for k, v in c.items() if v == 0.0]
        assert not zeros, f"{u} carries zero-valued keys {zeros}"
    print(f"  [3] pen.canton present on 1 of {len(by_url)} rows, no "
          f"zero-valued keys anywhere (absent == 0)  OK")


if __name__ == "__main__":
    if "--arm" in sys.argv:
        print(json.dumps(_arm(sys.argv[sys.argv.index("--arm") + 1])))
        sys.exit(0)
    print("task 583 -- per-component debug ledger (retrieval.py)")
    fails = 0
    for fn in (test_1_default_off_is_byte_identical_and_silent,
               test_2_components_reconcile_with_no_residual,
               test_3_a_lever_that_did_not_fire_writes_no_key):
        try:
            fn()
        except Exception as e:                              # noqa: BLE001
            fails += 1
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print("PASS" if not fails else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)
