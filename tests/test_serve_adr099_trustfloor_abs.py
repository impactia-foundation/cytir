#!/usr/bin/env python3
"""ADR-099 Ruling 3 — unit tests for the TRUSTFLOOR absolute exemption term.

Task 446 measured the pre-099 exemption granting the rank-1 candidate an
exemption in 43 of 43 queries, because `rel_core` is min-max normalized and the
BM25 leader therefore sits at ratio ~1.0 BY CONSTRUCTION. Ruling 3 keeps that
pool-relative test as NECESSARY and adds an ABSOLUTE one: raw dense cosine over
a floor, OR the candidate containing at least one surviving high-IDF query term.

Two levels, because each catches a different failure:

  A. the DECISION (`_trustfloor_exempt`) — a pure function, so grants/refusals
     can be asserted directly on mocked scores, including the flag-off identity
     with the pre-099 condition;
  B. the WIRING — the real `retrieval.search()` driven by a stub client (no
     qdrant), proving the term-presence probe actually reaches the scoring loop:
       * a FLOOD candidate (trust 0.95, dense-absent, BM25 leader, contains no
         high-IDF query term) keeps rank 1 with the flag off and loses its trust
         floor with it on — the measured defect;
       * a DENSE-ABSENT GOLD that DOES contain a high-IDF query term (the
         `ohear` / `rfc9103` class the exemption exists to protect) is scored
         IDENTICALLY in both arms.

Run: python3 tests/test_serve_adr099_trustfloor_abs.py
"""
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

# ----------------------------------------------------------------- level A ---
QUERY = "who invented the peculiar widget contraption thing anyway"
# hand-set so the test asserts the CODE's threshold logic, not live index data
IDF_TABLE = {"who": 1.2, "the": 0.44, "thing": 1.1, "anyway": 1.3,
             "invented": 3.5, "peculiar": 6.2, "widget": 7.1, "contraption": 8.4}


def test_A_decision_table():
    sys.path.insert(0, str(HERE))
    # match the shipped table's fold setting so importing schema does not warn
    os.environ.setdefault("AW_FOLD_ACCENTS", "1")
    from cytir.serve.fusion import _trustfloor_exempt as ex

    # pool-relative test FAILS -> never exempt, in either arm
    for abs_on in (False, True):
        assert ex(0.5, 1.0, 0.7, abs_on=abs_on, dense_raw=0.9,
                  abs_dense_floor=0.45, abs_term_hit=True) is False
    # pool-relative test PASSES, flag OFF -> exempt (the pre-099 behaviour)
    assert ex(0.8, 1.0, 0.7, abs_on=False, dense_raw=0.0,
              abs_dense_floor=0.45, abs_term_hit=False) is True
    # pool-relative PASSES, flag ON, no absolute evidence -> REFUSED (the fix)
    assert ex(1.0, 1.0, 0.7, abs_on=True, dense_raw=0.0,
              abs_dense_floor=0.45, abs_term_hit=False) is False
    # ... dense floor alone is enough
    assert ex(1.0, 1.0, 0.7, abs_on=True, dense_raw=0.45,
              abs_dense_floor=0.45, abs_term_hit=False) is True
    assert ex(1.0, 1.0, 0.7, abs_on=True, dense_raw=0.4499,
              abs_dense_floor=0.45, abs_term_hit=False) is False
    # ... a high-IDF term match alone is enough (the dense-absent gold class)
    assert ex(1.0, 1.0, 0.7, abs_on=True, dense_raw=0.0,
              abs_dense_floor=0.45, abs_term_hit=True) is True
    # empty pool (best_rel == 0) -> no exemption, unchanged by the flag
    assert ex(0.0, 0.0, 0.7, abs_on=True, dense_raw=1.0,
              abs_dense_floor=0.45, abs_term_hit=True) is False
    print("  [A] decision table: 9 grant/refusal cases, flag-off identical to "
          "the pre-099 condition  OK")


# ----------------------------------------------------------------- level B ---
FLOOD, GOLD, RARE, FILLER = 1, 2, 3, 4
# Titles carry NO query word: the title/exact features weigh 2.5/0.8 and would
# otherwise decide the ordering instead of the lever under test.
PAYLOAD = {
    FLOOD:  ("https://civic.example/inflation", "Annual report", 0.95),
    GOLD:   ("https://ref.example/history", "Reference page", 0.70),
    RARE:   ("https://spec.example/sheet", "Specification sheet", 0.90),
    FILLER: ("https://misc.example/other", "Notes", 0.60),
}
# Shaped like the measured defect (task 446 §2.1): the flooder LEADS the BM25
# leg on function words with no dense support at all, so after min-max it is the
# pool's rel_core leader-adjacent candidate at ratio ~0.9 — inside the
# pre-099 exemption, which is exactly why that exemption never fired.
BM25 = {FLOOD: 50.0, RARE: 48.0, GOLD: 14.0, FILLER: 10.0}
DENSE = {GOLD: 0.55, FILLER: 0.30}          # FLOOD and RARE are dense-absent
# which candidate contains which query token (the stored-BM25 presence oracle):
# only RARE carries a high-IDF term; FLOOD's BM25 mass is function words.
CONTAINS = {"contraption": {RARE}, "widget": {GOLD}, "peculiar": set(),
            "invented": set()}


def _arm(arm: str) -> dict:
    """Run the real search() under a stub client; return {url: score}."""
    import tempfile

    tbl = tempfile.mkstemp(suffix=".json")[1]
    os.environ.update({
        "AW_COLLECTION": "test_adr099", "AW_TWO_STAGE": "1", "AW_BATCH_LEGS": "0",
        "AW_BM25_PER_TOKEN": "0", "AW_RANKHEAD": "0", "AW_QHEAD": "0",
        "AW_AUTHLEG": "0", "AW_AUTHORITY": "0", "AW_COMMERCIAL": "0",
        "AW_DEDUP_V2": "0", "AW_RELFLOOR": "0", "AW_HUBFLOOR": "0",
        "AW_HUBGATE": "0", "AW_RERANK": "0", "AW_NAMEGATE": "0",
        "AW_EMBEDDER": "arctic", "AW_FOLD_ACCENTS": "0",
        "AW_TRUSTFLOOR": "1", "AW_TRUST_RELGATE": "1", "AW_TRUST_RELGATE_BETA": "0.5",
        "AW_BM25_QSTOP": "0", "AW_BM25_IDF_TABLE": tbl,
        "AW_TRUSTFLOOR_ABS": {"on": "1", "true": "true"}.get(arm, "0"),
    })
    if arm == "unset":
        os.environ.pop("AW_TRUSTFLOOR_ABS", None)
    sys.path.insert(0, str(HERE))
    from cytir.serve import models, retrieval, schema

    pathlib.Path(tbl).write_text(json.dumps({
        "fold_accents": False, "capture_max_idf": 9.0,
        "idf": {str(schema._hash(w)): v for w, v in IDF_TABLE.items()}}))
    schema.IDF = _reload_table(schema)
    hashed = {schema._hash(w): pids for w, pids in CONTAINS.items()}

    class P:
        def __init__(self, pid, score=0.0, payload=None):
            self.id, self.score, self.payload = pid, score, payload

    class Resp:
        def __init__(self, points):
            self.points = points

    def _pl(pid):
        url, title, trust = PAYLOAD[pid]
        return {"url": url, "domain": url.split("/")[2], "title": title,
                "snippet": None, "lang": "en", "source_type": "content",
                "trust_tier": trust, "last_seen": None}

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
            # the pool-filtered term-presence probe: one request per token
            out = []
            for r in requests:
                h = int(r.query.indices[0])
                out.append(Resp([P(i, score=1.0) for i in hashed.get(h, ())]))
            return out

        def retrieve(self, collection_name, ids, with_payload=True):
            return [P(i, payload=_pl(i)) for i in ids]

    retrieval.qdrant_io.client = lambda: Stub()
    models.embed_query = lambda q: ([0.1] * 128, None)
    models.last_full_query_vec = lambda: None
    hits = retrieval.search(QUERY, k=4, lang="en")
    return {h["url"]: round(float(h["score"]), 6) for h in hits}


def _reload_table(schema):
    """Re-read the idf table after writing it (the module read it at import)."""
    return schema._load_idf_table()


def test_B_wiring():
    outs = {}
    for arm in ("off", "on"):
        r = subprocess.run([sys.executable, str(HERE / "test_serve_adr099_trustfloor_abs.py"),
                            "--arm", arm], capture_output=True, text=True,
                       env=_child_env())
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
        assert line, f"arm {arm} produced no result:\n{r.stdout}\n{r.stderr}"
        outs[arm] = json.loads(line[-1])
    off, on = outs["off"], outs["on"]
    flood, gold, rare = (PAYLOAD[FLOOD][0], PAYLOAD[GOLD][0], PAYLOAD[RARE][0])

    assert max(off, key=off.get) == flood, \
        f"pre-099 arm should rank the flooder 1st, got {max(off, key=off.get)}"
    assert on[flood] < off[flood], "ABS on must attenuate the flooder's trust"
    assert on[flood] < on[gold], \
        f"flooder {on[flood]} should now lose to the gold {on[gold]}"
    assert on[rare] == off[rare], \
        (f"a dense-absent candidate that CONTAINS a high-IDF query term must be "
         f"scored identically: {off[rare]} -> {on[rare]}")
    assert on[gold] == off[gold], "a dense-supported candidate must not move"
    print(f"  [B] wiring: flood {off[flood]:.4f} -> {on[flood]:.4f} (loses the "
          f"floor), rare-term gold {on[rare]:.4f} unchanged, dense gold "
          f"{on[gold]:.4f} unchanged  OK")


def _child_arm(arm: str) -> dict:
    r = subprocess.run([sys.executable, str(HERE / "test_serve_adr099_trustfloor_abs.py"),
                        "--arm", arm], capture_output=True, text=True, env=_child_env())
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    assert line, f"arm {arm} produced no result:\n{r.stdout}\n{r.stderr}"
    return json.loads(line[-1])


def test_C_one_reading_true_never_arms():
    """Task 1188: AW_TRUSTFLOOR_ABS is a FLAG (== "1"). "true" is OFF for the floor, so
    search() output is identical to unset and "0"; only "1" changes it."""
    unset, zero, true, one = (_child_arm(a) for a in ("unset", "off", "true", "on"))
    assert json.dumps(unset, sort_keys=True) == json.dumps(zero, sort_keys=True) \
        == json.dumps(true, sort_keys=True)
    assert one != unset


if __name__ == "__main__":
    if "--arm" in sys.argv:
        print(json.dumps(_arm(sys.argv[sys.argv.index("--arm") + 1])))
        sys.exit(0)
    print("ADR-099 Ruling 3 — trust-floor absolute exemption (retrieval.py)")
    fails = 0
    for fn in (test_A_decision_table, test_B_wiring, test_C_one_reading_true_never_arms):
        try:
            fn()
        except Exception as e:                              # noqa: BLE001
            fails += 1
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print("PASS" if not fails else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)
