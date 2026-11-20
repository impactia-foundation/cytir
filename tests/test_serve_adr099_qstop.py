#!/usr/bin/env python3
"""ADR-099 Ruling 2 — unit tests for query-side IDF-threshold suppression.

Per verification-doctrine 1 these are TRIAGE, not coverage: the ranking effect
is measured by the gates in the task-461 doc. What is tested
here is what a gate cannot see —

  1. INERT at the default (flag off => byte-identical token bags);
  2. the THRESHOLD is a `< threshold` test, exactly, at both edges;
  3. the MINTOKENS seam (2..4-token queries keep their function words, because
     there a function word can BE the query — "the who");
  4. the NEVER-EMPTY guard (an all-function-word query keeps its lexical leg);
  5. an UNMEASURED token is high-IDF, i.e. kept — the table stores only the
     suppression candidates, so "absent" must never mean "suppress";
  6. the token-GROUP partition contract survives the flag (task 411 depends on
     `union(groups) == set(bm25_tokens)` and would silently require the wrong
     terms if the two views disagreed);
  7. a table that cannot be trusted (fold-setting mismatch, or a threshold above
     the table's capture bound) disables the lever instead of half-applying it;
  8. the SHIPPED table classifies the tokens ADR-099 names: `the` (idf 0.4442,
     the flooder's 60%) is suppressed, `framework`/`inflation` are not.

Run: python3 tests/test_serve_adr099_qstop.py   (or pytest tests/test_serve_adr099_qstop.py)
"""
import importlib.util
import json
import os
import pathlib
from cytir.serve import SERVER_DIR
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
SCHEMA = SERVER_DIR / "schema.py"
_n = [0]


def load_schema(**env):
    """A FRESH schema module under `env` (flags are read at import time)."""
    for k in ("AW_BM25_QSTOP", "AW_BM25_QSTOP_IDF", "AW_BM25_QSTOP_MINTOKENS",
              "AW_BM25_IDF_TABLE", "AW_FOLD_ACCENTS"):
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in env.items()})
    _n[0] += 1
    spec = importlib.util.spec_from_file_location(f"aw_schema_099_{_n[0]}", SCHEMA)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_EMPTY = tempfile.mkstemp(suffix=".json")[1]
pathlib.Path(_EMPTY).write_text(
    json.dumps({"fold_accents": False, "capture_max_idf": 6.0, "idf": {}}))


def write_table(entries, *, fold=False, capture=6.0, path=None):
    """entries: {surface word -> idf}, hashed exactly as the tokenizer does."""
    base = load_schema(AW_BM25_IDF_TABLE=_EMPTY)   # hashing helpers only
    idf = {}
    for word, v in entries.items():
        key = base._fold(word) if fold else word
        idf[str(base._hash(key))] = v
    p = path or tempfile.mkstemp(suffix=".json")[1]
    pathlib.Path(p).write_text(json.dumps({
        "fold_accents": fold, "capture_max_idf": capture, "idf": idf}))
    return p


# A deliberately small, hand-set table: the thresholds under test are the CODE's,
# not the live index's (that is what test 8 is for).
TABLE = {"the": 0.44, "of": 0.49, "is": 0.84, "as": 1.02, "or": 1.43,
         "edge": 1.50, "under": 1.49, "framework": 4.05}
QUERY = "the framework of an edge case is under review"


def test_1_inert_when_off():
    off = load_schema(AW_BM25_IDF_TABLE=write_table(TABLE))
    assert off._QSTOP is False, "AW_BM25_QSTOP must default OFF"
    assert off.bm25_tokens(QUERY, None) == off._bm25_tf(QUERY, None)
    print("  [1] flag off: token bag byte-identical to the raw tokenizer  OK")


def test_2_threshold_edges():
    s = load_schema(AW_BM25_QSTOP=1, AW_BM25_QSTOP_IDF=1.50,
                    AW_BM25_IDF_TABLE=write_table(TABLE))
    kept = set(s.bm25_tokens(QUERY, None))
    for word, expect_kept in (("the", False), ("of", False), ("is", False),
                              ("as", False), ("or", False),
                              ("under", False),      # 1.49 < 1.50 -> dropped
                              ("edge", True),        # 1.50 is NOT below 1.50
                              ("framework", True)):
        h = s._hash(word)
        assert (h in kept) is expect_kept, f"{word}: kept={h in kept}"
    print("  [2] threshold is a strict `idf < T` test; 1.49 drops, 1.50 stays  OK")


def test_3_mintokens_seam():
    s = load_schema(AW_BM25_QSTOP=1, AW_BM25_QSTOP_IDF=1.50,
                    AW_BM25_QSTOP_MINTOKENS=4, AW_BM25_IDF_TABLE=write_table(TABLE))
    for text, suppressed in (("the framework", False),           # 2 tokens
                             ("the framework of edge", False),   # 4 tokens
                             ("the framework of an edge", True)):  # 5 tokens
        raw, out = s._bm25_tf(text, None), s.bm25_tokens(text, None)
        assert (out != raw) is suppressed, (text, len(raw), len(out))
    print("  [3] mintokens seam: <=4-token queries untouched, 5+ suppressed  OK")


def test_4_never_empty():
    s = load_schema(AW_BM25_QSTOP=1, AW_BM25_QSTOP_IDF=1.50,
                    AW_BM25_IDF_TABLE=write_table(TABLE))
    text = "the of is as or the of"
    raw, out = s._bm25_tf(text, None), s.bm25_tokens(text, None)
    assert out == raw and out, "an all-function-word query must keep its leg"
    print("  [4] never-empty: all-suppressed query falls back to the full bag  OK")


def test_5_unknown_token_is_kept():
    s = load_schema(AW_BM25_QSTOP=1, AW_BM25_QSTOP_IDF=1.50,
                    AW_BM25_IDF_TABLE=write_table(TABLE))
    text = "the tocmacov of an obligationenrecht filing"
    kept = set(s.bm25_tokens(text, None))
    assert s._hash("tocmacov") in kept and s._hash("obligationenrecht") in kept
    assert s.token_idf(s._hash("tocmacov")) == s.UNKNOWN_IDF
    print("  [5] unmeasured tokens are high-IDF and survive  OK")


def test_6_group_partition_holds_under_the_flag():
    s = load_schema(AW_BM25_QSTOP=1, AW_BM25_QSTOP_IDF=1.50,
                    AW_BM25_IDF_TABLE=write_table(TABLE))
    for text in (QUERY, "the framework of Tim O'Hear as of the edge",
                 "the of is as or", "framework"):
        for lang in (None, "de", "fr", "en"):
            flat = set(s.bm25_tokens(text, lang))
            grouped = set()
            for _w, p, hs in s.bm25_token_groups(text, lang):
                assert p in hs, "primary must stay inside its own group"
                grouped |= set(hs)
            assert grouped == flat, (text, lang, grouped ^ flat)
    print("  [6] union(groups) == set(bm25_tokens) with suppression ON  OK")


def test_7_untrustworthy_table_disables_the_lever():
    folded = load_schema(AW_BM25_QSTOP=1, AW_FOLD_ACCENTS=1,
                         AW_BM25_IDF_TABLE=write_table(TABLE, fold=False))
    assert folded.IDF == {}, "a fold-setting mismatch must void the table"
    assert folded.bm25_tokens(QUERY, None) == folded._bm25_tf(QUERY, None)
    over = load_schema(AW_BM25_QSTOP=1, AW_BM25_QSTOP_IDF=9.0,
                       AW_BM25_IDF_TABLE=write_table(TABLE, capture=6.0))
    assert over.IDF == {}, "a threshold above capture_max must void the table"
    print("  [7] bad/insufficient table => lever inert, search unaffected  OK")


def test_8_shipped_table_classifies_the_adr099_tokens():
    s = load_schema(AW_BM25_QSTOP=1, AW_FOLD_ACCENTS=1)   # default table path
    assert s.IDF, "cytir/serve/bm25_idf.json failed to load"
    the = s.token_idf(s._hash("the"))
    assert abs(the - 0.4442) < 0.01, f"idf('the')={the}, task 446 measured 0.4442"
    assert the < s._QSTOP_IDF, "the flooder's 60%-of-score token must be suppressed"
    for word in ("framework", "inflation", "kuendigung", "obligationenrecht"):
        assert s.token_idf(s._hash(word)) >= s._QSTOP_IDF, f"{word} must survive"
    for numeral in ("10", "12", "336", "2025"):
        assert s.token_idf(s._hash(numeral)) >= s._QSTOP_IDF, \
            f"numeral {numeral} is load-bearing for article/version lookups"
    print(f"  [8] shipped table: {len(s.IDF)} tokens, idf('the')={the:.4f} "
          f"suppressed, content tokens + numerals survive  OK")


TESTS = [test_1_inert_when_off, test_2_threshold_edges, test_3_mintokens_seam,
         test_4_never_empty, test_5_unknown_token_is_kept,
         test_6_group_partition_holds_under_the_flag,
         test_7_untrustworthy_table_disables_the_lever,
         test_8_shipped_table_classifies_the_adr099_tokens]

if __name__ == "__main__":
    print("ADR-099 Ruling 2 — query-side IDF suppression (schema.py)")
    fails = 0
    for fn in TESTS:
        try:
            fn()
        except Exception as e:                              # noqa: BLE001
            fails += 1
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print("PASS" if not fails else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)
