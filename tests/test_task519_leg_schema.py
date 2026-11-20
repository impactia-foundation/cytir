"""Task 519 — pin what a `gate_authority` leg artifact actually contains.

The defect these tests exist for cost a whole attribution: `per_q[qid]["urls"]`
is the arm's TOP-TEN RESULT LIST, and task 516's `lost_gold_presence.py` read it
as the gold URLs. Nothing in the repo said so, so nothing caught it, and the
number it produced (126 ranking / 4 coverage / 41 dark) was quoted in a task
report and in `plan.md`'s MASTER STATE.

Two checks, none of which needs a collection (the script-pinning rest moved
beside its scripts, task 2544: the task 519 script folder):

  1. a source-level contract on `gate_authority.run` — `per_q[qid]["urls"]`
     comes from the SEARCH HITS, and no gold is written into a leg at all. This
     is the same pattern as task 516's anti-fork test: the assertion is on the
     code, because the code is the contract;
  2. `gold_of` reads the golds from the eval row, so gold URLs are only ever
     available from the YAML;

Run: python3 -m pytest on this file.
"""
import inspect

from cytir.gates import authority as GA


def test_leg_per_q_urls_are_the_search_hits_not_the_golds():
    src = inspect.getsource(GA.run)
    assert 'urls = [h.get("url") or "" for h in hits]' in src, (
        "gate_authority.run no longer derives `urls` from the search hits; if "
        "the leg schema changed, task 519 §3's falsification and every reader "
        "of a leg artifact needs to know")
    assert '"urls": urls' in src
    # and nothing gold-shaped is persisted per query
    for forbidden in ('"gold_urls"', '"gold_domains"', '"gold"'):
        assert forbidden not in src, (
            f"a leg now persists {forbidden}; update this test AND "
            "the task-519 gold_presence_attrib script, which exists only "
            "because the golds were NOT in the leg")


def test_gold_of_reads_the_eval_row_not_a_leg():
    row = {"query": "q", "target_url": "https://example.org/a",
           "gold_urls": ["https://example.org/b"]}
    urls, domains = GA.gold_of(row)
    assert urls == ["https://example.org/a", "https://example.org/b"]
    assert domains == []
    # a freshness row keeps its domain golds
    fresh = {"query": "q", "intent": "freshness",
             "gold_domains": ["www.example.org"]}
    urls, domains = GA.gold_of(fresh)
    assert urls == [] and domains == ["example.org"]
