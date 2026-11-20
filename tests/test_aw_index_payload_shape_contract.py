#!/usr/bin/env python3
"""The payload-shape contract (task 516), against REAL stage receipts.

`tests/testdata/task516_stage_receipts.json` holds three receipts copied
verbatim out of the task-516 staged tree:

  * `fedlex`       -- `task417_families/source=fedlex-r0-exact-current-v1`,
                      22,927 rows, the family the title contract exists for;
  * `wikipedia`    -- `wikipedia_413a/shard-0000`, 60,524 rows on the `twin`
                      payload route, whose 19.2M-row set shipped with NO BM25
                      posting in task 514;
  * `entity_cards` -- `task467_entity_card_1024_reembed/shard-0000`, 40,000
                      rows, the set whose `source_type` task 515 refused.

Each test is a defect this contract exists to have caught, stated as the number
that was actually shipped.

Run: python3 -m pytest tests/test_aw_index_payload_shape_contract.py
"""
from __future__ import annotations

import copy
import json
import pathlib

import pytest

from cytir.index import payload_shape_contract as PSC

RECEIPTS = json.loads(
    (pathlib.Path(__file__).resolve().parent / "testdata"
     / "task516_stage_receipts.json").read_text())
_CONTRACT_FILE = pathlib.Path(__file__).resolve().parents[1] / "data" / "payload-shape-contract.v1.json"
if not _CONTRACT_FILE.exists():
    pytest.skip("the public cut does not ship data/payload-shape-contract.v1.json (private-tree contract; task 1747)", allow_module_level=True)
CONTRACT = PSC.load()
ALL = [RECEIPTS["fedlex"], RECEIPTS["wikipedia"], RECEIPTS["entity_cards"]]

def _measured(receipt, provenance_heads=0):
    """Attach the task-517 provenance-head counter that a stage produces now.

    The three fixture receipts are verbatim task-516 copies and predate that
    counter. A receipt that carries none is a FAIL by design -- see
    `test_a_partition_that_never_measured_its_heads_is_a_FINDING` -- so every
    test that is about a DIFFERENT defect states this one as measured and clean.
    """
    r = copy.deepcopy(receipt)
    r["provenance_head"] = {"pattern": "(?:never-matches)",
                            "rows": r.get("staged") or 0,
                            "rows_with_provenance_head": provenance_heads,
                            "measured_at": "stage"}
    return r



def _with_title_shape(receipt, rows, matched,
                      source="fedlex-r0-exact-current-v1"):
    r = _measured(receipt)
    r["title_shape"] = {"source": source, "pattern": r"\bSR\s?\d",
                        "rows": rows, "matched": matched}
    return r


def _with_published_presence(
        receipt, rows, present,
        source="pubmed-exact-production-academic-markdown-v3"):
    """Task 536: same shape as `_with_title_shape`, for the OTHER opt-in
    per-family dimension the real tracked contract now declares."""
    r = _measured(receipt)
    r["published_presence"] = {"source": source, "rows": rows,
                               "present": present}
    return r


# The real contract declares `published_presence` for pubmed (task 536); any
# test below that asserts PASS against the real CONTRACT with a receipt set
# that never mentions pubmed must say so explicitly via `_with_published_presence`,
# same as `_with_title_shape` does for fedlex -- a declared-but-unmeasured
# dimension is a FAIL by design.


# --------------------------------------------------------------- the shipped
def test_the_real_receipts_pass_the_contract():
    report = PSC.evaluate(
        [_with_title_shape(RECEIPTS["fedlex"], 22927, 22927),
         _measured(RECEIPTS["wikipedia"]),
         _with_published_presence(RECEIPTS["entity_cards"], 1000, 995)],
        CONTRACT)
    assert report["verdict"] == "PASS", report["findings"]
    fedlex = next(f for f in report["families"]
                  if f["source"] == "fedlex-r0-exact-current-v1")
    assert fedlex["share"] == 1.0 and fedlex["verdict"] == "PASS"


def test_the_real_wikipedia_receipt_carries_its_postings():
    """Task 514 staged this same set with 0 carried and 60,524 without."""
    r = RECEIPTS["wikipedia"]
    assert r["sparse_carried_from_twin"] == r["staged"]
    assert r["staged_without_sparse"] == 0


def test_the_real_card_receipt_is_stamped_from_its_recipe():
    """Task 515 shipped 336,254 of these with no `source_type` at all."""
    meta = RECEIPTS["entity_cards"]["serving_metadata"]
    assert meta.get("meta_entity_card_stamped_from_recipe", 0) > 0
    assert meta.get("meta_refused_source_type_entity_card", 0) == 0


# ---------------------------------------------------------------- the defects
def test_a_family_whose_titles_lost_their_handle_is_REFUSED():
    """The task-515 measurement, replayed: 0 of 22,928 fedlex titles carried an
    SR number and every check passed."""
    report = PSC.evaluate([_with_title_shape(RECEIPTS["fedlex"], 22928, 0)]
                          + [_measured(r) for r in ALL[1:]], CONTRACT)
    assert report["verdict"] == "FAIL"
    assert any("fedlex-r0-exact-current-v1" in f and "0.00%" in f
               for f in report["findings"]), report["findings"]


def test_a_family_just_under_the_declared_share_is_REFUSED():
    report = PSC.evaluate([_with_title_shape(RECEIPTS["fedlex"], 10000, 9499)]
                          + [_measured(r) for r in ALL[1:]], CONTRACT)
    assert report["verdict"] == "FAIL"


def test_production_s_own_rate_would_pass():
    """23,495 of 23,872 = 98.42%. The bar is below production, so the contract
    fails on a regression and not on the 118 articles production itself titles
    with the bare docTitle."""
    report = PSC.evaluate(
        [_with_title_shape(RECEIPTS["fedlex"], 23872, 23495),
         _measured(RECEIPTS["wikipedia"]),
         _with_published_presence(RECEIPTS["entity_cards"], 1000, 995)],
        CONTRACT)
    assert report["verdict"] == "PASS", report["findings"]


def test_a_declared_contract_that_measured_nothing_is_a_FINDING():
    """A title contract nothing evaluates is not a contract. Task 515's
    staged tree had no title-shape counter at all, which is why 22,928
    unfindable statutes loaded without a word."""
    report = PSC.evaluate([_measured(r) for r in ALL], CONTRACT)
    assert report["verdict"] == "FAIL"
    assert any("measured no rows" in f for f in report["findings"])


# ------------------------------------------------------ published_presence (536)

def test_pubmed_shipping_with_no_published_at_all_is_REFUSED():
    """The task 535/536 defect, replayed: 0 of N pubmed rows carried
    `published` and every count check passed."""
    report = PSC.evaluate(
        [_with_title_shape(RECEIPTS["fedlex"], 22927, 22927),
         _measured(RECEIPTS["wikipedia"]),
         _with_published_presence(RECEIPTS["entity_cards"], 5_000_000, 0)],
        CONTRACT)
    assert report["verdict"] == "FAIL"
    pub = next(f for f in report["published_families"]
              if f["source"] == "pubmed-exact-production-academic-markdown-v3")
    assert pub["verdict"] == "FAIL" and pub["share"] == 0.0
    assert any("pubmed-exact-production-academic-markdown-v3" in f
              and "0.00%" in f for f in report["findings"]), report["findings"]


def test_the_twins_own_pubmed_rate_would_pass():
    """4,969,090 of 4,969,092 (task 535 sec.3.1) = 99.99996%, comfortably
    above the declared 99% floor."""
    report = PSC.evaluate(
        [_with_title_shape(RECEIPTS["fedlex"], 22927, 22927),
         _measured(RECEIPTS["wikipedia"]),
         _with_published_presence(RECEIPTS["entity_cards"],
                                  4_969_092, 4_969_090)],
        CONTRACT)
    assert report["verdict"] == "PASS", report["findings"]


def test_published_presence_declared_but_unmeasured_is_a_FINDING():
    """Same posture as title_shape: a contract nothing evaluates is not a
    contract."""
    report = PSC.evaluate(
        [_with_title_shape(RECEIPTS["fedlex"], 22927, 22927),
         _measured(RECEIPTS["wikipedia"]),
         _measured(RECEIPTS["entity_cards"])], CONTRACT)
    assert report["verdict"] == "FAIL"
    assert any("published-presence contract is declared" in f
              for f in report["findings"])
    pub = next(f for f in report["published_families"]
              if f["source"] == "pubmed-exact-production-academic-markdown-v3")
    assert pub["verdict"] == "NOT EVALUATED"


def test_an_undeclared_family_is_not_checked_for_published_presence():
    """OPT-IN, same as title_shape: a source with no `published_presence`
    entry (i.e. every family except pubmed today) is never failed on it, no
    matter how the field measures -- 84 of 86 declared families never
    stamped `published` at all (task 535 sec.3.5)."""
    receipt = _with_published_presence(RECEIPTS["wikipedia"], 1000, 0,
                                       source="wikipedia_413a")
    report = PSC.evaluate(
        [_with_title_shape(RECEIPTS["fedlex"], 22927, 22927), receipt,
         _with_published_presence(RECEIPTS["entity_cards"],
                                  4_969_092, 4_969_090)], CONTRACT)
    assert report["verdict"] == "PASS", report["findings"]
    assert not any(f["source"] == "wikipedia_413a"
                  for f in report["published_families"])


def test_a_set_shipping_rows_with_no_posting_is_REFUSED():
    """19,619,656 rows, 63.5% of task 514's collection, and nothing counted
    the absence."""
    broken = _measured(RECEIPTS["wikipedia"])
    broken["staged_without_sparse"] = broken["staged"]
    broken["sparse_carried_from_twin"] = 0
    report = PSC.evaluate(
        [_with_title_shape(RECEIPTS["fedlex"], 22927, 22927), broken,
         _measured(RECEIPTS["entity_cards"])], CONTRACT)
    assert report["verdict"] == "FAIL"
    assert any("no BM25 posting" in f and "wikipedia_413a" in f
               for f in report["findings"]), report["findings"]


def test_a_set_shipping_rows_with_no_serving_signal_is_REFUSED():
    """11,273,954 rows (36.5%) with no trust_tier and no source_type."""
    broken = _measured(RECEIPTS["wikipedia"])
    broken["serving_metadata"] = dict(broken.get("serving_metadata") or {})
    broken["serving_metadata"]["meta_row_missing_serving_signal"] = broken["staged"]
    report = PSC.evaluate(
        [_with_title_shape(RECEIPTS["fedlex"], 22927, 22927), broken,
         _measured(RECEIPTS["entity_cards"])], CONTRACT)
    assert report["verdict"] == "FAIL"
    assert any("carry no title/trust_tier/source_type/lang" in f
               for f in report["findings"]), report["findings"]


def test_the_per_set_limit_beats_the_default():
    """`wikipedia_413a` declares zero tolerance for a missing posting; the
    default allows 0.1%. One row must fail the set and pass the default."""
    broken = _measured(RECEIPTS["wikipedia"])
    broken["staged_without_sparse"] = 1
    report = PSC.evaluate([broken], CONTRACT)
    assert report["verdict"] == "FAIL"
    generic = copy.deepcopy(broken)
    generic["set_id"] = "some_other_set"
    assert PSC.evaluate([generic], CONTRACT)["sets"][0]["verdict"] == "PASS"


def test_the_contract_is_data_and_carries_its_evidence():
    assert CONTRACT["schema"] == PSC.SCHEMA
    for source, spec in CONTRACT["title_shape"].items():
        if isinstance(spec, dict) and spec.get("pattern"):
            assert spec.get("why"), f"{source} has a threshold and no evidence"
            assert spec.get("sample")


def test_title_pattern_is_opt_in_per_family():
    """Task 516 section 3 measured that jorf/kali/cantonal/gesetze/legifrance
    production titles carry NO SR number, so inventing one for them would fail
    a build over a rule nobody measured a need for."""
    assert PSC.title_pattern(CONTRACT, "dila-jorf-exact-production-article-bound-v1") is None
    assert PSC.title_pattern(CONTRACT, None) is None
    assert PSC.title_pattern(CONTRACT, "fedlex-r0-exact-current-v1") is not None


def test_the_unit_is_the_source_not_the_pooled_set():
    """`task417_families` pools 123 unrelated sources. On this build all 126,427
    context7 rows carry no `title` at all -- 100% of that source -- and the
    pooled set-level fraction is 1.2%. A threshold on the pooled number is a
    threshold on the wrong thing."""
    clean = copy.deepcopy(RECEIPTS["fedlex"])          # task417_families/fedlex
    broken = _measured(RECEIPTS["fedlex"])
    broken["shard"] = "source=context7-github-production-current-repos-4625-4874-v1.shard-0000"
    broken["staged"] = 7841
    broken["serving_metadata"] = {"meta_row_missing_serving_signal": 7841}
    report = PSC.evaluate([_with_title_shape(clean, 22927, 22927), broken], CONTRACT)
    units = {r["set_id"]: r for r in report["sets"]}
    assert "task417_families/source=fedlex-r0-exact-current-v1" in units
    assert units["task417_families/source=fedlex-r0-exact-current-v1"]["verdict"] == "PASS"
    ctx = units["task417_families/source=context7-github-production-current-repos-4625-4874-v1"]
    assert ctx["verdict"] == "FAIL" and ctx["missing_serving_signal_frac"] == 1.0


def test_a_source_level_threshold_overrides_the_set_level_one():
    contract = copy.deepcopy(CONTRACT)
    contract["per_set"] = dict(contract.get("per_set") or {})
    contract["per_set"]["task417_families"] = {"max_missing_serving_signal_frac": 0.0}
    contract["per_set"]["fedlex-r0-exact-current-v1"] = {
        "max_missing_serving_signal_frac": 1.0}
    r = copy.deepcopy(RECEIPTS["fedlex"])
    r["serving_metadata"] = {"meta_row_missing_serving_signal": 100}
    report = PSC.evaluate([_with_title_shape(r, 22927, 22927)], contract)
    assert report["sets"][0]["verdict"] == "PASS"


def test_measure_partition_titles_counts_a_real_payload_file(tmp_path):
    """The verify-time measurement reads the STAGED PAYLOAD, which is the exact
    row `load` upserts -- so a tree staged before the contract existed is
    measured rather than reported as unchecked."""
    import gzip
    import re
    path = tmp_path / "p.payload.jsonl.gz"
    titles = ["Obligationenrecht (OR; SR 220) - Art. 411",
              "Urheberrechtsgesetz (URG; SR 231.1) - Art. 21",
              "Bundesgesetz ueber Schuldbetreibung und Konkurs - Art. 1",
              ""]
    with gzip.open(path, "wt") as fh:
        for t in titles:
            fh.write(json.dumps({"id": "x", "payload": {"title": t}}) + "\n")
    rows, matched = PSC.measure_partition_titles(path, re.compile(r"\bSR\s?\d"))
    assert (rows, matched) == (4, 2)


def test_backfill_only_touches_a_source_that_has_a_contract(tmp_path):
    import gzip
    staged = tmp_path / "staged"
    staged.mkdir()
    name = "task417_families.source=fedlex-r0-exact-current-v1.shard-0000"
    with gzip.open(staged / (name + ".payload.jsonl.gz"), "wt") as fh:
        fh.write(json.dumps({"id": "x", "payload": {
            "title": "Obligationenrecht (OR; SR 220) - Art. 411"}}) + "\n")
    receipts = [
        {"set_id": "task417_families",
         "shard": "source=fedlex-r0-exact-current-v1.shard-0000", "staged": 1},
        {"set_id": "task417_families",
         "shard": "source=dila-jorf-exact-production-article-bound-v1.shard-0000",
         "staged": 1},
    ]
    filled = PSC.backfill_title_shape(receipts, staged, CONTRACT,
                                      log=lambda *a: None)
    assert filled == 1
    assert receipts[0]["title_shape"]["matched"] == 1
    assert "title_shape" not in receipts[1]


# ------------------------------------------- the provenance head (task 517)
def test_a_partition_that_never_measured_its_heads_is_a_FINDING():
    """A ceiling nothing evaluates is not a ceiling. Task 516 shipped 72,330
    rows whose snippet AND embedding head were a YAML provenance block, and
    every check passed because nothing looked at the shape of a chunk's head."""
    report = PSC.evaluate([_with_title_shape(RECEIPTS["fedlex"], 22927, 22927),
                           copy.deepcopy(RECEIPTS["wikipedia"])], CONTRACT)
    assert report["verdict"] == "FAIL"
    assert any("never evaluated" in f for f in report["findings"]), \
        report["findings"]


def test_the_context7_number_that_shipped_is_REFUSED():
    """72,330 of 126,427 = 57.2%, against a declared 0.5%."""
    broken = _measured(RECEIPTS["fedlex"], provenance_heads=72330)
    broken["shard"] = ("source=context7-github-production-current-repos-"
                       "4625-4874-v1.shard-0000")
    broken["staged"] = 126427
    broken["provenance_head"]["rows"] = 126427
    report = PSC.evaluate([broken], CONTRACT)
    assert report["verdict"] == "FAIL"
    assert any("open with a provenance block" in f and "57.2" in f
               for f in report["findings"]), report["findings"]


def test_the_fedlex_number_that_shipped_would_also_have_been_REFUSED():
    """22,913 of 22,928. The guard is the class, not the instance: it would
    have refused task 515's fedlex build a phase earlier."""
    broken = _with_title_shape(RECEIPTS["fedlex"], 22928, 0)
    broken["staged"] = 22928
    broken["provenance_head"].update(rows=22928, rows_with_provenance_head=22913)
    report = PSC.evaluate([broken], CONTRACT)
    assert report["verdict"] == "FAIL"
    assert any("open with a provenance block" in f for f in report["findings"])


def test_a_clean_partition_passes_the_head_contract():
    report = PSC.evaluate(
        [_with_title_shape(RECEIPTS["fedlex"], 22927, 22927),
         _measured(RECEIPTS["wikipedia"]),
         _with_published_presence(RECEIPTS["entity_cards"], 1000, 995)],
        CONTRACT)
    assert report["verdict"] == "PASS", report["findings"]
    row = next(r for r in report["sets"] if r["set_id"] == "wikipedia_413a")
    assert row["provenance_head_frac"] == 0.0


def test_every_declared_head_pattern_carries_its_evidence():
    pats = CONTRACT["provenance_head_patterns"]
    assert pats, "the contract declares no provenance-head pattern"
    for name, spec in pats.items():
        assert spec.get("pattern"), name
        assert spec.get("why"), f"{name} has a pattern and no evidence"
        assert spec.get("sample"), name


def test_measure_partition_provenance_heads_counts_a_real_payload_file(tmp_path):
    import gzip
    path = tmp_path / "p.payload.jsonl.gz"
    sealed = json.loads(
        (pathlib.Path(__file__).resolve().parent / "testdata"
         / "task517_sealed_context7_records.json").read_text())
    snippets = [" ".join(sealed["chunk_zero_with_the_front_matter"]["text"].split()),
                " ".join(sealed["chunk_one_without_it"]["text"].split()),
                "Official source identity - ELI: `cc/11/529_488_529`",
                ""]
    with gzip.open(path, "wt") as fh:
        for sn in snippets:
            fh.write(json.dumps({"id": "x", "payload": {"snippet": sn}}) + "\n")
    rows, hits = PSC.measure_partition_provenance_heads(
        path, PSC.provenance_head_pattern(CONTRACT))
    assert (rows, hits) == (4, 2)


def test_the_head_backfill_measures_a_tree_staged_before_the_contract(tmp_path):
    import gzip
    staged = tmp_path / "staged"
    staged.mkdir()
    name = "task417_families.source=context7-x.shard-0000"
    with gzip.open(staged / (name + ".payload.jsonl.gz"), "wt") as fh:
        fh.write(json.dumps({"id": "a", "payload": {
            "snippet": '--- aw_repository: "x/y" aw_revision: "z" ---'}}) + "\n")
        fh.write(json.dumps({"id": "b", "payload": {
            "snippet": "an actual sentence of the document"}}) + "\n")
    receipts = [{"set_id": "task417_families",
                 "shard": "source=context7-x.shard-0000", "staged": 2}]
    filled = PSC.backfill_provenance_head(receipts, staged, CONTRACT,
                                          log=lambda *a: None)
    assert filled == 1
    assert receipts[0]["provenance_head"]["rows_with_provenance_head"] == 1
    assert receipts[0]["provenance_head"]["measured_at"].startswith("backfilled")
