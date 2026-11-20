"""Tests for the convergence refusal (task 609).

The rule under test: two readings of ONE unchanged collection must agree per
gold, and anything that stops the comparison being possible is UNKNOWN rather
than a pass. Every case below is one of the ways this lane has actually been
fooled -- an arm of errors handed on as a reading (task 600), a count that
looked clean because the thing that would have shown the defect was absent
(ADR-053), a guard that computed a verdict and exited 0 (task 406).
"""
from __future__ import annotations

import json
import pathlib

import pytest

from cytir.gates import convergence as gc


def arm(ranks: dict, sets=gc.BAR_SETS, cache_miss=0) -> dict:
    """An arm scoring one gold per named set, at the given ranks."""
    out = {"cache_miss": cache_miss, "sets": {}}
    for name in sets:
        out["sets"][name] = {"per_q": {
            f"{name}-1": {"rank": ranks.get(name, 1),
                          "gold_urls": ["https://example.org/a"],
                          "gold_domains": []}}}
    return out


def write(tmp_path, name, doc):
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return path


def test_two_identical_readings_converge(tmp_path):
    a = write(tmp_path, "a.json", arm({}))
    b = write(tmp_path, "b.json", arm({}))
    result = gc.verdict(a, b)
    assert result["verdict"] == gc.CONVERGED
    assert result["disagreements"] == []
    assert result["queries_compared"] == len(gc.BAR_SETS)


def test_one_moved_gold_is_not_converged(tmp_path):
    # This is the whole point: on 2026-09-17 the SAME collection read 18 core
    # losses at wait_green+0 and 0 losses twenty minutes later. One gold moving
    # between two readings of one unchanged state means the state is moving.
    a = write(tmp_path, "a.json", arm({}))
    b = write(tmp_path, "b.json", arm({"authoritative": 4}))
    result = gc.verdict(a, b)
    assert result["verdict"] == gc.NOT_CONVERGED
    assert result["disagreements"] == [
        {"set": "authoritative", "qid": "authoritative-1",
         "first": 1, "second": 4}]


def test_a_gold_falling_out_of_the_top_ten_counts(tmp_path):
    a = write(tmp_path, "a.json", arm({}))
    b = write(tmp_path, "b.json", arm({"legal": None}))
    assert gc.verdict(a, b)["verdict"] == gc.NOT_CONVERGED


def test_the_default_tolerance_is_zero(tmp_path):
    a = write(tmp_path, "a.json", arm({}))
    b = write(tmp_path, "b.json", arm({"names": 2}))
    assert gc.verdict(a, b)["tolerance"] == 0
    assert gc.verdict(a, b)["verdict"] == gc.NOT_CONVERGED
    # a caller may widen it, and the record says it did
    assert gc.verdict(a, b, tolerance=1)["verdict"] == gc.CONVERGED


# ------------------------------------------------------- fails closed #

def test_a_missing_arm_is_unknown_not_a_pass(tmp_path):
    a = write(tmp_path, "a.json", arm({}))
    with pytest.raises(gc.ConvergenceUnknown, match="absent"):
        gc.verdict(a, tmp_path / "nope.json")


def test_an_unreadable_arm_is_unknown(tmp_path):
    a = write(tmp_path, "a.json", arm({}))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(gc.ConvergenceUnknown, match="unreadable"):
        gc.verdict(a, bad)


def test_an_arm_with_cache_misses_is_unknown(tmp_path):
    # Task 600: an arm whose 1,418 queries all errored was written and handed
    # on as a reading. "No evidence" and "evidence of agreement" are not the
    # same thing.
    a = write(tmp_path, "a.json", arm({}))
    b = write(tmp_path, "b.json", arm({}, cache_miss=16))
    with pytest.raises(gc.ConvergenceUnknown, match="cache_miss"):
        gc.verdict(a, b)


def test_an_arm_missing_a_bar_set_is_unknown(tmp_path):
    a = write(tmp_path, "a.json", arm({}))
    b = write(tmp_path, "b.json", arm({}, sets=("authoritative", "foundation")))
    with pytest.raises(gc.ConvergenceUnknown, match="does not score the set"):
        gc.verdict(a, b)


def test_two_different_populations_are_unknown_not_converged(tmp_path):
    # An aggregate cannot acquit a component: comparing 300 queries against 200
    # and finding no disagreement among the shared ones is not convergence.
    a = write(tmp_path, "a.json", arm({}))
    doc = arm({})
    doc["sets"]["authoritative"]["per_q"]["extra-9"] = {
        "rank": 1, "gold_urls": ["https://example.org/b"], "gold_domains": []}
    b = write(tmp_path, "b.json", doc)
    with pytest.raises(gc.ConvergenceUnknown, match="different populations"):
        gc.verdict(a, b)


def test_the_task370_arm_shape_is_actually_compared(tmp_path):
    """The regression that made the first version of this guard decoration.

    The Task370 arms the content gate produces carry only
    `{rank, urls, trace}` -- no gold metadata. Selecting the population on
    `gold_urls` made it empty, so nothing was compared and every pair came back
    CONVERGED. These arms must be compared, and a moved rank in them must fail.
    """
    def t370(ranks):
        return {"cache_miss": 0, "sets": {
            name: {"per_q": {f"{name}-1": {
                "rank": ranks.get(name, 1), "urls": ["https://example.org/a"],
                "trace": {}}}} for name in gc.BAR_SETS}}
    a = write(tmp_path, "a.json", t370({}))
    b = write(tmp_path, "b.json", t370({}))
    moved = write(tmp_path, "c.json", t370({"authoritative": 4}))
    assert gc.verdict(a, b)["queries_compared"] == len(gc.BAR_SETS)
    assert gc.verdict(a, b)["verdict"] == gc.CONVERGED
    assert gc.verdict(a, moved)["verdict"] == gc.NOT_CONVERGED


def test_an_arm_that_reports_no_rank_at_all_is_unknown(tmp_path):
    """Nothing to compare is UNKNOWN, never agreement."""
    empty = {"cache_miss": 0, "sets": {
        name: {"per_q": {f"{name}-1": {"urls": []}}} for name in gc.BAR_SETS}}
    a = write(tmp_path, "a.json", empty)
    b = write(tmp_path, "b.json", empty)
    with pytest.raises(gc.ConvergenceUnknown, match="nothing to compare"):
        gc.verdict(a, b)


# --------------------------------------------------- the exit code #

def test_main_carries_the_verdict_in_an_exit_code(tmp_path):
    a = write(tmp_path, "a.json", arm({}))
    b = write(tmp_path, "b.json", arm({}))
    moved = write(tmp_path, "c.json", arm({"foundation": 3}))
    assert gc.main(["--arm", str(a), "--arm", str(b)]) == 0
    assert gc.main(["--arm", str(a), "--arm", str(moved)]) == 1
    assert gc.main(["--arm", str(a), "--arm", str(tmp_path / "gone.json")]) == 2
    assert gc.main(["--arm", str(a)]) == 2                 # one arm is not two
    assert gc.main(["--arm", str(a), "--arm", str(b), "--arm", str(b)]) == 2


def test_main_writes_the_receipt_it_is_asked_for(tmp_path):
    a = write(tmp_path, "a.json", arm({}))
    b = write(tmp_path, "b.json", arm({"legal": 5}))
    out = tmp_path / "receipt.json"
    assert gc.main(["--arm", str(a), "--arm", str(b), "--out", str(out)]) == 1
    receipt = json.loads(out.read_text())
    assert receipt["verdict"] == gc.NOT_CONVERGED
    assert receipt["schema"] == "aw.gate-convergence.v1"
    assert len(receipt["disagreements"]) == 1


# ------------------------------------------------- the measured contract #

def contract(tmp_path, **overrides):
    doc = json.loads(gc.CONTRACT_PATH.read_text())
    doc.update(overrides)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(doc))
    return path


def test_the_shipped_contract_loads_and_carries_the_measured_defaults():
    doc = gc.load_contract()
    # The numbers themselves are ratified by the merge of ADR-043 am.5; what
    # this pins is that they are READ, and read from data.
    assert doc["settle_seconds"] == 480 and doc["max_rounds"] == 3
    assert tuple(doc["bar_sets"]) == gc.BAR_SETS
    assert doc["tolerance"] == 0


def test_a_missing_or_unreadable_contract_is_unknown(tmp_path):
    with pytest.raises(gc.ConvergenceUnknown, match="absent"):
        gc.load_contract(tmp_path / "gone.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(gc.ConvergenceUnknown, match="unreadable"):
        gc.load_contract(broken)


def test_the_contract_may_not_soften_the_standard(tmp_path):
    # The ways a run could get itself through by editing config.
    with pytest.raises(gc.ConvergenceUnknown, match="tolerance"):
        gc.load_contract(contract(tmp_path, tolerance=1))
    with pytest.raises(gc.ConvergenceUnknown, match="bar_sets"):
        gc.load_contract(contract(tmp_path, bar_sets=["authoritative"]))
    with pytest.raises(gc.ConvergenceUnknown, match="settle_seconds"):
        gc.load_contract(contract(tmp_path, settle_seconds=0))
    with pytest.raises(gc.ConvergenceUnknown, match="max_rounds"):
        gc.load_contract(contract(tmp_path, max_rounds=0))
    with pytest.raises(gc.ConvergenceUnknown, match="contract"):
        gc.load_contract(contract(tmp_path, schema="aw.something-else.v1"))


# ------------------------------------------------------- the protocol #

def protocol_arms(tmp_path, *docs):
    """A `cut_arm` that hands back these readings, in order."""
    paths = [write(tmp_path, f"arm{index}.json", doc)
             for index, doc in enumerate(docs)]
    seen = []

    def cut_arm(index):
        seen.append(index)
        return paths[len(seen) - 1]

    return cut_arm, seen, paths


def test_the_protocol_settles_between_the_two_readings(tmp_path):
    cut_arm, seen, paths = protocol_arms(tmp_path, arm({}), arm({}))
    slept = []
    result = gc.converged_arms(cut_arm, sleep=slept.append, log=lambda *_: None)
    assert result["verdict"] == gc.CONVERGED
    assert seen == [0, 1] and slept == [480]
    # The verdict is computed on the LATER reading -- the settled one.
    assert result["arm"] == str(paths[1])
    assert result["rounds"] == 1


def test_a_moving_state_is_re_scored_up_to_the_bound_then_unadjudicated(tmp_path):
    cut_arm, seen, _ = protocol_arms(
        tmp_path, arm({}), arm({"legal": 2}), arm({}), arm({"names": 4}))
    slept = []
    with pytest.raises(gc.ConvergenceUnadjudicated) as caught:
        gc.converged_arms(cut_arm, sleep=slept.append, log=lambda *_: None)
    assert seen == [0, 1, 2, 3]                    # 1 + 3 bounded rounds
    assert slept == [480, 480, 480]
    assert caught.value.receipt["verdict"] == gc.UNADJUDICATED
    assert len(caught.value.receipt["receipts"]) == 3
    # It is not a FAIL and it says so: nothing was measured about the pack.
    assert "UNADJUDICATED" in str(caught.value)
    assert "not a FAIL" in str(caught.value)


def test_a_state_that_settles_on_a_later_round_converges(tmp_path):
    cut_arm, seen, paths = protocol_arms(
        tmp_path, arm({}), arm({"legal": 2}), arm({"legal": 2}))
    result = gc.converged_arms(cut_arm, sleep=lambda _: None,
                               log=lambda *_: None)
    assert result["verdict"] == gc.CONVERGED and result["rounds"] == 2
    assert result["arm"] == str(paths[2]) and seen == [0, 1, 2]


def test_readiness_is_recorded_and_a_reading_that_cannot_be_taken_aborts(tmp_path):
    cut_arm, _, _ = protocol_arms(tmp_path, arm({}), arm({}))
    snapshot = {"ready": False, "problems": ["2 non-indexed segment(s)"]}
    result = gc.converged_arms(cut_arm, readiness=lambda: snapshot,
                               sleep=lambda _: None, log=lambda *_: None)
    # Task 609: the predicate was false at BOTH readings of this transition,
    # so an unready snapshot is evidence filed with the verdict, not a refusal.
    assert result["verdict"] == gc.CONVERGED
    assert result["readiness"] == snapshot

    cut_arm, _, _ = protocol_arms(tmp_path, arm({}), arm({}))

    def unreadable():
        raise RuntimeError("cannot read http://twin/telemetry")

    with pytest.raises(RuntimeError, match="cannot read"):
        gc.converged_arms(cut_arm, readiness=unreadable, sleep=lambda _: None,
                          log=lambda *_: None)


def test_an_uncomparable_pair_stays_unknown_inside_the_protocol(tmp_path):
    cut_arm, _, _ = protocol_arms(tmp_path, arm({}), arm({}, cache_miss=3))
    with pytest.raises(gc.ConvergenceUnknown, match="cache_miss"):
        gc.converged_arms(cut_arm, sleep=lambda _: None, log=lambda *_: None)


# --------------------------------- task 609's real arms, not a fixture #

FIXTURE = json.loads(
    (pathlib.Path(__file__).resolve().parent / "testdata" /
     "task609_convergence_arms.json").read_text())


def real_arm(tmp_path, name):
    return write(tmp_path, f"{name}.json", FIXTURE["arms"][name])


def test_the_real_pair_that_refused_the_news_pack_is_not_converged(tmp_path):
    """The measurement ADR-043 am.5 rests on, re-derived here.

    `applier_gate` is the content gate's own candidate arm at `wait_green`+0;
    `applier_recut` is the same script on the same collection eight minutes
    later. Task 609's committed receipt says 31 of 691; so does this.
    """
    result = gc.verdict(real_arm(tmp_path, "applier_gate"),
                        real_arm(tmp_path, "applier_recut"))
    assert result["verdict"] == gc.NOT_CONVERGED
    assert result["queries_compared"] == 691
    assert len(result["disagreements"]) == 31
    moved = {row["qid"]: (row["first"], row["second"])
             for row in result["disagreements"]}
    # Golds task 609 names in prose, each a recall event that reversed itself
    # with nothing about the corpus changed between the two readings.
    assert moved["ch-tax-003"][0] > moved["ch-tax-003"][1]
    assert moved["de-law-013"] == (7, 1)


def test_the_real_settled_pair_converges(tmp_path):
    result = gc.verdict(real_arm(tmp_path, "rollback_t0"),
                        real_arm(tmp_path, "rollback_settled"))
    assert result["verdict"] == gc.CONVERGED
    assert result["queries_compared"] == 691
    assert result["disagreements"] == []


def test_the_fixture_reproduces_task_609s_own_receipts(tmp_path):
    """Two derivations of the same numbers, from two different runs.

    Task 609 ran `gate_convergence` against the 17 MB and 63 MB arms on
    the build box and committed the receipts; this reruns it against the projection
    of those same files. If the projection ever stops carrying what the guard
    judges, these disagree.
    """
    receipts = gc.CONTRACT_PATH.parents[1] / "tasks" / "609-artifacts"
    if not receipts.is_dir():
        pytest.skip("the task tree is not in this checkout (the public cut)")
    applier = json.loads((receipts / "convergence-applier.json").read_text())
    endstate = json.loads((receipts / "convergence-endstate.json").read_text())
    here = gc.verdict(real_arm(tmp_path, "applier_gate"),
                      real_arm(tmp_path, "applier_recut"))
    assert here["verdict"] == applier["verdict"]
    assert here["queries_compared"] == applier["queries_compared"]
    assert ({(row["set"], row["qid"], row["first"], row["second"])
             for row in here["disagreements"]} ==
            {(row["set"], row["qid"], row["first"], row["second"])
             for row in applier["disagreements"]})
    settled = gc.verdict(real_arm(tmp_path, "rollback_t0"),
                         real_arm(tmp_path, "rollback_settled"))
    assert settled["verdict"] == endstate["verdict"]
    assert settled["queries_compared"] == endstate["queries_compared"]


def test_the_protocol_on_the_real_arms_refuses_the_transient_pair(tmp_path):
    """End to end on the real readings: the pair the applier actually cut."""
    cut_arm, _, _ = protocol_arms(tmp_path, FIXTURE["arms"]["applier_gate"],
                                  FIXTURE["arms"]["applier_recut"],
                                  FIXTURE["arms"]["rollback_t0"],
                                  FIXTURE["arms"]["rollback_settled"])
    result = gc.converged_arms(cut_arm, sleep=lambda _: None,
                               log=lambda *_: None)
    # Round 1 is the pair that refused the pack: 31 of 691 differ, so no
    # verdict is computed on it. Round 2 agrees on all 691 -- both of its
    # readings are settled ones, and they coincide because on a converged
    # collection the pack moves no bar-set gold, which is task 609's finding.
    assert [len(item["disagreements"]) for item in result["receipts"]] == [31, 0]
    assert result["verdict"] == gc.CONVERGED and result["rounds"] == 2


# ------------------------------------------------ am.9 layout fast path #

def fp(*, points=100, n=100, stale=0, opt_end="", startup="s1"):
    """A minimal `collection_layout.build`-shaped fingerprint for tests."""
    return {"points_count": points, "n_per_vector": n,
            "stale_postings": stale, "qdrant_startup": startup,
            "optimizer_entries": ([{"name": "merge", "start_at": "t0",
                                   "end_at": opt_end, "segment_ids": [0]}]
                                  if opt_end else [])}


def test_am9_is_off_unless_the_env_flag_is_exactly_layout(tmp_path, monkeypatch):
    """Default (unset) behaves exactly as before: settle_seconds, no am.9
    key claiming credit for a round it did not decide."""
    monkeypatch.delenv(gc.AM9_ENV, raising=False)
    cut_arm, _, _ = protocol_arms(tmp_path, arm({}), arm({}))
    slept = []
    result = gc.converged_arms(cut_arm, sleep=slept.append, log=lambda *_: None,
                               layout_fingerprint=lambda: fp())
    assert slept == [480]
    assert result["am9_layout_quiet"] is False
    assert result["receipts"][0]["am9"]["on"] is False


def test_am9_quiet_layout_converges_on_the_short_wait(tmp_path, monkeypatch):
    """The mechanism-justified fast path: no optimizer activity, identical
    fingerprint, 60s instead of 480s -- and it works even though the SCORES
    disagree, because am.9 is an ALTERNATIVE route to CONVERGED, not merely a
    faster version of the scored rule (task 627's mechanism: no layout
    movement -> no reason for the BM25 leg to have moved either)."""
    monkeypatch.setenv(gc.AM9_ENV, "layout")
    # Scores disagree (legal 1 -> 2) -- the classic rule alone would refuse
    # this pair. The layout fast path still declares it CONVERGED.
    cut_arm, seen, _ = protocol_arms(tmp_path, arm({}), arm({"legal": 2}))
    slept = []
    same = fp()
    result = gc.converged_arms(cut_arm, sleep=slept.append, log=lambda *_: None,
                               layout_fingerprint=lambda: same)
    assert slept == [60]
    assert result["verdict"] == gc.CONVERGED
    assert result["am9_layout_quiet"] is True
    assert result["receipts"][0]["am9"] == {
        "on": True, "settle_seconds": 60, "layout_quiet": True,
        "reason": "optimizer log quiet, layout fingerprint unchanged"}
    assert seen == [0, 1]


def test_am9_falls_back_to_the_full_wait_when_the_optimizer_ran(tmp_path, monkeypatch):
    monkeypatch.setenv(gc.AM9_ENV, "layout")
    cut_arm, _, _ = protocol_arms(tmp_path, arm({}), arm({}))
    slept = []
    reads = [fp(), fp(n=101, opt_end="t1")]  # optimizer entry appears
    result = gc.converged_arms(cut_arm, sleep=slept.append, log=lambda *_: None,
                               layout_fingerprint=lambda: reads.pop(0))
    # round 1's layout check fails (optimizer ran) -- falls back to scoring
    # the pair it already has, which happens to agree, so it still converges
    # this round, just not via the am.9 shortcut.
    assert slept == [60]
    assert result["verdict"] == gc.CONVERGED
    assert result["am9_layout_quiet"] is False
    assert result["receipts"][0]["am9"]["layout_quiet"] is False
    assert result["receipts"][0]["am9"]["reason"] == \
        "the optimizer ran between the two readings"


def test_am9_falls_back_when_the_fingerprint_is_unreadable(tmp_path, monkeypatch):
    monkeypatch.setenv(gc.AM9_ENV, "layout")
    cut_arm, _, _ = protocol_arms(tmp_path, arm({}), arm({}))
    slept = []

    def broken():
        raise RuntimeError("qdrant unreachable")

    result = gc.converged_arms(cut_arm, sleep=slept.append, log=lambda *_: None,
                               layout_fingerprint=broken)
    # the very first fingerprint read fails, so am9 never arms for round 1 --
    # the classic 480s wait is used, and it converges by the classic rule.
    assert slept == [480]
    assert result["verdict"] == gc.CONVERGED
    assert result["am9_layout_quiet"] is False


def test_am9_never_declares_convergence_from_a_raw_n_change_without_an_optimizer_entry(tmp_path, monkeypatch):
    """A moved fingerprint with NO optimizer log entry to explain it is not
    trusted as quiet -- it is exactly the kind of surprise am.9 exists to be
    conservative about, not the case it was written to fast-path."""
    monkeypatch.setenv(gc.AM9_ENV, "layout")
    # Only one pair is offered; scores disagree every round, so the bound (3
    # rounds) runs out and the protocol raises UNADJUDICATED -- exactly as
    # it would with am.9 off, proving the fast path is additive, not looser.
    cut_arm, _, _ = protocol_arms(
        tmp_path, arm({}), arm({"legal": 2}), arm({}), arm({"legal": 3}))
    slept = []
    # every CONSECUTIVE pair differs (never a repeat), so the layout check
    # never sees an unchanged fingerprint across all 4 reads this run needs
    reads = iter([fp(n=100), fp(n=105), fp(n=110), fp(n=115)])
    with pytest.raises(gc.ConvergenceUnadjudicated) as caught:
        gc.converged_arms(cut_arm, sleep=slept.append, log=lambda *_: None,
                          layout_fingerprint=lambda: next(reads))
    # scores disagree AND layout is not trusted quiet -> not converged on any
    # round; every round used the short 60s wait (the fast path stayed armed
    # since each fingerprint read succeeded, just never came back quiet).
    assert slept == [60, 60, 60]
    assert caught.value.receipt["verdict"] == gc.UNADJUDICATED
    assert caught.value.receipt["receipts"][0]["am9"]["layout_quiet"] is False
    assert caught.value.receipt["receipts"][0]["am9"]["reason"] == \
        "the layout fingerprint moved with no optimizer entry to explain it"
