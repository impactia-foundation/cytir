#!/usr/bin/env python3
"""Tests for `cytir/gates/gold_presence.py` (task 544, ported task 1820).

Every case replays a SEALED capture of what the twin and the `aw_e8` candidate
actually held on 2026-09-08 (`tests/testdata/gold_presence_sealed.json`), taken once
through the gate's own indexed-key backend. No test here opens a socket: a
guard whose test needs a running qdrant is a guard nobody runs.

The real record under test: three of the seven `names` golds --
`impactia.org/foundation-board/?lang=en`, `impactia.org/the-team/?lang=en` and
`impactia.org/robotme/` -- are held by the twin and absent from `aw_e8`, while
all sixteen `entity_card_golds` cards are present in both. Task 538 read the
same movement as one entity-card ranking defect.
"""
from __future__ import annotations

import json
import os

import pytest

from cytir.gates import gold_presence as G

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "testdata", "gold_presence_sealed.json")


@pytest.fixture(scope="module")
def sealed():
    with open(FIXTURE) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def backends(sealed):
    return (G.sealed_backend(sealed["candidate"]),
            G.sealed_backend(sealed["incumbent"]))


# The real gate sets (`names`, `entity_card_golds`) are repo-only: the public cut withholds
# them, so the tests that read them skip inside a cut tree instead of refusing.
_SETS_ON_DISK = all(os.path.exists(path) for name, path in G.GA.SETS
                    if name in ("names", "entity_card_golds"))
needs_gate_sets = pytest.mark.skipif(
    not _SETS_ON_DISK, reason="gate sets are not in this tree (public cut withholds them)")


CONTROL = ["https://www.vector8.com/en"]          # present in both, sealed
LOST = {"https://impactia.org/foundation-board/?lang=en",
        "https://impactia.org/the-team/?lang=en",
        "https://impactia.org/robotme/"}


def test_fixture_is_a_real_capture(sealed):
    assert sealed["schema"] == "aw.task544-gold-presence-fixture.v1"
    assert sealed["candidate_collection"] == "aw_e8"
    assert sealed["incumbent_collection"] == "aw_e3"
    # 7 names golds + 16 entity-card golds, deduplicated by url: the two
    # Vector8 queries and the two Elsys/Moviiu bridge pairs share a gold url.
    assert len(sealed["candidate"]) == len(sealed["incumbent"]) == 20


@needs_gate_sets
def test_the_gate_fails_on_the_measured_coverage_loss(backends):
    cand, inc = backends
    report, code = G.run(cand, inc, set_names=["names", "entity_card_golds"],
                         candidate_control=CONTROL, incumbent_control=CONTROL)
    assert code == 2, "three golds the twin holds are absent from aw_e8"
    assert report["verdict"] == "FAIL"
    assert {r["url"] for r in report["lost"]} == LOST
    assert {r["id"] for r in report["lost"]} == {"n02", "n04", "n07"}


@needs_gate_sets
def test_it_names_the_lane_per_set(backends):
    """The whole point: `names` is coverage, `entity_card_golds` is not."""
    cand, inc = backends
    report, _ = G.run(cand, inc, set_names=["names", "entity_card_golds"],
                      candidate_control=CONTROL, incumbent_control=CONTROL)
    assert report["per_set"]["names"]["LOST_BY_CANDIDATE"] == 3
    assert "LOST_BY_CANDIDATE" not in report["per_set"]["entity_card_golds"]
    # 16 QUERIES, 13 distinct cards: the two Vector8 rows and the two
    # crosslingual bridge pairs share a gold url, and the roster is per gold
    # ROW, not per distinct url -- so a lost card would show up once per query
    # that depends on it, which is the granularity a class score moves at.
    assert report["per_set"]["entity_card_golds"]["present_both"] == 16


@needs_gate_sets
def test_a_clean_candidate_passes(sealed):
    """No false alarm: candidate == incumbent must exit 0."""
    inc = G.sealed_backend(sealed["incumbent"])
    report, code = G.run(inc, inc, set_names=["names", "entity_card_golds"],
                         candidate_control=CONTROL, incumbent_control=CONTROL)
    assert code == 0 and report["verdict"] == "PASS" and report["n_lost"] == 0


@needs_gate_sets
def test_absent_from_both_is_not_a_failure(sealed):
    """A gold nobody holds is a pre-existing gap, not this build's regression."""
    gone = {u: False for u in sealed["candidate"]}
    gone["https://www.vector8.com/en"] = True      # keep the control alive
    both = G.sealed_backend(gone)
    report, code = G.run(both, both, set_names=["names", "entity_card_golds"],
                         candidate_control=CONTROL, incumbent_control=CONTROL)
    assert code == 0
    # 23 gold rows (7 names + 16 card queries); the two rows sharing the
    # still-present control url are the only survivors.
    assert report["n_golds"] == 23
    assert report["n_absent_both"] == 21 and report["n_lost"] == 0


def test_url_variants_are_all_tried(sealed):
    """A trailing slash is not a genuine absence (corpus-doctrine 5)."""
    stored = G.sealed_backend({"https://triskel.ch": True})
    assert stored("https://triskel.ch/") is True
    assert stored("https://triskel.ch") is True
    assert stored("https://triskel.ch/#anchor") is True
    assert stored("https://elsewhere.example/") is False


# --------------------------------------------------------------------------
# task 553 (task 545 sec 2.1) -- `ch-law-003`: a coverage test must use the
# GOLD's own url, never a twin's or a candidate's matched hit string. The
# twin's rank-1 hit was a finer fragment than the gold's own target; the
# candidate held the gold's exact url at rank 9 -- a ranking loss, not a
# coverage one -- and the OLD `twin_hit_presence.py` mislabeled it because
# it tested the wrong string.
# --------------------------------------------------------------------------
CH_LAW_003_TWIN_HIT = "https://www.fedlex.admin.ch/eli/cc/2009/615/en#art_10-c2"
CH_LAW_003_GOLD_URL = "https://www.fedlex.admin.ch/eli/cc/2009/615/en#art_10"


def test_ch_law_003_old_logic_misclassified_as_coverage():
    """Sealed proof the defect existed. The candidate never held the twin's
    exact fragment (`#art_10-c2`) -- only the gold's own, coarser one
    (`#art_10`). The OLD script's check is equivalent to
    `classify_lost_or_worse(present, [], twin_hit_url)`: it forces the
    domain-only fallback identity even for a gold that DOES carry its own
    url, so it tests `twin_hit_url` and reports COVERAGE."""
    present = G.sealed_backend({CH_LAW_003_GOLD_URL: True})
    assert present(CH_LAW_003_TWIN_HIT) is False, (
        "the candidate must NOT hold the twin's exact hit fragment -- "
        "that absence is what the old instrument mistook for a coverage gap")
    old_verdict = G.classify_lost_or_worse(present, [], CH_LAW_003_TWIN_HIT)
    assert old_verdict == "COVERAGE"


def test_ch_law_003_new_logic_classifies_as_ranking():
    """The fix. Given the gold's own url (exactly what `gate_authority.
    gold_of` returns for this row), the SAME candidate capture -- which
    holds the gold's exact fragment, just not the twin's finer one --
    classifies correctly as RANKING: the candidate has it, at a worse rank,
    not zero rows."""
    present = G.sealed_backend({CH_LAW_003_GOLD_URL: True})
    new_verdict = G.classify_lost_or_worse(
        present, [CH_LAW_003_GOLD_URL], CH_LAW_003_TWIN_HIT)
    assert new_verdict == "RANKING"


def test_coverage_identity_unions_gold_url_and_twin_hit():
    """`coverage_identity_urls` always includes the gold's own url when it
    has one (the fix), AND folds in the twin's hit alongside it rather than
    dropping it (`rank_of`'s page-level leniency runs both ways -- see
    `eu-001`/`eu-011`/`eu-003`, where the gold names no fragment at all but
    the candidate only stores per-article fragments, so gold-url-only
    probing would misreport a document task 547 just landed as absent). A
    gold with no url of its own has nothing but the twin's hit to test,
    same as before this fix."""
    assert G.coverage_identity_urls(
        ["https://gold.example/a"], "https://twin-hit.example/b"
    ) == ["https://gold.example/a", "https://twin-hit.example/b"]
    assert G.coverage_identity_urls([], "https://twin-hit.example/b") == [
        "https://twin-hit.example/b"]
    assert G.coverage_identity_urls([], None) == []
    # no duplicate probe when the twin's hit already IS the gold's own url
    assert G.coverage_identity_urls(
        ["https://gold.example/a"], "https://gold.example/a"
    ) == ["https://gold.example/a"]


def test_classify_lost_or_worse_is_unscorable_with_no_identity_at_all():
    present = G.sealed_backend({})
    assert G.classify_lost_or_worse(present, [], None) is None


def test_a_dead_channel_is_refused_not_recorded_as_zero(sealed):
    """verification-doctrine 6 -- an error is not an empty."""
    dead = G.sealed_backend({})
    live = G.sealed_backend(sealed["incumbent"])
    with pytest.raises(G.Refusal):
        G.run(dead, live, set_names=["names"],
              candidate_control=CONTROL, incumbent_control=CONTROL)


def test_no_control_url_is_refused(backends):
    """A gate that will record zeros from an unproven channel is decoration."""
    cand, inc = backends
    with pytest.raises(G.Refusal):
        G.run(cand, inc, set_names=["names"],
              candidate_control=[], incumbent_control=CONTROL)


def test_an_unknown_set_cannot_produce_a_vacuous_pass(backends):
    cand, inc = backends
    with pytest.raises(G.Refusal):
        G.run(cand, inc, set_names=["names", "no_such_set"],
              candidate_control=CONTROL, incumbent_control=CONTROL)


@needs_gate_sets
def test_roster_comes_from_the_gate_sets_not_a_hand_list():
    roster = G.gold_roster(["names"])
    assert {r["id"] for r in roster} == {f"n0{i}" for i in range(1, 8)}
    assert all(r["url"].startswith("https://impactia.org/") for r in roster)


def test_an_unindexed_presence_key_is_refused(monkeypatch):
    """A filter on an unindexed key is a full scan, not a lookup."""
    monkeypatch.setattr(
        G.IH, "fetch_json",
        lambda *a, **k: {"result": {"payload_schema": {"domain": {}}}})
    with pytest.raises(G.Refusal):
        G.qdrant_backend("http://unused.invalid", "aw_e8")


def test_main_returns_a_nonzero_exit_code_on_refusal(monkeypatch):
    """The verdict travels in the exit code, or it is not enforcement."""
    monkeypatch.setattr(G, "qdrant_backend",
                        lambda *a, **k: (_ for _ in ()).throw(
                            G.Refusal("collection unreachable")))
    assert G.main(["--qdrant", "http://x.invalid", "--collection", "aw_e8",
                   "--control-url", CONTROL[0]]) == 3


def _prod_scroll_fake(monkeypatch):
    """Target = prod (faked), a POST scroll; `fetch_json` and the audit spend are counters."""
    from cytir.index import prod_guard as PG
    calls = []
    monkeypatch.setattr(PG, "is_prod", lambda url, collection=None: True)
    monkeypatch.setattr(PG, "_spend", lambda *a, **k: None)
    monkeypatch.setattr(G.IH, "fetch_json",
                        lambda url, **k: calls.append((url, k)) or {"result": {"points": []}})
    return calls


def test_prod_scroll_without_a_reason_is_refused_before_any_http_call(monkeypatch):
    calls = _prod_scroll_fake(monkeypatch)
    with pytest.raises(G.Refusal):
        G._api("http://prod.invalid:6333", "/collections/aw_e8/points/scroll",
               {"limit": 1}, method="POST")
    assert calls == []


def test_prod_scroll_with_a_reason_makes_exactly_one_call_with_a_600s_timeout(monkeypatch):
    calls = _prod_scroll_fake(monkeypatch)
    G._api("http://prod.invalid:6333", "/collections/aw_e8/points/scroll",
           {"limit": 1}, method="POST", prod_scroll_reason="review 1828 test")
    assert len(calls) == 1
    assert calls[0][1]["timeout"] == 600.0


def test_every_api_call_carries_the_600s_timeout(monkeypatch):
    calls = []
    monkeypatch.setattr(G.PG, "check", lambda *a, **k: False)
    monkeypatch.setattr(G.IH, "fetch_json",
                        lambda url, **k: calls.append(k) or {"result": {}})
    G._api("http://local.invalid:6333", "/collections/aw_e8", None)
    assert calls and all(k.get("timeout") == 600.0 for k in calls)
