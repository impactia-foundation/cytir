#!/usr/bin/env python3
"""Task 627 — the layout fingerprint and the comparability gap, tested against
the REAL fingerprints read off the two live mirrors on 2026-09-21 at 09:54:39Z.

The fixture (`tests/testdata/task627_layout_fingerprints.json`) is sealed
from the pre-apply snapshot of 2026-09-17, taken while both
mirrors held a byte-equal membership of 33,734,809 points — the pair that was
measured to cost ZERO h@1 (206 on both, h@10 350 vs 349, 2 golds moved) and that
is therefore the calibration point behind
`collection_layout.MEASURED_SAFE_STALE_RATIO_GAP`. Everything asserted here is a
number the record actually produced; nothing is invented.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cytir.index import layout as CL  # noqa: E402

FIXTURE = json.loads(
    (ROOT / "tests" / "testdata"
     / "task627_layout_fingerprints.json").read_text())
FP = FIXTURE["fingerprints"]


def test_the_fixture_is_the_real_2026_09_21_reading():
    """Guard the provenance, so a later edit cannot quietly re-calibrate."""
    assert FIXTURE["captured_at"] == "2026-09-21T09:54:39Z"
    for name in ("prod", "twin"):
        assert FP[name]["points_count"] == 33_734_809, name
    assert FP["prod"]["indexed_vectors_count"] == 69_228_285
    assert FP["twin"]["indexed_vectors_count"] == 69_241_144


def test_n_is_not_points_count_and_the_gap_is_the_error_term():
    """The finding in one assertion: the IDF divisor is NOT the live count."""
    prod = FP["prod"]
    assert prod["n_per_vector"] == 69_228_285 / 2
    assert prod["n_per_vector"] > prod["points_count"]
    assert round(prod["stale_postings"]) == 879_334
    assert abs(prod["stale_ratio"] - 0.026066) < 1e-5


def test_threshold_is_derived_from_its_basis_not_typed():
    """MEASURED_SAFE_STALE_RATIO_GAP must equal the gap it was calibrated on.

    If someone re-types the constant as a round number, or edits the basis
    without re-deriving, this fails — which is the point. The threshold is a
    measurement, not a tolerance somebody chose.
    """
    measured = CL.gap(FP["prod"], FP["twin"])["stale_ratio_gap"]
    assert CL.MEASURED_SAFE_STALE_RATIO_GAP == measured
    # and it is a small number, not a licence: ~0.019% of the corpus
    assert 1e-4 < measured < 3e-4


def test_the_measured_safe_pair_is_adjudicable():
    """The pair that cost zero h@1 must pass, or the gate never opens."""
    g = CL.gap(FP["prod"], FP["twin"])
    assert g["same_membership"] is True
    assert g["same_target"] is False
    assert g["within_measured_safe"] is True
    assert round(g["n_gap"]) == 6_430


def test_a_wider_gap_is_refused():
    """Double the staleness on one side and the pair stops being comparable.

    The multiplier is deliberate and not a boundary-poke: the 2026-09-20 pair
    that cost 5 h@1 had a gap nobody captured, so the only honest statement is
    "wider than the one measured-safe point is not adjudicable".
    """
    worse = dict(FP["twin"],
                 indexed_vectors_count=FP["twin"]["indexed_vectors_count"]
                 + 2 * 6_430)
    worse = CL.build(
        collection=worse["collection"], url=worse["url"],
        physical=worse["physical"], points_count=worse["points_count"],
        indexed_vectors_count=worse["indexed_vectors_count"],
        named_vectors=worse["named_vectors"],
        segments_count=worse["segments_count"],
        segment_sizes=worse["segment_sizes"], status=worse["status"],
        optimizer_status=worse["optimizer_status"],
        qdrant_startup=worse["qdrant_startup"],
        optimizer_entries=worse["optimizer_entries"])
    assert CL.gap(FP["prod"], worse)["within_measured_safe"] is False


def test_green_and_ok_are_not_settle_predicates():
    """Both mirrors report green/ok and NEITHER is settled. This is the whole
    reason `status` was never enough."""
    for name in ("prod", "twin"):
        assert FP[name]["status"] == "green"
        assert FP[name]["optimizer_status"] == "ok"
        reasons = CL.settled(FP[name])
        assert reasons, f"{name} reads green/ok and carries 2.6% stale postings"
        assert "stale postings" in reasons[0]


def test_prods_883_point_segment_is_reported():
    """Prod really carries an 883-point segment beside twelve of 2.3-3.6M. A
    segment under the 10k indexing threshold scores a filtered sparse query
    differently (task 606) and `status: green` does not mention it."""
    assert 883 in FP["prod"]["segment_sizes"]
    assert any("plain segment" in r for r in CL.settled(FP["prod"]))
    # the twin's empty segment is not a plain segment and must not be flagged
    assert 0 in FP["twin"]["segment_sizes"]
    assert not any("plain segment" in r for r in CL.settled(FP["twin"]))


def test_the_twins_real_merges_are_detected_as_a_straddle():
    """Task 626c cut its twin baseline at 08:07Z and its candidate at 08:42Z,
    with merge[14,13,2] running 08:32:00Z->08:40:46Z between them. Those are the
    real log entries; the pair was never a converged reading."""
    ran = CL.optimizer_ran_between(FP["twin_at_baseline_0807z"], FP["twin"])
    assert len(ran) == 4
    names = [e["name"] for e in ran]
    assert names.count("merge") == 2
    first_merge = next(e for e in ran if e["name"] == "merge")
    assert first_merge["start_at"].startswith("2026-09-21T08:32:00")
    assert first_merge["segment_ids"] == [14, 13, 2]


def test_no_straddle_when_the_log_is_unchanged():
    assert CL.optimizer_ran_between(FP["twin"], FP["twin"]) == []
    assert CL.optimizer_ran_between(FP["prod"], FP["prod"]) == []


def test_quiescent_on_the_real_unchanged_pair():
    """task626 item 1/am.9: the one home `optimizer_ran_between` + field
    equality is bundled behind -- reads quiet on the SAME real reading
    against itself (no possible drift), used both by am.9's fast settle path
    and by the daily door's prod-baseline reuse decision."""
    quiet, reason = CL.quiescent(FP["prod"], FP["prod"])
    assert quiet is True
    assert reason == "optimizer log quiet, layout fingerprint unchanged"


def test_quiescent_on_the_real_straddled_pair():
    quiet, reason = CL.quiescent(FP["twin_at_baseline_0807z"], FP["twin"])
    assert quiet is False
    assert reason == "the optimizer ran between the two readings"


def test_quiescent_refuses_a_moved_fingerprint_with_no_optimizer_entry():
    moved = dict(FP["prod"])
    moved["n_per_vector"] = FP["prod"]["n_per_vector"] + 1000
    quiet, reason = CL.quiescent(FP["prod"], moved)
    assert quiet is False
    assert reason == "the layout fingerprint moved with no optimizer entry to explain it"


def test_quiescent_never_raises_on_missing_fingerprints():
    assert CL.quiescent(None, FP["prod"]) == (False, "fingerprint unavailable")
    blind = {"n_per_vector": None}
    assert CL.quiescent(blind, FP["prod"]) == (
        False, "fingerprint unreadable (n unknown)")


def test_a_restart_is_reported_rather_than_read_as_quiet():
    """A qdrant restart truncates the optimizer log, so an empty diff proves
    nothing. Absence of evidence must not read as evidence of absence."""
    restarted = dict(FP["twin"], qdrant_startup="2026-09-21T12:00:00Z",
                     optimizer_entries=[])
    ran = CL.optimizer_ran_between(FP["twin"], restarted)
    assert len(ran) == 1 and ran[0]["name"] == "qdrant restart"


def test_render_says_the_thing_an_operator_needs():
    line = CL.render(FP["prod"], "prod-baseline")
    for token in ("33,734,809", "34,614,142", "settled=NO"):
        assert token in line, line
    gap_line = CL.render_gap(CL.gap(FP["prod"], FP["twin"]))
    assert "within" in gap_line and "same_membership=True" in gap_line


def test_build_and_fingerprint_share_one_derivation():
    """The sealed fixture is re-derived through `build()`, so a change to the
    derived fields cannot make live reads and fixtures disagree."""
    rebuilt = CL.build(
        collection=FP["prod"]["collection"], url=FP["prod"]["url"],
        physical=FP["prod"]["physical"],
        points_count=FP["prod"]["points_count"],
        indexed_vectors_count=FP["prod"]["indexed_vectors_count"],
        named_vectors=FP["prod"]["named_vectors"],
        segments_count=FP["prod"]["segments_count"],
        segment_sizes=FP["prod"]["segment_sizes"],
        status=FP["prod"]["status"],
        optimizer_status=FP["prod"]["optimizer_status"],
        qdrant_startup=FP["prod"]["qdrant_startup"],
        optimizer_entries=FP["prod"]["optimizer_entries"])
    assert rebuilt == FP["prod"]


def test_a_missing_n_is_not_a_small_gap():
    """An unreadable indexed-vector count must never render as `gap 0`."""
    blind = CL.build(
        collection="x", url=None, physical=None, points_count=10,
        indexed_vectors_count=0, named_vectors=0, segments_count=1,
        segment_sizes=[10], status="green", optimizer_status="ok",
        qdrant_startup=None, optimizer_entries=[])
    assert blind["n_per_vector"] is None
    g = CL.gap(blind, FP["prod"])
    assert g["stale_ratio_gap"] is None
    assert g["within_measured_safe"] is False
    assert "UNAVAILABLE" in CL.render_gap(g)
    assert CL.settled(blind)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))


# --------------------------------------------------- render() is an OUTPUT path
def test_render_survives_a_partial_fingerprint():
    """Task 629 P4 found this the hard way: `render({})` raised on formatting a
    None `points_count`.

    A renderer that raises turns an unreadable telemetry response into a traceback
    in the middle of a gate run, losing the very reading the operator needed to see
    what was wrong. Deciding whether a collection is settled belongs to `settled()`
    and to the caller; `render()` only reports, and says UNKNOWN where it cannot.
    """
    line = CL.render({}, "empty")
    assert "points=UNKNOWN" in line
    assert "n=UNKNOWN" in line
    assert "settled=NO" in line


def test_render_survives_a_fingerprint_with_no_stale_figures():
    line = CL.render({"collection": "c", "url": "u", "points_count": 5,
                       "n_per_vector": 5}, "partial")
    assert "points=5" in line
    assert "stale=UNKNOWN" in line
    assert "settled=yes" in line


def test_render_of_a_complete_fingerprint_still_carries_every_number():
    fp = CL.build(collection="aw_e8", url="twin", physical="aw_e8",
                   points_count=33_736_546, indexed_vectors_count=69_246_982,
                   named_vectors=2, segments_count=13, segment_sizes=[],
                   status="green", optimizer_status="ok", qdrant_startup=None,
                   optimizer_entries=[])
    line = CL.render(fp, "twin")
    for fragment in ("points=33,736,546", "n=34,623,491", "stale=+886,945",
                     "segs=13", "settled=NO"):
        assert fragment in line, (fragment, line)
