#!/usr/bin/env python3
"""Guards for the gate IDF table sealer (task 629 P4 / ADR-109 R2).

The whole value of this module is its refusals, so that is what is tested: it must
refuse an unsettled collection, refuse a layout that moved mid-measurement, and
stamp `settled: true` ONLY when both assertions actually passed — because
`gate_leg_arms.load_idf_table` trusts that stamp.

No network: the layout reads and the measurement subprocess are monkeypatched, so
these test the decisions rather than the transport.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cytir.index import layout as LAYOUT  # noqa: E402
from cytir.gates import leg_arms as legs  # noqa: E402
from cytir.index import idf_table as S  # noqa: E402  (the seal half of the module)


def fingerprint(points=33_736_546, indexed=None, entries=(), startup="s1"):
    """A settled fingerprint by default: n == points, no optimizer entries."""
    return LAYOUT.build(
        collection="aw_e8", url="http://twin.invalid", physical="aw_e8",
        points_count=points,
        indexed_vectors_count=(points * 2 if indexed is None else indexed),
        named_vectors=2, segments_count=13, segment_sizes=[],
        status="green", optimizer_status="ok", qdrant_startup=startup,
        optimizer_entries=list(entries))


def fake_measure(doc):
    def run(*, qdrant, collection, out, capture_max_idf, extra=None):
        pathlib.Path(out).write_text(json.dumps(doc))
        return {"command": ["fake"], "stdout_tail": ""}
    return run


BASE_DOC = {"idf": {"art": 3.0, "co": 0.5}, "points_count": 33_736_546,
            "n_vocabulary": 2, "n_measured": 2,
            "measured_at": "2026-09-22T12:00:00Z"}


# ------------------------------------------------------------ the settle gate
def test_an_unsettled_collection_is_refused(monkeypatch, tmp_path):
    """n != points_count means stale postings still count toward every term's IDF,
    and a table sealed from that state ships the drift to every self-hoster."""
    stale = fingerprint(indexed=33_736_546 * 2 + 1_775_330)  # +887,665 per vector
    monkeypatch.setattr(LAYOUT, "fingerprint", lambda *a, **k: stale)
    with pytest.raises(S.SealError, match="may not be sealed"):
        S.seal(qdrant="http://twin.invalid", collection="aw_e8",
               out=tmp_path / "t.json")


def test_the_refusal_names_the_reason_from_settled(monkeypatch, tmp_path):
    stale = fingerprint(indexed=33_736_546 * 2 + 1_000)
    monkeypatch.setattr(LAYOUT, "fingerprint", lambda *a, **k: stale)
    with pytest.raises(S.SealError) as excinfo:
        S.seal(qdrant="http://twin.invalid", collection="aw_e8",
               out=tmp_path / "t.json")
    assert "stale postings" in str(excinfo.value)


def test_a_plain_segment_also_blocks_the_seal(monkeypatch, tmp_path):
    """task 606: a sub-threshold segment scores a filtered sparse query
    differently, and `status: green` does not report it."""
    fp = LAYOUT.build(collection="aw_e8", url="u", physical="aw_e8",
                      points_count=100, indexed_vectors_count=200,
                      named_vectors=2, segments_count=2, segment_sizes=[50, 50],
                      status="green", optimizer_status="ok", qdrant_startup="s",
                      optimizer_entries=[])
    monkeypatch.setattr(LAYOUT, "fingerprint", lambda *a, **k: fp)
    with pytest.raises(S.SealError, match="below the indexing threshold"):
        S.seal(qdrant="u", collection="aw_e8", out=tmp_path / "t.json")


# --------------------------------------------------- the layout must not move
def test_a_layout_that_moves_mid_measurement_is_refused(monkeypatch, tmp_path):
    """Measured across a merge is measured against two different collections."""
    before = fingerprint()
    after = fingerprint(entries=[{"name": "merge", "status": "done",
                                 "start_at": "t0", "end_at": "t1",
                                 "segment_ids": [1]}])
    seq = iter([before, after])
    monkeypatch.setattr(LAYOUT, "fingerprint", lambda *a, **k: next(seq))
    monkeypatch.setattr(S, "measure_table", fake_measure(BASE_DOC))
    with pytest.raises(S.SealError, match="layout MOVED"):
        S.seal(qdrant="u", collection="aw_e8", out=tmp_path / "t.json")


def test_a_qdrant_restart_mid_measurement_is_refused(monkeypatch, tmp_path):
    """A restart truncates the optimizer log, so its absence proves nothing —
    `optimizer_ran_between` reports the restart itself."""
    seq = iter([fingerprint(startup="s1"), fingerprint(startup="s2")])
    monkeypatch.setattr(LAYOUT, "fingerprint", lambda *a, **k: next(seq))
    monkeypatch.setattr(S, "measure_table", fake_measure(BASE_DOC))
    with pytest.raises(S.SealError, match="layout MOVED"):
        S.seal(qdrant="u", collection="aw_e8", out=tmp_path / "t.json")


# ------------------------------------------------------------------ the stamp
def test_a_clean_seal_stamps_settled_and_the_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setattr(LAYOUT, "fingerprint", lambda *a, **k: fingerprint())
    monkeypatch.setattr(S, "measure_table", fake_measure(BASE_DOC))
    out = tmp_path / "t.json"
    result = S.seal(qdrant="u", collection="aw_e8", out=out)
    doc = json.loads(out.read_text())
    assert doc["settled"] is True
    assert doc["schema"] == S.SCHEMA
    assert doc["layout_fingerprint"]["points_count"] == 33_736_546
    assert "A8" in doc["settled_predicate"]
    assert result["tokens"] == 2


def test_the_sealed_table_loads_through_the_leg_arms(monkeypatch, tmp_path):
    """The two halves must agree: what this seals is what the arm accepts."""
    monkeypatch.setattr(LAYOUT, "fingerprint", lambda *a, **k: fingerprint())
    monkeypatch.setattr(S, "measure_table", fake_measure(BASE_DOC))
    out = tmp_path / "t.json"
    S.seal(qdrant="u", collection="aw_e8", out=out)
    table = legs.load_idf_table(out)
    assert table["settled"] is True
    assert table["points_count"] == 33_736_546
    assert table["layout_fingerprint"]


def test_an_unsealed_table_is_rejected_by_the_arm_even_though_it_parses(tmp_path):
    """The hole this module closes: `measure_bm25_idf.py` writes no `settled` key,
    so a table that merely OMITS the question must be refused, not accepted."""
    out = tmp_path / "raw.json"
    out.write_text(json.dumps(BASE_DOC))
    with pytest.raises(legs.LegArmError, match="not stamped"):
        legs.load_idf_table(out)


def test_the_gate_ceiling_is_lifted_above_the_serving_one():
    """The serving table truncates at 6.0 with a stored max of 5.9967 — i.e. right
    where the distribution still has mass, which is why it weighs only 45% of the
    vocabulary. A gate table must weigh all of it."""
    assert S.GATE_CAPTURE_MAX_IDF > 6.0


def test_the_sealer_never_writes_the_serving_table(monkeypatch, tmp_path):
    """Asserted on BEHAVIOUR, not on the docstring: the only path it writes is the
    `--out` it was given. A gate artefact and a serving artefact sharing a path are
    one artefact with two meanings, which is how a gate change becomes a silent
    serving change."""
    written = []

    def record(*, qdrant, collection, out, capture_max_idf, extra=None):
        written.append(str(out))
        pathlib.Path(out).write_text(json.dumps(BASE_DOC))
        return {"command": ["fake"], "stdout_tail": ""}

    monkeypatch.setattr(LAYOUT, "fingerprint", lambda *a, **k: fingerprint())
    monkeypatch.setattr(S, "measure_table", record)
    out = tmp_path / "gate-idf.json"
    S.seal(qdrant="u", collection="aw_e8", out=out)
    assert written == [str(out)]
    assert all("cytir/serve" not in path for path in written)
    # and the module's default is not the serving path either
    assert "bm25_idf.json" not in str(getattr(S, "DEFAULT_OUT", ""))


def test_a_failed_measurement_is_a_refusal_not_an_empty_table(monkeypatch,
                                                              tmp_path):
    def boom(**kw):
        raise S.SealError("measure_bm25_idf failed (1)")
    monkeypatch.setattr(LAYOUT, "fingerprint", lambda *a, **k: fingerprint())
    monkeypatch.setattr(S, "measure_table", boom)
    with pytest.raises(S.SealError, match="failed"):
        S.seal(qdrant="u", collection="aw_e8", out=tmp_path / "t.json")
    assert not (tmp_path / "t.json").exists()


# ------------------------------------- the ruled residual allowance (gate only)
def test_the_strict_default_is_zero():
    """Strict is the default and is MANDATORY for a serving-path table; the
    allowance exists only because a settled twin un-settles the moment the daily
    news lane writes."""
    import inspect
    signature = inspect.signature(S.seal)
    assert signature.parameters["max_stale_ratio"].default == 0.0


def test_a_residual_inside_the_allowance_seals_and_is_STAMPED(monkeypatch,
                                                             tmp_path):
    """Tolerated is not the same as ignored: the measured residual is recorded so a
    verdict taken over a table with drift can be placed on that drift."""
    points = 33_736_546
    stale_per_vector = 13_000                     # ~ one news pack's overwrites
    fp = fingerprint(points=points,
                     indexed=(points + stale_per_vector) * 2)
    monkeypatch.setattr(LAYOUT, "fingerprint", lambda *a, **k: fp)
    monkeypatch.setattr(S, "measure_table", fake_measure(BASE_DOC))
    out = tmp_path / "t.json"
    S.seal(qdrant="u", collection="aw_e8", out=out, max_stale_ratio=0.001)
    doc = json.loads(out.read_text())
    assert doc["settled"] is True
    assert doc["stale_ratio_tolerated"] == 0.001
    measured = doc["stale_ratio_measured"]
    assert 0 < measured < 0.001, measured
    assert "ALLOWED" in doc["settled_predicate"]


def test_a_residual_over_the_allowance_still_refuses(monkeypatch, tmp_path):
    points = 33_736_546
    fp = fingerprint(points=points, indexed=(points + 887_665) * 2)   # 2.63%
    monkeypatch.setattr(LAYOUT, "fingerprint", lambda *a, **k: fp)
    with pytest.raises(S.SealError, match="exceed the allowed residual"):
        S.seal(qdrant="u", collection="aw_e8", out=tmp_path / "t.json",
               max_stale_ratio=0.001)


def test_the_allowance_cannot_be_raised_past_the_ruled_ceiling(tmp_path):
    """Raising it is a ruling, not a flag."""
    with pytest.raises(S.SealError, match="ruled gate ceiling"):
        S.seal(qdrant="u", collection="aw_e8", out=tmp_path / "t.json",
               max_stale_ratio=0.05)


def test_a_plain_segment_is_never_covered_by_the_allowance(monkeypatch, tmp_path):
    """A different defect from staleness: a plain segment scores a filtered sparse
    query differently (task 606), so no stale-ratio allowance reaches it."""
    fp = LAYOUT.build(collection="aw_e8", url="u", physical="aw_e8",
                      points_count=100, indexed_vectors_count=200,
                      named_vectors=2, segments_count=2, segment_sizes=[50, 50],
                      status="green", optimizer_status="ok", qdrant_startup="s",
                      optimizer_entries=[])
    monkeypatch.setattr(LAYOUT, "fingerprint", lambda *a, **k: fp)
    with pytest.raises(S.SealError, match="below the indexing threshold"):
        S.seal(qdrant="u", collection="aw_e8", out=tmp_path / "t.json",
               max_stale_ratio=0.001)


def test_an_unreadable_n_refuses_rather_than_sealing_against_the_unknown(
        monkeypatch, tmp_path):
    fp = LAYOUT.build(collection="aw_e8", url="u", physical="aw_e8",
                      points_count=100, indexed_vectors_count=0, named_vectors=0,
                      segments_count=1, segment_sizes=[], status="green",
                      optimizer_status="ok", qdrant_startup="s",
                      optimizer_entries=[])
    monkeypatch.setattr(LAYOUT, "fingerprint", lambda *a, **k: fp)
    with pytest.raises(S.SealError, match="unreadable"):
        S.seal(qdrant="u", collection="aw_e8", out=tmp_path / "t.json",
               max_stale_ratio=0.001)


def test_the_judgement_reads_the_NUMBERS_not_settleds_prose(monkeypatch,
                                                            tmp_path):
    """String-matching a reason to decide whether it is tolerable would break the
    first time someone improved the wording, so the decision comes from the
    fingerprint's own figures. Proven by rewording `settled()` and seeing the
    verdict hold."""
    points = 33_736_546
    fp = fingerprint(points=points, indexed=(points + 13_000) * 2)
    monkeypatch.setattr(LAYOUT, "fingerprint", lambda *a, **k: fp)
    monkeypatch.setattr(LAYOUT, "settled",
                        lambda f: ["totally different wording"])
    monkeypatch.setattr(S, "measure_table", fake_measure(BASE_DOC))
    out = tmp_path / "t.json"
    S.seal(qdrant="u", collection="aw_e8", out=out, max_stale_ratio=0.001)
    doc = json.loads(out.read_text())
    assert doc["settled"] is True
    # and the prose is still RECORDED -- on the PRE-measurement fingerprint, which
    # is where assert_settled returned it -- for the operator reading the receipt
    assert doc["layout_fingerprint_before"]["settled_reasons_at_seal"] == [
        "totally different wording"]
