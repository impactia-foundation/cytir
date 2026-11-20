"""The readiness predicate must fail closed, and it must be able to fail.

Every case here is one of the readings that would otherwise be mistaken for
"settled": a plain segment under a green collection (task 336 / task 606), a
segment with unindexed points, telemetry that does not mention the collection,
a shard with no segments, and a details_level too low to carry any of it.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cytir.index import readiness as readiness  # noqa: E402


def seg(num_points=1000, segment_type="indexed", num_indexed=None,
        num_vectors=None, deleted=0, appendable=False):
    return {"info": {
        "segment_type": segment_type,
        "num_points": num_points,
        "num_vectors": num_vectors if num_vectors is not None else num_points * 2,
        "num_indexed_vectors": (num_indexed if num_indexed is not None
                                else num_points),
        "num_deleted_vectors": deleted,
        "is_appendable": appendable}}


def telemetry(segments, collection="aw_e8"):
    return {"collections": {"collections": [
        {"id": collection, "shards": [{"local": {"segments": segments}}]}]}}


def info(status="green", optimizer="ok", queue=0, **extra):
    return {"status": status, "optimizer_status": optimizer,
            "update_queue_length": queue, "points_count": 33723480,
            "segments_count": 13, **extra}


# --------------------------------------------------------------- happy path
def test_all_indexed_and_quiet_is_ready():
    rows = readiness.segment_rows(telemetry([seg(), seg(500)]), "aw_e8")
    assert readiness.readiness_problems(info(), rows) == []


def test_segment_rows_carries_every_field_the_predicate_reads():
    rows = readiness.segment_rows(telemetry([seg(7, appendable=True)]), "aw_e8")
    assert rows == [{"shard": 0, "segment": 0, "segment_type": "indexed",
                     "num_points": 7, "num_vectors": 14,
                     "num_indexed_vectors": 7, "num_deleted_vectors": 0,
                     "is_appendable": True}]


# ------------------------------------------------- the defect it exists for
def test_plain_segment_under_a_green_collection_is_not_ready():
    """Task 336's incident shape: green + optimizer ok + a plain segment."""
    rows = readiness.segment_rows(
        telemetry([seg(), seg(14819, segment_type="plain")]), "aw_e8")
    problems = readiness.readiness_problems(info(), rows)
    assert problems and "non-indexed segment" in problems[0]
    assert "plain/14819pts" in problems[0]


def test_plain_segments_are_listed_not_just_counted():
    rows = readiness.segment_rows(
        telemetry([seg(12, segment_type="plain"), seg()]), "aw_e8")
    assert readiness.plain_segments(rows) == [
        {"shard": 0, "segment": 0, "segment_type": "plain", "num_points": 12,
         "num_vectors": 24, "num_indexed_vectors": 12,
         "num_deleted_vectors": 0, "is_appendable": False}]


def test_segment_with_unindexed_points_is_not_ready():
    rows = readiness.segment_rows(
        telemetry([seg(1000, num_indexed=400)]), "aw_e8")
    problems = readiness.readiness_problems(info(), rows)
    assert any("unindexed points" in problem for problem in problems)
    assert any("400/1000" in problem for problem in problems)


def test_indexed_type_with_all_points_indexed_is_not_flagged():
    rows = readiness.segment_rows(telemetry([seg(4373, num_indexed=4373,
                                                 num_vectors=8746)]), "aw_e8")
    assert readiness.unindexed_segments(rows) == []
    assert readiness.readiness_problems(info(), rows) == []


# --------------------------------------------------- the three quiet signals
@pytest.mark.parametrize("kwargs,needle", [
    ({"status": "yellow"}, "status='yellow'"),
    ({"optimizer": "optimizing"}, "optimizer_status='optimizing'"),
    ({"queue": 4}, "update_queue_length=4"),
])
def test_each_collection_level_signal_is_reported(kwargs, needle):
    rows = readiness.segment_rows(telemetry([seg()]), "aw_e8")
    problems = readiness.readiness_problems(info(**kwargs), rows)
    assert any(needle in problem for problem in problems), problems


def test_absent_update_queue_length_is_a_problem_not_a_pass():
    rows = readiness.segment_rows(telemetry([seg()]), "aw_e8")
    doc = info()
    del doc["update_queue_length"]
    assert readiness.readiness_problems(doc, rows) == [
        "update_queue_length absent"]


def test_rest_shape_nests_the_queue_under_update_queue():
    """`GET /collections/<name>` reports `update_queue.length`, not a flat key.

    Read as a flat key it comes back None and the collection can never be
    called settled -- which is how this was found, on the real twin.
    """
    rows = readiness.segment_rows(telemetry([seg()]), "aw_e8")
    doc = info()
    del doc["update_queue_length"]
    doc["update_queue"] = {"length": 0}
    assert readiness.readiness_problems(doc, rows) == []
    doc["update_queue"] = {"length": 3}
    assert readiness.readiness_problems(doc, rows) == [
        "update_queue_length=3 (want 0)"]


def test_flat_key_still_wins_when_both_shapes_are_present():
    assert readiness.update_queue_length(
        {"update_queue_length": 5, "update_queue": {"length": 0}}) == 5


def test_a_nested_queue_without_a_length_is_absent_not_zero():
    assert readiness.update_queue_length({"update_queue": {}}) is None


def test_every_unsettled_signal_is_reported_together():
    rows = readiness.segment_rows(
        telemetry([seg(9, segment_type="plain", num_indexed=0)]), "aw_e8")
    problems = readiness.readiness_problems(
        info(status="yellow", optimizer="optimizing", queue=2), rows)
    assert len(problems) == 5


# ----------------------------------------------------------- fail closed
def test_collection_absent_from_telemetry_raises():
    with pytest.raises(readiness.ReadinessError, match="not present in telemetry"):
        readiness.segment_rows(telemetry([seg()], collection="other"), "aw_e8")


def test_details_level_too_low_raises():
    with pytest.raises(readiness.ReadinessError, match="no collections list"):
        readiness.segment_rows({"collections": {}}, "aw_e8")


def test_shard_without_segments_raises():
    doc = {"collections": {"collections": [
        {"id": "aw_e8", "shards": [{"local": {"segments": []}}]}]}}
    with pytest.raises(readiness.ReadinessError, match="lists no segments"):
        readiness.segment_rows(doc, "aw_e8")


def test_collection_without_shards_raises():
    doc = {"collections": {"collections": [{"id": "aw_e8", "shards": []}]}}
    with pytest.raises(readiness.ReadinessError, match="lists no shards"):
        readiness.segment_rows(doc, "aw_e8")


def test_segment_missing_a_field_raises():
    broken = seg()
    del broken["info"]["num_indexed_vectors"]
    with pytest.raises(readiness.ReadinessError, match="missing"):
        readiness.segment_rows(telemetry([broken]), "aw_e8")


def test_segment_without_info_raises():
    with pytest.raises(readiness.ReadinessError, match="has no info"):
        readiness.segment_rows(telemetry([{}]), "aw_e8")


def test_remote_shard_is_skipped_but_a_collection_of_only_remotes_raises():
    doc = {"collections": {"collections": [
        {"id": "aw_e8", "shards": [{"remote": {}}]}]}}
    with pytest.raises(readiness.ReadinessError, match="no local segments"):
        readiness.segment_rows(doc, "aw_e8")


def test_unreachable_endpoint_raises_rather_than_passing():
    with pytest.raises(readiness.ReadinessError, match="cannot read"):
        readiness.get_json("http://127.0.0.1:1/collections/aw_e8", timeout=1)


# -------------------------------------------------------------- exit codes
def test_main_exits_nonzero_when_not_ready(monkeypatch, tmp_path):
    rows_doc = telemetry([seg(14819, segment_type="plain")])
    monkeypatch.setattr(readiness, "fetch_collection",
                        lambda *a, **k: info())
    monkeypatch.setattr(readiness, "fetch_telemetry", lambda *a, **k: rows_doc)
    out = tmp_path / "timeline.json"
    code = readiness.main(["--url", "http://x", "--collection", "aw_e8",
                           "--out", str(out)])
    assert code == 1
    written = json.loads(out.read_text())
    assert written["timeline"][-1]["ready"] is False


def test_main_exits_two_when_readiness_cannot_be_determined(monkeypatch):
    def boom(*a, **k):
        raise readiness.ReadinessError("telemetry unreachable")
    monkeypatch.setattr(readiness, "fetch_collection", boom)
    assert readiness.main(["--url", "http://x", "--collection", "aw_e8"]) == 2


def test_main_exits_zero_only_when_settled(monkeypatch):
    monkeypatch.setattr(readiness, "fetch_collection", lambda *a, **k: info())
    monkeypatch.setattr(readiness, "fetch_telemetry",
                        lambda *a, **k: telemetry([seg()]))
    assert readiness.main(["--url", "http://x", "--collection", "aw_e8"]) == 0


def test_wait_ready_returns_the_whole_timeline_and_stops_at_ready(monkeypatch):
    states = [telemetry([seg(9, segment_type="plain")]),
              telemetry([seg(9, segment_type="plain")]),
              telemetry([seg(9)])]
    monkeypatch.setattr(readiness, "fetch_collection", lambda *a, **k: info())
    monkeypatch.setattr(readiness, "fetch_telemetry",
                        lambda *a, **k: states.pop(0))
    monkeypatch.setattr(readiness.time, "sleep", lambda _: None)
    timeline = readiness.wait_ready("http://x", "aw_e8", timeout=60,
                                    interval=0)
    assert [row["ready"] for row in timeline] == [False, False, True]


def test_wait_ready_on_timeout_reports_unready_rather_than_ready(monkeypatch):
    monkeypatch.setattr(readiness, "fetch_collection", lambda *a, **k: info())
    monkeypatch.setattr(readiness, "fetch_telemetry",
                        lambda *a, **k: telemetry([seg(9, segment_type="plain")]))
    monkeypatch.setattr(readiness.time, "sleep", lambda _: None)
    timeline = readiness.wait_ready("http://x", "aw_e8", timeout=-1, interval=0)
    assert len(timeline) == 1 and timeline[-1]["ready"] is False
