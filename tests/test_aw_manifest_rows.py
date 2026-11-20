"""`manifest.rows` has one meaning per name (hygiene 2026-10-04): `rows_staged` and `rows_to_write_max`
are the two numbers, `rows` is the documented legacy alias, and a pack composed before the new
names exist is read exactly as it always was."""
from __future__ import annotations

import pathlib

import pytest

from cytir.index import manifest_rows as MR
from cytir.index import promote_queue as Q


def test_a_legacy_pack_answers_both_names_with_its_rows():
    legacy = {"rows": 100}
    assert MR.rows_to_write_max(legacy) == 100 and MR.rows_staged(legacy) == 100


def test_a_legacy_served_pack_knows_its_stage_count_from_rows_before_exclusion():
    served = {"rows": 1, "rows_before_exclusion": 2}
    assert MR.rows_to_write_max(served) == 1 and MR.rows_staged(served) == 2


def test_a_new_pack_answers_each_name_with_its_own_number_whatever_the_alias_says():
    new = {"rows": 900, "rows_staged": 900, "rows_to_write_max": 40}
    assert MR.rows_to_write_max(new) == 40 and MR.rows_staged(new) == 900


def test_absent_or_junk_counts_are_none_never_a_guess():
    assert MR.rows_to_write_max({}) is None and MR.rows_staged({}) is None
    assert MR.rows_to_write_max({"rows": "12", "rows_to_write_max": True}) is None


class _Locks:
    def acquire(self, *a, **k):
        pass

    def release(self, *a, **k):
        pass


@pytest.fixture()
def single(monkeypatch, tmp_path):
    """`single_apply` with its collaborators stubbed: only the written-vs-manifest check is real."""
    state = {"written": 0}
    monkeypatch.setattr(Q, "_load_run_batch", lambda: _Locks())
    monkeypatch.setattr(Q, "assert_single_target_not_live", lambda s: None)
    monkeypatch.setattr(Q.LAYOUT, "fingerprint",
                        lambda url, coll: {"points_count": state.get("before", 0)})
    monkeypatch.setattr(Q, "_single_persisted_check", lambda: None)

    def fake_apply(*a, **k):
        w = state["written"]
        return ({"written": w, "layout_after": {"points_count": w}, "rows": {"projected": w}},
                tmp_path / "ids.json")
    monkeypatch.setattr(Q, "apply_side", fake_apply)

    def run(manifest, written):
        state["written"] = written
        return Q.single_apply(pack_id="p", manifest=manifest,
                              single=Q.Single(url="http://127.0.0.1:1", collection="c"),
                              task="t", batch=1, artifact_root=pathlib.Path(tmp_path / "art"))
    return run


def test_single_apply_pins_written_to_rows_to_write_max_only_when_nothing_was_dropped(single):
    # no skip, exclusion or drop in the receipt: part (c) of check 5 binds, by the new name.
    # (A pack composed with an exclusion file reports those ids as `excluded_out_of_scope_language`,
    # which skips part (c); that case is pinned beside the real projection in the queue tests.)
    out = single({"rows": 2, "rows_staged": 2, "rows_to_write_max": 2}, written=2)
    assert out["check5"]["manifest_rows"] == 2
    with pytest.raises(Q.QueueError, match="declares 2 rows"):
        single({"rows": 2, "rows_staged": 2, "rows_to_write_max": 2}, written=1)


def test_single_apply_reads_a_legacy_pack_by_rows_exactly_as_before(single):
    assert single({"rows": 3}, written=3)["check5"]["manifest_rows"] == 3
    with pytest.raises(Q.QueueError, match="declares 3 rows"):
        single({"rows": 3}, written=2)
