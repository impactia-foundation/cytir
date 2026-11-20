"""The assembly binder's input loaders (task 2092, slice A1), cases copied from their old tests."""
from __future__ import annotations

import gzip
import json

import pytest

from cytir.index import assemble_inputs as AI


def _write_gz(tmp_path, rows, name="w.jsonl.gz"):
    p = tmp_path / name
    with gzip.open(p, "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def test_load_collision_winners_reads_gz_and_plain(tmp_path):
    rows = [{"point_id": "a", "winner_set_id": "setA"},
            {"point_id": "b", "winner_set_id": "setB"}]
    plain = tmp_path / "w.jsonl"
    plain.write_text("".join(json.dumps(r) + "\n" for r in rows))
    want = {"a": "setA", "b": "setB"}
    assert AI.load_collision_winners(_write_gz(tmp_path, rows)) == want
    assert AI.load_collision_winners(plain) == want


def test_load_collision_winners_refuses_two_winners_for_one_id(tmp_path):
    p = _write_gz(tmp_path, [{"point_id": "a", "winner_set_id": "x"},
                             {"point_id": "a", "winner_set_id": "y"}])
    with pytest.raises(SystemExit):
        AI.load_collision_winners(p)


def test_row_suppression_is_keyed_by_scope_not_by_id(tmp_path):
    p = _write_gz(tmp_path, [
        {"point_id": "a", "scope": "setA", "set_id": "setA"},
        {"point_id": "a", "scope": "setB", "set_id": "setB"},
        {"point_id": "b", "scope": "setA", "set_id": "setA"},
    ], "s.jsonl.gz")
    assert AI.load_row_suppression(p) == {"setA": {"a", "b"}, "setB": {"a"}}


def test_row_suppression_refuses_an_unscoped_entry(tmp_path):
    p = _write_gz(tmp_path, [{"point_id": "a", "set_id": "setA"}], "s.jsonl.gz")
    with pytest.raises(SystemExit) as excinfo:
        AI.load_row_suppression(p)
    assert "scope" in str(excinfo.value)


def test_row_suppression_absent_file_is_no_suppression():
    assert AI.load_row_suppression(None) == {}


def test_exclusions_refuse_the_wrong_schema_and_absent_means_none(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"schema": "something.else", "evidence": []}))
    with pytest.raises(SystemExit):
        AI.load_pack_route_exclusions(p)
    assert AI.load_pack_route_exclusions(tmp_path / "nope.json") == {}


def test_an_absent_override_file_declares_nothing_rather_than_raising(tmp_path, monkeypatch):
    monkeypatch.setenv("AW_REPO_ROOT", str(tmp_path))
    assert AI.load_pack_stage_overrides() == {}
