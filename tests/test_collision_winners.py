#!/usr/bin/env python3
"""Tests for the cross-set collision adjudication (task 513).

The defect these guard against: an id staged by two sets is upserted twice and
the survivor is whichever row lands last -- i.e. `sorted(glob("*.receipt.json"))`,
an accident of filenames. That is invisible whenever it is wrong, which is the
worst property a build step can have.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from cytir.index import assemble_inputs as AI  # noqa: E402

PRECEDENCE = REPO / "data" / "assembly" / "adr101-class-precedence.v1.json"
MANIFESTS = REPO / "data" / "assembly" / "claim-manifests.v1.json"


# --------------------------------------------------------------------------
# the winner map reader
# --------------------------------------------------------------------------

from tests.helpers.repo_files import repo_file_or_skip

repo_file_or_skip("data/assembly/claim-manifests.v1.json")  # the public cut withholds this file; skip there (task 2382)


def _write_winners(tmp_path, rows, gz=True):
    p = tmp_path / ("w.jsonl.gz" if gz else "w.jsonl")
    if gz:
        with gzip.open(p, "wt") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    else:
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_load_collision_winners_reads_gz_and_plain(tmp_path):
    rows = [{"point_id": "a", "winner_set_id": "setA"},
            {"point_id": "b", "winner_set_id": "setB"}]
    assert AI.load_collision_winners(_write_winners(tmp_path, rows)) == \
        {"a": "setA", "b": "setB"}
    assert AI.load_collision_winners(_write_winners(tmp_path, rows, gz=False)) == \
        {"a": "setA", "b": "setB"}


def test_load_collision_winners_refuses_a_row_with_no_winner(tmp_path):
    p = _write_winners(tmp_path, [{"point_id": "a"}])
    with pytest.raises(SystemExit):
        AI.load_collision_winners(p)


def test_load_collision_winners_refuses_two_winners_for_one_id(tmp_path):
    p = _write_winners(tmp_path, [{"point_id": "a", "winner_set_id": "x"},
                                  {"point_id": "a", "winner_set_id": "y"}])
    with pytest.raises(SystemExit):
        AI.load_collision_winners(p)


def test_load_collision_winners_tolerates_a_repeated_identical_row(tmp_path):
    p = _write_winners(tmp_path, [{"point_id": "a", "winner_set_id": "x"},
                                  {"point_id": "a", "winner_set_id": "x"}])
    assert AI.load_collision_winners(p) == {"a": "x"}


# --------------------------------------------------------------------------
# the precedence table itself
# --------------------------------------------------------------------------

def test_precedence_table_ranks_every_class_the_manifests_use():
    doc = json.loads(PRECEDENCE.read_text())
    ranked = {c for e in doc["ranks"] for c in e["classes"]}
    used = {s.get("class") for s in json.loads(MANIFESTS.read_text())["sets"]}
    used.discard(None)
    missing = sorted(used - ranked)
    assert not missing, (
        "these contributing-set classes have no rank, so a collision between "
        f"them would fall through to load order: {missing}")


def test_precedence_table_assigns_each_class_exactly_one_rank():
    doc = json.loads(PRECEDENCE.read_text())
    seen = {}
    for e in doc["ranks"]:
        for c in e["classes"]:
            assert c not in seen, f"{c} is ranked twice"
            seen[c] = e["rank"]


def test_rechunk_class_ranks_below_every_original_bytes_epoch4_class():
    """ADR-101 R1.1 / ADR-101b R2 / ADR-101c R2, as a class rule."""
    doc = json.loads(PRECEDENCE.read_text())
    rank = {c: e["rank"] for e in doc["ranks"] for c in e["classes"]}
    assert rank["rechunk_epoch4_from_l1b"] > rank["crawl30_capture_epoch4_chunk"]
    assert rank["rechunk_epoch4_from_l1b"] > rank["f2_l0_html_epoch4_chunk_1024_embed"]
    assert rank["rechunk_epoch4_from_l1b"] > rank["cc_materializer_epoch4_chunk_1024_embed"]
    assert rank["rechunk_epoch4_from_l1b"] > rank["direct_capture_epoch4_chunk"]
    # ADR-101b Ruling 1: the card set outranks every replacement
    assert rank["reembed_epoch3_in_place"] < rank["crawl30_capture_epoch4_chunk"]
    # task 456 s6: the Track-1A B2 residual re-chunks task 427's own L1-b output
    assert rank["narrow_lane_b2_track1a_retained_residual"] == rank["rechunk_epoch4_from_l1b"]
