"""`cytir.ops.cc_tranche_queue` (task 701): never-queued roster hosts -> best-first tranche queue files."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
for p in (str(HERE.parent),):
    if p not in sys.path:
        sys.path.insert(0, p)

from cytir.ops import cc_tranche_queue as Q  # noqa: E402

LISTS = HERE.parent / "data" / "spam-host-lists.v1.json"


def _require_host_lists():
    # The public cut withholds this data file (task 2097); skip when it is not a file.
    if not LISTS.is_file():
        pytest.skip("data/spam-host-lists.v1.json is withheld from the public cut")


def _roster(n=20):
    return [{"domain": f"h{i}.example", "class": "edu", "score": 100 - i} for i in range(n)]


def test_split_leaves_out_queued_excluded_and_hijacked_hosts_and_keeps_best_first():
    roster = _roster() + [{"domain": "www.tlb.or.tz", "class": "other", "score": 1}]
    queued = {"h0.example", "h1.example"}
    excluded = {"h2.example"}
    chunks, counts = Q.split(roster, queued, excluded, {"tlb.or.tz"}, parts=3)
    flat = [r["domain"] for c in chunks for r in c]
    assert flat == [f"h{i}.example" for i in range(3, 20)]  # contiguous, best-first, nothing reordered
    assert counts["already_queued"] == 2 and counts["excluded_publisher"] == 1 and counts["hijacked_holdout"] == 1
    assert sum(counts["hosts_per_part"]) == 17 and counts["parts"] == 3


def test_split_refuses_when_nothing_is_left():
    with pytest.raises(SystemExit, match="REFUSED"):
        Q.split(_roster(2), {"h0.example", "h1.example"}, set(), set(), 2)


def test_cli_writes_one_file_per_part_and_a_plan(tmp_path):
    _require_host_lists()
    (tmp_path / "roster.json").write_text(json.dumps(_roster(30)))
    (tmp_path / "q1.json").write_text(json.dumps([{"domain": "h0.example"}, {"domain": "h1.example"}]))
    (tmp_path / "ex.json").write_text(json.dumps(["h2.example"]))
    rc = Q.main(["--roster", str(tmp_path / "roster.json"), "--queued", str(tmp_path / "q1.json"),
                 "--exclude", str(tmp_path / "ex.json"), "--host-lists", str(LISTS),
                 "--out-dir", str(tmp_path / "out"), "--first-number", "7", "--parts", "3"])
    assert rc == 0
    plan = json.loads((tmp_path / "out" / "tranche_queue_plan.json").read_text())
    assert sorted(plan["files"]) == ["7", "8", "9"] and plan["never_queued"] == 27
    first = json.loads((tmp_path / "out" / "tranche7_never_queued.json").read_text())
    assert first[0]["domain"] == "h3.example" and len(first) == 9
