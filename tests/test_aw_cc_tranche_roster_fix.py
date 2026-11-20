"""Tests for cytir.ops.cc_tranche_roster_fix (task 716): the fix of the tranche 13 roster.

The fixture rows are a copy of REAL rows of `tranche13_never_queued.json` (7,516 rows, all `max_pages: 120`):
the 5 rows under an excluded publisher, the rows of the duplicate `www\\d*.` groups, rows that fold onto a
host of the tranche 2-12 queues, and 3 rows of each class/basis cell.
"""
import json
import shutil
from pathlib import Path

import pytest

from cytir.ops import cc_tranche_roster_fix as fx

FIX = Path(__file__).parent / "fixtures" / "cc_tranche_roster_fix"


@pytest.fixture
def rows():
    return json.loads((FIX / "tranche13_rows.json").read_text())


def _run(rows):
    queued = fx.queued_hosts([FIX / "earlier_queue.json"])
    excl = fx.excluded_regs([FIX / "excluded_publishers.json"])
    return fx.fix(rows, queued, excl)


def test_the_real_rows_all_start_at_120(rows):
    assert {r["max_pages"] for r in rows} == {120}


def test_endorsed_rows_lose_their_depth_and_self_signal_rows_keep_it(rows):
    out, _ = _run(rows)
    for r in out:
        if r["class_basis"] == "endorsed":
            assert "max_pages" not in r
        else:
            assert r["max_pages"] == 120
    assert {r["class_basis"] for r in out} == {"endorsed", "self-signal"}


def test_excluded_publisher_is_dropped_by_registrable_domain_not_exact_host(rows):
    exact = {e["domain"] for e in json.loads((FIX / "excluded_publishers.json").read_text())}
    out, rep = _run(rows)
    assert "ieeetv.ieee.org" not in exact  # the exact host is NOT on the list; its registrable domain is
    assert "ieeetv.ieee.org" in rep["dropped_excluded_publisher"]
    assert len(rep["dropped_excluded_publisher"]) == 5
    assert not [r for r in out if r["reg"] in ("wiley.com", "ieee.org")]


def test_rows_folding_onto_an_earlier_queue_are_dropped(rows):
    out, rep = _run(rows)
    queued = fx.queued_hosts([FIX / "earlier_queue.json"])
    assert rep["dropped_in_earlier_queue"] > 0
    assert not [r for r in out if fx.fold(r["domain"]) in queued]


def test_one_row_per_folded_domain_with_the_most_captures():
    mk = lambda d, cap, score=1.0: {"domain": d, "reg": "x.org", "class": "gov", "class_basis": "self-signal",
                                    "cap_n200": cap, "score": score, "max_pages": 120}
    out, rep = fx.fix([mk("www2.x.org", 3), mk("x.org", 9), mk("www.x.org", 9, 5.0), mk("y.org", 0)], set(), set())
    assert [r["domain"] for r in out] == ["www.x.org", "y.org"]  # tie on captures -> the better score
    assert rep["duplicate_groups"] == 1 and rep["dropped_duplicate_fold"] == 2


def test_fold_handles_numbered_www_hosts():
    assert fx.fold("WWW150.statcan.gc.ca") == "statcan.gc.ca"
    assert fx.fold("www2.census.gov") == "census.gov"
    assert fx.fold("wwwfoo.org") == "wwwfoo.org"


def test_report_counts_add_up_and_the_result_is_unique(rows):
    out, rep = _run(rows)
    dropped = len(rep["dropped_excluded_publisher"]) + rep["dropped_in_earlier_queue"] + rep["dropped_duplicate_fold"]
    assert rep["before"]["rows"] - dropped == rep["after"]["rows"] == len(out)
    folded = [fx.fold(r["domain"]) for r in out]
    assert len(folded) == len(set(folded))
    assert rep["after"]["by_depth"].get("120", 0) == sum(1 for r in out if r["class_basis"] == "self-signal")


def test_unknown_class_basis_is_refused():
    with pytest.raises(SystemExit):
        fx.fix([{"domain": "a.org", "reg": "a.org", "class": "gov", "class_basis": "guessed"}], set(), set())


def test_cli_overwrites_in_place_keeps_the_backup_and_refuses_a_second_run(tmp_path, capsys):
    roster = tmp_path / "tranche13_never_queued.json"
    shutil.copy(FIX / "tranche13_rows.json", roster)
    args = ["--roster", str(roster), "--queued", str(FIX / "earlier_queue.json"),
            "--exclude", str(FIX / "excluded_publishers.json"), "--backup-suffix", ".pre-fix"]
    assert fx.main(args) == 0
    rep = json.loads(capsys.readouterr().out)
    backup = tmp_path / "tranche13_never_queued.json.pre-fix"
    assert backup.read_bytes() == (FIX / "tranche13_rows.json").read_bytes()
    assert rep["sha256_before"] == fx.sha256(backup) and rep["sha256_after"] == fx.sha256(roster)
    assert len(json.loads(roster.read_text())) == rep["after"]["rows"]
    with pytest.raises(SystemExit):
        fx.main(args)
