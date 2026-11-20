"""Tests for cytir.ops.cc_tranche_roster (task 719): the roster maker of the CC outlink tranches.

Fixture rows = the REAL rows of tranche 13 kept by `tests/fixtures/cc_tranche_roster_fix` (46 rows: excluded
publishers, duplicate `www\\d*.` groups, hosts of earlier queues, 3 rows of each class/basis cell).
"""
import json
from pathlib import Path

import pytest

from cytir.ops import cc_tranche_queue as cq
from cytir.ops import spam_rows
from cytir.ops import cc_tranche_roster as rm
from cytir.ops import cc_tranche_roster_fix as fx

FIX = Path(__file__).parent / "fixtures" / "cc_tranche_roster_fix"


@pytest.fixture
def HOSTS(tmp_path):          # the cut does not ship the sealed list: a small file of the same schema
    f = tmp_path / "host-lists.json"
    f.write_text(json.dumps({"schema": spam_rows.HOST_LISTS_SCHEMA, "hold_out": [{"domain": "held.example.gov", "evidence": "t"}],
                             "exceptions": []}))
    return f


@pytest.fixture
def rows():
    out = json.loads((FIX / "tranche13_rows.json").read_text())
    for r in out:
        r.pop("max_pages", None)          # a ring file carries no depth
    return out


def _make(rows, held=frozenset()):
    return rm.make_roster(rows, queued=fx.queued_hosts([FIX / "earlier_queue.json"]),
                          excluded=fx.excluded_regs([FIX / "excluded_publishers.json"]), held=set(held))


def test_depth_120_for_self_signal_only_and_none_for_endorsed(rows):
    out, _ = _make(rows)
    assert {r["class_basis"] for r in out} == {"endorsed", "self-signal"}
    for r in out:
        if r["class_basis"] == "self-signal":
            assert r["max_pages"] == 120
        else:
            assert "max_pages" not in r


def test_drops_match_the_tranche_13_fix_and_are_counted_by_reason(rows):
    out, rep = _make(rows)
    _, fixrep = fx.fix([dict(r, max_pages=120) for r in rows], fx.queued_hosts([FIX / "earlier_queue.json"]),
                       fx.excluded_regs([FIX / "excluded_publishers.json"]))
    assert rep["dropped_excluded_publisher"] == len(fixrep["dropped_excluded_publisher"]) == 5
    assert rep["dropped_in_earlier_queue"] == fixrep["dropped_in_earlier_queue"] > 0
    assert rep["dropped_duplicate_fold"] == fixrep["dropped_duplicate_fold"] > 0
    assert len(out) == len(rows) - 5 - rep["dropped_in_earlier_queue"] - rep["dropped_duplicate_fold"]


def test_no_www_fold_duplicate_and_no_queued_host_survives(rows):
    out, _ = _make(rows)
    folded = [fx.fold(r["domain"]) for r in out]
    assert len(set(folded)) == len(folded)
    assert not set(folded) & fx.queued_hosts([FIX / "earlier_queue.json"])


def test_order_is_n_src_desc_then_host_name(rows):
    out, _ = _make(rows)
    assert out == sorted(out, key=lambda r: (-r["n_src"], r["domain"]))


def test_prescreen_drops_other_classes_held_hosts_spam_names_and_link_farms(rows):
    base = rows[0]
    bad = [dict(base, domain="a-casino.example.edu"),
           dict(base, domain="held.example.gov"),
           dict(base, domain="farm.example.gov", n_src=4, n_src_spammy=2),
           dict(base, domain="shop.example.com", **{"class": "news"}),
           dict(base, domain="fine.example.gov", n_src=4, n_src_spammy=1)]
    keep, drops = rm.prescreen(bad, {"held.example.gov"})
    assert [r["domain"] for r in keep] == ["fine.example.gov"]
    assert drops == {"hostname_spam_lexicon": 1, "held_host": 1, "link_farm": 1, "class_not_admitted": 1}


def test_held_host_is_dropped_by_its_www_fold(rows):
    host = rows[0]["domain"]
    keep, _ = rm.prescreen([dict(rows[0], domain="www3." + fx.fold(host))], {fx.fold(host)})
    assert keep == []


def test_nothing_left_is_refused(rows):
    with pytest.raises(SystemExit, match="REFUSED"):
        rm.make_roster([dict(rows[0], **{"class": "news"})], queued=set(), excluded=set(), held=set())


def test_split_share_takes_the_best_endorsed_first_and_loses_no_host(rows):
    out, _ = _make(rows)
    first, rest = rm.split_share(out, 0.5)
    assert len(first) == -(-len(out) // 2) and len(first) + len(rest) == len(out)
    assert min(r["n_src"] for r in first) >= max(r["n_src"] for r in rest)
    assert {r["domain"] for r in first}.isdisjoint(r["domain"] for r in rest)
    key = lambda r: (-r["n_src"], r["domain"])  # noqa: E731
    assert first == sorted(first, key=key) and key(first[-1]) < key(rest[0])


def test_cli_writes_both_halves_and_refuses_to_overwrite(rows, tmp_path, HOSTS):
    cand = tmp_path / "ring.json"
    cand.write_text(json.dumps(rows))
    argv = ["make", "--candidates", str(cand), "--queued", str(FIX / "earlier_queue.json"),
            "--exclude", str(FIX / "excluded_publishers.json"), "--host-lists", str(HOSTS),
            "--out", str(tmp_path / "t14.json"), "--keep-share", "0.5", "--rest-out", str(tmp_path / "t15.json"),
            "--report", str(tmp_path / "rep.json")]
    assert rm.main(argv) == 0
    a, b = json.loads((tmp_path / "t14.json").read_text()), json.loads((tmp_path / "t15.json").read_text())
    assert a and b and json.loads((tmp_path / "rep.json").read_text())["roster"]["rows"] == len(a) + len(b)
    with pytest.raises(SystemExit, match="exists"):
        rm.main(argv)


def test_cli_needs_rest_out_for_a_split(rows, tmp_path, HOSTS):
    cand = tmp_path / "ring.json"
    cand.write_text(json.dumps(rows))
    with pytest.raises(SystemExit, match="rest-out"):
        rm.main(["make", "--candidates", str(cand), "--queued", str(FIX / "earlier_queue.json"),
                 "--exclude", str(FIX / "excluded_publishers.json"), "--host-lists", str(HOSTS),
                 "--out", str(tmp_path / "x.json"), "--keep-share", "0.5"])


def test_queue_maker_folds_www_digits_onto_the_apex():
    roster = [{"domain": "www2.example.ch", "class": "gov"}, {"domain": "example.ch", "class": "gov"},
              {"domain": "www150.old.ch", "class": "gov"}, {"domain": "www12.bad.ch", "class": "gov"},
              {"domain": "new.ch", "class": "gov"}]
    chunks, c = cq.split(roster, queued={"old.ch"}, excluded=set(), held_out={"bad.ch"}, parts=1)
    assert [r["domain"] for r in chunks[0]] == ["www2.example.ch", "new.ch"]
    assert (c["already_queued"], c["hijacked_holdout"]) == (1, 1)


# ---- the declared allow-list of classes (task 1065, ruling 59; `health` closes the task 1057 gap) ---- #

CLS = Path(__file__).parent / "fixtures" / "cc_tranche_roster_classes"


def _make_classes(extra_queued=frozenset()):
    rows = json.loads((CLS / "candidates.json").read_text())
    return rm.make_roster(rows, queued=set(extra_queued), excluded=set(), held=set())


def test_class_allow_list_declares_health_and_corporate_besides_the_old_three():
    assert set(rm.CLASS_DEPTH) == {"edu", "gov", "ref", "health", "corporate"}
    assert rm.ADMITTED_CLASSES == tuple(rm.CLASS_DEPTH)


def test_health_and_corporate_pass_the_prescreen_and_other_classes_are_dropped_and_counted():
    out, rep = _make_classes()
    assert {r["class"] for r in out} == {"corporate", "health", "edu"}
    assert rep["dropped_prescreen"] == {"class_not_admitted": 2}      # `news` and `unlisted`


def test_depth_per_class_corporate_always_120_health_like_edu():
    out, _ = _make_classes()
    by = {r["domain"]: r.get("max_pages") for r in out}
    assert by["corp-a.example.com"] == by["corp-b.example.ch"] == by["corp-endorsed.example.com"] == 120
    assert by["health-self.example.org"] == 120
    assert by["health-endorsed.example.org"] is None and by["uni.example.edu"] is None


def test_a_class_added_to_the_table_is_admitted_without_code():
    rows = [{"domain": "x.example.com", "reg": "x.example.com", "class": "newclass", "class_basis": "self-signal", "n_src": 1}]
    with pytest.raises(SystemExit):
        rm.make_roster(rows, queued=set(), excluded=set(), held=set())
    rm.CLASS_DEPTH["newclass"] = {"*": 30}
    try:
        out, _ = rm.make_roster(rows, queued=set(), excluded=set(), held=set())
    finally:
        del rm.CLASS_DEPTH["newclass"]
    assert out[0]["max_pages"] == 30
