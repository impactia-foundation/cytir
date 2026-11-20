"""Scope classifier + two-half scoring on real hosts from the WebCode gold set (task 1026)."""
import json
from pathlib import Path

from cytir.bench import scope as sc

FIX = json.loads((Path(__file__).parent / "fixtures" / "scope_real_hosts.json").read_text())


def test_real_hosts_classify():
    assert len(FIX["cases"]) >= 10
    for c in FIX["cases"]:
        got = sc.classify_url(c["url"], FIX["served"])
        assert got.klass == c["klass"], (c["url"], got)
        if c["rule_prefix"]:
            assert got.rule.startswith(c["rule_prefix"]), (c["url"], got)
        assert got.clause, "every verdict names its clause"


def test_zero_served_is_not_in_scope_by_itself():
    assert sc.classify_url("https://hexdocs.pm/ecto/", {}).klass == sc.UNDECIDED


def test_www_variants_share_served_count():
    assert sc.served_rows("www.gnu.org", {"gnu.org": 5}) == 5


def test_doi_is_in_scope_by_family():
    assert sc.classify_doi("10.1590/x").klass == sc.IN
    assert sc.doi_prefix("10.1590/S0100") == "10.1590"


def test_half_table_wilson_and_out_share():
    i, o = sc.Scope(sc.IN, "r", "c"), sc.Scope(sc.OUT, "r", "c")
    t = sc.half_table([(i, True), (i, False), (i, False), (i, False), (o, False)])
    assert t["in"]["n"] == 4 and t["in"]["hits"] == 1 and t["in"]["rate"] == 0.25
    lo, hi = t["in"]["ci"]
    assert 0 < lo < 0.25 < hi < 1
    assert t["out_share"]["rate"] == 0.2 and t["all"]["n"] == 5


def test_missed_hosts_sorted_and_out_excluded():
    i, u, o = sc.Scope(sc.IN, "r", "c"), sc.Scope(sc.UNDECIDED, "r", "c"), sc.Scope(sc.OUT, "r", "c")
    rows = [("https://a.org/1", i, False), ("https://a.org/2", i, True), ("https://b.org/1", u, False),
            ("https://b.org/2", u, False), ("https://m.org/1", o, False)]
    got = sc.missed_hosts(rows, {"a.org": 3})
    assert [r["host"] for r in got] == ["b.org", "a.org"]
    assert got[0]["served_rows_on_twin"] == 0 and "admission" in got[0]["suggested_route"]
    assert got[1]["n_queries"] == 2 and got[1]["n_queries_missed"] == 1
