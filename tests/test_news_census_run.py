"""The news coverage census end to end on a scripted twin (task 676 / 700, BAU shift 5).

`cytir.ops.news_census` is the repo's one census of rows per month per publisher. The previous
shift's table came from it; this test pins the behaviour the backfill checks depend on: the
roster is DERIVED (news queue, news source packs, the HN arms), a publisher is probed in both
`d` and `www.d` spellings, rows are bucketed by the month of `published` (any format the
contract normaliser reads), and the markdown table has one column per month.
"""
from __future__ import annotations

import datetime as dt
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from cytir.ops import news_census as NC  # noqa: E402

TODAY = dt.date(2026, 10, 3)


class FakeTwin:
    """Serves count/facet over a {domain: {published: n}} table, the way Qdrant's exact
    count and facet answer a `domain` match filter (indexed keys only)."""
    collection = "aw_e8"

    def __init__(self, table):
        self.table = table

    def _domains(self, must):
        for cond in must:
            if cond["key"] == "domain":
                m = cond["match"]
                return set(m["any"]) if "any" in m else {m["value"]}
        return None

    def count(self, must):
        doms = self._domains(must)
        return sum(sum(v.values()) for d, v in self.table.items() if doms is None or d in doms)

    def facet(self, key, must, limit):
        doms = self._domains(must)
        agg = {}
        if key == "published":
            for d, v in self.table.items():
                if doms is None or d in doms:
                    for pub, n in v.items():
                        agg[pub] = agg.get(pub, 0) + n
        elif key == "domain":
            days = None
            for cond in must:
                if cond["key"] == "published":
                    days = set(cond["match"]["any"])
            for d, v in self.table.items():
                n = sum(c for pub, c in v.items() if days is None or pub in days)
                if n:
                    agg[d] = n
        return [{"value": k, "count": n} for k, n in sorted(agg.items(), key=lambda kv: -kv[1])][:limit]


TABLE = {
    "swissinfo.ch": {"2026-05-04": 10, "2026-06-10": 20, "2026-07-30": 30, "2026-07-29T10:00:00Z": 5},
    "www.rts.ch": {"2026-06-01": 7},
    "rts.ch": {"2026-09-15": 3, "Wed, 16 Sep 2026 10:00:00 GMT": 2},
    "example-extra.org": {"2026-08-01": 1},
}


def test_the_roster_is_derived_not_hand_listed():
    roster = NC.declared_roster()
    for must_have in ("swissinfo.ch", "rts.ch", "rsi.ch", "srf.ch", "news.ycombinator.com"):
        assert must_have in roster, must_have
    assert all(roster.values()), "every roster domain names the file that declares it"


def test_swissinfo_and_rts_rows_per_month_in_both_host_spellings():
    doc = NC.run(FakeTwin(TABLE), TODAY, 12, extra=["example-extra.org"], top=50)
    by = {d["domain"]: d for d in doc["domains"]}
    sw = by["swissinfo.ch"]
    assert sw["rows"] == 65
    assert sw["by_month"] == {"2026-05": 10, "2026-06": 20, "2026-07": 35}
    rts = by["rts.ch"]
    assert rts["per_variant"] == {"rts.ch": 5, "www.rts.ch": 7}, "both spellings are probed"
    assert rts["rows"] == 12
    assert rts["by_month"] == {"2026-06": 7, "2026-09": 5}, "an RFC-822 date lands in its month"
    assert by["example-extra.org"]["extra"] is True


def test_the_markdown_table_has_a_column_per_month_and_a_row_per_publisher():
    doc = NC.run(FakeTwin(TABLE), TODAY, 12, extra=[], top=50)
    md = NC.to_markdown(doc)
    header = next(line for line in md.splitlines() if line.startswith("| domain"))
    assert "25-10" in header and "26-10" in header
    assert "| swissinfo.ch | 65 |" in md and "| rts.ch | 12 |" in md
