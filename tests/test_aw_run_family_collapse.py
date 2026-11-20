"""Task 759: the stage door collapses identical-id identical-text rows INSIDE one unit (counted)
and quarantines only a real id collision (same id, different text, or the same id across units)."""
from __future__ import annotations

import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cytir.sources import registry, run_family  # noqa: E402
from cytir.sources.family import CHUNK_NATIVE, Captured, Doc, Unit  # noqa: E402
import test_aw_source_family_protocol as P  # noqa: E402


class Rows:
    """A unit's body is `id:text` lines separated by `|`; each line is one chunk row."""

    family_id = "toy_collapse_for_tests"
    route = "documented_source_api"
    direct_crawl = True
    chunk_policy_id = CHUNK_NATIVE

    def discover(self, ctx):
        return [Unit(key=f"u{i}", url=f"https://toy.example/{i}", meta={"body": b})
                for i, b in enumerate(ctx.options["bodies"])]

    def acquire(self, unit, ctx):
        return [Captured(unit, unit.meta["body"].encode("utf-8"), {"status": 200})]

    def extract(self, cap, ctx):
        return [Doc(url=cap.unit.url, text=cap.body.decode("utf-8"))]

    def chunk(self, doc, ctx):
        from cytir.core.ids import record_point_id
        out = []
        for line in doc.text.split("|"):
            pid, text = line.split(":", 1)
            out.append({"url": doc.url, "text": text,
                        "point_id": record_point_id({"point_id": pid})})
        return out


@pytest.fixture
def rows_family(monkeypatch, tmp_path):
    monkeypatch.setitem(registry._FACTORIES, Rows.family_id, Rows)

    def mutate(row):
        row["chunk_policy"].update(policy_id="unknown", sha256="unknown", epoch="unknown",
                                   policy_family=None)
        row["status"] = "backfilled"
        from cytir.sources import contract
        row["unknowns"] = sorted(contract.actual_unknowns(row))
    return P._contracts(tmp_path, Rows.family_id, mutate=mutate)


def _run(tmp_path, contracts, bodies, name="out"):
    return run_family.run_family(
        Rows.family_id, out_dir=tmp_path / name, today="2026-10-08", consent=P._consent_ok,
        options={"bodies": bodies}, contracts_path=contracts, log=lambda m: None)


def test_a_repeated_chunk_inside_a_document_is_kept_once_and_counted(tmp_path, rows_family):
    m = _run(tmp_path, rows_family, ["a:alpha|b:beta|a:alpha|b:beta|c:gamma", "d:delta"])
    t = m["terminals"]
    assert t["units_admitted"] == 2 and t["units_quarantined"] == 0
    assert m["records"]["count"] == 4 == t["records"]
    assert m["collapsed_identical_rows"] == 2
    assert sorted((d["unit"], d["dropped"]) for d in m["collapsed_detail"]) == [("u0", 1), ("u0", 1)]
    rows = run_family.load_pack(tmp_path / "out" / "pack")[1]
    assert [r["text"] for r in rows] == ["alpha", "beta", "gamma", "delta"]   # the first stays


def test_the_same_id_with_different_text_still_quarantines_the_unit(tmp_path, rows_family):
    m = _run(tmp_path, rows_family, ["a:alpha|a:ALPHA", "b:beta"])
    t = m["terminals"]
    assert t["quarantined_reasons"] == {"duplicate_point_id": 1}
    assert t["units_admitted"] == 1 and m["collapsed_identical_rows"] == 0


def test_a_collision_across_units_still_quarantines_even_when_the_text_is_equal(tmp_path, rows_family):
    m = _run(tmp_path, rows_family, ["a:alpha", "a:alpha"])
    assert m["terminals"]["quarantined_reasons"] == {"duplicate_point_id": 1}
    assert m["collapsed_identical_rows"] == 0


def test_a_month_without_duplicates_has_a_zero_count_and_the_same_rows(tmp_path, rows_family):
    m = _run(tmp_path, rows_family, ["a:alpha|b:beta"])
    assert m["collapsed_identical_rows"] == 0 and "collapsed_detail" not in m


def test_collapse_helper_keeps_the_first_and_leaves_a_real_collision_in_place():
    rows = [{"point_id": "1", "text": "x"}, {"point_id": "1", "text": "x"},
            {"point_id": "1", "text": "y"}, {"point_id": "2", "text": "x"}]
    out, dropped = run_family.collapse_identical_rows(rows)
    assert out == [rows[0], rows[2], rows[3]]       # the different-text row stays, to quarantine
    assert dropped == [{"point_id": "1", "url": None, "dropped": 1}]
