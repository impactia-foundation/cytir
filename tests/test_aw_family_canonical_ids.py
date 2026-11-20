"""Real names by alias (task 2855, RULING 78): schema checks, resolver, scoreboard, stamp site."""
from __future__ import annotations

import copy
import pytest

from cytir.sources import contract as C


def _doc():
    return {"families": [
        {"family_id": "scratch_a_exact", "canonical_id": "a_docs_exact", "legacy_ids": ["scratch_a_exact"]},
        {"family_id": "scratch_b_exact"},
        {"family_id": "gone_stage", "status": "retired"},
    ]}


def test_valid_alias_has_no_problem():
    assert C.alias_problems(_doc()) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d["families"][1].update(canonical_id="a_docs_exact", legacy_ids=["scratch_b_exact"]), "already used"),
    (lambda d: d["families"][0].update(legacy_ids=["other"]), "own family_id"),
    (lambda d: d["families"][0].update(canonical_id="Bad-Name"), "snake case"),
    (lambda d: d["families"][0].update(canonical_id="scratch_x_exact"), "task/scratch-named"),
    (lambda d: d["families"][0].update(canonical_id="scratch_b_exact"), "equals another row"),
    (lambda d: d["families"][1].update(legacy_ids=["x"]), "without canonical_id"),
    (lambda d: d["families"][0].update(legacy_ids=["scratch_a_exact", "scratch_b_exact"]), "another row's family_id"),
])
def test_bad_alias_is_refused(mutate, needle):
    d = copy.deepcopy(_doc())
    mutate(d)
    assert any(needle in p for p in C.alias_problems(d)), C.alias_problems(d)


def test_resolver_round_trip_and_passthrough():
    d = _doc()
    assert C.family_canonical_id("scratch_a_exact", d) == "a_docs_exact"
    assert C.legacy_family_id("a_docs_exact", d) == "scratch_a_exact"
    assert C.family_canonical_id("scratch_b_exact", d) == "scratch_b_exact"
    assert C.legacy_family_id("unknown_id", d) == "unknown_id"


def test_real_contracts_file_names_27_families_and_validates():
    doc = C.load()
    named = [f for f in doc["families"] if f.get("canonical_id")]
    assert len(named) == 27
    assert C.alias_problems(doc) == []
    for f in named:
        assert C.family_canonical_id(f["family_id"]) == f["canonical_id"]
        assert C.legacy_family_id(f["canonical_id"]) == f["family_id"]
    assert not [f for f in named if f["status"] == "retired"]


def test_the_door_stamps_the_legacy_family_id_even_when_a_canonical_id_exists(tmp_path, monkeypatch):
    """(a)/(c): sealed data and today's door key by family_id (ADR-124); the alias never reaches a row."""
    import pathlib
    from cytir.sources import run_family
    fix = pathlib.Path(__file__).resolve().parent / "fixtures" / "statcan_wds"
    monkeypatch.setitem(C._ALIAS_CACHE, str(C.data_path(pathlib.Path(C.DEFAULT_PATH).name)),
                        {"to_canonical": {"statcan_wds": "statcan_wds_real"}, "to_family": {"statcan_wds_real": "statcan_wds"}})
    assert C.family_canonical_id("statcan_wds") == "statcan_wds_real"
    ok = {"bucket": "ok", "search_ok": True, "train_ok": True, "status": "ok"}
    run_family.run_family("statcan_wds", out_dir=tmp_path / "o", today="2026-10-01", log=lambda m: None,
                          consent=lambda h, d: {**ok, "host": h}, options={"raw": str(fix)})
    _, recs = run_family.load_pack(tmp_path / "o" / "pack")
    assert recs and {r["family_id"] for r in recs} == {"statcan_wds"}
