"""`dila_juri` SourceFamily (task 696): real CASS + JADE decisions from the DILA dump on the build box
(`Freemium_{cass,jade}_global_20250713-140000.tar.gz`), run through `cyt source run`'s own door.

Pins: one record per decision; no year floor (the declared family) vs the task-140 floors (the
legacy CLI); every unit reaches a terminal with a reason; a wrong root and an all-empty run
refuse; the record carries the contract's licence and the canonical legifrance URL.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

FIX = HERE / "fixtures" / "dila_juri"
FAMILY = "dila_juri"
TODAY = "2026-10-03"


@pytest.fixture
def dump(tmp_path):
    root = tmp_path / "dump"
    for sub, prefix in (("cass/global/civile/JURI/TEXT/00", "JURITEXT"),
                        ("jade/global/publie/CETA/TEXT/00", "CETATEXT")):
        d = root / sub
        d.mkdir(parents=True)
        for f in sorted(FIX.glob(f"{prefix}*.xml")):
            shutil.copy(f, d / f.name)
    (root / "cass/README.txt").write_text("not a decision")
    return root


def _run(dump, out, **kw):
    from cytir.sources import run_family
    return run_family.run_family(
        FAMILY, out_dir=out, today=TODAY, consent=lambda h, d: pytest.fail(f"consent for {h}"),
        options={"root": str(dump)}, log=lambda m: None, **kw)


def _records(out):
    return [json.loads(line) for line in
            (out / "pack" / "records.jsonl").read_text(encoding="utf-8").splitlines()]


def test_one_record_per_real_decision_and_no_year_floor(dump, tmp_path):
    manifest = _run(dump, tmp_path / "o")
    rows = _records(tmp_path / "o")
    t = manifest["terminals"]
    assert t["units_in"] == 4 and t["units_admitted"] == 4 and len(rows) == 4
    assert manifest["discover"]["files_by_court"] == {"cass": 2, "jade": 2}
    assert {r["provenance"] for r in rows} == {"cass_juri", "ce_juri"}
    assert {r["domain"] for r in rows} == {"legifrance.gouv.fr"}
    assert {r["license"] for r in rows} == {"Etalab-2.0"}
    assert all(r["url"].startswith("https://www.legifrance.gouv.fr/") for r in rows)
    assert all(r["url"].rsplit("/", 1)[1] in ("JURITEXT000006967306", "JURITEXT000041668731",
                                              "CETATEXT000025402027", "CETATEXT000041735768")
               for r in rows)
    assert all(len(r["text"]) <= 3600 for r in rows)
    assert manifest["route"] == "source_owned_bulk_dump"
    assert manifest["consent"]["verdicts"] == []          # a dump contacts no host


def test_the_declared_family_keeps_what_the_p0_floors_dropped(dump):
    """The 1960s-2000s decision of each court is below the legacy floor and inside the family."""
    from cytir.extract.dila_juri import FLOORS_P0, parse_decision_bytes
    assert (FLOORS_P0["TEXTE_JURI_JUDI"], FLOORS_P0["TEXTE_JURI_ADMIN"]) == (2010, 2024)  # task-140 floors, kept from the deleted wrapper test
    dropped = 0
    for f in sorted(FIX.glob("*.xml")):
        body = f.read_bytes()
        rec, why = parse_decision_bytes(body, None)
        assert rec is not None and why == "kept", (f.name, why)
        legacy, why_p0 = parse_decision_bytes(body, FLOORS_P0)
        if legacy is None:
            assert why_p0.startswith("below_year_floor:")
            dropped += 1
    assert dropped >= 1            # at least one real fixture is older than its court's floor


def test_every_unit_reaches_one_terminal_with_a_reason(dump, tmp_path):
    cass = dump / "cass/global/civile/JURI/TEXT/00"
    (cass / "JURITEXT000000000001.xml").write_bytes(b"<TEXTE_JURI_JUDI><oops")      # not XML
    (cass / "JURITEXT000000000002.xml").write_bytes(b"<AUTRE><META_COMMUN/></AUTRE>")
    (cass / "JURITEXT000000000003.xml").write_bytes(
        b"<TEXTE_JURI_JUDI><META><META_COMMUN><ID>JURITEXT000000000003</ID></META_COMMUN>"
        b"</META></TEXTE_JURI_JUDI>")                                              # no content
    manifest = _run(dump, tmp_path / "o")
    t = manifest["terminals"]
    assert t["units_in"] == 7 and t["units_admitted"] == 4 and t["units_excluded"] == 3
    assert t["units_in"] == t["units_admitted"] + t["units_excluded"] + t["units_quarantined"]
    reasons = t["excluded_reasons"]
    assert any(r.startswith("xml_parse_error") for r in reasons)
    assert any(r.startswith("unknown_root") for r in reasons)
    assert sum(reasons.values()) == 3


def test_a_wrong_root_is_an_error_not_an_empty_source(tmp_path):
    from cytir.sources import run_family
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="wrong root"):
        run_family.run_family(FAMILY, out_dir=tmp_path / "o", today=TODAY,
                              consent=lambda h, d: {}, options={"root": str(empty)},
                              log=lambda m: None)
    with pytest.raises(ValueError, match="needs --opt root"):
        run_family.run_family(FAMILY, out_dir=tmp_path / "o2", today=TODAY,
                              consent=lambda h, d: {}, options={}, log=lambda m: None)


def test_a_run_that_admits_nothing_refuses(tmp_path):
    from cytir.sources import run_family
    root = tmp_path / "d" / "cass"
    root.mkdir(parents=True)
    (root / "JURITEXT000000000009.xml").write_bytes(b"<AUTRE/>")
    with pytest.raises(run_family.RunError, match="0 records admitted"):
        run_family.run_family(FAMILY, out_dir=tmp_path / "o", today=TODAY,
                              consent=lambda h, d: {}, options={"root": str(tmp_path / "d")},
                              log=lambda m: None)
