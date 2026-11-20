"""`named_slice:swisstopo-v2` on the family Protocol (task 1413).

No network. The fixture is a hand-written five-row SwissNAMES3D point CSV (three admitted
features in de/fr/it, one without a name, one without coordinates), zipped in the test. Pins:
registered and conformant; discover/acquire/extract on the sealed zip; a `plan` dry run through the
framework; consent and licence read from the row; and PARITY with the pre-move driver, whose output
on the fixture was digested before the move (driver blob 308c06b0e533821040f6cd9fa79728f29e9a68c5,
the file on origin/main at the task's start).
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import zipfile
from dataclasses import asdict

import pytest

from cytir.extract import swisstopo as S
from cytir.sources import contract as contract_mod
from cytir.sources import registry
from cytir.sources.family import RunContext, SourceFamily, UnitExcluded
from cytir.sources.families.swisstopo import SwisstopoFamily
from cytir.sources.run_family import run_family

FIXTURE = pathlib.Path(__file__).resolve().parent / "data" / "swisstopo" / "swissnames3d_sample_pkt.csv"
#: sha256 of json.dumps([{"passage", "record"}...], sort_keys=True, ensure_ascii=False) from the
#: PRE-MOVE driver over FIXTURE (member swissNAMES3D_PKT.csv, item swissnames3d_2026, datetime
#: 2026-04-29T00:00:00Z, archive sha "a"*64, archive url https://data.geo.admin.ch/archive.zip)
PRE_MOVE_DIGEST = "7808f6084a8b10ee9a3e7a93217120c5b99eca5dc429e80e6bf20f0c8060b2f9"
ITEM, STAMP, ARCHIVE_URL = "swissnames3d_2026", "2026-04-29T00:00:00Z", "https://data.geo.admin.ch/archive.zip"
OPTS = {"item_datetime": STAMP, "archive_url": ARCHIVE_URL}


def _zip_bytes(csv_bytes: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("swissNAMES3D_PKT.csv", csv_bytes)
    return buf.getvalue()


def _row() -> dict:
    return contract_mod.by_id(contract_mod.load())["named_slice:swisstopo-v2"]


def _ctx(root, tmp_path, contract=None, **opts) -> RunContext:
    return RunContext(out_dir=tmp_path / "out", today="2026-10-30", skip_consent=True,
                      options={"root": str(root), **OPTS, **opts}, contract=contract or _row(),
                      log=lambda m: None)


@pytest.fixture
def root(tmp_path):
    dst = tmp_path / "dump"
    dst.mkdir()
    (dst / "swissnames3d_2026_2056.csv.zip").write_bytes(_zip_bytes(FIXTURE.read_bytes()))
    return dst


def test_registered_and_conformant():
    assert "named_slice:swisstopo-v2" in registry.registered()
    adapter, row, spec = registry.admit("named_slice:swisstopo-v2")
    assert isinstance(adapter, SourceFamily)
    assert row["family_id"] == "named_slice:swisstopo-v2"
    assert not getattr(adapter, "PARTIAL", ())
    assert spec.declared == "chunk-policy-atomic-record.v1"


def test_parser_output_matches_the_pre_move_driver():
    zf = zipfile.ZipFile(io.BytesIO(_zip_bytes(FIXTURE.read_bytes())))
    out = []
    for member, row in S.rows(zf):
        p = S.build_passage(row, ITEM)
        rec = S.structured_record(row, item_id=ITEM, item_datetime=STAMP, parent_member=member,
                                  parent_archive_sha256="a" * 64, archive_url=ARCHIVE_URL)
        out.append({"passage": asdict(p) if p else None, "record": rec})
    assert [bool(o["passage"]) for o in out] == [True, True, True, False, False]
    digest = hashlib.sha256(json.dumps(out, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert digest == PRE_MOVE_DIGEST


def test_discover_acquire_extract_the_archive(root, tmp_path):
    fam, ctx = SwisstopoFamily(), _ctx(root, tmp_path)
    (unit,) = fam.discover(ctx)
    assert unit.key == "swissnames3d_2026_2056.csv.zip" and unit.url.startswith("file:")  # no host
    assert unit.meta["item_id"] == ITEM and ctx.notes["files"] == 1
    (cap,) = fam.acquire(unit, ctx)
    assert cap.meta["sha256"] == hashlib.sha256(cap.body).hexdigest()
    docs = list(fam.extract(cap, ctx))
    assert [d.title for d in docs] == ["Bern", "Lac Léman", "Piz Bernina"]
    assert [d.lang for d in docs] == ["de", "fr", "it"]
    rec = docs[0].extra["record"]
    assert rec["parent_archive"] == {"member": "swissNAMES3D_PKT.csv", "sha256": cap.meta["sha256"]}
    assert rec["source_release"] == {"id": ITEM, "datetime": STAMP}
    assert docs[0].url.startswith("https://map.geo.admin.ch/?E=2600000&N=1200000")


def test_terms_come_from_the_contract_row(root, tmp_path):
    fam, ctx = SwisstopoFamily(), _ctx(root, tmp_path)
    fam.discover(ctx)
    t, row = ctx.notes["contract_terms"], _row()
    assert t["search_ok"] is row["consent"]["search_ok"]
    assert t["consent_basis"] == row["consent"]["basis"]
    assert t["licence_basis"] == row["licence"]["basis"] == "instrument"
    with pytest.raises(ValueError, match="contract row"):
        SwisstopoFamily().discover(_ctx(root, tmp_path, {"family_id": "named_slice:swisstopo-v2"}))


def test_release_identity_is_required(root, tmp_path):
    ctx = _ctx(root, tmp_path)
    del ctx.options["archive_url"]
    with pytest.raises(ValueError, match="archive_url"):
        SwisstopoFamily().discover(ctx)


def test_an_archive_with_no_admitted_feature_is_excluded(tmp_path):
    d = tmp_path / "b"
    d.mkdir()
    head = FIXTURE.read_text(encoding="utf-8").splitlines()[0]
    (d / "x.csv.zip").write_bytes(_zip_bytes((head + "\n").encode()))
    fam, ctx = SwisstopoFamily(), _ctx(d, tmp_path)
    (unit,) = fam.discover(ctx)
    (cap,) = fam.acquire(unit, ctx)
    with pytest.raises(UnitExcluded):
        list(fam.extract(cap, ctx))


def test_an_empty_walk_is_a_wrong_root(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(RuntimeError, match="no .csv.zip"):
        SwisstopoFamily().discover(_ctx(tmp_path / "empty", tmp_path))


def test_plan_dry_run_on_the_sealed_fixture(root, tmp_path):
    out = tmp_path / "plan"
    plan = run_family("named_slice:swisstopo-v2", out_dir=out, today="2026-10-30", dry_run=True,
                      skip_consent=True, options={"root": str(root), **OPTS}, log=lambda m: None)
    assert plan is not None
    assert not (out / "pack").exists()


def test_atomic_policy_is_declared_and_a_run_composes_the_pack(root, tmp_path):
    # task 2260: the row declares chunk-policy-atomic-record.v1 (one record = one passage). A real
    # run on the committed fixture (the stage door's own entry point) must write a pack whose
    # records are the extracted records, one per passage.
    from cytir.sources import run_family as rf
    adapter, row, spec = registry.admit("named_slice:swisstopo-v2")
    assert row["chunk_policy"]["policy_id"] == "chunk-policy-atomic-record.v1"
    out = tmp_path / "o"
    run_family("named_slice:swisstopo-v2", out_dir=out, today="2026-10-30", skip_consent=True,
               options={"root": str(root), **OPTS}, log=lambda m: None)
    _, recs = rf.load_pack(out / "pack")
    assert recs and all(r.get("url") and r.get("text") for r in recs)
    assert len({r["url"] for r in recs}) == len(recs)
