"""`eric_official_pdf` on the family Protocol, slice 1: the inventory acquirer (task 1386).

No network. The fixture is one page of five real ERIC records (`data/samples/eric/README.md` records
where it was cut). Pins: the adapter is registered and conformant; it discovers and acquires the page
from the sealed receipt; consent and licence come from the contract row (a row without them refuses);
the whole-inventory seal runs and conserves the ids; a tampered page is refused.
"""
from __future__ import annotations

import json
import pathlib
import shutil

import pytest

from cytir.sources import contract as contract_mod
from cytir.sources import registry
from cytir.sources.eric import inventory as inv
from cytir.sources.family import RunContext, SourceFamily
from cytir.sources.families.eric_official_pdf import EricOfficialPdfFamily

REPO = pathlib.Path(__file__).resolve().parent.parent
FIXTURE = REPO / "data" / "samples" / "eric"
IDS = ["ED659153", "ED659156", "ED659161", "ED659172", "ED659173"]


def _row() -> dict:
    return contract_mod.by_id(contract_mod.load())["eric_official_pdf"]


def _ctx(root, tmp_path, contract=None) -> RunContext:
    return RunContext(out_dir=tmp_path / "out", today="2026-10-28", skip_consent=True,
                      options={"root": str(root)}, contract=contract or _row(), log=lambda m: None)


@pytest.fixture
def root(tmp_path):
    dst = tmp_path / "inventory"
    shutil.copytree(FIXTURE, dst)
    return dst


def test_registered_and_conformant():
    assert "eric_official_pdf" in registry.registered()
    adapter, row, spec = registry.admit("eric_official_pdf")
    assert isinstance(adapter, SourceFamily)
    assert row["family_id"] == "eric_official_pdf"
    assert spec.declared == row["chunk_policy"]["policy_id"]
    assert not hasattr(adapter, "PARTIAL")          # slice 2 ported extract and chunk
    assert callable(getattr(adapter, "l1", None))


def test_discover_and_acquire_the_real_page(root, tmp_path):
    fam, ctx = EricOfficialPdfFamily(), _ctx(root, tmp_path)
    units = list(fam.discover(ctx))
    assert [u.key for u in units] == ["page-000000000"]
    assert ctx.notes["catalog_records"] == 5 and ctx.notes["pages"] == 1
    assert units[0].url.startswith("file:")          # no host is contacted
    (cap,) = fam.acquire(units[0], ctx)
    docs = json.loads(cap.body)["response"]["docs"]
    assert [d["id"] for d in docs] == IDS
    assert all(d["e_fulltextauth"] == 1 for d in docs)
    assert cap.meta["status"] == 200 and cap.meta["documents"] == 5


def test_terms_come_from_the_contract_row(root, tmp_path):
    fam, ctx = EricOfficialPdfFamily(), _ctx(root, tmp_path)
    fam.discover(ctx)
    t, row = ctx.notes["contract_terms"], _row()
    assert t["search_ok"] is row["consent"]["search_ok"]
    assert t["train_ok"] is row["consent"]["train_ok"]
    assert t["consent_resolution"] == row["consent"]["resolution"]
    assert t["licence_basis"] == row["licence"]["basis"]
    assert t["licence_spdx"] == row["licence"]["spdx"]


def test_a_row_that_does_not_declare_search_ok_refuses(root, tmp_path):
    row = json.loads(json.dumps(_row()))
    row["consent"]["search_ok"] = False
    with pytest.raises(ValueError, match="search_ok"):
        EricOfficialPdfFamily().discover(_ctx(root, tmp_path, row))
    with pytest.raises(ValueError, match="contract row"):
        EricOfficialPdfFamily().discover(_ctx(root, tmp_path, {"family_id": "eric_official_pdf"}))


def test_a_tampered_page_is_refused(root, tmp_path):
    fam, ctx = EricOfficialPdfFamily(), _ctx(root, tmp_path)
    (unit,) = fam.discover(ctx)
    body = root / "pages" / "000000000.body.json"
    body.write_bytes(body.read_bytes().replace(b"ED659153", b"ED000000"))
    with pytest.raises(inv.EricInventoryError, match="digest-mismatched"):
        fam.acquire(unit, ctx)


def test_an_empty_walk_is_a_wrong_root(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(RuntimeError, match="no page receipt"):
        EricOfficialPdfFamily().discover(_ctx(tmp_path / "empty", tmp_path))


def test_seal_inventory_conserves_the_ids(root, tmp_path):
    fam, ctx = EricOfficialPdfFamily(), _ctx(root, tmp_path)
    run = fam.seal_inventory(ctx)
    assert run["counts"] == {"api_pages": 1, "catalog_records": 5, "unique_document_ids": 5}
    assert (root / "RUN.json").is_file()
    assert run["stages"]["original"]["records"] == 1


def test_an_inventory_page_is_not_a_document(root, tmp_path):
    fam, ctx = EricOfficialPdfFamily(), _ctx(root, tmp_path)
    (unit,) = fam.discover(ctx)
    (cap,) = fam.acquire(unit, ctx)
    with pytest.raises(ValueError, match="stage=pdf"):
        fam.extract(cap, ctx)
