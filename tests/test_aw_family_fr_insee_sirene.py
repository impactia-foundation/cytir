"""`fr_insee_sirene_cards` on the family Protocol (task 2728).

No network. The fixture is 40 REAL rows of INSEE's StockUniteLegale parquet with the real response
headers (provenance: tests/fixtures/fr_insee_sirene/PROVENANCE.md). Pins: registered, conformant,
PARTIAL (extract, chunk); discover/acquire on the sealed evidence set; the terms come from the contract
row; the seal blocks index admission; a tampered body is refused; the pinned identity is the extractor's.
"""
from __future__ import annotations

import json
import pathlib
import shutil

import pytest

from cytir.sources import contract as contract_mod
from cytir.sources import registry
from cytir.sources.family import RunContext, SourceFamily
from cytir.sources.families.fr_insee_sirene_cards import FrInseeSireneCardsFamily
from cytir.sources.official_bulk import sirene_legal_units as SL

FID = "fr_insee_sirene_cards"
FIX = pathlib.Path(__file__).parent / "fixtures" / "fr_insee_sirene"
BODY = "stock-unite-legale.slice.parquet"


def _row() -> dict:
    return contract_mod.by_id(contract_mod.load())[FID]


@pytest.fixture
def fetched(tmp_path):
    for n in (BODY, BODY + ".headers", BODY + ".transport.json"):
        shutil.copy(FIX / n, tmp_path / n)
    return tmp_path


def _ctx(d, tmp_path, contract=None) -> RunContext:
    opts = {"body": str(d / BODY), "headers": str(d / (BODY + ".headers")),
            "metadata": str(d / (BODY + ".transport.json")), "requested_url": SL.URL,
            "root": str(tmp_path / "run"), "source_id": "fr-sirene-legal-units",
            "source_name": "INSEE SIRENE stockUniteLegale", "country": "FR",
            "content_scope": SL.CONTENT_SCOPE, "next_projection": SL.NEXT_PROJECTION,
            "fetch_vantage": "fixture"}
    return RunContext(out_dir=tmp_path / "out", today="2026-11-12", skip_consent=True,
                      options=opts, contract=contract or _row(), log=lambda m: None)


def test_registered_conformant_and_partial():
    assert FID in registry.registered()
    adapter, row, spec = registry.admit(FID)
    assert isinstance(adapter, SourceFamily)
    assert row["family_id"] == FID and row["route"] == adapter.route == "bulk-dump"
    assert spec.declared == row["chunk_policy"]["policy_id"]
    assert adapter.PARTIAL == ("extract", "chunk")


def test_discover_and_acquire_on_the_real_slice(fetched, tmp_path):
    fam, ctx = FrInseeSireneCardsFamily(), _ctx(fetched, tmp_path)
    (unit,) = fam.discover(ctx)
    assert unit.url.startswith("file:")              # the adapter contacts no host
    (cap,) = fam.acquire(unit, ctx)
    assert cap.meta["status"] == 200
    assert cap.meta["source_url"].endswith("350182c9-148a-46e0-8389-76c2ec1374a3")
    assert cap.meta["bulk_body_bytes"] == (fetched / BODY).stat().st_size


def test_the_sealed_body_holds_real_legal_unit_rows(fetched):
    pq = pytest.importorskip("pyarrow.parquet")
    t = pq.read_table(fetched / BODY)
    assert t.num_rows == 40
    sirens = t.column("siren").to_pylist()
    assert sirens[0] == "009741984" and all(len(s) == 9 and s.isdigit() for s in sirens)
    assert len(set(sirens)) == 40
    row = t.slice(0, 1).to_pylist()[0]
    assert row["denominationUniteLegale"] == "ASS. COPROPRIETE"
    assert row["activitePrincipaleUniteLegale"] == "81.10Z"
    assert {"statutDiffusionUniteLegale", "categorieJuridiqueUniteLegale"} <= set(t.column_names)
    assert not {"communeEtablissement", "libelleCommuneEtablissement"} & set(t.column_names)


def test_terms_come_from_the_contract_row(fetched, tmp_path):
    ctx = _ctx(fetched, tmp_path)
    FrInseeSireneCardsFamily().discover(ctx)
    t, row = ctx.notes["contract_terms"], _row()
    assert t["search_ok"] is True and t["train_ok"] is True
    assert t["licence_spdx"] == row["licence"]["spdx"] == "etalab-2.0"
    assert t["consent_resolution"] == row["consent"]["resolution"] == "explicit-grant"


def test_a_row_that_does_not_declare_search_ok_refuses(fetched, tmp_path):
    row = json.loads(json.dumps(_row()))
    row["consent"]["search_ok"] = False
    with pytest.raises(ValueError, match="search_ok"):
        FrInseeSireneCardsFamily().discover(_ctx(fetched, tmp_path, row))


def test_seal_original_blocks_index_admission(fetched, tmp_path):
    run = FrInseeSireneCardsFamily().seal_original(_ctx(fetched, tmp_path))
    assert run["status"] == "source_acquired_pending_projection"
    assert run["index_admission"].startswith("blocked")
    assert (tmp_path / "run" / "RUN.json").is_file()


def test_a_body_that_disagrees_with_its_metadata_is_refused(fetched, tmp_path):
    fam, ctx = FrInseeSireneCardsFamily(), _ctx(fetched, tmp_path)
    (unit,) = fam.discover(ctx)
    (fetched / BODY).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="body disagrees"):
        fam.acquire(unit, ctx)


def test_extract_and_chunk_are_declared_partial(fetched, tmp_path):
    fam = FrInseeSireneCardsFamily()
    with pytest.raises(NotImplementedError, match="PARTIAL"):
        fam.extract(None, None)
    with pytest.raises(NotImplementedError, match="PARTIAL"):
        fam.chunk(None, None)
