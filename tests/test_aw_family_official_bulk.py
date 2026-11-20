"""`uk_legislation` and `it_senato_gazzetta` on the family Protocol (task 1394).

No network. The bulk response is a tiny synthetic body whose transport metadata is written the way
the operator's fetch recorder writes it (schema `aw.http-fetch-metadata.v1`). Pins: both families
are registered, conformant and PARTIAL; discover/acquire dry-run on the sealed fixture; consent and
licence come from the contract row; the seal writes RUN.json blocked from index admission; a body
that disagrees with its metadata is refused; the producer labels stay frozen.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil

import pytest

from cytir.sources import contract as contract_mod
from cytir.sources import registry
from cytir.sources.family import RunContext, SourceFamily
from cytir.sources.families.official_bulk import ItSenatoGazzettaFamily, UkLegislationFamily
from cytir.sources.official_bulk import bulk, direct

FAMILIES = {"uk_legislation": UkLegislationFamily, "it_senato_gazzetta": ItSenatoGazzettaFamily}
URL = "https://official.example/bulk.zip"
BODY = b"PK\x03\x04fixture"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _row(fid) -> dict:
    return contract_mod.by_id(contract_mod.load())[fid]


@pytest.fixture
def fetched(tmp_path):
    headers = b"HTTP/2 200\r\n\r\n"
    (tmp_path / "bulk.zip").write_bytes(BODY)
    (tmp_path / "bulk.headers").write_bytes(headers)
    meta = {"schema": "aw.http-fetch-metadata.v1", "requested_url": URL, "final_url": URL,
            "http_status": 200, "fetched_at": "2026-10-30T00:00:00Z",
            "body": {"bytes": len(BODY), "sha256": _sha(BODY)},
            "response_headers": {"bytes": len(headers), "sha256": _sha(headers)}}
    (tmp_path / "bulk.json").write_text(json.dumps(meta))
    return tmp_path


def _ctx(fid, d, tmp_path, contract=None, **extra) -> RunContext:
    opts = {"body": str(d / "bulk.zip"), "headers": str(d / "bulk.headers"),
            "metadata": str(d / "bulk.json"), "requested_url": URL, "root": str(tmp_path / "run"),
            "source_id": "x", "source_name": "X", "country": "GB", "content_scope": "dump",
            "next_projection": "akn to markdown", "fetch_vantage": "fixture", **extra}
    return RunContext(out_dir=tmp_path / "out", today="2026-10-30", skip_consent=True,
                      options=opts, contract=contract or _row(fid), log=lambda m: None)


@pytest.mark.parametrize("fid", FAMILIES)
def test_registered_conformant_and_partial(fid):
    assert fid in registry.registered()
    adapter, row, spec = registry.admit(fid)
    assert isinstance(adapter, SourceFamily)
    assert row["family_id"] == fid and row["route"] == adapter.route == "bulk-dump"
    assert spec.declared == row["chunk_policy"]["policy_id"]
    assert adapter.PARTIAL == (() if fid == "uk_legislation" else ("extract", "chunk"))  # uk: task 2181


def _parked(fid) -> bool:
    return bool((_row(fid).get("acquisition") or {}).get("parked"))


def _unparked_row(fid) -> dict:
    """The row with the park lifted: exercises the seal mechanics only, never the shipped row."""
    row = json.loads(json.dumps(_row(fid)))
    row["acquisition"].pop("parked", None)
    row["consent"].update(search_ok=True, train_ok=True)
    row["consent"].pop("consent_state", None)
    return row


@pytest.mark.parametrize("fid", FAMILIES)
def test_discover_and_acquire_dry_run(fid, fetched, tmp_path):
    if _parked(fid):
        with pytest.raises(ValueError, match="parked.*unreadable"):
            FAMILIES[fid]().discover(_ctx(fid, fetched, tmp_path))
        return
    fam, ctx = FAMILIES[fid](), _ctx(fid, fetched, tmp_path)
    (unit,) = fam.discover(ctx)
    assert unit.url.startswith("file:")              # the adapter contacts no host
    (cap,) = fam.acquire(unit, ctx)
    assert cap.meta["status"] == 200 and cap.meta["bulk_body_bytes"] == len(BODY)
    assert cap.meta["bulk_body_sha256"] == _sha(BODY)


@pytest.mark.parametrize("fid", FAMILIES)
def test_terms_come_from_the_contract_row(fid, fetched, tmp_path):
    ctx = _ctx(fid, fetched, tmp_path)
    if _parked(fid):
        with pytest.raises(ValueError, match="consent_state='unreadable'"):
            FAMILIES[fid]().discover(ctx)
        return
    FAMILIES[fid]().discover(ctx)
    t, row = ctx.notes["contract_terms"], _row(fid)
    assert t["search_ok"] is row["consent"]["search_ok"]
    assert t["licence_basis"] == row["licence"]["basis"]
    assert t["licence_spdx"] == row["licence"]["spdx"]


@pytest.mark.parametrize("fid", FAMILIES)
def test_a_row_that_does_not_declare_search_ok_refuses(fid, fetched, tmp_path):
    row = _unparked_row(fid) if _parked(fid) else json.loads(json.dumps(_row(fid)))
    row["consent"]["search_ok"] = False
    with pytest.raises(ValueError, match="search_ok"):
        FAMILIES[fid]().discover(_ctx(fid, fetched, tmp_path, row))
    with pytest.raises(ValueError, match="contract row"):
        FAMILIES[fid]().discover(_ctx(fid, fetched, tmp_path, {"family_id": fid}))


def test_seal_original_blocks_index_admission(fetched, tmp_path):
    run = UkLegislationFamily().seal_original(_ctx("uk_legislation", fetched, tmp_path))
    assert run["status"] == "source_acquired_pending_projection"
    assert run["index_admission"].startswith("blocked")
    assert (tmp_path / "run" / "RUN.json").is_file()


def test_direct_seal_takes_the_licence_from_the_row(fetched, tmp_path):
    ctx = _ctx("it_senato_gazzetta", fetched, tmp_path, _unparked_row("it_senato_gazzetta"),
               direct="1", artifact_root=str(fetched))
    run = ItSenatoGazzettaFamily().seal_original(ctx)
    assert run["license"] == _row("it_senato_gazzetta")["licence"]["spdx"] == "CC-BY-4.0"
    assert run["version"] == direct.VERSION


def test_a_body_that_disagrees_with_its_metadata_is_refused(fetched, tmp_path):
    fam, ctx = UkLegislationFamily(), _ctx("uk_legislation", fetched, tmp_path)
    (unit,) = fam.discover(ctx)
    (fetched / "bulk.zip").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="body disagrees"):
        fam.acquire(unit, ctx)


def test_frozen_producer_labels():
    assert bulk.PRODUCER_ID == "pipeline/acquire_official_bulk_source.py"
    assert direct.PRODUCER_ID == "pipeline/acquire_official_bulk_source_direct.py"


# Reference digests: produced by the ORIGINAL flat drivers (pipeline/acquire_official_bulk_source.py
# and ..._direct.py as of commit 9b23bc508, the tree before the move) on the committed fixture
# tests/fixtures/official_bulk/, each run twice with identical output. Nothing is excluded:
# `fetched_at` is read from the fixture's transport metadata, so RUN.json and the receipt are fully
# deterministic. Plain run: country XX; direct run: licence CC-BY-4.0, artifact_root = the work dir.
REFERENCE = {
    "bulk": {"run_json": "61224c0752b2bdeb86d296cec1fe4bf22a1afdc8d1f197470bbf15ed652971cf",
             "receipt": "14b8d00785c0baaea5526e7f342352a49c04fa5627cf2c3ddfc6d6db4624f101"},
    "direct": {"run_json": "44a23bc237cae2881067a04202c0046434008ce26cd7fda1f9aed849f1304aa7",
               "receipt": "8411145dc50c3ceb4b03db2c726f58e20da05d8ce78507788808c030d63f6df9"},
}
FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "official_bulk"


def _digests(root: pathlib.Path) -> dict:
    return {"run_json": _sha((root / "RUN.json").read_bytes()),
            "receipt": _sha((root / "stages/original/receipt.json").read_bytes())}


def test_seal_is_byte_identical_to_the_original_drivers(tmp_path):
    got = {}
    for mode, fid, extra in (("bulk", "uk_legislation", {"country": "XX"}),
                             ("direct", "it_senato_gazzetta", {"direct": "1", "artifact_root": str(tmp_path / "direct")})):
        work = tmp_path / mode
        shutil.copytree(FIXTURE, work / "in")
        opts = {"body": str(work / "in/bulk.zip"), "headers": str(work / "in/bulk.headers"),
                "metadata": str(work / "in/bulk.json"), "requested_url": "https://fixture-host/bulk.zip",
                "root": str(work / "run"), "source_id": "fixture-src", "source_name": "Fixture Source",
                "content_scope": "synthetic", "next_projection": "none", "fetch_vantage": "fixture",
                **extra}
        ctx = RunContext(out_dir=work / "out", today="2026-10-30", skip_consent=True, options=opts,
                         contract=_unparked_row(fid) if _parked(fid) else _row(fid),
                         log=lambda m: None)
        FAMILIES[fid]().seal_original(ctx)
        got[mode] = _digests(work / "run")
    assert got == REFERENCE
