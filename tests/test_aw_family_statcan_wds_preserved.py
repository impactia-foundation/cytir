"""`scratch_statcan_wds_exact` on the family Protocol: the preserved WDS originals (task 1405, S7).

No network, no live fixture: the inbox is built here from a synthetic catalogue. Pins: the adapter is
registered and conformant; discover/acquire validate the preserved evidence; consent and licence come
from the contract row as declared (unknown stays unknown); a partial or tampered inbox is refused; and
`seal_preserved` seals the SAME BYTES as the pre-move driver.

PARITY STANDARD (task 1152): `PINNED` holds the sha256 of every file the pre-move driver sealed for
this fixture, taken from the driver at origin/main commit c989add9b (fixed clock 2026-08-15, the same
fixture) and run twice to prove it deterministic. A change to the moved code that alters one byte
of a sealed receipt fails here.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from cytir.sources import contract as contract_mod
from cytir.sources import registry
from cytir.sources.family import RunContext, SourceFamily
from cytir.sources.families.statcan_wds_preserved import ENDPOINTS, StatcanWdsPreservedFamily
from cytir.sources.statcan import preserved as P

FID = "scratch_statcan_wds_exact"
CLOCK = "2026-08-15T00:00:00Z"
HEADERS = b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n"
CUBES = [
    {"productId": 2, "archived": "2", "cubeTitleEn": "Active English title",
     "cubeTitleFr": "Titre français actif", "cubeStartDate": "2020-01-01",
     "cubeEndDate": "2026-01-01", "releaseTime": "2026-08-01"},
    {"productId": 1, "archived": "1", "cubeTitleEn": "Archived", "cubeTitleFr": "Archivé"},
    {"productId": 3, "archived": "2", "cubeTitleEn": "Only English", "cubeTitleFr": ""},
]

PINNED: dict[str, str] = {
    "RUN.json": "5b5d732663c99b95a6e32c6b01ee7392dc0350f821e756dabf60f44f85d8f34a",
    "stages/markdown/data/a8/a8e2b328f2b1a5f6c6dc1d54d347582703e711e73983ed80856b12ad66517d0d.jsonl.gz": "1c1e073312cc2c4627b55edc248db7bce5408577c4680a00b2d034a67a3c6b06",
    "stages/markdown/receipt.json": "d5db257df88d2067e95bfedb0ba11a4059c04dbc9b9c7b0269cd5e4ff16d3655",
    "stages/original/data/body/4f/4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945.json.gz": "ad2d9bce436c0552820a7cbe7adcdad183162c2c5108739882624d9813b82bf7",
    "stages/original/data/body/78/784e3d2bbff276bd6dde9a3bf6aeece29b744bb405a8b6ecdac54e487865f428.json.gz": "a5ae0a20358a441923275121eba248f611f9d152732107249b84e4f88e85a2d7",
    "stages/original/data/headers/53/533af766959d485c4caf008ee7f966fa793d9fcbef8327e99a20b08799a00b64.headers.gz": "6fac36686735b0fecd8f2134f05f3ecefa3080ab5a19e9fc58169a9fb69891e5",
    "stages/original/data/metadata/46/462cfa9cc66c918ad7aa22aa49364f16fa3c262e358cf30004134fa63aa623ee.json.gz": "066ab7abe7c3242ba658b61a704edbf632cfde1ac7d3b6b38fe7b4b5ad6fadfd",
    "stages/original/data/metadata/53/53cf4667836800e9ac700850529bbffdc32ec575d7c682127a1132d82b069efc.json.gz": "13730dee7a4cb11a53bb52e7e7021ae10b4bbbc26a637bbf08ae803419b9e455",
    "stages/original/receipt.json": "2e9a95aceb9822cbe0261f86d5fe5b3992988463afbdb603043bfb065294a3f0",
    "stages/outlinks/data/6a/6a4ce41813681f3c2d979574545e72ffc76dfc6fb3a55e15a9be15f779d6b210.jsonl.gz": "3908d84579f2420eda7e053c5b71a43a5ea0c8b825fd1ef1347a185154e0a083",
    "stages/outlinks/receipt.json": "9ddf192fad6d1d6c7168941c6707b060d0dce1be2d6de805ae0245b9455640de"
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_inbox(inbox: pathlib.Path) -> pathlib.Path:
    """The six files the operator loop leaves per endpoint pair, from the synthetic catalogue."""
    inbox.mkdir(parents=True, exist_ok=True)
    bodies = {"getAllCubesListLite": json.dumps(CUBES).encode(), "getCodeSets": b"[]"}
    for endpoint, body in bodies.items():
        url = f"{P.WDS}/{endpoint}"
        (inbox / f"{endpoint}.json").write_bytes(body)
        (inbox / f"{endpoint}.json.headers").write_bytes(HEADERS)
        meta = {"schema": "aw.http-fetch-metadata.v1", "requested_url": url, "final_url": url,
                "http_status": 200, "fetched_at": CLOCK,
                "body": {"bytes": len(body), "sha256": _sha(body)},
                "response_headers": {"encoding": "curl-dump-header-raw", "bytes": len(HEADERS),
                                     "sha256": _sha(HEADERS)}}
        (inbox / f"{endpoint}.json.transport.json").write_bytes(
            (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode())
    return inbox


def tree_digest(root: pathlib.Path) -> dict[str, str]:
    return {str(p.relative_to(root)): _sha(p.read_bytes())
            for p in sorted(root.rglob("*")) if p.is_file()}


def _row() -> dict:
    return contract_mod.by_id(contract_mod.load())[FID]


def _ctx(inbox, tmp_path, contract=None, **opts) -> RunContext:
    options = {"inbox": str(inbox), **opts}
    return RunContext(out_dir=tmp_path / "out", today="2026-10-30", skip_consent=True,
                      options=options, contract=contract or _row(), log=lambda m: None)


@pytest.fixture
def inbox(tmp_path):
    return build_inbox(tmp_path / "inbox")


def test_registered_and_conformant():
    assert FID in registry.registered()
    adapter, row, spec = registry.admit(FID)
    assert isinstance(adapter, SourceFamily)
    assert row["family_id"] == FID
    assert adapter.PARTIAL == ("extract", "chunk")
    assert spec.declared == "dispatcher"      # the row declares no chunk policy


def test_discover_and_acquire_validate_the_evidence(inbox, tmp_path):
    fam, ctx = StatcanWdsPreservedFamily(), _ctx(inbox, tmp_path)
    units = list(fam.discover(ctx))
    assert [u.key for u in units] == list(ENDPOINTS)
    assert all(u.url.startswith("file:") for u in units)          # no host is contacted
    (cap,) = fam.acquire(units[0], ctx)
    assert json.loads(cap.body)[0]["productId"] == 2
    assert cap.meta["status"] == 200 and cap.meta["fetched_at"] == CLOCK


def test_terms_come_from_the_contract_row_as_declared(inbox, tmp_path):
    fam, ctx = StatcanWdsPreservedFamily(), _ctx(inbox, tmp_path)
    fam.discover(ctx)
    t, row = ctx.notes["contract_terms"], _row()
    assert t["search_ok"] is row["consent"]["search_ok"]
    assert t["train_ok"] is row["consent"]["train_ok"]
    assert t["consent_resolution"] == row["consent"]["resolution"]
    assert t["licence_basis"] == row["licence"]["basis"]
    assert t["licence_spdx"] == row["licence"]["spdx"]
    with pytest.raises(ValueError, match="contract row"):
        fam.discover(_ctx(inbox, tmp_path, {"family_id": FID}))


def test_a_partial_inbox_is_refused(inbox, tmp_path):
    (inbox / "getCodeSets.json.headers").unlink()
    with pytest.raises(RuntimeError, match="partial evidence set"):
        StatcanWdsPreservedFamily().discover(_ctx(inbox, tmp_path))


def test_a_tampered_body_is_refused(inbox, tmp_path):
    fam, ctx = StatcanWdsPreservedFamily(), _ctx(inbox, tmp_path)
    units = list(fam.discover(ctx))
    (inbox / "getCodeSets.json").write_bytes(b"{}")
    with pytest.raises(P.StatCanError, match="disagrees with transport receipt"):
        fam.acquire(units[1], ctx)


def test_later_stages_say_they_are_not_ported(inbox, tmp_path):
    fam, ctx = StatcanWdsPreservedFamily(), _ctx(inbox, tmp_path)
    (cap,) = fam.acquire(list(fam.discover(ctx))[0], ctx)
    with pytest.raises(NotImplementedError):
        fam.extract(cap, ctx)


def test_seal_preserved_counts_and_run_receipt(inbox, tmp_path):
    root = tmp_path / "root"
    ctx = _ctx(inbox, tmp_path, root=str(root), fetch_vantage="fixture", fetched_at=CLOCK)
    run = StatcanWdsPreservedFamily().seal_preserved(ctx)
    assert run["counts"] == {"cube_records": 3, "active_markdown_documents": 3,
                             "outlink_records": 3, "archived_cubes_skipped": 1,
                             "untitled_language_records_skipped": 1}
    assert (root / "RUN.json").is_file()


def test_seal_preserved_is_byte_identical_to_the_pre_move_driver(inbox, tmp_path):
    root = tmp_path / "root"
    ctx = _ctx(inbox, tmp_path, root=str(root), fetch_vantage="fixture", fetched_at=CLOCK)
    StatcanWdsPreservedFamily().seal_preserved(ctx)
    assert tree_digest(root) == PINNED
