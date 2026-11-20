"""`justice_exact` and `justice_residual_exact` on the family Protocol (task 1402).

No network. The fixture is a synthetic pinned snapshot built in `tmp_path` (a two-member tar.gz and
the operator's evidence files), the same shape the operator driver writes. Pins: both rows are
registered and conformant and declared PARTIAL; discover and acquire read the pinned archive; the
contract terms come from the row (`unknown` stays `unknown`); the seal writes the receipt and
`RUN.json`; a wrong commit, a tampered archive and a missing archive are refused.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import pathlib
import tarfile

import pytest

from cytir.sources import contract as contract_mod
from cytir.sources import registry
from cytir.sources.family import RunContext, SourceFamily
from cytir.sources.families.justice_canada import (EVIDENCE, JusticeExactFamily,
                                                JusticeResidualExactFamily, contract_terms)
from cytir.sources.justice_canada import snapshot as snap

COMMIT = "a" * 40
URL = f"https://codeload.github.com/justicecanada/laws-lois-xml/tar.gz/{COMMIT}"
IDS = ("justice_exact", "justice_residual_exact")


def _row(fid: str) -> dict:
    return contract_mod.by_id(contract_mod.load())[fid]


def _fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    inp = tmp_path / "source" / "input"
    inp.mkdir(parents=True)
    archive = inp / "snapshot.tar.gz"
    # fixed mtimes and a fixed gzip header: the archive bytes (and so every digest) repeat on every run
    with gzip.GzipFile(archive, "wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        root = tarfile.TarInfo(f"laws-lois-xml-{COMMIT}")
        root.type = tarfile.DIRTYPE
        root.mtime = 0
        tar.addfile(root)
        for name in ("eng/acts/A.xml", "fra/lois/A.xml"):
            body = b"<Statute><Section id='1'>Text</Section></Statute>"
            info = tarfile.TarInfo(f"laws-lois-xml-{COMMIT}/{name}")
            info.size = len(body)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(body))
    headers = inp / "snapshot.headers"
    headers.write_bytes(b"HTTP/2 200\r\n\r\n")
    (inp / "snapshot.curl.json").write_text(json.dumps({"http_code": 200, "url_effective": URL}))

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    (inp / "snapshot.transport.json").write_text(json.dumps({
        "schema": "aw.http-fetch-metadata.v1", "requested_url": URL, "final_url": URL,
        "http_status": 200, "fetched_at": "2026-08-15T00:00:01Z",
        "body": {"bytes": archive.stat().st_size, "sha256": sha(archive)},
        "response_headers": {"bytes": headers.stat().st_size, "sha256": sha(headers)}}))
    for name, text in (("git-ls-remote.txt", f"{COMMIT}\tHEAD\n"), ("git-ls-remote.raw", f"{COMMIT}\tHEAD\n"),
                       ("git-ls-remote.rc", "0\n"), ("fetch.rc", "0\n"),
                       ("git-ls-remote.started", "2026-08-15T00:00:00Z\n"),
                       ("git-ls-remote.completed", "2026-08-15T00:00:01Z\n"),
                       ("fetch.started", "2026-08-15T00:00:00Z\n"),
                       ("fetch.completed", "2026-08-15T00:00:01Z\n")):
        (inp / name).write_text(text)
    (inp / "git-ls-remote.stderr").write_bytes(b"")
    (inp / "snapshot.curl.stderr").write_bytes(b"")
    return inp


def _ctx(inp, tmp_path, fid, **over) -> RunContext:
    opts = {"root": str(tmp_path / "source"), "artifact_root": str(tmp_path), "input": str(inp),
            "commit": COMMIT, "requested_url": URL, **over}
    return RunContext(out_dir=tmp_path / "out", today="2026-10-30", skip_consent=True,
                      options=opts, contract=_row(fid), log=lambda m: None)


@pytest.fixture
def inp(tmp_path):
    return _fixture(tmp_path)


@pytest.mark.parametrize("fid", IDS)
def test_registered_conformant_and_partial(fid):
    assert fid in registry.registered()
    adapter, row, spec = registry.admit(fid)
    assert isinstance(adapter, SourceFamily)
    assert row["family_id"] == fid and spec.declared == row["chunk_policy"]["policy_id"]
    if fid == IDS[0]:
        assert adapter.PARTIAL == ()
    else:
        assert "extract" in adapter.PARTIAL and "chunk" in adapter.PARTIAL
    assert ("materialize" in adapter.PARTIAL) == (fid == "justice_residual_exact")


@pytest.mark.parametrize("cls,fid", [(JusticeExactFamily, IDS[0]), (JusticeResidualExactFamily, IDS[1])])
def test_discover_and_acquire_the_pinned_archive(cls, fid, inp, tmp_path):
    fam, ctx = cls(), _ctx(inp, tmp_path, fid)
    (unit,) = fam.discover(ctx)
    assert unit.key == f"snapshot-{COMMIT}" and unit.url.startswith("file:")   # no host is contacted
    (cap,) = fam.acquire(unit, ctx)
    assert cap.body == (inp / "snapshot.tar.gz").read_bytes()
    assert cap.meta["xml_members"] == 2 and ctx.notes["observed_language_paths"] == ["en", "fr"]


@pytest.mark.parametrize("fid", IDS)
def test_terms_come_from_the_row_and_unknown_stays_unknown(fid, inp, tmp_path):
    ctx = _ctx(inp, tmp_path, fid)
    JusticeExactFamily().discover(ctx)
    t, row = ctx.notes["contract_terms"], _row(fid)
    assert t["search_ok"] == row["consent"]["search_ok"] and t["train_ok"] == row["consent"]["train_ok"]
    assert t["consent_resolution"] == row["consent"]["resolution"]
    assert t["licence_basis"] == row["licence"]["basis"] and t["licence_spdx"] == row["licence"]["spdx"]


def test_a_run_without_a_contract_row_refuses(inp, tmp_path):
    ctx = _ctx(inp, tmp_path, IDS[0])
    ctx.contract = {"family_id": IDS[0]}
    with pytest.raises(ValueError, match="contract row"):
        contract_terms(ctx)


def test_wrong_commit_tampered_and_missing_archive_are_refused(inp, tmp_path):
    fam = JusticeExactFamily()
    ctx = _ctx(inp, tmp_path, IDS[0])
    (unit,) = fam.discover(ctx)
    with pytest.raises(ValueError, match="pinned to the exact commit"):
        fam.acquire(unit, _ctx(inp, tmp_path, IDS[0], requested_url=URL.replace("a" * 40, "b" * 40)))
    (inp / "snapshot.tar.gz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="disagrees with transport metadata"):
        fam.acquire(unit, ctx)
    (inp / "snapshot.tar.gz").unlink()
    with pytest.raises(RuntimeError, match="no snapshot archive"):
        fam.discover(ctx)


@pytest.mark.parametrize("cls,fid", [(JusticeExactFamily, IDS[0]), (JusticeResidualExactFamily, IDS[1])])
def test_seal_snapshot_writes_the_run(cls, fid, inp, tmp_path):
    run = cls().seal_snapshot(_ctx(inp, tmp_path, fid))
    assert run["counts"]["xml_members"] == 2 and run["commit"] == COMMIT
    assert (tmp_path / "source" / "RUN.json").is_file()
    assert run["stages"]["original"]["records"] == 1


def test_later_stages_say_they_are_not_ported(inp, tmp_path):
    fam, ctx = JusticeResidualExactFamily(), _ctx(inp, tmp_path, IDS[1])
    (unit,) = fam.discover(ctx)
    (cap,) = fam.acquire(unit, ctx)
    with pytest.raises(NotImplementedError):
        fam.extract(cap, ctx)
    with pytest.raises(NotImplementedError):
        fam.chunk(None, ctx)


def test_exact_extract_is_the_materializers_render_and_chunk_is_the_declared_policy(inp, tmp_path, word_only_chunking_allowed):
    import hashlib as _h
    import xml.etree.ElementTree as ET
    from cytir.sources.justice_canada import render as R
    fam, ctx = JusticeExactFamily(), _ctx(inp, tmp_path, IDS[0])
    fam.preflight(ctx)
    (unit,) = fam.discover(ctx)
    (cap,) = fam.acquire(unit, ctx)
    (doc,) = fam.extract(cap, ctx)           # the fixture holds one Act member (the fra/ member is not an Act)
    url = "https://laws-lois.justice.gc.ca/eng/acts/A/"
    xml_bytes = b"<Statute><Section id='1'>Text</Section></Statute>"
    sha = _h.sha256(xml_bytes).hexdigest()
    _t, md, _v = R.render_official_markdown(ET.fromstring(xml_bytes), url=url,
                                            member=f"laws-lois-xml-{COMMIT}/eng/acts/A.xml", member_sha256=sha)
    assert doc.url == url and doc.text == md and doc.source_text == md and doc.lang == "en"
    assert doc.extra["document_id"] == _h.sha256(f"justice-canada-law\0{url}".encode()).hexdigest()
    assert doc.extra["producer"] == R.PRODUCER and R.PRODUCER in _row(IDS[0])["producer_ids"]  # the contract's frozen label
    assert doc.extra["source_record_sha256"] == sha
    recs = list(fam.chunk(doc, ctx))
    assert recs and [r["chunk_ordinal"] for r in recs] == list(range(len(recs)))
    assert all(r["document_id"] == doc.extra["document_id"] and r["url"] == url for r in recs)
    assert any(url in str(r) and "Official URL" in str(r) for r in recs)


def test_exact_chunk_before_preflight_refuses(inp, tmp_path):
    fam, ctx = JusticeExactFamily(), _ctx(inp, tmp_path, IDS[0])
    (unit,) = fam.discover(ctx)
    (cap,) = fam.acquire(unit, ctx)
    (doc,) = fam.extract(cap, ctx)
    with pytest.raises(RuntimeError):
        list(fam.chunk(doc, ctx))


#: Digests produced by the ORIGINAL driver (the file as of the commit before the move, 9b23bc508, the
#: acquire_justice_canada_laws_snapshot.py in the pipeline directory, sha256 5f58657e802851b25bd86c6c332ec526be114cbf472646b6fa8e24746c3f338a) sealing the `_fixture` archive
#: above in a fresh directory. The old driver has no clock (the receipt carries only the evidence
#: files' own digests), so no field is excluded or injected.
REFERENCE = {
    "run_json": "c30d2275554d14006d0e25f8f6a71616ef10e7d11aa6688f93302a29e58b6200",
    "receipt_json": "1f91859332aa116d2834c49337c2873391f774b289c4d5c7583015fb467c2a65",
    "original_logical": "453162ea0342a9e335ca6e1d32dcad819cb822d436388832d618fb46b0f095f6",
}


def test_seal_is_byte_identical_to_the_original_driver(inp, tmp_path):
    kw = dict(root=tmp_path / "source", artifact_root=tmp_path, commit=COMMIT, requested_url=URL,
              **{k: inp / n for k, n in EVIDENCE.items()})
    run = snap.seal(**kw)
    receipt = tmp_path / "source" / "stages" / "original" / "receipt.json"
    got = {"run_json": hashlib.sha256((tmp_path / "source" / "RUN.json").read_bytes()).hexdigest(),
           "receipt_json": hashlib.sha256(receipt.read_bytes()).hexdigest(),
           "original_logical": run["stages"]["original"]["logical_sha256"]}
    assert got == REFERENCE


def test_residual_row_delegates_materialize_to_the_ported_module(inp, tmp_path, monkeypatch):
    from cytir.sources.justice_canada import residual
    seen = {}
    monkeypatch.setattr(residual, "materialize", lambda **kw: seen.update(kw) or {"ok": True})
    ctx = _ctx(inp, tmp_path, IDS[1], identity_run=str(tmp_path / "id.json"), output=str(tmp_path / "o"))
    assert JusticeResidualExactFamily().materialize_residual(ctx) == {"ok": True}
    assert seen["output"] == tmp_path / "o" and seen["identity_run_path"] == tmp_path / "id.json"
