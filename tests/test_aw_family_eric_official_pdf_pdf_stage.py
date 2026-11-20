"""`eric_official_pdf` slice 2 (task 1392): the PDF stages of the adapter, no network.

The acquisition is a small synthetic one cut like the real PDF acquisition's output (a receipt per
ERIC id whose body is an `artifact://` URI); the PDF extractor is injected, so the test pins the
adapter's wiring (discover, acquire, extract, chunk, l1, the framework's pack), not the extractor.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib

import pytest

from cytir.extract import pdf as extract_pdf
from cytir.sources import contract as contract_mod
from cytir.sources import run_family
from cytir.sources.eric import materialize as mat
from cytir.sources.family import RunContext
from cytir.sources.families.eric_official_pdf import EricOfficialPdfFamily

BODY = "# Teachers in rural schools\n\n" + " ".join(
    f"Sentence {i} about the education of teachers in rural schools." for i in range(400))


def _row() -> dict:
    return contract_mod.by_id(contract_mod.load())["eric_official_pdf"]


_STATE = {"markdown": BODY, "empty": ()}


def _extractor(_markdown: str = ""):
    def fn(_pdf):
        return extract_pdf.ExtractResult(markdown="" if _pdf.stem in _STATE["empty"] else _STATE["markdown"], extractor="liteparse",
                                         extractor_version="t", route_reason="t", page_count=1,
                                         is_complex=False, table_dense_runs=0)
    return fn


@pytest.fixture
def acquisition(tmp_path):
    root = tmp_path / "artifacts"
    receipts = root / "receipts"
    receipts.mkdir(parents=True)
    for eric_id in ("ED100001", "ED100002"):
        pdf = root / "original" / f"{eric_id}.bin"
        pdf.parent.mkdir(exist_ok=True)
        pdf.write_bytes(b"%PDF-1.7\n" + eric_id.encode())
        (receipts / f"{eric_id}.json").write_text(json.dumps({
            "schema": "aw.eric-official-pdf-receipt.v1", "status": "complete", "eric_id": eric_id,
            "source_url": f"https://files.eric.ed.gov/fulltext/{eric_id}.pdf",
            "artifacts": {"response_body": mat._artifact(pdf, root)}}))
    return root, receipts


def _ctx(acquisition, tmp_path, markdown=BODY, **opts):
    root, receipts = acquisition
    options = {"stage": "pdf", "artifact_root": str(root), "receipts": str(receipts), **opts}
    _STATE["markdown"], _STATE["empty"] = markdown, ()
    return RunContext(out_dir=tmp_path / "out", today="2026-10-30", skip_consent=True,
                      options=options, contract=_row(), log=lambda m: None)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(extract_pdf, "extract_pdf", _extractor())
    monkeypatch.setattr("cytir.acquire.fetch_process.detect_lang",
                        lambda text, url, default: ("en", True))


def test_discover_names_the_pdf_host_and_one_unit_per_receipt(acquisition, tmp_path):
    fam, ctx = EricOfficialPdfFamily(), _ctx(acquisition, tmp_path)
    units = list(fam.discover(ctx))
    assert [u.key for u in units] == ["ED100001", "ED100002"]
    assert all(u.consent_hosts == ("files.eric.ed.gov",) for u in units)
    assert ctx.notes["contract_terms"]["search_ok"] is True


def test_acquire_returns_the_validated_pdf_bytes_and_a_tampered_body_refuses(acquisition, tmp_path):
    fam, ctx = EricOfficialPdfFamily(), _ctx(acquisition, tmp_path)
    unit = list(fam.discover(ctx))[0]
    (cap,) = fam.acquire(unit, ctx)
    assert cap.body.startswith(b"%PDF") and cap.meta["content_type"] == "application/pdf"
    assert cap.meta["pdf_sha256"] == hashlib.sha256(cap.body).hexdigest()
    (acquisition[0] / "original" / "ED100001.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="mismatch"):
        fam.acquire(unit, ctx)


def test_extract_uses_the_one_record_shape_and_chunk_cuts_under_the_declared_policy(acquisition, tmp_path, word_only_chunking_allowed):
    fam, ctx = EricOfficialPdfFamily(), _ctx(acquisition, tmp_path)
    fam.preflight(ctx)
    unit = list(fam.discover(ctx))[0]
    (cap,) = fam.acquire(unit, ctx)
    (doc,) = fam.extract(cap, ctx)
    record, _ = mat.build_record(cap.meta["receipt"], BODY)
    assert doc.extra["record"] == record and doc.lang == "en"
    rows = list(fam.chunk(doc, ctx))
    assert rows and {r["document_id"] for r in rows} == {record["document_id"]}
    assert [r["chunk_ordinal"] for r in rows] == list(range(len(rows)))
    assert all(r["url"] == "https://files.eric.ed.gov/fulltext/ED100001.pdf" for r in rows)


def test_an_empty_markdown_is_quarantined_not_chunked(acquisition, tmp_path):
    fam, ctx = EricOfficialPdfFamily(), _ctx(acquisition, tmp_path, markdown="  \n")
    unit = list(fam.discover(ctx))[0]
    (cap,) = fam.acquire(unit, ctx)
    with pytest.raises(ValueError, match="quarantined"):
        fam.extract(cap, ctx)


def test_l1_files_the_markdown_shard_under_the_out_dir_and_says_when_it_does_not(
        acquisition, tmp_path):
    parent = tmp_path / "parent"
    (parent / "markdown").mkdir(parents=True)
    with gzip.open(parent / "markdown" / "documents.jsonl.gz", "wt") as fh:
        fh.write(json.dumps({"url": "https://files.eric.ed.gov/fulltext/ED100001.pdf",
                             "markdown_sha256": "x", "markdown": BODY}) + "\n")
    fam = EricOfficialPdfFamily()
    ctx = _ctx(acquisition, tmp_path, parent_root=str(parent), release_name="eric-t")
    manifest = fam.l1(ctx)
    assert manifest["release"] == "eric-t" and manifest["files"] == 1
    assert pathlib.Path(manifest["dest"]).is_relative_to(ctx.out_dir)
    assert "skipped" in fam.l1(_ctx(acquisition, tmp_path))


def test_the_framework_plans_and_runs_the_pdf_stage_end_to_end(acquisition, tmp_path, word_only_chunking_allowed):
    root, receipts = acquisition
    opts = {"stage": "pdf", "artifact_root": str(root), "receipts": str(receipts),
            }
    _STATE["markdown"] = BODY
    plan = run_family.run_family("eric_official_pdf", out_dir=tmp_path / "plan",
                                 today="2026-10-30", skip_consent=True, options=opts,
                                 log=lambda m: None, dry_run=True)
    assert plan["units"] == ["ED100001", "ED100002"] and plan["dry_run"] is True
    manifest = run_family.run_family("eric_official_pdf", out_dir=tmp_path / "run",
                                     today="2026-10-30", skip_consent=True, options=opts,
                                     log=lambda m: None)
    assert manifest["terminals"]["units_in"] == 2 and manifest["terminals"]["units_admitted"] == 2
    rows = [json.loads(line) for line in
            (tmp_path / "run" / "pack" / "records.jsonl").read_text().splitlines()]
    assert rows and {r["url"] for r in rows} == {
        "https://files.eric.ed.gov/fulltext/ED100001.pdf",
        "https://files.eric.ed.gov/fulltext/ED100002.pdf"}
    assert all(r["source_type"] == "reference" and r["trust_tier"] == 0.7 for r in rows)
    assert all(r["license"] == "none declared" and r["lang"] == "en" for r in rows)
    assert all(r["last_seen"] == "2026-10-30" for r in rows)
    for r in rows:
        assert r["anchor"] == f"document:{r['document_id']}:chunk-{r['chunk_ordinal']}"


def test_chunk_before_preflight_names_preflight(acquisition, tmp_path):
    fam, ctx = EricOfficialPdfFamily(), _ctx(acquisition, tmp_path)
    (cap,) = fam.acquire(list(fam.discover(ctx))[0], ctx)
    (doc,) = fam.extract(cap, ctx)
    with pytest.raises(RuntimeError, match="preflight"):
        fam.chunk(doc, ctx)


def test_an_unknown_stage_and_missing_roots_refuse(acquisition, tmp_path):
    fam = EricOfficialPdfFamily()
    with pytest.raises(ValueError, match="stage"):
        fam.discover(_ctx(acquisition, tmp_path, stage="nope"))
    ctx = _ctx(acquisition, tmp_path)
    del ctx.options["receipts"]
    with pytest.raises(ValueError, match="receipts"):
        fam.discover(ctx)


def test_the_stamped_values_are_the_served_rows_values(acquisition, tmp_path, word_only_chunking_allowed):
    from cytir.sources.families import eric_official_pdf as mod
    assert (mod.SOURCE_TYPE, mod.TRUST_TIER, mod.TIER, mod.LICENSE) == (
        "reference", 0.7, "B", "none declared")
    fam, ctx = EricOfficialPdfFamily(), _ctx(acquisition, tmp_path)
    fam.preflight(ctx)
    (cap,) = fam.acquire(list(fam.discover(ctx))[0], ctx)
    row = list(fam.chunk(fam.extract(cap, ctx)[0], ctx))[0]
    assert (row["source_type"], row["trust_tier"], row["license"]) == (
        "reference", 0.7, "none declared")


def test_a_run_quarantines_the_empty_unit_and_records_the_reason(acquisition, tmp_path, word_only_chunking_allowed):
    root, receipts = acquisition
    opts = {"stage": "pdf", "artifact_root": str(root), "receipts": str(receipts)}
    _STATE["markdown"], _STATE["empty"] = BODY, ("ED100002",)
    manifest = run_family.run_family("eric_official_pdf", out_dir=tmp_path / "run",
                                     today="2026-10-30", skip_consent=True, options=opts,
                                     log=lambda m: None)
    assert manifest["terminals"]["units_quarantined"] == 1
    assert manifest["terminals"]["units_admitted"] == 1
    (bad,) = [u for u in manifest["units"] if u["terminal"] == "quarantined"]
    assert bad["unit"] == "ED100002" and "extract_failed" in bad["reason"]
    assert bad["records"] == 0
