"""`named_slice:pubmed` on the family Protocol (task 1404).

No network. The fixture is a hand-written three-article MEDLINE XML (one plain, one structured with
links and a secondary abstract, one below the length floor). Pins: registered and conformant;
discover/acquire/extract on the sealed file; a `plan` dry run through the framework; consent and
licence read from the row; and PARITY with the pre-move driver, whose output on the fixture was
digested before the move (driver blob faf64a302592251e39ddcea7748460a817530111).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib

import pytest

from cytir.extract import pubmed as P
from cytir.sources import contract as contract_mod
from cytir.sources import registry
from cytir.sources.family import RunContext, SourceFamily, UnitExcluded
from cytir.sources.families.pubmed import PubmedFamily
from cytir.sources.run_family import run_family

FIXTURE = pathlib.Path(__file__).resolve().parent / "data" / "pubmed" / "pubmed_sample.xml"
#: sha256 of json.dumps(rows, sort_keys=True, ensure_ascii=False) from the PRE-MOVE driver
#: (blob faf64a30...) over FIXTURE with source_archive {"member": "pubmed_sample.xml.gz", "sha256": "a"*64}
PRE_MOVE_DIGEST = "ca1bb63236f620e43bf61202e7eb614af5240778059d871a02d3e97abffe4666"
ARCHIVE = {"member": "pubmed_sample.xml.gz", "sha256": "a" * 64}


def _row() -> dict:
    return contract_mod.by_id(contract_mod.load())["named_slice:pubmed"]


def _ctx(root, tmp_path, contract=None) -> RunContext:
    return RunContext(out_dir=tmp_path / "out", today="2026-10-30", skip_consent=True,
                      options={"root": str(root)}, contract=contract or _row(), log=lambda m: None)


@pytest.fixture
def root(tmp_path):
    dst = tmp_path / "baseline"
    dst.mkdir()
    (dst / "pubmed26n0001.xml.gz").write_bytes(gzip.compress(FIXTURE.read_bytes(), mtime=0))
    return dst


def test_registered_and_conformant():
    assert "named_slice:pubmed" in registry.registered()
    adapter, row, spec = registry.admit("named_slice:pubmed")
    assert isinstance(adapter, SourceFamily)
    assert row["family_id"] == "named_slice:pubmed"
    assert not getattr(adapter, "PARTIAL", ())
    assert spec.declared == "chunk-policy-atomic-record.v1"


def test_parser_output_matches_the_pre_move_driver():
    rows = list(P.iter_pubmed_articles(FIXTURE.read_bytes(), source_archive=ARCHIVE))
    assert [r["source_record_id"] for r in rows] == ["1001", "1002"]   # 1003 is under the floor
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert digest == PRE_MOVE_DIGEST


def test_discover_acquire_extract_the_file(root, tmp_path):
    fam, ctx = PubmedFamily(), _ctx(root, tmp_path)
    (unit,) = fam.discover(ctx)
    assert unit.key == "pubmed26n0001.xml.gz" and unit.url.startswith("file:")   # no host contacted
    assert ctx.notes["files"] == 1
    (cap,) = fam.acquire(unit, ctx)
    assert cap.meta["sha256"] == hashlib.sha256(cap.body).hexdigest()
    docs = fam.extract(cap, ctx)
    assert [d.url for d in docs] == ["https://pubmed.ncbi.nlm.nih.gov/1001/",
                                     "https://pubmed.ncbi.nlm.nih.gov/1002/"]
    row = docs[1].extra["row"]
    assert row["parent_archive"] == {"member": unit.key, "sha256": cap.meta["sha256"]}
    assert {"https://doi.org/10.1234/sample", "https://pubmed.ncbi.nlm.nih.gov/789/"} <= {
        link["url"] for link in row["source_links"]}


def test_terms_come_from_the_contract_row(root, tmp_path):
    fam, ctx = PubmedFamily(), _ctx(root, tmp_path)
    fam.discover(ctx)
    t, row = ctx.notes["contract_terms"], _row()
    assert t["search_ok"] is row["consent"]["search_ok"]
    assert t["consent_basis"] == row["consent"]["basis"]
    assert t["licence_basis"] == row["licence"]["basis"]
    with pytest.raises(ValueError, match="contract row"):
        PubmedFamily().discover(_ctx(root, tmp_path, {"family_id": "named_slice:pubmed"}))


def test_a_file_with_no_admitted_abstract_is_excluded(tmp_path):
    d = tmp_path / "b"
    d.mkdir()
    xml = (b"<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>9</PMID><Article>"
           b"<ArticleTitle>T</ArticleTitle><Abstract><AbstractText>Short.</AbstractText></Abstract>"
           b"</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>")
    (d / "x.xml.gz").write_bytes(gzip.compress(xml))
    fam, ctx = PubmedFamily(), _ctx(d, tmp_path)
    (unit,) = fam.discover(ctx)
    (cap,) = fam.acquire(unit, ctx)
    with pytest.raises(UnitExcluded):
        fam.extract(cap, ctx)


def test_an_empty_walk_is_a_wrong_root(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(RuntimeError, match="no .xml.gz"):
        PubmedFamily().discover(_ctx(tmp_path / "empty", tmp_path))


def test_plan_dry_run_on_the_sealed_fixture(root, tmp_path):
    out = tmp_path / "plan"
    plan = run_family("named_slice:pubmed", out_dir=out, today="2026-10-30", dry_run=True,
                      skip_consent=True, options={"root": str(root)}, log=lambda m: None)
    assert plan is not None
    assert not (out / "pack").exists()


def test_atomic_policy_is_declared_and_a_run_composes_the_pack(root, tmp_path):
    # task 2260: the row declares chunk-policy-atomic-record.v1 (one record = one passage). A real
    # run on the committed fixture (the stage door's own entry point) must write a pack whose
    # records are the extracted records, one per passage.
    from cytir.sources import run_family as rf
    adapter, row, spec = registry.admit("named_slice:pubmed")
    assert row["chunk_policy"]["policy_id"] == "chunk-policy-atomic-record.v1"
    out = tmp_path / "o"
    run_family("named_slice:pubmed", out_dir=out, today="2026-10-30", skip_consent=True,
               options={"root": str(root)}, log=lambda m: None)
    _, recs = rf.load_pack(out / "pack")
    assert recs and all(r.get("url") and r.get("text") for r in recs)
    assert len({r["url"] for r in recs}) == len(recs)


def test_producer_constant_is_the_label_the_sealed_receipts_carry():
    assert P.PRODUCER.endswith("ingest_pubmed.py:iter_pubmed_articles")
