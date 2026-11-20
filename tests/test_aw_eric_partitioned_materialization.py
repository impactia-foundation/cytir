import json
from pathlib import Path

from cytir.sources.eric import finalize_workers as F
from cytir.ops import eric_partition_preflight as A
from cytir.sources.eric import materialize as M
from cytir.sources.eric import prepare_workers as P
import cytir.extract.pdf as EXTRACT


class Converter:
    """Stands in for `cytir.extract.pdf.extract_pdf` (task 656): same
    `(pdf_path) -> ExtractResult` signature the old `converter.convert()`
    duck-type used to provide."""

    def __init__(self, text="# full\nhttps://example.test/doc"):
        self.text = text
        self.calls = 0

    def __call__(self, _pdf_path):
        self.calls += 1
        return EXTRACT.ExtractResult(
            markdown=self.text, extractor="docling",
            extractor_version="test", route_reason="test fixture",
            page_count=1, is_complex=False, table_dense_runs=0)

class EmptyConverter(Converter):
    def __init__(self):
        super().__init__(text="")

def fixture(tmp_path:Path,count=6):
    a=tmp_path/"acquisition";(a/"receipts").mkdir(parents=True)
    for i in range(count):
        eid=f"ED{i:06d}";pdf=a/f"original/{eid}.pdf";pdf.parent.mkdir(exist_ok=True);pdf.write_bytes(b"%PDF"+bytes([i]))
        (a/"receipts"/f"{eid}.json").write_text(json.dumps({"schema":"aw.eric-official-pdf-receipt.v1","status":"complete","eric_id":eid,"source_url":f"https://files.eric/{eid}.pdf","artifacts":{"response_body":M._artifact(pdf,tmp_path)}}))
    (a/"RUN.json").write_text(json.dumps({"schema":"aw.eric-official-pdf-acquisition-run.v1","status":"complete","counts":{"pdf_complete":count},"stages":{"original":{}}}))
    return a

def test_migrate_partition_and_strict_union_finalizer(tmp_path:Path,monkeypatch):
    acquisition=fixture(tmp_path)
    monkeypatch.setattr(M,"_validate_stage",lambda *a,**k:None)
    monkeypatch.setattr(F,"_validate_stage",lambda *a,**k:None)
    old=tmp_path/"old";c0=Converter();M.materialize(acquisition_root=acquisition,root=old,artifact_root=tmp_path,limit=1,extract_fn=c0)
    worker_stage_validations=[]
    def validate_worker_stage(_stage,name,**_kwargs):
        assert name!="eric.pdf-original", "workers must use shared L0 preflight"
        worker_stage_validations.append(name)
    monkeypatch.setattr(M,"_validate_stage",validate_worker_stage)
    (old/"RUN.json").unlink();outcome=tmp_path/"OUTCOME.json";outcome.write_text(json.dumps({"rc":143}))
    root=tmp_path/"v2";migration=P.prepare(acquisition_root=acquisition,old_root=old,new_root=root,artifact_root=tmp_path,limit=6,workers=3,old_outcome=outcome)
    assert migration["identity"]["migrated_terminal"]==1
    preflight=root/"PREFLIGHT.json"
    witness=A.audit(acquisition_root=acquisition,artifact_root=tmp_path,output=preflight,limit=6)
    assert witness["deep_verification"]["selected_response_bodies_rehashed"]==6
    calls=0
    for worker in range(3):
        converter=Converter();M.materialize(acquisition_root=acquisition,root=root/f"workers/worker-{worker:02d}",artifact_root=tmp_path,limit=6,partition_index=worker,partition_count=3,shared_preflight=preflight,extract_fn=converter);calls+=converter.calls
    assert calls==5
    run=F.finalize(acquisition_root=acquisition,root=root,artifact_root=tmp_path,limit=6,workers=3,shared_preflight=preflight)
    assert run["counts"]=={"original_documents":6,"markdown_documents":6,"docling_quarantine":0,"outlink_parents_including_empty":6}
    assert run["stages"]["original"]["records"]==6
    assert run["extraction"]=={"engine":"aw.extract.pdf","router_version":EXTRACT.VERSION,"docling_do_ocr":False,"docling_do_table_structure":True,"content_cap":None}
    assert run["selection"]=={"limit":6,"selected_complete_pdfs":6,"partition_count":3}
    assert (old/"SUPERSEDED.json").is_file()

def test_partition_arguments_fail_closed(tmp_path:Path,monkeypatch):
    acquisition=fixture(tmp_path,1);monkeypatch.setattr(M,"_validate_stage",lambda *a,**k:None)
    try:M.materialize(acquisition_root=acquisition,root=tmp_path/"bad",artifact_root=tmp_path,limit=1,partition_index=0,extract_fn=Converter())
    except ValueError as exc:assert "declared together" in str(exc)
    else:raise AssertionError("partial partition declaration must fail")

def test_fresh_partition_initializer_has_zero_migration_and_no_fake_supersession(
        tmp_path:Path):
    acquisition=fixture(tmp_path,3)
    old=tmp_path/"unused-old"
    root=tmp_path/"fresh-workers"
    migration=P.prepare(acquisition_root=acquisition,old_root=old,new_root=root,
      artifact_root=tmp_path,limit=3,workers=3,old_outcome=None,fresh=True)
    assert migration["status"]=="fresh_workers_initialized_no_migration"
    assert migration["identity"]["migrated_terminal"]==0
    assert migration["identity"]["old_outcome_sha256"] is None
    assert migration["receipt_pairs"]==[]
    assert not (old/"SUPERSEDED.json").exists()


def test_six_way_fresh_workers_conserve_one_exact_selection(
        tmp_path: Path, monkeypatch):
    acquisition = fixture(tmp_path, 7)
    monkeypatch.setattr(M, "_validate_stage", lambda *a, **k: None)
    monkeypatch.setattr(F, "_validate_stage", lambda *a, **k: None)
    root = tmp_path / "six-way"
    migration = P.prepare(
        acquisition_root=acquisition, old_root=tmp_path / "unused-old",
        new_root=root, artifact_root=tmp_path, limit=7, workers=6,
        old_outcome=None, fresh=True)
    assert migration["identity"]["workers"] == 6
    preflight = root / "PREFLIGHT.json"
    A.audit(acquisition_root=acquisition, artifact_root=tmp_path,
            output=preflight, limit=7)
    calls = 0
    for worker in range(6):
        converter = Converter()
        M.materialize(
            acquisition_root=acquisition,
            root=root / f"workers/worker-{worker:02d}",
            artifact_root=tmp_path, limit=7, partition_index=worker,
            partition_count=6, shared_preflight=preflight,
            extract_fn=converter)
        calls += converter.calls
    assert calls == 7
    run = F.finalize(
        acquisition_root=acquisition, root=root, artifact_root=tmp_path,
        limit=7, workers=6, shared_preflight=preflight)
    assert run["selection"]["partition_count"] == 6
    assert run["counts"] == {
        "original_documents": 7, "markdown_documents": 7,
        "docling_quarantine": 0, "outlink_parents_including_empty": 7}

def test_partition_rejects_unbound_or_missing_shared_preflight(tmp_path:Path):
    acquisition=fixture(tmp_path,1)
    try:M.materialize(acquisition_root=acquisition,root=tmp_path/"missing",artifact_root=tmp_path,limit=1,partition_index=0,partition_count=2,extract_fn=Converter())
    except ValueError as exc:assert "shared deep preflight" in str(exc)
    else:raise AssertionError("partition without shared preflight must fail")
    witness=tmp_path/"bad.json";witness.write_text(json.dumps({"schema":M.PREFLIGHT_SCHEMA,"status":"PASS","deep_verification":{"original_stage":True}}))
    try:M.materialize(acquisition_root=acquisition,root=tmp_path/"bad",artifact_root=tmp_path,limit=1,partition_index=0,partition_count=2,shared_preflight=witness,extract_fn=Converter())
    except ValueError as exc:assert "does not bind acquisition" in str(exc)
    else:raise AssertionError("unbound shared preflight must fail")

def test_finalizer_preserves_quarantine_page_diagnostics(tmp_path:Path,monkeypatch):
    acquisition=fixture(tmp_path,3)
    monkeypatch.setattr(M,"_validate_stage",lambda *a,**k:None)
    monkeypatch.setattr(F,"_validate_stage",lambda *a,**k:None)
    old=tmp_path/"old";old.mkdir();outcome=tmp_path/"OUTCOME.json"
    outcome.write_text(json.dumps({"rc":143}))
    root=tmp_path/"v2"
    P.prepare(acquisition_root=acquisition,old_root=old,new_root=root,
              artifact_root=tmp_path,limit=3,workers=3,old_outcome=outcome)
    preflight=root/"PREFLIGHT.json"
    A.audit(acquisition_root=acquisition,artifact_root=tmp_path,
            output=preflight,limit=3)
    for worker in range(3):
        converter=EmptyConverter() if worker==1 else Converter()
        M.materialize(acquisition_root=acquisition,
          root=root/f"workers/worker-{worker:02d}",artifact_root=tmp_path,
          limit=3,partition_index=worker,partition_count=3,
          shared_preflight=preflight,extract_fn=converter)
    run=F.finalize(acquisition_root=acquisition,root=root,
      artifact_root=tmp_path,limit=3,workers=3,shared_preflight=preflight)
    assert run["counts"]["markdown_documents"]==2
    assert run["counts"]["docling_quarantine"]==1
    assert run["stages"]["diagnostics"]["records"]==3
    quarantine_receipt=json.loads(
        (root/"stages/quarantine/receipt.json").read_text())
    uris={row["uri"] for row in quarantine_receipt["artifacts"]}
    assert any("/diagnostics/" in uri for uri in uris)
    assert any("/quarantine/" in uri for uri in uris)
