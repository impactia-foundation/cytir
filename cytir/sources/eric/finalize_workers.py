#!/usr/bin/env python3
"""Conserve disjoint ERIC Docling workers into one portable terminal stage."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from cytir.acquire.artifact_io import atomic_publish,canonical_json,sha256_file
from cytir.acquire.deterministic_gzip import deterministic_gzip_text
from cytir.sources.eric import materialize as M
from cytir.index.preservation_contract import _validate_stage
from cytir.acquire import stage_receipt as SS

VERSION="eric-pdf-partitioned-finalizer.v2-selected-original";PRODUCER="pipeline/finalize_eric_pdf_materialization_workers.py"

def finalize(*,acquisition_root:Path,root:Path,artifact_root:Path,limit:int,
             workers:int,shared_preflight:Path)->dict:
    if (root/"RUN.json").exists(): raise ValueError("ERIC aggregate already terminal")
    identity=json.loads((root/"IDENTITY.json").read_text());migration=json.loads((root/"MIGRATION-RECEIPT.json").read_text())
    if identity.get("limit")!=limit or identity.get("workers")!=workers or migration.get("identity")!=identity:
        raise ValueError("ERIC worker identity drift")
    arun_path=acquisition_root/"RUN.json";arun=json.loads(arun_path.read_text())
    sources=[p for p in sorted((acquisition_root/"receipts").glob("*.json")) if json.loads(p.read_text()).get("status")=="complete"][:limit]
    preflight_artifact=M._validate_shared_preflight(path=shared_preflight,
      acquisition_run_path=arun_path,acquisition_run=arun,
      source_receipts=sources,artifact_root=artifact_root)
    expected={json.loads(p.read_text())["eric_id"]:(i,p) for i,p in enumerate(sources)}
    terminals={};worker_extraction=None
    for worker in range(workers):
        wr=root/f"workers/worker-{worker:02d}";run=json.loads((wr/"RUN.json").read_text())
        sel=run.get("selection") or {}
        if sel.get("partition_index")!=worker or sel.get("partition_count")!=workers or sel.get("full_selection_complete_pdfs")!=limit:
            raise ValueError("ERIC worker partition identity drift")
        extraction=run.get("extraction") or {}
        if (extraction.get("engine")!="aw.extract.pdf"
                or extraction.get("docling_do_ocr") is not False
                or extraction.get("docling_do_table_structure") is not True
                or extraction.get("content_cap") is not None):
            raise ValueError("ERIC worker extraction contract drift")
        if worker_extraction is None:worker_extraction=extraction
        elif extraction!=worker_extraction:raise ValueError("ERIC workers disagree on extraction configuration")
        for name in ("markdown","outlinks","diagnostics","quarantine"):_validate_stage(run["stages"][name],f"worker{worker}.{name}",artifact_root=artifact_root,deep=True)
        for path in sorted((wr/"receipts").glob("*.json")):
            rec=json.loads(path.read_text());eid=rec.get("eric_id")
            if eid in terminals or eid not in expected or expected[eid][0]%workers!=worker:raise ValueError("ERIC worker receipt overlap/partition drift")
            M._validate_terminal(rec,source_receipt=json.loads(expected[eid][1].read_text()),artifact_root=artifact_root);terminals[eid]=(path,rec)
    if set(terminals)!=set(expected):raise ValueError("ERIC worker union does not conserve exact selection")
    ordered=[terminals[json.loads(p.read_text())["eric_id"]] for p in sources]
    complete=[x for x in ordered if x[1]["status"]=="complete"];quarantine=[x for x in ordered if x[1]["status"]=="quarantine"]
    outputs={"markdown":root/"markdown/documents.jsonl.gz","outlinks":root/"outlinks/parent-bound.jsonl.gz","diagnostics":root/"diagnostics/pages.jsonl.gz","quarantine":root/"quarantine/records.jsonl.gz"}
    for path in outputs.values():path.parent.mkdir(parents=True,exist_ok=True)
    for name,key,rows in (("markdown","markdown_parent",complete),("outlinks","parent_bound_outlinks",complete),("diagnostics","page_diagnostics",ordered)):
        with deterministic_gzip_text(outputs[name]) as stream:
            for _,rec in rows:stream.write((artifact_root/rec["artifacts"][key]["uri"].removeprefix("artifact://")).read_text())
    with deterministic_gzip_text(outputs["quarantine"]) as stream:
        for _,rec in quarantine:stream.write(canonical_json(rec)+"\n")
    evidence=[root/"IDENTITY.json",root/"MIGRATION-RECEIPT.json",
              shared_preflight,*[p for p,_ in ordered]]
    def artifacts(key,rows):return [artifact_root/rec["artifacts"][key]["uri"].removeprefix("artifact://") for _,rec in rows]
    source_artifacts=[shared_preflight]
    for source_path in sources:
        source_artifacts.append(source_path)
        source_receipt=json.loads(source_path.read_text())
        for artifact in source_receipt["artifacts"].values():
            source_artifacts.append(
                artifact_root/artifact["uri"].removeprefix("artifact://"))
    stages={"original":SS.seal(artifact_root=artifact_root,
      artifact_paths=source_artifacts,
      receipt_path=root/"stages/original/receipt.json",records=limit,
      fmt="ERIC-exact-selected-PDF-response+headers+transport+source-receipt",
      logical_sha256=json.loads(shared_preflight.read_text())["selection_logical_sha256"],
      producer_id=PRODUCER,producer_version=VERSION)}
    for name,key,rows,fmt in (("markdown","markdown_parent",complete,"docling-full-markdown"),("outlinks","parent_bound_outlinks",complete,"parent-bound-outlinks"),("diagnostics","page_diagnostics",ordered,"page-diagnostics-including-quarantine")):
        stages[name]=SS.seal(artifact_root=artifact_root,artifact_paths=[outputs[name],*evidence,*artifacts(key,rows)],receipt_path=root/f"stages/{name}/receipt.json",records=len(rows),fmt=fmt,logical_sha256=sha256_file(outputs[name]),producer_id=PRODUCER,producer_version=VERSION)
    qarts=[artifact_root/ref["uri"].removeprefix("artifact://")
           for _,rec in quarantine for ref in rec["artifacts"].values()]
    stages["quarantine"]=SS.seal(artifact_root=artifact_root,artifact_paths=[outputs["quarantine"],*qarts],receipt_path=root/"stages/quarantine/receipt.json",records=len(quarantine),fmt="explicit-docling-quarantine",logical_sha256=sha256_file(outputs["quarantine"]),producer_id=PRODUCER,producer_version=VERSION)
    run={"schema":"aw.eric-pdf-materialization-run.v1","version":VERSION,"status":"ready_for_web_pdf_profile_fidelity_and_current_chunking","source_profile":"web_pdf","selection":{"limit":limit,"selected_complete_pdfs":limit,"partition_count":workers},"counts":{"original_documents":limit,"markdown_documents":len(complete),"docling_quarantine":len(quarantine),"outlink_parents_including_empty":len(complete)},"extraction":worker_extraction,"stages":stages,"shared_deep_preflight":preflight_artifact,"parent_acquisition_run_sha256":sha256_file(arun_path),"migration_receipt_sha256":sha256_file(root/"MIGRATION-RECEIPT.json")}
    atomic_publish(root/"RUN.json",(json.dumps(run,indent=2,sort_keys=True)+"\n").encode())
    for name in stages:_validate_stage(stages[name],f"eric.{name}",artifact_root=artifact_root,deep=True)
    return run

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ("acquisition-root","root","artifact-root"):p.add_argument(f"--{n}",type=Path,required=True)
 p.add_argument("--shared-preflight",type=Path,required=True)
 p.add_argument("--limit",type=int,required=True);p.add_argument("--workers",type=int,required=True);a=p.parse_args();print(json.dumps(finalize(acquisition_root=a.acquisition_root.resolve(),root=a.root.resolve(),artifact_root=a.artifact_root.resolve(),limit=a.limit,workers=a.workers,shared_preflight=a.shared_preflight.resolve()),indent=2,sort_keys=True))
if __name__=="__main__":main()
