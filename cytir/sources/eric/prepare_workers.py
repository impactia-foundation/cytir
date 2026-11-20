#!/usr/bin/env python3
"""Initialize disjoint ERIC Docling workers, optionally migrating receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cytir.acquire.artifact_io import atomic_publish, sha256_file
from cytir.sources.eric import materialize as M

VERSION = "eric-pdf-materialization-worker-migration.v1"


def prepare(*, acquisition_root: Path, old_root: Path, new_root: Path,
            artifact_root: Path, limit: int, workers: int,
            old_outcome: Path | None, fresh: bool = False) -> dict:
    if new_root.exists(): raise ValueError("fresh ERIC v2 root already exists")
    if fresh:
        if old_outcome is not None:
            raise ValueError("fresh ERIC initialization cannot bind an old outcome")
    else:
        if old_outcome is None:
            raise ValueError("migration requires an old writer outcome")
        if (old_root / "RUN.json").exists(): raise ValueError("old ERIC root is terminal")
        outcome=json.loads(old_outcome.read_text())
        if outcome.get("rc") in {None, 0}: raise ValueError("old ERIC writer is not quiesced")
    acquisition_run=json.loads((acquisition_root/"RUN.json").read_text())
    if acquisition_run.get("status")!="complete": raise ValueError("ERIC acquisition not terminal")
    sources=[]
    for path in sorted((acquisition_root/"receipts").glob("*.json")):
        if json.loads(path.read_text()).get("status")=="complete": sources.append(path)
    sources=sources[:limit]
    if len(sources)!=limit or workers<2: raise ValueError("invalid ERIC worker selection")
    by_id={json.loads(path.read_text())["eric_id"]:(ordinal,path)
           for ordinal,path in enumerate(sources)}
    new_root.mkdir(parents=True)
    pairs=[]
    old_receipts = [] if fresh else sorted((old_root/"receipts").glob("*.json"))
    for old_path in old_receipts:
        receipt=json.loads(old_path.read_text()); eric_id=receipt.get("eric_id")
        if eric_id not in by_id: raise ValueError("old ERIC receipt outside exact selection")
        ordinal,source_path=by_id[eric_id]
        M._validate_terminal(receipt,source_receipt=json.loads(source_path.read_text()),
                             artifact_root=artifact_root)
        worker=ordinal%workers
        new_path=new_root/f"workers/worker-{worker:02d}/receipts"/old_path.name
        atomic_publish(new_path,old_path.read_bytes())
        pairs.append({"eric_id":eric_id,"ordinal":ordinal,"worker":worker,
          "old":{"uri":"artifact://"+old_path.relative_to(artifact_root).as_posix(),
                 "bytes":old_path.stat().st_size,"sha256":sha256_file(old_path)},
          "new":{"uri":"artifact://"+new_path.relative_to(artifact_root).as_posix(),
                 "bytes":new_path.stat().st_size,"sha256":sha256_file(new_path)}})
    identity={"schema":"aw.eric-pdf-partitioned-materialization-identity.v1",
      "version":VERSION,"status":"workers_ready","limit":limit,"workers":workers,
      "acquisition_run_sha256":sha256_file(acquisition_root/"RUN.json"),
      "old_outcome_sha256":None if fresh else sha256_file(old_outcome),
      "migrated_terminal":len(pairs),
      "selection_logical_sha256":hashlib.sha256(b"".join(
        (json.loads(path.read_text())["eric_id"]+"\n").encode() for path in sources)).hexdigest()}
    atomic_publish(new_root/"IDENTITY.json",(json.dumps(identity,indent=2,sort_keys=True)+"\n").encode())
    migration={"schema":"aw.eric-pdf-materialization-migration.v1","version":VERSION,
      "status":("fresh_workers_initialized_no_migration" if fresh
                else "terminal_receipts_migrated_no_reprocessing"),"identity":identity,
      "receipt_pairs":pairs}
    migration_path=new_root/"MIGRATION-RECEIPT.json"
    atomic_publish(migration_path,(json.dumps(migration,indent=2,sort_keys=True)+"\n").encode())
    if not fresh:
        superseded={"schema":"aw.eric-pdf-partial-supersession.v1","version":VERSION,
          "status":"superseded_partial_writer","replacement_root":new_root.relative_to(artifact_root).as_posix(),
          "migration_receipt":{"uri":"artifact://"+migration_path.relative_to(artifact_root).as_posix(),
           "bytes":migration_path.stat().st_size,"sha256":sha256_file(migration_path)}}
        atomic_publish(old_root/"SUPERSEDED.json",(json.dumps(superseded,indent=2,sort_keys=True)+"\n").encode())
    return migration

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("acquisition-root","old-root","new-root","artifact-root"):
        p.add_argument(f"--{n}",type=Path,required=True)
    p.add_argument("--old-outcome",type=Path)
    p.add_argument("--fresh",action="store_true")
    p.add_argument("--limit",type=int,required=True);p.add_argument("--workers",type=int,required=True)
    a=p.parse_args();print(json.dumps(prepare(acquisition_root=a.acquisition_root.resolve(),
      old_root=a.old_root.resolve(),new_root=a.new_root.resolve(),artifact_root=a.artifact_root.resolve(),
      limit=a.limit,workers=a.workers,
      old_outcome=None if a.old_outcome is None else a.old_outcome.resolve(),
      fresh=a.fresh),indent=2,sort_keys=True))
if __name__=="__main__":main()
