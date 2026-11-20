"""Direct-fetch only terminal exact-URL Common Crawl misses on Heimdall.

Moved from the pre-move driver by task 1410 (S7). The producer label `PRODUCER` is written into
sealed receipts, so it is frozen byte-for-byte (declared in `cytir.ops.public_cut.FROZEN_RECEIPT_LABELS`)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time
import threading
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx

from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire import preserved_stage_io as PS
from cytir.index.preservation_contract import _validate_stage
from cytir.acquire import stage_receipt as SS
from cytir.core.user_agent import user_agent  # task 432: one home for the crawler identity

VERSION="exact-url-direct-preserved.v4-consent-gated"
#: frozen: the producer id carried by every receipt this driver has sealed
PRODUCER="pipeline/acquire_exact_url_direct_preserved.py"
TEXTUAL_SOURCE_TYPES=("text/html","application/xhtml+xml","application/xml",
  "text/xml","application/json","text/json","text/plain","text/markdown")


def _json(value: Any) -> bytes: return (canonical_json(value)+"\n").encode()


def _frontier(cc_root: Path, artifact_root: Path) -> list[dict]:
    portable=cc_root/"RUN.portable.v2.json"
    run_path=portable if portable.exists() else cc_root/"RUN.json"; run=json.loads(run_path.read_text())
    if run.get("schema")!="aw.exact-url-cc-preserved-run.v1" or run.get("status")!="complete":
        raise ValueError("direct retrieval requires terminal exact-URL CC RUN")
    original=(run.get("stages") or {}).get("original")
    _validate_stage(original,"cc.original",artifact_root=artifact_root,deep=False)
    receipt_ref=original.get("receipt") or {}
    receipt_uri=str(receipt_ref.get("uri") or "")
    if not receipt_uri.startswith("artifact://"):
        raise ValueError("CC original stage lacks portable receipt")
    receipt_path=artifact_root/receipt_uri.removeprefix("artifact://")
    if (not receipt_path.is_file() or receipt_path.stat().st_size!=receipt_ref.get("bytes")
            or sha256_file(receipt_path)!=receipt_ref.get("sha256")):
        raise ValueError("CC original stage receipt binding failed")
    inventory=json.loads(receipt_path.read_text())
    if (inventory.get("schema")!="aw.stage-receipt.v1" or
            any(inventory.get(key)!=original.get(key) for key in
                ("records","bytes","format","logical_sha256","deterministic_encoding","producer"))):
        raise ValueError("CC original stage inventory differs from RUN")
    receipt_prefix=("artifact://"+cc_root.relative_to(artifact_root).as_posix()+"/receipts/")
    receipt_inventory={ref["uri"]:ref for ref in inventory.get("artifacts") or []
                       if str(ref.get("uri") or "").startswith(receipt_prefix)}
    plan_path=cc_root/"plan/captures.jsonl";plan_receipt_path=cc_root/"plan/RECEIPT.json"
    plan_receipt=json.loads(plan_receipt_path.read_text())
    plan=(plan_receipt.get("plan") or {})
    if (plan_receipt.get("schema")!="aw.exact-url-cc-plan-receipt.v1" or
            plan_receipt.get("status")!="complete" or
            plan.get("sha256")!=sha256_file(plan_path) or
            plan.get("misses")!=(run.get("counts") or {}).get("cc_misses")):
        raise ValueError("CC exact plan binding failed")
    rows=[]; captures=0
    with plan_path.open(encoding="utf-8") as source:
        for line in source:
            row=json.loads(line);state=row.get("state");url=row.get("url")
            if state=="CAPTURE":captures+=1;continue
            if state!="MISS" or not url:raise ValueError("invalid CC exact plan row")
            uri=receipt_prefix+hashlib.sha256(url.encode()).hexdigest()+".json"
            ref=receipt_inventory.get(uri)
            if ref is None:raise ValueError("CC miss lacks receipt inventory binding")
            rows.append({"source_url":url,"cc_receipt_sha256":ref["sha256"]})
    rows.sort(key=lambda row:row["source_url"])
    if (len(rows)!=(run.get("counts") or {}).get("cc_misses") or
            captures!=(run.get("counts") or {}).get("cc_captures")):
        raise ValueError("CC miss frontier conservation failed")
    return rows


miss_frontier = _frontier


def _fetch(client:httpx.Client,url:str)->tuple[bytes,dict,list[list[str]]]:
    last=None
    for attempt in range(1,3):
        fetched=datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
        try:
            response=client.get(url);body=bytes(response.content)
            headers=[[str(k),str(v)] for k,v in response.headers.multi_items()]
            return body,{"schema":"aw.direct-http-transport.v1","requested_url":url,
              "final_url":str(response.url),"retrieved_at":fetched,
              "http_status":response.status_code,"response_headers":headers,
              "request_headers":[["accept-encoding","identity"]],"attempt":attempt,
              "response_body":{"bytes":len(body),"sha256":hashlib.sha256(body).hexdigest()}},headers
        except httpx.HTTPError as exc:
            last=exc
            if attempt<2:time.sleep(1)
    raise RuntimeError(f"direct fetch exhausted retries: {last}")


def _validate(artifact_root:Path,receipt:dict,row:dict)->None:
    if (receipt.get("schema")!="aw.exact-url-direct-receipt.v1" or
            receipt.get("version")!=VERSION or
            receipt.get("source_url")!=row["source_url"] or
            receipt.get("cc_receipt_sha256")!=row["cc_receipt_sha256"] or
            receipt.get("status") not in {"complete","quarantine"}):
        raise ValueError("invalid exact-URL direct terminal receipt")
    validate_consent_gated_receipt(receipt, artifact_root)
    for artifact in (receipt.get("artifacts") or {}).values():PS.validate_artifact(artifact_root,artifact)


def validate_consent_gated_receipt(receipt:dict,artifact_root:Path)->dict:
    if (receipt.get("schema")!="aw.exact-url-direct-receipt.v1" or
            receipt.get("version")!=VERSION or
            receipt.get("status") not in {"complete","quarantine"}):
        raise ValueError("direct receipt is not current consent-gated evidence")
    consent_artifact=(receipt.get("artifacts") or {}).get("consent")
    if not isinstance(consent_artifact,dict):
        raise ValueError("exact-URL direct receipt lacks preserved consent evidence")
    PS.validate_artifact(artifact_root,consent_artifact)
    consent=json.loads(gzip.decompress(
      PS.artifact_path(artifact_root,consent_artifact).read_bytes()))
    if receipt.get("status")=="complete" and not (
            consent.get("status")=="ok" and consent.get("search_ok") is True):
        raise ValueError("complete direct receipt lacks affirmative search consent")
    return consent


def validate_consent_gated_run(root:Path,artifact_root:Path,*,deep:bool=True)->dict:
    run_path=root/"RUN.json";run=json.loads(run_path.read_text())
    if (run.get("schema")!="aw.exact-url-direct-preserved-run.v1" or
            run.get("version")!=VERSION or run.get("status")!="complete"):
        raise ValueError("direct RUN is not current consent-gated evidence")
    _validate_stage(run["stages"]["original"],"direct.original",
      artifact_root=artifact_root,deep=deep)
    _validate_stage(run["stages"]["quarantine"],"direct.quarantine",
      artifact_root=artifact_root,deep=deep)
    paths=sorted((root/"receipts").glob("*.json"))
    if len(paths)!=int(run["counts"]["frontier"]):
        raise ValueError("consent-gated direct receipt conservation failed")
    statuses={"complete":0,"quarantine":0}
    for path in paths:
        receipt=json.loads(path.read_text());validate_consent_gated_receipt(
          receipt,artifact_root);statuses[receipt["status"]]+=1
    if (statuses["complete"]!=int(run["counts"]["complete"]) or
            statuses["quarantine"]!=int(run["counts"]["quarantine"])):
        raise ValueError("consent-gated direct RUN status conservation failed")
    return run


validate_receipt = _validate


def _is_admissible_source_body(*,status:int,content_type:str,body:bytes)->bool:
    media_type=content_type.split(";",1)[0].strip().casefold()
    return 200<=status<300 and bool(body) and media_type in TEXTUAL_SOURCE_TYPES


def acquire(*,cc_root:Path,root:Path,artifact_root:Path,rate:float=1.0,workers:int=4,
            fetcher:Callable|None=None,consent_getter:Callable|None=None)->dict:
    if not 0<rate<=(100 if fetcher is not None else 1):
        raise ValueError("direct aggregate start rate must be >0 and <=1")
    if not 1<=workers<=4:raise ValueError("direct workers must be 1..4")
    frontier=_frontier(cc_root,artifact_root);root.mkdir(parents=True,exist_ok=True)
    relative_root=root.relative_to(artifact_root).as_posix()
    plan=root/"plan/direct-cc-misses.jsonl";data=b"".join(_json(row) for row in frontier)
    if plan.exists() and sha256_file(plan)!=hashlib.sha256(data).hexdigest():
        raise ValueError("immutable direct frontier drift")
    if not plan.exists():atomic_publish(plan,data)
    client=make_client(HttpSpec(follow_redirects=True,timeout=httpx.Timeout(30,connect=10),headers={
      "user-agent":user_agent(),
      "Accept-Encoding":"identity"}))
    # lazy: the resolver REFUSES at import without a crawler identity (see cytir/sources/runner.py)
    import cytir.consent.resolver as CR
    consent_client=None if consent_getter is not None else CR._client()
    consent_cache=root/"consent-cache"
    if consent_getter is None:
        consent_getter=lambda host: CR.resolve_cached_host(
            consent_client,host,cache_dir=str(consent_cache),direct_crawl=True)
    fetcher=fetcher or (lambda url:_fetch(client,url));receipts=[]
    gate_lock=threading.Lock(); next_start=[0.0]; host_locks:dict[str,threading.Lock]={}
    host_locks_lock=threading.Lock()
    def gate()->None:
        with gate_lock:
            now=time.monotonic();delay=max(0,next_start[0]-now)
            if delay:time.sleep(delay)
            next_start[0]=time.monotonic()+1/rate
    def process(row:dict)->dict:
        path=root/"receipts"/(hashlib.sha256(row["source_url"].encode()).hexdigest()+".json")
        if path.exists():
            receipt=json.loads(path.read_text());_validate(artifact_root,receipt,row);return receipt
        host=(urlsplit(row["source_url"]).hostname or "").casefold()
        with host_locks_lock:host_lock=host_locks.setdefault(host,threading.Lock())
        consent_artifact=None
        try:
            with host_lock:
                consent=consent_getter(host)
                consent_artifact=PS.store_artifact(
                    artifact_root,f"{relative_root}/consent","json",_json(consent))
                if consent.get("status")!="ok" or consent.get("search_ok") is not True:
                    classification=("consent_unknown_retryable" if
                      consent.get("status")=="unknown" else "consent_declined")
                    receipt={"schema":"aw.exact-url-direct-receipt.v1","version":VERSION,
                      "status":"quarantine","source_url":row["source_url"],
                      "cc_receipt_sha256":row["cc_receipt_sha256"],
                      "content_classification":classification,
                      "artifacts":{"consent":consent_artifact},
                      "quarantine_reason":classification}
                    atomic_publish(path,(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode())
                    _validate(artifact_root,receipt,row)
                    return receipt
                if not CR.direct_url_allowed(consent, row["source_url"]):
                    receipt={"schema":"aw.exact-url-direct-receipt.v1","version":VERSION,
                      "status":"quarantine","source_url":row["source_url"],
                      "cc_receipt_sha256":row["cc_receipt_sha256"],
                      "content_classification":"robots_path_declined",
                      "artifacts":{"consent":consent_artifact},
                      "quarantine_reason":"robots_path_declined"}
                    atomic_publish(path,(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode())
                    _validate(artifact_root,receipt,row)
                    return receipt
                gate();body,transport,response_headers=fetcher(row["source_url"])
            status=int(transport.get("http_status") or 0)
            content_type=next((v for k,v in response_headers if k.casefold()=="content-type"),"")
            admitted=_is_admissible_source_body(status=status,
              content_type=content_type,body=body)
            artifacts={"response_body":PS.store_artifact(artifact_root,f"{relative_root}/original/http-body","bin",body),
              "response_headers":PS.store_artifact(artifact_root,f"{relative_root}/original/response-headers","json",_json(response_headers)),
              "transport":PS.store_artifact(artifact_root,f"{relative_root}/original/transport","json",_json(transport)),
              "consent":consent_artifact}
            receipt={"schema":"aw.exact-url-direct-receipt.v1","version":VERSION,
              "status":"complete" if admitted else "quarantine","source_url":row["source_url"],
              "cc_receipt_sha256":row["cc_receipt_sha256"],"final_url":transport.get("final_url"),
              "content_classification":"current_or_newer_origin_textual_source" if admitted else "non_textual_empty_or_non_2xx",
              "artifacts":artifacts,"quarantine_reason":None if admitted else
                f"status_content_type_bytes:{status}:{content_type}:{len(body)}"}
        except Exception as exc:
            if consent_artifact is None:
                consent_artifact=PS.store_artifact(artifact_root,
                  f"{relative_root}/consent","json",_json({
                    "schema":"aw.consent-resolution-attempt.v1","status":"unknown",
                    "search_ok":False,"host":host,"error":repr(exc),
                    "failed_at":datetime.now(timezone.utc).isoformat()}))
            failure={"source_url":row["source_url"],"failed_at":datetime.now(timezone.utc).isoformat(),
                     "error":repr(exc)}
            receipt={"schema":"aw.exact-url-direct-receipt.v1","version":VERSION,
              "status":"quarantine","source_url":row["source_url"],
              "cc_receipt_sha256":row["cc_receipt_sha256"],"content_classification":"network_failure",
              "artifacts":{"consent":consent_artifact,
                "failure":PS.store_artifact(artifact_root,f"{relative_root}/quarantine/failure","json",_json(failure))},
              "quarantine_reason":"network_failure"}
        atomic_publish(path,(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode())
        _validate(artifact_root,receipt,row);return receipt
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(process,row) for row in frontier]
            for index,future in enumerate(as_completed(futures),1):
                receipts.append(future.result())
                if index%25==0 or index==len(frontier):print(f"PROGRESS terminal={index}/{len(frontier)}",flush=True)
    finally:
        client.close()
        if consent_client is not None:
            consent_client.close()
    receipts.sort(key=lambda row:row["source_url"])
    complete=sum(row["status"]=="complete" for row in receipts)
    quarantined=[row for row in receipts if row["status"]=="quarantine"]
    quarantine_path=root/"quarantine/records.jsonl"
    quarantine_bytes=b"".join(_json(x) for x in quarantined)
    if quarantine_path.exists():
        if quarantine_path.read_bytes()!=quarantine_bytes:
            raise ValueError(f"quarantine artifact collision: {quarantine_path}")
    else:
        atomic_publish(quarantine_path,quarantine_bytes)
    paths=[plan,quarantine_path]+sorted((root/"receipts").glob("*.json"))
    for receipt in receipts:paths.extend(PS.artifact_path(artifact_root,a) for a in receipt["artifacts"].values())
    paths=sorted(set(paths));inventory=[{"uri":f"artifact://{p.relative_to(artifact_root).as_posix()}",
      "bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in paths]
    original=SS.seal(artifact_root=artifact_root,artifact_paths=paths,receipt_path=root/"stages/original/receipt.json",
      records=complete,fmt="full-direct-http-body+canonical-response-headers+transport",
      logical_sha256=hashlib.sha256(b"".join(_json(x) for x in inventory)).hexdigest(),
      producer_id=PRODUCER,producer_version=VERSION)
    quarantine=SS.seal(artifact_root=artifact_root,artifact_paths=[quarantine_path],
      receipt_path=root/"stages/quarantine/receipt.json",records=len(quarantined),fmt="jsonl",
      logical_sha256=hashlib.sha256(quarantine_path.read_bytes()).hexdigest(),
      producer_id=PRODUCER,producer_version=VERSION)
    result={"schema":"aw.exact-url-direct-preserved-run.v1","version":VERSION,"status":"complete",
      "counts":{"frontier":len(frontier),"complete":complete,"quarantine":len(quarantined)},
      "cc_run_sha256":sha256_file(cc_root/"RUN.json"),"stages":{"original":original,"quarantine":quarantine},
      "next":"source-specific full Markdown plus parent-bound contextual outlinks"}
    run_path=root/"RUN.json";run_bytes=(json.dumps(result,indent=2,sort_keys=True)+"\n").encode()
    if run_path.exists():
        if run_path.read_bytes()!=run_bytes:raise ValueError(f"RUN artifact collision: {run_path}")
    else:atomic_publish(run_path,run_bytes)
    return result


def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cc-root",type=Path,required=True)
    p.add_argument("--root",type=Path,required=True);p.add_argument("--artifact-root",type=Path,required=True)
    p.add_argument("--rate",type=float,default=1);p.add_argument("--workers",type=int,default=4)
    a=p.parse_args();print(json.dumps(acquire(cc_root=a.cc_root.resolve(),root=a.root.resolve(),
      artifact_root=a.artifact_root.resolve(),rate=a.rate,workers=a.workers),indent=2,sort_keys=True))


if __name__=="__main__":main()
