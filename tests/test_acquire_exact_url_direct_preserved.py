import hashlib
import json
from pathlib import Path
import threading
import time

import cytir.sources.exact_url.direct_preserved as A
from cytir.acquire import stage_receipt as SS


def _open_consent(host: str) -> dict:
    return {"host": host, "status": "ok", "search_ok": True,
            "train_ok": False, "mode": "direct",
            "checked_at": "2026-08-16T00:00:00Z", "_robots_body": "",
            "signals": {"robots": {"fetched": True, "http_status": 404}}}


def _cc_fixture(tmp_path:Path, urls=None)->Path:
    root=tmp_path/"cc";receipts=root/"receipts";receipts.mkdir(parents=True)
    paths=[]
    pairs=urls or (("https://www.canada.ca/en/a.html","cc_miss"),
                   ("https://www.canada.ca/en/b.html","complete"))
    for url,status in pairs:
        value={"schema":"aw.exact-url-cc-capture-receipt.v1","version":"v",
               "status":status,"source_url":url,"artifacts":{}}
        path=receipts/(hashlib.sha256(url.encode()).hexdigest()+".json")
        path.write_text(json.dumps(value));paths.append(path)
    plan=root/"plan/captures.jsonl";plan.parent.mkdir()
    plan.write_text("".join(json.dumps({"url":url,"state":(
        "MISS" if status=="cc_miss" else "CAPTURE")})+"\n" for url,status in pairs))
    (root/"plan/RECEIPT.json").write_text(json.dumps({
        "schema":"aw.exact-url-cc-plan-receipt.v1","status":"complete",
        "plan":{"sha256":hashlib.sha256(plan.read_bytes()).hexdigest(),
                "misses":sum(status=="cc_miss" for _,status in pairs)}}))
    stage=SS.seal(artifact_root=tmp_path,artifact_paths=paths,
        receipt_path=root/"stages/original/receipt.json",records=1,fmt="fixture",
        logical_sha256="a"*64,producer_id="fixture",producer_version="v1")
    (root/"RUN.json").write_text(json.dumps({"schema":"aw.exact-url-cc-preserved-run.v1",
      "status":"complete","counts":{"cc_misses":sum(status=="cc_miss" for _,status in pairs),
        "cc_captures":sum(status!="cc_miss" for _,status in pairs)},
      "stages":{"original":stage}}))
    return root


def test_direct_frontier_is_only_terminal_cc_misses(tmp_path:Path):
    cc=_cc_fixture(tmp_path); root=tmp_path/"direct"
    result=A.acquire(cc_root=cc,root=root,artifact_root=tmp_path,rate=2,fetcher=lambda url:(
      b"<html><main>complete</main></html>",
      {"http_status":200,"final_url":url,"requested_url":url},
      [["content-type","text/html; charset=utf-8"]]),consent_getter=_open_consent)
    assert result["counts"]=={"frontier":1,"complete":1,"quarantine":0}
    row=json.loads(next((root/"receipts").glob("*.json")).read_text())
    assert row["source_url"].endswith("a.html") and row["status"]=="complete"
    assert set(row["artifacts"])=={"consent","response_body","response_headers","transport"}


def test_direct_requires_terminal_cc_run(tmp_path:Path):
    root=tmp_path/"cc";root.mkdir();(root/"RUN.json").write_text('{"status":"running"}')
    try:A._frontier(root,tmp_path)
    except ValueError as exc:assert "terminal" in str(exc)
    else:raise AssertionError("nonterminal CC run accepted")


def test_standard_catalog_textual_surfaces_are_admissible():
    for content_type in ("text/markdown", "text/plain; charset=utf-8",
                         "application/xml", "application/json"):
        assert A._is_admissible_source_body(
            status=200,content_type=content_type,body=b"source")
    assert not A._is_admissible_source_body(
        status=200,content_type="image/png",body=b"source")


def test_four_workers_keep_per_host_one_and_do_not_refetch_migrated_receipt(tmp_path:Path):
    urls=[("https://a.test/1","cc_miss"),("https://a.test/2","cc_miss"),
          ("https://b.test/1","cc_miss"),("https://c.test/1","cc_miss")]
    cc=_cc_fixture(tmp_path,urls);root=tmp_path/"direct";lock=threading.Lock()
    active={};max_host={};overall=[0,0];calls=[]
    def fetch(url):
        host=url.split("/")[2]
        with lock:
            calls.append(url);active[host]=active.get(host,0)+1;overall[0]+=1
            max_host[host]=max(max_host.get(host,0),active[host]);overall[1]=max(overall[1],overall[0])
        time.sleep(.05)
        with lock:active[host]-=1;overall[0]-=1
        return b"<html>ok</html>",{"http_status":200,"final_url":url},[["content-type","text/html"]]
    A.acquire(cc_root=cc,root=root,artifact_root=tmp_path,rate=100,workers=4,
              fetcher=fetch,consent_getter=_open_consent)
    assert max(max_host.values())==1 and overall[1]>=2 and len(calls)==4
    A.acquire(cc_root=cc,root=root,artifact_root=tmp_path,rate=100,workers=4,
      fetcher=lambda url:(_ for _ in ()).throw(AssertionError("terminal URL refetched")),
      consent_getter=_open_consent)


def test_consent_decline_is_preserved_and_never_fetches_origin(tmp_path:Path):
    cc=_cc_fixture(tmp_path);root=tmp_path/"direct"
    result=A.acquire(
        cc_root=cc,root=root,artifact_root=tmp_path,rate=2,
        fetcher=lambda _url:(_ for _ in ()).throw(
            AssertionError("declined origin fetched")),
        consent_getter=lambda host:{"host":host,"status":"ok",
            "search_ok":False,"train_ok":False,"mode":"direct",
            "signals":{"robots":{"wildcard_root":False}},
            "checked_at":"2026-08-16T00:00:00Z"})
    assert result["counts"]=={"frontier":1,"complete":0,"quarantine":1}
    receipt=json.loads(next((root/"receipts").glob("*.json")).read_text())
    assert receipt["content_classification"]=="consent_declined"
    assert set(receipt["artifacts"])=={"consent"}


def test_unknown_consent_is_explicit_retryable_quarantine(tmp_path:Path):
    cc=_cc_fixture(tmp_path);root=tmp_path/"direct"
    result=A.acquire(
        cc_root=cc,root=root,artifact_root=tmp_path,rate=2,
        fetcher=lambda _url:(_ for _ in ()).throw(
            AssertionError("unknown-consent origin fetched")),
        consent_getter=lambda host:{"host":host,"status":"unknown",
            "search_ok":True,"train_ok":True,"mode":"direct",
            "reasons":["policy endpoint unavailable"],"signals":{},
            "checked_at":"2026-08-16T00:00:00Z"})
    assert result["counts"]=={"frontier":1,"complete":0,"quarantine":1}
    receipt=json.loads(next((root/"receipts").glob("*.json")).read_text())
    assert receipt["quarantine_reason"]=="consent_unknown_retryable"


def test_path_specific_robots_decline_never_fetches_origin(tmp_path:Path):
    cc=_cc_fixture(tmp_path);root=tmp_path/"direct"
    result=A.acquire(
        cc_root=cc,root=root,artifact_root=tmp_path,rate=2,
        fetcher=lambda _url:(_ for _ in ()).throw(
            AssertionError("robots-disallowed path fetched")),
        consent_getter=lambda host:{"host":host,"status":"ok",
            "search_ok":True,"train_ok":False,"mode":"direct",
            "_robots_body":"User-agent: *\nDisallow: /en/a.html\n",
            "signals":{"robots":{"fetched":True,"http_status":200}},
            "checked_at":"2026-08-23T00:00:00Z"})
    assert result["counts"]=={"frontier":1,"complete":0,"quarantine":1}
    receipt=json.loads(next((root/"receipts").glob("*.json")).read_text())
    assert receipt["quarantine_reason"]=="robots_path_declined"


def test_v3_terminal_direct_root_is_not_current_consent_evidence(tmp_path:Path):
    root=tmp_path/"direct";receipts=root/"receipts";receipts.mkdir(parents=True)
    (receipts/"old.json").write_text(json.dumps({
      "schema":"aw.exact-url-direct-receipt.v1","version":"exact-url-direct-preserved.v3",
      "status":"complete","source_url":"https://maker.test/product","artifacts":{}}))
    (root/"RUN.json").write_text(json.dumps({
      "schema":"aw.exact-url-direct-preserved-run.v1","version":"exact-url-direct-preserved.v3",
      "status":"complete","counts":{"frontier":1,"complete":1,"quarantine":0},
      "stages":{"original":{},"quarantine":{}}}))
    try:A.validate_consent_gated_run(root,tmp_path)
    except ValueError as exc:assert "not current consent-gated" in str(exc)
    else:raise AssertionError("v3 terminal root was accepted")
