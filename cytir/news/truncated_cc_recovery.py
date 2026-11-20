#!/usr/bin/env python3
"""Recover legacy-truncated news HTML from pinned Common Crawl WARC records."""
from __future__ import annotations
import argparse
import importlib
import base64
import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable


from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire import preserved_stage_io as PS
from cytir.index.preservation_contract import _validate_stage
from cytir.acquire import stage_receipt as SS
from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client


VERSION = "news-truncated-cc-recovery.v1"
PRODUCER = "pipeline/acquire_news_truncated_cc_recovery.py"
from cytir.acquire.cc_exact_range import (  # noqa: E402  moved by task 1410
    _default_fetch, _default_parse, choose_exact, exact_url_variants)


def _json(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode()


def _source_frontier(source_root: Path) -> list[dict]:
    rows = []
    for path in sorted((source_root / "raw").glob("*.raw.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                html = str(value.get("html") or "")
                stored_sha1 = hashlib.sha1(html.encode()).hexdigest()
                if stored_sha1 == value.get("sha1"):
                    continue
                rows.append({"url": value["url"], "final_url": value["final_url"],
                             "recorded_full_response_sha1": value["sha1"],
                             "stored_prefix_sha1": stored_sha1,
                             "stored_prefix_bytes": len(html.encode()),
                             "legacy_artifact": path.relative_to(source_root).as_posix(),
                             "legacy_line": line_no})
    rows.sort(key=lambda row: row["url"])
    return rows


def _plan(*, source_root: Path, root: Path, crawl: str,
          expected_frontier: int, resolver: Callable | None = None) -> tuple[list[dict], str]:
    plan_path = root / "plan/captures.jsonl"
    identity_path = root / "plan/identity.json"
    frontier = _source_frontier(source_root)
    if len(frontier) != expected_frontier:
        raise ValueError("legacy truncated-parent frontier count mismatch")
    identity = {"schema": "aw.news-truncated-cc-plan.v1", "version": VERSION,
                "crawl": crawl, "frontier_records": len(frontier),
                "source_checksum_manifest_sha256": sha256_file(
                    source_root / "FILE-SHA256SUMS"),
                "selection": "exact-url-variants-newest-complete-html"}
    identity_sha = hashlib.sha256(_json(identity)).hexdigest()
    if plan_path.exists() or identity_path.exists():
        if not plan_path.is_file() or not identity_path.is_file():
            raise ValueError("partial immutable CC recovery plan")
        prior = json.loads(identity_path.read_text())
        if prior != {**identity, "identity_sha256": identity_sha,
                     "plan_sha256": sha256_file(plan_path)}:
            raise ValueError("CC recovery plan identity drift")
        return [json.loads(line) for line in plan_path.open() if line.strip()], identity_sha
    variants = {row["url"]: list(dict.fromkeys(
        exact_url_variants(row["url"]) + exact_url_variants(row["final_url"])))
        for row in frontier}
    resolver = resolver or (lambda urls: importlib.import_module("cytir.acquire.cc_columnar").resolve(
        crawl, urls, require_complete=True))
    resolved = resolver([url for values in variants.values() for url in values])
    plan = []
    for row in frontier:
        capture = choose_exact(resolved, variants[row["url"]])
        plan.append({**row, "state": "CAPTURE" if capture else "MISS",
                     "capture": capture})
    plan_body = b"".join(_json(row) for row in plan)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_publish(plan_path, plan_body)
    atomic_publish(identity_path, (json.dumps({
        **identity, "identity_sha256": identity_sha,
        "plan_sha256": hashlib.sha256(plan_body).hexdigest(),
    }, indent=2, sort_keys=True) + "\n").encode())
    return plan, identity_sha


def _validate_receipt(root: Path, receipt: dict, url: str) -> None:
    if (receipt.get("schema") != "aw.news-truncated-cc-recovery-receipt.v1" or
            receipt.get("source_url") != url or
            receipt.get("status") not in {"complete", "cc_miss"}):
        raise ValueError("invalid CC recovery receipt identity")
    for artifact in (receipt.get("artifacts") or {}).values():
        PS.validate_artifact(root, artifact)


def acquire(*, source_root: Path, root: Path, crawl: str,
            expected_frontier: int, rate: float = 4.0,
            transport: str = "signed-s3",
            resolver: Callable | None = None,
            fetcher: Callable | None = None,
            parser: Callable | None = None) -> dict:
    if not 0 < rate <= 4:
        raise ValueError("CC recovery rate must be >0 and <=4 requests/s")
    root.mkdir(parents=True, exist_ok=True)
    plan, identity_sha = _plan(source_root=source_root, root=root, crawl=crawl,
                               expected_frontier=expected_frontier,
                               resolver=resolver)
    client = make_client(HttpSpec(follow_redirects=True, timeout=120))
    transport_mode = transport
    fetcher = fetcher or (lambda capture: _default_fetch(
        client, capture, transport=transport_mode))
    parser = parser or _default_parse
    receipts = []
    try:
        for index, row in enumerate(plan, 1):
            receipt_path = root / "receipts" / (
                hashlib.sha256(row["url"].encode()).hexdigest() + ".json")
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                _validate_receipt(root, receipt, row["url"])
                receipts.append(receipt)
                continue
            if row["state"] == "MISS":
                receipt = {"schema": "aw.news-truncated-cc-recovery-receipt.v1",
                           "version": VERSION, "status": "cc_miss",
                           "source_url": row["url"], "plan_identity_sha256": identity_sha,
                           "artifacts": {}}
            else:
                raw_warc, transport_evidence = fetcher(row["capture"])
                http_headers, response_body, decoded_html = parser(raw_warc)
                if not decoded_html:
                    raise ValueError("decoded archived HTML is empty")
                digest = str(row["capture"].get("digest") or "")
                actual_digest = "sha1:" + base64.b32encode(
                    hashlib.sha1(response_body).digest()).decode().rstrip("=")
                transport_evidence["capture"] = row["capture"]
                transport_evidence["response_body"] = {
                    "bytes": len(response_body),
                    "sha256": hashlib.sha256(response_body).hexdigest(),
                    "content_digest_observed": actual_digest,
                    "content_digest_matches_index": actual_digest.casefold() == digest.casefold(),
                }
                artifacts = {
                    "raw_warc": PS.store_artifact(root, "original/warc", "warc.gz", raw_warc),
                    "http_headers": PS.store_artifact(root, "original/http-headers", "bin", http_headers),
                    "response_body": PS.store_artifact(root, "original/http-body", "bin", response_body),
                    "decoded_html": PS.store_artifact(root, "original/decoded-html", "html", decoded_html.encode()),
                    "transport": PS.store_artifact(root, "original/transport", "json", _json(transport_evidence)),
                }
                receipt = {"schema": "aw.news-truncated-cc-recovery-receipt.v1",
                           "version": VERSION, "status": "complete",
                           "source_url": row["url"], "plan_identity_sha256": identity_sha,
                           "capture_url": row["capture"]["url"], "artifacts": artifacts}
                if transport_mode == "https-mirror":
                    time.sleep(1 / rate)
            atomic_publish(receipt_path, (json.dumps(
                receipt, indent=2, sort_keys=True) + "\n").encode())
            _validate_receipt(root, receipt, row["url"])
            receipts.append(receipt)
            if index % 25 == 0 or index == len(plan):
                print(f"PROGRESS terminal={index}/{len(plan)}", flush=True)
    finally:
        client.close()
    complete = sum(row["status"] == "complete" for row in receipts)
    misses = len(receipts) - complete
    original_paths = [root / "plan/captures.jsonl", root / "plan/identity.json"]
    for receipt in receipts:
        original_paths.append(root / "receipts" / (
            hashlib.sha256(receipt["source_url"].encode()).hexdigest() + ".json"))
        original_paths.extend(PS.artifact_path(root, artifact)
                              for artifact in receipt["artifacts"].values())
    original_paths = sorted(set(original_paths))
    inventory = [{"uri": f"artifact://{p.relative_to(root).as_posix()}",
                  "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                 for p in original_paths]
    logical = hashlib.sha256(b"".join(_json(row) for row in inventory)).hexdigest()
    original = SS.seal(
        artifact_root=root, artifact_paths=original_paths,
        receipt_path=root / "stages/original/receipt.json", records=complete,
        fmt="exact-cc-warc-range+embedded-http+body+decoded-html+transport",
        logical_sha256=logical, producer_id=PRODUCER, producer_version=VERSION)
    result = {"schema": "aw.news-truncated-cc-recovery-run.v1", "version": VERSION,
              "status": "complete", "crawl": crawl,
              "counts": {"frontier": len(plan), "cc_captures": complete,
                         "cc_misses": misses}, "stages": {"original": original},
              "next": "direct_retrieval_for_cc_misses_only" if misses else
                      "merge_with_exact_legacy_news_originals"}
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    run_path = root / "RUN.json"
    if run_path.exists():
        if sha256_file(run_path) != hashlib.sha256(encoded).hexdigest():
            raise ValueError("immutable CC recovery RUN collision")
    else:
        atomic_publish(run_path, encoded)
    _validate_stage(original, "original", artifact_root=root, deep=True)
    return result


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--crawl", default="CC-MAIN-2026-30")
    parser.add_argument("--expected-frontier", type=int, default=1499)
    parser.add_argument("--rate", type=float, default=4.0)
    parser.add_argument("--transport", choices=("signed-s3", "https-mirror"),
                        default="signed-s3")
    args = parser.parse_args(argv)
    result = acquire(source_root=args.source_root.resolve(), root=args.root.resolve(),
                     crawl=args.crawl, expected_frontier=args.expected_frontier,
                     rate=args.rate, transport=args.transport)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
