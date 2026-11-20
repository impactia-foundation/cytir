"""Plan and preserve exact-URL Common Crawl captures for a sealed frontier.

Moved from the pre-move driver by task 1410 (S7). The producer label `PRODUCER` is written into
sealed receipts, so it is frozen byte-for-byte (declared in `cytir.ops.public_cut.FROZEN_RECEIPT_LABELS`)."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import gzip
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable


from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client

from cytir.acquire.artifact_io import atomic_publish, canonical_json, open_text_read, sha256_file
from cytir.acquire import cc_exact_range as CC
from cytir.acquire import cc_digest as CC_DIGEST
from cytir.acquire import preserved_stage_io as PS
from cytir.acquire import stage_receipt as SS


VERSION = "exact-url-cc-preserved.v1"
#: frozen: the producer id carried by every receipt this driver has sealed
PRODUCER = "pipeline/acquire_exact_url_cc_preserved.py"
TEXTUAL_MIME_PREFIXES = ("text/html", "application/xhtml+xml", "text/plain",
                         "text/markdown", "application/xml", "text/xml",
                         "application/json", "text/json")


def _json(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode()


def load_frontier(path: Path, receipt_path: Path) -> tuple[list[dict], str]:
    receipt = json.loads(receipt_path.read_text())
    declared = ((receipt.get("result") or {}).get("frontier") or {})
    if (receipt.get("schema") != "aw.exact-url-frontier-receipt.v1" or
            declared.get("sha256") != sha256_file(path)):
        raise ValueError("frontier is not bound to an exact-URL subtraction receipt")
    with open_text_read(path) as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if declared.get("urls") != len(rows):
        raise ValueError("frontier row conservation failed")
    urls = [row.get("url") for row in rows]
    if any(not isinstance(url, str) for url in urls) or len(set(urls)) != len(rows):
        raise ValueError("frontier URLs are absent or duplicated")
    return rows, sha256_file(receipt_path)


def _validate_checkpoint(root: Path, checkpoint: dict, expected: dict) -> dict:
    if (checkpoint.get("schema") != "aw.exact-url-cc-index-checkpoint.v1" or
            checkpoint.get("identity") != expected or
            checkpoint.get("identity_sha256") !=
            hashlib.sha256(_json(expected)).hexdigest()):
        raise ValueError("exact-URL CC index checkpoint identity drift")
    result = checkpoint.get("result")
    if (not isinstance(result, dict) or set(result) != {"complete", "any"} or
            not all(isinstance(result[key], dict) for key in result)):
        raise ValueError("invalid exact-URL CC index checkpoint result")
    requested = {url for _key, url in expected["exact"]}
    for kind in ("complete", "any"):
        if not set(result[kind]).issubset(requested):
            raise ValueError("exact-URL CC checkpoint returned an unrequested URL")
        for url, record in result[kind].items():
            if (not isinstance(record, dict) or record.get("url") is None or
                    not all(key in record for key in (
                        "warc_filename", "offset", "length", "digest",
                        "fetch_time", "content_truncated", "truncation_reason"))):
                raise ValueError(f"invalid exact-URL CC {kind} capture for {url}")
    return result


def _incremental_resolve(*, root: Path, crawl: str, variant_urls: list[str],
                         allowed_mime_prefixes: tuple[str, ...],
                         workers: int = 1) -> tuple[dict, dict]:
    """Resolve exact URLs one physical index file at a time, durably.

    Each signed-S3 Parquet query publishes one immutable checkpoint.  A killed
    planner therefore resumes without rereading completed files and the final
    preservation stage can bind every query result, rather than exposing no
    evidence until one monolithic DuckDB query returns.
    """
    from cytir.acquire import cc_columnar as cc_columnar

    if not 1 <= workers <= 4:
        raise ValueError("exact-URL CC index workers must be between 1 and 4")
    work_identity = {
        "schema": "aw.exact-url-cc-index-work-identity.v1",
        "version": VERSION,
        "crawl": crawl,
        "variant_urls": variant_urls,
    }
    work_identity_sha = hashlib.sha256(_json(work_identity)).hexdigest()
    work_path = root / "plan/index-work.json"
    if work_path.exists():
        work_doc = json.loads(work_path.read_text())
        if (work_doc.get("schema") != "aw.exact-url-cc-index-work.v1"
                or work_doc.get("identity") != work_identity
                or work_doc.get("identity_sha256") != work_identity_sha
                or not isinstance(work_doc.get("work"), list)):
            raise ValueError("exact-URL CC index work identity drift")
        work = work_doc["work"]
        resumed_work = True
    else:
        topology_con = cc_columnar._con()
        try:
            work = cc_columnar.index_local_work(
                crawl, variant_urls, [], topology_con)
        finally:
            topology_con.close()
        work_doc = {
            "schema": "aw.exact-url-cc-index-work.v1",
            "version": VERSION,
            "identity": work_identity,
            "identity_sha256": work_identity_sha,
            "work": work,
        }
        atomic_publish(work_path, _json(work_doc))
        resumed_work = False
    print(f"INDEX_WORK physical_files={len(work)} workers={workers} "
          f"resumed={str(resumed_work).lower()}", flush=True)
    entries: list[dict] = []
    merged: dict[str, dict] = {}

    def resolve_item(item: dict) -> tuple[dict, dict, Path, bool]:
        identity = {
            "schema": "aw.exact-url-cc-index-checkpoint-identity.v1",
            "version": VERSION,
            "crawl": crawl,
            "file": item["file"],
            "exact": [list(pair) for pair in item["exact"]],
            "allowed_mime_prefixes": list(allowed_mime_prefixes),
        }
        identity_sha = hashlib.sha256(_json(identity)).hexdigest()
        checkpoint_path = root / "plan/checkpoints" / f"{identity_sha}.json"
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text())
            result = _validate_checkpoint(root, checkpoint, identity)
            return identity, result, checkpoint_path, True
        con = cc_columnar._con()
        try:
            result = cc_columnar.resolve_exact_file(
                con, item["file"], item["exact"],
                allowed_mime_prefixes=allowed_mime_prefixes)
        finally:
            con.close()
        checkpoint = {
            "schema": "aw.exact-url-cc-index-checkpoint.v1",
            "version": VERSION,
            "identity": identity,
            "identity_sha256": identity_sha,
            "result": result,
        }
        _validate_checkpoint(root, checkpoint, identity)
        atomic_publish(checkpoint_path, _json(checkpoint))
        print(f"INDEX_CHECKPOINT file_sha256={identity_sha} "
              f"complete={len(result['complete'])}", flush=True)
        return identity, result, checkpoint_path, False

    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="aw-cc-index") as executor:
        for ordinal, (item, resolved_item) in enumerate(
                zip(work, executor.map(resolve_item, work)), 1):
            identity, result, checkpoint_path, resumed = resolved_item
            identity_sha = hashlib.sha256(_json(identity)).hexdigest()
            for requested, capture in result["complete"].items():
                prior = merged.get(requested)
                if prior is not None and prior != capture:
                    raise ValueError("divergent exact-URL capture across index files")
                merged[requested] = capture
            entries.append({
                "path": checkpoint_path.relative_to(root).as_posix(),
                "sha256": sha256_file(checkpoint_path),
                "identity_sha256": identity_sha,
                "file": item["file"],
                "exact_keys": len(item["exact"]),
                "complete": len(result["complete"]),
                "any": len(result["any"]),
            })
            print(
                f"INDEX_PROGRESS files={ordinal}/{len(work)} "
                f"complete={len(merged)} resumed={str(resumed).lower()}",
                flush=True)
    manifest = {
        "schema": "aw.exact-url-cc-index-checkpoint-manifest.v1",
        "version": VERSION,
        "crawl": crawl,
        "allowed_mime_prefixes": list(allowed_mime_prefixes),
        "physical_files": len(entries),
        "variant_urls": len(variant_urls),
        "index_work": {
            "path": work_path.relative_to(root).as_posix(),
            "sha256": sha256_file(work_path),
            "identity_sha256": work_identity_sha,
        },
        "checkpoints": entries,
    }
    manifest_path = root / "plan/checkpoints.manifest.json"
    encoded = _json(manifest)
    if manifest_path.exists():
        if manifest_path.read_bytes() != encoded:
            raise ValueError("immutable exact-URL CC checkpoint manifest collision")
    else:
        atomic_publish(manifest_path, encoded)
    return merged, manifest


def _checkpoint_manifest_result(root: Path) -> dict | None:
    manifest_path = root / "plan/checkpoints.manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "aw.exact-url-cc-index-checkpoint-manifest.v1" or
            manifest.get("physical_files") != len(manifest.get("checkpoints") or [])):
        raise ValueError("invalid exact-URL CC checkpoint manifest")
    index_work = manifest.get("index_work") or {}
    index_work_path = root / str(index_work.get("path") or "")
    if (not index_work_path.is_file()
            or index_work_path.resolve() != (root / "plan/index-work.json").resolve()
            or sha256_file(index_work_path) != index_work.get("sha256")):
        raise ValueError("exact-URL CC index work artifact drift")
    for entry in manifest["checkpoints"]:
        path = root / entry["path"]
        if (not path.is_file() or path.resolve().parent !=
                (root / "plan/checkpoints").resolve() or
                sha256_file(path) != entry["sha256"]):
            raise ValueError("exact-URL CC checkpoint manifest artifact drift")
    return {"path": manifest_path.relative_to(root).as_posix(),
            "sha256": sha256_file(manifest_path),
            "index_work": index_work,
            "physical_files": manifest["physical_files"],
            "checkpoints": len(manifest["checkpoints"])}


def plan(*, frontier_path: Path, frontier_receipt_path: Path, root: Path,
         crawl: str, content_class: str = "html",
         resolver: Callable | None = None,
         index_workers: int = 1, exact_url_only: bool = False) -> dict:
    if content_class not in {"html", "textual-source"}:
        raise ValueError("unsupported exact-URL CC content class")
    rows, frontier_receipt_sha = load_frontier(frontier_path, frontier_receipt_path)
    plan_path, identity_path = root / "plan/captures.jsonl", root / "plan/identity.json"
    identity = {"schema": "aw.exact-url-cc-plan-identity.v1", "version": VERSION,
                "crawl": crawl, "frontier_urls": len(rows),
                "frontier_sha256": sha256_file(frontier_path),
                "frontier_receipt_sha256": frontier_receipt_sha,
                "content_class": content_class,
                "selection": (("exact-url-newest-complete-html" if
                               content_class == "html" else
                               "exact-url-newest-complete-textual-source") if
                              exact_url_only else
                              ("exact-url-variants-newest-complete-html" if
                               content_class == "html" else
                               "exact-url-variants-newest-complete-textual-source"))}
    identity_sha = hashlib.sha256(_json(identity)).hexdigest()
    legacy_plan = False
    if plan_path.exists() or identity_path.exists():
        if not plan_path.is_file() or not identity_path.is_file():
            raise ValueError("partial immutable exact-URL CC plan")
        prior = json.loads(identity_path.read_text())
        expected = {**identity, "identity_sha256": identity_sha,
                    "plan_sha256": sha256_file(plan_path)}
        # Early v1 plans already recorded content_class in RECEIPT.json but
        # omitted the redundant identity field.  Preserve that hash-bound plan
        # on resume when (and only when) its selection is the identical HTML
        # policy; recomputing a 163k-URL columnar plan would fork evidence.
        legacy_identity = {key: value for key, value in identity.items()
                           if key != "content_class"}
        legacy_sha = hashlib.sha256(_json(legacy_identity)).hexdigest()
        legacy_expected = {**legacy_identity, "identity_sha256": legacy_sha,
                           "plan_sha256": sha256_file(plan_path)}
        if prior == legacy_expected and content_class == "html":
            identity_sha = legacy_sha
            legacy_plan = True
        elif prior != expected:
            raise ValueError("exact-URL CC plan identity drift")
        planned = [json.loads(line) for line in plan_path.open() if line.strip()]
    else:
        variants = {row["url"]: ([row["url"]] if exact_url_only else
                                 CC.exact_url_variants(row["url"]))
                    for row in rows}
        variant_urls = [url for values in variants.values() for url in values]
        if resolver is None:
            prefixes = (("text/html",) if content_class == "html"
                        else TEXTUAL_MIME_PREFIXES)
            resolved, _manifest = _incremental_resolve(
                root=root, crawl=crawl, variant_urls=variant_urls,
                allowed_mime_prefixes=prefixes, workers=index_workers)
        else:
            resolved = resolver(variant_urls)
        planned = []
        for row in rows:
            capture = CC.choose_exact(resolved, variants[row["url"]])
            planned.append({**row, "state": "CAPTURE" if capture else "MISS",
                            "capture": capture})
        body = b"".join(_json(row) for row in planned)
        atomic_publish(plan_path, body)
        atomic_publish(identity_path, (json.dumps({
            **identity, "identity_sha256": identity_sha,
            "plan_sha256": hashlib.sha256(body).hexdigest(),
        }, indent=2, sort_keys=True) + "\n").encode())
    if len(planned) != len(rows):
        raise ValueError("CC plan/frontier conservation failed")
    captures = sum(row["state"] == "CAPTURE" for row in planned)
    misses = len(planned) - captures
    result = {"schema": "aw.exact-url-cc-plan-receipt.v1", "version": VERSION,
              "status": "complete", "crawl": crawl,
              "identity_sha256": identity_sha,
              "frontier": {"path": str(frontier_path), "sha256": sha256_file(frontier_path),
                           "receipt_sha256": frontier_receipt_sha, "urls": len(rows)},
              "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path),
                       "captures": captures, "misses": misses},
              "conservation": {"passes": captures + misses == len(rows)}}
    if not legacy_plan:
        result["content_class"] = content_class
    checkpoint_manifest = _checkpoint_manifest_result(root)
    if checkpoint_manifest is not None:
        result["checkpoint_manifest"] = checkpoint_manifest
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    receipt_path = root / "plan/RECEIPT.json"
    if receipt_path.exists():
        if receipt_path.read_bytes() != encoded:
            raise ValueError("immutable exact-URL CC plan receipt collision")
    else:
        atomic_publish(receipt_path, encoded)
    return result


def _validate_terminal(artifact_root: Path, receipt: dict, url: str,
                       plan_sha: str) -> None:
    if (receipt.get("schema") != "aw.exact-url-cc-capture-receipt.v1" or
            receipt.get("source_url") != url or receipt.get("plan_sha256") != plan_sha or
            receipt.get("status") not in {"complete", "cc_miss"}):
        raise ValueError("invalid exact-URL CC terminal receipt")
    for artifact in (receipt.get("artifacts") or {}).values():
        PS.validate_artifact(artifact_root, artifact)
    if receipt.get("status") == "complete":
        transport_artifact = (receipt.get("artifacts") or {}).get("transport") or {}
        transport_path = PS.artifact_path(artifact_root, transport_artifact)
        transport = json.loads(gzip.decompress(transport_path.read_bytes()))
        if ((transport.get("response_body") or {}).get(
                "content_digest_matches_index") is not True):
            raise ValueError("terminal Common Crawl content digest did not match index")


validate_terminal = _validate_terminal


def _load_plan_partition(plan_path: Path, *, worker_index: int,
                         worker_count: int) -> tuple[list[dict], int, str]:
    """Hash the plan once while parsing only the rows assigned to this worker.

    The previous acquisition loop re-read and re-hashed the complete plan for
    every terminal receipt.  A 172 MiB plan therefore cost two full SHA-256
    passes per WARC body (Task 426, measured 2026-08-25).  Bind the immutable
    bytes once before network access and reuse that digest for every receipt.
    """
    digest = hashlib.sha256()
    selected: list[dict] = []
    row_count = 0
    with plan_path.open("rb") as source:
        for line in source:
            digest.update(line)
            if not line.strip():
                continue
            if row_count % worker_count == worker_index:
                selected.append(json.loads(line))
            row_count += 1
    return selected, row_count, digest.hexdigest()


def _bounded_results(executor: ThreadPoolExecutor, function: Callable,
                     rows: list[dict], *, concurrency: int):
    """Yield completed results with only two work windows in memory."""
    pending = set()
    iterator = iter(rows)

    def fill() -> None:
        while len(pending) < concurrency * 2:
            try:
                row = next(iterator)
            except StopIteration:
                return
            pending.add(executor.submit(function, row))

    fill()
    while pending:
        completed, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            yield future.result()
        fill()


def acquire(*, root: Path, artifact_root: Path, rate: float = 4.0,
            transport: str = "signed-s3",
            fetcher: Callable | None = None, parser: Callable | None = None,
            worker_index: int = 0, worker_count: int = 1,
            partition_only: bool = False,
            fetch_concurrency: int = 1) -> dict:
    if not 0 < rate <= 4:
        raise ValueError("CC rate must be >0 and <=4 requests/s")
    if worker_count < 1 or not 0 <= worker_index < worker_count:
        raise ValueError("invalid exact-URL CC worker partition")
    if partition_only != (worker_count > 1):
        raise ValueError("partition-only requires worker-count >1 and vice versa")
    if (not 1 <= fetch_concurrency <= 8
            or fetch_concurrency * worker_count > 32):
        raise ValueError(
            "signed-S3 fetch concurrency must be within 1..32 aggregate")
    if transport != "signed-s3" and fetch_concurrency != 1:
        raise ValueError("concurrent acquisition is restricted to signed S3")
    plan_receipt_path = root / "plan/RECEIPT.json"
    plan_receipt = json.loads(plan_receipt_path.read_text())
    if (plan_receipt.get("schema") != "aw.exact-url-cc-plan-receipt.v1" or
            plan_receipt.get("status") != "complete"):
        raise ValueError("terminal exact-URL CC plan is required")
    plan_path = root / "plan/captures.jsonl"
    rows, plan_rows, plan_sha = _load_plan_partition(
        plan_path, worker_index=worker_index, worker_count=worker_count)
    if plan_sha != (plan_receipt.get("plan") or {}).get("sha256"):
        raise ValueError("CC plan artifact drift")
    if plan_rows != (plan_receipt.get("frontier") or {}).get("urls"):
        raise ValueError("CC plan row conservation failed")
    client = make_client(HttpSpec(
        timeout=120, follow_redirects=True,
        headers={"user-agent": "authoritative-web/0.1 (+https://websear.ch)"}))
    transport_mode = transport
    fetcher = fetcher or (lambda capture: CC._default_fetch(
        client, capture, transport=transport_mode))
    parser = parser or CC._default_parse
    terminals = []
    captures = misses = terminal_count = 0

    def acquire_one(row: dict) -> dict:
        receipt_path = root / "receipts" / (
            hashlib.sha256(row["url"].encode()).hexdigest() + ".json")
        if receipt_path.exists():
            terminal = json.loads(receipt_path.read_text())
            _validate_terminal(artifact_root, terminal, row["url"], plan_sha)
            return terminal
        if row["state"] == "MISS":
            terminal = {"schema": "aw.exact-url-cc-capture-receipt.v1",
                        "version": VERSION, "status": "cc_miss",
                        "source_url": row["url"], "plan_sha256": plan_sha,
                        "artifacts": {}}
        else:
            raw_warc, transport_evidence = fetcher(row["capture"])
            http_headers, response_body, decoded_html = parser(raw_warc)
            if not decoded_html:
                raise ValueError("decoded archived HTML is empty")
            digest = str(row["capture"].get("digest") or "")
            observed = "sha1:" + base64.b32encode(
                hashlib.sha1(response_body).digest()).decode().rstrip("=")
            digest_matches = CC_DIGEST.matches_sha1(response_body, digest)
            if not digest_matches:
                raise ValueError("Common Crawl response body digest mismatch")
            transport_evidence["capture"] = row["capture"]
            transport_evidence["response_body"] = {
                "bytes": len(response_body),
                "sha256": hashlib.sha256(response_body).hexdigest(),
                "content_digest_observed": observed,
                "content_digest_matches_index": digest_matches}
            try:
                relative_root = root.relative_to(artifact_root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    "CC root must be below the global artifact root") from exc
            decoded_key = ("decoded_html" if
                           plan_receipt.get("content_class", "html") == "html"
                           else "decoded_source")
            decoded_suffix = "html" if decoded_key == "decoded_html" else "txt"
            artifacts = {
                "raw_warc": PS.store_artifact(
                    artifact_root, f"{relative_root}/original/warc",
                    "warc.gz", raw_warc),
                "http_headers": PS.store_artifact(
                    artifact_root, f"{relative_root}/original/http-headers",
                    "bin", http_headers),
                "response_body": PS.store_artifact(
                    artifact_root, f"{relative_root}/original/http-body",
                    "bin", response_body),
                decoded_key: PS.store_artifact(
                    artifact_root,
                    f"{relative_root}/original/{decoded_key.replace('_', '-')}",
                    decoded_suffix, decoded_html.encode()),
                "transport": PS.store_artifact(
                    artifact_root, f"{relative_root}/original/transport",
                    "json", _json(transport_evidence)),
            }
            terminal = {"schema": "aw.exact-url-cc-capture-receipt.v1",
                        "version": VERSION, "status": "complete",
                        "source_url": row["url"],
                        "capture_url": row["capture"]["url"],
                        "plan_sha256": plan_sha, "artifacts": artifacts}
            if transport_mode == "https-mirror":
                time.sleep(1 / rate)
        atomic_publish(
            receipt_path,
            (json.dumps(terminal, indent=2, sort_keys=True) + "\n").encode())
        _validate_terminal(artifact_root, terminal, row["url"], plan_sha)
        return terminal

    try:
        with ThreadPoolExecutor(
                max_workers=fetch_concurrency,
                thread_name_prefix=f"cc-fetch-{worker_index:02d}") as executor:
            for terminal in _bounded_results(
                    executor, acquire_one, rows,
                    concurrency=fetch_concurrency):
                terminal_count += 1
                if terminal["status"] == "complete":
                    captures += 1
                else:
                    misses += 1
                if not partition_only:
                    terminals.append(terminal)
                if terminal_count % 25 == 0 or terminal_count == len(rows):
                    print(f"PROGRESS terminal={terminal_count}/{len(rows)}",
                          flush=True)
    finally:
        client.close()
    if partition_only:
        worker_result = {
            "schema": "aw.exact-url-cc-worker-receipt.v1",
            "version": VERSION, "status": "complete",
            "worker_index": worker_index, "worker_count": worker_count,
            "fetch_concurrency": fetch_concurrency,
            "plan_sha256": plan_sha,
            "assigned": len(rows), "terminals": terminal_count,
            "cc_captures": captures, "cc_misses": misses,
        }
        worker_path = root / "workers" / f"worker-{worker_index:02d}-of-{worker_count:02d}.json"
        encoded = (json.dumps(worker_result, indent=2, sort_keys=True) + "\n").encode()
        if worker_path.exists():
            if worker_path.read_bytes() != encoded:
                raise ValueError("immutable exact-URL CC worker receipt collision")
        else:
            atomic_publish(worker_path, encoded)
        return worker_result
    artifact_paths = [root/"plan/captures.jsonl", root/"plan/identity.json",
                      root/"plan/RECEIPT.json"]
    manifest = _checkpoint_manifest_result(root)
    if manifest is not None:
        manifest_path = root / manifest["path"]
        artifact_paths.append(manifest_path)
        manifest_body = json.loads(manifest_path.read_text())
        artifact_paths.append(root / manifest_body["index_work"]["path"])
        artifact_paths.extend(root / row["path"]
                              for row in manifest_body["checkpoints"])
    for terminal in terminals:
        artifact_paths.append(root/"receipts"/(hashlib.sha256(terminal["source_url"].encode()).hexdigest()+".json"))
        artifact_paths.extend(PS.artifact_path(artifact_root, artifact)
                              for artifact in terminal["artifacts"].values())
    artifact_paths = sorted(set(artifact_paths))
    inventory = [{"uri": f"artifact://{p.relative_to(artifact_root).as_posix()}",
                  "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in artifact_paths]
    logical = hashlib.sha256(b"".join(_json(row) for row in inventory)).hexdigest()
    original = SS.seal(artifact_root=artifact_root, artifact_paths=artifact_paths,
        receipt_path=root/"stages/original/receipt.json", records=captures,
        fmt=("exact-cc-warc-range+embedded-http+body+decoded-html+transport"
             if plan_receipt.get("content_class", "html") == "html" else
             "exact-cc-warc-range+embedded-http+body+decoded-source+transport"),
        logical_sha256=logical, producer_id=PRODUCER, producer_version=VERSION)
    result = {"schema": "aw.exact-url-cc-preserved-run.v1", "version": VERSION,
              "status": "complete", "crawl": plan_receipt["crawl"],
              "counts": {"frontier": len(rows), "cc_captures": captures,
                         "cc_misses": misses}, "stages": {"original": original},
              "next": "direct_retrieval_for_exact_cc_misses_only_then_full_markdown_and_parent_bound_outlinks"}
    encoded = (json.dumps(result, indent=2, sort_keys=True)+"\n").encode()
    run_path = root / "RUN.json"
    if run_path.exists():
        if run_path.read_bytes() != encoded:
            raise ValueError("immutable exact-URL CC RUN collision")
    else:
        atomic_publish(run_path, encoded)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True,
                        help="global artifact:// resolution root containing --root")
    parser.add_argument("--mode", choices=("plan", "acquire", "all"), default="all")
    parser.add_argument("--frontier", type=Path)
    parser.add_argument("--frontier-receipt", type=Path)
    parser.add_argument("--crawl", default="CC-MAIN-2026-30")
    parser.add_argument("--content-class", choices=("html", "textual-source"),
                        default="html")
    parser.add_argument("--rate", type=float, default=4.0)
    parser.add_argument("--index-workers", type=int, default=4,
                        help="bounded signed-S3 Parquet planner workers (1-4)")
    parser.add_argument("--exact-url-only", action="store_true",
                        help="query only byte-exact frontier URLs; no variants")
    parser.add_argument("--transport", choices=("signed-s3", "https-mirror"),
                        default="signed-s3")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--partition-only", action="store_true")
    parser.add_argument("--fetch-concurrency", type=int, default=1,
                        help="bounded concurrent signed-S3 range fetches per worker (1-8)")
    args = parser.parse_args(); root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    if args.mode in {"plan", "all"}:
        if not args.frontier or not args.frontier_receipt:
            parser.error("--frontier and --frontier-receipt are required for planning")
        print(json.dumps(plan(frontier_path=args.frontier.resolve(),
                              frontier_receipt_path=args.frontier_receipt.resolve(),
                              root=root, crawl=args.crawl,
                              content_class=args.content_class,
                              index_workers=args.index_workers,
                              exact_url_only=args.exact_url_only), indent=2,
                         sort_keys=True), flush=True)
    if args.mode in {"acquire", "all"}:
        print(json.dumps(acquire(root=root, artifact_root=args.artifact_root.resolve(),
                                 rate=args.rate, transport=args.transport,
                                 worker_index=args.worker_index,
                                 worker_count=args.worker_count,
                                 partition_only=args.partition_only,
                                 fetch_concurrency=args.fetch_concurrency),
                         indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
