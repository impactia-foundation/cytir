#!/usr/bin/env python3
"""Acquire one bounded resumable tranche of official ERIC full-text PDF responses.

Ported from the deleted `acquire_eric_pdf_preserved` script (task 1226 removed it as "no port";
task 2182 restores it). Same flags, receipts, RUN.json and rate defaults (`--rate` .5, `--workers` 1).
One change: the transport. The old script shelled out to `curl`; this one uses the package's
consent-guarded client (`cytir.sources.http.make_client`), so a declined host or an unreadable consent
registry refuses BEFORE any socket opens, redirects included. A refusal is NOT turned into a quarantine
receipt: it ends the run. The old script had no per-path robots check and neither has this one; consent
is the host-level guard. The per-fetch transport file is now `aw.http-transport.v1` (key `http_metrics`);
nothing downstream reads it.

    python -m cytir.sources.eric.pdf_fetch --frontier-root D --root D --artifact-root D --offset N --limit N
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time

import httpx

from cytir.acquire import stage_receipt as SS
from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.consent.block import ConsentBlocked, ConsentRegistryError
from cytir.index.preservation_contract import _validate_stage
from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client

VERSION = "eric-official-pdf-acquisition.v2"
PRODUCER = "aw/sources/eric/pdf_fetch.py"


def _json(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode()


def _artifact(path: Path, artifact_root: Path) -> dict:
    return {"uri": f"artifact://{path.relative_to(artifact_root).as_posix()}",
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _publish(path: Path, value: bytes) -> None:
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"immutable ERIC PDF artifact collision: {path}")
    else:
        atomic_publish(path, value)


class StartRateGate:
    """One aggregate start-rate gate shared by every fetch worker."""

    def __init__(self, rate: float):
        self._interval = 1 / rate
        self._next_start = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delay = self._next_start - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next_start = time.monotonic() + self._interval


MAX_REDIRECTS = 50          # curl's default `--max-redirs`, which the old `curl --location` ran with
TOTAL_SECONDS = 600.0       # the old `curl --max-time 600`: a whole-transfer cap, per attempt, redirects included
_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})


class _Quarantine(Exception):
    """A row the fetcher refuses to fetch or to follow: it becomes a quarantine receipt with this reason."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason


def _pdf_host() -> str:
    from cytir.sources.families.eric_official_pdf import PDF_HOST    # lazy: the family imports this package
    return PDF_HOST


def _client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """The consent-guarded client: IPv4 only (the old `curl -4`), 20 s connect, identity.

    Redirects are NOT followed by the client: `_http_fetcher` follows them by hand, only on `PDF_HOST`.
    The whole-transfer 600 s cap is a wall-clock deadline in `_get_following_same_host` (httpx's timeout
    is per read)."""
    return make_client(
        HttpSpec(timeout=httpx.Timeout(600.0, connect=20.0), headers={"accept-encoding": "identity"},
                 follow_redirects=False),
        transport=transport or httpx.HTTPTransport(local_address="0.0.0.0"))


def _raw_headers(response: httpx.Response) -> bytes:
    """The status line and header block as bytes, the shape `curl --dump-header` wrote (final response only)."""
    lines = [f"{response.http_version} {response.status_code} {response.reason_phrase}"]
    lines += [f"{k.decode('latin-1')}: {v.decode('latin-1')}" for k, v in response.headers.raw]
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def _get_following_same_host(client: httpx.Client, url: str) -> tuple[httpx.Response, bytes, int]:
    """GET `url`; follow a redirect only while its target host is `PDF_HOST`. Whole transfer under TOTAL_SECONDS."""
    deadline = time.monotonic() + TOTAL_SECONDS
    host = _pdf_host()
    current, hops = url, 0
    while True:
        with client.stream("GET", current) as response:
            if response.status_code in _REDIRECT_STATUS and response.headers.get("location"):
                target = response.url.join(response.headers["location"])
                if (target.host or "").lower() != host:
                    raise _Quarantine("cross_host_redirect", f"{current} -> {target}")
                hops += 1
                if hops > MAX_REDIRECTS:
                    raise _Quarantine("network_failure", f"more than {MAX_REDIRECTS} redirects from {url}")
                current = str(target)
                continue
            chunks = []
            for chunk in response.iter_bytes():
                chunks.append(chunk)
                if time.monotonic() > deadline:
                    raise httpx.ReadTimeout(f"transfer exceeded {TOTAL_SECONDS:g} s", request=response.request)
            return response, b"".join(chunks), hops


def _http_fetcher(client: httpx.Client):
    def fetch(url: str, temporary: Path, rate_wait) -> tuple[bytes, bytes, dict]:
        failures = []
        for attempt in range(1, 3):
            rate_wait()
            fetched_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            try:
                response, body, hops = _get_following_same_host(client, url)
            except httpx.HTTPError as exc:      # a consent refusal is not an HTTPError: it propagates
                failures.append({"attempt": attempt, "attempted_at": fetched_at, "error": repr(exc)[-2000:]})
                continue
            metrics = {"url_effective": str(response.url), "http_code": response.status_code,
                       "size_download": len(body), "num_redirects": hops}
            return body, _raw_headers(response), {
                "schema": "aw.http-transport.v1", "requested_url": url,
                "final_url": str(response.url), "fetched_at": fetched_at,
                "http_status": response.status_code, "attempt": attempt,
                "http_metrics": metrics, "prior_transport_failures": failures,
                "request_headers": [["accept-encoding", "identity"]]}
        raise _Quarantine("network_failure", f"ERIC PDF fetch exhausted two bounded attempts: {failures}")
    return fetch


def _validate_receipt(receipt: dict, *, row: dict, artifact_root: Path) -> None:
    if (receipt.get("schema") != "aw.eric-official-pdf-receipt.v1"
            or receipt.get("source_url") != row["pdf_url"]
            or receipt.get("eric_id") != row["eric_id"]
            or receipt.get("status") not in {"complete", "quarantine"}):
        raise ValueError("existing ERIC PDF receipt identity/status drift")
    for item in receipt.get("artifacts", {}).values():
        uri = str(item.get("uri") or "")
        if not uri.startswith("artifact://"):
            raise ValueError("existing ERIC PDF receipt has a non-portable artifact")
        path = artifact_root / uri.removeprefix("artifact://")
        if (not path.is_file() or path.stat().st_size != item.get("bytes")
                or sha256_file(path) != item.get("sha256")):
            raise ValueError("existing ERIC PDF artifact mismatch")


def _acquire_one(*, row: dict, root: Path, artifact_root: Path, rate_wait, fetcher) -> dict:
    eric_id = row["eric_id"]
    url = row["pdf_url"]
    receipt_path = root / "receipts" / f"{eric_id}.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        _validate_receipt(receipt, row=row, artifact_root=artifact_root)
        return receipt
    try:
        if (httpx.URL(url).host or "").lower() != _pdf_host():
            raise _Quarantine("frontier_host_mismatch", f"{url} is not on {_pdf_host()}")
        with tempfile.TemporaryDirectory(prefix="aw-eric-pdf-") as directory:
            body, headers, transport = fetcher(url, Path(directory), rate_wait)
        status = int(transport.get("http_status") or 0)
        admitted = (status == 200 and body.startswith(b"%PDF-")
                    and b"application/pdf" in headers.lower())
        family = "original/pdf" if admitted else "quarantine/http-body"
        body_path = root / family / eric_id[:4] / f"{eric_id}.bin"
        header_path = root / "transport" / eric_id[:4] / f"{eric_id}.headers"
        transport["body"] = {
            "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()
        }
        transport["response_headers"] = {
            "bytes": len(headers), "sha256": hashlib.sha256(headers).hexdigest()
        }
        transport_path = root / "transport" / eric_id[:4] / f"{eric_id}.json"
        _publish(body_path, body)
        _publish(header_path, headers)
        _publish(transport_path, _json(transport))
        artifacts = {
            "response_body": _artifact(body_path, artifact_root),
            "response_headers": _artifact(header_path, artifact_root),
            "transport": _artifact(transport_path, artifact_root),
        }
        receipt = {
            "schema": "aw.eric-official-pdf-receipt.v1", "version": VERSION,
            "status": "complete" if admitted else "quarantine",
            "eric_id": eric_id, "source_url": url, "http_status": status,
            "artifacts": artifacts,
            "quarantine_reason": (None if admitted
                                  else "non_200_or_non_pdf_response"),
        }
    except (ConsentBlocked, ConsentRegistryError):
        raise                               # a consent refusal ends the run; no quarantine receipt
    except (_Quarantine, httpx.HTTPError) as exc:   # OSError (disk full, permissions) propagates: stop the run
        reason = exc.reason if isinstance(exc, _Quarantine) else "network_failure"
        failure_path = root / "quarantine/failure" / eric_id[:4] / f"{eric_id}.json"
        _publish(failure_path, _json({
            "eric_id": eric_id, "source_url": url,
            "failed_at": datetime.now(timezone.utc).isoformat(), "error": repr(exc),
        }))
        receipt = {
            "schema": "aw.eric-official-pdf-receipt.v1", "version": VERSION,
            "status": "quarantine", "eric_id": eric_id, "source_url": url,
            "artifacts": {"failure": _artifact(failure_path, artifact_root)},
            "quarantine_reason": reason,
        }
    _publish(receipt_path,
             (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    return receipt


def _migrate_predecessor(*, selected: list[dict], predecessor_root: Path,
                         root: Path, artifact_root: Path) -> tuple[int, Path]:
    rows = []
    for row in selected:
        source = predecessor_root / "receipts" / f"{row['eric_id']}.json"
        if not source.exists():
            continue
        receipt = json.loads(source.read_text())
        _validate_receipt(receipt, row=row, artifact_root=artifact_root)
        target = root / "receipts" / source.name
        _publish(target, source.read_bytes())
        rows.append({
            "eric_id": row["eric_id"],
            "source_receipt": _artifact(source, artifact_root),
            "migrated_receipt": _artifact(target, artifact_root),
        })
    migration_path = root / "migration/predecessor-receipts.jsonl"
    _publish(migration_path, b"".join(_json(row) for row in rows))
    return len(rows), migration_path


def acquire(*, frontier_root: Path, root: Path, artifact_root: Path,
            offset: int, limit: int, rate: float, workers: int = 1,
            predecessor_root: Path | None = None, fetcher=None) -> dict:
    """Fetch one tranche. `fetcher(url, tmpdir, rate_wait) -> (body, headers, transport)` is the test seam;
    None builds the consent-guarded client."""
    if (offset < 0 or not 0 < limit <= 10000 or not 0 < rate <= 1
            or not 0 < workers <= 8):
        raise ValueError("ERIC PDF tranche bounds/rate are invalid")
    frontier_run_path = frontier_root / "RUN.json"
    frontier_run = json.loads(frontier_run_path.read_text())
    if frontier_run.get("status") != "complete":
        raise ValueError("ERIC PDF frontier is not terminal")
    _validate_stage(frontier_run["stages"]["frontier"], "eric.pdf-frontier",
                    artifact_root=artifact_root, deep=True)
    selected = []
    frontier_count = 0
    with gzip.open(frontier_root / "frontier.jsonl.gz", "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if offset <= index < offset + limit:
                selected.append(json.loads(line))
            frontier_count = index + 1
    expected_frontier = int(frontier_run["counts"]["official_pdf_urls"])
    if frontier_count != expected_frontier:
        raise ValueError("ERIC PDF frontier row count disagrees with terminal RUN")
    if len(selected) != min(limit, max(0, frontier_count - offset)) or not selected:
        raise ValueError("ERIC PDF tranche selection is empty/non-conservative")
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.jsonl"
    _publish(plan_path, b"".join(_json(row) for row in selected))
    migration_count = 0
    migration_path = None
    if predecessor_root is not None:
        migration_count, migration_path = _migrate_predecessor(
            selected=selected, predecessor_root=predecessor_root,
            root=root, artifact_root=artifact_root,
        )
    gate = StartRateGate(rate)
    receipts = []
    client = None
    if fetcher is None:
        client = _client()
        fetcher = _http_fetcher(client)
    try:
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = [pool.submit(
                _acquire_one, row=row, root=root, artifact_root=artifact_root,
                rate_wait=gate.wait, fetcher=fetcher,
            ) for row in selected]
            for index, future in enumerate(as_completed(futures), 1):
                receipts.append(future.result())
                if index % 25 == 0:
                    print(f"PROGRESS terminal={index}/{len(selected)}", flush=True)
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)   # a refusal ends the run now, not after the queue
            raise
        pool.shutdown(wait=True)
    finally:
        if client is not None:
            client.close()
    complete = sum(row["status"] == "complete" for row in receipts)
    quarantine = len(receipts) - complete
    artifact_paths = [plan_path, *sorted((root / "receipts").glob("*.json"))]
    if migration_path is not None:
        artifact_paths.append(migration_path)
    for receipt in receipts:
        artifact_paths.extend(artifact_root / item["uri"].removeprefix("artifact://")
                              for item in receipt["artifacts"].values())
    artifact_paths = sorted(set(artifact_paths))
    logical = hashlib.sha256(b"".join(_json({"uri": str(path.relative_to(artifact_root)),
        "bytes": path.stat().st_size, "sha256": sha256_file(path)}) for path in artifact_paths)).hexdigest()
    stage = SS.seal(artifact_root=artifact_root, artifact_paths=artifact_paths,
        receipt_path=root / "stages/original/receipt.json", records=complete,
        fmt="exact-pdf-or-terminal-http-response+raw-headers+transport+per-url-receipt",
        logical_sha256=logical, producer_id=PRODUCER, producer_version=VERSION)
    run = {"schema": "aw.eric-official-pdf-acquisition-run.v1", "version": VERSION,
        "status": "complete", "frontier_run_sha256": sha256_file(frontier_run_path),
        "tranche": {"offset": offset, "limit": limit},
        "counts": {"frontier": frontier_count, "tranche": len(selected),
                   "pdf_complete": complete, "quarantine": quarantine,
                   "migrated_terminal": migration_count},
        "worker_policy": {"workers": workers, "aggregate_start_rate_per_second": rate},
        "content_scope": "exact source PDF responses; document count is pdf_complete only; no Markdown/point claim",
        "stages": {"original": stage},
        "downstream": "source-specific full PDF text to Markdown plus parent-bound outlinks; chunks only after fidelity/profile PASS"}
    _publish(root / "RUN.json", (json.dumps(run, indent=2, sort_keys=True) + "\n").encode())
    return run


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog or "pdf_fetch", description=__doc__)
    parser.add_argument("--frontier-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--predecessor-root", type=Path)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--rate", type=float, default=.5)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    print(json.dumps(acquire(frontier_root=args.frontier_root.resolve(), root=args.root.resolve(),
        artifact_root=args.artifact_root.resolve(), offset=args.offset, limit=args.limit,
        rate=args.rate, workers=args.workers,
        predecessor_root=(args.predecessor_root.resolve()
                          if args.predecessor_root else None)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
