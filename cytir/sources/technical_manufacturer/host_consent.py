"""Preserve one current consent decision per host in an exact URL frontier."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
import cytir.consent.resolver as CR
from cytir.acquire import preserved_stage_io as PS
from cytir.acquire import stage_receipt as SS


VERSION = "frontier-host-consent.v1"
PRODUCER = "aw/sources/technical_manufacturer/host_consent.py"


def _json(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode()


def acquire(*, frontier: Path, frontier_receipt: Path, root: Path,
            artifact_root: Path, rate: float = 1, workers: int = 4,
            resolver=None) -> dict:
    if not 0 < rate <= (100 if resolver is not None else 1) or not 1 <= workers <= 4:
        raise ValueError("host-consent rate/workers invalid")
    seal = json.loads(frontier_receipt.read_text())
    declared = ((seal.get("result") or {}).get("frontier") or {})
    if (seal.get("schema") != "aw.exact-url-frontier-receipt.v1"
            or declared.get("sha256") != sha256_file(frontier)):
        raise ValueError("host-consent frontier is not sealed")
    rows = [json.loads(line) for line in frontier.open() if line.strip()]
    if declared.get("urls") != len(rows):
        raise ValueError("host-consent frontier conservation failed")
    hosts = sorted({(urlsplit(row["url"]).hostname or "").casefold()
                    for row in rows})
    if not hosts or "" in hosts:
        raise ValueError("host-consent frontier has an invalid host")
    root.mkdir(parents=True, exist_ok=True)
    relative = root.relative_to(artifact_root).as_posix()
    client = None if resolver is not None else CR._client()
    if resolver is None:
        resolver = lambda host: CR.resolve_cached_host(
            client, host, cache_dir=str(root / "resolver-cache"), direct_crawl=True)
    lock = threading.Lock(); next_start = [0.0]
    def gate():
        with lock:
            delay = next_start[0] - time.monotonic()
            if delay > 0: time.sleep(delay)
            next_start[0] = time.monotonic() + 1 / rate
    def one(host: str) -> dict:
        path = root / "receipts" / f"{hashlib.sha256(host.encode()).hexdigest()}.json"
        if path.exists():
            value = json.loads(path.read_text())
            if value.get("host") != host or value.get("version") != VERSION:
                raise ValueError("existing host-consent receipt identity drift")
            PS.validate_artifact(artifact_root, value["consent_artifact"])
            return value
        gate(); consent = resolver(host)
        artifact = PS.store_artifact(
            artifact_root, f"{relative}/consent", "json", _json(consent))
        affirmative = consent.get("status") == "ok" and consent.get("search_ok") is True
        value = {"schema": "aw.frontier-host-consent-receipt.v1",
                 "version": VERSION, "status": "terminal", "host": host,
                 "affirmative_search_consent": affirmative,
                 "consent_status": consent.get("status"),
                 "search_ok": consent.get("search_ok"),
                 "consent_artifact": artifact}
        atomic_publish(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        return value
    values = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, host) for host in hosts]
            for future in as_completed(futures): values.append(future.result())
    finally:
        if client is not None: client.close()
    values.sort(key=lambda row: row["host"])
    paths = [frontier, frontier_receipt, *sorted((root / "receipts").glob("*.json"))]
    paths += [PS.artifact_path(artifact_root, row["consent_artifact"]) for row in values]
    paths = sorted(set(paths))
    logical = hashlib.sha256(b"".join(_json({
        "uri": "artifact://" + path.relative_to(artifact_root).as_posix(),
        "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        for path in paths)).hexdigest()
    stage = SS.seal(artifact_root=artifact_root, artifact_paths=paths,
        receipt_path=root / "stages/consent/receipt.json", records=len(values),
        fmt="per-host-current-live-consent+resolver-evidence",
        logical_sha256=logical, producer_id=PRODUCER, producer_version=VERSION)
    run = {"schema": "aw.frontier-host-consent-run.v1", "version": VERSION,
           "status": "complete", "frontier_sha256": sha256_file(frontier),
           "counts": {"hosts": len(values),
                      "affirmative_search_consent": sum(
                          row["affirmative_search_consent"] for row in values),
                      "not_affirmative": sum(
                          not row["affirmative_search_consent"] for row in values)},
           "stages": {"consent": stage}}
    encoded = (json.dumps(run, indent=2, sort_keys=True) + "\n").encode()
    run_path = root / "RUN.json"
    if run_path.exists():
        if run_path.read_bytes() != encoded: raise ValueError("host-consent RUN collision")
    else: atomic_publish(run_path, encoded)
    return run


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(prog=prog or "cyt source tech-product consent", description=__doc__)
    for name in ("frontier", "frontier-receipt", "root", "artifact-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=1)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    print(json.dumps(acquire(
        frontier=args.frontier.resolve(), frontier_receipt=args.frontier_receipt.resolve(),
        root=args.root.resolve(), artifact_root=args.artifact_root.resolve(),
        rate=args.rate, workers=args.workers), indent=2, sort_keys=True))
