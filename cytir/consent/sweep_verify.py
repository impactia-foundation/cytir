"""Independently verify a complete Task 427 private consent sweep."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit


DISPOSITIONS = {
    "admitted", "admitted_search_only", "declined",
    "held_invalid_evidence", "held_unknown",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def tree_sha256(root: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = bytes.fromhex(sha256_file(path))
        digest.update(content)
    return len(paths), digest.hexdigest()


def validate_nested_receipt_identity(source: dict, result: dict) -> None:
    """Bind resolver evidence to the exact scheme/authority/host/mode key."""
    receipt = result.get("receipt")
    if not isinstance(receipt, dict):
        raise ValueError("consent result has no nested receipt object")
    raw_origin = str(receipt.get("origin") or "")
    try:
        parsed = urlsplit(raw_origin)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("nested consent receipt has invalid origin") from exc
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    authority = host if port is None else f"{host}:{port}"
    normalized_origin = f"{scheme}://{authority}"
    required = {
        "origin": (normalized_origin, source.get("origin")),
        "origin_scheme": (receipt.get("origin_scheme"), source.get("scheme")),
        # The resolver's historical receipt field `host` is the authority,
        # including a non-default port.  Do not silently compare hostname only.
        "authority-as-host": (receipt.get("host"), source.get("authority")),
        "hostname": (host, source.get("host")),
        "mode": (receipt.get("mode"), source.get("mode")),
    }
    optional = {"scheme": source.get("scheme"),
                "authority": source.get("authority")}
    mismatches = [name for name, (actual, expected) in required.items()
                  if actual != expected]
    mismatches += [name for name, expected in optional.items()
                   if name in receipt and receipt[name] != expected]
    if raw_origin != normalized_origin:
        mismatches.append("origin-canonical-form")
    if mismatches:
        raise ValueError("nested consent receipt identity drift: "
                         + ",".join(sorted(mismatches)))


def verify(*, frontier: Path, run_dir: Path, shared_cache: Path,
           expected_shared_files: int, expected_shared_tree_sha256: str) -> dict:
    run_path = run_dir / "RUN.json"
    run = json.loads(run_path.read_text())
    if run.get("status") != "complete":
        raise ValueError("consent RUN is not complete")
    result_path = Path(run["results"]["path"])
    if sha256_file(frontier) != run["frontier"]["sha256"]:
        raise ValueError("frontier hash differs from consent RUN")
    if sha256_file(result_path) != run["results"]["sha256"]:
        raise ValueError("result hash differs from consent RUN")
    frontier_rows = rows(frontier)
    result_rows = rows(result_path)
    source = {(row["origin"], row["mode"]): row for row in frontier_rows}
    if len(source) != len(frontier_rows):
        raise ValueError("frontier contains duplicate keys")
    result_keys = [(row["origin"], row["mode"]) for row in result_rows]
    if len(set(result_keys)) != len(result_keys) or set(result_keys) != set(source):
        raise ValueError("result keys are not a unique exact frontier match")
    fields = ("origin", "scheme", "authority", "host", "mode", "documents",
              "waves", "sample_urls")
    disposition_keys = Counter()
    disposition_documents = Counter()
    for row in result_rows:
        key = (row["origin"], row["mode"])
        if any(row.get(field) != source[key].get(field) for field in fields):
            raise ValueError(f"result changed a frontier field: {key}")
        validate_nested_receipt_identity(source[key], row)
        disposition = row.get("disposition")
        if disposition not in DISPOSITIONS:
            raise ValueError(f"unknown disposition: {disposition}")
        if disposition in {"admitted", "admitted_search_only"}:
            receipt = row.get("receipt") or {}
            if (receipt.get("status") != "ok"
                    or receipt.get("search_ok") is not True
                    or not receipt.get("checked_at")):
                raise ValueError(f"admitted row lacks positive evidence: {key}")
        disposition_keys[disposition] += 1
        disposition_documents[disposition] += row["documents"]
    if dict(sorted(disposition_keys.items())) != run["counts"]["origin_mode_keys"]:
        raise ValueError("RUN key disposition counts differ from re-derivation")
    if dict(sorted(disposition_documents.items())) != run["counts"]["documents"]:
        raise ValueError("RUN document disposition counts differ from re-derivation")
    private_files = sum(path.is_file() for path in (run_dir / "private-cache").rglob("*"))
    if private_files != len(result_rows):
        raise ValueError("private cache file count differs from exact result keys")
    shared_files, shared_tree_sha = tree_sha256(shared_cache)
    if (shared_files != expected_shared_files
            or shared_tree_sha != expected_shared_tree_sha256):
        raise ValueError("shared consent cache changed during private sweep")
    return {
        "schema": "aw.task427-consent-reresolution-verification.v1",
        "status": "PASS",
        "inputs": {
            "frontier": {"path": str(frontier), "sha256": sha256_file(frontier)},
            "run": {"path": str(run_path), "sha256": sha256_file(run_path)},
            "results": {"path": str(result_path), "sha256": sha256_file(result_path)},
        },
        "counts": {
            "frontier_keys": len(frontier_rows), "result_keys": len(result_rows),
            "documents": sum(row["documents"] for row in result_rows),
            "private_cache_files": private_files,
            "shared_cache_files": shared_files,
            "origin_mode_keys": dict(sorted(disposition_keys.items())),
            "documents_by_disposition": dict(sorted(disposition_documents.items())),
        },
        "shared_cache_tree_sha256": shared_tree_sha,
        "checks": {
            "exact_key_bijection": True, "frontier_fields_unchanged": True,
            "nested_receipt_identity_exact": True,
            "admitted_evidence_positive": True, "run_counts_rederived": True,
            "private_cache_exact_count": True, "shared_cache_isolated": True,
        },
        "safety": {"production_writes": False, "verdict_application": False},
    }


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument("--frontier", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--shared-cache", required=True, type=Path)
    parser.add_argument("--expected-shared-files", required=True, type=int)
    parser.add_argument("--expected-shared-tree-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError(f"refusing existing verification output: {args.out}")
    result = verify(
        frontier=args.frontier, run_dir=args.run_dir,
        shared_cache=args.shared_cache,
        expected_shared_files=args.expected_shared_files,
        expected_shared_tree_sha256=args.expected_shared_tree_sha256)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
