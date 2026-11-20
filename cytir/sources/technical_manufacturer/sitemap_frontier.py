"""Convert a sealed manufacturer catalog decision into an exact-URL frontier.

`cyt source tech-product frontier` is step 1 of the technical-product preservation order
(the retired chain script ran it; no unit did). The order, CC first and direct only after
a CC miss: `tech-product frontier` -> `python -m cytir.sources.exact_url.cc_preserved
--content-class textual-source` -> `python -m cytir.sources.exact_url.direct_preserved` ->
`cyt source tech-product pages` -> `cyt source tech-product consent` -> cc_preserved
(`--content-class html`) -> direct_preserved.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from cytir.acquire.artifact_io import atomic_publish, sha256_file


VERSION = "technical-product-sitemap-frontier.v1"
PRODUCER = "aw/sources/technical_manufacturer/sitemap_frontier.py"


def _artifact(root: Path, path: Path) -> dict:
    return {"uri": "artifact://" + path.resolve().relative_to(root.resolve()).as_posix(),
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def prepare(*, catalog_receipt: Path, root: Path,
            artifact_root: Path) -> dict:
    catalog = json.loads(catalog_receipt.read_text())
    if (catalog.get("schema") !=
            "aw.technical-manufacturer-catalog-frontier-receipt.v1"
            or catalog.get("status") != "complete"):
        raise ValueError("terminal manufacturer catalog frontier is required")
    source_artifact = (catalog.get("artifacts") or {}).get(
        "product_sitemap_frontier")
    if source_artifact is None:
        raise ValueError("catalog receipt lacks product sitemap frontier")
    uri = str(source_artifact.get("uri") or "")
    if not uri.startswith("artifact://"):
        raise ValueError("catalog product sitemap frontier is not portable")
    source = artifact_root / uri.removeprefix("artifact://")
    if (not source.is_file() or source.stat().st_size != source_artifact.get("bytes")
            or sha256_file(source) != source_artifact.get("sha256")):
        raise ValueError("catalog product sitemap frontier failed byte/hash validation")
    rows = [json.loads(line) for line in source.open() if line.strip()]
    urls = [row.get("url") for row in rows]
    if (any(not isinstance(url, str) or not url for url in urls)
            or len(set(urls)) != len(rows)):
        raise ValueError("product sitemap URLs are absent or duplicated")
    encoded = source.read_bytes()
    expected_frontier = {
        "uri": "artifact://" + (root / "frontier.jsonl").resolve().relative_to(
            artifact_root.resolve()).as_posix(),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "urls": len(rows),
    }
    if root.exists():
        receipt_path = root / "RECEIPT.json"
        frontier_path = root / "frontier.jsonl"
        if not receipt_path.is_file() or not frontier_path.is_file():
            raise ValueError("existing product-sitemap frontier is not terminal")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("schema") != "aw.exact-url-frontier-receipt.v1"
                or receipt.get("status") != "complete"
                or receipt.get("producer") != {"id": PRODUCER, "version": VERSION}
                or receipt.get("logical_sha256") != expected_frontier["sha256"]
                or (receipt.get("result") or {}).get("frontier") != expected_frontier
                or (receipt.get("source") or {}).get("catalog_receipt")
                != _artifact(artifact_root, catalog_receipt)
                or (receipt.get("source") or {}).get("catalog_frontier")
                != source_artifact
                or frontier_path.read_bytes() != encoded):
            raise ValueError("existing product-sitemap frontier identity drift")
        return receipt
    root.mkdir(parents=True)
    frontier = root / "frontier.jsonl"
    atomic_publish(frontier, encoded)
    value = {
        "schema": "aw.exact-url-frontier-receipt.v1",
        "version": VERSION,
        "status": "complete",
        "source": {
            "catalog_receipt": _artifact(artifact_root, catalog_receipt),
            "catalog_frontier": source_artifact,
        },
        "selection": "every source-owned product sitemap emitted by the domain-independent authenticated catalog-surface gate",
        "result": {"frontier": expected_frontier},
        "logical_sha256": hashlib.sha256(encoded).hexdigest(),
        "producer": {"id": PRODUCER, "version": VERSION},
    }
    atomic_publish(root / "RECEIPT.json",
                   (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
    return value


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(prog=prog or "cyt source tech-product frontier", description=__doc__)
    parser.add_argument("--catalog-receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(
        catalog_receipt=args.catalog_receipt.resolve(), root=args.root.resolve(),
        artifact_root=args.artifact_root.resolve()), indent=2, sort_keys=True))
