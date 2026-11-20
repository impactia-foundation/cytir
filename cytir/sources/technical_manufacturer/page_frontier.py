"""Derive canonical first-party product URLs from preserved product sitemaps."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.sources.technical_manufacturer.catalog import _xml_locs
from cytir.sources.exact_url.direct_preserved import validate_consent_gated_run
from cytir.sources.technical_manufacturer.catalog_frontier import _load_preserved_surface
from cytir.sources.technical_manufacturer.seeds import manufacturer_url
from cytir.index.preservation_contract import _validate_stage


VERSION = "technical-product-page-frontier.v1"
PRODUCER = "aw/sources/technical_manufacturer/page_frontier.py"


def _artifact(root: Path, path: Path) -> dict:
    return {"uri": "artifact://" + path.resolve().relative_to(root.resolve()).as_posix(),
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def build(*, sitemap_frontier: Path, sitemap_receipt: Path, cc_root: Path,
          direct_root: Path | None, root: Path, artifact_root: Path) -> dict:
    if root.exists():
        raise ValueError("product page frontier root must be fresh")
    seal = json.loads(sitemap_receipt.read_text())
    declared = ((seal.get("result") or {}).get("frontier") or {})
    if (seal.get("schema") != "aw.exact-url-frontier-receipt.v1"
            or declared.get("sha256") != sha256_file(sitemap_frontier)):
        raise ValueError("product sitemap frontier is not sealed")
    sitemap_rows = [json.loads(line) for line in sitemap_frontier.open()
                    if line.strip()]
    if declared.get("urls") != len(sitemap_rows):
        raise ValueError("product sitemap frontier conservation failed")
    cc_run_path = cc_root / "RUN.json"
    cc_run = json.loads(cc_run_path.read_text())
    if (cc_run.get("schema") != "aw.exact-url-cc-preserved-run.v1"
            or cc_run.get("status") != "complete"):
        raise ValueError("terminal product-sitemap CC acquisition is required")
    _validate_stage(cc_run["stages"]["original"], "cc.original",
                    artifact_root=artifact_root, deep=True)
    direct_run_path = None
    if direct_root is not None:
        direct_run_path = direct_root / "RUN.json"
        validate_consent_gated_run(direct_root, artifact_root, deep=True)

    products = {}
    sitemap_evidence = []
    for sitemap in sitemap_rows:
        url = sitemap["url"]
        preserved = _load_preserved_surface(
            url=url, cc_root=cc_root, direct_root=direct_root,
            artifact_root=artifact_root)
        sitemap_evidence.append({
            "url": url, "domain": sitemap["domain"],
            "status": preserved["status"], "route": preserved["route"],
            "body_artifact": preserved["body_artifact"],
            "terminal_receipt": preserved["terminal_receipt"],
        })
        if preserved["status"] != 200:
            continue
        domain = str(sitemap["domain"]).casefold()
        for raw in _xml_locs(preserved["body"]):
            url_value = manufacturer_url(raw)
            parsed = urllib.parse.urlsplit(url_value)
            host = (parsed.hostname or "").casefold()
            if (not url_value or not (host == domain or host.endswith("." + domain))
                    or "/products/" not in parsed.path.casefold()):
                continue
            products.setdefault(url_value, {
                "url": url_value, "domain": domain,
                "surface": "canonical-product-page",
                "source_refs": [{
                    "product_sitemap_url": url,
                    "terminal_receipt": preserved["terminal_receipt"],
                }],
            })
    rows = [products[url] for url in sorted(products)]
    root.mkdir(parents=True)
    frontier = root / "frontier.jsonl"
    evidence = root / "sitemap-evidence.jsonl"
    encoded = b"".join((canonical_json(row) + "\n").encode() for row in rows)
    atomic_publish(frontier, encoded)
    atomic_publish(evidence, b"".join(
        (canonical_json(row) + "\n").encode() for row in sitemap_evidence))
    inputs = [sitemap_frontier, sitemap_receipt, cc_run_path]
    if direct_run_path:
        inputs.append(direct_run_path)
    value = {
        "schema": "aw.exact-url-frontier-receipt.v1",
        "version": VERSION, "status": "complete",
        "source": {"inputs": [_artifact(artifact_root, path) for path in inputs],
                   "product_sitemaps": len(sitemap_rows)},
        "selection": "canonical same-party /products/ URLs from every preserved authenticated source-owned product sitemap; tracking parameters removed; no domain allowlist",
        "result": {"frontier": {
            **_artifact(artifact_root, frontier), "urls": len(rows)},
            "sitemap_evidence": _artifact(artifact_root, evidence)},
        "logical_sha256": hashlib.sha256(encoded).hexdigest(),
        "producer": {"id": PRODUCER, "version": VERSION},
    }
    atomic_publish(root / "RECEIPT.json",
                   (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
    return value


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(prog=prog or "cyt source tech-product pages", description=__doc__)
    for name in ("sitemap-frontier", "sitemap-receipt", "cc-root", "root",
                 "artifact-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--direct-root", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build(
        sitemap_frontier=args.sitemap_frontier.resolve(),
        sitemap_receipt=args.sitemap_receipt.resolve(),
        cc_root=args.cc_root.resolve(),
        direct_root=args.direct_root.resolve() if args.direct_root else None,
        root=args.root.resolve(), artifact_root=args.artifact_root.resolve()),
        indent=2, sort_keys=True))
