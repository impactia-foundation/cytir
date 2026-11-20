"""Build a technical-catalog frontier from preserved standard surfaces."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.sources.technical_manufacturer.catalog import (
    CATALOG_MARKERS,
    _xml_locs,
    qualify_standard_catalog_surfaces,
)
from cytir.sources.exact_url.direct_preserved import (
    validate_consent_gated_receipt, validate_consent_gated_run)
from cytir.acquire import preserved_stage_io as PS
from cytir.index.preservation_contract import _validate_stage


VERSION = "technical-manufacturer-catalog-frontier.v1"
PRODUCER = "pipeline/build_technical_manufacturer_catalog_frontier.py"


def _json(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode()


def _artifact(artifact_root: Path, path: Path) -> dict[str, Any]:
    return {
        "uri": f"artifact://{path.resolve().relative_to(artifact_root.resolve()).as_posix()}",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _terminal_receipt(root: Path, url: str) -> Path:
    return root / "receipts" / f"{hashlib.sha256(url.encode()).hexdigest()}.json"


def _load_preserved_surface(*, url: str, cc_root: Path,
                            direct_root: Path | None,
                            artifact_root: Path) -> dict[str, Any]:
    cc_path = _terminal_receipt(cc_root, url)
    cc = json.loads(cc_path.read_text())
    if (cc.get("schema") != "aw.exact-url-cc-capture-receipt.v1"
            or cc.get("source_url") != url
            or cc.get("status") not in {"complete", "cc_miss"}):
        raise ValueError(f"invalid CC surface terminal: {url}")
    if cc["status"] == "complete":
        body_artifact = (cc["artifacts"].get("decoded_source")
                         or cc["artifacts"].get("decoded_html"))
        if body_artifact is None:
            raise ValueError(f"CC surface lacks decoded source bytes: {url}")
        PS.validate_artifact(artifact_root, body_artifact)
        body_path = PS.artifact_path(artifact_root, body_artifact)
        return {
            "status": 200,
            "route": "common-crawl-signed-s3",
            "body": gzip.decompress(body_path.read_bytes()),
            "body_artifact": body_artifact,
            "terminal_receipt": _artifact(artifact_root, cc_path),
        }
    if direct_root is None:
        return {"status": 0, "route": "cc-miss-direct-not-run", "body": b"",
                "body_artifact": None,
                "terminal_receipt": _artifact(artifact_root, cc_path)}
    direct_path = _terminal_receipt(direct_root, url)
    direct = json.loads(direct_path.read_text())
    if (direct.get("schema") != "aw.exact-url-direct-receipt.v1"
            or direct.get("source_url") != url
            or direct.get("status") not in {"complete", "quarantine"}):
        raise ValueError(f"invalid direct surface terminal: {url}")
    validate_consent_gated_receipt(direct, artifact_root)
    if direct["status"] != "complete":
        return {"status": 0, "route": "heimdall-direct-quarantine", "body": b"",
                "body_artifact": None,
                "terminal_receipt": _artifact(artifact_root, direct_path)}
    body_artifact = direct["artifacts"]["response_body"]
    PS.validate_artifact(artifact_root, body_artifact)
    body_path = PS.artifact_path(artifact_root, body_artifact)
    return {
        "status": 200,
        "route": "heimdall-residential-direct-after-cc-miss",
        "body": gzip.decompress(body_path.read_bytes()),
        "body_artifact": body_artifact,
        "terminal_receipt": _artifact(artifact_root, direct_path),
    }


def build(*, domain_frontier: Path, domain_receipt: Path,
          surface_frontier: Path, surface_receipt: Path, cc_root: Path,
          direct_root: Path | None, root: Path, artifact_root: Path) -> dict:
    if root.exists():
        raise ValueError("catalog-frontier output root must be fresh and immutable")
    domain_seal = json.loads(domain_receipt.read_text())
    surface_seal = json.loads(surface_receipt.read_text())
    if (domain_seal.get("schema") != "aw.technical-manufacturer-seed-receipt.v1"
            or (domain_seal.get("frontier") or {}).get("sha256")
            != sha256_file(domain_frontier)):
        raise ValueError("domain frontier is not sealed")
    if (surface_seal.get("schema") != "aw.exact-url-frontier-receipt.v1"
            or ((surface_seal.get("result") or {}).get("frontier") or {}).get(
                "sha256") != sha256_file(surface_frontier)):
        raise ValueError("standard-surface frontier is not sealed")
    cc_run_path = cc_root / "RUN.json"
    cc_run = json.loads(cc_run_path.read_text())
    if (cc_run.get("schema") != "aw.exact-url-cc-preserved-run.v1"
            or cc_run.get("status") != "complete"):
        raise ValueError("terminal CC surface acquisition is required")
    _validate_stage((cc_run.get("stages") or {}).get("original"), "cc.original",
                    artifact_root=artifact_root, deep=True)
    direct_run_path = None
    if direct_root is not None:
        direct_run_path = direct_root / "RUN.json"
        validate_consent_gated_run(direct_root, artifact_root, deep=True)

    domains = [json.loads(line) for line in domain_frontier.open() if line.strip()]
    surfaces = [json.loads(line) for line in surface_frontier.open() if line.strip()]
    if len(surfaces) != len(domains) * 3:
        raise ValueError("standard surface/domain conservation failed")
    by_domain: dict[str, dict[str, dict]] = {}
    for row in surfaces:
        by_domain.setdefault(row["domain"], {})[row["surface"]] = row
    decisions = []
    product_sitemaps = []
    for domain_row in domains:
        domain = domain_row["domain"]
        surface_rows = by_domain.get(domain) or {}
        if set(surface_rows) != {"agents-md", "ucp-well-known", "sitemap-index"}:
            raise ValueError(f"incomplete standard surface set: {domain}")
        preserved = {
            name: _load_preserved_surface(
                url=surface_rows[name]["url"], cc_root=cc_root,
                direct_root=direct_root, artifact_root=artifact_root)
            for name in sorted(surface_rows)
        }
        agents_text = preserved["agents-md"]["body"].decode("utf-8", "replace")
        ucp_text = preserved["ucp-well-known"]["body"].decode("utf-8", "replace")
        child_sitemaps = _xml_locs(preserved["sitemap-index"]["body"])
        markers = sorted(marker for marker in CATALOG_MARKERS
                         if marker in (agents_text + "\n" + ucp_text).casefold())
        qualification = qualify_standard_catalog_surfaces(
            agents_status=preserved["agents-md"]["status"],
            ucp_status=preserved["ucp-well-known"]["status"],
            sitemap_status=preserved["sitemap-index"]["status"],
            catalog_markers=markers, child_sitemaps=child_sitemaps)
        decision = {
            "domain": domain,
            "seed_evidence": domain_row["evidence"],
            "catalog_markers": markers,
            "child_sitemaps": sorted(set(child_sitemaps)),
            **qualification,
            "surfaces": {
                name: {key: value for key, value in preserved[name].items()
                       if key != "body"}
                for name in sorted(preserved)
            },
        }
        decisions.append(decision)
        if decision["qualified"]:
            for url in decision["product_sitemaps"]:
                product_sitemaps.append({
                    "url": url,
                    "domain": domain,
                    "surface": "source-owned-product-sitemap",
                    "source_refs": domain_row["evidence"],
                })
    product_sitemaps.sort(key=lambda row: (row["domain"], row["url"]))
    seen_urls: set[str] = set()
    product_sitemaps = [row for row in product_sitemaps
                        if not (row["url"] in seen_urls or seen_urls.add(row["url"]))]
    root.mkdir(parents=True)
    decisions_path = root / "catalog-domain-decisions.jsonl"
    frontier_path = root / "product-sitemap-frontier.jsonl"
    atomic_publish(decisions_path, b"".join(_json(row) for row in decisions))
    atomic_publish(frontier_path, b"".join(_json(row) for row in product_sitemaps))
    inputs = [domain_frontier, domain_receipt, surface_frontier, surface_receipt,
              cc_run_path]
    if direct_run_path is not None:
        inputs.append(direct_run_path)
    result = {
        "schema": "aw.technical-manufacturer-catalog-frontier-receipt.v1",
        "version": VERSION,
        "status": "complete",
        "selection": "any authenticated standard catalog surface (agents.md, well-known UCP, or first-party product sitemap); no domain allowlist",
        "counts": {
            "seed_domains": len(domains),
            "qualified_catalog_domains": sum(row["qualified"] for row in decisions),
            "product_sitemaps": len(product_sitemaps),
        },
        "inputs": [_artifact(artifact_root, path) for path in inputs],
        "artifacts": {
            "domain_decisions": _artifact(artifact_root, decisions_path),
            "product_sitemap_frontier": _artifact(artifact_root, frontier_path),
        },
        "producer": {"id": PRODUCER, "version": VERSION},
        "next": "preserve source-owned product sitemaps/catalog endpoints; then Product JSON-LD plus first-party docs admission",
    }
    atomic_publish(root / "RECEIPT.json",
                   (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("domain-frontier", "domain-receipt", "surface-frontier",
                 "surface-receipt", "cc-root", "root", "artifact-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--direct-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(
        domain_frontier=args.domain_frontier.resolve(),
        domain_receipt=args.domain_receipt.resolve(),
        surface_frontier=args.surface_frontier.resolve(),
        surface_receipt=args.surface_receipt.resolve(),
        cc_root=args.cc_root.resolve(),
        direct_root=args.direct_root.resolve() if args.direct_root else None,
        root=args.root.resolve(), artifact_root=args.artifact_root.resolve()),
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
