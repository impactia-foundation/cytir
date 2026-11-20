"""Apply the technical Product JSON-LD + documentation admission boundary."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from bs4 import BeautifulSoup

from cytir.sources.technical_manufacturer.catalog import TECHNICAL, page_semantics
from cytir.sources.exact_url.direct_preserved import validate_consent_gated_run
from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.sources.technical_manufacturer.catalog_frontier import _load_preserved_surface
from cytir.sources.technical_manufacturer.seeds import manufacturer_url
from cytir.index.preservation_contract import _validate_stage
from cytir.acquire import preserved_stage_io as PS


VERSION = "technical-product-admission-audit.v1"
PRODUCER = "pipeline/audit_technical_product_admission.py"
VOLATILE = {"price", "priceCurrency", "availability", "inventoryLevel",
            "lowPrice", "highPrice", "offerCount"}
REQUESTED_PIDOG_URL = (
    "https://sunfounder.com/products/sunfounder-pidog-robot-dog-kit-for-raspberry-pi")


def _artifact(root: Path, path: Path) -> dict:
    return {"uri": "artifact://" + path.resolve().relative_to(root.resolve()).as_posix(),
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _walk_keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_walk_keys(v) for v in value.values()),
                                       set())
    if isinstance(value, list):
        return set().union(*(_walk_keys(v) for v in value), set())
    return set()


def _requested_target_consequence(decisions: list[dict]) -> dict:
    target = manufacturer_url(REQUESTED_PIDOG_URL)
    matches = [row for row in decisions if manufacturer_url(
        str(row.get("canonical_url") or row.get("source_url") or "")) == target]
    if len(matches) > 1:
        raise ValueError("requested PiDog canonical target is duplicated")
    return {
        "requested_canonical_url": target,
        "frontier_selected": bool(matches),
        "source_preserved": bool(matches and matches[0].get("source_status") == 200),
        "admitted": bool(matches and matches[0].get("admitted")),
        "outcome": matches[0] if matches else None,
    }


def _host_consent(host_consent_root: Path, host: str,
                  artifact_root: Path) -> tuple[dict, dict]:
    path = host_consent_root / "receipts" / (
        hashlib.sha256(host.encode()).hexdigest() + ".json")
    receipt = json.loads(path.read_text())
    if (receipt.get("schema") != "aw.frontier-host-consent-receipt.v1"
            or receipt.get("version") != "frontier-host-consent.v1"
            or receipt.get("host") != host):
        raise ValueError("product host consent receipt identity drift")
    artifact = receipt.get("consent_artifact") or {}
    PS.validate_artifact(artifact_root, artifact)
    consent = json.loads(gzip.decompress(
        PS.artifact_path(artifact_root, artifact).read_bytes()))
    affirmative = (receipt.get("affirmative_search_consent") is True
                   and consent.get("status") == "ok"
                   and consent.get("search_ok") is True)
    return receipt, {"affirmative": affirmative,
                     "receipt": _artifact(artifact_root, path),
                     "evidence": artifact}


def audit(*, frontier: Path, frontier_receipt: Path, cc_root: Path,
          direct_root: Path | None, host_consent_root: Path,
          root: Path, artifact_root: Path) -> dict:
    if root.exists():
        raise ValueError("technical product admission root must be fresh")
    seal = json.loads(frontier_receipt.read_text())
    declared = ((seal.get("result") or {}).get("frontier") or {})
    if (seal.get("schema") != "aw.exact-url-frontier-receipt.v1"
            or declared.get("sha256") != sha256_file(frontier)):
        raise ValueError("product page frontier is not sealed")
    rows = [json.loads(line) for line in frontier.open() if line.strip()]
    if declared.get("urls") != len(rows):
        raise ValueError("product page frontier conservation failed")
    host_run_path = host_consent_root / "RUN.json"
    host_run = json.loads(host_run_path.read_text())
    if (host_run.get("schema") != "aw.frontier-host-consent-run.v1"
            or host_run.get("version") != "frontier-host-consent.v1"
            or host_run.get("status") != "complete"
            or host_run.get("frontier_sha256") != sha256_file(frontier)):
        raise ValueError("terminal current host-consent stage is required")
    _validate_stage(host_run["stages"]["consent"], "product.host-consent",
                    artifact_root=artifact_root, deep=True)
    cc_run_path = cc_root / "RUN.json"
    cc_run = json.loads(cc_run_path.read_text())
    if (cc_run.get("schema") != "aw.exact-url-cc-preserved-run.v1"
            or cc_run.get("status") != "complete"):
        raise ValueError("terminal product page CC acquisition is required")
    _validate_stage(cc_run["stages"]["original"], "cc.original",
                    artifact_root=artifact_root, deep=True)
    direct_run_path = None
    if direct_root is not None:
        direct_run_path = direct_root / "RUN.json"
        validate_consent_gated_run(direct_root, artifact_root, deep=True)

    decisions = []
    admitted = []
    outlink_rows = []
    for row in rows:
        preserved = _load_preserved_surface(
            url=row["url"], cc_root=cc_root, direct_root=direct_root,
            artifact_root=artifact_root)
        decision = {
            "source_url": row["url"], "domain": row["domain"],
            "route": preserved["route"], "source_status": preserved["status"],
            "terminal_receipt": preserved["terminal_receipt"],
            "body_artifact": preserved["body_artifact"],
            "admitted": False, "reason": "source_not_available",
        }
        host = str(row["domain"]).casefold()
        _consent_receipt, consent = _host_consent(
            host_consent_root, host, artifact_root)
        decision["host_consent"] = consent
        links = []
        if preserved["status"] == 200:
            semantics = page_semantics(
                preserved["body"], row["url"], row["domain"])
            soup = BeautifulSoup(preserved["body"], "html.parser")
            visible_text = " ".join(soup.get_text(" ", strip=True).split())
            product = semantics["jsonld_product"]
            links = semantics["documentation_links"]
            score = len(TECHNICAL.findall(visible_text))
            decision.update({
                "canonical_url": semantics["canonical_url"],
                "product_jsonld": bool(product),
                "technical_keyword_score": score,
                "documentation_links": links,
                "volatile_jsonld_fields_observed": sorted(
                    _walk_keys(product or {}) & VOLATILE),
            })
            if product and links and score and consent["affirmative"]:
                decision.update({
                    "admitted": True, "reason": "product_jsonld+technical_content+first_party_linked_docs",
                    "product_name": product.get("name"),
                    "sku": product.get("sku"),
                })
                admitted.append(decision)
            else:
                missing = []
                if not product: missing.append("product_jsonld")
                if not links: missing.append("documentation_link")
                if not score: missing.append("technical_content")
                if not consent["affirmative"]: missing.append("affirmative_search_consent")
                decision["reason"] = "missing:" + ",".join(missing)
        parent = decision.get("canonical_url") or row["url"]
        outlink_rows.append({
            "parent_url": parent,
            "links": [{"target_url": link["url"],
                       "relation": link["relation"],
                       "context": link["context"]} for link in links],
            "explicit_empty": not bool(links),
        })
        decisions.append(decision)

    root.mkdir(parents=True)
    decisions_path = root / "decisions.jsonl"
    admitted_path = root / "admitted-products.jsonl"
    outlinks_path = root / "parent-bound-outlinks.jsonl"
    atomic_publish(decisions_path, b"".join(
        (canonical_json(row) + "\n").encode() for row in decisions))
    atomic_publish(admitted_path, b"".join(
        (canonical_json(row) + "\n").encode() for row in admitted))
    atomic_publish(outlinks_path, b"".join(
        (canonical_json(row) + "\n").encode() for row in outlink_rows))
    inputs = [frontier, frontier_receipt, cc_run_path, host_run_path]
    if direct_run_path:
        inputs.append(direct_run_path)
    pidog = [row for row in admitted if
             str(row.get("sku") or "").upper() == "CN0481D"
             or "/pidog" in str(row.get("canonical_url") or "").casefold()]
    result = {
        "schema": "aw.technical-product-admission-audit.v1",
        "version": VERSION, "status": "complete",
        "policy": "canonical Product JSON-LD plus technical maker/robotics/electronics content plus documentation/manual/code link found on the preserved first-party page; no domain allowlist",
        "volatile_policy": "price, stock, availability, and inventory remain preserved only in source bytes and are excluded from durable embedding text",
        "counts": {"frontier": len(rows), "admitted": len(admitted),
                   "rejected": len(rows) - len(admitted),
                   "parents_with_links": sum(bool(row["links"]) for row in outlink_rows),
                   "explicit_empty_parents": sum(row["explicit_empty"] for row in outlink_rows)},
        "pidog_consequence": {"selected": bool(pidog), "records": pidog},
        "requested_pidog_target_consequence": _requested_target_consequence(decisions),
        "inputs": [_artifact(artifact_root, path) for path in inputs],
        "artifacts": {"decisions": _artifact(artifact_root, decisions_path),
                      "admitted_products": _artifact(artifact_root, admitted_path),
                      "parent_bound_outlinks": _artifact(artifact_root, outlinks_path)},
        "producer": {"id": PRODUCER, "version": VERSION},
        "next": "preserve linked manuals/docs/code, source-specific full Markdown, then profile/fidelity/token/UUID gates",
    }
    atomic_publish(root / "RECEIPT.json",
                   (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frontier", "frontier-receipt", "cc-root",
                 "host-consent-root", "root", "artifact-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--direct-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(
        frontier=args.frontier.resolve(),
        frontier_receipt=args.frontier_receipt.resolve(),
        cc_root=args.cc_root.resolve(),
        host_consent_root=args.host_consent_root.resolve(),
        direct_root=args.direct_root.resolve() if args.direct_root else None,
        root=args.root.resolve(), artifact_root=args.artifact_root.resolve()),
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
