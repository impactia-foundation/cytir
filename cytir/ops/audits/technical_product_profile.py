"""Deep semantic/profile audit for the technical manufacturer product pack."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import uuid

from cytir.sources.exact_url.direct_preserved import validate_consent_gated_run
from cytir.core.paths import data_path
from cytir.acquire.artifact_io import atomic_publish, sha256_file
from cytir.sources.technical_manufacturer.seeds import manufacturer_url
from cytir.index.preservation_contract import _validate_stage
from cytir.acquire import preserved_stage_io as PS


PROFILE = "technical_manufacturer_catalog"
POLICY_PATH = "data/chunk-policy-struct-md-768w-o0-adjacent-sections-tail-1152-tok2560.v1.json"
POLICY_ID = "struct-md-768w-o0-adjacent-sections-tail-1152-tok2560"
TARGET = manufacturer_url(
    "https://sunfounder.com/products/sunfounder-pidog-robot-dog-kit-for-raspberry-pi")
VOLATILE = {"price", "priceCurrency", "availability", "inventoryLevel",
            "lowPrice", "highPrice", "offerCount", "offers",
            "inventory_quantity", "available", "compare_at_price",
            "promotion", "discount"}


def _rows(root: Path, stage: dict) -> list[dict]:
    checked = _validate_stage(stage, "input", artifact_root=root, deep=True)
    receipt_path = root / checked["receipt"]["uri"].removeprefix("artifact://")
    receipt = json.loads(receipt_path.read_text())
    rows = []
    for artifact in receipt["artifacts"]:
        if artifact["uri"].endswith(".jsonl.gz"):
            with gzip.open(root / artifact["uri"].removeprefix("artifact://"),
                           "rt", encoding="utf-8") as source:
                rows.extend(json.loads(line) for line in source if line.strip())
    return rows


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        nested = set()
        for child in value.values():
            nested.update(_keys(child))
        return set(value) | nested
    if isinstance(value, list):
        nested = set()
        for child in value:
            nested.update(_keys(child))
        return nested
    return set()


def audit(*, artifact_root: Path, parent_run: Path, pack_root: Path,
          documentation_frontier_receipt: Path, documentation_cc_root: Path,
          documentation_direct_root: Path, output_root: Path,
          registry_path: Path, expected_products: int = 445,
          expected_sku_unavailable: int = 16) -> tuple[Path, Path]:
    run = json.loads(parent_run.read_text())
    counts = run.get("counts") or {}
    if (run.get("schema") != "aw.technical-product-materialized-run.v1"
            or run.get("status") != "full_product_markdown_complete_pending_profile_fidelity_chunk_measurement"
            or run.get("source_profile") != PROFILE
            or counts.get("markdown_documents") != expected_products
            or counts.get("outlink_parents") != expected_products
            or counts.get("quarantine") != 0
            or (run.get("content_fidelity") or {}).get("source_content_cap") is not None
            or not (run.get("content_fidelity") or {}).get("full_prechunk_markdown")):
        raise ValueError("technical product materialization conservation drifted")
    for name, stage in run["stages"].items():
        _validate_stage(stage, f"parent.{name}", artifact_root=artifact_root,
                        deep=True)

    frontier_receipt = json.loads(documentation_frontier_receipt.read_text())
    frontier = (frontier_receipt.get("result") or {}).get("frontier") or {}
    if (frontier_receipt.get("schema") != "aw.exact-url-frontier-receipt.v1"
            or frontier_receipt.get("status") != "exact_technical_documentation_frontier_complete"):
        raise ValueError("technical documentation frontier is not terminal")
    PS.validate_artifact(artifact_root, {
        **frontier, "compression": "gzip"
    }) if frontier.get("compression") == "gzip" else None
    frontier_path = artifact_root / frontier["uri"].removeprefix("artifact://")
    if (not frontier_path.is_file() or frontier_path.stat().st_size != frontier["bytes"]
            or sha256_file(frontier_path) != frontier["sha256"]):
        raise ValueError("technical documentation frontier artifact drifted")
    documentation_urls = {json.loads(line)["url"] for line in frontier_path.open()
                          if line.strip()}
    if len(documentation_urls) != frontier.get("urls"):
        raise ValueError("technical documentation URL conservation drifted")
    cc_run_path = documentation_cc_root / "RUN.json"
    cc_run = json.loads(cc_run_path.read_text())
    if (cc_run.get("schema") != "aw.exact-url-cc-preserved-run.v1"
            or cc_run.get("status") != "complete"):
        raise ValueError("technical documentation signed-S3 stage is not terminal")
    _validate_stage(cc_run["stages"]["original"], "docs.cc.original",
                    artifact_root=artifact_root, deep=True)
    direct_run_path = documentation_direct_root / "RUN.json"
    direct_run = validate_consent_gated_run(documentation_direct_root,
                                            artifact_root, deep=True)

    pack_path = pack_root / "PACK.json"
    pack = json.loads(pack_path.read_text())
    policy = pack.get("chunk_policy") or {}
    expected_policy = data_path(POLICY_PATH.removeprefix("data/"))
    if (pack.get("status") != "ready_for_gpu_queue"
            or pack.get("source_profile") != PROFILE
            or policy.get("path") != POLICY_PATH
            or policy.get("policy_id") != POLICY_ID
            or policy.get("sha256") != sha256_file(expected_policy)
            or (pack.get("token_distribution") or {}).get("over_hard_max") != 0
            or (pack.get("token_distribution") or {}).get("max", 2561) > 2560):
        raise ValueError("technical product pack policy/token gate failed")
    _validate_stage(pack["chunks"], "pack.chunks", artifact_root=artifact_root,
                    deep=True)

    markdown = _rows(artifact_root, run["stages"]["markdown"])
    outlinks = _rows(artifact_root, run["stages"]["outlinks"])
    chunks = _rows(artifact_root, pack["chunks"])
    parents = {}
    sku_unavailable = 0
    description_statuses: dict[str, int] = {}
    linked_docs = set()
    target = None
    for row in markdown:
        product = row.get("product") or {}
        document_id = str(row.get("document_id") or "")
        canonical = manufacturer_url(str(row.get("url") or ""))
        links = product.get("documentation_links") or []
        status = str(product.get("technical_description_status") or "")
        if (not document_id or document_id in parents or row.get("source_profile") != PROFILE
                or canonical != row.get("url") or product.get("canonical_url") != canonical
                or not product.get("name") or not product.get("manufacturer")
                or not product.get("stable_product_identity") or not links
                or status not in {"product_jsonld", "source_html_product_description", "docs_only"}
                or _keys(product.get("stable_jsonld") or {}) & VOLATILE
                or not (row.get("extraction") or {}).get("full_prechunk_content")
                or (row.get("extraction") or {}).get("content_cap") is not None
                or not (row.get("extraction") or {}).get(
                    "volatile_commerce_fields_excluded_from_markdown")):
            raise ValueError("technical product Markdown semantics are incomplete")
        if status != "docs_only" and "## Technical description\n\n" not in row["markdown"]:
            raise ValueError("technical product description was omitted from Markdown")
        if product.get("sku") is None:
            sku_unavailable += 1
            if (product.get("sku_status") != "unavailable"
                    or product.get("identity_kind") != "canonical_product_url"
                    or product.get("stable_product_identity") != canonical):
                raise ValueError("SKU-unavailable product identity is not explicit")
        elif (product.get("sku_status") != "source_declared"
              or product.get("stable_product_identity") != product.get("sku")):
            raise ValueError("source SKU product identity drifted")
        urls = {link["url"] for link in links}
        linked_docs.update(urls)
        if not urls.issubset(documentation_urls):
            raise ValueError("product documentation link absent from preserved frontier")
        description_statuses[status] = description_statuses.get(status, 0) + 1
        parents[document_id] = row
        if canonical == TARGET:
            target = row
    if (len(parents) != expected_products or sku_unavailable != expected_sku_unavailable
            or linked_docs != documentation_urls):
        raise ValueError("technical product parent/SKU/docs conservation failed")
    if (target is None or (target.get("product") or {}).get("sku") != "CN0481D"
            or (target.get("product") or {}).get("technical_description_status") != "product_jsonld"
            or len((target.get("product") or {}).get("documentation_links") or []) < 6):
        raise ValueError("requested PiDog consequence did not survive materialization")

    represented = set()
    for row in outlinks:
        parent = parents.get(str(row.get("document_id") or ""))
        if (parent is None or row.get("source_url") != parent.get("url")
                or row.get("links") != parent.get("source_links")
                or row.get("explicit_empty") is not False):
            raise ValueError("technical product outlink parent binding drifted")
        represented.add(row["document_id"])
    if represented != set(parents):
        raise ValueError("technical product outlink parent conservation failed")

    point_ids = set()
    ordinals: dict[str, list[int]] = {}
    parent_sha = {key: hashlib.sha256(value["markdown"].encode()).hexdigest()
                  for key, value in parents.items()}
    for row in chunks:
        point_id = str(row.get("point_id") or "")
        uuid.UUID(point_id)
        document_id = str(row.get("document_id") or "")
        parent = parents.get(document_id)
        if (point_id in point_ids or parent is None
                or row.get("source_profile") != PROFILE
                or row.get("parent_markdown_sha256") != parent_sha[document_id]
                or (row.get("source_metadata") or {}).get("product") != parent.get("product")
                or row.get("source_links") != parent.get("source_links")):
            raise ValueError("technical product chunk lost source identity")
        point_ids.add(point_id)
        ordinals.setdefault(document_id, []).append(row.get("chunk_ordinal"))
    if (set(ordinals) != set(parents)
            or any(sorted(values) != list(range(len(values)))
                   for values in ordinals.values())):
        raise ValueError("technical product chunk parent/ordinal conservation failed")

    registry = json.loads(registry_path.read_text())
    profile = registry["profiles"][PROFILE]
    methods = {
        "original": [
            "Deep-rehashed each admitted exact first-party product body plus transport, headers, and consent evidence.",
            "Verified requested/final URL, HTTP status, fetch-time transport, and body digest for every retained product.",
            "Verified canonical product URL plus source SKU/MPN or explicit canonical-URL identity when SKU is unavailable.",
            f"Bound all {len(documentation_urls)} source-declared manual/docs/code targets to terminal signed-S3 then consent-gated direct evidence."],
        "markdown": [
            f"Scanned {len(parents)} source-specific product Markdown records for manufacturer/name/identity and complete available technical description.",
            "Verified parent-bound manual/docs/code relations and body/receipt identity on every record.",
            "Recursively rejected volatile commerce keys from durable product metadata/text inputs while retaining them only in L0."],
        "chunks": [
            "Verified every chunk belongs to exactly one stable product model/URL identity.",
            "Recomputed every parent Markdown digest and checked pinned current policy/tokenizer evidence.",
            f"Validated {len(point_ids)} unique canonical UUID point IDs, contiguous ordinals, and exact max-token <=2560."],
        "outlinks": [
            "Verified every parent retains official product URL and stable product identity.",
            "Matched all manual/documentation/tutorial/code targets and contexts to the admission frontier.",
            "Verified an explicit parent-bound outlink row for every retained product."],
        "vectors": [],
    }
    checks = {}
    for stage in profile["required_stages"]:
        requirements = profile.get(stage, {}).get("requirements", [])
        if len(requirements) != len(methods[stage]):
            raise ValueError(f"technical product profile method mismatch: {stage}")
        checks[stage] = [{"requirement": requirement, "status": "PASS",
                          "verification_method": method}
                         for requirement, method in zip(requirements, methods[stage])]
    evidence = {
        "schema": "aw.technical-product-profile-evidence.v1",
        "source_id": "technical-manufacturer-catalog-admitted-products-v2",
        "counts": {"products": len(parents), "sku_unavailable": sku_unavailable,
                   "chunks": len(chunks), "unique_point_ids": len(point_ids),
                   "documentation_urls": len(documentation_urls),
                   "documentation_cc_complete": (cc_run.get("counts") or {}).get("complete"),
                   "documentation_direct_complete": (direct_run.get("counts") or {}).get("complete"),
                   "documentation_quarantine": (direct_run.get("counts") or {}).get("quarantine")},
        "description_statuses": description_statuses,
        "requested_pidog": {"url": TARGET, "sku": "CN0481D",
                            "documentation_links": len(target["product"]["documentation_links"])},
        "parent_run_sha256": sha256_file(parent_run),
        "pack_sha256": sha256_file(pack_path),
        "documentation_frontier_receipt_sha256": sha256_file(documentation_frontier_receipt),
        "documentation_cc_run_sha256": sha256_file(cc_run_path),
        "documentation_direct_run_sha256": sha256_file(direct_run_path),
        "profile_registry_sha256": sha256_file(registry_path),
        "classification": "measured admitted technical-product pack; no portfolio-scale point extrapolation",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_path = output_root / "verification-evidence.json"
    checks_path = output_root / "checks.json"
    for path, value in ((evidence_path, evidence), (checks_path, checks)):
        encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        if path.exists() and path.read_bytes() != encoded:
            raise ValueError(f"immutable technical product audit collision: {path}")
        if not path.exists():
            atomic_publish(path, encoded)
    return evidence_path, checks_path


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog or "technical_product_profile.py", description=__doc__)
    for name in ("artifact-root", "parent-run", "pack-root",
                 "documentation-frontier-receipt", "documentation-cc-root",
                 "documentation-direct-root", "output-root", "profile-registry"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-products", type=int, default=445)
    parser.add_argument("--expected-sku-unavailable", type=int, default=16)
    args = parser.parse_args(argv)
    evidence, checks = audit(
        artifact_root=args.artifact_root.resolve(),
        parent_run=args.parent_run.resolve(), pack_root=args.pack_root.resolve(),
        documentation_frontier_receipt=args.documentation_frontier_receipt.resolve(),
        documentation_cc_root=args.documentation_cc_root.resolve(),
        documentation_direct_root=args.documentation_direct_root.resolve(),
        output_root=args.output_root.resolve(), registry_path=args.profile_registry.resolve(),
        expected_products=args.expected_products,
        expected_sku_unavailable=args.expected_sku_unavailable)
    print(json.dumps({"evidence": str(evidence), "checks": str(checks)},
                     indent=2, sort_keys=True))
    return 0
