#!/usr/bin/env python3
"""Preserve and admit a domain-agnostic technical manufacturer catalog pilot (moved from the flat
task 1401 acquire script, now this module; the family adapter
`cytir.sources.families.technical_manufacturer_catalog` calls it). `acquire` is the whole run; `load_seeds`,
`acquire_domain` and `seal_run` are its three steps, so the adapter can run one domain per unit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import httpx
from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire import preserved_stage_io as PS
from cytir.acquire import stage_receipt as SS
from cytir.sources.technical_manufacturer.urls import manufacturer_url


VERSION = "technical-manufacturer-catalog-discovery.v2"
#: frozen label: stage receipts sealed before the move carry this producer string; a rewritten
#: label would seal the same run to different bytes (task 1401 parity test)
PRODUCER = "pipeline/acquire_technical_manufacturer_catalog.py"
TECHNICAL = re.compile(
    r"\b(robot|robotics|raspberry\s*pi|arduino|microcontroller|sensor|servo|"
    r"motor|camera|electronics?|embedded|maker|single[- ]board|iot|3d\s*print)",
    re.I)
DOC_LINK = re.compile(
    r"(docs?|documentation|manual|learn|wiki|github|readthedocs|tutorial|guide)",
    re.I)
CATALOG_MARKERS = ("universal commerce protocol", "ucp", "/products/{handle}.json")


def _json(value) -> bytes:
    return (canonical_json(value) + "\n").encode()


def _artifact_paths(artifact_root: Path, receipt: dict) -> list[Path]:
    return [PS.artifact_path(artifact_root, item)
            for item in (receipt.get("artifacts") or {}).values()]


class Fetcher:
    def __init__(self, *, root: Path, artifact_root: Path, rate: float, client=None):
        self.root = root
        self.artifact_root = artifact_root
        self.relative = root.relative_to(artifact_root).as_posix()
        self.interval = 1.0 / rate
        self.next = 0.0
        #: a client the caller owns (the family adapter hands over the run's `ctx.http`); the
        #: default builds the driver's own and closes it
        self._owns_client = client is None
        self.client = client if client is not None else make_client(HttpSpec(
            timeout=60, follow_redirects=True,
            headers={"user-agent": "authoritative-web/0.1 (+https://websear.ch)",
                     "Accept": "text/html,application/json,application/xml,text/markdown;q=0.9,*/*;q=0.5"}))

    def close(self):
        if self._owns_client:
            self.client.close()

    def fetch(self, url: str, purpose: str) -> dict:
        request_id = hashlib.sha256(url.encode()).hexdigest()
        receipt_path = self.root / "receipts/requests" / request_id[:2] / f"{request_id}.json"
        if receipt_path.exists():
            value = json.loads(receipt_path.read_text())
            if value.get("requested_url") != url or value.get("purpose") != purpose:
                raise ValueError("request receipt identity drift")
            for artifact in (value.get("artifacts") or {}).values():
                PS.validate_artifact(self.artifact_root, artifact)
            return value
        last = None
        for attempt in range(1, 4):
            delay = max(0.0, self.next - time.monotonic())
            if delay:
                time.sleep(delay)
            self.next = max(self.next, time.monotonic()) + self.interval
            started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            try:
                response = self.client.get(url)
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"server status {response.status_code}",
                        request=response.request, response=response)
                body = bytes(response.content)
                headers = _json(sorted((str(k), str(v))
                                       for k, v in response.headers.multi_items()))
                transport = _json({
                    "schema": "aw.http-transport.v1", "attempt": attempt,
                    "requested_url": url, "final_url": str(response.url),
                    "status": response.status_code, "fetched_at": started,
                    "response_body": {"bytes": len(body),
                                      "sha256": hashlib.sha256(body).hexdigest()},
                })
                artifacts = {
                    "body": PS.store_artifact(
                        self.artifact_root, f"{self.relative}/original/http-body",
                        "bin", body),
                    "headers": PS.store_artifact(
                        self.artifact_root, f"{self.relative}/original/http-headers",
                        "json", headers),
                    "transport": PS.store_artifact(
                        self.artifact_root, f"{self.relative}/original/transport",
                        "json", transport),
                }
                value = {
                    "schema": "aw.technical-manufacturer-http-receipt.v1",
                    "version": VERSION, "status": "complete",
                    "purpose": purpose, "requested_url": url,
                    "final_url": str(response.url),
                    "http_status": response.status_code,
                    "content_type": response.headers.get("content-type", ""),
                    "artifacts": artifacts,
                }
                atomic_publish(receipt_path,
                               (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
                return value
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
                time.sleep(attempt)
        raise RuntimeError(f"HTTP acquisition failed for {url}: {last}")

    def body(self, receipt: dict) -> bytes:
        artifact = receipt["artifacts"]["body"]
        return PS.load_validated_artifact(self.artifact_root, artifact)


def _xml_locs(body: bytes) -> list[str]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return []
    return [str(node.text).strip() for node in root.iter()
            if node.tag.endswith("loc") and str(node.text or "").strip()]


def _product(value: object) -> dict | None:
    if isinstance(value, dict):
        if str(value.get("@type") or "").casefold() == "product":
            return value
        for child in value.values():
            found = _product(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _product(child)
            if found:
                return found
    return None


def qualify_standard_catalog_surfaces(*, agents_status: int, ucp_status: int,
                                      sitemap_status: int,
                                      catalog_markers: list[str],
                                      child_sitemaps: list[str]) -> dict:
    """Apply the domain-independent catalog discovery boundary.

    No single emerging standard is mandatory.  A manufacturer can enter the
    product-evaluation cohort through an authenticated agents/UCP declaration
    or a first-party product sitemap.  Individual products still require the
    stricter Product JSON-LD plus substantive documentation gate downstream.
    """
    product_sitemaps = sorted({url for url in child_sitemaps
                               if "product" in url.casefold()})
    authenticated = (
        (agents_status == 200 and bool(catalog_markers))
        or (ucp_status == 200 and bool(catalog_markers))
        or (sitemap_status == 200 and bool(product_sitemaps))
    )
    return {
        "authenticated_standard_catalog_surface": authenticated,
        "product_sitemaps": product_sitemaps,
        "qualified": sitemap_status == 200 and authenticated,
    }


def page_semantics(body: bytes, url: str, domain: str) -> dict:
    soup = BeautifulSoup(body, "html.parser")
    canonical = manufacturer_url(url)
    link = soup.find("link", rel=lambda value: value and "canonical" in value)
    if link and link.get("href"):
        canonical = manufacturer_url(urllib.parse.urljoin(url, link["href"])) or canonical
    product = None
    for script in soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)}):
        try:
            product = _product(json.loads(script.string or script.get_text() or ""))
        except (json.JSONDecodeError, TypeError):
            continue
        if product:
            break
    links = []
    for anchor in soup.find_all("a", href=True):
        target = manufacturer_url(urllib.parse.urljoin(url, anchor["href"]))
        context = " ".join(anchor.get_text(" ", strip=True).split())
        if target and DOC_LINK.search(target + " " + context):
            links.append({"url": target, "context": context,
                          "relation": "technical-documentation"})
    unique = {row["url"]: row for row in links}
    return {"canonical_url": canonical, "jsonld_product": product,
            "documentation_links": [unique[key] for key in sorted(unique)]}


def load_seeds(frontier: Path, frontier_receipt: Path) -> list[dict]:
    """The sealed seed domains; a frontier that is not the receipt's, or lost a domain, refuses."""
    seed_receipt = json.loads(frontier_receipt.read_text())
    if ((seed_receipt.get("frontier") or {}).get("sha256") != sha256_file(frontier)
            or seed_receipt.get("schema") != "aw.technical-manufacturer-seed-receipt.v1"):
        raise ValueError("technical manufacturer frontier is not sealed")
    seeds = [json.loads(line) for line in frontier.open() if line.strip()]
    if len(seeds) != (seed_receipt.get("frontier") or {}).get("domains"):
        raise ValueError("technical manufacturer seed conservation failed")
    return seeds


def acquire_domain(fetcher: Fetcher, seed: dict, *, root: Path, max_products_per_domain: int,
                   max_technical_pages_per_domain: int) -> dict:
    """Fetch and qualify one seed domain; publish its domain receipt under `root`.

    Returns `{"domain_result", "qualified", "product_rows", "admitted", "receipt_path"}`.
    """
    product_rows: list[dict] = []
    admitted: list[dict] = []
    domain = seed["domain"]
    agents_url = f"https://{domain}/agents.md"
    ucp_url = f"https://{domain}/.well-known/ucp"
    sitemap_url = f"https://{domain}/sitemap.xml"
    agents = fetcher.fetch(agents_url, "agent-instructions")
    ucp = fetcher.fetch(ucp_url, "well-known-ucp-catalog")
    sitemap = fetcher.fetch(sitemap_url, "root-sitemap")
    agents_text = fetcher.body(agents).decode("utf-8", "replace")
    ucp_text = fetcher.body(ucp).decode("utf-8", "replace")
    sitemap_body = fetcher.body(sitemap)
    child_sitemaps = [manufacturer_url(url) for url in _xml_locs(sitemap_body)
                      if "sitemap" in url.casefold()]
    markers = sorted(marker for marker in CATALOG_MARKERS
                     if marker in (agents_text + "\n" + ucp_text).casefold())
    qualification = qualify_standard_catalog_surfaces(
        agents_status=agents["http_status"], ucp_status=ucp["http_status"],
        sitemap_status=sitemap["http_status"], catalog_markers=markers,
        child_sitemaps=child_sitemaps)
    product_sitemaps = qualification["product_sitemaps"]
    authenticated_surface = qualification["authenticated_standard_catalog_surface"]
    is_qualified = qualification["qualified"]
    domain_result = {"domain": domain, "seed_evidence": seed["evidence"],
                     "agents_receipt": agents, "ucp_receipt": ucp,
                     "sitemap_receipt": sitemap,
                     "catalog_markers": markers,
                     "product_sitemaps": product_sitemaps,
                     "authenticated_standard_catalog_surface": authenticated_surface,
                     "child_sitemaps": sorted(set(child_sitemaps)),
                     "qualified": is_qualified}
    if is_qualified:
        product_urls = set()
        for child_url in sorted(set(child_sitemaps)):
            child = fetcher.fetch(child_url, "catalog-sitemap")
            if child["http_status"] != 200:
                continue
            for url in _xml_locs(fetcher.body(child)):
                value = manufacturer_url(url)
                parsed = urllib.parse.urlsplit(value)
                if parsed.hostname and parsed.hostname.endswith(domain) and "/products/" in parsed.path:
                    product_urls.add(value)
        urls = sorted(product_urls)[:max_products_per_domain]
        domain_result["product_urls_discovered"] = len(product_urls)
        domain_result["product_urls_piloted"] = len(urls)
        technical = []
        for product_url in urls:
            parsed = urllib.parse.urlsplit(product_url)
            json_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc,
                                               parsed.path.rstrip("/") + ".json", "", ""))
            response = fetcher.fetch(json_url, "ucp-product-json")
            if response["http_status"] != 200:
                continue
            try:
                value = json.loads(fetcher.body(response))
            except json.JSONDecodeError:
                continue
            product = value.get("product", value)
            title = str(product.get("title") or "")
            body_html = str(product.get("body_html") or "")
            product_type = str(product.get("product_type") or "")
            score = len(TECHNICAL.findall(" ".join((title, body_html, product_type))))
            variants = product.get("variants") or []
            skus = sorted({str(row.get("sku") or "").strip()
                           for row in variants if str(row.get("sku") or "").strip()})
            row = {"domain": domain, "product_url": product_url,
                   "product_json_url": json_url, "title": title,
                   "product_type": product_type, "skus": skus,
                   "technical_keyword_score": score,
                   "product_json_receipt": response,
                   "volatile_fields_excluded_downstream": [
                       "price", "compare_at_price", "available", "inventory_quantity"]}
            product_rows.append(row)
            if score:
                technical.append(row)
        technical.sort(key=lambda row: (-row["technical_keyword_score"],
                                        row["product_url"]))
        for row in technical[:max_technical_pages_per_domain]:
            page = fetcher.fetch(row["product_url"], "canonical-product-html")
            if page["http_status"] != 200:
                continue
            semantics = page_semantics(fetcher.body(page), row["product_url"], domain)
            product = semantics["jsonld_product"]
            docs = semantics["documentation_links"]
            if product and docs:
                admitted.append({**row, "page_receipt": page,
                                 "canonical_url": semantics["canonical_url"],
                                 "jsonld_product_name": product.get("name"),
                                 "jsonld_sku": product.get("sku"),
                                 "documentation_links": docs})
    receipt_path = root / "receipts/domains" / f"{hashlib.sha256(domain.encode()).hexdigest()}.json"
    atomic_publish(receipt_path, (json.dumps(
        domain_result, indent=2, sort_keys=True) + "\n").encode())
    return {"domain_result": domain_result, "qualified": is_qualified,
            "product_rows": product_rows, "admitted": admitted, "receipt_path": receipt_path}


def seal_run(*, frontier: Path, frontier_receipt: Path, root: Path, artifact_root: Path,
             seeds: list[dict], qualified: list[str], product_rows: list[dict],
             admitted: list[dict], domain_receipts: list[Path]) -> dict:
    """Deduplicate the admitted products by SKU, publish the inventories and outlinks, seal the
    `original` and `outlinks` stages and write `RUN.json`."""
    # Product variants/legacy handles are deduplicated by authoritative SKU.
    by_identity = {}
    for row in admitted:
        identity = str(row.get("jsonld_sku") or "").strip() or (
            row["skus"][0] if row["skus"] else row["canonical_url"])
        current = by_identity.get(identity)
        if current is None or (-row["technical_keyword_score"], row["canonical_url"]) < (
                -current["technical_keyword_score"], current["canonical_url"]):
            by_identity[identity] = row
    admitted = [by_identity[key] for key in sorted(by_identity)]
    products_path = root / "inventory/products.jsonl"
    admitted_path = root / "inventory/admitted-technical-products.jsonl"
    atomic_publish(products_path, b"".join(_json(row) for row in product_rows))
    atomic_publish(admitted_path, b"".join(_json(row) for row in admitted))
    outlinks_path = root / "outlinks/parent-bound.jsonl"
    outlink_rows = []
    for row in admitted:
        outlink_rows.append({"parent_url": row["canonical_url"],
                             "target_url": row["canonical_url"],
                             "relation": "official-product"})
        outlink_rows.extend({"parent_url": row["canonical_url"],
                             "target_url": link["url"],
                             "relation": link["relation"],
                             "context": link["context"]}
                            for link in row["documentation_links"])
    atomic_publish(outlinks_path, b"".join(_json(row) for row in outlink_rows))
    request_receipts = sorted((root / "receipts/requests").rglob("*.json"))
    artifacts = [frontier, frontier_receipt, products_path, admitted_path,
                 *domain_receipts, *request_receipts]
    for receipt_path in request_receipts:
        artifacts.extend(_artifact_paths(artifact_root,
                                         json.loads(receipt_path.read_text())))
    logical = hashlib.sha256(b"".join(_json({
        "uri": f"artifact://{path.relative_to(artifact_root).as_posix()}",
        "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        for path in sorted(set(artifacts)))).hexdigest()
    original = SS.seal(
        artifact_root=artifact_root, artifact_paths=sorted(set(artifacts)),
        receipt_path=root / "stages/original/receipt.json", records=len(request_receipts),
        fmt="query-gap-seeds+agents-md+source-sitemaps+ucp-product-json+canonical-product-html+transport",
        logical_sha256=logical, producer_id=PRODUCER, producer_version=VERSION)
    outlinks = SS.seal(
        artifact_root=artifact_root, artifact_paths=[outlinks_path],
        receipt_path=root / "stages/outlinks/receipt.json", records=len(outlink_rows),
        fmt="parent-bound-official-product-and-technical-documentation-links",
        logical_sha256=hashlib.sha256(outlinks_path.read_bytes()).hexdigest(),
        producer_id=PRODUCER, producer_version=VERSION)
    pidog = [row for row in admitted if "CN0481D" in row["skus"]
             or row.get("jsonld_sku") == "CN0481D"]
    result = {
        "schema": "aw.technical-manufacturer-catalog-discovery-run.v1",
        "version": VERSION, "status": "source_only_complete",
        "selection_policy": "query-gap domain discovery -> any authenticated standard catalog surface (agents.md, well-known UCP, or product sitemap) -> Product JSON-LD + substantive first-party docs/manual/code linkage; no domain allowlist",
        "counts": {"seed_domains": len(seeds), "qualified_catalog_domains": len(qualified),
                   "product_records": len(product_rows),
                   "admitted_unique_products": len(admitted), "outlinks": len(outlink_rows)},
        "qualified_domains": sorted(qualified),
        "pidog_consequence": {"selected": bool(pidog), "records": pidog},
        "volatile_price_stock_policy": "preserved in L0 originals when source supplies it; excluded from durable Markdown/embedding text",
        "stages": {"original": original, "outlinks": outlinks},
        "next": "source-specific full Markdown projection, profile/fidelity/token/UUID gates; no GPU admission from this source-only RUN",
    }
    atomic_publish(root / "RUN.json",
                   (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    return result


def acquire(*, frontier: Path, frontier_receipt: Path, root: Path,
            artifact_root: Path, rate: float, max_products_per_domain: int,
            max_technical_pages_per_domain: int) -> dict:
    if root.exists():
        raise ValueError("technical manufacturer output root must be a fresh immutable root")
    seeds = load_seeds(frontier, frontier_receipt)
    root.mkdir(parents=True)
    fetcher = Fetcher(root=root, artifact_root=artifact_root, rate=rate)
    qualified: list[str] = []
    product_rows: list[dict] = []
    admitted: list[dict] = []
    domain_receipts: list[Path] = []
    try:
        for seed in seeds:
            one = acquire_domain(fetcher, seed, root=root,
                                 max_products_per_domain=max_products_per_domain,
                                 max_technical_pages_per_domain=max_technical_pages_per_domain)
            if one["qualified"]:
                qualified.append(seed["domain"])
            product_rows.extend(one["product_rows"])
            admitted.extend(one["admitted"])
            domain_receipts.append(one["receipt_path"])
    finally:
        fetcher.close()
    return seal_run(frontier=frontier, frontier_receipt=frontier_receipt, root=root,
                    artifact_root=artifact_root, seeds=seeds, qualified=qualified,
                    product_rows=product_rows, admitted=admitted, domain_receipts=domain_receipts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--frontier-receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--max-products-per-domain", type=int, default=1000)
    parser.add_argument("--max-technical-pages-per-domain", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(acquire(
        frontier=args.frontier.resolve(),
        frontier_receipt=args.frontier_receipt.resolve(), root=args.root.resolve(),
        artifact_root=args.artifact_root.resolve(), rate=args.rate,
        max_products_per_domain=args.max_products_per_domain,
        max_technical_pages_per_domain=args.max_technical_pages_per_domain),
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
