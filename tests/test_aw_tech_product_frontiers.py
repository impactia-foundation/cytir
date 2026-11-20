import hashlib
import json
from pathlib import Path

from cytir.sources.exact_url import cc_preserved as EXACT
from cytir.ops.audits import technical_product_admission as ADMISSION
import cytir.sources.technical_manufacturer.page_frontier as PAGES
import cytir.sources.technical_manufacturer.sitemap_frontier as SITEMAPS
from cytir.acquire import preserved_stage_io as PS
from cytir.acquire import stage_receipt as SS


def _artifact(root: Path, path: Path) -> dict:
    return {"uri": "artifact://" + path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _host_consent_fixture(root: Path, host: str,
                          affirmative: bool) -> tuple[Path, dict]:
    consent_root = root / ("host-consent-ok" if affirmative else "host-consent-no")
    evidence = PS.store_artifact(root, consent_root.relative_to(root).as_posix() +
        "/consent", "json", (json.dumps({"status": "ok", "search_ok": affirmative}) +
        "\n").encode())
    receipt = consent_root / "receipts" / (hashlib.sha256(host.encode()).hexdigest() + ".json")
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({
        "schema": "aw.frontier-host-consent-receipt.v1",
        "version": "frontier-host-consent.v1", "status": "terminal",
        "host": host, "affirmative_search_consent": affirmative,
        "consent_artifact": evidence}))
    stage = SS.seal(artifact_root=root,
        artifact_paths=[receipt, PS.artifact_path(root, evidence)],
        receipt_path=consent_root / "stages/consent/receipt.json", records=1,
        fmt="fixture", logical_sha256="c" * 64,
        producer_id="fixture", producer_version="v1")
    return consent_root, stage


def test_catalog_to_sitemap_to_canonical_product_frontier(tmp_path: Path):
    catalog_root = tmp_path / "catalog"
    catalog_root.mkdir()
    source = catalog_root / "product-sitemap-frontier.jsonl"
    source.write_text(json.dumps({
        "url": "https://maker.test/sitemap_products_1.xml",
        "domain": "maker.test", "surface": "source-owned-product-sitemap",
        "source_refs": [{"query_id": "q1"}],
    }) + "\n")
    catalog_receipt = catalog_root / "RECEIPT.json"
    catalog_receipt.write_text(json.dumps({
        "schema": "aw.technical-manufacturer-catalog-frontier-receipt.v1",
        "status": "complete",
        "artifacts": {"product_sitemap_frontier": _artifact(tmp_path, source)},
    }))
    sealed_root = tmp_path / "sealed-sitemaps"
    seal = SITEMAPS.prepare(catalog_receipt=catalog_receipt,
                            root=sealed_root, artifact_root=tmp_path)
    sitemap_frontier = sealed_root / "frontier.jsonl"
    sitemap_receipt = sealed_root / "RECEIPT.json"
    rows, _receipt_sha = EXACT.load_frontier(sitemap_frontier, sitemap_receipt)
    assert len(rows) == seal["result"]["frontier"]["urls"] == 1

    url = rows[0]["url"]
    cc_root = tmp_path / "cc"
    body = b'''<?xml version="1.0"?><urlset>
      <url><loc>https://maker.test/products/pidog?srsltid=tracking</loc></url>
      <url><loc>https://maker.test/blog/not-a-product</loc></url>
      <url><loc>https://other.test/products/not-first-party</loc></url>
    </urlset>'''
    stored = PS.store_artifact(tmp_path, "cc/decoded", "xml", body)
    receipt_path = cc_root / "receipts" / (
        hashlib.sha256(url.encode()).hexdigest() + ".json")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps({
        "schema": "aw.exact-url-cc-capture-receipt.v1",
        "status": "complete", "source_url": url,
        "artifacts": {"decoded_source": stored},
    }))
    stage = SS.seal(
        artifact_root=tmp_path,
        artifact_paths=[receipt_path, PS.artifact_path(tmp_path, stored)],
        receipt_path=cc_root / "stages/original/receipt.json", records=1,
        fmt="fixture", logical_sha256="a" * 64,
        producer_id="fixture", producer_version="v1")
    (cc_root / "RUN.json").write_text(json.dumps({
        "schema": "aw.exact-url-cc-preserved-run.v1", "status": "complete",
        "stages": {"original": stage},
    }))
    product_root = tmp_path / "products"
    result = PAGES.build(
        sitemap_frontier=sitemap_frontier, sitemap_receipt=sitemap_receipt,
        cc_root=cc_root, direct_root=None, root=product_root,
        artifact_root=tmp_path)
    product_rows, _ = EXACT.load_frontier(
        product_root / "frontier.jsonl", product_root / "RECEIPT.json")
    assert result["result"]["frontier"]["urls"] == 1
    assert product_rows[0]["url"] == "https://maker.test/products/pidog"
    assert "srsltid" not in product_rows[0]["url"]

    product_url = product_rows[0]["url"]
    product_cc = tmp_path / "product-cc"
    product_body = b'''<html><head>
      <link rel="canonical" href="https://maker.test/products/pidog?srsltid=x">
      <script type="application/ld+json">{"@type":"Product","name":"PiDog Robot Dog","sku":"CN0481D","offers":{"price":"99","availability":"InStock"}}</script>
      </head><body><h1>PiDog robotics kit for Raspberry Pi</h1>
      <a href="https://docs.maker.test/pidog/manual">PiDog manual and documentation</a>
      </body></html>'''
    product_stored = PS.store_artifact(
        tmp_path, "product-cc/decoded", "html", product_body)
    product_receipt = product_cc / "receipts" / (
        hashlib.sha256(product_url.encode()).hexdigest() + ".json")
    product_receipt.parent.mkdir(parents=True, exist_ok=True)
    product_receipt.write_text(json.dumps({
        "schema": "aw.exact-url-cc-capture-receipt.v1",
        "status": "complete", "source_url": product_url,
        "artifacts": {"decoded_source": product_stored},
    }))
    product_stage = SS.seal(
        artifact_root=tmp_path,
        artifact_paths=[product_receipt,
                        PS.artifact_path(tmp_path, product_stored)],
        receipt_path=product_cc / "stages/original/receipt.json", records=1,
        fmt="fixture", logical_sha256="b" * 64,
        producer_id="fixture", producer_version="v1")
    (product_cc / "RUN.json").write_text(json.dumps({
        "schema": "aw.exact-url-cc-preserved-run.v1", "status": "complete",
        "stages": {"original": product_stage},
    }))
    host_consent, host_stage = _host_consent_fixture(tmp_path, "maker.test", True)
    (host_consent / "RUN.json").write_text(json.dumps({
        "schema": "aw.frontier-host-consent-run.v1",
        "version": "frontier-host-consent.v1", "status": "complete",
        "frontier_sha256": hashlib.sha256(
            (product_root / "frontier.jsonl").read_bytes()).hexdigest(),
        "stages": {"consent": host_stage}}))
    admission = ADMISSION.audit(
        frontier=product_root / "frontier.jsonl",
        frontier_receipt=product_root / "RECEIPT.json",
        cc_root=product_cc, direct_root=None,
        host_consent_root=host_consent, root=tmp_path / "admission",
        artifact_root=tmp_path)
    assert admission["counts"]["admitted"] == 1
    assert admission["pidog_consequence"]["selected"] is True
    record = admission["pidog_consequence"]["records"][0]
    assert record["sku"] == "CN0481D"
    assert record["canonical_url"] == "https://maker.test/products/pidog"
    assert record["volatile_jsonld_fields_observed"] == ["availability", "price"]
    outlinks = (tmp_path / "admission/parent-bound-outlinks.jsonl").read_text()
    assert "docs.maker.test/pidog/manual" in outlinks
    assert '"price"' not in outlinks

    denied, denied_stage = _host_consent_fixture(tmp_path, "maker.test", False)
    (denied / "RUN.json").write_text(json.dumps({
        "schema": "aw.frontier-host-consent-run.v1",
        "version": "frontier-host-consent.v1", "status": "complete",
        "frontier_sha256": hashlib.sha256(
            (product_root / "frontier.jsonl").read_bytes()).hexdigest(),
        "stages": {"consent": denied_stage}}))
    rejected = ADMISSION.audit(
        frontier=product_root / "frontier.jsonl",
        frontier_receipt=product_root / "RECEIPT.json",
        cc_root=product_cc, direct_root=None,
        host_consent_root=denied, root=tmp_path / "admission-no-consent",
        artifact_root=tmp_path)
    assert rejected["counts"]["admitted"] == 0
    decision = json.loads((tmp_path / "admission-no-consent/decisions.jsonl").read_text())
    assert "affirmative_search_consent" in decision["reason"]
