import hashlib
import json
from pathlib import Path

import cytir.sources.technical_manufacturer.catalog_frontier as subject
import cytir.sources.technical_manufacturer.seeds as SEEDS
from cytir.acquire import preserved_stage_io as PS
from cytir.acquire import stage_receipt as SS
from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _seal_surfaces(*, domain_frontier, domain_receipt, frontier, receipt,
                   artifact_root):
    """Fixture-only: the sealed standard-surface frontier the builder consumes."""
    rows = []
    for line in domain_frontier.open():
        if not line.strip():
            continue
        d = json.loads(line)
        for surface, path in (("agents-md", "/agents.md"),
                              ("ucp-well-known", "/.well-known/ucp"),
                              ("sitemap-index", "/sitemap.xml")):
            rows.append({"ordinal": len(rows),
                         "url": f"https://{d['domain']}{path}",
                         "domain": d["domain"], "surface": surface,
                         "source_refs": d["evidence"]})
    encoded = b"".join((canonical_json(r) + "\n").encode() for r in rows)
    atomic_publish(frontier, encoded)
    loc = lambda p: "artifact://" + p.resolve().relative_to(
        artifact_root.resolve()).as_posix()
    value = {
        "schema": "aw.exact-url-frontier-receipt.v1", "status": "complete",
        "source": {"uri": loc(domain_frontier),
                   "sha256": sha256_file(domain_frontier),
                   "receipt_sha256": sha256_file(domain_receipt),
                   "domains": len(rows) // 3},
        "result": {"frontier": {"uri": loc(frontier), "bytes": len(encoded),
                                "sha256": hashlib.sha256(encoded).hexdigest(),
                                "urls": len(rows)}},
    }
    atomic_publish(receipt, (json.dumps(value, indent=2, sort_keys=True)
                             + "\n").encode())


def _fixture(tmp_path: Path, *, agents: bytes, sitemap: bytes):
    source = tmp_path / "queries.jsonl"
    source.write_text(json.dumps({
        "id": "q1", "query": "robotics kit",
        "preferred_urls": ["https://maker.test/blog"],
    }) + "\n")
    domain_frontier = tmp_path / "domains.jsonl"
    domain_receipt = tmp_path / "domains-receipt.json"
    SEEDS.derive(source=source, frontier=domain_frontier,
                 receipt=domain_receipt, artifact_root=tmp_path)
    surface_frontier = tmp_path / "surfaces.jsonl"
    surface_receipt = tmp_path / "surfaces-receipt.json"
    _seal_surfaces(domain_frontier=domain_frontier,
                   domain_receipt=domain_receipt, frontier=surface_frontier,
                     receipt=surface_receipt, artifact_root=tmp_path)
    cc_root = tmp_path / "cc"
    cc_root.mkdir()
    artifacts = []
    bodies = {"agents-md": agents, "ucp-well-known": b"",
              "sitemap-index": sitemap}
    for row in (json.loads(line) for line in surface_frontier.open()):
        body = bodies[row["surface"]]
        stored = PS.store_artifact(tmp_path, "cc/body", "bin", body or b"missing")
        receipt = {
            "schema": "aw.exact-url-cc-capture-receipt.v1",
            "status": "complete", "source_url": row["url"],
            "artifacts": {"decoded_html": stored},
        }
        path = cc_root / "receipts" / (
            hashlib.sha256(row["url"].encode()).hexdigest() + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_json(receipt))
        artifacts.extend((path, PS.artifact_path(tmp_path, stored)))
    stage = SS.seal(
        artifact_root=tmp_path, artifact_paths=sorted(set(artifacts)),
        receipt_path=cc_root / "stages/original/receipt.json", records=3,
        fmt="fixture", logical_sha256="a" * 64,
        producer_id="fixture", producer_version="v1")
    (cc_root / "RUN.json").write_text(json.dumps({
        "schema": "aw.exact-url-cc-preserved-run.v1", "status": "complete",
        "stages": {"original": stage},
    }))
    return domain_frontier, domain_receipt, surface_frontier, surface_receipt, cc_root


def test_product_sitemap_qualifies_without_agents_and_is_global_portable(tmp_path: Path):
    sitemap = b'''<?xml version="1.0"?><sitemapindex>
      <sitemap><loc>https://maker.test/sitemap_products_1.xml</loc></sitemap>
      <sitemap><loc>https://maker.test/sitemap_blogs_1.xml</loc></sitemap>
    </sitemapindex>'''
    args = _fixture(tmp_path, agents=b"missing", sitemap=sitemap)
    result = subject.build(
        domain_frontier=args[0], domain_receipt=args[1],
        surface_frontier=args[2], surface_receipt=args[3], cc_root=args[4],
        direct_root=None, root=tmp_path / "output", artifact_root=tmp_path)
    assert result["counts"] == {
        "seed_domains": 1, "qualified_catalog_domains": 1,
        "product_sitemaps": 1,
    }
    frontier = json.loads((tmp_path / "output/product-sitemap-frontier.jsonl").read_text())
    assert frontier["url"] == "https://maker.test/sitemap_products_1.xml"
    for artifact in result["inputs"] + list(result["artifacts"].values()):
        assert artifact["uri"].startswith("artifact://")
        path = tmp_path / artifact["uri"].removeprefix("artifact://")
        assert path.stat().st_size == artifact["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_agents_catalog_marker_without_product_sitemap_does_not_invent_products(tmp_path: Path):
    args = _fixture(tmp_path, agents=b"Universal Commerce Protocol /products/{handle}.json",
                    sitemap=b"<urlset></urlset>")
    result = subject.build(
        domain_frontier=args[0], domain_receipt=args[1],
        surface_frontier=args[2], surface_receipt=args[3], cc_root=args[4],
        direct_root=None, root=tmp_path / "output", artifact_root=tmp_path)
    assert result["counts"]["qualified_catalog_domains"] == 1
    assert result["counts"]["product_sitemaps"] == 0
