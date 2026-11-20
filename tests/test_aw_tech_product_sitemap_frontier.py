import hashlib
import json
from pathlib import Path

import pytest

from cytir.sources.technical_manufacturer.sitemap_frontier import prepare


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_terminal_frontier_is_verified_and_resumable(tmp_path: Path):
    source = tmp_path / "catalog" / "product-sitemaps.jsonl"
    source.parent.mkdir()
    body = b'{"url":"https://maker.example/products-sitemap.xml"}\n'
    source.write_bytes(body)
    catalog = tmp_path / "catalog" / "RECEIPT.json"
    catalog.write_text(json.dumps({
        "schema": "aw.technical-manufacturer-catalog-frontier-receipt.v1",
        "status": "complete",
        "artifacts": {"product_sitemap_frontier": {
            "uri": "artifact://catalog/product-sitemaps.jsonl",
            "bytes": len(body), "sha256": _sha(body),
        }},
    }))
    root = tmp_path / "frontier"

    first = prepare(catalog_receipt=catalog, root=root, artifact_root=tmp_path)
    second = prepare(catalog_receipt=catalog, root=root, artifact_root=tmp_path)
    assert second == first

    (root / "frontier.jsonl").write_bytes(body + b"{}\n")
    with pytest.raises(ValueError, match="identity drift"):
        prepare(catalog_receipt=catalog, root=root, artifact_root=tmp_path)
