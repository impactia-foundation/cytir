import cytir.sources.technical_manufacturer.catalog as subject
import json
from pathlib import Path


def test_product_semantics_requires_jsonld_and_docs_and_canonicalizes_tracking():
    body = b'''<html><head><link rel="canonical" href="/products/robot?utm_source=x">
<script type="application/ld+json">{"@type":"Product","name":"Robot Kit","sku":"R1"}</script>
</head><body><a href="https://docs.example.com/robot?utm_campaign=x">Six-language docs</a></body></html>'''
    value = subject.page_semantics(body, "https://example.com/products/robot?ref=x",
                                   "example.com")
    assert value["canonical_url"] == "https://example.com/products/robot"
    assert value["jsonld_product"]["sku"] == "R1"
    assert value["documentation_links"] == [{
        "url": "https://docs.example.com/robot",
        "context": "Six-language docs",
        "relation": "technical-documentation"}]


def test_non_product_page_is_not_semantically_admissible():
    value = subject.page_semantics(b"<html><a href='/manual'>manual</a></html>",
                                   "https://example.com/blog/x", "example.com")
    assert value["jsonld_product"] is None


def test_product_sitemap_qualifies_without_agents_or_ucp():
    value = subject.qualify_standard_catalog_surfaces(
        agents_status=404, ucp_status=404, sitemap_status=200,
        catalog_markers=[],
        child_sitemaps=["https://maker.test/sitemap_products_1.xml"])
    assert value == {
        "authenticated_standard_catalog_surface": True,
        "product_sitemaps": ["https://maker.test/sitemap_products_1.xml"],
        "qualified": True,
    }


def test_generic_sitemap_without_catalog_surface_does_not_qualify():
    value = subject.qualify_standard_catalog_surfaces(
        agents_status=404, ucp_status=404, sitemap_status=200,
        catalog_markers=[],
        child_sitemaps=["https://maker.test/sitemap_blogs_1.xml"])
    assert value["qualified"] is False


def test_technical_manufacturer_profile_excludes_volatile_fields():
    registry = json.loads((Path(__file__).resolve().parents[1]
                           / "data/source-preservation-profiles.v1.json").read_text())
    profile = registry["profiles"]["technical_manufacturer_catalog"]
    markdown = " ".join(profile["markdown"]["requirements"])
    assert "volatile price" in markdown
    assert profile["chunks"]["boundary"] == "product-model"
