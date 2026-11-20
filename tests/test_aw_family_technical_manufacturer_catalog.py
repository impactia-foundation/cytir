"""`technical_manufacturer_catalog` on the family Protocol (task 1401).

No network, no fixture from live: a synthetic two-domain catalog behind `httpx.MockTransport`, a fixed
clock. The parity pin is the sealed output (every file, sha256) of the pre-move acquire driver on that
same fixture (`tests/data/technical_manufacturer_catalog_parity.v1.json` names the reference run); the
moved module and the adapter path must both seal the same bytes.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import pathlib

import httpx
import pytest

from cytir.sources import contract as contract_mod
from cytir.sources import registry
from cytir.sources.families.technical_manufacturer_catalog import TechnicalManufacturerCatalogFamily
from cytir.sources.family import RunContext, SourceFamily
from cytir.sources.run_family import run_family
from cytir.sources.technical_manufacturer import catalog, urls

PARITY = json.loads((pathlib.Path(__file__).parent / "data"
                     / "technical_manufacturer_catalog_parity.v1.json").read_text())

_SITEMAP_INDEX = (b'<?xml version="1.0"?><sitemapindex><sitemap><loc>https://shop.example/sitemap_products_1.xml'
                  b'</loc></sitemap><sitemap><loc>https://shop.example/sitemap_blogs_1.xml</loc></sitemap></sitemapindex>')
_PRODUCTS = (b'<?xml version="1.0"?><urlset><url><loc>https://shop.example/products/robot-kit?utm_source=x</loc></url>'
             b'<url><loc>https://shop.example/products/mug</loc></url></urlset>')
_PAGE = (b'<html><head><link rel="canonical" href="/products/robot-kit"><script type="application/ld+json">'
         b'{"@type":"Product","name":"Robot Kit","sku":"RK1"}</script></head><body>'
         b'<a href="https://docs.shop.example/robot-kit">Robot kit manual</a></body></html>')


def _handler(request):
    table = {
        "https://shop.example/agents.md": (200, b"Universal Commerce Protocol catalog at /products/{handle}.json"),
        "https://shop.example/sitemap.xml": (200, _SITEMAP_INDEX),
        "https://shop.example/sitemap_products_1.xml": (200, _PRODUCTS),
        "https://shop.example/products/robot-kit.json": (200, json.dumps({"product": {
            "title": "Robot Kit with servo motor", "body_html": "<p>Raspberry Pi sensor</p>",
            "product_type": "Electronics", "variants": [{"sku": "RK1"}]}}).encode()),
        "https://shop.example/products/mug.json": (200, json.dumps({"product": {
            "title": "Mug", "body_html": "ceramic", "product_type": "Home",
            "variants": [{"sku": "M1"}]}}).encode()),
        "https://shop.example/products/robot-kit": (200, _PAGE),
    }
    status, body = table.get(str(request.url), (404, b"not found"))
    return httpx.Response(status, content=body, headers={"content-type": "text/plain"})


class _FixedDT(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 2, 3, 4, 5, tzinfo=tz)


def _seeds(art: pathlib.Path):
    (art / "seeds").mkdir(parents=True)
    rows = [{"domain": "shop.example", "evidence": ["q1"]}, {"domain": "other.example", "evidence": ["q2"]}]
    fr = art / "seeds/frontier.jsonl"
    fr.write_bytes(b"".join((json.dumps(r, sort_keys=True) + "\n").encode() for r in rows))
    rc = art / "seeds/receipt.json"
    rc.write_text(json.dumps({"schema": "aw.technical-manufacturer-seed-receipt.v1",
                              "frontier": {"sha256": hashlib.sha256(fr.read_bytes()).hexdigest(),
                                           "domains": 2}}))
    return fr, rc


def _digests(art: pathlib.Path) -> dict:
    return {p.relative_to(art).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(art.rglob("*")) if p.is_file()}


@pytest.fixture
def clock(monkeypatch):
    monkeypatch.setattr(catalog, "datetime", _FixedDT)


def _row() -> dict:
    return contract_mod.by_id(contract_mod.load())["technical_manufacturer_catalog"]


def _ctx(tmp_path, art, fr, rc, **options) -> RunContext:
    opts = {"frontier": str(fr), "frontier_receipt": str(rc), "root": str(art / "run"),
            "artifact_root": str(art), "rate": "1000", "max_products_per_domain": "10",
            "max_technical_pages_per_domain": "5", **options}
    http = httpx.Client(transport=httpx.MockTransport(_handler))
    return RunContext(out_dir=tmp_path / "out", today="2026-10-30", skip_consent=True, http=http,
                      options=opts, contract=_row(), log=lambda m: None)


def test_registered_and_conformant():
    assert "technical_manufacturer_catalog" in registry.registered()
    adapter, row, spec = registry.admit("technical_manufacturer_catalog")
    assert isinstance(adapter, SourceFamily)
    assert spec.declared == row["chunk_policy"]["policy_id"]
    assert not getattr(adapter, "PARTIAL", ())
    assert adapter.route == row["route"] == "direct" and adapter.direct_crawl is True


def test_moved_module_seals_the_pre_move_bytes(tmp_path, clock, monkeypatch):
    art = tmp_path / "art"
    fr, rc = _seeds(art)
    monkeypatch.setattr(httpx, "HTTPTransport", lambda *a, **k: httpx.MockTransport(_handler))
    result = catalog.acquire(frontier=fr, frontier_receipt=rc, root=art / "run", artifact_root=art,
                             rate=1000.0, max_products_per_domain=10, max_technical_pages_per_domain=5)
    assert result["counts"]["admitted_unique_products"] == 1       # a yield, not an empty run
    assert result["qualified_domains"] == ["shop.example"]
    assert _digests(art) == PARITY["files"]


def test_adapter_path_seals_the_same_bytes(tmp_path, clock):
    art = tmp_path / "art"
    fr, rc = _seeds(art)
    ctx, fam = _ctx(tmp_path, art, fr, rc), TechnicalManufacturerCatalogFamily()
    units = list(fam.discover(ctx))
    assert [u.key for u in units] == ["domain:shop.example", "domain:other.example"]
    for unit in units:
        (cap,) = fam.acquire(unit, ctx)
        assert json.loads(cap.body)["domain"] == unit.meta["seed"]["domain"]
    sealed = fam.finalize(ctx)
    assert sealed["counts"]["admitted_unique_products"] == 1
    assert _digests(art) == PARITY["files"]


def test_plan_is_a_dry_run_over_the_seed_domains(tmp_path):
    art = tmp_path / "art"
    fr, rc = _seeds(art)
    seen = []

    def consent(host, direct):
        seen.append((host, direct))
        return {"host": host, "bucket": "ok"}

    plan = run_family("technical_manufacturer_catalog", out_dir=tmp_path / "o", today="2026-10-30",
                      dry_run=True, consent=consent, log=lambda m: None,
                      options={"frontier": str(fr), "frontier_receipt": str(rc)})
    assert plan["dry_run"] and plan["units"] == ["domain:shop.example", "domain:other.example"]
    assert plan["runnable"] == plan["units"]
    assert sorted(seen) == [("other.example", True), ("shop.example", True)]
    assert not (tmp_path / "o").exists() and not (art / "run").exists()   # a plan writes and fetches nothing


def test_terms_come_from_the_contract_row(tmp_path):
    art = tmp_path / "art"
    fr, rc = _seeds(art)
    ctx = _ctx(tmp_path, art, fr, rc)
    TechnicalManufacturerCatalogFamily().discover(ctx)
    t, row = ctx.notes["contract_terms"], _row()
    assert t["search_ok"] is row["consent"]["search_ok"]
    assert t["train_ok"] is row["consent"]["train_ok"]
    assert t["consent_resolution"] == row["consent"]["resolution"] == "live-per-origin"
    assert t["licence_basis"] == row["licence"]["basis"] == "instrument"
    bare = RunContext(out_dir=tmp_path, today="2026-10-30", contract={}, options=ctx.options)
    with pytest.raises(ValueError, match="contract row"):
        TechnicalManufacturerCatalogFamily().discover(bare)


def test_an_unsealed_frontier_refuses(tmp_path):
    art = tmp_path / "art"
    fr, rc = _seeds(art)
    fr.write_bytes(fr.read_bytes() + b'{"domain": "x.example", "evidence": []}\n')
    with pytest.raises(ValueError, match="not sealed"):
        TechnicalManufacturerCatalogFamily().discover(_ctx(tmp_path, art, fr, rc))


def test_manufacturer_url_has_one_home():
    assert urls.manufacturer_url("http://www.Shop.example//a?utm_x=1&b=2#f") == "https://shop.example/a?b=2"
    assert catalog.manufacturer_url is urls.manufacturer_url


def test_default_http_client_keeps_the_crawler_identity(tmp_path):
    """run_family builds no client (ctx.http is None unless a caller injects one), so the
    adapter's Fetcher builds the pre-port driver's own: crawler UA, Accept, 60 s, redirects."""
    ctx = RunContext(out_dir=tmp_path, today="2026-10-30")
    assert ctx.http is None
    fetcher = catalog.Fetcher(root=tmp_path / "a" / "r", artifact_root=tmp_path / "a", rate=1.0,
                              client=ctx.http)
    try:
        c = fetcher.client
        assert c.headers["User-Agent"] == "authoritative-web/0.1 (+https://websear.ch)"
        assert c.headers["Accept"] == ("text/html,application/json,application/xml,"
                                       "text/markdown;q=0.9,*/*;q=0.5")
        assert c.timeout == httpx.Timeout(60)
        assert c.follow_redirects is True
    finally:
        fetcher.close()


# --- extract + chunk on the Protocol (task 2138) -----------------------------------------------

_PRODUCT_PAGE = (
    b'<html lang="en-GB"><head><link rel="canonical" href="/products/robot-kit">'
    b'<script type="application/ld+json">{"@type":"Product","name":"Robot Kit","sku":"RK1",'
    b'"description":"A servo robot kit with a Raspberry Pi header.","category":"Robotics",'
    b'"brand":{"name":"ShopCo"},"offers":{"price":"49.00"}}</script></head><body>'
    b'<a href="https://docs.shop.example/robot-kit">Robot kit manual</a></body></html>')


def _product_cap():
    from cytir.sources.family import Captured, Unit
    url = "https://shop.example/products/robot-kit"
    page = catalog.page_semantics(_PRODUCT_PAGE, url, "shop.example")
    decision = {"admitted": True, "source_url": url, "domain": "shop.example",
                "canonical_url": page["canonical_url"], "sku": "RK1",
                "documentation_links": page["documentation_links"],
                "terminal_receipt": {"uri": "artifact://t", "sha256": "0" * 64, "bytes": 1},
                "body_artifact": {"uri": "artifact://b", "sha256": "1" * 64, "bytes": 1},
                "volatile_jsonld_fields_observed": ["offers"]}
    return Captured(Unit(key="p", url=url), _PRODUCT_PAGE, {"decision": decision})


def _chunk_ctx(tmp_path) -> RunContext:
    return RunContext(out_dir=tmp_path / "out", today="2026-10-31", skip_consent=True,
                      contract=_row(), log=lambda m: None)


def test_extract_projects_the_admitted_page_to_product_markdown(tmp_path):
    fam = TechnicalManufacturerCatalogFamily()
    (doc,) = fam.extract(_product_cap(), _chunk_ctx(tmp_path))
    assert doc.text.startswith("# Robot Kit\n\nManufacturer: ShopCo\n\nSKU/model: RK1\n\n"
                               "Official product page: https://shop.example/products/robot-kit\n\n")
    assert "## Technical description\n\nA servo robot kit" in doc.text
    assert "- Robot kit manual: https://docs.shop.example/robot-kit" in doc.text
    assert "49.00" not in doc.text and doc.lang == "en" and doc.title == "Robot Kit"
    row = doc.extra["row"]
    from cytir.sources.technical_manufacturer import product_markdown
    assert row["extraction"]["producer"] == product_markdown.PRODUCER   # the frozen sealed label
    assert row["extraction"]["version"] == "technical-product-markdown.v1"
    assert row["product"]["volatile_fields_observed"] == ["offers"]
    assert row["document_id"] == hashlib.sha256(b"https://shop.example/products/robot-kit").hexdigest()


def test_extract_refuses_a_page_without_its_admission_decision(tmp_path):
    from cytir.sources.family import Captured, Unit
    with pytest.raises(ValueError, match="admission decision"):
        TechnicalManufacturerCatalogFamily().extract(
            Captured(Unit(key="p", url="https://shop.example/"), b"<html/>", {}), _chunk_ctx(tmp_path))


def test_chunk_runs_under_the_declared_policy_with_the_stage_anchor(tmp_path, word_only_chunking_allowed):
    fam, ctx = TechnicalManufacturerCatalogFamily(), _chunk_ctx(tmp_path)
    (doc,) = fam.extract(_product_cap(), ctx)
    with pytest.raises(RuntimeError, match="before preflight"):
        fam.chunk(doc, ctx)
    fam.preflight(ctx)
    recs = fam.chunk(doc, ctx)
    row = doc.extra["row"]
    assert recs and [r["chunk_ordinal"] for r in recs] == list(range(len(recs)))
    assert all(r["anchor"] == f"document:{row['document_id']}:chunk-{r['chunk_ordinal']}" for r in recs)
    assert all(r["stamps"]["chunk"] == 4 for r in recs)
    assert all(r["parent_markdown_sha256"] == row["markdown_sha256"] for r in recs)
    assert recs[0]["source_metadata"].keys() == {"product", "publisher", "title"}
    # the sealed stage rows hold none of these: they are absent, never invented
    assert not {"tier", "source_type", "trust_tier"} & recs[0].keys()
