"""The technical product Markdown projection: one admitted product page -> one source-specific Markdown row.

Ported verbatim from the retired `materialize_technical_product_catalog` script (task 2138; the script was
deleted in task 1219, and its `build` wrote the sealed `markdown` stage `technical-product-markdown.v1`
of `admitted-products-v2` on the build box). Only `render_product` and its helpers are ported; the
stage sealing (`build`, `_publish_gzip`, receipts) is not.

NON-VERBATIM LINES: (1) `from cytir.sources.technical_manufacturer.catalog import page_semantics` replaces
`from acquire_technical_manufacturer_catalog import page_semantics` (the same function, moved in task
1401); (2) `canonical_json` comes from `cytir.acquire.artifact_io`, as in the original. Nothing else
differs; `PRODUCER` and `VERSION` stay the frozen labels the sealed rows carry.
"""
from __future__ import annotations

import hashlib

from bs4 import BeautifulSoup

from cytir.acquire.artifact_io import canonical_json
from cytir.sources.technical_manufacturer.catalog import page_semantics



VERSION = "technical-product-markdown.v1"
PRODUCER = "pipeline/materialize_technical_product_catalog.py"
PROFILE = "technical_manufacturer_catalog"
VOLATILE_KEYS = {
    "price", "priceCurrency", "availability", "inventoryLevel", "lowPrice",
    "highPrice", "offerCount", "offers", "inventory_quantity", "available",
    "compare_at_price", "promotion", "discount",
}
TECHNICAL_BODY_SELECTOR = (
    "[class*='description'], [id*='description'], [class*='specification'], "
    "[id*='specification'], [class*='product-details'], [id*='product-details']")
NON_PRODUCT_BODY = ("refund policy", "shipping policy", "privacy policy",
                    "newsletter", "return policy")


def _json_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode()


def _strip_volatile(value: object) -> object:
    if isinstance(value, dict):
        return {key: _strip_volatile(child) for key, child in value.items()
                if key not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_strip_volatile(child) for child in value]
    return value


def _plain_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = canonical_json(value)
    return "\n".join(line.strip() for line in
                      BeautifulSoup(str(value), "html.parser").get_text("\n").splitlines()
                      if line.strip())


def _brand(product: dict) -> str:
    value = product.get("brand")
    if isinstance(value, dict):
        return str(value.get("name") or "").strip()
    return str(value or "").strip()


def _body_technical_description(soup: BeautifulSoup) -> str:
    """Return an explicit source product-description block, never page chrome."""
    values = []
    for node in soup.select(TECHNICAL_BODY_SELECTOR):
        text = "\n".join(line.strip() for line in
                         node.get_text("\n").splitlines() if line.strip())
        folded = text.casefold()
        if (len(text) >= 80 and not any(marker in folded
                                        for marker in NON_PRODUCT_BODY)):
            values.append(text)
    # Prefer the smallest block containing a larger duplicate; it has less UI
    # chrome while preserving the same source-declared product prose.
    values = sorted(set(values), key=lambda value: (len(value), value))
    selected = []
    for value in values:
        if any(value in prior for prior in selected):
            continue
        selected.append(value)
    return "\n\n".join(selected)


def render_product(*, decision: dict, body: bytes) -> dict:
    """Render one already-admitted page without commerce volatility."""
    if decision.get("admitted") is not True:
        raise ValueError("technical product renderer requires admitted input")
    semantics = page_semantics(body, decision["source_url"], decision["domain"])
    product = semantics.get("jsonld_product") or {}
    stable = _strip_volatile(product)
    canonical = semantics.get("canonical_url") or decision["source_url"]
    if (canonical != decision.get("canonical_url")
            or str(product.get("sku") or "") != str(decision.get("sku") or "")):
        raise ValueError("technical product admission/source semantics drifted")
    name = _plain_text(product.get("name"))
    soup = BeautifulSoup(body, "html.parser")
    description = _plain_text(product.get("description"))
    description_origin = "product_jsonld"
    if not description:
        description = _body_technical_description(soup)
        description_origin = "source_html_product_description" if description else "docs_only"
    source_sku = str(product.get("sku") or product.get("mpn") or "").strip()
    stable_identity = source_sku or canonical
    identity_kind = "source_sku_or_mpn" if source_sku else "canonical_product_url"
    manufacturer = _brand(product) or decision["domain"]
    if not name or not stable_identity:
        raise ValueError("technical product lacks stable name/product identity")
    links = semantics.get("documentation_links") or []
    admitted_links = decision.get("documentation_links") or []
    if links != admitted_links or not links:
        raise ValueError("technical product documentation relation drifted")
    sections = [
        f"# {name}",
        f"Manufacturer: {manufacturer}",
        f"Official product page: {canonical}",
    ]
    sections.insert(2, (f"SKU/model: {source_sku}" if source_sku
                        else "SKU/model: not declared by source"))
    category = _plain_text(product.get("category"))
    model = _plain_text(product.get("model") or product.get("mpn"))
    if category:
        sections.append(f"Category: {category}")
    if model and model != source_sku:
        sections.append(f"Source-declared model: {model}")
    if description:
        sections.extend(["## Technical description", description])
    sections.append("## Documentation, manuals, tutorials, and code")
    sections.extend(f"- {str(link.get('context') or 'Documentation').strip()}: {link['url']}"
                    for link in links)
    markdown = "\n\n".join(section for section in sections if section).strip() + "\n"
    # Volatile values may remain in exact L0 bytes but cannot leak to durable text.
    for key in VOLATILE_KEYS:
        value = product.get(key)
        if value is not None and str(value).strip() and str(value).strip() in markdown:
            raise ValueError(f"volatile product field leaked into Markdown: {key}")
    document_id = hashlib.sha256(canonical.encode()).hexdigest()
    source_links = [{"url": canonical, "relation": "official-product"}, *[
        {"url": link["url"], "relation": link["relation"],
         "context": link.get("context", "")} for link in links]]
    lang = str((soup.html or {}).get("lang") or "und").split("-", 1)[0].casefold()
    return {
        "document_id": document_id,
        "source_record_id": stable_identity,
        "source_profile": PROFILE,
        "url": canonical,
        "lang": lang,
        "title": name,
        "publisher": manufacturer,
        "markdown": markdown,
        "markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
        "source_links": source_links,
        "parent_identity": {
            "terminal_receipt": decision["terminal_receipt"],
            "body_artifact": decision["body_artifact"],
            "sku": source_sku or None,
        },
        "product": {
            "name": name, "manufacturer": manufacturer,
            "sku": source_sku or None,
            "sku_status": "source_declared" if source_sku else "unavailable",
            "stable_product_identity": stable_identity,
            "identity_kind": identity_kind,
            "canonical_url": canonical, "stable_jsonld": stable,
            "documentation_links": links,
            "technical_description_status": description_origin,
            "volatile_fields_observed": decision.get(
                "volatile_jsonld_fields_observed", []),
        },
        "extraction": {
            "producer": PRODUCER, "version": VERSION,
            "full_prechunk_content": True, "content_cap": None,
            "volatile_commerce_fields_excluded_from_markdown": True,
        },
    }

