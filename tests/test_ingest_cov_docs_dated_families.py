import json

from cytir.sources.cov.engine import (collapse_latest_dated_families, enum_listing,
                             close_fetch_l0, gate_dated_family_winners,
                             inventory_metadata, preferred_pdf_title,
                             register_gate_record, url_in_route_scope)


class FakeL0:
    def __init__(self):
        self.complete = None

    def close(self, complete):
        self.complete = complete


def test_fetch_l0_seals_only_after_frontier_drains():
    drained = FakeL0()
    close_fetch_l0(drained, drained=True)
    assert drained.complete is True

    resumable = FakeL0()
    close_fetch_l0(resumable, drained=False)
    assert resumable.complete is False

    close_fetch_l0(None, drained=True)


def test_opaque_pdf_transport_titles_yield_to_extracted_document_heading():
    markdown = "# de7d6e3e 383b 4ae5 bdaf e629158a9183\n## Empfehlung nach Art. 14 des Öffentlichkeitsgesetzes\nText"
    assert preferred_pdf_title("de7d6e3e 383b 4ae5 bdaf e629158a9183", markdown,
                               "https://backend.example/files/x.pdf") == "Empfehlung nach Art. 14 des Öffentlichkeitsgesetzes"
    assert preferred_pdf_title("Basistext d mit%20Logo", "# Basistext d mit%20Logo\n## Neue Kaufmännische Grundbildung\nText",
                               "https://example/record.pdf") == "Neue Kaufmännische Grundbildung"
    assert preferred_pdf_title("Human supplied title", markdown, "https://example/x.pdf") == "Human supplied title"
    assert preferred_pdf_title("content", "## Rauch- und Wärmeabzugsanlagen\nText",
                               "https://backend.example/85.pdf/content") == "Rauch- und Wärmeabzugsanlagen"


class ListingResponse:
    status_code = 200
    url = "https://www.bsvonline.ch/de/list"

    def __init__(self, text):
        self.text = text
        self.content = text.encode()
        self.headers = {"content-type": "text/html"}


class ListingClient:
    def __init__(self, text):
        self.text = text

    def get(self, _url):
        return ListingResponse(self.text)


def test_listing_named_groups_retain_authoritative_title(monkeypatch):
    monkeypatch.setattr("cytir.sources.cov.engine.time.sleep", lambda _seconds: None)
    text = ('&quot;title&quot;:[0,&quot;c - Richtlinien&quot;],'
            '&quot;fileLeafRef&quot;:[0,&quot;BSPUB-85.pdf&quot;],'
            '&quot;title&quot;:[0,&quot;16-15 Flucht- und Rettungswege&quot;],'
            '&quot;vkgPubVersion&quot;:[0,&quot;3&quot;],'
            '&quot;dokumentUrl&quot;:[0,&quot;https://services2.vkf.ch/rest/'
            'public/georg/bs/publikation/documents/BSPUB-85.pdf/content&quot;]')
    cfg = {
        "doc_re": (r'(?s)&quot;fileLeafRef&quot;:\[0,&quot;[^&]*?&quot;\],'
                   r'&quot;title&quot;:\[0,&quot;(?P<title>.*?)&quot;\]'
                   r'.{0,5000}?&quot;dokumentUrl&quot;:\[0,&quot;(?P<url>'
                   r'https://services2\.vkf\.ch/rest/public/georg/bs/'
                   r'publikation/documents/BSPUB-[0-9-]+\.pdf/content)'
                   r'&quot;\]'),
        "seeds": ["https://www.bsvonline.ch/de/list"],
        "host": "www.bsvonline.ch",
        "ok_hosts": ["services2.vkf.ch"],
        "delay": 0,
        "lang_default": "en",
    }
    assert enum_listing(ListingClient(text), cfg, None, lambda _msg: None) == [{
        "url": ("https://services2.vkf.ch/rest/public/georg/bs/publikation/"
                "documents/BSPUB-85.pdf/content"),
        "kind": "html",
        "lang": "en",
        "title": "16-15 Flucht- und Rettungswege",
    }]


def test_inventory_metadata_supports_resumed_gate(tmp_path):
    inv = tmp_path / "inv"
    inv.mkdir()
    (inv / "vkf.jsonl").write_text(json.dumps({
        "url": "https://example.test/85.pdf/content",
        "title": "16-15 Flucht- und Rettungswege",
        "lang": "de",
    }) + "\n")
    assert inventory_metadata(str(tmp_path)) == {
        "https://example.test/85.pdf/content": {
            "title": "16-15 Flucht- und Rettungswege", "lang": "de"}}


def row(url):
    return {"url": url, "kind": "html", "lang": "en"}


def test_latest_stable_collapses_dated_and_draft_logical_family():
    prefix = "https://modelcontextprotocol.io/docs/"
    values = [
        row(prefix + "2024-11-05/tutorials/security/security_best_practices"),
        row(prefix + "2025-11-25/tutorials/security/security_best_practices"),
        row(prefix + "2026-07-28/tutorials/security/security_best_practices"),
        row(prefix + "draft/tutorials/security/security_best_practices"),
        row(prefix + "draft/develop/new-draft-only-page"),
        row("https://modelcontextprotocol.io/community/security"),
    ]
    actual = [item["url"] for item in collapse_latest_dated_families(values)]
    assert actual == [
        prefix + "2026-07-28/tutorials/security/security_best_practices",
        prefix + "draft/develop/new-draft-only-page",
        "https://modelcontextprotocol.io/community/security",
    ]


def test_date_elsewhere_is_not_a_document_version_family():
    values = [row("https://example.test/blog/2025-01-01/release"),
              row("https://example.test/blog/2026-01-01/release")]
    assert collapse_latest_dated_families(values) == values


def test_gate_resume_reapplies_route_scope_and_family_policy(tmp_path):
    cfg = {"include": r"^https://example\.test/docs/agent(?:/|$)",
           "dated_family_policy": "latest_stable"}
    assert url_in_route_scope("https://example.test/docs/agent/guide", cfg)
    assert not url_in_route_scope("https://example.test/blog/agent", cfg)

    docs = tmp_path / "docs.jsonl"
    docs.write_text("\n".join(json.dumps(value) for value in [
        {"source": "mcp", "url": "https://example.test/docs/agent/2025-01-01/x"},
        {"source": "mcp", "url": "https://example.test/docs/agent/2026-01-01/x"},
    ]) + "\n")
    # The generic family syntax intentionally begins directly below /docs or
    # /specification. This path is scoped but is not a version family.
    assert gate_dated_family_winners(str(docs), {"sources": {"mcp": cfg}}) == {
        "mcp": {
            "https://example.test/docs/agent/2025-01-01/x",
            "https://example.test/docs/agent/2026-01-01/x",
        }
    }


def test_gate_route_scope_keeps_only_reviewed_harvested_pdf_assets():
    cfg = {
        "include": r"^https://www\.ech\.ch/(de|fr)/",
        "pdf_include": r"(?i)^https://www\.ech\.ch/sites/default/files/.*\.pdf$",
    }
    # eCH's page route is DE/FR, while the reviewed document assets are served
    # from a shared files path.  Both are intentional route members.
    assert url_in_route_scope("https://www.ech.ch/de/ech/ech-0003/11.0.0", cfg)
    assert url_in_route_scope(
        "https://www.ech.ch/sites/default/files/dosvers/eCH-0003.pdf", cfg)
    # The asset exception is narrow: neither an arbitrary same-host page nor
    # a non-PDF file can bypass the declared page route.
    assert not url_in_route_scope("https://www.ech.ch/about/board", cfg)
    assert not url_in_route_scope(
        "https://www.ech.ch/sites/default/files/dosvers/eCH-0003.docx", cfg)


def test_gate_deduplicates_identical_shared_identity_before_staging():
    rec = {"url": "https://example.test/a", "title": "A", "anchor": None,
           "text": "A real authoritative paragraph.", "source_type": "official",
           "provenance": "fixture"}
    seen = {}
    assert register_gate_record(seen, rec)
    assert not register_gate_record(seen, dict(rec))
    changed = dict(rec, provenance="different-payload")
    try:
        register_gate_record(seen, changed)
    except RuntimeError as exc:
        assert "non-identical records share point identity" in str(exc)
    else:
        raise AssertionError("identity collision must fail loud")


def test_gate_resume_selects_latest_mcp_family_without_loading_bodies(tmp_path):
    docs = tmp_path / "docs.jsonl"
    docs.write_text("\n".join(json.dumps(value) for value in [
        {"source": "mcp", "url": "https://mcp.test/docs/2025-01-01/security",
         "html": "large predecessor body"},
        {"source": "mcp", "url": "https://mcp.test/docs/2026-01-01/security",
         "html": "large current body"},
        {"source": "mcp", "url": "https://mcp.test/docs/draft/new-page",
         "html": "draft-only body"},
    ]) + "\n")
    pack = {"sources": {"mcp": {
        "include": r"^https://mcp\.test/",
        "dated_family_policy": "latest_stable",
    }}}
    assert gate_dated_family_winners(str(docs), pack) == {"mcp": {
        "https://mcp.test/docs/2026-01-01/security",
        "https://mcp.test/docs/draft/new-page",
    }}
