#!/usr/bin/env python3
"""Offline unit test for the unified extractor aw_common.process_html (ADR-020).

Validates that ONE HTML parse yields both the endorsement link path and the
content passage path, that the BOILERPLATE / same-domain filter is applied, and
that the two consuming scripts (cc_ingest, extract_outlinks) drive it correctly.

  python3 -m pytest tests/test_process_html.py

Needs only bs4 for the link path; the content path additionally needs
trafilatura (auto-skipped if absent).
"""
from __future__ import annotations

import os
import sys

# Offline unit test: the lid.176 model isn't present here, so the language
# scope-gate is inactive. Opt into degraded mode explicitly (URL-heuristic
# fallback) rather than tripping the task-238 hard-fail — this is exactly the
# documented AW_LANGGATE_OPTIONAL contract for offline tooling/tests.
os.environ.setdefault("AW_LANGGATE_OPTIONAL", "1")

import cytir.core.common as ac

PASS, FAIL = [], []


def chk(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("  ok   " if cond else " FAIL ") + name)


SEED = "hug.ch"
PAGE = "https://www.hug.ch/some/page"
HTML = """
<html><head><title>HUG — guideline</title></head><body>
<main>
  <h1>Clinical guideline on something</h1>
  <p>This is a reasonably long paragraph of authoritative content that should be
     extracted by trafilatura as the main passage text for grounding. It needs
     to be long enough to clear the minimum length and quality gates that the
     pipeline applies, so we keep writing a few more words here to be safe and
     ensure the chunker emits at least one passage of useful prose.</p>
  <p>A second paragraph also carrying real prose so the extractor has enough
     substance to keep and chunk into a passage for the index, well beyond the
     fifteen word floor used by the chunker downstream in the build.</p>
  <a href="https://www.planetesante.ch/article">external endorsement</a>
  <a href="https://addictionsuisse.ch/page">another endorsement</a>
  <a href="https://www.hug.ch/other">internal (same seed) — drop</a>
  <a href="https://www.facebook.com/hug">social chrome — drop</a>
  <a href="#section">fragment — drop</a>
  <a href="mailto:info@hug.ch">mailto — drop</a>
</main>
</body></html>
"""


def test_links():
    _, links = ac.process_html(PAGE, HTML,
                               {"content": False, "links": True, "seed_reg": "hug.ch"})
    chk("external endorsement kept", "https://www.planetesante.ch/article" in links)
    chk("second endorsement kept", "https://addictionsuisse.ch/page" in links)
    chk("internal same-seed dropped",
        not any("hug.ch" in u for u in links))
    chk("social boilerplate dropped",
        not any("facebook" in u for u in links))
    chk("fragment/mailto excluded",
        not any(u.startswith(("#", "mailto:")) for u in links))


def test_links_default_seed_reg():
    # seed_reg derived from the url when not supplied
    _, links = ac.process_html(PAGE, HTML, {"content": False, "links": True})
    chk("derived seed_reg drops internal",
        not any("hug.ch" in u for u in links))


def test_short_html_noop():
    p, l = ac.process_html(PAGE, "<html></html>", {"content": True, "links": True})
    chk("too-short html → empty", p == [] and l == set())


def test_content():
    try:
        import trafilatura  # noqa: F401
        from cytir.acquire import fetch_process  # noqa: F401
    except Exception:
        print("  skip content path (trafilatura/fetch_process absent)")
        return
    common = dict(domain="hug.ch", tier="A", trust_tier=1.0,
                  source_type="official", last_seen="2026-06-28")
    passages, links = ac.process_html(PAGE, HTML, {
        "content": True, "links": True, "seed_reg": "hug.ch",
        "passage_common": common, "lang_default": "fr", "digest": "abc123"})
    chk("content path yields >=1 passage", len(passages) >= 1)
    if passages:
        p = passages[0]
        chk("passage url set", p.url == PAGE)
        chk("passage carries common fields", p.domain == "hug.ch" and p.tier == "A")
        chk("passage digest forwarded", p.content_digest == "abc123")
        chk("passage snippet non-empty", bool(p.snippet))
    chk("one-pass also returns links", "https://www.planetesante.ch/article" in links)


def test_outbound_links_helper():
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(HTML, "html.parser")
    links = ac.outbound_links(soup, PAGE, "hug.ch")
    chk("outbound_links helper filters correctly",
        "https://addictionsuisse.ch/page" in links
        and not any("hug.ch" in u for u in links))


def test_reg_domain():
    chk("reg_domain 2-label", ac.reg_domain("www.hug.ch") == "hug.ch")
    chk("reg_domain 2-level eTLD", ac.reg_domain("foo.gov.uk") == "foo.gov.uk")


def test_extraction_junk():
    """task 097 guard: machine-content classes must be rejected; prose kept."""
    # fedpol.report class: embedded JSON search-index leaked as content
    j = ('[{"id":4856,"permalink":"\\/de\\/report-2025\\/editorial\\/",'
         '"breadcrumbs":[{"title":"Report 2025","permalink":"\\/de\\/"}],'
         '"title":"Editorial","lead":"Liebe Leserin","content":"' + "lorem " * 200 + '"}]')
    chk("json_dump caught (envelope)", ac.extraction_junk(j) == "json_dump")
    mid = ('und dennoch schwer zu fassen","permalink":"\\/de\\/report\\/a\\/",'
           '"next":"\\/de\\/b\\/"' + ',"p":"\\/de\\/x\\/"' * 10)
    chk("json_dump caught (mid-chunk \\/)", ac.extraction_junk(mid) == "json_dump")
    # southampton class: sitemap URL list
    urls = "\n".join(f"https://example.ac.uk/courses/module-{i}" for i in range(40))
    chk("url_list caught", ac.extraction_junk(urls) == "url_list")
    # BFS class: whitespace/comma shell and i18n placeholder
    chk("no_prose caught", ac.extraction_junk("DE\n" + " , \n  ,  \n" * 20) == "no_prose")
    ph = ("Momentan können keine Daten angezeigt werden. "
          "Momentan können keine Daten angezeigt werden.")
    chk("shell placeholder caught", ac.extraction_junk(ph) == "shell_boilerplate")
    lic = ("Utilisation libre. Obligation d’indiquer la source. Utilisation "
           "commerciale uniquement avec l’autorisation du fournisseur des données.\n\n"
           "Utilisation libre. Obligation d’indiquer la source. Utilisation "
           "commerciale uniquement avec l’autorisation du fournisseur des données.")
    chk("license placeholder caught", ac.extraction_junk(lic) == "shell_boilerplate")
    # breadcrumb soup
    chk("breadcrumb soup caught",
        ac.extraction_junk("Home > Themen > Daten / de / fr / it | " * 5
                           + "/" * 200) == "breadcrumb")
    # legit prose — incl. a page that merely QUOTES a placeholder — passes
    prose = ("Die Bundesverfassung ist die Verfassung der Schweizerischen "
             "Eidgenossenschaft. Sie wurde am 18. April 1999 von Volk und "
             "Ständen angenommen und ist seit dem 1. Januar 2000 in Kraft. "
             "Sie ersetzt die Bundesverfassung von 1874 und gliedert sich in "
             "sechs Titel mit insgesamt 197 Artikeln zu Grundrechten und mehr.")
    chk("plain prose passes", ac.extraction_junk(prose) is None)
    chk("prose quoting a placeholder passes",
        ac.extraction_junk(prose + " Momentan können keine Daten angezeigt "
                           "werden, meldet die Seite.") is None)
    chk("markdown link-table passes",
        ac.extraction_junk("| Jahr | Satz |\n| 2024 | [8.1%](https://estv."
                           "admin.ch/mwst) gilt für die meisten Lieferungen "
                           "und Dienstleistungen in der Schweiz |") is None)


def test_listing_shell():
    """task 167 gate: tag/category/archive/pagination shells (URL pattern AND
    listing-boilerplate title) are rejected; real content hubs are kept."""
    # URL-pattern detector: listing paths match, article/hub paths don't.
    chk("tag path detected",
        ac.listing_shell_url("https://mlflow.org/articles/tags/ai-testing-tools")
        == "tag")
    chk("category path detected",
        ac.listing_shell_url("https://blog.example/category/news/") == "category")
    chk("label path detected",
        ac.listing_shell_url("https://x.blogspot.com/search/label/python") == "label")
    chk("archive path detected",
        ac.listing_shell_url("https://blog.example/archives/2025/06") == "archive")
    chk("pagination path detected",
        ac.listing_shell_url("https://blog.example/page/3") == "page")
    chk("query pagination detected",
        ac.listing_shell_url("https://blog.example/?page=4") == "query-page")
    chk("article path NOT a listing",
        ac.listing_shell_url("https://mlflow.org/articles/the-role-of-api-gateway")
        is None)
    chk("'/tagliatelle' not a tag path",
        ac.listing_shell_url("https://food.example/tagliatelle-recipe") is None)
    chk("'/topical' not a topic path",
        ac.listing_shell_url("https://x.example/topical-guide") is None)
    # Title-boilerplate detector.
    chk("'One post tagged' title is boilerplate",
        ac.listing_shell_title('One post tagged with "AI testing tools" | MLflow'))
    chk("'Category:' title is boilerplate",
        ac.listing_shell_title("Category: News | Example Blog"))
    chk("'Page N of M' title is boilerplate",
        ac.listing_shell_title("News — Page 2 of 9"))
    chk("real article title is NOT boilerplate",
        not ac.listing_shell_title("The Role of API Gateway AI Services in 2026"))
    chk("'Archive :' colon title IS boilerplate",
        ac.listing_shell_title("Phys.org | Archive : 05/10/2022"))
    chk("'Archives - <real title>' dash is NOT boilerplate (canada.ca lesson)",
        not ac.listing_shell_title(
            "Archives - Guide de la détermination de l'admissibilité - Chapitre 10"))
    chk("'Archive Deployment - API Reference' is NOT boilerplate",
        not ac.listing_shell_title("Archive Deployment - Claude API Reference"))
    # Combined gate requires BOTH (protects real content hubs on listing paths).
    chk("tag path + boilerplate title → listing_shell",
        ac.listing_shell("https://mlflow.org/articles/tags/ai-testing-tools",
                         'One post tagged with "AI testing tools" | MLflow')
        == "listing_shell")
    chk("topics hub path + real title → KEPT (not a shell)",
        ac.listing_shell("https://github.com/topics/machine-learning",
                         "machine-learning · GitHub Topics") is None)
    chk("article path + boilerplate-ish title → KEPT (path not a listing)",
        ac.listing_shell("https://mlflow.org/articles/the-role-of-api-gateway",
                         "The Role of API Gateway AI Services") is None)
    # task 349 / probe 2026-07-28-postcut: the Bund-CMS /newnsb rolling
    # news-list shells (identical teaser boilerplate across ≥12 admin.ch
    # subdomains). Conjunction: /newnsb path + bare news-list label title.
    chk("/newnsb + 'News' title → listing_shell",
        ac.listing_shell("https://www.aboutswitzerland.eda.admin.ch/de/newnsb",
                         "News") == "listing_shell")
    chk("/newnsb + 'Aktuell | BAG' title → listing_shell",
        ac.listing_shell("https://www.bag.admin.ch/de/newnsb",
                         "Aktuell | BAG") == "listing_shell")
    chk("/newnsb + real prose title → KEPT (conjunction)",
        ac.listing_shell("https://x.example/de/newnsb",
                         "Bundesrat verabschiedet Botschaft zur AHV-Reform")
        is None)
    chk("'News' title on a non-newnsb path → KEPT",
        ac.listing_shell("https://x.example/de/aktuelles-projekt",
                         "News") is None)
    chk("'/newnsbla' does not match the newnsb segment",
        ac.listing_shell_url("https://x.example/newnsbla") is None)


def test_sitemap_shell():
    """task 184 gate: sitemap/link-dump pages (sitemap-family URL AND
    sitemap-boilerplate title), plus Apache autoindex titles; real pages
    ABOUT sitemaps (docs, glossary, wiki, tools) are kept."""
    # The trigger: evernote.com/sitemap ranked top-5 on three unrelated live
    # queries (18,964 anchors, anchor-text ratio 0.994 — extraction_junk-proof).
    chk("evernote.com/sitemap → sitemap_shell",
        ac.listing_shell("https://evernote.com/sitemap", "Sitemap")
        == "sitemap_shell")
    chk("gov Plan du site → sitemap_shell",
        ac.listing_shell("https://www.kmu.admin.ch/kmu/fr/home/sitemap.html",
                         "Plan du site") == "sitemap_shell")
    chk("Inhaltsverzeichnis sitemap.php → sitemap_shell",
        ac.listing_shell("https://www.dresden.de/de/sonstiges/sitemap.php",
                         "Inhaltsverzeichnis") == "sitemap_shell")
    chk("'U.S. Access Board - Site Map' → sitemap_shell",
        ac.listing_shell("https://www.access-board.gov/sitemap.html",
                         "U.S. Access Board - Site Map") == "sitemap_shell")
    chk("html-sitemap segment + Sitemap title → sitemap_shell",
        ac.listing_shell("https://www.caa.co.uk/website-policies/html-sitemap",
                         "Sitemap") == "sitemap_shell")
    chk("generic 'Übersicht' ONLY at exact /sitemap.html → sitemap_shell",
        ac.listing_shell("https://www.sem.admin.ch/sem/de/home/sitemap.html",
                         "Übersicht") == "sitemap_shell")
    chk("'Elastic Sitemap' suffix at exact /sitemap → sitemap_shell",
        ac.listing_shell("https://www.elastic.co/sitemap", "Elastic Sitemap")
        == "sitemap_shell")
    chk("A-Z index page → sitemap_shell",
        ac.listing_shell("https://www.utah.edu/a-z/", "A-Z Index")
        == "sitemap_shell")
    chk("'Public holidays A-Z' at /a-z.php → sitemap_shell",
        ac.listing_shell("https://feiertagskalender.ch/a-z.php?hl=en",
                         "Public holidays A-Z") == "sitemap_shell")
    # Apache/nginx autoindex: title-alone signal.
    chk("'Index of /…' autoindex → autoindex_shell",
        ac.listing_shell("https://www.kernel.org/pub/linux/kernel/people/",
                         "Index of /pub/linux/kernel/people/")
        == "autoindex_shell")
    # KEEPS — one signal only, or excluded trees (each is a real indexed page
    # verified in the 184 audit).
    chk("Django sitemaps framework docs KEPT",
        ac.listing_shell("https://docs.djangoproject.com/en/5.2/ref/contrib/"
                         "sitemaps/", "The sitemap framework | Django "
                         "documentation") is None)
    chk("Wikipedia 'Sitemaps' article KEPT (/wiki/ excluded)",
        ac.listing_shell("https://en.wikipedia.org/wiki/Sitemaps", "Sitemaps")
        is None)
    chk("de-wiki 'Sitemap' article KEPT (/wiki/ excluded)",
        ac.listing_shell("https://de.wikipedia.org/wiki/Sitemap", "Sitemap")
        is None)
    chk("MDN Glossary/Site_map KEPT (glossary excluded)",
        ac.listing_shell("https://developer.mozilla.org/en-US/docs/Glossary/"
                         "Site_map", "Site map") is None)
    chk("mural.co sitemap TEMPLATE product page KEPT (title not boilerplate)",
        ac.listing_shell("https://www.mural.co/templates/sitemap",
                         "Sitemap template | Mural") is None)
    chk("sitemap-checker tool page KEPT",
        ac.listing_shell("https://www.blazehive.io/tools/sitemap-checker",
                         "Sitemap Checker | Free | BlazeHive") is None)
    chk("real prose under /sitemap/ path KEPT (kmu impressum lesson)",
        ac.listing_shell("https://www.kmu.admin.ch/kmu/de/home/sitemap/"
                         "impressum.html", "Impressum") is None)
    chk("glossary entry under /a-z/ KEPT",
        ac.listing_shell("https://www.blackwellpublishing.com/ridley/a-z/"
                         "Biological_species_concept_b.asp", "Evolution - A-Z")
        is None)
    chk("article with 'sitemap' inside slug + real title KEPT",
        ac.listing_shell("https://blog.example/wordpress-sitemap-best-practices",
                         "Understanding the WordPress Sitemap") is None)
    chk("generic 'Übersicht' NOT at a sitemap URL KEPT",
        ac.listing_shell("https://www.example.ch/themen/uebersicht.html",
                         "Übersicht") is None)


def test_listing_shell_end2end():
    """The listing-shell gate drops a tag shell inside process_html and counts it."""
    try:
        import trafilatura  # noqa: F401
        from cytir.acquire import fetch_process  # noqa: F401
    except Exception:
        print("  skip listing_shell end2end (trafilatura/fetch_process absent)")
        return
    common = dict(domain="mlflow.org", tier="B", trust_tier=0.6,
                  source_type="blog", last_seen="2026-07-01")
    # A tag shell: listing URL + "One post tagged with…" <title>, real prose body
    # (copied from the underlying article) so extraction_junk would pass it.
    shell = ("<html><head><title>One post tagged with \"AI testing tools\" | "
             "MLflow</title></head><body><main><p>" + (
             "What I have learned is that the teams who get AI testing right "
             "treat evaluation as a first-class engineering discipline with "
             "reproducible datasets and clear metrics for every model change. ")
             * 4 + "</p></main></body></html>")
    p, _, s = ac.process_html("https://mlflow.org/articles/tags/ai-testing-tools",
                              shell, {"content": True, "links": False,
                                      "passage_common": common,
                                      "want_stats": True})
    chk("tag shell → 0 passages", p == [])
    chk("tag shell → listing_shell reject counted",
        s["extraction_rejects"].get("listing_shell", 0) >= 1)


def test_want_stats():
    """want_stats=True returns the 3-tuple with extraction_rejects counted."""
    try:
        import trafilatura  # noqa: F401
        from cytir.acquire import fetch_process  # noqa: F401
    except Exception:
        print("  skip want_stats (trafilatura/fetch_process absent)")
        return
    common = dict(domain="hug.ch", tier="A", trust_tier=1.0,
                  source_type="official", last_seen="2026-06-28")
    res = ac.process_html(PAGE, HTML, {"content": True, "links": True,
                                       "passage_common": common,
                                       "want_stats": True})
    chk("want_stats returns 3-tuple", len(res) == 3)
    passages, links, stats = res
    chk("good page: no rejects", stats == {"extraction_rejects": {}})
    chk("good page: passages unaffected", len(passages) >= 1)
    # a JSON-shell page body must be rejected AND counted
    shell = ("<html><head><title>Report</title></head><body><div id='app'>"
             '[{"id":1,"permalink":"\\/de\\/a\\/","breadcrumbs":[{"title":"R",'
             '"permalink":"\\/de\\/"}],"title":"E","content":"'
             + "wort " * 300 + '"}]</div></body></html>')
    p2, _, s2 = ac.process_html("https://shell.example/de/", shell,
                                {"content": True, "links": False,
                                 "passage_common": common,
                                 "want_stats": True})
    chk("json shell page → 0 passages", p2 == [])
    chk("json shell page → counted reject",
        s2["extraction_rejects"].get("json_dump", 0) >= 1)
    # legacy 2-tuple contract intact without want_stats
    res2 = ac.process_html(PAGE, HTML, {"content": True, "links": True,
                                        "passage_common": common})
    chk("legacy 2-tuple contract intact", len(res2) == 2)


ANCHOR_HTML = """
<html><head><title>Anchors</title></head><body>
<a href="https://www.planetesante.ch/article?utm_source=nl&x=1">Guide santé</a>
<a href="https://www.planetesante.ch/article?x=1&utm_medium=web"><img src="p.png"/></a>
<a href="https://addictionsuisse.ch/page">""" + ("A" * 150) + """</a>
<a href="https://www.hug.ch/other">internal</a>
</body></html>
"""


def test_outlink_anchors_helper():
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(ANCHOR_HTML, "html.parser")
    anchors = ac.outbound_link_anchors(soup, PAGE, "hug.ch")
    chk("anchors keyed by target url",
        "https://addictionsuisse.ch/page" in anchors)
    chk("anchor text capped at 100 chars",
        len(anchors["https://addictionsuisse.ch/page"]) == 100)
    chk("non-empty anchor wins over empty (dup target)",
        anchors.get("https://www.planetesante.ch/article?utm_source=nl&x=1")
        == "Guide santé")
    chk("internal target excluded from anchors",
        not any("hug.ch" in u for u in anchors))
    chk("outbound_links == anchor keys",
        ac.outbound_links(soup, PAGE, "hug.ch") == set(anchors))


def test_outlink_capture_sink():
    """task 194: the sink harvests links on links:False calls, streams
    discovery-schema rows, strips tracking params, and never alters the
    return contract."""
    import json as _json
    import os as _os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        side = ac.enable_outlink_capture(d)
        chk("sink path is <dir>/outlinks.jsonl",
            side == _os.path.join(d, "outlinks.jsonl"))
        try:
            p, links = ac.process_html(PAGE, ANCHOR_HTML,
                                       {"content": False, "links": False})
            chk("links:False contract intact (empty set returned)",
                links == set())
            # a second page with NO external links must still emit a row
            ac.process_html("https://www.hug.ch/plain",
                            "<html><body><p>" + "prose " * 60 + "</p>"
                            "<a href='https://www.hug.ch/x'>int</a></body></html>",
                            {"content": False, "links": False})
        finally:
            ac.disable_outlink_capture()
        rows = [_json.loads(l) for l in open(side)]
        chk("one row per parsed page (incl. 0-link page)", len(rows) == 2)
        r = rows[0]
        chk("row schema matches discovery sidecar",
            {"url", "domain", "links"} <= set(r)
            and r["url"] == PAGE and r["domain"] == "hug.ch"
            and isinstance(r["links"], list))
        chk("tracking params stripped + variants deduped in sink",
            "https://www.planetesante.ch/article?x=1" in r["links"]
            and not any("utm" in u for u in r["links"]))
        chk("anchors carried (≤100 chars, empty omitted)",
            r["anchors"].get("https://www.planetesante.ch/article?x=1")
            == "Guide santé"
            and all(len(t) <= 100 for t in r["anchors"].values()))
        chk("0-link page row has empty links", rows[1]["links"] == [])
        # after disable, nothing is captured
        ac.process_html(PAGE, ANCHOR_HTML, {"content": False, "links": False})
        chk("disable stops capture", len(open(side).readlines()) == 2)


def test_outlink_capture_env():
    """AW_OUTLINKS env arms the sink with zero code change in the ingester."""
    import os as _os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = _os.path.join(d, "side.jsonl")
        _os.environ["AW_OUTLINKS"] = path
        # reset the lazy env check (a prior test already tripped it)
        ac._OUTLINK_SINK["env_checked"] = False
        try:
            ac.process_html(PAGE, ANCHOR_HTML, {"content": False, "links": False})
            chk("env-armed sink wrote the sidecar",
                _os.path.exists(path) and len(open(path).readlines()) == 1)
        finally:
            ac.disable_outlink_capture()
            _os.environ.pop("AW_OUTLINKS", None)


# --- task 312: outlink COLUMN (aw_common.extract_outlink_records + the
# process_html(meta["outlinks_column"]) wiring) -----------------------------
# Distinct fixtures from ANCHOR_HTML/HTML above: those exercise the
# endorsement-graph EXTERNAL-only path (outbound_link_anchors); these exercise
# the raw, unfiltered corpus-row inventory (extract_outlink_records), which
# deliberately KEEPS internal/same-domain links.

OUTLINK_REL_HTML = """
<html><head><title>Relative + dedup</title></head><body>
<a href="/docs/guide">relative internal (resolves against PAGE)</a>
<a href="other/page.html">relative, no leading slash</a>
<a href="https://addictionsuisse.ch/page?utm_source=nl&x=1">tracked variant A</a>
<a href="https://addictionsuisse.ch/page?x=1&utm_campaign=spring">tracked variant B (dup target)</a>
<a rel="nofollow" href="https://sponsor.example/landing">sponsored (nofollow)</a>
<a href="https://sponsor.example/landing">same target, no rel this time</a>
<a href="#section">fragment — drop</a>
<a href="mailto:info@hug.ch">mailto — drop</a>
<a href="tel:+41123456789">tel — drop</a>
<a href="javascript:void(0)">js — drop</a>
</body></html>
"""

OUTLINK_CANON_HTML = """
<html><head><title>Canonicalization</title></head><body>
<a href="http://WWW.Example.CH/Path/">http, www, mixed case, trailing slash</a>
<a href="https://example.ch/Path">https, no www, same path (should collapse with the row above)</a>
<a href="https://example.ch/other?ref=newsletter&keep=1">tracking param stripped, functional param kept</a>
</body></html>
"""


def test_extract_outlink_records_relative_and_dedup():
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(OUTLINK_REL_HTML, "html.parser")
    recs = ac.extract_outlink_records(soup, PAGE)
    by_url = {r["url"]: r for r in recs}
    chk("relative /docs/guide resolves against PAGE's host",
        "https://hug.ch/docs/guide" in by_url)
    chk("relative other/page.html resolves against PAGE's directory",
        "https://hug.ch/some/other/page.html" in by_url)
    chk("internal (same-domain) links ARE kept (unlike outbound_link_anchors)",
        any(u.startswith("https://hug.ch/") for u in by_url))
    chk("tracking-param variants of one target dedup to one record",
        sum(1 for u in by_url if u.startswith(
            "https://addictionsuisse.ch/page")) == 1)
    chk("fragment/mailto/tel/javascript targets excluded",
        not any(u.startswith(("#", "mailto:", "tel:", "javascript:"))
               for u in by_url))
    chk("result sorted by url", [r["url"] for r in recs] == sorted(by_url))


def test_extract_outlink_records_nofollow_flag():
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(OUTLINK_REL_HTML, "html.parser")
    recs = ac.extract_outlink_records(soup, PAGE)
    by_url = {r["url"]: r for r in recs}
    chk("nofollow flag captured from the FIRST occurrence's rel attribute",
        by_url.get("https://sponsor.example/landing", {}).get("nofollow") is True)
    chk("a normal link has nofollow=False",
        by_url.get("https://addictionsuisse.ch/page?x=1", {}).get("nofollow") is False)


def test_extract_outlink_records_canonicalization():
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(OUTLINK_CANON_HTML, "html.parser")
    recs = ac.extract_outlink_records(soup, PAGE)
    urls = [r["url"] for r in recs]
    chk("http+www+trailing-slash and https+no-www+no-slash collapse to one url",
        urls.count("https://example.ch/Path") == 1)
    chk("every url matches aw_common.canonical_url()'s own output",
        all(u == ac.canonical_url(u) for u in urls))
    chk("tracking param (?ref=) stripped, functional param (?keep=) kept",
        any(u.startswith("https://example.ch/other?") and "keep=1" in u
           and "ref=" not in u for u in urls))


def test_extract_outlink_records_empty_soup():
    chk("None soup returns []", ac.extract_outlink_records(None, PAGE) == [])


def test_process_html_outlinks_column_default_off():
    """Regression guard for the task-312 'additive-only' requirement: every
    EXISTING call shape (no outlinks_column key) must leave Passage.outlinks
    unset, so process_html is byte-identical for the ~20 other callers."""
    try:
        import trafilatura  # noqa: F401
        from cytir.acquire import fetch_process  # noqa: F401
    except Exception:
        print("  skip (trafilatura/fetch_process absent)")
        return
    common = dict(domain="hug.ch", tier="A", trust_tier=1.0,
                 source_type="official", last_seen="2026-06-28")
    passages, _links = ac.process_html(PAGE, HTML, {
        "content": True, "links": True, "seed_reg": "hug.ch",
        "passage_common": common})
    chk("outlinks_column omitted -> Passage.outlinks stays None",
        passages and all(p.outlinks is None for p in passages))


def test_process_html_outlinks_column_wiring():
    """End-to-end: meta['outlinks_column']=True attaches the SAME outlink
    inventory to every chunk of one page (mirrors title/site_name, task 274
    bug #2), and the endorsement `links` return value is unaffected."""
    try:
        import trafilatura  # noqa: F401
        from cytir.acquire import fetch_process  # noqa: F401
    except Exception:
        print("  skip (trafilatura/fetch_process absent)")
        return
    common = dict(domain="hug.ch", tier="A", trust_tier=1.0,
                 source_type="official", last_seen="2026-06-28")
    passages, links = ac.process_html(PAGE, HTML, {
        "content": True, "links": True, "seed_reg": "hug.ch",
        "passage_common": common, "outlinks_column": True})
    chk("outlinks_column=True -> at least one passage produced", bool(passages))
    if passages:
        chk("every passage carries a non-None outlinks list",
            all(p.outlinks is not None for p in passages))
        chk("outlinks identical across every chunk of the page",
            all(p.outlinks == passages[0].outlinks for p in passages))
        chk("outlinks includes the internal hug.ch link (unfiltered inventory)",
            any(r["url"].startswith("https://hug.ch/")
               for r in passages[0].outlinks))
        chk("outlinks includes the external endorsement link too",
            any("planetesante.ch" in r["url"] for r in passages[0].outlinks))
    chk("the endorsement `links` return value is untouched by outlinks_column",
        "https://www.planetesante.ch/article" in links)
    chk("payload()['outlinks'] round-trips the same list",
        passages and passages[0].payload().get("outlinks") == passages[0].outlinks)
    chk("compact_payload() strips outlinks (master-only, ADR-039)",
        passages and "outlinks" not in ac.compact_payload(passages[0].payload()))


# --- task 634: the exact discovery_ingest.py `common` shape, end to end ----
# The regression this pins: test_content() above (task 020) built its own
# `common` dict, which never included class_basis/n_endorser_regs -- so it
# never caught that task 440 (2026-08-20) stamped those two fields into
# process_domain's `common`, which is spread as **fields straight into
# aw_common.Passage(**fields). Passage has no such constructor arguments;
# EVERY real fetch since task 440 merged raised TypeError inside the
# extraction pool worker, silently swallowed and mislabeled
# "extract-timeout/skip" by discovery_ingest._extract_in_pool's necessarily-
# broad except. Zero passages were produced by any real run in that whole
# window. Uses plain `assert` (not this file's local chk()/PASS/FAIL
# convention, which does not fail a pytest collection run on its own) so this
# specific guard cannot silently go green the same way.
#
# No mock below Passage construction: trafilatura, the langgate (opted
# degraded via AW_LANGGATE_OPTIONAL, set at module import time above), the
# chunker and Passage() itself all run for real.
def test_process_domains_exact_common_shape_yields_at_least_one_passage():
    """Real end-to-end guard, driven from the SOURCE OF TRUTH rather than a
    hand-copied dict (a hand copy is exactly how test_content() above missed
    this): builds `common` the exact way process_domain does (via
    class_trust_for), and calls the REAL aw_common.process_html directly --
    no multiprocessing pool. process_domain's own pool boundary is
    deliberately not exercised here: a full pytest session (400+ modules)
    reproduced TWO distinct pool-sharing hazards unrelated to this guard's
    actual target (Passage construction) -- a stale pre-env-change pool
    (fixed, reset_extraction_pool()) and a 'spawn' pickling failure from a
    process_html binding reassigned elsewhere in the same long-lived
    process. This guard exists for the class_basis/n_endorser_regs
    regression, which lives entirely inside process_html/Passage
    construction; calling it directly still fully covers that, without the
    pool's cross-test fragility. If a future field is ever added to
    `common`/passage_common that Passage's constructor does not accept,
    this fails immediately and LOUDLY -- never silently swallowed as
    "extract-timeout/skip" the way task 440's regression was."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cytir.acquire import discovery_ingest as di

    common = dict(domain="hug.ch", tier="C",
                  trust_tier=di.class_trust_for("docs", "endorsed", 3),
                  source_type=di.CLASS_SOURCE.get("docs", "docs"),
                  last_seen="2026-09-22")
    passages, links = ac.process_html(PAGE, HTML, {
        "content": True, "links": True,
        "passage_common": common, "lang_default": "fr", "digest": "abc123"})

    assert len(passages) >= 1, (
        "process_domain's real common shape produced zero passages -- see task 634")
    assert passages[0].domain == "hug.ch"


if __name__ == "__main__":
    test_links()
    test_links_default_seed_reg()
    test_short_html_noop()
    test_content()
    test_outbound_links_helper()
    test_reg_domain()
    test_extraction_junk()
    test_listing_shell()
    test_sitemap_shell()
    test_listing_shell_end2end()
    test_want_stats()
    test_outlink_anchors_helper()
    test_outlink_capture_sink()
    test_outlink_capture_env()
    test_extract_outlink_records_relative_and_dedup()
    test_extract_outlink_records_nofollow_flag()
    test_extract_outlink_records_canonicalization()
    test_extract_outlink_records_empty_soup()
    test_process_html_outlinks_column_default_off()
    test_process_html_outlinks_column_wiring()
    test_process_domains_exact_common_shape_yields_at_least_one_passage()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)
