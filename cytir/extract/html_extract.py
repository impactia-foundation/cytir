"""HTML -> passages extraction (moved out of `cytir.core.common`, task 2333).

Outlink capture, `process_html`, title/site-name resolution, markup repair, trafilatura
markdown extraction, metadata date guard, `passages_from_markdown`.
"""
from __future__ import annotations

from typing import Optional
from urllib.parse import urljoin
from urllib.parse import urlparse
import json
import os
import re
import threading

from cytir.core.canon import canonical_url
from cytir.core.canon import fold_www
from cytir.core.canon import strip_tracking_url
from cytir.core.prose_guard import binary_content as _binary_content
from cytir.core.common import (  # noqa: E402
    Passage,
    extraction_junk,
    extractive_snippet,
    listing_shell,
    low_quality,
    passage_from_chunk,
)
from cytir.chunk.dispatch import (
    ChunkerSelection,
    chunk_markdown_channels,
)


# ---------------------------------------------------------------------------
# Unified extraction — content + link-graph in ONE pass (ADR-020 decision).
#
# Tim's decision (2026-06-28): extraction must do CONTENT passages AND the
# endorsement/link-graph in a single parse, never re-fetch CC twice or discard
# links. `process_html` is the shared entry point both `cc_ingest.py` (content)
# and `extract_outlinks.py` (links) call. See notes/unified-extractor.md.
# ---------------------------------------------------------------------------

# Chrome we never count as an endorsement (social/CDN/analytics/auth/etc.).
# Single source of truth, reused by extract_outlinks.py. (Was duplicated there.)
BOILERPLATE = re.compile(
    r"(facebook|twitter|x\.com|linkedin|youtube|instagram|tiktok|pinterest|"
    r"google\.|gstatic|googleapis|gravatar|gstatic|doubleclick|cloudflare|"
    r"fonts\.|cdn\.|jsdelivr|unpkg|addthis|sharethis|paypal|cookiebot|"
    r"whatsapp|telegram|apple\.com|play\.google|siteimprove|sabacloud|"
    r"saba\.com|hubspot|salesforce|adobe\.com|microsoft\.com|office\.com|"
    r"vimeo|issuu|slideshare|eventbrite|mailchimp|typeform|surveymonkey)", re.I)


def reg_domain(host: str) -> str:
    """Crude registrable-domain: last 2 labels (3 for known 2-level eTLDs).
    Shared so content + link paths agree on what 'same site' means."""
    p = (host or "").lower().lstrip(".").split(".")
    if len(p) >= 3 and p[-2] in ("co", "gov", "ac", "org", "com") and p[-1] in ("uk", "au", "nz"):
        return ".".join(p[-3:])
    return ".".join(p[-2:]) if len(p) >= 2 else host


def outbound_link_anchors(soup, page_url: str, seed_reg: str) -> dict[str, str]:
    """External endorsement links WITH their anchor text from a parsed page.

    Returns `{target_url: anchor_text}` for `http(s)` `<a href>` targets whose
    registrable domain differs from `seed_reg` and that are not boilerplate/
    chrome (BOILERPLATE). Fragments stripped; target URLs are otherwise kept
    exactly as `outbound_links` always has (no tracking-strip here — sidecar
    identity must stay comparable across waves; the capture sink strips).
    Anchor text is whitespace-normalized and capped at 100 chars (may be "",
    e.g. image links); the first non-empty text per target wins.

    The cross-page template-frequency filter (drop near-ubiquitous nav links)
    stays in the callers — see extract_outlinks.process_seed /
    aggregate_harvested_links."""
    out: dict[str, str] = {}
    if soup is None:
        return out
    for a in soup.find_all("a", href=True):
        href = (a["href"] or "").strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        try:
            u = urljoin(page_url, href)
        except ValueError:      # a malformed bracketed host (`Invalid IPv6 URL`): skip the link, keep the page
            continue
        pr = urlparse(u)
        if pr.scheme not in ("http", "https") or not pr.netloc:
            continue
        if reg_domain(pr.netloc) == seed_reg or BOILERPLATE.search(pr.netloc):
            continue
        u = u.split("#")[0]
        text = " ".join(a.get_text(" ", strip=True).split())[:100]
        if u not in out or (text and not out[u]):
            out[u] = text
    return out


def outbound_links(soup, page_url: str, seed_reg: str) -> set[str]:
    """External endorsement links from a parsed page.

    Returns the set of `http(s)` `<a href>` targets whose registrable domain
    differs from `seed_reg` and that are not boilerplate/chrome (BOILERPLATE).
    Fragments stripped. This is the *per-page* link set; the cross-page
    template-frequency filter (drop near-ubiquitous nav links) is applied by
    the caller over all of a seed's sampled pages (it cannot be decided from a
    single page) — see extract_outlinks.process_seed.
    """
    return set(outbound_link_anchors(soup, page_url, seed_reg))


# ---------------------------------------------------------------------------
# Outlink COLUMN (task 312) — distinct from outbound_link_anchors/outbound_links
# above (the endorsement-graph EXTERNAL-only, BOILERPLATE-filtered set) and from
# the outlink capture SINK below (a sidecar file). This is the raw, UNFILTERED
# per-page link inventory folded into the corpus row itself (Passage.outlinks)
# so a future re-extraction/discovery pass — or the epochs-DAG extract@N+1 case
# architecture.md §4.2 exists for — can read outlinks back from the corpus
# without re-parsing HTML. Same-domain nav is deliberately KEPT here (the
# endorsement path drops it); this is inventory, not endorsement signal.
# ---------------------------------------------------------------------------

def outlinks_column_enabled() -> bool:
    """AW_OUTLINKS_COLUMN=1 default ON — the kill switch the task-312 ingest
    chokepoints (sitemap_ingest / discovery_ingest / cc_ingest) check before
    passing meta['outlinks_column']=True to process_html. process_html itself
    stays opt-in-only (meta default False) so every OTHER existing caller of
    process_html is byte-identical regardless of this env var."""
    return os.environ.get("AW_OUTLINKS_COLUMN", "1") not in ("0", "false", "False", "")


def extract_outlink_records(soup, page_url: str) -> list[dict]:
    """All absolute outlinks on one parsed page, for the corpus row `outlinks`
    column (task 312; Passage.outlinks).

    Returns a list of `{"url": <canonical_url()-normalized target>, "nofollow":
    bool}`, deduped by the canonicalized url and sorted by it. `nofollow` is
    True iff the FIRST occurrence of that target carries `rel="nofollow"` on
    its `<a>` tag (cheap: read straight off the already-parsed tag, no extra
    fetch, per the task-312 brief). Uses `canonical_url` — the same
    canonicalization AW uses everywhere else to dedup/join urls — so a corpus
    row's outlinks are directly comparable to `url`/`point_id` derivation
    elsewhere. `#fragment`/`mailto:`/`tel:`/`javascript:` targets are dropped;
    http(s) only. No same-domain or BOILERPLATE filtering (contrast
    `outbound_link_anchors` above) — this is a raw inventory, not an
    endorsement signal."""
    out: dict[str, bool] = {}
    if soup is None:
        return []
    for a in soup.find_all("a", href=True):
        href = (a["href"] or "").strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        try:
            u = urljoin(page_url, href)
        except ValueError:
            continue
        pr = urlparse(u)
        if pr.scheme not in ("http", "https") or not pr.netloc:
            continue
        cu = canonical_url(u)
        rel = a.get("rel") or []
        if isinstance(rel, str):
            rel = rel.split()
        nofollow = any(str(r).lower() == "nofollow" for r in rel)
        if cu not in out:
            out[cu] = nofollow
    return [{"url": u, "nofollow": nf} for u, nf in sorted(out.items())]


def extract_outlink_context_records(soup, page_url: str) -> list[dict]:
    """All canonical outlinks with first-occurrence anchor/context evidence.

    This is the preservation-profile companion to ``extract_outlink_records``:
    it uses the same URL resolution/canonicalization and first-occurrence
    nofollow contract, while retaining the human-visible evidence required to
    replay authoritative-source link expansion without parsing HTML again.
    Internal links are deliberately retained; callers can classify them using
    ``reg_domain`` without losing the original inventory.
    """
    out: dict[str, dict] = {}
    if soup is None:
        return []
    for anchor in soup.find_all("a", href=True):
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        try:
            resolved = urljoin(page_url, href)
        except ValueError:
            continue
        parsed = urlparse(resolved)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            continue
        target = canonical_url(resolved)
        rel = anchor.get("rel") or []
        if isinstance(rel, str):
            rel = rel.split()
        context = (" ".join(anchor.get_text(" ", strip=True).split()) or
                   " ".join(str(anchor.get("aria-label") or "").split()) or
                   " ".join(str(anchor.get("title") or "").split()) or
                   target)
        out.setdefault(target, {
            "url": target,
            "nofollow": any(str(value).casefold() == "nofollow" for value in rel),
            "anchor_or_context": context,
        })
    return [out[url] for url in sorted(out)]


# ---------------------------------------------------------------------------
# Outlink capture sink (task 194) — restore harvesting in the pack path.
#
# The discovery waves harvested every fetched page's outlinks for the
# endorsement graph (ADR-020 one-pass contract), but the bespoke pack
# ingesters since 175 call process_html with {"links": False} and the links
# were DISCARDED — exactly on the packs crawling the best endorsement sources
# (WHO→national bodies, EIGE→equality institutes, PG→startups). The sink
# closes that gap at the shared seam: once enabled, EVERY process_html call
# in this process streams one row per parsed page to <workdir>/outlinks.jsonl
#
#     {"url": <source_url>, "domain": <source_host, www-stripped>,
#      "links": [<target_url, tracking-stripped, deduped, sorted>, …],
#      "anchors": {<target_url>: <anchor_text ≤100 chars>, …}}
#
# schema-compatible with discovery_ingest's sidecar ({url, domain, links} —
# aggregate_harvested_links.py consumes it unchanged; "anchors" is an
# additive key it ignores). A row is written for every parsed page (links may
# be []) so "parsed, no links" stays distinguishable from "never fetched",
# and n_pages stays correct for the aggregator's template filter. Passage
# extraction is untouched, and the (passages, links) return contract is
# unchanged — callers that asked {"links": False} still get an empty set.
#
# Activation: `enable_outlink_capture(out_dir)` (one line in a pack's main),
# or env `AW_OUTLINKS=<path.jsonl|dir>` for a zero-code-change run of an
# unmodified ingester. Appends (resumable packs keep appending across runs;
# the aggregator dedups pages by url). Single-process writers only — the
# in-process lock serializes threads, but concurrent *processes* sharing one
# sink file may interleave rows larger than the stdio buffer.
# ---------------------------------------------------------------------------

_OUTLINK_SINK: dict = {"fh": None, "path": None, "lock": None,
                       "env_checked": False}

# `fold_www` MOVED to cytir.core.canon (task 646 S2) and is re-exported at the
# top of this file; see that module's docstring for why it deliberately
# differs from `canonical_url`'s own www-fold.


def enable_outlink_capture(path: str) -> str:
    """Enable streaming outlink capture for this process. `path` is either a
    work DIRECTORY (rows go to <path>/outlinks.jsonl — the pack convention)
    or an explicit *.jsonl file path. Opens in append mode; returns the
    resolved sidecar path."""
    disable_outlink_capture()
    if not path.endswith(".jsonl"):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "outlinks.jsonl")
    else:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
    _OUTLINK_SINK.update(fh=open(path, "a", encoding="utf-8"), path=path,
                         lock=threading.Lock(), env_checked=True)
    return path


def disable_outlink_capture() -> None:
    """Close and disarm the sink (also blocks later AW_OUTLINKS auto-enable —
    an explicit disable wins over the env)."""
    fh = _OUTLINK_SINK.get("fh")
    if fh is not None:
        try:
            fh.close()
        except Exception:  # noqa: BLE001
            pass
    _OUTLINK_SINK.update(fh=None, path=None, lock=None, env_checked=True)


def _outlink_sink_active() -> bool:
    if _OUTLINK_SINK["fh"] is None and not _OUTLINK_SINK["env_checked"]:
        _OUTLINK_SINK["env_checked"] = True
        env = os.environ.get("AW_OUTLINKS")
        if env:
            enable_outlink_capture(env)
    return _OUTLINK_SINK["fh"] is not None


def _capture_outlinks(url: str, anchors: dict[str, str]) -> None:
    fh = _OUTLINK_SINK["fh"]
    if fh is None:
        return
    host = fold_www(urlparse(url).netloc or "")
    tgt: dict[str, str] = {}
    for u, text in anchors.items():
        cu = strip_tracking_url(u)          # ?ref=/utm_* variants collapse
        if cu not in tgt or (text and not tgt[cu]):
            tgt[cu] = text
    row = {"url": url, "domain": host, "links": sorted(tgt),
           "anchors": {u: t for u, t in sorted(tgt.items()) if t}}
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _OUTLINK_SINK["lock"]:
        fh.write(line)
        fh.flush()


def _parse_soup(html: str):
    """Parse HTML once with BeautifulSoup. Lazy import so callers that only
    need one path don't hard-require bs4 at module import time."""
    try:
        from bs4 import BeautifulSoup
    except Exception:  # noqa: BLE001
        return None
    try:
        return BeautifulSoup(html, "html.parser")
    except Exception:  # noqa: BLE001
        return None


def parse_html_soup(html: str):
    """Public single-pass HTML parse for master-side evidence stages.

    `extract_outlink_context_records` / `outbound_link_anchors` take an
    already parsed document so a caller needing several views of one page
    parses it once. Preservation stages need that seam from outside this
    module, and a private-name import at each call site is how "one home"
    erodes.
    """
    return _parse_soup(html)


def process_html(url: str, html: str, meta: dict):
    """Unified one-pass extractor: CONTENT passage(s) + endorsement OUTLINKS.

    Contract
    --------
    Parse `html` (already fetched, e.g. a CC WARC body) exactly once and return

        (passages, links)

    where
      * `passages` is a list of `Passage` content records (possibly empty —
        empty if the page is low-quality, too short, or `meta["content"]` is
        falsy). A list (not a single Passage) because one page chunks into
        several passages.
      * `links` is the set of external endorsement URLs on the page (after the
        BOILERPLATE/same-domain filter); empty if `meta["links"]` is falsy.

    Independent of `meta["links"]`, an armed outlink-capture sink (task 194:
    `enable_outlink_capture` / env `AW_OUTLINKS`) streams every parsed page's
    external links to a sidecar jsonl — see the sink block below.

    Neither path re-fetches; both read the same parsed DOM / same HTML string.
    The link path's *template-frequency* filter is cross-page and stays in the
    caller (it needs the whole seed sample), so this function returns the raw
    per-page link set.

    `meta` keys
    -----------
      content   : bool   — produce content passages (default True)
      links     : bool   — produce outbound links (default True)
      seed_reg  : str    — registrable domain to treat as 'internal' for the
                           link filter (default: derived from `url`'s host)
      lang_default : str — fallback language if not inferable from the URL
      passage_common : dict — fields forwarded to Passage (domain, tier,
                           trust_tier, source_type, last_seen, …)
      digest    : str|None — CC content SHA1 for cross-crawl dedup
      want_stats : bool  — return (passages, links, stats) instead, where
                           stats = {"extraction_rejects": {reason: count}}
                           from the extraction_junk gate (task 097). Default
                           False keeps the 2-tuple contract for existing
                           callers (cc_ingest, extract_outlinks, fetch_hn_cc).
      outlinks_column : bool — task 312: also compute the raw, UNFILTERED
                           per-page outlink inventory (extract_outlink_records)
                           and attach it as `Passage.outlinks` on every passage
                           this call returns. Default False — every existing
                           caller is byte-identical; the task-312 ingest
                           chokepoints (sitemap_ingest / discovery_ingest /
                           cc_ingest) opt in via outlinks_column_enabled().
                           Independent of `links`/the sink above: this is a
                           corpus-row field, not the endorsement-graph set.
    """
    meta = meta or {}
    do_content = meta.get("content", True)
    do_links = meta.get("links", True)
    want_outlinks = meta.get("outlinks_column", False)
    passages: list[Passage] = []
    links: set[str] = set()
    rejects: dict[str, int] = {}

    if not html or len(html) < 200:
        return (passages, links, {"extraction_rejects": rejects}) \
            if meta.get("want_stats") else (passages, links)

    # Outlink capture sink (task 194): when armed, harvest links on EVERY
    # parsed page — even for {"links": False} callers (the bespoke packs) —
    # without changing the return contract.
    sink = _outlink_sink_active()
    soup = _parse_soup(html) if (do_links or sink or want_outlinks) else None

    if do_links or sink:
        seed_reg = meta.get("seed_reg") or reg_domain(urlparse(url).netloc)
        anchors = outbound_link_anchors(soup, url, seed_reg)
        if sink:
            _capture_outlinks(url, anchors)
        if do_links:
            links = set(anchors)

    outlink_records = extract_outlink_records(soup, url) if want_outlinks else None

    if do_content:
        passages = _content_passages(url, html, meta, rejects)
        if outlink_records is not None:
            for p in passages:
                p.outlinks = outlink_records

    if meta.get("want_stats"):
        return passages, links, {"extraction_rejects": rejects}
    return passages, links


def _raw_html_title(html: str) -> Optional[str]:
    """Verbatim <title> text (whitespace-collapsed, entities unescaped). Unlike
    trafilatura's metadata title it KEEPS the site-name suffix (task 274)."""
    m = re.search(r"<title[^>]*>(.*?)</title>", html or "", re.I | re.S)
    if not m:
        return None
    import html as _htmlmod
    t = _htmlmod.unescape(re.sub(r"\s+", " ", m.group(1)).strip())
    return t or None


def html_title(html: str) -> Optional[str]:
    """Public verbatim HTML-title extractor for master-side evidence stages."""
    return _raw_html_title(html)


# separators sites put between the page title and the site/entity name
_TITLE_SEP_SPLIT = re.compile(r"\s*(?:::|\|\||[|:·•»›]|[–—-]| - )\s*")
_ENTITY_SOURCE_TYPES = frozenset({"official", "reference", "entity-self",
                                  "entity_card"})


# Registrable labels that name no entity. `admin`/`gouv` are the state, not the
# body; the rest are path/plumbing words. Kept beside the one function that uses
# them so a second copy cannot drift from it (engineering doctrine 1).
_GENERIC_SITE_LABELS = frozenset({
    "www", "admin", "gouv", "government", "commune", "site", "index"})


def site_entity_label(url: str) -> Optional[str]:
    """ADR-061's domain fallback for the entity/site name, on its own.

    "impactia.org" -> "Impactia", "dardagny.ch" -> "Dardagny", "www.admin.ch"
    -> None. Public because it is a MIRROR CONTRACT: `_site_entity_name` uses
    it to compose the surface, and `representation_contract`'s
    `identity.site_name` check uses it to decide whether the composed surface
    still carries the site's own name token. Two implementations of "what is
    this site called" would mean the guard could pass a row the composer never
    could, and vice versa (engineering doctrine 6).

    Returns None when the registrable label names nothing distinctive -- which
    the caller must report as UNASSERTABLE, never as absent.
    """
    try:
        host = urlparse(url or "").netloc.lower()
    except Exception:  # noqa: BLE001
        host = ""
    label = reg_domain(host).split(".")[0] if host else ""
    if label and len(label) >= 4 and label not in _GENERIC_SITE_LABELS:
        return label.capitalize()
    return None


def _site_entity_name(raw_title, clean_title, url, source_type) -> Optional[str]:
    """The entity/site-name suffix trafilatura stripped from the title (task 274,
    bug #1).

    Primary: the part of the raw <title> the cleaned title dropped
    ("Gestion des déchets – Dardagny" − "Gestion des déchets" → "Dardagny").
    Fallback (official/reference/entity types only, where the domain IS the
    entity — communes/gov): the registrable-domain label ("dardagny.ch" →
    "Dardagny"). This is what makes the targeted re-embed cheap: no raw title
    needed, the domain suffices. Bounded to a NAME, not a tagline (≤6 words / 60
    chars). Returns None when nothing distinctive is found."""
    suffix = ""
    if raw_title and clean_title and clean_title.strip():
        rl, cl = raw_title.strip(), clean_title.strip()
        low_r, low_c = rl.lower(), cl.lower()
        if low_c in low_r and low_c != low_r:
            i = low_r.find(low_c)
            residue = (rl[:i] + " " + rl[i + len(cl):]).strip()
            parts = [p.strip() for p in _TITLE_SEP_SPLIT.split(residue)
                     if p.strip() and p.strip().lower() not in low_c]
            suffix = " ".join(parts).strip(" -–—:|·•»›")
    if not suffix and source_type in _ENTITY_SOURCE_TYPES:
        suffix = site_entity_label(url) or ""
    if suffix and (len(suffix.split()) > 6 or len(suffix) > 60):
        suffix = ""      # a tagline/description, not a name — drop it
    return suffix or None


def resolve_title_and_site_name(html, clean_title, url, source_type):
    """ADR-061's title pair -- `(display_title, site_name)` -- in ONE place.

    `process_html` has done this since task 274; the retired small-site ingester (task 1433) did not, and
    for two years emitted whatever `trafilatura.extract_metadata().title`
    returned with no `site_name` beside it. Whether a page kept the token every
    query for it is made of was therefore decided by trafilatura, per page:
    task 547's `impactia.org/robotme/` row kept `robotMe - impactIA` and task
    427's `la-fondation/` row did not (task 548). Two call sites, one function,
    so a future third cannot invent a third answer (engineering doctrine 1).

    `html` may be None when the raw markup is gone (a re-embed from stored
    text): the raw-<title> path is then skipped and the domain fallback in
    `_site_entity_name` carries it, which is exactly what ADR-061 designed the
    fallback for.
    """
    raw_title = _raw_html_title(html) if html else None
    site_name = _site_entity_name(raw_title, clean_title, url, source_type)
    title = _entity_display_title(clean_title, raw_title, site_name, source_type)
    return title, site_name


def _entity_display_title(clean_title, raw_title, site_name, source_type) -> str:
    """Display / title-centrality title (task 274). For gov/entity source types
    the stripped site-name suffix IS the disambiguating entity token, so prefer
    the full raw <title> when it is a sane length and contains the cleaned title;
    else synthesize `"<clean> – <site_name>"` (the re-embed path — raw title gone,
    domain-derived name survives). Non-entity types keep the CLEANED title (a
    "... | CNN" suffix is outlet noise that would pollute title-centrality)."""
    if source_type not in _ENTITY_SOURCE_TYPES or not clean_title:
        return clean_title
    cl = clean_title.strip()
    if raw_title:
        rt = raw_title.strip()
        if cl.lower() in rt.lower() and cl.lower() != rt.lower() and len(rt) <= 90:
            return rt
    if site_name and site_name.lower() not in cl.lower():
        return f"{cl} – {site_name}"
    return cl


# --- markup-repair pass (task 523: the F2-HTML extraction collapse) ---------
# INCIDENT. 17,850 rows (14.05%) of the task472_f2_html lane -- 10,143 distinct
# URLs over 846 digest cohorts -- carried ONE markdown body per cohort. On
# eeoc.gov, 3,010 captures (median 73 KB, distinct body_sha256) all extracted to
# the same 483-character, 72-token ".gov websites use HTTPS" USWDS banner.
#
# MECHANISM, measured on the sealed captures (task 523 diagnose_
# extraction_collapse.py). eeoc.gov's USWDS mega-menu leaves 6 <div>s unclosed
# inside a <ul> in <nav>. libxml2 (lxml's HTML parser, what trafilatura uses)
# then refuses the matching </ul>, so <main id="main-content"> is parsed as a
# DESCENDANT OF <nav> -- ancestors measured: main < ul < li < div < ul < ... <
# nav < header. trafilatura's tree_cleaning drops nav/header wholesale, taking
# the page with it (36,443 chars of DOM text -> 1,173 chars cleaned), every
# BODY_XPATH then matches nothing, and `recover_wild_text` salvages the only
# <p> elements left standing: the banner. Every page of the host yields the
# same bytes, so it is one content_digest and one parent_markdown_sha256.
#
# `<main>` as a descendant of nav/header/footer/aside is FORBIDDEN by the HTML
# spec, so observing it is proof of a mis-parse, not a page-design choice. That
# is the first trigger. The second is quantitative and host-free: an extraction
# that recovers a tiny fraction of the document's own DOM text.
#
# THE REPAIR is a re-parse, not a re-extraction: html.parser (stdlib, via bs4 --
# already a pipeline dependency, so self-hostability is unchanged) closes the
# stray <div>s at </ul> the way the HTML5 tree-construction algorithm requires,
# and re-serializing its tree gives trafilatura a correctly nested document.
# Measured on the four sealed eeoc.gov captures: 483 -> 1,797 / 22,746 / 9,999 /
# 6,784 chars, four distinct bodies where there had been one.
#
# The repaired candidate is compared against the SAME extractor mode and is
# taken only when it is decisively longer, so a page trafilatura already handles
# is byte-identical by construction. Knobs are config, not literals.
_REPAIR_THIN_RATIO = float(os.environ.get("AW_EXTRACT_REPAIR_THIN_RATIO", "0.10"))
_REPAIR_MIN_DOM_CHARS = int(os.environ.get("AW_EXTRACT_REPAIR_MIN_DOM", "2000"))
_REPAIR_MIN_GAIN = float(os.environ.get("AW_EXTRACT_REPAIR_MIN_GAIN", "1.25"))
_REPAIR_MIN_GAIN_CHARS = int(os.environ.get("AW_EXTRACT_REPAIR_MIN_GAIN_CHARS", "200"))

# `<main>` must not descend from these (HTML Living Standard, 4.4.14).
_MISPARSE_FORBIDDEN_ANCESTORS = frozenset({"nav", "header", "footer", "aside"})


def landmark_misparse(html: str) -> Optional[str]:
    """Reason string if the document's content landmarks parse into a position
    the HTML spec forbids -- proof the byte stream defeated the parser -- else
    None. Returns e.g. "main_in_nav". Cheap: one lxml parse, no extraction."""
    try:
        from lxml import html as _lh
        tree = _lh.fromstring(html)
    except Exception:
        return None
    for tag in ("main", "article"):
        for el in tree.iter(tag):
            parent = el.getparent()
            while parent is not None:
                if parent.tag in _MISPARSE_FORBIDDEN_ANCESTORS:
                    return f"{tag}_in_{parent.tag}"
                parent = parent.getparent()
    return None


def _dom_text_chars(html: str) -> int:
    """Characters of text the parsed document holds, boilerplate included. The
    denominator of the thin-yield trigger; 0 when the HTML will not parse."""
    try:
        from lxml import html as _lh
        return len(_lh.fromstring(html).text_content())
    except Exception:
        return 0


def repair_markup(html: str) -> Optional[str]:
    """Re-serialize `html` through the stdlib HTML parser, which closes stray
    elements at their enclosing end tag the way HTML5 tree construction
    requires. Returns None if the repair fails or changes nothing."""
    try:
        from bs4 import BeautifulSoup
        repaired = str(BeautifulSoup(html, "html.parser"))
    except Exception:
        return None
    return repaired if repaired and repaired != html else None


def _decisively_longer(candidate: Optional[str], incumbent: Optional[str]) -> bool:
    """True when `candidate` recovers materially more than `incumbent` did. A
    nudge is not enough: a re-parse must earn the change it makes."""
    if not candidate or low_quality(candidate) or len(candidate) < 120:
        return False
    base = len(incumbent or "")
    return (len(candidate) >= base * _REPAIR_MIN_GAIN
            and len(candidate) >= base + _REPAIR_MIN_GAIN_CHARS)


def _not_markup(html: str) -> bool:
    """True when the "html" handed to the HTML extractor is a binary file's bytes decoded as text.
    ROOT CAUSE of the binary-PDF rows (task 767): a lane that fetched `application/pdf` and ran
    this trafilatura path anyway got `7 0 obj\\n<>\\nstream\\nx\u00bdMo...` back (the `<<...>>`
    dictionaries read as tags and dropped), and `low_quality` passed it because Latin-1
    mojibake letters are alphabetic. Reproduced on a real WEKO decision PDF
    (tests/fixtures/prose_guard). A `%PDF-` magic at the start is conclusive; anything else is
    left to the prose test."""
    head = (html or "")[:4096].lstrip("\ufeff \t\r\n")
    return head.startswith("%PDF-") or bool(_binary_content(head))


def _extract_content_md(html: str, url: str):
    """Precision-first markdown extraction with a TRIGGERED recall fallback for
    nav-heavy pages trafilatura over-strips to near-empty (task 274, bug #3),
    and a TRIGGERED markup-repair re-parse for documents whose markup defeats
    libxml2's recovery (task 523, the F2-HTML extraction collapse).

    Returns (md, mode) with mode ∈ {"precision", "recall_fallback",
    "precision+markup_repair", "recall_fallback+markup_repair"}, or (None, None)
    if every pass fails the length floor / low_quality. Both fallbacks fire ONLY
    on their trigger, and the repair is accepted only when the SAME extractor
    mode reads decisively more from the repaired parse -- so any page
    trafilatura already handles is byte-identical, and the mode string always
    names the path that produced the bytes."""
    if _not_markup(html):
        return None, None               # task 767: bytes of a PDF (or other binary), not markup
    import trafilatura
    _MISSING = object()
    repaired = _MISSING                 # repaired markup, computed at most once
    dom_chars = None

    def _repaired_html():
        nonlocal repaired
        if repaired is _MISSING:
            repaired = repair_markup(html)
        return repaired

    def _repair_warranted(md: Optional[str]) -> bool:
        nonlocal dom_chars
        if dom_chars is None:
            dom_chars = _dom_text_chars(html)
        if dom_chars < _REPAIR_MIN_DOM_CHARS:
            return False
        if len(md or "") < dom_chars * _REPAIR_THIN_RATIO:
            return True
        return landmark_misparse(html) is not None

    passes = (("precision", dict(favor_precision=True)),
              ("recall_fallback", dict(favor_recall=True, include_comments=False)))
    for mode, kwargs in passes:
        md = trafilatura.extract(html, url=url, output_format="markdown",
                                 include_tables=True, **kwargs)
        if _repair_warranted(md):
            fixed = _repaired_html()
            if fixed is not None:
                alt = trafilatura.extract(fixed, url=url, output_format="markdown",
                                          include_tables=True, **kwargs)
                if _decisively_longer(alt, md):
                    md, mode = alt, mode + "+markup_repair"
        if md and len(md) >= 120 and not low_quality(md):
            return md, mode
    return None, None


def extract_content_markdown(html: str, url: str):
    """Public precision-first principal extractor for master-side stages."""
    return _extract_content_md(html, url)


# ---------------------------------------------------------------------------
# Page metadata with a bounded date stage (task 760).
#
# `trafilatura.extract_metadata` runs htmldate's `find_date` on every page, and
# htmldate raises on pages the web actually serves: a page with a time zone
# offset of 24 h or more ("+25:00", a `<time datetime>` typo) raises ValueError
# in `is_valid_date`; a Turkish month name the regex matches but its English /
# DE/FR/IT-only `TEXT_MONTHS` table lacks ("mayis", "nis", "haziran") raises
# KeyError in `regex_parse`. Either one escaped `_extract_in_pool` (which
# re-raises real worker exceptions on purpose) and marked the WHOLE DOMAIN
# `error` for a field -- the page date -- that no AW row carries. Ten domains of
# CC tranches 7-12 sat un-fetched on this.
#
# The date is the only stage allowed to fail here. On any exception we re-run
# the SAME extraction with `find_date` stubbed to "no date", so title / author /
# sitename are exactly what an undated page would give, and the caller gets a
# reason string to count. No locale table is patched: AW stores no page date on
# this path, so teaching a third-party table Turkish would buy nothing.
_METADATA_LOCK = threading.Lock()
DATE_PARSE_FAILED = "_date_parse_failed"   # non-drop COUNTER, like _recall_fallback


def extract_metadata_dateguard(html: str):
    """`(trafilatura Document | None, date_error | None)`.

    `date_error` is `"ExcType: message"` when the date stage raised and the
    metadata was re-derived without a date (`document.date` is then None);
    None when extraction was clean (output then byte-identical to a direct
    `trafilatura.extract_metadata(html)` call)."""
    import trafilatura
    try:
        return trafilatura.extract_metadata(html), None
    except Exception as first:  # noqa: BLE001 — only the date stage is expected to raise
        reason = f"{type(first).__name__}: {str(first)[:120]}"
    from trafilatura import metadata as _tm
    with _METADATA_LOCK:
        real = _tm.find_date
        _tm.find_date = lambda *a, **k: None
        try:
            return trafilatura.extract_metadata(html), reason
        finally:
            _tm.find_date = real


def coerce_title(value) -> Optional[str]:
    """A metadata title as text. trafilatura passes JSON-LD through verbatim,
    and publishers write `"headline": {"title": "..."}` (fibe-berlin.com, task
    760); a dict, list or number reaching `.strip()` crashed the domain. A
    string is returned UNCHANGED; a mapping/list is searched (title, headline,
    name, text, @value, then any value) for its first non-empty string,
    whitespace-collapsed; anything else is None."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for k in ("title", "headline", "name", "text", "@value"):
            got = coerce_title(value.get(k))
            if got and got.strip():
                return re.sub(r"\s+", " ", got).strip()
        for v in value.values():
            got = coerce_title(v)
            if got and got.strip():
                return re.sub(r"\s+", " ", got).strip()
        return None
    if isinstance(value, (list, tuple)):
        for v in value:
            got = coerce_title(v)
            if got and got.strip():
                return re.sub(r"\s+", " ", got).strip()
    return None



def _content_passages(url: str, html: str, meta: dict,
                      rejects: dict[str, int] | None = None) -> list[Passage]:
    """Content path: trafilatura markdown → chunks → Passage records.

    Mirrors the prior cc_ingest.process_record body. trafilatura + the chunker
    are imported lazily so the link-only caller (extract_outlinks) need not
    install them. `rejects` (optional counter) tallies extraction_junk drops
    per reason — surfaced by ingesters as the extraction_rejects stat."""
    if rejects is None:
        rejects = {}

    def _reject(reason: str):
        rejects[reason] = rejects.get(reason, 0) + 1

    try:
        import trafilatura
        from cytir.acquire import fetch_process as fp
    except Exception:  # noqa: BLE001
        return []
    # Precision-first extraction with a triggered recall fallback (task 274, bug
    # #3): nav-heavy commune/gov pages trafilatura precision-strips to near-empty
    # get a second, recall-oriented pass. Working pages hit the precision branch
    # and are byte-identical.
    # A source-preserving caller may extract and durably seal L1 before asking
    # this function to chunk it.  In that mode the supplied bytes are the one
    # and only chunk input: do not run the extractor a second time and merely
    # assume that it reproduced the retained Markdown.
    if "content_markdown" in meta:
        md = meta["content_markdown"]
        if not isinstance(md, str) or not md:
            raise ValueError("meta.content_markdown must be a non-empty string")
        extract_mode = meta.get("content_markdown_mode") or "preserved"
    else:
        md, extract_mode = _extract_content_md(html, url)
    if md is None:
        return []
    page_reason = extraction_junk(md)
    if page_reason:
        _reject(page_reason)
        return []
    if extract_mode == "recall_fallback" and meta.get("want_stats"):
        _reject("_recall_fallback")   # non-drop COUNTER (task 274 fallback fires)
    tmeta, date_error = extract_metadata_dateguard(html)
    if date_error:
        _reject(DATE_PARSE_FAILED)   # counted, page and domain continue (task 760)
    # Store the tracking-stripped url so `?ref=`/`?utm_*` capture variants of one
    # page collapse to a single point_id (dedup at ingest — task #31). Keep host/
    # scheme as captured; only referral/campaign params are dropped.
    store_url = strip_tracking_url(url)
    clean_title = (coerce_title(tmeta.title) if tmeta else None) or store_url
    # task 274 (bug #1): capture the entity/site-name suffix trafilatura strips
    # off the title (the commune name). site_name is stored separately and folded
    # into BOTH vectors (compose_index_text). For gov/entity types the DISPLAY
    # title also keeps the suffix (validated on the dardagny prod test: full title
    # "Gestion des déchets – Dardagny" lifted the page #20->#1); news/blog/docs
    # keep the cleaned title. This perturbs title-centrality inputs for official/
    # reference pages, so a REBUILD-WIDE apply must pass the gold zero-loss gate
    # first (ADR-053) — watch fedlex precise-cite focus.
    common = meta.get("passage_common", {})
    title, site_name = resolve_title_and_site_name(
        html, clean_title, store_url, common.get("source_type"))
    # Listing-shell gate (task 167): tag/category/archive/pagination taxonomy
    # pages carry a copied article snippet (so extraction_junk passes them) but
    # are near-identical nav shells that let one blog post occupy many top-10
    # slots. Drop when the URL is a listing path AND the title is listing
    # boilerplate ("One post tagged with…", "Category:…", "Page N of M").
    shell_reason = listing_shell(store_url, title)
    if shell_reason:
        _reject(shell_reason)
        return []
    return passages_from_markdown(
        store_url, md, title=title, site_name=site_name, common=common,
        lang_default=meta.get("lang_default", "en"), digest=meta.get("digest"),
        rejects=rejects, lang_url=url,
        selection=(meta["chunk_selection_for"](md) if meta.get("chunk_selection_for") else None))


def passages_from_markdown(store_url: str, md: str, *, title: str, site_name: Optional[str],
                           common: dict, lang_default: str = "en", digest: Optional[str] = None,
                           rejects: dict[str, int] | None = None,
                           lang_url: Optional[str] = None,
                           selection: Optional[ChunkerSelection] = None) -> list[Passage]:
    """The chunk half of the content path, from markdown that is already extracted: the language
    scope gate, the declared chunker, the per-chunk quality and junk gates, and one `Passage` per
    kept chunk. `_content_passages` calls it after its HTML-only steps (extraction, title, listing
    shell); a family whose bytes are not HTML (a PDF: `cytir.sources.families.pdf_replay`) calls it
    with the markdown its own extractor made. One tail, so a PDF row and an HTML row are chunked,
    gated and stamped by the same code (architecture-doctrine: one way to do one thing).

    `selection` (default: the process-wide `chunker_selection()`) lets a lane that picks its
    chunk policy PER PAGE (the ratified reference/generic selector) say which verified policy cut
    this page; the rows then stamp that policy's epoch."""
    if rejects is None:
        rejects = {}

    def _reject(reason: str):
        rejects[reason] = rejects.get(reason, 0) + 1

    from cytir.acquire import fetch_process as fp
    url = lang_url or store_url      # the URL as captured, for the language fallback
    # Language scope gate (task 166): detect on the extracted text, not the URL.
    # Confidently out-of-scope (non EN/DE/FR/IT) pages are dropped here so a CC
    # locale-mirror capture (e.g. blogs.nvidia.co.jp) can no longer be stamped
    # 'en' by the URL-default and leak into the index. Low-confidence → URL
    # heuristic fallback, kept (precision-favoring).
    lang, lang_conf = fp.detect_lang(md, url, lang_default)
    if lang_conf and lang not in fp.LANG_SCOPE:
        _reject("lang_out_of_scope")
        return []
    out: list[Passage] = []
    units = chunk_markdown_channels(md, selection)
    for unit in units:
        ch = unit.text
        if low_quality(ch):
            continue
        ch_reason = extraction_junk(ch)
        if ch_reason:
            _reject(ch_reason)
            continue
        # title + site_name are attached to EVERY chunk (task 274, bug #2): each
        # chunk is retrieved independently and needs the page-level entity token;
        # compose_index_text folds them into the dense + BM25 input at embed time.
        out.append(passage_from_chunk(
            unit, url=store_url, title=title, site_name=site_name,
            snippet=extractive_snippet(ch), lang=lang,
            content_digest=digest, **common))
    return out
