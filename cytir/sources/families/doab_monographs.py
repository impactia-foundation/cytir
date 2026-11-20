"""`doab_monographs_mitpress_p0` -- DOAB/OAPEN open-access monograph POINTER CARDS (task 558; on the Protocol, task 1237).

The register is the Directory of Open Access Books' own open REST API (`/rest/search`, `/rest/items/<uuid>`),
never the publisher's site: `direct.mit.edu` content pages are Cloudflare-challenged for every user agent
(task 556 S1.2), a technical barrier POLICY.md S2 U4 forbids circumventing. DOAB carries every field a pointer
card needs (title, authors, subject classification, language, publisher, a resolvable DOI).

A CARD, NOT A DEEP CARD (specs/oa-monographs.md, ratified by Tim's merge of PR #124, Option B):
  * url      the publisher's landing page, from the register's DOI by a redirect-only hop (`doi.org/<doi>` ->
             `Location:`); only the 30x header is read, never a byte of the gated content.
  * text     `title \n authors \n subjects`. NO abstract and no book prose of any kind (the representation
             contract is RULED). `sparse_text` = title + authors + publisher + doab_id (a register hook bag).
  * license  the BOOK's own instrument, read from the item page's `rel="license"` anchor (never DOAB's
             unrelated plain `CC BY 4.0` catalogue link); an absent or unreadable badge excludes the title.
  * consent  the pointer's own host (`direct.mit.edu`), resolved live by the framework BEFORE any DOAB request of
             a unit, never inherited from the register. The verdicts are in the pack manifest (the legacy driver
             stamped them on every row; its per-row `search_ok`/`consent_*` fields are the framework's now).

Units: one DOAB item (`uuid`). `discover` pages the publisher search (`offset=`, never `start=`: DSpace
silently ignores `start=`). `acquire` reads the item's metadata, applies the filter and seals what it found
(metadata, licence badge URL, resolved landing URL) as one JSON capture. A unit ends `excluded(no_doi |
no_license_badge | no_license_spdx | wrong_publisher_or_missing_fields | no_url | no_title_or_authors |
resolved_off_pointer_host)`: the legacy driver's own counted reasons. A transport failure (DOAB unreachable,
status 0) is NOT a reason: it raises and the unit is QUARANTINED (corpus-doctrine #3; the legacy driver counted
a failed licence-page GET as `no_license_badge`).

Options (`--opt KEY=VALUE`): `publisher` (default `The MIT Press`), `pointer_hosts` (comma list the resolved DOI must land
on, default `direct.mit.edu`: 8 of 317 MIT DOIs landed on a Crossref chooser, 1 on a microsite), `max_req_s` (default
2.0: the task 558 politeness budget, one pacer for every DOAB and doi.org request).
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse

from cytir.core import projection
from cytir.core.common import TIER_TRUST, extractive_snippet, sha1
from cytir.sources import api_fetch as AF
from cytir.sources.family import CHUNK_DECLARED, HttpSpec, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

REST_BASE = "https://directory.doabooks.org/rest"
OAI_BASE = "https://directory.doabooks.org/oai/request"

SOURCE_TYPE = "oa-monograph"
LEGACY_TIER = "B"          # sources/POLICY.md legacy A/B scheme (task 556 S4):
                            # authoritative peer-reviewed academic press, not
                            # a primary legal/gov source (reserved for A).
TRUST_TIER = TIER_TRUST[LEGACY_TIER]   # aw_common.TIER_TRUST -- one home for
                                        # the tier->trust_tier table (engineering
                                        # doctrine 1); never a local literal.

LANG_MAP = {"english": "en", "german": "de", "french": "fr", "italian": "it"}

# creativecommons.org/licenses/<slug>/<version>[/] -> SPDX-ish string.
# Case-insensitive on the slug (task 556's own sample mixes "BY-NC-ND" and
# "by-nc-nd" across otherwise-identical badges on different item pages).
_CC_RE = re.compile(
    r"creativecommons\.org/licenses/([a-zA-Z-]+)/([0-9.]+)/?", re.I)


class DoabError(RuntimeError):
    """A DOAB/doi.org response could not be read as expected."""




# ---------------------------------------------------------------------------
# transport: one paced httpx client for every DOAB / doi.org request
# ---------------------------------------------------------------------------
def make_fetch(client, pacer: AF.Pacer, retries: int = 3, sleep=None):
    """A GET `fetch(url) -> (status, body_bytes, final_url, headers)` over an `httpx.Client`.

    Redirects are followed; `final_url` is where the chain ended REGARDLESS of the terminal status, which is
    what `resolve_doi_url` reads (a Cloudflare 403 on the landing page still carries the landing URL). A
    transport failure is retried with backoff under the same pacer and, if it never recovers, reported as
    status 0: a CALLER failure, never a publisher decision."""
    sleep = sleep or time.sleep

    def fetch(url: str) -> tuple[int, bytes, str, dict]:
        for attempt in range(retries):
            pacer.wait()
            try:
                r = client.get(url, follow_redirects=True)
            except Exception:  # noqa: BLE001 -- retried, then reported as status 0
                if attempt == retries - 1:
                    return 0, b"", url, {}
                sleep(2 * (attempt + 1))
                continue
            return r.status_code, r.content, str(r.url), dict(r.headers)
        return 0, b"", url, {}
    return fetch

# ---------------------------------------------------------------------------
# DOAB REST search (enumeration) -- the ``offset=``/``limit=`` quirk
# ---------------------------------------------------------------------------
def doab_search_page(fetch, query: str, offset: int, limit: int
                     ) -> tuple[list[dict], int]:
    """One page of ``/rest/search`` -> ``(items, x_total_count)``.

    DOAB's DSpace REST API paginates on ``offset=``, NOT ``start=`` --
    ``start=`` is silently ignored and every page repeats page 1 (measured
    2026-09-09: two requests differing only in ``start=0`` vs ``start=300``
    returned the identical 300 uuids in the identical order). Regression-
    tested in
    ``tests/test_aw_doab_monographs.py::test_search_page_uses_offset_not_start``
    so this cannot be silently reintroduced by a future "cleanup".
    """
    url = (f"{REST_BASE}/search?"
          + urllib.parse.urlencode({"query": query, "offset": offset,
                                     "limit": limit}))
    status, body, _, headers = fetch(url)
    if status != 200:
        raise DoabError(f"DOAB search {url} -> HTTP {status}")
    items = json.loads(body)
    total = int(headers.get("x-total-count", 0) or headers.get("X-Total-Count", 0)
               or len(items))
    return items, total


def enumerate_all_uuids(fetch, query: str, page_size: int = 100
                        ) -> tuple[list[str], int]:
    """Every uuid the search returns, paged by ``offset=`` (see above).

    Returns ``(uuids, x_total_count)`` so the caller can log enumerated vs
    the register's own declared total (verification-doctrine 8: a headline
    count is re-derived, not trusted on its own).
    """
    uuids: list[str] = []
    offset = 0
    total = None
    while total is None or offset < total:
        items, page_total = doab_search_page(fetch, query, offset, page_size)
        total = page_total if total is None else total
        if not items:
            break
        uuids.extend(it["uuid"] for it in items if it.get("uuid"))
        offset += len(items)
    return uuids, (total or 0)


# ---------------------------------------------------------------------------
# item metadata (pure parse -- unit-tested against sealed fixtures)
# ---------------------------------------------------------------------------
def fetch_item_metadata(fetch, uuid: str) -> list[dict]:
    url = f"{REST_BASE}/items/{uuid}?expand=metadata"
    status, body, _, _headers = fetch(url)
    if status != 200:
        raise DoabError(f"DOAB item {uuid} -> HTTP {status}")
    doc = json.loads(body)
    return doc.get("metadata") or []


def parse_item_metadata(metadata: list[dict]) -> dict:
    """Flatten the REST ``metadata`` array into the fields this family needs.

    Pure function: input is exactly the JSON list the REST API returns under
    ``metadata``; no I/O. Repeated keys (``dc.contributor.author``,
    ``dc.subject.classification``) collect into lists; everything else keeps
    its last value (DOAB items do not repeat scalar keys in practice).
    """
    authors: list[str] = []
    subjects: list[str] = []
    flat: dict[str, str] = {}
    for m in metadata:
        key, value = m.get("key"), m.get("value")
        if value is None:
            continue
        if key == "dc.contributor.author":
            authors.append(value)
        elif key == "dc.subject.classification":
            subjects.append(value)
        else:
            flat[key] = value
    return {
        "title": flat.get("dc.title"),
        "authors": authors,
        "subjects": subjects,
        "language": flat.get("dc.language"),
        "dc_type": flat.get("dc.type"),
        "doi": flat.get("oapen.identifier.doi"),
        "handle_uri": flat.get("dc.identifier.uri"),
        "publisher_name": flat.get("publisher.name"),
        "publisher_identifier": flat.get("publisher.identifier"),
    }


def has_title_and_authors(meta: dict) -> bool:
    """Both a non-empty title and >=1 non-empty author name.

    ONE home for this rule (run()'s metadata-filter pass and build_record
    both call it) so the two can never drift the way they did before this
    was extracted: build_record silently excluded these rows with no named
    terminal in run()'s counted `excluded` dict, and the real MIT Press
    tranche run (task 558) lost 62 of 379 metadata-admitted candidates this
    way before it was caught. The real shape: DOAB edited volumes carry
    `dc.contributor.editor`, not `.author` -- `parse_item_metadata`
    deliberately does not fall back to it (an editor is not an author), so
    these have a title but no authors.
    """
    title_ok = bool((meta.get("title") or "").strip())
    authors_ok = bool([a for a in (meta.get("authors") or []) if a and a.strip()])
    return title_ok and authors_ok


def resolved_host_is_admissible(url: str | None, pointer_hosts) -> bool:
    """Does a resolved DOI URL land on one of the vetted pointer host(s)?

    Real shape found on the full MIT Press run (task 558): not every "The
    MIT Press" DOI resolves to direct.mit.edu. 8 of 317 metadata-admitted
    candidates terminated at ``chooser.crossref.org/?doi=...`` (a Crossref
    content-negotiation interstitial -- a page ABOUT the DOI, not a landing
    page an agent could use to reach the book) and 1 at a book's own
    third-party microsite (``islamic-pasts-futures.org``, not
    ``direct.mit.edu``). Neither is the allowlisted, vetted pointer host
    this family's POLICY PR covers; admitting either would silently widen
    the family past its declared/allowlisted scope. `pointer_hosts` is a
    collection of lower-cased hostnames.
    """
    if not url:
        return False
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host in pointer_hosts


def simplify_subject(thema: str) -> str:
    """``"thema EDItEUR::U ...::UYQM Machine learning"`` -> ``"Machine learning"``.

    Keeps only the leaf label (after the last ``::``) and strips any leading
    thema code token (``"UYQM "``) -- the code is a classification-scheme
    artifact, not a searchable term.
    """
    leaf = thema.split("::")[-1].strip()
    parts = leaf.split(" ", 1)
    if len(parts) == 2 and re.match(r"^[A-Z0-9]{2,6}$", parts[0]):
        return parts[1]
    return leaf


# ---------------------------------------------------------------------------
# per-title licence -- read the item's OWN badge, never the hosting
# ---------------------------------------------------------------------------
def extract_license_badge(html: str) -> str | None:
    """The ``href`` of the anchor carrying ``rel="license"`` -- the BOOK's
    own instrument. DOAB's page also carries a plain (no ``rel="license"``)
    ``CC BY 4.0`` link for its own compiled-metadata right; this function
    deliberately does not match it (corpus-doctrine: read the instrument,
    never the hosting -- here, never the wrong instrument on the same page).
    """
    m = re.search(r'<a\b[^>]*\brel="license"[^>]*\bhref="([^"]+)"', html, re.I)
    if not m:
        m = re.search(r'<a\b[^>]*\bhref="([^"]+)"[^>]*\brel="license"', html, re.I)
    return m.group(1) if m else None


def normalize_cc_license(url: str | None) -> str | None:
    """``creativecommons.org/licenses/by-nc-nd/4.0`` -> ``"CC-BY-NC-ND-4.0"``.

    Case-insensitive on the slug (task 556's own sample mixes cases across
    item pages). Returns ``None`` for anything not a recognisable CC URL --
    an unreadable badge excludes the title (fail-closed), it never guesses.
    """
    if not url:
        return None
    m = _CC_RE.search(url)
    if not m:
        return None
    slug, version = m.group(1).upper(), m.group(2)
    return f"CC-{slug}-{version}"


# ---------------------------------------------------------------------------
# DOI -> publisher landing URL, via the redirect HEADER ONLY
# ---------------------------------------------------------------------------
def resolve_doi_url(fetch, doi: str | None) -> str | None:
    """``https://doi.org/<doi>`` -> the publisher's landing URL.

    ``fetch`` follows redirects itself (see ``make_fetch``), so this is one
    call: urllib walks every ``Location:`` header internally and lands on
    the publisher's own URL, which is returned in ``final_url`` REGARDLESS
    of the terminal status code. This is why a Cloudflare content challenge
    on ``direct.mit.edu`` does not block pointer-card construction: the
    chain terminates in a 403, but the 403's own url (the thing Cloudflare
    challenged) is exactly the landing URL wanted, and this function never
    reads a byte of that response's body.

    Verified 2026-09-09 from the operator build box: ``doi.org/10.7551/mitpress/
    14207.001.0001`` -> 302 -> ``direct.mit.edu/books/book/5590/
    Distributional-Reinforcement-Learning`` (403, Cloudflare challenge on
    the body; the URL itself came through clean).
    """
    if not doi:
        return None
    _status, _body, final_url, _headers = fetch(f"https://doi.org/{doi}")
    if final_url and final_url.startswith("http") and "doi.org" not in final_url:
        return final_url
    return None


# ---------------------------------------------------------------------------
# record assembly (pure -- unit-tested)
# ---------------------------------------------------------------------------
def build_record(meta: dict, license_spdx: str | None, url: str | None,
                 *, target_publisher: str, consent: dict | None,
                 today: str) -> dict | None:
    """One pointer-card record, or ``None`` if this title cannot be admitted.

    Exclusion is EXPLICIT and every reason is distinct (never a bare
    ``return None`` with no trace) so a caller can report why a candidate
    dropped rather than just a shrinking count (corpus-doctrine principle 2:
    every row reaches exactly one terminal).
    """
    if not has_title_and_authors(meta):
        return None
    title = (meta.get("title") or "").strip()
    authors = [a.strip() for a in (meta.get("authors") or []) if a and a.strip()]
    if (meta.get("publisher_name") or "").strip() != target_publisher:
        return None
    if not url or not url.startswith("http"):
        return None
    if not license_spdx:
        return None
    if consent is not None and consent.get("search_ok") is not True:
        return None

    subjects = [simplify_subject(s) for s in (meta.get("subjects") or [])]
    author_str = "; ".join(authors)
    subject_str = "; ".join(dict.fromkeys(subjects))  # de-dup, keep order
    doab_id = meta.get("handle_uri", "").rsplit("/handle/", 1)[-1] or meta.get("handle_uri")

    text = "\n".join(p for p in (title, author_str, subject_str) if p)
    sparse_text = " ".join(p for p in (
        title, author_str, meta.get("publisher_name"), doab_id) if p)

    lang = LANG_MAP.get((meta.get("language") or "").strip().lower(), "en")
    host = urllib.parse.urlparse(url).hostname or ""

    rec = {
        "url": url,
        "anchor": None,
        "title": title,
        "authors": authors,
        "subjects": subjects,
        "snippet": extractive_snippet(text, 25),
        "text": text,
        "dense_text": text,
        "sparse_text": sparse_text,
        "lang": lang,
        "domain": host.lower(),
        "tier": LEGACY_TIER,
        "source_type": SOURCE_TYPE,
        "trust_tier": TRUST_TIER,
        "last_seen": today,
        "content_digest": sha1(text),
        "license": license_spdx,
        "publisher": meta.get("publisher_name"),
        "doab_id": doab_id,
        # ---- master-only (ADR-039: stripped from the compact dump; carried
        # here only for the zero-re-fetch escape hatch and audit) ----
        "_master": {
            "attribution": f"{meta.get('publisher_name')} (open-access monograph; "
                          "register: DOAB/OAPEN)",
            "license_version": license_spdx.rsplit("-", 1)[-1] if license_spdx else None,
            "doi": meta.get("doi"),
            "doab_handle_uri": meta.get("handle_uri"),
            "doab_catalogue_license": "CC-BY-4.0",  # DOAB's OWN compiled-
            # record right (task 556 S2.2) -- distinct from `license` above,
            # which is the BOOK's instrument. Never the served field.
        },
    }
    if consent is not None:
        rec["search_ok"] = consent.get("search_ok")
        rec["train_ok"] = consent.get("train_ok")
        rec["consent_state"] = "live"
        rec["consent_basis"] = consent.get("basis")
        rec["consent_source"] = consent.get("source")
        rec["consent_checked_at"] = consent.get("checked_at")
    return rec




# ---------------------------------------------------------------------------
# the family
# ---------------------------------------------------------------------------
DEFAULT_PUBLISHER = "The MIT Press"
DEFAULT_POINTER_HOSTS = ("direct.mit.edu",)
DEFAULT_MAX_REQ_S = 2.0
FILTER_NOTE = ("publisher.name exact match; dc.title and >=1 author required; a resolvable DOI required; a "
               "per-title rel=license badge required (fail-closed, no default license); consent resolved "
               "live per origin by the framework, never inherited.")


@register_family
class DoabMonographsFamily:
    family_id = "doab_monographs_mitpress_p0"
    route = "documented_source_api"
    direct_crawl = True
    chunk_policy_id = CHUNK_DECLARED
    http_spec = HttpSpec(timeout=30, follow_redirects=False)   # task 1556: the runner builds and closes the client
    require_records = True

    def _fetch(self, ctx: RunContext):
        if getattr(ctx, "_doab_fetch", None) is None:
            pacer = AF.Pacer(float(ctx.options.get("max_req_s", DEFAULT_MAX_REQ_S)))
            ctx._doab_fetch = make_fetch(ctx.http, pacer)
        return ctx._doab_fetch

    @staticmethod
    def _publisher(ctx: RunContext) -> str:
        return ctx.options.get("publisher", DEFAULT_PUBLISHER)

    @staticmethod
    def _hosts(ctx: RunContext) -> tuple[str, ...]:
        raw = ctx.options.get("pointer_hosts")
        return tuple(h.strip().lower() for h in raw.split(",") if h.strip()) if raw else DEFAULT_POINTER_HOSTS

    def discover(self, ctx: RunContext):
        query = f'publisher:"{self._publisher(ctx)}"'
        uuids, total = enumerate_all_uuids(self._fetch(ctx), query)
        ctx.notes.update({"query": query, "x_total_count": total, "enumerated": len(uuids),
                          "endpoint": f"{REST_BASE}/search", "filter_note": FILTER_NOTE,
                          "pointer_hosts": list(self._hosts(ctx))})
        return [Unit(key=u, url=f"{REST_BASE}/items/{u}?expand=metadata",
                     consent_hosts=self._hosts(ctx)) for u in uuids]

    def acquire(self, unit: Unit, ctx: RunContext):
        fetch = self._fetch(ctx)
        meta = parse_item_metadata(fetch_item_metadata(fetch, unit.key))   # DoabError -> quarantined
        if (meta.get("publisher_name") or "").strip() != self._publisher(ctx):
            raise UnitExcluded("wrong_publisher_or_missing_fields")
        if not has_title_and_authors(meta):
            raise UnitExcluded("no_title_or_authors")
        if not meta.get("doi"):
            raise UnitExcluded("no_doi")
        handle_uri = meta.get("handle_uri")
        status, body, _, _ = fetch(handle_uri) if handle_uri else (200, b"", "", {})
        if status == 0:
            raise DoabError(f"DOAB item page {handle_uri} unreachable")
        badge = extract_license_badge(body.decode("utf-8", "replace")) if status == 200 else None
        if not badge:
            raise UnitExcluded("no_license_badge")
        spdx = normalize_cc_license(badge)
        if not spdx:
            raise UnitExcluded("no_license_spdx")
        url = resolve_doi_url(fetch, meta["doi"])
        if not url:
            raise UnitExcluded("no_url")
        if not resolved_host_is_admissible(url, self._hosts(ctx)):
            raise UnitExcluded("resolved_off_pointer_host")
        sealed = {"meta": meta, "license_badge": badge, "license_spdx": spdx, "url": url}
        body = json.dumps(sealed, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return [Captured(unit, body, {"status": 200, "source_url": unit.url})]

    def extract(self, cap: Captured, ctx: RunContext):
        s = json.loads(cap.body)
        m = s["meta"]
        return [Doc(url=s["url"], text=m["title"], title=m["title"],
                    lang=LANG_MAP.get((m.get("language") or "").strip().lower(), "en"),
                    extra={"meta": m, "spdx": s["license_spdx"]})]

    def chunk(self, doc: Doc, ctx: RunContext):
        rec = build_record(doc.extra["meta"], doc.extra["spdx"], doc.url,
                           target_publisher=self._publisher(ctx), consent=None, today=ctx.today)
        if rec is None:
            raise RuntimeError(f"build_record returned None for a unit that passed every gate "
                               f"(url={doc.url!r}): a new exclusion path has no counted reason")
        # The framework's stage door wants an id on the row. The legacy card carried none (the b1 stage
        # minted it); the id is taken from that same projection (`cytir.core.projection`, the module the
        # contract's `identity.minted_by_modules` names: Tier B, digest-as-anchor), never re-composed here.
        projected = projection.project_record(rec)
        if projected is None:
            # Rejected at admission (site-search result URL, task 2048): the predicate's reject counter
            # already counted it inside project_record; the unit yields no row.
            return []
        rec["point_id"] = projected[0]
        return [rec]
