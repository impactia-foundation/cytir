#!/usr/bin/env python3
"""ADR-060 shared two-channel entity-card recipe.

Every independently refreshed content assertion first becomes an immutable
``aw.card-facet-event.v1`` event. The recipe reconciles only declared
``(source, facet)`` scopes against a compacted current-state bundle, then
projects the merged bundle into one ordinary chunk:

* dense = bounded natural-language company description;
* sparse = identity/taxonomy hooks plus every word in dense content.

Homepage, L1, Wikipedia, Wikidata, register purpose/people, FINMA,
taxonomy/category/product and predecessor assertions therefore share the same
retrieval semantics. Source provenance never becomes a ranking feature, and
readable source fields never enter the serving payload; only ``bundle_digest``
does. De-dilution remains: commerce/taxonomy/brand vocabulary is earned only
by curated commercial categories or explicit source evidence.

Channel primitives are imported from the task-299 gate-lineage implementation
rather than forked. Zefix ``Zweck`` still uses the task-253 LINDAS route; QIDs
still require P856 registrable-domain validation upstream.

HARD CONSTRAINTS unchanged from 299: ADR-040 (Wikidata/Wikipedia/LINDAS only,
NO competitor search APIs) · ADR-039 (honest per-record license; attribution/
license_version master-only) · ADR-027 (snippet <=25 words) · ADR-050
(agent_policy declared, not inferred) · QID by VALIDATION (P856 registrable-
domain == card domain), never a bare guess.
"""
from __future__ import annotations

import json
import hashlib
import re
import time
import urllib.parse

import cytir.core.ids as identity
from cytir.core import common as aw_common
from cytir.corpus import entity_card_website as ECW
from cytir.corpus import facet_bundle as FB

# ---- channel primitives: ONE home (task 299 gate-lineage code), imported ----
from cytir.extract.retailer_card_channels import (
    BASE_COMMERCE, CAT_TERMS, GEO_TERMS,
    brands_for, claims_qids, claims_strings, domain_label, drop_stopwords,
    first_sentences, registrable, sells_phrase, wd_api,
    wp_summary,
)

try:
    from cytir.corpus.sparse_subtok import expand_name_subtokens  # task-282 / ADR-063
except Exception:  # noqa: BLE001
    def expand_name_subtokens(_):
        return []

RECIPE = "adr060-two-channel-v2"
TODAY = __import__("datetime").date.today().isoformat()
CATEGORY_EVIDENCE_SCHEMA = "aw.card-category-evidence.v1"
CATEGORY_EVIDENCE_SOURCES = frozenset(
    {"finma_license", "predecessor_inherited"})

# ADR-039 honest per-record license, by dense description source
# (license -> compact dump; attribution/license_version -> master-only).
LICENSE = {
    "wikipedia": ("CC-BY-SA-4.0", "CC BY-SA 4.0", "Wikipedia contributors"),
    "wikidata": ("CC0-1.0", "CC0 1.0 Universal", "Wikidata contributors"),
    "zefix_zweck": ("Swiss public commercial register (Zefix/LINDAS, OGD)",
                    None, "Zefix — Federal Office of Justice, via LINDAS"),
    "aw_category_template": ("CC0-1.0", "CC0 1.0 Universal",
                             "AW category template (own derivation)"),
}

# A card is a compact, derived chunk -- not a second payload schema for every
# upstream feed.  Source-specific text can participate in the two universal
# retrieval channels, but it must never become a readable serving field.  The
# receipt is deliberately content-free and is retained master-only by the
# projection boundary below.
WEBSITE_RECEIPT_SCHEMA = ECW.WEBSITE_RECEIPT_SCHEMA


def _clean_content(value: str, cap: int | None = None) -> str:
    """Normalize bounded source text before it enters a card chunk.

    ``cap`` is a remaining Arctic-token allowance (not characters); ``None``
    spends the full ratified website budget. The unit and the default both
    live in ``cytir.corpus.entity_card_website`` so capture and composition cannot drift
    apart again (task 393 §5).
    """
    return ECW.normalized_content_tokens(value, cap)


def _validated_website_receipt(seed: dict, text: str, receipt: dict) -> dict:
    """Validate one exact, content-free website search-index receipt."""
    return ECW.validate_receipt(seed, text, receipt)


def admitted_website_content(seed: dict) -> tuple[str, dict | None]:
    """Return search-admitted website chunk text and a compact receipt.

    The Zefix-to-site join historically proved only site ownership and retained
    title/meta evidence; it did *not* carry page body text.  A future join may
    supply ``website_content`` only with this explicit receipt.  Keeping this
    check at the recipe boundary prevents an accidental URL-only join from
    silently becoming an unconsented web-content ingest.  A restrictive
    copyright status is not mislabeled as a reuse grant: it is admitted only
    for minimal search indexing, with ``train_ok=false`` retained.
    """
    text = _clean_content(seed.get("website_content") or "")
    if not text:
        return "", None
    receipt = seed.get("website_receipt")
    if not isinstance(receipt, dict):
        raise ValueError("website_content requires website_receipt")
    # The receipt is traceable but contains no body text, so it can safely be
    # retained under projection's master-only provenance envelope.
    return text, _validated_website_receipt(seed, text, receipt)


def complete_sparse_content(sparse_hooks_text: str, dense_content: str) -> str:
    """Keep hooks additive while making BM25 cover the complete chunk content.

    Sparse may contain identity aliases or translated taxonomy terms that are
    absent from prose, but it may not be a curated subset of the actual chunk.
    This is source-neutral: register roles, a website body and a Wikipedia lead
    all reach BM25 by exactly the same projection.
    """
    return drop_stopwords(" ; ".join(
        bit for bit in (sparse_hooks_text, dense_content) if bit))


def segmented_sparse_content(sparse_hooks_text: str, dense_content: str,
                             evidence_segments: list[str]) -> str:
    """Preserve source-event boundaries for source-neutral BM25 saturation.

    The complete dense chunk remains a segment, so sparse can never omit a
    dense word.  Each canonical evidence assertion is also a segment, allowing
    a short exact assertion to retain its own length-normalized weight.  The
    projection layer combines duplicate postings by max, so repeated sources
    do not manufacture term frequency.
    """
    from cytir.core.projection import SPARSE_SEGMENT_SEPARATOR  # lazy: no cycle

    segments: list[str] = []
    for value in [sparse_hooks_text, dense_content, *evidence_segments]:
        cleaned = drop_stopwords(value or "")
        if cleaned and cleaned not in segments:
            segments.append(cleaned)
    return SPARSE_SEGMENT_SEPARATOR.join(segments)


# ---------------------------------------------------------------------------
# Zefix Zweck via LINDAS (the task-253 scale route, promoted to its pipeline
# home; the task-253 pull script is the provenance).
# ---------------------------------------------------------------------------
LINDAS = "https://ld.admin.ch/query"
LINDAS_GRAPH = "https://lindas.admin.ch/foj/zefix"


def norm_uid(s: str | None) -> str | None:
    m = re.search(r"CHE[-.]?(\d{3})[-.]?(\d{3})[-.]?(\d{3})", s or "")
    return "CHE" + "".join(m.groups()) if m else None


def _sparql(q: str, timeout: int = 180):
    from cytir.consent.block import ConsentBlocked
    from cytir.sources.family import HttpSpec          # lazy: the client fails closed without an operator identity
    from cytir.sources.http import make_client
    data = urllib.parse.urlencode({"query": q}).encode()
    last = None
    for att in range(6):
        try:
            # consent-guarded (task 1621); the same Accept, identity encoding as urllib sent
            with make_client(HttpSpec(timeout=timeout, headers={
                    "accept": "application/sparql-results+json", "accept-encoding": "identity",
                    "content-type": "application/x-www-form-urlencoded"})) as client:
                r = client.post(LINDAS, content=data)
                r.raise_for_status()
                return json.loads(r.content)
        except ConsentBlocked:
            raise                                    # a declined host is not retried
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(4 * (att + 1))
    raise RuntimeError(f"LINDAS sparql failed: {last!r}")


def lindas_zweck(uids, batch: int = 250, sleep: float = 1.0) -> dict:
    """{CHEnnnnnnnnn: zweck text} for normalized UIDs. Indexed schema:value
    match (NOT a CONTAINS scan). Politeness ~1 query/s, 250 UIDs each."""
    uids = sorted({u for u in (norm_uid(x) for x in uids) if u})
    out: dict[str, str] = {}
    for i in range(0, len(uids), batch):
        vals = " ".join(f'"{u}"' for u in uids[i:i + batch])
        q = f"""PREFIX schema: <http://schema.org/>
SELECT ?uid ?desc WHERE {{ VALUES ?uid {{ {vals} }}
  GRAPH <{LINDAS_GRAPH}> {{
    ?idn schema:name "CompanyUID" ; schema:value ?uid .
    ?org schema:identifier ?idn ; schema:description ?desc .
}} }}"""
        d = _sparql(q)
        for b in d["results"]["bindings"]:
            u, desc = b["uid"]["value"], b["desc"]["value"]
            if u and desc and (u not in out or len(desc) > len(out[u])):
                out[u] = desc
        time.sleep(sleep)
    return out


# ---------------------------------------------------------------------------
# Wikidata: full entity + MULTILINGUAL facet labels
# ---------------------------------------------------------------------------
FACET_LANGS = ("en", "de", "fr", "it")


def wd_entity(qid: str) -> dict:
    return wd_api({"action": "wbgetentities", "ids": qid,
                   "props": "labels|aliases|claims|descriptions|sitelinks",
                   "languages": "|".join(FACET_LANGS), "format": "json"}
                  ).get("entities", {}).get(qid, {})


def multilang_labels(qids, aliases: bool = False) -> dict:
    """{qid: {lang: label}} in EN/DE/FR/IT — ALL four languages per facet QID
    (v1's batch_labels collapsed to one label; multilingual hooks need all).
    With aliases=True each entry also carries {"aliases": [..]} — the category
    NODE's own alternate names in all four languages."""
    qids = [q for q in dict.fromkeys(qids) if q]
    out: dict[str, dict] = {}
    props = "labels|aliases" if aliases else "labels"
    for i in range(0, len(qids), 45):
        chunk = qids[i:i + 45]
        d = wd_api({"action": "wbgetentities", "ids": "|".join(chunk),
                    "props": props, "languages": "|".join(FACET_LANGS),
                    "format": "json"})
        for qid, ent in d.get("entities", {}).items():
            labs = ent.get("labels", {})
            row = {lg: labs[lg]["value"] for lg in FACET_LANGS if lg in labs}
            if aliases:
                al = []
                for lg in FACET_LANGS:
                    for a in ent.get("aliases", {}).get(lg, []):
                        v = a.get("value", "").strip()
                        if v and v not in al:
                            al.append(v)
                row["aliases"] = al[:12]
            out[qid] = row
        time.sleep(1.0)
    return out


# Cross-language category bridge (task-351 board reinforcement, 2026-07-28):
# a card must be findable by its CATEGORY's terms in any mission language
# ("comptable genève" must reach a fiduciaire, not only "fiduciaire genève").
# The GENERAL mechanism — never hand-mapped word pairs — is the taxonomy NODE:
# each Wikidata category item (P452 industry / P1056 product / P106-class
# occupation nodes) carries labels + aliases in DE/FR/IT/EN; NOGA sector codes
# are the register-side equivalent (each NOGA node has official labels in all
# four languages). We emit the node's full 4-language label+alias set as
# discrete sparse hooks, stoplist-filtered downstream.
#
# P31 (instance-of) added 2026-07-30 (task 365 regression fix, Tim's
# evidence-graded ruling): the entity's own CLASS nodes ("Regionalbank",
# "Versicherer", …) are category evidence the entity itself carries — the
# pre-ADR-060 cards' category descriptors came from exactly these labels, and
# rebuilding without P31 left ~89k cards with name-only sparse hooks (the
# comm-personalloan-ch gate loss). De-dilution-compliant: a card's OWN class
# labels are never borrowed vocabulary.
BRIDGE_PROPS = ("P31", "P452", "P1056", "P106")


# ---------------------------------------------------------------------------
# EVIDENCE-GRADED term expansion (Tim's design ruling, 2026-07-30, task 365):
# product terms require product evidence; category terms require category
# evidence. A bare Zweck "Betrieb einer Bank" earns BANKING category terms
# only — never the consumer-credit vocabulary ("personal loan", mortgage,
# leasing …). Product-level vocab is earned ONLY when the evidence names it
# (Zweck listing Kredite/Hypotheken/Leasing, Wikidata P1056, a FINMA licence
# category that IS the product, or a predecessor card that earned them from
# such evidence). Every term channel below states its evidence class; the
# audit tags each card with the sources that (re)earned its terms
# (finma_license / zweck_product / zweck_category / wikidata_facet /
# predecessor_inherited) — foreshadows the facet-bundle provenance
# architecture (mini-spec in draft; tags only here).
# ---------------------------------------------------------------------------
def finma_category_terms(auth_rows) -> list[str]:
    """FINMA register (finma.ch uid.csv) = TOP category evidence for the
    financial class: authoritative licence categories, official labels in
    DE/FR/IT/EN. `auth_rows` = the entity's uid.csv rows (multi-licence
    entities have several). Returns the licence-category labels as discrete
    category terms — the category itself, never product vocab (a "Bank"
    licence earns banking-category terms only)."""
    terms: list[str] = []
    for row in auth_rows or ():
        for k in ("AuthorisationTypeDE", "AuthorisationTypeFR",
                  "AuthorisationTypeIT", "AuthorisationTypeEN"):
            v = (row.get(k) or "").strip()
            if v and v not in terms:
                terms.append(v)
    return terms


_PRED_DESCRIPTOR = re.compile(
    r"^\s*[^—]+—\s*(?P<cats>[^.]+?)(?P<terminated>\.|$)")


def predecessor_category_terms(old_snippet: str) -> list[str]:
    """Graded predecessor inheritance (task 365 ruling #3): the pre-ADR-060
    card snippet's descriptor segment ("Name — cat1, cat2, cat3. Official
    website.") holds CATEGORY descriptors that originally came from Wikidata
    class labels — category evidence, so they still qualify and may be
    inherited. Boilerplate ("Official website") and register-UID noise are
    dropped. Anything beyond category descriptors (none observed in the
    audited class) must NOT be inherited via this channel — over-broad
    predecessor vocab is dropped and REPORTED, never silently carried."""
    m = _PRED_DESCRIPTOR.match(old_snippet or "")
    if not m:
        return []
    out = []
    candidates = m.group("cats").split(",")
    if m.group("terminated") != ".":
        # A 25-word predecessor snippet can truncate before the descriptor
        # sentence's period. In that case only the leading comma-delimited
        # class is safe evidence; later fragments may already be narrative.
        candidates = candidates[:1]
    for cat in candidates:
        cat = cat.strip()
        if not cat or re.search(r"official website|CHE[-.\s]?\d|MWST|\bLEI\b",
                                cat, re.I):
            continue
        if cat not in out:
            out.append(cat)
    return out


def category_evidence_payload(*, sources=(), retained_terms=()) -> dict:
    """Durable category-evidence carried by a card across later refreshes.

    Task 365 originally wrote FINMA/predecessor evidence only to a one-off
    audit JSONL.  A later phase-3 refresh could therefore re-derive P31 but
    silently discard the two evidence-graded recovery channels.  Persist only
    the reviewed CATEGORY terms here (never product vocabulary); current
    Wikidata P31/P452/P1056/P106 terms are deliberately re-derived live.
    """
    sources = list(sources)
    unsupported = [s for s in sources if s not in CATEGORY_EVIDENCE_SOURCES]
    if unsupported:
        raise ValueError(
            f"unsupported category_evidence sources: {unsupported!r}")
    clean_sources = list(dict.fromkeys(sources))
    clean_terms = []
    seen = set()
    for raw in retained_terms:
        term = " ".join(str(raw).split()).strip()
        key = term.casefold()
        if term and key not in seen:
            seen.add(key)
            clean_terms.append(term)
    return {
        "schema": CATEGORY_EVIDENCE_SCHEMA,
        "sources": clean_sources,
        "category_terms": clean_terms,
    }


def retained_category_evidence(rec: dict) -> tuple[list[str], list[str]]:
    """Validate/read durable evidence from a prior canonical card.

    Unknown schemas or source classes fail closed instead of laundering an
    arbitrary old sparse bag into evidence-backed category hooks.
    """
    value = rec.get("category_evidence")
    if value is None:
        return [], []
    if not isinstance(value, dict) or value.get("schema") != CATEGORY_EVIDENCE_SCHEMA:
        raise ValueError("invalid category_evidence schema")
    sources = value.get("sources")
    terms = value.get("category_terms")
    if (not isinstance(sources, list) or not all(isinstance(x, str) for x in sources)
            or not isinstance(terms, list) or not all(isinstance(x, str) for x in terms)):
        raise ValueError("invalid category_evidence values")
    if any(x not in CATEGORY_EVIDENCE_SOURCES for x in sources):
        raise ValueError("unsupported category_evidence source")
    clean = category_evidence_payload(sources=sources, retained_terms=terms)
    return clean["category_terms"], clean["sources"]


def category_bridge_terms(cat_qids) -> list[str]:
    """All-language labels + aliases for a set of category/taxonomy node QIDs
    — the cross-language bridge tokens. CC0 (Wikidata)."""
    terms: list[str] = []
    for _qid, row in multilang_labels(cat_qids, aliases=True).items():
        for lg in FACET_LANGS:
            v = row.get(lg)
            if v and v not in terms:
                terms.append(v)
        for v in row.get("aliases", []):
            if v not in terms:
                terms.append(v)
    return terms


def wd_facet_terms(ent: dict) -> list[str]:
    """Discrete multilingual industry/product terms the entity itself carries:
    the 4-language category bridge over its P452/P1056/P106 nodes, plus the
    entity's own aliases. All CC0. Discrete terms, never sentences."""
    fq = []
    for prop in BRIDGE_PROPS:
        fq += claims_qids(ent, prop)
    terms = category_bridge_terms(fq) if fq else []
    for lg in FACET_LANGS:
        for a in ent.get("aliases", {}).get(lg, []):
            v = a.get("value", "").strip()
            if v and v not in terms:
                terms.append(v)
    return terms


def entity_aliases(ent: dict, name: str) -> list[str]:
    out: list[str] = []
    seen = {name.lower()}
    for lg in FACET_LANGS:
        for a in ent.get("aliases", {}).get(lg, []):
            v = a.get("value", "").strip()
            if v and v.lower() not in seen:
                seen.add(v.lower())
                out.append(v)
    return out[:8]


def wp_lead_for(ent: dict, pref_lang: str) -> str:
    """Wikipedia lead first sentence(s) via the entity's own sitelinks
    (P856-verified QID upstream). ~1-2 req/s politeness is the caller's rate."""
    sl = ent.get("sitelinks", {})
    for wk in (pref_lang + "wiki", "enwiki", "dewiki", "frwiki", "itwiki"):
        if wk in sl:
            lead = first_sentences(wp_summary(sl[wk]["title"], wk[:2]), n=2)
            if lead:
                return lead
    return ""


# ---------------------------------------------------------------------------
# The two channels
# ---------------------------------------------------------------------------
def dense_prose(name, kind_word, base_desc, *, cats=(), industry=(),
                products=(), year="", parent=None, geo="", domain="") -> str:
    """Natural-language prose profile (the 299 assembly, priority reordered
    upstream). NOT a keyword menu — ADR-060 dense channel."""
    bits = [f"{name} — {base_desc.rstrip('.')}."]
    if kind_word and not re.search(
            r"retail|shop|commerce|market|store|magasin|Händl|Handel|détail|vendo|negozi",
            base_desc, re.I):
        bits.append(f"It is an {kind_word}." if kind_word[0] in "aeiou"
                    else f"It is a {kind_word}.")
    if industry:
        bits.append(f"Industry: {', '.join(dict.fromkeys(industry))}.")
    if products:
        bits.append(f"Products/services: {', '.join(dict.fromkeys(products))}.")
    sp = sells_phrase(cats)
    if sp:
        bits.append(f"Product range: {sp}.")
    brands = brands_for(cats)
    if brands:
        bits.append("Carries major brands including " + ", ".join(brands[:22]) + ".")
    if year:
        bits.append(f"Founded {year}.")
    if parent:
        bits.append(f"Parent/owner: {parent}.")
    if geo:
        bits.append(f"Headquarters/market: {geo}.")
    if domain:
        bits.append(f"Official website of {name} ({domain}); AW indexes this "
                    f"entity as a source pointer, not its pages.")
    return " ".join(bits)[:1400]


def sparse_hooks(name, domain, *, aliases=(), facet_terms=(), cats=(),
                 country=None, commercial=False) -> str:
    """The BM25 hook bag: discriminative multi-char terms only, function-word
    stoplist applied (drop_stopwords = the ADR-060 filter), deduped.
    De-dilution: commerce/taxonomy/brand/geo vocab ONLY when commercial."""
    parts = [name, " ".join(aliases), domain_label(domain), " ".join(facet_terms)]
    if commercial:
        parts.append(BASE_COMMERCE)
        parts += [CAT_TERMS.get(c, "") for c in cats]
        parts.append(" ".join(brands_for(cats)))
        if country:
            parts.append(GEO_TERMS.get(country, ""))
    subtoks = expand_name_subtokens(" ".join([name, domain_label(domain)]
                                             + list(aliases)))
    if subtoks:
        parts.append(" ".join(subtoks))
    return drop_stopwords(" ; ".join(p for p in parts if p))


def snippet_for(name, base_desc, note) -> str:
    snip = f"{name}: {base_desc}".rstrip(".")
    return " ".join((snip + f". {note}").split()[:25])


# ---------------------------------------------------------------------------
# Register-facet extension (task 363): the four register facets — people
# (register-published roles), register mutation date, founding date, register
# status — layered onto a card produced by ANY recipe generation (this
# project's cards were built across tasks 224/253/275/296/351 with several
# base-description sources; this extension does not re-derive the base
# description, it enriches whatever text already exists). Governance: the
# PEOPLE facet is gated by `sources/POLICY.md@v3 §4.9.1` (board PR, task 363)
# — the dates/status facets carry no such gate (Tim's ruling, 2026-07-30) and
# may be applied on their own, independent of the people facet's PR status.
# See `cytir.sources.zefix.facets` for extraction (status/dates/people
# from the register's own ZefixREST/LINDAS data) — this half is TEXT assembly
# only, kept here (the one home for dense/sparse channel construction) so a
# future card generation inherits the same append-shape automatically.
def add_people_facet(dense_text: str, sparse_text: str,
                     people: dict[str, str]) -> tuple[str, str]:
    """Legacy sealed-artifact adapter for Tasks 363/383 only.

    New/recurring card builds must pass ``seed.register_people`` through
    ``build_two_channel`` so the people assertion enters the canonical event
    log before projection. This compatibility function appends
    register-published signatories/board/management to an
    EXISTING card's two channels. `people` = {name_key: display} from
    `cytir.sources.zefix.facets.current_people` (already natural-persons-only,
    already register-published-role-only — POLICY §4.9.1 scope is enforced
    upstream in the extractor, not here). Idempotent-safe to call once per
    card per pass; does not touch anything else in either channel.

    HELD BEHIND POLICY §4.9.1 (task 363): a caller must not fold this into
    a card that ships before the governance PR merges. This function itself
    has no gate — the gate is procedural (which artifact gets applied)."""
    if not people:
        return dense_text, sparse_text
    names = list(people.values())
    sentence = "Registered signatories/officers (commercial register): " + \
        "; ".join(names) + "."
    # NEVER cut (ADR-060 am., Tim 2026-10-07, ruling 27 T6): the old 1,600-char slice dropped officer
    # names (a card ended `... Marc Maret (...); Ni`). The register composition stays: purpose
    # first, then the named officers. No upper bound is ruled yet.
    new_dense = (dense_text.rstrip() + " " + sentence).strip()
    tokens = person_sparse_tokens_local(people)
    # Names remain useful additive hooks, but the complete role sentence must
    # enter sparse too.  Otherwise a register field gets privileged lexical
    # treatment while ordinary chunk content is invisible to BM25.
    new_sparse = complete_sparse_content(
        sparse_text + " ; " + " ".join(tokens), new_dense)
    return new_dense, new_sparse


def person_sparse_tokens_local(people: dict[str, str]) -> list[str]:
    """Local mirror-free wrapper: import lazily to avoid a hard dependency
    for callers that only use the identity/Wikidata half of this module."""
    from cytir.sources.zefix.facets import person_sparse_tokens
    return person_sparse_tokens(people)


def enrich_register_facets(rec: dict, facets: dict) -> dict:
    """Merge the non-gated facets (published/founded/reg_status/dissolved)
    into a card record's payload. Pure payload fields, no vector impact —
    safe to apply independently of the people-facet governance gate."""
    if facets.get("published"):
        rec["published"] = facets["published"]
    if facets.get("founded"):
        rec["founded"] = facets["founded"]
    if facets.get("reg_status") is not None:
        rec["reg_status"] = facets["reg_status"]
    if facets.get("reg_status_delete_date"):
        rec["reg_status_delete_date"] = facets["reg_status_delete_date"]
    if "dissolved" in facets:
        rec["dissolved"] = facets["dissolved"]
    return rec


#: task 502 defect 3 / ADR-094 §6a / ADR-103. An entity card's evidence unit is
#: the register row, not a crawled page (`build_two_channel` never fetches the
#: card's own `url` to build it -- only an optional homepage PROSE facet goes
#: through a live-consent-admitted fetch, tracked separately below). The old
#: card lane instead joined consent by URL against the *crawl* ledger, which
#: is why the task-467 anatomy needed a manual receipt join over 7,925 rows
#: to explain a "declined" count that was 61% an unreadable robots.txt, never
#: a refusal (task 502, declined anatomy). Stamping the basis here means a
#: future census reads it off the record instead of re-deriving it.
CONSENT_STATES = ("unreadable", "declined", "not_applicable", "ok")

#: Register scheme (folded: upper-case alphanumerics) -> ISO-2 of the register's
#: own country (task 2008).  A card whose payload `country` is empty gets the
#: country of the register that issued its `register_id`.
_SCHEME_COUNTRY = {
    "CHE": "CH", "CHEUID": "CH", "CHUID": "CH", "UID": "CH",
    "SIREN": "FR", "DEREGISTERNUMMER": "DE",
    "CABUSINESSNUMBER": "CA",
}
#: Wikidata P17 (country) QID -> ISO-2, for cards with no register field.
_P17_COUNTRY = {"Q39": "CH", "Q142": "FR", "Q183": "DE", "Q16": "CA",
                "Q38": "IT", "Q40": "AT", "Q30": "US", "Q145": "GB"}


def p17_country(ent: dict | None) -> str | None:
    """ISO-2 of a Wikidata entity's single mapped P17 country, else None
    (none, unmapped, or two different countries: ambiguity stays null)."""
    if not ent:
        return None
    found = {_P17_COUNTRY[q] for q in claims_qids(ent, "P17") if q in _P17_COUNTRY}
    return next(iter(found)) if len(found) == 1 else None


def register_country(rec: dict) -> str | None:
    """Payload `country` from the register source, in precedence order: the
    register scheme of `register_id`, then the `wikidata_<cc>` provenance of a
    country-head card.  None when neither names a country."""
    if rec.get("register_id"):
        scheme = "".join(c for c in str(rec.get("register_scheme") or "").upper()
                         if c.isalnum())
        if scheme in _SCHEME_COUNTRY:
            return _SCHEME_COUNTRY[scheme]
    m = re.fullmatch(r"wikidata_([a-z]{2})", str(rec.get("provenance") or ""))
    return m.group(1).upper() if m else None


def finish(rec, existing, recipe: str = RECIPE, phase: str | None = None,
           ent: dict | None = None):
    """Enrich-in-place override (reuse existing point_id/url so a promote
    MODIFY replaces in place) or net-new; digest forces the digest-keyed delta
    path to MODIFY, not additive-SKIP (task-299 §4 lesson)."""
    rec["fx351_recipe"] = recipe
    # Payload-only stamp (task 2008): never an input of the digest below or of
    # the dense/sparse text.  Ruled order (review 2016): explicit -> register
    # scheme -> wikidata_<cc> provenance -> Wikidata P17 -> earlier stamp, kept
    # never erased -> null.
    if not rec.get("country"):
        rec["country"] = (register_country(rec) or p17_country(ent)
                          or (existing or {}).get("country") or None)
    rec["digest"] = identity.content_digest(
        (rec.get("dense_text") or "") + "\n" + (rec.get("sparse_text") or ""))
    if phase:
        rec["adr060_phase"] = phase
    rec.setdefault("consent_basis", "register_row")
    rec.setdefault("consent_state", "not_applicable")
    rec.setdefault("non_refreshable", False)
    if rec["consent_state"] not in CONSENT_STATES:
        raise ValueError(f"unknown card consent_state: {rec['consent_state']!r}")
    if existing:
        rec["enrich_in_place"] = True
        rec["base_point_id"] = existing["point_id"]
        rec["base_url"] = existing["url"]
        rec["base_provenance"] = existing.get("provenance")
        rec["point_id"] = existing["point_id"]
        rec["url"] = existing["url"]
        rec["all_sites"] = [existing["url"]]
        if existing.get("trust_tier") is not None:
            rec["trust_tier"] = existing["trust_tier"]   # never regress authority
        # An earlier pass (or the ADR-103 backfill) may already have recorded
        # that this card's URL sits on an unreadable/declined origin. A fresh
        # re-emit that never checked crawl consent at all must not silently
        # erase that finding just because IT stamped the honest default.
        if (existing.get("non_refreshable") is True
                and rec["consent_state"] == "not_applicable"):
            rec["non_refreshable"] = True
            rec["consent_state"] = existing.get("consent_state", "unreadable")
        # Same never-regress precedent for a prior "ok" finding (a source-
        # scoped rebuild, or the site dropping out of
        # `admitted_website_content`, means THIS pass gathered no
        # website_policies at all): a fresh pass that stamped nothing must
        # not silently downgrade an already-admitted origin fetch to
        # "not_applicable" just because it didn't re-check this time.
        elif (existing.get("consent_state") == "ok"
                and rec["consent_state"] == "not_applicable"):
            rec["consent_state"] = "ok"
    else:
        rec["enrich_in_place"] = False
        canonical = aw_common.canonical_url(rec["url"])
        rec["point_id"] = (
            rec.get("point_id") or identity.mint_point_id(canonical, ""))
    return rec


def build_two_channel(seed: dict, existing: dict | None, *,
                      ent: dict | None, qid: str | None,
                      zweck: str | None, phase: str,
                      current_bundle: list[dict] | None = None) -> tuple[dict, dict]:
    """Generic ADR-060 card assembly. `seed` needs: domain, url, country, lang,
    kind ('retailer'|'marketplace'|'org'|...), cats (may be empty), optional
    parent/provenance/search. `ent` = wd_entity(qid) for a P856-VERIFIED qid
    (or None). `zweck` = LINDAS Zweck text (or None). Returns (record, audit).
    Commercial vocab fires only when seed['cats'] is non-empty (de-dilution)."""
    dom, cats = seed["domain"], list(seed.get("cats") or [])
    commercial = bool(cats)
    kind = seed.get("kind") or ("org" if not commercial else "retailer")
    kind_word = {"marketplace": "online marketplace",
                 "retailer": "online retailer"}.get(kind, "")
    audit = dict(domain=dom, qid=qid, enrich=bool(existing), phase=phase)

    name = (seed.get("name")
            or (existing or {}).get("title")
            or domain_label(dom).title())
    aliases, facets, industry, products, geo, year, parent_wd = [], [], [], [], "", "", []
    wp_lead = wd_desc = ""
    if ent:
        labels = ent.get("labels", {})
        name = (labels.get("en") or labels.get("de") or labels.get("fr")
                or labels.get("it") or {}).get("value", name)
        aliases = entity_aliases(ent, name)
        facets = wd_facet_terms(ent)
        wd_desc = (ent.get("descriptions", {}).get(seed["lang"])
                   or ent.get("descriptions", {}).get("en")
                   or ent.get("descriptions", {}).get("de")
                   or ent.get("descriptions", {}).get("fr")
                   or ent.get("descriptions", {}).get("it") or {}
                   ).get("value", "").strip()
        wp_lead = wp_lead_for(ent, seed["lang"])
        vlab = multilang_labels(claims_qids(ent, "P17") + claims_qids(ent, "P159")
                                + claims_qids(ent, "P127") + claims_qids(ent, "P749"))

        def lab1(prop):
            return [next(iter(vlab.get(q, {}).values()), "")
                    for q in claims_qids(ent, prop) if vlab.get(q)]
        f_hq, f_country = lab1("P159"), lab1("P17")
        parent_wd = lab1("P749") or lab1("P127")
        geo = ", ".join(dict.fromkeys(x for x in (f_hq[:1] + f_country) if x))
        inception = claims_strings(ent, "P571")
        year = inception[0].lstrip("+")[:4] if inception else ""
        # industry/products readable lines (first-available label is fine here;
        # the multilingual full set lives in the SPARSE facets)
        ml = multilang_labels(
            claims_qids(ent, "P31")
            + claims_qids(ent, "P452")
            + claims_qids(ent, "P1056"))
        industry = [next(iter(ml.get(q, {}).values()), "")
                    for q in (claims_qids(ent, "P31")
                              + claims_qids(ent, "P452")) if ml.get(q)]
        products = [next(iter(ml.get(q, {}).values()), "")
                    for q in claims_qids(ent, "P1056") if ml.get(q)]

    website_text, website_receipt = admitted_website_content(seed)
    now = seed.get("effective_at") or TODAY
    retrieved = seed.get("retrieved_at") or now
    entity = norm_uid(seed.get("register_id")) or qid or dom
    has_current = bool(current_bundle)
    proposed: list[dict] = []
    refreshed = {
        tuple(value) if isinstance(value, list) else value
        for value in (seed.get("refreshed_sources") or ())
    }

    def mark(source: str, facet: str) -> None:
        refreshed.add((source, facet))

    def emit(*, facet, value, source, source_url, source_id, evidence=None,
             lang=None, license, consent, attribution, extractor,
             effective_at=None, retrieved_at=None, artifact_pointer=None,
             selection=None, source_receipt=None):
        value = " ".join(str(value or "").split()).strip()
        if not value:
            return
        mark(source, facet)
        proposed.append(FB.make_event(
            entity=entity, facet=facet, value=value, lang=lang or seed["lang"],
            source=source, evidence=evidence or value, source_url=source_url,
            source_id=str(source_id), effective_at=effective_at or now,
            retrieved_at=retrieved_at or retrieved, extractor_version=extractor,
            license=license, consent=consent, attribution=attribution, task=phase,
            artifact_pointer=artifact_pointer, selection=selection,
            source_receipt=source_receipt))

    if website_text:
        emit(facet="prose", value=website_text, source="homepage",
             source_url=website_receipt["source_url"],
             source_id=website_receipt["source_url"],
             license=website_receipt["license"], consent="resolved-search-index",
             attribution=website_receipt["attribution"],
             extractor="entity-card-website-v1",
             retrieved_at=website_receipt["fetched_at"],
             artifact_pointer=website_receipt["raw_artifact_pointer"],
             selection={"depth": 0, "rule": "canonical homepage"},
             source_receipt=website_receipt)
    l1_pages = seed.get("l1_pages") or []
    if "l1_pages" in seed:
        mark("homepage_l1", "prose")
    # Homepage and depth-1 pages share ONE total website allowance (task 388's
    # ratified budget is a total, not a per-source cap), spent in the canonical
    # homepage -> L1 order. A page that arrives with no allowance left is
    # dropped from the projection, not silently granted its own budget.
    website_budget_left = (ECW.WEBSITE_TOKEN_BUDGET
                           - ECW.website_token_count(website_text))
    for page in l1_pages:
        receipt = page.get("receipt") or {}
        if website_budget_left <= 0:
            break
        content = _clean_content(page.get("content") or "",
                                 cap=website_budget_left)
        required = ("token_count", "selection_rule")
        if (any(receipt.get(key) in (None, "") for key in required)
                or receipt.get("depth") != 1
                or receipt.get("selection_rule") not in {"about", "services", "products"}
                or not content):
            raise ValueError(
                "L1 content requires a depth=1 approved receipt carrying "
                "selection_rule and an exact Arctic token_count")
        validated = _validated_website_receipt(seed, content, receipt)
        actual_tokens = ECW.website_token_count(content)
        try:
            receipt_tokens = int(receipt["token_count"])
        except (TypeError, ValueError) as exc:
            raise ValueError("L1 receipt token_count must be an integer") from exc
        if receipt_tokens != actual_tokens:
            raise ValueError("L1 receipt token_count mismatch")
        website_budget_left -= actual_tokens
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if receipt.get("content_sha256") != expected:
            raise ValueError("L1 receipt content hash mismatch")
        if registrable(receipt.get("source_url")) != registrable(seed["url"]):
            raise ValueError("L1 receipt must remain on card domain")
        emit(facet="prose", value=content, source="homepage_l1",
             source_url=validated["source_url"], source_id=validated["source_url"],
             license=validated["license"], consent="resolved-search-index",
             attribution=validated["attribution"], extractor="entity-card-website-v1",
             retrieved_at=validated["fetched_at"],
             artifact_pointer=(page.get("artifact_pointer")
                               or validated["raw_artifact_pointer"]),
             selection={"depth": 1, "rule": receipt["selection_rule"],
                        "token_count": receipt_tokens},
             source_receipt=validated)
    if ent is not None:
        if not qid:
            raise ValueError("Wikidata refresh requires qid")
        for facet in ("prose", "alias", "industry", "product", "taxonomy",
                      "geo", "founded", "parent"):
            mark("wikidata_facet", facet)
    if wp_lead:
        emit(facet="prose", value=wp_lead, source="wikipedia_lead",
             source_url=f"https://www.wikidata.org/wiki/{qid}", source_id=qid,
             license=LICENSE["wikipedia"][0], consent="dump-license",
             attribution=LICENSE["wikipedia"][2], extractor="wp-lead-v1")
    elif seed.get("wikipedia_refreshed") is True:
        mark("wikipedia_lead", "prose")
    if wd_desc:
        emit(facet="prose", value=wd_desc, source="wikidata_facet",
             source_url=f"https://www.wikidata.org/wiki/{qid}", source_id=qid,
             license=LICENSE["wikidata"][0], consent="CC0",
             attribution=LICENSE["wikidata"][2], extractor="wikidata-entity-v1")
    for value in aliases:
        emit(facet="alias", value=value, source="wikidata_facet",
             source_url=f"https://www.wikidata.org/wiki/{qid}", source_id=qid,
             license=LICENSE["wikidata"][0], consent="CC0",
             attribution=LICENSE["wikidata"][2], extractor="wikidata-entity-v1")
    for value in industry:
        emit(facet="industry", value=value, source="wikidata_facet",
             source_url=f"https://www.wikidata.org/wiki/{qid}", source_id=qid,
             license=LICENSE["wikidata"][0], consent="CC0",
             attribution=LICENSE["wikidata"][2], extractor="wikidata-entity-v1")
    for value in products:
        emit(facet="product", value=value, source="wikidata_facet",
             source_url=f"https://www.wikidata.org/wiki/{qid}", source_id=qid,
             license=LICENSE["wikidata"][0], consent="CC0",
             attribution=LICENSE["wikidata"][2], extractor="wikidata-entity-v1")
    for value in facets:
        emit(facet="taxonomy", value=value, source="wikidata_facet",
             source_url=f"https://www.wikidata.org/wiki/{qid}", source_id=qid,
             license=LICENSE["wikidata"][0], consent="CC0",
             attribution=LICENSE["wikidata"][2], extractor="wikidata-entity-v1")
    if geo:
        emit(facet="geo", value=geo, source="wikidata_facet",
             source_url=f"https://www.wikidata.org/wiki/{qid}", source_id=qid,
             license=LICENSE["wikidata"][0], consent="CC0",
             attribution=LICENSE["wikidata"][2], extractor="wikidata-entity-v1")
    if year:
        emit(facet="founded", value=year, source="wikidata_facet",
             source_url=f"https://www.wikidata.org/wiki/{qid}", source_id=qid,
             license=LICENSE["wikidata"][0], consent="CC0",
             attribution=LICENSE["wikidata"][2], extractor="wikidata-entity-v1")
    for value in parent_wd:
        emit(facet="parent", value=value, source="wikidata_facet",
             source_url=f"https://www.wikidata.org/wiki/{qid}", source_id=qid,
             license=LICENSE["wikidata"][0], consent="CC0",
             attribution=LICENSE["wikidata"][2], extractor="wikidata-entity-v1")
    if zweck is not None:
        mark("register_field", "prose")
    if zweck:
        emit(facet="prose", value=first_sentences(zweck, n=2, cap=340),
             source="register_field", source_url="https://lindas.admin.ch/foj/zefix",
             source_id=norm_uid(seed.get("register_id")) or dom,
             license=LICENSE["zefix_zweck"][0], consent="open-data-license",
             attribution=LICENSE["zefix_zweck"][2], extractor="lindas-zefix-zweck-v1")
    if "zweck_category_terms" in seed:
        mark("zweck_category", "category")
    for value in seed.get("zweck_category_terms") or ():
        emit(facet="category", value=value, source="zweck_category",
             source_url="https://lindas.admin.ch/foj/zefix", source_id=entity,
             license=LICENSE["zefix_zweck"][0], consent="open-data-license",
             attribution=LICENSE["zefix_zweck"][2], extractor="zweck-grade-v1")
    if "zweck_product_terms" in seed:
        mark("zweck_product", "product")
    for value in seed.get("zweck_product_terms") or ():
        emit(facet="product", value=value, source="zweck_product",
             source_url="https://lindas.admin.ch/foj/zefix", source_id=entity,
             license=LICENSE["zefix_zweck"][0], consent="open-data-license",
             attribution=LICENSE["zefix_zweck"][2], extractor="zweck-grade-v1")
    taxonomy_refreshed = (not has_current or seed.get("taxonomy_refreshed") is True)
    if taxonomy_refreshed:
        mark("taxonomy_expansion", "category")
        mark("taxonomy_expansion", "taxonomy")
        mark("taxonomy_expansion", "product")
    for value in cats if taxonomy_refreshed else ():
        emit(facet="category", value=value, source="taxonomy_expansion",
             source_url="aw://adr060-taxonomy", source_id="adr060-taxonomy-v1",
             license="CC0-1.0", consent="own-derivation", attribution="AW taxonomy",
             extractor="adr060-taxonomy-v1")
        if CAT_TERMS.get(value):
            emit(facet="taxonomy", value=CAT_TERMS[value], source="taxonomy_expansion",
                 source_url="aw://adr060-taxonomy", source_id="adr060-taxonomy-v1",
                 license="CC0-1.0", consent="own-derivation",
                 attribution="AW taxonomy", extractor="adr060-taxonomy-v1")
    for value in brands_for(cats) if taxonomy_refreshed else ():
        emit(facet="product", value=value, source="taxonomy_expansion",
             source_url="aw://adr060-taxonomy", source_id="adr060-taxonomy-v1",
             license="CC0-1.0", consent="own-derivation", attribution="AW taxonomy",
             extractor="adr060-taxonomy-v1")
    people = seed.get("register_people") or {}
    if "register_people" in seed:
        mark("register_field", "person")
    for value in (people.values() if isinstance(people, dict) else people):
        emit(facet="person", value=value, source="register_field",
             source_url="https://www.zefix.ch/", source_id=entity,
             license=LICENSE["zefix_zweck"][0], consent="open-data-license",
             attribution=LICENSE["zefix_zweck"][2], extractor="zefix-people-v1")
    if "finma_terms" in seed:
        mark("finma_license", "finma")
    for value in seed.get("finma_terms") or ():
        emit(facet="finma", value=value, source="finma_license",
             source_url="https://www.finma.ch/", source_id=entity,
             license=seed.get("finma_license") or "FINMA public register",
             consent="public-register", attribution="FINMA", extractor="finma-register-v1")
    if "predecessor_terms" in seed:
        mark("predecessor_inherited", "predecessor")
    for value in seed.get("predecessor_terms") or ():
        emit(facet="predecessor", value=value, source="predecessor_inherited",
             source_url=seed.get("predecessor_artifact") or "aw://legacy-card-artifact",
             source_id=seed.get("predecessor_point_id") or entity,
             license=seed.get("predecessor_license") or "derived-existing-card",
             consent="legacy-reconstruction", attribution="AW legacy artifact",
             extractor="predecessor-reconstruction-v1")

    register_identity_refreshed = (
        bool(seed.get("register_id"))
        and (not has_current or seed.get("register_identity_refreshed") is True))
    if register_identity_refreshed and seed.get("country"):
        emit(facet="geo", value=seed["country"], source="register_field",
             source_url="https://www.zefix.ch/", source_id=entity,
             license=LICENSE["zefix_zweck"][0], consent="open-data-license",
             attribution=LICENSE["zefix_zweck"][2], extractor="zefix-identity-v1")
    if register_identity_refreshed and "founded" in seed:
        mark("register_field", "founded")
        if seed.get("founded"):
            emit(facet="founded", value=seed["founded"], source="register_field",
                 source_url="https://www.zefix.ch/", source_id=entity,
                 license=LICENSE["zefix_zweck"][0], consent="open-data-license",
                 attribution=LICENSE["zefix_zweck"][2], extractor="zefix-identity-v1")
    if register_identity_refreshed and "legal_form" in seed:
        mark("register_field", "legal_form")
        if seed.get("legal_form"):
            emit(facet="legal_form", value=seed["legal_form"], source="register_field",
                 source_url="https://www.zefix.ch/", source_id=entity,
                 license=LICENSE["zefix_zweck"][0], consent="open-data-license",
                 attribution=LICENSE["zefix_zweck"][2], extractor="zefix-identity-v1")
    if register_identity_refreshed and "parent" in seed:
        mark("register_field", "parent")
        if seed.get("parent"):
            emit(facet="parent", value=seed["parent"], source="register_field",
                 source_url="https://www.zefix.ch/", source_id=entity,
                 license=LICENSE["zefix_zweck"][0], consent="open-data-license",
                 attribution=LICENSE["zefix_zweck"][2], extractor="zefix-identity-v1")

    current_rows = [dict(row) for row in (current_bundle or [])]
    if any(row.get("entity") != entity for row in current_rows):
        raise ValueError(f"{entity}: current bundle contains another entity")
    changes = FB.reconcile(current_rows, proposed, refreshed, task=phase,
                           effective_at=now, retrieved_at=retrieved)
    bundle_rows = FB.compact(current_rows + changes)
    if not any(row["facet"] == "prose" for row in bundle_rows):
        emit(facet="prose", value=f"{seed['country']} {kind_word or 'organization'}",
             source="aw_template", source_url="aw://entity-card-template",
             source_id="category-template-v1", license="CC0-1.0",
             consent="own-derivation", attribution="AW category template",
             extractor="adr060-v2")
        changes = FB.reconcile(current_rows, proposed, refreshed, task=phase,
                               effective_at=now, retrieved_at=retrieved)
        bundle_rows = FB.compact(current_rows + changes)
    entity_rows, digest = FB.entity_bundle(bundle_rows, entity)
    if not entity_rows:
        raise ValueError(f"{entity}: facet bundle has no current assertions")
    by_facet = {}
    for row in sorted(entity_rows, key=FB.projection_key):
        by_facet.setdefault(row["facet"], []).append(row["value"])
    prose = list(by_facet.get("prose") or [])
    prose += by_facet.get("predecessor", [])
    base_desc = " ".join(prose) if prose else f"{seed['country']} {kind_word or 'organization'}"
    aliases = list(dict.fromkeys(by_facet.get("alias", [])))
    industry = list(dict.fromkeys(by_facet.get("industry", [])
                                  + by_facet.get("category", [])
                                  + by_facet.get("legal_form", [])))
    products = list(dict.fromkeys(by_facet.get("product", []) + by_facet.get("finma", [])))
    facet_terms = list(dict.fromkeys(
        value for facet, values in by_facet.items() if facet != "prose"
        for value in values))
    source = "+".join(sorted({row["source"] for row in entity_rows}))
    licenses = sorted({row["license"] for row in entity_rows})
    lic = (licenses[0] if len(licenses) == 1 else
           "Composite source licenses/statuses: " + "; ".join(licenses))
    lic_ver = None
    attr = "; ".join(dict.fromkeys(row["attribution"] for row in entity_rows))

    projected_geo = ", ".join(dict.fromkeys(by_facet.get("geo", [])))
    projected_year = (by_facet.get("founded") or [""])[0]
    projected_parent = (by_facet.get("parent") or [None])[0]
    dense_text = dense_prose(
        name, kind_word, base_desc, cats=(), industry=industry,
        products=products, year=projected_year,
        parent=projected_parent, geo=projected_geo, domain=dom)
    if by_facet.get("person"):
        # never cut (ADR-060 am., ruling 27 T6): officer names are kept whole
        dense_text = (dense_text + " Registered signatories/officers: "
                      + "; ".join(by_facet["person"]) + ".")
    sparse_text = sparse_hooks(
        name, dom, aliases=aliases, facet_terms=facet_terms, cats=(),
        country=seed.get("country"), commercial=False)
    # Project all source text through the same sparse representation.  Hooks
    # remain additive; source provenance has no retrieval/ranking semantics.
    # Event boundaries are retained only for BM25 length normalization: every
    # event is treated identically and duplicate postings merge by max.
    sparse_text = segmented_sparse_content(
        sparse_text, dense_text, [row["value"] for row in entity_rows])
    snippet = snippet_for(
        name, base_desc,
        (f"{kind_word.capitalize()} ({seed['country']}); " if kind_word else "")
        + f"described from {source}.")

    rec = {
        "point_id": None, "url": seed["url"], "title": name, "snippet": snippet,
        # A partial refresh may intentionally leave identity aliases untouched.
        # Keep a supplied canonical identity string rather than deleting it
        # merely because its Wikidata source was outside this source scope.
        "names": seed.get("names") or " | ".join([name] + aliases),
        "register_id": seed.get("register_id"),
        "register_scheme": seed.get("register_scheme"), "lei": None,
        "lang": seed["lang"], "domain": dom, "anchor": None,
        "tier": seed.get("tier", "A"),
        "source_type": "entity_card", "entity_kind": "org",
        "provenance": seed.get("provenance", "retailer_card"),
        "category": seed.get("category", "ecommerce" if commercial else None),
        "retail_kind": kind if commercial else None, "cats": cats or None,
        "country": seed.get("country"), "qid": qid,
        "trust_tier": seed.get("trust_tier", 0.7), "last_seen": TODAY,
        "resolve_confidence": seed.get("resolve_confidence"),
        "all_sites": [seed["url"]], "paywalled": False,
        "agent_policy": "unknown",
        "description_source": source, "license": lic,
        "attribution": attr, "license_version": lic_ver, "wikidata_qid": qid,
        "parent_note": seed.get("parent"),
        # Shared stage/rebuild contract requires ``text`` for admission and
        # identity; explicit channels preserve the card representation.
        "text": dense_text, "dense_text": dense_text, "sparse_text": sparse_text,
    }
    website_policies = [
        row["source_receipt"] for row in entity_rows
        if row.get("source_receipt") is not None
    ]
    if website_policies:
        rec["search_ok"] = all(row["search_ok"] is True for row in website_policies)
        rec["train_ok"] = all(row["train_ok"] is True for row in website_policies)
        # The oldest check bounds the freshness of the composite representation.
        rec["consent_checked_at"] = min(row["checked_at"] for row in website_policies)
        # `admitted_website_content` only ever includes a receipt that already
        # passed `require_resolved_search_index_receipt` (search_ok=True,
        # status=ok) -- a card built with website prose therefore always has a
        # live-resolved, admitted origin fetch behind it: "ok", not the
        # register-only default `finish()` will otherwise stamp.
        rec["consent_state"] = "ok"
    # These are compact register lifecycle fields, not index text.  Passing
    # them through the shared recipe keeps a source-scoped re-emit aligned
    # with the full-rebuild payload contract without a task-specific patcher.
    if seed.get("register_facets") is not None:
        if not isinstance(seed["register_facets"], dict):
            raise ValueError("register_facets must be an object")
        enrich_register_facets(rec, seed["register_facets"])
    # Event log is the canonical master.  Emitters append these immutable rows
    # per source wave, compact them, and pass the resulting current bundle back
    # on a subsequent re-emit.  The card itself carries only its digest.
    rec["bundle_digest"] = digest
    rec = finish(rec, existing, phase=phase, ent=ent)
    audit.update(status="OK", title=name, description_source=source,
                 license=lic, n_facets=len(facets),
                 dense=dense_text[:150], sparse=sparse_text[:180],
                 facet_events=changes, bundle_digest=rec["bundle_digest"],
                 current_bundle=entity_rows)
    return rec, audit
