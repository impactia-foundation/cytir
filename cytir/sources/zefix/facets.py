#!/usr/bin/env python3
"""Task 363 — register-facet extraction: status, dates, register-published
PEOPLE from the Zefix register's own data. One home for all four facets;
imported by the fetch/build scripts, never re-implemented locally.

## Sourcing (ADR-040-clean: register-native, no competitor search)

LINDAS `foj/zefix` graph (the established batch-SPARQL pattern, reused
verbatim from `pull_zweck_275.py`/`cytir.corpus.card_recipe.lindas_zweck`) was
re-verified for this task (task 363) with an EXHAUSTIVE distinct-predicate
scan (`SELECT DISTINCT ?p WHERE { GRAPH <...> { ?s ?p ?o } }`, 57 predicates
total, none truncated) and confirmed to carry ONLY identity/purpose/address:
`type, identifier(s), name, legalName, description (Zweck), address,
municipality`. No status, no date, no person field exists at the company
level in this graph — task 223's "no website field" finding generalizes to
"no status/date/person field" too. LINDAS is used HERE only for its one
remaining service: batched UID -> EHRAID resolution (the `CompanyEHRAID`
identifier sibling of the `CompanyUID` identifier already joined for Zweck),
same 250-UID-batch / indexed schema:value pattern, same politeness.

The actual facet DATA (status, SHAB mutation/registration dates, and the
register-published person roster) comes from `https://www.zefix.ch/ZefixREST
/api/v1/firm/{ehraid}` — the SAME backend the public zefix.ch website itself
calls to render a firm's public page to any anonymous visitor (verified live,
task 363: no API key, no session, no auth header; distinct from the
API-key-gated commercial "ZefixPublicREST" developer product, which 401s).
This is the register's own publication channel, not a competitor search API
(ADR-040 is about ranked web search results, not a government registry's own
data service — the same category LINDAS itself is in).

## What each field means (identity/spec notes)

- `published` = the MOST RECENT SHAB mutation date for the entity
  (`shabDate` top-level field == `shabPub[0].shabDate`). Record-currency
  semantics: "when did the register last say something changed about this
  entity", NOT "when was this card written" (that stays `last_seen`). For
  `source_type=entity_card` / `provenance=zefix_*`, `published` therefore
  means *register mutation recency*, a DIFFERENT semantic than the news
  layer's `published` (article publication date) or the docs layer's
  `published` (doc revision date) — same field name, per-source_type meaning,
  documented here per the task-363 brief; `published` is already a
  cross-class field (see `build_aw_b1`, `cytir/serve/
  rankhead_v2.py` FEATURES `has_published`) precisely because "how current is
  this record" is a general recency signal regardless of source_type.
- `founded` = the SHAB date of the entity's *Neueintragung* (new-registration)
  publication, when that publication is still within SHAB's queryable
  history. Coverage is PARTIAL by construction: SHAB's online archive does
  not reach back to every company's actual incorporation (older firms only
  show recent mutations, e.g. "infonic AG" task-363 sample: oldest visible
  shabPub is a 2019 address change, not the founding entry). We do NOT
  synthesize a `founded` date when no Neueintragung entry is visible
  (Fail Loud > guessed default) — `founded` is null for those, honestly.
  **`founded` is an entity ATTRIBUTE and a future HEAD FEATURE candidate —
  it is explicitly NOT wired as a hand-weighted ranking boost.** Per task
  283's finding on the ADR-019 authority-signal lane (`task 283 authority-
  signal.md`: "no scalar config reaches 0 regressions with non-zero lift" —
  every hand-tuned weight on a structural signal traded one regression class
  for another, repeatedly, across that lane's iterations before the fusion
  moved to a LEARNED per-query head, ADR-036/rankhead_v2), a bare "older is
  more authoritative" or "younger is riskier" scalar multiplier on `founded`
  would recreate exactly that failure mode. `founded` ships as a payload
  field only; if it ever earns a place in ranking it goes in as a rankhead
  feature (`cytir/serve/rankhead_v2.py` FEATURES list), the same route
  `published` already took (`has_published`), never a hand weight.
- `reg_status` = the raw Zefix register status string, passed through
  VERBATIM (not re-coded into an invented taxonomy — a status value this
  pipeline hasn't seen yet is preserved, not dropped or mis-mapped). Values
  observed live during the task-363 bulk pass: `EXISTIEREND` (active),
  `GELOESCHT` (deleted — the entity no longer exists), `IN_AUFLOESUNG`
  (dissolution/liquidation in progress — still a legal entity, winding
  down). Others may exist and are preserved as-seen. `dissolved` = a
  DELIBERATELY NARROW derived boolean (`status == "GELOESCHT" or
  deleteDate is not None`) meaning "no longer a legal entity" — an
  `IN_AUFLOESUNG` company is NOT flagged `dissolved` (it still exists; the
  distinction between "active", "in liquidation", and "gone" is exactly
  what `reg_status` carries raw, and collapsing it into one boolean would
  lose that). Per the task-363 brief: serving treatment of any of these
  states (demote? suppress? no-op?) is a rankhead-p2 FEATURE QUESTION, not
  a hand rule — this module only ships the fields, no serving logic
  changes here.
- people (register-published roles) — reconstructed by REPLAYING each
  entity's `shabPub` mutation history oldest-to-newest and tracking who is
  CURRENTLY registered: a person named in an "Eingetragene Personen[...]"
  segment is added/updated (name + role/signature text extracted); a person
  named in an "Ausgeschiedene Personen[...]" segment is removed. Only
  NATURAL PERSONS are kept (an entry is an organization, not a person, if it
  carries a UID `(CHE-...)` inline or is followed by "Revisionsstelle" with
  a company-suffix name pattern — see `_looks_like_org`). This reproduces
  exactly what SOGC/Zefix itself publishes as the entity's current
  signatories/board — nothing aggregated beyond the register's own content,
  matching the policy scope (`sources/POLICY.md` "Register-published
  individuals").

  Coverage note: person-segment parsing is regex/keyword-based (no LLM),
  verified against real DE-language mutation text (Vector8 AG, this task's
  acceptance instance) and cross-checked against real live FR/IT samples
  pulled from the SHAB feed (task 363 exploration). DE coverage is
  high-confidence; FR/IT segment-marker keyword sets are a best-effort
  multilingual translation of the same official marker phrases and were
  spot-checked but not exhaustively fuzzed — flagged here for the first
  post-apply audit sample to include FR/IT cantons specifically.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def dissolved(status: str | None, delete_date: str | None) -> bool:
    return bool(delete_date) or (status or "").upper() == "GELOESCHT"


# ---------------------------------------------------------------------------
# founded — the Neueintragung shabPub entry, if still in SHAB's window
# ---------------------------------------------------------------------------
_NEU_KEYS = {"status.neu"}
_NEU_TEXT_RX = re.compile(r"\(Neueintragung\)|\(Nouvelle inscription\)|\(Nuova iscrizione\)", re.I)


def founding_date(shab_pub: list[dict] | None) -> str | None:
    """shabPub is newest-first (Zefix's own ordering) — the founding entry,
    if visible at all, is the OLDEST (last in the list) AND explicitly a
    new-registration entry (mutationTypes key `status.neu`, or the
    `(Neueintragung)`-class marker in the free text — belt and suspenders,
    the structured key is preferred when present)."""
    for entry in reversed(shab_pub or []):
        keys = {m.get("key") for m in (entry.get("mutationTypes") or [])}
        if keys & _NEU_KEYS or _NEU_TEXT_RX.search(entry.get("message") or ""):
            return entry.get("shabDate")
    return None


# ---------------------------------------------------------------------------
# people — replay shabPub oldest -> newest, track current registered persons
# ---------------------------------------------------------------------------
# Section markers, DE verified live; FR/IT = translated official phrasing
# (see module docstring coverage note).
_ADD_MARK = re.compile(
    r"Eingetragene Personen(?: neu oder mutierend)?\s*:|"
    r"Personnes? (?:inscrite|nouvelle)s?(?: ou modifi\w+)?\s*:|"
    r"Person[ae] (?:iscritt|nuov)[ae](?:\s+o\s+modificat[ae])?\s*:",
    re.I)
_REMOVE_MARK = re.compile(
    r"Ausgeschiedene Personen(?: und erloschene Unterschriften)?\s*:|"
    r"Personnes? (?:radi[ée]e|ayant quitt\w+)[^:]*:|"
    r"Person[ae] (?:dimissionari[ae]|uscit[ae])[^:]*:",
    re.I)
# Any recognized section marker (used as the right boundary of a segment).
_ANY_MARK = re.compile(
    "|".join([_ADD_MARK.pattern, _REMOVE_MARK.pattern,
              r"Neue Adresse\s*:", r"Nouvelle adresse\s*:", r"Nuovo indirizzo\s*:",
              r"Notiz\s*:", r"Bemerkung\s*:"]), re.I)

# A natural-person entry starts "Lastname, Firstname," (possibly with
# particles). An organization entry carries an inline UID or legal-form
# suffix before the first comma-clause closes — excluded (natural persons
# only, per policy scope).
_PERSON_RX = re.compile(
    r"([A-ZÀ-ÖØ-Ý][\wÀ-ÖØ-öø-ÿ''.-]*(?:[\s-][A-ZÀ-ÖØ-Ý][\wÀ-ÖØ-öø-ÿ''.-]*)*),\s*"
    r"([A-ZÀ-ÖØ-Ý][\wÀ-ÖØ-öø-ÿ''.-]*(?:[\s-][A-ZÀ-ÖØ-Ý][\wÀ-ÖØ-öø-ÿ''.-]*)*),")
_ORG_HINT_RX = re.compile(r"\(CHE-|Revisionsstelle|AG\)|SA\)|GmbH\)|Sàrl\)|Sagl\)")


def _split_segments(text: str, marker_rx: "re.Pattern") -> list[str]:
    """Every occurrence of marker_rx, sliced to the next ANY recognized
    marker (or end of message)."""
    segs = []
    for m in marker_rx.finditer(text):
        start = m.end()
        nxt = _ANY_MARK.search(text, start)
        end = nxt.start() if nxt else len(text)
        segs.append(text[start:end])
    return segs


def _entries(segment: str) -> list[str]:
    # Person entries are ";"-separated in SHAB free text.
    return [e.strip() for e in segment.split(";") if e.strip()]


def _looks_like_org(entry: str) -> bool:
    return bool(_ORG_HINT_RX.search(entry[:120]))


def _parse_person(entry: str) -> tuple[str, str] | None:
    """-> (folded_name_key, display 'Firstname Lastname (role text)') or None."""
    if _looks_like_org(entry):
        return None
    m = _PERSON_RX.match(entry.strip())
    if not m:
        return None
    last, first = m.group(1).strip(), m.group(2).strip()
    if len(last) < 2 or len(first) < 2:
        return None
    role = entry[m.end():].strip(" ,.")
    # role text runs up to the signature clause or end; keep it short
    role = re.split(r"\.\s|\bmit\b\s+(?:Einzel|Kollektiv)|$", role)[0].strip(" ,.")
    key = f"{first.lower()}|{last.lower()}"
    display = f"{first} {last}" + (f" ({role})" if role else "")
    return key, display


def current_people(shab_pub: list[dict] | None) -> dict[str, str]:
    """Replay oldest -> newest; return {name_key: display} for persons
    CURRENTLY registered (never removed by a later Ausgeschiedene segment).
    Register-published roles ONLY — no enrichment from any non-register
    source (policy scope)."""
    current: dict[str, str] = {}
    for entry in reversed(shab_pub or []):  # oldest first
        text = entry.get("message") or ""
        for seg in _split_segments(text, _ADD_MARK):
            for raw in _entries(seg):
                parsed = _parse_person(raw)
                if parsed:
                    key, display = parsed
                    current[key] = display
        for seg in _split_segments(text, _REMOVE_MARK):
            for raw in _entries(seg):
                parsed = _parse_person(raw)
                if parsed:
                    current.pop(parsed[0], None)
    return current


def person_sparse_tokens(people: dict[str, str]) -> list[str]:
    """Discrete BM25 name tokens (first, last, full) for the sparse channel —
    NOT sentences, matches the ADR-060 sparse-channel contract."""
    toks: list[str] = []
    for key in people:
        first, last = key.split("|", 1)
        for t in (first, last, f"{first} {last}"):
            t = t.strip()
            if t and t not in toks:
                toks.append(t)
    return toks


def extract_facets(zefixrest_json: dict) -> dict:
    """The one call site: raw ZefixREST /firm/{ehraid} JSON -> the four
    facets, ready to merge into a card payload."""
    status = zefixrest_json.get("status")
    delete_date = zefixrest_json.get("deleteDate")
    shab_pub = zefixrest_json.get("shabPub") or []
    people = current_people(shab_pub)
    return {
        "published": zefixrest_json.get("shabDate"),
        "founded": founding_date(shab_pub),
        "reg_status": status,
        "reg_status_delete_date": delete_date,
        "dissolved": dissolved(status, delete_date),
        "people": people,  # {name_key: display}
        "people_tokens": person_sparse_tokens(people),
    }
