#!/usr/bin/env python3
"""The one home for "what is this fedlex law called, short?".

THE DEFECT THIS EXISTS TO FIX
-----------------------------
Production's fedlex records are titled

    Obligationenrecht (OR; SR 220) - Art. 411 - IV. Kreditnehmer und Auftraggeber

The abbreviation and the SR number are METADATA -- fedlex SPARQL's
`jolux:titleShort` and `jolux:historicalLegalId` -- and the short common name is
a curated table (`data/fedlex-law-names.v1.json`). The live loader
(`cytir.sources.families.fedlex`) resolves and composes them. The epoch-4
re-materialization (task 417's `materialize_r0_fedlex_anchored.py`) is
"from pinned AKN XML only" and therefore never resolved them at all: it took the
AKN `<docTitle>` and shipped 22,928 statutes whose own abbreviation does not
appear anywhere in their title or their BM25 surface. 0 of 22,928 carried an SR
number against 23,495 of the twin's 23,872 (task 515 sec.7.3), and every legal
gold the 1024 build lost is abbreviation-led.

WHY A SEALED ARTIFACT AND NOT A LIVE CALL
-----------------------------------------
The 417 design's offline-determinism property is worth keeping: a
re-materialization from pinned XML must be replayable byte-for-byte, years
later, with no network. A live SPARQL call at build time would make the title a
function of what fedlex answered that morning.

So the resolution happens ONCE, into an artifact
(`aw.fedlex-law-meta.v1`) that records what SPARQL answered, when, and from
which endpoint; the materializer takes it as an INPUT bound by sha256. Producing
it is the retired `seal_fedlex_law_meta` script; consuming it is `load_sealed()` below.

Both the live path and the sealed path compose the prefix through
`compose_title_prefix`, so there is one definition of what a fedlex title looks
like and `tests/test_fedlex_law_meta.py` holds it to production's formula.
"""
from __future__ import annotations

import json
import pathlib
import time

SCHEMA = "aw.fedlex-law-meta.v1"
CURATED_NAMES = "data/fedlex-law-names.v1.json"
SPARQL_ENDPOINT = "https://fedlex.data.admin.ch/sparqlendpoint"
JOLUX = "http://data.legilux.public.lu/resource/ontology/jolux#"
# fedlex's SPARQL language codes. `rm` is carried because the endpoint has it;
# Romansh is out of AW's scope (dropped 2026-06-21) and no caller requests it.
LANG_SPARQL = {"de": "DEU", "fr": "FRA", "it": "ITA", "en": "ENG", "rm": "ROH"}


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def load_curated_names(path: str | pathlib.Path | None = None) -> dict:
    """ELI -> {lang: short common name}. The one reader of the curated table."""
    p = pathlib.Path(path) if path else (repo_root() / CURATED_NAMES)
    doc = json.loads(p.read_text())
    if doc.get("schema") != "aw.fedlex-law-names.v1":
        raise ValueError(f"{p}: not aw.fedlex-law-names.v1")
    return doc["names"]


def compose_title_prefix(name: str, abbrev: str | None, sr: str | None) -> str:
    """`Name (ABBR; SR n)` -- byte-identical to `fedlex_dump.parse_akn`.

        paren = "; ".join(x for x in (abbrev, ("SR %s" % sr) if sr else None) if x)
        prefix = "%s (%s)" % (name, paren) if paren else name

    Both parts optional: a law with no abbreviation still gets its SR number
    ("Alkoholgrenzwerte im Strassenverkehr (SR 741.13)"), and a law with
    neither is its bare name rather than a name with empty parentheses.
    """
    paren = "; ".join(x for x in (abbrev, ("SR %s" % sr) if sr else None) if x)
    return "%s (%s)" % (name, paren) if paren else name


def short_name(curated: dict, eli: str, lang: str, abbrev: str | None,
               doc_title: str, warn=None) -> str:
    """The short common name, with the fallback ladder production uses.

    curated table -> the bare SPARQL abbreviation -> the first 60 characters of
    the formal docTitle. An ELI outside the table still gets a record; the
    fallback is LOUD, never silent, because a docTitle prefix is exactly the
    shape the epoch-4 defect shipped.
    """
    name = (curated.get(eli) or {}).get(lang)
    if name:
        return name
    if warn is not None:
        warn(f"fedlex {eli}/{lang}: no curated short name -- falling back to "
             f"the abbreviation/docTitle. Add a row to {CURATED_NAMES} if this "
             f"ELI is staying in the corpus.")
    return abbrev or doc_title[:60]


# --------------------------------------------------------------------------
# resolution (network) -- used only by the sealer, never at build time
# --------------------------------------------------------------------------
def resolve_law_meta(eli: str, langs: list[str], http, *,
                     endpoint: str = SPARQL_ENDPOINT,
                     min_interval_s: float = 0.5, log=None) -> dict[str, dict]:
    """Live SPARQL: {lang: {"title", "short", "sr"}}.

    Probes de/fr/it ALWAYS, even when not requested, so the SR-number
    cross-language backfill has real data to draw from: the SR number is
    language-invariant but fedlex answers it for only some languages (SchKG/en
    and DSG/de-fr-it each independently omit `historicalLegalId` while a sibling
    carries it -- verified 2026-07-30, task 358b).

    `min_interval_s` is the fedlex politeness floor (<= 2 req/s), enforced here
    rather than left to each caller.
    """
    probe = sorted(set(langs) | {"de", "fr", "it"})
    out: dict[str, dict] = {}
    for lang in probe:
        code = LANG_SPARQL.get(lang)
        if not code:
            continue
        query = """
        PREFIX jolux: <%s>
        SELECT ?title ?short ?sr WHERE {
          <https://fedlex.data.admin.ch/eli/%s/%s> jolux:title ?title .
          OPTIONAL { <https://fedlex.data.admin.ch/eli/%s/%s> jolux:titleShort ?short . }
          OPTIONAL { <https://fedlex.data.admin.ch/eli/%s/%s> jolux:historicalLegalId ?sr . }
        } LIMIT 1
        """ % (JOLUX, eli, lang, eli, lang, eli, lang)
        started = time.monotonic()
        response = http.post(endpoint, data={"query": query},
                             headers={"Accept": "application/sparql-results+json"})
        if response.status_code != 200:
            if log:
                log(f"  ! {eli}/{lang}: http {response.status_code}")
        else:
            rows = response.json()["results"]["bindings"]
            if rows:
                row = rows[0]
                out[lang] = {"title": row.get("title", {}).get("value"),
                             "short": row.get("short", {}).get("value"),
                             "sr": row.get("sr", {}).get("value")}
        slept = min_interval_s - (time.monotonic() - started)
        if slept > 0:
            time.sleep(slept)
    sr = next((m["sr"] for m in out.values() if m.get("sr")), None)
    for m in out.values():
        if not m.get("sr") and sr:
            m["sr"] = sr
    return out


# --------------------------------------------------------------------------
# the sealed artifact
# --------------------------------------------------------------------------
def load_sealed(path: str | pathlib.Path, expected_sha256: str | None = None) -> dict:
    """Read `aw.fedlex-law-meta.v1`, optionally bound to a digest.

    A build-time consumer passes the digest. A digest that does not match is a
    different set of titles, which is a different collection.
    """
    import hashlib
    p = pathlib.Path(path)
    raw = p.read_bytes()
    if expected_sha256:
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"{p}: sha256 {actual} != declared {expected_sha256}; the "
                f"sealed law-meta artifact is not the one this run was "
                f"authorised against")
    doc = json.loads(raw)
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{p}: not {SCHEMA}")
    if not doc.get("elis"):
        raise ValueError(f"{p}: carries no ELIs")
    return doc


def sealed_title_prefix(sealed: dict, eli: str, lang: str) -> str | None:
    """The composed `Name (ABBR; SR n)` for one (ELI, language), or None."""
    row = ((sealed.get("elis") or {}).get(eli) or {}).get(lang)
    if not row:
        return None
    prefix = row.get("title_prefix")
    return prefix if isinstance(prefix, str) and prefix.strip() else None
