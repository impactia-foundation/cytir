#!/usr/bin/env python3
"""The register-first entity-card engine: verify, grade, compose, declare.

Task 621. The candidate list is the company register (`card_discovery_scope`);
this module is the three steps that turn a (register entity, candidate domain)
pair into a card, with every ratified clause the July one-off batches satisfied
in scripts and the rebuild then silently dropped:

  1. **verify** — the ADR-041 §4 verify contract, evaluated against page text a
     caller supplies. `UID-on-page` (checksum-valid) is a certainty override; a
     name match alone is never sufficient; one address token out of
     {street, postcode, city} rules out the foreign homonym.
  2. **grade** — ADR-041 §5. A pair carries a `resolve_confidence ∈ [0,1]` built
     from name-match × name-distinctiveness, a TLD factor, the best address token
     and the parked/redirect penalties. Admission is `>= threshold`; sub-threshold
     pairs are RETAINED with their score for the offline candidate pool, never
     discarded, so the threshold stays re-tunable without re-resolving.
  3. **compose** — the ADR-060a register-only surface, via the real
     `card_recipe_adr060` primitives. Identity, purpose and officers come from the
     register evidence and nowhere else (`card-officers-from-register-only`).

and the part that is new here, because its absence is what task 621 measured:

  4. **declare** — `assert_payload_contract()`. Every card this engine emits
     carries `register_id`, `register_scheme`, `country` and
     `resolve_confidence`, plus `legal_form` / `legal_form_code` once the
     representation contract declares them. The check FAILS CLOSED on a missing
     field and refuses a field that is present but empty, because the defect it
     exists to catch was a field that survived as a key and lost its value.

Why 4 exists, measured on production 2026-09-19 (`aw_e8`, 337,702 entity cards):

    legal_form         0 / 337,702      resolve_confidence  0 / 337,702
    legal_form_code    0 / 337,702      canton              0 / 337,702

ADR-041 §5 ratified *"Store the surviving score in each card payload
(`resolve_confidence`) so we can also re-gate live if needed."* It is unmet for
100% of the live population: nothing in production records at what confidence a
card's website association was proved, so a threshold change today requires
re-resolving 194,255 entities. The July generation (`aw_v4_sat`) carried the
score on all 6,172 rows of the batch this task audited. The field was not removed
by a decision; it stopped being written, and no contract could see it go.

`legal_form` is written exactly when `data/representation-contract.v1.json`
declares it, and not before — `legal_form_enabled()` reads the contract rather
than a hand-flipped flag, so the founder's merge IS the switch and there is no
window in which the contract says one thing and the producer does another. That
window is what this defect came out of: the task-161 card builder (line 257) emitted
the field, the ADR-060a recipe dissolved it into hook text, and because the
contract declared nothing either way nothing could tell that it had stopped
existing. `AW_CARD_LEGAL_FORM` stays as an explicit override in both directions
for a staging pack that wants the field early, receipted as such.

Run the tests: python3 -m pytest tests/test_aw_corpus_register_cards.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata

from cytir.core import ids as aw_b1_identity
from cytir.core.paths import data_path
import cytir.corpus.card_recipe as CR

__all__ = [
    "DEFAULT_THRESHOLD",
    "TLD_FACTOR",
    "PayloadContractRefusal",
    "uid_checksum_ok",
    "normalize_register_id",
    "fold_ascii",
    "name_distinctiveness",
    "find_address_tokens",
    "verify_pair",
    "grade_pair",
    "compose_register_card",
    "assert_payload_contract",
    "legal_form_enabled",
    "contract_declares",
    "CONTRACT_PATH",
]

#: ADR-041 admission threshold, calibrated in task 224 (`calib_result.json`).
DEFAULT_THRESHOLD = 0.80

#: ADR-041 §5 TLD factor: a jurisdiction-locked TLD carries lower homonym risk,
#: so it needs less corroboration. `.swiss` is an OFCOM-operated restricted TLD
#: (holder must be a Swiss entity); `.ch` is Swiss-registry but unrestricted;
#: a gTLD carries none of that and must earn its confidence from evidence.
TLD_FACTOR = {"swiss": 1.00, "ch": 0.95}
TLD_FACTOR_DEFAULT = 0.85

#: Aggregator/directory hosts a redirect to which disqualifies a pair. These are
#: proprietary competing directory products; harvesting them is the ADR-040
#: violation, and being redirected INTO one means the candidate is not the
#: entity's own site.
AGGREGATOR_HOSTS = frozenset({
    "moneyhouse.ch", "local.ch", "search.ch", "help.ch", "swissfirms.ch",
    "zefix.ch", "easymonitoring.ch", "monetas.ch", "firmenwissen.de",
    "northdata.com", "opencorporates.com", "dnb.com",
})

_UID_WEIGHTS = (5, 4, 3, 2, 7, 6, 5, 4)
_UID_RE = re.compile(r"CHE[\s\-]?(\d{3})[.\s\-]?(\d{3})[.\s\-]?(\d{3})", re.I)
_PARKED_RE = re.compile(
    r"coming\s+soon|under\s+construction|domain\s+(?:is\s+)?for\s+sale|"
    r"diese\s+domain|parkingcrew|sedoparking|this\s+domain\s+is\s+parked|"
    r"website\s+coming", re.I)

#: Legal-form tokens stripped before slug/name comparison. Not a legal-form
#: FILTER: Tim ratified "active company (any legal form) + validated live site ->
#: card" (specs/entity-cards-phase2-companies.md D7, 2026-07-10). These are
#: removed only so "Muster AG" and "muster.ch" compare as the same name.
_LEGAL_FORM_TOKENS = frozenset({
    "ag", "sa", "sarl", "sagl", "gmbh", "llc", "ltd", "limited", "inc",
    "holding", "group", "gruppe", "groupe", "co", "company", "kg", "ohg",
    "genossenschaft", "cooperative", "verein", "association", "associazione",
    "stiftung", "fondation", "fondazione", "foundation", "sca", "snc",
    "kollektivgesellschaft", "einzelfirma", "societe", "société", "societa",
    "società", "en", "liquidation", "the", "und", "and", "et", "e",
})


class PayloadContractRefusal(RuntimeError):
    """A card was built without a field the register-first contract requires."""


# --------------------------------------------------------------------- identity
def uid_checksum_ok(uid9: str) -> bool:
    """eCH-0097 CHE-UID mod-11 checksum. A valid UID on a page is a certainty."""
    if len(uid9) != 9 or not uid9.isdigit():
        return False
    s = sum(int(d) * w for d, w in zip(uid9[:8], _UID_WEIGHTS))
    chk = 11 - (s % 11)
    if chk == 11:
        chk = 0
    if chk == 10:
        return False
    return chk == int(uid9[8])


def normalize_register_id(value) -> str | None:
    """Strip every separator and upper-case. THE join key, not a nicety.

    Production carries the Swiss scheme under three spellings — `CHE` 92,187
    (undotted), `CHE-UID` 88,801 (dotted) and `CH-UID` 587 (measured in `aw_e8`,
    2026-09-19). A raw `register_id ==` join across two stores misses roughly
    half the population in one direction or the other. Every count that crosses
    a store boundary must normalise first (corpus-doctrine 5).
    """
    if value is None:
        return None
    s = re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()
    return s or None


def fold_ascii(s: str) -> str:
    """NFKD-fold to ASCII lower-case, for name<->slug comparison only."""
    if not s:
        return ""
    d = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in d if not unicodedata.combining(c)).lower()


def _name_tokens(name: str) -> list[str]:
    toks = [t for t in re.split(r"[^a-z0-9]+", fold_ascii(name)) if t]
    kept = [t for t in toks if t not in _LEGAL_FORM_TOKENS and len(t) > 1]
    return kept or toks


def name_distinctiveness(name: str) -> float:
    """How much a name identifies on its own: rare/long high, generic low.

    ADR-041's principled fix for the generic-name flood that sank task 220's
    SIREN precision to 55%. "Swiss Solutions AG" must not clear on a name match;
    a coined token like "Galaxus" may.
    """
    toks = _name_tokens(name)
    if not toks:
        return 0.0
    longest = max(len(t) for t in toks)
    # length of the most distinctive token, saturating at 12 chars
    length_score = min(longest, 12) / 12.0
    # multi-token coined names are more distinctive than a single dictionary word
    breadth = min(len(toks), 3) / 3.0
    generic = {"swiss", "suisse", "schweiz", "solutions", "services", "consulting",
               "immobilien", "immobilier", "trading", "invest", "management",
               "partners", "technologies", "systems", "digital", "media", "food",
               "bau", "transport", "treuhand", "beratung", "atelier", "studio",
               "sport", "garage", "shop", "store", "clinic", "praxis"}
    generic_share = sum(1 for t in toks if t in generic) / len(toks)
    score = 0.55 * length_score + 0.45 * breadth
    return max(0.0, min(1.0, score * (1.0 - 0.6 * generic_share)))


def find_address_tokens(page_text: str, *, street=None, postcode=None,
                        city=None) -> list[str]:
    """Which of {street, postcode, city} appear on the page, best first.

    A single address token rules out the foreign homonym: a US or German site
    does not carry the Swiss town, NPA or street (ADR-041 §4). The page text must
    be the WHOLE document — task 224's BUG-2 was a `html[:150000]` clip that
    discarded the footer, where imprints live; impactIA's address tokens start at
    offset 161,507 of a 260,317-char homepage.
    """
    hay = fold_ascii(page_text or "")
    out = []
    # street first (most specific), then postcode, then city
    st = fold_ascii(street or "").strip()
    if st:
        # compare on the street NAME, not the number: registers and sites
        # disagree on "10" vs "10bis" vs "no 10" far more than on the name
        core = re.sub(r"[\d,\.]+", " ", st).strip()
        core = re.sub(r"\s+", " ", core)
        if len(core) >= 4 and core in hay:
            out.append(f"street:{street}")
    pc = re.sub(r"\s+", "", str(postcode or ""))
    if pc and re.search(r"(?<!\d)" + re.escape(pc) + r"(?!\d)", hay):
        out.append(f"postcode:{postcode}")
    ct = fold_ascii(city or "").strip()
    if len(ct) >= 3 and re.search(r"\b" + re.escape(ct) + r"\b", hay):
        out.append(f"city:{city}")
    return out


# ----------------------------------------------------------------------- verify
def verify_pair(*, register_id: str, legal_name: str, domain: str,
                page_text: str, http_status: int | None = None,
                final_host: str | None = None,
                street=None, postcode=None, city=None) -> dict:
    """The ADR-041 §4 verify contract. Returns the evidence, not a verdict.

    Accept a (entity, domain) pair iff
        UID-on-page (checksum-valid)  OR  name-match AND >=1 address token.
    Name alone is never sufficient — that is the foreign-homonym failure
    (`kitchenproject.com`, `anwerina.com`, and the `comodo.com` -> "Comodo Sàrl"
    false positive task 224 caught in its own 25-card sample).
    """
    uid_norm = normalize_register_id(register_id) or ""
    uid_digits = uid_norm[3:] if uid_norm.startswith("CHE") else ""
    uid_on_page = False
    for m in _UID_RE.finditer(page_text or ""):
        found = "".join(m.groups())
        if uid_checksum_ok(found) and found == uid_digits:
            uid_on_page = True
            break

    hay = fold_ascii(page_text or "")
    full = fold_ascii(legal_name)
    name_full = bool(full) and full in hay
    toks = _name_tokens(legal_name)
    clean = " ".join(toks)
    name_clean = bool(clean) and clean in hay
    # every significant token present somewhere (handles reordered brandings)
    name_tokens_all = bool(toks) and all(
        re.search(r"\b" + re.escape(t) + r"\b", hay) for t in toks)
    name_match = name_full or name_clean or name_tokens_all

    addr = find_address_tokens(page_text, street=street, postcode=postcode,
                               city=city)
    parked = bool(_PARKED_RE.search(page_text or "")) and len(page_text or "") < 20000
    host = (final_host or "").lower().lstrip(".")
    redirect_aggregator = any(host == a or host.endswith("." + a)
                              for a in AGGREGATOR_HOSTS)

    return {
        "uid_on_page": uid_on_page,
        "name_full": name_full,
        "name_clean": name_clean,
        "name_tokens_all": name_tokens_all,
        "name_match": name_match,
        "address_tokens": addr,
        "parked": parked,
        "redirect_aggregator": redirect_aggregator,
        "http_status": http_status,
        "page_chars": len(page_text or ""),
        "contract_satisfied": bool(
            not redirect_aggregator and not parked
            and (uid_on_page or (name_match and addr))),
    }


# ------------------------------------------------------------------------ grade
def grade_pair(evidence: dict, *, legal_name: str, domain: str) -> float:
    """ADR-041 §5 graded confidence. Stored in the payload; never thrown away.

    The score subsumes the per-TLD rule automatically, so there is no brittle
    if/else chain: a `.swiss` name-match clears on less corroboration than a
    `.com` name-match, because the TLD factor is a multiplier and the address
    token is an additive term.
    """
    if evidence.get("redirect_aggregator"):
        return 0.0
    if evidence.get("uid_on_page"):
        return 1.0
    if not evidence.get("name_match"):
        return 0.0
    tld = (domain or "").rsplit(".", 1)[-1].lower()
    tld_f = TLD_FACTOR.get(tld, TLD_FACTOR_DEFAULT)
    base = name_distinctiveness(legal_name)
    if evidence.get("name_full"):
        base = min(1.0, base + 0.15)
    score = base * tld_f
    addr = evidence.get("address_tokens") or []
    kinds = {a.split(":", 1)[0] for a in addr}
    if "street" in kinds:
        score += 0.45
    elif "postcode" in kinds:
        score += 0.35
    elif "city" in kinds:
        score += 0.22
    if evidence.get("parked"):
        score -= 0.5
    return max(0.0, min(1.0, round(score, 4)))


# ---------------------------------------------------------------------- compose
#: The one authority on whether a payload field may be written.
CONTRACT_PATH = data_path("representation-contract.v1.json")


def contract_declares(field: str, *, path=None) -> bool:
    """Does `data/representation-contract.v1.json` declare this payload field?

    Fails CLOSED: an unreadable or malformed contract means "not declared", so a
    producer never writes a field on the strength of a file it could not parse.
    """
    try:
        doc = json.loads((path or CONTRACT_PATH).read_text(encoding="utf-8"))
        fam = doc["families"]["entity_card"]
    except Exception:                                       # noqa: BLE001
        return False
    declared = set(fam.get("declared_payload_fields") or [])
    declared |= set(fam.get("identity_fields") or [])
    declared |= set(fam.get("identity_fields_each") or [])
    declared |= set(fam.get("sparse_identity_fields") or [])
    return field in declared


def legal_form_enabled(env: dict | None = None, *, path=None) -> bool:
    """`legal_form` is written exactly when the representation contract declares it.

    The ratification IS the switch. The alternative — a flag somebody flips by hand
    after the contract PR merges — leaves a window in which the contract says one
    thing and the producer does another, which is the disagreement this project has
    already paid for twice: the task-161 card builder (line 257) emitted `legal_form`,
    the ADR-060a recipe dissolved it into hook text, and because the contract
    declared nothing either way, no check could tell that the field had stopped
    existing (0 non-null on all 337,702 live cards).

    `AW_CARD_LEGAL_FORM` remains as an explicit override in both directions: `1` to
    write the field before the declaration lands (a staging pack, receipted as such),
    `0` to withhold it after. An unset variable defers to the contract.
    """
    src = os.environ if env is None else env
    raw = str(src.get("AW_CARD_LEGAL_FORM", "")).strip().lower()
    if raw in {"1", "true", "yes"}:
        return True
    if raw in {"0", "false", "no"}:
        return False
    return contract_declares("legal_form", path=path)


def compose_register_card(seed: dict, *, evidence: dict, confidence: float,
                          env: dict | None = None) -> dict:
    """The ADR-060a register-only surface + the aw_b1 identity + the payload.

    `seed` keys (all from the REGISTER, never the website):
        register_id, register_scheme, legal_name, purpose (Zweck), officers
        (list[str], may be empty), legal_form, legal_form_code, canton, locality,
        postcode, street, country, lang, domain, url, license, provenance,
        register_url, trust_tier
    """
    name = seed["legal_name"]
    purpose = " ".join(str(seed.get("purpose") or "").split())
    officers = [o for o in (seed.get("officers") or []) if o]

    # DENSE = register composition, head-first (ADR-060a). The entity's own
    # website contributes nothing to identity, purpose or officers.
    #
    # When the register publishes no purpose the composition falls back to the
    # register's OTHER identity fields — legal form and seat — never to the
    # website and never to a bare title. A card whose entire dense surface is its
    # own title has no surface at all: it is retrievable only by its name, which
    # is what the sparse channel is for. 78 of the 715 candidates in this task's
    # own shortlist have no Zefix `Zweck`, so this branch is the measured case,
    # not a defensive one. The shape matches what the July generation used
    # ("<name> — AG, Baar. Registered Swiss company.").
    if purpose:
        dense = f"{name} — {purpose}"
    else:
        seat = ", ".join(str(x) for x in (seed.get("locality"), seed.get("canton"))
                         if x)
        bits = [b for b in (seed.get("legal_form"), seat) if b]
        dense = (f"{name} — {'. '.join(bits)}" if bits else name)
        dense = dense.rstrip(".") + ". Registered in the "
        dense += {"CH": "Swiss commercial register"}.get(
            seed.get("country"), "public company register") + "."
    if not dense.endswith("."):
        dense += "."
    if officers:
        dense += (" Registered signatories/officers (commercial register): "
                  + ", ".join(officers) + ".")

    # SPARSE = the register hook bag (ADR-060 §b): hooks, never sentences.
    # Built by the RATIFIED primitive, not by hand — `CR.sparse_hooks` is what
    # applies the domain label, the ADR-063 alphanumeric sub-token expansion and
    # the ADR-060 stop-list. `commercial=False`: a register card gets no
    # commerce/brand/geo vocabulary it did not earn (the de-dilution rule).
    rid_norm = normalize_register_id(seed["register_id"])
    facet_terms = [seed["register_id"], rid_norm]
    facet_terms += [str(t) for t in (seed.get("locality"), seed.get("canton"),
                                     seed.get("legal_form")) if t]
    facet_terms += officers
    hook_text = CR.sparse_hooks(name, seed["domain"],
                                aliases=tuple(seed.get("aliases") or ()),
                                facet_terms=tuple(facet_terms),
                                country=seed.get("country"), commercial=False)
    sparse = CR.segmented_sparse_content(hook_text, dense, [])

    # IDENTITY — the ratified scheme. entity_card is Tier C, so the anchor
    # component is "" and the id is uuid5(AW_NAMESPACE, url + "#"). This is the
    # id production already holds for a card's URL: an upsert REPLACES the row
    # at that URL rather than adding a second point beside it.
    tier, anchor = aw_b1_identity.classify_tier(
        "entity_card", None, True, digest=None)
    point_id = aw_b1_identity.mint_point_id(seed["url"], anchor)

    snippet = dense if len(dense) <= 220 else dense[:220].rsplit(" ", 1)[0] + "…"
    rec = {
        "point_id": point_id,
        "url": seed["url"],
        "title": name,
        "snippet": snippet,
        "dense_text": dense,
        "sparse_text": sparse,
        "lang": seed.get("lang") or "de",
        "domain": seed["domain"],
        "tier": tier,
        "source_type": "entity_card",
        "trust_tier": seed.get("trust_tier", 0.85),
        "last_seen": seed["last_seen"],
        "country": seed["country"],
        "license": seed["license"],
        "content_digest": hashlib.sha256(dense.encode("utf-8")).hexdigest(),
        "register_id": seed["register_id"],
        "register_scheme": seed["register_scheme"],
        "provenance": seed["provenance"],
        "entity_kind": "org",
        # ADR-041 §5 — the clause production does not currently satisfy.
        "resolve_confidence": round(float(confidence), 4),
        "resolve_evidence_class": ("uid_on_page" if evidence.get("uid_on_page")
                                   else "name_address"),
        "resolve_uid_on_page": bool(evidence.get("uid_on_page")),
    }
    # Omitted rather than written empty when the admission rests on a UID match and
    # no address token was needed: an absent field says "not applicable", an empty
    # list says "we looked and there were none", and only one of those is true here.
    if evidence.get("address_tokens"):
        rec["resolve_address_tokens"] = list(evidence["address_tokens"])
    if seed.get("register_url"):
        rec["register_url"] = seed["register_url"]
    if seed.get("canton"):
        rec["canton"] = seed["canton"]
    if legal_form_enabled(env):
        if seed.get("legal_form"):
            rec["legal_form"] = seed["legal_form"]
        if seed.get("legal_form_code"):
            rec["legal_form_code"] = str(seed["legal_form_code"])
    return rec


# ---------------------------------------------------------------------- declare
#: Fields every register-first card must carry, unconditionally.
REQUIRED_PAYLOAD_FIELDS = (
    "url", "title", "snippet", "lang", "domain", "tier", "source_type",
    "trust_tier", "last_seen", "country", "license", "content_digest",
    "register_id", "register_scheme", "provenance", "entity_kind",
    "resolve_confidence", "point_id",
)

#: Fields required only once the representation contract declares them.
GATED_PAYLOAD_FIELDS = ("legal_form", "legal_form_code")


def assert_payload_contract(rec: dict, *, env: dict | None = None,
                            stage: str = "register_first_cards") -> dict:
    """Fail closed on a missing OR empty required field. Returns the record.

    Presence is not enough: the July generation carried `legal_form_code` as a
    key on all 6,172 rows of the batch with 0 non-null values, and
    `resolve_confidence` reached production as a key on none of the 337,702 live
    cards. Both failures are invisible to a key-presence check.
    """
    missing, empty = [], []
    for f in REQUIRED_PAYLOAD_FIELDS:
        if f not in rec:
            missing.append(f)
        elif rec[f] in (None, "", [], {}):
            empty.append(f)
    if legal_form_enabled(env):
        for f in GATED_PAYLOAD_FIELDS:
            if f not in rec:
                missing.append(f)
            elif rec[f] in (None, "", [], {}):
                empty.append(f)
    # ANY payload key that is present and empty is refused, not just a required one.
    # An empty key is never informative and it is how a field stops meaning anything
    # while every presence check stays green. Found the hard way: task 621's own pack
    # driver read a consent field the resolver does not emit and wrote `consent_state`
    # as null on 506 of 507 rows, past a check that only looked at its own required
    # list. Leave a field OUT rather than write it empty.
    for f, v in rec.items():
        if v in (None, "", [], {}) and f not in empty:
            empty.append(f)
    if missing or empty:
        raise PayloadContractRefusal(
            f"{stage}: register-first card payload contract violated for "
            f"url={rec.get('url')!r} — missing={missing} empty={empty}. "
            f"ADR-041 §5 requires the resolution score to travel with the card; "
            f"the fields above are what a re-gate, a subtraction and a per-country "
            f"census all need and what production lost in the rebuild "
            f"(task 621 §1.6)."
        )
    if rec["source_type"] != "entity_card":
        raise PayloadContractRefusal(
            f"{stage}: source_type={rec['source_type']!r}, not 'entity_card'. "
            f"876 CH points in production carry a register_id under a non-card "
            f"source_type — cards collapsed into the site row at the same URL "
            f"(task 621 §1.5). This engine never emits one of those."
        )
    if not (0.0 <= float(rec["resolve_confidence"]) <= 1.0):
        raise PayloadContractRefusal(
            f"{stage}: resolve_confidence={rec['resolve_confidence']!r} outside [0,1].")
    return rec
