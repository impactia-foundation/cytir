"""The ONE place per-record consent flags are computed (ADR-115, ruling 17).

`derive_consent_flags(row, ctx)` decides, for one payload, what `search_ok`,
`train_ok`, `consent_checked_at` and `consent_basis` should be, and from which
rule. Nothing else in the repo may compute those four fields for an existing row;
a caller that needs them calls this function (`cytir.consent.backfill` is the first).

Rules, first match wins (each is a clause of ADR-115; ids are the `rule` field):

    E   row already carries both flags           -> never rewritten
    r1  dump-route family, licence is the instrument
        wikipedia / arxiv / pubmed / stack exchange (r2)
    r5  official-dump family whose host verdict is irrelevant (ADR-076 pattern)
    r0  contract `explicit-grant` / `not-applicable-synthesized` with both flags
    r3  the host's CURRENT consent-cache verdict (exact host; `www.` -> apex only, r4)
    r6  (PROPOSED, task 684 / ruling 20, sealed per-host probe) only when the row's host has
        NO consent-cache record of ANY status; the host must appear in the family's sealed
        probe artifact (status ok, both flags boolean, a checked_at, flags EQUAL to the
        contract's, not older than 30 days). The row gets the sealed record's own
        checked_at. A dump-route family takes r1, never r6.
    D   anything else: NO flags, never a guess, the row stays visible as class D

Pure: no network, no clock (the caller passes `as_of`), no Qdrant. Stdlib only, so
the same module runs next to the twin on the build box (Python 3.10).
"""
from __future__ import annotations

import json
import os
import urllib.parse
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field

RULES_VERSION = "adr115.v2"   # v2 adds r6 (task 684, PROPOSED); the v1 rules are unchanged

# ADR-076 / task 433 (mirrors consent_gate.DUMP_CONSENT_BASES; pinned by a test).
DUMP_CONSENT_BASES = frozenset({"official-dump-license"})

# Closed set of `consent_basis` values this module writes.
BASIS_DUMP_LICENCE = "dump-licence"
BASIS_OFFICIAL_DUMP = "official-dump-licence-family"
BASIS_CONTRACT = "contract-grant"
BASIS_HOST_CACHE = "host-cache"
BASIS_HOST_CACHE_WWW = "host-cache-www-apex"
BASIS_CONTRACT_SEALED = "contract-sealed-probe"      # r6 (task 684, ruling 20; PROPOSED)
WRITTEN_BASES = frozenset({BASIS_DUMP_LICENCE, BASIS_OFFICIAL_DUMP, BASIS_CONTRACT,
                           BASIS_HOST_CACHE, BASIS_HOST_CACHE_WWW, BASIS_CONTRACT_SEALED})

#: A sealed probe older than this is no evidence (= `cytir.consent.resolver.MAX_AGE_DAYS`;
#: pinned by the consent-stamp compose test).
SEALED_MAX_AGE_DAYS = 30

MECHANICAL_RESOLUTIONS = frozenset({"explicit-grant", "not-applicable-synthesized"})
DUMP_ROUTES = frozenset({"dump", "bulk-dump"})

# --- family mapping (moved here from the task-678 census; one copy) -----------

# provenance value -> contract family_id. ONLY where the contract's own source_unit
# or id names the same source. (`hn-discussion-firebase` is NOT here: the contract
# `hn_firebase_pointer` is a RETIRED, refused 406-row pack, not the ADR-108
# `hn_discussion` rows; the census alias was a mis-mapping, ADR-115 B.)
ALIASES = {
    "kali_article": "kali_exact",
    "ce_juri": "scratch_ce_juri_exact",
    "legi_code": "named_slice:legifrance-codes",
    "bfs_dam": "bfs_dam_exact",
    "us_fed_cc": "scratch_us_fed_cc_exact",
    "ca_canadaca_cc": "scratch_ca_canadaca_cc_exact",
    "ca_laws_bilingual": "justice_exact",
    "zefix_ccjoin": "scratch_ch_business_exact",
    "zefix_slug_live": "scratch_ch_business_exact",
    "bazl_docs": "scratch_bazl_docs_exact",
    "bav_docs": "scratch_bav_docs_exact",
    "context7-github": "context7_pinned_docs",
    "iris_meta": "iris_exact",
    "scc_a2aj_si975": "scratch_scc_a2aj_exact",
    "rii_courts": "task472_rii_courts_1024_embed",
}

# Rows with NO provenance key map by domain AND a licence prefix, so a domain alone
# never decides. domain -> (family_id, licence prefix).
_SE = ("stackexchange_nonso_qa_p0", "CC-BY-SA")
DOMAIN_HINTS = {
    **{d: _SE for d in (
        "math.stackexchange.com", "superuser.com", "askubuntu.com", "serverfault.com",
        "unix.stackexchange.com", "electronics.stackexchange.com",
        "codereview.stackexchange.com", "stats.stackexchange.com",
        "softwareengineering.stackexchange.com", "dba.stackexchange.com",
        "apple.stackexchange.com", "security.stackexchange.com")},
    "stackoverflow.com": ("stackoverflow_qa", "CC-BY-SA"),
    "en.wikipedia.org": ("wikipedia_413a", "CC-BY-SA"),
    "de.wikipedia.org": ("wikipedia_413a", "CC-BY-SA"),
    "fr.wikipedia.org": ("wikipedia_413a", "CC-BY-SA"),
    "it.wikipedia.org": ("wikipedia_413a", "CC-BY-SA"),
    "arxiv.org": ("named_slice:arxiv", "arXiv-metadata"),
    "openalex.org": ("openalex_academic_dump", "CC0"),
    "pubmed.ncbi.nlm.nih.gov": ("named_slice:pubmed", "NLM-PubMed"),
    "de.openlegaldata.io": ("de_openlegaldata_v1", ""),
    "files.eric.ed.gov": ("eric_official_pdf", ""),
    "swisstopo.admin.ch": ("named_slice:swisstopo-v2", ""),
}

# ADR-115 r1/r2: family -> (rule id, licence prefix the ROW must carry).
INSTRUMENT_FAMILIES = {
    "wikipedia_413a": ("r1-wikipedia", "CC-BY-SA"),
    "named_slice:arxiv": ("r1-arxiv", "arXiv-metadata"),
    "named_slice:pubmed": ("r1-pubmed", "NLM-PubMed"),
    "stackexchange_nonso_qa_p0": ("r2-stackexchange", "CC-BY-SA"),
    "stackoverflow_qa": ("r2-stackexchange", "CC-BY-SA"),
    # CC0-1.0 dedication read from OpenAlex's own licence statement (contract instrument);
    # search and train follow the licence (task 684, PROPOSED with r6, reviewer fix 3).
    "openalex_academic_dump": ("r1-openalex", "CC0"),
    # DILA's fiche archive, published by the source itself under Licence Ouverte / Etalab-2.0:
    # the licence is the consent instrument (POLICY.md@v3.2 U1, Tim 2026-10-05, ruling 26 item 22).
    # The rows carry `Etalab-2.0` (cytir/sources/families/service_public_fiche.py LICENSE).
    "service_public_fiche": ("r1-service-public", "Etalab-2.0"),
}

# ADR-115 r5 (ADR-076 pattern): official dumps whose canonical host's robots says no.
# provenance -> (licence the row must carry, declared basis text).  search_ok and
# train_ok are both true: the licence (Etalab-2.0; UrhG section 5 official work)
# permits unrestricted reuse, and ADR-076 says train_ok follows the licence terms.
OFFICIAL_DUMP_FAMILIES = {
    "jorf_recent": ("Etalab-2.0", "DILA JORF open dump, Licence Ouverte / Etalab-2.0"),
    "kali_article": ("Etalab-2.0", "DILA KALI open dump, Licence Ouverte / Etalab-2.0"),
    "kali_conv": ("Etalab-2.0", "DILA KALI open dump, Licence Ouverte / Etalab-2.0"),
    "ce_juri": ("Etalab-2.0", "DILA CE open-data dump, Licence Ouverte / Etalab-2.0"),
    "cass_juri": ("Etalab-2.0", "DILA CASS open-data dump, Licence Ouverte / Etalab-2.0"),
    "legi_code": ("Etalab-2.0", "DILA LEGI open dump, Licence Ouverte / Etalab-2.0"),
    "loda": ("Etalab-2.0", "DILA LEGI/LODA open dump, Licence Ouverte / Etalab-2.0"),
    "rii_courts": ("public-domain-official-work-UrhG-5",
                   "rechtsprechung-im-internet.de federal-court dump, UrhG section 5 official work"),
}


def family_candidates(provenance: str) -> list:
    t = provenance.replace("-", "_")
    return [t, t + "_exact", "scratch_" + t + "_exact", "scratch_" + t]


def build_family_map(families: list, aliases: dict | None = None,
                     domain_hints: dict | None = None):
    """Return fam_of(provenance, domain, licence) -> (family_id, method).

    Method: alias | provenance-name | domain-hint | none | provenance-unmatched. A
    mapping is an INFERENCE (the payload carries no family key), so the method
    travels with every count."""
    ids = {f["family_id"] for f in families}
    alias = dict(aliases or {})
    hints = dict(domain_hints or {})

    def fam_of(provenance: str, domain: str = "", license_: str = ""):
        if provenance:
            if provenance in alias and alias[provenance] in ids:
                return alias[provenance], "alias"
            for c in family_candidates(provenance):
                if c in ids:
                    return c, "provenance-name"
            return None, "provenance-unmatched"
        h = hints.get(domain)
        if h and h[0] in ids and license_.startswith(h[1]):
            return h[0], "domain-hint"
        return None, "none"
    return fam_of


# --- host + consent cache -------------------------------------------------------

def host_of(url) -> str:
    try:
        return (urllib.parse.urlsplit(str(url)).hostname or "").lower().strip(".")
    except ValueError:
        return ""


def load_cache_index(cache_dir: str) -> dict:
    """{host: {variant: {status, search_ok, train_ok, checked_at}}}; read-only.

    Variant is the file's own suffix ('cc', 'direct', ...). The host comes from the
    FILE NAME (the key `resolver._cache_path` writes). `__unparseable__` counts files
    that could not be read; none is skipped silently."""
    idx: dict = {}
    bad = 0
    for name in os.listdir(cache_dir):
        if not name.endswith(".json") or name.startswith("."):
            continue
        host, _, variant = name[:-5].partition("@")
        try:
            with open(os.path.join(cache_dir, name)) as f:
                r = json.load(f)
        except Exception:  # noqa: BLE001
            bad += 1
            continue
        idx.setdefault(host, {})[variant or "cc"] = {
            "status": r.get("status"), "search_ok": r.get("search_ok"),
            "train_ok": r.get("train_ok"), "checked_at": r.get("checked_at")}
    idx["__unparseable__"] = bad
    return idx


def _usable(rec) -> bool:
    return bool(rec and rec.get("status") == "ok"
                and isinstance(rec.get("search_ok"), bool)
                and isinstance(rec.get("train_ok"), bool)
                and rec.get("checked_at"))


def _host_record(idx: dict, host: str):
    """The host's CURRENT verdict (ADR-115 r3): among the USABLE cache variants, the one with
    the newest `checked_at`. Ties (equal timestamp) break by a fixed order: 'cc', then
    'direct', then the other variant names alphabetically. A fixed variant order alone was
    wrong: github.com cc=2026-09-08 train_ok=True vs direct=2026-09-28 train_ok=False must
    stamp False."""
    variants = idx.get(host) or {}
    order = {"cc": 0, "direct": 1}
    floor = datetime.min.replace(tzinfo=timezone.utc)      # an unparseable stamp ranks oldest
    usable = [(v, rec) for v, rec in variants.items() if _usable(rec)]
    if not usable:
        return None, ""
    v, rec = min(usable, key=lambda p: (-(_parse_utc(p[1]["checked_at"]) or floor).timestamp(),
                                        order.get(p[0], 2), p[0]))
    return rec, v


def cache_lookup(idx: dict, host: str):
    """(record, label). Exact host first. ADR-115 r4: a `www.<host>` record stands for
    the apex `<host>` ONLY when the apex has no usable record of its own. The other
    direction (a `www.` row borrowing the apex's verdict) is NOT approved and is never
    tried. A parent-domain record is never used."""
    host = (host or "").lower().strip(".")
    if not host:
        return None, "no-host"
    rec, v = _host_record(idx, host)
    if rec is not None:
        return rec, f"exact:{v}"
    if not host.startswith("www."):
        rec, v = _host_record(idx, "www." + host)
        if rec is not None:
            return rec, f"www-apex:{v}"
    return None, "miss"


# --- licence classifiers ------------------------------------------------------------

_CC_TRAIN_TRUE = {"by", "by-sa"}
_CC_TRAIN_FALSE = {"by-nc", "by-nc-sa", "by-nc-nd", "by-nd"}


def paper_licence_train_ok(licence) -> tuple:
    """(train_ok, label) for a per-paper licence URL. FAIL CLOSED.

    true : CC-BY, CC-BY-SA (any version), CC0, public-domain-mark and the old
           CC public-domain dedication
    false: CC-BY-NC*, CC-BY-ND*, the arXiv non-exclusive licence, a missing licence,
           and any licence this function does not recognise."""
    if not licence or not isinstance(licence, str):
        return False, "missing"
    u = licence.strip().lower().split("://", 1)[-1].rstrip("/")
    parts = u.split("/")
    if parts[0] == "creativecommons.org" and len(parts) >= 3:
        if parts[1] == "licenses":
            if parts[2] in _CC_TRAIN_TRUE:
                return True, "cc-" + parts[2]
            if parts[2] in _CC_TRAIN_FALSE:
                return False, "cc-" + parts[2]
            if parts[2] == "publicdomain":          # the older CC public-domain dedication
                return True, "public-domain"
        if parts[1] == "publicdomain" and parts[2] in ("zero", "mark"):
            return True, "cc0" if parts[2] == "zero" else "public-domain-mark"
    if parts[0].endswith("arxiv.org") and "nonexclusive-distrib" in parts:
        return False, "arxiv-nonexclusive"
    return False, "unrecognised"


def arxiv_id_of(url) -> str:
    u = str(url)
    marker = "arxiv.org/abs/"
    return u.split(marker, 1)[1].strip("/") if marker in u else ""


def pmid_of(url) -> str:
    u = str(url).rstrip("/")
    return u.rsplit("/", 1)[-1] if "pubmed.ncbi.nlm.nih.gov/" in u else ""


# --- the derivation -----------------------------------------------------------------

@dataclass(frozen=True)
class Derivation:
    klass: str                      # E | A | B | D
    rule: str                       # rule id, or the D reason
    search_ok: bool | None = None
    train_ok: bool | None = None
    consent_checked_at: str | None = None
    consent_basis: str | None = None
    detail: str = ""                # coverage label (licence bucket, cache variant)

    def writes(self) -> dict:
        """Payload fields to set. Empty for E and D: unknown writes NO flags."""
        if self.klass not in ("A", "B"):
            return {}
        return {"search_ok": self.search_ok, "train_ok": self.train_ok,
                "consent_checked_at": self.consent_checked_at,
                "consent_basis": self.consent_basis}


@dataclass
class DeriveContext:
    families: list
    cache_index: dict
    as_of: str                                   # UTC stamp for instrument-derived rows
    arxiv_licences: dict | None = None           # arxiv id -> licence URL
    pmc_licences: dict | None = None             # pmid -> licence URL (none available today)
    sealed_probes: dict | None = None            # family_id -> {host: sealed probe record} (r6)
    fam_of: object = field(init=False)
    by_id: dict = field(init=False)

    def __post_init__(self):
        self.by_id = {f["family_id"]: f for f in self.families}
        self.fam_of = build_family_map(self.families, ALIASES, DOMAIN_HINTS)


def _flag(v):
    return v if isinstance(v, bool) else None


def family_of_row(row: dict, ctx: DeriveContext):
    return ctx.fam_of(str(row.get("provenance") or ""), str(row.get("domain") or ""),
                      str(row.get("license") or ""))[0]


def derive_consent_flags(row: dict, ctx: DeriveContext, *,
                         family_id: str | None = None) -> Derivation:
    """See the module docstring. Never raises on a odd payload; odd -> class D.

    `family_id`: the family a writer DECLARES for the row (a queue pack's manifest
    `family`). It replaces the provenance/domain inference, which a chunk-stage row
    cannot support (it carries no provenance). The backfill passes nothing."""
    s, t = row.get("search_ok"), row.get("train_ok")
    if isinstance(s, bool) and isinstance(t, bool):
        return Derivation("E", "explicit-flags")
    if s is not None or t is not None:
        return Derivation("D", "one-flag-only-never-coerced")

    licence = str(row.get("license") or "")
    prov = str(row.get("provenance") or "")
    fid = family_id or family_of_row(row, ctx)
    fam = ctx.by_id.get(fid) if fid else None
    host = host_of(row.get("url")) or str(row.get("domain") or "").lower()
    state = str(row.get("consent_state") or "")

    # r1 / r2 -- the dump's licence is the consent instrument (ADR-103 am.2, ADR-115).
    if fid in INSTRUMENT_FAMILIES:
        rule, prefix = INSTRUMENT_FAMILIES[fid]
        if not licence.startswith(prefix):
            return Derivation("D", f"D:instrument-licence-absent-or-mismatch ({fid})")
        if rule == "r1-arxiv":
            lic = (ctx.arxiv_licences or {}).get(arxiv_id_of(row.get("url")))
            train, label = paper_licence_train_ok(lic)
            return _instrument(ctx, rule, True, train, label)
        if rule == "r1-pubmed":
            lic = (ctx.pmc_licences or {}).get(pmid_of(row.get("url")))
            train, label = paper_licence_train_ok(lic) if lic else (False, "no-pmc-oa-record")
            return _instrument(ctx, rule, True, train, label)
        return _instrument(ctx, rule, True, True, licence)

    # r5 -- official dump families: the dump basis, never the host verdict.
    if prov in OFFICIAL_DUMP_FAMILIES:
        want, _basis = OFFICIAL_DUMP_FAMILIES[prov]
        if not licence.startswith(want):
            return Derivation("D", f"D:official-dump-licence-absent-or-mismatch ({prov})")
        basis = row.get("consent_basis") if (
            state == "dump" and row.get("consent_basis") in DUMP_CONSENT_BASES) else BASIS_OFFICIAL_DUMP
        return Derivation("A", "r5-official-dump", True, True, ctx.as_of, basis, prov)

    # r0 -- a declared, non-retired contract grant with both flags.
    if fam:
        cons = fam.get("consent") or {}
        if (fam.get("status") != "retired"
                and cons.get("resolution") in MECHANICAL_RESOLUTIONS
                and _flag(cons.get("search_ok")) is not None
                and _flag(cons.get("train_ok")) is not None):
            basis = row.get("consent_basis") if (
                state == "dump" and row.get("consent_basis") in DUMP_CONSENT_BASES) else BASIS_CONTRACT
            return Derivation("A", "r0-contract-grant", cons["search_ok"], cons["train_ok"],
                              ctx.as_of, basis, fid)
        # A dump-route family whose contract declares no instrument: the host's robots
        # is the wrong evidence there (ADR-103 am.2). A ruling decides, not a probe.
        if fam.get("route") in DUMP_ROUTES and cons.get("resolution") != "live-per-origin":
            return Derivation("D", f"D:dump-route-instrument-undeclared ({fid})")

    # r3 / r4 -- the host's CURRENT consent-cache verdict, with its own checked_at.
    rec, label = cache_lookup(ctx.cache_index, host)
    if rec is None:
        if _any_cache_record(ctx.cache_index, host):
            # A record exists but is not usable (unknown / error / held): that is OUR
            # instrument failing or the publisher unreadable, never a licence to fall back.
            return Derivation("D", f"D:held-cache-record-not-usable [{label}]")
        sealed, why = _sealed_probe(ctx, fid, fam, host)
        if sealed is not None:
            return Derivation("A", "r6-contract-sealed-probe", sealed["search_ok"],
                              sealed["train_ok"], str(sealed["checked_at"]),
                              BASIS_CONTRACT_SEALED, f"{fid}@{host}")
        return Derivation("D", f"D:no-usable-cache-record [{label}; r6: {why}]")
    carried = state == "carried"
    if not rec["search_ok"] and not carried:
        # A served, non-carried row whose host now declines: not ruled. Stays D.
        return Derivation("D", f"D:cache-search-false-on-served-row [{label}]")
    basis = BASIS_HOST_CACHE_WWW if label.startswith("www-apex") else BASIS_HOST_CACHE
    return Derivation("B", "r3-host-cache" + ("-carried" if carried else ""),
                      rec["search_ok"], rec["train_ok"], str(rec["checked_at"]), basis, label)


def _any_cache_record(idx: dict, host: str) -> bool:
    """True when the cache holds ANY record (any status, any variant) for the host, or for
    `www.<host>` standing for an apex row (the one approved direction)."""
    hosts = [host] + ([] if host.startswith("www.") else ["www." + host])
    return any(isinstance(idx.get(h), dict) and idx[h] for h in hosts)


def _parse_utc(v):
    try:
        return datetime.strptime(str(v), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def sealed_family_refusal(fam) -> str:
    """"" when the family's contract makes it an r6 (sealed probe) input, else the reason."""
    if not fam or fam.get("status") != "declared":
        return "family-not-declared"
    cons = fam.get("consent") or {}
    if not (cons.get("resolution") == "live-per-origin" and cons.get("search_ok") is True
            and isinstance(cons.get("train_ok"), bool) and cons.get("evidence")):
        return "contract-not-a-live-per-origin-verdict"
    if fam.get("route") in DUMP_ROUTES:
        return "dump-route-takes-r1"
    return ""


def _sealed_probe(ctx: DeriveContext, fid, fam, host: str):
    """(sealed record, "") or (None, reason). All of: a declared family (not backfilled or
    retired) whose contract is `live-per-origin` with search_ok true, train_ok boolean and
    evidence; NOT a dump-route family (it takes r1); `host` present in the family's loaded
    sealed probe artifact with status ok, both flags boolean, a checked_at, flags EQUAL to
    the contract's, and age <= 30 days at `ctx.as_of`. A pointer string proves nothing:
    only a loaded record counts."""
    why = sealed_family_refusal(fam)
    if why:
        return None, why
    cons = fam["consent"]
    rec = ((ctx.sealed_probes or {}).get(fid) or {}).get(host)
    if rec is None:
        return None, "host-not-in-sealed-artifact"
    if not (rec.get("status") == "ok" and isinstance(rec.get("search_ok"), bool)
            and isinstance(rec.get("train_ok"), bool) and rec.get("checked_at")):
        return None, "sealed-record-unusable"
    if rec["search_ok"] != cons["search_ok"] or rec["train_ok"] != cons["train_ok"]:
        return None, "sealed-flags-differ-from-contract"
    seen, now = _parse_utc(rec["checked_at"]), _parse_utc(ctx.as_of)
    if seen is None or now is None:
        return None, "sealed-checked_at-unparseable"
    if now - seen > timedelta(days=SEALED_MAX_AGE_DAYS) or seen > now:
        return None, f"sealed-record-older-than-{SEALED_MAX_AGE_DAYS}d"
    return rec, ""


_EVIDENCE_POINTER = "#host_consent_resolution"


class SealedEvidenceRefused(RuntimeError):  # alias SealedProbeRefused below
    """`strict=True`: a contract names a sealed probe artifact that does not load, or that lacks
    the section the pointer names. A run without it would class every row of the family D."""


SealedProbeRefused = SealedEvidenceRefused


def load_sealed_probes(families: list, repo_root: str, *, strict: bool | None = None) -> dict:
    """{family_id: {host: record}} from the sealed artifacts the contracts' `consent.evidence`
    name (`<path>#host_consent_resolution`). A host belongs to a family only if the
    artifact's own `family_verdicts[family].hosts` lists it. An artifact that does not load,
    or lacks either section, contributes nothing (never an error read as an empty verdict:
    the row then falls to D and compose refuses).

    `strict=None` (the default, task 738) means strict wherever the tree can hold sealed
    evidence, i.e. `<repo_root>/tasks` exists: compose must never read a missing or truncated
    probe file as an empty verdict. The public cut ships no `tasks/`; there the pointer cannot
    be checked and the loader stays lenient. Pass True/False to force either.
    `strict=True` (the rebuild, task 720; task 713 section 6 finding: a first dry run without
    the ERIC file put 2,375,444 rows in D, silently) RAISES `SealedEvidenceRefused` instead,
    naming every pointer that did not load and every family the artifact does not list. A
    retired family's pointer is not read (the family takes no rule)."""
    if strict is None:
        strict = os.path.isdir(os.path.join(repo_root, "tasks"))
    out: dict = {}
    cache: dict = {}
    problems: list = []
    for f in families:
        if strict and f.get("status") == "retired":
            continue
        cons = f.get("consent") or {}
        for ev in cons.get("evidence") or []:
            if not (isinstance(ev, str) and ev.endswith(_EVIDENCE_POINTER)):
                continue
            f["family_id"]  # precondition: a family carrying evidence has an id (KeyError otherwise)
            path = ev[:-len(_EVIDENCE_POINTER)]
            if path not in cache:
                try:
                    with open(os.path.join(repo_root, path)) as fh:
                        cache[path] = json.load(fh)
                except (OSError, ValueError) as exc:
                    cache[path] = None
                    problems.append(f"{f['family_id']}: {path} does not load ({type(exc).__name__}: {exc})")
            art = cache[path]
            if not isinstance(art, dict):
                if art is not None:
                    problems.append(f"{f['family_id']}: {path} is not a JSON object")
                continue
            hosts = ((art.get("family_verdicts") or {}).get(f["family_id"]) or {}).get("hosts")
            records = art.get("host_consent_resolution")
            if not isinstance(hosts, list) or not hosts or not isinstance(records, dict):
                problems.append(f"{f['family_id']}: {path} has no family_verdicts[{f['family_id']}].hosts "
                                "list or no host_consent_resolution object")
                continue
            for h in hosts:
                if isinstance(records.get(h), dict):
                    out.setdefault(f["family_id"], {})[h] = records[h]
                else:
                    problems.append(f"{f['family_id']}: {path} lists host {h} with no record")
    if strict and problems:
        raise SealedEvidenceRefused(
            "REFUSED: %d sealed consent evidence problem(s); a run without them would class the "
            "family's rows D without a word: %s" % (len(problems), "; ".join(sorted(set(problems)))))
    return out


def sealed_probe_expiries(sealed_probes: dict, as_of: str) -> list:
    """Every loaded sealed record that would NOT pass `_sealed_probe`'s age bound at `as_of`:
    `[(family_id, host, checked_at, reason)]`, plus the earliest deadline in the second slot of
    the returned tuple. The deadline is the latest `as_of` at which every record still passes
    (task 713: the ERIC r6 record of 2026-09-22T06:59:38Z expires 2026-10-22T06:59Z)."""
    now = _parse_utc(as_of)
    bad, deadline = [], None
    for fid, hosts in sorted((sealed_probes or {}).items()):
        for host, rec in sorted(hosts.items()):
            seen = _parse_utc((rec or {}).get("checked_at"))
            if seen is None or now is None:
                bad.append((fid, host, (rec or {}).get("checked_at"), "checked_at-unparseable"))
                continue
            end = seen + timedelta(days=SEALED_MAX_AGE_DAYS)
            deadline = end if deadline is None or end < deadline else deadline
            if now - seen > timedelta(days=SEALED_MAX_AGE_DAYS):
                bad.append((fid, host, rec["checked_at"], f"older-than-{SEALED_MAX_AGE_DAYS}d"))
            elif seen > now:
                bad.append((fid, host, rec["checked_at"], "newer-than-as_of"))
    return bad, (deadline.strftime("%Y-%m-%dT%H:%M:%SZ") if deadline else None)


def _instrument(ctx: DeriveContext, rule: str, search: bool, train: bool, label: str) -> Derivation:
    return Derivation("A", rule, search, train, ctx.as_of, BASIS_DUMP_LICENCE, label)
