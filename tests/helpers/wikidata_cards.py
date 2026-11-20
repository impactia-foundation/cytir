"""Wikidata ENTITY-CARD construction: the `cards` step of the country generator (task 155, ported task 1517).

One card per notable ENTITY (company/org or person) that has an official website in
Wikidata, pointing an agent at the entity's authoritative home. Keyed on notability and
an official website (P856), not on consent. The embedding text is identity-centric
(name / aliases / type), not content, so a card wins "who is X / official site of X"
and does not crowd substantive content queries.

`cmd_cards(args)` reads an `enriched.json` (country, langs, entities) and writes
`cards.jsonl`; the text recipe is `cytir.corpus.card_recipe`. No prod or qdrant writes here.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from urllib.parse import urlparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cytir.corpus import card_recipe  # noqa: E402

TODAY = time.strftime("%Y-%m-%d", time.gmtime())


# Country profiles — the generator rolls CH -> FR/DE/CA. `primary` label langs are
# tried in order for the card title; `langs` is the multilingual recall set.
# `register` = the country company-register identifier Wikidata carries:
#   CH Swiss-UID P4829 · FR SIREN P1616 (+SIRET P3215 folded) · DE Registernummer
#   Deutschland P12012 (Handelsregister) · CA Canada Business Number P8860.
# `name_en` = country display name folded into the card text.
COUNTRIES = {
    "CH": {"qid": "Q39", "tld": ".ch", "name_en": "Switzerland",
           "langs": ["de", "fr", "it", "en"], "primary": ["de", "fr", "it", "en"],
           "register": {"var": "uid", "prop": "P4829", "scheme": "CHE-UID"}},
    "FR": {"qid": "Q142", "tld": ".fr", "name_en": "France",
           "langs": ["fr", "en", "de"], "primary": ["fr", "en"],
           "register": {"var": "siren", "prop": "P1616", "scheme": "SIREN"}},
    "DE": {"qid": "Q183", "tld": ".de", "name_en": "Germany",
           "langs": ["de", "en", "fr"], "primary": ["de", "en"],
           "register": {"var": "dereg", "prop": "P12012", "scheme": "DE-Registernummer"}},
    "CA": {"qid": "Q16", "tld": ".ca", "name_en": "Canada",
           "langs": ["en", "fr"], "primary": ["en", "fr"],
           "register": {"var": "cabn", "prop": "P8860", "scheme": "CA-BusinessNumber"}},
}


def pick_canonical(sites: list[str], tld: str) -> str:
    """Prefer https + root/base domain + country TLD, else first."""
    def score(u: str) -> tuple:
        p = urlparse(u)
        host = (p.netloc or "").lower()
        path = (p.path or "/").rstrip("/")
        depth = len([s for s in path.split("/") if s])
        return (
            0 if p.scheme == "https" else 1,       # https first
            0 if host.endswith(tld) else 1,        # country TLD first
            depth,                                  # shallower (root) first
            len(u),                                 # shorter first
        )
    return sorted(sites, key=score)[0]


def label_in(d: dict, langs: list[str], prefix="label_") -> tuple[str, str]:
    for l in langs:
        v = d.get(prefix + l)
        if v:
            return v, l
    return "", ""


def class_label(labels: dict, qids: list[str], primary: list[str]) -> str:
    for q in qids:
        ld = labels.get(q, {})
        for l in primary:
            if ld.get(l):
                return ld[l]
        if ld:
            return next(iter(ld.values()))
    return ""


def class_terms(labels: dict, qids: list[str], langs: list[str]) -> list[str]:
    """All available language labels for the entity's own class nodes."""
    out = []
    for qid in qids:
        row = labels.get(qid, {})
        for lang in langs:
            value = row.get(lang)
            if value and value not in out:
                out.append(value)
    return out


def class_primary_labels(
        labels: dict, qids: list[str], primary: list[str]) -> list[str]:
    """One readable label per class node for the natural-language channel."""
    out = []
    for qid in qids:
        value = class_label(labels, [qid], primary)
        if value and value not in out:
            out.append(value)
    return out


_CHE_DIGITS = None


def normalize_che(raw: str) -> str:
    """Normalize a Swiss UID to canonical CHE-###.###.###. Wikidata stores it
    already-formatted, but be defensive (strip MWST/HR suffixes, spaces)."""
    import re as _re
    if not raw:
        return ""
    s = raw.upper().replace(" ", "")
    m = _re.search(r"CHE-?(\d{3})\.?(\d{3})\.?(\d{3})", s)
    if not m:
        return ""
    return f"CHE-{m.group(1)}.{m.group(2)}.{m.group(3)}"


def normalize_register(cc: str, raw: str) -> str:
    """Normalize the country register id to a canonical string, else "".
      CH  CHE-###.###.### (via normalize_che)
      FR  SIREN = 9 digits (strip spaces); a 14-digit SIRET is kept whole
      DE  Registernummer (Handelsregister), e.g. 'HRB 12345' -> collapse spaces
      CA  Canada Business Number, 9-digit root + optional program suffix"""
    import re as _re
    if not raw:
        return ""
    s = raw.strip()
    if cc == "CH":
        return normalize_che(s)
    if cc == "FR":
        digits = _re.sub(r"\D", "", s)
        return digits if len(digits) in (9, 14) else (digits or "")
    if cc == "DE":
        return _re.sub(r"\s+", " ", s.upper()).strip()
    if cc == "CA":
        return _re.sub(r"\s+", "", s).upper()
    return s


def all_names(d: dict, langs: list[str]) -> list[str]:
    names: list[str] = []
    for l in langs:
        v = d.get("label_" + l)
        if v and v not in names:
            names.append(v)
    for l in langs:
        for v in d.get("alias_" + l, []):
            if v and v not in names:
                names.append(v)
    return names


def cmd_cards(args):
    data = json.loads(pathlib.Path(args.enriched).read_text())
    cc = data["country"]
    langs = data["langs"]
    primary = COUNTRIES[cc]["primary"]
    tld = COUNTRIES[cc]["tld"]
    name_en = COUNTRIES[cc]["name_en"]
    reg_cfg = COUNTRIES[cc]["register"]
    labels = data["class_labels"]
    ents = data["entities"]

    # cross-run / cross-country dedup: skip any qid that already has a card
    # (existing prod cards + qids minted earlier in this multi-country run).
    existing: set[str] = set()
    if getattr(args, "existing_qids", None):
        p = pathlib.Path(args.existing_qids)
        if p.exists():
            existing = {l.strip() for l in p.read_text().splitlines() if l.strip()}
    print(f"# cards {cc}: {len(existing)} qids already carded (will skip)", flush=True)

    by_pid: dict[str, dict] = {}
    n_nolabel = n_nosite = n_existing = 0
    url_collisions = 0
    for qid, d in ents.items():
        if qid in existing:
            n_existing += 1
            continue
        sites = d.get("sites", [])
        if not sites:
            n_nosite += 1
            continue
        title, plang = label_in(d, primary)
        if not title:
            title, plang = label_in(d, langs)
        if not title:
            n_nolabel += 1
            continue
        canonical = pick_canonical(sites, tld)

        kind = d["kind"]
        if kind == "org":
            typ = class_label(labels, d.get("type", []), primary)
        else:
            typ = class_label(labels, d.get("occupation", []), primary)
        industry = class_label(labels, d.get("industry", []), primary) if kind == "org" else ""

        desc, _ = label_in(d, primary, prefix="desc_")
        if not desc:
            desc, _ = label_in(d, langs, prefix="desc_")

        names = all_names(d, langs)
        aliases = [n for n in names if n != title][:12]

        # register identifiers (orgs): country register id (CH UID / FR SIREN /
        # DE Registernummer / CA Business Number) -> register_id + register_scheme;
        # LEI (P1278) for all; FR SIRET + CH VAT form folded into `names`/text.
        register_id = register_scheme = ""
        vat_form = siret = ""
        lei = ""
        if kind == "org":
            register_id = normalize_register(cc, d.get(reg_cfg["var"], ""))
            register_scheme = reg_cfg["scheme"] if register_id else ""
            lei = (d.get("lei", "") or "").strip().upper()
            if cc == "CH" and register_id:
                vat_form = f"{register_id} MWST"
            if cc == "FR":
                siret = normalize_register("FR", d.get("siret", ""))
        idents = [x for x in (register_id, vat_form, siret, lei) if x]

        type_part = typ or ("company" if kind == "org" else "person")
        if industry and industry.lower() != type_part.lower():
            type_part = f"{type_part}, {industry}"

        # snippet <=25 words, identity-centric
        snip = f"{title} — {type_part}"
        if desc:
            snip += f", {desc}"
        snip += ". Official website."
        snip = " ".join(snip.split()[:25])

        host = (urlparse(canonical).netloc or "").lower()
        dom = host[4:] if host.startswith("www.") else host
        facet_qids = (
            list(d.get("type", []))
            + list(d.get("industry", []))
            + list(d.get("product", []))
            + list(d.get("occupation", [])))
        facet_terms = class_terms(labels, facet_qids, langs)
        dense_class_qids = list(d.get("type", []))
        dense_class_qids += (
            list(d.get("industry", [])) if kind == "org"
            else list(d.get("occupation", [])))
        dense_classes = class_primary_labels(
            labels, dense_class_qids, primary)
        dense_products = class_primary_labels(
            labels, list(d.get("product", [])), primary)
        dense_text = card_recipe.dense_prose(
            title, "", desc or type_part, industry=dense_classes[:4],
            products=dense_products[:4], geo=name_en, domain=dom)
        if idents:
            dense_text += " Identifiers: " + ", ".join(idents) + "."
        sparse_text = card_recipe.sparse_hooks(
            title, dom, aliases=aliases + idents,
            facet_terms=facet_terms, commercial=False)

        rec = {
            "point_id": None,
            "url": canonical,
            "title": title[:300],
            "snippet": snip,
            "names": " | ".join(names + idents),
            "register_id": register_id or None,
            "register_scheme": register_scheme or None,
            "lei": lei or None,
            # `text` remains for older inspection/export consumers. The
            # canonical embedder consumes the explicit two channels below.
            "text": dense_text[:1400],
            "dense_text": dense_text[:1400],
            "sparse_text": sparse_text,
            "lang": plang or "mul",
            "domain": dom,
            "anchor": None,
            "tier": "A",
            "source_type": "entity_card",
            "entity_kind": kind,
            "provenance": f"wikidata_{cc.lower()}",
            "country": cc,
            "qid": qid,
            "trust_tier": 0.9,
            "last_seen": TODAY,
            "all_sites": sites,
        }
        rec = card_recipe.finish(
            rec, None, phase="wikidata_country_head")
        pid = rec["point_id"]
        if pid in by_pid:
            url_collisions += 1
            # keep the one with the richer name set (proxy for notability)
            if len(names) <= len(by_pid[pid]["names"].split(" | ")):
                continue
        by_pid[pid] = rec

    recs = list(by_pid.values())
    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    korg = sum(1 for r in recs if r["entity_kind"] == "org")
    kper = sum(1 for r in recs if r["entity_kind"] == "person")
    n_reg = sum(1 for r in recs if r.get("register_id"))
    n_lei = sum(1 for r in recs if r.get("lei"))
    print(f"cards: {len(recs)} (org {korg} / person {kper}); "
          f"url-collisions collapsed {url_collisions}; "
          f"skipped no-label {n_nolabel}, no-site {n_nosite}, "
          f"already-carded(qid overlap) {n_existing}", flush=True)
    print(f"  register_id ({reg_cfg['scheme']}): {n_reg}/{korg} orgs "
          f"({100*n_reg/max(korg,1):.1f}%); LEI: {n_lei}", flush=True)
    # multilingual coverage
    for l in langs:
        n = sum(1 for r in recs if any(x for x in ents.get(r["qid"], {})
                                       if x == "label_" + l))
        print(f"  label_{l} present on {n} cards", flush=True)
    print(f"  -> {args.out}", flush=True)

