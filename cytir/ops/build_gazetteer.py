"""ADR-062 Phase-2b (task 281) — build the query-side sub-national name→country
GAZETTEER (`cytir/serve/geo_gazetteer.json`).

Purpose (amended Q2, spec §7 + §1 knob-13): a curated, prominence-ranked,
MULTILINGUAL sub-national gazetteer (provinces / states / cantons / Länder /
regions + major cities of the indexed countries) that serves THREE roles:

  1. the COLD-START prior for the learned `country_scope` head (its labels),
  2. a cheap deterministic INPUT FEATURE / fallback at serve time (Q5),
  3. NOT the sole driver — the learned head generalises beyond it.

ABSTAIN on homographs (the spec's hard rule — "a wrong scope is worse than no
scope", ADR-054): country-vs-subnational collisions ("georgia" = sovereign country
vs US state) and balanced minor-city collisions ("paris" = FR vs Paris TX;
"nice" = FR vs the English word) get an abstain entry → NEVER rolled up, and they
act as a HARD abstain override on the head at serve time.

PROMINENCE ranking resolves the NON-balanced homographs in favour of the dominant
sense: "ontario" → CA (the province vastly outweighs Ontario, California in
grounding queries), "london" → GB, "rome"/"roma" → IT. Documented per entry.

The gazetteer deliberately does NOT re-list anything `_query_jurisdiction` already
detects (the 26 Swiss cantons + the CH/FR/DE/CA/US country markers): the keyword
detector OWNS those and takes precedence at serve time, so the gazetteer only ADDS
new sub-national coverage. Sub-national CODES stay deferred (ADR-045 FR/BE
two-letter collision); every entry rolls up to a COUNTRY-level scope only.

Run:  cyt ops build-gazetteer [--out FILE]
      → writes cytir/serve/geo_gazetteer.json
"""
from __future__ import annotations

import json
import os

from cytir.core.serve_layout import SERVER_REL

# --------------------------------------------------------------------------- #
# ROLLUP ENTRIES.  (names, country, kind, prominence, note)
#   names       — multilingual surface forms, lowercased, word-boundary matched.
#   prominence  — 1..5 (5 = unmistakable at country scope); used for match order
#                 (higher first) and as the head's cold-start label weight.
#   Only NON-ambiguous names live here.  Homographs go in ABSTAIN below.
# --------------------------------------------------------------------------- #
ROLLUP = [
    # ---- CANADA (provinces + territories + major cities) — the Ontario case ----
    (["ontario"], "CA", "province", 5,
     "province >> Ontario, California by grounding prominence"),
    (["quebec", "québec"], "CA", "province", 5, ""),
    (["british columbia", "colombie-britannique"], "CA", "province", 5, ""),
    (["alberta"], "CA", "province", 5, ""),
    (["manitoba"], "CA", "province", 5, ""),
    (["saskatchewan"], "CA", "province", 5, ""),
    (["nova scotia", "nouvelle-écosse", "nouvelle-ecosse"], "CA", "province", 5, ""),
    (["new brunswick", "nouveau-brunswick"], "CA", "province", 5, ""),
    (["newfoundland and labrador", "newfoundland", "terre-neuve"], "CA", "province", 4, ""),
    (["prince edward island", "île-du-prince-édouard"], "CA", "province", 4, ""),
    (["northwest territories"], "CA", "territory", 4, ""),
    (["nunavut"], "CA", "territory", 4, ""),
    (["yukon"], "CA", "territory", 4, ""),
    (["toronto"], "CA", "city", 4, ""),
    (["montreal", "montréal"], "CA", "city", 4, ""),
    (["vancouver"], "CA", "city", 4, ""),
    (["calgary"], "CA", "city", 4, ""),
    (["edmonton"], "CA", "city", 4, ""),
    (["ottawa"], "CA", "city", 4, "national capital"),
    (["winnipeg"], "CA", "city", 4, ""),
    (["mississauga"], "CA", "city", 3, ""),
    # ---- UNITED STATES (states + major cities) ----
    (["california"], "US", "state", 5, ""),
    (["texas"], "US", "state", 5, ""),
    (["florida"], "US", "state", 5, ""),
    (["new york state", "new york"], "US", "state", 4,
     "NY state and NYC are both US; disambiguation to US is safe"),
    (["pennsylvania"], "US", "state", 5, ""),
    (["illinois"], "US", "state", 5, ""),
    (["ohio"], "US", "state", 5, ""),
    (["massachusetts"], "US", "state", 5, ""),
    (["michigan"], "US", "state", 5, ""),
    (["north carolina"], "US", "state", 5, ""),
    (["new jersey"], "US", "state", 5, ""),
    (["arizona"], "US", "state", 5, ""),
    (["colorado"], "US", "state", 5, ""),
    (["los angeles"], "US", "city", 4, ""),
    (["san francisco"], "US", "city", 4, ""),
    (["chicago"], "US", "city", 4, ""),
    (["houston"], "US", "city", 4, ""),
    (["seattle"], "US", "city", 4, ""),
    (["miami"], "US", "city", 4, ""),
    (["philadelphia"], "US", "city", 4, ""),
    # ---- GERMANY (Länder + major cities) ----
    (["bayern", "bavaria", "bavière", "baviera"], "DE", "land", 5, ""),
    (["baden-württemberg", "baden-wurttemberg", "baden-wuerttemberg"], "DE", "land", 5, ""),
    (["nordrhein-westfalen", "north rhine-westphalia", "nrw"], "DE", "land", 5, ""),
    (["hessen", "hesse"], "DE", "land", 4, ""),
    (["sachsen", "saxony", "saxe"], "DE", "land", 4, ""),
    (["niedersachsen", "lower saxony"], "DE", "land", 4, ""),
    (["rheinland-pfalz", "rhineland-palatinate"], "DE", "land", 4, ""),
    (["thüringen", "thueringen", "thuringia"], "DE", "land", 4, ""),
    (["brandenburg"], "DE", "land", 4, ""),
    (["schleswig-holstein"], "DE", "land", 4, ""),
    (["mecklenburg-vorpommern"], "DE", "land", 4, ""),
    (["berlin"], "DE", "city", 4, "national capital"),
    (["münchen", "muenchen", "munich"], "DE", "city", 4, ""),
    (["hamburg"], "DE", "city", 4, ""),
    (["köln", "koeln", "cologne", "colonia"], "DE", "city", 4, ""),
    (["frankfurt"], "DE", "city", 3, ""),
    (["stuttgart"], "DE", "city", 4, ""),
    (["düsseldorf", "duesseldorf", "dusseldorf"], "DE", "city", 4, ""),
    (["dortmund"], "DE", "city", 3, ""),
    (["leipzig"], "DE", "city", 4, ""),
    (["dresden"], "DE", "city", 4, ""),
    # ---- FRANCE (regions + major cities) ----
    (["île-de-france", "ile-de-france"], "FR", "region", 5, ""),
    (["provence-alpes-côte d'azur", "provence", "provence-alpes-cote d'azur"], "FR", "region", 4, ""),
    (["auvergne-rhône-alpes", "auvergne-rhone-alpes"], "FR", "region", 4, ""),
    (["nouvelle-aquitaine"], "FR", "region", 4, ""),
    (["occitanie"], "FR", "region", 4, ""),
    (["bretagne", "brittany"], "FR", "region", 4, ""),
    (["normandie", "normandy"], "FR", "region", 4, ""),
    (["grand est"], "FR", "region", 4, ""),
    (["hauts-de-france"], "FR", "region", 4, ""),
    (["pays de la loire"], "FR", "region", 4, ""),
    (["bourgogne-franche-comté", "bourgogne-franche-comte"], "FR", "region", 4, ""),
    (["centre-val de loire"], "FR", "region", 4, ""),
    (["corse", "corsica"], "FR", "region", 4, ""),
    (["lyon"], "FR", "city", 4, ""),
    (["marseille", "marsiglia"], "FR", "city", 4, ""),
    (["toulouse"], "FR", "city", 4, ""),
    (["nantes"], "FR", "city", 4, ""),
    (["strasbourg", "strassburg"], "FR", "city", 4, ""),
    (["bordeaux"], "FR", "city", 4, ""),
    (["lille"], "FR", "city", 4, ""),
    (["rennes"], "FR", "city", 3, ""),
    (["montpellier"], "FR", "city", 4, ""),
    # ---- ITALY (regions + major cities) ----
    (["lombardia", "lombardy", "lombardie"], "IT", "region", 5, ""),
    (["lazio", "latium"], "IT", "region", 4, ""),
    (["campania"], "IT", "region", 4, ""),
    (["sicilia", "sicily", "sicile", "sizilien"], "IT", "region", 4, ""),
    (["veneto"], "IT", "region", 4, ""),
    (["piemonte", "piedmont", "piémont"], "IT", "region", 4, ""),
    (["toscana", "tuscany", "toscane", "toskana"], "IT", "region", 4, ""),
    (["emilia-romagna"], "IT", "region", 4, ""),
    (["puglia", "apulia"], "IT", "region", 4, ""),
    (["calabria"], "IT", "region", 4, ""),
    (["roma", "rome", "rom"], "IT", "city", 4,
     "Rome, Italy >> Rome, NY/GA by prominence"),
    (["milano", "milan", "mailand"], "IT", "city", 4, ""),
    (["napoli", "naples", "neapel"], "IT", "city", 4, ""),
    (["torino", "turin"], "IT", "city", 4, ""),
    (["palermo"], "IT", "city", 4, ""),
    (["firenze", "florence", "florenz"], "IT", "city", 4, ""),
    (["venezia", "venice", "venise", "venedig"], "IT", "city", 4, ""),
    (["bologna"], "IT", "city", 4, ""),
    # ---- UNITED KINGDOM (nations + major cities) ----
    (["scotland", "écosse", "ecosse", "schottland", "scozia"], "GB", "nation", 5, ""),
    (["wales", "pays de galles", "galles"], "GB", "nation", 5, ""),
    (["england", "angleterre", "inghilterra"], "GB", "nation", 4, ""),
    (["northern ireland", "irlande du nord"], "GB", "nation", 4, ""),
    (["london", "londres", "londra"], "GB", "city", 4,
     "London, UK >> London, Ontario/KY by prominence"),
    (["manchester"], "GB", "city", 4, ""),
    (["glasgow"], "GB", "city", 4, ""),
    (["edinburgh", "édimbourg", "edimburgo"], "GB", "city", 4, ""),
    (["liverpool"], "GB", "city", 4, ""),
    (["leeds"], "GB", "city", 3, ""),
    (["bristol"], "GB", "city", 3, ""),
]

# --------------------------------------------------------------------------- #
# ABSTAIN (ambiguous → NEVER roll up; HARD abstain override on the head too).
#   Two flavours the spec calls out:
#     * country-vs-subnational  ("georgia" sovereign country vs US state) — the
#       most dangerous conflation (ADR-054), always abstain.
#     * balanced minor-city / common-word collisions where no sense dominates.
#   NB: names `_query_jurisdiction` already owns (e.g. "geneva" → GE → CH) are NOT
#   listed — the keyword detector wins there, so the gazetteer never reaches them.
# --------------------------------------------------------------------------- #
ABSTAIN = [
    ("georgia", "sovereign country (GE) vs US state — country-vs-subnational, never scope"),
    ("washington", "US state vs DC vs surname — no dominant single scope"),
    ("paris", "Paris FR vs Paris TX/… + city-scope ambiguity (spec example)"),
    ("nice", "Nice FR vs the English word 'nice' — common-word collision"),
    ("york", "York GB vs New York US — ambiguous without qualifier"),
    ("cambridge", "Cambridge GB vs Cambridge MA — balanced"),
    ("birmingham", "Birmingham GB vs Birmingham AL — roughly balanced"),
    ("hamilton", "Hamilton CA (ON) vs Hamilton NZ vs Hamilton surname — balanced"),
]
_ABSTAIN_NAMES = {a for a, _ in ABSTAIN}
# de-conflict: a single-form rollup name that is also abstained → ABSTAIN wins.
ROLLUP = [e for e in ROLLUP if not (len(e[0]) == 1 and e[0][0] in _ABSTAIN_NAMES)]


def main(argv=None, prog=None):
    import argparse
    ap = argparse.ArgumentParser(prog=prog, description=__doc__)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                                  *SERVER_REL.split("/"), "geo_gazetteer.json"))
    args = ap.parse_args(argv)
    entries = []
    for names, country, kind, prom, note in ROLLUP:
        entries.append({
            "names": names, "country": country, "kind": kind,
            "prominence": prom, **({"note": note} if note else {}),
        })
    doc = {
        "_comment": (
            "ADR-062 Phase-2b (task 281) query-side sub-national name→country "
            "gazetteer. Cold-start prior + input feature + fallback for the learned "
            "country_scope head (spec §7 Q2 AMENDED / §1 knob-13). Prominence-ranked, "
            "multilingual, COUNTRY-level rollup only (sub-national codes deferred, "
            "ADR-045). ABSTAIN on homographs (country-vs-subnational + balanced "
            "collisions) — never scoped, and a hard abstain override on the head. Does "
            "NOT duplicate _query_jurisdiction (cantons + country markers), which owns "
            "its names and takes precedence at serve time. Generated by "
            "the gazetteer builder script — edit there, regenerate."),
        "adr": "ADR-062", "phase": "2b-scope",
        "n_rollup": len(entries),
        "n_abstain": len(ABSTAIN),
        "entries": entries,
        "abstain": [{"name": a, "reason": r} for a, r in ABSTAIN],
    }
    out = args.out
    with open(out, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    from cytir.ops._provenance import write_provenance
    write_provenance(out, "build-gazetteer", {})
    countries = {}
    for e in entries:
        countries[e["country"]] = countries.get(e["country"], 0) + 1
    print(f"wrote {out}")
    print(f"  rollup entries={len(entries)}  abstain={len(ABSTAIN)}")
    print("  per country:", dict(sorted(countries.items())))


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main() or 0)
