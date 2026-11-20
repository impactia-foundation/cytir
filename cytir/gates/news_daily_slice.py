"""Freeze the ADR-043 am.4 daily-gate slice from the gold sets themselves.

The slice is a SUBSET SELECTOR over the immutable gold sets, never a new gold
set: every query in it is already one of the 1,418 the battery scores, under
the same id, with the same gold urls.  It is written to
`data/news-daily-gate-slice.v1.json` by this script and changed only by a PR,
so a run cannot choose the queries it is judged on.

The rule, and why it is not smaller:

  * the WHOLE 663-gold core -- `authoritative`, `foundation`, `legal`,
    `names` -- because that is the population ADR-043's net-non-negative
    condition is written over, and a daily gate scored on a subset would be
    applying a different bar under the same name;
  * the whole `rel_vs_trust` set -- ADR-043's flood condition;
  * the three acceptance probes, unchanged.

Task 605 first cut a 329-query slice of the classes task 603's mechanism
predicts (statute/tax/case-law + foundation), and it did reproduce the
battery's verdict on the 09-16 pack exactly.  It was still the wrong design:
it left 361 golds unmeasured between batteries and made the daily number a
detector rather than ADR-043's own.  Measured on the twin, the whole core
costs **45 s against that slice's 23 s** -- twenty-two seconds a day to keep
the bar identical, and to make all four ADR-043 conditions (net, flood,
no-class-broken, tripwires) computable daily instead of two of them.

So what the weekly battery adds is no longer "the golds you skipped": it is
the other 755 held-out queries -- slice_b, the dom_* sets, jurisdiction,
freshness, commercial, general_flood, entity cards, zero_signal, vocab
robustness -- plus ADR-043 am.3's null arm.  That is the thing Tim's ruling
moved off the daily path.

The ids are read from the yaml files, never from a list someone typed
(memory: census-roster-from-manifest).

    cyt gate news-daily-slice [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from cytir.core.paths import REPO_ROOT as ROOT, data_path
from cytir.gates import authority as gate

OUT = data_path("news-daily-gate-slice.v1.json")
PROBE_FIXTURE = data_path("gates", "acceptance_probes.v1.json")

#: The sealed slice file carries the label of the script that first wrote it; the contract pins
#: that file's sha256, so the label is FROZEN there. A regeneration (a PR that re-pins the
#: contract) writes GENERATED_BY. `--check` compares the selection content only: the three prose
#: labels that name the generator (`generated_by`, `note`, `probe_fixture.path`) are left out, every
#: id list, count and sha256 is compared.
GENERATED_BY = "aw.gates.news_daily_slice"

SCHEMA = "aw.news-daily-gate-slice.v1"

#: Gold sets whose h@1 / h@10 the daily gate adds up, and the selector each
#: one is taken with. `None` means the whole set -- which is all of them:
#: these four ARE ADR-043's 663, and the comparator's own `GOLD663`.
#: The selector machinery stays because a narrower slice is a PR away if the
#: cost ever stops being 45 s, and because it is what keeps the roster a rule
#: rather than a list.
GOLD_SETS = {
    "authoritative": None,
    "foundation": None,
    "legal": None,
    "names": None,
}
#: ADR-043 condition 2's set. Scored, never summed into h@1/h@10.
FLOOD_SET = "rel_vs_trust"

FAMILY_RE = re.compile(r"-\d+$")


def family(qid: str) -> str:
    return FAMILY_RE.sub("", qid)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qids_of(name: str, path: Path) -> list[str]:
    """Every gold id of one set, derived exactly as a full arm derives it.

    `cytir/gates/run_corpus_arm.py` falls back to `f"{name}-{index}"`
    for a row with no id, and the index is its position in the FULL set.  A
    slice that enumerated only its own rows would therefore invent ids that
    no battery arm has, and the two instruments would stop being comparable
    at exactly the queries with no id of their own.  So the ids are always
    computed over the whole file and filtered afterwards.
    """
    rows = gate.load_rows(str(path))
    return [str(row.get("id") or row.get("qid") or f"{name}-{index}")
            for index, row in enumerate(rows)]


def build() -> dict:
    sets: dict[str, list[str]] = {}
    sources = []
    paths = dict(gate.SETS)
    for name in (*GOLD_SETS, FLOOD_SET):
        path = Path(paths[name])
        if not path.is_file():
            raise SystemExit(f"gold set absent from this release: {path}")
        every = qids_of(name, path)
        selector = GOLD_SETS.get(name)
        chosen = ([q for q in every if any(family(q).endswith(suffix)
                                          for suffix in selector)]
                  if selector else list(every))
        sets[name] = sorted(chosen)
        sources.append({"path": path.relative_to(ROOT).as_posix(),
                        "sha256": digest(path), "rows": len(every),
                        "selected": len(chosen),
                        "family_suffixes": list(selector) if selector else None})
    probes = sorted(item["id"] for item in
                    json.loads(PROBE_FIXTURE.read_text()))
    return {
        "schema": SCHEMA,
        "adr": "ADR-043 amendment 4",
        "generated_by": GENERATED_BY,
        "note": __doc__.split("\n\n", 1)[1].strip(),
        "gold_sets": sorted(GOLD_SETS),
        "flood_set": FLOOD_SET,
        "tripwire_sets": ["legal", "names"],
        "sets": sets,
        "probes": probes,
        "probe_fixture": {"path": PROBE_FIXTURE.relative_to(ROOT).as_posix(),
                          "sha256": digest(PROBE_FIXTURE)},
        "sources": sources,
        "counts": {"gold": sum(len(sets[name]) for name in GOLD_SETS),
                   "flood": len(sets[FLOOD_SET]),
                   "queries": sum(len(value) for value in sets.values()),
                   "probes": len(probes)},
    }


def _without_label(text: str) -> dict:
    body = json.loads(text) if text else {}
    for key in ("generated_by", "note"):
        body.pop(key, None)
    (body.get("probe_fixture") or {}).pop("path", None)
    return body


def main(argv=None, prog=None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the committed slice is stale")
    args = parser.parse_args(argv)
    payload = json.dumps(build(), indent=1, sort_keys=True) + "\n"
    if args.check:
        current = OUT.read_text() if OUT.is_file() else ""
        if _without_label(current) != _without_label(payload):
            print(f"STALE: {OUT.name} is not what the rule "
                  f"produces from today's gold sets -- regenerate it in a PR")
            return 2
        print(f"OK: {OUT.name} matches the rule")
        return 0
    OUT.write_text(payload)
    counts = json.loads(payload)["counts"]
    print(f"wrote {OUT} ({counts['queries']} queries, "
          f"{counts['gold']} gold, {counts['probes']} probes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
