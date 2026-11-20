#!/usr/bin/env python3
"""COVERAGE vs RANKING, measured before a class regression is named.

WHY THIS EXISTS (task 544, 2026-09-08). The 1,418 gate reported the `names`
class at h@1 5 -> 2 between the twin and the `aw_e8` candidate, and task 538
attributed the loss -- together with `entity_card_golds` and Tim's "Marc Maret"
check -- to the entity-card sparse channel. It was not that. Three of the seven
`names` golds are simply NOT IN the candidate collection:

    https://impactia.org/foundation-board/?lang=en   twin 5 points, aw_e8 0
    https://impactia.org/the-team/?lang=en           twin 6 points, aw_e8 0
    https://impactia.org/robotme/                    twin 3 points, aw_e8 0

A gold that is absent cannot be ranked, so those three were never a ranking
finding, and no amount of surface work or re-tuning could ever have reached
them. The gate had no way to say so: it scores ranks, and an absent gold and a
badly-ranked gold produce the same `rank: null`. Task 538 separated the two by
hand, per query, after the fact.

THIS GATE MAKES THAT SEPARATION MECHANICAL AND FAIL-CLOSED.

  * The roster is the gate's OWN scored sets (`gate_authority.SETS`) and their
    own gold semantics (`gate_authority.gold_of`) -- never a hand-typed list
    (corpus-doctrine 8, "absence is measured, never inferred").
  * Presence is an INDEXED-key lookup on `url`. If `url` is not an indexed
    payload key on the collection, the gate REFUSES rather than running a full
    scan (a full scan on a serving box is an outage, and a filter nobody
    checked is a join nobody proved -- corpus-doctrine 5).
  * URL variants (bare, fragment-stripped, with and without a trailing slash)
    are all tried: one trailing slash once took a population from 1 matched
    point to 133,179.
  * A LIVENESS CONTROL runs first, per collection: a URL known to be present
    must resolve. Without it every miss below could be an outage recorded as a
    genuine negative (verification-doctrine 6, "an error is not an empty").
  * The verdict is an EXIT CODE, and the failing condition is a gold the
    INCUMBENT holds and the CANDIDATE does not. Absent from both is reported
    separately: it is a pre-existing gap, not a regression this build caused.

    cyt gate gold-presence \\
        --qdrant <twin-url> --collection <twin-collection> \\
        --incumbent-qdrant http://localhost:16444 --incumbent aw_e3 \\
        --sets names,entity_card_golds --out report.json

    exit 0 = no gold lost   2 = golds lost (coverage)   3 = refused

ADDENDUM (task 553, 2026-09-08). Task 545 sec 2.1 found a SECOND instrument
carrying the coverage-vs-ranking question wrong: task 538's
`twin_hit_presence` probe tested presence of the TWIN's matched hit url instead
of the GOLD's own url. For `ch-law-003`, the twin's rank-1 hit was a finer
fragment (`#art_10-c2`) than the gold's own target (`#art_10`); the candidate
held the gold's exact url at rank 9 -- a genuine RANKING loss (1 -> 9), filed
as COVERAGE because the probe checked the wrong string. `rank_of`'s own
page-level leniency (an exact-fragment match OR a fragment-STRIPPED page
match) is exactly what let the twin's hit drift to a different fragment of
the same page than the gold names; a coverage test that does not carry the
same leniency will keep mislabeling this class. `coverage_identity_urls` and
`classify_lost_or_worse` below are the one home for that identity, for this
gate and for any per-query lost/worse classifier -- `twin_hit_presence.py`
now calls them rather than re-deriving its own.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from cytir.core import internal_http as IH
from cytir.gates import authority as GA
from cytir.index import prod_guard as PG  # no prod scroll without a reason (2026-10-07)
import cytir.index.supersede as US  # THE scheme/www fold, one home

#: The payload key presence is read from. It must be INDEXED on every
#: collection this gate touches; see `assert_indexed`.
PRESENCE_KEY = "url"


class Refusal(RuntimeError):
    """The gate cannot measure. Never downgraded to a miss."""


def url_variants(url: str) -> list[str]:
    """Every literal payload-`url` spelling a gold's own page may be stored
    under. Two independent folds compose here, because a Qdrant payload
    filter is an exact-string match -- it cannot apply `gate_authority.
    norm_url`'s normalisation to the stored value, only try more literal
    spellings of the query:

      * fragment forms -- as-given, fragment-stripped (the PAGE), and both
        with/without a trailing slash. This is what lets a presence check
        see `rank_of`'s own page-level leniency (`n in gu or
        n.split('#')[0] in gu_page`) rather than a stricter one this gate
        invented -- one trailing slash once took a population from 1
        matched point to 133,179.
      * scheme/www forms -- `upsert_supersede.url_variants`'s own fold
        (stored urls keep scheme+www as captured; reused here, not
        re-derived -- engineering doctrine 1).
    """
    base = (url or "").split("#")[0]
    out, seen = [], set()
    for frag_form in (url, base, base.rstrip("/"), base + "/"):
        if not frag_form:
            continue
        for v in ({frag_form} | US.url_variants(frag_form)):
            if v and v not in seen:
                seen.add(v)
                out.append(v)
    return out


def coverage_identity_urls(gold_urls, twin_hit_url=None) -> list[str]:
    """The url(s) a COVERAGE-vs-RANKING test must probe for one gold the
    incumbent beat the candidate on. The gold's own target url(s)
    (`gate_authority.gold_of`'s `gold_urls`) are ALWAYS included -- never
    dropped in favour of a twin's or a candidate's hit string, which was
    task 538's `twin_hit_presence.py` bug (task 553; task 545 sec 2.1,
    `ch-law-003`: the twin matched a FINER fragment, `#art_10-c2`, than the
    gold's own `#art_10`; the candidate held the gold's exact url at rank 9;
    testing only the twin's hit reported COVERAGE for what was really a
    ranking loss).

    The twin's hit url is folded in too, ADDITIONALLY, not as a fallback:
    `rank_of`'s own page-level leniency is bidirectional, and it goes the
    OTHER way just as often -- a gold with NO fragment at all (a
    document-level "any part of this law counts" target) over a family
    that only ever stores per-article fragments can never literally match
    any payload row, however much of the document is genuinely present
    (eur-lex CELEX golds `eu-001`/`eu-011`/`eu-003`: task 547 fetched the
    exact consolidated edition the gold names, but every stored row carries
    an `#art_N` fragment the gold doesn't, so gold-url-only probing would
    report COVERAGE for a document task 547 just landed). A literal-string
    probe cannot enumerate every equivalence `rank_of`'s `norm_url()`
    tolerates (whole-url case folding, percent-encoding, fragment
    granularity in either direction) -- but it does not need to: the twin's
    hit is a CONCRETE payload string `rank_of` has already proven matches
    this exact gold, at the exact rank the classifier is explaining, so
    testing it directly covers every one of those equivalence classes
    without inventing a new normalisation for each. Only a gold with NO url
    at all (a pure domain gold -- most `freshness`-intent rows) has nothing
    but the twin's hit to test, same as before this fix.
    """
    urls = list(gold_urls or [])
    if twin_hit_url and twin_hit_url not in urls:
        urls.append(twin_hit_url)
    return urls


def classify_lost_or_worse(present, gold_urls, twin_hit_url=None):
    """COVERAGE vs RANKING verdict for one gold the incumbent beat the
    candidate on, tested against `coverage_identity_urls`'s correct
    identity. `present` is a presence function as returned by
    `qdrant_backend` / `sealed_backend` below. Returns None (unscorable)
    only when neither a gold url nor a twin hit url survives -- a row with
    no identity to test at all, which should not happen for anything
    carrying a scored twin rank."""
    urls = coverage_identity_urls(gold_urls, twin_hit_url)
    if not urls:
        return None
    return "RANKING" if any(present(u) for u in urls) else "COVERAGE"


def gold_roster(set_names=None) -> list[dict]:
    """[{set, id, url}] over the gate's own sets. Never a hand-list."""
    rows = []
    for name, path in GA.SETS:
        if set_names and name not in set_names:
            continue
        if not os.path.exists(path):
            raise Refusal(f"gate set {name!r} is not on disk at {path!r} -- an "
                          f"absent roster is not an empty one")
        for j, r in enumerate(GA.load_rows(path)):
            g_urls, _g_domains = GA.gold_of(r)
            qid = str(r.get("id") or r.get("qid") or f"{name}-{j}")
            for u in g_urls:
                rows.append({"set": name, "id": qid, "url": u})
    if set_names:
        missing = set(set_names) - {r["set"] for r in rows}
        if missing:
            raise Refusal(f"asked for sets {sorted(missing)} and none of their "
                          f"rows carry a gold url -- refusing to report a "
                          f"vacuous PASS over an empty roster")
    return rows


# ---------------------------------------------------------------- backends -- #

def _api(base: str, path: str, body: dict | None = None, *, method: str | None = None,
         prod_scroll_reason: str | None = None) -> dict:
    """One qdrant JSON call through the internal client; the prod-read guard runs first."""
    try:
        PG.check(base, method or ("POST" if body is not None else "GET"), path,
                 allow_prod_scroll=bool(prod_scroll_reason), reason=prod_scroll_reason)
    except PG.ProdReadRefused as exc:
        raise Refusal(str(exc)) from exc
    # 600 s, as the retired qdrant_http api used: a scroll on a large collection is slow
    # and the internal client's 30 s default would fail it.
    return IH.fetch_json(base.rstrip("/") + path, body=body, method=method, timeout=600.0)


def qdrant_backend(qdrant: str, collection: str, prod_scroll_reason: str | None = None):
    """A presence function over a live collection, indexed-key only."""
    def api(base, path, body=None, method=None):
        return _api(base, path, body, method=method, prod_scroll_reason=prod_scroll_reason)

    schema = (api(qdrant, f"/collections/{collection}", None,
                  method="GET")["result"].get("payload_schema") or {})
    if PRESENCE_KEY not in schema:
        raise Refusal(
            f"{PRESENCE_KEY!r} is not an indexed payload key on {collection} "
            f"(indexed: {sorted(schema)}). A filter on an unindexed key is a "
            f"full scan; refusing.")

    def present(url: str) -> bool:
        for v in url_variants(url):
            pts = api(qdrant, f"/collections/{collection}/points/scroll",
                      {"filter": {"must": [{"key": PRESENCE_KEY,
                                            "match": {"value": v}}]},
                       "limit": 1, "with_payload": False})["result"]["points"]
            if pts:
                return True
        return False

    return present


def sealed_backend(mapping: dict):
    """A presence function over a sealed capture: {url: bool}.

    Tests run against this, never against a live collection: a guard whose
    test needs a running qdrant is a guard nobody runs.
    """
    norm = {GA.norm_url(u): bool(v) for u, v in mapping.items()}

    def present(url: str) -> bool:
        return any(norm.get(GA.norm_url(v), False) for v in url_variants(url))

    return present


# ------------------------------------------------------------------- gate --- #

def check_liveness(present, control_urls, label):
    """Prove the channel answers before any zero is recorded."""
    if not control_urls:
        raise Refusal(
            f"no liveness control for {label}. Every miss below would be "
            f"indistinguishable from an outage; refusing to record zeros "
            f"from an unproven channel.")
    for u in control_urls:
        if present(u):
            return u
    raise Refusal(
        f"liveness control FAILED for {label}: none of {list(control_urls)} "
        f"resolved. An error is not an empty -- fix the channel, do not read "
        f"these as genuine absences.")


def run(candidate_present, incumbent_present=None, set_names=None,
        candidate_control=(), incumbent_control=(), max_lost=0,
        candidate_label="candidate", incumbent_label="incumbent"):
    """Returns (report, exit_code). Raises Refusal when it cannot measure."""
    roster = gold_roster(set_names)
    check_liveness(candidate_present, candidate_control, candidate_label)
    if incumbent_present is not None:
        check_liveness(incumbent_present, incumbent_control, incumbent_label)

    rows = []
    for g in roster:
        c = bool(candidate_present(g["url"]))
        i = None if incumbent_present is None else bool(incumbent_present(g["url"]))
        if i is None:
            verdict = "present_candidate" if c else "absent_candidate"
        elif c and i:
            verdict = "present_both"
        elif c and not i:
            verdict = "gained_by_candidate"
        elif i and not c:
            verdict = "LOST_BY_CANDIDATE"
        else:
            verdict = "absent_both"
        rows.append(dict(g, candidate=c, incumbent=i, verdict=verdict))

    per_set = {}
    for r in rows:
        s = per_set.setdefault(r["set"], {})
        s[r["verdict"]] = s.get(r["verdict"], 0) + 1
    lost = [r for r in rows if r["verdict"] == "LOST_BY_CANDIDATE"]
    absent_both = [r for r in rows if r["verdict"] == "absent_both"]
    absent_cand = [r for r in rows if r["verdict"] == "absent_candidate"]

    report = {
        "schema": "aw.gold-presence-gate.v1",
        "gate_set_version": GA.GATE_SET_VERSION,
        "candidate": candidate_label, "incumbent": incumbent_label,
        "sets_scored": sorted({r["set"] for r in rows}),
        "n_golds": len(rows),
        "per_set": per_set,
        "n_lost": len(lost),
        "n_absent_both": len(absent_both),
        "n_absent_candidate_no_incumbent": len(absent_cand),
        "max_lost_allowed": max_lost,
        "lost": lost,
        "absent_both": absent_both,
        "rows": rows,
    }
    # Absent-from-both is NOT a failure: the candidate did not cause it, and
    # failing on it would make the gate unpassable for reasons outside the
    # change under test. It is reported so it can never be discovered later
    # as a surprise.
    report["verdict"] = "PASS" if len(lost) <= max_lost else "FAIL"
    return report, (0 if len(lost) <= max_lost else 2)


def main(argv=None, prog: str = "cyt gate gold-presence") -> int:
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--qdrant", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--incumbent-qdrant")
    ap.add_argument("--incumbent")
    ap.add_argument("--sets", help="comma-separated; default = every scored set")
    ap.add_argument("--control-url", action="append", default=[],
                    help="a URL known to be present; repeatable. Required.")
    ap.add_argument("--incumbent-control-url", action="append", default=[])
    ap.add_argument("--max-lost", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--prod-scroll-reason",
                    help="name why a PROD target may be read; without it the prod-read guard refuses")
    a = ap.parse_args(argv)

    set_names = ([s.strip() for s in a.sets.split(",") if s.strip()]
                 if a.sets else None)
    try:
        cand = qdrant_backend(a.qdrant, a.collection, a.prod_scroll_reason)
        inc = None
        if a.incumbent:
            inc = qdrant_backend(a.incumbent_qdrant or a.qdrant, a.incumbent, a.prod_scroll_reason)
        report, code = run(
            cand, inc, set_names=set_names,
            candidate_control=a.control_url,
            incumbent_control=(a.incumbent_control_url or a.control_url),
            max_lost=a.max_lost,
            candidate_label=a.collection,
            incumbent_label=a.incumbent or "none")
    except Refusal as exc:
        print(f"GOLD-PRESENCE REFUSED: {exc}", file=sys.stderr)
        return 3

    print(f"gold-presence {report['verdict']}: {report['n_lost']} gold(s) held "
          f"by {report['incumbent']} and absent from {report['candidate']}; "
          f"{report['n_absent_both']} absent from both (pre-existing); "
          f"{report['n_golds']} golds over {len(report['sets_scored'])} sets")
    for r in report["lost"][:20]:
        print(f"  LOST  {r['set']:<18} {r['id']:<14} {r['url']}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("->", a.out)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
