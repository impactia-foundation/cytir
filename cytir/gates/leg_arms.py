#!/usr/bin/env python3
# Moved from the retired `gate_leg_arms` module (task 677, task 643 S5); the old path was removed (task 683 S9).
"""The DENSE-ONLY and SEALED-IDF BM25-ONLY gate arms (ADR-043 A8 rule 4).

Task 629 P4. A8 rule 4: *"The gate reads three arms: dense-only, BM25-only, fused.
The per-leg split is what made the layout channel attributable in task 627; it
becomes routine so the next instrument move attributes itself. The BM25-only arm
runs under ADR-109's sealed IDF table once that ships (R2), which makes it
layout-invariant by construction."*

Task 627 §12.6 recorded honestly that neither arm was built, and why: the shipped
`cytir/serve/bm25_idf.json` covers only 3,117 of 6,894 query tokens as a
*weight*, and a table sealed from a drifted collection would bake that drift in
permanently. This module is the scoring half — pure, testable, and independent of
when the table gets sealed.

WHY A PER-LEG ARM IS THE THING THAT CARRIES A VERDICT
-----------------------------------------------------
ADR-109: the BM25 leg's `idf(t) = ln((n+1)/(df+0.5))` is computed from each
segment's sparse-index file header, and the on-disk inverted index cannot
represent a deletion, so `n` and `df` count every point deleted or overwritten
since that segment was built. Consequence: **the BM25 leg's ranking of rows nobody
touched is a function of the segment layout**, an optimizer merge moves it with
zero writes, and it moved the 639-gold daily slice by 5 h@1 and 32 golds.

So a FUSED reading across two collections mixes the pack's effect with a layout
artefact, and task 627 §7.4 measured that the artefact is not bandable: at a fixed
layout the instrument is exactly deterministic (0 golds moved, twice, on two
different collections), so there is no run-to-run noise to band — all of the
movement is layout, and a band wide enough to cover it (±5 on 639 golds) would
blind the gate to the loss class it exists to catch.

The two arms here answer the question the fused reading cannot:

* **dense-only** is untouched by the IDF channel entirely — the dense leg does not
  read sparse statistics — so a dense-only delta is attributable to the content.
* **BM25-only under a SEALED table** is layout-invariant *by construction*: the
  score becomes a pure function of (query, stored vectors, sealed table). Task 627
  §3.5 measured it: 30/30 golds' leg order collapses onto each other under one
  shared IDF vector, against 4/30 as Qdrant scores it live.

WHAT THIS MODULE IS NOT
-----------------------
It does not seal the table (that needs a settled collection and the build box's
the fetched German splitter table (`cytir.core.decompound`) — task 627 §6-R3 cost 3), it does not touch
`cytir/serve/bm25_idf.json` (overwriting the serving table would be a ranking
change under ADR-109 decision 2, needing the ADR-043 ranking bar and a fusion
re-tune), and it grants no band. It scores.
"""
from __future__ import annotations

import json
import math
import pathlib
from typing import Iterable, Mapping, Sequence

SCHEMA = "aw.gate-leg-arms.v1"

#: A leg the arms score. `country_*` are rescue legs, reported separately by the
#: trace and deliberately not folded in here: a rescue changes which pool a gold
#: could be found in, which is a different question from how a leg ordered it.
SCORED_LEGS = ("dense", "bm25")

#: The default for a query token absent from the sealed table. Task 627 §6-R3
#: cost (2) is explicit that this has to be DECIDED, not inherited: the shipped
#: loader treats an absent token as *high*-IDF, which is right for ADR-099's
#: suppression threshold ("when in doubt, keep the term") and **wrong** for a
#: scoring weight ("when in doubt, do not silently make the term the most
#: important one in the query"). So the default here is 0.0 — an unmeasured token
#: contributes nothing rather than dominating — and `coverage()` reports how much
#: of a query fell through, so a verdict taken over a thin table is visibly thin
#: instead of quietly wrong.
ABSENT_TOKEN_IDF = 0.0

#: Below this share of a query's tokens present in the table, the arm refuses to
#: score that query rather than returning a number computed from a fragment.
#: Reported per query either way.
DEFAULT_COVERAGE_FLOOR = 0.80


class LegArmError(RuntimeError):
    """A refusal. Never downgraded to a score."""


# --------------------------------------------------------------- the gold match
def _norm(url: str) -> str:
    """Compare urls the way the gate already does, via `gate_authority.norm_url`
    when it is importable, and by a conservative fallback otherwise. One rule, so
    a leg rank and a fused rank cannot disagree about what counts as the gold."""
    try:
        from cytir.gates import authority as gate_authority  # noqa: PLC0415
        return gate_authority.norm_url(url)
    except Exception:  # noqa: BLE001
        value = (url or "").strip().lower().rstrip("/")
        for prefix in ("https://", "http://"):
            if value.startswith(prefix):
                value = value[len(prefix):]
        return value[4:] if value.startswith("www.") else value


def rank_in_order(ids: Sequence[str], urls: Mapping[str, str],
                  gold_urls: Iterable[str],
                  gold_domains: Iterable[str] = ()) -> int | None:
    """1-based position of the first gold in a leg's own order, or None.

    `urls` is the trace's pid->url map. An id with no url is SKIPPED but still
    occupies its rank, because the leg really did return it — silently closing the
    gap would report a better rank than the leg produced.
    """
    wanted = {_norm(u) for u in gold_urls if u}
    domains = {(d or "").strip().lower().lstrip(".") for d in gold_domains if d}
    for position, pid in enumerate(ids, 1):
        url = urls.get(str(pid))
        if not url:
            continue
        norm = _norm(url)
        if norm in wanted:
            return position
        if domains and any(norm == d or norm.startswith(d + "/")
                           or norm.split("/")[0].endswith("." + d)
                           for d in domains):
            return position
    return None


# ------------------------------------------------------ the sealed-IDF BM25 leg
def load_idf_table(path: str | pathlib.Path) -> dict:
    """Read a sealed IDF table and refuse one that cannot be used as a weight.

    The refusals are the costs ADR-109 names, checked rather than trusted:
    a table with no `points_count` cannot be placed on a collection's history, and
    a table measured on an UNSETTLED collection has baked that collection's stale
    postings in permanently (task 627 §6-R3 cost 3) — which is worse than having no
    table, because it ships the drift to every self-hoster.
    """
    doc = json.loads(pathlib.Path(path).read_text())
    idf = doc.get("idf") or doc.get("tokens") or {}
    if not isinstance(idf, Mapping) or not idf:
        raise LegArmError(f"{path}: no `idf` mapping — not a usable table")
    if doc.get("points_count") is None:
        raise LegArmError(
            f"{path}: no `points_count`. A sealed table is only meaningful "
            "against the collection state it was measured on; without that "
            "number it cannot be placed on the collection's history at all.")
    # `is True`, not `is not False`. `measure_bm25_idf.py` does not write this
    # field at all, so accepting its ABSENCE would let the producer bypass the
    # refusal simply by not mentioning it — the check would then only catch a
    # table honest enough to admit the problem. A gate table must carry positive
    # evidence of the state it was measured on; `cytir.index.idf_table seal` stamps
    # it from a live layout fingerprint.
    if doc.get("settled") is not True:
        raise LegArmError(
            f"{path}: not stamped `settled: true`. A table sealed from a drifted "
            "layout bakes that collection's stale postings in permanently and "
            "ships them to every self-hoster (task 627 §6-R3 cost 3), so a gate "
            "table must PROVE it was measured on a settled collection rather than "
            "omit the question. Seal it with `python -m cytir.index.idf_table seal`, "
            "which asserts the A8 predicate and records the fingerprint.")
    return {"idf": {str(k): float(v) for k, v in idf.items()},
            "points_count": int(doc["points_count"]),
            "settled": True,
            "measured_at": doc.get("measured_at"),
            "layout_fingerprint": doc.get("layout_fingerprint"),
            "vocabulary": len(idf),
            "source": str(path)}


def coverage(query_tokens: Iterable[str], table: Mapping) -> dict:
    """How much of one query the sealed table can actually weigh."""
    tokens = [str(t) for t in query_tokens]
    idf = table["idf"]
    present = [t for t in tokens if t in idf]
    return {"tokens": len(tokens), "present": len(present),
            "absent": sorted(set(tokens) - set(present)),
            "share": (len(present) / len(tokens)) if tokens else 0.0}


def sealed_bm25_score(query: Mapping[str, float], stored: Mapping[str, float],
                      table: Mapping) -> float:
    """One document's BM25 score under the SEALED table.

    Qdrant scores a sparse pair as `sum_t idf(t) * q(t) * d(t)` (ADR-109). Replace
    its `idf(t)` — read from segment headers, and therefore a function of the
    layout — with the sealed table's, and the score becomes a pure function of
    (query, stored vector, table). That is the whole layout-invariance property,
    and it is one line of arithmetic.
    """
    idf = table["idf"]
    total = 0.0
    for token, q_value in query.items():
        d_value = stored.get(token)
        if not d_value:
            continue
        total += idf.get(str(token), ABSENT_TOKEN_IDF) * float(q_value) * float(d_value)
    return total


def sealed_bm25_order(query: Mapping[str, float],
                      pool: Mapping[str, Mapping[str, float]],
                      table: Mapping) -> list[str]:
    """The pool re-ordered under the sealed table, best first.

    Ties break on the point id so the order is total and reproducible — an
    arbitrary tie order would reintroduce, as a coin flip, exactly the
    irreproducibility this arm exists to remove.
    """
    scored = [(sealed_bm25_score(query, stored, table), str(pid))
              for pid, stored in pool.items()]
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [pid for _score, pid in scored]


# ----------------------------------------------------------------- arm scoring
def score_leg(per_q: Mapping[str, Mapping], leg: str, golds: Mapping[str, dict],
              ) -> dict:
    """h@1 / h@10 and per-gold ranks for ONE leg of one already-cut arm.

    Reads a stored arm; cuts nothing. That is deliberate: the leg arms must be
    derivable from the arms the gate ALREADY cuts, or every batch pays for two more
    full readings and the gate cadence — which the 100M spec §5.3 already names as
    the limiter on W1, not compute — gets worse.

    **The refusal that matters is `urls`.** An arm cut before `trace()` carried the
    pid->url map has full leg id lists and no way to resolve any of them, so every
    rank would come back `None` and the leg would read as *every gold lost* — a
    -639 h@1 produced entirely by a missing input. That is the failure mode this
    project keeps meeting (memory `verify-liveness-before-trusting-misses`: never
    record an error as an empty), so a leg present WITHOUT a usable url map is
    counted `unresolvable` and refused, never scored as a miss.
    """
    if leg not in SCORED_LEGS:
        raise LegArmError(f"{leg!r} is not a scored leg ({SCORED_LEGS})")
    ranks, h1, h10, scored = {}, 0, 0, 0
    no_trace, unresolvable = [], []
    for qid, gold in golds.items():
        row = per_q.get(qid)
        if row is None:
            continue
        trace = row.get("trace")
        if not isinstance(trace, Mapping) or leg not in trace:
            no_trace.append(qid)
            continue
        ids = trace.get(leg) or []
        urls = trace.get("urls") or {}
        if ids and not any(str(pid) in urls for pid in ids):
            # The leg returned candidates and not one of them can be resolved to a
            # url. That is an arm cut without TRACE_EXTRA_KEYS, not a leg that
            # missed.
            unresolvable.append(qid)
            continue
        rank = rank_in_order(ids, urls, gold.get("urls") or (),
                             gold.get("domains") or ())
        ranks[qid] = rank
        scored += 1
        h1 += rank == 1
        h10 += bool(rank and rank <= 10)
    verdict = {"leg": leg, "h1": h1, "h10": h10, "scored": scored,
               "ranks": ranks, "queries_without_a_trace": no_trace,
               "queries_unresolvable": unresolvable}
    if unresolvable and not scored:
        verdict["status"] = "UNSCOREABLE"
        verdict["why"] = (
            f"{len(unresolvable)} quer(y|ies) carry a `{leg}` leg whose ids "
            "resolve to no url, and none scored. This arm was cut before "
            "`run_corpus_arm.trace()` carried `urls` (TRACE_EXTRA_KEYS); re-cut "
            "it rather than reading this as a leg that lost every gold.")
    else:
        verdict["status"] = "OK"
    return verdict


SEALED_LEG = "bm25_sealed"


def score_sealed_leg(per_q: Mapping[str, Mapping], golds: Mapping[str, dict],
                     table: Mapping) -> dict:
    """The BM25 leg RE-RANKED under the sealed table. A8 rule 4's third arm.

    Reads the same already-cut arm as `score_leg`, plus the two inputs
    `run_corpus_arm.trace(capture_sparse=True)` records: the query's bm25 vector
    and each traced candidate's stored bm25 values. The pool is re-ordered by
    `sealed_bm25_order`, so the resulting rank is a pure function of (query,
    stored vectors, table) -- no segment header, hence no layout.

    Returns the SAME shape as `score_leg`, so `compare_leg` works on it unchanged
    and the null-arm correction applies over one shape rather than two.

    Refusals, both of which would otherwise read as "the leg lost every gold":
      * a query whose trace carries no `sparse` at all -> `queries_unresolvable`,
        and UNSCOREABLE if that is every query. An arm cut without
        --capture-sparse is a missing input, not a leg that missed.
      * a query whose sealed re-order produces an EMPTY pool while the live leg
        returned candidates -> also unresolvable. That happens when no candidate
        shares a token with the query, which for a leg that returned those
        candidates BY bm25 score means the capture is wrong, not the corpus.
    """
    ranks, h1, h10, scored = {}, 0, 0, 0
    no_trace, unresolvable = [], []
    for qid, gold in golds.items():
        row = per_q.get(qid)
        if row is None:
            continue
        trace = row.get("trace")
        if not isinstance(trace, Mapping) or "bm25" not in trace:
            no_trace.append(qid)
            continue
        query = trace.get("bm25_query")
        pool = trace.get("sparse")
        if not isinstance(query, Mapping) or not isinstance(pool, Mapping):
            unresolvable.append(qid)
            continue
        live_ids = trace.get("bm25") or []
        if live_ids and not pool:
            unresolvable.append(qid)
            continue
        order = sealed_bm25_order(query, pool, table)
        urls = trace.get("urls") or {}
        rank = rank_in_order(order, urls, gold.get("urls") or (),
                             gold.get("domains") or ())
        ranks[qid] = rank
        scored += 1
        h1 += rank == 1
        h10 += bool(rank and rank <= 10)
    verdict = {"leg": SEALED_LEG, "h1": h1, "h10": h10, "scored": scored,
               "ranks": ranks, "queries_without_a_trace": no_trace,
               "queries_unresolvable": unresolvable,
               "idf_table": table.get("source"),
               "idf_points_count": table.get("points_count")}
    if unresolvable and not scored:
        verdict["status"] = "UNSCOREABLE"
        verdict["why"] = (
            f"{len(unresolvable)} quer(y|ies) carry a bm25 leg with no usable "
            "`bm25_query`/`sparse` capture, and none scored. Cut the arms with "
            "`run_corpus_arm.py --capture-sparse` rather than reading this as a "
            "leg that lost every gold.")
    else:
        verdict["status"] = "OK"
    return verdict


def compare_leg(baseline: Mapping, candidate: Mapping) -> dict:
    """Per-gold movement on one leg between two arms.

    `losses` and `gains` are RAW, exactly as ADR-043 A3 requires: the null-arm
    correction is applied by the caller over the same shape, and a module that
    pre-corrected them would hide the raw list A3 rule 2 insists is printed in
    full.
    """
    if baseline["leg"] != candidate["leg"]:
        raise LegArmError(
            f"comparing different legs: {baseline['leg']} vs {candidate['leg']}")
    for label, side in (("baseline", baseline), ("candidate", candidate)):
        if side.get("status") == "UNSCOREABLE":
            raise LegArmError(
                f"{label} leg arm is UNSCOREABLE: {side.get('why')} A delta "
                "computed against it would be a number about the instrument, not "
                "about the pack.")
    losses, gains, moved = [], [], []
    shared = set(baseline["ranks"]) & set(candidate["ranks"])
    for qid in sorted(shared):
        before, after = baseline["ranks"][qid], candidate["ranks"][qid]
        if before == after:
            continue
        moved.append({"qid": qid, "before": before, "after": after})
        key = 10 ** 9
        b = key if before is None else before
        a = key if after is None else after
        (losses if a > b else gains).append(
            {"qid": qid, "before": before, "after": after})
    return {
        "leg": baseline["leg"],
        "h1_delta": candidate["h1"] - baseline["h1"],
        "h10_delta": candidate["h10"] - baseline["h10"],
        "losses": losses, "gains": gains, "moved": moved,
        "compared": len(shared),
        "only_in_baseline": sorted(set(baseline["ranks"]) - shared),
        "only_in_candidate": sorted(set(candidate["ranks"]) - shared),
    }


def implied_df(idf_value: float, points: int) -> float:
    """Invert Qdrant's IDF for REPORTING only (same formula measure_bm25_idf
    documents). Handy when a table's numbers have to be sanity-checked against a
    collection's size, and never used in scoring."""
    return (points + 0.5) / math.exp(idf_value) - 0.5
