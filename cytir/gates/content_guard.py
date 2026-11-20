#!/usr/bin/env python3
"""The CONTENT GUARD -- task 626, Tim's standing catch-up waiver
(`GO-626-standing-catchup`), condition 3.

WHY THIS EXISTS
----------------
Task 627 root-caused the daily gate's catch-up-era REFUSEs to a BM25-leg
instrument property (stale IDF segment-header statistics, ADR-109): a write
moves the ranking of rows NOBODY TOUCHED, on the sparse leg only, with the
dense leg unaffected. Tim's standing waiver lets a REFUSE/UNADJUDICATED
verdict proceed under that finding, PACK BY PACK -- but only after an
independent check that THIS pack's particular losses are consistent with
that mechanism and not a real content defect: for every gold whose fused
rank is WORSE in the candidate than in the concurrent production baseline,
the DENSE leg's retrieved id list (ids AND order) must be IDENTICAL between
the two arms. Any dense difference on a lost gold means something besides
the known instrument property moved that ranking, and the day must STOP,
not proceed under the standing waiver.

WHERE THE TRACE COMES FROM
---------------------------
`cytir.gates.news_daily.cut_arm` records `trace.dense`/`trace.bm25` per
gold (task626 catch-up item 2), the same shape Task370's own
`run_corpus_arm.trace()` uses -- one home, not a second, separately-cut
instrument. This guard runs directly off the daily gate's OWN baseline and
candidate arms; it needs no additional arm-cutting pass.

WHAT THIS MODULE DOES NOT DO
------------------------------
It does not decide whether to proceed -- that is the standing waiver plus
`--waiver-ref`, a human ruling recorded as its own fact. It does not touch a
band, and it does not run when the verdict is already PASS (Tim's ruling,
condition 3, is "mandatory when the verdict is not PASS" -- a clean pass
needs no guard, there is no loss to check).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Iterable, Mapping

import yaml

from cytir.gates import news_daily as daily  # noqa: E402
from cytir.gates import convergence as gc  # noqa: E402

SCHEMA = "aw.news-content-guard.v1"
SCHEMA_POOL = "aw.news-content-guard-pool.v1"
SCHEMA_NEWS = "aw.news-content-guard-news.v1"

#: task626/627 refinement (Tim, 2026-09-21, behind AW_GUARD_MODE=pool,
#: default off): ADR-035's k-stability guarantee is that results at every
#: served k in {3,5,8,10} -- which covers every k this lane's daily gate
#: ever scores at -- are a strict PREFIX of the top-50 per-leg ordering
#: ("results@k in {3,5,8,10} == prefix of results@k=50", decisions.md
#: ADR-035). A dense-leg difference that first appears AFTER this depth
#: could not have been visible to fusion at any k this gate scores, so it
#: cannot be the mechanism behind a lost gold -- it is exactly the kind of
#: harmless HNSW tail reordering task 627 found between mirrors (same
#: class as ai-api-001's 99/100 overlap). Overridable for a future,
#: differently-scored door via AW_GUARD_POOL_DEPTH.
POOL_DEPTH_ENV = "AW_GUARD_POOL_DEPTH"
DEFAULT_POOL_DEPTH = 50
GUARD_MODE_ENV = "AW_GUARD_MODE"


class ContentGuardUnknown(RuntimeError):
    """The guard cannot be computed -- never read as a pass."""


def _dense_trace(arm: Mapping, name: str, qid: str) -> list | None:
    per_q = (arm.get("sets") or {}).get(name, {}).get("per_q") or {}
    row = per_q.get(qid)
    if row is None:
        raise ContentGuardUnknown(f"{name}/{qid}: absent from this arm")
    trace = row.get("trace")
    if not isinstance(trace, Mapping):
        raise ContentGuardUnknown(
            f"{name}/{qid}: no trace recorded on this arm -- cut under the "
            "old door (task626 item 2 pre-dates it) or a different tool "
            "entirely; the guard cannot be computed from it")
    dense = trace.get("dense")
    if not isinstance(dense, list):
        raise ContentGuardUnknown(f"{name}/{qid}: trace.dense is not a list")
    return dense


def evaluate(baseline: Mapping, candidate: Mapping,
            names: Iterable[str] = gc.BAR_SETS) -> dict:
    """Every gold whose fused rank is worse in `candidate` than `baseline`:
    the dense leg (ids AND order) must be identical between the two arms.

    Returns a receipt; never raises for a content violation (that is the
    receipt's `passed: False`) -- only for a guard that could not be
    computed at all (`ContentGuardUnknown`, e.g. a missing trace), which is
    UNKNOWN and must never be read as a pass by a caller (verification-
    doctrine 12, same discipline as `gate_convergence`).
    """
    try:
        losses = daily.loss_rows(baseline, candidate, names)
    except KeyError as exc:
        raise ContentGuardUnknown(
            f"{exc}: absent from this arm -- baseline and candidate must "
            "score the same query composition") from exc
    violations = []
    checked = []
    for qkey, before_rank, after_rank in losses:
        name, qid = qkey.split("/", 1)
        baseline_dense = _dense_trace(baseline, name, qid)
        candidate_dense = _dense_trace(candidate, name, qid)
        identical = baseline_dense == candidate_dense
        checked.append({"qid": qkey, "before_rank": before_rank,
                        "after_rank": after_rank, "dense_identical": identical})
        if not identical:
            violations.append({
                "qid": qkey, "before_rank": before_rank,
                "after_rank": after_rank,
                "baseline_dense": baseline_dense,
                "candidate_dense": candidate_dense})
    return {"schema": SCHEMA,
           "lost_golds": len(losses),
           "checked": checked,
           "violations": violations,
           "passed": not violations}


def pool_depth() -> int:
    """The fusion pool cut depth to use for the `pool` guard mode's
    criterion (b), read from the environment (never hardcoded past this
    one function -- one home) and always printed by the caller that uses
    it, per Tim's instruction."""
    return int(os.environ.get(POOL_DEPTH_ENV, DEFAULT_POOL_DEPTH))


def _first_diff_position(a: list, b: list) -> int | None:
    """The first index at which two ordered id lists disagree, or `None`
    if one is a prefix of the other (or they are identical)."""
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _pool_violation(baseline_row: Mapping, candidate_row: Mapping,
                    pack_ids: set, depth: int) -> dict:
    """task626/627 refinement: a lost gold is a violation only if the
    dense/BM25 movement behind it is ATTRIBUTABLE to the pack, not to
    harmless tail reordering between mirrors. Three independent routes to
    "yes, attributable" (any one is enough): (a) a dense id that is new in
    the candidate's list is a pack row; (b) the dense leg's first
    disagreement is within the fusion pool cut depth (ADR-035
    k-stability -- see `pool_depth()`); (c) a BM25 id that is new in the
    candidate's list is a pack row. Never raises; always returns the full
    diagnostic record Tim asked for, violation or not.
    """
    b_dense = baseline_row.get("trace", {}).get("dense") or []
    c_dense = candidate_row.get("trace", {}).get("dense") or []
    b_bm25 = baseline_row.get("trace", {}).get("bm25") or []
    c_bm25 = candidate_row.get("trace", {}).get("bm25") or []
    new_dense = set(c_dense) - set(b_dense)
    new_bm25 = set(c_bm25) - set(b_bm25)
    dense_pack_count = len(new_dense & pack_ids)
    bm25_pack_count = len(new_bm25 & pack_ids)
    first_diff = _first_diff_position(b_dense, c_dense)
    cond_a = dense_pack_count > 0
    cond_b = first_diff is not None and first_diff < depth
    cond_c = bm25_pack_count > 0
    violation = cond_a or cond_b or cond_c
    reason = ("dense_new_id_is_pack_row" if cond_a else
             "first_dense_diff_within_pool_depth" if cond_b else
             "bm25_new_id_is_pack_row" if cond_c else
             "no_route_to_attribution" if first_diff is not None else None)
    return {"before_rank": baseline_row.get("rank"),
           "after_rank": candidate_row.get("rank"),
           "first_dense_diff_position": first_diff,
           "pool_depth": depth,
           "new_dense_ids": len(new_dense),
           "dense_pack_row_count": dense_pack_count,
           "new_bm25_ids": len(new_bm25),
           "bm25_pack_row_count": bm25_pack_count,
           "violation": violation, "reason": reason}


def evaluate_pool(baseline: Mapping, candidate: Mapping, pack_ids: set,
                  names: Iterable[str] = gc.BAR_SETS,
                  depth: int | None = None) -> dict:
    """The `AW_GUARD_MODE=pool` refinement of `evaluate()` (task626/627,
    Tim 2026-09-21) -- same lost-gold population (`daily.loss_rows`), but a
    lost gold is only a violation when the movement is ATTRIBUTABLE to the
    pack (`_pool_violation`), not merely dense/BM25-different. `pack_ids`
    is the union of the artifact's own row ids and every tombstoned
    predecessor id (source-group and aged-out) -- everything this pack
    actually touched.
    """
    depth = pool_depth() if depth is None else depth
    try:
        losses = daily.loss_rows(baseline, candidate, names)
    except KeyError as exc:
        raise ContentGuardUnknown(
            f"{exc}: absent from this arm -- baseline and candidate must "
            "score the same query composition") from exc
    checked, violations = [], []
    for qkey, before_rank, after_rank in losses:
        name, qid = qkey.split("/", 1)
        b_row = (baseline.get("sets") or {}).get(name, {}).get("per_q", {}).get(qid)
        c_row = (candidate.get("sets") or {}).get(name, {}).get("per_q", {}).get(qid)
        if b_row is None or c_row is None:
            raise ContentGuardUnknown(f"{qkey}: absent from this arm")
        if not isinstance((b_row.get("trace") or {}).get("dense"), list) or \
           not isinstance((c_row.get("trace") or {}).get("dense"), list):
            raise ContentGuardUnknown(f"{qkey}: no trace recorded on this arm")
        detail = _pool_violation(b_row, c_row, pack_ids, depth)
        detail["qid"] = qkey
        checked.append(detail)
        if detail["violation"]:
            violations.append(detail)
    return {"schema": SCHEMA_POOL, "mode": "pool", "pool_depth": depth,
           "lost_golds": len(losses), "checked": checked,
           "violations": violations, "passed": not violations}


# task626 (Tim via coordinator, 2026-09-22, GO-626-standing-catchup
# AMENDMENT 2): "changes in gold that are explainable by valuable news
# articles shouldn't block." The rule, literally:
#
#   a lost gold is a violation ONLY if a PACK row is among its new dense
#   or BM25 candidates (clauses a/c; clause b -- within-pool reorder of
#   pre-existing rows -- is dropped) AND that pack row is not a relevant
#   news article for the query. Relevance is measured, not assumed: the
#   displacing pack row must be on-topic for the query (dense similarity
#   to the query not below the displaced gold's minus the measured
#   margin) and carry a publisher-declared publication date.
#
# THE 2026-09-23 CORRECTION (task 626, day 2026-09-21). The first cut of
# this mode measured relevance on the id occupying the dense leg's FIRST
# divergent slot, and called any other shape a violation
# (`pack_row_not_at_divergence_point`). That is clause (b) wearing a
# different hat: the first divergent slot is usually a PRE-EXISTING row
# that merely reordered, which Amendment 2 says is not a violation at
# all. On 09-21 it stopped the day over `authoritative/de-fresh-015` --
# where the divergence point (dense rank 5) was a pre-existing row and
# the pack's two actual new candidates sat at dense ranks 61 and 94,
# never reaching the fused top-10. The rule names the PACK ROWS among the
# new candidates as the things to test, and the DISPLACED GOLD as what to
# measure them against; this module now does exactly that.
#
# Margin measured 2026-09-22 from the sealed 09-15/09-21/09-13 guard arms'
# ACTUAL live dense cosine scores (query_points against twin aw_e8 with the
# MRL-256 truncated+renormalized query vector -- serving_config.mrl_trunc;
# see the task 626 margin-measuring script (exploration, not shipped), that
# produced these numbers, kept for reproducibility). The adjacent-rank
# score gap AT the position where authoritative/de-fresh-005's dense list
# first diverges was 0.00028 (rank index 9->10) and 0.00024 (13->14); for
# foundation/eu-001, 0.00035/0.00047 nearby -- both near-tied, exactly the
# class of harmless HNSW tail reordering task 627 diagnosed between
# mirrors. The LARGEST single-step adjacent gap observed anywhere in dense
# ranks 5-20 (excluding the top few, naturally more separated) across both
# golds was 0.04537. The margin is set generously above that observed
# ceiling -- 0.05 -- so a genuine near-tie still passes while a displacer
# meaningfully less relevant than what it replaced does not.
NEWS_RELEVANCE_MARGIN = 0.05


class DenseScorer:
    """Live dense-similarity lookup for the news-mode relevance check --
    the sealed arms carry ORDERED id lists per leg, so scoring a specific
    pack row against a specific query needs a live query. Uses
    `query_points` with a `HasIdCondition` filter restricting the search to
    exactly that one point, so Qdrant computes the score through its OWN
    internal pipeline -- the identical number a real top-K search would
    return for that point. A manual retrieve + dot-product was tried first
    and found to disagree with the live search score by ~0.006 (float16
    storage precision, confirmed live against twin aw_e8 2026-09-22) --
    small next to the 0.05 margin, but real enough that "the same number
    the margin was measured with" beats "a close approximation" for a
    comparison the margin is calibrated against.

    ONE instrument for every number in the comparison: the displaced
    gold's score and every candidate displacer's score are read the same
    way, from the same collection, so the margin means what it was
    measured to mean. Caches per (point, query) so a query that loses the
    same gold on multiple checked rows never re-queries the same point.
    """

    PAYLOAD_FIELDS = ["published", "url"]

    def __init__(self, qdrant_client, collection: str):
        self._client = qdrant_client
        self._collection = collection
        self._cache: dict[tuple, dict] = {}

    def describe(self, point_id: str, query_vector) -> dict:
        """`{"score", "published", "url"}` for one point against one query
        vector. `score` is None if the point does not exist in the
        collection (deleted, or a genuinely missing id)."""
        from qdrant_client import models
        key = (point_id, tuple(query_vector) if hasattr(query_vector, "__iter__") else query_vector)
        if key not in self._cache:
            points = self._client.query_points(
                self._collection, query=query_vector, using="dense", limit=1,
                query_filter=models.Filter(
                    must=[models.HasIdCondition(has_id=[point_id])]),
                with_payload=self.PAYLOAD_FIELDS).points
            if not points:
                self._cache[key] = {"score": None, "published": None, "url": None}
            else:
                payload = points[0].payload or {}
                self._cache[key] = {"score": float(points[0].score),
                                    "published": payload.get("published"),
                                    "url": payload.get("url")}
        return self._cache[key]


def _eval_rows(yaml_path) -> list:
    """The query rows of one eval YAML, in the two shapes this lane's eval
    sources use (a `queries:`/`eval:` block, or a bare list)."""
    doc = yaml.safe_load(pathlib.Path(yaml_path).read_text())
    return (doc.get("queries") if isinstance(doc, dict) and "queries" in doc else
            doc.get("eval") if isinstance(doc, dict) and "eval" in doc else
            doc if isinstance(doc, list) else [])


def load_query_vectors(qvecs_path: pathlib.Path, eval_yaml_sources: Mapping[str, pathlib.Path],
                       mrl_dim: int = 256) -> dict[str, list]:
    """{qid: MRL-truncated+renormalized query vector} for every query named
    in `eval_yaml_sources` ({set_name: yaml_path}, e.g. {"authoritative":
    ".../eval_authoritative.yaml"}) that also has a cached embedding in
    `qvecs_path` (keyed by query TEXT, task 617's gate-inputs cache).
    Silently skips a query with no cached vector -- the caller
    (`evaluate_news`) raises `ContentGuardUnknown` if a LOST gold's own
    qid is missing from the result, which is the only case that matters.
    """
    from cytir.core.serving_config import mrl_trunc  # local: pulls in numpy
    qvecs = json.loads(qvecs_path.read_text())
    out: dict[str, list] = {}
    for set_name, yaml_path in eval_yaml_sources.items():
        for row in _eval_rows(yaml_path):
            qid, text = row.get("id"), row.get("query")
            if not qid or not text or text not in qvecs:
                continue
            out[f"{set_name}/{qid}"] = mrl_trunc([qvecs[text]], mrl_dim)[0]
    return out


def load_query_texts(eval_yaml_sources: Mapping[str, pathlib.Path]) -> dict[str, str]:
    """{qid: query TEXT} over the same sources -- Amendment 2 asks for the
    QUERY in every recorded displacement, and a qid alone does not tell a
    human what was being asked."""
    out: dict[str, str] = {}
    for set_name, yaml_path in eval_yaml_sources.items():
        for row in _eval_rows(yaml_path):
            qid, text = row.get("id"), row.get("query")
            if qid and text:
                out[f"{set_name}/{qid}"] = text
    return out


def _new_pack_rows(baseline_row: Mapping, candidate_row: Mapping,
                   pack_ids: set) -> list[dict]:
    """Every PACK row newly present among this gold's candidate lists --
    the exact population Amendment 2's clauses (a)/(c) name. Ordered by
    dense position then BM25 position, de-duplicated across the two legs,
    each carrying where it entered (so the receipt can say so)."""
    b_dense = baseline_row.get("trace", {}).get("dense") or []
    c_dense = candidate_row.get("trace", {}).get("dense") or []
    b_bm25 = baseline_row.get("trace", {}).get("bm25") or []
    c_bm25 = candidate_row.get("trace", {}).get("bm25") or []
    before = set(b_dense), set(b_bm25)
    rows: dict[str, dict] = {}
    for pos, pid in enumerate(c_dense):
        if pid in pack_ids and pid not in before[0]:
            rows.setdefault(pid, {"id": pid, "dense_position": None,
                                  "bm25_position": None})["dense_position"] = pos
    for pos, pid in enumerate(c_bm25):
        if pid in pack_ids and pid not in before[1]:
            rows.setdefault(pid, {"id": pid, "dense_position": None,
                                  "bm25_position": None})["bm25_position"] = pos
    return sorted(rows.values(),
                  key=lambda r: (r["dense_position"] if r["dense_position"] is not None
                                 else 10 ** 6,
                                 r["bm25_position"] if r["bm25_position"] is not None
                                 else 10 ** 6))


def _displaced_gold_ids(baseline_row: Mapping) -> tuple[list, str | None]:
    """(candidate point ids, url) for the gold hit that occupied
    `before_rank` in the BASELINE fused list -- the thing the rule calls
    "the displaced gold". A page can be present as several chunks; the
    caller scores them all and takes the best, which makes the relevance
    floor as HIGH (i.e. as hard on the displacer) as the evidence allows.
    """
    rank = baseline_row.get("rank")
    urls = baseline_row.get("urls") or []
    if not isinstance(rank, int) or rank < 1 or rank > len(urls):
        return [], None
    gold_url = urls[rank - 1]
    by_id = (baseline_row.get("trace") or {}).get("urls") or {}
    return [pid for pid, url in by_id.items() if url == gold_url], gold_url


def _news_violation(baseline_row: Mapping, candidate_row: Mapping, pack_ids: set,
                    scorer: "DenseScorer", query_vector) -> dict:
    """One lost gold's news-mode disposition, per Amendment 2.

    Clause (b) is DROPPED: a dense/BM25 difference with no pack row in it
    is never a violation, at any position. When pack rows ARE newly
    present among the candidate lists, EVERY one of them is measured
    against the displaced gold: it must score within
    `NEWS_RELEVANCE_MARGIN` of the gold's own dense similarity to the
    query and carry a publisher-declared `published` date. All of them
    relevant and dated -> EXPLAINED news displacement, not a violation.
    Any one of them off-topic, undated or unscoreable -> violation,
    conservatively, for a human to look at.
    """
    b_dense = baseline_row.get("trace", {}).get("dense") or []
    c_dense = candidate_row.get("trace", {}).get("dense") or []
    b_bm25 = baseline_row.get("trace", {}).get("bm25") or []
    c_bm25 = candidate_row.get("trace", {}).get("bm25") or []
    new_dense = set(c_dense) - set(b_dense)
    new_bm25 = set(c_bm25) - set(b_bm25)
    displacers = _new_pack_rows(baseline_row, candidate_row, pack_ids)
    base = {"before_rank": baseline_row.get("rank"),
            "after_rank": candidate_row.get("rank"),
            "first_dense_diff_position": _first_diff_position(b_dense, c_dense),
            "new_dense_ids": len(new_dense),
            "dense_pack_row_count": sum(1 for r in displacers
                                        if r["dense_position"] is not None),
            "new_bm25_ids": len(new_bm25),
            "bm25_pack_row_count": sum(1 for r in displacers
                                       if r["bm25_position"] is not None),
            "margin": NEWS_RELEVANCE_MARGIN}
    if not displacers:
        base.update(violation=False, explained=None, reason="no_pack_row_involved",
                    displacements=[])
        return base

    gold_ids, gold_url = _displaced_gold_ids(baseline_row)
    gold_scored = [(scorer.describe(pid, query_vector)["score"], pid) for pid in gold_ids]
    gold_scored = [(score, pid) for score, pid in gold_scored if score is not None]
    if not gold_scored:
        base.update(violation=True, explained=False, displacements=[],
                    gold={"url": gold_url, "id": None, "score": None,
                          "rank": baseline_row.get("rank")},
                    reason=("displaced_gold_unresolvable" if not gold_ids
                            else "displaced_gold_unscoreable"))
        return base
    gold_score, gold_id = max(gold_scored)
    floor = gold_score - NEWS_RELEVANCE_MARGIN
    base["gold"] = {"url": gold_url, "id": gold_id, "score": gold_score,
                    "rank": baseline_row.get("rank")}
    base["relevance_floor"] = floor

    displacements = []
    for row in displacers:
        found = scorer.describe(row["id"], query_vector)
        row = dict(row, score=found["score"], published=found["published"],
                   url=found["url"])
        if found["score"] is None:
            row.update(relevant=False, reason="displacer_unscoreable")
        elif found["published"] is None:
            row.update(relevant=False, reason="displacer_has_no_publication_date")
        elif found["score"] < floor:
            row.update(relevant=False, reason="displacer_below_relevance_margin")
        else:
            row.update(relevant=True, reason="relevant_dated_news_article")
        displacements.append(row)
    base["displacements"] = displacements
    offending = [row for row in displacements if not row["relevant"]]
    if offending:
        base.update(violation=True, explained=False,
                    reason=offending[0]["reason"])
    else:
        base.update(violation=False, explained=True,
                    reason="explained_news_displacement")
    return base


def evaluate_news(baseline: Mapping, candidate: Mapping, pack_ids: set,
                  scorer: "DenseScorer", query_vectors: Mapping[str, object],
                  names: Iterable[str] = gc.BAR_SETS,
                  query_texts: Mapping[str, str] | None = None) -> dict:
    """AW_GUARD_MODE=news (task626, Tim via coordinator, 2026-09-22,
    corrected 2026-09-23). `query_vectors` is `{qid: mrl-truncated query
    vector}` (see `load_query_vectors`) -- must cover every lost gold or
    this raises ContentGuardUnknown rather than silently skip the
    relevance check for one. `query_texts` (optional, see
    `load_query_texts`) puts the QUERY itself in every recorded
    displacement, as Amendment 2 asks.
    """
    try:
        losses = daily.loss_rows(baseline, candidate, names)
    except KeyError as exc:
        raise ContentGuardUnknown(
            f"{exc}: absent from this arm -- baseline and candidate must "
            "score the same query composition") from exc
    query_texts = query_texts or {}
    checked, violations = [], []
    for qkey, before_rank, after_rank in losses:
        name, qid = qkey.split("/", 1)
        b_row = (baseline.get("sets") or {}).get(name, {}).get("per_q", {}).get(qid)
        c_row = (candidate.get("sets") or {}).get(name, {}).get("per_q", {}).get(qid)
        if b_row is None or c_row is None:
            raise ContentGuardUnknown(f"{qkey}: absent from this arm")
        if not isinstance((b_row.get("trace") or {}).get("dense"), list) or \
           not isinstance((c_row.get("trace") or {}).get("dense"), list):
            raise ContentGuardUnknown(f"{qkey}: no trace recorded on this arm")
        qvec = query_vectors.get(qkey)
        if qvec is None:
            raise ContentGuardUnknown(
                f"{qkey}: no cached query vector for news-mode scoring -- "
                "cannot evaluate the relevance check for this lost gold")
        detail = _news_violation(b_row, c_row, pack_ids, scorer, qvec)
        detail["qid"] = qkey
        detail["query"] = query_texts.get(qkey)
        checked.append(detail)
        if detail["violation"]:
            violations.append(detail)
    return {"schema": SCHEMA_NEWS, "mode": "news", "margin": NEWS_RELEVANCE_MARGIN,
            "lost_golds": len(losses), "checked": checked,
            "violations": violations, "passed": not violations}


def _render_displacements(row: Mapping) -> list[str]:
    """Amendment 2, condition: "Every displacement is recorded in the
    receipt with query, gold, displacer url, similarity, date" -- and
    printed, so the day's log carries it without opening the receipt."""
    lines = []
    gold = row.get("gold") or {}
    for hit in row.get("displacements") or []:
        score = hit.get("score")
        lines.append(
            f"    displacer={hit.get('url')} "
            f"similarity={'n/a' if score is None else format(score, '.4f')} "
            f"published={hit.get('published')} "
            f"dense_pos={hit.get('dense_position')} bm25_pos={hit.get('bm25_position')} "
            f"-> {hit.get('reason')}")
    if gold:
        lines.insert(0, "    gold={url} (rank {rank}, similarity {score})".format(
            url=gold.get("url"), rank=gold.get("rank"),
            score=("n/a" if gold.get("score") is None
                   else format(gold["score"], ".4f"))))
    return lines


def render(receipt: Mapping) -> list[str]:
    mode = receipt.get("mode", "strict")
    lines = [f"CONTENT GUARD ({mode}"
            + (f", pool_depth={receipt['pool_depth']}" if mode == "pool" else "")
            + (f", margin={receipt['margin']}" if mode == "news" else "")
            + f"): {'PASS' if receipt['passed'] else 'STOP'} -- "
            f"{receipt['lost_golds']} lost gold(s), "
            f"{len(receipt['violations'])} violation(s)"]
    explained = [row for row in receipt.get("checked", [])
                if row.get("explained") is True]
    for row in explained:
        lines.append(
            f"  EXPLAINED (news displacement, not a violation): {row['qid']} "
            f"{row['before_rank']} -> {row['after_rank']} "
            f"query={row.get('query')!r}")
        lines.extend(_render_displacements(row))
    for row in receipt["violations"]:
        if mode == "news":
            lines.append(
                f"  UNEXPLAINED on a lost gold: {row['qid']} "
                f"{row['before_rank']} -> {row['after_rank']} "
                f"(reason={row['reason']}) query={row.get('query')!r}")
            lines.extend(_render_displacements(row))
        elif mode == "pool":
            lines.append(
                f"  ATTRIBUTABLE TO PACK on a lost gold: {row['qid']} "
                f"{row['before_rank']} -> {row['after_rank']} "
                f"(reason={row['reason']}, first_dense_diff="
                f"{row['first_dense_diff_position']}, "
                f"dense_pack_rows={row['dense_pack_row_count']}, "
                f"bm25_pack_rows={row['bm25_pack_row_count']})")
        else:
            lines.append(f"  DENSE DIFFERS on a lost gold: {row['qid']} "
                         f"{row['before_rank']} -> {row['after_rank']}")
    return lines


def _pack_ids(artifact: pathlib.Path, tombstone_pointers: pathlib.Path) -> set:
    import gzip
    ids = set()
    with gzip.open(artifact, "rt") as f:
        for line in f:
            ids.add(str(json.loads(line)["id"]))
    tp = json.loads(tombstone_pointers.read_text())
    for item in (tp.get("source_group") or []) + (tp.get("aged_out") or []):
        ids.add(str(item["predecessor_id"]))
    return ids


def main(argv=None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[2])
    ap.add_argument("--baseline", required=True, type=pathlib.Path,
                    help="the concurrent production baseline arm")
    ap.add_argument("--candidate", required=True, type=pathlib.Path,
                    help="the post-write twin candidate arm")
    ap.add_argument("--mode", choices=("strict", "pool", "news"),
                    default=os.environ.get(GUARD_MODE_ENV, "strict"),
                    help=f"defaults to ${GUARD_MODE_ENV} or 'strict'")
    ap.add_argument("--pack-artifact", type=pathlib.Path,
                    help="required for --mode pool/news: the pack's own artifact")
    ap.add_argument("--pack-tombstone-pointers", type=pathlib.Path,
                    help="required for --mode pool/news: the pack's tombstone pointers")
    ap.add_argument("--pool-depth", type=int, default=None)
    ap.add_argument("--twin-url", help="required for --mode news")
    ap.add_argument("--twin-collection", help="required for --mode news")
    ap.add_argument("--qvecs", type=pathlib.Path, help="required for --mode news")
    ap.add_argument("--eval-yaml", action="append", default=[],
                    help="required for --mode news: SET_NAME=PATH, repeatable "
                         "(e.g. --eval-yaml authoritative=data/eval_authoritative.yaml)")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args(argv)
    if args.mode in ("pool", "news") and not (args.pack_artifact and args.pack_tombstone_pointers):
        ap.error(f"--mode {args.mode} needs --pack-artifact and --pack-tombstone-pointers")
    if args.mode == "news" and not (args.twin_url and args.twin_collection
                                    and args.qvecs and args.eval_yaml):
        ap.error("--mode news needs --twin-url, --twin-collection, --qvecs, "
                 "and at least one --eval-yaml")
    try:
        baseline = json.loads(args.baseline.read_text())
        candidate = json.loads(args.candidate.read_text())
        if args.mode == "pool":
            depth = pool_depth() if args.pool_depth is None else args.pool_depth
            print(f"[content_guard] pool mode, depth={depth} "
                 f"(ADR-035 k-stability; override via ${POOL_DEPTH_ENV})")
            pack_ids = _pack_ids(args.pack_artifact, args.pack_tombstone_pointers)
            receipt = evaluate_pool(baseline, candidate, pack_ids, depth=depth)
        elif args.mode == "news":
            from cytir.core.qdrant import qdrant_connect
            print(f"[content_guard] news mode, margin={NEWS_RELEVANCE_MARGIN}")
            pack_ids = _pack_ids(args.pack_artifact, args.pack_tombstone_pointers)
            eval_yaml_sources = dict(pair.split("=", 1) for pair in args.eval_yaml)
            query_vectors = load_query_vectors(args.qvecs, eval_yaml_sources)
            client = qdrant_connect(args.twin_url, timeout=120, check_compatibility=False)
            scorer = DenseScorer(client, args.twin_collection)
            receipt = evaluate_news(baseline, candidate, pack_ids, scorer,
                                    query_vectors,
                                    query_texts=load_query_texts(eval_yaml_sources))
        else:
            receipt = evaluate(baseline, candidate)
    except (OSError, ValueError, ContentGuardUnknown) as exc:
        print(f"UNKNOWN: {exc}", file=sys.stderr)
        return 2
    if args.out:
        args.out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    for line in render(receipt):
        print(line)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(None, prog="python -m cytir.gates.content_guard"))
