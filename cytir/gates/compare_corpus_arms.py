#!/usr/bin/env python3
"""Fail-closed ADR-043 comparator for Task370 policy-off corpus arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cytir.gates import authority as gate  # noqa: E402
from cytir.gates import content_gate_null_arm as null  # noqa: E402
from cytir.gates import leg_arms as legs  # noqa: E402
from cytir.core import serving_config  # noqa: E402

GOLD663 = ("authoritative", "foundation", "legal", "names")
TRIPWIRE = ("legal", "names")
FLOOD = "rel_vs_trust"
PROBES = {"art257-de", "art335c-en", "art257-en"}
PROBE_FIXTURE = ROOT / "data" / "gates" / "acceptance_probes.v1.json"
# DERIVED, never restated. An arm scores every set in `gate_authority.SETS`
# plus `slice_b`, so a set added to the gate can no longer skip the comparator
# silently. Task 394 added `entity_card_golds` to SETS and it reached neither
# this list nor the qvec cache; the omission surfaced only in task 452, when it
# made the gate structurally unable to accept ANY current arm -- first as
# `cache_miss=16`, then, once the cache was repaired, as an "incomplete
# expected gate-set composition" refusal. Deriving both ends kills that class:
# the cache's own coverage is already guarded, because
# `build_qvec_query_source_manifest.py` requires the cache to equal the gate's
# query set exactly.
EXPECTED_SETS = {name for name, _ in gate.SETS} | {"slice_b"}


class EvidenceError(RuntimeError): pass
class ProbeFixtureMissing(EvidenceError): pass


@lru_cache(maxsize=None)
def probe_specs(path=PROBE_FIXTURE):
    """Probe specs by id. Loaded on first call, not at import: the fixture is not in the public cut."""
    path = Path(path)
    if not path.is_file(): raise ProbeFixtureMissing(f"probe fixture missing: {path}")
    return {item["id"]: item for item in json.loads(path.read_text())}


def current_gold_total() -> int:
    """The gold-core total THIS checkout's sets carry -- never a literal.

    Mirrors how `cytir/gates/news_daily.py` derives its own population
    (`cytir/gates/news_daily_slice.py`'s rule over `gate.SETS`,
    counted the way an arm's own `n_gold` is counted -- `gate.load_rows` +
    `gate.gold_of`, over exactly GOLD663): a checked-in count is exactly what
    ADR-107 am.3 broke. `data/eval_authoritative.yaml` lost 24 contaminated
    freshness rows (478 -> 454) and a hardcoded "663" here did not move with
    it, so a battery arm cut from main after that PR was refused as a
    measurement error over a number the gold sets no longer produce.
    """
    paths = dict(gate.SETS)
    total = 0
    for name in GOLD663:
        for row in gate.load_rows(paths[name]):
            gold_urls, gold_domains = gate.gold_of(row)
            if gold_urls or gold_domains:
                total += 1
    return total


def golds_of(name: str, path: str) -> dict:
    """{qid: {urls, domains}} for one gate set, derived the way the ARM derived it.

    The qid rule is `run_corpus_arm.main()`'s, character for character
    (`row["id"] or row["qid"] or f"{name}-{index}"`), because a leg arm scores
    rows an arm already cut and a different key would silently score nothing.
    Golds are re-read from `gate.SETS` rather than taken from the arm, for the
    reason task 607 gives: the population is whatever this checkout's sets carry.
    """
    out = {}
    for index, row in enumerate(gate.load_rows(path)):
        qid = str(row.get("id") or row.get("qid") or f"{name}-{index}")
        gold_urls, gold_domains = gate.gold_of(row)
        if gold_urls or gold_domains:
            out[qid] = {"urls": gold_urls, "domains": gold_domains}
    return out


def sealed_arm_available(arm: dict) -> tuple[bool, str]:
    """Can the BM25 leg be RE-SCORED under the sealed table, or only read live?

    ADR-043 A8 rule 4 wants the BM25 arm run under ADR-109's sealed table, which
    makes it layout-invariant BY CONSTRUCTION. That needs three inputs:
    the query's sparse vector, each candidate's STORED sparse vector, and the
    table. `run_corpus_arm.trace()` carries ids, urls and the live per-leg
    scores -- not the stored vectors -- so the re-score is not derivable from an
    arm cut today, and this function says so instead of quietly reporting the
    LIVE bm25 leg under the sealed arm's name.

    That substitution is the failure this returns a reason for: the live leg is
    exactly the layout-variant reading the sealed arm exists to replace, so
    labelling it "sealed" would make the gate report its own artefact as its own
    control.
    """
    for entry in arm.get("sets", {}).values():
        for row in entry.get("per_q", {}).values():
            trace = row.get("trace") or {}
            if isinstance(trace.get("sparse"), dict) and trace["sparse"]:
                return True, "arms carry stored sparse vectors"
    return False, (
        "arms carry no `trace.sparse` (stored BM25 vectors). "
        "run_corpus_arm.trace() records ids, urls and the LIVE per-leg scores; "
        "re-scoring under a sealed table needs each candidate's stored sparse "
        "vector as well. Until trace() captures them, the BM25 reading below is "
        "the LIVE leg -- layout-variant, and reported as such rather than under "
        "the sealed arm's name.")


def leg_arm_report(before: dict, after: dict, null_doc, idf_path,
                   collect: dict | None = None) -> list[str]:
    """ADR-043 A8 rule 4: read the arms per leg, not only fused.

    Returns printable lines and never raises on a missing optional input: a
    comparator that refused to print the fused verdict because a leg arm was
    unavailable would make the optional mandatory by accident. An UNSCOREABLE leg
    is reported as unscoreable -- never as a leg that lost every gold, which is
    the -639 h@1 `gate_leg_arms.score_leg` exists to refuse.
    """
    lines = ["", "LEG ARMS (ADR-043 A8 rule 4)"]
    paths = dict(gate.SETS)
    golds = {}
    for name in GOLD663:
        golds.update(golds_of(name, paths[name]))

    def pooled(arm):
        merged = {}
        for name in GOLD663:
            merged.update(arm["sets"][name]["per_q"])
        return merged

    table = None
    if idf_path is not None:
        try:
            table = legs.load_idf_table(idf_path)
        except legs.LegArmError as exc:
            lines.append(f"  sealed IDF table REFUSED: {exc}")
    sealed_ok = False
    if collect is not None:
        collect["_sealed_table"] = (
            None if table is None else
            {"source": table["source"], "vocabulary": table["vocabulary"],
             "points_count": table["points_count"]})
    if table is not None:
        lines.append(
            f"  sealed IDF table {Path(table['source']).name}: "
            f"{table['vocabulary']:,} tokens at points_count="
            f"{table['points_count']:,}")
        sealed_ok, why = sealed_arm_available(after)
        if collect is not None:
            collect["_sealed_available"] = sealed_ok
            collect["_sealed_unavailable_why"] = None if sealed_ok else why
        lines.append(f"  sealed BM25 re-score: "
                     f"{'AVAILABLE' if sealed_ok else 'NOT AVAILABLE'}"
                     + ("" if sealed_ok else f" -- {why}"))

    def record(leg_name, payload):
        """Machine-readable half of the same reading, for --json-out.

        A driver that adjudicates on these legs must not have to parse the prose
        above: the same numbers reach an operator and a program, from one
        computation. A second, separately-derived JSON path would be a second
        instrument.
        """
        if collect is not None:
            collect[leg_name] = payload

    def report(base, cand, label, leg_name):
        """One leg's reading, with the null-arm split. Shared by all three legs."""
        if base.get("status") == "UNSCOREABLE" or cand.get("status") == "UNSCOREABLE":
            side = base if base.get("status") == "UNSCOREABLE" else cand
            record(leg_name, {"label": label, "status": "UNSCOREABLE",
                              "why": side.get("why")})
            return [f"  {label}: UNSCOREABLE -- {side.get('why')}"]
        delta = legs.compare_leg(base, cand)
        moved_by_null = set()
        out = []
        if null_doc is not None:
            try:
                null_leg = (legs.score_sealed_leg(pooled(null_doc), golds, table)
                            if leg_name == legs.SEALED_LEG
                            else legs.score_leg(pooled(null_doc), leg_name, golds))
                if null_leg.get("status") != "UNSCOREABLE":
                    moved_by_null = {row["qid"] for row
                                     in legs.compare_leg(base, null_leg)["moved"]}
                else:
                    out.append(f"  {label}: null arm UNSCOREABLE -- "
                               f"{null_leg.get('why')}")
            except legs.LegArmError as exc:
                out.append(f"  {label}: null arm not scoreable -- {exc}")
        attributable = [r for r in delta["losses"] if r["qid"] not in moved_by_null]
        instrument = [r for r in delta["losses"] if r["qid"] in moved_by_null]
        record(leg_name, {
            "label": label, "status": "OK",
            "h1_before": base["h1"], "h1_after": cand["h1"],
            "h1_delta": delta["h1_delta"],
            "h10_before": base["h10"], "h10_after": cand["h10"],
            "h10_delta": delta["h10_delta"],
            "compared": delta["compared"],
            "losses": delta["losses"], "gains": delta["gains"],
            "attributable": attributable, "instrument": instrument,
            "null_arm_adjudicated": null_doc is not None,
        })
        out.append(
            f"  {label}: h@1 {base['h1']}->{cand['h1']} "
            f"({delta['h1_delta']:+d}), h@10 {base['h10']}->{cand['h10']} "
            f"({delta['h10_delta']:+d}) over {delta['compared']} golds; "
            f"losses={len(delta['losses'])} "
            f"(attributable={len(attributable)}, instrument={len(instrument)})")
        out += [f"    LEG-LOSS {label} {r['qid']} {r['before']}->{r['after']}"
                for r in attributable]
        return out

    for leg in legs.SCORED_LEGS:
        try:
            base = legs.score_leg(pooled(before), leg, golds)
            cand = legs.score_leg(pooled(after), leg, golds)
        except legs.LegArmError as exc:
            lines.append(f"  {leg}: NOT SCOREABLE -- {exc}")
            continue
        # The bm25 leg is labelled LIVE whenever a sealed table is in play, so a
        # reader cannot mistake Qdrant's layout-dependent order for the sealed arm.
        label = leg if leg != "bm25" or table is None else "bm25 (LIVE idf)"
        lines += report(base, cand, label, leg)

    if table is not None and sealed_ok:
        try:
            base = legs.score_sealed_leg(pooled(before), golds, table)
            cand = legs.score_sealed_leg(pooled(after), golds, table)
        except legs.LegArmError as exc:
            lines.append(f"  {legs.SEALED_LEG}: NOT SCOREABLE -- {exc}")
        else:
            lines += report(base, cand, "bm25 (SEALED idf)", legs.SEALED_LEG)
    return lines


def rank_key(rank): return 10**9 if rank is None else rank

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(label, doc, *, bind_serving=False):
    meta = doc.get("task370") or {}
    required = {"target_manifest", "target_manifest_sha256", "corpus_manifest", "corpus_manifest_sha256",
                "cache_sha256", "cache_vectors", "qvec_query_source_manifest_sha256", "qvec_query_source",
                "retrieval_sha256", "qhead_sha256", "pinned_env", "effective_env", "probe_fixture_sha256"}
    if bind_serving:
        # Task 604. A content-pack verdict is about production, so its arms
        # must have measured production's ranker -- and must be able to PROVE
        # which one they measured. Pre-604 arms record no binding at all and
        # are refused here rather than re-interpreted: they scored the
        # `fusion_weights.json` fallback with rescore off, 82 golds away from
        # what websear.ch serves. The standard lives in
        # `cytir/core/serving_config.py` and cannot be softened from inside a run.
        required = required | {"serving_binding"}
    if required - set(meta): raise EvidenceError(f"{label}: missing metadata {sorted(required-set(meta))}")
    if bind_serving:
        problems = serving_config.gate_serving_binding_problems(meta.get("serving_binding"))
        if problems: raise EvidenceError(f"{label}: {'; '.join(problems)}")
    if doc.get("cache_miss") != 0: raise EvidenceError(f"{label}: cache_miss={doc.get('cache_miss')}")
    if meta["probe_fixture_sha256"] != digest(PROBE_FIXTURE): raise EvidenceError(f"{label}: probe fixture hash drift")
    if meta["pinned_env"] != meta["effective_env"]: raise EvidenceError(f"{label}: pinned environment drift")
    if meta["effective_env"].get("AW_COUNTRY_SCOPE_MODE") != "policy_only" or meta["effective_env"].get("AW_QHEAD_COUNTRY_LEGS") != "0":
        raise EvidenceError(f"{label}: not policy_only with country legs disabled")
    target, corpus = meta["target_manifest"], meta["corpus_manifest"]
    for key in ("identity", "qdrant_url", "collection", "status", "points_count"):
        if target.get(key) != corpus.get(key): raise EvidenceError(f"{label}: target/corpus {key} mismatch")
    if target.get("status") != "green" or not isinstance(target.get("points_count"), int) or target["points_count"] <= 0:
        raise EvidenceError(f"{label}: target not green/exact")
    if not re.fullmatch(r"[0-9a-f]{64}", str(corpus.get("supersede_sha256"))) or not re.fullmatch(r"[0-9a-f]{64}", str(corpus.get("source_sha256"))):
        raise EvidenceError(f"{label}: no supersede/source hash")
    source = meta["qvec_query_source"]
    if source.get("cache_sha256") != meta["cache_sha256"] or source.get("query_count") != meta["cache_vectors"]:
        raise EvidenceError(f"{label}: qvec source does not bind cache exact count")
    if not source.get("query_keys_sha256") or not isinstance(source.get("query_sources"), list) or not source["query_sources"]:
        raise EvidenceError(f"{label}: incomplete qvec query-source provenance")
    for item in source["query_sources"]:
        if not isinstance(item, dict) or {"path", "sha256", "rows"} - set(item):
            raise EvidenceError(f"{label}: invalid qvec query-source row")
    expected = set(doc.get("sets", {}))
    if expected != EXPECTED_SETS:
        raise EvidenceError(f"{label}: incomplete expected gate-set composition")
    for name, result in doc["sets"].items():
        if not isinstance(result.get("per_q"), dict): raise EvidenceError(f"{label}/{name}: missing per_q")
        if result.get("n") != len(result["per_q"]): raise EvidenceError(f"{label}/{name}: set count does not match per_q")
        if name in GOLD663:
            if result.get("n_gold") != len(result["per_q"]): raise EvidenceError(f"{label}/{name}: incomplete gold composition")
            ranks=[row.get("rank") for row in result["per_q"].values()]
            if result.get("h1") != sum(rank == 1 for rank in ranks) or result.get("h10") != sum(bool(rank and rank <= 10) for rank in ranks): raise EvidenceError(f"{label}/{name}: metrics do not match per_q")
        if any(not isinstance(v,dict) or not isinstance(v.get("urls"),list) or not v["urls"] or str(v["urls"][0]).startswith("__ERROR__") for v in result["per_q"].values()):
            raise EvidenceError(f"{label}/{name}: scorer error")
    if set(doc.get("probes", {})) != PROBES: raise EvidenceError(f"{label}: probe fixture composition mismatch")
    return meta


def probe_acceptance(doc):
    for probe_id, row in doc["probes"].items():
        if not isinstance(row.get("urls"),list) or not row["urls"] or str(row["urls"][0]).startswith("__ERROR__"): raise EvidenceError(f"probe {probe_id}: scorer error")
        spec, trace = row["spec"], row.get("trace") or {}
        if spec != probe_specs()[probe_id]: raise EvidenceError(f"probe {probe_id}: fixture contract drift")
        if trace.get("fast_path"): raise EvidenceError(f"probe {probe_id}: fast path has no retrieval trace")
        if row.get("rank") is None or row["rank"] > spec["max_rank"]: raise EvidenceError(f"probe {probe_id}: exact anchor rank {row.get('rank')}")
        for leg in spec.get("require_legs", []):
            if spec["point_id"] not in trace.get(leg, []): raise EvidenceError(f"probe {probe_id}: exact point absent from {leg}")


def _document_key(url):
    """host + path of a url with the fragment dropped (the document, not the anchor)."""
    parts = urllib.parse.urlsplit((url or "").strip())
    host = parts.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    return host, urllib.parse.unquote(parts.path).rstrip("/")


def same_document_slots(target_url, urls):
    """Ranks (1-based) in `urls` held by the target's DOCUMENT: same host+path, any fragment."""
    want = _document_key(target_url)
    return [i for i, u in enumerate(urls, 1) if _document_key(u) == want]


def _document_slot_urls(target_url, urls):
    return [urls[i - 1] for i in same_document_slots(target_url, urls)]


ANCHOR_TIES_MARK = "ANCHOR_TIES_JSON "


def print_anchor_ties(ties):
    """Human line per tie, then ONE machine line the apply gate parses into its receipt."""
    for tie in ties:
        print(f"[anchor_tie] probe {tie['probe']}: exact anchor {tie['gold_rank']}->"
              f"{tie['found_rank']} is a same-document anchor swap, tied at rank "
              f"{tie['gold_rank']} (not a loss): {tie['target_url']}")
    if ties:
        print(ANCHOR_TIES_MARK + json.dumps(ties, sort_keys=True))


def anchor_tie(target_url, old_urls, new_urls):
    """Task 898 (T22 follow-up iii): a swap of anchors inside ONE document is a tie, not a loss.

    True when the target's exact-anchor rank worsened but the target's document holds exactly
    the same slots in both readings and the target is still among them: only the anchors
    inside the document were permuted (art257-de 874: `#art_257` and `#art_257_a`, 5 -> 6).
    A document that gained or lost a slot, or lost the target anchor, stays a regression.
    Reporting only: the exact-anchor rank is still what the gate and the daily gate score."""
    old_slots = same_document_slots(target_url, old_urls)
    new_slots = same_document_slots(target_url, new_urls)
    if not old_slots or old_slots != new_slots:
        return False
    # Every slot outside the document must hold the same url at the same rank (review 900):
    # a set comparison alone lets the target fall behind foreign pages while a sibling
    # anchor swaps into its old slot.
    if len(old_urls) != len(new_urls):
        return False
    held = set(old_slots)
    for i, (a, b) in enumerate(zip(old_urls, new_urls), 1):
        if i not in held and gate.norm_url(a) != gate.norm_url(b):
            return False
    return any(gate.norm_url(new_urls[i - 1]) == gate.norm_url(target_url) for i in new_slots)


def probe_non_regression(before, after, allow_anchor_tie=False):
    """Coverage packs preserve Task370 probes; they do not re-admit them.

    The channel-split experiment owns an absolute exact-anchor contract.  A
    later unrelated content pack may run after a manually accepted Task370
    state where one absolute anchor is already absent.  Treating that
    predecessor state as a new pack regression is a profile error.  The
    content bar instead requires the same immutable probes to stay measurable
    and no exact-anchor rank to worsen -- except a same-document anchor swap,
    which is returned as an `anchor_tie` (Task 898) and never raised.

    Returns the list of anchor_tie reports (empty when none)."""
    ties = []
    for probe_id in sorted(PROBES):
        old, new = before["probes"][probe_id], after["probes"][probe_id]
        for label, row in (("baseline", old), ("candidate", new)):
            if row.get("spec") != probe_specs()[probe_id]:
                raise EvidenceError(f"probe {probe_id}: {label} fixture contract drift")
            urls = row.get("urls")
            if (not isinstance(urls, list) or not urls or
                    str(urls[0]).startswith("__ERROR__")):
                raise EvidenceError(f"probe {probe_id}: {label} scorer error")
        if rank_key(new.get("rank")) > rank_key(old.get("rank")):
            if allow_anchor_tie and anchor_tie(old["spec"]["target_url"], old["urls"], new["urls"]):
                ties.append({"probe": probe_id, "kind": "anchor_tie",
                             "gold_rank": old.get("rank"), "found_rank": new.get("rank"),
                             "target_url": old["spec"]["target_url"],
                             "old_anchors": _document_slot_urls(old["spec"]["target_url"], old["urls"]),
                             "new_anchors": _document_slot_urls(old["spec"]["target_url"], new["urls"])})
                continue
            raise EvidenceError(
                f"probe {probe_id}: exact anchor regressed "
                f"{old.get('rank')}->{new.get('rank')}")
    return ties


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument(
        "--profile", default="task370-channel-split",
        choices=("task370-channel-split", "content-pack"),
        help=("named promotable representation pair; scoring and ADR-043 "
              "arithmetic are shared unchanged"))
    ap.add_argument(
        "--anchor-tie", action="store_true",
        help=("Task 898: report a same-document anchor swap in a probe as an "
              "anchor_tie instead of failing the probe guard. Opt-in; the daily "
              "probe passes it, cytir/index/apply.py does not."))
    ap.add_argument("--baseline-representation")
    ap.add_argument("--candidate-representation")
    ap.add_argument(
        "--null-arm", type=Path,
        help=("ADR-043 am.3: a third arm, read on a state holding the same "
              "content as the baseline. Its own losses and net delta are the "
              "instrument's, and are taken off the pack's before the verdict "
              "is read -- am.6 extends that to the LEGAL/NAMES tripwires, "
              "which keep zero tolerance on the attributable ones. "
              "Content-pack profile only."))
    ap.add_argument(
        "--null-basis", type=Path,
        help=("the receipt that proves the null arm's state held the "
              "baseline's content (schema aw.content-gate-null-basis.v1). "
              "Required with --null-arm."))
    ap.add_argument(
        "--idf-table", type=Path,
        help=("ADR-109 R2 / ADR-043 A8 rule 4: a SEALED gate IDF table "
              "(`python -m cytir.index.idf_table seal`). Enables the sealed BM25 arm "
              "where the arms carry stored sparse vectors, and is reported as "
              "unavailable -- never silently as the live leg -- where they do "
              "not."))
    ap.add_argument(
        "--json-out", type=Path,
        help=("write the verdict and the per-leg readings as JSON, for a driver "
              "that must ADJUDICATE on them. Same numbers as the printed report, "
              "from the same computation -- never a second derivation."))
    ap.add_argument(
        "--leg-arms", action="store_true",
        help=("print the per-leg readings (dense-only, BM25-only) alongside the "
              "fused verdict. A8 rule 4 reads three arms; the fused number alone "
              "mixes the pack's effect with the layout channel ADR-109 "
              "describes."))
    args = ap.parse_args()
    try: probe_specs()
    except ProbeFixtureMissing as exc:
        print(f"MEASUREMENT ERROR: {exc} (the public cut ships none)"); return 3
    expected = {
        "task370-channel-split": ("r5b", "task370-channel-split"),
        "content-pack": ("current-b1", "content-pack"),
    }[args.profile]
    baseline_representation = args.baseline_representation or expected[0]
    candidate_representation = args.candidate_representation or expected[1]
    before, after = json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text())
    null_doc = null_basis = None
    bind = args.profile == "content-pack"
    try:
        a = validate("baseline", before, bind_serving=bind)
        b = validate("candidate", after, bind_serving=bind)
        for key in ("cache_sha256", "cache_vectors", "qvec_query_source_manifest_sha256", "qvec_query_source", "retrieval_sha256", "qhead_sha256", "pinned_env", "probe_fixture_sha256", *(("serving_binding",) if bind else ())):
            if a[key] != b[key]: raise EvidenceError(f"arm mismatch: {key}")
        if a["target_manifest"]["identity"] == b["target_manifest"]["identity"]:
            raise EvidenceError("arms do not have separate target identities")
        # Task370 compares two simultaneously materialized scratch targets.
        # A content-pack transaction instead measures one canonical twin at
        # two immutable count-bound states, immediately before and after its
        # write.  Same URL is therefore expected only for that named profile.
        if (args.profile == "task370-channel-split" and
                a["target_manifest"]["qdrant_url"] == b["target_manifest"]["qdrant_url"]):
            raise EvidenceError("arms do not have separate target URLs")
        if a["corpus_manifest"].get("representation") != baseline_representation or b["corpus_manifest"].get("representation") != candidate_representation:
            raise EvidenceError("wrong baseline/candidate corpus representation")
        if set(before["sets"]) != set(after["sets"]): raise EvidenceError("gate-set composition differs")
        for name in before["sets"]:
            if set(before["sets"][name]["per_q"]) != set(after["sets"][name]["per_q"]): raise EvidenceError(f"{name}: query composition differs")
        if args.profile == "task370-channel-split":
            probe_acceptance(after)
        else:
            print_anchor_ties(probe_non_regression(before, after, allow_anchor_tie=args.anchor_tie))
        if args.null_arm or args.null_basis:
            if args.profile != "content-pack":
                raise EvidenceError("a null arm is a content-pack control (ADR-043 am.3)")
            if not (args.null_arm and args.null_basis):
                raise EvidenceError("--null-arm and --null-basis are one control, pass both")
            null_doc = json.loads(args.null_arm.read_text())
            n = validate("null", null_doc)
            null.assert_arm_bound(a, n)
            if set(null_doc["sets"]) != set(before["sets"]):
                raise EvidenceError("null: gate-set composition differs")
            for name in before["sets"]:
                if set(before["sets"][name]["per_q"]) != set(null_doc["sets"][name]["per_q"]):
                    raise EvidenceError(f"null/{name}: query composition differs")
            null_basis = null.validate_basis(
                json.loads(args.null_basis.read_text()),
                collection=a["target_manifest"]["collection"],
                points_count=a["target_manifest"]["points_count"],
                # ADR-043 am.6 rule 3: the control belongs to THIS baseline.
                # Every bound field in `assert_arm_bound` is also equal for a
                # null arm cut in another run on the same collection, and those
                # readings differ (task 616: 32 bar golds, 5 h@1), so without
                # this citation an operator could shop for an excuse.
                baseline_identity=a["target_manifest"]["identity"],
                baseline_arm_sha256=digest(args.baseline))
    except (EvidenceError, null.NullArmError) as exc:
        print(f"MEASUREMENT ERROR: {exc}"); return 3
    h1a = h10a = h1b = h10b = total = 0; trip_losses = []; loss_rows = []
    for name, old in before["sets"].items():
        new = after["sets"][name]
        if name in GOLD663:
            total += old["n_gold"]; h1a += old["h1"]; h10a += old["h10"]; h1b += new["h1"]; h10b += new["h10"]
        for qid, row in old["per_q"].items():
            if rank_key(new["per_q"][qid].get("rank")) > rank_key(row.get("rank")):
                loss_rows.append((f"{name}/{qid}", row.get("rank"), new["per_q"][qid].get("rank")))
                # The tripwire list is the RAW one, and stays raw: am.6
                # adjudicates it against the null arm below (the ranks
                # themselves are never corrected), and an unadjudicated
                # tripwire loss still refuses.
                if name in TRIPWIRE: trip_losses.append(loss_rows[-1])
    current_total = current_gold_total()
    if total != current_total:
        print(f"MEASUREMENT ERROR: arm total {total} ≠ current sets "
              f"{current_total} -- ADR-107 am.3 changed the gold-core count "
              f"this checkout's sets carry; cut a fresh arm against today's "
              f"gold files")
        return 3
    strict_losses = [f"{key} {old}->{new}" for key, old, new in loss_rows]
    null_h1 = null_h10 = None
    if null_doc is not None:
        h1n = sum(null_doc["sets"][name]["h1"] for name in GOLD663)
        h10n = sum(null_doc["sets"][name]["h10"] for name in GOLD663)
        null_h1, null_h10 = h1n - h1a, h10n - h10a
    metrics = {"h@1": null.metric(h1b - h1a, null_h1),
               "h@10": null.metric(h10b - h10a, null_h10)}
    null_moved = null.moved(before["sets"], null_doc["sets"]) if null_doc else set()
    attributable, instrument = null.split_losses(loss_rows, null_moved)
    # ADR-043 am.6. Zero tolerance on the tripwires is unchanged; what the null
    # arm decides is WHOSE loss it is. With no null arm a tripwire loss is
    # unadjudicated and still refuses.
    tripwire = null.tripwire_verdict(trip_losses, null_moved,
                                     adjudicated=null_doc is not None)
    net_ok = all(value["ok"] for value in metrics.values())
    flood_ok = after["sets"][FLOOD]["h1"] >= before["sets"][FLOOD]["h1"] and after["sets"][FLOOD]["h10"] >= before["sets"][FLOOD]["h10"]
    broken = [n for n in GOLD663 if before["sets"][n]["h10"] > 0 and after["sets"][n]["h10"] < before["sets"][n]["h10"] / 2]
    promotable = (baseline_representation, candidate_representation) == expected
    passed = (promotable and net_ok and flood_ok and not broken
              and not tripwire["refuses"])
    print(f"TOTAL-663 h@1 {h1a}->{h1b}; h@10 {h10a}->{h10b}; strict per-gold losses={len(strict_losses)}")
    for loss in strict_losses: print("  RANK-DRIFT", loss)
    if args.profile == "content-pack":
        for line in null.render(
                (null_doc or {}).get("task370", {}).get("target_manifest", {}).get("identity", ""),
                metrics, attributable, instrument, null_basis,
                null.null_ranks(null_doc["sets"]) if null_doc else None,
                tripwire):
            print(line)
    leg_data: dict = {}
    if args.leg_arms or args.idf_table:
        for line in leg_arm_report(before, after, null_doc, args.idf_table,
                                   collect=leg_data):
            print(line)
    print(f"ADR-043 net={net_ok} flood={flood_ok} broken={broken or 'none'} "
          f"legal/names-losses={len(trip_losses)}"
          + (f" (attributable={len(tripwire['attributable'])} "
             f"instrument={len(tripwire['instrument'])})"
             if trip_losses and null_doc is not None else ""))
    unadjudicated = (args.profile == "content-pack" and not passed
                     and null_doc is None and not net_ok)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps({
            "schema": "aw.corpus-arm-comparison.v1",
            "profile": args.profile,
            "verdict": "PASS" if passed else "FAIL",
            "promotable": promotable,
            "total_golds": total,
            "fused": {"h1_before": h1a, "h1_after": h1b,
                      "h10_before": h10a, "h10_after": h10b,
                      "h1_delta": h1b - h1a, "h10_delta": h10b - h10a},
            "adr043": {"net_ok": net_ok, "flood_ok": flood_ok,
                       "broken": broken,
                       "tripwire_refuses": tripwire["refuses"],
                       "tripwire_unadjudicated": tripwire["unadjudicated"]},
            "strict_losses": strict_losses,
            "null_arm": null_doc is not None,
            "legs": {k: v for k, v in leg_data.items()
                     if not k.startswith("_")},
            "sealed_table": leg_data.get("_sealed_table"),
            "sealed_available": leg_data.get("_sealed_available"),
            "sealed_unavailable_why": leg_data.get("_sealed_unavailable_why"),
        }, indent=1, sort_keys=True) + "\n")
        print(f"[json] {args.json_out}")
    print("VERDICT:", "PASS" if passed else
          "FAIL (ADR-043 corpus bar"
          + ("; non-promotable comparison" if not promotable else "")
          + ("; UNADJUDICATED -- no null arm, this net is not yet the pack's"
             if unadjudicated else "")
          + ("; TRIPWIRE UNADJUDICATED -- no null arm, this legal/names loss "
             "is not yet the pack's (am.6)"
             if args.profile == "content-pack" and tripwire["unadjudicated"]
             else "") + ")")
    return 0 if passed else 2


if __name__ == "__main__": raise SystemExit(main())
