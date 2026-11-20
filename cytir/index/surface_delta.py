"""Apply a re-surfaced pack family to a collection as an ADR-083 vector delta.

Same point ids, vectors only: the dense vector recomputed from the
document-first surface, and the `bm25` posting recomputed from the family's own
sparse channel through the project's OWN tokeniser
(`build_aw_b1._retokenize` -> `aw_b1_projection.saturated_bm25`), never a second
implementation of it (engineering doctrine 1/6).

Because `canonical_text` never moved, no id can move either
(`aw_common.mint_passage_point_id` keys on it), so `points_count` is invariant
by construction -- and it is asserted before and after anyway, because "by
construction" is how silent divergence gets shipped.

    cyt index surface-delta \
        --stage <resurfaced stage> --vectors <vector dir> \
        --qdrant <twin-url> --collection <twin-collection> \
        --mrl 256 --report <json> [--apply] [--limit N]

Without `--apply` it is a DRY RUN: it reads, recomputes, compares against what
the collection holds today, and writes the report -- no write of any kind.

TASK 546 -- the join-rate guard and the two-surface detector.

Task 538 found that a family's delta ids are minted from `canonical_text`
(`aw_common.mint_passage_point_id`); when the pack's canonical text drifts
between versions, the delta's ids address points the collection does not
hold, while the collection's REAL rows for the same URLs sit under other,
untouched ids. `context7` shipped this way: 53,822 of 84,961 staged rows
addressed nothing, and this module's own `points_count` conservation check
could not see it -- a family can gain a second embedding surface without
`points_count` moving by even one. `rfc_editor` showed a milder 16.5% version
of the same defect the same way.

Two guards close that gap, both against the collection this run targets,
never against a payload-unindexed scan:

  1. A JOIN-RATE GUARD, run before any write. It joins the delta's own ids
     against the live collection (batched retrieve, the full population) and,
     separately, the delta's URLs against the collection's rows for those
     URLs (paged `scroll` on the indexed `url` field -- `domain`/`url`/
     `source_type`/... are the only keyword-indexed fields; `url` is the one
     that lets a page's whole point set be found without a full scan). The
     apply is refused -- non-zero exit, report still written -- unless the id
     join is 100% of the delta, or the caller passes `--allow-partial-join
     <reason>`, which lands verbatim in the report. Both rates are reported
     unconditionally, refused or not.

  2. A TWO-SURFACE DETECTOR, run after the write, when the caller supplies
     `--arm-a-vectors` (the family's prior surface, staged in the exact same
     shard/row-id layout as `--vectors`, over the exact same ids in the exact
     same order -- e.g. task 538's arm-classification pair `arma/<family>` /
     `armb/<family>`). It samples >=200 written ids UNIFORMLY (never 1
     batch in 20 -- that sampling reports `0/0` for every family too small to
     reach a 20th batch, which reads as "nothing moved" and means "nothing
     was looked at"), reads the STORED vector back from the collection, and
     classifies it against both arms with L2 RENORMALISATION FIRST: the
     stored vector is only unit-length to ~1e-5, and comparing it raw is the
     exact mistake `which_arm.py`'s first run made -- it put a third of every
     family, including one that had never been touched, into a spurious
     "neither" bucket. Renormalised, a row is `armA` (the write did not take),
     `armB` (correctly applied), both at once folded into `armB` when the two
     arms are themselves indistinguishable (a genuine no-op row carries no
     evidence either way), or `neither` (matches nothing -- a real anomaly).
     `conservation_ok` is refused if any sampled row is `neither`, or if the
     family holds evidence of BOTH arms at once (a split family, task 538's
     actual failure mode).

Both guards report always; neither guard fires, and `conservation_ok` keeps
its old (`points_count`-only) meaning, for any invocation that does not pass
`--arm-a-vectors` and whose join is 100% -- which is every existing caller
until they choose otherwise (engineering doctrine 9: fix the class, but a
clean delta must still see the byte-for-byte old behaviour).
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import random
import time
import numpy as np

import cytir.core.projection as PROJ  # noqa: E402
from cytir.core.common import compose_index_text  # noqa: E402
from cytir.core.qdrant_http import api, points_count  # noqa: E402
from cytir.core import serving_config as SC  # noqa: E402
import cytir.corpus.persisted as PI  # noqa: E402
from cytir.index import prod_guard as PG  # noqa: E402
from cytir.index import twin_write_hold as TWH  # noqa: E402

DIM = 1024
VERSION = "apply-pack-surface-delta.v2"
# Absolute renormalised-cosine threshold for "this stored vector IS this arm".
# Same value which_arm.py uses for arm-vs-arm indistinguishability; reused
# here as the stored-vs-arm match bar because it is the same instrument
# (aw_e8/aw_prod ship dense at 1024-d fp16, sliced to --mrl and renormalised
# for compare -- one truth, one home, per engineering doctrine 1/6).
ARM_MATCH_COS = 0.99999


def iter_shards(stage: str, vectors: str):
    """Yield (tag, texts_path, vectors_path, row_ids_path, manifest_path) for
    every shard.

    The roster is a glob of the vector directory joined to the stage, and each
    shard's row_ids are checked against the staged order before a single vector
    is read: row i of the .f16 IS line i of the texts, and that is the only
    thing binding a vector to an id.
    """
    for man in sorted(glob.glob(os.path.join(vectors, "shard-*.vectors.manifest.json"))):
        tag = os.path.basename(man).removesuffix(".vectors.manifest.json")
        yield (tag, os.path.join(stage, tag + ".texts.jsonl.gz"),
               os.path.join(vectors, tag + ".vectors.f16"),
               os.path.join(vectors, tag + ".row_ids.jsonl"), man)


def selected_shards(stage: str, vectors: str, world: int, rank: int,
                     resume_from: str):
    """`iter_shards`, filtered to the shard-strided slice and resume point
    this invocation owns -- the ONE definition of "which shards does this
    run touch", shared by the join guard's preflight and the apply loop
    (engineering doctrine 1). A rank's guard must measure exactly the rows
    its own apply loop will process, or the two disagree about what "100% of
    the delta" means.
    """
    seen_resume = not resume_from
    for shard_no, shard in enumerate(iter_shards(stage, vectors)):
        tag = shard[0]
        if world > 1 and shard_no % world != rank:
            continue
        if not seen_resume:
            if tag != resume_from:
                continue
            seen_resume = True
        yield shard


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def join_guard(stage: str, vectors: str, qdrant: str, collection: str,
               world: int, rank: int, resume_from: str,
               url_sample: int = 200, seed: int = 546538) -> dict:
    """The preflight: prove the join before trusting the delta (corpus
    doctrine 5). Reads only `row_ids.jsonl` + the texts (never the vectors --
    this runs before a single vector is touched, and must be cheap).

    Forward: every id in this run's shard slice, batched against the live
    collection -- the FULL population, not a sample, because the refusal bar
    is "100% of the delta", not an estimate of it.

    Reverse: a uniform sample of this slice's distinct URLs, each resolved to
    every point the collection holds for it via the indexed `url` field
    (`points/scroll`, paged to exhaustion -- a `limit`-bounded single page
    undercounts any URL with more chunks than the page, which is exactly the
    bug the reverse join in task 538's family-join script had before
    family_join_reverse.py fixed it). This is the direction that catches
    "the collection's real rows for this URL sit under ids the delta never
    mentions" even on a family whose OWN ids all happen to be present.
    """
    all_ids: list[str] = []
    by_url: dict[str, list[str]] = collections.defaultdict(list)
    for tag, texts_p, _vec_p, rid_p, _man_p in selected_shards(
            stage, vectors, world, rank, resume_from):
        rid = [json.loads(l)["point_id"] for l in open(rid_p)]
        all_ids.extend(rid)
        with gzip.open(texts_p, "rt", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                rec = json.loads(line)
                if rec["point_id"] != rid[i]:
                    raise SystemExit(f"{tag}: staged order does not match row_ids")
                by_url[rec.get("url") or ""].append(rec["point_id"])

    found: set[str] = set()
    for batch in _batched(all_ids, 1024):
        got = api(qdrant, f"/collections/{collection}/points",
                  {"ids": batch, "with_payload": False, "with_vector": False})["result"]
        found |= {str(r["id"]) for r in got}
    ids_total, ids_found = len(all_ids), len(found)

    urls = sorted(u for u in by_url if u)
    rng = random.Random(seed)
    usamp = rng.sample(urls, min(url_sample, len(urls))) if urls else []
    stg = coll = ov = 0
    for u in usamp:
        cids: set[str] = set()
        offset = None
        while True:
            body = {"filter": {"must": [{"key": "url", "match": {"value": u}}]},
                    "limit": 256, "with_payload": False}
            if offset is not None:
                body["offset"] = offset
            PG.check(qdrant, "POST", f"/collections/{collection}/points/scroll",
                     collection=collection)  # a prod scroll is refused (task 1869)
            res = api(qdrant, f"/collections/{collection}/points/scroll",
                      body)["result"]
            cids |= {str(p["id"]) for p in res["points"]}
            offset = res.get("next_page_offset")
            if offset is None:
                break
        sids = set(by_url[u])
        stg += len(sids); coll += len(cids); ov += len(sids & cids)

    return {
        "ids_total": ids_total, "ids_found": ids_found,
        "id_join_rate": (ids_found / ids_total) if ids_total else 1.0,
        "url_sample": len(usamp), "url_population": len(urls),
        "url_join_staged_points": stg, "url_join_collection_points": coll,
        "url_join_overlap": ov,
        "url_join_collection_only": coll - ov, "url_join_staged_only": stg - ov,
        "url_join_rate": (ov / stg) if stg else 1.0,
    }


def load_full_arm(vec_dir: str, mrl: int) -> tuple[np.ndarray, list[str]]:
    """Load one arm's full mrl-dim, L2-renormalised dense matrix + point ids,
    in shard order -- the exact transform this module writes to the
    collection (`sub = vec[:, :mrl]; sub /= norm`), so a stored vector can be
    compared to it without the collection's ~1e-5 write slack producing a
    false non-match. Mirrors task 538's arm loader (one truth,
    one home for this transform)."""
    mats: list[np.ndarray] = []
    ids: list[str] = []
    for man_p in sorted(glob.glob(os.path.join(vec_dir, "shard-*.vectors.manifest.json"))):
        man = json.load(open(man_p))
        tag = os.path.basename(man_p).removesuffix(".vectors.manifest.json")
        raw = np.fromfile(os.path.join(vec_dir, tag + ".vectors.f16"), dtype="<f2")
        mats.append(raw.reshape(man["n"], DIM).astype("float32"))
        ids += [json.loads(x)["point_id"]
                for x in open(os.path.join(vec_dir, tag + ".row_ids.jsonl"))]
    m = (np.concatenate(mats, 0) if mats else np.zeros((0, DIM), dtype="float32"))[:, :mrl]
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    return m, ids


def two_surface_check(qdrant: str, collection: str, written_ids: list[str],
                      arm_a_dir: str, arm_b_dir: str, mrl: int,
                      sample_n: int = 200, seed: int = 546538) -> dict:
    """After the write: does the family hold arm A, arm B, both, or neither?

    `written_ids` is THIS run's own written population (never the ids the
    join guard already found missing -- those were never candidates for a
    write, and sampling them would relabel "known-missing" as "neither",
    which is a different finding this module already reports separately).

    A row where arm A and arm B are themselves indistinguishable carries no
    evidence and is counted toward `armB` (the applier's own intended target)
    rather than manufacturing a false split; a row that matches BOTH arms
    while they disagree, or matches NEITHER, is real evidence and is what the
    refusal below exists for.
    """
    a_vecs, a_ids = load_full_arm(arm_a_dir, mrl)
    b_vecs, b_ids = load_full_arm(arm_b_dir, mrl)
    if a_ids != b_ids:
        raise SystemExit(
            "--arm-a-vectors and --vectors must stage the SAME ids in the "
            "SAME order (which_arm.py's own precondition) -- they do not, so "
            "no stored-vs-arm comparison can be trusted")
    idx = {pid: i for i, pid in enumerate(a_ids)}
    universe = [pid for pid in written_ids if pid in idx]
    rng = random.Random(seed)
    samp = rng.sample(universe, min(sample_n, len(universe))) if universe else []

    found = matches_a = matches_b = indistinguishable = neither = 0
    for batch in _batched(samp, 300):
        got = api(qdrant, f"/collections/{collection}/points",
                  {"ids": batch, "with_vector": ["dense"],
                   "with_payload": False})["result"]
        for r in got:
            v = np.asarray((r.get("vector") or {}).get("dense"), dtype="float32")
            if v.size != mrl:
                continue
            found += 1
            j = idx[str(r["id"])]
            v = v / np.maximum(np.linalg.norm(v), 1e-12)
            ca = float(v @ a_vecs[j]); cb = float(v @ b_vecs[j])
            is_a, is_b = ca > ARM_MATCH_COS, cb > ARM_MATCH_COS
            if is_a and is_b:
                indistinguishable += 1
            elif is_a:
                matches_a += 1
            elif is_b:
                matches_b += 1
            else:
                neither += 1

    ok = (neither == 0) and not (matches_a > 0 and matches_b > 0)
    return {"skipped": False, "sampled": len(samp), "found": found,
            "matches_armA": matches_a, "matches_armB": matches_b,
            "arms_indistinguishable": indistinguishable, "neither": neither,
            "ok": ok}


def sparse_for(rec: dict, payload: dict, field: str = "auto") -> tuple[list, list]:
    """The posting the binder would build, from the family's OWN sparse channel.

    `assembly_binder.py` (task 528a) now hands `record_text_channels`'s sparse
    side to the projection; this reproduces the same composition -- payload
    title and site_name over the sparse surface, folded and tf-saturated -- so
    the delta's postings and a future rebuild's agree.
    """
    if field == "text":
        # The SHIPPED rule: before task 528a, assembly_binder.py:1389 fed
        # `rec["text"] or rec["dense_text"]` to BOTH legs. Reproducing it is
        # what makes a rollback of the posting exact rather than approximate.
        surface = rec.get("text") or rec.get("dense_text") or ""
    else:
        surface = rec.get("sparse_text") or rec.get("text") or ""
    rec_for_compose = {"text": surface, "title": payload.get("title"),
                       "site_name": payload.get("site_name"),
                       "source_type": payload.get("source_type")}
    composed = compose_index_text(rec_for_compose)
    return PROJ.saturated_bm25(composed, payload.get("lang"))


def main(argv=None, prog: str = "cyt index surface-delta") -> int:
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--qdrant", required=True, help="target Qdrant URL (no default: the twin is never an implicit target)")
    ap.add_argument("--collection", required=True, help="target collection (no default)")
    ap.add_argument("--mrl", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--compare-every", type=int, default=20,
                    help="fetch the OLD vector on 1 batch in N to measure how "
                         "much actually moved; fetching it on every batch "
                         "doubles the payload of 7M round trips for a "
                         "statistic, so it is sampled and the sample size is "
                         "reported with the number")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds between batches; the throttle that keeps a "
                         "7M-point in-place rewrite from becoming an "
                         "optimization storm (memory: "
                         "aw-heavy-rewrite-degrades-prod)")
    ap.add_argument("--report", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--require-persisted", default="l1",
                    choices=sorted(PI.REQUIRE_LEVELS),
                    help="ADR-105 bar for the staged pages: 'l1' (a persisted "
                         "L1 markdown record) or 'l1b' (additionally admits "
                         "pages whose text survives only as L2 passages, the "
                         "ADR-093 route). No level admits a page with no "
                         "stored text at all.")
    ap.add_argument("--no-sparse", action="store_true")
    ap.add_argument("--no-dense", action="store_true",
                    help="write ONLY the bm25 posting; the dense vector is left "
                         "exactly as it is. This is what makes a sparse-leg "
                         "rollback control possible without disturbing the "
                         "dense leg the same gate is measuring")
    ap.add_argument("--sparse-field", default="auto", choices=["auto", "text"],
                    help="'text' reproduces the pre-528a binder rule, one "
                         "surface for both legs -- the shipped posting")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume-from", default="",
                    help="shard tag to resume at, inclusive")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1,
                    help="shard-strided parallelism, same convention as "
                         "the shard embedder: worker r takes "
                         "shards r, r+world, r+2*world ... Each worker writes "
                         "its own report; the reports are summed by "
                         "task 528b's report summer so a partial "
                         "fleet can never be read as a whole one")
    ap.add_argument("--allow-partial-join", default="", metavar="REASON",
                    help="proceed even though the join-rate guard did not "
                         "find every one of the delta's ids in the "
                         "collection; REASON is required and is written "
                         "into the report verbatim (task 546 -- this is "
                         "the escape hatch, used deliberately, never a "
                         "default)")
    ap.add_argument("--join-url-sample", type=int, default=200,
                    help="distinct URLs sampled for the reverse join-rate "
                         "check (points/scroll on the indexed `url` field, "
                         "paged to exhaustion per URL)")
    ap.add_argument("--arm-a-vectors", default="",
                    help="the family's PRIOR surface, staged in the exact "
                         "same shard/row-id layout as --vectors over the "
                         "exact same ids in the exact same order (e.g. "
                         "task 538's arma/<family>). "
                         "When given, the two-surface detector runs after "
                         "the write; when omitted, it is skipped and "
                         "conservation_ok keeps its old points_count-only "
                         "meaning")
    ap.add_argument("--surface-sample", type=int, default=200,
                    help="ids sampled per family for the two-surface "
                         "detector, uniformly over the written population "
                         "(never 1 batch in 20)")
    ap.add_argument("--sample-seed", type=int, default=546538,
                    help="seed shared by the reverse-join URL sample and the "
                         "two-surface id sample, for a reproducible report")
    args = ap.parse_args(argv)
    if args.apply:
        # task 756 / review 776: no receipt-less twin write while an aw_e10 build holds the twin
        try:
            TWH.refuse_if_held(args.collection, tool="cyt index surface-delta --apply")
        except TWH.TwinWriteHeld as exc:
            raise SystemExit(str(exc)) from exc

    # -- 0. dim guard: before ANY write, before the join-rate guard even
    # (task 563). This module's declared embedding width is `--mrl` -- the
    # width `sub = vec[:, :args.mrl]` actually writes -- sanity-named against
    # the first shard's own manifest (model + raw output width), then
    # reconciled against the target through
    # serving_config.assert_dim_matches_shape, never a second comparison.
    # Reads the target's shape over the SAME raw-HTTP `api()` this module
    # already uses for everything else -- no second client stack, so the
    # existing fake-qdrant test harness covers this guard too. Report is
    # untouched: a matching pack must produce the exact same report this
    # module wrote before task 563.
    first_shard = next(iter(iter_shards(args.stage, args.vectors)), None)
    if first_shard is not None:
        man = json.load(open(first_shard[4]))
        source = (f"--mrl {args.mrl} (pack manifest "
                 f"{os.path.basename(first_shard[4])}: model="
                 f"{man.get('model')!r} raw_dim={man.get('dim')})")
        target_shape = SC.shape_from_json(
            api(args.qdrant, f"/collections/{args.collection}")["result"])
        try:
            SC.assert_dim_matches_shape(target_shape, args.collection,
                                        args.mrl, source=source)
        except RuntimeError as exc:
            raise SystemExit(f"[dim-guard] REFUSED: {exc}") from None

    # -- 0b. persisted-inputs guard: before ANY write (ADR-105, task 581) --
    # This module rewrites the EMBEDDING SURFACE of existing rows from staged
    # text. If that text's page is not on disk, the surface it writes cannot
    # be re-derived by anyone -- including us, on the next rebuild -- and the
    # row joins the 492,211-page population task 579 measured. The roster is
    # this invocation's OWN shards (`selected_shards`), not the whole stage:
    # a rank must be checked over exactly the rows it will write.
    def _staged_pages():
        for _tag, texts_p, _v, _r, _m in selected_shards(
                args.stage, args.vectors, args.world, args.rank,
                args.resume_from):
            with gzip.open(texts_p, "rt", encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    yield {"url": rec.get("url"),
                           "sha": rec.get("content_digest")}  # canonical row

    try:
        pv = PI.assert_pages_persisted(
            _staged_pages(), require=args.require_persisted,
            label=f"surface delta {args.stage} -> {args.collection} "
                  f"(rank {args.rank}/{args.world})")
    except PI.PersistedInputsError as exc:
        raise SystemExit(f"[persisted-inputs] REFUSED: {exc}") from None
    print(f"[persisted-inputs] OK: {pv['pages']:,} page(s), "
          f"{pv['persisted']:,} with L1/L0, {pv['l1b_reconstructible']:,} "
          f"L2-only (require={pv['require']})", flush=True)

    before = points_count(args.qdrant, args.collection)
    stats = {
        "persisted_inputs": pv,
        "version": VERSION, "collection": args.collection,
        "stage": os.path.abspath(args.stage),
        "vectors": os.path.abspath(args.vectors),
        "points_count_before": before, "mrl": args.mrl,
        "rank": args.rank, "world": args.world,
        "apply": bool(args.apply), "no_dense": bool(args.no_dense),
        "sparse_field": args.sparse_field, "shards": [],
        "rows": 0, "missing_in_collection": 0, "dense_changed": 0,
        "sparse_changed": 0, "sparse_weights_only_changed": 0,
        "dense_compared": 0, "sparse_compared": 0, "sparse_terms_before": 0, "sparse_terms_after": 0,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # -- 1. the join-rate guard: before ANY write (task 546) ---------------
    guard = join_guard(args.stage, args.vectors, args.qdrant, args.collection,
                       args.world, args.rank, args.resume_from,
                       url_sample=args.join_url_sample, seed=args.sample_seed)
    guard["allow_partial_join_reason"] = args.allow_partial_join or None
    refused = guard["id_join_rate"] < 1.0 and not args.allow_partial_join
    guard["refused"] = refused
    stats["join_guard"] = guard
    if refused:
        stats["two_surface_check"] = {"skipped": True,
                                      "reason": "refused by the join guard"}
        stats["conservation_ok"] = False
        stats["points_count_after"] = before
        stats["points_count_delta"] = 0
        stats["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(json.dumps({k: v for k, v in stats.items() if k != "shards"},
                         indent=2, sort_keys=True))
        print(f"[join-guard] REFUSED: id_join_rate="
              f"{guard['id_join_rate']:.4f} ({guard['ids_found']:,}/"
              f"{guard['ids_total']:,}) < 100% and no --allow-partial-join "
              f"was given. No write was attempted.", flush=True)
        return 1
    if guard["id_join_rate"] < 1.0:
        print(f"[join-guard] proceeding on --allow-partial-join "
              f"{args.allow_partial_join!r}: id_join_rate="
              f"{guard['id_join_rate']:.4f} ({guard['ids_found']:,}/"
              f"{guard['ids_total']:,})", flush=True)

    written_ids: list[str] = []

    for shard_no, (tag, texts_p, vec_p, rid_p, man_p) in enumerate(
            selected_shards(args.stage, args.vectors, args.world, args.rank,
                            args.resume_from)):
        man = json.load(open(man_p))
        vec = np.fromfile(vec_p, dtype="<f2").reshape(-1, DIM)
        if vec.shape[0] != man["n"]:
            raise SystemExit(f"{tag}: {vec.shape[0]} rows vs manifest {man['n']}")
        rid = [json.loads(l)["point_id"] for l in open(rid_p)]
        recs = []
        with gzip.open(texts_p, "rt", encoding="utf-8") as fh:
            for line in fh:
                recs.append(json.loads(line))
        if [r["point_id"] for r in recs] != rid:
            raise SystemExit(f"{tag}: staged order does not match row_ids")

        sub = vec[:, :args.mrl].astype("float32")
        sub /= np.linalg.norm(sub, axis=1, keepdims=True)

        shard_stat = {"tag": tag, "rows": len(rid), "missing": 0,
                      "dense_changed": 0, "sparse_changed": 0,
                      "sparse_weights_only_changed": 0,
                      "dense_compared": 0, "sparse_compared": 0}
        batch_no = 0
        for start in range(0, len(rid), args.batch):
            batch_no += 1
            ids = rid[start:start + args.batch]
            want_old = (batch_no % max(args.compare_every, 1)) == 0
            got = api(args.qdrant, f"/collections/{args.collection}/points",
                      {"ids": ids, "with_payload": True,
                       "with_vector": (["dense", "bm25"] if want_old
                                       else False)})["result"]
            have = {str(r["id"]): r for r in got}
            updates = []
            for j, pid in enumerate(ids):
                cur = have.get(pid)
                if cur is None:
                    shard_stat["missing"] += 1
                    continue
                payload = cur.get("payload") or {}
                newv = sub[start + j]
                if want_old:
                    shard_stat["dense_compared"] += 1
                    oldv = (cur.get("vector") or {}).get("dense")
                    if oldv is None or not np.allclose(
                            np.asarray(oldv, dtype="float32"), newv, atol=2e-3):
                        shard_stat["dense_changed"] += 1
                vector = {} if args.no_dense else {"dense": newv.tolist()}
                if not args.no_sparse:
                    sidx, sval = sparse_for(recs[start + j], payload,
                                            args.sparse_field)
                    if want_old:
                        old_sparse = (cur.get("vector") or {}).get("bm25") or {}
                        stats["sparse_terms_before"] += len(
                            old_sparse.get("indices") or [])
                        stats["sparse_terms_after"] += len(sidx)
                        shard_stat["sparse_compared"] += 1
                        # TASK 564 -- compare the WEIGHTS, not only the term
                        # set. Two surfaces of the same record can carry
                        # identical tokens at different term frequencies (a
                        # law pack's `sparse_text` repeats the title once per
                        # section, its `text` once), and BM25 stores the
                        # SATURATED weight, which depends on tf and on the
                        # document length. The old index-only comparison
                        # reported `sparse_changed=0/5,120` for a run that
                        # rewrote every weight in the family, and the rewrite
                        # moved 8 of 1,420 held-out ranks. A "nothing changed"
                        # that cannot see a weight change is not a check.
                        old_map = dict(zip(
                            [int(i) for i in (old_sparse.get("indices") or [])],
                            [float(v) for v in (old_sparse.get("values") or [])]))
                        new_map = dict(zip([int(i) for i in sidx],
                                           [float(v) for v in sval]))
                        if set(old_map) != set(new_map):
                            shard_stat["sparse_changed"] += 1
                        elif any(abs(old_map[k] - new_map[k]) > 1e-4
                                 for k in new_map):
                            shard_stat["sparse_changed"] += 1
                            shard_stat["sparse_weights_only_changed"] += 1
                    vector["bm25"] = {"indices": list(sidx),
                                      "values": [float(v) for v in sval]}
                updates.append({"id": pid, "vector": vector})
            updates = [u for u in updates if u["vector"]]
            written_ids.extend(u["id"] for u in updates)
            if args.apply and updates:
                api(args.qdrant,
                    f"/collections/{args.collection}/points/vectors?wait=true",
                    {"points": updates}, method="PUT")
            if args.sleep:
                time.sleep(args.sleep)
            if args.limit and stats["rows"] + start + len(ids) >= args.limit:
                break
        stats["shards"].append(shard_stat)
        stats["rows"] += shard_stat["rows"]
        stats["missing_in_collection"] += shard_stat["missing"]
        stats["dense_changed"] += shard_stat["dense_changed"]
        stats["sparse_changed"] += shard_stat["sparse_changed"]
        stats["sparse_weights_only_changed"] += shard_stat["sparse_weights_only_changed"]
        stats["dense_compared"] += shard_stat["dense_compared"]
        stats["sparse_compared"] += shard_stat["sparse_compared"]
        print(f"[delta] {tag} rows={shard_stat['rows']:,} "
              f"missing={shard_stat['missing']} "
              f"dense_changed={shard_stat['dense_changed']:,}"
              f"/{shard_stat['dense_compared']:,} sampled "
              f"sparse_changed={shard_stat['sparse_changed']:,}"
              f"/{shard_stat['sparse_compared']:,} "
              f"(weights-only {shard_stat['sparse_weights_only_changed']:,})",
              flush=True)
        if args.limit and stats["rows"] >= args.limit:
            break

    after = points_count(args.qdrant, args.collection)
    stats["points_count_after"] = after
    stats["points_count_delta"] = after - before
    conservation_ok = (after == before)

    # -- 2. the two-surface detector: after the write (task 546) -----------
    if args.apply and args.arm_a_vectors and not args.no_dense:
        two_surface = two_surface_check(
            args.qdrant, args.collection, written_ids, args.arm_a_vectors,
            args.vectors, args.mrl, sample_n=args.surface_sample,
            seed=args.sample_seed)
        conservation_ok = conservation_ok and two_surface["ok"]
        if not two_surface["ok"]:
            print(f"[two-surface] REFUSED: matches_armA="
                  f"{two_surface['matches_armA']} matches_armB="
                  f"{two_surface['matches_armB']} neither="
                  f"{two_surface['neither']} of {two_surface['sampled']} "
                  f"sampled -- the family holds evidence of both surfaces, "
                  f"or of neither.", flush=True)
    else:
        reason = ("--no-dense: the detector is dense-vector-only" if
                  args.no_dense else
                  "not applied (dry run)" if not args.apply else
                  "--arm-a-vectors not given")
        two_surface = {"skipped": True, "reason": reason}
    stats["two_surface_check"] = two_surface
    stats["conservation_ok"] = conservation_ok
    stats["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "shards"},
                     indent=2, sort_keys=True))
    return 0 if stats["conservation_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
