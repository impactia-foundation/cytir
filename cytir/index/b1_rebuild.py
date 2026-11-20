"""The b1 builder's full-rebuild run modes (moved from the builder script, task 2114).

`run_full` drives the phases (kv, coverage, base, deferred, staged, finalize, manifest);
`run_base`, `run_deferred`, `run_staged`, `run_finalize` and `run_full_manifest` are the
phase bodies; `_consent_census` summarises the consent cache for the manifest. The helpers
they call (coverage gate, delete set, exclusion accounting, corpus-row lookup, seen ids,
the explicit dense re-embed, the deferred tripwire) and the knobs they read move with them.

The code below is the original block, with these differences: the dense re-embed gets
`load_embedder` from `cytir.embed.model` and `mrl_trunc` from `cytir.core.serving_config`
(the builder's `index_bakeoff` import re-exported both); `run_full` takes the corpus
directory and the resume-command prefix as keywords (the builder's `main` passes the
value that was a module constant and the literal that was in `run_full`).

Every write site here (`_flush`, the slice-collection delete in `run_base`) is reached only
through the builder's `main`, which calls `refuse_cli_write` before it dispatches."""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import cytir.core.ids as ID
from cytir.core import bm25_saturate as SAT
from cytir.consent.build_gate import ConsentGate
from cytir.index.b1_corpus_kv import (
    _canon_key, _kv_get_channels, _kv_get_texts, _kv_key, build_corpus_kv,
    kv_open, load_published_map, load_resolve_conf_map)
from cytir.index.b1_manifest import (
    TWIN_BASE_COUNT, create_collection, finalize_collection, load_state, qc,
    save_state, validate_manifest, write_manifest)
from cytir.index.b1_projection import project_record
from qdrant_client import models as qm

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
TWIN_URL = os.environ.get("AW_B1_TWIN_URL", "http://localhost:6333")
STAGING_URL = os.environ.get("AW_B1_STAGING_URL", "http://localhost:6398")
BUILD_URL = os.environ.get("AW_B1_BUILD_URL", "http://localhost:6390")
BASE_COLL = os.environ.get("AW_B1_BASE_COLL", "aw_v4_sat")

# --- task-319 fail-loud knobs (each paid for by the 318 gate FAIL) -----------
# Deferred-pass tripwire: the deferred/unknown consent class resolving to more
# than this fraction EXCLUDED is an error signature (the 318 build excluded
# 100% of 929,826 deferred points — census under-coverage, not opt-outs).
DEFERRED_EXCL_MAX = float(os.environ.get("AW_B1_DEFERRED_EXCL_MAX", "0.20"))
# Base id-collision tripwire: upserted-vs-final count shortfall beyond this
# fraction means an identity collapse (318: ~1.8M silent collisions, GAP A33).
ID_COLLISION_MAX = float(os.environ.get("AW_B1_ID_COLLISION_MAX", "0.005"))
# Explicit channel-bearing corpus rows deliberately change their dense input.
# Keep this configuration visible and shared with the normal pack-stage path
# (stage_pack_b1 uses the same arctic model, MRL-128, and 512-token ceiling).
DENSE_REEMBED_MODEL = os.environ.get("AW_B1_DENSE_REEMBED_MODEL", "arctic-l-v2")
DENSE_REEMBED_BATCH = int(os.environ.get("AW_B1_DENSE_REEMBED_BATCH", "256"))
DENSE_REEMBED_MAX_SEQ = int(os.environ.get("AW_B1_DENSE_REEMBED_MAX_SEQ", "512"))

# 10 staged packs (:6398).
STAGING_PACKS = [
    "aw_dwave2_cov", "aw_dtier1_cov", "aw_dzefix_cov", "aw_dsitemap_cov",
    "aw_dgithub_cov", "aw_dswissinfo_cov", "aw_dchmedia_cov", "aw_dhn_cov",
    "aw_dretailcards_cov", "aw_dnewscards_cov",
]


# ==========================================================================
# FULL RUN — resumable, checkpointed streaming builder (47.8M base + staged)
# ==========================================================================
# Phases (each idempotent / resumable via STATE):
#   kv       — build the LMDB corpus-text KV (all parquet waves + fedlex.jsonl)
#   coverage — sampled twin coverage -> tok=2 vs tok=1-tail split (GATE report)
#   base     — stream twin WITH vectors, project, upsert to aw_b1 (checkpoint offset)
#   staged   — merge the 10 staging packs (adds/modifies by minted id)
#   finalize — PATCH hnsw m=16 + sparse keep-warm
#   manifest — write + validate MANIFEST.json
BASE_AVGDL_NOTE = SAT.BASE_AVGDL


# --- coverage / tok-split gate (sampled) -----------------------------------
LAW_SOURCE_TYPES = {"official", "gov-law", "gov-court", "case_law", "official-docs"}
# Accent-bearing languages where an un-folded (tok=1) BM25 leg is a REAL recall
# loss. For pure-ASCII English, folded == unfolded byte-for-byte, so EN law in the
# tail is behaviorally equivalent to tok=2 and is tolerated (coordinator ruling,
# hybrid option C). The gate STOPs only on DE/FR/IT Tier-A law in the tail.
ACCENT_LANGS = {"de", "fr", "it"}
# Domains mis-typed source_type=official but NOT statute/court content — excluded
# from the gate's law set (coordinator decision A+B, 2026-07-24). They were never
# law; reclassified out of the accent-law-in-tail STOP check. Documented in the task 306 doc.
NON_LAW_MISTYPED = {"impactia.org", "psychedelos.ch", "fondationcsm.org"}
# Gate tolerance: genuine accent-law stragglers (BVerfG/legifrance/fedlex few-offs
# with no clean dump route) below this fraction of the sample ship tok=1, are
# appended to the tail-refetch list + counted in the MANIFEST exceptions.
ACCENT_LAW_TOLERANCE = 0.0001   # 0.01%


def measure_coverage(twin, kv_path, sample, log):
    env = kv_open(kv_path, readonly=True)
    by_st = defaultdict(lambda: [0, 0])   # source_type -> [have_text, tail]
    law_tail_by_lang = defaultdict(lambda: defaultdict(int))  # st -> lang -> tail
    n = 0
    offset = None
    with env.begin() as txn:
        while n < sample:
            pts, offset = twin.scroll(
                BASE_COLL, limit=5000, offset=offset,
                with_payload=["url", "anchor", "source_type", "lang", "domain"],
                with_vectors=False)
            for p in pts:
                pl = p.payload or {}
                key = _kv_key(_canon_key(pl.get("url"), pl.get("anchor")))
                have = txn.get(key) is not None
                st = pl.get("source_type") or "?"
                by_st[st][0 if have else 1] += 1
                if (not have) and st in LAW_SOURCE_TYPES:
                    dom = (pl.get("domain") or "").lower().lstrip(".")
                    if dom.startswith("www."):
                        dom = dom[4:]
                    if dom in NON_LAW_MISTYPED:
                        by_st["_reclassified_nonlaw"][1] += 1  # not law -> excluded
                    else:
                        law_tail_by_lang[st][(pl.get("lang") or "?").lower()] += 1
                n += 1
            if offset is None or n >= sample:
                break
    env.close()
    tot_have = sum(v[0] for v in by_st.values())
    tot_tail = sum(v[1] for v in by_st.values())
    overall_tail = tot_tail / max(tot_have + tot_tail, 1)
    # accent-bearing law in tail is the STOP trigger; EN law tail is tolerated
    accent_law_tail = {}
    en_law_tail = 0
    for st, langs in law_tail_by_lang.items():
        for lang, c in langs.items():
            if lang in ACCENT_LANGS:
                accent_law_tail[f"{st}:{lang}"] = accent_law_tail.get(f"{st}:{lang}", 0) + c
            else:
                en_law_tail += c
    log(f"[coverage] sample={n:,}  tok2(have_text)={tot_have:,}  "
        f"tok1(tail)={tot_tail:,}  overall_tail={overall_tail:.3%}")
    for st in sorted(by_st, key=lambda k: -sum(by_st[k])):
        h, t = by_st[st]
        log(f"  {st:16} have={h:>8,} tail={t:>7,} tail%={t/max(h+t,1):.2%}")
    log(f"  ACCENT-LAW tail (de/fr/it, STOP trigger): {accent_law_tail}")
    log(f"  EN-law tail (tolerated, tok=1==tok=2 for ASCII): {en_law_tail:,}")
    return {"sample": n, "overall_tail": round(overall_tail, 4),
            "accent_law_in_tail": accent_law_tail, "en_law_tail": en_law_tail,
            "law_tail_by_lang": {k: dict(v) for k, v in law_tail_by_lang.items()}}


# --- base projection / assembly (streaming, checkpointed) ------------------
def load_delete_set(path, log):
    ids = set()
    if path:
        # fail loud (task 319 / preflight A40): a missing or empty delete-id
        # list must not silently skip the Romansh eviction — the build would
        # ship 26k records the scope decision evicted, discovered only at gate.
        if not os.path.exists(path):
            raise SystemExit(f"[FATAL] --romansh delete-id list not found: {path}")
        for ln in open(path):
            ln = ln.strip()
            if ln:
                ids.add(ln)
        if not ids:
            raise SystemExit(f"[FATAL] --romansh delete-id list is EMPTY: {path}")
    log(f"[romansh] loaded {len(ids):,} delete ids")
    return ids


def _account_exclusion(st: dict, reason: str) -> str:
    """Classify + count ONE excluded record into STATE, truthfully (task 319:
    the 318 build's STATE said consent_excluded: 0 while 1.79M were excluded).
    Returns the class: 'deferred' | 'declined' | 'undated_gov' | 'drop'."""
    if reason.startswith("defer"):
        st["deferred_count"] = st.get("deferred_count", 0) + 1
        return "deferred"
    if reason.startswith("undated_gov"):
        # ch-media admin.ch row with no published date (task 321 ruling,
        # preflight T8/M21) — mislabeled gov press release, not news.
        st["staged_excluded_undated_gov"] = \
            st.get("staged_excluded_undated_gov", 0) + 1
        return "undated_gov"
    if "declined" in reason:
        st["consent_declined"] = st.get("consent_declined", 0) + 1
        st["consent_excluded"] = st.get("consent_excluded", 0) + 1
        return "declined"
    st["drops"] = st.get("drops", 0) + 1
    return "drop"


def _corpus_row_for(kv_txn, url, anchor):
    key = _kv_key(_canon_key(url, anchor))
    texts = _kv_get_texts(kv_txn, key)
    channels = _kv_get_channels(kv_txn, key)
    if not texts and not channels:
        return None
    return {"texts": texts, **(channels or {})}


def load_seen_ids(build, collection, log) -> set:
    """Task 325: rebuild the in-build minted-id set from the target collection
    (id-only scroll). Needed on RESUME — the collision-demotion chain in
    project_record must know every id already upserted. Fresh builds return
    an empty set instantly (empty collection)."""
    seen: set = set()
    if not build.collection_exists(collection):
        return seen
    offset = None
    while True:
        pts, offset = build.scroll(collection, limit=10000, offset=offset,
                                   with_payload=False, with_vectors=False)
        if not pts:
            break
        for p in pts:
            seen.add(str(p.id))
        if len(seen) % 5_000_000 < 10000:
            log(f"[seen-ids] loaded {len(seen):,}")
        if offset is None:
            break
    log(f"[seen-ids] {len(seen):,} existing ids loaded from {collection}")
    return seen


def run_base(build, twin, kv_path, gate, pub_map, conf_map, romansh, st,
             out_dir, args, log, resume_cmd, seen_ids=None):
    env = kv_open(kv_path, readonly=True)
    if not build.collection_exists(args.collection):
        create_collection(build, args.collection, log=log)
    elif st.get("base_processed", 0) == 0:
        # fresh STATE but the target collection already holds points (e.g. the
        # failed build's artifact): starting over would double-upsert into a
        # poisoned collection. Fail loud (task 319); delete/rename it first,
        # or set AW_B1_REUSE_COLLECTION=1 to resume deliberately.
        existing = build.count(args.collection, exact=True).count
        if existing > 0 and not os.environ.get("AW_B1_REUSE_COLLECTION"):
            raise SystemExit(
                f"[FATAL] collection {args.collection!r} already holds "
                f"{existing:,} points but STATE says base_processed=0 — "
                f"refusing to build into a pre-existing artifact. Delete or "
                f"rename it (or set AW_B1_REUSE_COLLECTION=1 deliberately).")
    stats = defaultdict(int)
    offset = st.get("base_offset")
    batch = []
    processed = st.get("base_processed", 0)
    deferred_fh = open(os.path.join(out_dir, "consent_deferred_queue.txt"), "a")
    # tail-refetch list: no-text (tok=1) base points — incl. the genuine accent-law
    # stragglers the gate tolerated — for a future delta to fold (coordinator A+B).
    tail_fh = open(os.path.join(out_dir, "aw_b1.tail_refetch.jsonl"), "a")
    if gate.mode == "pipelined":
        gate.reload()
    with env.begin() as txn:
        while True:
            pts, offset = twin.scroll(BASE_COLL, limit=args.scroll_batch,
                                      offset=offset, with_payload=True, with_vectors=True)
            if not pts:
                break
            for rec in pts:
                if str(rec.id) in romansh:
                    st["deletes"] = st.get("deletes", 0) + 1
                    continue
                pl = rec.payload or {}
                crow = _corpus_row_for(txn, pl.get("url"), pl.get("anchor"))
                point, meta = project_record(rec, BASE_COLL, crow, pub_map,
                                             conf_map, gate, stats,
                                             seen_ids=seen_ids)
                if point is None:
                    cls = _account_exclusion(st, meta.get("reason", ""))
                    if cls == "deferred":
                        # consent not resolved yet -> queue for the deferred pass
                        deferred_fh.write(str(rec.id) + "\n")
                    continue
                if (not point.get("has_text")) and point.get("tok_epoch") == 1:
                    tail_fh.write(json.dumps({"url": point["payload"].get("url"),
                                              "point_id": point["id"]}) + "\n")
                    st["tail_tok1"] = st.get("tail_tok1", 0) + 1
                batch.append(point)
            processed += len(pts)
            if len(batch) >= args.upsert_batch:
                _flush(build, args.collection, batch)
                st["base_added"] = st.get("base_added", 0) + len(batch)
                batch = []
            st["base_processed"] = processed
            st["base_offset"] = offset if offset is None else str(offset)
            if processed % (args.scroll_batch * 20) < args.scroll_batch:
                deferred_fh.flush()
                st["consent_stats"] = dict(gate.stats)
                st["consent_grace_retained"] = stats.get("consent_grace_retained", 0)
                save_state(out_dir, st, resume_cmd)
                log(f"[base] processed {processed:,}/{TWIN_BASE_COUNT:,} "
                    f"added {st['base_added']:,} del {st['deletes']:,} "
                    f"deferred {st.get('deferred_count',0):,} "
                    f"declined {st.get('consent_declined',0):,} "
                    f"grace-retained {stats.get('consent_grace_retained',0):,} "
                    f"drops {st.get('drops',0):,} | {dict(stats)}")
                gate.save()
                if gate.mode == "pipelined":
                    gate.reload()   # pick up newly-rsync'd consent verdicts
                if (not st.get("slice_deleted")) and st["base_added"] > 1_000_000:
                    try:
                        build.delete_collection("aw_b1_slice")
                        st["slice_deleted"] = True
                        log("[base] deleted aw_b1_slice (>1M points reached)")
                    except Exception:  # noqa: BLE001
                        st["slice_deleted"] = True
            if offset is None:
                break
    if batch:
        _flush(build, args.collection, batch)
        st["base_added"] = st.get("base_added", 0) + len(batch)
    deferred_fh.close()
    tail_fh.close()
    env.close()
    gate.save()
    st["base_stats"] = dict(stats)
    st["consent_stats"] = dict(gate.stats)
    st["consent_grace_retained"] = stats.get("consent_grace_retained", 0)
    st["base_offset"] = None
    dcount = st.get("deferred_count", 0)
    log(f"[base] MAIN-PASS DONE processed {processed:,} added {st['base_added']:,} "
        f"del {st['deletes']:,} deferred {dcount:,} declined "
        f"{st.get('consent_declined',0):,} grace-retained "
        f"{stats.get('consent_grace_retained',0):,} drops {st.get('drops',0):,} "
        f"| consent classes {dict(gate.stats)}")
    # >15% deferred at main-pass end => the record-weight reorder didn't work; PAUSE
    if dcount > 0.15 * max(processed, 1):
        st["phase"] = "deferred_paused"
        save_state(out_dir, st, resume_cmd)
        log(f"[base] DEFERRED {dcount:,} > 15% of {processed:,} — PAUSE + report "
            f"(reorder ineffective) before the deferred pass")
        return
    st["phase"] = "deferred"
    save_state(out_dir, st, resume_cmd)


_DENSE_REEMBEDDER = None


def _reembed_explicit_dense(batch: list[dict]) -> int:
    """Re-embed only records whose corpus input explicitly changed.

    Most of a full build retains its frozen dense vector. This narrow path is
    for source-specific two-channel representations such as Fedlex R2; using
    the old vector there would make the full rebuild silently disagree with
    the regular stage-pack path.
    """
    global _DENSE_REEMBEDDER
    selected = [p for p in batch if p.get("dense_reembed_text")]
    if not selected:
        return 0
    if _DENSE_REEMBEDDER is None:
        import torch
        from cytir.core.serving_config import mrl_trunc
        from cytir.embed.model import load_embedder
        _DENSE_REEMBEDDER = (load_embedder(
            DENSE_REEMBED_MODEL, use_fp16=torch.cuda.is_available(),
            max_seq=DENSE_REEMBED_MAX_SEQ), mrl_trunc)
    (encode_docs, _encode_query, _spec), mrl_trunc = _DENSE_REEMBEDDER
    for start in range(0, len(selected), DENSE_REEMBED_BATCH):
        group = selected[start:start + DENSE_REEMBED_BATCH]
        dense, _ = encode_docs([p["dense_reembed_text"] for p in group],
                               batch=DENSE_REEMBED_BATCH)
        dense = mrl_trunc(dense, 128)
        for point, vector in zip(group, dense):
            point["vector"]["dense"] = vector
            point["dense_reembedded"] = True
    return len(selected)


def _flush(build, coll, batch):
    _reembed_explicit_dense(batch)
    build.upsert(coll, points=[
        qm.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
        for p in batch], wait=True)


def check_deferred_tripwire(processed: int, excluded: int,
                            threshold: float = DEFERRED_EXCL_MAX,
                            min_processed: int = 20_000) -> None:
    """ABORT the build if the deferred/unknown consent class resolves to more
    than `threshold` exclusion (task 319 hardening). A near-total exclusion of
    the deferred class is an ERROR SIGNATURE (census under-coverage, cache not
    synced, wrong cache file), not a clean count: real opt-out rates are a few
    percent (census: 3.9% declined). The 318 build resolved 100% of 929,826
    deferred points to excluded and kept going. Raises SystemExit(4)."""
    if processed < min_processed:
        return
    frac = excluded / max(processed, 1)
    if frac > threshold:
        raise SystemExit(
            f"[TRIPWIRE] deferred consent pass excluded {excluded:,}/"
            f"{processed:,} = {frac:.1%} > {threshold:.0%} "
            f"(AW_B1_DEFERRED_EXCL_MAX). A 100%-exclusion is an error "
            f"signature, not a clean count — check census coverage / cache "
            f"sync before resuming. BUILD ABORTED.")


def run_deferred(build, twin, kv_path, gate, pub_map, conf_map, st, out_dir,
                 args, log, resume_cmd, seen_ids=None):
    """Deferred consent pass: re-check the hosts that were cache-miss during the
    main pass, now that consent is 100% cached (gated on CONSENT.DONE). Returns
    False if consent isn't complete yet (driver re-invokes later).

    post_census: a host STILL absent from the completed census grace-retains
    (task 319 fix 1) — exclusion here is only a resolved DECLINE."""
    if not os.path.exists(os.path.join(out_dir, "CONSENT.DONE")):
        log("[deferred] CONSENT.DONE absent — consent not 100% cached yet; will resume")
        return False
    gate.reload()
    gate.post_census = True
    dq = os.path.join(out_dir, "consent_deferred_queue.txt")
    ids, seen = [], set()
    if os.path.exists(dq):
        for ln in open(dq):
            i = ln.strip()
            if i and i not in seen:
                seen.add(i)
                ids.append(i)
    log(f"[deferred] {len(ids):,} deferred ids to re-check (post_census=True: "
        f"census-absent hosts grace-retain, tripwire at "
        f"{DEFERRED_EXCL_MAX:.0%} exclusion)")
    env = kv_open(kv_path, readonly=True)
    stats = defaultdict(int)
    excluded = st.get("deferred_excluded", 0)
    start = st.get("deferred_processed", 0)
    try:
        with env.begin() as txn:
            for i in range(start, len(ids), args.scroll_batch):
                chunk = ids[i:i + args.scroll_batch]
                recs = twin.retrieve(BASE_COLL, ids=chunk, with_payload=True,
                                     with_vectors=True)
                batch = []
                for rec in recs:
                    pl = rec.payload or {}
                    crow = _corpus_row_for(txn, pl.get("url"), pl.get("anchor"))
                    point, meta = project_record(rec, BASE_COLL, crow, pub_map,
                                                 conf_map, gate, stats,
                                                 seen_ids=seen_ids)
                    if point is None:
                        excluded += 1
                        _account_exclusion(st, meta.get("reason", ""))
                        continue
                    batch.append(point)
                if batch:
                    _flush(build, args.collection, batch)
                    st["deferred_added"] = st.get("deferred_added", 0) + len(batch)
                st["deferred_processed"] = i + len(chunk)
                st["deferred_excluded"] = excluded
                check_deferred_tripwire(st["deferred_processed"], excluded)
                if (i // max(args.scroll_batch, 1)) % 20 == 0:
                    st["consent_stats"] = dict(gate.stats)
                    save_state(out_dir, st, resume_cmd)
                    gate.save()
                    log(f"[deferred] {st['deferred_processed']:,}/{len(ids):,} "
                        f"added {st.get('deferred_added',0):,} excl {excluded:,} "
                        f"grace {stats.get('consent_grace_retained',0):,}")
    except SystemExit:
        st["phase"] = "deferred_aborted"
        st["consent_stats"] = dict(gate.stats)
        save_state(out_dir, st, resume_cmd)
        gate.save()
        raise
    # end-of-pass tripwire (no min_processed floor: even a small deferred
    # queue resolving to near-total exclusion is the same error signature)
    try:
        check_deferred_tripwire(st.get("deferred_processed", 0), excluded,
                                min_processed=1000)
    except SystemExit:
        st["phase"] = "deferred_aborted"
        save_state(out_dir, st, resume_cmd)
        raise
    env.close()
    gate.save()
    st["deferred_stats"] = dict(stats)
    st["consent_stats"] = dict(gate.stats)
    st["phase"] = "staged"
    save_state(out_dir, st, resume_cmd)
    log(f"[deferred] DONE added {st.get('deferred_added',0):,} excluded "
        f"{excluded:,} grace-retained {stats.get('consent_grace_retained',0):,} "
        f"| classes {dict(gate.stats)}")
    return True


def run_staged(build, stag, kv_path, gate, pub_map, conf_map, st, out_dir,
               args, log, resume_cmd):
    env = kv_open(kv_path, readonly=True)
    per_pack = st.setdefault("staged_per_pack", {})
    # staged runs after the deferred phase => census complete: cache-miss
    # hosts grace-retain; exclusions here are counted, never silent (task 319 —
    # the old path dropped consent-unresolved staged points with no accounting).
    gate.post_census = True
    with env.begin() as txn:
        for pack in STAGING_PACKS:
            if pack in st.get("staged_done", []):
                continue
            stats = defaultdict(int)
            added = modified = excluded = 0
            offset = None
            batch = []
            # 325: within-PACK collision demotion only. A staged id matching an
            # EXISTING collection id is the intended add/modify semantics; two
            # staged records of one pack minting the same id is chunk collapse.
            pack_seen: set = set()
            while True:
                pts, offset = stag.scroll(pack, limit=args.scroll_batch, offset=offset,
                                          with_payload=True, with_vectors=True)
                for rec in pts:
                    pl = rec.payload or {}
                    crow = _corpus_row_for(txn, pl.get("url"), pl.get("anchor"))
                    point, meta = project_record(rec, pack, crow, pub_map,
                                                 conf_map, gate, stats,
                                                 seen_ids=pack_seen)
                    if point is None:
                        excluded += 1
                        _account_exclusion(st, meta.get("reason", ""))
                        continue
                    batch.append(point)
                if len(batch) >= args.upsert_batch or offset is None:
                    if batch:
                        ids = [p["id"] for p in batch]
                        existing = {r.id for r in build.retrieve(args.collection, ids=ids,
                                    with_payload=False, with_vectors=False)}
                        for p in batch:
                            if p["id"] in existing:
                                modified += 1
                            else:
                                added += 1
                        _flush(build, args.collection, batch)
                        batch = []
                if offset is None:
                    break
            per_pack[pack] = {"added": added, "modified": modified,
                              "excluded": excluded,
                              "grace": stats.get("consent_grace_retained", 0),
                              "undated_gov":
                                  stats.get("staged_excluded_undated_gov", 0)}
            st["staged_added"] = st.get("staged_added", 0) + added
            st["staged_modified"] = st.get("staged_modified", 0) + modified
            st["staged_excluded"] = st.get("staged_excluded", 0) + excluded
            st.setdefault("staged_done", []).append(pack)
            st["consent_stats"] = dict(gate.stats)
            gate.save()
            save_state(out_dir, st, resume_cmd)
            log(f"[staged] {pack}: +{added:,} ~{modified:,} excl {excluded:,} "
                f"grace {stats.get('consent_grace_retained',0):,} | {dict(stats)}")
    env.close()
    st["phase"] = "finalize"
    save_state(out_dir, st, resume_cmd)


def run_finalize(build, st, out_dir, args, log, resume_cmd):
    finalize_collection(build, args.collection, log=log)
    # keep-warm sparse postings (aw-write-churn-anon-bloat): a cheap warm query
    try:
        build.query_points(args.collection, query=qm.SparseVector(indices=[1], values=[1.0]),
                           using="bm25", limit=1)
    except Exception:  # noqa: BLE001
        pass
    for _ in range(720):
        info = build.get_collection(args.collection)
        if str(info.status) in ("green", "CollectionStatus.GREEN"):
            break
        time.sleep(10)
    st["phase"] = "manifest"
    save_state(out_dir, st, resume_cmd)
    log(f"[finalize] {args.collection} green, m=16")


def _consent_census(cache_path, log=print):
    """The consent census — a first-of-its-kind artifact for this index: the live
    verdict distribution over the full distinct-host universe."""
    try:
        c = json.load(open(cache_path))
    except Exception:  # noqa: BLE001
        return {}
    cnt = defaultdict(int)
    for v in c.values():
        so = v.get("search_ok")
        status = v.get("status")
        reason = v.get("reason")
        if status == "ok" and so is True:
            cnt["search_ok_true"] += 1
        elif so is False:
            cnt["declined"] += 1
        elif reason == "unreachable" or status != "ok":
            cnt["unreachable_retryable"] += 1
        else:
            cnt["other"] += 1
    cnt["total_hosts"] = len(c)
    return dict(cnt)


def run_full_manifest(build, st, out_dir, args, log):
    final = build.count(args.collection, exact=True).count
    deletes = st.get("deletes", 0)
    counts = {"points": final, "adds": final + deletes, "changes": 0, "deletes": deletes}
    sources = []
    # base
    sources.append({"pack": "base", "source": BASE_COLL, "version": "aw_v4_sat-2026-07",
                    "count": st.get("base_added", 0) + st.get("deferred_added", 0),
                    "policy": "POLICY.md@v1"})
    sources.append({"pack": "ch", "source": "fedlex-reextract",
                    "version": "2026-07-23", "count": 19516, "policy": "POLICY.md@v1"})
    for pack, d in (st.get("staged_per_pack", {}) or {}).items():
        p = "ch" if any(x in pack for x in ("zefix", "chmedia", "swissinfo")) else "mixed"
        sources.append({"pack": p, "source": pack, "version": "staged-2026-07",
                        "count": d.get("added", 0), "policy": "POLICY.md@v1"})
    # exceptions block (task 319 policy): unreachable/uncensused hosts are
    # RETAINED under grace (consent_state=grace payload stamp) + retry-queued;
    # only resolved DECLINES are excluded. The retry queue drives the weekly
    # delta re-resolve (evict on decline or grace expiry).
    retry_path = os.path.join(out_dir, "consent_retry_queue.txt")
    retry_hosts = [ln.strip() for ln in open(retry_path)] if os.path.exists(retry_path) else []
    retry_hosts = [h for h in retry_hosts if h]
    bs = st.get("base_stats", {}) or {}
    cs = st.get("consent_stats", {}) or {}
    exceptions = {
        "consent_declined_excluded": st.get("consent_declined", 0),
        "consent_deferred_recovered": st.get("deferred_added", 0),
        "consent_deferred_excluded": st.get("deferred_excluded", 0),
        "_legacy_declined_stat": bs.get("declined", 0),
        "consent_grace_retained": st.get("consent_grace_retained", 0),
        "consent_grace_classes": {k: v for k, v in cs.items()
                                  if k.endswith("_grace")},
        "consent_retry_queue": retry_path if retry_hosts else None,
        "consent_retry_hosts": len(retry_hosts),
        "chmedia_undated_gov_excluded": st.get("staged_excluded_undated_gov", 0),
        "romansh_evicted": st.get("deletes", 0),
        "en_law_tok1_tail": (st.get("coverage", {}) or {}).get("en_law_tail", 0),
        "accent_law_stragglers_tok1": (st.get("coverage", {}) or {}).get("accent_law_in_tail", {}),
        "tail_refetch_list": os.path.join(out_dir, "aw_b1.tail_refetch.jsonl"),
        "tail_refetch_count": st.get("tail_tok1", 0),
        "consent_census": _consent_census(args.consent_cache, log),
    }
    man = write_manifest(
        os.path.join(out_dir, "aw_b1.MANIFEST.json"), "aw_b1", None,
        sources, counts,
        {"golds": 0, "pass": False, "report": "PENDING (promote gate not yet run)"},
        exceptions=exceptions, log=log)
    errs = validate_manifest(man)
    log(f"[manifest] validate: {'OK' if not errs else 'ERRORS: ' + str(errs)}")
    st["phase"] = "done"
    st["counts"] = counts
    save_state(out_dir, st)
    log(f"[manifest] counts={counts}")
    return man, errs


def run_full(args, log=print, *, corpus_dir, resume_prefix):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    kv_path = os.path.join(out_dir, "corpus_text.lmdb")
    st = load_state(out_dir)
    resume_cmd = (f"{resume_prefix} "
                  f"--mode full --collection {args.collection} --out-dir {out_dir} "
                  f"--consent-mode {args.consent_mode} --git-sha {args.git_sha} "
                  f"--fedlex {args.fedlex} --romansh {args.romansh} "
                  f"--cards-audit {args.cards_audit} "
                  + " ".join(f"--news-published {p}" for p in (args.news_published or [])))
    ID.self_test()
    log(f"[full] resuming at phase={st['phase']}")

    if st["phase"] == "kv":
        build_corpus_kv(kv_path, args.fedlex, out_dir, st, log,
                        extra_jsonls=args.extra_jsonl,
                        corpus_dir=corpus_dir, save_state=save_state)
        st["phase"] = "coverage"
        save_state(out_dir, st, resume_cmd)

    if st["phase"] == "coverage":
        twin = qc(TWIN_URL)
        cov = measure_coverage(twin, kv_path, args.coverage_sample, log)
        st["coverage"] = cov
        save_state(out_dir, st, resume_cmd)
        # GATE (hybrid option C): stop if tail > 8% OR any DE/FR/IT Tier-A law in
        # tail. EN law tail is tolerated (folded==unfolded for ASCII) + counted.
        if cov["overall_tail"] > 0.08:
            log(f"[GATE] STOP: overall tail {cov['overall_tail']:.2%} > 8%")
            return st, "STOP_TAIL"
        accent_ct = sum(cov["accent_law_in_tail"].values())
        accent_frac = accent_ct / max(cov["sample"], 1)
        if accent_frac > ACCENT_LAW_TOLERANCE:
            log(f"[GATE] STOP: accent-bearing (de/fr/it) Tier-A law in tail "
                f"{accent_ct}/{cov['sample']} = {accent_frac:.4%} > tol "
                f"{ACCENT_LAW_TOLERANCE:.4%}: {cov['accent_law_in_tail']}")
            return st, "STOP_LAW"
        log(f"[GATE] PASS: tail {cov['overall_tail']:.2%} <= 8%; accent-law tail "
            f"{accent_ct}/{cov['sample']}={accent_frac:.4%} <= tol "
            f"{ACCENT_LAW_TOLERANCE:.4%} ({cov['accent_law_in_tail']} genuine "
            f"stragglers ship tok=1, tail-refetch+exceptions); EN-law tail "
            f"{cov['en_law_tail']:,} tolerated")
        if args.gate_only:
            log("[full] --gate-only: stopping after coverage gate for review")
            return st, "GATE_PASS"
        st["phase"] = "base"
        save_state(out_dir, st, resume_cmd)

    twin = qc(TWIN_URL)
    stag = qc(STAGING_URL)
    build = qc(BUILD_URL)
    pub_map = load_published_map(args.news_published or [], log=log)
    conf_map = load_resolve_conf_map(args.cards_audit or "", log=log)
    gate = ConsentGate(args.consent_mode, args.consent_grace_days,
                       args.consent_cache, log=log)
    romansh = load_delete_set(args.romansh, log)

    # 325: in-build minted-id set for the collision-demotion chain. Fresh build
    # -> empty; resume -> rebuilt from the collection (id-only scroll).
    seen_ids: set = set()
    if st["phase"] in ("base", "deferred", "deferred_paused") and (
            st.get("base_processed", 0) > 0 or st.get("deferred_processed", 0) > 0):
        seen_ids = load_seen_ids(build, args.collection, log)
    if st["phase"] == "base":
        run_base(build, twin, kv_path, gate, pub_map, conf_map, romansh, st,
                 out_dir, args, log, resume_cmd, seen_ids=seen_ids)
    if st["phase"] == "deferred_paused":
        log("[full] deferred>15% PAUSE — awaiting review; not proceeding")
        return st, "STOP_DEFERRED"
    if st["phase"] == "deferred":
        if not run_deferred(build, twin, kv_path, gate, pub_map, conf_map, st,
                            out_dir, args, log, resume_cmd, seen_ids=seen_ids):
            return st, "WAIT_CONSENT"   # driver re-invokes when CONSENT.DONE lands
    if st["phase"] == "staged" and not st.get("id_collision_check"):
        # id-collision tripwire (task 319 / GAP A33): every base+deferred
        # upsert minted a distinct id, so the collection count must match the
        # upsert count. The 318 build silently lost ~1.8M points here.
        expected = st.get("base_added", 0) + st.get("deferred_added", 0)
        actual = build.count(args.collection, exact=True).count
        missing = expected - actual
        rate = missing / max(expected, 1)
        st["id_collision_check"] = {"expected": expected, "actual": actual,
                                    "missing": missing, "rate": round(rate, 5)}
        save_state(out_dir, st, resume_cmd)
        log(f"[id-collision] base+deferred upserted {expected:,} -> collection "
            f"holds {actual:,} (missing {missing:,} = {rate:.3%}, "
            f"max {ID_COLLISION_MAX:.1%})")
        if rate > ID_COLLISION_MAX:
            st["phase"] = "id_collision_aborted"
            save_state(out_dir, st, resume_cmd)
            raise SystemExit(
                f"[TRIPWIRE] id-collision rate {rate:.3%} > "
                f"{ID_COLLISION_MAX:.1%} (AW_B1_ID_COLLISION_MAX): distinct "
                f"source records are minting duplicate point ids (chunk "
                f"collapse, GAP A33 signature). BUILD ABORTED before staged merge.")
    if st["phase"] == "staged":
        run_staged(build, stag, kv_path, gate, pub_map, conf_map, st, out_dir,
                   args, log, resume_cmd)
    if st["phase"] == "finalize":
        run_finalize(build, st, out_dir, args, log, resume_cmd)
    if st["phase"] == "manifest":
        run_full_manifest(build, st, out_dir, args, log)
    log(f"[full] phase now: {st['phase']}")
    return st, "DONE" if st["phase"] == "done" else st["phase"]
