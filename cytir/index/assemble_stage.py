"""Stage step of the assembly binder (task 2105, binder slice A3).

Moved verbatim from the binder: `cmd_stage`, its partition workers, the
staged-row projection per partition (`_stage_partition`), the twin payload
reader, `_SrcShim`, the stage summary and the assertions that refuse a staged
tree (payload shape, serving metadata, persisted inputs). Staged rows and
refusals are unchanged; the only edits are that the pieces the binder's other
stages share are imported rather than defined here, a twin client is built
with `cytir.core.qdrant.qdrant_connect`, and the repo root comes from
`cytir.core.paths`.
"""
from __future__ import annotations

import functools
import gzip
import json
import multiprocessing
import pathlib
import time

import numpy as np

from cytir.consent.build_gate import ConsentGate
from cytir.core.qdrant import qdrant_connect
from cytir.index.assemble_inputs import (
    load_adr103_carried_flags, load_pack_route_exclusions, read_jsonl_gz_ids,
    sha256_file)
from cytir.index.assemble_plan import (
    FULL_DIM, PINNED_PROJECTION_COMMIT, assert_family_contract, log)
from cytir.index.b1_corpus_kv import load_published_map
from cytir.index.b1_projection import project_record


def staged_pages(outdir, receipts):
    """(url, content_digest) of every row staged under `outdir`, streamed.

    The roster is the receipts' own staged payload files, not a list anyone
    typed: what gets checked is exactly what gets upserted.
    """
    for r in receipts:
        pp = outdir / f"{r['set_id']}.{r['shard']}.payload.jsonl.gz"
        if not pp.exists():
            continue
        with gzip.open(pp, "rt") as fh:
            for line in fh:
                pl = json.loads(line).get("payload") or {}
                yield {"url": pl.get("url") or pl.get("canonical_url"),
                       "sha": pl.get("content_digest")}  # canonical row


def assert_persisted_inputs(pages, args, stage: str, label: str = ""):
    """Refuse rows whose PAGE is not persisted in a registered store (ADR-105).

    Task 581. `assert_family_contract` asks whether a set declared what it is;
    `assert_payload_shape` asks whether its rows are shaped right. Neither asks
    whether the page behind a row still exists anywhere but in the row. Task
    426 found 1,636,193 points whose only surviving input was their own legacy
    production passage, and task 579 found 492,211 pages with no L0, no L1 and
    no L2 text at all -- rows for which ADR-026's "every collection must be
    rebuildable from the store" is false, and has been false since the day they
    were written.

    Same posture as the two guards beside it: at STAGE (before the load) and
    again per partition in LOAD (so a tree staged before this guard existed
    cannot be loaded unchecked).
    """
    import cytir.corpus.persisted as PI
    require = getattr(args, "require_persisted", "l1")
    try:
        v = PI.assert_pages_persisted(pages, require=require, label=label or stage)
    except PI.UnpersistedInputs as exc:
        raise SystemExit(str(exc)) from None
    except PI.PersistedInputsError as exc:
        raise SystemExit(
            f"REFUSING: the persisted-inputs guard cannot run ({exc}). A guard "
            "that skips on a missing input reports green over unchecked "
            "ground.") from None
    log(f"persisted inputs [{stage}] {label}: {v['pages']:,} page(s), "
        f"{v['persisted']:,} L1/L0, {v['l1b_reconstructible']:,} L2-only "
        f"(require={require})")
    return v


# stage 2 -- stage (partitioned, resumable)
# ===========================================================================
def cmd_stage(args):
    work = pathlib.Path(args.work)
    man = json.loads((work / "binder-plan.json").read_text())
    # Task 529, again at stage and not only at plan: a work tree planned
    # before the contract existed must not be stageable unchecked. Same
    # reasoning as `assert_payload_shape` running in both stage and verify.
    assert_family_contract(
        sorted({p["set_id"] for p in man["partitions"]}
               & (set(args.sets.split(",")) if args.sets
                  else {p["set_id"] for p in man["partitions"]})),
        args, "stage")
    outdir = work / "staged"
    outdir.mkdir(parents=True, exist_ok=True)
    supp_path = work / "suppressed-ids.jsonl.gz"
    suppressed = read_jsonl_gz_ids(supp_path) if supp_path.exists() else set()

    if args.skip_adr103_flags:
        log("WARNING: --skip-adr103-flags; the 7,935 ADR-103 carried rows "
            "will NOT get their consent_state/non_refreshable stamp. Do not "
            "use this for a shipping build.")
        adr103_flags, adr103_stats = {}, {"total": 0}
    else:
        adr103_flags, adr103_stats = load_adr103_carried_flags(args.census)
        log(f"ADR-103 flags loaded: {adr103_stats['total']:,} points "
            f"({adr103_stats['by_set']}), "
            f"{adr103_stats['unresolved_folded_into_unreadable']} "
            f"'unresolved' folded to 'unreadable'")

    parts = man["partitions"]
    if args.sets:
        keep = set(args.sets.split(","))
        parts = [p for p in parts if p["set_id"] in keep]

    todo = []
    for p in parts:
        rp = outdir / f"{p['set_id']}.{p['shard']}.receipt.json"
        if rp.exists() and not args.force:
            r = json.loads(rp.read_text())
            log(f"skip {p['set_id']}/{p['shard']} (staged {r['staged']:,})")
            continue
        todo.append(p)

    jobs = max(1, int(getattr(args, "jobs", 1) or 1))
    if jobs == 1:
        registry = _open_metadata_registry(args)
        for p in todo:
            _stage_partition(p, outdir, suppressed, args, adr103_flags,
                             registry)
    else:
        # Partitions are independent by construction: each reads its own
        # sealed triplet and writes its own three output files plus a receipt,
        # and the per-partition conservation check is what makes the stage
        # correct. Nothing is shared but read-only inputs, so the only reason
        # this was serial is that nobody needed it parallel; at 725 rows/s a
        # 38.5M-row restage is nine hours of wall clock spent proving the same
        # thing 747 times.
        _PARALLEL_STATE.update(outdir=outdir, suppressed=suppressed,
                               args=args, adr103_flags=adr103_flags)
        log(f"staging {len(todo)} partitions with {jobs} workers")
        with multiprocessing.get_context("fork").Pool(
                jobs, initializer=_stage_worker_init) as pool:
            for _ in pool.imap_unordered(_stage_worker, todo, 1):
                pass
    log("STAGE complete for the requested partitions")
    _stage_summary(work)
    _assert_serving_metadata_complete(work, args)
    _assert_persisted_inputs_staged(work, args)


# Parallel staging shares only read-only inputs. They are handed to the workers
# through module state at fork time rather than pickled per task: `proj` is an
# imported module and `suppressed` is a set of millions of ids, neither of which
# should cross a pipe 747 times.
_PARALLEL_STATE: dict = {}
_WORKER_REGISTRY = None


def _stage_worker_init():
    global _WORKER_REGISTRY
    _WORKER_REGISTRY = _open_metadata_registry(_PARALLEL_STATE["args"])


def _stage_worker(p):
    st = _PARALLEL_STATE
    _stage_partition(p, st["outdir"], st["suppressed"], st["args"],
                     st["adr103_flags"], _WORKER_REGISTRY)
    return p["shard"]


def assert_payload_shape(receipts, contract):
    """Evaluate `data/payload-shape-contract.v1.json` and refuse on a FAIL.

    Used at STAGE (before a row is upserted) and again in VERIFY (so a tree
    staged before the contract existed cannot be loaded and shipped unchecked).
    """
    from cytir.index import payload_shape_contract as PSC
    report = PSC.evaluate(receipts, contract)
    PSC.format_report(report, log=log)
    if report["verdict"] != "PASS":
        raise SystemExit(
            "REFUSING: the payload-shape contract FAILS. "
            + " | ".join(report["findings"])
            + f"\nThe thresholds and their evidence are in {report['contract']}; "
              "changing one is a deliberate, visible edit, not a flag.")
    return report


def _assert_serving_metadata_complete(work, args):
    """Refuse a stage tree that would ship rows the ranker cannot read.

    Task 514's collection had `trust_tier` and `source_type` blank on 36.5% of
    its rows and every count check passed, because conservation counts rows and
    says nothing about what is IN them. This is the check that would have
    caught it, and it is deliberately at the END of stage rather than inside
    the projection: an individual row with no resolvable stamp is a fact to
    measure, a whole build of them is a refusal.

    Sources the load will exclude are exempt -- they are staged but never
    upserted, so their metadata is not a property of the shipped collection.
    The exclusion list is the same file `load` uses; there is no second list.
    """
    allowed = float(getattr(args, "max_missing_serving_signal_frac", 0.0) or 0.0)
    if not getattr(args, "serving_metadata_registry", None):
        log("NOTE: no serving-metadata registry declared; completeness check "
            "not evaluated (this is what task 514 shipped)")
        return
    excluded = set(load_pack_route_exclusions())
    total = missing = refused = no_sparse = 0
    worst = []
    receipts = []
    for rp in sorted((work / "staged").glob("*.receipt.json")):
        r = json.loads(rp.read_text())
        src = r["shard"].split(".shard-")[0].removeprefix("source=")
        if src in excluded:
            continue
        receipts.append(r)
        sm = r.get("serving_metadata") or {}
        m = sm.get("meta_row_missing_serving_signal", 0)
        refused += sm.get("meta_row_source_type_refused_entity_card", 0)
        no_sparse += r.get("staged_without_sparse", 0)
        total += r["staged"]
        missing += m
        if m:
            worst.append((m, r["set_id"], r["shard"]))
    frac = (missing / total) if total else 0.0
    log(f"serving-metadata completeness: {total - missing:,}/{total:,} rows "
        f"carry every serving signal; {missing:,} do not ({frac:.4%}), "
        f"allowance {allowed:.4%}")
    if refused:
        log(f"  {refused:,} rows carry every OTHER signal but no `source_type`, "
            f"because the snapshot's answer was `entity_card` and that label is "
            f"not inheritable through a join (aw_serving_metadata); a ruling, "
            f"not a coverage gap")
    log(f"  {no_sparse:,} staged rows carry NO bm25 posting "
        f"({(no_sparse / total if total else 0):.4%}); task 514 shipped "
        f"19,619,656 (63.5%) and said nothing")
    for m, sid, shard in sorted(worst, reverse=True)[:15]:
        log(f"  [!] {m:>9,} {sid}/{shard}")
    if frac > allowed:
        raise SystemExit(
            f"REFUSING: {missing:,} staged rows ({frac:.4%}) carry no "
            f"trust_tier/source_type/title/lang after hydration, above the "
            f"declared allowance {allowed:.4%}. Raise "
            f"--max-missing-serving-signal-frac deliberately, with the number, "
            f"or fix the registry coverage. Task 514 shipped 36.5%.")

    # The same tree, against the DECLARED per-set and per-family contract
    # (task 516). The fraction above is one number over the whole build; this
    # is the one that can see a single family shipping unrankable titles while
    # the total stays comfortable.
    from cytir.index import payload_shape_contract as PSC
    contract = (getattr(args, "_payload_shape_contract", None)
                or PSC.load(getattr(args, "payload_shape_contract", None)))
    # A partition this run SKIPPED (its receipt already existed) was staged
    # before a contract dimension existed and carries no counter for it. That
    # is "not measured", which this contract treats as a failure -- correctly --
    # so it is measured here, from the staged payload the load will upsert,
    # rather than left to fail as though the tree were defective. Same posture
    # as `verify`: checked late is not the same as unchecked, and both say so.
    PSC.backfill_title_shape(receipts, work / "staged", contract, log=log)
    PSC.backfill_provenance_head(receipts, work / "staged", contract, log=log)
    PSC.backfill_extraction_collapse(receipts, work / "staged", contract, log=log)
    PSC.backfill_chunk_width(receipts, work / "staged", contract, log=log)
    report = assert_payload_shape(receipts, contract)
    (work / "payload-shape-report.json").write_text(json.dumps(report, indent=1))


def _assert_persisted_inputs_staged(work, args):
    """ADR-105 (task 581): the pages behind the staged rows must exist on disk.

    Deliberately NOT folded into `_assert_serving_metadata_complete`: that
    function returns early when no serving-metadata registry is declared, and a
    guard that inherits somebody else's early return is a guard that reports
    green over unchecked ground (verification-doctrine 12). It has its own
    call site in `cmd_stage` and its own roster -- every staged receipt, not
    the subset that survives the exclusion list, because `load` checks each
    partition again anyway and a stage that refuses early is cheaper than one
    that refuses at hour six.
    """
    receipts = [json.loads(rp.read_text())
                for rp in sorted((work / "staged").glob("*.receipt.json"))]
    if not receipts:
        raise SystemExit(
            "REFUSING: no staged receipts under "
            f"{work / 'staged'} -- nothing to check is not the same as "
            "nothing to write.")
    v = assert_persisted_inputs(staged_pages(work / "staged", receipts), args,
                                "stage", label=str(work))
    (work / "persisted-inputs-report.json").write_text(
        json.dumps(v, indent=1))


class _SrcShim:
    """The minimal `qdrant Record` surface `project_record` reads: id/payload/vector.

    This docstring used to say "the sealed stage rows are already flat payload
    dicts (url, title, snippet, lang, domain, source_type, tier, trust_tier,
    country, last_seen, text ...)". That was ASSERTED, never checked, and it is
    true of six of the nineteen contributing sets. The other thirteen ship
    sealed EMBEDDING-STAGE records -- identity + text, no serving payload -- in
    three different shapes (chunk@2 materializer, task417 family pack,
    entity-card re-embed). `project_record` passes serving metadata through, so
    it emitted what it was given: nothing. Task 514 loaded 11,273,954 rows
    (36.5%) with no `trust_tier` and no `source_type`. Task 515 hydrates those
    fields from the sealed membership snapshot before projection
    (`aw_serving_metadata`), and the stage receipt now says per field where
    each value came from.

    `project_record` wants a Record because its normal input is a point
    scrolled out of a source collection; this adapter lets the binder reuse the
    projection unchanged rather than growing a second one.

    `vector` is NOT optional: `project_record` returns
    `{"reason": "no dense vector on source"}` and excludes the row when
    `src.vector` carries no `dense` key. Passing the sealed 1024-dim vector here
    is also what lets the projection own the SPARSE leg -- see `_stage_partition`.
    """
    __slots__ = ("id", "payload", "vector")

    def __init__(self, pid, payload, dense, bm25=None):
        self.id = pid
        self.payload = payload
        # `bm25` is the CARRIED posting for a row with no text surface. Task
        # 515: `project_record`'s no-text tail reads `src.vector["bm25"]` and
        # keeps it -- that is the branch's entire purpose -- but this shim only
        # ever set `dense`, so every row the binder staged without text got
        # `sidx, sval = [], []` and reached the collection with NO sparse
        # vector at all. Measured on task 514's staged tree: 19,619,656 rows,
        # 63.5% of `aw_e4`, including the whole 19.2M Wikipedia slice, i.e. the
        # BM25 leg of two thirds of the index, silently absent.
        self.vector = {"dense": dense}
        if bm25 is not None:
            self.vector["bm25"] = bm25


def _payload_from_twin(ids, args):
    """Payload for an IN-PLACE RE-EMBED, joined from the read-only twin by id.

    413a is a re-embed: the vector changes, the payload does not, so the payload
    legitimately still lives in `aw_e3` under the SAME point ids. This is an
    INDEXED point lookup (`retrieve` by id), not a scroll or a filter -- the
    `prod-serving-box-fragile-scrolls` incident is about unindexed full-scan
    scroll/filter on the LIVE box, and neither applies here: the twin is the
    designated place to inspect, and retrieve-by-id is the cheapest read Qdrant
    offers.

    READ ONLY. The twin is never written by this binder.

    A missing id is returned as an empty payload and COUNTED, never silently
    dropped: an in-place re-embed whose base point has vanished is a real
    finding (the vector would be orphaned), and `project_record` will exclude it
    on `no_url` so it lands in the exclusion census under its own name.
    """
    if not args.twin_url:
        raise SystemExit(
            "payload_route='twin' needs --twin-url (read-only), e.g. "
            "http://localhost:6444")
    client = qdrant_connect(args.twin_url, timeout=300)
    out, sparse, missing = [], [], 0
    B = args.twin_batch
    for lo in range(0, len(ids), B):
        chunk = ids[lo:lo + B]
        # `with_vectors=["bm25"]` (task 515): an in-place re-embed changes the
        # DENSE vector and nothing else -- not the text, so not the posting.
        # Re-deriving the posting is impossible here (there is no text on this
        # route, which is the whole reason the payload comes from the twin),
        # so the only truthful options are to carry it or to ship none. Task
        # 514 shipped none for 19,227,106 Wikipedia rows.
        got = client.retrieve(args.twin_collection, ids=chunk,
                              with_payload=True, with_vectors=["bm25"])
        by_id = {str(r.id): r for r in got}
        for pid in chunk:
            r = by_id.get(str(pid))
            if r is None:
                missing += 1
                out.append({})
                sparse.append(None)
            else:
                out.append(dict(r.payload or {}))
                sparse.append((r.vector or {}).get("bm25")
                              if isinstance(r.vector, dict) else None)
    return out, sparse, missing


def _open_metadata_registry(args):
    """One `ServingMetadataRegistry` per process, or None.

    Task 515. sqlite connections are not fork-safe, so this is opened INSIDE
    each stage worker rather than passed across the fork. `None` when the
    caller declared no registry -- the historical behaviour, byte-for-byte.
    """
    if not getattr(args, "serving_metadata_registry", None):
        return None
    from cytir.index import serving_metadata as SM
    return SM.ServingMetadataRegistry(
        args.serving_metadata_registry,
        domain_registry_path=getattr(args, "serving_metadata_domain_registry",
                                     None) or None,
        manifest_path=getattr(args, "serving_metadata_manifest", None) or None,
        min_domain_share=getattr(args, "serving_metadata_min_domain_share",
                                 0.90))


@functools.lru_cache(maxsize=8)
def _news_published_map(paths: tuple):
    """`load_published_map`, cached per paths so every
    partition in a `--jobs`>1 run doesn't re-read the sidecar files.

    Task 536: `assembly_binder.py:1504` used to pass `published_map={}`
    unconditionally, so the `NEWS_PACKS` branch in `project_record` could
    never resolve a date for a row whose own sealed record had none. `paths`
    is a tuple (hashable, for lru_cache) of `args.news_published`; empty ->
    an empty map, same as historical behaviour.
    """
    return load_published_map(list(paths), log=print)


@functools.lru_cache(maxsize=1)
def _representation_registry():
    """`data/representation-contract.v1.json`, loaded once per process."""
    from cytir.index import representation_contract as RC
    return RC.load_registry()


def _representation_family(set_id: str, source: str | None) -> dict:
    """The representation-contract family governing this partition, or {}.

    Resolution order is the registry's own `prov_id_note`: by `set_id` from
    `sealed-populations.v1.json`, then by a pack's `source=` dir. An unlisted
    set resolves to {} and keeps the pre-528a single-surface behaviour, which
    is the safe default -- but the receipt records the resolution, so "this
    family had no contract" is a countable fact rather than a silence.
    """
    reg = _representation_registry()["families"]
    for key, fam in reg.items():
        if set_id in (fam.get("applies_to") or []) or key == set_id:
            return fam
    if source:
        for fam in reg.values():
            if source in (fam.get("applies_to") or []):
                return fam
    return {}


def _stage_partition(p, outdir, suppressed, args, adr103_flags=None,
                     registry=None):
    from collections import defaultdict

    adr103_flags = adr103_flags or {}

    sid, stem = p["set_id"], p["shard"]
    n = p["rows"]
    t0 = time.time()
    # The per-family TITLE-SHAPE contract (task 516). A title can be present,
    # non-empty and still not carry the handle the query uses: 22,928 fedlex
    # statutes shipped titled by their formal docTitle, with the SR number and
    # the abbreviation -- the two things a legal query is made of -- nowhere in
    # the title or in the BM25 surface composed from it. Counted here, per
    # partition, so the refusal below is a measurement and not a spot check.
    from cytir.index import payload_shape_contract as PSC
    _contract = getattr(args, "_payload_shape_contract", None)
    _source = str(p.get("source") or "").removeprefix("source=") or None
    _title_rule = PSC.title_pattern(_contract, _source) if _contract else None
    n_title_rows = n_title_match = 0
    # task 536: the `published`-presence dimension, same opt-in shape as the
    # title-shape contract above -- a family is checked only once its
    # declared min_share exists, so a family that never carried `published`
    # in production is not failed over a rule nobody measured a need for.
    _pub_rule = (PSC.published_presence_min_share(_contract, _source)
                if _contract else None)
    n_pub_rows = n_pub_present = 0
    # task 517: the same question one level down -- is this row's snippet (and
    # therefore the head of its dense embedding) the document, or the
    # materializer's own provenance block? Measured on EVERY partition, because
    # the two instances of this defect were each found in a family nobody
    # suspected until after the collection was built.
    _prov_rule = PSC.provenance_head_pattern(_contract) if _contract else None
    n_prov_rows = n_prov_head = 0
    # task 528a: does THIS family declare its own BM25 surface? Resolved from
    # the representation contract by set_id (or by `source=` dir for a pack),
    # exactly as `data/representation-contract.v1.json` says a family is
    # resolved until ADR-089 lands. Recorded on the receipt so a build says
    # which sets ran two-channel and which ran one.
    _sparse_channel_declared = bool(
        _representation_family(sid, _source).get("sparse_surface_field"))
    vecs = np.fromfile(p["vectors"], dtype="<f2").reshape(n, FULL_DIM)
    ids = [json.loads(l)["point_id"] for l in open(p["row_ids"])]
    route = p.get("payload_route", "stage")
    n_missing_payload = 0
    carried_sparse = [None] * n
    if route in ("stage", "pack"):
        # "pack" differs only in WHERE the payload file lives and what it is
        # called (texts.jsonl.gz in the source's stage dir, resolved through its
        # STAGE-RECEIPT); the format and the positional binding are identical.
        with gzip.open(p["records"], "rt") as fh:
            recs = [json.loads(l) for l in fh]
        # A row with NO text surface cannot regenerate its BM25 posting -- the
        # projection's no-text tail can only keep the one the point already
        # has. Two sealed sets on this route carry no text at all
        # (task467 cards carry `dense_text`, task505_narrow_d carries neither),
        # and in task 514 they reached the collection with an empty sparse
        # vector. Fetching the existing posting is a read-only retrieve-by-id
        # against the twin, exactly as `payload_route="twin"` already does, and
        # it is legal precisely because these sets are IN-PLACE re-embeds: the
        # text did not change, so the posting did not either. Decided from the
        # DATA (does this row have a text surface?), never from a set-id list.
        need = [i for i, r in enumerate(recs)
                if not (r.get("text") or r.get("dense_text"))]
        if need and args.twin_url:
            _pl, sp, _miss = _payload_from_twin([ids[i] for i in need], args)
            for i, s in zip(need, sp):
                carried_sparse[i] = s
        elif need:
            log(f"{sid}/{stem}: [!] {len(need):,} rows have no text surface and "
                f"no --twin-url to carry their posting from; they will ship "
                f"with an EMPTY bm25 vector")
    elif route == "twin":
        recs, carried_sparse, n_missing_payload = _payload_from_twin(ids, args)
    else:
        raise SystemExit(f"{sid}/{stem}: unsupported payload_route {route!r}")
    if not (len(ids) == len(recs) == len(carried_sparse) == n):
        raise SystemExit(f"{sid}/{stem}: positional binding broken "
                         f"({len(ids)} ids / {len(recs)} recs / n={n})")

    gate = ConsentGate(mode=args.consent_mode, grace_days=args.grace_days,
                       cache_path=args.consent_cache, log=lambda *a: None)
    stats = defaultdict(int)
    kept = 0
    n_suppressed = 0
    n_excluded = 0
    id_divergence = 0
    reasons = defaultdict(int)
    n_adr103_stamped = 0
    n_adr103_conflicts = 0
    n_no_sparse = 0
    # Rows whose producer sealed a DISTINCT sparse channel and that therefore
    # tokenise a different surface from the one they embedded. Counted so the
    # receipt says how many rows exercised the two-channel path -- an absence
    # nothing counted is exactly how the collapse shipped (task 527 sec.4.2).
    n_sparse_channel = 0
    n_carried_sparse = sum(1 for s in carried_sparse if s is not None)
    adr103_states = defaultdict(int)

    vout = open(outdir / f"{sid}.{stem}.vectors.f16", "wb")
    rout = gzip.open(outdir / f"{sid}.{stem}.payload.jsonl.gz", "wt")
    kept_ids = []

    # task 536: the news-published sidecar (see --news-published above). Loaded
    # once per partition, cached across partitions in this worker by
    # `_news_published_map`; `{}` when the flag was never passed, matching the
    # historical `published_map={}` exactly.
    pub_map = _news_published_map(tuple(getattr(args, "news_published", None)
                                        or []))

    for i, (pid, rec) in enumerate(zip(ids, recs)):
        if pid in suppressed:
            n_suppressed += 1
            continue
        # The sealed stage row's `text` IS the corpus row for this point -- it is
        # the exact text the lane embedded, positionally bound to the vector. So
        # it is passed as the corpus join rather than left to the no-text tail:
        # that is what routes the sparse leg through `_retokenize` ->
        # `PROJ.saturated_bm25`, i.e. compose_index_text(title/site_name + body),
        # folded bm25 tokens, ADR-028 saturation, and the segmented entity-card
        # path with per-term max merge. The binder therefore does NOT derive BM25
        # itself; the projection owns it, and there is one implementation.
        # `dense_text` is the same surface under the name its producer used:
        # the task467 entity-card stage seals `dense_text` and no `text` (its
        # own stage.json: "dense_text_contract: record['dense_text']"), so
        # reading only `text` sent 339,307 cards down the no-text tail.
        #
        # TASK 528a. That single `surface` fed BOTH legs. A producer that
        # sealed a distinct SPARSE channel -- every task-417 law pack row, and
        # every ADR-060 entity card -- had it discarded here, so its BM25
        # posting was built from the dense prose instead: task 527 sec.5
        # measured production's card postings at 9-46 terms reconstructing from
        # `new_sparse_text`, and this build's at 42-133 reconstructing from the
        # dense prose, jaccard 0.168 against each other, 0 of 40 matching. The
        # register UID (`CHE341859062`) is in the hook bag and not in the prose,
        # and a query term with no posting cannot match at any weight.
        # `aw_common.record_text_channels` is the declared selection contract
        # ("a card carrying distinct channels must never silently collapse back
        # to its legacy `text` field") and `build_aw_b1._channel_surfaces` is
        # the consumer; the pair is atomic there, so both keys are set together
        # or neither is.
        #
        # SCOPED BY THE CONTRACT, not applied globally. Eight epoch-4 chunk
        # populations also seal a `sparse_text`, distinct from `text` on
        # 30-99% of their rows and a few characters longer (the chunker's
        # breadcrumb-scoped variant) -- reading it everywhere would change the
        # BM25 posting of millions of rows in lanes nobody measured, which is
        # a second unruled representation change, not a fix for the first.
        # A family gets its own sparse channel by DECLARING one:
        # `sparse_surface_field` in data/representation-contract.v1.json
        # (task 528a; its stage-channel probe).
        surface = rec.get("text") or rec.get("dense_text")
        corpus_row = {"text": surface} if surface else None
        sparse_surface = (rec.get("sparse_text") or "").strip()
        if (_sparse_channel_declared and corpus_row is not None
                and sparse_surface and sparse_surface != surface):
            corpus_row["dense_text"] = surface
            corpus_row["sparse_text"] = sparse_surface
            n_sparse_channel += 1
        point, meta = project_record(
            _SrcShim(pid, dict(rec), vecs[i].astype("float32").tolist(),
                     carried_sparse[i]),
            origin=sid, corpus_row=corpus_row,
            published_map=pub_map, conf_map={}, gate=gate, stats=stats,
            metadata_registry=registry)
        if point is None:
            n_excluded += 1
            reasons[str(meta.get("reason", "?")).split(":")[0]] += 1
            continue
        # RULE 2: the sealed id wins. Measure, do not silently accept.
        if str(point["id"]) != str(pid):
            id_divergence += 1
        payload = point["payload"]
        # ADR-103 (task 505 §2.3, task 511 §0): a carried row on a declined
        # or unreadable origin gets an honest `consent_state` and
        # `non_refreshable: true` stamped onto the SERVING payload -- census
        # v34 measured and recorded these 7,935 points but never made them
        # reach the payload (task 505 §7's named gap). Every other row is
        # untouched: this block only ever touches a payload whose `pid` is a
        # key in `adr103_flags`, so a row absent from that dict is
        # byte-identical to what this function produced before this change.
        adr103 = adr103_flags.get(pid)
        if adr103 is not None:
            prior = payload.get("consent_state")  # canonical row
            if prior is not None and prior != adr103["consent_state"]:  # canonical row
                # Not expected on the ADR-103 population (its sealed cards
                # carry no pre-existing consent_state), but if the pinned
                # projection ever DOES set one first (e.g. a future dump- or
                # grace-basis branch), the ADR-103 receipt join is the more
                # authoritative, per-point-resolved verdict and wins -- this
                # is measured and counted, never silently overwritten.
                n_adr103_conflicts += 1
            payload["consent_state"] = adr103["consent_state"]  # canonical row
            payload["non_refreshable"] = True
            adr103_states[adr103["consent_state"]] += 1  # canonical row
            n_adr103_stamped += 1
        if _title_rule is not None:
            n_title_rows += 1
            if _title_rule[0].search(payload.get("title") or ""):
                n_title_match += 1
        if _pub_rule is not None:
            n_pub_rows += 1
            if payload.get("published"):
                n_pub_present += 1
        if _prov_rule is not None:
            n_prov_rows += 1
            if _prov_rule.match(payload.get("snippet") or ""):
                n_prov_head += 1
        row = {"id": pid, "payload": payload}
        sp = (point.get("vector") or {}).get("bm25")
        if sp is not None:
            row["bm25"] = {"indices": list(sp.indices), "values": list(sp.values)}
        else:
            # A row with no posting is invisible to the BM25 leg. It is a legal
            # outcome (a genuinely textless point), never an unremarkable one --
            # task 514 shipped 19,619,656 of them without a line anywhere
            # saying so, because nothing counted the absence.
            n_no_sparse += 1
        rout.write(json.dumps(row, ensure_ascii=False) + "\n")
        vout.write(vecs[i].tobytes())
        kept_ids.append(pid)
        kept += 1

    vout.close()
    rout.close()
    with gzip.open(outdir / f"{sid}.{stem}.ids.jsonl.gz", "wt") as fh:
        for i in kept_ids:
            fh.write(i + "\n")

    receipt = {
        "schema": "aw.binder-stage-receipt.v1",
        "set_id": sid, "shard": stem,
        "input_rows": n, "staged": kept,
        "suppressed": n_suppressed, "excluded_by_projection": n_excluded,
        "exclusion_reasons": dict(reasons),
        "id_divergence_vs_project_record": id_divergence,
        "payload_route": route,
        "payload_missing_in_twin": n_missing_payload,
        "adr103_stamped": n_adr103_stamped,
        "adr103_conflicts": n_adr103_conflicts,
        "adr103_states": dict(adr103_states),
        "sparse_carried_from_twin": n_carried_sparse,
        "sparse_own_channel": n_sparse_channel,
        "sparse_channel_declared": _sparse_channel_declared,
        "staged_without_sparse": n_no_sparse,
        "title_shape": ({"source": _source, "pattern": _title_rule[0].pattern,
                         "rows": n_title_rows, "matched": n_title_match}
                        if _title_rule is not None else None),
        "published_presence": ({"source": _source, "rows": n_pub_rows,
                                "present": n_pub_present}
                               if _pub_rule is not None else None),
        "provenance_head": ({"pattern": _prov_rule.pattern,
                             "rows": n_prov_rows,
                             "rows_with_provenance_head": n_prov_head,
                             "measured_at": "stage"}
                            if _prov_rule is not None else None),
        "consent_mode": args.consent_mode,
        # task 515: how each serving-signal field got its value, per partition.
        # A receipt that cannot say "0 rows reached the collection without a
        # trust_tier" is the receipt task 514 had.
        "serving_metadata": {k: v for k, v in stats.items()
                             if k.startswith("meta_")},
        "serving_metadata_registry": getattr(
            args, "serving_metadata_registry", None) or None,
        "projection_commit": PINNED_PROJECTION_COMMIT,
        "vectors_sha256": sha256_file(outdir / f"{sid}.{stem}.vectors.f16"),
        "payload_sha256": sha256_file(outdir / f"{sid}.{stem}.payload.jsonl.gz"),
        "elapsed_s": round(time.time() - t0, 1),
        "staged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # PER-PARTITION CONSERVATION. Everything that came in must be accounted for
    # by exactly one of: staged, suppressed, excluded. No residue.
    acc = kept + n_suppressed + n_excluded
    if acc != n:
        raise SystemExit(f"{sid}/{stem}: conservation FAIL — {n:,} in but "
                         f"{kept:,} staged + {n_suppressed:,} suppressed + "
                         f"{n_excluded:,} excluded = {acc:,}")
    (outdir / f"{sid}.{stem}.receipt.json").write_text(json.dumps(receipt, indent=1))
    log(f"{sid}/{stem}: {kept:,} staged / {n_suppressed:,} suppressed / "
        f"{n_excluded:,} excluded  ({receipt['elapsed_s']}s)"
        + (f"  [!] {id_divergence:,} id divergences" if id_divergence else "")
        + (f"  [adr103] {n_adr103_stamped:,} stamped {dict(adr103_states)}"
           if n_adr103_stamped else "")
        + (f"  [!] {n_adr103_conflicts:,} adr103 conflicts"
           if n_adr103_conflicts else "")
        + (f"  sparse carried {n_carried_sparse:,}" if n_carried_sparse else "")
        + (f"  [!] {n_no_sparse:,} rows with NO bm25" if n_no_sparse else ""))


def _stage_summary(work):
    outdir = work / "staged"
    per = {}
    for rp in sorted(outdir.glob("*.receipt.json")):
        r = json.loads(rp.read_text())
        d = per.setdefault(r["set_id"], {"input": 0, "staged": 0, "suppressed": 0,
                                         "excluded": 0, "divergent": 0,
                                         "shards": 0, "adr103_stamped": 0,
                                         "adr103_conflicts": 0})
        d["input"] += r["input_rows"]
        d["staged"] += r["staged"]
        d["suppressed"] += r["suppressed"]
        d["excluded"] += r["excluded_by_projection"]
        d["adr103_stamped"] += r.get("adr103_stamped", 0)
        d["adr103_conflicts"] += r.get("adr103_conflicts", 0)
        d["divergent"] += r["id_divergence_vs_project_record"]
        d["shards"] += 1
    (work / "stage-summary.json").write_text(json.dumps(per, indent=1))
    tot = {k: sum(v[k] for v in per.values())
           for k in ("input", "staged", "suppressed", "excluded", "divergent",
                     "adr103_stamped", "adr103_conflicts")}
    log("STAGE SUMMARY")
    for s, v in sorted(per.items()):
        log(f"  {s:44s} in {v['input']:>10,}  staged {v['staged']:>10,}  "
            f"supp {v['suppressed']:>8,}  excl {v['excluded']:>8,}"
            + (f"  adr103 {v['adr103_stamped']:,}" if v["adr103_stamped"] else ""))
    log(f"  {'TOTAL':44s} in {tot['input']:>10,}  staged {tot['staged']:>10,}  "
        f"supp {tot['suppressed']:>8,}  excl {tot['excluded']:>8,}"
        + (f"  adr103 {tot['adr103_stamped']:,}" if tot["adr103_stamped"] else ""))
    if tot["adr103_conflicts"]:
        log(f"  [!] {tot['adr103_conflicts']:,} ADR-103 stamp conflicts "
            f"(pre-existing consent_state overwritten -- see per-shard "
            f"receipts)")
    return per
