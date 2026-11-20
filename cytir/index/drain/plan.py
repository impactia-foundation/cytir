"""Drain plan: the read-only `dry_plan`, plan kwargs, mechanical-content status, persisted-index refresh.

Task 2897 (RULING 80, drain split steps 7-9): moved verbatim out of `cytir/index/promote_queue.py`,
which re-exports every name here. A moved body that calls a name a test patches on the facade (or a
name that stays there) reads it at call time through `_Q` (the facade module).
"""
from __future__ import annotations

import os
import pathlib
import subprocess

from cytir.corpus import persisted as PI
from cytir.gates import mechanical_content_check as MCC
from cytir.index import exclude_file as XF
from cytir.index import id_rule as IR
from cytir.index import payload_profile as PP
from cytir.index import prod_guard as PG
from cytir.index import promote_sealed_pack as PSP

#: Concurrency of plan()'s read-only already-present / already-served lookups
#: per target (task 689). Measured 2026-10-01: the prod-side plan() of a
#: ~141k-row OpenAlex pack took 265-300 s against 55 s on the twin, all of it
#: sequential round trips over the build box->serving box tunnel (283 id
#: retrieves + 553 filtered counts). Twin is local and not RAM-limited; prod
#: gets a modest width because the 30 GB box also serves search.
LOOKUP_WORKERS = {"twin": 8, "prod": PG.PROD_LOOKUP_WORKERS}


# ---------------------------------------------- the persisted-input index
#: The registered-store roots `persisted_inputs.build_index` globs (from
#: `data/persisted-input-stores.v1.json`'s own layer globs, restated here
#: only as a cheap staleness PROBE -- never as a second copy of what the
#: index actually covers; `refresh_persisted_index` always rebuilds through
#: `persisted_inputs.build_index` itself, never a partial/local guess).
PERSISTED_INDEX_ROOTS = ("l0", "l0_direct", "l1", "l2")


def persisted_index_is_stale(index_dir: str, *, corpus_root: str | None = None,
                            timeout_s: int = 300) -> tuple[bool, str]:
    """Whether any file under the registered-store roots is newer than the
    index's own `INDEX.json`.

    Task 641 (2026-09-23): the default shared index was built 2026-09-17,
    six days before LEGI and most of SE were sealed -- every one of their
    pages genuinely IS persisted, but the CACHED index had never seen them,
    so `plan()`'s ADR-105 check refused 82% of that session's enqueued rows
    on a false negative. `find -newer ... -quit` stops at the first hit, so
    the common "nothing changed" case costs a stat-walk (seconds), not a
    full re-parse (the ~2900s the real rebuild measured 2026-09-17 with 12
    workers) -- cheap enough to run before every drain, unconditionally.
    """
    man_path = pathlib.Path(index_dir) / "INDEX.json"
    if not man_path.exists():
        return True, f"{man_path} does not exist"
    croot = corpus_root or PI.corpus_root()
    roots = [os.path.join(croot, r) for r in PERSISTED_INDEX_ROOTS
            if os.path.isdir(os.path.join(croot, r))]
    if not roots:
        return True, f"none of {PERSISTED_INDEX_ROOTS} exist under {croot}"
    result = subprocess.run(
        ["find", *roots, "-newer", str(man_path), "-print", "-quit"],
        capture_output=True, text=True, timeout=timeout_s)
    newer = result.stdout.strip()
    if newer:
        return True, f"{newer} is newer than {man_path}"
    return False, f"no file under {roots} is newer than {man_path}"


def refresh_persisted_index(index_dir: str | None = None, *,
                           force: bool = False, full: bool = False) -> dict:
    """Merge every new-or-changed store file into the sqlite persisted-
    input index. Every `drain()` calls this before touching a single pack.

    Task 664, 2026-09-26: replaced the old "stale? -> full in-memory
    rebuild" scheme. Even the incremental per-FILE scan cache (task
    641/662) still re-gathered and Python-sorted ALL cached records (88.8M)
    into one binary blob on every call -- measured live: ~1h of dead time
    reaching `_sort_dedup_pages`, whether 14 files changed or 0.
    `persisted_inputs.update_index_sqlite` is a PERSISTENT indexed store
    (sqlite, `(layer, key)` primary key): an unchanged file costs one
    indexed lookup by `(path, size, mtime_ns)`, a changed one updates ONLY
    the rows it contributed, and nothing is ever re-sorted -- there is no
    "stale, so rebuild everything" state left, so this now runs
    unconditionally, every drain, and the common case (nothing changed) is
    the same handful of indexed lookups either way.

    `force` is kept for CLI/call-site compatibility and is now a no-op (the
    merge is already cheap enough to always run). `full=True` wipes the
    sqlite tables first -- a genuine from-scratch rebuild, for debugging or
    a registry migration, not the normal path.
    """
    index_dir = index_dir or PI.default_index_dir()
    if full:
        db_path = pathlib.Path(PI.sqlite_index_path(index_dir))
        for p in (db_path, pathlib.Path(str(db_path) + "-wal"),
                 pathlib.Path(str(db_path) + "-shm")):
            if p.exists():
                p.unlink()
        PI.reset_index_cache()  # a cached handle to the deleted file must not survive
    result = PI.update_index_sqlite(index_dir)
    reason = (f"{result['files_changed']} file(s) changed, "
             f"{result['rows_indexed']:,} row(s) merged"
             if result["files_changed"]
             else "no store file changed since the last merge")
    return {"rebuilt": result["files_changed"] > 0, "reason": reason,
           "index_dir": index_dir, "seconds": result["seconds"]}






def plan_kwargs(manifest: dict) -> dict:
    return dict(
        consent=manifest.get("consent"),
        attribution=manifest.get("attribution"),
        license_version=manifest.get("license_version"),
        allow_supersede=manifest.get("allow_supersede"),
        require_persisted=manifest.get("require_persisted", "l1"),
        identity_check=manifest.get("identity_check", "reproduce"),
        identity_reason=manifest.get("identity_reason"),
        published_from=manifest.get("published_from"),
        published_status_from=manifest.get("published_status_from"),
        published_status_requires=manifest.get(
            "published_status_requires", PSP.DEFAULT_PUBLISHED_STATUS_REQUIRES),
        require_published=manifest.get("require_published", False),
        family_payload=manifest.get("family_payload"),
        title_from=tuple(manifest.get("title_from", ())),
        license_from=manifest.get("license_from"),
        family_license=manifest.get("family_license"),
    )


def language_scope_and_waiver_exclusions(
        manifest: dict) -> tuple[dict[str, str], dict | None, dict | None]:
    """Every `exclude_ids` source a manifest can opt into
    (`language_scope_filter`, `lang_presence_waiver`), merged into ONE
    `{point_id: reason}` map, plus the two report dicts `dry_plan()`
    surfaces on its result (`language_scope_filter`, `waiver`) -- the ONE
    place this is derived, called by BOTH `dry_plan()` (the read-only
    proof) and `apply_side()` (the actual write).

    Task 660, 2026-09-25: before this, `apply_side()`'s own `PSP.plan()`
    call (at write time) applied NEITHER exclusion -- only `dry_plan()`
    did. For CC tranche 1 that meant the dry-run (scoped, population
    clears 99%) and the live write (unscoped, population back down to the
    raw 92-94%) computed DIFFERENT populations from the identical
    manifest, so a pack `dry_plan()` had just proved clean refused again
    seconds later inside the locked twin-write transaction -- a
    mirror-contract violation (engineering-doctrine #6: the write must
    reuse the read side's derivation, never re-derive its own). Both
    exclusions are deterministic over the same raw rows, so computing them
    the same way on both sides guarantees the population the write
    actually carries is provably the one the dry-run already proved.
    """
    exclude_map: dict[str, str] = {}
    language_scope_report = None
    if manifest.get("language_scope_filter"):
        raw_rows = PSP.read_records(manifest["pack_chunks"])
        language_scope_report = MCC.language_scope_filter(raw_rows)
        for pid in language_scope_report["out_of_scope_ids"]:
            exclude_map[str(pid)] = "out_of_scope_language"
    lang_presence_waiver = manifest.get("lang_presence_waiver")
    waiver_report = None
    if lang_presence_waiver:
        raw_rows2 = PSP.read_records(manifest["pack_chunks"])
        drop_detail = MCC.lang_drop_detail(raw_rows2)
        for entry in drop_detail:
            exclude_map.setdefault(entry["point_id"], "no_confident_lang")
        waiver_report = {
            **lang_presence_waiver,
            "dropped_count": len(drop_detail),
            "dropped_rows": drop_detail,
        }
    if manifest.get("exclude_ids_file"):
        # Task 680 (OpenAlex ADR-102 ladder trim): a sha-pinned list of point
        # ids the admission floor does not admit. `setdefault`, so a more
        # specific reason a language rule already gave an id is kept. The
        # dropped rows are reported by `plan()` with this reason (ADR-086).
        for pid, why in XF.load_exclude_ids(
                manifest["exclude_ids_file"],
                sha256=manifest.get("exclude_ids_sha256"),
                reason=manifest.get("exclude_ids_reason")).items():
            exclude_map.setdefault(pid, why)
    return exclude_map, language_scope_report, waiver_report


def plan_kwargs_with_scope(
        manifest: dict) -> tuple[dict, dict | None, dict | None]:
    """`plan_kwargs()` plus every `exclude_ids` source this manifest opts
    into (see `language_scope_and_waiver_exclusions`'s docstring) -- the
    ONE place `exclude_ids` is derived for a `plan()` call, so the
    read-only proof and the write never disagree about which rows are in
    scope."""
    kwargs = plan_kwargs(manifest)
    exclude_map, language_scope_report, waiver_report = \
        language_scope_and_waiver_exclusions(manifest)
    if exclude_map:
        kwargs["exclude_ids"] = exclude_map
    return kwargs, language_scope_report, waiver_report



def dry_plan(manifest: dict, url: str, *, collection: str | None = None,
             persisted_check: bool = True) -> dict:
    """A read-only `promote_sealed_pack.plan()` — proves the mechanical
    checks (counts reconcile, consent+licence per row, dedup/supersede
    clean) without writing anything. This is what a "battery-required" pack
    gets instead of a write: proof it WOULD clear, not a promotion.

    Keeps `_joined`/`_matrix`/`_id_index` in the result (unlike `promote_
    sealed_pack.main()`'s own use of `plan()`) -- `drain()` needs `_joined`
    to run the mechanical-content-v1 check against the real dense text, and
    pops all three itself before the receipt is written to disk.

    `manifest["language_scope_filter"] = true` (orchestrator ruling,
    2026-09-23, CC tranche 1, task 641): OPT-IN per family, not automatic
    for every pack -- a second fasttext pass over every raw row before
    `plan()` even runs is real per-row cost this module will not spend
    unasked, mirroring the coverage check's own opt-in shape. When set,
    runs `mechanical_content_check.language_scope_filter` over the pack's
    RAW rows and passes the out-of-scope ids into `plan(exclude_ids=...)`
    so they are gone before the payload-carry population census ever scores
    them -- for a MULTI-LANGUAGE family, "language matches" means the
    detected language is in the product's set (EN/DE/FR/IT), and a row
    detected outside it is out of scope, not a content defect.

    `manifest["lang_presence_waiver"]` (Tim, 2026-09-24, CC tranche 1, task
    652): a per-pack, Tim-issued waiver of the `lang` payload-carry
    presence floor, `{issuer, at, scope, rule}`. NOT a new default --
    OPT-IN per pack, same shape as `language_scope_filter` above, and it
    covers exactly one thing: rows the tracked detector could not place
    confidently in the product's language set AT ALL (empty `lang` even
    after `normalize_sealed_pack_shape.py --backfill-lang` ran), never a
    row that declares a lang or one the detector confidently classified.
    When set, `mechanical_content_check.lang_drop_detail` runs over the
    pack's RAW rows and every one of THOSE ids is excluded from `plan()`'s
    population too (merged into the same `exclude_ids` mapping, with its
    own `"no_confident_lang"` reason -- a row already excluded as
    `out_of_scope_language` keeps that more specific reason). The full
    per-row detail (id, url, text length, detector score) rides in the
    returned `waiver` block so the promotion receipt is the accountability
    record Tim asked for, not just a count.
    """
    from cytir.index import promote_queue as _Q  # noqa: PLC0415 -- patched names resolve at call time
    kwargs, language_scope_report, waiver_report = _Q.plan_kwargs_with_scope(manifest)
    if persisted_check is not True:       # only a Single transaction ever sets this
        kwargs["persisted_check"] = persisted_check
    result = PSP.plan(chunks_dir=manifest["pack_chunks"],
                      vectors_dir=manifest["vectors"], qdrant=url,
                      collection=collection or manifest["collection"], mrl=int(manifest["mrl"]),
                      lookup_workers=LOOKUP_WORKERS["twin"], **kwargs)
    if language_scope_report is not None:
        result["language_scope_filter"] = {
            k: v for k, v in language_scope_report.items()
            if k != "out_of_scope_ids"} | {
            "out_of_scope_ids_count": len(language_scope_report["out_of_scope_ids"])}
    if waiver_report is not None:
        result["waiver"] = waiver_report
    return result


def mechanical_content_rows(joined: dict) -> list[dict]:
    """`joined` (from `plan()`, `{point_id: (dense_text, sparse_text,
    payload, rec)}`) -> the merged row shape `mechanical_content_check.
    evaluate()` needs. The payload alone carries only a 25-word extractive
    `snippet` (ADR-039 minimum-content) -- the real word count needs the
    dense embed text merged back in, under `canonical_text` (matching
    `promote_sealed_pack.py`'s own field name for it)."""
    return [{**payload, "point_id": pid, "canonical_text": dense_text}
           for pid, (dense_text, _sparse, payload, _rec) in joined.items()]


def mechanical_content_status(manifest: dict, joined: dict) -> dict:
    """Run `mechanical_content_check.evaluate()` over this pack's real
    joined rows. Manifest keys `word_floor`/`coverage_floor`/
    `fail_share_max`/`l1_sources`/`detect_language` are all optional
    per-pack overrides of MCC's defaults."""
    rows = mechanical_content_rows(joined)
    kwargs = {}
    for key in ("word_floor", "coverage_floor", "fail_share_max",
               "l1_sources", "detect_language"):
        if key in manifest:
            kwargs[key] = manifest[key]
    return MCC.evaluate(rows, **kwargs)



def payload_profile_preflight_both(twin_collection: str, prod_collection: str,
                                   payloads) -> dict:
    """Task 675: prove, BEFORE the twin leg, that every row projects for BOTH
    targets' declared payload profiles. Twin and prod may differ (aw_e8 raw, aw_e9
    normalized), and a refusal found at the prod leg would leave the twin ahead.
    `payloads` is consumed once per target (pass a re-iterable, not a generator,
    if it must be reused: this materialises it)."""
    # Task 864 (review 869 M1): each leg's `plan()` mints the id ITS collection declares
    # (`id_rule`; aw_e10: I1). Two legs with different rules land disjoint id sets (and the
    # twin's drop set misses prod), and check 5 compares counts only, so the pack would land
    # DONE. Refused here, before the twin leg: nothing is written. Today neither leg declares a
    # rule (aw_e8, aw_e9), so this never fires before the aw_e10 flip commit.
    twin_rule, prod_rule = IR.declared_id_rule(twin_collection), IR.declared_id_rule(prod_collection)
    if twin_rule != prod_rule:
        raise PP.PayloadProfileError(
            f"REFUSED: the twin {twin_collection!r} declares id_rule {twin_rule!r} and prod "
            f"{prod_collection!r} declares {prod_rule!r}: the two legs would write different ids "
            "for one row. Declare the same id_rule for both (the aw_e10 flip commit, F7)")
    rows = list(payloads)
    return {role: PSP.payload_profile_preflight(collection, rows)
            for role, collection in (("twin", twin_collection),
                                     ("prod", prod_collection))}
