"""Family pack -> sealed chunk stage -> L1 filing, for atomic-record families (task 696).

THE VERB. `cytir.sources.run_family` ends at a pack (`pack/manifest.json` + `records.jsonl`, schema
`aw.family-pack.v1`). The promotion queue starts at a sealed CHUNK STAGE
(`<stage>/chunks/shard-NNNN.texts.jsonl.gz`, one row per point, `point_id` + `text` + `url`)
whose pages are filed into the registered L1 tree (ADR-105, `require=l1`). This module is the
bridge, so a family that declares `chunk.atomic` (one record = one point; the record IS the
chunk) reaches `aw-growth-pending-compose` by the same road as OpenLegalData and Stack
Overflow:

    run_family -> pack  --stage-->  stages/<slug>/chunks-shardNN  (+ L1 release, + overlap report)
               --aw.ops.growth_pending_compose (embed -> compose -> enqueue)--> promotion queue

A stage is a slice of at most `MAX_SHARD_ROWS` (150,000) rows, so one queue pack stays inside
the 1,000,000-row queue cap with room to spare and one embed run is ~40 minutes.

WHAT IS CHECKED BEFORE A BYTE IS STAGED (corpus-doctrine: name the stage with every count)
  * manifest schema + family id, terminals balanced (`in = admitted + excluded + quarantined`),
    `records.count > 0`, and `records.jsonl` re-hashed against the manifest's sha256;
  * every row carries `url`, `text`, `point_id`, and its own non-empty `license` (ADR-039:
    the per-record licence ships in the dump; it is never filled in downstream);
  * the contract's declared consent basis is not `unknown` and the contract has no unknowns.

CONSENT is NOT decided here. It is stamped at compose from the family's declared basis
(`cytir.consent.derive`, ADR-115 r5 for the DILA provenances, ruling 17(5)); a row the engine
cannot stamp refuses the pack. This module only refuses to stage a pack whose contract is
unresolved.

OVERLAP is measured, never dropped silently. `overlap` asks the twin and prod, by id and by url
(indexed retrieves and counts; never a scan), what they already serve of each stage and writes
`OVERLAP.json` with the counts, plus `overlap-ids.jsonl` when any row is to be excluded.
`cytir.ops.growth_pending_compose` passes that file to compose as the declared `--exclude-ids-file` (the #495
path, reason recorded) and REFUSES a stage that has no `OVERLAP.json` (somebody looked) or whose
`needs_ruling` is true (a url share above the drain's 5% ceiling with no exclusion declared: the
drain would refuse the whole pack). Rows whose url is served under another id are excluded only
when the operator runs `overlap --exclude-served-urls`; whether that may become automatic is Tim's
ruling, not this module's.

CLI: `python -m cytir.sources.family_stage {stage,overlap,status} ...`
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import argparse
import gzip
from cytir.acquire.gzip_pinned import deterministic_gzip_open
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Callable

from cytir.core import twin as TWIN
from cytir.core.canon import strip_tracking_url  # noqa: F401
from cytir.sources import url_admission as UA
from cytir.core.prose_guard import binary_content
from cytir.index import prod_guard as PG
from cytir.index import promote_sealed_pack as PSP
from cytir.sources import staged_text_guard
from cytir.sources.family import CHUNK_NATIVE, CHUNK_NATIVE_CHUNKED

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_SCHEMA = "aw.family-pack.v1"
STAGE_SCHEMA = "aw.family-stage.v1"
OVERLAP_SCHEMA = "aw.family-stage-overlap.v1"
MAX_SHARD_ROWS = 150_000
OVERLAP_REASON = "already_served_id_at_stage_time_task696"
OVERLAP_REASON_URL = "already_served_id_or_url_at_stage_time_task696"
_RETRIEVE_BATCH = 500


class StageError(RuntimeError):
    """Refused: nothing (more) is staged and the message says why."""


# ----------------------------------------------------------------------------- names

def slug(family_id: str) -> str:
    """`dila_juri` -> `dila-juri-v1` (the stage tree, the vectors release, the pack ids)."""
    return family_id.replace("_", "-") + "-v1"


def stages_root(corpus_root: str | Path, family_id: str) -> Path:
    return Path(corpus_root) / "stages" / slug(family_id)


def stage_dir(corpus_root: str | Path, family_id: str, shard: int) -> Path:
    return stages_root(corpus_root, family_id) / f"chunks-shard{shard}"


def pack_id(family_id: str, shard: int) -> str:
    return f"{slug(family_id)}-shard{shard}"


def vectors_dir(corpus_root: str | Path, family_id: str, shard: int) -> Path:
    return (Path(corpus_root) / "vectors" / f"release=aw-growth-{slug(family_id)}"
            / f"source={pack_id(family_id, shard)}")


def l1_release(family_id: str, shard: int) -> str:
    return pack_id(family_id, shard)


def is_l1_filed(corpus_root: str | Path, family_id: str, shard: int) -> bool:
    return (Path(corpus_root) / "l1" / f"release={l1_release(family_id, shard)}"
            / "_register_manifest.json").is_file()


def list_stages(corpus_root: str | Path, family_id: str) -> list[tuple[int, Path]]:
    """The roster is the TREE (census-roster-from-manifest), never a typed list."""
    root = stages_root(corpus_root, family_id)
    if not root.is_dir():
        return []
    out = []
    for d in root.iterdir():
        name = d.name
        if d.is_dir() and name.startswith("chunks-shard") and name[12:].isdigit():
            out.append((int(name[12:]), d))
    return sorted(out)


# ----------------------------------------------------------------------------- hashing

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_gz_deterministic(path: Path, lines: list[bytes]) -> None:
    with open(path, "wb") as raw:
        with deterministic_gzip_open(raw, compresslevel=6) as gz:
            for ln in lines:
                gz.write(ln)


# ----------------------------------------------------------------------------- pack checks

#: The suffix a split article's anchor carries, for the native families only: loda
#: `article:<id>:chunk-<n>` (cytir.extract.legi; anchored on `article:` so a Stack Overflow
#: `...:chunk-<n>` anchor is not stripped), fedlex `<eId>-c<n>` (families/fedlex.window).
#: The door calls `article_row_counts` for the `native` / `native-chunked` policies only.
_SPLIT_SUFFIX = re.compile(r"^(article:.*):chunk-\d+$|(-c\d+)$")


def article_row_counts(records_path: str | Path) -> dict:
    """How many rows each source article became in a pack: `{articles, split_articles,
    rows_in_split, examples}`. An article is one `(url without fragment, base anchor)`; the base
    anchor is the row's `base_anchor` (fedlex), else its `anchor` with the split suffix removed,
    else the url fragment. A row with none of them counts as its own article (an atomic record)."""
    per: dict[tuple[str, str], int] = {}
    with open(records_path, "rb") as fh:
        for n, raw in enumerate(fh):
            if not raw.strip():
                continue
            row = json.loads(raw)
            url = row.get("url") or ""
            base = row.get("base_anchor") or _SPLIT_SUFFIX.sub(
                r"\1", row.get("anchor") or url.partition("#")[2])
            key = ((url.partition("#")[0], base) if base
                   else (url, row.get("point_id") or f"row:{n}"))
            per[key] = per.get(key, 0) + 1
    split = {k: n for k, n in per.items() if n > 1}
    return {"articles": len(per), "split_articles": len(split),
            "rows_in_split": sum(split.values()),
            "examples": [f"{u}#{a} ({n} rows)" for (u, a), n in sorted(split.items())[:5]]}


def chunk_policy_refusal(declared: str | None, counts: dict) -> str | None:
    """The door's reading of a pack's chunk policy. `native` means one row per article, so a pack
    that holds several rows for one article is refused; `native-chunked` may (an over-window
    article is split, never cut). Any other policy is not judged here. Returns text or None."""
    if declared == CHUNK_NATIVE and counts["split_articles"] > 0:
        return (f"chunk policy 'native' is one row per article, but {counts['split_articles']} "
                f"article(s) became {counts['rows_in_split']} rows (e.g. {counts['examples']!r}). "
                "Declare `native-chunked` (POLICY PR; POLICY.md §4.1) or fix the extractor")
    return None


def check_pack(pack_dir: str | Path, family_id: str) -> dict:
    """Validate the family pack and return its manifest. Raises `StageError` on any failure."""
    pack = Path(pack_dir)
    mpath, rpath = pack / "manifest.json", pack / "records.jsonl"
    if not mpath.is_file() or not rpath.is_file():
        raise StageError(f"{pack}: manifest.json / records.jsonl missing -- a run that did not "
                         "finish has no pack (FULL.rc must be 0)")
    man = json.loads(mpath.read_text(encoding="utf-8"))
    if man.get("schema") != PACK_SCHEMA:
        raise StageError(f"{mpath}: schema {man.get('schema')!r}, expected {PACK_SCHEMA!r}")
    if man.get("family_id") != family_id:
        raise StageError(f"{mpath}: family_id {man.get('family_id')!r} != {family_id!r}")
    t = man.get("terminals") or {}
    if t.get("units_in") != (t.get("units_admitted", 0) + t.get("units_excluded", 0)
                             + t.get("units_quarantined", 0)):
        raise StageError(f"{mpath}: terminals do not balance: {t!r}")
    n = (man.get("records") or {}).get("count", 0)
    if n <= 0:
        raise StageError(f"{mpath}: 0 records -- a run that admits nothing is a wrong input, not "
                         "an empty source")
    contract = man.get("contract") or {}
    if contract.get("unknowns"):
        raise StageError(f"{mpath}: contract unknowns {contract['unknowns']!r}; declare them first")
    if not (contract.get("consent_basis") or "").strip():
        raise StageError(f"{mpath}: contract carries no consent basis")
    if not (man.get("consent") or {}).get("checked"):
        raise StageError(f"{mpath}: consent was not checked for this run (offline replay); "
                         "such a pack cannot be staged")
    want = (man.get("records") or {}).get("sha256")
    got = _sha256_file(rpath)
    if want != got:
        raise StageError(f"{rpath}: sha256 {got} != manifest {want}; the pack was altered")
    refusal = staged_text_guard.door_refusal(man.get("staged_text_guard"))
    if refusal:                       # a pack staged before the guard has no key and passes
        raise StageError(f"{mpath}: {refusal}")
    declared = (contract.get("chunk") or {}).get("declared")
    if declared in (CHUNK_NATIVE, CHUNK_NATIVE_CHUNKED):     # a pack with no chunk key passes
        counts = article_row_counts(rpath)
        refusal = chunk_policy_refusal(declared, counts)
        if refusal:
            raise StageError(f"{mpath}: {refusal}")
        man["door_article_rows"] = {"policy": declared, **counts}   # in memory: reported, not stored
    return man


# ----------------------------------------------------------------------------- stage

def stage_family_pack(pack_dir: str | Path, *, corpus_root: str | Path, family_id: str,
                      shard_rows: int = MAX_SHARD_ROWS, first_shard: int = 0,
                      drop_served: dict[str, tuple[str, str]] | None = None,
                      python_bin: str | None = None,
                      retrieve: Callable | None = None, served_urls: Callable | None = None,
                      runner: Callable = subprocess.run, log: Callable = print,
                      single_slice: bool = False) -> list[dict]:
    """Cut the pack into slices of <= `shard_rows` rows, stage each slice, file its L1, return a
    receipt per slice. Idempotent: a slice already staged from THIS pack (same source sha and
    `row_start`) is reused, never rewritten; a different slicing of the same pack refuses. A new
    stage takes the next free number, never below `first_shard`, so a later pack of the same family
    (an increments-only run) continues the numbering and every stage keeps one number for good.

    `drop_served` (`name -> (url, collection)`, e.g. twin and prod): rows whose id or url the target
    already serves are NOT staged, so the GPU never embeds a page the index already holds. Every
    dropped id is written to the stage's `dropped-served-ids.jsonl` (sha-pinned in `PACK.json`) and
    counted in the receipt: staged + dropped == the slice, always. Without it nothing is dropped
    here and `overlap` declares the exclusion at compose instead.

    `single_slice` (ruling 27 T6): refuse a pack that would be cut into more than one slice. A
    restage pack (the run's option `only_split` is true) is always one slice: its stage-guard block
    is pack-level, and a url-group supersede checks `docs_checked` against the URL groups of the
    ONE pack it drains (task 712 follow-up 1)."""
    if not 0 < shard_rows <= MAX_SHARD_ROWS:
        raise StageError(f"shard_rows must be in 1..{MAX_SHARD_ROWS}")
    man = check_pack(pack_dir, family_id)
    if man.get("door_article_rows"):
        d = man["door_article_rows"]
        log(f"[family-stage] {family_id}: chunk policy {d['policy']}: {d['articles']:,} articles, "
            f"{d['split_articles']:,} split into {d['rows_in_split']:,} rows")
    total = man["records"]["count"]
    restage = str(((man.get("run") or {}).get("options") or {}).get("only_split", "")
                  ).strip().lower() in ("1", "true", "yes")
    if restage and drop_served:
        raise StageError("--drop-served refuses a restage pack (only_split): it drops every row "
                         "whose URL is already served, which is every row of a restage")
    if (single_slice or restage) and total > shard_rows:
        raise StageError(
            f"{pack_dir}: {total:,} rows would be staged in {-(-total // shard_rows)} slices of "
            f"{shard_rows:,}; a restage / url-group pack is ONE slice (its guard block is "
            "pack-level). Narrow the selector; never re-cut")
    rpath = Path(pack_dir) / "records.jsonl"
    held: list[tuple[str, str, str]] = []             # task 777: counted per-row hold-out
    rows_seen = _validate_rows(rpath, held)
    held_ids = {i for i, _u, _w in held}
    if rows_seen != total:
        raise StageError(f"{rpath}: {rows_seen} rows read, manifest says {total}")
    stages_root(corpus_root, family_id).mkdir(parents=True, exist_ok=True)
    manifest_sha = _sha256_file(Path(pack_dir) / "manifest.json")
    source_sha = man["records"]["sha256"]
    mine: dict[int, int] = {}                  # row_start -> shard, stages of THIS pack
    last = first_shard - 1
    for n, d in list_stages(corpus_root, family_id):
        last = max(last, n)
        pj = d / "PACK.json"
        if pj.is_file():
            have = json.loads(pj.read_text())
            if have.get("source_records_sha256") == source_sha:
                if have.get("slice_rows") not in (None, shard_rows):
                    raise StageError(f"{d}: this pack was staged with slices of "
                                     f"{have['slice_rows']} rows, not {shard_rows}; never re-cut")
                mine[have["row_start"]] = n
    receipts: list[dict] = []
    buf: list[bytes] = []
    row0 = 0
    totals = {"staged": 0, "dropped": 0, "refused": 0}
    if held:
        _write_refused_binary(corpus_root, family_id, source_sha, held)
        log(f"[family-stage] {family_id}: {len(held):,} row(s) held out as not prose "
            f"({held[0][2]}); first: {held[0][0]} {held[0][1]}")
    next_shard = last + 1

    def flush() -> None:
        nonlocal row0, buf, next_shard
        lines, dropped, refused = buf, [], []
        if row0 not in mine:
            # task 2713: a url the source never published as a page is REFUSED, not "already served"
            lines, refused = drop_refused_urls(buf)
            # task 767: a tracking/session variant never joins a page it duplicates
            lines, dropped = drop_url_variants(lines)
        if drop_served and row0 not in mine:
            lines, more = _split_served(lines, drop_served, retrieve, served_urls)
            dropped = dropped + more
        if row0 in mine:
            shard = mine[row0]
        elif lines:
            shard, next_shard = next_shard, next_shard + 1
        else:
            shard = None
        if shard is not None:
            receipts.append(_publish_stage(
                corpus_root=corpus_root, family_id=family_id, shard=shard, lines=lines,
                dropped=dropped, refused=refused, row_start=row0, source_rows=len(buf),
                slice_rows=shard_rows,
                source_manifest_sha=manifest_sha, source_records_sha=source_sha,
                python_bin=python_bin, runner=runner, log=log,
                pack_guard=man.get("staged_text_guard")))
            totals["staged"] += receipts[-1]["rows"]
            totals["dropped"] += receipts[-1]["dropped_served"]
            totals["refused"] += sum(receipts[-1]["refused_urls"].values())
        else:
            totals["dropped"] += len(dropped)
            totals["refused"] += len(refused)
            _log_refused(log, refused)
            log(f"[family-stage] slice at row {row0:,}: no row staged ({len(dropped):,} already served "
                f"or variants, {len(refused):,} refused urls); no stage")
        row0 += len(buf)
        buf = []

    with open(rpath, "rb") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            if held_ids and json.loads(raw)["point_id"] in held_ids:
                continue
            buf.append(raw if raw.endswith(b"\n") else raw + b"\n")
            if len(buf) >= shard_rows:
                flush()
    if buf:
        flush()
    if totals["staged"] + totals["dropped"] + totals["refused"] + len(held) != total:
        raise StageError(f"conservation: staged {totals['staged']} + dropped-served "
                         f"{totals['dropped']} + refused-urls {totals['refused']} + "
                         f"refused-binary {len(held)} != pack {total}")
    for r in receipts if held else ():                # absent on a clean pack: receipts unchanged
        r["refused_binary"] = len(held)               # pack-level count; ids in the pack's file
    log(f"[family-stage] {family_id}: pack {total:,} rows -> {len(receipts)} stage(s) "
        f"(<= {shard_rows:,} rows each), {totals['staged']:,} staged, {totals['dropped']:,} dropped "
        f"(already served); refused at the stage door: {totals['refused']:,} urls; L1 filed")
    return receipts


def _write_refused_binary(corpus_root, family_id: str, source_sha: str,
                          held: "list[tuple[str, str, str]]") -> Path:
    """The pack's counted terminal for rows that are not prose: one file per source pack, beside
    the stages (not a stage dir, so `list_stages` never sees it). Deterministic; rewriting it on a
    re-run writes the same bytes."""
    root = stages_root(corpus_root, family_id)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"refused-binary-{source_sha[:16]}.jsonl"
    path.write_text("".join(json.dumps({"point_id": i, "url": u, "why": w}, sort_keys=True) + "\n"
                            for i, u, w in sorted(held)))
    return path


def drop_refused_urls(buf: list[bytes]) -> tuple[list[bytes], list[dict]]:
    """`(rows kept, refused)`: rows whose url `cytir.sources.url_admission.refusal` names (a search-result
    page, the site chrome of the statute host). Task 2713. `refused` is `{point_id, url, reason}` per row:
    a refusal is counted by reason and never reported as "already served"."""
    kept, refused = [], []
    for b in buf:
        r = json.loads(b)
        why = UA.refusal(r["url"])
        if why:
            refused.append({"point_id": r["point_id"], "url": r["url"], "reason": why})
        else:
            kept.append(b)
    return kept, refused


def _refused_counts(refused: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in refused:
        out[r["reason"]] = out.get(r["reason"], 0) + 1
    return dict(sorted(out.items()))


def _log_refused(log, refused: list[dict]) -> None:
    if refused:
        by = " ".join(f"{k}={v}" for k, v in _refused_counts(refused).items())
        log(f"[family-stage] refused {len(refused):,} urls at the stage door: {by}")


def drop_url_variants(buf: list[bytes], served_hits: "set[str] | None" = None
                      ) -> tuple[list[bytes], list[str]]:
    """`(rows kept, ids dropped)`: the stage door's duplicate-variant guard (task 767, defect A).

    A VARIANT row is one whose url differs from `strip_tracking_url(url)` (`?authuser=1`,
    `?utm_*`, `;jsessionid`, ...). It is dropped when the canonical form of its url is already
    held: by another row of this slice, or in `served_hits` (urls a target serves). When no
    canonical row exists, ONE variant spelling per canonical url is kept (the first seen) and the
    other spellings of that page are dropped, so a page crawled per account index enters once.
    Rows of one kept spelling (the chunk rows of one page) all stay. Drops are ids, counted by the
    caller as `dropped_served` (conservation: staged + dropped == slice)."""
    served_hits = served_hits or set()
    rows = [json.loads(b) for b in buf]
    canon = {r["url"]: UA.fold_url(r["url"]) for r in rows}
    held = {u for u in canon if canon[u] == u}               # canonical spellings in this slice
    keep_spelling: dict[str, str] = {}                       # canonical url -> the kept variant
    kept, dropped = [], []
    for b, r in zip(buf, rows):
        u, c = r["url"], canon[r["url"]]
        if c == u:
            kept.append(b)
        elif c in held or c in served_hits or keep_spelling.setdefault(c, u) != u:
            dropped.append(r["point_id"])
        else:
            kept.append(b)
    return kept, dropped


def _split_served(buf: list[bytes], targets: dict[str, tuple[str, str]], retrieve, served_urls
                  ) -> tuple[list[bytes], list[str]]:
    """`(rows to stage, ids dropped)`: a row is dropped when a target serves its id or its url,
    or (task 767) when it is a tracking-variant of a url a target serves."""
    retrieve = retrieve or _retrieve_present
    served_urls = served_urls or _served_urls
    rows = [json.loads(b) for b in buf]
    ids = [r["point_id"] for r in rows]
    urls = sorted({r["url"] for r in rows} | {UA.fold_url(r["url"]) for r in rows})
    ids_hit: set[str] = set()
    urls_hit: set[str] = set()
    for _name, (turl, collection) in sorted(targets.items()):
        for i in range(0, len(ids), _RETRIEVE_BATCH):
            ids_hit |= retrieve(turl, collection, ids[i:i + _RETRIEVE_BATCH])
    for hit in _ask_urls(targets, urls, served_urls)[0].values():
        urls_hit |= hit
    var_keep, var_dropped = drop_url_variants(buf, urls_hit)
    var_dropped = set(var_dropped)
    gone = lambda r: (r["point_id"] in ids_hit or r["url"] in urls_hit  # noqa: E731
                      or r["point_id"] in var_dropped)
    keep = [b for b, r in zip(buf, rows) if not gone(r)]
    dropped = [r["point_id"] for r in rows if gone(r)]
    return keep, dropped


def _validate_rows(rpath: Path, held: "list[tuple[str, str, str]] | None" = None) -> int:
    """One pass BEFORE anything is written: shape, per-record licence, unique ids.

    A row that is a binary file read as text (`cytir.core.prose_guard`) is NOT a pack defect (ADR-094:
    quarantine the row, never abort the family; task 777). With `held` given, each such row is
    appended to it as `(point_id, url, why)` and the pass goes on; without it (the one-shot
    `stage_out` tool) the pack is refused as before."""
    seen: set[str] = set()
    binary: list[tuple[str, str, str]] = []
    n = 0
    with open(rpath, "rb") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            row = json.loads(raw)
            n += 1
            for key in ("url", "text", "point_id"):
                if not row.get(key):
                    raise StageError(f"{rpath} row {n}: no {key!r}")
            if not str(row.get("license") or "").strip():
                raise StageError(f"{rpath} row {n} ({row['url']}): no per-record `license` "
                                 "(ADR-039); nothing is staged")
            if row["point_id"] in seen:
                raise StageError(f"{rpath} row {n}: duplicate point_id {row['point_id']}")
            seen.add(row["point_id"])
            why = binary_content(str(row["text"]))      # task 767: a binary file read as text
            if why:
                binary.append((row["point_id"], row["url"], why))
    if held is not None:
        held.extend(binary)
    elif binary:
        raise StageError(
            f"{rpath}: {len(binary)} row(s) are not prose ({binary[0][2]}); nothing is staged. "
            "first: " + "; ".join(f"{i} {u}" for i, u, _w in binary[:5]))
    return n


def _publish_stage(*, corpus_root, family_id, shard, lines, dropped, row_start, source_rows,
                   slice_rows, source_manifest_sha, source_records_sha, python_bin, runner,
                   log, pack_guard=None, refused=()) -> dict:
    final = stage_dir(corpus_root, family_id, shard)
    pack_json = final / "PACK.json"
    meta = {"schema": STAGE_SCHEMA, "family_id": family_id, "pack_id": pack_id(family_id, shard),
            "shard": shard, "row_start": row_start, "source_rows": source_rows,
            "slice_rows": slice_rows, "rows": len(lines), "dropped_served": len(dropped),
            "refused_urls": _refused_counts(list(refused)),
            "source_manifest_sha256": source_manifest_sha,
            "source_records_sha256": source_records_sha,
            "l1_release": l1_release(family_id, shard)}
    if pack_guard is not None:       # pack-level counts (task 706); absent on a pre-guard pack
        meta["pack_staged_text_guard"] = pack_guard
    if pack_json.is_file():
        have = json.loads(pack_json.read_text())
        for k in ("source_records_sha256", "row_start", "source_rows"):
            if have.get(k, meta[k] if k == "source_rows" else None) != meta[k]:
                raise StageError(f"{final}: existing stage differs from this pack on {k!r} "
                                 f"({have.get(k)!r} vs {meta[k]!r}); never overwritten")
        meta = have
        log(f"[family-stage] {meta['pack_id']}: stage exists ({have['rows']:,} rows)")
    else:
        tmp = final.with_name(final.name + ".part")
        if tmp.exists():
            raise StageError(f"{tmp}: a half-written stage is never overwritten; remove it by hand")
        (tmp / "chunks").mkdir(parents=True)
        shard_path = tmp / "chunks" / "shard-0000.texts.jsonl.gz"
        _write_gz_deterministic(shard_path, lines)
        meta["chunks_sha256"] = _sha256_file(shard_path)
        if dropped:
            dpath = tmp / "dropped-served-ids.jsonl"
            dpath.write_text("".join(json.dumps({"point_id": i}) + "\n" for i in sorted(dropped)))
            meta["dropped_served_sha256"] = _sha256_file(dpath)
        if refused:
            rpath_ = tmp / "refused-urls.jsonl"
            rpath_.write_text("".join(json.dumps(r, sort_keys=True) + "\n"
                                      for r in sorted(refused, key=lambda r: r["point_id"])))
            meta["refused_urls_sha256"] = _sha256_file(rpath_)
        (tmp / "PACK.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
        tmp.rename(final)        # published by rename: PACK.json never describes a partial dir
        log(f"[family-stage] {meta['pack_id']}: staged {len(lines):,} rows"
            + (f", dropped {len(dropped):,} already served" if dropped else ""))
        _log_refused(log, list(refused))
    if not is_l1_filed(corpus_root, family_id, shard):
        _file_l1(final, corpus_root, family_id, shard, python_bin=python_bin, runner=runner,
                 log=log)
    return {"pack_id": meta["pack_id"], "shard": shard, "rows": meta["rows"],
            "dropped_served": meta.get("dropped_served", 0),
            "refused_urls": meta.get("refused_urls", {}), "stage": str(final),
            "l1_release": meta["l1_release"]}


def _file_l1(stage: Path, corpus_root, family_id: str, shard: int, *, python_bin, runner,
             log) -> None:
    """ADR-105: the drain refuses a page with no L1 record. Subprocess, like every other family
    (a fresh process, `python -m`); the copy is sha-verified by the tool."""
    release = l1_release(family_id, shard)
    argv = [python_bin or sys.executable, "-m", "cytir.corpus.register_l1_release",
            "--source", str(stage / "chunks"), "--file-suffix", ".texts.jsonl.gz",
            "--release", release, "--corpus-root", str(corpus_root)]
    log(f"[family-stage] $ {' '.join(argv[-6:])}")
    res = runner(argv, env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)), cwd=str(REPO_ROOT))
    if res.returncode != 0:
        raise StageError(f"L1 filing of {release} failed rc={res.returncode}")
    if not is_l1_filed(corpus_root, family_id, shard):
        raise StageError(f"L1 filing of {release} left no _register_manifest.json")


# ----------------------------------------------------------------------------- overlap

#: The drain refuses a pack WHOLE when more than this share of its distinct urls is already served
#: under other point ids (`promote_sealed_pack.ALREADY_SERVED_FAIL_SHARE_MAX`); below it the
#: drain drops those rows itself. Mirrored, not restated: one number, one home.
URL_SHARE_MAX = PSP.ALREADY_SERVED_FAIL_SHARE_MAX


def _retrieve_present(url: str, collection: str, ids: list[str], opener=IH.urlopen
                      ) -> set[str]:
    PG.check(url, "POST", f"/collections/{collection}/points", collection=collection)
    req = urllib.request.Request(
        f"{url.rstrip('/')}/collections/{collection}/points",
        data=json.dumps({"ids": ids, "with_payload": False, "with_vector": False}).encode(),
        headers={"content-type": "application/json"}, method="POST")
    with opener(req, timeout=120) as resp:
        body = json.load(resp)
    PG.pace(url, collection)
    return {str(p["id"]) for p in body["result"]}


def _ask_urls(targets: dict[str, tuple[str, str]], urls: list[str], served_urls: Callable
              ) -> tuple[dict[str, set[str]], list[str]]:
    """`({target: urls it serves}, [targets asked])` for the "is this URL served" question.

    Prod is NEVER asked (incident 2026-10-07: the answer needs a scroll for every batch with a
    hit, and 187k scrolls made search answer 503); the TWIN answers, equal to prod by the
    twin-mirror rule. Ids are still asked of every target, by id retrieve. A target set with no
    non-prod target cannot answer and is refused: there is no fallback to prod."""
    asked = {n: (u, c) for n, (u, c) in targets.items() if not PG.is_prod(u, c)}
    if not asked:
        raise StageError("the url question needs a non-prod target (the twin); prod is never "
                         f"scrolled (cytir.index.prod_guard); targets: {sorted(targets)}")
    return ({n: set(served_urls(u, c, urls)) for n, (u, c) in sorted(asked.items())},
            sorted(asked))


def _served_urls(url: str, collection: str, urls: list[str]) -> set[str]:
    """The drain's own question, asked the drain's own way (indexed filtered counts)."""
    return set(PSP.already_served_urls(url, collection, urls, workers=8))


def stage_urls(stage: str | Path) -> set[str]:
    """Every record URL of one stage (its sealed text shard)."""
    out: set[str] = set()
    with gzip.open(Path(stage) / "chunks" / "shard-0000.texts.jsonl.gz", "rt", encoding="utf-8") as fh:
        for ln in fh:
            out.add(json.loads(ln)["url"])
    return out


def other_stage_urls(corpus_root: str | Path, family_id: str, stage: str | Path) -> set[str]:
    """The URLs every OTHER stage of the family holds: staged, not yet served. A second route or
    window that writes a URL a sibling stage already holds would be a parallel representation."""
    out: set[str] = set()
    for _n, d in list_stages(corpus_root, family_id):
        if d != Path(stage):
            out |= stage_urls(d)
    return out


def check_overlap(stage: str | Path, *, targets: dict[str, tuple[str, str]],
                  exclude_served_urls: bool = False, staged_elsewhere: set[str] | None = None,
                  retrieve: Callable = _retrieve_present,
                  served_urls: Callable = _served_urls, log: Callable = print) -> dict:
    """Measure what each target (`name -> (url, collection)`) already serves of this stage:

      * point ids  (a collision: the drain would refuse the pack as AlreadyLanded);
      * urls under OTHER ids (the old representation of the same page: the drain drops those rows
        up to `URL_SHARE_MAX` of the stage's urls and refuses the whole pack beyond it).

    Writes `OVERLAP.json` (always). With `exclude_served_urls` the exclusion file lists the id
    collisions AND the rows whose url is already served (never a second parallel representation of
    a page); without it only the id collisions, and a url share above the drain's ceiling is
    recorded as `needs_ruling` -- discovery then refuses the stage rather than let the drain refuse
    it. Everything is a by-id retrieve or an indexed count: never a scan."""
    stage = Path(stage)
    pack_json = json.loads((stage / "PACK.json").read_text())
    ids: list[str] = []
    url_of: dict[str, str] = {}
    with gzip.open(stage / "chunks" / "shard-0000.texts.jsonl.gz", "rt", encoding="utf-8") as fh:
        for ln in fh:
            row = json.loads(ln)
            ids.append(row["point_id"])
            url_of[row["point_id"]] = row["url"]
    if len(ids) != pack_json["rows"]:
        raise StageError(f"{stage}: {len(ids)} ids read, PACK.json says {pack_json['rows']}")
    urls = sorted(set(url_of.values()))
    ids_hit: set[str] = set()
    urls_hit: set[str] = set()
    staged_hit = set(urls) & staged_elsewhere if staged_elsewhere else set()
    per_target: dict[str, dict] = {}
    for name, (turl, collection) in sorted(targets.items()):
        present: set[str] = set()
        for i in range(0, len(ids), _RETRIEVE_BATCH):
            present |= retrieve(turl, collection, ids[i:i + _RETRIEVE_BATCH])
        per_target[name] = {"url": turl, "collection": collection, "ids_present": len(present)}
        ids_hit |= present
    url_hits, asked = _ask_urls(targets, urls, served_urls)
    for name, hit_urls in url_hits.items():
        per_target[name]["urls_served"] = len(hit_urls)
        urls_hit |= hit_urls
    for name in set(per_target) - set(asked):
        per_target[name]["urls_served"] = None
        per_target[name]["urls_checked_against"] = "twin (prod is never scrolled)"
    urls_hit |= staged_hit            # a URL a sibling stage holds is as taken as a served one
    share = len(urls_hit) / len(urls) if urls else 0.0
    exclude = set(ids_hit)
    if exclude_served_urls:
        exclude |= {pid for pid, u in url_of.items() if u in urls_hit}
    out = {"schema": OVERLAP_SCHEMA, "pack_id": pack_json["pack_id"], "rows": len(ids),
           "targets": per_target, "urls_checked_against": asked, "ids_served": len(ids_hit), "urls": len(urls),
           "urls_served": len(urls_hit), "urls_staged_elsewhere": len(staged_hit),
           "url_share": round(share, 6),
           "url_share_max": URL_SHARE_MAX, "exclude_served_urls": bool(exclude_served_urls),
           "rows_excluded": len(exclude), "rows_net_new": len(ids) - len(exclude),
           "needs_ruling": bool(not exclude_served_urls and share > URL_SHARE_MAX)}
    ex = stage / "overlap-ids.jsonl"
    if exclude:
        ex.write_text("".join(json.dumps({"point_id": i}) + "\n" for i in sorted(exclude)))
        out["exclude_file"] = "overlap-ids.jsonl"
        out["exclude_sha256"] = _sha256_file(ex)
        out["exclude_reason"] = (OVERLAP_REASON_URL if exclude_served_urls else OVERLAP_REASON)
    elif ex.exists():
        ex.unlink()
    (stage / "OVERLAP.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    log(f"[family-stage] {pack_json['pack_id']}: {len(ids):,} rows; ids already served "
        f"{len(ids_hit):,}; urls already served {len(urls_hit):,} of {len(urls):,} "
        f"({share:.2%}; {len(staged_hit):,} of them staged by a sibling stage); excluded "
        f"{len(exclude):,}; net new {out['rows_net_new']:,}"
        + ("; NEEDS RULING (over the drain's ceiling, no exclusion declared)"
           if out["needs_ruling"] else ""))
    return out


def read_overlap(stage: str | Path) -> dict | None:
    p = Path(stage) / "OVERLAP.json"
    if not p.is_file():
        return None
    doc = json.loads(p.read_text())
    if doc.get("schema") != OVERLAP_SCHEMA:
        raise StageError(f"{p}: schema {doc.get('schema')!r}")
    if doc.get("rows_excluded") and _sha256_file(Path(stage) / doc["exclude_file"]) != doc[
            "exclude_sha256"]:
        raise StageError(f"{stage}: overlap-ids.jsonl does not match OVERLAP.json (edited?)")
    return doc


# ----------------------------------------------------------------------------- CLI

def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stage", help="cut a family pack into stages and file each stage's L1")
    st.add_argument("family")
    st.add_argument("--pack", required=True, help="the run's pack/ directory")
    st.add_argument("--corpus-root", required=True)
    st.add_argument("--shard-rows", type=int, default=MAX_SHARD_ROWS)
    st.add_argument("--first-shard", type=int, default=0,
                    help="lowest stage number to use (a later pack of the same family "
                         "continues after the last stage by itself)")
    st.add_argument("--single-slice", action="store_true",
                    help="refuse a pack that would be cut into more than one slice (always on "
                         "for a restage pack, option only_split=true)")
    st.add_argument("--drop-served", action="store_true",
                    help="do not stage rows whose id or url the twin/prod already serve (needs "
                         "--twin and --prod); counted and listed, never silent")
    st.add_argument("--twin", help="twin URL (default: the twin_url setting)")
    st.add_argument("--twin-collection", default=TWIN.collection())
    st.add_argument("--prod", help="prod URL, e.g. http://127.0.0.1:6334")
    st.add_argument("--prod-collection", default="aw_prod")
    ov = sub.add_parser("overlap", help="measure which stage ids are already served")
    ov.add_argument("family")
    ov.add_argument("--corpus-root", required=True)
    ov.add_argument("--twin", required=True, help="twin URL (default: the twin_url setting)")
    ov.add_argument("--twin-collection", default=TWIN.collection())
    ov.add_argument("--prod", required=True, help="prod URL, e.g. http://127.0.0.1:6334")
    ov.add_argument("--prod-collection", default="aw_prod")
    ov.add_argument("--exclude-served-urls", action="store_true",
                    help="also exclude rows whose url is already served under another point id "
                         "(the drain's own drop, made explicit before its 5%% ceiling can refuse "
                         "the whole pack); recorded in OVERLAP.json")
    ov.add_argument("--recheck", action="store_true",
                    help="measure again a stage that already has OVERLAP.json (ONLY before it is "
                         "composed: the queue manifest pins the exclusion file's sha)")
    ss = sub.add_parser("status", help="the roster of stages and their state")
    ss.add_argument("family")
    ss.add_argument("--corpus-root", required=True)
    return ap


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "stage":
        drop = None
        if args.drop_served:
            if not (args.twin and args.prod):
                raise SystemExit("--drop-served needs --twin and --prod")
            drop = {"twin": (args.twin, args.twin_collection),
                    "prod": (args.prod, args.prod_collection)}
        rec = stage_family_pack(args.pack, corpus_root=args.corpus_root, family_id=args.family,
                                shard_rows=args.shard_rows, first_shard=args.first_shard,
                                drop_served=drop, single_slice=args.single_slice)
        print(json.dumps(rec, indent=1))
        return 0
    stages = list_stages(args.corpus_root, args.family)
    if args.cmd == "overlap":
        targets = {"twin": (args.twin, args.twin_collection),
                   "prod": (args.prod, args.prod_collection)}
        for n, d in stages:
            if read_overlap(d) is not None and not args.recheck:
                # a stage that was already measured may already be COMPOSED: its queue manifest pins
                # the sha of overlap-ids.jsonl, so rewriting it would make the drain refuse the pack
                print(f"[family-stage] {pack_id(args.family, n)}: already measured, left alone "
                      "(--recheck only before compose)")
                continue
            check_overlap(d, targets=targets,
                          exclude_served_urls=args.exclude_served_urls)
        return 0
    for n, d in stages:
        ov = read_overlap(d)
        vec = vectors_dir(args.corpus_root, args.family, n)
        print(json.dumps({"pack_id": pack_id(args.family, n),
                          "rows": json.loads((d / "PACK.json").read_text())["rows"],
                          "l1_filed": is_l1_filed(args.corpus_root, args.family, n),
                          "overlap_checked": ov is not None,
                          "ids_served": (ov or {}).get("ids_served"),
                          "urls_served": (ov or {}).get("urls_served"),
                          "rows_excluded": (ov or {}).get("rows_excluded"),
                          "needs_ruling": (ov or {}).get("needs_ruling"),
                          "vectors": vec.is_dir()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
