"""Family run -> one sealed chunk stage on disk, for the stranger's path (task 696, slice 7 step 1).

    cyt source run dila_kali --from data/samples/dila-kali --out ./stage --today 2026-10-03
    cyt index build --pack ./stage ...

`cyt source run` makes a PACK (`pack/manifest.json` + `records.jsonl`). `cyt index build` reads a
STAGE (`PACK.json` + `chunks/shard-0000.texts.jsonl.gz`). This module is the bridge for a
person with one directory and no corpus tree: it makes ONE stage from ONE pack, nowhere else.

It is not the production bridge. `cytir.sources.family_stage` cuts a pack into numbered stages under
a corpus root, files each one in the L1 register and measures its overlap with the served index.
A single local stage has none of that: no corpus root, no L1 filing, no overlap. It applies the same
checks BEFORE a byte is written (`family_stage.check_pack`: the schema, the balanced terminals, the
declaration with no `unknown`, the consent checked, the records re-hashed; and every row's `url`,
`text`, `point_id` and its own `license`), writes the same bytes (a deterministic gzip of the rows,
untouched) and the same `PACK.json` keys. A pack of more than one stage's worth of rows is refused,
not cut.

`--from LISTING` for a family with the replay-l0 route (`cytir.sources.families.pdf_replay`) reads a
sealed L0 listing of PDF bytes instead of a directory, fetches no page, and still resolves its
host's consent live (the one network call); see that module.

`--from DIR` is OFFLINE by construction. It is allowed only for a family whose route is a source-owned
bulk dump and that contacts no host: the family then reads `DIR` (its `root` option) and nothing else.
No consent is asked (a dump route has no host to ask about; the licence is the consent basis, ADR-076)
and no network client is passed. A family that needs a host is refused with the reason.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from cytir.sources import family_stage as FS
from cytir.sources import registry

OFFLINE_ROUTE = "source_owned_bulk_dump"
SINGLE_SHARD = 0
LOCAL_MARK = "local"
DIGEST_HEX = 12


def local_pack_id(family_id: str, records_sha256: str) -> str:
    """`<family>-local-<12 hex of the rows' sha>`: a local stage never carries a production pack id
    (`<family>-shardN`), because `cyt index build` keys its state and receipts by pack id. The digest
    is of the rows, so the id is the same on every machine."""
    return f"{FS.slug(family_id)}-{LOCAL_MARK}-{records_sha256[:DIGEST_HEX]}"


class StageOutRefused(RuntimeError):
    """Nothing was written; the message names the rule."""


def queue_refusal(path: str | Path) -> str | None:
    """Why `path` may not receive a stage: it is, lies inside, or contains the live promotion queue
    (the one test the promotion verbs use, `cytir.index.promote_queue.live_queue_refusal`: the
    settings, the box file, the `LIVE-QUEUE` marker). A stage written there would be read by the
    hourly drain as a pack."""
    from cytir.index import promote_queue      # lazy: a large module the stage door needs only here

    return promote_queue.live_queue_refusal(Path(path))


def is_replay_family(family_id: str, *, contracts_path=None) -> bool:
    """True when `--from` for this family names a sealed L0 listing (the replay-l0 route, task
    782) and not a directory of a source-owned dump."""
    adapter, _row, _spec = registry.admit(family_id, contracts_path=contracts_path)
    return bool(getattr(adapter, "replay_l0", False))


def offline_refusal(family_id: str, *, contracts_path=None) -> str | None:
    """Why `family_id` cannot run from a local input, or None when it can. A family with the
    replay-l0 route reads a sealed L0 listing and fetches no page, so `--from` is allowed for it;
    its host's consent is still resolved live (see `run_to_stage`). A live sitemap family without
    that route keeps the refusal below."""
    adapter, _row, _spec = registry.admit(family_id, contracts_path=contracts_path)
    if getattr(adapter, "replay_l0", False):
        return None
    if adapter.direct_crawl or adapter.route != OFFLINE_ROUTE:
        return (f"{family_id} has route {adapter.route!r} and "
                f"{'contacts hosts' if adapter.direct_crawl else 'no host'}: `--from` is for a "
                f"family that reads a local dump ({OFFLINE_ROUTE}, no host). Run it without `--from` "
                "(the network, with the consent check) instead.")
    return None


def _no_consent(host: str, direct: bool) -> dict:      # pragma: no cover - guarded by offline_refusal
    raise AssertionError(f"an offline family asked for the consent of {host}")


def stage_from_pack(pack_dir: str | Path, family_id: str, out: str | Path) -> dict:
    """Write `out/PACK.json` and `out/chunks/shard-0000.texts.jsonl.gz` from a family pack.
    `out` must not exist. Returns the `PACK.json` content."""
    out = Path(out)
    why_live = queue_refusal(out)
    if why_live:
        raise StageOutRefused(f"--out refused: {why_live}")
    if out.exists():
        raise StageOutRefused(f"{out} exists; a stage is never overwritten (pick a new --out)")
    try:
        man = FS.check_pack(pack_dir, family_id)
        rpath = Path(pack_dir) / "records.jsonl"
        n = FS._validate_rows(rpath)
    except FS.StageError as exc:
        raise StageOutRefused(str(exc)) from exc
    total = man["records"]["count"]
    if n != total:
        raise StageOutRefused(f"{rpath}: {n} rows read, manifest says {total}")
    if n > FS.MAX_SHARD_ROWS:
        raise StageOutRefused(f"{n:,} rows is more than one stage ({FS.MAX_SHARD_ROWS:,}); "
                              "`cytir.sources.family_stage` cuts a large pack into stages")
    lines = [ln if ln.endswith(b"\n") else ln + b"\n"
             for ln in rpath.read_bytes().splitlines(keepends=True) if ln.strip()]
    meta = {"schema": FS.STAGE_SCHEMA, "family_id": family_id,
            "pack_id": local_pack_id(family_id, man["records"]["sha256"]), "shard": SINGLE_SHARD,
            "row_start": 0, "source_rows": n, "slice_rows": FS.MAX_SHARD_ROWS, "rows": len(lines),
            "dropped_served": 0,
            # no sha of manifest.json: it names the checkout and the input directory, so the
            # stage would differ from machine to machine; the rows' own sha is the identity
            "source_records_sha256": man["records"]["sha256"]}
    tmp = out.with_name(out.name + ".part")
    if tmp.exists():
        raise StageOutRefused(f"{tmp}: a half-written stage is never overwritten; remove it")
    (tmp / "chunks").mkdir(parents=True)
    shard = tmp / "chunks" / "shard-0000.texts.jsonl.gz"
    FS._write_gz_deterministic(shard, lines)
    meta["chunks_sha256"] = FS._sha256_file(shard)
    (tmp / "PACK.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    tmp.rename(out)                 # published by rename: PACK.json never describes a partial dir
    return meta


def run_to_stage(family_id: str, *, out: str | Path, today: str, from_dir: str | Path | None = None,
                 work_dir: str | Path | None = None, limit: int = 0, options: dict | None = None,
                 contracts_path=None, log: Callable = print, **run_kw) -> dict:
    """Run the family, then stage its pack into `out`. With `from_dir` the run is offline (see the
    module doc). `work_dir` keeps the sealed L0 and the pack; without it they live in a temporary
    directory that is removed. Returns the stage's `PACK.json` content."""
    from cytir.sources import run_family

    out = Path(out)
    why_live = queue_refusal(out)
    if why_live:
        raise StageOutRefused(f"--out refused: {why_live}")
    if out.exists():
        raise StageOutRefused(f"{out} exists; a stage is never overwritten (pick a new --out)")
    if from_dir is None and work_dir is None:
        raise StageOutRefused("--out without --from needs --out-dir: a network run's sealed source "
                              "data is kept in a directory you name, never in a temporary one that "
                              "is deleted")
    opts = dict(options or {})
    kw = dict(run_kw)
    if from_dir is not None:
        why = offline_refusal(family_id, contracts_path=contracts_path)
        if why:
            raise StageOutRefused(why)
        if is_replay_family(family_id, contracts_path=contracts_path):
            # the replay-l0 route: the input is a sealed L0 listing, the host's consent is resolved
            # live by the default resolver (the contract says "live per origin, at stage time"; the
            # stage door refuses a pack whose consent was not checked). No page is fetched.
            from cytir.sources.families import pdf_replay

            if not pdf_replay.listing_file(from_dir).is_file():
                raise StageOutRefused(f"--from {from_dir!r} is not a sealed L0 listing (a "
                                      f"{pdf_replay.LISTING_NAME} file, or a directory holding one)")
            if pdf_replay.LISTING_OPTION in opts:
                raise StageOutRefused("--from and --opt l0_listing=... both name the input")
            opts[pdf_replay.LISTING_OPTION] = str(from_dir)
        else:
            if not Path(from_dir).is_dir():
                raise StageOutRefused(f"--from {from_dir!r} is not a directory")
            if "root" in opts:
                raise StageOutRefused("--from and --opt root=... both name the input; give one")
            opts["root"] = str(from_dir)
            kw.setdefault("consent", _no_consent)
    tmp = None
    if work_dir is None:
        out.parent.mkdir(parents=True, exist_ok=True)
        # beside the result (same filesystem), not in $TMPDIR
        tmp = tempfile.mkdtemp(prefix=f".{out.name}.work-", dir=out.parent)
        work_dir = tmp
    try:
        run_family.run_family(family_id, out_dir=work_dir, today=today, limit=limit,
                              options=opts, log=log, contracts_path=contracts_path, **kw)
        meta = stage_from_pack(Path(work_dir) / "pack", family_id, out)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    log(f"[{family_id}] stage: {meta['rows']} row(s) -> {out}")
    return meta
