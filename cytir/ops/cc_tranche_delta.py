#!/usr/bin/env python3
"""Seal the DELTA of a CC tranche whose fetch kept growing after it was sealed (task 698).

`deploy/scripts/cc_tranche_seal_watchdog.sh` (formerly pipeline/, deleted task 1962) seals a tranche exactly once (the `.sealed` marker). The
tranche's `passages.jsonl` is append-only and the fetch resumes: after the CC index mirror was
repaired (2026-10-02) tranches 2-5 gained ~2.1M raw rows, 1.15M of them past the consent gate,
that no step ever staged (measured 2026-10-03: t3 +158k, t5 +786k ok rows; 99% / 89% of a URL
sample not served). This module is that missing step, and the same chain as the watchdog:

    1. filter   `gated/passages_ok.jsonl` minus every row whose URL the twin already serves
                (indexed reads, `family_stage` overlap code). The count is recorded; this is the
                recorded form of the exclusion step, never an automatic drop inside compose.
    2. stage    `cytir.corpus.stage_cc_tranche` on the delta file.
    3. L1       `cytir.corpus.register_l1_release` (ADR-105) + `cytir.corpus.verify_l1_coverage`.
    4. seal     the stage directory is renamed to `t<N>_gpu/delta` LAST. The compose timer
                (`discover_cc_pending`) sees a part directory only then, so a half-done run is
                invisible to it. Pack id: `cc-authority-frontier-w7-tranche<N>-delta`.

Resumable: a finished `delta` part is a no-op; a half-written `_delta_stage` is removed and
redone. A delta of 0 rows is a refusal, never a quiet success.

What a delta is FOR: rows that reached `gated/passages_ok.jsonl` AFTER the tranche's first pack was
staged (the fetch of tranches 2-6 is relaunched for as long as the CC index mirror grows). It is
not always small (tranches 3 and 5 sealed early and grew through relaunches: 99 % and 86 % of
their gated rows were new). Two guards (task 703, 2026-10-04; tranche 7 was staged twice, 0.9M rows
embedded twice, because part 1 ran in the same firing as the first seal):

    * part 1 waits until the FIRST pack (`cc-authority-frontier-w7-tranche<N>`) is SERVED, by its
      promotion-queue receipt: `LANDED`, `DONE` carried by a flip, or `SUPERSEDED` by packs that are
      themselves served. A `REFUSED` first pack is not served: no delta, a dated line each firing.
      A wait that lasts `LOUD_AFTER_WAITS` firings (6 h at the 30 min cadence) turns LOUD;
    * part 1 is refused unless the twin serves at least `MIN_SERVED_SHARE` of the first pack's URLs
      (receipt `mechanical_plan.rows.distinct_urls`; else
      `twin.rows` / `prod.rows`, so a pack landed by a prod-only resume is measurable, task 801). Tranche 7's "delta" was the first pack again:
      1 of 156,560 of its URLs was served. The count is the filter's own `urls_served` (no extra
      request); a human can override with `--allow-unserved-first-pack` (it skips every part-1 first-pack
      gate: the wait, the missing-`distinct_urls` block and the floor).

Exit codes of `main` (none of 3-9 is an error and none is stamped `.delta-checked`; each is retried on
the next firing and LOUD from the 12th in a row): 3 waiting, 5 waiting LOUD, 4 first pack REFUSED,
6 below the 95 % floor, 7 floor LOUD, 8 receipt has no `distinct_urls`, 9 that LOUD; 2 = a refusal
about the gated file (stamped); 0 = ok or nothing to do.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Started by path, sys.path[0] is this file's own directory, not the repo root: put the root first
# so `import cytir` resolves.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cytir.core import twin as TWIN

REPO_ROOT = Path(__file__).resolve().parents[2]
URL_BATCH = 200
#: Part 1 is staged only when the twin serves this share of the first pack's URLs (the receipt's
#: `distinct_urls`). Below it, "not served" means "the first pack is not in yet", not "new rows".
MIN_SERVED_SHARE = 0.95
#: Consecutive part-1 waits after which the line is LOUD (30 min cadence: 12 waits = 6 h).
LOUD_AFTER_WAITS = 12
WAITS_FILE = ".delta-waits"


class DeltaError(RuntimeError):
    """Refused: nothing was sealed and the message says why."""


def part_name(k: int) -> str:
    """Part 1 is `delta` (the name the first run of 2026-10-03 sealed); later parts `delta2`..."""
    return "delta" if k == 1 else f"delta{k}"


def pack_id_for(n: int, k: int = 1) -> str:
    return f"cc-authority-frontier-w7-tranche{n}-{part_name(k)}"


def first_pack_id(n: int) -> str:
    """The pack `cytir.ops.growth_pending_compose` files for the first seal (`t<N>_gpu/all`)."""
    return f"cc-authority-frontier-w7-tranche{n}"


def tranche_paths(data_root: str | Path, n: int, k: int = 1) -> dict:
    t = Path(data_root) / "tranches"
    name = part_name(k)
    return {"gated": t / f"t{n}_out" / "gated" / "passages_ok.jsonl",
            "delta_dir": t / f"t{n}_out" / name,
            "gpu": t / f"t{n}_gpu", "stage": t / f"t{n}_gpu" / f"_{name}_stage",
            "part": t / f"t{n}_gpu" / name}


def _count_lines(path: Path) -> int:
    with open(path, "rb") as fh:
        return sum(1 for line in fh if line.strip())


def _previous_parts_landed(queue_dir: Path, n: int, sealed: int) -> str | None:
    """The first earlier part whose pack is not LANDED yet (rows still in flight), else None."""
    for j in range(1, sealed + 1):
        rec = Path(queue_dir) / pack_id_for(n, j) / "receipt.json"
        try:
            status = json.loads(rec.read_text()).get("status")
        except (OSError, ValueError):
            status = None
        if status != "LANDED":
            return pack_id_for(n, j)
    return None


def _receipt(queue_dir: Path, pack: str) -> dict | None:
    try:
        rec = json.loads((Path(queue_dir) / pack / "receipt.json").read_text())
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def _rows_served(rec: dict | None) -> bool:
    """A receipt that says the pack's rows are on the twin: `LANDED`, or `DONE` finalized from
    `CARRIED-BY-FLIP` (`promote_queue.finalize_carried_by_flip`; a bare `CARRIED-BY-FLIP` is not yet)."""
    if not rec:
        return False
    return rec.get("status") == "LANDED" or (
        rec.get("status") == "DONE" and rec.get("finalized_from") == "CARRIED-BY-FLIP")


def _distinct_urls(rec: dict | None) -> int | None:
    """The first pack's URL count: `mechanical_plan.rows.distinct_urls` (a twin-first receipt), else the
    first present of `twin.rows.distinct_urls`, `prod.rows.distinct_urls`
    (a prod-only resume receipt, task 801). None when none holds an integer."""
    if not isinstance(rec, dict):
        return None
    def at(*path):
        cur = rec
        for key in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur
    for path in (("mechanical_plan", "rows", "distinct_urls"),
                 ("twin", "rows", "distinct_urls"), ("prod", "rows", "distinct_urls")):
        try:
            return int(at(*path))
        except (TypeError, ValueError):
            continue
    return None


def first_pack_state(queue_dir: Path, n: int) -> dict:
    """Where the first pack of tranche `n` stands: `state` is `served`, `waiting` or `refused`.

    `served` carries `expected_urls`, the sum of the receipts' `distinct_urls` over `packs` (None
    when a receipt lacks it). A `SUPERSEDED` first pack (tranche 2: restaged as `-a`..`-d`) is served
    when the packs named in the receipt's `superseded_by` list are; without that list it WAITS (a human
    names them; sibling names are never guessed).
    """
    first = first_pack_id(n)
    rec = _receipt(queue_dir, first)
    status = (rec or {}).get("status")
    if _rows_served(rec):
        return {"state": "served", "packs": [first], "expected_urls": _distinct_urls(rec)}
    if status == "REFUSED":
        return {"state": "refused", "pack": first, "stage": rec.get("stage") or "unknown",
                "reason": str(rec.get("reason") or "")[:200]}
    if status == "SUPERSEDED":
        named = rec.get("superseded_by")
        if not (isinstance(named, list) and named):
            # Never inferred from sibling names (`-a`..`-z`): a later unrelated pack, or only some of
            # the split packs, would read as "served". A human names the superseders.
            return {"state": "waiting", "pack": first,
                    "why": f"{first} is SUPERSEDED and its receipt has no superseded_by list; a human "
                           "must name the superseding packs in it (superseded_by) or use "
                           "--allow-unserved-first-pack"}
        by = [str(x) for x in named]
        recs = {b: _receipt(queue_dir, b) for b in by}
        open_ = next((b for b, r in recs.items() if not _rows_served(r)), None)
        if open_:
            return {"state": "waiting", "pack": open_,
                    "why": f"{first} is SUPERSEDED; its superseding pack {open_} is not served "
                           f"(status {(recs[open_] or {}).get('status')})"}
        counts = [_distinct_urls(r) for r in recs.values()]
        return {"state": "served", "packs": by,
                "expected_urls": None if None in counts else sum(counts)}
    return {"state": "waiting", "pack": first,
            "why": f"{first} has not LANDED (receipt status {status or 'none'})"}


def _stamp(now) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now()))


#: One counter file per blocked state, all under `t<N>_gpu`; a firing that finds another state clears them.
BLOCKS = {"waiting": WAITS_FILE, "below_floor": ".delta-below-floor", "unmeasurable": ".delta-unmeasurable"}


def _bump_waits(gpu_dir: Path, kind: str = "waiting") -> int:
    f = Path(gpu_dir) / BLOCKS[kind]
    try:
        count = int(f.read_text().strip()) + 1
    except (OSError, ValueError):
        count = 1
    f.write_text(f"{count}\n")
    return count


def _clear_waits(gpu_dir: Path, keep: str | None = None) -> None:
    """Clear the counters of every blocked state except `keep` (the one that is firing now)."""
    for kind, name in BLOCKS.items():
        if kind != keep:
            (Path(gpu_dir) / name).unlink(missing_ok=True)


def _block(n: int, gpu_dir: Path, kind: str, text: str, now, log, **extra) -> dict:
    """A part-1 block that retries (never stamped): one dated line per firing, LOUD from the 12th
    consecutive one. `kind` picks the counter and the status the caller maps to an exit code."""
    _clear_waits(gpu_dir, keep=kind)
    count = _bump_waits(gpu_dir, kind)
    loud = count >= LOUD_AFTER_WAITS
    log(f"{_stamp(now)} [cc-tranche-delta] tranche {n}: "
        + (f"LOUD: STILL BLOCKED after {count} firings (6 h or more), nothing stages: " if loud else "")
        + f"{text} (wait {count})")
    return {"tranche": n, "status": {"waiting": "waiting_first_pack",
                                     "below_floor": "first_pack_below_floor",
                                     "unmeasurable": "first_pack_unmeasurable"}[kind],
            "waits": count, "loud": loud, **extra}


def filter_delta(passages_ok: Path, out_file: Path, served_urls, *, log=print) -> dict:
    """Write the rows of `passages_ok` whose URL is not served; return the counts.

    `served_urls(list[str]) -> set[str]` answers for a batch. Two passes: the URLs, then the rows.
    """
    urls: dict[str, None] = {}
    rows_in = 0
    with open(passages_ok, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows_in += 1
            urls.setdefault(json.loads(line)["url"])
    served: set[str] = set()
    batch = list(urls)
    for i in range(0, len(batch), URL_BATCH):
        served |= served_urls(batch[i:i + URL_BATCH])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(".tmp")
    kept = 0
    with open(passages_ok, encoding="utf-8") as fh, open(tmp, "w", encoding="utf-8") as out:
        for line in fh:
            if not line.strip():
                continue
            if json.loads(line)["url"] in served:
                continue
            out.write(line if line.endswith("\n") else line + "\n")
            kept += 1
    tmp.replace(out_file)
    counts = {"rows_in": rows_in, "urls_in": len(urls), "urls_served": len(served),
              "rows_served_excluded": rows_in - kept, "delta_rows": kept}
    log(f"[cc-tranche-delta] filter: {counts}")
    return counts


def screen_delta(delta_dir: Path, host_lists: Path | None, *, log=print) -> dict:
    """Task 701: drop spam rows and whole hijacked hosts from the delta BEFORE it is staged or sealed
    (`cytir.ops.spam_rows.screen_jsonl`). The unscreened file stays as `passages_ok.prescreen.jsonl`, the
    dropped rows with their reason in `spam_screen.jsonl`."""
    from cytir.ops import spam_rows
    src = delta_dir / "passages_ok.jsonl"
    tmp = delta_dir / "passages_ok.screened.jsonl"
    counts = spam_rows.screen_jsonl(src, tmp, delta_dir / "spam_screen.jsonl", host_lists=host_lists)
    if counts["rows_kept"] == 0:
        raise DeltaError(f"the spam screen left 0 rows of {counts['rows_in']}")
    src.rename(delta_dir / "passages_ok.prescreen.jsonl")
    tmp.rename(src)
    log(f"[cc-tranche-delta] spam screen: {counts}")
    return counts


def run_delta(n: int, *, data_root, python_bin: str, served_urls, runner=subprocess.run,
              log=print, now=time.time, queue_dir=None, spam_host_lists: Path | None = None,
              allow_unserved_first_pack: bool = False) -> dict:
    """Seal the next delta part of tranche `n`. Part 1 is the rows the one-time seal missed; a
    later part exists only when `gated/passages_ok.jsonl` GREW past what the last part read
    (`rows_in` in its DELTA.json) AND every earlier part's pack LANDED (`queue_dir` receipts), so
    rows still in flight are never sealed twice."""
    sealed = 0
    while tranche_paths(data_root, n, sealed + 1)["part"].is_dir():
        sealed += 1
    k = sealed + 1
    p = tranche_paths(data_root, n, k)
    first_pack = None
    if not sealed and not allow_unserved_first_pack:
        # Part 1: the first pack must be served before "the rows not served" can mean "new rows".
        if queue_dir is None:
            raise DeltaError(f"tranche {n}: part 1 needs --queue-dir to check the first pack is served")
        first_pack = first_pack_state(Path(queue_dir), n)
        if first_pack["state"] == "refused":
            _clear_waits(p["gpu"])
            log(f"{_stamp(now)} [cc-tranche-delta] tranche {n}: FIRST PACK REFUSED, no delta is staged: "
                f"{first_pack['pack']} was REFUSED at stage {first_pack['stage']} "
                f"({first_pack['reason']}); it needs a human or a re-queue")
            return {"tranche": n, "status": "first_pack_refused", "pack_id": first_pack["pack"],
                    "stage": first_pack["stage"]}
        if first_pack["state"] == "waiting":
            return _block(n, p["gpu"], "waiting",
                          "waiting for the first pack to be served: " + first_pack["why"], now, log,
                          pack_id=first_pack["pack"])
        if first_pack["expected_urls"] is None:       # free here: no 14 GB read, no twin request
            return _block(n, p["gpu"], "unmeasurable",
                          f"the first pack {first_pack['packs']} cannot be measured: its receipt has no "
                          "mechanical_plan.rows.distinct_urls; a human must add it or use "
                          "--allow-unserved-first-pack", now, log, pack_id=first_pack["packs"][0])
        _clear_waits(p["gpu"], keep="below_floor")
    if sealed:
        last = tranche_paths(data_root, n, sealed)
        grown = _count_lines(p["gated"]) - json.loads(
            (last["delta_dir"] / "DELTA.json").read_text())["rows_in"]
        if grown <= 0:
            log(f"[cc-tranche-delta] tranche {n}: {part_name(sealed)} already sealed and the "
                "gated file has not grown, nothing to do")
            return {"tranche": n, "status": "already_sealed"}
        if queue_dir is None:
            raise DeltaError(f"tranche {n}: the gated file grew by {grown} rows; a later delta "
                             "needs --queue-dir to check the earlier part landed")
        waiting = _previous_parts_landed(Path(queue_dir), n, sealed)
        if waiting:
            log(f"{_stamp(now)} [cc-tranche-delta] tranche {n}: grew by {grown} rows, waiting for "
                f"{waiting} to land first")
            return {"tranche": n, "status": "waiting_previous_delta", "pack_id": waiting}
    if not p["gated"].is_file():
        raise DeltaError(f"tranche {n}: no gated passages_ok.jsonl at {p['gated']}")
    if not (p["gpu"] / ".sealed").is_file():
        raise DeltaError(f"tranche {n}: the first seal is not done; the watchdog owns it")
    counts = filter_delta(p["gated"], p["delta_dir"] / "passages_ok.jsonl", served_urls, log=log)
    if counts["delta_rows"] == 0:
        raise DeltaError(f"tranche {n}: the delta is 0 rows ({counts}); nothing to stage")
    if first_pack is not None:
        need, got = first_pack["expected_urls"], counts["urls_served"]
        floor = math.ceil(MIN_SERVED_SHARE * need)
        counts["first_pack_check"] = {"packs": first_pack["packs"], "expected_urls": need,
                                      "urls_served": got, "floor": floor}
        if got < floor:
            (p["delta_dir"] / "passages_ok.jsonl").unlink()   # tens of GB; a rerun rebuilds it
            return _block(n, p["gpu"], "below_floor",
                          f"the first pack is not served enough: the twin serves {got} gated URLs, the "
                          f"first pack {first_pack['packs']} holds {need} (floor {floor}, "
                          f"{MIN_SERVED_SHARE:.0%}); this delta would stage the first pack again; no "
                          f"delta staged, asked again every firing ({counts}); override: "
                          "--allow-unserved-first-pack", now, log, pack_id=first_pack["packs"][0],
                          first_pack_check=counts["first_pack_check"])
        _clear_waits(p["gpu"])
    counts["spam_screen"] = screen_delta(p["delta_dir"], spam_host_lists, log=log)
    label = f"task698-cc-tranche{n}-w7-{part_name(k)}"
    if p["stage"].exists():
        shutil.rmtree(p["stage"])
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    steps = [
        ("stage", [python_bin, "-m", "cytir.corpus.stage_cc_tranche", "--in",
                   str(p["delta_dir"] / "passages_ok.jsonl"), "--out", str(p["stage"]),
                   "--packs", "1", "--shard-size", "20000"]),
        ("register_l1", [python_bin, "-m", "cytir.corpus.register_l1_release", "--source",
                         str(p["delta_dir"]), "--release", label,
                         "--file-suffix", "passages_ok.jsonl"]),
        ("verify_l1", [python_bin, "-m", "cytir.corpus.verify_l1_coverage", "--passages",
                       str(p["delta_dir"] / "passages_ok.jsonl"), "--label", label]),
    ]
    for name, argv in steps:
        log(f"[cc-tranche-delta] {name}: {' '.join(argv)}")
        r = runner(argv, env=env)
        if r.returncode != 0:
            raise DeltaError(f"tranche {n}: step {name} failed rc={r.returncode}")
    staged = p["stage"] / "all" / "chunks"
    if not any(staged.glob("*.texts.jsonl.gz")):
        raise DeltaError(f"tranche {n}: stage wrote no chunk shards under {staged}")
    (p["stage"] / "all").rename(p["part"])            # the seal: visible to the timer only now
    shutil.copy2(p["stage"] / "stage_report.json", p["part"] / "stage_report.json")
    shutil.rmtree(p["stage"])
    (p["part"] / ".sealed").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now())) + "\n")
    counts.update(tranche=n, status="sealed", part=part_name(k), pack_id=pack_id_for(n, k),
                  l1_release=label)
    (p["delta_dir"] / "DELTA.json").write_text(json.dumps(counts, indent=1) + "\n")
    log(f"[cc-tranche-delta] SEALED {counts}")
    return counts


def _twin_served_urls(twin_url: str, collection: str):
    from cytir.sources import family_stage   # the same indexed URL read the DILA overlap uses

    def fn(urls):
        return family_stage._served_urls(twin_url, collection, urls)
    return fn


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog or "cc_tranche_delta", description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path, required=True,
                    help="the task634 tree that holds tranches/ (under the work root)")
    ap.add_argument("--tranche", type=int, required=True)
    ap.add_argument("--twin", default=TWIN.url())
    ap.add_argument("--twin-collection", default=TWIN.collection())
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--queue-dir", type=Path, default=None,
                    help="the promotion queue (default: the `queue_dir` setting); a delta needs it to check "
                         "the first pack, and every earlier part, landed")
    ap.add_argument("--allow-unserved-first-pack", action="store_true",
                    help="HUMAN OVERRIDE: stage part 1 although the twin serves less than "
                         f"{MIN_SERVED_SHARE * 100:.0f}%% of the first pack's URLs")
    ap.add_argument("--spam-host-lists", type=Path, default=REPO_ROOT / "data" / "spam-host-lists.v1.json",
                    help="sealed hold-out / exception host lists for the spam screen (cytir.ops.spam_rows)")
    args = ap.parse_args(argv)
    if not args.spam_host_lists.is_file():
        print(f"REFUSED: spam host lists {args.spam_host_lists} not found", file=sys.stderr)
        return 2
    if not os.environ.get("AW_PERSISTED_INDEX_SQLITE_DIR"):
        print("FATAL: AW_PERSISTED_INDEX_SQLITE_DIR is not set (the NVMe persisted index the drain "
              "uses); verify_l1_coverage would fall back to the HDD copy", file=sys.stderr)
        return 2
    if args.queue_dir is None:
        from cytir.core import paths
        args.queue_dir = Path(paths.get_value("queue_dir"))
    try:
        res = run_delta(args.tranche, data_root=args.data_root, python_bin=args.python,
                        served_urls=_twin_served_urls(args.twin, args.twin_collection),
                        queue_dir=args.queue_dir, spam_host_lists=args.spam_host_lists,
                        allow_unserved_first_pack=args.allow_unserved_first_pack)
        if res.get("status") == "first_pack_refused":
            return 4            # nothing staged, not stamped as checked: named again each firing
        status = res.get("status")
        for blocked, (quiet, loud) in (("waiting_first_pack", (3, 5)),
                                       ("first_pack_below_floor", (6, 7)),
                                       ("first_pack_unmeasurable", (8, 9))):
            if status == blocked:   # retried, never stamped; the loud code is from the 12th firing
                return loud if res.get("loud") else quiet
        if status == "waiting_previous_delta":
            return 3            # the scheduled caller retries; 2 (a refusal) is re-checked on change
    except DeltaError as exc:
        print(f"{_stamp(time.time)} REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
