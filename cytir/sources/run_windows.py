"""Run a windowed API family end to end, one window at a time (task 699).

    python -m cytir.sources.run_windows <family> --out-root R --corpus-root C --opt leg=fr \\
        --twin <twin-url> --prod http://127.0.0.1:6334

`cytir.sources.run_family` composes ONE pack at the end of a run and holds its records in memory: right
for a dump of an hour, wrong for a 56-hour API fetch (nothing reaches the promotion queue until the
end, and a kill loses everything). This driver cuts the fetch into date WINDOWS the adapter plans
(`adapter.plan_windows(ctx)`: calendar months packed to <= `--cap` listed documents) and for every
window runs `run_family` -> `family_stage stage` (L1 filing) -> `family_stage overlap`, so the
compose timer embeds and enqueues window 1 while window 2 is still being read.

Resumable and idempotent: the plan is written once (`R/WINDOWS.json`) and reused; a window with a
`DONE.json` is skipped; a window whose pack exists is not re-fetched, only re-staged; a window with
no pack is removed and re-run whole. Stage and overlap run under ONE lock shared with every other
family stage run of the corpus (`<C>/stages/.family-stage.lock`), because `overlap` walks every
unmeasured stage of the family. A window that admits nothing is recorded `empty` (not fatal); a
run that stages 0 rows in total refuses (corpus-doctrine: a run that yields nothing is not done).
Units quarantined by an instrument failure are listed in `STATE.json` (`quarantined_units`) and
the window is still staged: those months need a re-run, never a waiver.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import datetime as dt
import fcntl
import json
import os
import shlex
import shutil
import sys
import time
from pathlib import Path

from cytir.core import twin as TWIN
from cytir.sources import family_stage as FS
from cytir.sources import registry, run_family
from cytir.sources.family import RunContext


class WindowsError(RuntimeError):
    """Refused: the message says why and what was left standing."""


def _write(path: Path, doc) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def plan(family_id: str, root: Path, options: dict, today: str, cap: int, http=None) -> list[dict]:
    """The windows of this run: read from `WINDOWS.json` when it exists, else planned and sealed."""
    wfile = root / "WINDOWS.json"
    if wfile.is_file():
        doc = json.loads(wfile.read_text())
        if doc["options"] != options:
            raise WindowsError(f"{wfile} was planned with options {doc['options']}, not {options}; "
                               "use a new --out-root (windows never change under a running fetch)")
        return doc["windows"]
    adapter, _row, spec = registry.admit(family_id, strict=False)
    ctx = RunContext(out_dir=root, today=today, options=dict(options), chunk_spec=spec, http=http,
                     contract=_row)
    with ExitStack() as stack:
        hs = getattr(adapter, "http_spec", None)
        if ctx.http is None and hs is not None:     # the runner's builder, closed after planning
            ctx.http = run_family.build_http_client(hs)
            stack.callback(ctx.http.close)
        ctx.http = run_family.ensure_guarded(ctx.http)
        wins = adapter.plan_windows(ctx, cap=cap)
    if not wins:
        raise WindowsError(f"{family_id}: the plan holds 0 windows; an empty plan is a wrong "
                           "option or a broken API, not an empty source")
    _write(wfile, {"family": family_id, "planned": today, "cap": cap, "options": options,
                   "windows": wins})
    return wins


def _locked(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")
    fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


def run_window(family_id: str, win: dict, *, root: Path, corpus_root: Path, options: dict,
               today: str, twin: str, prod: str, twin_collection: str, prod_collection: str,
               http=None, consent=None, log=print) -> dict:
    out = root / win["key"]
    done = out / "DONE.json"
    if done.is_file():
        return json.loads(done.read_text())
    pack = out / "pack"
    if not (pack / "manifest.json").is_file():
        if out.exists():
            shutil.rmtree(out)            # a half-run window: start it whole (no pack = no claim)
        opts = {**options, "from": win["from"], "to": win["to"]}
        try:
            run_family.run_family(family_id, out_dir=out, today=today, options=opts, http=http,
                                  consent=consent, log=log)
        except run_family.RunError as exc:
            if "0 records admitted" not in str(exc):
                raise
            out.mkdir(parents=True, exist_ok=True)
            rec = {"window": win["key"], "status": "empty", "rows": 0, "why": str(exc)[:500]}
            _write(done, rec)
            return rec
    man = json.loads((pack / "manifest.json").read_text())
    lock = _locked(corpus_root / "stages" / ".family-stage.lock")
    overlap = None
    try:
        staged = FS.stage_family_pack(pack, corpus_root=corpus_root, family_id=family_id, log=log)
        targets = {"twin": (twin, twin_collection), "prod": (prod, prod_collection)}
        for n, d in FS.list_stages(corpus_root, family_id):
            if FS.read_overlap(d) is None:
                ov = FS.check_overlap(
                    d, targets=targets, exclude_served_urls=True,
                    staged_elsewhere=FS.other_stage_urls(corpus_root, family_id, d))
                overlap = {k: ov[k] for k in ("rows", "ids_served", "urls", "urls_served",
                                              "urls_staged_elsewhere", "rows_excluded", "rows_net_new")}
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    t = man["terminals"]
    rec = {"window": win["key"], "status": "staged", "rows": man["records"]["count"],
           "stages": [s["pack_id"] for s in staged], "units": t["units_in"],
           "quarantined": t["units_quarantined"], "quarantined_reasons": t["quarantined_reasons"],
           "documents": (man.get("discover") or {}).get("documents"), "overlap": overlap,
           "collapsed_identical_rows": man.get("collapsed_identical_rows", 0),
           "collapsed_detail": man.get("collapsed_detail") or [],
           "quarantined_units": [u["unit"] for u in man["units"] if u["terminal"] == "quarantined"]}
    held = max((x.get("refused_binary", 0) for x in staged), default=0)
    if held:                                           # task 777: absent on a clean window
        rec["refused_binary"] = held
    _write(done, rec)
    return rec


def run_all(family_id: str, *, root: Path, corpus_root: Path, options: dict, today: str, cap: int,
            twin: str, prod: str, twin_collection=TWIN.collection(), prod_collection="aw_prod",
            newest_first=True, max_windows=0, http=None, consent=None, log=print) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    wins = plan(family_id, root, options, today, cap, http=http)
    order = list(reversed(wins)) if newest_first else list(wins)
    state_path = root / "STATE.json"
    state = {"family": family_id, "windows_total": len(wins), "done": [], "rows_staged": 0,
             "quarantined_units": [], "status": "running", "started": today}
    (root / "NEXT.md").write_text(next_md(family_id, root, corpus_root, options, twin, prod))
    ran = 0
    for win in order:
        state["current"] = win["key"]
        _write(state_path, state)
        t0 = time.time()
        rec = run_window(family_id, win, root=root, corpus_root=corpus_root, options=options,
                         today=today, twin=twin, prod=prod, twin_collection=twin_collection,
                         prod_collection=prod_collection, http=http, consent=consent, log=log)
        state["done"].append({"window": win["key"], "status": rec["status"], "rows": rec["rows"],
                              "seconds": round(time.time() - t0)})
        state["rows_staged"] += rec["rows"]
        state["quarantined_units"] += rec.get("quarantined_units") or []
        ran += 1
        log(f"[run_windows] {win['key']}: {rec['status']} {rec['rows']:,} rows "
            f"(collapsed {rec.get('collapsed_identical_rows', 0):,} identical) "
            f"({len(state['done'])}/{len(wins)} windows)")
        if max_windows and ran >= max_windows:
            break
    complete = len(state["done"]) == len(wins)
    state.pop("current", None)
    if complete and state["rows_staged"] == 0:
        state["status"] = "refused_zero_rows"
        _write(state_path, state)
        raise WindowsError(f"{family_id}: all {len(wins)} windows done and 0 rows staged")
    state["status"] = "complete" if complete else "partial"
    _write(state_path, state)
    return state


def _resume_prefix() -> str:
    """The checkout and the environment of THIS process, shell-quoted, for a pasteable resume.

    An unquoted `<placeholder>` is parsed by bash as a redirection (`FOO=<x` reads a file named
    x), so everything that is not a known value is quoted. The identity, the overlay file and the
    work root are the running process's effective ones, not ours."""
    from cytir.core import user_agent as UA
    app = Path(__file__).resolve().parents[2]
    url, contact, configured = UA.operator()
    env = [f"AW_OPERATOR_URL={shlex.quote(url if configured else '<your-url>')}",
           f"AW_OPERATOR_CONTACT={shlex.quote(contact if configured else '<you@example.org>')}"]
    for name in ("AW_OPERATOR_OVERLAY", "AW_WORK_ROOT"):
        if os.environ.get(name):
            env.append(f"{name}={shlex.quote(os.environ[name])}")
    env.append("AW_PERSISTED_INDEX_SQLITE_DIR="
               + shlex.quote(os.environ.get("AW_PERSISTED_INDEX_SQLITE_DIR") or "<persisted>"))
    return f"cd {shlex.quote(str(app))} && " + " ".join(env)


def next_md(family_id, root, corpus_root, options, twin, prod) -> str:
    opts = " ".join(f"--opt {k}={v}" for k, v in sorted(options.items()))
    return (f"# {family_id} windows -- NEXT.md\n\n"
            f"State: `{root}/STATE.json`. Success predicate: `status` == `complete` and every window has "
            f"`{root}/<window>/DONE.json`.\n\nResume (idempotent; never delete a DONE window):\n\n"
            f"    {_resume_prefix()} \\\n"
            f"      python -m cytir.sources.run_windows {family_id} --out-root {root} "
            f"--corpus-root {corpus_root} {opts} --twin {twin} --prod {prod}\n\n"
            "The compose timer embeds and enqueues each staged window; check the receipts of "
            f"`{FS.slug(family_id)}-shardN` in the promotion queue. `quarantined_units` in STATE.json "
            "lists months an instrument failure left out: re-run them, never waive.\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("family")
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--corpus-root", required=True, type=Path)
    ap.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--cap", type=int, default=25_000, help="listed documents per window")
    ap.add_argument("--twin", default=TWIN.url())
    ap.add_argument("--twin-collection", default=TWIN.collection())
    ap.add_argument("--prod", default="http://127.0.0.1:6334")
    ap.add_argument("--prod-collection", default="aw_prod")
    ap.add_argument("--oldest-first", action="store_true")
    ap.add_argument("--max-windows", type=int, default=0, help="stop after N windows (smoke)")
    ap.add_argument("--today", default=dt.datetime.now(dt.timezone.utc).date().isoformat())
    a = ap.parse_args(argv)
    options = dict(kv.split("=", 1) for kv in a.opt)
    st = run_all(a.family, root=a.out_root, corpus_root=a.corpus_root, options=options,
                 today=a.today, cap=a.cap, twin=a.twin, prod=a.prod,
                 twin_collection=a.twin_collection, prod_collection=a.prod_collection,
                 newest_first=not a.oldest_first, max_windows=a.max_windows)
    print(json.dumps({k: st[k] for k in ("status", "windows_total", "rows_staged")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
