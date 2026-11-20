"""Retry NAMED domains of a SEALED CC tranche (task 760): `cyt ops cc-tranche retry-domains`.

A never-queued tranche (7-14) is fetched once. A domain that errored stays out of `state.json`'s `done`, and
nothing in the package ever fetched it again: `watchdog_cc_tranches.sh` relaunches tranches 2-6 only and leaves
a sealed never-queued tranche alone. Task 714 retried 15 such domains with a hand script
(a task-714 exploration script). This verb is that sequence, in the package, with its guards.

    cyt ops cc-tranche retry-domains --tranche N --domains FILE|a.com,b.org [--dry-run]

What it does, in this order:
  1. reads the tranche's roster (`tranches/tranche<N>_never_queued.json`) and refuses a domain that is not in it;
  2. refuses an unsealed tranche (`t<N>_gpu/.sealed`): an unsealed tranche is the seal watchdog's, and a partial
     retry run would stamp `.fetch-ok` on a tranche that never finished;
  3. drops a domain already in `state.json` `done` (named, not an error);
  4. refuses while ANY never-queued CC fetch runs (the watchdog's rule: one at a time, CC <= 5 req/s) or while the
     watchdog's own pass holds `.watchdog.lock`;
  5. runs ONE `discovery_ingest fetch --only <domains>` with the watchdog's flags (`--resume`: it appends to
     `t<N>_out/passages.jsonl`, `outlinks.jsonl` and `state.json`), through `aw-cc-fetch-run.sh`, so the seal
     watchdog's late-growth pass (fetch not alive, `passages.jsonl` newer than the gated file) gates the new
     rows into a `-delta` pack unchanged. The fetch command line names `tranches/tranche<N>_never_queued.json`,
     so the watchdog sees it as that tranche's live fetch and starts no second one.
  6. prints one line per domain (done / still not done, pages, rows) and appends it to
     `logs/tranche<N>-retry-domains.log`. Exit 0 when every requested domain is done, 4 when one is not.

`--dry-run` prints the plan (and whether it is blocked now) and runs nothing. Nothing here touches prod or the twin.
The flags below mirror `watchdog_cc_tranches.sh`; `tests/test_aw_cc_tranche_retry.py` fails when they drift.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FETCH_RUN = REPO_ROOT / "deploy" / "scripts" / "aw-cc-fetch-run.sh"

#: The watchdog's fetch recipe (watchdog_cc_tranches.sh), one home here for the retry.
CRAWL = "CC-MAIN-2026-39"
MIRROR_SUBPATH = ("aw-work", "cc-index-mirror", CRAWL)
FETCH_FLAGS = ("--max-per-domain", "60", "--rate", "4", "--workers", "10", "--heartbeat-pages", "20",
               "--resolver", "columnar", "--crawl", CRAWL, "--columnar-threads", "3", "--egress-mbps", "4.5")
NICE = ("taskset", "-c", "12-23", "nice", "-n15", "ionice", "-c3")
#: The watchdog's "a never-queued fetch is running" test (NEW_FETCH_RE), for `pgrep -f`.
NEW_FETCH_RE = r"discovery_ingest(\.py)? fetch --queue tranches/tranche[0-9]+_never_queued"
EXIT_NOT_DONE = 4


class Refused(SystemExit):
    def __init__(self, why: str):
        super().__init__(f"REFUSED: {why}")


def parse_domains(spec: str, tranche: int) -> list[str]:
    """`spec` is a file or a comma list. A file is a JSON list, a JSON object keyed by tranche number
    (`{"7": [...], "8": [...]}`, the shape of `retry714_domains.json`), or text with one domain per line."""
    p = Path(spec)
    if p.is_file():
        text = p.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except ValueError:
            data = [ln.split("#")[0].strip() for ln in text.splitlines()]
        if isinstance(data, dict):
            data = data.get(str(tranche), [])
    else:
        data = spec.split(",")
    out: list[str] = []
    for d in data:
        d = str(d).strip().lower()
        if d and d not in out:
            out.append(d)
    if not out:
        raise Refused(f"no domain for tranche {tranche} in --domains {spec!r}")
    return out


def running_fetches() -> list[str]:
    """Command lines of the never-queued CC fetches running now (this process excluded)."""
    r = subprocess.run(["pgrep", "-af", NEW_FETCH_RE], capture_output=True, text=True)
    me = str(os.getpid())
    return [ln for ln in r.stdout.splitlines() if ln.split(" ", 1)[0] != me and "pgrep" not in ln]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_plan(data_root: Path, tranche: int, domains: list[str]) -> dict:
    """Everything that can be decided from files: roster check, sealed check, which domains are still to fetch."""
    queue = data_root / "tranches" / f"tranche{tranche}_never_queued.json"
    out_dir = data_root / "tranches" / f"t{tranche}_out"
    gpu_dir = data_root / "tranches" / f"t{tranche}_gpu"
    if not queue.is_file():
        raise Refused(f"{queue} not found: tranche {tranche} has no never-queued roster")
    roster = {(r["domain"] if isinstance(r, dict) else r).lower() for r in _read_json(queue)}
    unknown = [d for d in domains if d not in roster]
    if unknown:
        raise Refused(f"not in the roster of tranche {tranche} ({queue.name}): {', '.join(unknown)}")
    if not (gpu_dir / ".sealed").is_file():
        raise Refused(f"tranche {tranche} is not sealed ({gpu_dir / '.sealed'} missing): the seal watchdog owns it")
    state_path = out_dir / "state.json"
    if not state_path.is_file():
        raise Refused(f"{state_path} not found")
    done = _read_json(state_path).get("done", {})
    already = [d for d in domains if d in done]
    todo = [d for d in domains if d not in done]
    return {"tranche": tranche, "queue": str(queue), "out_dir": str(out_dir), "state": str(state_path),
            "requested": domains, "already_done": already, "to_fetch": todo}


def fetch_command(plan: dict, *, python: str, data_root: Path, mirror: Path) -> list[str]:
    """The `aw-cc-fetch-run.sh ... -- <fetch>` argv. Paths are relative to `data_root` (the cwd of the run),
    exactly as the watchdog writes them, so the process line matches `NEW_FETCH_RE`."""
    n = plan["tranche"]
    out = f"tranches/t{n}_out"
    return ["bash", str(FETCH_RUN), out, f"logs/tranche{n}-retry760.log", "logs/watchdog.log", f"tranche{n}-retry760",
            "--", *NICE, python, "-u", "-m", "cytir.acquire.discovery_ingest", "fetch",
            "--queue", f"tranches/tranche{n}_never_queued.json", "--only", ",".join(plan["to_fetch"]),
            "--out", f"{out}/passages.jsonl", "--links-out", f"{out}/outlinks.jsonl", "--state", f"{out}/state.json",
            *FETCH_FLAGS, "--egress-state", str(data_root / "EGRESS-BUCKET.json"),
            "--require-local-index", str(mirror)]


def _line(ts: str, tranche: int, dom: str, status: str, **kw) -> str:
    return json.dumps({"ts": ts, "tranche": tranche, "domain": dom, "status": status, **kw}, ensure_ascii=False)


def report(plan: dict, state_path: Path, log_path: Path, now) -> int:
    """One line per requested domain from the state file; appended to the retry log. Exit code of the run."""
    done = _read_json(state_path).get("done", {})
    lines, missing = [], 0
    for d in plan["requested"]:
        if d in done:
            e = done[d]
            fresh = d in plan["to_fetch"]
            lines.append(_line(now(), plan["tranche"], d, "done" if fresh else "already-done",
                               result=e.get("status"), pages=e.get("pages"), rows=e.get("rows")))
        else:
            missing += 1
            lines.append(_line(now(), plan["tranche"], d, "NOT-DONE"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return EXIT_NOT_DONE if missing else 0


def _defaults():
    from cytir.core.paths import operator_settings
    s = operator_settings()
    return Path(s.work_root) / "task634", s.langgate_py, Path(s.hdd_root)


def retry(tranche: int, domains: list[str], *, data_root: Path, python: str, hdd_root: Path, dry_run: bool,
          probe=running_fetches, runner=subprocess.run, now=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())) -> int:
    plan = build_plan(data_root, tranche, domains)
    mirror = hdd_root.joinpath(*MIRROR_SUBPATH)
    cmd = fetch_command(plan, python=python, data_root=data_root, mirror=mirror) if plan["to_fetch"] else []
    busy = probe()
    print(json.dumps({**plan, "command": " ".join(cmd), "cwd": str(data_root), "running_never_queued_fetches": busy,
                      "blocked": bool(busy)}, indent=1, ensure_ascii=False))
    if not plan["to_fetch"]:
        print(f"nothing to do: every named domain of tranche {tranche} is already done")
        return 0
    if dry_run:
        print("DRY RUN: " + ("BLOCKED now (a never-queued fetch runs); would refuse" if busy else "ready; nothing launched"))
        return 0
    lock_path = data_root / ".watchdog.lock"
    lock = open(lock_path, "a+")
    try:
        # The watchdog's own lock, held only from the check to the moment the fetch is visible to pgrep: it
        # cannot start a second never-queued fetch in between, and it is not blocked for the length of the run.
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Refused(f"the watchdog's pass holds {lock_path}; try again in a minute")
        busy = probe()
        if busy:
            raise Refused("a never-queued CC fetch is running (one at a time, CC <= 5 req/s): " + busy[0][:160])
        (data_root / "logs").mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONPATH": f"{REPO_ROOT}{os.pathsep}{os.environ['PYTHONPATH']}"
               if os.environ.get("PYTHONPATH") else str(REPO_ROOT)}
        print(f"{now()} retry-domains: tranche {tranche}: fetching {len(plan['to_fetch'])} domain(s)", flush=True)
        proc = runner(cmd, cwd=str(data_root), env=env, stdin=subprocess.DEVNULL)
    finally:
        lock.close()
    print(f"{now()} retry-domains: tranche {tranche}: fetch rc={getattr(proc, 'returncode', 0)}", flush=True)
    return report(plan, Path(plan["state"]), data_root / "logs" / f"tranche{tranche}-retry-domains.log", now)


def main(argv=None, prog: str = "cyt ops cc-tranche retry-domains") -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    ap.add_argument("--tranche", type=int, required=True)
    ap.add_argument("--domains", required=True, help="a file (JSON list, JSON {tranche: [...]}, or one per line) or a,b,c")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; launch nothing")
    ap.add_argument("--data-root", type=Path, default=None, help="the task634 tree (default: <work_root>/task634)")
    ap.add_argument("--python", default=None, help="the lang-gate interpreter (default: the langgate_py setting)")
    ap.add_argument("--hdd-root", type=Path, default=None, help="holds the CC index mirror (default: the hdd_root setting)")
    a = ap.parse_args(argv)
    if not re.fullmatch(r"\d+", str(a.tranche)) or a.tranche < 7:
        raise Refused("--tranche must be a never-queued tranche (7 or higher)")
    if a.data_root is None or a.python is None or a.hdd_root is None:
        d_root, d_py, d_hdd = _defaults()
        a.data_root, a.python, a.hdd_root = a.data_root or d_root, a.python or d_py, a.hdd_root or d_hdd
    return retry(a.tranche, parse_domains(a.domains, a.tranche), data_root=a.data_root, python=a.python,
                 hdd_root=a.hdd_root, dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main(prog="python -m cytir.ops.cc_tranche_retry"))
