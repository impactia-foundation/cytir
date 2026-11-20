"""`cyt ops writer-locks --check`: is a drain writer lock stale? A READ-ONLY look at the registry.

Reads the two registry lock records (`TWIN-WRITE-LOCK`, `PROD-WRITE-LOCK`) in the configured lock
directory (`cytir.ops.write_lock.effective_location()`: the operator settings, no path of its own) and
reports, for each: held or free, task, owner, host, pid, taken-at, age.

It NEVER changes anything: no lock is released, removed, renamed or rewritten, and no network
connection is opened (a registry on another host is refused by name: run the check on the lock
host). A test enforces this by `ast` over this file. The serving-box token lock
(`/tmp/aw_prod_writer.lock` on the production host) is NOT checked: it lives on another host and
carries no host or age, so a read would need ssh and could not say "stale".

STALE means ALL of:
  1. the lock is held (the record exists);
  2. its age (now minus `at=`) is over `--max-age-min` (default 120: no setting carries the drain
     unit's budget, which only the systemd unit knows, so the default is a floor under a pack's
     normal hold, and the pid test below, not the age, is what proves a writer dead);
  3. the record's host is this host AND its pid is not alive (`os.kill(pid, 0)`), or the record's
     host is another host / its pid is not a number (liveness "unknown": stale by age only).
A pid counts as the holder only while it is alive AND its command line is python (argv0 basename starts
with `python`), `-m cytir`, `cytir.cli`, or argv0 is `cyt`, AND (when /proc is readable) it started no more than
60 s after the record's `at=`. A live pid that fails those tests is not the holder: "pid reused", and
STALE when the lock is old. A zombie (empty command line) is dead. An `at=` more than 5 min in the
future is an unreadable record (rc 2).

Exit codes: 0 nothing held, or held and live (or young and reused); 1 at least one STALE lock;
2 a record cannot be read or parsed (never a silent pass).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import socket

from cytir.ops import write_lock as WL

DEFAULT_MAX_AGE_MIN = 120
START_SLACK_S = 60          # a holder cannot start more than this long after its record's at=
FUTURE_SLACK_MIN = 5        # an at= further ahead of now than this is unreadable
DEAD_LIKE = ("dead", "unknown", "reused", "started-after")
NOT_CHECKED = ("serving-box token lock /tmp/aw_prod_writer.lock: not checked (another host, no host or age "
               "in its record; reading it needs ssh and this check opens no connection)")
STALE_STEPS = (
    "STALE writer lock: only a human releases it, and only after checking:",
    "  1. the unit is inactive and no `cyt index promote` / drain process runs (systemctl status, ps);",
    "  2. twin == prod (the transaction did not die between the two legs);",
    "  3. then remove the file by hand and log it in LOCK-RELEASES.log; never by TTL or auto-release.",
)


class RecordUnreadable(Exception):
    """A lock record is there but cannot be read or does not hold host, pid and time."""


def parse_record(text: str) -> dict:
    fields = {}
    for tok in text.split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            fields.setdefault(k, v)
    for need in ("task", "host", "pid", "at"):
        if not fields.get(need):
            raise RecordUnreadable(f"record has no {need}=: {text.strip()!r}")
    try:
        fields["_at"] = dt.datetime.strptime(fields["at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except ValueError as exc:
        raise RecordUnreadable(f"record at={fields['at']!r} is not a UTC ISO time") from exc
    return fields


def _cmdline(pid: int) -> str | None:
    try:
        return pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return None


def _is_holder_cmdline(cmd: str) -> bool:
    argv = cmd.split()
    base = os.path.basename(argv[0]) if argv else ""
    return (base.startswith("python") or base == "cyt" or "cytir.cli" in cmd
            or re.search(r"(^|\s)-m\s+cytir(\s|$)", cmd) is not None)


def _start_epoch(pid: int) -> float | None:
    """Process start time in epoch seconds from /proc (field 22 after the last ')'), or None if unreadable."""
    try:
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        boot = next(float(line.split()[1]) for line in pathlib.Path("/proc/stat").read_text(encoding="utf-8").splitlines()
                    if line.startswith("btime "))
        ticks = int(stat[stat.rindex(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError, StopIteration):
        return None
    return boot + ticks / os.sysconf("SC_CLK_TCK")


def pid_liveness(pid_text: str, at_epoch: float | None = None) -> str:
    """'dead', 'live', 'reused' (alive, but not a python / aw process), 'started-after' (alive, but started
    after the record's at=: a reused pid) or 'unknown' (not an ASCII pid)."""
    if not (pid_text.isascii() and pid_text.isdigit()):
        return "unknown"
    pid = int(pid_text)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "dead"
    except PermissionError:
        return "live"          # exists, owned by someone else
    cmd = _cmdline(pid)
    if cmd is None:
        return "live"          # /proc unreadable: today's behaviour
    if cmd == "":
        return "dead"          # zombie / defunct: no command line
    if not _is_holder_cmdline(cmd):
        return "reused"
    if at_epoch is not None:
        started = _start_epoch(pid)
        if started is not None and started > at_epoch + START_SLACK_S:
            return "started-after"
    return "live"


def inspect(name: str, text: str | None, *, now: dt.datetime, max_age_min: float, this_host: str) -> dict:
    if not text:
        return {"lock": name, "held": False, "status": "free"}
    rec = parse_record(text)
    age_min = (now - rec["_at"]).total_seconds() / 60.0
    if age_min < -FUTURE_SLACK_MIN:
        raise RecordUnreadable(f"{name}: record at= is in the future ({rec['at']}, more than "
                               f"{FUTURE_SLACK_MIN} min ahead of now): the record or the clock is wrong")
    live = pid_liveness(rec["pid"], rec["_at"].timestamp()) if rec["host"] == this_host else "unknown"
    old = age_min > max_age_min
    note = None
    if live == "reused":
        note = "pid reused (command line is not python or aw)"
    elif live == "started-after":
        note = "pid reused (process started after the lock was taken)"
    if live in DEAD_LIKE and old:
        status = "STALE"
    elif live == "live":
        status = "held and live"
    elif live == "reused":
        status = "held, pid reused: unknown"
    else:
        status = "held, younger than the limit"
    return {"lock": name, "held": True, "task": rec["task"], "owner": rec.get("owner"), "host": rec["host"],
            "pid": rec["pid"], "at": rec["at"], "age_min": round(age_min, 1), "liveness": live, "status": status,
            "note": note}


def _read(directory: str, name: str) -> str | None:
    path = pathlib.Path(directory) / name
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise RecordUnreadable(f"{name}: cannot read {path}: {exc}") from exc


def check(max_age_min: float, *, now: dt.datetime | None = None, this_host: str | None = None) -> tuple[int, dict]:
    now = now or dt.datetime.now(dt.timezone.utc)
    this_host = this_host or socket.gethostname()
    host, directory = WL.effective_location()
    out = {"lock_host": host, "lock_dir": directory, "max_age_min": max_age_min, "locks": [],
           "not_checked": NOT_CHECKED, "errors": []}
    if host not in (WL.LOCAL_HOST, this_host):
        out["errors"].append(f"the registry is on host {host!r}, this host is {this_host!r}: run the check "
                             "there (this verb opens no network connection)")
        return 2, out
    if not pathlib.Path(directory).is_dir():
        out["errors"].append(f"the lock directory {directory!r} does not exist: refusing to read it as free")
        return 2, out
    for name in WL.LOCKS:
        try:
            text = _read(directory, name)
            if text == "":
                raise RecordUnreadable(f"{name}: the file exists and is empty (a token-less lock)")
            out["locks"].append(inspect(name, text, now=now, max_age_min=max_age_min, this_host=this_host))
        except RecordUnreadable as exc:
            out["errors"].append(str(exc))
            out["locks"].append({"lock": name, "held": True, "status": "UNREADABLE"})
    if out["errors"]:
        return 2, out
    return (1 if any(r["status"] == "STALE" for r in out["locks"]) else 0), out


def _print(rc: int, out: dict) -> None:
    print(f"writer-locks: registry {out['lock_dir']} on {out['lock_host']}; limit {out['max_age_min']} min")
    for r in out["locks"]:
        if r["status"] == "free":
            print(f"  {r['lock']}: free")
        elif r["status"] == "UNREADABLE":
            print(f"  {r['lock']}: UNREADABLE")
        else:
            print(f"  {r['lock']}: {r['status']}: task={r['task']} owner={r['owner']} host={r['host']} "
                  f"pid={r['pid']} (pid {r['liveness']}) at={r['at']} age={r['age_min']} min"
                  + (f" [{r['note']}]" if r.get("note") else ""))
    for e in out["errors"]:
        print(f"writer-locks: ERROR: {e}")
    print(f"  {out['not_checked']}")
    if rc == 1:
        print("\n".join(STALE_STEPS))


def main(argv: list[str] | None = None, prog: str = "cyt ops writer-locks") -> int:
    ap = argparse.ArgumentParser(prog=prog, description="read-only check for a stale drain writer lock")
    ap.add_argument("--check", action="store_true", required=True, help="exit 1 on a stale lock, 2 if unreadable")
    ap.add_argument("--max-age-min", type=float, default=DEFAULT_MAX_AGE_MIN,
                    help=f"age over which a held lock can be stale (default {DEFAULT_MAX_AGE_MIN}: no setting "
                         "carries the drain unit's budget)")
    ap.add_argument("--json", action="store_true", help="print the same facts as one JSON object")
    a = ap.parse_args(argv)
    try:
        rc, out = check(a.max_age_min)
    except Exception as exc:          # settings refused by name, etc.: never a silent pass
        rc, out = 2, {"lock_host": None, "lock_dir": None, "max_age_min": a.max_age_min, "locks": [],
                      "not_checked": NOT_CHECKED, "errors": [f"{type(exc).__name__}: {exc}"]}
    out["rc"] = rc
    if a.json:
        print(json.dumps(out, indent=1, default=str))
    else:
        _print(rc, out)
    return rc
