#!/usr/bin/env python3
"""The incumbent's dense-only (and later sparse-only) gate arms, read on PROD (task 865, P12).

WHY (ruling 29 T8 (f), spec `aw_e10` section 5 (b)): the flip gate's control is production itself,
read in the same hours as the candidate. The public `/search` gives the FUSED ranking only, so the leg
arms of the incumbent (`data/flip-gate-contract.v1.json` `legs`) had no instrument on prod. A twin
reading is another index: task 862 asked 20 chooser queries on live prod right after the 8eda143d
release and only 4/20 equalled the gate's answers computed on the twin with the same code -- physical
state, not code (the 627 lesson: the BM25 top-100 is a function of the segment layout).

WHAT IT IS. The gate's own scorer -- the serving code in `cytir/serve`, in-process -- pointed at
prod's Qdrant through the existing read-only tunnel on the build box (`aw-prod-tunnel.service`:
127.0.0.1:6334 -> the serving box's :6333), with prod's own api flags (an env file from `docker
inspect` of the api container) and the LEG taken from `cytir.gates.leg_definition` (the one place both
sides read it: overrides, k = 10, per-row lang/country). Nothing on the serving box changes.

THE BOUNDS (enforced here, not promised):
  * every httpx request of the process (Qdrant and TEI) passes `cytir.index.prod_guard.check` first
    (`httpx.Client.send` is wrapped): a SCROLL to prod is refused before it is sent; every prod query
    POST counts against prod_guard's budget. The 4+ metadata GETs of `read_target` (urllib, uncounted
    by prod_guard by design) are counted HERE and recorded.
  * one prod request in flight at a time (a second is refused), and at most `RATE_PER_S` prod POSTs per
    second (a token bucket per POST, not per query);
  * the WHOLE read aborts on any prod_guard refusal, any prod response >= 500, any transport error to
    prod (timeout, connect, read), and when the rolling p95 of the prod ROUND TRIP (timed inside the
    guard, TEI excluded) over the last 100 POSTs exceeds `--prod-p95-max-s`;
  * a refused, errored or aborted query is never scored;
  * never inside :15-:40 past the hour; never while the drain service is active; never across two prod
    states (target + points_count re-read every `RECHECK_EVERY` queries and at the end);
  * the TEI pin: the TEI's `/info` must serve `cytir.core.model_pin`'s model at revision
    `EMBED_MODEL_REVISION` BEFORE the first prod POST; both are written into the artifact.

BUDGETS. Without a ruling: `--max-prod-calls` 1..400 (prod_guard's per-process budget). With
`--budget-ruling <id>` (the id must appear in `--plan-file` before the run): up to `RULED_CAP_MAX`
POSTs, enforced ACROSS processes by an append-only ledger keyed by the ruling (one line per prod POST,
reserved before it is sent): a re-run after an abort continues the same count, it does not get a fresh
budget. The ledger is keyed by the ruling STRING: an operator-only tool, used with exactly one ruling
string, which the orchestrator writes in plan.md before the run (a misspelt id fails the plan check;
two spellings of one ruling would be two ledgers -- accepted, review 866 round 2). A ruled read is the
gate's arm only: `--k` is refused (k = 10 from `cytir.gates.leg_definition`), and the leg's overrides are
applied after prod's env file, so nothing in that file can change them. A ruled read also needs a `--stop-file` (the orchestrator's public-health watcher writes it;
the command is in the task 865 doc, section 3.5), re-checks the prod api env at the end (`--prod-env-cmd`),
and stops after the first `PRECOMMIT_QUERIES` queries when the mean POSTs per search exceeds
`PRECOMMIT_MAX_MEAN`.

Usage (on the build box, from a checkout of the RELEASE commit prod serves):
    python3 -m cytir.gates.prod_legs --qdrant http://127.0.0.1:6334 \\
        --leg dense-only --weights <fusion_weights.dense_only.json> \\
        --env-file <prod-api.env> --tei http://127.0.0.1:8902 \\
        --queries <file.jsonl: id, query, lang?, country?, gold_urls?> \\
        --max-prod-calls 5 --out <result.json>
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import argparse
import collections
import contextlib
import datetime as _dt
import fcntl
import hashlib
import json
import os
from cytir.core import envnames as _envnames
import pathlib
import re
import subprocess
import sys
import threading
import time
from typing import Callable

from cytir.gates import leg_definition as LD
from cytir.core.serve_layout import PACKAGE_NAMES, SERVER_REL
from cytir.index import prod_guard

SCHEMA = "aw.prod-leg-read.v2"
LEGS = LD.LEGS
#: minutes past the hour when no prod battery read runs (the drain lands at :20)
FORBIDDEN_MINUTES = range(15, 40)
ROOT = pathlib.Path(__file__).resolve().parents[2]
RATE_PER_S = 2.0
RULED_CAP_MAX = 20_000
RECHECK_EVERY = 50
P95_WINDOW = 100
P95_MIN_SAMPLES = 20
PRECOMMIT_QUERIES = 50
PRECOMMIT_MAX_MEAN = 2.4
DRAIN_UNIT = "aw-promotion-nightly.service"
DEFAULT_LEDGER_DIR = pathlib.Path(os.environ.get("AW_PROD_LEGS_LEDGER_DIR",
                                                 str(pathlib.Path.home() / ".aw" / "prod-legs-ledger")))


class ProdLegRefused(RuntimeError):
    """A refusal of the whole read before it starts (wrong target, window, budget, pin, leg)."""


class ProdLegAbort(RuntimeError):
    """The read must stop now; every row from here on is not run."""


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def in_forbidden_window(now: _dt.datetime | None = None) -> bool:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    return now.minute in FORBIDDEN_MINUTES


class Ledger:
    """Append-only, cross-process count of prod POSTs spent under one budget ruling."""

    def __init__(self, directory: pathlib.Path, ruling: str, cap: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", ruling)[:80] + "-" + sha256_text(ruling)[:12]
        self.path = directory / f"{safe}.jsonl"
        self.ruling, self.cap = ruling, cap
        self.path.touch()

    def used(self) -> int:
        with open(self.path) as f:
            return sum(1 for line in f if line.strip())

    def reserve(self) -> int:
        """Reserve one POST; raises ProdLegAbort at the cap. A reserved POST counts even if it fails."""
        with open(self.path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            n = sum(1 for line in f if line.strip())
            if n >= self.cap:
                raise ProdLegAbort(f"budget ruling {self.ruling!r}: {n} of {self.cap} prod POSTs already "
                                   f"spent (ledger {self.path}) -- the cap holds across processes")
            f.write(json.dumps({"at": time.time(), "pid": os.getpid()}) + "\n")
            f.flush()
            os.fsync(f.fileno())
            return n + 1


class Guard:
    """Routes every httpx request of this process through `prod_guard.check` and the read's bounds.

    Records refusals and abort causes, so a caller tells 'refused/aborted' from 'found nothing'."""

    def __init__(self, *, rate_per_s: float | None = RATE_PER_S, ledger: Ledger | None = None,
                 p95_max_s: float | None = None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.prod_posts = 0
        self.prod_requests = 0
        self.requests_total = 0
        self.refusals: list[str] = []
        self.abort: str | None = None
        self.latencies: collections.deque = collections.deque(maxlen=P95_WINDOW)
        self.rate_per_s, self.ledger, self.p95_max_s = rate_per_s, ledger, p95_max_s
        self._clock, self._sleep = clock, sleep
        self._last_post: float | None = None
        self._inflight = threading.Lock()

    def _set_abort(self, why: str) -> None:
        if self.abort is None:
            self.abort = why

    def p95(self) -> float | None:
        if len(self.latencies) < P95_MIN_SAMPLES:
            return None
        xs = sorted(self.latencies)
        return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]

    def _before(self, request) -> bool:
        """Checks before a request is sent; returns whether it goes to prod as a POST."""
        self.requests_total += 1
        url = f"{request.url.scheme}://{request.url.host}:{request.url.port or 80}"
        try:
            is_prod = prod_guard.check(url, request.method, request.url.path)
        except prod_guard.ProdReadRefused as exc:
            self.refusals.append(str(exc))
            self._set_abort(f"prod_guard refused: {exc}")
            raise
        if not is_prod:
            return False
        self.prod_requests += 1
        post = request.method.upper() == "POST"
        if post:
            if self.ledger is not None:
                try:
                    self.ledger.reserve()
                except ProdLegAbort as exc:
                    self.refusals.append(str(exc))
                    self._set_abort(str(exc))
                    raise prod_guard.ProdReadRefused(str(exc)) from exc
            if self.rate_per_s:
                gap = 1.0 / self.rate_per_s
                now = self._clock()
                if self._last_post is not None and now - self._last_post < gap:
                    self._sleep(gap - (now - self._last_post))
                self._last_post = self._clock()
            self.prod_posts += 1
        return post

    @contextlib.contextmanager
    def installed(self):
        import httpx
        orig = httpx.Client.send
        guard = self

        def send(client, request, *a, **kw):
            to_prod = _targets_prod(request)
            if to_prod and not guard._inflight.acquire(blocking=False):
                msg = "a second prod request while one is in flight: this read is strictly sequential"
                guard.refusals.append(msg)
                guard._set_abort(msg)
                raise prod_guard.ProdReadRefused(msg)
            try:
                post = guard._before(request)
                t0 = guard._clock()
                try:
                    resp = orig(client, request, *a, **kw)
                except httpx.TransportError as exc:
                    if to_prod:
                        guard._set_abort(f"transport error to prod: {type(exc).__name__}: {exc}")
                    raise
                if to_prod:
                    if post:
                        guard.latencies.append(guard._clock() - t0)
                        p95 = guard.p95()
                        if guard.p95_max_s and p95 is not None and p95 > guard.p95_max_s:
                            guard._set_abort(f"prod round-trip p95 {p95:.2f}s > {guard.p95_max_s}s over the "
                                             f"last {len(guard.latencies)} POSTs")
                    if resp.status_code >= 500:
                        guard._set_abort(f"prod answered {resp.status_code} to {request.method} {request.url.path}")
                return resp
            finally:
                if to_prod:
                    guard._inflight.release()

        httpx.Client.send = send
        restore_check = _silence_qdrant_version_check()
        try:
            yield self
        finally:
            restore_check()
            httpx.Client.send = orig


def _silence_qdrant_version_check():
    """qdrant-client starts a daemon thread at construction that sends `GET /` (a version check
    that only ever warns). It raced the first search POST, and the strictly-sequential guard
    refused whichever came second (task 916). The check is not part of any read: skip it for the
    duration of the guard; the serving code under test is not edited. Returns the restore."""
    try:
        from qdrant_client import qdrant_remote
    except ImportError:
        return lambda: None
    orig = qdrant_remote.QdrantRemote.__dict__["_check_compatibility"]
    qdrant_remote.QdrantRemote._check_compatibility = staticmethod(lambda *a, **kw: None)

    def restore():
        qdrant_remote.QdrantRemote._check_compatibility = orig
    return restore


def _targets_prod(request) -> bool:
    url = f"{request.url.scheme}://{request.url.host}:{request.url.port or 80}"
    m = re.match(r"^/collections/([^/?]+)", request.url.path or "")
    return prod_guard.is_prod(url, m.group(1) if m else None)


class MetaReader:
    """The uncounted metadata GETs (`/aliases`, `/collections/<alias>`), counted here."""

    def __init__(self, opener=IH.urlopen) -> None:
        self.gets = 0
        self.opener = opener

    def read_target(self, qdrant: str, alias: str) -> dict:
        base = qdrant.rstrip("/")
        self.gets += 1
        with self.opener(base + "/aliases", timeout=10) as r:
            aliases = {a["alias_name"]: a["collection_name"] for a in json.load(r)["result"]["aliases"]}
        self.gets += 1
        with self.opener(f"{base}/collections/{alias}", timeout=10) as r:
            info = json.load(r)["result"]
        return {"qdrant": base, "alias": alias, "collection": aliases.get(alias),
                "points_count": info.get("points_count"), "status": info.get("status")}


def read_target(qdrant: str, alias: str, *, opener=IH.urlopen) -> dict:
    return MetaReader(opener).read_target(qdrant, alias)


def load_env_file(path: str) -> dict[str, str]:
    """The `AW_*` lines of a KEY=VALUE file (as `docker inspect -f '{{range .Config.Env}}{{println
    .}}{{end}}'` prints). Only `AW_*`: the container's PATH, HOME, LANG and friends are the
    container's, not serving flags, and must not replace this process's."""
    return parse_env_text(pathlib.Path(path).read_text())


def parse_env_text(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if _envnames.is_knob(k):
            out[k] = v
    return out


#: The api image's layout (server/Dockerfile: WORKDIR /app, the package copied under /app/<package>, which holds
#: <package>/serve). A flag that names a file inside the image names the same file in this checkout. The image on
#: the box is NOT this tree's: the live one was built before the rename and holds the OLD package name under
#: /app; the first image after the rename holds the new one. A live env dump (`docker inspect`) shows whichever runs, so BOTH
#: package names map, to this tree's own serve dir and package dir. (The parity gate's own-compose rule,
#: `serve_parity.refuse_foreign_image_prefix`, reads a different input -- the tree's own compose -- and is not used here.)
IMAGE_PREFIXES = {**{f"/app/{name}/serve/": ROOT / SERVER_REL for name in PACKAGE_NAMES},
                  **{f"/app/{name}/": ROOT / SERVER_REL.split("/")[0] for name in PACKAGE_NAMES}}


def map_image_paths(env: dict[str, str]) -> dict[str, str]:
    """Rewrite `/app/...` values to this checkout's files; refuse one that is missing (a flag
    that silently names nothing would read as a different serving config) and one under an
    `/app/` prefix no image of ours has (the serving loader answers an unreadable table with `{}`,
    so a value passed through unmapped would score an arm without the feature it configures)."""
    out = dict(env)
    for k, v in env.items():
        for prefix, local in IMAGE_PREFIXES.items():
            if v.startswith(prefix):
                path = local / v[len(prefix):]
                if not path.exists():
                    raise ProdLegRefused(f"{k}={v}: {path} does not exist in this checkout")
                out[k] = str(path)
                break
        else:
            if v.startswith("/app/"):
                raise ProdLegRefused(f"{k}={v}: an image path under none of the known prefixes "
                                     f"{sorted(IMAGE_PREFIXES)}; refused, not passed through unmapped")
    return out


def check_tei_pin(tei: str, *, get=None) -> dict:
    """The TEI on this box must serve the pinned model at the pinned revision (spec section 5 (b));
    the serving code only LOGS a mismatch. Read before the first prod POST; returned for the artifact."""
    from cytir.core import model_pin
    if model_pin.embed_model_revision() != model_pin.EMBED_MODEL_REVISION:
        raise ProdLegRefused("the embed revision is overridden in this environment "
                             f"({model_pin.embed_model_revision()}); the gate reads the pin "
                             f"{model_pin.EMBED_MODEL_REVISION}")
    if get is None:
        get = lambda url: IH.fetch_json(url, timeout=10)   # noqa: E731
    info = get(tei.rstrip("/") + "/info")
    got = {"model_id": info.get("model_id"), "model_sha": info.get("model_sha")}
    if got["model_id"] != model_pin.EMBED_MODEL_ID or got["model_sha"] != model_pin.EMBED_MODEL_REVISION:
        raise ProdLegRefused(f"TEI {tei} serves {got}, the pin is {model_pin.EMBED_MODEL_ID} @ "
                             f"{model_pin.EMBED_MODEL_REVISION} -- refused before any prod request")
    return {"tei": tei, **got}


def rank_of(urls: list[str], gold_urls, gold_domains=()) -> int | None:
    from cytir.gates import authority as gate
    return gate.rank_of(urls, list(gold_urls or []), list(gold_domains or []))


def read_legs(queries: list[dict], *, search: Callable[..., list[dict]], k: int, guard: Guard,
              pace: Callable[[], None], clock: Callable[[], _dt.datetime] | None = None,
              before_query: Callable[[int], str | None] | None = None,
              precommit: tuple[int, float] | None = None) -> list[dict]:
    """One row per query, sequentially. The read STOPS at the first refusal, abort, window or
    `before_query` objection; rows not reached are NOT-RUN. A refused/errored row is never scored."""
    rows: list[dict] = []
    stop: str | None = None
    for i, q in enumerate(queries):
        if stop is None and in_forbidden_window(clock() if clock else None):
            stop = "inside :15-:40 (the drain's window)"
        if stop is None and before_query is not None:
            stop = before_query(i)
        if stop is None and guard.abort:
            stop = guard.abort
        if stop is None and precommit and i == precommit[0]:
            ok = [r for r in rows if r["status"] == "OK"]
            mean = sum(r["prod_posts"] for r in ok) / len(ok) if ok else 0.0
            if mean > precommit[1]:
                stop = (f"pre-commit: {mean:.2f} prod POSTs per search over the first {precommit[0]} "
                        f"queries > {precommit[1]}")
        if stop is not None:
            rows.append({"id": q.get("id"), "query": q["query"], "status": "NOT-RUN", "reason": stop[:300]})
            continue
        if i:
            pace()
        before_posts, before_ref = guard.prod_posts, len(guard.refusals)
        t0 = time.monotonic()
        kw = LD.search_kwargs(q, k)
        row: dict = {"id": q.get("id"), "query": q["query"], "lang": kw["lang"], "country": kw["country"]}
        if q.get("cls") is not None:      # the class the flip gate scores this row under (task 926)
            row["cls"] = q["cls"]
        try:
            hits = search(q["query"], **kw)
            err = None
        except Exception as exc:   # noqa: BLE001 -- recorded; a failed query is never a miss
            hits, err = [], f"{type(exc).__name__}: {exc}"
        row["prod_posts"] = guard.prod_posts - before_posts
        row["seconds"] = round(time.monotonic() - t0, 3)
        refused = guard.refusals[before_ref:]
        if refused:
            row.update(status="REFUSED", reason=refused[0][:300])
            stop = "a refusal: the code path or the budget changed (review 866 S4)"
        elif guard.abort:
            row.update(status="ABORTED", reason=guard.abort[:300])
            stop = guard.abort
        elif err:
            row.update(status="ERROR", reason=err[:300])
        else:
            urls = [h.get("url") or "" for h in hits]
            row.update(status="OK", urls=urls)
            if q.get("gold_urls") or q.get("gold_domains"):
                row["gold_rank"] = rank_of(urls, q.get("gold_urls"), q.get("gold_domains"))
        rows.append(row)
    return rows


def _gate_set_version() -> str:
    from cytir.gates import authority as gate
    return gate.GATE_SET_VERSION


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        rev = ROOT / "REVISION"          # a `git archive` copy carries its commit here
        return rev.read_text().strip() if rev.is_file() else "unknown"


def unit_active(unit: str) -> str:
    try:
        return subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unreadable"


def parse_args(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    ap.add_argument("--qdrant", required=True, help="prod's Qdrant as this box reaches it (the tunnel)")
    ap.add_argument("--alias", default=prod_guard.DEFAULT_PROD_ALIAS)
    ap.add_argument("--leg", required=True, choices=LEGS)
    ap.add_argument("--weights", help="the leg's fusion-weights file (dense-only: the contract's)")
    ap.add_argument("--env-file", required=True, help="prod's api environment (docker inspect of the api)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--qvec-cache", help="the gate's immutable query-vector cache")
    src.add_argument("--tei", help="a revision-pinned TEI on THIS box (never prod's)")
    qs = ap.add_mutually_exclusive_group(required=True)
    qs.add_argument("--query", action="append")
    qs.add_argument("--queries", help="jsonl: {id, query, lang?, country?, gold_urls?}")
    ap.add_argument("--k", type=int, default=None,
                    help=f"default {LD.GATE_K} (the gate's); refused on a --budget-ruling read")
    ap.add_argument("--max-prod-calls", type=int, required=True,
                    help=f"prod POST budget: 1..{prod_guard.PROD_READ_BUDGET}, or up to {RULED_CAP_MAX} "
                         "with --budget-ruling")
    ap.add_argument("--budget-ruling", help="the ruling id (in --plan-file) that grants a budget above 400")
    ap.add_argument("--plan-file", default=str(ROOT / "plan.md"))
    ap.add_argument("--ledger-dir", default=str(DEFAULT_LEDGER_DIR))
    ap.add_argument("--stop-file", help="the watcher's stop file (required with --budget-ruling)")
    ap.add_argument("--prod-p95-max-s", type=float,
                    help="abort when the prod round-trip p95 exceeds this (default: the monitor's "
                         "AW_MONITOR_MAX_SMOKE_LATENCY)")
    ap.add_argument("--qdrant-timeout-s", type=float, default=None,
                    help="the serving code's Qdrant client timeout for this read (default: prod's own "
                         "AW_QDRANT_TIMEOUT); a cold query takes 8-12 s inside Qdrant, so a ruled read sets 30. "
                         "Ranking is timeout-independent; recorded in the receipt")
    ap.add_argument("--prod-env-cmd", help="prints prod's live api env (re-read at the end with a ruling)")
    ap.add_argument("--drain-unit", default=DRAIN_UNIT)
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def _monitor_latency() -> float | None:
    p = ROOT / "deploy" / "aw-monitor.env"
    if p.is_file():
        for line in p.read_text().splitlines():
            if line.startswith("AW_MONITOR_MAX_SMOKE_LATENCY="):
                try:
                    return float(line.split("=", 1)[1].strip())
                except ValueError:
                    return None
    return None


def main(argv=None, prog=None) -> int:
    a = parse_args(argv, prog)
    if not prod_guard.is_prod(a.qdrant, a.alias):
        raise ProdLegRefused(f"{a.qdrant} ({a.alias}) is not prod by prod_guard: this tool reads PROD only")
    ledger = None
    if a.budget_ruling and a.k is not None:
        raise ProdLegRefused(f"--k on a --budget-ruling read: a ruled read is the gate's arm only, k = {LD.GATE_K} "
                             "(cytir.gates.leg_definition); no arm-definition override is accepted")
    a.k = LD.GATE_K if a.k is None else a.k
    if a.budget_ruling:
        if not 0 < a.max_prod_calls <= RULED_CAP_MAX:
            raise ProdLegRefused(f"--max-prod-calls {a.max_prod_calls}: a ruled read is 1..{RULED_CAP_MAX}")
        plan = pathlib.Path(a.plan_file)
        if not plan.is_file() or a.budget_ruling not in plan.read_text():
            raise ProdLegRefused(f"--budget-ruling {a.budget_ruling!r} is not written in {plan}: the ruling "
                                 "is recorded before the run")
        if not a.stop_file:
            raise ProdLegRefused("a ruled read needs --stop-file (the orchestrator's health watcher writes it)")
        if not a.prod_env_cmd:
            raise ProdLegRefused("a ruled read needs --prod-env-cmd (the api env is re-read at the end)")
        ledger = Ledger(pathlib.Path(a.ledger_dir), a.budget_ruling, a.max_prod_calls)
        if ledger.used() >= a.max_prod_calls:
            raise ProdLegRefused(f"budget ruling {a.budget_ruling!r} is spent: {ledger.used()} of "
                                 f"{a.max_prod_calls} POSTs in {ledger.path}")
    elif not 0 < a.max_prod_calls <= prod_guard.PROD_READ_BUDGET:
        raise ProdLegRefused(f"--max-prod-calls {a.max_prod_calls}: must be 1..{prod_guard.PROD_READ_BUDGET} "
                             "(prod_guard's per-process budget); more needs --budget-ruling")
    if in_forbidden_window():
        raise ProdLegRefused("inside :15-:40 past the hour (the drain's window): start outside it")
    if unit_active(a.drain_unit) != "inactive":
        raise ProdLegRefused(f"{a.drain_unit} is not inactive: a drain may be writing prod")
    weights_sha = sha256_file(a.weights) if a.weights else None
    overrides = LD.leg_env(a.leg, os.path.abspath(a.weights) if a.weights else None, weights_sha)
    queries = ([{"id": f"q{i + 1}", "query": q} for i, q in enumerate(a.query)] if a.query else
               [json.loads(l) for l in pathlib.Path(a.queries).read_text().splitlines() if l.strip()])

    env_text = pathlib.Path(a.env_file).read_text()
    env = map_image_paths(parse_env_text(env_text))
    live_env_start = None
    if a.prod_env_cmd:
        live_env_start = subprocess.run(a.prod_env_cmd, shell=True, capture_output=True, text=True,
                                        timeout=60, check=True).stdout
        if parse_env_text(live_env_start) != parse_env_text(env_text):
            raise ProdLegRefused("--env-file differs from prod's live api env (--prod-env-cmd): re-capture it")
    for key in ("AW_QDRANT_URL", "AW_COLLECTION", "AW_FUSION_WEIGHTS", "AW_TEI_URL"):
        _envnames.pop_knob(env, key)
    os.environ.update(env)
    os.environ.update(overrides)
    remaining = (a.max_prod_calls - ledger.used()) if ledger else a.max_prod_calls
    os.environ.update({"AW_QDRANT_URL": a.qdrant, "AW_COLLECTION": a.alias,
                       prod_guard.BUDGET_ENV: str(remaining)})
    tei_pin = None
    if a.tei:
        os.environ["AW_TEI_URL"] = a.tei
        tei_pin = check_tei_pin(a.tei)
    p95_max = a.prod_p95_max_s if a.prod_p95_max_s is not None else _monitor_latency()
    meta = MetaReader()
    target_before = meta.read_target(a.qdrant, a.alias)
    sys.path[:0] = [str(ROOT)]
    guard = Guard(ledger=ledger, p95_max_s=p95_max)

    def before_query(i: int) -> str | None:
        if a.stop_file and pathlib.Path(a.stop_file).exists():
            return f"stop file {a.stop_file} present (the health watcher)"
        st = unit_active(a.drain_unit)
        if st != "inactive":
            return f"{a.drain_unit} is {st}"
        if i and i % RECHECK_EVERY == 0:
            now = meta.read_target(a.qdrant, a.alias)
            if (now["collection"], now["points_count"]) != (target_before["collection"],
                                                             target_before["points_count"]):
                return f"prod changed during the read: {target_before} -> {now}"
        return None

    with guard.installed():
        if a.qvec_cache:
            from cytir.gates import authority as gate
            gate.install_cached_embedder(a.qvec_cache)
        from cytir.serve import retrieval
        qdrant_timeout = LD.apply_qdrant_timeout(retrieval, a.qdrant_timeout_s)
        effective = retrieval.effective_config()
        rows = read_legs(queries, search=lambda q, **kw: retrieval.search(q, **kw), k=a.k, guard=guard,
                         pace=lambda: prod_guard.pace(a.qdrant, a.alias), before_query=before_query,
                         precommit=(PRECOMMIT_QUERIES, PRECOMMIT_MAX_MEAN) if ledger else None)
    target_after = meta.read_target(a.qdrant, a.alias)
    invalid = []
    if (target_after["collection"], target_after["points_count"]) != (target_before["collection"],
                                                                       target_before["points_count"]):
        invalid.append("prod's target or count changed between the start and the end")
    live_env_end = None
    if a.prod_env_cmd:
        live_env_end = subprocess.run(a.prod_env_cmd, shell=True, capture_output=True, text=True,
                                      timeout=60).stdout
        if sha256_text(live_env_end) != sha256_text(live_env_start or ""):
            invalid.append("prod's api env changed during the read")
    counts = {s: sum(1 for r in rows if r["status"] == s)
              for s in ("OK", "REFUSED", "ERROR", "ABORTED", "NOT-RUN")}
    doc = {
        "schema": SCHEMA,
        "is_gate_arm": False,
        "is_gate_arm_note": ("not an arm the flip gate reads: a sealer must refuse any REFUSED/ERROR/"
                             "ABORTED/NOT-RUN row and pair rows by id with the candidate arm"),
        "leg": a.leg,
        "leg_overrides": overrides,
        "k": a.k,
        "gate_set_version": _gate_set_version(),
        "measured_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_before": target_before, "target_after": target_after,
        "weights": {"path": os.path.abspath(a.weights), "sha256": weights_sha} if a.weights else None,
        "env_file": {"path": os.path.abspath(a.env_file), "sha256": sha256_file(a.env_file), "keys": len(env)},
        "live_env": ({"start_sha256": sha256_text(live_env_start or ""),
                      "end_sha256": sha256_text(live_env_end or "")} if a.prod_env_cmd else None),
        "embedder": ({"qvec_cache": a.qvec_cache, "sha256": sha256_file(a.qvec_cache)} if a.qvec_cache
                     else tei_pin),
        "effective_config": effective,
        "qdrant_timeout_s": qdrant_timeout,
        "qdrant_timeout_s_override": a.qdrant_timeout_s,
        "code": {"git_head": _git_head(),
                 "retrieval_sha256": sha256_file(ROOT / SERVER_REL / "retrieval.py")},
        "budget": {"max_prod_calls": a.max_prod_calls, "ruling": a.budget_ruling,
                   "ledger": str(ledger.path) if ledger else None,
                   "ledger_used_after": ledger.used() if ledger else None},
        "prod_posts": guard.prod_posts,
        "prod_requests_via_client": guard.prod_requests,
        "prod_metadata_gets": meta.gets,
        "requests_total_via_client": guard.requests_total,
        "prod_round_trip_p95_s": guard.p95(),
        "prod_p95_max_s": p95_max,
        "abort": guard.abort,
        "invalid": invalid,
        "refusals": len(guard.refusals),
        "rows": rows,
        "counts": counts,
    }
    pathlib.Path(a.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"out": a.out, "prod_posts": guard.prod_posts, "abort": guard.abort,
                      "invalid": invalid, **counts}))
    return 0 if counts["OK"] == len(queries) and not invalid else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ProdLegRefused as exc:
        sys.exit(f"REFUSED: {exc}")
