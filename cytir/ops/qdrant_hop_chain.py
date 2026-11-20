"""qdrant_hop_chain.py — the staged Qdrant version walk, behind `cyt ops qdrant-hop-verify chain`.

Carries a collection across a version gap that one upgrade cannot cross (ADR-070, task 305: Qdrant
moved RocksDB -> Gridstore between 1.13 and 1.18; a 1.12.x snapshot restored straight into 1.18.3
fails). Self-hosted has to reproduce the staged walk by hand; this is that walk. It replaces the
shell driver of the same procedure (task 2370). Every hop is verified by `cytir.ops.qdrant_hop_verify`
(count, config, indexes, content digest, replayed queries) against one baseline fingerprint; any
difference stops the chain.

Two modes (they answer different questions):

  snapshot  build side. Each hop = a fresh server of version N with an EMPTY storage dir, restore the
            previous hop's snapshot, verify, re-snapshot. Non-destructive: the source snapshot is never
            modified, so any hop can be re-run.
  inplace   serving side. Each hop = the SAME storage dir handed to a newer binary, which migrates it on
            start-up; verify; stop. No downgrade exists afterwards: needs
            --i-understand-inplace-is-irreversible. Run it on a COPY of the storage dir first.

Both modes resume: each finished hop writes a marker under WORK/state and a re-run skips it.

Operator runbook (steps the code does NOT do for you):
  1. Take a snapshot (or a stopped copy of the storage dir) of the collection first. Never aim this at
     the serving Qdrant or a live twin: the hop container must listen on a port that is not the serving
     port (6333/6334 or AW_PROD_URL); the verifier refuses the scroll there.
  2. A baseline taken with --baseline-url off a LIVE prod server is REFUSED by the verifier
     (prod_guard forbids the scroll). Use --baseline FILE (captured earlier from a stopped copy with
     `cyt ops qdrant-hop-verify capture`), or snapshot mode, which fingerprints the source itself.
  3. Pick the ladder with --chain (default is the ladder validated in task 305). Do not skip a minor
     version.
  4. Run with --dry-run first: it prints every docker command and touches nothing.
  5. After a green chain, restore FINAL_SNAPSHOT (snapshot mode) into the target version, or restart the
     serving container on the migrated dir (inplace mode), then run the promote gates. This tool does
     not flip anything.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from cytir.ops import qdrant_hop_verify as HV

DEFAULT_CHAIN = "v1.13.6,v1.16.3,v1.18.3"   # validated ladder, task 305
HOP_COPY_REASON = "stopped hop copy, not serving"
SERVING_PORTS = (HV.SERVING_PORT, 6334)


class HopError(RuntimeError):
    """A hop failed. The chain stops; nothing downstream is trustworthy."""


# ------------------------------------------------------------------- helpers --
def container_name(prefix: str, ver: str) -> str:
    return f"{prefix}-{ver.replace('.', '-')}"


def size_kb(path: Path) -> int:
    """du -sk equivalent (file or dir), 0 if missing."""
    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return (p.stat().st_size + 1023) // 1024
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return (total + 1023) // 1024


def free_kb(path: Path) -> int:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free // 1024


def disk_gate(input_kb: int, avail_kb: int, factor: float) -> int:
    """Raise unless avail_kb covers factor x input. Returns the KiB needed."""
    need = int(input_kb * factor)
    if avail_kb < need:
        raise HopError(f"insufficient free space: need ~{need // 1048576} GiB "
                       f"({factor}x input), have {avail_kb // 1048576} GiB")
    return need


def docker_run_args(*, name: str, ver: str, port: int, image_repo: str, storage: Path, snaps: Path,
                    input_dir: Path | None, input_subdir: str, nofile: int) -> list[str]:
    """The `docker run` argv for one hop container (loopback-only publish).

    The input snapshot is mounted INSIDE the snapshots dir: 1.18.3 rejects a recover location outside
    it (measured in the task-339 dry run). The nested mount works on every hop.
    """
    args = ["docker", "run", "-d", "--name", name, "--ulimit", f"nofile={nofile}:{nofile}",
            "-p", f"127.0.0.1:{port}:6333",
            "-v", f"{storage}:/qdrant/storage", "-v", f"{snaps}:/qdrant/snapshots"]
    if input_dir is not None:
        args += ["-v", f"{input_dir}:/qdrant/snapshots/{input_subdir}:ro"]
    return args + [f"{image_repo}:{ver}"]


def recover_body(container_path: str) -> dict:
    return {"location": f"file://{container_path}"}


def snapshot_name(resp: dict) -> str:
    name = ((resp or {}).get("result") or {}).get("name")
    if not name:
        raise HopError(f"snapshot create returned no name: {json.dumps(resp)[:300]}")
    return name


@contextlib.contextmanager
def hop_copy_env():
    """Name the reason for scrolling the hop container (stopped copy, loopback). Only the calls that
    target the hop container run inside this; a --baseline-url capture never does."""
    old = os.environ.get(HV.HOP_COPY_ENV)
    os.environ[HV.HOP_COPY_ENV] = HOP_COPY_REASON
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(HV.HOP_COPY_ENV, None)
        else:
            os.environ[HV.HOP_COPY_ENV] = old


def baseline_summary(b: dict) -> str:
    return (f"{b['exact_count']:,} pts, server {b['server_version']}, {len(b['queries'])} queries, "
            f"sha {b['content']['sha256'][:16]}")


class PeakSampler:
    """Peak transient disk of a dir during an in-place migration (the number that decides whether a
    single-disk box can do it at all: measure, never assume)."""

    def __init__(self, path: Path, every: float = 5.0):
        self.path, self.every, self.peak = path, every, 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, size_kb(self.path))
            self._stop.wait(self.every)

    def start(self):
        self._t.start()

    def stop(self) -> int:
        self._stop.set()
        if self._t.is_alive():
            self._t.join()
        return self.peak


# --------------------------------------------------------------------- runner --
class Chain:
    def __init__(self, a: argparse.Namespace, *, sh=None, http=None, sleep=time.sleep, say=None):
        self.a = a
        self.work = Path(a.work_dir)
        self.state, self.logs = self.work / "state", self.work / "logs"
        self.snaps, self.store = self.work / "snapshots", self.work / "storage"
        self.hops = [h for h in a.chain.split(",") if h]
        self.sh = sh or self._sh
        self.http = http or HV._req
        self.sleep = sleep
        self.say = say or (lambda m: print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {m}", flush=True))
        self.url = f"http://127.0.0.1:{a.port}"
        self.cur: str | None = None

    # -- process / http
    @staticmethod
    def _sh(argv: list[str], check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(argv, capture_output=True, text=True, check=check)

    def run(self, argv: list[str], check: bool = True):
        if self.a.dry_run:
            print("  DRY: " + " ".join(argv))
            return subprocess.CompletedProcess(argv, 0, "", "")
        return self.sh(argv, check)

    def _coll_url(self, tail: str = "") -> str:
        return f"{self.url}/collections/{self.a.collection}{tail}"

    # -- docker
    def image_present(self, ver: str) -> bool:
        return self.sh(["docker", "image", "inspect", f"{self.a.image_repo}:{ver}"], False).returncode == 0

    def pull_missing(self) -> None:
        for ver in dict.fromkeys([self.a.source_version, *self.hops]):
            if self.a.dry_run or not self.image_present(ver):
                self.say(f"pulling {self.a.image_repo}:{ver}")
                try:
                    self.run(["docker", "pull", f"{self.a.image_repo}:{ver}"])
                except subprocess.CalledProcessError as e:
                    raise HopError(f"cannot pull {self.a.image_repo}:{ver}: {(e.stderr or '')[:300]}") from None

    def start(self, ver: str, storage: Path, snaps: Path, input_dir: Path | None = None) -> None:
        name = container_name(self.a.container_prefix, ver)
        self.run(["docker", "rm", "-f", name], check=False)
        if not self.a.dry_run:
            storage.mkdir(parents=True, exist_ok=True)
            snaps.mkdir(parents=True, exist_ok=True)
        self.run(docker_run_args(name=name, ver=ver, port=self.a.port, image_repo=self.a.image_repo,
                                 storage=storage, snaps=snaps, input_dir=input_dir,
                                 input_subdir=self.a.input_subdir, nofile=self.a.nofile))
        self.cur = None if self.a.dry_run else name
        self.say(f"started {name} ({self.a.image_repo}:{ver}) storage={storage}")

    def stop(self) -> None:
        """Graceful: a forced kill makes the next start replay a long WAL and can drop writes."""
        if self.a.dry_run or not self.cur:
            self.cur = None
            return
        t0 = time.monotonic()
        self.sh(["docker", "stop", "-t", "300", self.cur])
        self.say(f"stopped {self.cur} cleanly in {int(time.monotonic() - t0)}s")
        logs = self.sh(["docker", "logs", self.cur], False)
        (self.logs / f"{self.cur}.docker.log").write_text((logs.stdout or "") + (logs.stderr or ""))
        self.sh(["docker", "rm", self.cur], False)
        self.cur = None

    def cleanup(self) -> None:
        if self.cur:
            self.say(f"cleanup: removing {self.cur}")
            self.sh(["docker", "rm", "-f", self.cur], False)
            self.cur = None

    def container_alive(self) -> bool:
        out = self.sh(["docker", "ps", "--format", "{{.Names}}"], False).stdout or ""
        return self.cur in out.split()

    def _tail_logs(self) -> str:
        r = self.sh(["docker", "logs", "--tail", "40", self.cur], False)
        return (r.stdout or "") + (r.stderr or "")

    # -- qdrant
    def wait_ready(self) -> None:
        if self.a.dry_run:
            return
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.a.ready_timeout:
            try:
                self.http(f"{self.url}/collections", timeout=10)
                ver = self.http(f"{self.url}/", timeout=10).get("version")
                self.say(f"server ready in {int(time.monotonic() - t0)}s (version={ver})")
                return
            except HV.QdrantError:
                pass
            if self.cur and not self.container_alive():
                raise HopError(f"{self.cur} exited during startup (this is what a REFUSED storage "
                               f"format looks like):\n{self._tail_logs()}")
            self.sleep(3)
        raise HopError(f"server not ready after {self.a.ready_timeout}s")

    def wait_green(self) -> None:
        if self.a.dry_run:
            return
        t0 = time.monotonic()
        st = "unreadable"
        while time.monotonic() - t0 < self.a.green_timeout:
            try:
                st = self.http(self._coll_url(), timeout=30)["result"]["status"]
            except (HV.QdrantError, KeyError, TypeError):
                st = "unreadable"
            if st == "green":
                self.say(f"collection green in {int(time.monotonic() - t0)}s")
                return
            if st == "red":
                raise HopError(f"collection went RED:\n{self._tail_logs()}")
            self.sleep(10)
        raise HopError(f"collection not green after {self.a.green_timeout}s (last status={st})")

    def restore(self, container_path: str) -> None:
        if self.a.dry_run:
            print(f"  DRY: restore {container_path}")
            return
        t0 = time.monotonic()
        try:
            r = self.http(self._coll_url("/snapshots/recover?wait=true"), "PUT",
                          recover_body(container_path), timeout=86400)
        except HV.QdrantError as e:
            raise HopError(f"snapshot recover FAILED (the RocksDB->Gridstore wall if the chain "
                           f"skipped a version): {str(e)[:600]}") from None
        if r.get("result") is not True:
            raise HopError(f"snapshot recover FAILED: {json.dumps(r)[:600]}")
        self.say(f"restore ok in {int(time.monotonic() - t0)}s")

    def create_snapshot(self) -> str:
        if self.a.dry_run:
            return "DRY.snapshot"
        try:
            return snapshot_name(self.http(self._coll_url("/snapshots?wait=true"), "POST", None, timeout=86400))
        except HV.QdrantError as e:
            raise HopError(f"snapshot create FAILED: {str(e)[:600]}") from None

    # -- verifier (in-process; the same functions the capture/compare modes call)
    def capture(self, url: str, out: Path, hop_copy: bool) -> None:
        ctx = hop_copy_env() if hop_copy else contextlib.nullcontext()
        with ctx:
            fp = HV.fingerprint(url, self.a.collection, self.a.seed, self.a.topk, self.a.digest_points,
                                self.a.timeout, per_leg=self.a.queries_per_leg)
        out.write_text(json.dumps(fp, indent=1))

    def verify_hop(self, label: str) -> None:
        if self.a.dry_run:
            print(f"  DRY: verify {label}")
            return
        base = json.loads((self.work / "baseline.json").read_text())
        with hop_copy_env():
            cur = HV.fingerprint(self.url, self.a.collection, self.a.seed, self.a.topk,
                                 self.a.digest_points, self.a.timeout, queries=base["queries"],
                                 sample_ids=base["sample_ids"])
            rc = HV.compare(base, cur, label, self.a.score_tolerance, self.a.ann_drift)
        if rc != 0:
            raise HopError(f"VERIFY FAILED at {label} - chain stopped, nothing downstream is trustworthy")

    def ensure_baseline(self) -> None:
        dest = self.work / "baseline.json"
        a = self.a
        if dest.is_file():
            self.say(f"baseline: reusing {dest} (resume)")
        elif a.baseline:
            if not Path(a.baseline).is_file():
                raise HopError(f"--baseline {a.baseline} not found")
            if not a.dry_run:
                shutil.copyfile(a.baseline, dest)
            self.say(f"baseline: copied from {a.baseline}")
        elif a.baseline_url:
            self.say(f"baseline: capturing read-only from {a.baseline_url}")
            if not a.dry_run:
                self.capture(a.baseline_url, dest, hop_copy=False)
        elif a.mode != "snapshot":
            raise HopError("inplace mode needs --baseline or --baseline-url")
        else:
            self.say(f"baseline: hop0 - restoring source snapshot into {a.source_version} to fingerprint it")
            src = Path(a.source_snapshot)
            disk_gate(size_kb(src), free_kb(self.store), a.disk_factor)
            hop0 = f"hop0-{a.source_version}"
            self.start(a.source_version, self.store / hop0, self.snaps / hop0, src.parent)
            self.wait_ready()
            self.restore(f"/qdrant/snapshots/{a.input_subdir}/{src.name}")
            self.wait_green()
            if not a.dry_run:
                self.capture(self.url, dest, hop_copy=True)
            self.stop()
            if not a.keep_artifacts and not a.dry_run:
                shutil.rmtree(self.store / hop0, ignore_errors=True)
        if not a.dry_run:
            self.say("baseline: " + baseline_summary(json.loads(dest.read_text())))

    # -- chains
    def chain_snapshot(self) -> Path:
        a = self.a
        input_file = Path(a.source_snapshot)
        for i, ver in enumerate(self.hops):
            n = i + 1
            marker = self.state / f"hop{n}-{ver}.done"
            sdir, ndir = self.store / f"hop{n}-{ver}", self.snaps / f"hop{n}-{ver}"
            if marker.is_file():
                input_file = Path(json.loads(marker.read_text())["out_snapshot"])
                if not input_file.is_file():
                    raise HopError(f"hop{n} marked done but its snapshot {input_file} is gone - "
                                   f"delete {marker} to redo")
                self.say(f"hop{n} {ver}: SKIP (already done) -> {input_file}")
                continue
            self.say(f"=== hop{n}: {ver}  <- {input_file.name} ===")
            disk_gate(size_kb(input_file), free_kb(self.store), a.disk_factor)
            t0 = time.monotonic()
            if not a.dry_run:
                shutil.rmtree(sdir, ignore_errors=True)   # a half-finished hop must not be reused
            self.start(ver, sdir, ndir, input_file.parent)
            self.wait_ready()
            t_ready = time.monotonic() - t0
            self.restore(f"/qdrant/snapshots/{a.input_subdir}/{input_file.name}")
            self.wait_green()
            t_restore = time.monotonic() - t0 - t_ready
            self.verify_hop(f"hop{n}:{ver}")
            t_verify = time.monotonic() - t0 - t_ready - t_restore
            snap = self.create_snapshot()
            out = ndir / a.collection / snap
            if not a.dry_run and not out.is_file():
                raise HopError(f"snapshot {out} not on disk after create")
            t_snap = time.monotonic() - t0 - t_ready - t_restore - t_verify
            self.stop()
            if not a.dry_run:
                total = time.monotonic() - t0
                marker.write_text(json.dumps({
                    "version": ver, "out_snapshot": str(out),
                    "seconds": {"ready": int(t_ready), "restore_green": int(t_restore),
                                "verify": int(t_verify), "snapshot": int(t_snap), "total": int(total)},
                    "storage_kb": size_kb(sdir), "snapshot_kb": size_kb(out)}, indent=1))
                self.say(f"hop{n} {ver} DONE in {int(total)}s -> {out}")
                input_file = out
                if not a.keep_artifacts and i > 0:   # the previous hop's storage is dead weight
                    shutil.rmtree(self.store / f"hop{i}-{self.hops[i - 1]}", ignore_errors=True)
        self.say(f"FINAL snapshot (restorable by {self.hops[-1]}): {input_file}")
        if not a.dry_run:
            (self.work / "FINAL_SNAPSHOT").write_text(f"{input_file}\n")
        return input_file

    def chain_inplace(self) -> None:
        a = self.a
        src = Path(a.source_storage)
        for i, ver in enumerate(self.hops):
            n = i + 1
            marker = self.state / f"inplace-hop{n}-{ver}.done"
            if marker.is_file():
                self.say(f"inplace hop{n} {ver}: SKIP (already done)")
                continue
            self.say(f"=== inplace hop{n}: {ver} on {src} ===")
            t0 = time.monotonic()
            before = size_kb(src)
            sampler = PeakSampler(src, a.sample_seconds)
            if not a.dry_run:
                sampler.start()
            try:
                self.start(ver, src, self.snaps / f"inplace-{ver}")
                self.wait_ready()
                t_ready = time.monotonic() - t0
                self.wait_green()
                self.verify_hop(f"inplace-hop{n}:{ver}")
                self.stop()
            finally:
                peak = sampler.stop() if not a.dry_run else 0
            if not a.dry_run:
                total = time.monotonic() - t0
                marker.write_text(json.dumps({
                    "version": ver, "seconds": int(total), "ready_seconds": int(t_ready),
                    "storage": str(src), "kb_before": before, "kb_after": size_kb(src),
                    "kb_peak": peak}, indent=1))
                self.say(f"inplace hop{n} {ver} DONE in {int(total)}s (ready {int(t_ready)}s, "
                         f"peak {peak // 1024} MiB)")
        self.say(f"FINAL: {src} now readable by {self.hops[-1]}")

    def go(self) -> int:
        a = self.a
        if not a.dry_run:
            for d in (self.state, self.logs, self.snaps, self.store):
                d.mkdir(parents=True, exist_ok=True)
        self.say(f"=== qdrant hop chain mode={a.mode} collection={a.collection} chain={a.chain} ===")
        self.say(f"work={self.work} port={a.port}")
        try:
            self.pull_missing()
            self.ensure_baseline()
            if a.mode == "snapshot":
                self.chain_snapshot()
            else:
                self.chain_inplace()
        except HopError as e:
            print(f"FATAL: {e}", file=sys.stderr)
            return 1
        except HV.QdrantError as e:
            print(f"TRANSPORT/SERVER ERROR: {e}", file=sys.stderr)
            return 2
        finally:
            self.cleanup()
        self.say(f"=== chain complete: {a.collection} {a.source_version} -> {self.hops[-1]} ===")
        return 0


# ------------------------------------------------------------------------ cli --
def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    e = os.environ.get
    ap = argparse.ArgumentParser(prog=prog or "qdrant_hop_chain", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["snapshot", "inplace"], default=e("AW_HOP_MODE", "snapshot"))
    ap.add_argument("--collection", default=e("AW_HOP_COLLECTION"))
    ap.add_argument("--work-dir", default=e("AW_HOP_WORK"))
    ap.add_argument("--chain", default=e("AW_HOP_CHAIN", DEFAULT_CHAIN), help="comma list of image tags, in order")
    ap.add_argument("--source-version", default=e("AW_HOP_SOURCE_VERSION", "v1.12.4"))
    ap.add_argument("--source-snapshot", default=e("AW_HOP_SOURCE_SNAPSHOT"))
    ap.add_argument("--source-storage", default=e("AW_HOP_SOURCE_STORAGE"))
    ap.add_argument("--port", type=int, default=int(e("AW_HOP_PORT", "6421")),
                    help="hop container port; never the serving port")
    ap.add_argument("--image-repo", default=e("AW_HOP_IMAGE_REPO", "qdrant/qdrant"))
    ap.add_argument("--container-prefix", default=e("AW_HOP_CONTAINER_PREFIX", "aw-qdrant-hop"))
    ap.add_argument("--baseline", default=e("AW_HOP_BASELINE"), help="reuse an existing fingerprint file")
    ap.add_argument("--baseline-url", default=e("AW_HOP_BASELINE_URL"), help="capture read-only off a server")
    ap.add_argument("--digest-points", type=int, default=int(e("AW_HOP_DIGEST_POINTS", "500")))
    ap.add_argument("--score-tolerance", type=float, default=float(e("AW_HOP_SCORE_TOLERANCE", "0.001")),
                    help="max per-hit score drift; rank order is always compared strictly")
    ap.add_argument("--ann-drift", choices=["fail", "warn"], default=e("AW_HOP_ANN_DRIFT", "fail"),
                    help="'warn' only with a real eval gate downstream")
    ap.add_argument("--queries-per-leg", type=int, default=int(e("AW_HOP_QUERIES_PER_LEG", "2")))
    ap.add_argument("--seed", type=int, default=339)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--input-subdir", default=e("AW_HOP_INPUT_SUBDIR", "_hopin"))
    ap.add_argument("--nofile", type=int, default=int(e("AW_HOP_NOFILE", "1048576")))
    ap.add_argument("--ready-timeout", type=int, default=int(e("AW_HOP_READY_TIMEOUT", "600")))
    ap.add_argument("--green-timeout", type=int, default=int(e("AW_HOP_GREEN_TIMEOUT", "86400")))
    ap.add_argument("--disk-factor", type=float, default=float(e("AW_HOP_DISK_FACTOR", "2.5")),
                    help="free space required per hop, x input size")
    ap.add_argument("--sample-seconds", type=float, default=5.0, help="peak-disk sampling period (inplace)")
    ap.add_argument("--keep-artifacts", action="store_true", default=e("AW_HOP_KEEP_ARTIFACTS") == "1")
    ap.add_argument("--dry-run", action="store_true", help="print the docker commands; touch nothing")
    ap.add_argument("--i-understand-inplace-is-irreversible", dest="inplace_ack", action="store_true")
    return ap


def validate(a: argparse.Namespace, ap: argparse.ArgumentParser) -> None:
    if not a.collection:
        ap.error("--collection required")
    if not a.work_dir:
        ap.error("--work-dir required")
    if not [h for h in a.chain.split(",") if h]:
        ap.error("empty --chain")
    if a.port in SERVING_PORTS:
        ap.error(f"--port {a.port} is the serving port; pick a hop port")
    if a.mode == "snapshot":
        if not a.source_snapshot:
            ap.error("snapshot mode needs --source-snapshot")
        if not Path(a.source_snapshot).is_file():
            ap.error(f"no such snapshot: {a.source_snapshot}")
    else:
        if not a.source_storage:
            ap.error("inplace mode needs --source-storage")
        if not Path(a.source_storage).is_dir():
            ap.error(f"no such storage dir: {a.source_storage}")
        if not a.inplace_ack:
            ap.error(f"inplace mode rewrites {a.source_storage} with no downgrade path. Pass "
                     "--i-understand-inplace-is-irreversible (and take a snapshot first).")
    if not a.dry_run and shutil.which("docker") is None:
        ap.error("docker not found")


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = build_parser(prog)
    a = ap.parse_args(argv)
    validate(a, ap)
    return Chain(a).go()
