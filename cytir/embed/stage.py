"""The one door to embed a sealed chunk stage: GPU lock + `cytir.embed.shards`.

MOVED HERE from the lock discipline in task 633's `run_embed_lease.sh`
(task 657). That script's own comment named two lock disciplines honoured
together: (1) the house convention every embed run has used since
task 472's `run_embed.sh` -- `flock` on `$GPU_LOCK`
(`/tmp/aw_gpu0.lock`), yielding to `$PRIORITY_SENTINEL`; (2) the
`<work-root>/GPU-LOCK` directory courtesy-marker another
concurrently running task may poll for. Both are reproduced here, in
Python, as the one door package code and the ERIC materialization chain
call -- task 633's `run_embed_lease.sh` is left in place (shell
drivers may still call it directly; it is not on any package import path),
but new automated callers use `embed_stage()` so there is one implementation
of the lock discipline, not a shell copy and a Python copy that can drift
(engineering-doctrine #1).

`check_gpu_idle()` is `nvidia-smi --query-compute-apps` -- called BEFORE the
lock is acquired, so a caller gets a clear refusal (not a silent long wait)
when another lane is already holding the GPU outside this lock's own
discipline (the repo convention named in task 657's brief: "check
nvidia-smi first").
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

from cytir.core.paths import operator_settings

DEFAULT_GPU_LOCK = "/tmp/aw_gpu0.lock"
DEFAULT_PRIORITY_SENTINEL = "/tmp/aw_gpu0.priority.production"


def default_legacy_lock_dir() -> str:
    """The legacy GPU courtesy lock: operator setting `gpu_lock_dir` (refused by name when unconfigured)."""
    return operator_settings().gpu_lock_dir


def __getattr__(name):
    # `DEFAULT_LEGACY_LOCK_DIR` stays readable but resolves on first ACCESS, never at import.
    if name == "DEFAULT_LEGACY_LOCK_DIR":
        return default_legacy_lock_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

DEFAULT_POLL_SECONDS = 30.0


class GPUBusyError(RuntimeError):
    """Raised when the GPU already has a compute process on it."""


def gpu_compute_pids() -> list[int]:
    """PIDs of processes currently holding GPU compute memory (any GPU).

    Mirrors `gpu_writer_pids` in task 417's `run_admitted_stage_embed.py`
    (the same query) -- ONE query shape, not two copies of the nvidia-smi
    invocation (engineering-doctrine #1). Returns [] if nvidia-smi itself is
    absent (a CPU-only dev box), which the caller treats as "idle", not a
    silent skip of a real check -- there is no GPU to be busy on.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid",
             "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=True)
    except FileNotFoundError:
        return []
    return [int(line) for line in result.stdout.splitlines() if line.strip()]


def check_gpu_idle() -> None:
    """Fail loud, before any lock or lease, if the GPU already has a writer.

    This is a courtesy pre-check, not the safety mechanism -- the flock in
    `embed_stage()` is. It exists so a caller sees "GPU busy: pids [...]"
    immediately instead of waiting silently inside the lock queue (fail
    loud or don't fail, engineering-doctrine #3).
    """
    pids = gpu_compute_pids()
    if pids:
        raise GPUBusyError(
            f"GPU already has compute process(es) {pids} -- check "
            "nvidia-smi before claiming the lock (repo convention).")


@contextlib.contextmanager
def gpu_lease(*, gpu_lock: str = DEFAULT_GPU_LOCK,
             priority_sentinel: str = DEFAULT_PRIORITY_SENTINEL,
             legacy_lock_dir: str | None = None,
             poll_seconds: float = DEFAULT_POLL_SECONDS,
             log=lambda msg: print(msg, flush=True)):
    """Yield-once context manager holding the GPU flock (task 633's two-lock
    discipline). Yields to `priority_sentinel` before even trying the lock."""
    import fcntl

    legacy_lock_dir = legacy_lock_dir or default_legacy_lock_dir()
    while os.path.exists(priority_sentinel):
        log(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"state=YIELDING sentinel={priority_sentinel}")
        time.sleep(poll_seconds)

    Path(gpu_lock).parent.mkdir(parents=True, exist_ok=True)
    lock_fh = open(gpu_lock, "a+")
    fcntl.flock(lock_fh, fcntl.LOCK_EX)
    legacy_created = False
    try:
        try:
            os.mkdir(legacy_lock_dir)
            legacy_created = True
        except FileExistsError:
            pass
        except OSError:
            pass  # courtesy marker only -- never block the real lock on it
        yield
    finally:
        if legacy_created:
            with contextlib.suppress(OSError):
                os.rmdir(legacy_lock_dir)
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()


def embed_stage(chunks_dir: str, vectors_dir: str, *,
                model: str | None = None, revision: str | None = None,
                max_seq: int = 2560, batch: int = 16, chunk: int = 4096,
                device: str = "cuda:0", compute_dtype: str = "float16",
                python_bin: str | None = None,
                gpu_lock: str = DEFAULT_GPU_LOCK,
                priority_sentinel: str = DEFAULT_PRIORITY_SENTINEL,
                legacy_lock_dir: str | None = None,
                poll_seconds: float = DEFAULT_POLL_SECONDS,
                skip_idle_check: bool = False,
                log=lambda msg: print(msg, flush=True)) -> int:
    """Embed one staged chunk directory under the GPU lease.

    `model`/`revision` default to `cytir.core.model_pin` (the committed pin) --
    pass an override ONLY for a deliberate re-embed campaign (mirrors
    `AW_MODEL_REVISION`, honoured loudly, never silently, by
    `cytir.core.model_pin.embed_model_revision`).

    Runs `cytir.embed.shards` in a FRESH subprocess (not an in-process call)
    deliberately: a long-running chain driver calling this once per tranche
    must not accumulate GPU/CUDA context or model weights across calls, and
    a subprocess crash (OOM, driver reset) must not take the calling chain
    process down with it.
    """
    from cytir.consent.embed_gate import assert_stage_consent
    from cytir.core.model_pin import EMBED_MODEL_ID, embed_model_revision

    # Refuse a recorded consent decline BEFORE the GPU check or the lock queue;
    # the shard worker enforces the same gate again for its own shards.
    assert_stage_consent(chunks_dir, log=log)

    if not skip_idle_check:
        check_gpu_idle()

    model = model or EMBED_MODEL_ID
    revision = revision or embed_model_revision()
    python_bin = python_bin or sys.executable

    argv = [python_bin, "-m", "cytir.embed.shards",
           "--in", chunks_dir, "--out", vectors_dir,
           "--model", model, "--revision", revision,
           "--max-seq", str(max_seq), "--batch", str(batch),
           "--chunk", str(chunk), "--device", device,
           "--compute-dtype", compute_dtype, "--rank", "0", "--world", "1"]
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    with gpu_lease(gpu_lock=gpu_lock, priority_sentinel=priority_sentinel,
                  legacy_lock_dir=legacy_lock_dir, poll_seconds=poll_seconds,
                  log=log):
        log(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"state=EMBEDDING chunks_dir={chunks_dir}")
        result = subprocess.run(argv, env=env)
    log(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
        f"state=COMPLETE rc={result.returncode}")
    return result.returncode


def build_parser():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("chunks_dir")
    ap.add_argument("vectors_dir")
    ap.add_argument("--model", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--gpu-lock", default=DEFAULT_GPU_LOCK)
    ap.add_argument("--priority-sentinel", default=DEFAULT_PRIORITY_SENTINEL)
    ap.add_argument("--legacy-lock-dir", default=None,
                    help="legacy GPU courtesy lock (default: the operator setting gpu_lock_dir)")
    ap.add_argument("--skip-idle-check", action="store_true",
                    help="skip the pre-lock nvidia-smi check (testing only)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return embed_stage(args.chunks_dir, args.vectors_dir, model=args.model,
                       revision=args.revision, gpu_lock=args.gpu_lock,
                       priority_sentinel=args.priority_sentinel,
                       legacy_lock_dir=args.legacy_lock_dir,
                       skip_idle_check=args.skip_idle_check)


if __name__ == "__main__":
    raise SystemExit(main())
