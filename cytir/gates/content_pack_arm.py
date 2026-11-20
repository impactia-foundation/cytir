#!/usr/bin/env python3
"""Capture and score one immutable current-twin/content-pack corpus state.

This is orchestration only: Task370 remains the sole scorer and comparator.
The wrapper makes its generic corpus contract usable for serialized coverage
packs without inventing another eval runner or another ADR-043 calculation.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys

# Started by path (`python aw/...py`), sys.path[0] is this file's own directory, not the repo
# root: put the root first so `import cytir` resolves (the start-guard test).
_REPO_ROOT = str(_pathlib.Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cytir.core.serve_layout import package_name  # noqa: E402 -- the RELEASE tree's own package dir
from cytir.core import internal_http as IH  # noqa: E402 -- the one door to the operator's own services (consent step 5, task 1673)
import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_content_json(out_dir: Path, stem: str, payload: dict) -> Path:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path = out_dir / f"{stem}.{hashlib.sha256(data).hexdigest()}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def collection_state(url: str, collection: str) -> dict:
    endpoint = f"{url.rstrip('/')}/collections/{collection}"
    with IH.urlopen(endpoint, timeout=30) as response:
        result = json.load(response)["result"]
    queue = int((result.get("update_queue") or {}).get("length", 0))
    state = {
        "status": result.get("status"),
        "optimizer_status": result.get("optimizer_status"),
        "points_count": int(result["points_count"]),
        "update_queue_length": queue,
    }
    if state["status"] != "green" or state["optimizer_status"] != "ok" or queue:
        raise SystemExit(f"target is not quiet-green: {state}")
    return state


def refuse_error_arm(arm: Path) -> None:
    """An arm made of errors is not a measurement.

    Task 600: every one of this arm's 1,418 queries came back
    `__ERROR__:UnexpectedResponse` -- the gate was querying a 256-dim
    collection at the shipped 128-dim default -- and the arm was written and
    reported as success. Only `compare_corpus_arms.py`, one step later, said
    so, and only about the first set it happened to look at. A zero here is
    never a clean miss (verification-doctrine: an error is not an empty), so
    the arm refuses at the point the errors exist, names how many there are,
    and exits non-zero.
    """
    document = json.loads(arm.read_text())
    failures = []
    for name, result in sorted((document.get("sets") or {}).items()):
        rows = (result or {}).get("per_q") or {}
        bad = sorted(query for query, row in rows.items()
                     if not isinstance(row, dict) or
                     not isinstance(row.get("urls"), list) or
                     not row["urls"] or
                     str(row["urls"][0]).startswith("__ERROR__"))
        if bad:
            failures.append(f"{name}: {len(bad)} of {len(rows)} "
                            f"(e.g. {bad[0]} -> "
                            f"{str((rows[bad[0]] or {}).get('urls'))[:60]})")
    for probe_id, row in sorted((document.get("probes") or {}).items()):
        if (not isinstance(row, dict) or
                not isinstance(row.get("urls"), list) or not row["urls"] or
                str(row["urls"][0]).startswith("__ERROR__")):
            failures.append(f"probe {probe_id}: scorer error")
    if failures:
        arm.unlink()
        raise SystemExit(
            "refusing an arm whose queries errored -- this is an outage, not "
            "a result: " + "; ".join(failures))


def refuse_unbound_arm(arm: Path, release_dir: Path) -> None:
    """An arm that did not run production's ranker is not a measurement of it.

    Task 603 found the gate scoring the dense leg with rescore OFF while
    production serves it ON x3; task 604 found `AW_FUSION_WEIGHTS` unset as
    well, so the arm loaded `fusion_weights.json` while production served
    candidate533 -- 82 golds in a different position and h@1 213 -> 219 on the
    live twin. Both were invisible because the arm recorded neither.

    The arm now records what it ran (`run_corpus_arm.serving_binding`), and
    this refuses it at the point the mismatch exists, exactly as
    `refuse_error_arm` does for a wall of errors: non-zero exit, arm deleted,
    the difference named. The standard is `cytir/core/serving_config.py`, which
    no run can edit on its way past.
    """
    # The release's copy, not the orchestrator box's: loaded from the release dir by path.
    spec = importlib.util.spec_from_file_location(
        "release_serving_config",
        release_dir / "cytir/core/serving_config.py")
    serving_config = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = serving_config   # dataclasses resolve their module here
    spec.loader.exec_module(serving_config)

    binding = (json.loads(arm.read_text()).get("task370") or {}).get(
        "serving_binding")
    problems = serving_config.gate_serving_binding_problems(binding)
    if problems:
        arm.unlink()
        raise SystemExit(
            "refusing an arm that did not measure what production serves: "
            + "; ".join(problems))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--representation", required=True,
                        choices=("current-b1", "content-pack"))
    parser.add_argument("--identity", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--supersede-sha256", required=True)
    parser.add_argument("--cache", required=True, help="SHA256=PATH")
    parser.add_argument("--qvec-query-source-manifest", required=True,
                        help="SHA256=PATH")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--capture-sparse", action="store_true",
        help=("pass --capture-sparse to the Task370 runner: also record each "
              "query's bm25 vector and its traced candidates' STORED bm25 "
              "values, which are the inputs ADR-109's sealed BM25 gate arm needs "
              "(ADR-043 A8 rule 4). Costs one extra qdrant retrieve per query "
              "and a larger artifact, so it is opt-in here too."))
    args = parser.parse_args()
    for label, value in (("source", args.source_sha256),
                         ("supersede", args.supersede_sha256)):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            parser.error(f"{label} SHA256 is invalid")
    runner = args.release_dir / package_name(args.release_dir) / "gates" / "run_corpus_arm.py"
    if not runner.is_file():
        parser.error(f"Task370 runner absent from release: {runner}")
    state = collection_state(args.qdrant_url, args.collection)
    target = atomic_content_json(args.out_dir, "target", {
        "schema": "aw.content-pack-target.v1",
        "identity": args.identity,
        "qdrant_url": args.qdrant_url,
        "collection": args.collection,
        "status": state["status"],
        "points_count": state["points_count"],
        "optimizer_status": state["optimizer_status"],
        "update_queue_length": state["update_queue_length"],
        "write_policy": "read_only-gate",
    })
    corpus = atomic_content_json(args.out_dir, "corpus", {
        "schema": "aw.content-pack-corpus-wrapper.v1",
        "identity": args.identity,
        "qdrant_url": args.qdrant_url,
        "collection": args.collection,
        "status": state["status"],
        "points_count": state["points_count"],
        "representation": args.representation,
        "source_sha256": args.source_sha256,
        "supersede_sha256": args.supersede_sha256,
    })
    before = set(args.out_dir.glob(f"task370-arm.{args.arm_id}.*.json"))
    command = [
        args.python, str(runner), "--cache", args.cache,
        "--qvec-query-source-manifest", args.qvec_query_source_manifest,
        "--target-manifest", f"{digest(target)}={target}",
        "--corpus-manifest", f"{digest(corpus)}={corpus}",
        "--out-dir", str(args.out_dir), "--arm-id", args.arm_id,
    ]
    if args.capture_sparse:
        command.append("--capture-sparse")
    completed = subprocess.run(command, cwd=args.release_dir, text=True,
                               capture_output=True)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    created = set(args.out_dir.glob(f"task370-arm.{args.arm_id}.*.json")) - before
    if len(created) != 1:
        raise SystemExit(f"expected one new Task370 arm, found {len(created)}")
    arm = created.pop()
    refuse_error_arm(arm)
    refuse_unbound_arm(arm, args.release_dir)
    print(f"TARGET_PATH={target}")
    print(f"CORPUS_PATH={corpus}")
    print(f"ARM_PATH={arm}")


if __name__ == "__main__":
    main()
