#!/usr/bin/env python3
"""Reusable canonical-dump coverage-pack transaction.

The shared twin-first protocol owns the real production lock and ordering.
This adapter owns only the point-dump format, target-local supersede apply,
exact readback proof, and the post-twin Task370 corpus gate.  Twin and
production collection names are independently configurable because aliases on
the two machines are not required to share a name.

Moved from the task 351 scratch script `apply_content_pack_shared.py` (task 646 S4:
"nothing in production runs from scratch dirs"). A shim at the old path re-exports
this module unchanged -- the ~20 task scratch-script callers that still
`importlib`-load it by that path (and its own sibling tests) keep working
against the shim untouched. `cyt news routine-apply` (cytir/news/routine_apply.py) and
the task 598 `news_tier2_apply.sh` `apply-armed` step now point at
this path directly.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pathlib as _pathlib

# Started by path (`python aw/...py`), sys.path[0] is this file's own directory, not the repo
# root: put the root first so `import cytir` resolves (pinned by the script start-guard test).
_REPO_ROOT = str(_pathlib.Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

ROOT = Path(__file__).resolve().parents[2]
from cytir.core import internal_http as IH  # after the bootstrap above (task 1738)
from cytir.core.digest import canonical_line_bytes as canonical
from cytir.core.serve_layout import SERVER_REL
from cytir.core.qdrant import qdrant_connect
from cytir.index.twin_first import (ApplyAbort, ProductionWriterLock, append_ledger,
                                 resume_after_verified_twin_gate, twin_first)
from cytir.index.content_pack_preservation import (
    ContractError as PreservationContractError,
    verify_publish_environment as verify_content_pack_publish_environment)
from cytir.index.supersede import (CardOverwriteRefused, Superseder, client_url, guard_card_overwrites,
                                load_card_waivers)

#: The supersede pass scrolls its target by url group. On PROD that is a scroll of the serving box
#: (incident 2026-10-07), refused by `cytir.index.prod_guard` unless the operator names a reason here.
PROD_SCROLL_REASON = os.environ.get("AW_PROD_SUPERSEDE_SCROLL_REASON") or None
from cytir.index.supersede import payload_group_key
from cytir.index.upsert import to_point
# Package-relative, not a bare sys.path import: these two moved INTO cytir/index/
# alongside this file (task 646 S4, "351 applier + lineage + noop"), so they
# are real siblings now rather than task 351 scratch-dir neighbours reachable
# only via a script's own directory being sys.path[0].
from cytir.index.lineage import LineageError, load_failed_twin_lineage
from cytir.index.noop import Comparison, compare_batches
from cytir.index.dense_precision import (DensePrecisionError,
                                      dense_datatype_of, dense_size_of)
from cytir.gates import convergence as gate_convergence
from cytir.corpus import persisted as PI
from cytir.index import readiness as qdrant_readiness
from cytir.core import serving_config
from cytir.gates import news_daily as news_daily_gate
from cytir.index import layout as collection_layout

from qdrant_client import QdrantClient, models

ARM = "APPLY-CONTENT-PACK"
RESUME_ARM = "RESUME-CONTENT-PACK-AFTER-VERIFIED-TWIN-GATE"


def resolve_target_collections(collection: str | None,
                               twin_collection: str | None,
                               prod_collection: str | None) -> dict[str, str]:
    """Resolve the legacy common name or the explicit per-target names."""
    explicit = bool(twin_collection or prod_collection)
    if collection and explicit:
        raise ApplyAbort(
            "--collection cannot be combined with per-target collections")
    if collection:
        return {"twin": collection, "prod": collection}
    if not twin_collection or not prod_collection:
        raise ApplyAbort(
            "supply --collection or both --twin-collection and "
            "--prod-collection")
    return {"twin": twin_collection, "prod": prod_collection}


def target_binding(role: str, url: str, collection: str) -> dict[str, str]:
    return {"role": role, "url": url.rstrip("/"),
            "collection": collection}


def plan_without_target(plan: dict) -> dict:
    return {key: value for key, value in plan.items() if key != "target"}


def require_equivalent_mirror_plans(plans: dict[str, dict]) -> None:
    """Require equal mutations while allowing distinct target identities."""
    if set(plans) != {"twin", "prod"}:
        raise ApplyAbort("mirror plan must contain twin and prod")
    for role in ("twin", "prod"):
        target = plans[role].get("target")
        if not isinstance(target, dict) or target.get("role") != role:
            raise ApplyAbort(f"{role} plan lacks its target binding")
        if not target.get("url") or not target.get("collection"):
            raise ApplyAbort(f"{role} plan has an incomplete target binding")
    if plan_without_target(plans["twin"]) != plan_without_target(plans["prod"]):
        raise ApplyAbort(f"starting mirror plans differ: {plans}")


def require_resume_preflight_invariants(
        *, plan: dict, rows: int, membership: dict[str, int],
        prod_comparison: dict) -> None:
    """Refuse to resume unless the twin holds the pack and production is
    still exactly at its pre-apply baseline.

    The original form of this check was ``membership == {"twin": rows,
    "prod": 0}``, which silently assumed an INSERT-ONLY pack. A pack that
    updates points production already holds -- the ordinary daily-news case,
    where ``existing_ids`` is non-zero -- could never satisfy it, so a
    perfectly safe state was unresumable (task 452).

    The invariant is therefore stated against the bound plan instead, and it
    is strictly stronger than the old one rather than weaker:

    * the twin holds **every** pack row (unchanged);
    * production holds **exactly** the plan's ``existing_ids`` -- for an
      insert-only pack that is 0, i.e. the old check is the special case;
    * production's full artifact comparison still equals the plan's, digests
      included. Membership alone cannot see an in-place update that had
      begun rewriting points production already had; the content digests can,
      and they cover every one of the pack's rows rather than a sample.
    """
    if membership.get("twin") != rows:
        raise ApplyAbort(
            f"resume twin membership is not exact: {membership.get('twin')} "
            f"of {rows} pack rows present")
    existing = plan["existing_ids"]
    if membership.get("prod") != existing:
        raise ApplyAbort(
            f"resume production membership is not its pre-apply baseline: "
            f"{membership.get('prod')} pack ids present, plan says {existing} "
            "already existed")
    expected = plan.get("comparison")
    if not isinstance(expected, dict) or not expected:
        raise ApplyAbort("predecessor plan carries no comparison evidence")
    if prod_comparison != expected:
        differing = sorted(
            key for key in set(expected) | set(prod_comparison)
            if expected.get(key) != prod_comparison.get(key))
        raise ApplyAbort(
            "production is no longer at its pre-apply baseline; the prod "
            f"write may have started. Differing: {differing}")


def proof_without_target(proof: dict) -> dict:
    return {key: value for key, value in proof.items() if key != "target"}


def load_persisted_overlay(path: Path | None) -> dict | None:
    """``--persisted-overlay``'s file, turned into what `assert_pages_persisted`
    wants -- or ``None`` if the caller passed nothing (unchanged behavior).

    Not a bypass: `PI.overlay_from_paths` still verifies every path exists on
    disk and matches a store the registry actually declares (task 598).
    """
    if path is None:
        return None
    try:
        url_to_path = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ApplyAbort(f"--persisted-overlay {path}: {exc}") from None
    if not isinstance(url_to_path, dict):
        raise ApplyAbort("--persisted-overlay must be a JSON object of url: path")
    try:
        return PI.overlay_from_paths(url_to_path)
    except PI.PersistedInputsError as exc:
        raise ApplyAbort(f"--persisted-overlay rejected: {exc}") from None


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def runtime_release_identity(release_dir: Path) -> dict:
    """Bind preservation to the immutable code and blocked manifest in use."""
    git_path = release_dir / "GIT_SHA"
    try:
        git_sha = git_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise PreservationContractError(
            f"running release GIT_SHA is unreadable: {exc}") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", git_sha):
        raise PreservationContractError(
            "running release GIT_SHA must be lowercase 40-hex")
    global_path = release_dir / "data" / "production-rebuild-lock.v1.json"
    try:
        global_manifest = json.loads(global_path.read_text(encoding="utf-8"))
        size = global_path.stat().st_size
    except (OSError, json.JSONDecodeError) as exc:
        raise PreservationContractError(
            f"running global rebuild manifest is unreadable: {exc}") from exc
    if (not isinstance(global_manifest, dict) or
            global_manifest.get("schema") != "aw.preservation-manifest.v1" or
            global_manifest.get("status") != "blocked"):
        raise PreservationContractError(
            "running global rebuild manifest must remain explicitly blocked")
    return {
        "git_sha": git_sha,
        "global_rebuild_manifest": {
            "uri": "runtime://data/production-rebuild-lock.v1.json",
            "bytes": size,
            "sha256": digest(global_path),
        },
    }


def records(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ApplyAbort(f"artifact line {line_number}: {exc}") from exc
            if set(("id", "vector", "payload")) - set(row):
                raise ApplyAbort(f"artifact line {line_number}: incomplete point")
            yield row


def vector_json(vector) -> dict:
    result = {}
    for name, value in (vector or {}).items():
        if hasattr(value, "indices"):
            result[name] = {"indices": list(value.indices),
                            "values": list(value.values)}
        elif isinstance(value, dict) and {"indices", "values"} <= set(value):
            result[name] = {"indices": list(value["indices"]),
                            "values": list(value["values"])}
        else:
            result[name] = list(value)
    return result


def point_fingerprint(point) -> str:
    value = {"id": str(point.id), "vector": vector_json(point.vector),
             "payload": point.payload or {}}
    return hashlib.sha256(canonical(value)).hexdigest()


def load_tombstone_pointers(path: Path, expected_sha256: str,
                            artifact_sha256: str,
                            artifact_groups: dict[str, set[str]]) -> dict:
    if not path.is_file() or digest(path) != expected_sha256:
        raise ApplyAbort("tombstone pointer path/hash mismatch")
    value = json.loads(path.read_text())
    if (value.get("schema") != "aw.explicit-tombstone-pointers.v1" or
            value.get("artifact_sha256") != artifact_sha256):
        raise ApplyAbort("tombstone pointers do not bind this artifact")
    source = value.get("source_group")
    aged = value.get("aged_out")
    if not isinstance(source, list) or not isinstance(aged, list):
        raise ApplyAbort("tombstone pointer classes must be lists")
    all_ids = []
    for pointer in source:
        group = pointer.get("group")
        if (pointer.get("reason") != "source_group_reemit" or
                sorted(pointer.get("replacement_ids") or []) !=
                sorted(artifact_groups.get(group, set())) or
                not pointer.get("predecessor_fingerprint")):
            raise ApplyAbort("invalid source-group tombstone pointer")
        all_ids.append(str(pointer.get("predecessor_id")))
    for pointer in aged:
        if (pointer.get("reason") != "news_window_aged_out" or
                pointer.get("replacement_ids") != [] or
                not pointer.get("predecessor_fingerprint")):
            raise ApplyAbort("invalid aged-out tombstone pointer")
        all_ids.append(str(pointer.get("predecessor_id")))
    if len(all_ids) != len(set(all_ids)) or "None" in all_ids:
        raise ApplyAbort("tombstone predecessor ids are missing/duplicated")
    counts = value.get("counts") or {}
    if counts != {"source_group": len(source), "aged_out": len(aged)}:
        raise ApplyAbort("tombstone pointer counts mismatch")
    return value


def state(client: QdrantClient, collection: str) -> dict:
    info = client.get_collection(collection)
    queue = int(getattr(getattr(info, "update_queue", None), "length", 0) or 0)
    return {"status": str(info.status).lower().split(".")[-1],
            "optimizer_status": str(info.optimizer_status).lower().split(".")[-1],
            "points_count": client.count(collection, exact=True).count,
            "update_queue_length": queue}


def require_green(value: dict, label: str) -> None:
    if value["status"] != "green" or value["optimizer_status"] != "ok" or value["update_queue_length"]:
        raise ApplyAbort(f"{label} is not quiet-green: {value}")


def retrieve_count(client, collection: str, ids: list[str], batch: int = 512) -> int:
    return sum(len(client.retrieve(collection, ids=ids[i:i + batch],
                                   with_payload=False, with_vectors=False))
               for i in range(0, len(ids), batch))


def projected_batches(path: Path, batch_size: int):
    """Re-open and project an artifact without retaining its full rows."""
    batch = []
    for row in records(path):
        batch.append(to_point(row))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def compare_target(client, collection: str, artifact: Path,
                   batch_size: int) -> Comparison:
    """Compare the artifact against one target at THAT target's precision.

    The datatype is read from the collection being compared, here, on every
    comparison -- never passed in by a caller who could hand in a softer one.
    Against a float16-storage collection (`aw_e8`, `256-pq8`) a float32
    artifact vector cannot round-trip byte-identically; narrowing the artifact
    side to the declared storage datatype makes the equality the check always
    meant, and keeps it an equality.  See the retired `dense_storage_precision.py` script.
    """
    return compare_batches(
        projected_batches(artifact, batch_size),
        lambda point_ids: client.retrieve(
            collection, ids=point_ids, with_payload=True, with_vectors=True),
        dense_datatype=dense_datatype_of(client.get_collection(collection)))


def guard_cards_before_writes(clients: dict, collections: dict, artifact: Path,
                              comparisons: dict, *, waivers: dict | None, ledger: Path,
                              pack: str) -> dict:
    """Task 820 / review 823 M2: the drain's card check for EVERY target this run will write,
    BEFORE the first write on ANY of them (called from `preflight` and `recovery_preflight`, so a
    dry run refuses too). A stored entity card overwritten by id with a row that is not a card
    replacing it, on the twin or on prod, refuses the run while nothing is written anywhere.
    Only the rows in each target's `write_ids` are checked: one indexed id retrieve per 500 ids
    per target (`prod_guard` paces prod, refuses it under `AW_REPLAY=1`). Any failed read is a
    refusal too (ledgered `CARD-GUARD-ERROR`), never a pass. Returns `{target: report}`; each
    report names its collection, which `apply_elided_points` checks."""
    reports: dict = {}
    for name, comparison in comparisons.items():
        collection = collections[name]
        try:
            report = guard_card_overwrites(
                ({"id": str(r["id"]), "payload": r["payload"]}
                 for r in records(artifact) if str(r["id"]) in comparison.write_ids),
                client_url(clients[name]), collection, writer="cytir.index.apply", waivers=waivers)
        except CardOverwriteRefused as exc:
            append_ledger(ledger, {"status": "CARD-GUARD-REFUSED", "pack": pack, "target": name,
                                   "writer": "cytir.index.apply", "reason": str(exc)[:4000]})
            raise ApplyAbort(f"{name}: {exc}") from exc
        except ApplyAbort:
            raise
        except Exception as exc:  # noqa: BLE001 -- a failed read is a refusal, ledgered, re-raised
            append_ledger(ledger, {"status": "CARD-GUARD-ERROR", "pack": pack, "target": name,
                                   "error": f"{type(exc).__name__}: {exc}"[:4000]})
            raise
        reports[name] = {**report, "collection": collection}
        append_ledger(ledger, {"status": "CARD-GUARD-OK", "pack": pack, "target": name,
                               "card_guard": reports[name]})
    return reports


def apply_elided_points(client, collection: str, artifact: Path,
                        batch_size: int, comparison: Comparison,
                        superseder: Superseder, *, card_guard: dict | None) -> int:
    """Observe the full pack but write only points proven non-identical.

    Task 820: refuses (ApplyAbort, nothing written) without `card_guard`, the report
    `guard_cards_before_writes` returned for THIS collection. The read happens once, in the
    preflight, for both targets, before either is written."""
    if not card_guard or card_guard.get("collection") != collection:
        raise ApplyAbort(
            f"{collection}: no card-overwrite check ran for this target before its write")
    batch = []
    for row in records(artifact):
        payload = row["payload"] or {}
        point_id = str(row["id"])
        superseder.observe_payload(point_id, payload)
        if point_id in comparison.write_ids:
            batch.append(to_point(row))
        if len(batch) >= batch_size:
            client.upsert(collection, points=batch, wait=True)
            batch = []
    if batch:
        client.upsert(collection, points=batch, wait=True)
    return superseder.run()["superseded"]


def wait_green(client, collection: str, timeout: int) -> dict:
    deadline = time.time() + timeout
    while True:
        value = state(client, collection)
        if value["status"] == "green" and value["optimizer_status"] == "ok" and not value["update_queue_length"]:
            return value
        if time.time() >= deadline:
            raise ApplyAbort(f"settle timeout: {value}")
        time.sleep(15)


def parse_arm(stdout: str) -> Path:
    values = [Path(line.split("=", 1)[1]) for line in stdout.splitlines()
              if line.startswith("ARM_PATH=")]
    if len(values) != 1 or not values[0].is_file():
        raise ApplyAbort("content arm runner did not return one materialized arm")
    return values[0]


def gate_query_width(url: str, collection: str) -> int:
    """The dense width this collection MUST be queried at.

    `cytir/serve/models.py` truncates the embedder's 1024-dim output to
    `AW_MRL_DIM` and Qdrant rejects every other width outright, so this is not
    a knob the gate may inherit, default or pin -- it is a property of the
    target.  Task 600: pointed at 256-dim `aw_e8` with the shipped default of
    128, all 1,418 gate queries returned `__ERROR__:UnexpectedResponse`, and
    the arm was written anyway.  Pre-flip the default happened to equal
    production's width, so nothing ever exercised the assumption.
    """
    endpoint = f"{url.rstrip('/')}/collections/{collection}"
    try:
        body = IH.fetch_json(endpoint, timeout=30)["result"]
        return dense_size_of(body)
    except (OSError, ValueError, KeyError, DensePrecisionError) as exc:
        raise ApplyAbort(
            f"cannot read the gate target's dense width from {endpoint}: "
            f"{exc}") from None


def twin_readiness(url: str, collection: str) -> dict:
    """Task 608's predicate, asserted after `wait_green` and filed.

    A reading that cannot be TAKEN aborts the transaction -- an unreadable
    input must never read as clean.  A reading that says `not ready` is
    recorded and does not by itself refuse: task 609 measured the predicate
    false at BOTH the failing and the passing reading of this transition, so it
    is evidence, not the discriminator.  The discriminator is the pair of
    answers `converged_candidate` compares.
    """
    try:
        return qdrant_readiness.snapshot(url, collection)
    except qdrant_readiness.ReadinessError as exc:
        raise ApplyAbort(
            f"twin readiness could not be read, so the gate cannot say what "
            f"state it scored: {exc}") from None


ANCHOR_TIES_MARK = "ANCHOR_TIES_JSON "


def content_gate_argv(comparator, baseline_arm, candidate, null_flags):
    """The ADR-043 content-pack comparator call. `--anchor-tie` (task 933, T25) reports a
    same-document anchor swap as a tie instead of exiting 3; every other movement still does."""
    return [sys.executable, str(comparator), str(baseline_arm), str(candidate),
            "--profile", "content-pack", "--anchor-tie", *null_flags]


def parse_anchor_ties(stdout: str) -> list:
    """The ties the comparator reported (probe, old/new rank, old/new anchors); [] when none."""
    ties = []
    for line in stdout.splitlines():
        if line.startswith(ANCHOR_TIES_MARK):
            ties.extend(json.loads(line[len(ANCHOR_TIES_MARK):]))
    return ties


def converged_candidate(cut_arm, gate_dir: Path, readiness,
                        sleep=time.sleep,
                        layout_fingerprint=None) -> tuple[Path, dict]:
    """ADR-043 am.5: score twice, agree, and only then rule.

    Task 609 measured the applier's OWN instrument reading core-639
    206 -> 201 (18 core losses, FAIL) at `wait_green` + 0 and 206 -> 206 (0
    losses) eight minutes later, on the same collection with the same pack.
    255 of 1,420 queries had lost a top-10 url in the first reading and 82.8%
    of them were back in the second.  A pack refused by a gate that had not
    converged has been refused by the clock.

    Returns the LATER of the two agreeing readings -- the one the ADR-043
    comparison is then computed on -- and the receipt it is filed with.  It
    never returns a FAIL: a state that will not settle within the contract's
    bound aborts the transaction as UNADJUDICATED, with its receipt on disk,
    and an uncomparable pair aborts as UNKNOWN.
    """
    try:
        convergence = gate_convergence.converged_arms(
            cut_arm, readiness=readiness, sleep=sleep,
            log=lambda line: print(line, flush=True),
            layout_fingerprint=layout_fingerprint)
    except gate_convergence.ConvergenceUnadjudicated as exc:
        receipt = gate_dir / "convergence.unadjudicated.json"
        receipt.write_text(json.dumps(exc.receipt, indent=1,
                                      sort_keys=True) + "\n")
        raise ApplyAbort(
            f"ADR-043 content gate UNADJUDICATED (am.5) -- this is not a FAIL "
            f"and not a PASS: {exc} Receipt: {receipt}") from None
    except gate_convergence.ConvergenceUnknown as exc:
        raise ApplyAbort(
            f"ADR-043 content gate convergence UNKNOWN (am.5): {exc}") from None
    receipt = gate_dir / "convergence.json"
    receipt.write_text(json.dumps(convergence, indent=1, sort_keys=True) + "\n")
    return Path(convergence["arm"]), {
        "verdict": convergence["verdict"],
        "rounds": convergence["rounds"],
        "settle_seconds": convergence["settle_seconds"],
        "arms": convergence["arms"],
        "queries_compared": convergence["receipts"][-1]["queries_compared"],
        "readiness": convergence["readiness"],
        "receipt": str(receipt),
        "receipt_sha256": digest(receipt)}


def gate_environment(release_dir: Path, twin_url: str,
                     collection: str) -> dict[str, str]:
    """Render Task370's pinned policy-off environment from its one home."""
    source = release_dir / "cytir" / "gates" / "run_arm.py"
    spec = importlib.util.spec_from_file_location("content_pack_run369", source)
    if spec is None or spec.loader is None:
        raise ApplyAbort(f"cannot load Task369 pinned environment: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    environment = dict(__import__("os").environ)
    environment.update(module.PINNED_ENV)
    environment.update({
        "AW_COUNTRY_SCOPE_MODE": "policy_only",
        "AW_QHEAD_COUNTRY_LEGS": "0",
        "AW_QHEAD_COUNTRY_TRACE": "1",
        "AW_QDRANT_URL": twin_url,
        "AW_COLLECTION": collection,
        # Bound to the target, never inherited: see `gate_query_width`.
        "AW_MRL_DIM": str(gate_query_width(twin_url, collection)),
        # Bound to the SERVING CONFIG, never inherited (task 604). The width
        # above is a property of the collection; the dense rescore policy and
        # the fusion weights are properties of what production serves, and
        # the retired `serving_config.py` script is their one home. Left to the shipped
        # defaults, a gate arm scores rescore off and `fusion_weights.json` --
        # 82 golds and +6 h@1 away from the ranker on websear.ch.
        **serving_config.gate_dense_env(release_dir / SERVER_REL),
        "AW_QHEAD_PATH": str(release_dir / SERVER_REL / "qhead.json"),
        "AW_HUBFLOOR_TABLE": str(release_dir / SERVER_REL /
                                  "hub_hosts.json"),
    })
    for key in ("AW_QHEAD_PATH", "AW_HUBFLOOR_TABLE"):
        if not Path(environment[key]).is_file():
            raise ApplyAbort(f"gate artifact absent: {key}={environment[key]}")
    return environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--staged-verification", type=Path, required=True)
    parser.add_argument("--staged-verification-sha256", required=True)
    parser.add_argument("--staged-collection", required=True)
    parser.add_argument("--tombstone-pointers", type=Path)
    parser.add_argument("--tombstone-pointers-sha256")
    parser.add_argument("--twin-url", required=True)
    parser.add_argument("--prod-url", required=True)
    parser.add_argument(
        "--collection",
        help="legacy common collection name for both targets")
    parser.add_argument("--twin-collection")
    parser.add_argument("--prod-collection")
    parser.add_argument("--card-waivers", help="sealed card-waiver file (task 728 format) naming "
                        "a stored card this pack may overwrite (task 820)")
    parser.add_argument("--card-waivers-sha256")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--baseline-arm", type=Path, required=True)
    parser.add_argument("--baseline-arm-sha256", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--qvec-query-source-manifest", required=True)
    parser.add_argument("--prod-lock-host", required=True)
    parser.add_argument("--prod-lock-path", default="/tmp/aw_prod_writer.lock")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--settle-timeout", type=int, default=21600)
    parser.add_argument("--resume-after-verified-twin-gate", action="store_true")
    parser.add_argument(
        "--stop-after-twin-gate", action="store_true",
        help="task 626: write and gate the TWIN for real, then stop -- "
             "production is never touched, on a PASS or otherwise (a REFUSE "
             "already stops there). For proving a fixed gate on a real write "
             "before a human-gated go-ahead (a GO file) authorizes the "
             "production leg. No cadence 'pack' row is recorded either: "
             "the pack has not landed (twin_first_apply.stop_after_twin_gate).")
    parser.add_argument("--failed-ledger", type=Path)
    parser.add_argument("--failed-ledger-sha256")
    parser.add_argument("--candidate-arm", type=Path)
    parser.add_argument("--candidate-arm-sha256")
    parser.add_argument("--arm", default="")
    parser.add_argument(
        "--gate-profile", default="content-pack",
        choices=("content-pack", "news-daily"),
        help="'content-pack' (default) runs the 1,418-query ADR-043 battery "
             "this transaction has always run. 'news-daily' runs ADR-043 "
             "am.4's daily gate -- the whole gold core, the flood set and "
             "the probes (667 queries), all four ADR-043 conditions, and a "
             "cadence ledger that refuses the pack once a battery is due. It refuses any pack outside the "
             "routine scope in data/news-daily-gate-contract.v1.json.")
    parser.add_argument(
        "--daily-gate-intent", type=Path,
        help="the sealed preservation intent whose scope, rows and source "
             "profiles decide whether this pack is routine at all. Defaults "
             "to AW_CONTENT_PACK_PRESERVATION_INTENT, which the armed step "
             "already exports; required with --gate-profile news-daily.")
    parser.add_argument(
        "--cadence-ledger", type=Path,
        help="the ADR-043 am.4 cadence ledger (append-only JSONL). Required "
             "with --gate-profile news-daily; a missing one is a refusal, "
             "never a zero.")
    parser.add_argument(
        "--gate-null-arm", type=Path,
        help="ADR-043 am.3/am.6: the null arm this run cut on a state proved "
             "to hold the baseline's content -- in practice the arm cut AFTER "
             "a previous attempt's rollback, with the same cutter the "
             "baseline used, read under am.5 like any other arm. Its own "
             "movement is the instrument's and is taken off the pack's "
             "account, LEGAL/NAMES tripwires included (am.6). Both doors pass "
             "it straight through to the judge; neither adjudicates anything "
             "in band.")
    parser.add_argument(
        "--gate-null-basis", type=Path,
        help="the aw.content-gate-null-basis.v2 receipt for --gate-null-arm: "
             "the protocol, the membership digest that proves content "
             "equality, the convergence receipt, and the baseline arm it "
             "adjudicates. Required with --gate-null-arm, and vice versa.")
    parser.add_argument(
        "--prod-baseline-arm", type=Path, default=None,
        help="ADR-043 am.7 (task 625/626): a CONVERGED reading of production "
             "on the same daily slice, cut concurrently with the twin's "
             "pre-apply baseline. When given, --gate-profile news-daily "
             "scores the candidate against THIS, not the twin's own "
             "pre-apply reading -- concurrent production is ground truth for "
             "what the pack costs, and a twin/prod disagreement is reported "
             "in the verdict (`baseline_disagreement`) rather than traded "
             "silently. Only valid with --gate-profile news-daily.")
    parser.add_argument(
        "--prod-baseline-arm-sha256",
        help="sha256 of --prod-baseline-arm; required together with it, "
             "same one-control rule as every other sealed input here.")
    parser.add_argument(
        "--waiver-ref", type=Path, default=None,
        help="task 626: a Tim-issued GO file waiving a daily-gate REFUSE "
             "for THIS pack only (memory: gate-waivers-escalate-to-tim -- "
             "never self-issued; this flag makes the escalation an input, "
             "not a code path that can soften its own standard). When "
             "given alongside --waiver-issuer, a REFUSE verdict is still "
             "computed and logged VERBATIM (the raw verdict is never "
             "altered, no band is ever changed), but the transaction "
             "proceeds anyway, with the waiver reference, its content hash, "
             "the issuer and the moment it was applied recorded in the "
             "apply ledger as their own row. A PASS verdict needs no "
             "waiver and this flag changes nothing for one. Only valid "
             "with --gate-profile news-daily.")
    parser.add_argument(
        "--waiver-issuer",
        help="who ruled the waiver in --waiver-ref (e.g. 'Tim'). Required "
             "together with --waiver-ref -- a waiver with no named human "
             "behind it is not a waiver, it is a gate that can excuse "
             "itself.")
    parser.add_argument(
        "--prod-baseline-reuse-proof", type=Path, default=None,
        help="task626 catch-up item 1 (Tim, 2026-09-21): when --prod-"
             "baseline-arm was REUSED from a previous transaction's "
             "post-apply reading rather than freshly cut for today (no "
             "write to production and the #218 layout fingerprint "
             "unchanged since that reading -- the retired `collection_layout.py` script."
             "quiescent), this is the JSON proof of both conditions, folded "
             "verbatim into the daily door's ledger row so the reuse is "
             "auditable, never silent. Optional: absent means this "
             "transaction cut its own baseline fresh. Only valid with "
             "--gate-profile news-daily, same as the arm it documents.")
    parser.add_argument(
        "--persisted-overlay", type=Path, default=None,
        help="optional JSON {url: path} of pages this run itself just filed "
             "under an ADR-105 store, e.g. via cytir/corpus/persist_l1_records.py. Passed "
             "to persisted_inputs.assert_pages_persisted as an evidence "
             "overlay -- never a bypass: overlay_from_paths still verifies "
             "every path exists and matches a REGISTERED store glob. Lets a "
             "pack whose pages were persisted in this same run pass without "
             "waiting for the next full corpus-wide index rebuild to see "
             "them, and without needing that index at all if the overlay "
             "covers every page (persisted_inputs.py's lazy index load).")
    args = parser.parse_args()
    # ADR-043 am.3 rule 1 / am.6: an arm and its basis are ONE control. Half of
    # it is not a weaker control, it is an unreadable one -- the arm without the
    # receipt cannot prove its state held the baseline's content, and the
    # receipt without the arm adjudicates nothing.
    if bool(args.gate_null_arm) != bool(args.gate_null_basis):
        parser.error("--gate-null-arm and --gate-null-basis are one control, "
                     "pass both (ADR-043 am.3 rule 1)")
    for path, label in ((args.gate_null_arm, "gate null arm"),
                        (args.gate_null_basis, "gate null basis")):
        if path is not None and not path.is_file():
            parser.error(f"{label} absent: {path} -- a missing control is a "
                         f"refusal, never an unadjudicated pass")
    if bool(args.prod_baseline_arm) != bool(args.prod_baseline_arm_sha256):
        parser.error("--prod-baseline-arm and --prod-baseline-arm-sha256 are "
                     "one control, pass both")
    if args.prod_baseline_arm is not None and args.gate_profile != "news-daily":
        parser.error("--prod-baseline-arm is ADR-043 am.7's daily-gate "
                     "reading; it has no meaning against the battery "
                     "comparator (--gate-profile content-pack)")
    if bool(args.waiver_ref) != bool(args.waiver_issuer):
        parser.error("--waiver-ref and --waiver-issuer are one control, "
                     "pass both (memory: gate-waivers-escalate-to-tim)")
    if args.waiver_ref is not None:
        if args.gate_profile != "news-daily":
            parser.error("--waiver-ref only has meaning against the daily "
                         "gate (--gate-profile news-daily)")
        if not args.waiver_ref.is_file():
            parser.error(f"--waiver-ref absent: {args.waiver_ref} -- a "
                         f"waiver that cannot be read is not a waiver")
    if args.prod_baseline_reuse_proof is not None:
        if args.prod_baseline_arm is None:
            parser.error("--prod-baseline-reuse-proof documents a REUSED "
                         "--prod-baseline-arm; it has no meaning without one")
        if args.gate_profile != "news-daily":
            parser.error("--prod-baseline-reuse-proof only has meaning "
                         "against the daily gate (--gate-profile news-daily)")
        if not args.prod_baseline_reuse_proof.is_file():
            parser.error(f"--prod-baseline-reuse-proof absent: "
                         f"{args.prod_baseline_reuse_proof} -- a reuse "
                         f"claim that cannot be read is not a proof")
    if args.gate_profile == "news-daily":
        if args.daily_gate_intent is None:
            value = os.environ.get("AW_CONTENT_PACK_PRESERVATION_INTENT")
            if not value:
                parser.error("--gate-profile news-daily needs "
                             "--daily-gate-intent (or "
                             "AW_CONTENT_PACK_PRESERVATION_INTENT)")
            args.daily_gate_intent = Path(value)
        if args.cadence_ledger is None:
            parser.error("--gate-profile news-daily needs --cadence-ledger: "
                         "a routine pack is only routine relative to the "
                         "last passing battery (ADR-043 am.4)")
        if args.resume_after_verified_twin_gate:
            parser.error("the resume path re-judges an EXISTING candidate arm "
                         "against the battery's comparator; it has no daily "
                         "profile. Re-run the transaction instead.")
    if args.stop_after_twin_gate and args.resume_after_verified_twin_gate:
        parser.error("--stop-after-twin-gate stops BEFORE the production leg; "
                     "--resume-after-verified-twin-gate resumes only that "
                     "leg. They name opposite halves of one transaction and "
                     "cannot both apply to the same call.")
    target_collections = resolve_target_collections(
        args.collection, args.twin_collection, args.prod_collection)
    targets = {
        "twin": target_binding("twin", args.twin_url,
                               target_collections["twin"]),
        "prod": target_binding("prod", args.prod_url,
                               target_collections["prod"]),
    }
    expected_arm = RESUME_ARM if args.resume_after_verified_twin_gate else ARM
    armed = args.arm == expected_arm
    if args.arm and not armed:
        raise ApplyAbort(f"bad arm token; expected {expected_arm}")
    if armed:
        # task 864 (review 854 S6): an aw_e10 build holds the twin from T0 to the flip; only the
        # drain, whose receipts the catch-up reads, may write it then
        from cytir.index import twin_write_hold as TWH
        try:
            TWH.refuse_if_held(target_collections["twin"], tool="cytir.index.apply")
        except TWH.TwinWriteHeld as exc:
            raise ApplyAbort(str(exc)) from exc
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.pack):
        raise ApplyAbort("invalid pack name")
    for path, expected, label in (
        (args.artifact, args.artifact_sha256, "artifact"),
        (args.staged_verification, args.staged_verification_sha256, "staged verification"),
        (args.baseline_arm, args.baseline_arm_sha256, "baseline arm"),
    ):
        if not path.is_file() or digest(path) != expected:
            raise ApplyAbort(f"{label} path/hash mismatch")
    if args.prod_baseline_arm is not None:
        if (not args.prod_baseline_arm.is_file() or
                digest(args.prod_baseline_arm) != args.prod_baseline_arm_sha256):
            raise ApplyAbort("prod baseline arm path/hash mismatch")
    if args.resume_after_verified_twin_gate:
        for path, expected, label in (
            (args.failed_ledger, args.failed_ledger_sha256,
             "failed predecessor ledger"),
            (args.candidate_arm, args.candidate_arm_sha256,
             "candidate arm"),
        ):
            if path is None or not expected or not path.is_file() or digest(path) != expected:
                raise ApplyAbort(f"{label} path/hash mismatch")
    verification = json.loads(args.staged_verification.read_text())
    if (verification.get("collection") != args.staged_collection or
            verification.get("status") != "green" or
            verification.get("points_count") != args.rows or
            verification.get("distinct_minted_ids") != args.rows or
            not verification.get("count_match")):
        raise ApplyAbort("staged verification does not prove exact pack membership")
    # IDs are the only artifact-wide row state retained.  Payloads and vectors
    # are streamed afresh for comparison/apply, which keeps the 388k-row packs
    # from doubling their already-large in-memory footprint.
    ids = []
    artifact_groups: dict[str, set[str]] = {}
    pack_pages: list[dict] = []
    for row in records(args.artifact):
        point_id = str(row["id"])
        ids.append(point_id)
        payload = row["payload"] or {}
        group = payload_group_key(payload)
        artifact_groups.setdefault(group, set()).add(point_id)
        pack_pages.append({"url": payload.get("url"),
                           "sha": payload.get("content_digest")})
    if len(ids) != args.rows or len(set(ids)) != args.rows:
        raise ApplyAbort("artifact row/distinct-id count mismatch")

    # ADR-105 (task 581) -- persisted inputs are a precondition to load, and
    # this is the LIVE route: the one applier in the repo that writes the twin
    # and then production. ADR-083 mode 1 ("live delta-apply") said nothing
    # about the input being on disk, so every pack this path has ever carried
    # was free to seal nothing. Refused here, before the writer lock, before
    # the preservation intent is bound, and before either mirror is contacted.
    overlay = load_persisted_overlay(args.persisted_overlay)
    try:
        persisted_verdict = PI.assert_pages_persisted(
            pack_pages, require="l1", label=f"content pack {args.pack}",
            overlay=overlay)
    except PI.PersistedInputsError as exc:
        raise ApplyAbort(str(exc)) from None
    print(f"[persisted-inputs] {args.pack}: {persisted_verdict['pages']:,} "
          f"page(s), {persisted_verdict['persisted']:,} L1/L0, "
          f"{persisted_verdict['l1b_reconstructible']:,} L2-only", flush=True)
    if bool(args.tombstone_pointers) != bool(args.tombstone_pointers_sha256):
        raise ApplyAbort("tombstone pointer path and SHA must be supplied together")
    pointers = (load_tombstone_pointers(
        args.tombstone_pointers, args.tombstone_pointers_sha256,
        args.artifact_sha256, artifact_groups)
        if args.tombstone_pointers else
        {"source_group": [], "aged_out": [],
         "counts": {"source_group": 0, "aged_out": 0}})
    source_pointer_ids = [str(item["predecessor_id"])
                          for item in pointers["source_group"]]
    aged_pointer_ids = [str(item["predecessor_id"])
                        for item in pointers["aged_out"]]
    if args.twin_url.rstrip("/") == args.prod_url.rstrip("/"):
        raise ApplyAbort("fake mirror endpoints")
    card_waivers = load_card_waivers(args.card_waivers, args.card_waivers_sha256)
    clients = {"twin": qdrant_connect(args.twin_url, timeout=900),
               "prod": qdrant_connect(args.prod_url, timeout=900)}
    plans: dict[str, dict] = {}
    proofs: dict[str, dict] = {}
    applies: dict[str, dict] = {}

    def verify_preservation_intent() -> dict:
        """Bind the sealed incremental authority to this exact live plan."""
        require_equivalent_mirror_plans(plans)
        tombstone_artifact = (
            {"uri": "runtime://tombstone-pointers",
             "bytes": args.tombstone_pointers.stat().st_size,
             "sha256": args.tombstone_pointers_sha256}
            if args.tombstone_pointers else None)
        # Portable URIs belong to the sealed intent.  Runtime paths are
        # independently hash-checked above, so equality here deliberately
        # compares only immutable size/hash identity.
        runtime = {
            **runtime_release_identity(args.release_dir),
            "pack": {
                "name": args.pack,
                "rows": args.rows,
                "artifact": {
                    "uri": "runtime://artifact",
                    "bytes": args.artifact.stat().st_size,
                    "sha256": args.artifact_sha256,
                },
                "staged_verification": {
                    "uri": "runtime://staged-verification",
                    "bytes": args.staged_verification.stat().st_size,
                    "sha256": args.staged_verification_sha256,
                },
                "tombstone_artifact": tombstone_artifact,
            },
            "targets": targets,
            "mutation_plan": plan_without_target(plans["twin"]),
        }
        return verify_content_pack_publish_environment(runtime=runtime)

    def validate_pointer_readbacks(name: str) -> dict:
        client = clients[name]
        collection = target_collections[name]
        pointer_rows = pointers["source_group"] + pointers["aged_out"]
        pointer_ids = [str(item["predecessor_id"]) for item in pointer_rows]
        retrieved = (client.retrieve(collection, ids=pointer_ids,
                                     with_payload=True, with_vectors=True)
                     if pointer_ids else [])
        observed = {str(point.id): point_fingerprint(point) for point in retrieved}
        expected = {str(item["predecessor_id"]):
                    item["predecessor_fingerprint"] for item in pointer_rows}
        if observed != expected:
            raise ApplyAbort(f"{name} tombstone predecessor readback mismatch")
        return {"rows": len(pointer_ids),
                "fingerprint_digest": hashlib.sha256(canonical(expected)).hexdigest()}

    def exact_supersede_ids(name: str, client) -> list[str]:
        superseder = Superseder(
            client, target_collections[name], dry_run=True,
            prod_scroll_reason=PROD_SCROLL_REASON)
        superseder.ensure_url_index(create=False)
        observe_artifact(superseder)
        superseder.run()
        return superseder.last_stale_ids

    def observe_artifact(superseder: Superseder) -> None:
        for row in records(args.artifact):
            payload = row["payload"] or {}
            superseder.observe_payload(str(row["id"]), payload)

    def exact_comparison(name: str) -> Comparison:
        comparison = compare_target(
            clients[name], target_collections[name], args.artifact, args.batch)
        if comparison.rows != args.rows:
            raise ApplyAbort(
                f"{name} comparison rows {comparison.rows} != {args.rows}")
        return comparison

    card_guards: dict[str, dict] = {}     # kept OUT of `applies`: twin/prod proofs compare equal

    def preflight():
        comparisons: dict[str, Comparison] = {}
        for name, client in clients.items():
            before = state(client, target_collections[name])
            require_green(before, name)
            comparison = exact_comparison(name)
            comparisons[name] = comparison
            supersede_ids = exact_supersede_ids(name, client)
            if args.tombstone_pointers and supersede_ids != sorted(source_pointer_ids):
                raise ApplyAbort(
                    f"{name} explicit source-group tombstones drifted")
            superseded = len(supersede_ids)
            pointer_proof = validate_pointer_readbacks(name)
            plans[name] = {"target": targets[name],
                           "before": before, "rows": args.rows,
                           "existing_ids": comparison.existing,
                           "new_ids": comparison.missing,
                           "point_identical": comparison.identical,
                           "point_noops": comparison.identical,
                           "point_changed": comparison.changed,
                           "point_writes": (comparison.changed +
                                            comparison.missing),
                           "comparison": comparison.evidence(),
                           "superseded": superseded,
                           "aged_tombstones": len(aged_pointer_ids),
                           "tombstone_pointer_sha256":
                               args.tombstone_pointers_sha256,
                           "tombstone_predecessor_proof": pointer_proof,
                           "planned_topology_mutations": (
                               comparison.changed + comparison.missing +
                               superseded + len(aged_pointer_ids)),
                           "expected_delta": (comparison.missing - superseded -
                                              len(aged_pointer_ids))}
        require_equivalent_mirror_plans(plans)
        # task 820 / review 823 M2: the card check of BOTH targets, before the twin is written
        card_guards.update(guard_cards_before_writes(
            clients, target_collections, args.artifact, comparisons,
            waivers=card_waivers, ledger=args.ledger, pack=args.pack))
        append_ledger(args.ledger, {
            "status": "CONTENT-PACK-PLAN",
            "pack": args.pack,
            "artifact_sha256": args.artifact_sha256,
            "staged_verification_sha256": args.staged_verification_sha256,
            "tombstone_pointers_sha256": args.tombstone_pointers_sha256,
            "proof": plans,
        })
        return {"artifact_sha256": args.artifact_sha256,
                "staged_verification_sha256": args.staged_verification_sha256,
                "targets": targets,
                "plan": plan_without_target(plans["twin"])}

    def apply_target(name, _transaction):
        client = clients[name]
        collection = target_collections[name]
        # The original preflight happens before the real writer lock.  Repeat
        # the complete content comparison while locked and fail before the
        # first write if any ID changed in that interval.
        comparison = exact_comparison(name)
        expected = plans[name].get("comparison")
        if expected is None and isinstance(_transaction, dict):
            expected = _transaction.get(f"{name}_comparison")
        if expected is not None and comparison.evidence() != expected:
            raise ApplyAbort(
                f"{name} exact point comparison drifted after preflight")
        if args.tombstone_pointers:
            if exact_supersede_ids(name, client) != sorted(source_pointer_ids):
                raise ApplyAbort(
                    f"{name} locked explicit source-group tombstones drifted")
            validate_pointer_readbacks(name)
        superseder = Superseder(client, collection, prod_scroll_reason=PROD_SCROLL_REASON)
        superseder.ensure_url_index(create=False)
        observed = apply_elided_points(
            client, collection, args.artifact, args.batch, comparison,
            superseder, card_guard=card_guards.get(name))
        if observed != plans[name]["superseded"]:
            raise ApplyAbort(f"{name} supersede plan drift {observed} != {plans[name]['superseded']}")
        if aged_pointer_ids:
            client.delete(
                collection,
                points_selector=models.PointIdsList(points=aged_pointer_ids),
                wait=True)
            remaining = client.retrieve(
                collection, ids=aged_pointer_ids,
                with_payload=False, with_vectors=False)
            if remaining:
                raise ApplyAbort(
                    f"{name} aged tombstones remain after delete: {len(remaining)}")
        applies[name] = {
            "comparison": comparison.evidence(),
            "point_writes": len(comparison.write_ids),
            "point_noops": comparison.identical,
            "superseded": observed,
            "aged_tombstones": len(aged_pointer_ids),
            "tombstone_pointers_sha256": args.tombstone_pointers_sha256,
            "topology_mutations": (len(comparison.write_ids) + observed +
                                   len(aged_pointer_ids)),
        }

    def verify_target(name, _transaction):
        client = clients[name]
        collection = target_collections[name]
        after = wait_green(client, collection, args.settle_timeout)
        delta = after["points_count"] - plans[name]["before"]["points_count"]
        if delta != plans[name]["expected_delta"]:
            raise ApplyAbort(f"{name} count delta {delta} != {plans[name]['expected_delta']}")
        present = retrieve_count(client, collection, ids)
        if present != args.rows:
            raise ApplyAbort(f"{name} artifact membership {present} != {args.rows}")
        aged_remaining = retrieve_count(
            client, collection, aged_pointer_ids) if aged_pointer_ids else 0
        if aged_remaining:
            raise ApplyAbort(f"{name} retained {aged_remaining} aged tombstones")
        terminal = exact_comparison(name)
        if (terminal.identical != args.rows or terminal.changed or
                terminal.missing):
            raise ApplyAbort(
                f"{name} artifact content not exact after apply: "
                f"{terminal.evidence()}")
        proof = {"target": targets[name],
                 "artifact_sha256": args.artifact_sha256, "rows_present": present,
                 "before": plans[name]["before"], "after": after, "delta": delta,
                 "superseded": plans[name]["superseded"],
                 "aged_tombstones_absent": len(aged_pointer_ids),
                 "tombstone_pointers_sha256": args.tombstone_pointers_sha256,
                 "apply": applies[name],
                 "terminal_comparison": terminal.evidence()}
        if (name == "prod" and
                proof_without_target(proofs.get("twin", {})) !=
                proof_without_target(proof)):
            raise ApplyAbort("terminal twin/prod proof mismatch")
        proofs[name] = proof
        return proof

    def readiness() -> dict:
        return twin_readiness(args.twin_url, target_collections["twin"])

    def cut_candidate_arm(points_count: int, round_index: int) -> Path:
        """One battery reading of the twin as it is NOW -- am.5 needs two.

        Each round gets its own out-dir and arm id: `run_content_pack_arm`
        writes its manifests with `O_EXCL` under a content-addressed name, and
        two readings of one state produce byte-identical corpus and target
        manifests, so a shared directory would make the second round collide
        rather than measure.
        """
        helper = args.release_dir / "cytir" / "gates" / "content_pack_arm.py"
        out_dir = args.gate_dir / f"candidate-r{round_index}"
        command = [sys.executable, str(helper), "--release-dir", str(args.release_dir),
                   "--qdrant-url", args.twin_url, "--collection",
                   target_collections["twin"],
                   "--representation", "content-pack",
                   "--identity", f"content-pack:{args.pack}:{args.artifact_sha256}:{points_count}",
                   "--source-sha256", args.staged_verification_sha256,
                   "--supersede-sha256", (args.tombstone_pointers_sha256 or
                                          args.artifact_sha256),
                   "--cache", args.cache,
                   "--qvec-query-source-manifest", args.qvec_query_source_manifest,
                   "--out-dir", str(out_dir),
                   "--arm-id", f"{args.pack}-candidate-r{round_index}"]
        arm_run = subprocess.run(
            command, cwd=args.release_dir, text=True, capture_output=True,
            env=gate_environment(args.release_dir, args.twin_url,
                                 target_collections["twin"]))
        if arm_run.returncode:
            raise ApplyAbort(
                f"candidate arm r{round_index} failed: "
                f"{arm_run.stdout}{arm_run.stderr}")
        return parse_arm(arm_run.stdout)

    def cut_daily_candidate_arm(points_count: int, round_index: int) -> Path:
        """One DAILY reading of the twin as it is now -- am.5 needs two.

        The same per-round out-dir and arm id the battery cutter uses, for the
        same reason one level down: `news_daily_gate.arm` names its file after
        the sha256 of its own payload and creates it with `O_EXCL`, and two
        readings of one unchanged state that agree write byte-identical
        payloads -- so a shared directory would make the CONVERGED round
        collide instead of measure.
        """
        out_dir = args.gate_dir / f"daily-candidate-r{round_index}"
        command = [sys.executable,
                   str(args.release_dir / "cytir" / "gates" / "news_daily.py"),
                   "--contract",
                   str(args.release_dir / "data" /
                       "news-daily-gate-contract.v1.json"),
                   "arm", "--release-dir", str(args.release_dir),
                   "--qdrant-url", args.twin_url,
                   "--collection", target_collections["twin"],
                   "--identity",
                   f"content-pack:{args.pack}:{args.artifact_sha256}:"
                   f"{points_count}",
                   "--cache", args.cache,
                   "--out-dir", str(out_dir),
                   "--arm-id",
                   f"{args.pack}-daily-candidate-r{round_index}".lower()]
        arm_run = subprocess.run(
            command, cwd=args.release_dir, text=True, capture_output=True,
            env=gate_environment(args.release_dir, args.twin_url,
                                 target_collections["twin"]))
        if arm_run.returncode:
            raise ApplyAbort(f"daily-gate arm r{round_index} failed: "
                             f"{arm_run.stdout}{arm_run.stderr}")
        return parse_arm(arm_run.stdout)

    def null_arm_proof() -> dict:
        """What the gate proof records about the control it was judged with."""
        if not args.gate_null_arm:
            return {"null_arm": None, "null_basis": None}
        return {"null_arm": str(args.gate_null_arm),
                "null_arm_sha256": digest(args.gate_null_arm),
                "null_basis": str(args.gate_null_basis),
                "null_basis_sha256": digest(args.gate_null_basis)}

    def waiver_proof() -> dict:
        """What the apply ledger records about a Tim-issued gate waiver
        (task 626, memory: gate-waivers-escalate-to-tim). The waiver
        reference's own bytes are captured by content hash and copied in
        full, so the ledger row is self-contained -- readable years later
        without needing the GO file to still exist on disk at that path."""
        if not args.waiver_ref:
            return {"waiver_ref": None, "waiver_issuer": None}
        return {"waiver_ref": str(args.waiver_ref),
                "waiver_ref_sha256": digest(args.waiver_ref),
                "waiver_ref_text": args.waiver_ref.read_text(),
                "waiver_issuer": args.waiver_issuer,
                "waived_at": _utcnow().isoformat()}

    def null_arm_flags() -> list[str]:
        """ADR-043 am.3/am.6 through the battery door, when this run has one.

        Task 601 sec.11 and task 613 sec.8 found this door passing no null arm
        at all: the comparator has carried `--null-arm`/`--null-basis` since
        am.3 and the applier never handed them over, so every battery verdict
        the lane has taken was unadjudicated by construction -- and am.6, which
        judges the LEGAL/NAMES tripwires against the same control, would have
        been unreachable from the door that writes production.
        """
        if not args.gate_null_arm:
            return []
        return ["--null-arm", str(args.gate_null_arm),
                "--null-basis", str(args.gate_null_basis)]

    def post_twin_daily_gate(_transaction, twin_proof):
        """ADR-043 am.4: the routine-pack gate, on 667 queries not 1,418.

        Same transaction, same twin, same pinned and serving-bound gate
        environment, same arm-refusal rules -- a different QUERY POPULATION
        and a smaller set of conditions, by Tim's ruling of 2026-09-17.  The
        full battery still runs, weekly, and this refuses the pack outright
        once the cadence ledger says one is due, so the interval cannot be
        skipped by never running it.

        ADR-043 am.5 binds this door on the same terms as the battery's.  am.4
        chose the POPULATION a routine pack is judged on; am.5 says when a
        reading of that population counts as a reading at all, and the routine
        path is the one that runs every day on a collection that has just taken
        a write -- so it is the door where a transient reading would be cashed
        most often, not the one to leave outside the protocol.  The convergence
        population (`gate_convergence.BAR_SETS`) is the daily slice's own five
        sets: its 639 gold core and the 28-query flood tripwire.

        A refusal here aborts the transaction before production is written,
        exactly as the battery's refusal does.  It is never adjudicated in
        band by this code path (memory: gate-waivers-escalate-to-tim).
        """
        contract = news_daily_gate.load_contract(
            args.release_dir / "data" / "news-daily-gate-contract.v1.json")
        intent = json.loads(args.daily_gate_intent.read_text())
        ledger_rows = news_daily_gate.read_ledger(args.cadence_ledger)
        state = news_daily_gate.cadence_state(
            ledger_rows, now=_utcnow())
        release_sha = runtime_release_identity(args.release_dir)["git_sha"]
        cadence = {
            **state,
            "max_packs": contract["cadence"]["max_packs_between_batteries"],
            "problems": news_daily_gate.cadence_problems(
                state, contract, release_sha=release_sha,
                pack_rows=int((intent.get("pack") or {}).get("rows") or 0))}
        candidate, convergence = converged_candidate(
            lambda index: cut_daily_candidate_arm(
                twin_proof["after"]["points_count"], index),
            args.gate_dir, readiness,
            layout_fingerprint=lambda: collection_layout.fingerprint(
                args.twin_url, target_collections["twin"]))
        try:
            verdict = news_daily_gate.judge(
                baseline=json.loads(args.baseline_arm.read_text()),
                candidate=json.loads(candidate.read_text()),
                contract=contract,
                slice_doc=news_daily_gate.load_slice(contract,
                                                     args.release_dir),
                intent=intent, cadence=cadence,
                null_arm=(json.loads(args.gate_null_arm.read_text())
                          if args.gate_null_arm else None),
                null_basis=(json.loads(args.gate_null_basis.read_text())
                            if args.gate_null_basis else None),
                baseline_arm_sha256=digest(args.baseline_arm),
                prod_baseline=(json.loads(args.prod_baseline_arm.read_text())
                              if args.prod_baseline_arm else None))
        except (news_daily_gate.DailyGateError,
                news_daily_gate.null.NullArmError) as exc:
            raise ApplyAbort(f"ADR-043 am.4 daily gate is not a "
                             f"measurement: {exc}") from None
        log_path = args.gate_dir / (
            f"daily-gate.{digest(args.baseline_arm)}.{digest(candidate)}.json")
        if log_path.exists():
            raise ApplyAbort(f"refusing to overwrite gate log: {log_path}")
        log_path.write_text(json.dumps(verdict, indent=1, sort_keys=True) + "\n")
        waived = False
        if verdict["verdict"] != "PASS":
            if args.waiver_ref is None:
                raise ApplyAbort("ADR-043 am.4 daily gate failed: "
                                 + "\n".join(news_daily_gate.render(verdict)))
            # Task 626: a Tim-issued waiver of THIS refusal, for THIS pack
            # only. The verdict above is already logged verbatim, unedited,
            # to verdict_log -- nothing here rewrites it, softens a band or
            # skips the gate; the gate ran, refused, and that refusal is
            # the permanent record. The waiver is a human decision to
            # proceed past a true REFUSE, and it is recorded as its own
            # fact, not folded into the verdict as if it had passed.
            waived = True
        return {"status": "PASS", "profile": "news-daily",
                "gate_verdict": verdict["verdict"],
                "baseline_arm_sha256": digest(args.baseline_arm),
                "prod_baseline_arm": (str(args.prod_baseline_arm)
                                      if args.prod_baseline_arm else None),
                "prod_baseline_arm_sha256": (digest(args.prod_baseline_arm)
                                            if args.prod_baseline_arm else None),
                "baseline_disagreement": verdict["baseline_disagreement"],
                "candidate_arm": str(candidate),
                "candidate_arm_sha256": digest(candidate),
                "verdict_log": str(log_path),
                "verdict_log_sha256": digest(log_path),
                "slice_sha256": verdict["slice_sha256"],
                "slice_queries": verdict["slice_queries"],
                "cadence": cadence, "metrics": verdict["metrics"],
                "convergence": convergence, "waived": waived,
                "prod_baseline_reuse": (
                    json.loads(args.prod_baseline_reuse_proof.read_text())
                    if args.prod_baseline_reuse_proof else None),
                **waiver_proof(), **null_arm_proof()}

    def post_twin_battery_gate(_transaction, twin_proof):
        comparator = args.release_dir / "cytir" / "gates" / "compare_corpus_arms.py"
        candidate, convergence = converged_candidate(
            lambda index: cut_candidate_arm(
                twin_proof["after"]["points_count"], index),
            args.gate_dir, readiness,
            layout_fingerprint=lambda: collection_layout.fingerprint(
                args.twin_url, target_collections["twin"]))
        comparison = subprocess.run(
            content_gate_argv(comparator, args.baseline_arm, candidate, null_arm_flags()),
            cwd=args.release_dir,
            text=True, capture_output=True)
        log_path = args.gate_dir / f"compare.{digest(args.baseline_arm)}.{digest(candidate)}.log"
        if log_path.exists():
            raise ApplyAbort(f"refusing to overwrite gate log: {log_path}")
        log_path.write_text(comparison.stdout + comparison.stderr)
        if comparison.returncode:
            raise ApplyAbort(f"ADR-043 content gate failed: {comparison.stdout}{comparison.stderr}")
        ties = parse_anchor_ties(comparison.stdout)
        return {"status": "PASS", "profile": "content-pack",
                "anchor_tie_count": len(ties), "anchor_ties": ties,
                "baseline_arm_sha256": digest(args.baseline_arm),
                "candidate_arm": str(candidate),
                "candidate_arm_sha256": digest(candidate),
                "comparison_log": str(log_path),
                "comparison_log_sha256": digest(log_path),
                "convergence": convergence, **null_arm_proof()}

    # One transaction, two gate profiles -- the battery (every non-routine
    # pack, every rebuild, the default) and ADR-043 am.4's daily gate. The
    # profile is an argument the caller passes, and the DAILY one refuses
    # anything outside the routine scope its contract names, so choosing it
    # for a pack it does not cover is a refusal rather than a shortcut.
    post_twin_gate = (post_twin_daily_gate
                      if args.gate_profile == "news-daily"
                      else post_twin_battery_gate)

    if args.resume_after_verified_twin_gate:
        try:
            lineage = load_failed_twin_lineage(
                args.failed_ledger, args.failed_ledger_sha256,
                pack=args.pack, artifact_sha256=args.artifact_sha256,
                staged_verification_sha256=args.staged_verification_sha256,
                rows=args.rows,
                twin_target=targets["twin"], prod_target=targets["prod"])
        except LineageError as exc:
            raise ApplyAbort(str(exc)) from exc
        plans.update(lineage["plans"])

        def recovery_preflight():
            try:
                bound = load_failed_twin_lineage(
                    args.failed_ledger, args.failed_ledger_sha256,
                    pack=args.pack, artifact_sha256=args.artifact_sha256,
                    staged_verification_sha256=args.staged_verification_sha256,
                    rows=args.rows,
                    twin_target=targets["twin"], prod_target=targets["prod"])
            except LineageError as exc:
                raise ApplyAbort(str(exc)) from exc
            current = {name: state(client, target_collections[name])
                       for name, client in clients.items()}
            for name, value in current.items():
                require_green(value, name)
            if current["twin"] != bound["twin"]["after"]:
                raise ApplyAbort(
                    f"current twin no longer matches predecessor proof: {current['twin']}")
            if current["prod"] != bound["plan"]["before"]:
                raise ApplyAbort(
                    f"current prod no longer matches predecessor plan: {current['prod']}")
            membership = {name: retrieve_count(
                              client, target_collections[name], ids)
                          for name, client in clients.items()}
            prod_comparison = exact_comparison("prod")
            require_resume_preflight_invariants(
                plan=bound["plan"], rows=args.rows, membership=membership,
                prod_comparison=prod_comparison.evidence())
            # task 820 / review 823 M2: the twin is already written; prod is the leg left to guard
            card_guards.update(guard_cards_before_writes(
                {"prod": clients["prod"]}, target_collections, args.artifact,
                {"prod": prod_comparison}, waivers=card_waivers, ledger=args.ledger,
                pack=args.pack))
            superseder = Superseder(
                clients["prod"], target_collections["prod"], dry_run=True,
                prod_scroll_reason=PROD_SCROLL_REASON)
            superseder.ensure_url_index(create=False)
            observe_artifact(superseder)
            superseded = superseder.run()["superseded"]
            if superseded != bound["plan"]["superseded"]:
                raise ApplyAbort(
                    f"resume prod supersede plan drift {superseded} != "
                    f"{bound['plan']['superseded']}")
            return {
                "artifact_sha256": args.artifact_sha256,
                "failed_ledger": str(args.failed_ledger),
                "failed_ledger_sha256": args.failed_ledger_sha256,
                "candidate_arm": str(args.candidate_arm),
                "candidate_arm_sha256": args.candidate_arm_sha256,
                "targets": targets, "current": current,
                "membership": membership,
                "prod_comparison": prod_comparison.evidence(),
                "plan": bound["plan"], "twin_proof": bound["twin"],
            }

        def recovery_gate(_state):
            comparator = (args.release_dir / "cytir" / "gates" /
                          "compare_corpus_arms.py")
            # ADR-043 am.5 binds this door too. The bound candidate arm is ONE
            # reading, cut by a predecessor whose gate refused -- exactly the
            # reading am.5 says may have been taken in a post-insert transient.
            # So it enters the protocol as round 0 and is admitted only if a
            # fresh reading of this same (preflight-verified, unchanged) twin
            # agrees with it per gold; otherwise the protocol keeps scoring and
            # rules on the settled pair. A guard the caller can route around is
            # not a guard.
            candidate, convergence = converged_candidate(
                lambda index: (args.candidate_arm if index == 0 else
                               cut_candidate_arm(
                                   lineage["twin"]["after"]["points_count"],
                                   index)),
                args.gate_dir, readiness,
                layout_fingerprint=lambda: collection_layout.fingerprint(
                    args.twin_url, target_collections["twin"]))
            comparison = subprocess.run(
                [sys.executable, str(comparator), str(args.baseline_arm),
                 str(candidate), "--profile", "content-pack",
                 *null_arm_flags()],
                cwd=args.release_dir, text=True, capture_output=True)
            log_path = args.gate_dir / (
                f"compare.{digest(args.baseline_arm)}."
                f"{digest(candidate)}.resume.log")
            if log_path.exists():
                raise ApplyAbort(f"refusing to overwrite gate log: {log_path}")
            log_path.write_text(comparison.stdout + comparison.stderr)
            if comparison.returncode:
                raise ApplyAbort(
                    f"ADR-043 content gate failed: "
                    f"{comparison.stdout}{comparison.stderr}")
            return {
                "status": "PASS", "profile": "content-pack",
                "recovery": "existing candidate bound to verified twin",
                "failed_ledger_sha256": args.failed_ledger_sha256,
                "baseline_arm_sha256": digest(args.baseline_arm),
                "bound_candidate_arm": str(args.candidate_arm),
                "bound_candidate_arm_sha256": args.candidate_arm_sha256,
                "candidate_arm": str(candidate),
                "candidate_arm_sha256": digest(candidate),
                "comparison_log": str(log_path),
                "comparison_log_sha256": digest(log_path),
                "convergence": convergence, **null_arm_proof(),
            }

        def recovery_apply_prod(_state):
            apply_target("prod", _state)

        def recovery_verify_mirrors(_state):
            expected = lineage["twin"]["after"]
            current = {"twin": state(clients["twin"],
                                     target_collections["twin"]),
                       "prod": wait_green(clients["prod"],
                                          target_collections["prod"],
                                          args.settle_timeout)}
            for name, value in current.items():
                require_green(value, name)
            if current != {"twin": expected, "prod": expected}:
                raise ApplyAbort(f"terminal mirror states differ: {current}")
            membership = {name: retrieve_count(
                              client, target_collections[name], ids)
                          for name, client in clients.items()}
            if membership != {"twin": args.rows, "prod": args.rows}:
                raise ApplyAbort(
                    f"terminal mirror membership differs: {membership}")
            terminal = {name: exact_comparison(name).evidence()
                        for name in clients}
            for name, comparison in terminal.items():
                if (comparison["identical"] != args.rows or
                        comparison["changed"] or comparison["missing"]):
                    raise ApplyAbort(
                        f"terminal {name} artifact content not exact: "
                        f"{comparison}")
            return {
                "status": "PASS", "artifact_sha256": args.artifact_sha256,
                "rows_present": args.rows, "targets": targets,
                "states": current,
                "membership": membership,
                "expected_delta": lineage["plan"]["expected_delta"],
                "superseded": lineage["plan"]["superseded"],
                "prod_apply": applies["prod"],
                "terminal_comparison": terminal,
            }

        resume_after_verified_twin_gate(
            args.ledger,
            ProductionWriterLock(args.prod_lock_host, args.prod_lock_path,
                                 f"coverage-pack-resume:{args.pack}"),
            preflight=recovery_preflight, post_twin_gate=recovery_gate,
            apply_prod=recovery_apply_prod,
            verify_mirrors=recovery_verify_mirrors, armed=armed,
            preservation_verifier=verify_preservation_intent)
        print("RESUME-DRY-RUN-PREFLIGHT-OK" if not armed else
              "TWIN-PROD-VERIFIED-OK")
        return

    twin_first(
        args.ledger,
        ProductionWriterLock(args.prod_lock_host, args.prod_lock_path,
                             f"coverage-pack:{args.pack}"),
        preflight=preflight, apply_target=apply_target,
        verify_target=verify_target, armed=armed,
        post_twin_gate=post_twin_gate,
        preservation_verifier=verify_preservation_intent,
        stop_after_twin_gate=args.stop_after_twin_gate)
    if armed and args.gate_profile == "news-daily" and not args.stop_after_twin_gate:
        # Counted only once production is written and both mirrors verified:
        # the ledger says how much has accumulated SINCE the last battery, so
        # a pack that never landed must not age the interval. A row written
        # here can still make the next battery come early (if a later step
        # fails after this point); it can never make one come late.
        # --stop-after-twin-gate never reaches here on a PASS (production is
        # deliberately never touched), so the pack has not landed and must
        # not age the cadence interval either.
        row = news_daily_gate.append_ledger(args.cadence_ledger, {
            "event": "pack", "at": _utcnow().isoformat(),
            "release_sha": runtime_release_identity(args.release_dir)["git_sha"],
            "pack": args.pack, "rows": args.rows,
            "evidence": str(args.ledger)})
        print("CADENCE-ROW " + json.dumps(row, sort_keys=True))
    if armed and args.stop_after_twin_gate:
        print("TWIN-GATE-PASS-STOPPED (task 626: production not touched, "
              "pending a GO file)")
    else:
        print("DRY-RUN-PREFLIGHT-OK" if not armed else "TWIN-PROD-VERIFIED-OK")


if __name__ == "__main__":
    main()
