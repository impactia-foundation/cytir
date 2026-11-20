#!/usr/bin/env python3
"""Pure validation for a content-pack resume predecessor ledger.

Moved from task 351's `content_pack_lineage.py` (task 646 S4, part of
the "351 applier + lineage + noop" group -- `cytir/index/apply.py` imports this
module directly as `cytir.index.lineage`). A shim at the old path re-exports it
unchanged for any caller not yet updated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class LineageError(RuntimeError):
    pass


PLAN = "CONTENT-PACK-PLAN"
ARMED = "PREFLIGHT-OK"
PRESERVED = "PRESERVATION-VERIFIED"
LOCKED = "LOCKED-PREFLIGHT-OK"
TWIN = "TWIN-VERIFIED"
FAILED = "FAILED"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def quiet_green(state: dict) -> bool:
    return (state.get("status") == "green" and
            state.get("optimizer_status") == "ok" and
            state.get("update_queue_length") == 0 and
            isinstance(state.get("points_count"), int))


def load_failed_twin_lineage(
        path: Path, expected_sha256: str, *, pack: str,
        artifact_sha256: str, staged_verification_sha256: str,
        rows: int, twin_target: dict | None = None,
        prod_target: dict | None = None) -> dict:
    """Return the bound plan/proof for one terminal gate-only failure.

    The predecessor may contain an earlier dry preflight, but its final armed
    transaction must be exactly plan, armed preflight, verified twin, failed
    gate.  A production or gate-pass event can therefore never be mistaken for
    a resumable state.
    """
    if not path.is_file() or digest(path) != expected_sha256:
        raise LineageError("failed predecessor ledger path/hash mismatch")
    try:
        events = [json.loads(line) for line in path.read_text().splitlines()
                  if line.strip()]
    except (json.JSONDecodeError, OSError) as exc:
        raise LineageError(f"failed predecessor ledger is unreadable: {exc}") from exc
    if len(events) < 4:
        raise LineageError("failed predecessor ledger is incomplete")
    statuses = [item.get("status") for item in events]
    # `twin_first` emits PRESERVATION-VERIFIED between the armed preflight and
    # the writer lock whenever a preservation verifier runs -- which, under
    # ADR-098, is EVERY tier-2 content-pack apply. This validator predates that
    # event, so without it here no tier-2 transaction could ever be recognised
    # as resumable (task 452 hit exactly that). Admitting the event is pure
    # vocabulary: the required topology is unchanged, and every check below
    # still runs against the same four events.
    tails = [
        [PLAN, ARMED, PRESERVED, LOCKED, TWIN, FAILED],
        [PLAN, ARMED, LOCKED, TWIN, FAILED],
        [PLAN, ARMED, PRESERVED, TWIN, FAILED],
        [PLAN, ARMED, TWIN, FAILED],
    ]
    matched = next((tail for tail in tails
                    if statuses[-len(tail):] == tail), None)
    if matched is None:
        raise LineageError("predecessor is not a terminal verified-twin gate failure")
    window = events[-len(matched):]
    by_status = {item.get("status"): item for item in window}
    plan_event = by_status[PLAN]
    armed_event = by_status[ARMED]
    twin_event = by_status[TWIN]
    failed_event = by_status[FAILED]
    if armed_event.get("armed") is not True:
        raise LineageError("predecessor is not a terminal verified-twin gate failure")
    # Task385 composes a pack-local held-out acceptance into the same
    # post-twin callback as ADR-043.  Both failures have the identical safe
    # topology (verified twin, untouched prod); admit that explicit failure
    # class without relaxing lineage to arbitrary errors such as a prod write
    # or network failure.
    error = str(failed_event.get("error", ""))
    if not any(marker in error for marker in (
            "ADR-043 content gate failed",
            "held-out Marc Maret acceptance failed")):
        raise LineageError("predecessor failure was not the post-twin content gate")
    if (plan_event.get("pack") != pack or
            plan_event.get("artifact_sha256") != artifact_sha256 or
            plan_event.get("staged_verification_sha256") != staged_verification_sha256):
        raise LineageError("predecessor plan artifact identity mismatch")
    plans = plan_event.get("proof")
    if not isinstance(plans, dict) or set(plans) != {"twin", "prod"}:
        raise LineageError("predecessor plan does not bind both mirrors")
    expected_targets = {"twin": twin_target, "prod": prod_target}
    observed_targets = {role: plans[role].get("target")
                        for role in ("twin", "prod")}
    # Pre-target-binding ledgers are safe to resume only through the legacy
    # one-name contract.  A split-name resume must never infer where either
    # side of an old transaction landed.
    unbound_legacy = (
        observed_targets == {"twin": None, "prod": None} and
        twin_target is not None and prod_target is not None and
        twin_target.get("collection") == prod_target.get("collection"))
    for role, expected_target in expected_targets.items():
        observed_target = observed_targets[role]
        if (expected_target is not None and observed_target != expected_target
                and not unbound_legacy):
            raise LineageError(
                f"predecessor {role} target binding mismatch")
        if observed_target is not None and (
                not isinstance(observed_target, dict) or
                observed_target.get("role") != role or
                not observed_target.get("url") or
                not observed_target.get("collection")):
            raise LineageError(
                f"predecessor {role} target binding is invalid")
    twin_plan = {key: value for key, value in plans["twin"].items()
                 if key != "target"}
    prod_plan = {key: value for key, value in plans["prod"].items()
                 if key != "target"}
    if twin_plan != prod_plan:
        raise LineageError("predecessor starting mirror plans differ")
    plan = twin_plan
    before = plan.get("before", {})
    existing = plan.get("existing_ids")
    new = plan.get("new_ids")
    if (not quiet_green(before) or plan.get("rows") != rows or
            not isinstance(existing, int) or not 0 <= existing <= rows or
            new != rows - existing or
            not isinstance(plan.get("superseded"), int) or
            plan.get("superseded") < 0 or
            plan.get("expected_delta") != new - plan["superseded"]):
        raise LineageError("predecessor plan counts/state are inconsistent")
    twin = twin_event.get("proof", {})
    if (twin_target is not None and twin.get("target") != twin_target and
            not unbound_legacy):
        raise LineageError("predecessor twin proof target binding mismatch")
    after = twin.get("after", {})
    if (twin.get("artifact_sha256") != artifact_sha256 or
            twin.get("rows_present") != rows or twin.get("before") != before or
            twin.get("delta") != plan["expected_delta"] or
            twin.get("superseded") != plan["superseded"] or
            not quiet_green(after) or
            after["points_count"] != before["points_count"] + plan["expected_delta"]):
        raise LineageError("predecessor twin proof is inconsistent")
    return {"failed_ledger": str(path),
            "failed_ledger_sha256": expected_sha256,
            "plan": plan, "plans": plans, "twin": twin}
