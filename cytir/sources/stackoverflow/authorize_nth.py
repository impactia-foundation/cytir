#!/usr/bin/env python3
"""Authorize one exact disjoint Stack Overflow production tranche (generic, any ordinal).

Every ordinal-named predecessor (`authorize_stackoverflow_qa_second_tranche.py`
through `..._seventh_tranche.py`) is the same shape with one more prior tranche
to re-validate each time: the frontier is immutable and deterministically
ranked, so the only things that ever changed per tranche were (a) the offset,
(b) the set of prior tranches whose terminal evidence must be re-validated,
and (c) a hardcoded `OFFSET`/`LIMIT` constant the caller's stated offset was
checked against. That pattern does not scale past the 7 tranches already
authorized -- ~48 more hand-written near-duplicate modules would be needed to
exhaust the 551,326-pair frontier one 10k window at a time.

This module replaces the pattern: it takes any number of prior tranches (as a
list, not N separate named parameter blocks), re-validates every one of them
from its own receipts in a loop (`_validate_terminal_tranche`, reused
unforked from `authorize_stackoverflow_qa_third_tranche`, exactly as every
prior successor already did), and *derives* the offset from the sum of the
priors' own selection-key counts rather than trusting a hardcoded literal.
A caller-stated `--offset` is cross-checked against that derived value and
the run refuses if they disagree -- which is exactly what a gap (a missing or
short prior) or an overlap (a duplicated or mis-ordered prior, or a caller
asking to start before the priors' claimed prefix ends) looks like.

Reproduces every existing tranche script's *behavior* (offsets 10000,
20000, ..., 60000 map to what `..._second_tranche.py` through
`..._seventh_tranche.py` each authorize); see
`tests/test_aw_sources_stackoverflow_authorize_nth.py` for a byte-for-byte
selection-content check of ordinal 7. Ordinal 8+
(offset 70000+) is new -- no hand-written predecessor exists for it, which is
the entire point of this module.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from cytir.acquire.artifact_io import atomic_publish, sha256_file
from cytir.sources.stackoverflow.authorize_second import _smallest, _stream_gzip, _binding
from cytir.sources.stackoverflow.authorize_third import _validate_terminal_tranche
from cytir.index.preservation_contract import _validate_stage
import cytir.acquire.stage_receipt as SS


VERSION = "stackoverflow-qa-bounded-tranche-authorization.v2-generic"
PRODUCER = "pipeline/authorize_stackoverflow_qa_nth_tranche.py"

_EVIDENCE_FIELDS = ("tranche", "parent", "pack", "profile", "fidelity", "stage", "vector")


def _load_keys(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return [json.loads(line)["selection_key"] for line in source if line.strip()]


def build(*, frontier_run: Path, frontier: Path, priors: list[dict[str, Path]],
          ordinal: int, root: Path, artifact_root: Path,
          offset: int | None = None, limit: int = 10_000) -> dict:
    """Authorize the `ordinal`-th 10k-question Stack Overflow tranche.

    `priors` is the ordered list (1st tranche first) of the `ordinal - 1`
    already-authorized tranches this one succeeds. Each entry is a dict with
    keys `tranche, parent, pack, profile, fidelity, stage, vector` -> Path,
    matching exactly the 7-artifact evidence shape every prior tranche script
    required per predecessor (one selection file + the 6-file terminal
    evidence chain `_validate_terminal_tranche` checks).

    `offset`, if given, must equal the offset *derived* from summing the
    priors' own selection-key counts -- a mismatch means a gap (a missing or
    short prior) or an overlap (a duplicated prior, or a caller trying to
    start the new window before the priors' claimed prefix actually ends),
    and the run refuses rather than silently authorizing the wrong slice.
    When omitted, the derived offset is used directly.
    """
    if root.exists():
        raise ValueError("Stack Overflow bounded authorization root must be fresh")
    if ordinal < 2:
        raise ValueError("ordinal must be >= 2 (the first tranche has no priors)")
    if len(priors) != ordinal - 1:
        raise ValueError(
            f"ordinal {ordinal} requires exactly {ordinal - 1} prior tranche(s), "
            f"got {len(priors)}")
    for i, prior in enumerate(priors, start=1):
        missing = [f for f in _EVIDENCE_FIELDS if f not in prior]  # vector may be None
        if missing:
            raise ValueError(f"prior #{i} is missing evidence field(s): {missing}")

    frontier_seal = json.loads(frontier_run.read_text())
    if (frontier_seal.get("schema") != "aw.stackoverflow-qa-production-frontier-run.v1"
            or frontier_seal.get("status") != "exact_frontier_complete_first_tranche_authorized"):
        raise ValueError("terminal Stack Overflow full selection frontier required")
    _validate_stage(frontier_seal["stages"]["frontier"],
                    "stackoverflow.production_frontier",
                    artifact_root=artifact_root, deep=True)

    # Re-validate EVERY prior from its own receipts, in a loop -- not forked
    # per-ordinal. A single bad/incomplete prior anywhere in the chain fails
    # the whole authorization; label names which one broke.
    ordinal_words = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh",
                     "eighth", "ninth", "tenth")
    prior_keys_per: list[list[str]] = []
    evidence_paths_flat: list[Path] = []
    evidence: dict[str, dict] = {}
    for i, prior in enumerate(priors, start=1):
        label = ordinal_words[i - 1] if i <= len(ordinal_words) else f"prior-#{i}"
        vector_path = prior["vector"]  # None: embed pending, see _validate_terminal_tranche
        eval_paths = [prior["parent"], prior["pack"], prior["profile"],
                      prior["fidelity"], prior["stage"]]
        parent_j, pack_j, profile_j, fidelity_j, stage_j = [
            json.loads(p.read_text()) for p in eval_paths]
        vector_j = json.loads(vector_path.read_text()) if vector_path else None
        if vector_path:
            eval_paths.append(vector_path)
        _validate_terminal_tranche(parent=parent_j, pack=pack_j, profile=profile_j,
                                   fidelity=fidelity_j, stage=stage_j, vector=vector_j,
                                   label=label)
        keys = _load_keys(prior["tranche"])
        prior_keys_per.append(keys)
        evidence_paths_flat.append(prior["tranche"])
        evidence_paths_flat.extend(eval_paths)
        for field, path in zip(_EVIDENCE_FIELDS[1:], eval_paths):
            evidence[f"{label}_{field}"] = _binding(path, artifact_root)
        if vector_path is None:
            evidence[f"{label}_vector"] = {
                "absent": "embed pending: sealed by the growth_pending_compose timer, "
                          "not by the authorizer; stage chain validated"}

    prior_keys = [key for group in prior_keys_per for key in group]
    if len(set(prior_keys)) != len(prior_keys):
        raise ValueError(
            "prior Stack Overflow tranches overlap each other (duplicate "
            "selection_key across two or more priors) -- refusing")

    derived_offset = sum(len(group) for group in prior_keys_per)
    if offset is not None and offset != derived_offset:
        raise ValueError(
            f"stated offset {offset} does not match the offset derived from "
            f"{len(priors)} prior tranche receipt(s) ({derived_offset}) -- "
            "this is a gap (a missing/short prior) or an overlap (a "
            "duplicated prior, or a request to start before the priors' "
            "claimed prefix ends); refusing rather than authorizing the "
            "wrong window")
    resolved_offset = derived_offset

    ranked, full_records = _smallest(frontier, resolved_offset + limit)
    if full_records != frontier_seal["stages"]["frontier"]["records"]:
        raise ValueError("Stack Overflow full frontier count drift")
    if prior_keys != [row["selection_key"] for row in ranked[:resolved_offset]]:
        raise ValueError("prior Stack Overflow tranches no longer match rank prefix")
    selected = ranked[resolved_offset:resolved_offset + limit]
    if len(selected) != limit or (set(prior_keys) & {row["selection_key"] for row in selected}):
        raise ValueError(
            f"{ordinal_words[ordinal - 1] if ordinal <= len(ordinal_words) else ordinal}"
            " Stack Overflow tranche is incomplete or overlaps a prior tranche")
    user_ids = {int(value) for row in selected for value in
                (row.get("question_owner_user_id"), row.get("answer_owner_user_id"))
                if value is not None}
    expected_source_rows = 2 * limit + len(user_ids)

    root.mkdir(parents=True)
    end = resolved_offset + limit - 1
    tranche = root / f"frontier/tranche-{resolved_offset:06d}-{end:06d}.jsonl.gz"
    records, logical = _stream_gzip(tranche, selected)
    decision = {
        "schema": "aw.stackoverflow-qa-bounded-tranche-decision.v1",
        "version": VERSION,
        "status": f"exact_tranche_{ordinal:02d}_authorized_only",
        "ordinal": ordinal,
        "classification": "bounded measured continuation; no full-frontier document, chunk, vector, cost, or point extrapolation; Track-2 expansion lane, not the Sep-8 flip",
        "selector": {"ordering": "ascending selection_key then question_id",
                     "offset": resolved_offset, "limit": limit,
                     "full_frontier_records": full_records,
                     "disjoint_from_prior_tranches": True,
                     "selection_logical_sha256": logical},
        "counts": {"questions": records, "unique_user_rows": len(user_ids),
                   "expected_source_rows_question_answer_user": expected_source_rows},
        "prior_tranches_terminal_evidence": evidence,
        "next": (f"materialize only rank interval {resolved_offset}:{resolved_offset + limit} "
                 "with complete official XML rows, attribution, licenses, uncapped Markdown, "
                 "and profile/fidelity/token/UUID gates"),
    }
    decision_path = root / "DECISION.json"
    atomic_publish(decision_path,
                   (json.dumps(decision, indent=2, sort_keys=True) + "\n").encode())
    artifacts = [decision_path, tranche, frontier_run, frontier, *evidence_paths_flat]
    sealed = SS.seal(
        artifact_root=artifact_root, artifact_paths=artifacts,
        receipt_path=root / "stages/frontier/receipt.json", records=records,
        fmt="exact-disjoint-source-native-Q+accepted-A+user-selection-tranche",
        logical_sha256=hashlib.sha256((logical + sha256_file(decision_path)).encode()).hexdigest(),
        producer_id=PRODUCER, producer_version=VERSION)
    run = {
        "schema": "aw.stackoverflow-qa-production-frontier-run.v1",
        "version": VERSION, "status": f"exact_tranche_{ordinal:02d}_authorized_only",
        "decision": decision, "stages": {"frontier": sealed},
    }
    atomic_publish(root / "RUN.json",
                   (json.dumps(run, indent=2, sort_keys=True) + "\n").encode())
    return run


def _cli_priors(args: argparse.Namespace) -> list[dict[str, Path]]:
    lists = {field: getattr(args, f"prior_{field}") or [] for field in _EVIDENCE_FIELDS}
    # `--prior-vector -` = this prior has no vector rollup receipt (embed pending).
    lists["vector"] = [None if str(v) == "-" else v for v in lists["vector"]]
    lengths = {field: len(paths) for field, paths in lists.items()}
    if len(set(lengths.values())) != 1:
        raise SystemExit(f"all --prior-* flags must repeat the same number of times, got {lengths}")
    n = lengths[_EVIDENCE_FIELDS[0]]
    return [{field: (lists[field][i].resolve() if lists[field][i] is not None else None)
             for field in _EVIDENCE_FIELDS} for i in range(n)]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frontier-run", type=Path, required=True)
    p.add_argument("--frontier", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--ordinal", type=int, required=True,
                    help="which tranche this authorizes (8 for the 8th, etc.)")
    p.add_argument("--offset", type=int, default=None,
                    help="sanity cross-check only; derived from prior receipts if omitted")
    p.add_argument("--limit", type=int, default=10_000)
    for field in _EVIDENCE_FIELDS:
        p.add_argument(f"--prior-{field}", type=Path, action="append", default=[],
                        help="repeat once per prior tranche, in tranche order (1st..Nth-1)")
    a = p.parse_args()
    priors = _cli_priors(a)
    print(json.dumps(build(
        frontier_run=a.frontier_run.resolve(), frontier=a.frontier.resolve(),
        priors=priors, ordinal=a.ordinal,
        root=a.root.resolve(), artifact_root=a.artifact_root.resolve(),
        offset=a.offset, limit=a.limit),
        indent=2, sort_keys=True))
