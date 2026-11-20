"""The recorded-decline refusal, enforced at the embed stage (task 677, S5).

Until this module the recorded consent decisions
(`data/consent_declined_hosts.json`, read by `cytir.consent.block`) were checked
by a handful of ingesters BEFORE they fetched. Nothing looked at them again
between a sealed chunk stage and the GPU, so a stage that reached the embedder
by any other road (an adapter, a re-stage, a hand-built pack) was embedded
whatever its hosts had decided. Task 643 S5 requires "consent refusal enforced
in the embed stage too".

Contract:
  * Input: a stage directory holding `*.texts.jsonl.gz` shards (each row a
    full record with `point_id`, `text`, `url`, ...; see
    `assembly_binder`'s stage description).
  * Every distinct host is checked with `cytir.consent.block.lookup`, the same
    registry and the same rule (`declined` and `unknown` both block) the
    ingesters use. NO NETWORK: this gate reads what was already decided, and
    fails closed if the registry is unreadable (`ConsentRegistryError`).
  * A blocked host raises `ConsentBlocked` naming EVERY blocked host with its
    row count and the shards it sits in, before any model is loaded or any GPU
    lease taken. There is no bypass: un-blocking is a registry PR.
  * A row with no `url` cannot be checked. It is COUNTED and returned in the
    summary and logged loudly, not silently passed. (It is not refused: the
    stage contract says rows carry a url, and whether a url-less row must be
    refused is a policy decision the founder has not made -- recorded in
    task 677.)

This is a refusal of the recorded decision only. It does not re-resolve
consent (a live per-host verdict is `cytir.consent.resolver`'s job, at acquire
time) and it does not touch retention: consent governs fetch and refresh,
never the retention of an already-derived row (ADR-103).
"""
from __future__ import annotations

import glob
import gzip
import json
import os
from collections import Counter, defaultdict
from urllib.parse import urlsplit

from cytir.consent import block

__all__ = ["assert_stage_consent", "assert_records_consent", "stage_hosts"]


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def stage_hosts(stage_dir: str, shards_only: list[str] | None = None
                ) -> tuple[Counter, dict[str, set], int, int]:
    """(rows per host, shards per host, rows without a usable url, total rows).

    `shards_only` names the `*.texts.jsonl.gz` files to read (default: all)."""
    per_host: Counter = Counter()
    shards: dict[str, set] = defaultdict(set)
    no_url = total = 0
    paths = (sorted(os.path.join(stage_dir, s) for s in shards_only)
             if shards_only is not None else
             sorted(glob.glob(os.path.join(stage_dir, "*.texts.jsonl.gz"))))
    if not paths:
        raise FileNotFoundError(
            f"no *.texts.jsonl.gz under {stage_dir!r}: nothing to "
            "consent-check, and an empty stage is not a clean one")
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                total += 1
                host = _host(json.loads(line).get("url") or "")
                if not host:
                    no_url += 1
                    continue
                per_host[host] += 1
                shards[host].add(os.path.basename(path))
    return per_host, shards, no_url, total


def _refuse_blocked(per_host: Counter, shards: dict[str, set], *, where: str,
                    registry: str | None, how_to_remove: str) -> None:
    blocked = []
    for host, rows in sorted(per_host.items()):
        hit = block.lookup(host, registry)
        if hit and hit[1].get("state") in block.BLOCKING_STATES:
            matched, entry = hit
            blocked.append(
                f"{host} rows={rows} shards={sorted(shards.get(host, ()))} "
                f"(registry {matched!r} state={entry.get('state')!r} "
                f"decided {entry.get('decided_on')}; un-block: "
                f"{entry.get('unblock_condition')})")
    if blocked:
        raise block.ConsentBlocked(
            "EMBED REFUSED: " + where + " holds rows "
            "from host(s) with a recorded blocking consent decision:\n  "
            + "\n  ".join(blocked) +
            "\nThis enforces a decision already on record, not a bug; there "
            "is no bypass. " + how_to_remove + ", or un-block the "
            f"host by PR against {block.registry_path()}.")


def assert_records_consent(records, *, registry: str | None = None,
                           label: str = "the input records",
                           log=lambda msg: None) -> dict:
    """The same refusal for records held in memory (`cyt embed index`, a composed family
    pack): every distinct host is checked against the recorded decisions, no network.

    Task 683 (S7). S6 left the generic embed door unenforced ("the generic records carry no
    consent stamp the stage can check"); a record does carry a `url`, and its host is enough
    to consult the registry. Returns `{"rows", "hosts", "rows_without_url"}` when clean."""
    per_host: Counter = Counter()
    no_url = total = 0
    for rec in records:
        total += 1
        host = _host(rec.get("url") or "")
        if not host:
            no_url += 1
            continue
        per_host[host] += 1
    _refuse_blocked(per_host, {}, where=label, registry=registry,
                    how_to_remove="Remove the rows from the input")
    if no_url:
        log(f"[consent] WARNING: {no_url} of {total} records carry no url "
            "and could NOT be consent-checked")
    return {"rows": total, "hosts": len(per_host), "rows_without_url": no_url}


def assert_stage_consent(stage_dir: str, *, registry: str | None = None,
                         shards_only: list[str] | None = None,
                         log=lambda msg: None) -> dict:
    """Refuse a stage that contains a host with a recorded blocking decision.

    Returns `{"rows", "hosts", "rows_without_url"}` when clean.
    """
    per_host, shards, no_url, total = stage_hosts(stage_dir, shards_only)
    _refuse_blocked(per_host, shards, where="the stage at " + repr(stage_dir),
                    registry=registry,
                    how_to_remove="Remove the rows from the stage")
    if no_url:
        log(f"[consent] WARNING: {no_url} of {total} stage rows carry no url "
            "and could NOT be consent-checked")
    return {"rows": total, "hosts": len(per_host), "rows_without_url": no_url}
