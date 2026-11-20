"""Exclude, at COMPOSE, the ids the twin and prod already serve (Tim's ruling 23.5, ADR-116, task 702).

WHY
---
A pack that repeats ids both targets already hold is refused by the drain ("partial overlap with
an earlier pack, not a resume") and an operator reseals it with `--exclude-ids-file`. That is a
mechanical fix a human was doing by hand. It is done here, once, where the pack is made.

WHAT
----
`declare_already_served` asks, by id (indexed batched retrieves, never a scan):

  1. the TWIN, for all the pack's ids, with a short payload projection (`PAYLOAD_COMPARE_FIELDS`);
  2. PROD, only about the ids the twin holds. A net-new pack asks prod nothing; tranche 6 asks it
     one question of 203 ids instead of 287 of 143,493.

The ids both hold are written to `already-served-ids.jsonl` in the pack directory, each row with
its OWN reason (`SERVED_REASON`, ADR-116's code; an id a caller's own exclusion file already named
keeps that file's reason), and the manifest fields come back:

    already_served_excluded         count of ids declared (not counting ids a caller already excluded)
    already_served_excluded_file / _sha256
    already_served_checked_against  the targets asked, with their per-target hit counts
    already_served_payload_differs  how many excluded ids hold a DIFFERENT payload on the twin
    already_served_payload_differs_file / _sha256 / _fields   the sidecar and the per-field counts
    exclude_ids_file / _sha256 / _reason   the SAME keys a `--exclude-ids-file` reseal writes,
                                    so the drain needs no new path
    excluded_ids_rechecked_at_drain false: the drain does not re-ask; an id deleted between
                                    compose and drain still reads "excluded" in the record

"ALREADY SERVED" IS NOT "SAME CONTENT" (review H1, an open question for Tim)
---------------------------------------------------------------------------
An id is minted from url, anchor and chunk text. A shared id means the same TEXT, not the same
payload. In tranche 6 all 203 excluded ids are old `aw_b1` rows whose title, source_type, trust_tier
differ from the pack's rows. Excluding them is what ADR-116 says, so nothing changes in WHAT is
excluded; the difference is MEASURED and recorded instead (the sidecar is the input for whatever
rule Tim picks: leave, reconcile at a ship+flip, supersede). `content_digest` is not compared: the
raw pack record's digest differs from the stored one on all 203 only because the two encodings
differ (base32 vs sha256 hex). The digest of the drain's projection of the pack row equals the
stored digest on 203 of 203, so the TEXT is the same and the digest says nothing about the payload.

THE RESUME GUARD (a resume must stay a resume)
----------------------------------------------
Three things keep a half-applied pack from being shrunk under its own resume:

  1. only ids on BOTH targets are excluded. A pack this drain half-applied has its ids on the
     twin and not (yet) on prod, so they are never declared;
  2. a drain artifact for this pack id (the sealed pack's `{pack_id}-twin-written-ids.json`, or the
     content pack's twin-state file) means the drain has begun this pack: nothing is excluded and
     the fact is recorded;
  3. a pack directory that already holds a manifest with an already-served declaration keeps that
     declaration byte for byte (a recompose must not change the manifest sha the resume is keyed
     on).

A pack whose ids are ALL already served composes to nothing. It never enters the promotion queue
as a pack (a directory with no manifest is a refusal the drain writes every hour, for ever): the
record `<queue>/_composed-to-nothing/<pack_id>.json` (+ its ids file and sha256) is kept where the
drain does not look (it skips `_`-prefixed entries) and the 15-minute compose timer reads it as
settled.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping

from cytir.index import exclude_file as XF
from cytir.index import row_decode as RD

SERVED_FILE = "already-served-ids.jsonl"
DIFFERS_FILE = "already-served-payload-differs.jsonl"
SETTLED_DIR = "_composed-to-nothing"
SETTLED_SCHEMA = "aw.pack-composed-to-nothing.v1"
#: ADR-116 section 2: the reason every excluded id ends with.
SERVED_REASON = "already_on_twin_and_prod"
#: The payload fields compared between the pack's row and the twin's stored row for an id both
#: targets hold. One named place. `content_digest` is NOT here: the raw pack record's digest and
#: the stored one differ only by encoding (base32 vs sha256 hex); the digest of the drain's projection
#: equals the stored one (203 of 203 in tranche 6), so it says nothing about the payload.
PAYLOAD_COMPARE_FIELDS = ("title", "source_type", "trust_tier", "tier", "lang", "license",
                          "published")


class ServedExclusionError(RuntimeError):
    """The check could not be answered; the pack must not be composed on a guess."""


_CARRIED_KEYS = (
    "already_served_excluded", "already_served_excluded_file", "already_served_excluded_sha256",
    "already_served_checked_against", "already_served_check",
    "already_served_payload_differs", "already_served_payload_differs_file",
    "already_served_payload_differs_sha256", "already_served_payload_differs_fields",
    "already_served_payload_compared_fields", "already_served_payload_unchecked",
    "excluded_ids_rechecked_at_drain",
    "exclude_ids_file", "exclude_ids_sha256", "exclude_ids_reason", "rows", "rows_before_exclusion",
    "rows_staged", "rows_to_write_max")


def read_declared(pack_dir: Path) -> dict | None:
    """The already-served fields of a previously composed manifest, or None."""
    m = Path(pack_dir) / "manifest.json"
    if not m.is_file():
        return None
    old = json.loads(m.read_text())
    if "already_served_excluded" not in old:
        return None
    return {k: old[k] for k in _CARRIED_KEYS if k in old}


def settled_path(queue_dir: str | Path, pack_id: str) -> Path:
    """The record of a pack that composed to nothing. `<queue_dir>/_composed-to-nothing/` is skipped
    by the drain (it ignores `_`-prefixed entries), so the pack never reaches it."""
    return Path(queue_dir) / SETTLED_DIR / f"{pack_id}.json"


def is_settled(queue_dir: str | Path, pack_id: str) -> bool:
    return settled_path(queue_dir, pack_id).is_file()


def _write_rows(path: Path, rows: Iterable[dict]) -> str:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return XF.sha256_file(path)


def compare_payloads(pack: Mapping, stored: Mapping) -> list[str]:
    """The names of `PAYLOAD_COMPARE_FIELDS` on which the pack's row and the stored row differ.
    An absent, None or empty value reads the same on both sides. Both rows are compared as
    CANONICAL rows (`row_decode`, task 855: `published` in its contract form on both sides), and a
    field the stored row's profile does not store (`tier` on the coded twin) is not compared: its
    absence says nothing about the payload."""
    def norm(v):
        return None if v in (None, "", [], {}) else v
    pack, stored = RD.as_canonical_row(pack), RD.as_canonical_row(stored)
    return [f for f in PAYLOAD_COMPARE_FIELDS
            if f not in stored.absent and norm(pack.get(f)) != norm(stored.get(f))]


def declare_already_served(*, pack_dir: str | Path, pack_id: str, ids: list[str],
                           targets: dict[str, tuple[str, str]],
                           twin_rows: Callable[[list[str]], Mapping[str, Mapping]],
                           prod_present: Callable[[list[str]], Iterable[str]],
                           pack_payloads: Callable[[list[str]], Mapping[str, Mapping]],
                           resume_artifact_exists: bool = False,
                           explicit_exclude: tuple[str, str] | None = None,
                           projection_rejected: Mapping[str, str] | None = None) -> dict:
    """Manifest fields for the compose-time already-served exclusion.

    `targets`: `name -> (url, collection)`, exactly `twin` and `prod` (recorded, not used to ask).
    `twin_rows(ids)`: the twin's stored `PAYLOAD_COMPARE_FIELDS` per id it holds (an id retrieve);
    its keys are the twin's membership answer. `prod_present(ids)`: the ids prod holds, asked ONLY
    of the twin's hits. `pack_payloads(ids)`: the payload the drain would write for each given id
    (an id it cannot project is absent). `explicit_exclude`: `(path, reason)` of a file the caller
    already declared; its ids stay excluded, each with its own reason, in the union file.
    `projection_rejected` (task 2220): `declared point_id -> reason` of pack rows the drain's
    projection refuses (no minted id, no row, but a vector); they join the same union file by
    DECLARED id so `plan()` subtracts their vectors, and `fields["projection_rejected"]` counts
    them per reason. Empty or None: nothing is added to the fields or the file.
    Returns `{"fields", "all_served", "carried_over", "excluded_ids", ...}`."""
    pack_dir = Path(pack_dir)
    if set(targets) != {"twin", "prod"}:
        raise ServedExclusionError(f"{pack_id}: targets must be twin and prod, got {sorted(targets)}")
    prior = read_declared(pack_dir)
    if prior is not None:
        return {"fields": prior, "all_served": False, "carried_over": True, "excluded_ids": []}
    explicit_ids: dict[str, str] = {}
    if explicit_exclude:
        explicit_ids = XF.load_exclude_ids(
            explicit_exclude[0], sha256=XF.sha256_file(explicit_exclude[0]),
            reason=explicit_exclude[1])
    rejected = {str(i): why for i, why in (projection_rejected or {}).items()}
    for i, why in rejected.items():
        explicit_ids.setdefault(i, why)       # a caller's own reason for the id wins
    if resume_artifact_exists:
        return {"fields": {"already_served_excluded": 0,
                           "already_served_check": "skipped: a drain artifact exists for this pack "
                                                   "(the drain has begun it; resume guard)"},
                "all_served": False, "carried_over": False, "excluded_ids": [],
                # the caller's own file still leaves `rows` on this path
                "explicit_in_pack": sorted(set(explicit_ids) & set(ids))}

    def ask(name, fn, arg):
        url, collection = targets[name]
        try:
            return fn(arg)
        except Exception as exc:  # noqa: BLE001 -- an unanswered check is a refusal, never "0 served"
            raise ServedExclusionError(
                f"{pack_id}: cannot ask {name} ({url} {collection}) which ids it holds: "
                f"{type(exc).__name__}: {exc}") from exc

    twin = dict(ask("twin", twin_rows, ids))
    twin_hits = sorted(twin)
    prod_hits = set(ask("prod", prod_present, twin_hits)) if twin_hits else set()
    both = prod_hits & set(twin_hits)
    checked = {"twin": {"url": targets["twin"][0], "collection": targets["twin"][1],
                        "ids_asked": len(ids), "ids_present": len(twin_hits)},
               "prod": {"url": targets["prod"][0], "collection": targets["prod"][1],
                        "ids_asked": len(twin_hits), "ids_present": len(both)},
               "twin_only_not_excluded": len(twin_hits) - len(both)}
    new = sorted(both - set(explicit_ids))
    all_served = bool(ids) and bool(new) and set(ids) <= (both | set(explicit_ids))
    fields: dict = {"already_served_excluded": len(new),
                    "already_served_checked_against": checked,
                    "excluded_ids_rechecked_at_drain": False}
    if rejected:
        by_reason: dict[str, int] = {}
        for why in rejected.values():
            by_reason[why] = by_reason.get(why, 0) + 1
        fields["projection_rejected"] = dict(sorted(by_reason.items()))
    if not new and not rejected:
        return {"fields": fields, "all_served": False, "carried_over": False, "excluded_ids": [],
                "explicit_in_pack": sorted(set(explicit_ids) & set(ids))}

    mine = pack_payloads(new)
    differs, unchecked, by_field = [], 0, {}
    for pid in new:
        if pid not in mine:
            unchecked += 1
            continue
        names = compare_payloads(mine[pid], twin[pid])
        if names:
            differs.append({"point_id": pid, "fields": names})
            for n in names:
                by_field[n] = by_field.get(n, 0) + 1
    fields.update({"already_served_payload_differs": len(differs),
                   "already_served_payload_differs_fields": dict(sorted(by_field.items())),
                   "already_served_payload_compared_fields": list(PAYLOAD_COMPARE_FIELDS),
                   "already_served_payload_unchecked": unchecked})
    if all_served:
        # a pack that composes to nothing keeps its files beside the settled record, not here
        return {"fields": fields, "all_served": True, "carried_over": False, "excluded_ids": new,
                "payload_differs": differs}
    pack_dir.mkdir(parents=True, exist_ok=True)
    rows = ([{"point_id": i, "reason": SERVED_REASON} for i in new]
            + [{"point_id": i, "reason": why} for i, why in sorted(explicit_ids.items())])
    sha = _write_rows(pack_dir / SERVED_FILE, rows)
    fields.update({"already_served_excluded_file": SERVED_FILE,
                   "already_served_excluded_sha256": sha,
                   "exclude_ids_file": str(pack_dir / SERVED_FILE),
                   "exclude_ids_sha256": sha,
                   "exclude_ids_reason": (SERVED_REASON if not explicit_ids else
                                          f"per id, see each row's `reason` ({SERVED_REASON}; "
                                          + "; ".join(sorted({explicit_exclude[1]} if explicit_exclude
                                                             else set()) + sorted(set(rejected.values())))
                                          + ")")})
    if differs:
        fields["already_served_payload_differs_file"] = DIFFERS_FILE
        fields["already_served_payload_differs_sha256"] = _write_rows(pack_dir / DIFFERS_FILE,
                                                                      differs)
    return {"fields": fields, "all_served": False, "carried_over": False, "excluded_ids": new,
            "explicit_in_pack": sorted(set(explicit_ids) & set(ids))}


def write_composed_to_nothing(queue_dir: str | Path, *, pack_id: str, rows: int, fields: dict,
                              excluded_ids: list[str], payload_differs: list[dict],
                              family: str, pack_chunks: str, vectors: str,
                              explicit_exclude_file: str | None = None,
                              projection_rejected: Mapping[str, str] | None = None) -> Path:
    """The terminal record of a pack whose every id is already served. The ids file and the
    payload-differs sidecar are written beside it with their sha256, then the record LAST, so its
    existence means the other files are whole. `projection_rejected` (task 2222): `declared id ->
    reason` of the rows the drain's projection refuses; the record lists the counts per reason and a
    `<pack>.projection-rejected.jsonl` sidecar (with sha256) lists the ids."""
    d = Path(queue_dir) / SETTLED_DIR
    d.mkdir(parents=True, exist_ok=True)
    ids_file, differs_file = f"{pack_id}.ids.jsonl", f"{pack_id}.payload-differs.jsonl"
    # what an audit needs to find a pack whose content was "already served" at compose time, and to
    # compose it again without a search (review of #572): the family, the stage and its vectors,
    # the stage's chunk count, when, and the caller's own exclusion file (path + sha256) if any
    record = {"schema": SETTLED_SCHEMA, "pack_id": pack_id, "rows": rows,
              "state": "composed-to-nothing", "reason": SERVED_REASON, **fields,
              "family": family, "pack_chunks": pack_chunks, "vectors": vectors,
              "composed_at": datetime.datetime.now(datetime.timezone.utc).strftime(
                  "%Y-%m-%dT%H:%M:%SZ"),
              **({"explicit_exclude_file": str(explicit_exclude_file),
                  "explicit_exclude_sha256": XF.sha256_file(explicit_exclude_file)}
                 if explicit_exclude_file else {}),
              "excluded_ids_file": ids_file,
              "excluded_ids_sha256": _write_rows(
                  d / ids_file, ({"point_id": i, "reason": SERVED_REASON} for i in excluded_ids))}
    if projection_rejected:
        rej_file = f"{pack_id}.projection-rejected.jsonl"
        by_reason: dict[str, int] = {}
        for why in projection_rejected.values():
            by_reason[why] = by_reason.get(why, 0) + 1
        record["projection_rejected"] = dict(sorted(by_reason.items()))
        record["projection_rejected_file"] = rej_file
        record["projection_rejected_sha256"] = _write_rows(
            d / rej_file, ({"point_id": i, "reason": why}
                           for i, why in sorted(projection_rejected.items())))
    if payload_differs:
        record["payload_differs_file"] = differs_file
        record["payload_differs_sha256"] = _write_rows(d / differs_file, payload_differs)
    path = settled_path(queue_dir, pack_id)
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return path
