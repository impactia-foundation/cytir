"""Payload-profile probe and reconciliation (task 675).

Two verbs on one declaration (`data/collection-payload-profiles.v1.json`):

``roundtrip``  READ-ONLY. Sample ids from the twin (random uuid offsets, limit 1:
               never an unindexed scroll of prod), retrieve the same ids from
               prod and twin by id, and prove that prod's stored row equals
               ``project(twin row)`` and that ``decode(prod row)`` equals the
               twin's raw row. A raw row in prod is counted, not hidden.

``reconcile``  DRY RUN by default. Lists the rows the promotion drain landed in
               prod since a point in time (from the drain receipts and the
               twin-write-state artifacts, never from a scan of prod), re-projects
               them through ``cytir.index.payload_profile`` and reports how many
               would change and why. ``--apply`` writes; it needs the production
               writer lock, reads each row's CURRENT prod payload, overwrites only
               the payload (no vector moves, no point is added or removed), throttles
               between batches and reads the rows back. The twin is not touched:
               its declared profile is `prod` (raw), so the twin-mirror rule holds
               by construction (point counts on both sides are unchanged).

``published``  (task 676) rewrites the `published` of rows NOT in contract form
               (ISO datetime, dd.mm.yyyy, RFC-822: 36,175 rows on 2026-09-30) to
               `YYYY-MM-DD`. Ids come from the twin by exact match on each non-contract
               value of an exhaustive facet (indexed), checked against the facet count;
               the same ids are then rewritten on prod by id (never a scan of prod).
               Dry run by default; `--apply` = twin leg, then prod leg, under the
               production writer lock, with the promotion timer stopped and restarted
               and a prod health guard between batches. Only `published` is set.

Neither pack verb assigns a code. A closed key with an unseen value is reported as a
refusal and ``--apply`` refuses to start while any exist.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import pathlib
import random
import time
import uuid

from cytir.core.stopped_timer import StoppedTimer
from cytir.index import payload_profile as PP
from cytir.index import point_delete as PD
from cytir.index import promote_sealed_pack as PSP
from cytir.index.box_health_guard import BoxHealthGuard, GuardTripped  # noqa: F401 -- re-exported for callers (no in-repo attribute reader found 2026-10-30; kept for the module contract)

SCHEMA = "aw.payload-profile-reconcile.v1"
DEFAULT_PAUSE_S = 0.5
DEFAULT_APPLY_BATCH = 200


# ------------------------------------------------------------------- roundtrip
def sample_ids(url: str, collection: str, n: int, seed: int, *, api=None) -> list[str]:
    """n distinct point ids, each read with one `limit=1` scroll from a random uuid
    offset. Point ids are ordered, so this is an indexed seek, not a scan."""
    api = api or PSP._api
    rng = random.Random(seed)
    out: list[str] = []
    seen: set[str] = set()
    guard = 0
    while len(out) < n and guard < n * 20:
        guard += 1
        offset = str(uuid.UUID(int=rng.getrandbits(128), version=4))
        got = api(url, f"/collections/{collection}/points/scroll",
                  {"limit": 1, "offset": offset, "with_payload": False,
                   "with_vector": False})
        for row in (got or {}).get("points", []):
            pid = str(row["id"])
            if pid not in seen:
                seen.add(pid)
                out.append(pid)
    return out


def compare_row(prod_row: dict, twin_row: dict, projector: "PP.Projector") -> dict:
    """One row: which form prod holds, and whether it equals the twin's row in the
    target's representation.

    `forward_equal` and `decode_equal` are judged on the keys the profile REWRITES
    (the contract under test): forward = `project(twin) == prod` on those keys,
    decode = `decode(prod) == twin` on those keys, with `stamps` absent and the
    digest compared by its 32-hex prefix. `payload_equal` is the whole payload, and
    `other_diff_keys` names any key OUTSIDE the profile that differs (twin and prod
    drift, which the profile neither causes nor can cure).
    """
    form = PP.form_of(prod_row, projector.profile, projector.open_keys)
    want = projector.project(twin_row)
    rules = PP.PROFILE_RULES[projector.profile]
    decoded = PP.decode_payload(prod_row, projector.profile, projector.tables)
    forward = decode_ok = True
    for k, rule in rules.items():
        if want.get(k) != prod_row.get(k):
            forward = False
        if rule == "drop":
            decode_ok &= k not in prod_row
        elif rule == "prefix16":
            b = twin_row.get(k)
            decode_ok &= (isinstance(b, str) and decoded.get(k) == b[:PP.DIGEST_PREFIX_HEX]) \
                or (k not in twin_row and k not in decoded)
        else:
            decode_ok &= decoded.get(k) == twin_row.get(k)
    other = sorted(k for k in set(want) | set(prod_row)
                   if k not in rules and want.get(k) != prod_row.get(k))
    return {"form": form, "forward_equal": forward, "decode_equal": decode_ok,
            "payload_equal": want == prod_row, "other_diff_keys": other}


def roundtrip(prod_url: str, prod_collection: str, twin_url: str,
              twin_collection: str, ids: list[str], *, projector=None) -> dict:
    projector = projector or PP.Projector.for_collection(prod_collection)
    prod = PD.fetch_payload_fields(prod_url, prod_collection, ids, True)
    twin = PD.fetch_payload_fields(twin_url, twin_collection, ids, True)
    forms: collections.Counter = collections.Counter()
    ok: collections.Counter = collections.Counter()
    drift: collections.Counter = collections.Counter()
    bad: list[dict] = []
    codes: dict[str, set] = collections.defaultdict(set)
    missing = {"prod": [], "twin": []}
    for pid in ids:
        if pid not in prod:
            missing["prod"].append(pid)
        if pid not in twin:
            missing["twin"].append(pid)
        if pid not in prod or pid not in twin:
            continue
        res = compare_row(prod[pid], twin[pid], projector)
        forms[res["form"]] += 1
        if res["form"] != "projected":
            continue
        for k, r in PP.PROFILE_RULES[projector.profile].items():
            if r == "enum" and isinstance(prod[pid].get(k), int):
                codes[k].add(prod[pid][k])
        ok["forward"] += res["forward_equal"]
        ok["decode"] += res["decode_equal"]
        ok["payload"] += res["payload_equal"]
        drift.update(res["other_diff_keys"])
        if not (res["forward_equal"] and res["decode_equal"]):
            bad.append({"id": pid, **res})
    n_coded = forms["projected"]
    return {"schema": SCHEMA, "verb": "roundtrip", "prod_collection": prod_collection,
            "twin_collection": twin_collection, "ids": len(ids),
            "missing": {k: len(v) for k, v in missing.items()},
            "prod_form": dict(forms), "coded_rows": n_coded,
            "forward_equal": ok["forward"], "decode_equal": ok["decode"],
            "round_trip_pct": (round(100.0 * min(ok["forward"], ok["decode"]) / n_coded, 4)
                               if n_coded else None),
            "whole_payload_equal": ok["payload"],
            "twin_prod_drift_keys": dict(drift),
            "distinct_codes_seen": {k: len(v) for k, v in sorted(codes.items())},
            "mismatches": bad[:20], "mismatch_count": len(bad)}


# ------------------------------------------------------------------- reconcile
def landed_packs(queue_dir: pathlib.Path, since: str) -> list[dict]:
    """The packs the drain landed in prod at or after `since` (ISO, Z), from each
    pack's own receipt. Roster = a listing of the queue dir, printed with the
    result."""
    out = []
    for d in sorted(p for p in queue_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        rp = d / "receipt.json"
        if not (d / "DONE").exists() or not rp.exists():
            continue
        try:
            r = json.loads(rp.read_text())
        except (OSError, ValueError):
            continue
        if r.get("status") == "LANDED" and (r.get("checked_at") or "") >= since:
            out.append({"pack_id": d.name, "checked_at": r["checked_at"],
                        "dir": str(d), "prod_collection": (r.get("prod") or {}).get("collection")})
    return out


def pack_written_ids(pack: dict, artifact_root: pathlib.Path) -> tuple[list[str], str]:
    """The ids the drain actually wrote for this pack: the content-pack
    twin-write-state's `pack_ids` (word-coverage skips excluded), else the sealed
    path's `<pack>-prod-written-ids.json`. Never the manifest's declared set."""
    pid = pack["pack_id"]
    state = artifact_root / f"{pid}-content-pack-twin-state.json"
    if state.exists():
        rec = json.loads(state.read_text())
        if rec.get("pack_ids") is not None:
            return list(rec["pack_ids"]), "content-pack-twin-state"
    ids_path = artifact_root / f"{pid}-prod-written-ids.json"
    if ids_path.exists():
        return list(json.loads(ids_path.read_text())), "prod-written-ids"
    raise PP.PayloadProfileError(
        f"{pid}: no written-id artifact under {artifact_root}; cannot list its rows")


def _pack_payloads(pack: dict, ids: set[str]) -> dict[str, dict]:
    """The raw payloads of a CONTENT pack's written rows, from the pack file."""
    manifest = json.loads((pathlib.Path(pack["dir"]) / "manifest.json").read_text())
    path = manifest.get("pack_points")
    if manifest.get("kind") != "content_pack" or not path:
        return {}
    opener = gzip.open if path.endswith(".gz") else open
    out: dict[str, dict] = {}
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["id"] in ids:
                out[row["id"]] = row.get("payload") or {}
    return out


def plan_reconcile(queue_dir: pathlib.Path, artifact_root: pathlib.Path, since: str,
                   prod_collection: str, *, prod_url: str | None = None,
                   probe: int = 0, projector=None) -> dict:
    """The dry run. Counts per pack: rows written, rows whose projection differs
    from the raw pack payload (the rows a reconcile would rewrite), rows a closed
    key refuses. With `probe`, that many ids per pack are also read from prod to
    confirm they are stored in the form the plan assumes."""
    projector = projector or PP.Projector.for_collection(prod_collection)
    packs = landed_packs(queue_dir, since)
    report = []
    totals: collections.Counter = collections.Counter()
    for pack in packs:
        ids, ids_source = pack_written_ids(pack, artifact_root)
        payloads = _pack_payloads(pack, set(ids))
        entry = {"pack_id": pack["pack_id"], "landed_at": pack["checked_at"],
                 "ids_source": ids_source, "written_ids": len(ids),
                 "payload_source": "pack-file" if payloads else "prod-read-at-apply"}
        changes = unknown = 0
        by_key: collections.Counter = collections.Counter()
        unknown_keys: collections.Counter = collections.Counter()
        examples = []
        for pid, payload in payloads.items():
            try:
                new = projector.project(payload)
            except PP.UnknownValueError as exc:
                unknown += 1
                unknown_keys[exc.key] += 1
                if len(examples) < 3:
                    examples.append({"id": pid, "key": exc.key})
                continue
            if new != payload:
                changes += 1
                for k in set(payload) | set(new):
                    if payload.get(k) != new.get(k):
                        by_key[k] += 1
        entry.update({"rows_to_rewrite": changes, "rows_refused": unknown,
                      "changed_keys": dict(sorted(by_key.items())),
                      "refused_keys": dict(unknown_keys), "refused_examples": examples,
                      "rows_without_pack_payload": len(ids) - len(payloads)})
        if probe and prod_url and ids:
            step = max(1, len(ids) // probe)
            chosen = ids[::step][:probe]
            stored = PD.fetch_payload_fields(prod_url, prod_collection, chosen, True)
            forms = collections.Counter(
                PP.form_of(stored[i], projector.profile, projector.open_keys)
                if i in stored else "absent"
                for i in chosen)
            # a probed row the plan would leave alone must already be what projecting it gives
            still = sum(1 for i in chosen if i in stored
                        and projector.project(stored[i]) != stored[i])
            entry["prod_probe"] = {"sampled": len(chosen), "forms": dict(forms),
                                   "would_change": still}
        report.append(entry)
        for k in ("written_ids", "rows_to_rewrite", "rows_refused",
                  "rows_without_pack_payload"):
            totals[k] += entry[k]
    return {"schema": SCHEMA, "verb": "reconcile", "mode": "dry-run",
            "queue_dir": str(queue_dir), "since": since,
            "prod_collection": prod_collection, "profile": projector.profile,
            "open_keys": sorted(projector.open_keys),
            "roster": [p["pack_id"] for p in packs], "packs": report,
            "totals": dict(totals)}


def apply_reconcile(report: dict, artifact_root: pathlib.Path, prod_url: str,
                    prod_collection: str, *, pause_s: float = DEFAULT_PAUSE_S,
                    batch: int = DEFAULT_APPLY_BATCH, projector=None,
                    api=None) -> dict:
    """WRITE. Payload-only. Refuses while the plan holds a refused row. The CALLER
    holds the production writer lock (`main`), which keeps this testable."""
    api = api or PSP._api
    projector = projector or PP.Projector.for_collection(prod_collection)
    if report["totals"].get("rows_refused"):
        raise PP.PayloadProfileError(
            "REFUSED: the plan holds %d row(s) a closed key cannot encode; add the "
            "codes by a data PR first" % report["totals"]["rows_refused"])
    if projector.identity:
        raise PP.PayloadProfileError("REFUSED: %s stores the raw form; nothing to do"
                                     % prod_collection)
    count_before = api(prod_url, f"/collections/{prod_collection}")["points_count"]
    rewritten = 0
    for pack in report["packs"]:
        ids, _ = pack_written_ids({"pack_id": pack["pack_id"]}, artifact_root)
        for i in range(0, len(ids), batch):
            chunk = ids[i:i + batch]
            current = PD.fetch_payload_fields(prod_url, prod_collection, chunk, True)
            ops, expect = [], {}
            for pid in chunk:
                if pid not in current:
                    continue  # tombstoned since; not ours to resurrect
                new = projector.project(current[pid])
                if new != current[pid]:
                    ops.append({"overwrite_payload": {"payload": new, "points": [pid]}})
                    expect[pid] = new
            if ops:
                api(prod_url, f"/collections/{prod_collection}/points/batch?wait=true",
                    {"operations": ops}, method="POST")
                back = PD.fetch_payload_fields(prod_url, prod_collection, list(expect), True)
                wrong = [p for p, want in expect.items() if back.get(p) != want]
                if wrong:
                    raise PP.PayloadProfileError(
                        "readback mismatch on %d row(s), e.g. %s" % (len(wrong), wrong[:3]))
                rewritten += len(ops)
                if pause_s:
                    time.sleep(pause_s)
    count_after = api(prod_url, f"/collections/{prod_collection}")["points_count"]
    if count_after != count_before:
        raise PP.PayloadProfileError(
            "points_count moved %d -> %d during a payload-only rewrite" % (count_before, count_after))
    return {"rewritten": rewritten, "points_count": count_after}


# --------------------------------------------------- published backfill (task 676)
#: Format classes (cytir.core.published.classify_published) already in contract form.
#: `missing` has no value to rewrite; `other` names no readable day and is left
#: exactly as written (counted, never dropped).
CONTRACT_PUBLISHED_CLASSES = frozenset({"iso_date", "iso_year_month", "iso_year",
                                        "missing"})
FACET_LIMIT = 1_000_000
VALUE_BATCH = 100
SCROLL_PAGE = 1000
PUBLISHED_SCHEMA = "aw.published-backfill.v1"


def facet_published(url: str, collection: str, *, api=None) -> dict[str, int]:
    """Every distinct `published` value and its row count, exhaustively (an indexed
    keyword facet, exact). The roster of values is read from the store, never typed."""
    api = api or PSP._api
    got = api(url, f"/collections/{collection}/facet",
              {"key": "published", "limit": FACET_LIMIT, "exact": True})
    return {str(h["value"]): int(h["count"]) for h in (got or {}).get("hits", [])}


def classify_facet(facet: dict[str, int]) -> dict:
    """Split the facet into the values a backfill rewrites and the ones it leaves.
    Returns `{"rewrite": {value: (class, rows)}, "classes": {class: {values, rows}},
    "unreadable": {value: rows}}`. A value whose class is not in contract form and
    that `normalize_published` can read is rewritten; one it cannot read is
    `unreadable` (kept as written)."""
    from cytir.core.published import classify_published, normalize_published
    rewrite: dict[str, tuple[str, int]] = {}
    classes: dict[str, dict] = {}
    unreadable: dict[str, int] = {}
    for value, rows in facet.items():
        cls = classify_published(value)
        slot = classes.setdefault(cls, {"values": 0, "rows": 0})
        slot["values"] += 1
        slot["rows"] += rows
        if cls in CONTRACT_PUBLISHED_CLASSES:
            continue
        if normalize_published(value) is None:
            unreadable[value] = rows
        else:
            rewrite[value] = (cls, rows)
    return {"rewrite": rewrite, "classes": classes, "unreadable": unreadable}


def ids_for_published_values(url: str, collection: str, values: list[str], *,
                             api=None) -> dict[str, list[str]]:
    """{value -> ids} for points whose `published` is exactly one of `values`: an
    indexed `match any` scroll per batch of values, ids and `published` only. The
    value a point carries is read back from the point itself, so the caller can
    check the selection against the facet count in both directions."""
    api = api or PSP._api
    out: dict[str, list[str]] = collections.defaultdict(list)
    ordered = sorted(values)
    for i in range(0, len(ordered), VALUE_BATCH):
        chunk = ordered[i:i + VALUE_BATCH]
        offset = None
        while True:
            body = {"filter": {"must": [{"key": "published", "match": {"any": chunk}}]},
                    "limit": SCROLL_PAGE, "with_payload": ["published"],
                    "with_vector": False}
            if offset is not None:
                body["offset"] = offset
            got = api(url, f"/collections/{collection}/points/scroll", body) or {}
            for pt in got.get("points", []):
                out[str((pt.get("payload") or {}).get("published"))].append(str(pt["id"]))
            offset = got.get("next_page_offset")
            if offset is None:
                break
    return dict(out)


def plan_published(url: str, collection: str, *, api=None, fetch=None,
                   verify_sample: int = 300, seed: int = 676) -> dict:
    """The dry run, on ONE store (the twin): which rows carry a `published` that is
    not in contract form, what each becomes, and proof that the selection is exact.

    * selection = exact match on every non-contract value in the facet (indexed);
    * proof 1: for every selected value the ids found == the facet count (the join
      neither missed nor over-matched), else the plan refuses;
    * proof 2: on a random sample of the selected rows the FULL payload is read and
      `project(row)` differs from the row in `published` only (nothing else moves).
    The plan carries `rows` = [{id, before, after}] so the prod leg rewrites the same
    ids without any scan of prod."""
    from cytir.core.published import normalize_published
    api = api or PSP._api
    fetch = fetch or PD.fetch_payload_fields
    facet = facet_published(url, collection, api=api)
    cls = classify_facet(facet)
    found = ids_for_published_values(url, collection, list(cls["rewrite"]), api=api)
    mismatches = []
    for value, (klass, want) in sorted(cls["rewrite"].items()):
        got = len(found.get(value, []))
        if got != want:
            mismatches.append({"value": value, "class": klass, "facet": want, "ids": got})
    stray = sorted(set(found) - set(cls["rewrite"]))
    if mismatches or stray:
        raise PP.PayloadProfileError(
            "REFUSED: selection is not exact -- %d value(s) differ from the facet "
            "(e.g. %s), %d value(s) returned that were not asked for"
            % (len(mismatches), mismatches[:3], len(stray)))
    rows, by_class = [], collections.Counter()
    seen: set[str] = set()
    for value, ids in sorted(found.items()):
        after = normalize_published(value)
        for pid in ids:
            if pid in seen:
                raise PP.PayloadProfileError(f"REFUSED: id {pid} selected under two values")
            seen.add(pid)
            rows.append({"id": pid, "before": value, "after": after})
            by_class[cls["rewrite"][value][0]] += 1
    projector = PP.Projector.for_collection(collection)
    sample_ids_ = sorted(seen)
    random.Random(seed).shuffle(sample_ids_)
    sample_ids_ = sample_ids_[:verify_sample]
    full = fetch(url, collection, sample_ids_, True) if sample_ids_ else {}
    after_of = {r["id"]: r["after"] for r in rows}
    off = []
    for pid, payload in full.items():
        want = dict(payload)
        want["published"] = after_of[pid]
        if projector.project(payload) != want:
            off.append(pid)
    if off:
        raise PP.PayloadProfileError(
            "REFUSED: projecting %d sampled row(s) changes more than `published`, e.g. %s"
            % (len(off), off[:3]))
    import hashlib
    digest = hashlib.sha256("\n".join(sorted(after_of)).encode()).hexdigest()
    return {"schema": PUBLISHED_SCHEMA, "verb": "published", "mode": "plan",
            "collection": collection, "facet_distinct_values": len(facet),
            "facet_rows": sum(facet.values()),
            "classes": cls["classes"], "selected_values": len(cls["rewrite"]),
            "selected_rows": len(rows), "rows_by_class": dict(by_class),
            "expected_rows": sum(n for _, n in cls["rewrite"].values()),
            "unreadable_values": cls["unreadable"],
            "sample_checked": len(full), "sample_only_published_moves": len(full) - len(off),
            "ids_sha256": digest, "rows": rows}


def plan_summary(plan: dict) -> dict:
    return {k: v for k, v in plan.items() if k != "rows"}


def apply_published_leg(rows: list[dict], url: str, collection: str, *,
                        pause_s: float = 1.0, batch: int = DEFAULT_APPLY_BATCH,
                        guard: "BoxHealthGuard | None" = None, api=None, fetch=None,
                        explicit: bool = False) -> dict:
    """WRITE one store. Sets ONLY `published` (Qdrant `set_payload`: every other key,
    every vector and every id stays), reads each batch back, leaves `points_count`
    unchanged. Idempotent: a row already in contract form is counted, not rewritten.
    Each row's CURRENT value is re-read and normalised (the contract function), so a
    row that changed since the plan is handled by what it holds now, not by stale
    `before`/`after` text.

    `explicit=True` (the page-date plan, `published_page_date`): the row's `after` is a
    date read from the page, not derivable from the stored value. It is written only
    where the store still holds the planned `before`; a row that holds `after` is
    `already`; a row holding anything else changed since the plan and is `moved`
    (counted, left alone)."""
    from cytir.core.published import normalize_published
    api = api or PSP._api
    fetch = fetch or PD.fetch_payload_fields
    count_before = api(url, f"/collections/{collection}")["points_count"]
    res = collections.Counter()
    for i in range(0, len(rows), batch):
        if guard:
            guard.tick()
        chunk_rows = rows[i:i + batch]
        ids = [r["id"] for r in chunk_rows]
        planned = {r["id"]: r for r in chunk_rows}
        current = fetch(url, collection, ids, ["published"])
        ops, expect = [], {}
        for pid in ids:
            if pid not in current:
                res["absent"] += 1
                continue
            have = current[pid].get("published")
            if explicit:
                want = planned[pid]["after"]
                if have == want:
                    res["already"] += 1
                elif have == planned[pid]["before"]:
                    ops.append({"payload": {"published": want}, "points": [pid]})
                    expect[pid] = want
                else:
                    res["moved"] += 1
                continue
            want = normalize_published(have)
            if want is None:
                res["unreadable_kept"] += 1
            elif want == have:
                res["already"] += 1
            else:
                ops.append({"payload": {"published": want}, "points": [pid]})
                expect[pid] = want
        if ops:
            PD.set_payload_batch(url, collection, ops, api=api)
            back = fetch(url, collection, list(expect), ["published"])
            wrong = [p for p, want in expect.items()
                     if (back.get(p) or {}).get("published") != want]
            if wrong:
                raise PP.PayloadProfileError(
                    "readback mismatch on %d row(s), e.g. %s" % (len(wrong), wrong[:3]))
            res["rewritten"] += len(ops)
            if pause_s:
                time.sleep(pause_s)
    count_after = api(url, f"/collections/{collection}")["points_count"]
    if count_after != count_before:
        raise PP.PayloadProfileError(
            "points_count moved %d -> %d during a payload-only rewrite"
            % (count_before, count_after))
    accounted = sum(res[k] for k in ("rewritten", "already", "absent", "unreadable_kept",
                                     "moved"))
    if accounted != len(rows):
        raise PP.PayloadProfileError(
            "conservation: %d planned rows, %d accounted (%s)" % (len(rows), accounted, dict(res)))
    return {"planned": len(rows), **{k: res[k] for k in
            ("rewritten", "already", "absent", "unreadable_kept")},
            **({"moved": res["moved"]} if explicit else {}),
            "points_count": count_after}


def verify_published(url: str, collection: str, *, ids: list[str] | None = None,
                     api=None, fetch=None) -> dict:
    """Read-only. With `ids`: each row's current `published` class (a by-id read, safe
    on prod). Without: the exhaustive facet's format classes (an indexed facet)."""
    from cytir.core.published import classify_published
    api = api or PSP._api
    fetch = fetch or PD.fetch_payload_fields
    if ids is not None:
        got = fetch(url, collection, ids, ["published"])
        classes = collections.Counter(
            classify_published((got.get(i) or {}).get("published")) if i in got else "absent"
            for i in ids)
        return {"collection": collection, "sampled": len(ids), "classes": dict(classes),
                "non_contract": sum(v for k, v in classes.items()
                                    if k not in CONTRACT_PUBLISHED_CLASSES | {"absent"})}
    facet = facet_published(url, collection, api=api)
    cls = classify_facet(facet)
    return {"collection": collection, "distinct_values": len(facet),
            "rows_with_published": sum(facet.values()), "classes": cls["classes"],
            "non_contract_rows": sum(n for _, n in cls["rewrite"].values()),
            "unreadable_rows": sum(cls["unreadable"].values())}


def verify_rows(url: str, collection: str, rows: list[dict], *, n: int, seed: int,
                fetch=None) -> dict:
    """Read-only, by id: of `n` planned rows sampled at random, how many hold their
    planned `after` (equal), still hold `before`, or hold something else."""
    fetch = fetch or PD.fetch_payload_fields
    sample = random.Random(seed).sample(rows, min(n, len(rows)))
    got = fetch(url, collection, [r["id"] for r in sample], ["published"])
    out = collections.Counter()
    for r in sample:
        if r["id"] not in got:
            out["absent"] += 1
            continue
        have = got[r["id"]].get("published")
        out["equal_after" if have == r["after"] else "still_before"
            if have == r["before"] else "other"] += 1
    return {"collection": collection, "sampled": len(sample), **dict(out)}


# --------------------------------------------------------------------------- CLI
def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _published_args(p) -> None:
    """The arguments both `published` verbs take."""
    p.add_argument("--twin-url", required=True)
    p.add_argument("--twin-collection", required=True, help="PHYSICAL name")
    p.add_argument("--prod-url", required=True)
    p.add_argument("--prod-alias", default="aw_prod")
    p.add_argument("--plan-out", default="", help="write the full plan (ids, before, after)")
    p.add_argument("--plan-in", default="", help="reuse a plan written earlier")
    p.add_argument("--probe", type=int, default=200,
                   help="ids sampled from prod (by id) before and after")
    p.add_argument("--apply", action="store_true",
                   help="WRITE twin then prod; needs the production writer lock")
    p.add_argument("--pause-s", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=DEFAULT_APPLY_BATCH)
    p.add_argument("--stop-timer", default="",
                   help="systemd unit to stop for the apply and start again at the end")
    p.add_argument("--wait-service", default="",
                   help="with --stop-timer: wait for this service to finish before writing")
    p.add_argument("--wait-s", type=float, default=30000.0,
                   help="longest wait for --wait-service (default 30000 s)")
    p.add_argument("--guard-ssh-host", default="",
                   help="override: ssh host of the prod box the health guard samples "
                        "(default: the operator setting `prod_lock_host`)")
    p.add_argument("--guard-container", default="",
                   help="override: qdrant container whose RestartCount the guard watches "
                        "(default: the operator setting `prod_qdrant_container`)")
    p.add_argument("--guard-search-url", default="",
                   help="override: the one search URL the guard probes between batches "
                        "(default: the operator setting `prod_health_search_url`)")
    p.add_argument("--state-out", default="")


def _write_json(path: str, obj) -> None:
    if path:
        pathlib.Path(path).write_text(json.dumps(obj, indent=1) + "\n")


def _health_guard(a) -> BoxHealthGuard:
    """The between-batches prod health guard. Its targets are operator settings (a flag
    overrides one); no host, container or URL is built in (the old defaults named our box)."""
    from cytir.core.paths import SettingRefused, operator_settings  # noqa: PLC0415
    got, missing = {}, []
    for key, flag, setting in (("ssh_host", "--guard-ssh-host", "prod_lock_host"),
                               ("container", "--guard-container", "prod_qdrant_container"),
                               ("search_url", "--guard-search-url", "prod_health_search_url")):
        try:
            got[key] = getattr(a, flag[2:].replace("-", "_")) or getattr(operator_settings(), setting)
        except SettingRefused:
            got[key] = ""
        if not got[key]:
            missing.append(f"{flag} (or the setting `{setting}`)")
    if missing:
        raise SystemExit("REFUSED: --apply needs " + " and ".join(missing))
    return BoxHealthGuard(**got)


def _main_published(a) -> int:
    if a.verb == "published-from-pages" and not a.work:
        raise SystemExit("REFUSED: published-from-pages needs --work (the news work dir)")
    guard = _health_guard(a) if a.apply else None     # refuse before the plan, not after it
    if a.plan_in:
        plan = json.loads(pathlib.Path(a.plan_in).read_text())
        if plan.get("schema") != PUBLISHED_SCHEMA or plan.get("collection") != a.twin_collection:
            raise SystemExit("REFUSED: --plan-in is not a plan for %s" % a.twin_collection)
    elif a.verb == "published-from-pages":
        import cytir.news.published_page_date as PPD  # noqa: WPS433
        plan = PPD.plan_from_pages(a.work, a.twin_url, a.twin_collection,
                                   limit_urls=a.limit_urls or None)
    else:
        plan = plan_published(a.twin_url, a.twin_collection)
    explicit = plan.get("mode") == "explicit"
    plan["measured_at"] = plan.get("measured_at") or _iso_now()
    _write_json(a.plan_out, plan)
    print(json.dumps(plan_summary(plan), indent=1))
    ids = [r["id"] for r in plan["rows"]]
    rng = random.Random(2)
    probe_ids = rng.sample(ids, min(a.probe, len(ids)))
    from cytir.index import promote_queue as Q  # noqa: WPS433 -- alias resolver and lock constants
    prod_collection = Q.resolve_prod_alias(a.prod_url, a.prod_alias)
    if explicit:
        before = verify_rows(a.prod_url, prod_collection, plan["rows"], n=a.probe, seed=2)
    else:
        before = verify_published(a.prod_url, prod_collection, ids=probe_ids)
    print(json.dumps({"prod_probe_before": before}, indent=1))
    if not a.apply:
        return 0
    from cytir.index import twin_first as TFA  # noqa: WPS433
    state = {"plan_ids_sha256": plan["ids_sha256"], "planned": len(ids), "legs": {}}
    timer = (StoppedTimer(a.stop_timer, wait_unit=a.wait_service, wait_s=a.wait_s)
             if a.stop_timer else None)
    lock = TFA.ProductionWriterLock(host=Q.PROD_LOCK_HOST, lock_path=Q.PROD_LOCK_PATH,
                                    purpose="task676-published-backfill")
    try:
        if timer:
            timer.__enter__()
        lock.acquire()
        Q._revalidate_prod_collection("published-backfill", a.prod_url,
                                      a.prod_alias, prod_collection)
        state["guard_start"] = guard.start()
        for leg, url, coll, g in (("twin", a.twin_url, a.twin_collection, None),
                                  ("prod", a.prod_url, prod_collection, guard)):
            state["legs"][leg] = apply_published_leg(
                plan["rows"], url, coll, pause_s=a.pause_s, batch=a.batch, guard=g,
                explicit=explicit)
            _write_json(a.state_out, state)
            print(json.dumps({leg: state["legs"][leg]}, indent=1), flush=True)
    except BaseException as exc:
        state["stopped"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        state["guard_log"] = guard.log[-50:]
        try:
            lock.release()
        finally:
            if timer:
                timer.__exit__(None, None, None)
            state["timer_restored"] = bool(timer and timer.was_active) or not timer
            _write_json(a.state_out, state)
    if explicit:
        twin_after = verify_rows(a.twin_url, a.twin_collection, plan["rows"],
                                 n=max(a.probe, 500), seed=3)
        prod_after = verify_rows(a.prod_url, prod_collection, plan["rows"], n=a.probe, seed=3)
        bad = (twin_after.get("still_before", 0) + twin_after.get("other", 0)
               + prod_after.get("still_before", 0) + prod_after.get("other", 0))
    else:
        twin_after = verify_published(a.twin_url, a.twin_collection)
        prod_after = verify_published(a.prod_url, prod_collection, ids=probe_ids)
        bad = twin_after["non_contract_rows"] or prod_after["non_contract"]
    print(json.dumps({"twin_after": twin_after, "prod_probe_after": prod_after}, indent=1))
    state.update(twin_after=twin_after, prod_probe_after=prod_after)
    _write_json(a.state_out, state)
    return 1 if bad else 0


def main(argv=None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="verb", required=True)

    rt = sub.add_parser("roundtrip", help="read-only decode probe, prod vs twin")
    rt.add_argument("--prod-url", required=True)
    rt.add_argument("--prod-collection", required=True, help="PHYSICAL name")
    rt.add_argument("--twin-url", required=True)
    rt.add_argument("--twin-collection", required=True)
    rt.add_argument("-n", type=int, default=200)
    rt.add_argument("--seed", type=int, default=675)
    rt.add_argument("--out", default="")

    rc = sub.add_parser("reconcile", help="re-project rows the drain landed (dry run)")
    rc.add_argument("--queue-dir", default="",
                    help="promotion queue (default: the operator setting `queue_dir`)")
    rc.add_argument("--artifact-root", default="")
    rc.add_argument("--since", required=True, help="ISO time, e.g. 2026-09-29T11:39:09Z")
    rc.add_argument("--prod-url", required=True)
    rc.add_argument("--prod-alias", default="aw_prod")
    rc.add_argument("--probe", type=int, default=0,
                    help="also read this many ids per pack from prod (by id)")
    rc.add_argument("--out", default="")
    rc.add_argument("--apply", action="store_true",
                    help="WRITE. Not run by task 675; needs the prod writer lock.")
    rc.add_argument("--pause-s", type=float, default=DEFAULT_PAUSE_S)
    pb = sub.add_parser("published", help="rewrite non-contract `published` values")
    _published_args(pb)
    pp = sub.add_parser("published-from-pages",
                        help="set `published` to the page's own date (from sealed originals)")
    _published_args(pp)
    pp.add_argument("--work", default="",
                    help="news work dir holding stage-*/news_documents.jsonl")
    pp.add_argument("--limit-urls", type=int, default=0, help="smoke: plan only N urls")
    a = ap.parse_args(argv)
    # task 864 (review 869 S1): from an aw_e10 T0 to the flip only the drain writes the twin
    from cytir.index import twin_write_hold as _TWH
    _TWH.refuse_cli_write(a, tool='cyt ops payload-profile')

    if a.verb in ("published", "published-from-pages"):
        return _main_published(a)

    if a.verb == "roundtrip":
        ids = sample_ids(a.twin_url, a.twin_collection, a.n, a.seed)
        res = roundtrip(a.prod_url, a.prod_collection, a.twin_url, a.twin_collection, ids)
        res["measured_at"] = _iso_now()
        res["seed"] = a.seed
        res["tables_sha256"] = PP.file_sha256(PP.TABLES_PATH)
        text = json.dumps(res, indent=1)
        print(text)
        if a.out:
            pathlib.Path(a.out).write_text(text + "\n")
        return 0 if (res["mismatch_count"] == 0 and res["coded_rows"]) else 1

    from cytir.index import promote_queue as Q  # noqa: WPS433 -- the queue's alias resolver and default dir
    queue_dir = pathlib.Path(a.queue_dir or Q.default_queue_dir())
    artifact_root = pathlib.Path(a.artifact_root) if a.artifact_root else queue_dir / "_artifacts"
    prod_collection = Q.resolve_prod_alias(a.prod_url, a.prod_alias)
    report = plan_reconcile(queue_dir, artifact_root, a.since, prod_collection,
                            prod_url=a.prod_url, probe=a.probe)
    report["measured_at"] = _iso_now()
    text = json.dumps(report, indent=1)
    print(text)
    if a.out:
        pathlib.Path(a.out).write_text(text + "\n")
    if not a.apply:
        return 0
    from cytir.index import twin_first as TFA  # noqa: WPS433
    lock = TFA.ProductionWriterLock(host=Q.PROD_LOCK_HOST, lock_path=Q.PROD_LOCK_PATH,
                                    purpose="task675-payload-profile-reconcile")
    lock.acquire()
    try:
        Q._revalidate_prod_collection("payload-profile-reconcile", a.prod_url,
                                      a.prod_alias, prod_collection)
        res = apply_reconcile(report, artifact_root, a.prod_url, prod_collection,
                              pause_s=a.pause_s)
    finally:
        lock.release()
    print(json.dumps({"applied": res}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(prog="python -m cytir.index.payload_profile_reconcile"))
