"""Reseal a queued pack: record why, mint a new manifest sha, re-arm READY.

WHY THIS EXISTS (task 686, 2026-10-01). `promote_queue.py` caches a REFUSED
pack at a cacheable stage (`plan`, `content_pack_check`) for as long as its
manifest sha is unchanged: only a changed seal retries it. The ten OpenAlex
packs refused with `UnpersistedInputs` (ADR-105) because their markdown was
never filed into L1. After the L1 filing (`cytir/corpus/register_l1_release.py
--recursive`) the pack content is the same, so nothing else would change the
manifest, and the drain would skip the packs forever.

This writes `manifest["resealed"]` -- the reason, the time and the L1
releases that now cover the pack -- so the new sha carries the record of why
it changed (corpus-doctrine: provenance travels with the row). Everything
else in the manifest and the pack's vectors are left exactly as sealed. A
`PARKED`/`DONE`/`SUPERSEDED` pack is refused: those are terminal and must
not be re-armed.

    python -m cytir.index.reseal --queue-dir Q --pack-id P \\
        --reason "L1 filed" --l1-release task686-openalex-x [--l1-release ...]

EXCLUDING A PARTIAL OVERLAP (task 690, 2026-10-02). `--exclude-ids-file F
--exclude-ids-reason R` also pins a task-680 exclusion file into the manifest
(`exclude_ids_file` / `exclude_ids_sha256` / `exclude_ids_reason`, the sha
computed here from the bytes on disk). It exists for a pack that shares a few
point ids with an earlier pack that already landed them on twin AND prod:
`plan()` refuses any already-present id, so the drain refuses the whole pack
until those ids are excluded. The file is validated with the same loader the
drain uses, so a file the drain would refuse is refused here first. A manifest
that already names an exclusion file is refused: merging two id lists is a
decision, not a reseal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from cytir.index import exclude_file as XF

_TERMINAL = ("DONE", "SUPERSEDED", "PARKED")


class ResealError(Exception):
    pass


def reseal_pack(queue_dir: str, pack_id: str, *, reason: str,
                l1_releases: list[str] | None = None,
                now: str | None = None,
                exclude_ids_file: str | None = None,
                exclude_ids_reason: str | None = None) -> dict:
    pack = Path(queue_dir) / pack_id
    manifest_path = pack / "manifest.json"
    if not manifest_path.is_file() or not (pack / "READY").is_file():
        raise ResealError(f"{pack_id}: not a sealed, queued pack")
    done = [m for m in _TERMINAL if (pack / m).exists()]
    if done:
        raise ResealError(f"{pack_id}: terminal marker {done[0]} present")
    if not reason:
        raise ResealError("a reseal needs a reason")
    old_bytes = manifest_path.read_bytes()
    old_sha = hashlib.sha256(old_bytes).hexdigest()
    ready_sha = (pack / "READY").read_text().strip().split()[0].lower()
    if ready_sha != old_sha:
        raise ResealError(f"{pack_id}: READY {ready_sha[:12]} != manifest "
                          f"{old_sha[:12]}; the pack is already inconsistent")
    manifest = json.loads(old_bytes)
    excluded = None
    if exclude_ids_file:
        if manifest.get("exclude_ids_file"):
            raise ResealError(
                f"{pack_id}: manifest already names exclude_ids_file "
                f"{manifest['exclude_ids_file']}; merge the lists by hand")
        path = str(Path(exclude_ids_file).resolve())
        sha = XF.sha256_file(path)
        try:
            ids = XF.load_exclude_ids(path, sha256=sha, reason=exclude_ids_reason)
        except XF.ExcludeFileError as exc:
            raise ResealError(f"{pack_id}: {exc}") from exc
        excluded = len(ids)
        if "rows_to_write_max" in manifest:
            # a pack composed with the two row names keeps `rows_to_write_max` true; a legacy pack has
            # only `rows` (the alias, left as sealed) and is not touched. No exclusion existed
            # before (refused above), so the stage count less the new ids that are in it.
            from cytir.index import promote_sealed_pack as PSP
            try:
                inside = sum(1 for r in PSP.read_records(manifest["pack_chunks"])
                             if str(r["point_id"]) in ids)
            except Exception as exc:  # noqa: BLE001 -- a count we cannot make is a refusal
                raise ResealError(f"{pack_id}: cannot count the excluded ids in the pack's "
                                  f"chunks ({type(exc).__name__}: {exc})") from exc
            manifest["rows_to_write_max"] = manifest["rows_to_write_max"] - inside
        manifest["exclude_ids_file"] = path
        manifest["exclude_ids_sha256"] = sha
        manifest["exclude_ids_reason"] = exclude_ids_reason
    elif exclude_ids_reason:
        raise ResealError("--exclude-ids-reason needs --exclude-ids-file")
    manifest["resealed"] = {
        "reason": reason,
        "at": now or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "previous_manifest_sha": old_sha,
        "l1_releases": sorted(l1_releases or []),
    }
    if excluded is not None:
        manifest["resealed"]["exclude_ids_added"] = excluded
    new_bytes = json.dumps(manifest, indent=1, sort_keys=True).encode()
    new_sha = hashlib.sha256(new_bytes).hexdigest()
    manifest_path.write_bytes(new_bytes)
    (pack / "READY").write_text(new_sha + "\n")
    # Retire a cached REFUSED receipt (task 688). `cytir.ops.growth_pending_compose`
    # treats a REFUSED receipt older than the compose code's last commit as
    # "stale" and RECOMPOSES the pack, which overwrites this reseal with the
    # original manifest. With no receipt the pack is settled and the drain
    # tries it fresh. The old receipt is kept beside it, not deleted.
    receipt = pack / "receipt.json"
    if receipt.is_file():
        try:
            prior = json.loads(receipt.read_text())
        except (OSError, ValueError):
            prior = {}
        if prior.get("status") == "REFUSED":
            stamp = str(prior.get("checked_at") or "unknown").replace(":", "")
            receipt.rename(pack / f"receipt.refused-{stamp}.json")
    out = {"pack_id": pack_id, "old_sha": old_sha, "new_sha": new_sha}
    if excluded is not None:
        out["exclude_ids_added"] = excluded
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--queue-dir", required=True)
    ap.add_argument("--pack-id", action="append", required=True)
    ap.add_argument("--reason", required=True)
    ap.add_argument("--l1-release", action="append", default=[])
    ap.add_argument("--exclude-ids-file")
    ap.add_argument("--exclude-ids-reason")
    args = ap.parse_args(argv)
    for pid in args.pack_id:
        print(json.dumps(reseal_pack(
            args.queue_dir, pid, reason=args.reason,
            l1_releases=args.l1_release,
            exclude_ids_file=args.exclude_ids_file,
            exclude_ids_reason=args.exclude_ids_reason)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
