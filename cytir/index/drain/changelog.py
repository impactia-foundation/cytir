"""Drain change log: the landing records the drain appends to the delta change log.

Task 2875 (RULING 80): moved verbatim out of `cytir/index/promote_queue.py`, which re-exports
every name here.
"""
from __future__ import annotations

import pathlib

from cytir.index import change_log as CL
from cytir.index import url_group_supersede as UGS


def _log_landing(record: dict, plan: dict, *, manifest_sha: str | None = None) -> None:
    """Task 2314: write the pack's change-log segment, BEFORE the DONE marker. A no-op unless the
    `change_log_dir` setting is set. A failure here is loud and propagates: the pack stays without
    DONE and the next pass rewrites the same segment. Wired at the fresh-landing sites ONLY: the
    resume and "already landed" DONE sites use `_log_resumed_landing` (task 2337); the Single path logs nothing."""
    out = CL.configured_dir()
    if out is not None:
        CL.record_landing(dict(record, pack_sha=manifest_sha or ""), plan, out)


def _log_sealed_landing(record: dict, *, manifest_sha: str | None, artifact_root: pathlib.Path,
                        superseded_ids=()) -> None:
    """A sealed pack on the twin-first transaction: the added ids are exactly what the prod leg's own
    written-ids artifact holds; `superseded_ids` are the old prod rows a `supersede: url-group` pack
    deleted (one `tombstone`, reason `superseded`, each). A Single drain logs nothing: its target is
    never the live collection (`assert_single_target_not_live`)."""
    if CL.configured_dir() is None:
        return
    ids = CL.read_written_ids(CL.written_ids_path(artifact_root, record["pack_id"], "prod"))
    _log_landing(record, {"upserts": ids, "op": "add",
                          "tombstones": [{"id": i, "reason": "superseded"} for i in sorted(superseded_ids)]},
                 manifest_sha=manifest_sha)


def _supersede_tombstones_on_resume(manifest: dict, manifest_sha: str | None,
                                    artifact_root: pathlib.Path) -> list[dict]:
    """The `superseded` tombstones of a pack that reached an "already held" exit. The answer is the
    url-group state of THIS manifest: `applied` -> the ids prod deleted (`prod.to_delete`); none ->
    no delete ran, `[]` is true; `deleting` -> unknown (prod may still hold rows the twin lost):
    `ChangeLogError`, with or without the log (task 2337 review 2342, finding 4; an older bug too: a
    pack killed between the prod upsert and the prod delete was marked DONE here)."""
    if not manifest.get("supersede"):
        return []
    from cytir.index import promote_queue as _Q
    try:
        st = UGS.read_state(artifact_root, manifest["pack_id"], manifest_sha)
    except _Q.QueueError as exc:
        raise CL.ChangeLogError(str(exc)) from exc
    if st is None:
        return []
    if st.get("stage") == "applied":
        gone = ((st.get("prod") or {}).get("to_delete")) or ()
        return [{"id": i, "reason": "superseded"} for i in sorted(gone)]
    raise CL.ChangeLogError(
        f"{manifest['pack_id']}: the url-group supersede stopped at stage {st.get('stage')!r}; prod may "
        "still hold old rows the twin already lost, so the pack is not DONE. The drain settles this "
        "state itself (`settle_supersede_deleting`); reaching here means that step did not run. "
        f"state {UGS.state_path(artifact_root, manifest['pack_id'])}")


def _log_resumed_landing(record: dict, ids, *, manifest: dict, manifest_sha: str | None,
                         tombstones=(), op: str = "add") -> None:
    """Task 2337: the segment of a pack that reached DONE by a RESUME or an "already held" exit.
    `ids` are the ids PROD holds after the step (the drain's own written-ids artifact or the
    membership set it already computed; no new Qdrant read). An existing segment is kept only when
    its `pack_sha` is this manifest's (a crash between segment and DONE); one of another manifest
    (a reseal under the same id) is rewritten. `tombstones`: `{id, reason}` the caller proved."""
    out = CL.configured_dir()
    if out is None:
        return
    seg = CL.segment_for(out, record["pack_id"])
    if seg is not None and CL.read_segment(seg)[0].get("pack_sha") == (manifest_sha or ""):
        return
    _log_landing(record, {"upserts": sorted(ids), "op": op,
                          "tombstones": [{"id": t["id"], "reason": t.get("reason")} for t in tombstones]},
                 manifest_sha=manifest_sha)
