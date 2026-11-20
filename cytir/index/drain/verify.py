"""Drain verify: written-form readback, twin==prod reasons, prod-alias resolution and re-validation.

Task 2897 (RULING 80, drain split steps 7-9): moved verbatim out of `cytir/index/promote_queue.py`,
which re-exports every name here. A moved body that calls a name a test patches on the facade (or a
name that stays there) reads it at call time through `_Q` (the facade module).
"""
from __future__ import annotations

import json
import pathlib

from cytir.index import mirror_compare as MC
from cytir.index import payload_profile as PP
from cytir.index import point_delete as PD
from cytir.index import promote_sealed_pack as PSP
from cytir.index import sparse_rule as SR
from cytir.index import twin_first as TFA
from cytir.index.drain.common import QueueError


def _already_landed_overlap_note(exc, prod_missing, never_written) -> str:
    """The part of a no-artifact resume refusal that names the ids which
    actually triggered it (task 690). Pure; the refusal itself is unchanged.

    `plan()` raises `AlreadyLandedError` when ANY id is already on twin. When
    those ids are also on prod and every other id is on neither target, the
    pack is not a half-landed resume: an earlier, different pack landed a few
    of the same ids on both targets, and the rest are new. Measured live,
    2026-10-02: `eric-official-pdf-tranche-130000` shared 1 id of 169,070
    with `eric-official-pdf-smoke-tranche-130000-first-000004` (landed
    2026-09-28), and the old text only said "169069 of 169069 id(s) ... on
    NEITHER twin nor prod" -- it named the ids it was not about."""
    present = sorted(getattr(exc, "present", ()) or ())
    if not present:
        return ""
    on_prod = sorted(set(present) - set(prod_missing))
    note = (f" TRIGGER: twin already holds {len(present)} of "
            f"{len(exc.ids)} id(s) (e.g. {present[:3]}), {len(on_prod)} of "
            "them on prod too.")
    if len(on_prod) == len(present) and len(never_written) == len(prod_missing):
        note += (" This is a PARTIAL OVERLAP with an earlier pack, not a "
                 "resume: those ids already landed on both targets and every "
                 "other id is on neither. Recovery: exclude exactly those ids "
                 "(manifest `exclude_ids_file`, `python -m cytir.index.reseal "
                 "--exclude-ids-file ...`) so the rest takes the normal "
                 "twin-first write. UNLESS this very pack already landed and "
                 "its twin leg's gate dropped exactly those 'neither' ids: then "
                 "its `<pack>-twin-written-ids.json` is missing from this "
                 "artifact root (a replay on a queue copy must carry it) and "
                 "there is nothing to do (task 788).")
    return note


def landed_with_twin_drops(exc_ids, prod_missing, twin_ids_path: pathlib.Path) -> dict | None:
    """The real state of a LANDED sealed pack whose twin leg dropped rows (task 788; task 790's replay).

    `plan()` re-projects the whole pack, so a pack whose twin leg's gate dropped N rows (mechanical-
    content problems, prose/binary hold-outs, already-served urls) shows N ids "missing from prod" for
    ever. With the pack's own twin-written-ids artifact that is decidable: when the projected ids prod
    holds are EXACTLY the twin-written ids (none missing, none of the dropped ones present), the pack
    has landed and there is nothing to do. Returns `{"dropped_by_twin_decision", "artifact", "note"}`,
    or None for every other shape (no artifact; prod lacks a twin-written id, a real resume; prod holds
    a dropped id, a divergence the resume path refuses). Pure but for one read."""
    if not prod_missing or not twin_ids_path.exists():
        return None
    twin_ids = set(json.loads(twin_ids_path.read_text()))
    missing = set(prod_missing)
    # Landed means prod holds EXACTLY the twin's set of this pack's projection: no twin id missing,
    # and no projected id the twin leg dropped present on prod (review 797 M1: a dropped id on prod is
    # a divergence, never "nothing to do").
    if missing & twin_ids or set(exc_ids) - missing != twin_ids:
        return None
    return {"dropped_by_twin_decision": len(missing), "artifact": str(twin_ids_path),
            "note": (f"pack landed; {len(missing)} ids were dropped by the twin leg's gate "
                     f"(see {twin_ids_path.name}); nothing to do")}


def twin_set_matches_prod_pack_ids(exc_ids, prod_missing, twin_ids_path: pathlib.Path) -> bool:
    """May the OLDER "twin and prod already hold every id" exit mark a pack DONE? (task 808; review 797 S4)

    That exit fires when `prod_missing` is empty: prod holds EVERY projected id. It never asked whether
    the twin leg dropped some of them. With the pack's twin-written-ids artifact it is decidable and
    mirrors #720's rule in `landed_with_twin_drops`: the pack ids on prod must EQUAL the twin's written
    set (prod holds every twin id AND no projected id the twin dropped). Incident, 2026-10-09: tranche
    14, twin = pack minus 213 dropped ids, prod = the whole pack -> the old exit would have closed DONE
    over 213 rows the twin never held. False falls through to the resume path, which refuses with
    `ResumeCollisionError` (nothing written).

    NO artifact -> True, today's behaviour: there is no twin truth to compare against (task-664
    incident-2 shape, another pack_id's apply wrote the twin; ruling T14 is open). Pure but for one read."""
    if not twin_ids_path.exists():
        return True
    twin_ids = set(json.loads(twin_ids_path.read_text()))
    return set(exc_ids) - set(prod_missing) == twin_ids


def outside_twin_set_reason(exc_ids, prod_missing, twin_ids_path: pathlib.Path) -> str:
    """The refusal text for `twin_set_matches_prod_pack_ids` == False (task 808; review 811 S1)."""
    twin_ids = set(json.loads(twin_ids_path.read_text()))
    outside = sorted((set(exc_ids) - set(prod_missing)) - twin_ids)
    return (f"prod holds {len(outside)} id(s) of this pack that the twin leg did not write "
            f"(e.g. {outside[:3]}; twin-written set: {len(twin_ids)} ids, {twin_ids_path.name}). "
            "This is a divergence, not a landed pack: nothing was written. Reconcile by hand "
            "(task-664 style: compare prod to the twin, purge or mirror the extra ids on both "
            "targets), then drain again.")




#: The written-form readback (task 675; widened by task 833, guard G2). Rows read back per leg:
#: the in-place rows FIRST, then up to `VERIFY_FORM_SAMPLE` evenly spaced others, never more than
#: `VERIFY_FORM_MAX` in all -- so the readback is ONE id retrieve per side.
VERIFY_FORM_SAMPLE = 200
VERIFY_FORM_MAX = 500


def _readback_sample(points: list[dict], inplace_ids, *, sample: int,
                     cap: int = VERIFY_FORM_MAX) -> tuple[list[dict], int]:
    """(chosen rows, how many of them are in-place). In-place rows (ids the target held BEFORE this
    write: the rows a write OVERWRITES) come first: an even spacing picks one only by chance, and
    they are the class a stale mirror hides in (task 830: 6,247 of them, 0.02 % of prod). More
    in-place rows than `cap`: evenly spaced over them. Then even spacing over the rest."""
    inplace = set(inplace_ids or ())
    old = [p for p in points if p["id"] in inplace]
    new = [p for p in points if p["id"] not in inplace]
    if len(old) > cap:
        old = [old[i * len(old) // cap] for i in range(cap)]
    room = min(sample, cap - len(old))
    picked_new = new[::max(1, len(new) // room)][:room] if (new and room > 0) else []
    return old + picked_new, len(old)


def verify_written_form(url: str, collection: str, points: list[dict], *,
                        sample: int = VERIFY_FORM_SAMPLE, inplace_ids=(),
                        twin: tuple[str, str] | None = None) -> dict:
    """Read back the rows just written and compare them with what was meant to land, BY CONTENT.

    Task 675 compared, on an evenly spaced sample, only the keys the target's profile rewrites.
    Task 833 (guard G2 of task 831) widens it:
    * the sample takes the `inplace_ids` FIRST (the rows this write overwrote on the target; see
      `_readback_sample`), at most `VERIFY_FORM_MAX` rows = one retrieve per side;
    * EVERY payload key and the dense vector are compared (`mirror_compare`), the pack row put in
      the target's form by the writer's own steps (BM25 epoch stamp, then projection);
    * `twin` = (twin url, twin collection), given for the PROD leg: the same ids are read on the
      twin and twin vs prod is compared too, so the receipt says "twin == prod by content on N
      rows, M of them in-place", not only "prod == pack".
    A mismatch raises `QueueError`: the leg FAILS before DONE, as a membership failure does. The
    raw twin (identity profile) keeps its own leg's check at `checked: 0`: G2 is the prod leg.
    """
    projector = PP.Projector.for_collection(collection)
    if projector.identity or not points:
        return {"collection": collection, "profile": projector.profile, "checked": 0,
                "mismatches": 0}
    chosen, n_inplace = _readback_sample(points, inplace_ids, sample=sample)
    ids = [p["id"] for p in chosen]
    stored = PD.fetch_rows(url, collection, ids, batch=VERIFY_FORM_MAX)
    cmp = MC.Comparator(collection, projector=projector)
    vs_pack = cmp.compare(ids, {p["id"]: p for p in chosen}, stored)
    vs_twin = None
    if twin is not None:
        twin_url, twin_collection = twin
        vs_twin = cmp.compare(ids, PD.fetch_rows(twin_url, twin_collection, ids,
                                                 batch=VERIFY_FORM_MAX), stored)
    problems = []
    if vs_pack["mismatched"]:
        problems.append(
            f"{vs_pack['mismatched']} of {len(ids)} sampled written row(s) are not in the declared "
            f"{projector.profile!r} form of the pack row ({n_inplace} of the sample in-place), "
            f"first {len(vs_pack['examples'])}: {vs_pack['examples']}")
    if vs_twin is not None and vs_twin["mismatched"]:
        problems.append(
            f"{vs_twin['mismatched']} of {len(ids)} sampled row(s) differ by content between twin "
            f"{twin[1]!r} and {collection!r}, first {len(vs_twin['examples'])}: "
            f"{vs_twin['examples']}")
    if problems:
        raise QueueError(f"{collection}: " + "; ".join(problems)
                         + " -- payload-profile/mirror content verify failed")
    out = {"collection": collection, "profile": projector.profile,
           "checked": len(chosen), "mismatches": 0, "compared": "all payload keys + dense",
           "inplace_checked": n_inplace, "inplace_total": len(set(inplace_ids or ())),
           "twin_checked": vs_twin["checked"] if vs_twin is not None else 0}
    if collection in SR.LAST_READBACK:     # the rule's own readback of what the collection stores
        out["sparse_rule_readback"] = SR.LAST_READBACK[collection]
    return out




#: Task 833 (guard G1 of task 831). `finalize_carried_by_flip` used to compare one field,
#: `content_digest` (`_row_content_signature`, removed here), and fell back to a whole-payload
#: signature only for a row without one. The comment that stood here said an l1 row "can be
#: legitimately digest-less". That was FALSE: the projection sets `content_digest` on every row it
#: makes (`cytir.core.projection.project_record`), and every L1, hn and news row carries one (task
#: 831 section 1.3: 100 % of four packs), so the fallback never ran for them. And for a Tier-B row
#: the point id IS `uuid5(canonical_url + "#" + content_digest)` (`cytir.core.ids.classify_tier`,
#: `mint_point_id`): "the same digest under a present id" is "the same id", a membership check. It
#: passed 2,317 of 2,317 stale rows of `l1-backfill-batch-00001`. The finalize now compares every
#: payload key and the dense vector, twin vs prod (`cytir.index.mirror_compare`). The digest is one
#: key of that compare: it carries information only where the id is NOT made from it (Tier A: a
#: source-native per-chunk anchor; Tier C: an entity card, id from the url only; and the few
#: Tier-B rows whose id predates I1 or uses another url spelling).


def resolve_prod_alias(url: str, alias_name: str = "aw_prod") -> str:
    """Resolve a qdrant alias to its CURRENT physical collection name
    (task 673 review round 2, finding 3): after the `aw_e9` flip, prod
    serves through the `aw_prod` alias, not the `aw_e8` name a pack's own
    manifest declares -- checking a stale manifest-declared collection
    after a flip would silently check nothing, or the wrong retired
    collection. Qdrant's alias-listing endpoint, verified live (the build box
    `:6334` tunnel, 2026-09-29): the raw HTTP body of `GET /aliases` is
    `{"result": {"aliases": [{"alias_name": "aw_prod", "collection_name":
    "aw_e9"}]}, "status": "ok"}` (NOT `/collections/aliases`, which qdrant
    reads as a collection literally named "aliases" and 404s).

    `PSP._api` already UNWRAPS `result` (it returns `payload.get("result")`),
    so what arrives here is `{"aliases": [...]}` -- reading `result` again
    found no alias ever and refused every drain and finalize right after
    the aw_e9 flip (2026-09-29, live)."""
    result = PSP._api(url, "/aliases")
    for entry in (result or {}).get("aliases", []):
        if entry.get("alias_name") == alias_name:
            return entry["collection_name"]
    raise QueueError(
        f"alias {alias_name!r} not found on {url} (GET /aliases) -- "
        "cannot resolve prod's live collection")


def _revalidate_prod_collection(pack_id: str, prod_url: str, prod_alias: str,
                                expected_collection: str) -> None:
    """Re-resolve the `aw_prod` alias and require it still names
    `expected_collection` (task 673 review round 4, finding 2).

    `drain()` resolves the alias once per run (round 3 finding 6), but an
    hourly drain can run for hours and a flip can repoint `aw_prod` to a
    different physical collection mid-run -- writing to the run-start
    collection after that would silently write to one prod no longer
    serves. Every prod-facing write function (`live_apply`,
    `live_apply_content_pack`, `resume_prod_only`,
    `resume_content_pack_prod_only`) calls this immediately before its own
    prod write, i.e. AFTER the production writer lock is already held
    (`apply_target`/`apply_prod` run only after `lock.acquire()`), so no
    other writer can move the alias out from under an in-flight write
    without this catching it on the very next pack.

    Raises `twin_first_apply.ApplyAbort` -- the same exception type an
    outright prod outage raises here (a down tunnel makes the re-resolve
    itself fail) -- so `drain()`'s existing mid-write `except BaseException`
    handling marks the pack ABORTED and stops the drain exactly as it
    already does for any other prod-write failure, loudly, via the
    exception message it prints and persists to the pack's receipt."""
    from cytir.index import promote_queue as _Q  # noqa: PLC0415 -- patched names resolve at call time
    current = _Q.resolve_prod_alias(prod_url, prod_alias)
    if current != expected_collection:
        raise TFA.ApplyAbort(
            f"{pack_id}: prod alias {prod_alias!r} moved mid-drain -- was "
            f"{expected_collection!r} earlier this run, now {current!r}. "
            "ABORTED: refusing to write to a collection prod may no "
            "longer serve.")


def _read_ledger_twin_counts(artifact_root: pathlib.Path, pack_id: str
                             ) -> tuple[int, int] | None:
    """The pack's own recorded twin-leg `written`/`tombstoned` counts, read
    from its apply ledger (task 673 review round 4, finding 1).

    `finalize_carried_by_flip`'s twin-membership-reconstruction fallback
    (no twin-write-state artifact -- the real `l1-backfill-batch-00001`
    case) derives "written" from pack ids CURRENTLY present on twin and
    "deleted" from tombstones CURRENTLY absent from twin, then verifies
    that reconstruction against PROD -- which, after the very flip this
    status exists for, IS a copy of twin. That check alone is circular: a
    twin leg that was only PARTLY applied would reconstruct a smaller
    "written" set (only the ids that actually landed) and a correspondingly
    smaller "deleted" set, both of which trivially match a post-flip prod
    that mirrors that same partial twin -- finalizing a half-landed pack as
    DONE with no way to notice.

    The ledger `twin_first()` appends at write time (`TWIN-VERIFIED`,
    written before any resume, flip, or later reconstruction could touch
    it) records what the twin leg ACTUALLY wrote/deleted, independent of
    anything read back afterwards -- a fixed point the reconstruction can
    be checked against. Verified live against the real ledger:
    `l1-backfill-batch-00001-ledger.jsonl`'s `TWIN-VERIFIED` entry reads
    `"written": 218208, "tombstoned": 131285`, matching the pack's own
    receipt exactly (task 673's investigation).

    Returns `None` when the ledger file is missing, unreadable/corrupt, or
    has no `TWIN-VERIFIED` entry for the twin role -- callers must refuse
    to finalize in that case, never proceed as if it matched."""
    path = artifact_root / f"{pack_id}-ledger.jsonl"
    if not path.exists():
        return None
    written: int | None = None
    tombstoned: int | None = None
    try:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("status") != "TWIN-VERIFIED":
                continue
            proof = entry.get("proof")
            if not isinstance(proof, dict) or proof.get("role") != "twin":
                continue
            if "written" not in proof or "tombstoned" not in proof:
                continue
            # Last matching entry wins -- a re-run/replay appends a fresh
            # TWIN-VERIFIED record; the most recent one reflects what the
            # twin leg holds now.
            written, tombstoned = proof["written"], proof["tombstoned"]
    except (OSError, json.JSONDecodeError):
        return None
    if written is None or tombstoned is None:
        return None
    return written, tombstoned
