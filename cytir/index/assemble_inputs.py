"""Input loaders for the assembly binder (task 2092, binder slice A1).

Moved verbatim from the binder: the declared stage overrides, extra pack sources,
pack-route exclusions, cross-set collision winners, row-suppression map and the
ADR-103 carried consent flags. These decide which rows the assembly includes; the
only change is that the `data/` files are located with `cytir.core.paths.data_path`
instead of the binder's own `REPO`.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import sys

from cytir.core.paths import data_path


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_pack_stage_overrides() -> dict:
    """Declared stage dirs for pack sources whose receipt names bare filenames.

    `adapt_pack_to_413a_stage.py` v1 wrote `artifact://shard-0000.texts.jsonl.gz`
    -- a filename relative to a stage dir the receipt never records. Resolved
    against the artifact root it points at a file that does not exist, so the
    binder refuses those sources by name rather than guessing (task 478 sections
    2.5 and 6.6).

    This makes the hop DECLARED and checkable instead of guessed. It is not a
    bypass: the caller applies an override only to a receipt that is actually
    bare, and then re-checks that the override covers every shard the receipt
    names and that the receipt still declares the record count the override was
    verified against. An absent file is not an error -- there is simply nothing
    declared, and every bare source stays unresolved.
    """
    path = data_path("assembly", "pack-stage-overrides.v1.json")
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    return {row["source"]: row for row in doc.get("overrides") or []}


def read_jsonl_gz_ids(p) -> set[str]:
    out = set()
    with gzip.open(p, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line[0] == "{":
                d = json.loads(line)
                out.add(str(d.get("point_id") or d.get("id")))
            else:
                out.add(line.strip('"'))
    return out


# ===========================================================================
# ADR-103 -- consent_state / non_refreshable stamps on the carried E-terminal
# rows (task 505 landed the measurement; task 511 §0 makes it reach the
# payload -- task 505 §7 names this exact gap as a phase-B obligation)
# ===========================================================================

def _absent_note(path, effect: str) -> None:
    """The ONE named reason a per-build assembly record degrades to its empty default (task 2285)."""
    print(f"assemble: {path} not found -> {effect}; the file is a per-build record of the master's "
          "assembly and is not shipped (a rebuild declares its own)", file=sys.stderr)


def load_pack_route_extra_sources(path=None) -> list[dict]:
    """Pack sources the build takes from outside the globbed release directory.

    Declared by name in `data/assembly/pack-route-extra-sources.v1.json`, never
    by pattern, with the row count two independent artifacts already agree on.
    Absent file = no extras, which is the behaviour before this existed.
    """
    p = pathlib.Path(path) if path else (
        data_path("assembly", "pack-route-extra-sources.v1.json"))
    if not p.exists():
        _absent_note(p, "no extras")
        return []
    doc = json.loads(p.read_text())
    if doc.get("schema") != "aw.pack-route-extra-sources.v1":
        raise SystemExit(f"{p}: unexpected schema {doc.get('schema')!r}")
    rows = doc.get("sources") or []
    for row in rows:
        for field in ("set_id", "release", "source", "expected_rows", "vectors_uri"):
            if not row.get(field):
                raise SystemExit(f"{p}: an extra source declares no {field}")
    return rows


def load_pack_route_exclusions(path=None) -> dict[str, dict]:
    """Pack sources that are staged but are NOT part of the flip build.

    `_plan_pack_sources` plans every `source=` directory under the release --
    task 478 §2.5's deliberate over-collection, on the understanding that the
    claim manifests' `claimed(S)` term would narrow it. For the pack route it
    never narrows: 122 sources stage under one binder set id, so nothing asks
    whether a source belongs to the flip. Task 513 measured 6,521,047 staged
    rows whose own claim fragments say "NOT flip-lane" and "are NOT in the
    Sep-8 flip's membership set".

    The artifact is derived from those fragments by
    a retired deriver (task 1308, in the repo history), never hand-written; each entry
    carries the fragment and field that excluded it. Absent file = no
    exclusions, which is the behaviour before this existed.
    """
    p = pathlib.Path(path) if path else (
        data_path("assembly", "pack-route-exclusions.v1.json"))
    if not p.exists():
        _absent_note(p, "no exclusions")
        return {}
    doc = json.loads(p.read_text())
    if doc.get("schema") != "aw.pack-route-exclusions.v1":
        raise SystemExit(f"{p}: unexpected schema {doc.get('schema')!r}")
    return {row["source"]: row for row in doc.get("evidence") or []}


def load_collision_winners(path) -> dict[str, str]:
    """Read `{point_id: winner_set_id}` from a cross-set collision winner map.

    Produced by a retired deriver (task 1308, in the repo history) from ratified precedence
    (ADR-101 Ruling 1 / ADR-101b Rulings 2 and 3a / ADR-101c Ruling 2 and its
    equal-rank tie-break). Refuses a row that names no winner rather than
    letting an unadjudicated id fall through to load order.
    """
    out: dict[str, str] = {}
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid, win = row.get("point_id"), row.get("winner_set_id")
            if not pid or not win:
                raise SystemExit(f"{path}:{ln}: row names no point_id/winner_set_id")
            if pid in out and out[pid] != win:
                raise SystemExit(f"{path}:{ln}: {pid} has two different winners")
            out[pid] = win
    return out


def load_row_suppression(path) -> dict[str, set]:
    """Read `{set_id: {point_id, ...}}` from a SET-SCOPED row-suppression map.

    Set-scoped, never a bare id list, and that is the whole point. Task 511 §3
    hit the alternative: a global id union is applied by id with no set scope,
    so where one co-owner forfeits an id and another legitimately keeps its own
    copy, the row vanishes from the collection entirely instead of loading once
    from the keeper -- 91 measured ids that had to be vetoed out of the artifact
    to avoid data loss. A scoped map states "this SET does not load this row",
    which is what every ratified precedence rule actually says.

    Produced by the task 514 one-shot derivation script (ADR-101
    Ruling 1, both directions) and the ADR-101b/c derivation task 511 §3 could
    not apply at stage time. Absent file = no suppression.

    The scope key is `scope`: the bare pack source name for a `pack` partition
    and the binder set id otherwise -- i.e. exactly what `cmd_load` knows about
    the partition it is reading, because 122 pack sources stage under the single
    binder set id `task417_families` and a claim-manifest set id would not
    address them.  `set_id` rides along for provenance.
    """
    out: dict[str, set] = {}
    if not path:
        return out
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid, scope = row.get("point_id"), row.get("scope")
            if not pid or not scope:
                raise SystemExit(
                    f"{path}:{ln}: a row-suppression entry names no "
                    "point_id/scope; an unscoped entry would drop the id from "
                    "every set that stages it")
            out.setdefault(scope, set()).add(pid)
    return out


def _adr103_ruled_states() -> tuple[str, ...]:
    """ADR-103's three ruled values, DERIVED from `card_recipe_adr060.
    CONSENT_STATES` rather than re-declared -- that is the one closed set
    every other writer of this field already validates against (it
    additionally allows "ok", which ADR-103 never mints), and a second,
    separately-typed list here is exactly the fork engineering doctrine
    #1/#6 forbid. Imported inside the function, matching this file's own
    pattern for its other heavy/optional imports (`build_aw_b1`,
    `serving_config`, `qdrant_client` are all function-local for the same
    reason: the pure helpers stay importable without their dependency
    chains -- see `test_assembly_binder_pack_route.py`'s docstring).
    """
    from cytir.corpus.card_recipe import CONSENT_STATES as card_states
    ruled = tuple(s for s in card_states if s != "ok")
    assert ruled == ("unreadable", "declined", "not_applicable"), (
        "card_recipe_adr060.CONSENT_STATES changed shape; ADR-103's mapping "
        "needs a look, not a silent pass-through")
    return ruled


def _map_adr103_consent_state(raw: str, point_id: str,
                              ruled_states: tuple[str, ...]) -> tuple[str, bool]:
    """The flags file's raw `consent_state` -> the value this binder stamps.

    Returns (state, was_folded). ADR-103 rules three values (see
    `_adr103_ruled_states`). Census v34 measured a fourth on the same join,
    `unresolved` (4 of 7,925 -- "no consent result found": the receipt join
    found nothing to read at all, not a verdict). ADR-103's text does not
    reach it, and the closed set every consumer of this field validates
    against does not include it either -- minting a fifth value on this
    build's own initiative is exactly the drift the doctrine forbids. Folded
    into `unreadable`, the existing fail-closed bucket and the conservative
    reading of "we have no read on this origin at all"; the fold is counted
    (`unresolved_folded_into_unreadable`) so it is visible in the receipt
    rather than silent, and is named in the phase-B report for Tim -- 4 of
    38.5M rows, not blocking, but not mine to quietly decide either.
    """
    if raw == "unresolved":
        return "unreadable", True
    if raw not in ruled_states:
        raise SystemExit(
            f"REFUSING: unmapped ADR-103 consent_state {raw!r} on point "
            f"{point_id} -- not in {ruled_states} and not the known "
            f"'unresolved' fold. Refusing rather than defaulting it to "
            f"'declined' (task 505 §2.3's own rule for this join).")
    return raw, False


def _iter_jsonl_maybe_gz(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_adr103_carried_flags(census_path=None):
    """The ADR-103 per-point consent_state / non_refreshable stamp.

    Config-driven from what census v34 itself recorded -- no hand list, no
    count typed into this file. `data/r0-1024-coverage-census.v1.json`'s
    `consent_corrections.carried_declined_rows_adr103` block names both the
    flags file AND the set-by-set row counts it is supposed to cover; both
    sources below are read from paths and hashes THAT block pins, and every
    count is cross-checked against it before this function will return.

    Two sources, because the census's own artifact does not cover the whole
    ruling (task 505 §7 names the gap):

    1. `per_row_flags.uri` -- the task467 entity-card declined-anatomy join,
       7,925 rows, each already carrying its own `consent_state` and
       `non_refreshable: true`.
    2. The 10 `task500_news_1024_in_place` rows the census's `sets` total
       counts but that file does not contain. ADR-103's ruling text is
       explicit for exactly this population ("Page-derived rows [the 10 news
       rows] carry declined"), so they are re-derived from the SAME
       per-point bucket assignment the census cites as its own evidence
       (`basis.bucket_assignment.uri`) -- filtered to rows this set covers
       whose `census_member` is the E cc-consent-declined terminal, which is
       the exact member census v34 used to count them at 10 (task 502 §0 /
       task 505 §2.3).

    Refuses rather than guessing if: the census carries no ADR-103 block; a
    source file's sha256 does not match what the census pinned; a row's
    `consent_state` is not in the recognized set; or the derived per-set (or
    total) count disagrees with what the census's own `sets` dict declares.

    Returns (flags, stats) where `flags` is
    `{point_id: {"consent_state": str, "non_refreshable": True}}` and `stats`
    names the histogram and the fold count for the phase-B report.
    """
    census_path = pathlib.Path(census_path or data_path("r0-1024-coverage-census.v1.json"))
    census = json.loads(census_path.read_text())
    try:
        cc = census["consent_corrections"]["carried_declined_rows_adr103"]
    except KeyError as e:
        raise SystemExit(
            f"REFUSING: {census_path} carries no consent_corrections."
            f"carried_declined_rows_adr103 block (ADR-103) -- the binder "
            f"will not stamp what the census never recorded ({e}).")

    ruled_states = _adr103_ruled_states()
    flags: dict[str, dict] = {}
    per_set_loaded: dict[str, int] = {}
    histogram: dict[str, int] = {}
    n_folded = 0

    # --- source 1: the entity-card declined-anatomy join ------------------
    prf = cc["per_row_flags"]
    prf_path = pathlib.Path(prf["uri"])
    got_sha = sha256_file(prf_path)
    if got_sha != prf["sha256"]:
        raise SystemExit(
            f"REFUSING: {prf_path} sha256 {got_sha} != census-pinned "
            f"{prf['sha256']} -- the flags file drifted since census v34.")
    n1 = 0
    for row in _iter_jsonl_maybe_gz(prf_path):
        raw = row["consent_state"]
        state, folded = _map_adr103_consent_state(raw, row["point_id"],
                                                   ruled_states)
        flags[row["point_id"]] = {"consent_state": state,
                                   "non_refreshable": True}
        histogram[raw] = histogram.get(raw, 0) + 1
        n_folded += int(folded)
        n1 += 1
    if n1 != prf["rows"]:
        raise SystemExit(
            f"REFUSING: {prf_path} holds {n1:,} rows, census pins "
            f"{prf['rows']:,}.")
    # Every row of this file belongs to task467 -- verified below against the
    # census's own per-set total, not assumed because the file has no set_id
    # column of its own.
    per_set_loaded["task467_entity_card_1024_reembed"] = n1

    # --- source 2: rows the census's `sets` total declares but source 1 ---
    # --- does not contain (the 10 task500 news rows) -----------------------
    expected_sets = cc["sets"]
    remaining = {sid: n for sid, n in expected_sets.items()
                 if n > per_set_loaded.get(sid, 0)}
    if remaining:
        ba = census["basis"]["bucket_assignment"]
        ba_path = pathlib.Path(ba["uri"])
        got_sha = sha256_file(ba_path)
        if got_sha != ba["sha256"]:
            raise SystemExit(
                f"REFUSING: {ba_path} sha256 {got_sha} != census-pinned "
                f"{ba['sha256']}.")
        MEMBER_SUFFIX = "consent-declined"
        for row in _iter_jsonl_maybe_gz(ba_path):
            covered_by = row.get("covered_by") or []
            member = row.get("census_member") or ""
            if not member.endswith(MEMBER_SUFFIX):
                continue
            for sid, need in remaining.items():
                if sid not in covered_by:
                    continue
                pid = row["point_id"]
                if pid in flags:
                    continue
                # ADR-103's own text: page-derived rows carry `declined`
                # (they were never joined against a robots/receipt anatomy
                # the way the task467 join was -- there is no unreadable/
                # not_applicable distinction to make for this population).
                flags[pid] = {"consent_state": "declined",
                              "non_refreshable": True}
                histogram["declined"] = histogram.get("declined", 0) + 1
                per_set_loaded[sid] = per_set_loaded.get(sid, 0) + 1

    for sid, expect in expected_sets.items():
        got = per_set_loaded.get(sid, 0)
        if got != expect:
            raise SystemExit(
                f"REFUSING: ADR-103 carried set {sid!r} -- census declares "
                f"{expect:,} rows, this run derived {got:,}.")

    total_expected = sum(expected_sets.values())
    if len(flags) != total_expected:
        raise SystemExit(
            f"REFUSING: ADR-103 flags total {len(flags):,} != census sets "
            f"total {total_expected:,}.")

    stats = {"total": len(flags), "by_set": per_set_loaded,
              "consent_state_histogram_raw": histogram,
              "unresolved_folded_into_unreadable": n_folded}
    return flags, stats
