#!/usr/bin/env python3
"""The ADR-101 assembly-guard suite -- the flip gate for the 1024-dim build.

ADR-101 Ruling 2 makes eight checks a precondition of the index flip: the
builder MUST run them before loading a single vector, and the flip is
prohibited while any fails.  This module is that gate.

Design, and why it is shaped this way
-------------------------------------
The gate never touches the corpus.  It consumes an **assembly plan** -- the
builder's own declaration of what it intends to load, delete, and carry -- and
checks that declaration against evidence produced independently:

  * ``data/r0-427-collision-ledger.v1.json`` (task 456), sha-pinned.  Measured
    by walking real point IDs on the corpus.
  * ``data/r0-1024-coverage-census.v1.json`` (task 453), sha-pinned.
  * the claim-manifest index and the per-guard resolution records emitted by
    ``cyt ops produce-assembly-artifacts``, a second, independent walk of
    the same corpus.

The plan is not derived from the ledger.  If it were, every reconciliation
check would be a tautology -- a filter selecting on what it reports.  The
builder declares; the ledger and census attest; the guard refuses when they
disagree.

Three statuses, not two
-----------------------
``FAIL`` means the population is terminal and the check is broken -- a defect.
``WAITING`` means the population is not terminal yet: a lane named in
``pending_lanes`` still has to seal before the check can even be evaluated.
Distinguishing them is what turns this suite into the burn-down instrument for
the ~Sep-8 assembly: WAITING counts go to zero as lanes seal, and only then can
the gate be green.  ``PASS`` is neither.

Fail-closed
-----------
A missing input, an unreadable artifact, a sha mismatch, an unrecognised field,
or an exception inside a guard is a ``FAIL``.  Nothing is skipped, and no guard
may be waived from inside this file -- gate waivers are Tim's, by PR.

Exit codes: 0 = every guard PASS (flip permitted).  1 = at least one FAIL.
2 = no FAIL, at least one WAITING (burn-down in progress; flip prohibited).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

SCHEMA_REPORT = "aw.assembly-guard-report.v1"
SCHEMA_PLAN = "aw.assembly-plan.v1"
SCHEMA_MANIFESTS = "aw.assembly-claim-manifests.v1"

PASS = "PASS"
FAIL = "FAIL"
WAITING = "WAITING"
_RANK = {PASS: 0, WAITING: 1, FAIL: 2}

TASK427 = "task427_postconsent"

#: A set whose rows were never membership points.  ``disposition`` is an
#: identity over the membership, so these rows fall into none of its buckets
#: and the plan carries an ``additions`` term for them instead.  The evidence
#: burden is not reduced, it is moved: G1 wants the enumerated row IDs it would
#: otherwise want of the claim, G3 wants the plan's term to agree with the
#: manifests and to state the resulting collection size, G5 wants
#: rows == minted IDs == nothing claimed, and G7 wants a witness proving the
#: minted IDs are absent from the membership.  Only G8 skips them, by name.
ADDITION_MODE = "net_new_addition"

#: The witness an addition publishes in place of a replacement point witness.
SCHEMA_ADDITION_WITNESS = "aw.net-new-addition-census-witness.v1"

#: ADR-101 Ruling 1 as a table over the competing set's chunk class.  The plan
#: must declare exactly this table; a plan that applies a different precedence
#: is not the ratified one.
ADR101_PRECEDENCE = {
    "exact_epoch4": "this_set",
    "reembed_epoch3_in_place": "this_set",
    "named_slice_payload_rule": "this_set",
    "atomic_epoch1": TASK427,
    "adr092_remainder": TASK427,
    "tombstone": "nobody",
}

#: The schema of a precedence amendment layered on top of the sha-pinned
#: task-456 collision ledger.  The ledger is a measurement and is never edited
#: to carry policy; rulings for pairs it never measured arrive in their own
#: pinned file, and are consumable only through the ratification gate below.
SCHEMA_PRECEDENCE_SUPPLEMENT = "aw.collision-ledger-precedence-supplement.v1"

#: Appliers that lose or duplicate points when a new ID collides with an ID in
#: the same build's delete set (ADR-101 Ruling 2, guard G7).
FORBIDDEN_APPLY_ORDERINGS = {"delete_then_insert", "insert_then_delete"}

VALID_UNIT_TIEBREAKS = {"majority", "tie-break:sealed-set-wins"}

#: The winner class the ledger's owner map assigns to an ADR-091 policy
#: tombstone.  A URL that mixes one of these with a loaded replacement row is
#: an unruled policy edge, not a defect -- see guard G4's escalation check.
TOMBSTONE_WINNER = "nobody"


class GuardInputError(Exception):
    """A guard input is missing, unreadable, or does not match its digest."""


# ---------------------------------------------------------------------------


@dataclass
class GuardResult:
    id: str
    name: str
    checks: list[dict] = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    remedy: str | None = None
    evidence: list[str] = field(default_factory=list)

    def check(self, name: str, status: str, detail: str,
              **counts: Any) -> None:
        self.checks.append({"name": name, "status": status, "detail": detail,
                            **({"counts": counts} if counts else {})})
        if status == FAIL:
            self.findings.append(f"{name}: {detail}")

    def ok(self, name: str, detail: str, **counts: Any) -> None:
        self.check(name, PASS, detail, **counts)

    def fail(self, name: str, detail: str, **counts: Any) -> None:
        self.check(name, FAIL, detail, **counts)

    def wait(self, name: str, detail: str, **counts: Any) -> None:
        self.check(name, WAITING, detail, **counts)

    @property
    def status(self) -> str:
        if not self.checks:
            return FAIL
        return max((row["status"] for row in self.checks), key=_RANK.__getitem__)

    def as_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "status": self.status,
            "checks": self.checks, "counts": self.counts,
            "findings": self.findings, "remedy": self.remedy,
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pinned(root: Path, binding: dict, label: str) -> dict:
    """Read a JSON input and refuse it unless its sha256 matches the pin."""
    if not isinstance(binding, dict) or "uri" not in binding:
        raise GuardInputError(f"{label}: binding must carry a uri")
    path = root / str(binding["uri"]).removeprefix("artifact://")
    if not path.is_file():
        raise GuardInputError(f"{label}: missing input {path}")
    expected = binding.get("sha256")
    if not expected:
        raise GuardInputError(f"{label}: binding must pin a sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise GuardInputError(
            f"{label}: digest drift {actual} != {expected} ({path})")
    if path.suffix == ".gz":
        # A gzipped id LIST is a legitimate plan record (the ADR-101b/c row
        # suppression, the cross-set collision winners). It is not a JSON
        # document, and parsing it as one made the whole guard suite fail
        # `inputs-loadable` with a UnicodeDecodeError on the gzip magic byte --
        # eight FAILs whose real cause was a file extension. The digest pin is
        # the part that matters here; the guards that need the CONTENT of one
        # of these read it themselves, with the reader the format needs.
        return {"uri": str(binding["uri"]), "sha256": actual,
                "bytes": path.stat().st_size, "binary": True,
                "role": binding.get("role")}
    return json.loads(path.read_text())


class Context:
    """Everything the guards read, loaded and digest-checked exactly once."""

    def __init__(self, root: Path, plan: dict):
        self.root = root
        self.plan = plan
        if plan.get("schema") != SCHEMA_PLAN:
            raise GuardInputError(
                f"plan schema must be {SCHEMA_PLAN}, got {plan.get('schema')!r}")
        self.ledger = load_pinned(root, plan["inputs"]["collision_ledger"],
                                  "collision_ledger")
        self.census = load_pinned(root, plan["inputs"]["coverage_census"],
                                  "coverage_census")
        self.manifests = load_pinned(root, plan["inputs"]["manifest_index"],
                                     "manifest_index")
        if self.manifests.get("schema") != SCHEMA_MANIFESTS:
            raise GuardInputError("manifest index has the wrong schema")
        self.sets = {row["set_id"]: row for row in self.manifests["sets"]}
        if len(self.sets) != len(self.manifests["sets"]):
            raise GuardInputError("duplicate set_id in the manifest index")
        self.records: dict[str, dict] = {}
        for key, binding in (plan.get("records") or {}).items():
            self.records[key] = load_pinned(root, binding, f"records.{key}")
        self.pending = list(plan.get("pending_lanes") or [])

    # -- convenience -------------------------------------------------------
    @property
    def membership_points(self) -> int:
        return int(self.plan["membership"]["points"])

    def contributing(self) -> list[dict]:
        return [row for row in self.manifests["sets"]
                if row.get("contributes_vectors")]

    def additions(self) -> list[dict]:
        """Sets whose rows were never membership points."""
        return [row for row in self.manifests["sets"]
                if row.get("mode") == ADDITION_MODE]

    def pending_blocking(self, guard_id: str) -> list[dict]:
        return [lane for lane in self.pending
                if guard_id in (lane.get("blocks_guards") or [])]


# ---------------------------------------------------------------------------
# Precedence amendments, and the gate that decides whether they may be read.
# ---------------------------------------------------------------------------


def _draft_marker_cleared(root: Path, gate: dict) -> tuple[bool, str]:
    """Has the ADR this supplement encodes actually been ratified?

    The repo's legislative convention is that a POLICY/ADR change lands only
    through a PR Tim merges, and that merging IS ratification: the ADR text
    goes into ``decisions.md`` under a heading marked
    ``(DRAFT -- ratified by merging this PR)``, and the merge is what removes
    the marker.  So the ratified instrument itself is the gate, and it is
    readable offline -- which matters, because this suite makes no network
    request by design.

    What this deliberately does NOT do is read a status field out of the
    supplement.  A file that can declare itself ratified is not gated at all.
    """
    document = root / str(gate.get("document") or "")
    prefix = str(gate.get("heading_prefix") or "")
    marker = str(gate.get("draft_marker") or "")
    if not document.is_file() or not prefix or not marker:
        raise GuardInputError(
            "precedence supplement: the ratification gate must name a readable "
            "document, a heading prefix and a draft marker")
    heading = next((line for line in document.read_text().splitlines()
                    if line.startswith(prefix)), None)
    if heading is None:
        return False, (f"{prefix} is not in {gate['document']} yet -- the "
                       "amendment has not merged")
    if marker in heading:
        return False, (f"{prefix} is still marked {marker!r} in "
                       f"{gate['document']} -- the amendment has not merged")
    return True, f"{prefix} in {gate['document']} no longer carries the draft marker"


def load_precedence_supplement(ctx: Context) -> tuple[dict | None, bool, str]:
    """Load the precedence amendment and resolve its ratification gate.

    Returns ``(record, ratified, reason)``.  ``record`` is ``None`` when the
    plan binds no supplement, which is not an error: a build with no
    unadjudicated pairs needs none.
    """
    record = ctx.records.get("precedence_supplement")
    if record is None:
        return None, False, "the plan binds no precedence supplement"
    if record.get("schema") != SCHEMA_PRECEDENCE_SUPPLEMENT:
        raise GuardInputError(
            f"precedence supplement has the wrong schema: {record.get('schema')!r}")
    ratification = record.get("ratification") or {}
    gate = ratification.get("gate") or {}
    if gate.get("kind") != "adr_draft_marker_cleared":
        raise GuardInputError(
            f"unrecognised ratification gate {gate.get('kind')!r}; a gate this "
            "suite cannot evaluate is not a gate")
    ratified, reason = _draft_marker_cleared(ctx.root, gate)
    # Self-ratification refusal: the supplement may not claim a ratifier the
    # instrument does not show.  This is the check that makes the gate real.
    if not ratified and (ratification.get("ratified_by")
                         or ratification.get("ratified_at")):
        raise GuardInputError(
            f"precedence supplement names a ratifier "
            f"({ratification.get('ratified_by')!r}) while {reason}. A record "
            "does not ratify itself.")
    return record, ratified, reason



SCHEMA_CONVERGENT_IDENTITY = "aw.assembly-convergent-identity-record.v1"


def _convergent_row_ids(ctx: Context, overlaps: dict
                        ) -> tuple[dict[str, int], dict[str, int]]:
    """Split measured row-ID collisions into proven-convergent and not.

    The record has to earn each pair.  It must be the right schema, declare a
    single load, and -- the part that matters -- carry one verified row per
    collision the fold actually measured, every one of them byte-identical in
    both sets' embed stages.  A record that covers 16 of 17 rows covers no
    pair: the seventeenth is exactly the row that would load a vector nobody
    chose.
    """
    record = ctx.records.get("g7_convergent_identity")
    if record is None:
        return {}, dict(overlaps)
    if record.get("schema") != SCHEMA_CONVERGENT_IDENTITY:
        raise GuardInputError(
            f"convergent-identity record has the wrong schema: "
            f"{record.get('schema')!r}")
    if record.get("load") != "single":
        raise GuardInputError(
            "a convergent-identity record must declare a single load; "
            f"this one declares {record.get('load')!r}")
    proven: dict[str, int] = {}
    for row in (record.get("rows") or []):
        if not row.get("embedded_text_byte_identical"):
            continue
        try:
            uuid.UUID(str(row.get("point_id")))
        except (ValueError, AttributeError, TypeError):
            raise GuardInputError(
                f"convergent-identity record carries an invalid point ID: "
                f"{row.get('point_id')!r}") from None
        proven[str(row["pair"])] = proven.get(str(row["pair"]), 0) + 1
    covered, missed = {}, {}
    for pair, count in overlaps.items():
        # every measured collision on the pair, not merely some of them
        (covered if proven.get(pair) == int(count) else missed)[pair] = int(count)
    return covered, missed


def fold_overlaps_for(ctx: Context, set_id: str) -> list[str]:
    """The fold report's overlap pair keys that involve one set."""
    overlaps = (ctx.records.get("fold_report") or {}).get("claim_overlaps") or {}
    return [pair for pair in overlaps if set_id in pair.split(" x ")]


def precedence_rulings(record: dict | None) -> dict[frozenset[str], dict]:
    """The supplement's lines, keyed by the unordered set pair they rule."""
    out: dict[frozenset[str], dict] = {}
    for line in ((record or {}).get("lines") or []):
        left, separator, right = str(line.get("pair") or "").partition(" x ")
        if not separator:
            raise GuardInputError(
                f"precedence line {line.get('id')!r} names no 'LEFT x RIGHT' pair")
        if line.get("winner") not in {left, right}:
            raise GuardInputError(
                f"precedence line {line.get('id')!r} names a winner that is not "
                "one of the two sets it rules")
        out[frozenset({left, right})] = line
    return out


CLASS_PRECEDENCE = "data/assembly/adr101-class-precedence.v1.json"


def class_rule_rulings(ctx, measured_pairs) -> dict[frozenset[str], dict]:
    """Rule a measured pair the enumeration never reached, by CLASS rank.

    ADR-101 Ruling 1, ADR-101b Ruling 2 and ADR-101c Ruling 2 are written as
    class rules -- ADR-101c says so in as many words: "The guard consumes these
    as class rules; no future pair enumeration needed within either class." The
    guard has never evaluated one. It keys precedence by literal pair string, so
    a ratified class rule reaches the gate only if somebody enumerated the pair,
    and the precedence supplement's own `derivation` field records that gap.

    It stopped being theoretical when task 513b bound the 13 payload-route sets:
    three pairs the ledger had always counted became measurable for the first
    time -- datatracker_canonical_exact x task427 (3,568), named_slice:arxiv x
    task427 (176), task427 x wikipedia_413a (5,085), 8,829 points, exactly the
    shortfall G2's multi-claim-population had been carrying as "sets that
    publish no point-ID claim and so cannot be intersected".

    A pair is ruled here ONLY when the two sets sit at different ranks. Equal
    rank decides nothing and stays unruled, so the failure direction is closed:
    this can turn a FAIL into a PASS only where a ratified class rule already
    said who wins.
    """
    path = ctx.root / CLASS_PRECEDENCE
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text())
    if doc.get("schema") != "aw.adr101-class-precedence.v1":
        raise GuardInputError(f"{CLASS_PRECEDENCE}: unexpected schema")
    rank = {cls: entry["rank"] for entry in doc["ranks"]
            for cls in entry["classes"]}
    basis = {cls: entry["basis"] for entry in doc["ranks"]
             for cls in entry["classes"]}
    class_of = {row["set_id"]: row.get("class") for row in ctx.manifests["sets"]}

    out: dict[frozenset[str], dict] = {}
    for pair in measured_pairs:
        left, separator, right = str(pair).partition(" x ")
        if not separator:
            continue
        cl, cr = class_of.get(left), class_of.get(right)
        if cl not in rank or cr not in rank or rank[cl] == rank[cr]:
            continue
        winner, loser = (left, right) if rank[cl] < rank[cr] else (right, left)
        out[frozenset({left, right})] = {
            "id": f"class-rule:{doc['schema']}",
            "pair": pair,
            "winner": winner,
            "loser": loser,
            "ruling": "class",
            "basis": basis[class_of[winner]],
            "source": CLASS_PRECEDENCE,
        }
    return out


# ---------------------------------------------------------------------------
# The eight guards.
# ---------------------------------------------------------------------------


def _binds_old_points(binding: dict | None) -> bool:
    """Does this binding assert a relationship to an existing production point?

    An explicit ``{"count": 0}`` does not; a uri or a non-zero count does.
    """
    binding = binding or {}
    return bool(binding.get("uri")) or int(binding.get("count") or 0) > 0


def _addition_manifest_defects(row: dict) -> list[str]:
    """Everything a net-new-addition manifest owes, checked.

    This is the replacement lane's evidence burden moved on to the IDs an
    addition actually owns -- not a lighter one.  A set that fails any line
    here is either a replacement wearing the wrong label, or an addition whose
    minted IDs nobody can name, and both are worse than a set with no vectors
    yet: an unnamed ID cannot be intersected against anything.
    """
    defects: list[str] = []
    if row.get("contributes_production_points") is not False:
        defects.append("does not declare contributes_production_points: false")
    claims = row.get("claims") or {}
    claimed = int(claims.get("count") or 0)
    if claimed:
        defects.append(f"claims {claimed:,} production point(s); a net-new "
                       "addition binds none")
    if claims.get("claims_are_production_points") is not False:
        defects.append("its claim does not declare "
                       "claims_are_production_points: false")
    for name in ("delete_set", "identity_collapse"):
        if _binds_old_points(row.get(name)):
            defects.append(f"declares a non-empty {name}")
    minted = row.get("new_point_ids") or row.get("additions") or {}
    if not minted.get("uri") or not minted.get("sha256"):
        defects.append("does not enumerate its minted row IDs under a digest")
    if int(minted.get("count") or 0) <= 0:
        defects.append("enumerates no minted row ID")
    return defects


def guard_g1(ctx: Context) -> GuardResult:
    """Every set that contributes a vector publishes point-ID-level claims."""
    result = GuardResult("G1", "claim-manifest completeness")
    result.evidence.append(str(ctx.plan["inputs"]["manifest_index"]["uri"]))

    expected = set(ctx.plan.get("expected_sets") or [])
    if not expected:
        result.fail("roster-declared",
                    "the plan declares no expected_sets roster; completeness "
                    "cannot be checked and the guard fails closed")
        return result
    missing = sorted(expected - set(ctx.sets))
    extra = sorted(set(ctx.sets) - expected)
    if missing:
        result.fail("roster-complete",
                    f"{len(missing)} expected set(s) publish no manifest: "
                    f"{missing[:5]}", missing=len(missing))
    elif extra:
        result.fail("roster-complete",
                    f"{len(extra)} manifest(s) for sets outside the declared "
                    f"roster: {extra[:5]}", extra=len(extra))
    else:
        result.ok("roster-complete",
                  f"all {len(expected)} declared sets publish a manifest",
                  sets=len(expected))

    bad: list[str] = []
    counts_only: list[str] = []
    additions_ok: list[str] = []
    for row in ctx.manifests["sets"]:
        set_id = row["set_id"]
        claims = row.get("claims") or {}
        if row.get("contributes_vectors"):
            if row.get("mode") == ADDITION_MODE:
                # An addition's claim is empty because it binds nothing, not
                # because it enumerated nothing -- and those two look identical
                # from here unless the set publishes the IDs it DOES own.  So
                # the point-ID requirement is not relaxed, it is redirected on
                # to the row-ID artifact, with the same digest and the same
                # verified flag.
                defects = _addition_manifest_defects(row)
                if defects:
                    bad.append(f"{set_id} ({'; '.join(defects)})")
                else:
                    additions_ok.append(set_id)
                continue
            if claims.get("granularity") != "point_id":
                counts_only.append(
                    f"{set_id} (granularity={claims.get('granularity')!r})")
            elif not claims.get("verified") or not claims.get("count"):
                bad.append(f"{set_id} (claims unverified or empty)")
            elif not claims.get("sha256") and not claims.get("partitions"):
                bad.append(f"{set_id} (claims artifact carries no digest)")
        else:
            mode = row.get("mode")
            if mode not in {"tombstone", "cancelled"}:
                bad.append(f"{set_id} (contributes no vector but mode={mode!r})")
            elif mode == "cancelled":
                record = (row.get("cancellation") or {}).get("uri")
                cancel = ctx.records.get(f"cancellation:{set_id}")
                if not record or cancel is None:
                    bad.append(f"{set_id} (cancelled with no cancellation record)")
                elif not (cancel.get("authority") or {}).get("adr"):
                    bad.append(f"{set_id} (cancellation names no ratified ADR)")

    replacement_side = len(ctx.contributing()) - len(ctx.additions())
    if counts_only:
        result.fail("point-id-granularity",
                    f"{len(counts_only)} contributing set(s) publish counts or "
                    f"URLs only: {counts_only}", offending=len(counts_only))
    else:
        result.ok("point-id-granularity",
                  f"all {replacement_side} contributing sets that bind old "
                  "points publish point-ID claims",
                  contributing=replacement_side)
    if ctx.additions():
        if additions_ok and len(additions_ok) == len(ctx.additions()):
            result.ok("addition-rows-enumerated",
                      f"all {len(additions_ok)} net-new addition set(s) "
                      "enumerate every minted row ID under a digest, declare "
                      "`contributes_production_points: false`, and claim no "
                      "production point", sets=len(additions_ok))
        else:
            # the defects are already in `bad`, and manifest-well-formed below
            # reports them by name; this line exists so the guard's own count
            # of clean additions never reads as "all of them"
            result.fail("addition-rows-enumerated",
                        f"{len(ctx.additions()) - len(additions_ok)} of "
                        f"{len(ctx.additions())} net-new addition set(s) do not "
                        "meet the addition contract; see manifest-well-formed",
                        offending=len(ctx.additions()) - len(additions_ok))
    if bad:
        result.fail("manifest-well-formed", "; ".join(bad), offending=len(bad))
    else:
        result.ok("manifest-well-formed",
                  "every manifest is well formed; every non-contributing set "
                  "carries an explicit tombstone or cancellation record")

    lanes = ctx.pending_blocking("G1")
    if lanes:
        result.wait("pending-lanes",
                    f"{len(lanes)} lane(s) not sealed and will publish claims "
                    f"later: {[lane['lane_id'] for lane in lanes]}",
                    lanes=len(lanes),
                    points=sum(int(lane.get("points") or 0) for lane in lanes))

    result.counts = {
        "sets": len(ctx.sets),
        "contributing": len(ctx.contributing()),
        "net_new_addition_sets": len(ctx.additions()),
        "cancelled": sum(1 for row in ctx.manifests["sets"]
                         if row.get("mode") == "cancelled"),
        "tombstone_sets": sum(1 for row in ctx.manifests["sets"]
                              if row.get("mode") == "tombstone"),
    }
    result.remedy = ("publish a point-ID claim manifest for every listed set, "
                     "or record an explicit cancellation naming the ADR")
    return result


def guard_g2(ctx: Context) -> GuardResult:
    """No point is claimed twice without a named winner."""
    result = GuardResult("G2", "single-claim")
    ledger = ctx.ledger
    result.evidence.append(ctx.plan["inputs"]["collision_ledger"]["uri"])

    # Every double claim involves task 427 -- the ledger measured every other
    # set pair disjoint at point-ID level -- so the contested counts sum to
    # the whole multi-claimed population.  A shortfall means some set is
    # claiming points it never enumerated.
    # The ledger measured the population that existed when task 456 ran.  Sets
    # folded in since then bring their own collisions; counting those against
    # the ledger's total would compare two different populations and call the
    # difference a defect.  They are reported separately, below.
    attested_multi = int(ledger["problem"]["outside_adr092_remainder"])
    # The ledger-era population is the sets that existed when task 456 measured
    # it, NOT the sets that are currently unfolded. Those were the same set
    # until task 513b, and using `folded_from` as the partition was a latent
    # coincidence: binding the 13 payload-route sets folds three LEDGER-ERA sets
    # (datatracker_canonical_exact, named_slice:arxiv, wikipedia_413a, 8,829
    # points), which dropped `declared` below `attested` and reported a
    # shortfall that is really a re-partition. The correct, fold-invariant axis
    # is membership in the ADR-101b/c supplement -- its own title is "Precedence
    # for the POST-ADR-101 claim sets", so its participants are exactly the
    # collisions the 456 ledger did NOT measure. Verified: summing
    # contested_with_task427 over the non-supplement sets reproduces the ledger's
    # 976,351 exactly, before the fold and after.
    supplement_sets: set[str] = set()
    for line in ((supplement := (load_precedence_supplement(ctx)[0]) or {})
                 .get("lines") or []):
        left, _, right = str(line.get("pair") or "").partition(" x ")
        supplement_sets.update({left, right})
    ledger_era = [row for row in ctx.manifests["sets"]
                  if row["set_id"] not in supplement_sets]
    declared = sum(int((row.get("adjudication") or {}).get(
        "contested_with_task427") or 0) for row in ledger_era)
    folded_contest = sum(int((row.get("adjudication") or {}).get(
        "contested_with_task427") or 0) for row in ctx.manifests["sets"]
        if row["set_id"] in supplement_sets)
    if declared != attested_multi:
        gap = attested_multi - declared
        unenumerated = sorted(
            row["set_id"] for row in ledger_era
            if (row.get("claims") or {}).get("granularity") != "point_id")
        result.fail("multi-claim-population",
                    f"the manifest index adjudicates {declared:,} double-claimed "
                    f"points; the sha-pinned ledger measured {attested_multi:,} "
                    f"-- {gap:+,} unaccounted. Sets that publish no point-ID "
                    f"claim and so cannot be intersected: {unenumerated}",
                    declared=declared, attested=attested_multi, gap=gap)
    else:
        result.ok("multi-claim-population",
                  f"{declared:,} double-claimed points, independently measured "
                  "by task 456 and re-measured here, agree exactly"
                  + (f"; {folded_contest:,} further contested points arrived "
                     "with the folded sets and are adjudicated below"
                     if folded_contest else ""),
                  points=declared, folded_contest=folded_contest)

    # A precedence amendment may rule pairs the ledger never measured.  It is
    # readable only through its own ratification gate: while the ADR it encodes
    # is still a draft, its lines name the pairs but decide nothing.
    supplement, ratified, gate_reason = load_precedence_supplement(ctx)
    rulings = precedence_rulings(supplement)
    if supplement is not None:
        result.evidence.append(ctx.plan["records"]["precedence_supplement"]["uri"])
    # The enumerated lines rule first; the ratified CLASS rules rule what the
    # enumeration never reached. Supplement lines are never overridden.
    measured_pairs = set((ctx.records.get("fold_report") or {})
                         .get("claim_overlaps") or {})
    for set_id in {row["set_id"] for row in ctx.manifests["sets"]}:
        measured_pairs |= set(fold_overlaps_for(ctx, set_id))
    by_class = class_rule_rulings(ctx, measured_pairs)
    class_ruled = {p: r for p, r in by_class.items() if p not in rulings}
    if class_ruled:
        rulings = {**by_class, **rulings}
        result.evidence.append(CLASS_PRECEDENCE)

    unnamed = [row["set_id"] for row in ctx.manifests["sets"]
               if int((row.get("adjudication") or {}).get(
                   "contested_with_task427") or 0) > 0
               and not (row.get("adjudication") or {}).get("winner_on_contested")]
    # A set is named by the amendment only if the amendment rules EVERY pair it
    # is contested in -- and only if it is contested in at least one measured
    # pair.  A set with no measured pairs is not covered vacuously; it is a set
    # whose collisions nobody has located, which is worse, not better.
    covered_sets = set()
    for set_id in unnamed:
        pairs = fold_overlaps_for(ctx, set_id)
        if pairs and all(frozenset(pair.split(" x ")) in rulings
                         for pair in pairs):
            covered_sets.add(set_id)
    if unnamed and ratified and covered_sets >= set(unnamed):
        result.ok("every-collision-named",
                  f"every contested set names a winner: {len(unnamed)} under "
                  f"{supplement['adr']} ({gate_reason})",
                  ruled_by_amendment=len(unnamed))
    elif unnamed:
        still_unnamed = sorted(set(unnamed) - covered_sets)
        if still_unnamed:
            result.fail("every-collision-named",
                        f"{len(still_unnamed)} contested set(s) name no winner "
                        f"and no ratified or pending amendment rules them: "
                        f"{still_unnamed}", offending=len(still_unnamed))
        else:
            result.wait("every-collision-named",
                        f"{len(unnamed)} contested set(s) are ruled by "
                        f"{supplement['adr']}, which is not ratified: "
                        f"{gate_reason}", pending=len(unnamed))
    else:
        result.ok("every-collision-named",
                  "every contested set names a winner under ADR-101 Ruling 1")

    # Any collision the ledger measured OUTSIDE the 427 axis must be zero, or
    # adjudicated.  A non-zero, unnamed pair is a hard failure.
    unadjudicated = []
    for pair, row in (ledger.get("other_collisions") or {}).items():
        if pair.endswith("_x_task427_postconsent") and pair.startswith("track1a"):
            continue  # the cancelled lane; G1 owns it
        value = row.get("intersect")
        if value is None:
            value = row.get("intersect_task427")
        if value not in (0, None):
            unadjudicated.append(f"{pair}={value}")
    if unadjudicated:
        result.fail("no-unadjudicated-pairs",
                    "; ".join(unadjudicated), pairs=len(unadjudicated))
    else:
        result.ok("no-unadjudicated-pairs",
                  "every non-427 set pair measured disjoint at point-ID level")

    fold = ctx.records.get("fold_report") or {}
    if not fold:
        result.wait("folded-claims-disjoint",
                    "no fold report is bound, so the sets folded in since the "
                    "ledger was measured have not been intersected pairwise")
    else:
        overlaps = fold.get("claim_overlaps") or {}
        ruled = {pair: count for pair, count in overlaps.items()
                 if frozenset(pair.split(" x ")) in rulings}
        unruled = {pair: count for pair, count in overlaps.items()
                   if pair not in ruled}
        if unruled:
            result.fail("folded-claims-disjoint",
                        f"{len(unruled)} pair(s) of folded sets claim the same "
                        f"old point(s) with no adjudicated winner and no "
                        f"amendment before the ratifier: {unruled}",
                        pairs=len(unruled), ruled_pending=len(ruled))
        elif ruled and not ratified:
            result.wait("folded-claims-disjoint",
                        f"{len(ruled)} pair(s) ({sum(ruled.values()):,} points) "
                        f"are ruled by {supplement['adr']}, which is not "
                        f"ratified: {gate_reason}",
                        pairs=len(ruled), points=sum(ruled.values()))
        elif ruled:
            result.ok("folded-claims-disjoint",
                      f"{len(ruled)} pair(s) ({sum(ruled.values()):,} points) "
                      f"adjudicated by {supplement['adr']}; "
                      f"{gate_reason}",
                      pairs=len(ruled), points=sum(ruled.values()))
        else:
            measured = len(fold.get("tombstone_intersection") or {})
            result.ok("folded-claims-disjoint",
                      f"every pair among the {measured} folded set(s) and task "
                      "427 measured disjoint at point-ID level",
                      sets=measured)

    lanes = ctx.pending_blocking("G2")
    if lanes:
        result.wait("pending-lanes",
                    f"{len(lanes)} unsealed lane(s) may add claims: "
                    f"{[lane['lane_id'] for lane in lanes]}", lanes=len(lanes))

    result.counts = {
        "multi_claimed_points": declared,
        "ledger_winner_breakdown":
            ledger["task427_disposition"]["lost_breakdown"],
        "precedence_amendment": ({
            "adr": supplement["adr"],
            "uri": ctx.plan["records"]["precedence_supplement"]["uri"],
            # the ENUMERATED lines only -- the class-ruled pairs are counted
            # separately below, so "pairs_ruled" stays equal to the supplement's
            # own line count (a test pins it to len(supplement["lines"])).
            "pairs_ruled": len(precedence_rulings(supplement)),
            "class_ruled_pairs": len(class_ruled),
            "ratified": ratified,
            "gate": gate_reason,
            "pull_request":
                (supplement.get("ratification") or {}).get("pull_request"),
            "pairs_not_covered": [
                {"pair": row.get("pair"), "points": row.get("points"),
                 "status": row.get("status")}
                for row in (supplement.get("not_covered") or [])],
        } if supplement is not None else None),
    }
    result.remedy = "adjudicate every double claim in the ledger, or drop the set"
    return result


def _check_additions_term(ctx: Context, result: GuardResult,
                          membership: int) -> None:
    """G3's additions arithmetic: the term the disposition identity cannot hold.

    ``replaced + re_embedded_in_place + tombstoned + carried_legacy + pending +
    contested = membership`` is an identity OVER the membership.  A row that
    was never a member satisfies none of those predicates, so before this term
    existed the built collection held ``membership + additions`` points while
    the plan declared ``membership`` -- and nothing in the suite could see the
    difference, because every check was an equation the additions were absent
    from on both sides.

    The plan states the term; the manifest index attests it; this refuses when
    they disagree.  A plan that simply omits it while an addition set exists
    fails closed, which is the same treatment ``expected_sets`` gets and for
    the same reason: silence is not a passing answer.
    """
    addition_sets = ctx.additions()
    declared = ctx.plan.get("additions")

    # An empty term and no term say the same thing, and the honest producer is
    # the one that would otherwise fail.  This is the same reading the fold
    # gives an explicit `{"count": 0}` delete set.
    if declared is not None and not (declared.get("sets")
                                     or int(declared.get("count") or 0)):
        declared = None

    if declared is None:
        if addition_sets:
            result.fail(
                "additions-declared",
                f"{len(addition_sets)} set(s) contribute rows that were never "
                f"membership points "
                f"({sorted(row['set_id'] for row in addition_sets)}) and the "
                "plan declares no `additions` term. The disposition is an "
                "identity over the membership and has no bucket for them, so "
                "the collection would hold points the plan does not account "
                "for", sets=len(addition_sets))
        else:
            result.ok("additions-declared",
                      "no set adds net-new rows; the dispositions account for "
                      "every point the collection will hold")
        return

    if not addition_sets:
        result.fail("additions-declared",
                    "the plan declares an `additions` term but no manifest "
                    "declares mode 'net_new_addition'; the term describes a "
                    "population that does not exist")
        return

    manifest_rows = {row["set_id"]:
                     int((row.get("vector") or {}).get("rows") or 0)
                     for row in addition_sets}
    minted_ids = {row["set_id"]:
                  int(((row.get("new_point_ids") or row.get("additions")
                        or {}).get("count")) or 0)
                  for row in addition_sets}
    entries = {str(row.get("set_id")): row
               for row in (declared.get("sets") or [])}

    problems: list[str] = []
    if set(entries) != set(manifest_rows):
        problems.append(
            f"the plan lists {sorted(entries)} and the manifest index declares "
            f"{sorted(manifest_rows)}")
    for set_id, entry in sorted(entries.items()):
        rows = int(entry.get("rows") or 0)
        if set_id in manifest_rows and rows != manifest_rows[set_id]:
            problems.append(f"{set_id}: plan says {rows:,} rows, the manifest "
                            f"declares {manifest_rows[set_id]:,}")
        if set_id in minted_ids and rows != minted_ids[set_id]:
            problems.append(f"{set_id}: plan says {rows:,} rows, the set "
                            f"enumerates {minted_ids[set_id]:,} minted IDs")
        if int(entry.get("claimed_production_points", -1)) != 0:
            problems.append(f"{set_id}: the plan declares "
                            f"{entry.get('claimed_production_points')!r} "
                            "claimed production points; an addition claims 0")
        binding = entry.get("row_ids") or {}
        if not binding.get("uri") or not binding.get("sha256"):
            problems.append(f"{set_id}: the plan pins no digested row-ID "
                            "artifact for its minted IDs")
        elif int(binding.get("count") or 0) != rows:
            problems.append(f"{set_id}: the pinned row-ID artifact carries "
                            f"{int(binding.get('count') or 0):,} IDs for "
                            f"{rows:,} declared rows")

    total_declared = sum(int(entry.get("rows") or 0)
                         for entry in entries.values())
    count = int(declared.get("count", -1))
    if count != total_declared:
        problems.append(f"additions.count is {count:,}, its own sets sum to "
                        f"{total_declared:,}")
    if count != sum(manifest_rows.values()):
        problems.append(f"additions.count is {count:,}, the manifest index's "
                        f"addition sets carry {sum(manifest_rows.values()):,} "
                        "rows")
    collection = int(declared.get("collection_points_after_additions", -1))
    if collection != membership + count:
        problems.append(
            f"the plan declares a {collection:,}-point collection after "
            f"additions; {membership:,} membership + {count:,} additions is "
            f"{membership + count:,}")

    if problems:
        result.fail("additions-declared", "; ".join(problems),
                    offending=len(problems))
    else:
        result.ok("additions-declared",
                  f"{count:,} net-new row(s) across {len(entries)} set(s), "
                  "each enumerated under a digest and claiming no production "
                  f"point; the collection holds {collection:,} points, which "
                  f"the disposition's {membership:,} could not have said",
                  additions=count, sets=len(entries), collection=collection)


def guard_g3(ctx: Context) -> GuardResult:
    """Every membership point lands in exactly one disposition."""
    result = GuardResult("G3", "conservation / no-zero-claim")
    total = ctx.membership_points
    disposition = ctx.plan.get("disposition") or {}
    required = ["replaced", "re_embedded_in_place", "tombstoned",
                "carried_legacy", "pending", "contested"]
    missing = [key for key in required if key not in disposition]
    if missing:
        result.fail("disposition-complete",
                    f"the plan declares no {missing} bucket(s)")
        return result

    values = {key: int(disposition[key]) for key in required}
    got = sum(values.values())
    if got != total:
        result.fail("sums-to-membership",
                    f"{got:,} disposed != {total:,} membership points "
                    f"(delta {got - total:+,})", **values, total=total)
    else:
        result.ok("sums-to-membership",
                  f"the six dispositions sum to {total:,} exactly", **values)

    # A point two sets both claim has no owner, so it has no disposition.
    # Folding it into `replaced` would close the arithmetic while asserting an
    # ownership nobody has ruled on.
    if values["contested"]:
        pairs = (ctx.plan.get("contested") or {}).get("pairs") or {}
        result.wait("no-contested-points",
                    f"{values['contested']:,} point(s) are claimed by two "
                    f"contributing sets with no adjudicated winner: {pairs}",
                    points=values["contested"], pairs=len(pairs))
    else:
        result.ok("no-contested-points",
                  "every point has exactly one claiming set")

    _check_additions_term(ctx, result, total)

    expected_tombstones = int(
        ctx.census["buckets"]["E_no_vector_will_exist"]["points"])
    if values["tombstoned"] != expected_tombstones:
        result.fail("tombstone-count",
                    f"plan tombstones {values['tombstoned']:,} != census bucket E "
                    f"{expected_tombstones:,}",
                    declared=values["tombstoned"], census=expected_tombstones)
    else:
        result.ok("tombstone-count",
                  f"{expected_tombstones:,} tombstoned points match census "
                  "bucket E", points=expected_tombstones)

    carried = ctx.plan.get("carried_legacy") or {}
    if values["carried_legacy"] != int(carried.get("count", -1)):
        result.fail("carried-legacy-declared",
                    "disposition.carried_legacy and carried_legacy.count "
                    "disagree")
    elif values["carried_legacy"] == 0:
        result.ok("carried-legacy-enumerated",
                  "no point is silently carried on a legacy vector")
    elif not carried.get("enumeration_uri"):
        result.fail("carried-legacy-enumerated",
                    f"{values['carried_legacy']:,} points would be carried on "
                    "legacy vectors and are NOT enumerated; this is the "
                    "census-453 bucket-F failure mode",
                    points=values["carried_legacy"])
    else:
        waiver = carried.get("waiver")
        if not waiver:
            result.fail("carried-legacy-waived",
                        f"{values['carried_legacy']:,} points enumerated but no "
                        "named waiver; the promotion gate refuses",
                        points=values["carried_legacy"])
        elif not waiver.get("ratified_by"):
            result.fail("carried-legacy-waived",
                        "the carried-legacy waiver names no ratifier; gate "
                        "waivers are never self-issued")
        else:
            result.ok("carried-legacy-waived",
                      f"{values['carried_legacy']:,} enumerated and waived by "
                      f"{waiver['ratified_by']}", points=values["carried_legacy"])

    lane_points = sum(int(lane.get("points") or 0) for lane in ctx.pending)
    reconciliation = ctx.plan.get("pending_lane_reconciliation")
    if values["pending"] != lane_points and reconciliation:
        # The lanes are on census 453's bucket basis and `pending` is measured
        # off the census remainder.  Two bases may differ -- but the difference
        # has to be attributed, line by line, to named populations.  A declared
        # gap with an unexplained remainder is a balancing entry wearing a
        # label, so a residue is a hard failure and only a residue of zero
        # leaves this an open-lane question rather than an accounting one.
        residue = reconciliation.get("unexplained_residue")
        lines = reconciliation.get("lines") or []
        if residue is None or not lines:
            result.fail("pending-accounted",
                        f"disposition.pending {values['pending']:,} and the "
                        f"named lanes {lane_points:,} differ by "
                        f"{lane_points - values['pending']:+,} with no "
                        "line-by-line reconciliation",
                        pending=values["pending"], lanes=lane_points)
        elif int(residue) != 0:
            result.fail("pending-accounted",
                        f"the two accounting bases differ by "
                        f"{int(reconciliation.get('gap') or 0):+,}, of which "
                        f"{int(residue):+,} is attributed to no named "
                        "population",
                        pending=values["pending"], lanes=lane_points,
                        residue=int(residue))
        else:
            result.wait("pending-accounted",
                        f"disposition.pending {values['pending']:,} and the "
                        f"named lanes {lane_points:,} differ by "
                        f"{int(reconciliation.get('gap') or 0):+,}, attributed "
                        f"in full across {len(lines)} named population(s) with "
                        "zero residue; the lanes themselves are still open",
                        pending=values["pending"], lanes=lane_points,
                        gap=int(reconciliation.get("gap") or 0),
                        reconciled_lines=len(lines))
    elif values["pending"] != lane_points:
        result.fail("pending-accounted",
                    f"disposition.pending {values['pending']:,} != the sum of "
                    f"pending_lanes {lane_points:,}",
                    pending=values["pending"], lanes=lane_points)
    elif values["pending"]:
        result.wait("pending-accounted",
                    f"{values['pending']:,} points across "
                    f"{len(ctx.pending)} unsealed lane(s) have no terminal "
                    "disposition yet",
                    points=values["pending"], lanes=len(ctx.pending))
    else:
        result.ok("pending-accounted", "no lane is still in flight")

    result.counts = {**values, "membership": total,
                     "additions": int((ctx.plan.get("additions")
                                       or {}).get("count") or 0),
                     "pending_lanes": len(ctx.pending)}
    result.remedy = ("seal every pending lane, then enumerate and either empty "
                     "or have Tim waive the carried_legacy set")
    return result


def guard_g4(ctx: Context) -> GuardResult:
    """A tombstoned point accepts no replacement."""
    result = GuardResult("G4", "tombstone exclusivity")
    record = ctx.records.get("g4_tombstone_suppression")
    tombstone_sets = [row for row in ctx.manifests["sets"]
                      if row.get("mode") == "tombstone"]
    if not tombstone_sets:
        result.fail("tombstone-sets-present",
                    "the manifest index declares no tombstone set; ADR-091's "
                    "policy drop must be represented")
        return result

    reclaimed = sum(int((row.get("tombstones") or {}).get(
        "reclaimed_by_a_winning_set") or 0) for row in tombstone_sets)
    attested = int(ctx.ledger["by_class"]["tombstone"]["contested_with_task427"])
    # Every set folded in since the ledger was measured must be intersected
    # against the tombstones too, not assumed disjoint.
    fold = ctx.records.get("fold_report") or {}
    hits = {name: count for name, count
            in (fold.get("tombstone_intersection") or {}).items()
            if count and name != "task427_postconsent"}
    if not fold:
        result.wait("folded-sets-intersected",
                    "no fold report is bound, so the sets folded in since the "
                    "ledger was measured have not been intersected against the "
                    "tombstones")
    elif hits:
        result.fail("folded-sets-intersected",
                    f"{len(hits)} folded set(s) claim a tombstoned point: "
                    f"{hits}", sets=len(hits))
    else:
        measured = len(fold.get("tombstone_intersection") or {})
        result.ok("folded-sets-intersected",
                  f"all {measured} folded set(s) intersect the ADR-091 "
                  "tombstones at zero", sets=measured)
    if reclaimed != attested:
        result.fail("reclaim-count",
                    f"manifest index reports {reclaimed} tombstoned point(s) "
                    f"re-claimed; ledger measured {attested}",
                    manifest=reclaimed, ledger=attested)
    else:
        result.ok("reclaim-count",
                  f"{reclaimed} tombstoned point(s) re-claimed by a winning "
                  "set, agreed by two independent walks", points=reclaimed)

    if reclaimed == 0:
        result.ok("suppression-record",
                  "no suppression needed: tombstones intersect no winning claim")
    elif record is None:
        result.fail("suppression-record",
                    f"{reclaimed} tombstoned point(s) are re-claimed and no "
                    "suppression record is bound to the plan",
                    points=reclaimed)
    else:
        ids = record.get("point_ids") or []
        problems = []
        if record.get("guard") != "G4":
            problems.append("record is not a G4 record")
        if int(record.get("count", -1)) != len(ids):
            problems.append("count does not match the listed IDs")
        if len(ids) != reclaimed:
            problems.append(f"{len(ids)} IDs listed, {reclaimed} re-claimed")
        if len(set(ids)) != len(ids):
            problems.append("duplicate point IDs in the suppression record")
        for point_id in ids:
            try:
                uuid.UUID(str(point_id))
            except (ValueError, AttributeError, TypeError):
                problems.append(f"invalid point ID {point_id!r}")
                break
        if problems:
            result.fail("suppression-record", "; ".join(problems))
        else:
            result.ok("suppression-record",
                      f"all {len(ids)} re-claimed tombstoned point(s) are "
                      "struck from the winning set's claim by name",
                      points=len(ids))
            result.evidence.append(
                ctx.plan["records"]["g4_tombstone_suppression"]["uri"])
            result.ok("residual-intersection",
                      "tombstones INTERSECT winning claims = 0 after suppression",
                      residual=0)

    # -- ADR-101a Correction 2: a URL bearing an ADR-091 tombstone accepts no
    # replacement rows.  Ratified, so this is now a RULE, not an escalation:
    # a URL that still loads a row is a privacy defect, not an open question.
    escalations = _tombstone_url_escalations(ctx)
    record = ctx.records.get("g4_tombstone_url_suppression")
    if record is None:
        result.fail("adr101a-url-suppression",
                    "ADR-101a Correction 2 is ratified but the plan binds no "
                    "URL-level suppression record")
    elif escalations:
        result.fail("adr101a-url-suppression",
                    f"{len(escalations)} URL(s) still load a replacement row "
                    "while another point on the same URL is tombstoned by the "
                    f"ADR-091 policy drop: {sorted(escalations)[:5]}",
                    urls=len(escalations))
    else:
        suppressed = record.get("suppressed") or []
        bad = [row.get("url") for row in suppressed
               if int(row.get("rows_loaded_after", -1)) != 0]
        if bad:
            result.fail("adr101a-url-suppression",
                        f"{len(bad)} suppressed URL(s) still declare loaded "
                        f"rows: {bad[:5]}")
        else:
            result.ok("adr101a-url-suppression",
                      f"{len(suppressed)} tombstoned URL(s) suppress all "
                      f"{int(record.get('rows_suppressed', 0))} replacement "
                      "row(s); privacy beats coverage",
                      urls=len(suppressed),
                      rows=int(record.get("rows_suppressed", 0)))
        # every OTHER contributing set has to be asked the same question
        other = record.get("other_sets") or {}
        if not other.get("measured"):
            result.wait("adr101a-other-sets",
                        "only task 427's URLs have been joined against the "
                        "tombstoned URL set; every other contributing set is "
                        "unmeasured, and an entity-card tombstone can share a "
                        "URL with a crawled page")
        else:
            still_open = record.get("open_obligations") or []
            if still_open:
                result.fail(
                    "adr101a-other-sets",
                    f"{len(still_open)} URL(s) carry a tombstone and a "
                    "replacement row from a set that has not suppressed it: "
                    + "; ".join(f"{row['set_id']} -> {row['url']}"
                                for row in still_open[:5]),
                    urls=len(still_open))
            else:
                result.ok("adr101a-other-sets",
                          f"every URL the membership join found is covered: "
                          f"{len(record.get('suppressed') or [])} corrected "
                          f"here, {len(record.get('other_entries') or [])} "
                          f"already satisfied, "
                          f"{len(other.get('clean') or [])} set(s) clean",
                          clean_sets=len(other.get("clean") or []))

    lanes = ctx.pending_blocking("G4")
    if lanes:
        result.wait("pending-lanes",
                    f"{len(lanes)} unsealed lane(s) may re-claim a tombstone",
                    lanes=len(lanes))
    result.counts = {"tombstone_sets": len(tombstone_sets),
                     "reclaimed": reclaimed,
                     "urls_loading_on_a_tombstoned_page": len(escalations)}
    result.remedy = ("strike every re-claimed tombstoned point from the winning "
                     "set and delete the point")
    return result


def _tombstone_url_escalations(ctx: Context) -> list[str]:
    """URLs still loading a replacement row despite carrying a tombstone.

    The G6 unit resolution is the *measurement* -- it records what the ratified
    tie-break decided before ADR-101a existed, and it is deliberately not
    rewritten.  A URL it shows as loading is only a violation if the ADR-101a
    suppression record does not cover it.  Reading the measurement without the
    correction on top would make this guard fail forever.
    """
    record = ctx.records.get("g6_mixed_url_resolution")
    if not record:
        return []
    suppression = ctx.records.get("g4_tombstone_url_suppression") or {}
    covered = {row.get("url") for row in (suppression.get("suppressed") or [])
               if int(row.get("rows_loaded_after", -1)) == 0}
    out = []
    for row in record.get("resolutions") or []:
        tally = row.get("point_winner_tally") or {}
        if (tally.get(TOMBSTONE_WINNER)
                and int(row.get("task427_rows_loaded", 0)) > 0
                and row.get("url") not in covered):
            out.append(row.get("url"))
    return out


def guard_g5(ctx: Context) -> GuardResult:
    """Rows whose bound old points all lost are never loaded."""
    result = GuardResult("G5", "row-level loser suppression")

    declared_rule = (ctx.plan.get("precedence") or {})
    if declared_rule.get("adr") != "ADR-101":
        result.fail("precedence-is-ratified",
                    f"the plan applies {declared_rule.get('adr')!r}, not ADR-101")
    elif declared_rule.get("table") != ADR101_PRECEDENCE:
        diff = {key: (declared_rule.get("table") or {}).get(key)
                for key, value in ADR101_PRECEDENCE.items()
                if (declared_rule.get("table") or {}).get(key) != value}
        result.fail("precedence-is-ratified",
                    f"the plan's precedence table diverges from ADR-101 "
                    f"Ruling 1 on {diff}")
    else:
        result.ok("precedence-is-ratified",
                  "the plan applies ADR-101 Ruling 1 chunk-epoch-then-lineage "
                  "verbatim")

    # The suppression arithmetic for the atomic families rests on one premise:
    # strictly one vector row per old point.  Assert it rather than assume it.
    broken_premise = [
        f"{row['set_id']}: {int(row['vector']['rows']):,} rows for "
        f"{int(row['claims']['count']):,} claimed points"
        for row in ctx.contributing()
        if row.get("class") == "atomic_epoch1"
        and (row.get("vector") or {}).get("declared")
        and int(row["vector"]["rows"]) != int(row["claims"]["count"])]
    atomic = sum(1 for row in ctx.contributing()
                 if row.get("class") == "atomic_epoch1")
    if broken_premise:
        result.fail("atomic-families-are-one-row-per-point",
                    "; ".join(broken_premise), offending=len(broken_premise))
    elif atomic:
        result.ok("atomic-families-are-one-row-per-point",
                  f"all {atomic} atomic families emit exactly one row per "
                  "claimed old point, so a forfeited point is a forfeited row",
                  sets=atomic)

    broken: list[str] = []
    waiting: list[str] = []
    checked = 0
    for row in ctx.contributing():
        adj = row.get("adjudication") or {}
        if "rows_declared" not in adj:
            waiting.append(row["set_id"])
            continue
        if "rows_pending_adjudication" in adj:
            # The set claims points another set also claims.  Declaring a row
            # split would assert an ownership nobody has ruled on.
            waiting.append(row["set_id"])
            continue
        declared = int(adj["rows_declared"])
        loaded = int(adj.get("rows_loaded", 0))
        suppressed = int(adj.get("rows_suppressed", 0))
        checked += 1
        if loaded + suppressed != declared:
            broken.append(f"{row['set_id']}: {loaded:,}+{suppressed:,} != "
                          f"{declared:,}")
    if broken:
        result.fail("rows-conserved-per-set", "; ".join(broken),
                    offending=len(broken))
    else:
        result.ok("rows-conserved-per-set",
                  f"rows_loaded + rows_suppressed = rows_declared for all "
                  f"{checked} set(s) with vectors", sets=checked)

    # -- net-new additions conserve, row for row ---------------------------
    # The replacement lane's conservation is `delete_set + identity_collapse =
    # superseded`, and G7 checks it.  An addition has none of those three, so
    # its conservation is the other identity its own fragment states:
    # rows == minted IDs == nothing claimed.  Two of the three numbers agreeing
    # is what a partially-embedded set looks like, and it is exactly the case
    # where a shard is missing and nobody notices.
    addition_broken: list[str] = []
    for row in ctx.additions():
        vector_rows = int((row.get("vector") or {}).get("rows", -1))
        minted = int(((row.get("new_point_ids") or row.get("additions")
                       or {}).get("count", -2)))
        claimed = int((row.get("claims") or {}).get("count", -3))
        if vector_rows != minted or claimed != 0:
            addition_broken.append(
                f"{row['set_id']}: {vector_rows:,} vector rows, {minted:,} "
                f"minted row IDs, {claimed:,} production points claimed -- an "
                "addition must satisfy rows == minted IDs and claim nothing")
    fold_additions = (ctx.records.get("fold_report") or {}).get("additions")
    if addition_broken:
        result.fail("additions-conserve", "; ".join(addition_broken),
                    offending=len(addition_broken))
    elif ctx.additions() and fold_additions is None:
        result.wait("additions-conserve",
                    f"{len(ctx.additions())} addition set(s) conserve as "
                    "declared, but no fold report is bound, so the minted IDs "
                    "have not been counted on disk")
    elif ctx.additions():
        unconserved = sorted(name for name, row in fold_additions.items()
                             if not row.get("conserves"))
        if unconserved:
            result.fail("additions-conserve",
                        f"the fold counted the minted IDs on disk and "
                        f"{len(unconserved)} addition set(s) do not conserve: "
                        f"{unconserved}", offending=len(unconserved))
        else:
            result.ok("additions-conserve",
                      f"all {len(ctx.additions())} addition set(s) satisfy "
                      "rows == minted IDs == vectors and claim no production "
                      "point, counted on disk by the fold",
                      sets=len(ctx.additions()))

    measured = ((ctx.ledger.get("assembly_guard") or {}).get("checks") or [])
    g5 = next((row for row in measured if row.get("id") == "G5"), {})
    ledger_rows = g5.get("measured_today") or {}
    plan427 = (ctx.sets.get(TASK427) or {}).get("adjudication") or {}
    if ledger_rows and "rows_declared" in plan427:
        expected = int(ledger_rows["task427_rows_declared"])
        got = int(plan427["rows_declared"])
        expected_loaded = (int(ledger_rows["task427_rows_loaded"])
                           + int(ledger_rows.get("task427_rows_mixed", 0)))
        if got != expected:
            result.fail("task427-rows-agree",
                        f"plan declares {got:,} task-427 rows, ledger {expected:,}")
        elif int(plan427["rows_loaded"]) > expected_loaded:
            result.fail("task427-rows-agree",
                        f"plan loads {plan427['rows_loaded']:,} task-427 rows, "
                        f"more than the ledger's ceiling {expected_loaded:,} "
                        "(wins + every mixed-URL row)")
        else:
            result.ok("task427-rows-agree",
                      f"task 427 loads {plan427['rows_loaded']:,} of "
                      f"{got:,} rows; the ledger's win/loss split is honoured",
                      loaded=int(plan427["rows_loaded"]), declared=got)

    if waiting:
        result.wait("rows-known",
                    f"{len(waiting)} contributing set(s) have no stateable row "
                    f"split -- no vectors yet, or ownership unadjudicated: "
                    f"{sorted(waiting)}", sets=len(waiting))
    lanes = ctx.pending_blocking("G5")
    if lanes:
        result.wait("pending-lanes",
                    f"{len(lanes)} unsealed lane(s) will add rows",
                    lanes=len(lanes))

    result.counts = {"sets_checked": checked, "sets_without_rows": len(waiting)}
    result.remedy = "suppress every row whose bound old points all lost"
    return result


def guard_g6(ctx: Context) -> GuardResult:
    """A re-chunk replaces a whole page, never half of one."""
    result = GuardResult("G6", "URL-unit integrity")
    measured = ((ctx.ledger.get("assembly_guard") or {}).get("checks") or [])
    g6 = next((row for row in measured if row.get("id") == "G6"), {})
    attested = g6.get("measured_today") or {}
    record = ctx.records.get("g6_mixed_url_resolution")

    if not attested:
        result.fail("ledger-binds-mixed-urls",
                    "the ledger carries no G6 measurement to check against")
        return result

    if record is None:
        result.fail("resolution-record",
                    f"{attested['mixed_urls']} URL(s) are split between a "
                    "re-chunk and a sealed set and no unit resolution is bound",
                    urls=int(attested["mixed_urls"]))
        result.remedy = ("emit a unit resolution for every mixed URL using the "
                         "ADR-101 tie-break")
        return result

    result.evidence.append(ctx.plan["records"]["g6_mixed_url_resolution"]["uri"])
    resolutions = record.get("resolutions") or []
    problems: list[str] = []
    if int(record.get("urls", -1)) != len(resolutions):
        problems.append("urls count does not match the listed resolutions")
    if len(resolutions) != int(attested["mixed_urls"]):
        problems.append(f"{len(resolutions)} URLs resolved, ledger measured "
                        f"{attested['mixed_urls']}")
    if int(record.get("old_points", -1)) != int(attested["old_points"]):
        problems.append(f"{record.get('old_points')} old points covered, ledger "
                        f"measured {attested['old_points']}")
    if int(record.get("task427_new_rows", -1)) != int(attested["new_rows"]):
        problems.append(f"{record.get('task427_new_rows')} rows covered, ledger "
                        f"measured {attested['new_rows']}")
    seen_urls = set()
    for item in resolutions:
        url = item.get("url")
        if url in seen_urls:
            problems.append(f"duplicate URL in the resolution: {url}")
            break
        seen_urls.add(url)
        if not item.get("unit_winner"):
            problems.append(f"{url}: no unit winner")
            break
        if item.get("basis") not in VALID_UNIT_TIEBREAKS:
            problems.append(f"{url}: basis {item.get('basis')!r} is not the "
                            "ratified tie-break")
            break
        rows = int(item.get("task427_new_rows", 0))
        if (int(item.get("task427_rows_loaded", 0))
                + int(item.get("task427_rows_suppressed", 0))) != rows:
            problems.append(f"{url}: row split does not conserve")
            break
    loaded = sum(int(item.get("task427_rows_loaded", 0))
                 for item in resolutions)
    suppressed = sum(int(item.get("task427_rows_suppressed", 0))
                     for item in resolutions)
    if loaded != int(record.get("task427_rows_loaded", -1)):
        problems.append(f"per-URL loaded rows sum to {loaded}, record declares "
                        f"{record.get('task427_rows_loaded')}")
    if suppressed != int(record.get("task427_rows_suppressed", -1)):
        problems.append(f"per-URL suppressed rows sum to {suppressed}, record "
                        f"declares {record.get('task427_rows_suppressed')}")
    if problems:
        result.fail("resolution-record", "; ".join(problems))
    else:
        result.ok("resolution-record",
                  f"all {len(resolutions)} mixed URL(s) resolved to a single "
                  "unit winner by the ratified tie-break",
                  urls=len(resolutions),
                  old_points=int(record["old_points"]),
                  rows=int(record["task427_new_rows"]))
        result.ok("no-half-rechunked-page",
                  "no URL is left half re-chunked and half re-embedded",
                  unresolved=0)

    lanes = ctx.pending_blocking("G6")
    if lanes:
        result.wait("pending-lanes",
                    f"{len(lanes)} unsealed lane(s) may split further URLs",
                    lanes=len(lanes))
    result.counts = {"mixed_urls": len(resolutions),
                     "rows_loaded": int(record.get("task427_rows_loaded", 0)),
                     "rows_suppressed": int(
                         record.get("task427_rows_suppressed", 0))}
    result.remedy = "resolve every mixed URL to one unit winner"
    return result


def _check_addition_identity(ctx: Context, result: GuardResult) -> None:
    """G7's addition clause: a minted ID must not already be somebody's point.

    The pairwise loops prove an addition's row IDs collide with no other
    contributing set's ROWS.  That is a different question from whether one of
    them is already a point in the collection being rebuilt: these IDs are
    ``uuid5(canonical_url # content_digest)`` values, so an ID equal to a
    membership point means same URL and same content -- an in-place re-embed
    that has declared itself an addition, and an applier told to load it
    net-new would upsert over a live vector while nothing subtracted the point
    it landed on.

    The guard never walks the corpus, so it cannot join 17,959 IDs against
    34.4M points itself.  It requires the join to have been done, against the
    membership THIS PLAN builds against -- a disjointness proof over some other
    membership dump proves nothing about this build -- and it refuses the
    absence of that proof rather than reading it as zero.
    """
    additions = ctx.additions()
    if not additions:
        return
    plan_basis = str((ctx.plan.get("membership") or {}).get("sha256") or "")
    problems: list[str] = []
    unmeasured: list[str] = []
    for row in additions:
        set_id = row["set_id"]
        declared_rows = int((row.get("vector") or {}).get("rows", -1))
        basis = (row.get("membership_basis") or {}).get("receipt_sha256")
        if not basis:
            problems.append(f"{set_id}: names no membership basis, so its "
                            "disjointness cannot be tied to this build")
        elif basis != plan_basis:
            problems.append(
                f"{set_id}: proves disjointness against membership "
                f"{basis[:16]}..., the plan builds against {plan_basis[:16]}...")

        witness = ctx.records.get(f"addition_witness:{set_id}")
        if witness is None:
            problems.append(
                f"{set_id}: no addition witness is bound, so the claim that its "
                f"{max(declared_rows, 0):,} minted IDs are absent from the "
                f"{ctx.membership_points:,}-point membership is unmeasured")
            continue
        if witness.get("schema") != SCHEMA_ADDITION_WITNESS:
            problems.append(f"{set_id}: witness has schema "
                            f"{witness.get('schema')!r}, not "
                            f"{SCHEMA_ADDITION_WITNESS}")
            continue
        if str(witness.get("set_id")) != set_id:
            problems.append(
                f"{set_id}: witness declares set_id "
                f"{witness.get('set_id')!r}; binding a witness to the wrong "
                "set is how one population is proven and another credited")
            continue

        completeness = witness.get("guard_1_enumeration_completeness") or {}
        disjointness = witness.get("guard_2_disjointness") or {}
        vector_rows = int(completeness.get("vector_rows", -1))
        split = int(completeness.get("split_net_new", -2))
        orphans = int(completeness.get("orphans_either_direction", -1))
        if not (vector_rows == split == declared_rows) or orphans != 0:
            problems.append(
                f"{set_id}: witness covers {vector_rows:,} vector rows and "
                f"{split:,} net-new rows with {orphans} orphan(s) for a set "
                f"that declares {declared_rows:,}; a witness over part of a "
                "population proves nothing about the rest of it")
        hits = int(disjointness.get("membership_hits", -1))
        if hits < 0:
            problems.append(f"{set_id}: witness reports no membership "
                            "intersection at all")
        elif hits:
            problems.append(
                f"{set_id}: {hits:,} minted ID(s) are already membership "
                "points, so the set is not a net-new addition")
        cross = disjointness.get("cross_set") or {}
        if not cross.get("checked"):
            problems.append(f"{set_id}: witness ran no cross-set intersection")
        elif cross.get("collisions"):
            problems.append(f"{set_id}: witness reports cross-set collisions "
                            f"{cross['collisions']}")
        elif cross.get("unchecked_artifacts"):
            # Not a defect and not a pass: an artifact nobody could read is an
            # open question, and WAITING is the status this suite has for that.
            unmeasured.append(
                f"{set_id} ({len(cross['unchecked_artifacts'])} artifact(s) "
                "could not be read)")

    fold_additions = (ctx.records.get("fold_report") or {}).get("additions") or {}
    for set_id, row in sorted(fold_additions.items()):
        collisions = row.get("minted_ids_in_another_sets_claims") or {}
        if collisions:
            problems.append(f"{set_id}: the fold measured {sum(collisions.values()):,} "
                            f"minted ID(s) inside another set's claim: {collisions}")
        if int(row.get("minted_ids_tombstoned") or 0):
            problems.append(f"{set_id}: {int(row['minted_ids_tombstoned']):,} "
                            "minted ID(s) are ADR-091 tombstoned points")
        inside = row.get("minted_ids_in_census_remainder")
        if inside:
            problems.append(f"{set_id}: {int(inside):,} minted ID(s) are "
                            "inside the census remainder and so are membership "
                            "points")

    if problems:
        result.fail("addition-ids-are-net-new", "; ".join(problems[:6]),
                    offending=len(problems))
    elif unmeasured:
        result.wait("addition-ids-are-net-new",
                    f"{len(unmeasured)} addition set(s) prove disjointness "
                    f"against every artifact they could read: {unmeasured}",
                    sets=len(unmeasured))
    else:
        result.ok("addition-ids-are-net-new",
                  f"all {len(additions)} addition set(s) prove every minted ID "
                  "absent from the membership this plan builds against, and "
                  "from every other set's claim, delete and collapse artifact",
                  sets=len(additions))


def guard_g7(ctx: Context) -> GuardResult:
    """Loaded IDs are distinct, and a replaced ID is upserted, never deleted."""
    result = GuardResult("G7", "identity uniqueness and delete/insert ordering")
    apply = ctx.plan.get("apply") or {}
    # ADR-101b Ruling 3a rules the convergent-identity case; it is read through
    # the same ratification gate as its precedence lines, and for the same
    # reason -- a draft is a proposal, whichever guard is asking.
    supplement_101b, ratified_101b, gate_101b = load_precedence_supplement(ctx)
    adr_101b = (supplement_101b or {}).get("adr") or "the precedence amendment"

    # -- the machine-checkable upsert-in-place property ---------------------
    problems: list[str] = []
    if apply.get("mode") != "upsert_in_place":
        problems.append(f"apply.mode is {apply.get('mode')!r}, not "
                        "'upsert_in_place'")
    if apply.get("id_collision_policy") != "upsert_in_place":
        problems.append(f"apply.id_collision_policy is "
                        f"{apply.get('id_collision_policy')!r}")
    ordering = apply.get("ordering")
    if ordering in FORBIDDEN_APPLY_ORDERINGS:
        problems.append(f"apply.ordering {ordering!r} loses or duplicates every "
                        "identity-collapsed point")
    elif not ordering:
        problems.append("apply.ordering is not declared")
    if apply.get("delete_set_excludes_loaded_ids") is not True:
        problems.append("apply.delete_set_excludes_loaded_ids is not true")
    if problems:
        result.fail("upsert-in-place-property", "; ".join(problems))
    else:
        result.ok("upsert-in-place-property",
                  "the plan declares upsert-in-place, an ordering that is not "
                  "delete-then-insert, and a delete set that excludes every "
                  "loaded ID")

    # -- the collapse population itself ------------------------------------
    ledger_collapse = int(ctx.ledger["task427_disposition"]["new_point_ids"][
        "reused_as_an_old_point_it_replaces"])
    collapse = apply.get("identity_collapse") or {}
    declared = int(collapse.get("count", -1))
    if declared != ledger_collapse:
        result.fail("collapse-population",
                    f"the plan declares {declared:,} identity-collapsed IDs; "
                    f"the ledger measured {ledger_collapse:,}",
                    declared=declared, ledger=ledger_collapse)
    elif not collapse.get("uri") or not collapse.get("sha256"):
        result.fail("collapse-population",
                    "the identity-collapsed IDs are counted but not enumerated; "
                    "the applier cannot exclude what it cannot name",
                    count=declared)
    else:
        result.ok("collapse-population",
                  f"{declared:,} identity-collapsed IDs enumerated and bound by "
                  "digest, and excluded from the delete set", count=declared)
        result.evidence.append(str(collapse["uri"]))

    delete_set = apply.get("delete_set") or {}
    if "count" not in delete_set:
        result.fail("delete-set-declared", "the plan declares no delete set")
    elif delete_set.get("excludes") != collapse.get("uri"):
        result.fail("delete-set-declared",
                    "the delete set does not name the identity-collapse "
                    "artifact as its exclusion list")
    elif delete_set["count"] is None:
        if not delete_set.get("pending_reason"):
            result.fail("delete-set-declared",
                        "the delete set has no count and no pending reason")
        else:
            result.wait("delete-set-declared",
                        "the delete set is correctly shaped -- it excludes the "
                        "identity-collapse artifact -- but its size is not "
                        f"knowable yet: {delete_set['pending_reason']}")
    else:
        result.ok("delete-set-declared",
                  f"delete set of {int(delete_set['count']):,} IDs, excluding "
                  "the identity-collapse artifact by digest",
                  count=int(delete_set["count"]))

    # -- per-set supersession conservation ---------------------------------
    # delete_set + identity_collapse must be exactly the set of old points a
    # replacement supersedes.  A point in neither is silently orphaned: its
    # vector is superseded and it is never deleted, so it survives at the old
    # dimension.  A point in both is deleted while it is being loaded.
    fold = ctx.records.get("fold_report") or {}
    conservation = fold.get("conservation") or {}
    if not conservation:
        result.wait("supersession-conserves",
                    "no fold report is bound, so no folded set's "
                    "delete_set + identity_collapse = superseded identity has "
                    "been checked")
    else:
        broken = [f"{name}: superseded {row['superseded']:,} != delete "
                  f"{row['delete_set']:,} + collapse {row['identity_collapse']:,}"
                  for name, row in sorted(conservation.items())
                  if not row.get("conserves")]
        if broken:
            result.fail("supersession-conserves", "; ".join(broken),
                        offending=len(broken))
        else:
            result.ok("supersession-conserves",
                      f"all {len(conservation)} folded set(s) satisfy "
                      "delete_set + identity_collapse = superseded, as sets and "
                      "not merely as counts", sets=len(conservation))
        overlaps = fold.get("row_id_overlaps") or {}
        # Two sets emitting one row ID is normally a defect: the row is counted
        # twice and the applier's delete-set arithmetic double-counts it.  It is
        # NOT a defect when the collision is convergent -- two lanes acquiring
        # the same bytes, so the ID is the same because the row is the same.
        # That distinction is only worth making if it is PROVEN, so a record
        # covers a pair only when it shows every colliding row embedding
        # byte-identical text.  An assertion of convergence is not convergence.
        convergent, missed = _convergent_row_ids(ctx, overlaps)
        if missed:
            result.fail("folded-row-ids-distinct",
                        f"{len(missed)} pair(s) of folded sets emit identical "
                        f"row IDs with no proven-convergent record: {missed}",
                        pairs=len(missed), convergent=len(convergent))
        elif convergent and not ratified_101b:
            result.wait("folded-row-ids-distinct",
                        f"{len(convergent)} pair(s) "
                        f"({sum(convergent.values())} row(s)) are convergent "
                        "identities -- verified byte-identical in both sets' "
                        f"embed stages -- resolved single-load by {adr_101b} "
                        f"Ruling 3a, which is not ratified: {gate_101b}",
                        pairs=len(convergent), rows=sum(convergent.values()))
        elif convergent:
            result.ok("folded-row-ids-distinct",
                      f"{len(convergent)} pair(s) "
                      f"({sum(convergent.values())} row(s)) emit an identical "
                      "row ID because they emit an identical row: each is "
                      "byte-identical in both sets' embed stages, and "
                      f"{adr_101b} Ruling 3a loads it once",
                      pairs=len(convergent), rows=sum(convergent.values()))
        else:
            result.ok("folded-row-ids-distinct",
                      "no two folded sets emit the same row ID")

    _check_addition_identity(ctx, result)

    # -- global row-ID distinctness ----------------------------------------
    rows_loaded = 0
    unknown: list[str] = []
    for row in ctx.contributing():
        adj = row.get("adjudication") or {}
        if "rows_declared" not in adj:
            unknown.append(row["set_id"])
        else:
            rows_loaded += int(adj.get("rows_loaded", 0))
    distinct = (apply.get("loaded_row_ids") or {}).get("distinct")
    if unknown:
        result.wait("row-ids-globally-distinct",
                    f"{len(unknown)} contributing set(s) have no rows yet, so "
                    f"the final loaded-ID set is not knowable: {unknown}",
                    sets=len(unknown), rows_known=rows_loaded)
    elif distinct is None:
        result.fail("row-ids-globally-distinct",
                    "the plan does not declare the distinct loaded-row-ID count")
    elif int(distinct) != rows_loaded:
        result.fail("row-ids-globally-distinct",
                    f"{rows_loaded:,} rows loaded but only {int(distinct):,} "
                    "distinct IDs -- some row ID is loaded twice",
                    rows=rows_loaded, distinct=int(distinct))
    else:
        result.ok("row-ids-globally-distinct",
                  f"all {rows_loaded:,} loaded row IDs are distinct",
                  rows=rows_loaded)

    lanes = ctx.pending_blocking("G7")
    if lanes:
        result.wait("pending-lanes",
                    f"{len(lanes)} unsealed lane(s) will mint further row IDs",
                    lanes=len(lanes))
    result.counts = {"identity_collapsed": declared,
                     "rows_loaded_known": rows_loaded,
                     "sets_without_rows": len(unknown)}
    result.remedy = ("the applier must UPSERT a replaced ID in place; it must "
                     "never delete then insert it")
    return result


def guard_g8(ctx: Context) -> GuardResult:
    """The census must be able to see every replacement, without double-counting.

    A witness the census has already consumed is disjoint by construction --
    ``census_r0_remainder.py`` raises on any replacement/replacement overlap,
    so v31 completing is the proof.  A witness the census has NOT consumed yet
    must cover the set's WINNING subset only: task 427's losing points are
    already inside another set's witness, and a witness over all 1,211,392
    claims would make the next census raise instead of subtract.

    A ``net_new_addition`` set is the one kind this guard skips, and it is
    skipped for exactly the same reason: this ledger tracks the drawdown of
    membership points, an addition draws down none, and a witness over its
    rows would name points that are not in the remainder at all.  The skip is
    scoped to this ledger and to nothing else -- the set still owes G1 its
    enumeration, G3 and G5 its conservation, and G7 the proof that its minted
    IDs are absent from the membership -- and the skipped sets are named in a
    PASS line, because a guard that quietly excludes a population is
    indistinguishable from one that never noticed it.
    """
    result = GuardResult("G8", "census witness parity")
    missing: list[str] = []
    mismatched: list[str] = []
    waiting: list[str] = []
    additions: list[str] = []
    consumed = 0
    fresh = 0
    for row in ctx.contributing():
        if row.get("mode") == "in_place_reembed":
            continue  # the census subtracts 413-A by its own point witness
        if row.get("mode") == ADDITION_MODE:
            # A replacement point witness states which MEMBERSHIP points to
            # subtract from the census remainder.  An addition claims none, so
            # a witness over it would name points that are not in the
            # remainder and census_r0_remainder.py would raise.  The skip is
            # about this ledger only -- the set still owes its enumeration
            # (G1), its conservation (G3, G5) and its identity disjointness
            # (G7), and it is listed by name below so the skip is a statement
            # rather than a silence.
            additions.append(row["set_id"])
            continue
        adj = row.get("adjudication") or {}
        witness = row.get("census_witness")
        set_id = row["set_id"]
        if witness is None:
            (waiting if row.get("pending") else missing).append(set_id)
            continue
        if witness.get("shape") != "membership/points.jsonl.gz":
            mismatched.append(f"{set_id}: witness is not in the shape "
                              "census_r0_remainder.py::_replacement_point_"
                              "witness requires")
            continue
        records = int(witness.get("records", -2))
        if witness.get("consumed_by_census"):
            consumed += 1
            expected = int((row.get("claims") or {}).get("count", -1))
            if records != expected:
                mismatched.append(
                    f"{set_id}: consumed witness carries {records} of "
                    f"{expected} claimed points")
        else:
            fresh += 1
            expected = int(adj.get("points_won", -1))
            scope = witness.get("scope")
            if scope not in {"winning_subset", "unwitnessed_winning_subset"}:
                mismatched.append(
                    f"{set_id}: a witness the census has not consumed must be "
                    f"scoped to its winning subset, not {scope!r}")
            elif scope == "unwitnessed_winning_subset":
                # The subtractable part: winning points no other witness holds.
                declared = int(witness.get("winning_points_total", -1))
                if declared != expected:
                    mismatched.append(
                        f"{set_id}: witness declares {declared} winning points, "
                        f"the adjudication says {expected}")
                elif records > expected:
                    mismatched.append(
                        f"{set_id}: witness carries {records} points, more than "
                        f"the {expected} it wins")
            elif records != expected:
                mismatched.append(
                    f"{set_id}: witness {records} != winning points {expected}")

    if missing:
        result.fail("witness-present",
                    f"{len(missing)} sealed replacement set(s) publish no census "
                    f"point witness: {missing}", offending=len(missing))
    else:
        result.ok("witness-present",
                  f"all {consumed + fresh} sealed replacement set(s) publish a "
                  "census point witness", sets=consumed + fresh)
    if mismatched:
        result.fail("witness-scope", "; ".join(mismatched[:6]),
                    offending=len(mismatched))
    elif consumed or fresh:
        result.ok("witness-scope",
                  f"{consumed} witness(es) already consumed by the census and "
                  f"disjoint by construction; {fresh} new witness(es) scoped to "
                  "their winning subset", consumed=consumed, fresh=fresh)
    if additions:
        result.ok("additions-out-of-census-scope",
                  f"{len(additions)} net-new addition set(s) publish no "
                  "replacement point witness, and must not: this guard's "
                  "ledger is the drawdown of membership points from the census "
                  "remainder, and an addition draws down nothing. Their "
                  "enumeration is G1's, their conservation G3's and G5's, "
                  f"their identity disjointness G7's: {sorted(additions)}",
                  sets=len(additions))
    if waiting:
        result.wait("witness-pending",
                    f"{len(waiting)} set(s) are not embedded yet and cannot "
                    f"publish a witness: {waiting}", sets=len(waiting))
    lanes = ctx.pending_blocking("G8")
    if lanes:
        result.wait("pending-lanes",
                    f"{len(lanes)} unsealed lane(s) still to witness",
                    lanes=len(lanes))
    result.counts = {"consumed": consumed, "fresh": fresh,
                     "missing": len(missing), "pending": len(waiting),
                     "net_new_additions_out_of_scope": len(additions)}
    result.remedy = ("mint a replacement point witness over the winning subset "
                     "only, in the membership/points.jsonl.gz shape, and re-run "
                     "the remainder census with it")
    return result


GUARDS: list[Callable[[Context], GuardResult]] = [
    guard_g1, guard_g2, guard_g3, guard_g4,
    guard_g5, guard_g6, guard_g7, guard_g8,
]

_GUARD_NAMES = {
    "guard_g1": ("G1", "claim-manifest completeness"),
    "guard_g2": ("G2", "single-claim"),
    "guard_g3": ("G3", "conservation / no-zero-claim"),
    "guard_g4": ("G4", "tombstone exclusivity"),
    "guard_g5": ("G5", "row-level loser suppression"),
    "guard_g6": ("G6", "URL-unit integrity"),
    "guard_g7": ("G7", "identity uniqueness and delete/insert ordering"),
    "guard_g8": ("G8", "census witness parity"),
}


# ---------------------------------------------------------------------------


def evaluate(root: Path, plan_path: Path) -> dict:
    """Run all eight guards.  Never raises: an input error is a FAIL."""
    started = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)
    try:
        plan = json.loads(plan_path.read_text())
        ctx: Context | None = Context(root, plan)
        setup_error = None
    except Exception as exc:  # fail closed on any input defect
        ctx, plan, setup_error = None, {}, f"{type(exc).__name__}: {exc}"

    results: list[GuardResult] = []
    for guard in GUARDS:
        guard_id, name = _GUARD_NAMES[guard.__name__]
        if ctx is None:
            result = GuardResult(guard_id, name)
            result.fail("inputs-loadable",
                        f"guard inputs could not be loaded: {setup_error}")
            results.append(result)
            continue
        try:
            results.append(guard(ctx))
        except Exception as exc:  # a guard that errors is a guard that fails
            result = GuardResult(guard_id, name)
            result.fail("guard-executed",
                        f"{type(exc).__name__}: {exc}")
            results.append(result)

    statuses = [row.status for row in results]
    summary = {status: statuses.count(status)
               for status in (PASS, WAITING, FAIL)}
    if summary[FAIL]:
        verdict, permitted = "BLOCKED", False
    elif summary[WAITING]:
        verdict, permitted = "BURNING-DOWN", False
    else:
        verdict, permitted = "GREEN", True

    pending = list((plan.get("pending_lanes") or []))
    return {
        "schema": SCHEMA_REPORT,
        "evaluated_utc": started.isoformat().replace("+00:00", "Z"),
        "plan": {"uri": str(plan_path), "build_id": plan.get("build_id"),
                 "sha256": sha256_file(plan_path) if plan_path.is_file() else None},
        "adr": "ADR-101 Ruling 2",
        "verdict": verdict,
        "flip_permitted": permitted,
        "summary": summary,
        "guards": [row.as_dict() for row in results],
        "pending_lane_accounting": {
            "lanes": len(pending),
            "points": sum(int(lane.get("points") or 0) for lane in pending),
            "detail": pending,
        },
        "setup_error": setup_error,
    }


def render(report: dict) -> str:
    lines = [
        f"ADR-101 assembly guards -- {report['verdict']} "
        f"(flip {'PERMITTED' if report['flip_permitted'] else 'PROHIBITED'})",
        f"plan: {report['plan']['build_id']}  evaluated {report['evaluated_utc']}",
        "",
        f"{'id':<4}{'status':<10}{'guard':<48}findings",
        "-" * 96,
    ]
    for guard in report["guards"]:
        first = guard["findings"][0] if guard["findings"] else ""
        lines.append(f"{guard['id']:<4}{guard['status']:<10}"
                     f"{guard['name'][:46]:<48}{first[:44]}")
        for check in guard["checks"]:
            mark = {PASS: "  +", FAIL: "  x", WAITING: "  ~"}[check["status"]]
            lines.append(f"{mark} {check['name']:<40} {check['detail'][:100]}")
    summary = report["summary"]
    lines += ["", f"PASS={summary[PASS]}  WAITING={summary[WAITING]}  "
                  f"FAIL={summary[FAIL]}",
              f"pending lanes: {report['pending_lane_accounting']['lanes']} "
              f"({report['pending_lane_accounting']['points']:,} points)"]
    return "\n".join(lines)


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--plan", type=Path, required=True,
                        help="the builder's assembly plan (aw.assembly-plan.v1)")
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                        help="root the plan's relative URIs resolve against")
    parser.add_argument("--out", type=Path,
                        help="write the machine-readable report here")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    report = evaluate(args.root, args.plan)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True)
          if args.format == "json" else render(report))
    if report["summary"][FAIL]:
        return 1
    return 2 if report["summary"][WAITING] else 0


if __name__ == "__main__":
    sys.exit(main())
