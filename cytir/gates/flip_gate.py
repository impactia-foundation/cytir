#!/usr/bin/env python3
"""The flip gate: the ratified battery, and the sealed PASS a flip requires.

Task 529, from Tim's 2026-09-06 ruling: *hold the flip until no regression vs
prod on the dense and sparse legs* -- "no class loses vs prod on any leg
without a named, ruled reason."

WHY THE FUSED NUMBER IS NOT ENOUGH
----------------------------------
Task 524 scored the candidate and the incumbent at **276/629 h@1 each** on the
full stack -- a dead heat -- while the dense-only leg underneath was **141
against 189**. Tasks 526/527 then found the mechanism on two families: the
rebuild had changed WHAT TEXT they embed, and BM25 was carrying the query. A
gate that scores only the fusion cannot see a leg that stopped working, and a
flip is precisely the change that swaps every leg at once.

And two instruments, because they have disagreed: on the same build, 629
queries said +3 golds and the held-out 1,418 said -36 (task 516 §7.2).

WHAT THIS MODULE IS
-------------------
`data/flip-gate-contract.v1.json` states the battery -- 3 legs x 2 sides x 2
instruments = 12 arms, per class, plus tier-2, plus latency at prod's own
deadline. This module

  * `evaluate(arms, contract)` -- normalises the two sealed arm shapes the
    project already produces (`run_arm_524.py` output and `seal_arm.py`
    output), refuses a battery with a missing arm / a moved instrument / a
    nonzero error count / two sides on one collection, and then applies the
    regression-free rule per (leg, instrument, class, metric);
  * `seal(verdict, path)` -- writes the verdict with every arm re-digested;
  * `assert_sealed_pass(...)` -- the refusal `cytir.gates.flip_config_verify`
    and the flip-day runbook call. No sealed PASS, no flip.

It does NOT score anything. Scoring is `run_arm_524.py` (task 524) and
`cytir/gates/authority.py`, which run against a replica and seal their own
arms; a gate that both produces and judges its evidence is a filter selecting
on what it reports.

There is no waiver flag. A loss is covered only by an entry in the contract's
`named_rulings`, which is a legislative change (memory: legislative-prs).

WHAT A NAMED RULING DOES, AND WHAT IT CANNOT DO (task 590)
----------------------------------------------------------
Tim ruled the flip on 2026-09-16 over a battery that reads HOLD on 87 per-class
cells. His ruling is one entry in `named_rulings`, and it changes the VERDICT
LINE only:

    flip gate: PASS — HOLD overridden by named ruling FLIP-2026-09-16-aw_e8 (87 cells)
      LOSS  1418:full-stack authoritative.hit1: 149 -> 145 (-4)   [overridden by …]

Every cell still prints its raw incumbent -> candidate pair, the seal still
carries `measured_verdict: FAIL`, and the ruling is bound three ways: to the
candidate collection it was ruled on, to the size of each loss it was shown
(a re-run that loses MORE on a covered cell is a fresh refusal), and to an
evidence obligation that `assert_sealed_pass` refuses the flip over until it
lands. A ruling that covered any future loss on those cells would be the
waiver flag this module refuses to have, wearing Tim's name.

TWO PRESERVATION TIERS, NOT ONE (ADR-107 amendment 2, task 592)
---------------------------------------------------------------
ADR-098 Ruling 1 writes one preservation law per operation class: the
content-pack contract gates INCREMENTAL applies, and "any full rebuild,
ship+flip, or promotion" is gated by the sealed GLOBAL preservation manifest
instead. This module used to demand the first of a flip, which is the wrong
tier and which no flip battery in the record has ever satisfied. With
`tier2.scope` set to `incremental-content-pack-apply` the tier-2 clause steps
aside for a flip and `rebuild_tier` takes its place: the gate checks, cheaply
and at the front, the two facts `deploy/scripts/alias_swap.sh` re-checks
deeply at the alias swap -- the manifest is `status: sealed` and it names the
candidate collection. Both keys fail CLOSED: an absent `scope` leaves tier-2
exactly as it was, and an absent, unreadable or unsealed manifest is a
refusal, never a skip.

THE NOISE FLOOR, IF IT IS EVER ENABLED (ADR-107 amendment 2 Option 2)
---------------------------------------------------------------------
`regression_free.noise_floor` is inert unless `enabled` is true. Enabled, it
widens a named ruling's per-cell bound by that cell's MEASURED re-run movement
between two seals of the same comparison on unchanged collections -- and by
nothing else. It cannot create coverage: a cell with no named ruling is still
a refusal, and every non-regression clause is untouched.

Usage
-----
    python3 -m cytir.gates.flip_gate --evaluate \\
        --arm 629:full-stack:incumbent=524-artifacts/full-stack-twin.json \\
        --arm 629:dense-only:candidate=524-artifacts/dense-only-build.json \\
        ... --tier2 <sealed tier-2 pass> --seal data/flip-gate-verdict.v1.json

    python3 -m cytir.gates.flip_gate --assert-pass --collection <twin-collection>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

from cytir.core.paths import REPO_ROOT, data_path
from cytir.gates import flip_cells as FC  # noqa: E402 -- ruling 31 (T21): component
# arms read against null pairs, and named accepted losses on the strict bar.
from cytir.core import serving_config as sc  # noqa: E402 -- task 552: the ONE home for the
# flip's fusion-weights pointer (FLIP_FUSION_WEIGHTS_FILE/SHA256). This
# module never keeps its own copy of that sha256; see
# `_candidate_weights_problems` below.

SCHEMA = "aw.flip-gate-contract.v1"
VERDICT_SCHEMA = "aw.flip-gate-verdict.v1"
DEFAULT_CONTRACT = "data/flip-gate-contract.v1.json"

REPO = REPO_ROOT

PASS = "PASS"
FAIL = "FAIL"

# The one `tier2.scope` value that takes the content-pack contract out of a
# flip battery (ADR-098 Ruling 1 / ADR-107 am.2). Any other value, and the
# absence of the key, leaves tier-2 required exactly as before.
INCREMENTAL_SCOPE = "incremental-content-pack-apply"


class FlipGateError(Exception):
    """The battery is incomplete, inconsistent, or shows a regression."""


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load(path: str | pathlib.Path | None = None) -> dict:
    p = pathlib.Path(path) if path else data_path("flip-gate-contract.v1.json")
    doc = json.loads(p.read_text())
    if doc.get("schema") != SCHEMA:
        raise FlipGateError(f"{p}: not {SCHEMA}")
    doc["_path"] = str(p)
    return doc


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def latency_ceiling_s(repo: pathlib.Path = REPO,
                      contract: dict | None = None) -> float:
    """Prod's own deadline, read from the file the monitor reads.

    `deploy/aw-monitor.env` is the one home for it; a second copy here would
    let the gate and the alarm disagree about what 'too slow' means.
    """
    src = ((contract or {}).get("latency") or {}).get(
        "source", "deploy/aw-monitor.env#AW_MONITOR_MAX_SMOKE_LATENCY")
    filename, _, key = src.partition("#")
    key = key or "AW_MONITOR_MAX_SMOKE_LATENCY"
    p = repo / filename
    if not p.exists():
        raise FlipGateError(
            f"{filename} does not exist, so prod's latency deadline cannot be "
            "read. It is deliberately not duplicated in this module.")
    for line in p.read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{key}="):
            return float(line.split("=", 1)[1].strip().strip('"'))
    raise FlipGateError(f"{filename} does not set {key}")


# --------------------------------------------------------------------------
# arm normalisation -- two sealed shapes the project already produces
# --------------------------------------------------------------------------
LEG_READ_SCHEMAS = ("aw.prod-leg-read.v2", "aw.scratch-leg-read.v1")


def _leg_read_arm(doc: dict, instrument: str) -> dict:
    """An arm from a leg read: `cytir.gates.prod_legs` (incumbent) or `cytir.gates.scratch_legs` (candidate).

    Class counts are computed here, from the rows, the same way for both sides. A read that aborted,
    was refused or is marked invalid is no arm: refused, never scored short.
    """
    if doc.get("abort") or doc.get("invalid"):
        raise FlipGateError(f"leg read aborted or invalid ({doc.get('abort')!r} / {doc.get('invalid')!r}): "
                            "not an arm")
    prod = doc["schema"] == "aw.prod-leg-read.v2"
    rows = doc.get("rows") or []
    ok = [r for r in rows if r.get("status") == "OK"]
    k = doc.get("k") or 10
    buckets: dict[str, list] = {}
    norm = []
    for r in ok:
        rank = r.get("gold_rank") if prod else r.get("rank")
        cls = r.get("cls")
        if cls is None:
            raise FlipGateError(f"leg read row {r.get('id')!r} carries no class (cls)")
        rid = str(r.get("id"))
        if rid.startswith(f"{cls}|"):
            rid = rid[len(cls) + 1:]
        buckets.setdefault(str(cls), []).append(rank)
        norm.append({"id": rid, "cls": str(cls), "rank": rank})
    classes = {c: {"hit1": sum(1 for x in rs if x == 1),
                   "hitk": sum(1 for x in rs if x is not None and x <= k),
                   "mrr": round(sum(1.0 / x for x in rs if x) / len(rs), 4),
                   "n_gold": len(rs)}
               for c, rs in buckets.items()}
    emb = doc.get("embedder") or {}
    digest = emb.get("sha256") if emb.get("qvec_cache") else None
    if digest is None and str(instrument) == "629":   # the 629 reads its pinned file, no vector cache
        chooser = REPO / "data" / "eval_tasks.v2.yaml"
        if not chooser.is_file():
            raise FlipGateError(f"the 629 chooser set is missing: {chooser} (the public cut ships none)")
        digest = sha256_file(chooser)
    target = doc.get("target_before") or {}
    return {
        "shape": doc["schema"],
        "instrument_digest": digest,
        "n_total": len(rows),
        "n_errors": len(rows) - len(ok),
        "cache_miss": 0,
        "collection": target.get("collection") if prod else doc.get("collection"),
        "latency_p95_ms": None,
        "classes": classes,
        "overall": None,
        "per_gold": FC.per_gold_from_rows(norm, class_key="cls"),
        "weights_sha256": ((doc.get("effective_config") or {}).get("fusion_weights") or {}).get("sha256"),
        "gate_set_version": doc.get("gate_set_version"),
    }


def normalise_arm(doc: dict, *, instrument: str, leg: str, side: str,
                  path: str | None = None) -> dict:
    """One arm, in one shape, whatever it was sealed as.

    Refuses an unrecognised shape rather than guessing which keys mean what:
    a silently mis-read arm scores as a regression or a pass at random.
    """
    if "by_type" in doc and "eval_sha256" in doc:          # run_arm_524.py
        classes = {k: {"hit1": v.get("hit1"), "hitk": v.get("hitk"),
                       "mrr": v.get("mrr"), "n_gold": v.get("n")}
                   for k, v in (doc.get("by_type") or {}).items()}
        arm = {
            "shape": "aw.arm-524",
            "instrument_digest": doc.get("eval_sha256"),
            "n_total": doc.get("n_total"),
            "n_errors": doc.get("n_errors"),
            "cache_miss": 0,
            "collection": ((doc.get("server_health") or {}).get("config")
                           or {}).get("collection"),
            "latency_p95_ms": (doc.get("latency_ms_warm") or {}).get("p95"),
            "classes": classes,
            "overall": doc.get("overall"),
            # ruling 31: per-gold ranks, which the strict bar needs to consult
            # an accepted loss (`regression_free.accepted_losses`).
            "per_gold": (FC.per_gold_from_rows(doc["rows"])
                         if isinstance(doc.get("rows"), list) else None),
        }
    elif doc.get("schema") in LEG_READ_SCHEMAS:             # prod_legs / scratch_legs (task 926)
        arm = _leg_read_arm(doc, instrument)
    elif "sets" in doc and "cache" in doc:                  # seal_arm.py
        classes = {}
        for k, v in (doc.get("sets") or {}).items():
            classes[k] = {"hit1": v.get("h1"), "hitk": v.get("h10"),
                          "mrr": v.get("mrr"), "n_gold": v.get("n_gold")}
        arm = {
            "shape": "aw.arm-519-seal",
            "instrument_digest": (doc.get("cache") or {}).get("sha256"),
            "n_total": sum((v.get("n") or 0)
                           for v in (doc.get("sets") or {}).values()),
            "n_errors": 0,
            "cache_miss": doc.get("cache_miss"),
            "collection": (doc.get("collection") or {}).get("name"),
            "latency_p95_ms": None,
            "classes": classes,
            "overall": None,
            # A seal carries no per-gold ranks; `load_arm` reads them from the
            # leg json the seal pins by sha256 (`leg_sha256`), when present.
            "per_gold": None,
        }
    else:
        raise FlipGateError(
            f"arm {instrument}:{leg}:{side} ({path or '<inline>'}) is neither "
            "a run_arm_524 result (by_type + eval_sha256) nor a seal_arm seal "
            "(sets + cache). Refusing to guess which keys are the scores.")
    # task 552: which fusion weights this arm actually served, whichever of
    # the two independent stamps the arm carries -- `harness_weights` (the
    # shell driver's own resolved $W, task 551, present on the raw run_arm_524
    # output) or `fusion_weights` (gate_authority.py's own process-env
    # self-report, task 544, present on a seal_arm.py seal). Neither key
    # exists in both shapes, so this is a fallback between two disjoint
    # sources, not a preference between two that could disagree here -- a
    # real cross-check between them happens at the harness level instead
    # (task 551 §3's stamper docstring).
    weights = doc.get("harness_weights") or doc.get("fusion_weights") or {}
    arm["weights_sha256"] = weights.get("sha256") or arm.get("weights_sha256")
    # ADR-107 amendment 3: which ROW SET this arm scored. The cache digest
    # above pins the query VECTORS and cannot see the rows, so an arm run
    # before the 24 contaminated queries were removed and an arm run after
    # carry the same digest and the same cache_miss = 0. gate_authority.py
    # stamps this on every leg it writes; the contract pins it per instrument.
    arm["gate_set_version"] = doc.get("gate_set_version") or arm.get("gate_set_version")
    arm.update(instrument=instrument, leg=leg, side=side, path=path)
    return arm


def load_arm(spec: str, repo: pathlib.Path = REPO) -> dict:
    """`<instrument>:<leg>:<side>=<path>`."""
    key, _, path = spec.partition("=")
    parts = key.split(":")
    if len(parts) != 3 or not path:
        raise FlipGateError(
            f"--arm {spec!r} is not '<instrument>:<leg>:<side>=<path>'")
    instrument, leg, side = parts
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = repo / p
    doc = json.loads(p.read_text())
    # Record the arm REPO-RELATIVE whenever it lives in the repo. A verdict is
    # sealed in whatever worktree ran the battery and is read back on flip day
    # from the main checkout (and, for anyone reproducing this, from a fresh
    # clone -- memory: cold-start-reproducibility). An absolute
    # `/.claude/worktrees/agent-…/` path turns `assert_sealed_pass`'s
    # re-digest of the evidence into "arm artifact no longer exists", which
    # reads as tampering and is really just a moved checkout.
    try:
        rel = str(p.resolve().relative_to(pathlib.Path(repo).resolve()))
    except ValueError:
        rel = str(p)
    arm = normalise_arm(doc, instrument=instrument, leg=leg, side=side,
                        path=rel)
    arm["sha256"] = sha256_file(p)
    if arm["shape"] == "aw.arm-519-seal":
        arm["per_gold"], arm["per_gold_source"] = _seal_per_gold(doc, p)
    return arm


def load_null_pair(spec: str, repo: pathlib.Path = REPO) -> dict:
    """`<name>:<kind>:<instrument>:<leg>=<arm A>,<arm B>` (ruling 31, T21).

    Both arms are loaded exactly as battery arms are (same shapes, same
    digests); the gate computes B - A itself rather than trusting a delta file.
    """
    key, _, paths = spec.partition("=")
    parts = key.split(":")
    pa, _, pb = paths.partition(",")
    if len(parts) != 4 or not pa or not pb:
        raise FlipGateError(
            f"--null-pair {spec!r} is not "
            "'<name>:<kind>:<instrument>:<leg>=<arm A>,<arm B>'")
    name, kind, instrument, leg = parts
    return {"name": name, "kind": kind, "instrument": instrument, "leg": leg,
            "a": load_arm(f"{instrument}:{leg}:null-a={pa}", repo),
            "b": load_arm(f"{instrument}:{leg}:null-b={pb}", repo)}


def _seal_per_gold(doc: dict, seal_path: pathlib.Path):
    """Per-gold ranks of a held-out seal, from the leg json the seal pins.

    A seal (`seal_arm.py`) records the leg json it sealed by `leg_sha256`, at
    an absolute path on the box that ran it. The leg json is committed next to
    the seal (`X.seal.json` beside `X.json`), so it is read from there and
    accepted ONLY when its sha256 equals the seal's: a per-gold list that is
    not the one sealed is not evidence. Absent or mismatched -> (None, reason);
    an accepted loss on that arm then cannot be consulted, and the cell stays a
    refusal (ruling 31; fail closed).
    """
    want = doc.get("leg_sha256")
    name = seal_path.name
    if not want or not name.endswith(".seal.json"):
        return None, "the seal names no leg json"
    leg = seal_path.with_name(name[: -len(".seal.json")] + ".json")
    if not leg.exists():
        return None, f"leg json {leg.name} is not beside the seal"
    if sha256_file(leg) != want:
        return None, f"leg json {leg.name} sha256 is not the seal's leg_sha256"
    return FC.per_gold_from_sets(json.loads(leg.read_text()).get("sets") or {}), leg.name


# --------------------------------------------------------------------------
# the battery
# --------------------------------------------------------------------------
def _candidate_weights_problems(arms: list[dict]) -> list[str]:
    """Refuse a battery whose candidate full-stack arm did not measure the
    weights the flip would actually ship (task 552).

    Task 551 found the harnesses defaulting to a stale candidate whenever
    `AW_BUILD_WEIGHTS` was unset; task 552 found the SEALED EVIDENCE side of
    the same defect -- `cytir/core/serving_config.py:FLIP_FUSION_WEIGHTS_FILE`
    named `candidate519` while task 538's gate arms measured `candidate533`,
    with nothing in the repo able to say the two had diverged. This is the
    mechanism: import the one home directly (never a second copy of its
    sha256) and compare it to what each candidate full-stack arm actually
    recorded serving.

    Only the `full-stack` leg is checked -- `dense-only`/`sparse-only` legs
    always serve `fusion_weights.dense_only.json`/`.sparse_only.json`
    regardless of which candidate is promoted; they are not governed by this
    pointer and must not be newly blocked by it (the same scoping task 551
    §3 used for its guard).
    """
    problems: list[str] = []
    want = sc.FLIP_FUSION_WEIGHTS_SHA256
    for arm in arms:
        if arm.get("side") != "candidate" or arm.get("leg") != "full-stack":
            continue
        label = f"{arm.get('instrument')}:{arm.get('leg')}:{arm.get('side')}"
        got = arm.get("weights_sha256")
        if not got:
            problems.append(
                f"{label}: carries no weights_sha256 -- it was not run "
                "through the harness that stamps `harness_weights` "
                "(stamp_harness_weights_551.py, task 551) or "
                "`fusion_weights` (cytir/gates/authority.py, task 544), so "
                "this arm cannot prove which fusion weights it served and "
                "is not evidence for a flip that ships a specific file.")
            continue
        if got != want:
            problems.append(
                f"{label}: served fusion weights sha256 {got}, but the "
                f"flip's ratified weights (cytir/core/serving_config.py:"
                f"FLIP_FUSION_WEIGHTS_SHA256, file "
                f"{sc.FLIP_FUSION_WEIGHTS_FILE!r}) is {want}. An arm scored "
                "with different weights than the flip would ship is not "
                "evidence for this flip.")
    return problems


class _RulingExceeded(Exception):
    """A named ruling covers the cell, but the measured loss outgrew it."""


def _ruling_for(contract: dict, leg, instrument, klass, metric, *,
                delta: float | None = None, eps: float = 0.0,
                candidate_collection: str | None = None):
    """The named ruling covering ONE (leg, instrument, class, metric) cell.

    Two shapes, because a ruling is written at the altitude the founder ruled
    at. The original per-cell shape -- `{leg, instrument, class, metric,
    ruling, cite}` -- says "this one cell". Task 590's shape carries an
    `overrides` LIST under one `id`, because Tim ruled once, on one battery,
    over 87 cells at a time; writing that as 87 unrelated waivers would lose
    the fact that they are one decision and would let 86 of them survive a
    revocation of the 87th.

    A ruling of either shape is bound, never open:

      * `cite` is mandatory -- a ruling nobody can find is not a ruling;
      * an override entry that carries the pair it was ruled over covers the
        cell only while the observed loss is NO WORSE than the ruled one
        (`delta >= ruled_delta`), so a re-run that degrades further is a
        fresh refusal and goes back to Tim;
      * `candidate_collection`, when the ruling names one, must be the
        collection the candidate arm actually scored. A ruling on `aw_e8` is
        not a ruling on whatever is staged next.

    MULTIPLE RULINGS, SCOPED BY COLLECTION, because two flips can regress the
    same-SHAPED cell for unrelated reasons (task 651: both
    `629/full-stack/coding/hit1` and `629/full-stack/research/hitk` regressed
    in the aw_e3 -> aw_e8 flip AND, independently, in the aw_e8 -> aw_e9 one).
    Only rulings bound to THIS `candidate_collection` (or unbound) are ever
    consulted; a cell no same-collection ruling covers is unruled, full stop
    -- there is no cross-collection "this shape is ruled elsewhere" diagnostic
    (task 651 removed it: with a second real ruling in the contract it fired
    on ordinary aw_e8 batteries whose cells coincidentally matched an aw_e9
    cell shape, which is a false alarm, not a safety net).
    """
    want = (leg, str(instrument), klass, metric)
    rulings = (contract.get("regression_free") or {}).get(
        "named_rulings") or []

    def _match(cell: dict) -> bool:
        got = (cell.get("leg"), str(cell.get("instrument")),
               cell.get("class"), cell.get("metric"))
        return got == want

    def _mandatory_fields(r: dict) -> None:
        if not r.get("cite"):
            raise FlipGateError(
                f"named_rulings entry for {leg}/{instrument}/{klass}/"
                f"{metric} carries no `cite`. A ruling nobody can find is "
                "not a ruling.")
        if r.get("overrides") is not None and not r.get("id"):
            raise FlipGateError(
                "a named ruling with an `overrides` list carries no `id`. "
                "The id is what the verdict line names and what "
                "`assert_sealed_pass` re-checks against the contract in "
                "force at flip time.")

    def _resolve(r: dict, cell: dict):
        _mandatory_fields(r)
        ruled_delta = _ruled_delta(cell, metric)
        allowance = noise_allowance(contract, leg, instrument, klass, metric)
        if (ruled_delta is not None and delta is not None
                and delta < ruled_delta - allowance - eps):
            # A ruling exists for this cell and the loss has since grown
            # past it. That is NOT covered, and the caller must say so in
            # the refusal rather than report the cell as unruled: the two
            # are different conversations with Tim.
            raise _RulingExceeded(
                f"named ruling {_ruling_id(r)} covers {leg}/{instrument}/"
                f"{klass}/{metric} at {cell.get('incumbent')} -> "
                f"{cell.get('candidate')} ({ruled_delta:+g})"
                + (f", widened by a measured noise floor of {allowance:+g}"
                   if allowance else "")
                + f"; this battery measures {delta:+g}, a LARGER loss than "
                "was ruled. A ruling is granted over the loss it was "
                "shown, not over the cell forever.")
        return r

    # Rulings bound to this candidate, or bound to nothing. A ruling never
    # covers a cell for the WRONG collection -- that guarantee is what the
    # `want_coll` filter below enforces, unconditionally.
    #
    # Earlier (single-ruling) versions of this function also raised a loud
    # FlipGateError when a cell's SHAPE matched a DIFFERENT collection's
    # ruling, as a diagnostic. Task 651 removed that: with two real rulings
    # in the contract, two UNRELATED rebuild generations turned out to share
    # cell shapes by coincidence (629/full-stack/research/hitk regressed in
    # both aw_e3->aw_e8, task 524, and aw_e8->aw_e9, task 637, for unrelated
    # reasons), and the diagnostic fired on ordinary aw_e8 batteries that have
    # nothing to do with aw_e9. A cell no same-collection ruling covers is
    # simply unruled -- reported as a plain, named problem below, exactly as
    # it would be if no other collection's ruling existed at all.
    for r in rulings:
        want_coll = r.get("candidate_collection")
        if (want_coll and candidate_collection
                and want_coll != candidate_collection):
            continue
        overrides = r.get("overrides")
        for cell in ([r] if overrides is None else overrides):
            if _match(cell):
                return _resolve(r, cell)
    return None



def noise_allowance(contract: dict, leg, instrument, klass, metric) -> float:
    """How far this cell's ruled loss may move before the ruling stops covering it.

    Zero unless `regression_free.noise_floor.enabled` is true -- the block can
    sit in the contract, fully derived and reviewable, without changing a
    verdict (ADR-107 amendment 2 Option 2 is Tim's to take or leave).

    Enabled, the allowance is that cell's MEASURED movement between two seals
    of the same comparison on collections that were not written between them.
    It is never a fraction, a percentage or a default: a cell with no measured
    pair gets nothing, which is why this returns 0.0 rather than a tolerance.
    """
    nf = (contract.get("regression_free") or {}).get("noise_floor") or {}
    if not nf.get("enabled"):
        return 0.0
    want = (str(instrument), leg, klass, metric)
    for cell in nf.get("cells") or []:
        got = (str(cell.get("instrument")), cell.get("leg"),
               cell.get("class"), cell.get("metric"))
        if got == want:
            return abs(float(cell.get("allowance") or 0.0))
    return 0.0


def _rebuild_tier_manifest(rb: dict, candidate_collection: str | None) -> str | None:
    """Which global manifest governs THIS candidate.

    `manifest` is the default and governs everything not named. ADR-098
    amendment 1's ruling is bound to one collection, so its manifest is bound
    to one collection too: a candidate with an entry in `candidate_manifests`
    is checked against that entry, and every other candidate keeps being
    checked against the default -- which is `data/production-rebuild-lock.v1.json`,
    still blocked, still for `aw_e3`. A ruling for one flip can therefore never
    quietly become the preservation basis for the next one.
    """
    per_candidate = rb.get("candidate_manifests") or {}
    if candidate_collection and isinstance(per_candidate, dict):
        named = per_candidate.get(candidate_collection)
        if named:
            return named
    return rb.get("manifest")


def _preservation_ruling(contract: dict, candidate_collection: str | None,
                         repo: pathlib.Path) -> dict | None:
    """The ruling a PASS rests on, for the seal. Never a verdict of its own.

    `_preservation_problems` decides whether the flip may proceed. This only
    records WHAT it proceeded on, so a sealed verdict read a year from now
    says whether the preservation basis was a proof or a promise.
    """
    rb = contract.get("rebuild_tier") or {}
    rel = _rebuild_tier_manifest(rb, candidate_collection)
    if not rb.get("required") or not rel:
        return None
    p = pathlib.Path(rel)
    if not p.is_absolute():
        p = repo / p
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if doc.get("status") != "ruled":
        return None
    ruling = doc.get("ruling") or {}
    return {
        "manifest": rel,
        "id": ruling.get("id"),
        "collection": ruling.get("collection"),
        "ruled_by": ruling.get("ruled_by"),
        "ruled_on": ruling.get("ruled_on"),
        "ratified_by": ruling.get("ratified_by"),
        "does_not_cover": ruling.get("does_not_cover"),
        "carried_gaps": [g.get("id") for g in (ruling.get("carried_gaps") or [])],
    }


def _preservation_problems(contract: dict, candidate_collection: str | None,
                           repo: pathlib.Path) -> list[str]:
    """The rebuild tier's gate: a sealed global manifest naming the candidate.

    ADR-098 Ruling 1: "any full rebuild, ship+flip, or promotion keeps
    requiring the sealed global preservation manifest". This is the cheap
    front-of-process form of the check `deploy/scripts/alias_swap.sh` runs
    deeply (`preservation_contract.py verify --deep --expect-collection`)
    at the alias swap itself, so a flip that would be refused at the swap is
    refused by the battery instead -- days earlier, on a laptop, for free.

    Fails closed on every ambiguity: no manifest path, no file, unreadable
    JSON, wrong schema, any status but `sealed` or a candidate-bound `ruled`
    (ADR-098 amendment 1), a manifest for some other collection, or a battery
    whose candidate collection cannot be determined.
    """
    rb = contract.get("rebuild_tier") or {}
    if not rb.get("required"):
        return []
    # The label, not the contract's `adr` prose: a refusal line is read in a
    # terminal at 06:00, and the reasoning lives in the contract for whoever
    # goes looking.
    adr = "ADR-098 Ruling 1 (rebuild tier)"
    rel = _rebuild_tier_manifest(rb, candidate_collection)
    if not rel:
        return [f"{adr}: the rebuild tier is required and the contract names "
                "no `rebuild_tier.manifest`. A gate with no input is not a "
                "gate."]
    p = pathlib.Path(rel)
    if not p.is_absolute():
        p = repo / p
    if not p.exists():
        return [f"{adr}: no global preservation manifest at {rel}. A ship+flip "
                "is a rebuild-tier operation and requires one."]
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        return [f"{adr}: {rel} could not be read as JSON ({exc}). An "
                "unreadable gate input is a refusal, not a skip."]
    schema = rb.get("manifest_schema") or "aw.preservation-manifest.v1"
    if doc.get("schema") != schema:
        return [f"{adr}: {rel} is schema {doc.get('schema')!r}, not "
                f"{schema!r}."]
    out: list[str] = []
    status = doc.get("status")
    if status == "ruled":
        # ADR-098 amendment 1. A ruling is admissible here only if it is bound
        # to THIS candidate and says who ratified it; everything else about it
        # -- the carried gaps, the rollback, the receipts -- is checked deeply
        # by `preservation_contract.verify_manifest` at the alias swap. This
        # gate does not re-adjudicate a ruling, and it does not accept one that
        # names another collection or nobody at all.
        ruling = doc.get("ruling")
        if not isinstance(ruling, dict):
            out.append(
                f"{adr}: {rel} is `ruled` and carries no `ruling` block. A "
                "status that is granted by a ruling cannot be granted "
                "without one.")
        else:
            ratified = str(ruling.get("ratified_by") or "").strip()
            if not ratified:
                out.append(
                    f"{adr}: the ruling in {rel} names no `ratified_by`. A "
                    "ruling nobody merged is a waiver.")
            elif ratified.upper().startswith(("PENDING", "TBD", "TODO")):
                out.append(
                    f"{adr}: the ruling in {rel} says its ratification is "
                    f"{ratified!r}. A ruling that still says it is pending "
                    "has not been ratified.")
            if candidate_collection and ruling.get("collection") != candidate_collection:
                out.append(
                    f"{adr}: the ruling in {rel} covers "
                    f"{ruling.get('collection')!r}, and the candidate is "
                    f"{candidate_collection!r}.")
    elif status != "sealed":
        gaps = doc.get("blocking_gaps") or []
        names = ", ".join(str(g.get("id") or g)[:40] for g in gaps[:5])
        out.append(
            f"{adr}: the global preservation manifest {rel} is "
            f"{status!r}, not `sealed`"
            + (f" ({len(gaps)} open gap(s): {names})" if gaps else "")
            + ". `deploy/scripts/alias_swap.sh` refuses the alias swap on "
              "the same fact, deeply; this is the same refusal, earlier.")
    named = ((doc.get("collection") or {}).get("name")
             if isinstance(doc.get("collection"), dict) else None)
    if not candidate_collection:
        out.append(f"{adr}: the battery does not name ONE candidate "
                   "collection, so the preservation manifest cannot be "
                   "checked against it.")
    elif named != candidate_collection:
        out.append(
            f"{adr}: the global preservation manifest {rel} names collection "
            f"{named!r}, and the candidate is {candidate_collection!r}. A "
            "manifest for one collection is not a manifest for another.")
    return out


def _ruled_delta(cell: dict, metric: str) -> float | None:
    """The loss the ruling was granted over, or None if it stated no bound."""
    inc, cand = cell.get("incumbent"), cell.get("candidate")
    if inc is None or cand is None:
        return None
    return float(cand) - float(inc)


def _ruling_id(r: dict | None) -> str | None:
    if not r:
        return None
    return r.get("id") or r.get("ruling")


def pending_evidence(contract: dict, ruling_ids) -> list[str]:
    """Unmet evidence obligations on the named rulings a verdict rests on.

    A ruling may be granted while one measurement is still in flight -- Tim
    ruled the flip on 2026-09-16 with task 586 (the same battery restated with
    the hand-authored `refresh_manifest` rows removed) still running. Recording
    that as prose would make it a promise; recording it here makes it a
    refusal, and `assert_sealed_pass` is where it bites, so the battery can be
    sealed today and the alias cannot move until the result lands.

    Fails closed: any status other than `landed`, or a `landed` entry with no
    `cite`, is unmet.
    """
    wanted = {str(i) for i in ruling_ids}
    out: list[str] = []
    for r in ((contract.get("regression_free") or {})
              .get("named_rulings") or []):
        rid = _ruling_id(r)
        if rid is None or str(rid) not in wanted:
            continue
        for item in r.get("pending_evidence") or []:
            status = str(item.get("status") or "").lower()
            if status == "landed" and item.get("cite"):
                continue
            out.append(
                f"named ruling {rid}: evidence obligation "
                f"{item.get('id') or item.get('task') or '<unnamed>'} is "
                f"{status or 'unstated'}"
                + ("" if status != "landed" else " but carries no `cite`")
                + f" -- {item.get('requires') or 'no requirement stated'}")
    return out


def _component_reading(inst_key, leg, klass, metric, ci, cc, delta,
                       pairs: list, eps: float) -> dict:
    """One losing component cell, read against its null pairs (ruling 31).

    Gold units (`cytir.gates.flip_cells`): counts as counts, MRR as the
    reciprocal-rank sum. Never a refusal: the verdict carries every reading,
    flagged or not, with the numbers that decided it.
    """
    d = FC.delta_gold_units(ci, cc, metric)
    n_gold = ci.get("n_gold") or 0
    eps_gold = eps * n_gold if metric == "mrr" else 0.0
    cells = [(name, kind, deltas.get(klass, {}).get(metric))
             for name, kind, deltas in pairs]
    sp = FC.spread([deltas for _, _, deltas in pairs], metric)
    read = FC.read_component_cell(d if d is not None else float("-inf"),
                                  [c for _, _, c in cells], sp, eps_gold)
    return {"instrument": inst_key, "leg": leg, "class": klass,
            "metric": metric, "incumbent": ci.get(metric),
            "candidate": cc.get(metric), "delta": round(delta, 6),
            "n_gold": n_gold,
            "delta_gold_units": None if d is None else round(d, 6),
            "nulls": [{"name": n, "kind": k,
                       "delta_gold_units": None if c is None else round(c, 6)}
                      for n, k, c in cells],
            "spread": round(sp, 6), **read}


def _accepted_cover(contract, inst_key, leg, klass, metric, ci, inc, cand,
                    delta, eps) -> dict | None:
    """Whether named accepted losses cover one losing strict cell (ruling 31).

    None when no accepted loss names this cell. Else {covered, residual,
    applied, problems}: the class's loss with the accepted golds' own changes
    removed must clear the strict bar, and every accepted gold must sit no
    lower than the rank it was shown falling to.
    """
    try:
        entries = FC.accepted_entries(
            contract, instrument=inst_key, leg=leg, klass=klass,
            candidate_collection=cand.get("collection"))
    except FC.CellRuleError as exc:
        raise FlipGateError(str(exc)) from None
    if not entries:
        return None
    d = FC.delta_gold_units(ci, (cand.get("classes") or {}).get(klass) or {},
                            metric)
    res = FC.residual_after_accepted(
        d, entries, (inc.get("per_gold") or {}).get(klass),
        (cand.get("per_gold") or {}).get(klass), metric)
    n_gold = ci.get("n_gold") or 0
    tol = -(eps * n_gold) if metric == "mrr" else 0.0
    covered = (not res["problems"] and len(res["applied"]) == len(entries)
               and res["residual"] >= tol)
    if not covered and not res["problems"]:
        res["problems"].append(
            f"with the accepted gold(s) removed the class still loses "
            f"{res['residual']:+g} (gold units)")
    return {"covered": covered, "residual": round(res["residual"], 6),
            "applied": res["applied"], "problems": res["problems"]}


def _null_pair_readings(null_pairs: list[dict], contract: dict,
                        component_legs: set[str]) -> tuple[dict, list[str]]:
    """{(instrument, leg): [(name, kind, deltas)]} for the usable null pairs,
    and the reasons any supplied pair is unusable (ruling 31, T21).

    A null pair is two arms of one instrument x leg that differ in no variable
    under test (`component_arms.null_pair_kinds`). It meets the same instrument
    integrity as a battery arm: an A/B and its null scored on two query sets
    are not one experiment. An unusable pair is reported, never a refusal: the
    reading it would have fed is informational.
    """
    instruments = contract.get("instruments") or {}
    block = (contract.get("regression_free") or {}).get("component_arms") or {}
    kinds = set(block.get("null_pair_kinds") or FC.NULL_PAIR_KINDS)
    out: dict = {}
    notes: list[str] = []
    for np_ in null_pairs or []:
        label = (f"null pair {np_.get('name')} ({np_.get('kind')}) "
                 f"{np_.get('instrument')}:{np_.get('leg')}")
        if np_.get("kind") not in kinds:
            notes.append(f"{label}: kind is not one of {sorted(kinds)}")
            continue
        if np_.get("leg") not in component_legs:
            notes.append(f"{label}: not a component leg; the strict bar reads "
                         "no null")
            continue
        inst = instruments.get(str(np_.get("instrument"))) or {}
        want = inst.get("sha256") or inst.get("query_cache_sha256")
        bad = [side for side in ("a", "b")
               if want and (np_.get(side) or {}).get("instrument_digest") != want]
        want_v = inst.get("gate_set_version")
        bad += [f"{side} (gate-set)" for side in ("a", "b")
                if want_v and (np_.get(side) or {}).get("gate_set_version")
                != want_v]
        bad += [f"{side} (n_errors)" for side in ("a", "b")
                if ((np_.get(side) or {}).get("n_errors") or 0) > 0]
        if bad:
            notes.append(f"{label}: arm(s) {bad} not scored on the pinned "
                         "instrument, or with errors")
            continue
        out.setdefault((str(np_["instrument"]), np_["leg"]), []).append(
            (np_.get("name"), np_.get("kind"),
             FC.null_pair_deltas(np_["a"], np_["b"])))
    return out, notes


def evaluate(arms: list[dict], contract: dict, *,
             tier2_seal: dict | None = None,
             repo: pathlib.Path = REPO,
             null_pairs: list[dict] | None = None) -> dict:
    """The whole battery. Returns a verdict; never raises on a regression."""
    problems: list[str] = []
    findings: list[dict] = []
    overridden: list[dict] = []
    accepted: list[dict] = []
    component: list[dict] = []
    applied_ids: set[str] = set()
    instruments = contract.get("instruments") or {}
    rf = contract.get("regression_free") or {}
    metrics = list(rf.get("metrics") or ["hit1", "hitk", "mrr"])
    eps = float(rf.get("mrr_epsilon") or 0.0)
    # Ruling 31 (Tim 2026-10-12), T21: the strict bar reads `strict_legs`
    # only; every other leg is a COMPONENT arm, read against its null pairs
    # and informational. A contract without `strict_legs` is the contract
    # before ruling 31: every leg strict.
    try:
        split = FC.component_block(contract)
    except FC.CellRuleError as exc:
        raise FlipGateError(str(exc)) from None
    all_legs = {req["leg"] for req in contract.get("required_arms") or []}
    strict_legs = set(split[0]) if split else all_legs
    component_legs = all_legs - strict_legs
    nulls, null_notes = _null_pair_readings(null_pairs or [], contract,
                                            component_legs)

    index = {(a["instrument"], a["leg"], a["side"]): a for a in arms}
    if len(index) != len(arms):
        problems.append("two arms were supplied for the same "
                        "(instrument, leg, side); the battery is ambiguous")

    # -- the candidate full-stack arm served the flip's actual weights ----
    problems.extend(_candidate_weights_problems(arms))

    # -- every required arm present --------------------------------------
    missing = []
    for req in contract.get("required_arms") or []:
        key = (str(req["instrument"]), req["leg"], req["side"])
        if key not in index:
            missing.append(":".join(key))
    if missing:
        problems.append(
            f"{len(missing)} required arm(s) were not run: {missing}. The "
            "battery is stated in full so a missing arm refuses the flip "
            "instead of being a question nobody asked.")

    # -- arm integrity ----------------------------------------------------
    for arm in arms:
        inst = instruments.get(str(arm["instrument"]))
        label = f"{arm['instrument']}:{arm['leg']}:{arm['side']}"
        if inst is None:
            problems.append(f"{label}: instrument {arm['instrument']!r} is not "
                            "in the contract")
            continue
        want = inst.get("sha256") or inst.get("query_cache_sha256")
        if want and arm.get("instrument_digest") != want:
            problems.append(
                f"{label}: scored against instrument digest "
                f"{arm.get('instrument_digest')!r}, the pinned one is {want!r}. "
                "Two arms scored against different query sets are not an A/B.")
        want_version = inst.get("gate_set_version")
        if want_version and arm.get("gate_set_version") != want_version:
            problems.append(
                f"{label}: scored on gate-set "
                f"{arm.get('gate_set_version') or '<unstamped>'}, the pinned "
                f"row set is {want_version}. The cache digest pins the query "
                "vectors, not the rows -- two arms scored on two row sets are "
                "not an A/B (ADR-107 amendment 3).")
        if (arm.get("n_errors") or 0) > 0:
            problems.append(
                f"{label}: n_errors={arm['n_errors']}. An error is not an "
                "empty -- a leg that timed out scores like a leg that "
                "answered wrongly.")
        if arm.get("cache_miss") not in (0, None):
            problems.append(
                f"{label}: cache_miss={arm['cache_miss']}; the two legs no "
                "longer share query vectors.")

    # -- one (instrument, leg) pair is ONE comparison ----------------------
    # `required_arms` names each pair twice, once per side. Iterating it
    # directly reported every regression twice, which reads as two defects.
    pairs = []
    for req in contract.get("required_arms") or []:
        pair = (str(req["instrument"]), req["leg"])
        if pair not in pairs:
            pairs.append(pair)

    # -- the two sides must be two collections ----------------------------
    for inst_key, leg in pairs:
        inc = index.get((inst_key, leg, "incumbent"))
        cand = index.get((inst_key, leg, "candidate"))
        if inc and cand and inc.get("collection") and cand.get("collection"):
            if inc["collection"] == cand["collection"]:
                problems.append(
                    f"{inst_key}:{leg}: both sides report "
                    f"collection {inc['collection']!r}. ADR-098 Ruling 3 -- "
                    "a twin that is secretly production is the measured "
                    "failure (task 447), and it looks exactly like this.")

    # -- the regression-free rule -----------------------------------------
    for inst_key, leg in pairs:
        inc = index.get((inst_key, leg, "incumbent"))
        cand = index.get((inst_key, leg, "candidate"))
        if not inc or not cand:
            continue
        inst = instruments.get(inst_key) or {}
        skip = set(inst.get("tripwire_sets_without_golds") or [])
        for klass, ci in sorted(inc["classes"].items()):
            if klass in skip:
                continue
            cc = cand["classes"].get(klass)
            if cc is None:
                problems.append(
                    f"{inst_key}:{leg}: class {klass!r} is scored on the "
                    "incumbent and absent from the candidate")
                continue
            if not (ci.get("n_gold") or 0):
                continue  # no gold to lose; excluded, and the contract says so
            for metric in metrics:
                a, b = ci.get(metric), cc.get(metric)
                if a is None or b is None:
                    continue
                delta = b - a
                tol = -eps if metric == "mrr" else 0.0
                if delta < tol and leg in component_legs:
                    component.append(_component_reading(
                        inst_key, leg, klass, metric, ci, cc, delta,
                        nulls.get((inst_key, leg)) or [], eps))
                    continue
                if delta < tol:
                    covered = _accepted_cover(
                        contract, inst_key, leg, klass, metric, ci, inc, cand,
                        delta, eps)
                    if covered is not None and covered["covered"]:
                        finding = {
                            "instrument": inst_key, "leg": leg,
                            "class": klass, "metric": metric,
                            "incumbent": a, "candidate": b,
                            "delta": round(delta, 6),
                            "accepted_losses": covered["applied"],
                            "residual_delta": covered["residual"],
                        }
                        findings.append(finding)
                        accepted.append(finding)
                        continue
                    outgrew = None
                    try:
                        ruling = _ruling_for(
                            contract, leg, inst_key, klass, metric,
                            delta=delta,
                            eps=(eps if metric == "mrr" else 0.0),
                            candidate_collection=cand.get("collection"))
                    except _RulingExceeded as exc:
                        ruling, outgrew = None, str(exc)
                    finding = {
                        "instrument": inst_key, "leg": leg, "class": klass,
                        "metric": metric, "incumbent": a, "candidate": b,
                        "delta": round(delta, 6),
                        "ruling": (ruling or {}).get("ruling"),
                        "ruling_id": _ruling_id(ruling),
                        "cite": (ruling or {}).get("cite"),
                    }
                    # A cell that is covered only because the noise floor is
                    # on says so, in the seal, next to its numbers. A verdict
                    # that leant on a tolerance and does not record it is the
                    # invisible state this file exists to refuse.
                    lean = noise_allowance(contract, leg, inst_key, klass,
                                           metric)
                    if ruling is not None and lean:
                        finding["noise_allowance"] = lean
                    findings.append(finding)
                    if ruling is not None:
                        overridden.append(finding)
                        applied_ids.add(_ruling_id(ruling))
                    else:
                        problems.append(outgrew or (
                            f"{inst_key}:{leg}: class {klass} loses on "
                            f"{metric} ({a} -> {b}, {delta:+g}) with no named "
                            "ruling"
                            + (" and no accepted loss that covers it ("
                               + "; ".join(covered["problems"]) + ")"
                               if covered is not None else "")
                            + ". Tim 2026-09-06: no class loses vs prod on "
                            "any leg without a named, ruled reason."))

    # -- known contamination in an instrument -----------------------------
    # Task 586 found 24 queries in the 1,418 held-out set whose answers were
    # written into the corpus by a freshness lane. Removing them is Tim's, and
    # deferred on purpose (it flatters the incumbent). Until then the fact
    # travels with every verdict this gate seals, rather than living in a
    # contract section a battery-runner may not read: a yardstick's known
    # defect belongs next to the number it produced.
    contamination = []
    for inst_key, inst in instruments.items():
        c = inst.get("contamination")
        if not c:
            continue
        if not any(a["instrument"] == inst_key for a in arms):
            continue
        contamination.append({
            "instrument": inst_key,
            "n": len(c.get("eval_ids") or []),
            "disposition": c.get("disposition"),
            "cite": c.get("cite"),
        })

    # -- latency at prod's own deadline -----------------------------------
    lat = contract.get("latency") or {}
    ceiling_s = latency_ceiling_s(repo, contract)
    for leg in lat.get("applies_to_legs") or []:
        for inst_key in instruments:
            cand = index.get((inst_key, leg, "candidate"))
            if not cand:
                continue
            p95 = cand.get("latency_p95_ms")
            if p95 is None:
                continue
            if p95 / 1000.0 > ceiling_s:
                problems.append(
                    f"{inst_key}:{leg}: candidate p95 {p95:.0f} ms exceeds "
                    f"prod's deadline {ceiling_s} s "
                    f"({lat.get('source')})")

    # -- preservation: the right tier for the operation ---------------------
    # ADR-098 Ruling 1 keeps two laws. A flip is a rebuild-tier operation, so
    # the content-pack contract below steps aside for it once the contract
    # scopes it (`tier2.scope`), and the global manifest gate takes its place.
    # An unscoped contract keeps the old behaviour: the default is the strict
    # one, because a preservation clause that disappears when a field is
    # missing is a clause that disappears on a typo.
    cand_colls = {a.get("collection") for a in arms
                  if a.get("side") == "candidate" and a.get("collection")}
    one_candidate = sorted(cand_colls)[0] if len(cand_colls) == 1 else None
    problems.extend(_preservation_problems(contract, one_candidate, repo))
    preservation_ruling = _preservation_ruling(contract, one_candidate, repo)

    t2 = contract.get("tier2") or {}
    if t2.get("required") and t2.get("scope") != INCREMENTAL_SCOPE:
        if not tier2_seal:
            problems.append(
                "no sealed tier-2 contract PASS was presented "
                f"({t2.get('instrument')}, {t2.get('adr')}). A failing source "
                "is excluded by regenerating the pack, never waived.")
        elif str(tier2_seal.get("verdict", "")).upper() != PASS:
            problems.append(
                f"the tier-2 seal reports "
                f"{tier2_seal.get('verdict')!r}, not PASS")

    # -- the verdict line, and the measurement it never edits ---------------
    # A named ruling changes what the battery CONCLUDES. It does not change,
    # soften, hide or re-round a single number: every overridden cell is still
    # in `regressions` with its raw incumbent -> candidate pair, and
    # `measured_verdict` records what this battery would have said with an
    # empty `named_rulings`. Someone reading the seal a year from now must be
    # able to see the HOLD that was ruled over, not just the flip that
    # happened.
    verdict = PASS if not problems else FAIL
    measured = FAIL if (problems or overridden or accepted) else PASS
    ids = sorted(i for i in applied_ids if i)
    if not overridden:
        line = verdict
    elif verdict == PASS:
        line = (f"{PASS} — HOLD overridden by named ruling "
                f"{', '.join(ids)} ({len(overridden)} cells)")
    else:
        line = (f"{FAIL} — HOLD overridden by named ruling "
                f"{', '.join(ids)} ({len(overridden)} cells), "
                f"{len(problems)} problem(s) it does not cover")
    accepted_applied = sorted(
        {tuple(e[k] for k in ("gold", "instrument", "leg", "class", "ruling",
                              "candidate_collection"))
         for f in accepted for e in f["accepted_losses"]})
    if accepted:
        rulings = sorted({k[4] for k in accepted_applied})
        line += (f"; {len(accepted)} strict cell(s) covered by accepted "
                 f"loss(es) {[k[0] for k in accepted_applied]} "
                 f"({', '.join(rulings)})")
    flagged = [c for c in component if c["flagged"]]
    if component:
        line += (f"; component arms (informational): {len(flagged)} of "
                 f"{len(component)} losing cell(s) flagged against the null "
                 "pairs")
    pending = pending_evidence(contract, ids)

    return {
        "schema": VERDICT_SCHEMA,
        "verdict": verdict,
        "verdict_line": line,
        "measured_verdict": measured,
        "named_rulings_applied": ids,
        "overridden": overridden,
        "accepted_losses_applied": [dict(zip(
            ("gold", "instrument", "leg", "class", "ruling",
             "candidate_collection"), k)) for k in accepted_applied],
        "accepted": accepted,
        "strict_legs": sorted(strict_legs),
        "component_readings": component,
        "component_flagged": len(flagged),
        "null_pairs": [{"name": n.get("name"), "kind": n.get("kind"),
                        "instrument": n.get("instrument"), "leg": n.get("leg"),
                        "a": {k: (n.get("a") or {}).get(k)
                              for k in ("path", "sha256", "collection")},
                        "b": {k: (n.get("b") or {}).get(k)
                              for k in ("path", "sha256", "collection")}}
                       for n in (null_pairs or [])],
        "null_pair_notes": null_notes,
        "pending_evidence": pending,
        "preservation_ruling": preservation_ruling,
        "instrument_contamination": contamination,
        "contract": contract.get("_path", DEFAULT_CONTRACT),
        "arms": [{k: a.get(k) for k in
                  ("instrument", "leg", "side", "shape", "collection",
                   "instrument_digest", "gate_set_version", "path",
                   "sha256", "n_total", "n_errors", "latency_p95_ms",
                   "weights_sha256")}
                 for a in arms],
        "arms_required": len(contract.get("required_arms") or []),
        "arms_present": len(arms),
        "latency_ceiling_s": ceiling_s,
        "regressions": findings,
        "problems": problems,
    }


# --------------------------------------------------------------------------
# sealing and the refusal
# --------------------------------------------------------------------------
def seal(verdict: dict, path: str | pathlib.Path, *,
         collection: str | None = None,
         serving_config: str | None = None) -> pathlib.Path:
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = REPO / p
    doc = dict(verdict)
    cands = {a.get("collection") for a in verdict.get("arms") or []
             if a.get("side") == "candidate" and a.get("collection")}
    doc["candidate_collection"] = collection or (
        sorted(cands)[0] if len(cands) == 1 else None)
    if collection is None and len(cands) > 1:
        raise FlipGateError(
            f"the candidate arms name {len(cands)} different collections "
            f"{sorted(cands)}; a verdict is bound to ONE candidate. Pass "
            "--collection to say which, or re-run the battery against one.")
    doc["serving_config"] = serving_config
    p.write_text(json.dumps(doc, indent=1) + "\n")
    return p


def assert_sealed_pass(path: str | pathlib.Path | None = None, *,
                       collection: str | None = None,
                       serving_config: str | None = None,
                       repo: pathlib.Path = REPO) -> dict:
    """No sealed PASS, no flip. Raises `FlipGateError` with the reason.

    Re-digests every arm the verdict names: a verdict that cites an artifact
    which has since changed is describing a battery nobody ran.
    """
    contract = load()
    p = pathlib.Path(path) if path else (
        repo / (contract.get("seal") or {}).get(
            "path", "data/flip-gate-verdict.v1.json"))
    if not p.exists():
        raise FlipGateError(
            f"no sealed flip-gate verdict at {p}. The flip is refused: the "
            "ratified battery (data/flip-gate-contract.v1.json) has not been "
            "run and sealed. `python3 -m cytir.gates.flip_gate --evaluate ...`")
    doc = json.loads(p.read_text())
    if doc.get("schema") != VERDICT_SCHEMA:
        raise FlipGateError(f"{p}: not {VERDICT_SCHEMA}")
    if doc.get("verdict") != PASS:
        problems = doc.get("problems") or []
        raise FlipGateError(
            f"{p}: verdict is {doc.get('verdict')!r}, not PASS"
            + (f" — {len(problems)} problem(s):\n  - "
               + "\n  - ".join(problems[:10]) if problems else ""))
    if collection and doc.get("candidate_collection") != collection:
        raise FlipGateError(
            f"{p}: sealed for candidate collection "
            f"{doc.get('candidate_collection')!r}, but the flip would serve "
            f"{collection!r}. A verdict is bound to the collection the arms "
            "actually scored.")
    if serving_config and doc.get("serving_config") not in (None,
                                                            serving_config):
        raise FlipGateError(
            f"{p}: sealed for serving config {doc.get('serving_config')!r}, "
            f"not {serving_config!r}")
    for arm in doc.get("arms") or []:
        ap, want = arm.get("path"), arm.get("sha256")
        if not ap or not want:
            continue
        f = pathlib.Path(ap)
        if not f.is_absolute():
            f = repo / f
        if not f.exists():
            raise FlipGateError(
                f"{p}: arm artifact {ap} no longer exists; the verdict cites "
                "a battery that cannot be re-read.")
        got = sha256_file(f)
        if got != want:
            raise FlipGateError(
                f"{p}: arm artifact {ap} sha256 {got} != sealed {want}. The "
                "evidence changed after the verdict was sealed.")

    for np_ in doc.get("null_pairs") or []:
        for side in ("a", "b"):
            ref = np_.get(side) or {}
            ap, want = ref.get("path"), ref.get("sha256")
            if not ap or not want:
                continue
            f = pathlib.Path(ap)
            if not f.is_absolute():
                f = repo / f
            if not f.exists() or sha256_file(f) != want:
                raise FlipGateError(
                    f"{p}: null-pair arm {ap} is missing or changed after "
                    "the verdict was sealed; the component readings it "
                    "recorded can no longer be re-read.")

    # Ruling 31: a PASS that rests on accepted losses is re-checked against
    # the contract in force NOW, as a named ruling is: an accepted loss that
    # was withdrawn, re-bound to another collection, or narrowed to a smaller
    # fall after sealing no longer covers the cell it covered.
    for f in doc.get("accepted") or []:
        for e in f.get("accepted_losses") or []:
            try:
                live = FC.accepted_entries(
                    contract, instrument=e.get("instrument"),
                    leg=e.get("leg"), klass=e.get("class"),
                    candidate_collection=doc.get("candidate_collection"))
            except FC.CellRuleError as exc:
                raise FlipGateError(f"{p}: {exc}") from None
            match = [x for x in live if str(x["gold"]) == str(e.get("gold"))
                     and x["ruling"] == e.get("ruling")]
            if not match:
                raise FlipGateError(
                    f"{p}: the sealed PASS rests on accepted loss "
                    f"{e.get('gold')} ({e.get('ruling')}) on "
                    f"{e.get('instrument')}:{e.get('leg')} {e.get('class')}, "
                    "which the contract in force no longer carries for this "
                    "candidate. Re-run the battery against the current "
                    "contract.")
            rc = e.get("rank_candidate")
            if FC.accepted_outside_top_k(match[0]):
                continue
            if rc is None or rc > int(match[0]["shown"]["candidate_rank"]):
                raise FlipGateError(
                    f"{p}: accepted loss {e.get('gold')} sealed at candidate "
                    f"rank {rc}, beyond the rank the contract now accepts "
                    f"({match[0]['shown']['candidate_rank']}).")

    # task 552: re-check against the CURRENT ratified weights pointer, not
    # just the one in force when the verdict was sealed. A verdict sealed
    # PASS against candidate533 is not evidence for a flip that would now
    # ship candidate6xx -- the pointer moving after sealing is exactly the
    # kind of drift this function re-digests every other arm for.
    weights_problems = _candidate_weights_problems(doc.get("arms") or [])
    if weights_problems:
        raise FlipGateError(
            f"{p}: sealed PASS no longer matches the ratified fusion "
            "weights pointer (cytir/core/serving_config.py:"
            f"FLIP_FUSION_WEIGHTS_SHA256) -- {'; '.join(weights_problems)}")

    # task 590: a PASS that rests on named rulings is re-checked against the
    # contract IN FORCE NOW, for the same reason as the weights pointer above.
    # A seal is a record of one evaluation; the standard it was judged against
    # is legislative and can move under it. Three ways it can move, all fatal:
    # the ruling is revoked, it is narrowed so a cell it used to cover no
    # longer is, or an evidence obligation attached to it has not landed yet.
    ids = [i for i in (doc.get("named_rulings_applied") or []) if i]
    if ids:
        known = {_ruling_id(r) for r in
                 ((contract.get("regression_free") or {})
                  .get("named_rulings") or [])}
        gone = sorted(set(ids) - known)
        if gone:
            raise FlipGateError(
                f"{p}: sealed PASS rests on named ruling(s) {gone} which the "
                "contract no longer carries. The flip is refused: the ruling "
                "that covered those losses was withdrawn after sealing.")
        for cell in doc.get("overridden") or []:
            try:
                still = _ruling_for(
                    contract, cell.get("leg"), cell.get("instrument"),
                    cell.get("class"), cell.get("metric"),
                    delta=cell.get("delta"),
                    eps=float((contract.get("regression_free") or {})
                              .get("mrr_epsilon") or 0.0)
                    if cell.get("metric") == "mrr" else 0.0,
                    candidate_collection=doc.get("candidate_collection"))
            except _RulingExceeded as exc:
                raise FlipGateError(f"{p}: {exc}") from None
            if still is None:
                raise FlipGateError(
                    f"{p}: the sealed PASS overrode {cell.get('instrument')}:"
                    f"{cell.get('leg')} {cell.get('class')}."
                    f"{cell.get('metric')}, and the contract in force now does "
                    "not cover that cell. Re-run the battery against the "
                    "current contract.")
        unmet = pending_evidence(contract, ids)
        if unmet:
            raise FlipGateError(
                f"{p}: the named ruling(s) this PASS rests on carry evidence "
                "obligations that have not landed. The flip is refused until "
                "they do:\n  - " + "\n  - ".join(unmet))
    return doc


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _outside_text(e: dict, contract: dict):
    """The candidate rank of an accepted gold, or 'outside top k' for one shown outside the top k."""
    if e.get("outside_top_k") and e.get("rank_candidate") is None:
        return f"outside top {contract.get('k') or 10}"
    return e["rank_candidate"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", default=None)
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--arm", action="append", default=[],
                    help="<instrument>:<leg>:<side>=<path to a sealed arm>")
    ap.add_argument("--tier2", default=None,
                    help="path to the sealed tier-2 contract result")
    ap.add_argument("--null-pair", action="append", default=[],
                    help="<name>:<kind>:<instrument>:<leg>=<arm A>,<arm B> -- "
                         "a null pair for a component leg (ruling 31); the "
                         "gate reads B - A per class")
    ap.add_argument("--seal", default=None,
                    help="write the verdict here (default: the contract's "
                         "seal.path)")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--serving-config", default=None)
    ap.add_argument("--assert-pass", action="store_true",
                    help="refuse unless a sealed PASS exists and still "
                         "matches its arms")
    ap.add_argument("--verdict", default=None)
    args = ap.parse_args()

    if args.assert_pass:
        try:
            doc = assert_sealed_pass(args.verdict, collection=args.collection,
                                     serving_config=args.serving_config)
        except FlipGateError as exc:
            print(f"FLIP REFUSED: {exc}", file=sys.stderr)
            return 1
        print(f"OK: sealed flip-gate PASS for "
              f"{doc.get('candidate_collection')!r} "
              f"({doc.get('arms_present')}/{doc.get('arms_required')} arms).")
        return 0

    if not args.evaluate:
        ap.error("nothing to do: pass --evaluate or --assert-pass")

    contract = load(args.contract)
    arms = [load_arm(spec) for spec in args.arm]
    tier2 = json.loads(pathlib.Path(args.tier2).read_text()) if args.tier2 \
        else None
    verdict = evaluate(arms, contract, tier2_seal=tier2,
                       null_pairs=[load_null_pair(x) for x in args.null_pair])

    print(f"flip gate: {verdict.get('verdict_line') or verdict['verdict']} "
          f"({verdict['arms_present']}/{verdict['arms_required']} arms, "
          f"latency ceiling {verdict['latency_ceiling_s']} s)")
    # Every losing cell prints as a LOSS with its raw numbers whether or not a
    # ruling covers it. A ruling moves the verdict line above; it never edits
    # a line below.
    for f in verdict["regressions"]:
        ruled = (f"   [overridden by {f['ruling_id']}]"
                 if f.get("ruling_id") else "")
        print(f"  LOSS  {f['instrument']}:{f['leg']} {f['class']}."
              f"{f['metric']}: {f['incumbent']} -> {f['candidate']} "
              f"({f['delta']:+g}){ruled}")
    for f in verdict.get("accepted") or []:
        print(f"  ACCEPTED  {f['instrument']}:{f['leg']} {f['class']}."
              f"{f['metric']}: {f['incumbent']} -> {f['candidate']} "
              f"({f['delta']:+g}); residual {f['residual_delta']:+g} after "
              f"{[(e['gold'], e['rank_incumbent'], _outside_text(e, contract)) for e in f['accepted_losses']]}"
              f" [{', '.join(sorted({e['ruling'] for e in f['accepted_losses']}))}]")
    for c in verdict.get("component_readings") or []:
        nulls = ", ".join(f"{n['name']} {n['delta_gold_units']}"
                          for n in c["nulls"]) or "none"
        print(f"  {'FLAG' if c['flagged'] else 'info'}  {c['instrument']}:"
              f"{c['leg']} {c['class']}.{c['metric']}: {c['incumbent']} -> "
              f"{c['candidate']} ({c['delta_gold_units']:+g} gold units; "
              f"nulls {nulls}; spread {c['spread']:g}) -- {c['reason']}")
    for note in verdict.get("null_pair_notes") or []:
        print(f"  NULL-PAIR {note}")
    for prob in verdict["problems"]:
        print(f"  FAIL {prob}")
    if verdict.get("overridden"):
        print(f"  measured verdict: {verdict['measured_verdict']} (HOLD) — "
              f"{len(verdict['overridden'])} cell(s) covered by "
              f"{', '.join(verdict['named_rulings_applied'])}; the "
              "measurements above are unchanged by it")
    for item in verdict.get("pending_evidence") or []:
        print(f"  PENDING {item}")
    for c in verdict.get("instrument_contamination") or []:
        # The flag does not retire when the queries leave: a verdict has to
        # say which set produced it, and "24 were removed from this one" is
        # part of that answer.
        head = ("REMEDIED" if (c.get("disposition") or "").upper() == "REMOVED"
                else "CONTAMINATED")
        print(f"  {head} instrument {c['instrument']}: {c['n']} "
              f"quer(ies) whose answers were written into the corpus "
              f"({c['disposition']}; {c['cite']})")

    out = args.seal or (contract.get("seal") or {}).get("path")
    if out:
        p = seal(verdict, out, collection=args.collection,
                 serving_config=args.serving_config)
        print(f"sealed -> {p}")
    return 0 if verdict["verdict"] == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
