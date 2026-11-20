"""Two cell rules of the flip gate beyond the strict bar: component arms read against null pairs, and
named accepted losses (Ruling 31 (Tim 2026-10-12), plan T21, ADR-043 amendment A10).

The flip gate (`data/flip-gate-contract.v1.json`, scored by the flip gate's `evaluate`) compares the
candidate with the incumbent per leg x instrument x class x metric. Until ruling 31 every leg met the
strict bar: tolerance 0, a loss refuses unless a named ruling covers that cell. Task 863 then showed
that the NULL pairs themselves -- two arms that differ in no variable under test -- move one gold or
one reciprocal rank in single classes (rebuild null `ch-stats-010`, `ho-fr-009`; id null `en14`
rank 1 -> 2). A zero-tolerance bar on a component arm therefore flags the instrument, not the
change. This module holds the two rules the contract now states:

1. COMPONENT arms (`regression_free.component_arms`, the legs NOT in `regression_free.strict_legs`)
   are informational. A losing cell is FLAGGED only when it is worse than BOTH null pairs' cells of
   the same instrument x leg x class, by more than the nulls' own spread:

       flagged  <=>  d_arm < min(0, d_null_a, d_null_b) - spread - eps

   in GOLD UNITS (h@1 and h@10 as gold counts; MRR as the reciprocal-rank SUM, i.e. the class's mean
   MRR x its gold count), where `spread` = the largest |d| either null pair shows in ANY class of
   the same instrument x leg x metric, and `eps` = the contract's `mrr_epsilon` x the class's gold
   count for MRR (the arms publish MRR to 4 decimals), 0 for counts. The 0 inside the min makes the
   reading one-sided, as ADR-043 A3 rule 4: a null that GAINED is noise, not a raised bar, so this
   rule is never stricter than "zero minus the spread". A flag refuses nothing; it is printed,
   sealed and reported to Tim. A cell without two usable null readings is flagged with the reason.

2. ACCEPTED LOSSES (`regression_free.accepted_losses`): a strict (full-stack) cell that loses is
   covered when, after removing the named golds' own rank changes, the rest of the class does not
   lose. Each entry names one gold, its instrument, leg, class, the ruling string and the candidate
   collection, and the rank it was SHOWN falling to: a re-run where that gold falls further is a
   fresh refusal, never coverage. A gold shown falling OUT of the top k carries
   `shown.candidate_rank: null` with `shown.candidate_outside_top_k: true` (never one without the
   other): it is scored beyond k, and any candidate position at re-check satisfies the bound. Per-gold ranks of both arms are required; without them the entry
   cannot be consulted and the cell stays a refusal.

Pure functions; no I/O. The gate's arm loader passes the per-gold ranks it read.
"""
from __future__ import annotations

COUNT_METRICS = ("hit1", "hitk")
METRICS = ("hit1", "hitk", "mrr")
#: the closed list of null-pair kinds (ADR-043 A3 rule 6: a new way of producing a null is an ADR
#: change, not a string a run may pass in)
NULL_PAIR_KINDS = ("rebuild", "id", "reread")


class CellRuleError(ValueError):
    """The contract's component-arm or accepted-loss block is malformed."""


# ------------------------------------------------------------------------------------------------ #
# units
# ------------------------------------------------------------------------------------------------ #

def gold_units(cls: dict, metric: str) -> float | None:
    """One class reading in gold units: counts as they are, MRR as mean x gold count."""
    v = cls.get(metric)
    if v is None:
        return None
    if metric == "mrr":
        n = cls.get("n_gold")
        if not n:
            return None
        return float(v) * float(n)
    return float(v)


def delta_gold_units(a: dict, b: dict, metric: str) -> float | None:
    """b - a for one class, in gold units; None when either side has no reading."""
    x, y = gold_units(a, metric), gold_units(b, metric)
    if x is None or y is None:
        return None
    return y - x


def null_pair_deltas(arm_a: dict, arm_b: dict) -> dict[str, dict[str, float | None]]:
    """Per class, per metric: arm_b - arm_a in gold units (both normalised arms of one null pair)."""
    out: dict[str, dict[str, float | None]] = {}
    for klass, ca in (arm_a.get("classes") or {}).items():
        cb = (arm_b.get("classes") or {}).get(klass)
        if cb is None or not (ca.get("n_gold") or 0):
            continue
        out[klass] = {m: delta_gold_units(ca, cb, m) for m in METRICS}
    return out


def spread(deltas: list[dict[str, dict[str, float | None]]], metric: str) -> float:
    """The nulls' own spread on one instrument x leg x metric: the largest |d| any null pair shows
    in any class. Which class a null's noisy gold lands in is chance, so one class's own null
    reading is too small a sample to be its noise; the leg's is."""
    vals = [abs(c[metric]) for d in deltas for c in d.values() if c.get(metric) is not None]
    return max(vals) if vals else 0.0


# ------------------------------------------------------------------------------------------------ #
# rule 1: component cells against the null pairs
# ------------------------------------------------------------------------------------------------ #

def read_component_cell(d_arm: float, null_cells: list[float | None], spread_: float,
                        eps: float = 0.0, *, min_nulls: int = 2) -> dict:
    """Read one losing component cell (gold units) against its null pairs' cells of the same class.

    Returns {flagged, bar, reason}. `bar` = min(0, nulls...) - spread; flagged when d_arm < bar - eps.
    Fewer than `min_nulls` usable null readings -> flagged, with the reason: an unread cell is
    reported, never silently passed.
    """
    usable = [x for x in null_cells if x is not None]
    if len(usable) < min_nulls:
        return {"flagged": True, "bar": None,
                "reason": f"{len(usable)} usable null reading(s), {min_nulls} required: "
                          "the cell cannot be read against its nulls"}
    bar = min([0.0, *usable]) - spread_
    flagged = d_arm < bar - eps
    return {"flagged": flagged, "bar": round(bar, 6),
            "reason": (f"worse than every null pair by more than their spread "
                       f"({d_arm:+g} < {bar:+g})") if flagged else
                      (f"inside the null band ({d_arm:+g} >= {bar:+g})")}


def component_block(contract: dict) -> tuple[tuple[str, ...], dict] | None:
    """(strict legs, component block) when the contract declares strict legs; None = every leg is
    strict (the contract before ruling 31). A declared split with no component block is malformed:
    a leg that is neither strict nor read against nulls would be read by nothing."""
    rf = contract.get("regression_free") or {}
    strict = rf.get("strict_legs")
    if strict is None:
        return None
    if not isinstance(strict, list) or not strict:
        raise CellRuleError("regression_free.strict_legs must be a non-empty list of legs")
    block = rf.get("component_arms")
    if not isinstance(block, dict):
        raise CellRuleError("regression_free.strict_legs is set but regression_free.component_arms is "
                            "missing: a leg outside the strict bar must be read by the component rule")
    if block.get("bar") != "informational":
        raise CellRuleError(f"component_arms.bar is {block.get('bar')!r}; the only bar ruling 31 "
                            "defines is 'informational'")
    kinds = block.get("null_pair_kinds")
    if kinds is not None and not set(kinds) <= set(NULL_PAIR_KINDS):
        raise CellRuleError(f"component_arms.null_pair_kinds {kinds} is not within the closed list "
                            f"{list(NULL_PAIR_KINDS)}")
    return tuple(strict), block


# ------------------------------------------------------------------------------------------------ #
# rule 2: accepted losses on the strict bar
# ------------------------------------------------------------------------------------------------ #

ENTRY_KEYS = ("gold", "instrument", "leg", "class", "ruling")


def entry_key(e: dict) -> tuple:
    return (str(e.get("gold")), str(e.get("instrument")), e.get("leg"), e.get("class"),
            e.get("ruling"), e.get("candidate_collection"))


def accepted_entries(contract: dict, *, instrument: str, leg: str, klass: str,
                     candidate_collection: str | None) -> list[dict]:
    """The accepted losses naming this strict cell (instrument, leg, class), bound to this candidate
    (an entry bound to another collection never applies). Malformed entries raise."""
    rf = contract.get("regression_free") or {}
    strict = rf.get("strict_legs")
    out = []
    for e in rf.get("accepted_losses") or []:
        missing = [k for k in ENTRY_KEYS if not e.get(k)]
        if missing or not e.get("cite"):
            raise CellRuleError(f"accepted_losses entry {e.get('gold')!r} lacks {missing or ['cite']}: "
                                "an accepted loss names its gold, instrument, leg, class, ruling and cite")
        shown = e.get("shown") or {}
        outside = shown.get("candidate_outside_top_k") is True
        if outside and shown.get("candidate_rank") is not None:
            raise CellRuleError(f"accepted_losses entry {e['gold']!r} says shown.candidate_outside_top_k "
                                "but also carries an integer shown.candidate_rank: contradictory")
        if not outside and not isinstance(shown.get("candidate_rank"), int):
            raise CellRuleError(f"accepted_losses entry {e['gold']!r} carries no shown.candidate_rank: "
                                "a loss is accepted as it was shown, not open-ended")
        if strict is not None and e["leg"] not in strict:
            raise CellRuleError(f"accepted_losses entry {e['gold']!r} names leg {e['leg']!r}, which is "
                                "not a strict leg: only the strict bar consults accepted losses")
        if (str(e["instrument"]), e["leg"], e["class"]) != (str(instrument), leg, klass):
            continue
        want = e.get("candidate_collection")
        if want and candidate_collection and want != candidate_collection:
            continue
        out.append(e)
    return out


def accepted_outside_top_k(e: dict) -> bool:
    """True when the entry was shown with its gold OUTSIDE the top k (no candidate rank)."""
    return (e.get("shown") or {}).get("candidate_outside_top_k") is True


def _hit(rank, cut: int) -> int:
    return 1 if (rank is not None and rank <= cut) else 0


def gold_contribution(r_inc, r_cand, metric: str, k: int = 10) -> float:
    """One gold's change (candidate - incumbent) in gold units."""
    if metric == "hit1":
        return float(_hit(r_cand, 1) - _hit(r_inc, 1))
    if metric == "hitk":
        return float(_hit(r_cand, k) - _hit(r_inc, k))
    rr = lambda r: (1.0 / r) if (r is not None and r <= k) else 0.0  # noqa: E731
    return rr(r_cand) - rr(r_inc)


def residual_after_accepted(delta_gold: float, entries: list[dict], ranks_inc: dict | None,
                            ranks_cand: dict | None, metric: str, k: int = 10) -> dict:
    """The class's loss with the accepted golds' own changes removed.

    Returns {residual, applied: [entry...], problems: [str...]}. A problem means the entry could not
    be applied (no per-gold ranks, gold not scored, or it fell further than shown); the caller keeps
    the cell a refusal and names the problem.
    """
    problems: list[str] = []
    applied: list[dict] = []
    contrib = 0.0
    if ranks_inc is None or ranks_cand is None:
        return {"residual": delta_gold, "applied": [],
                "problems": ["the arms carry no per-gold ranks for this class, so accepted loss(es) "
                             f"{[e['gold'] for e in entries]} cannot be checked"]}
    for e in entries:
        g = str(e["gold"])
        outside = accepted_outside_top_k(e)
        # An entry shown OUTSIDE the top k scores an absent candidate gold as beyond k (0 units).
        if g not in ranks_inc or (g not in ranks_cand and not outside):
            problems.append(f"accepted gold {g} is not scored in both arms of this class")
            continue
        ri, rc = ranks_inc[g], ranks_cand.get(g)
        bound = None if outside else int(e["shown"]["candidate_rank"])
        if not outside and (rc is None or rc > bound):
            problems.append(f"accepted gold {g} ranks {rc} on the candidate; {e['ruling']} accepted it "
                            f"falling to rank {bound} -- a loss larger than the one shown is a fresh "
                            "refusal")
            continue
        contrib += gold_contribution(ri, rc, metric, k)
        applied.append({**{x: e.get(x) for x in ("gold", "instrument", "leg", "class", "ruling",
                                                   "candidate_collection")},
                        "rank_incumbent": ri, "rank_candidate": rc,
                        **({"outside_top_k": True} if outside else {})})
    return {"residual": delta_gold - contrib, "applied": applied, "problems": problems}


def per_gold_from_rows(rows: list[dict], *, class_key: str = "type") -> dict[str, dict]:
    """{class: {gold id: rank}} from a run-arm `rows` list (id, type, rank, error)."""
    out: dict[str, dict] = {}
    for r in rows or []:
        if r.get("error"):
            continue
        out.setdefault(str(r.get(class_key)), {})[str(r.get("id"))] = r.get("rank")
    return out


def per_gold_from_sets(sets: dict) -> dict[str, dict]:
    """{class: {gold id: rank}} from a held-out leg json's `sets[*].per_q`."""
    out: dict[str, dict] = {}
    for klass, s in (sets or {}).items():
        pq = (s or {}).get("per_q")
        if isinstance(pq, dict):
            out[klass] = {str(q): (v or {}).get("rank") for q, v in pq.items()}
    return out
