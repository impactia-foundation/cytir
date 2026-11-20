"""Compare two arms: per class, per gold, with a null band. Pure functions over arm documents.

An arm document (`aw.retune-arm.v1`, written by `arm.py`) holds `sets[name] = {n, n_gold, h1, h10,
mrr, per_q: {qid: {urls, rank}}}`. The chooser is one set whose rows carry a `type`; it is split
into one class per type here, as the flip-gate contract does (`class_field: by_type`).

The bar is ADR-043's RANKING/serving bar: a fusion-weights change keeps the strict per-gold
zero-loss rule (a per-gold regression is a real defect, not corpus drift). The null band is the
number of golds that moved between two runs of the SAME weights: a delta no larger than that is
noise (verification-doctrine: an A/B needs a same-arm null).
"""
from __future__ import annotations

from cytir.gates.authority import rank_regressed
from cytir.retune.score import aggregate

MRR_TOLERANCE = 0.0005     # the same rounding slack as `cytir.gates.authority.compare`
CHOOSER = "chooser"


def classes(arm: dict) -> dict[str, dict]:
    """{class name: set document}; the chooser split by `type`."""
    out = {}
    for name, doc in (arm.get("sets") or {}).items():
        if name != CHOOSER:
            out[name] = doc
            continue
        by_type: dict[str, dict] = {}
        for qid, row in doc["per_q"].items():
            by_type.setdefault(row.get("type") or "?", {})[qid] = row
        for t, rows in sorted(by_type.items()):
            out[f"{CHOOSER}:{t}"] = {**aggregate(rows), "per_q": rows}
    return out


def _agg(doc: dict) -> dict:
    if "h1" in doc and "h10" in doc:
        return {"n": doc.get("n", len(doc["per_q"])), "n_gold": doc.get("n_gold", 0), "h1": doc["h1"],
                "h10": doc["h10"], "mrr": doc.get("mrr")}
    return aggregate(doc["per_q"])


def compare(a: dict, b: dict) -> dict:
    """Arm `a` (incumbent) against arm `b`. Raises if the two did not score the same rows: a
    missing class or query is a broken instrument, never a zero."""
    ca, cb = classes(a), classes(b)
    if set(ca) != set(cb):
        raise ValueError(f"arms scored different classes: {sorted(set(ca) ^ set(cb))}")
    rows, regressed, improved, moved = [], [], [], 0
    for name in ca:
        qa, qb = ca[name]["per_q"], cb[name]["per_q"]
        if set(qa) != set(qb):
            raise ValueError(f"class {name}: arms scored different queries "
                             f"({len(set(qa) ^ set(qb))} differ)")
        reordered = 0
        for qid in qa:
            ra, rb = qa[qid].get("rank"), qb[qid].get("rank")
            if qa[qid].get("urls") != qb[qid].get("urls"):
                reordered += 1
            if ra != rb:
                moved += 1
            if rank_regressed(ra, rb):
                regressed.append({"class": name, "id": qid, "from": ra, "to": rb})
            elif ra is not None and rb is not None and rb < ra or (ra is None and rb is not None):
                improved.append({"class": name, "id": qid, "from": ra, "to": rb})
        ga, gb = _agg(ca[name]), _agg(cb[name])
        rows.append({"class": name, "n": ga["n"], "n_gold": ga["n_gold"],
                     "a": {k: ga[k] for k in ("h1", "h10", "mrr")},
                     "b": {k: gb[k] for k in ("h1", "h10", "mrr")},
                     "d_h1": gb["h1"] - ga["h1"], "d_h10": gb["h10"] - ga["h10"],
                     "d_mrr": round((gb["mrr"] or 0.0) - (ga["mrr"] or 0.0), 4),
                     "reordered": reordered})
    chooser = [r for r in rows if r["class"].startswith(CHOOSER + ":")]
    return {"rows": rows, "regressed": regressed, "improved": improved, "moved": moved,
            "chooser_net": sum(r["d_h1"] + r["d_h10"] for r in chooser),
            "chooser_d_mrr": round(sum(r["d_mrr"] * r["n_gold"] for r in chooser)
                                   / max(1, sum(r["n_gold"] for r in chooser)), 4)}


def bar(cmp: dict, null: dict) -> dict:
    """The ADR-043 ranking bar for `cmp` (incumbent vs candidate), read through `null`
    (incumbent vs incumbent, same weights, same twin). Returns {verdict, reasons, band}.

    The twin is live (hourly drain, qdrant segment merges) and near-tied pages swap places between
    two runs of the same weights, so the null pair moves a few golds. `band` is that count. The
    bar is strict per-gold zero-loss; a loss no larger than the band cannot be told from noise.
      PASS          no gold lost rank, no class dropped, and the chooser gain exceeds the band
      NOT_BETTER    no loss, but the gain does not exceed the band
      INCONCLUSIVE  the worst loss (golds, or a class's h@1/h@10) is at most the band: repeat on
                    a quieter twin; do not promote on this run
      FAIL          a loss larger than the band"""
    band = null["moved"]
    class_drops = [(r["class"], r["d_h1"], r["d_h10"], r["d_mrr"]) for r in cmp["rows"]
                   if r["d_h1"] < 0 or r["d_h10"] < 0 or r["d_mrr"] < -MRR_TOLERANCE]
    worst = max([len(cmp["regressed"])] + [max(-d1, -d10, 0) for _, d1, d10, _ in class_drops])
    reasons = []
    if cmp["regressed"]:
        reasons.append(f"{len(cmp['regressed'])} gold(s) lost rank (ADR-043 ranking bar is per-gold zero-loss)")
    reasons += [f"class {c}: h@1 {d1:+d}, h@10 {d10:+d}, MRR {dm:+.4f}" for c, d1, d10, dm in class_drops]
    if worst > band:
        return {"verdict": "FAIL", "band": band, "reasons": reasons}
    if reasons:
        return {"verdict": "INCONCLUSIVE", "band": band,
                "reasons": reasons + [f"the null pair moved {band} gold(s): a loss of {worst} is inside the noise"]}
    if cmp["chooser_net"] <= band:
        return {"verdict": "NOT_BETTER", "band": band,
                "reasons": [f"chooser net gain {cmp['chooser_net']:+d} (h@1 + h@10) does not exceed the null band {band}"]}
    return {"verdict": "PASS", "band": band, "reasons": []}


def pick_winner(arms: dict[str, dict], incumbent: dict) -> tuple[str | None, dict]:
    """Among proposal arms (name -> arm doc, chooser only), the one to validate: fewest regressed
    golds, then the largest chooser net gain, then MRR. None when no arm gains at all."""
    scored = {}
    for name, arm in arms.items():
        c = compare(incumbent, arm)
        scored[name] = c
    live = {n: c for n, c in scored.items() if c["chooser_net"] > 0}
    if not live:
        return None, scored
    best = min(live, key=lambda n: (len(live[n]["regressed"]), -live[n]["chooser_net"],
                                    -live[n]["chooser_d_mrr"], n))
    return best, scored


def perquery_csv(inc: dict, null: dict, cand: dict) -> str:
    """One line per gold-bearing query: class, id, live rank, same-weights-again rank, candidate
    rank (blank = not in the top 10). ADR-099 amendment 3 point 3: the report is per query."""
    ci, cn, cc = classes(inc), classes(null), classes(cand)
    lines = ["class,id,live,null,candidate"]
    for name in ci:
        for qid, row in ci[name]["per_q"].items():
            r = [row.get("rank"), cn[name]["per_q"][qid].get("rank"), cc[name]["per_q"][qid].get("rank")]
            if all(x is None for x in r):
                continue            # no gold, or missed by all three: no signal
            lines.append(",".join([name, qid] + ["" if x is None else str(x) for x in r]))
    return "\n".join(lines) + "\n"


ERROR_PREFIX = "__ERROR__"


class ArmIntegrityError(SystemExit):
    """An arm that holds an error row is refused, not averaged over (flip-gate contract,
    `arm_integrity.no_errors`; verification-doctrine 6). `cytir.gates.authority.run` turns every
    exception into an `__ERROR__` row with rank None, which would read as a miss."""
    def __init__(self, msg: str):
        super().__init__(f"REFUSED: {msg}")
        self.code = 4


def error_rows(doc: dict) -> list[tuple[str, str]]:
    out = []
    for name, s in (doc.get("sets") or {}).items():
        for qid, row in (s.get("per_q") or {}).items():
            urls = row.get("urls") or []
            if urls and str(urls[0]).startswith(ERROR_PREFIX):
                out.append((name, qid))
    return out


def assert_no_errors(doc: dict, label: str) -> None:
    bad = error_rows(doc)
    if bad:
        raise ArmIntegrityError(f"arm {label} has {len(bad)} error row(s), e.g. {bad[:3]}: an error is not a miss")
