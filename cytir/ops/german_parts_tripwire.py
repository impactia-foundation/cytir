#!/usr/bin/env python3
"""Guard 2 of task 710: the daily twin tripwire for German BM25 parts.

Read-only on ONE Qdrant: the twin, which the caller names (`--url`, `--collection`; there is no
default, so it never reads a live endpoint by accident). For each
GENERATION of German rows it reads the share of rows whose stored sparse vector carries
half-integer tf values (a de-compounded part adds exactly 0.5, task 710 section 5.1) and
alarms when a generation reads below `--alarm-below` (50 %).

Generation = pool x source_type x provenance x prov_id x chunk epoch x build_id x last_seen month.
A pool is a domain that holds >= `--min-generation` German rows (found with the facet API,
not a hand list), or the long tail (every other domain). The payload names no writer and no pack
id, so the key is built only from fields that a LATER pass over the payload does not rewrite:
the source fields, `rule_epochs.chunk`, `build_id`, and `last_seen` (the month the row was last
fetched: a re-fetch that moves it also rewrote the vector, so the row truly changes generation).
`last_seen` is the FETCH day, not the day the vector was written: for drained CC rows the drain
computes the sparse leg later, so one month can hold a blind and a sighted writer (task 732; hence
the growth judgement below). The consent fields
(`consent_checked_at`, `consent_basis`, `search_ok`, `train_ok`, `consent_state`, `stamps`) are
NOT in the key: the consent backfill of the twin adds or restamps them on rows long after the
writer finished (2026-10-07: it moved 596,520 openlegaldata rows from month "-" to "2026-10" and
raised a false GREW alarm). `rule_epochs` (except `chunk`) and `build_id` do not tell the drained
from the built generation on their own; `last_seen` does where the payload has it (CC-w7 and
other tail rows: drained 2026-09/10, built 2026-06/07). Where the payload has no `last_seen`
(openlegaldata, Wikipedia DE) the month reads "-": those pools hold ONE generation per source
fields, and the pack receipts (task 710 section 2) are the only finer grain, not carried in the
payload. Self-check: every scanned row falls in exactly one generation (see `self_check`).

Sampling: random uuid offsets under the filter `lang = de` (+ the pool's domain). A generation
whose estimated population is > `--min-generation` needs >= `--sample` sampled rows; when the
random windows left it short and its pool is small enough, a payload-only scan of the pool
finds its ids and the vectors are read by id. A generation still short is listed as
UNDERSAMPLED, never alarmed on.

Rows with < 20 postings are undecidable on the lattice and are not counted (task 710 proposed
this floor; the report gives how many rows it dropped). Entity cards are excluded (their
segment-max weights are off the lattice).

The acknowledged list binds each ruling to ONE MEASURED generation, never to a family: a line is
`<generation key> ; rows=<estimated rows when measured> ; share=<measured share> # <reason>`
(a pattern may use fnmatch wildcards). A generation matched by a line is acknowledged only while its
estimated rows stay within `rows` (+ `--ack-slack`, default 20 %, for the sampling estimate): a new
pack the drain writes into an acknowledged generation GROWS it. The GROWTH is then judged on its own
(`growth_share`, from the line's `rows` and `share`): growth that reads below the floor alarms (more
rows without parts); growth that reads at or above it is GREW WITH PARTS, state `warn`, and the line
must be re-measured (task 732: a sighted pack in the same month as a blind one). A re-measured line
says so (`; remeasured=<date>:<tracked report>`, task 742) and a test pins its rows and share to that
report. A generation with
no line alarms. A line that matches no alarmed generation is STALE (state `result=warn`): the
rows were rebuilt, so the line is deleted and the list shrinks.

CODED TWIN (task 855). Every payload the tripwire reads is its CANONICAL row
(`cytir.index.row_decode`, the collection's declared profile): an integer `provenance`, `prov_id` or
`rule_epochs` reads as its value, so a raw twin (aw_e8) and a coded one (the twin after the aw_e10
flip) give the same generations. One key differs by profile: `rebuild_e10` does not store
`build_id`. Such a row's key carries `ABSENT` (`~`) in the build_id field, and an acknowledged line
matches it whatever build_id the line names (`key_matches`): the table has ONE build_id value
(`aw_b1`), so the field told no generation from another, and the lines measured on the raw twin keep
matching the same rows on the coded twin.

The lattice reads a row's base weight off the most common weight, so it assumes tf = 1 is the modal
weight; rows where another tf dominates are mis-read, and the 20-posting floor leaves most of those
out.

Output: a JSON report, a key=value state file that the operator's monitor (`deploy/scripts/aw-monitor-check.sh`) reads over
ssh (check `german_parts`), and a one-line summary on stdout. Exit 0 = nothing new alarmed,
3 = a generation not on the acknowledged list (or grown past it) reads below the floor, 2 = could
not measure.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from cytir.core import internal_http as IH  # noqa: E402 -- the one door to the operator's own services (consent step 5, task 1673)
import argparse
import collections
import datetime
import fnmatch
import json
import os
import random
import time
import urllib.request
import uuid

from cytir.index.german_parts_guard import LATTICE_MIN_POSTINGS, bm25_of, lattice  # noqa: E402
from cytir.index import prod_guard as PG  # noqa: E402 -- no prod scroll, no prod replay (2026-10-07)
from cytir.index import row_decode as RD  # noqa: E402 -- canonical stored rows, any profile (task 855)

SCHEMA = "aw.german-parts-tripwire.v1"
PAYLOAD_KEYS = ["domain", "source_type", "provenance", "prov_id", "rule_epochs", "build_id",
                "last_seen", "lang"]
WINDOW = 500
#: The build_id field of a generation key when the row's profile does not store `build_id`.
ABSENT = "~"
#: The position of build_id in a generation key (pool|type|prov|prov_id|chunk|build_id|month).
BUILD_ID_FIELD = 5


class Qdrant:
    def __init__(self, url: str, collection: str, timeout: float = 120.0):
        self.url, self.collection, self.timeout = url.rstrip("/"), collection, timeout
        self.decode = RD.reader(collection)     # every payload read is a canonical row (task 855)

    def _decoded(self, points):
        return [dict(p, payload=self.decode(p.get("payload"))) for p in points]

    def post(self, path: str, body: dict):
        PG.check(self.url, "POST", f"/collections/{self.collection}{path}")
        req = urllib.request.Request(
            f"{self.url}/collections/{self.collection}{path}", data=json.dumps(body).encode(),
            headers={"content-type": "application/json"})
        with IH.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["result"]

    def count(self, flt: dict) -> int:
        return self.post("/points/count", {"filter": flt, "exact": True})["count"]

    def facet(self, key: str, flt: dict, limit: int):
        return self.post("/facet", {"key": key, "filter": flt, "limit": limit, "exact": True})["hits"]

    def scroll(self, flt, *, offset=None, limit=WINDOW, vector=True):
        body = {"filter": flt, "limit": limit, "with_payload": PAYLOAD_KEYS,
                "with_vector": ["bm25"] if vector else False}
        if offset is not None:
            body["offset"] = offset
        res = self.post("/points/scroll", body)
        return dict(res, points=self._decoded(res.get("points") or []))

    def retrieve(self, ids):
        return self._decoded(self.post("/points", {"ids": ids, "with_payload": PAYLOAD_KEYS,
                                                   "with_vector": ["bm25"]}))


def gen_key(pl: dict, pool: str) -> str:
    """The generation key of a CANONICAL row (`row_decode`; a raw payload reads the same)."""
    pl = RD.as_canonical_row(pl)
    ep = pl.get("rule_epochs") or {}  # canonical row
    build = ABSENT if "build_id" in pl.absent else str(pl.get("build_id") or "-")  # canonical row
    return "|".join((pool, str(pl.get("source_type") or "-"), str(pl.get("provenance") or "-"),  # canonical row
                     str(pl.get("prov_id") or "-")[:8], f"chunk{ep.get('chunk', '-')}",  # canonical row
                     build, str(pl.get("last_seen") or "-")[:7]))


def key_matches(key: str, pattern: str) -> bool:
    """An acknowledged line's pattern against a generation key. A key whose build_id is `ABSENT`
    (the profile does not store it) matches the line whatever build_id the line names."""
    if fnmatch.fnmatchcase(key, pattern):
        return True
    kf, pf = key.split("|"), pattern.split("|")
    if len(kf) != len(pf) or len(kf) <= BUILD_ID_FIELD or kf[BUILD_ID_FIELD] != ABSENT:
        return False
    pf[BUILD_ID_FIELD] = "*"
    return fnmatch.fnmatchcase(key, "|".join(pf))


def row_has_parts(point: dict) -> bool | None:
    """True/False for a decidable row, None when the lattice cannot decide it."""
    lat = lattice(bm25_of(point).get("values") or [])
    if lat is None or lat[0] < LATTICE_MIN_POSTINGS:
        return None
    return lat[1] > 0


class Gen:
    __slots__ = ("sampled", "decidable", "with_parts", "ids", "exact_pop")

    def __init__(self):
        self.sampled = self.decidable = self.with_parts = 0
        self.ids = set()
        self.exact_pop = None

    def add(self, point):
        if point["id"] in self.ids:
            return
        self.ids.add(point["id"])
        self.sampled += 1
        v = row_has_parts(point)
        if v is not None:
            self.decidable += 1
            self.with_parts += v


def sample_pool(q: Qdrant, flt: dict, pool: str, pool_rows: int, rng, *, windows: int,
                gens: dict, seen: set, stop_when=None):
    """Random windows under `flt`; a pool that fits one scan is read whole."""
    if pool_rows <= 4 * WINDOW:
        offsets = [None]
    else:
        offsets = [str(uuid.UUID(int=rng.getrandbits(128))) for _ in range(windows)]
    for off in offsets:
        cursor = off
        while True:
            res = q.scroll(flt, offset=cursor)
            for p in res["points"]:
                if p["id"] in seen:
                    continue
                seen.add(p["id"])
                gens.setdefault(gen_key(p["payload"], pool), Gen()).add(p)
            cursor = res.get("next_page_offset")
            if off is not None or not cursor:
                break
        if stop_when and stop_when():
            return


def fill_undersampled(q: Qdrant, flt: dict, pool: str, pool_rows: int, want: int, gens: dict,
                      scan_cap: int, rng):
    """Payload-only scan of a small pool to find the ids of a generation the random windows
    left short, then read their vectors by id. Returns the keys it could not fill."""
    if pool_rows > scan_cap:
        return
    ids_by_key = collections.defaultdict(list)
    cursor = None
    while True:
        res = q.scroll(flt, offset=cursor, limit=1000, vector=False)
        for p in res["points"]:
            ids_by_key[gen_key(p["payload"], pool)].append(p["id"])
        cursor = res.get("next_page_offset")
        if not cursor:
            break
    for key, ids in ids_by_key.items():
        g = gens.get(key)
        if len(ids) <= 1000 or (g is not None and g.sampled >= want):
            continue
        g = gens.setdefault(key, Gen())
        g.exact_pop = len(ids)
        todo = [i for i in ids if i not in g.ids]
        rng.shuffle(todo)
        for k in range(0, min(len(todo), want * 2), 250):
            for p in q.retrieve(todo[k:k + 250]):
                g.add(p)
            if g.sampled >= want:
                break


EST_TOLERANCE = 0.05


def self_check(gens: dict, out: list, total_de: int, pool_rows: dict) -> dict:
    """A row falls in exactly ONE generation, and the generations hold every German row.
    Sampled level: no id sits in two generations, and the per-generation sampled counts add up to
    the distinct ids scanned. Population level: the estimated rows of all generations add up to the
    twin's German row count (the estimate is pool rows x sampled share, so a mismatch beyond
    rounding and the exact-count generations means a row was dropped or counted twice).
    `ok` False makes the run an error (state `error`), never a silent ok."""
    per_gen = sum(len(g.ids) for g in gens.values())
    distinct = len(set().union(*(g.ids for g in gens.values()))) if gens else 0
    est_total = sum(g["est_rows"] for g in out)
    pool_total = sum(pool_rows.values())
    rel = abs(est_total - total_de) / max(1, total_de)
    ok = (per_gen == distinct and per_gen == sum(g.sampled for g in gens.values())
          and pool_total == total_de and rel <= EST_TOLERANCE)
    return {"ok": ok, "scanned_rows": distinct, "rows_in_generations": per_gen,
            "rows_in_two_generations": per_gen - distinct, "pool_rows_total": pool_total,
            "estimated_rows_total": est_total, "twin_german_rows": total_de,
            "estimate_vs_twin": round(rel, 5), "tolerance": EST_TOLERANCE}


def run(q: Qdrant, *, min_generation: int, sample: int, alarm_below: float, windows: int,
        tail_windows: int, scan_cap: int, seed: int) -> dict:
    rng = random.Random(seed)
    de = {"must": [{"key": "lang", "match": {"value": "de"}}]}
    total_de = q.count(de)
    hits = q.facet("domain", de, 5000)
    named = {h["value"]: h["count"] for h in hits if h["count"] >= min_generation}
    gens: dict[str, Gen] = {}
    pool_rows: dict[str, int] = {}
    keys_of_pool: dict[str, set] = {}
    t0 = time.time()
    for dom, n in sorted(named.items(), key=lambda kv: -kv[1]):
        flt = {"must": de["must"] + [{"key": "domain", "match": {"value": dom}}]}
        seen: set = set()
        sample_pool(q, flt, dom, n, rng, windows=windows, gens=gens, seen=seen)
        pool_rows[dom] = n
        keys_of_pool[dom] = {k for k in gens if k.split("|", 1)[0] == dom}
        short = [k for k in keys_of_pool[dom]
                 if gens[k].sampled < sample and
                 (gens[k].sampled / max(1, len(seen))) * n > min_generation]
        if short:
            fill_undersampled(q, flt, dom, n, sample, gens, scan_cap, rng)
    tail_rows = total_de - sum(named.values())
    if tail_rows > 0:
        flt = {"must": de["must"],
               "must_not": [{"key": "domain", "match": {"any": list(named)}}]} if named else de
        seen = set()
        sample_pool(q, flt, "(tail)", tail_rows, rng, windows=tail_windows, gens=gens, seen=seen)
        pool_rows["(tail)"] = tail_rows
        short = [k for k in gens if k.startswith("(tail)|") and gens[k].sampled < sample
                 and gens[k].sampled / max(1, len(seen)) * tail_rows > min_generation]
        if short:
            fill_undersampled(q, flt, "(tail)", tail_rows, sample, gens, scan_cap, rng)
    # estimated population of a generation = pool rows x its share of the pool's random sample
    pool_sampled = collections.Counter()
    for k, g in gens.items():
        pool_sampled[k.split("|", 1)[0]] += g.sampled
    out = []
    for key, g in sorted(gens.items()):
        pool = key.split("|", 1)[0]
        est = g.exact_pop or round(pool_rows[pool] * g.sampled / max(1, pool_sampled[pool]))
        entity = key.split("|")[1] == "entity_card"
        share = g.with_parts / g.decidable if g.decidable else None
        if est <= min_generation:
            verdict = "small"
        elif entity:
            verdict = "excluded-entity-card"
        elif g.sampled < min(sample, est) or g.decidable < 100:
            verdict = "undersampled"
        else:
            verdict = "ALARM" if share < alarm_below else "ok"
        out.append({"generation": key, "est_rows": est, "sampled": g.sampled,
                    "decidable": g.decidable, "share_half_integer": None if share is None else round(share, 4),
                    "verdict": verdict})
    check = self_check(gens, out, total_de, pool_rows)
    return {"schema": SCHEMA, "twin_german_rows": total_de, "self_check": check, "named_domains": len(named),
            "alarm_below": alarm_below, "min_generation": min_generation, "sample": sample,
            "lattice_min_postings": LATTICE_MIN_POSTINGS, "seconds": round(time.time() - t0),
            "generations": out}


ACK_SLACK = 0.2


class Ack(collections.namedtuple("Ack", "pattern label rows share remeasured", defaults=(None,))):
    """One acknowledged line. `rows` None = no row cap (a line with no measured count).
    `remeasured` = None, or `(date, artefact path)`: the line was measured again after the 710 baseline
    (task 742) and its rows and share equal that tracked report's values for this exact key."""
    __slots__ = ()


def parse_remeasured(text: str) -> tuple[str, str]:
    """`2026-10-08:reports/x.json` -> ("2026-10-08", "reports/x.json").
    Raises ValueError on a bad date or an empty / absolute / `..` path (a re-measure names a tracked file)."""
    day, sep, path = str(text).strip().partition(":")
    try:
        datetime.date.fromisoformat(day)
    except ValueError:
        raise ValueError(f"remeasured needs <YYYY-MM-DD>:<artefact path>, got {text!r}") from None
    if not sep or not path or path.startswith("/") or ".." in path.split("/"):
        raise ValueError(f"remeasured needs a relative artefact path, got {text!r}")
    return day, path


def acknowledged(path: str | None) -> list[Ack]:
    """`Ack(pattern, label, rows, share)` per line of the acknowledged file:
    `<pattern> ; rows=N ; share=S [; remeasured=<date>:<artefact>] # <label>` (the `; key=value` parts
    are optional; `remeasured` needs `rows` and `share` and is parsed strictly)."""
    if not path or not os.path.exists(path):
        return []
    rows = []
    for line in open(path, encoding="utf-8"):
        body, _, label = line.partition("#")
        if not body.strip():
            continue
        pat, *kvs = (x.strip() for x in body.split(";"))
        kv = dict(x.split("=", 1) for x in kvs if "=" in x)
        n = int(kv["rows"].replace(",", "")) if "rows" in kv else None
        rem = None
        if "remeasured" in kv:
            if n is None or "share" not in kv:
                raise ValueError(f"{pat}: remeasured needs rows and share on the same line")
            rem = parse_remeasured(kv["remeasured"])
        rows.append(Ack(pat, label.strip() or pat, n, kv.get("share"), rem))
    return rows


def parse_share(text) -> float | None:
    """`1.40%` -> 0.014, `0.014` -> 0.014; None when absent or unreadable."""
    if text is None:
        return None
    t = str(text).strip()
    try:
        return float(t[:-1]) / 100 if t.endswith("%") else float(t)
    except ValueError:
        return None


def growth_share(a: Ack, gens: list, total: int) -> float | None:
    """Share of rows with parts among the rows a generation GAINED since its line was measured:
    (rows with parts now - rows with parts then) / (rows now - rows then), from the line's own
    `rows` and `share` and today's estimate. None when the line or a generation has no share.

    Why: the key cannot tell two writers apart inside one month (the payload names no writer;
    `rule_epochs` is the same for the blind and the sighted drain, because the rule did not
    change, only the interpreter). A sighted pack landing in an acknowledged blind generation
    grows it, and the mixture still reads below the floor. Judging the GROWTH tells "more rows
    without parts" (alarm) from "new rows with parts" (warn, re-measure the line).
    Limit: rows repaired in place inside the generation would count as growth with parts."""
    then = parse_share(a.share)
    if then is None or a.rows is None or total <= a.rows:
        return None
    if any(g.get("share_half_integer") is None for g in gens):
        return None
    with_now = sum(g["share_half_integer"] * g["est_rows"] for g in gens)
    return max(0.0, min(1.0, (with_now - then * a.rows) / (total - a.rows)))


def judge(report: dict, ack: list, *, slack: float = ACK_SLACK) -> dict:
    ack = [a if isinstance(a, Ack) else Ack(a[0], a[1], None, None) for a in ack]
    alarms = [g for g in report["generations"] if g["verdict"] == "ALARM"]
    by_label: dict[str, list[int]] = {}
    matched: dict[int, list[dict]] = {}
    new = []
    for g in alarms:
        i = next((i for i, a in enumerate(ack) if key_matches(g["generation"], a.pattern)), None)
        if i is None:
            new.append(g)
        else:
            matched.setdefault(i, []).append(g)
    grown, grown_with_parts = [], []
    for i, gens in sorted(matched.items()):
        a, total = ack[i], sum(g["est_rows"] for g in gens)
        if a.rows is not None and total > a.rows * (1 + slack):
            gshare = growth_share(a, gens, total)
            if gshare is not None and gshare >= report["alarm_below"]:
                # The rows written into it since the line was measured carry parts (task 732: tranche 13
                # landed 59,751 sighted German rows in the acknowledged blind October tail; the mixture read
                # 36 %). Not an alarm, but the line no longer describes the generation: state warn.
                grown_with_parts.append((a, total, gshare))
                acc = by_label.setdefault(a.label, [0, 0])
                acc[0] += len(gens)
                acc[1] += total
                continue
            grown.append((a, total, gshare))
            new.extend(gens)          # past what was measured: new rows without parts were written into it
        else:
            acc = by_label.setdefault(a.label, [0, 0])
            acc[0] += len(gens)
            acc[1] += total
    stale = [a for i, a in enumerate(ack) if i not in matched]
    under = [g for g in report["generations"] if g["verdict"] == "undersampled"]
    est_alarm = sum(g["est_rows"] for g in alarms)
    parts = [f"{len(alarms)} generation(s) below {report['alarm_below']:.0%} "
             f"(~{est_alarm:,} rows), {len(new)} not acknowledged"]
    if new:
        parts.append("NEW: " + "; ".join(
            f"{g['generation']} {g['share_half_integer']:.1%} of {g['decidable']}" for g in new[:5]))
    if grown:
        parts.append("GREW past the acknowledged rows: " + "; ".join(
            f"{a.pattern} acknowledged {a.rows:,}, now ~{t:,}"
            + ("" if gs is None else f", the growth reads {gs:.1%}") for a, t, gs in grown[:5]))
    if grown_with_parts:
        parts.append("GREW WITH PARTS (not an alarm; re-measure the line): " + "; ".join(
            f"{a.pattern} acknowledged {a.rows:,}, now ~{t:,}, the growth reads {gs:.1%}"
            for a, t, gs in grown_with_parts[:5]))
    if stale:
        parts.append(f"{len(stale)} STALE acknowledged line(s), match nothing now (delete them): "
                     + "; ".join(a.pattern for a in stale[:3]) + (" ..." if len(stale) > 3 else ""))
    if under:
        parts.append(f"{len(under)} undersampled")
    return {"alarms": len(alarms), "new_alarms": len(new), "undersampled": len(under),
            "grown": [{"pattern": a.pattern, "acknowledged_rows": a.rows, "est_rows": t,
                       "growth_share": gs} for a, t, gs in grown],
            "grown_with_parts": [{"pattern": a.pattern, "acknowledged_rows": a.rows, "est_rows": t,
                                  "growth_share": gs} for a, t, gs in grown_with_parts],
            "stale_lines": [a.pattern for a in stale],
            "match_counts": {a.pattern: len(matched.get(i, [])) for i, a in enumerate(ack)},
            "acknowledged_rollup": {k: {"generations": v[0], "est_rows": v[1]}
                                    for k, v in sorted(by_label.items(), key=lambda kv: -kv[1][1])},
            "summary": " | ".join(parts)}


def write_state(path: str, result: str, summary: str, extra: dict) -> None:
    now = datetime.datetime.now(datetime.timezone.utc)
    lines = [f"result={result}", f"checked_at={now.strftime('%Y-%m-%dT%H:%M:%SZ')}",
             f"checked_epoch={int(now.timestamp())}", f"summary={summary.replace(chr(10), ' ')}"]
    lines += [f"{k}={v}" for k, v in extra.items()]
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True, help="the TWIN's Qdrant base URL (never a live endpoint)")
    ap.add_argument("--collection", required=True, help="the twin collection")
    ap.add_argument("--state", required=True, help="key=value state file the monitor reads")
    ap.add_argument("--report", help="JSON report path (default: next to the state file)")
    ap.add_argument("--acknowledged", help="file of acknowledged generations (key ; rows ; share # reason)")
    ap.add_argument("--ack-slack", type=float, default=ACK_SLACK,
                    help="growth allowed past an acknowledged line's rows before it alarms")
    ap.add_argument("--min-generation", type=int, default=1000)
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--alarm-below", type=float, default=0.5)
    ap.add_argument("--windows", type=int, default=12, help="random windows per named domain")
    ap.add_argument("--tail-windows", type=int, default=300)
    ap.add_argument("--scan-cap", type=int, default=1_500_000)
    ap.add_argument("--seed", type=int, default=None, help="default: the day (varies daily)")
    a = ap.parse_args(argv)
    seed = a.seed if a.seed is not None else int(datetime.date.today().strftime("%Y%m%d"))
    try:
        report = run(Qdrant(a.url, a.collection), min_generation=a.min_generation,
                     sample=a.sample, alarm_below=a.alarm_below, windows=a.windows,
                     tail_windows=a.tail_windows, scan_cap=a.scan_cap, seed=seed)
    except Exception as exc:  # noqa: BLE001 -- state says error; never a silent "ok"
        write_state(a.state, "error", f"{type(exc).__name__}: {exc}", {})
        print(f"[german_parts] ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if not report["self_check"]["ok"]:
        msg = "self-check failed (a row is in two generations or the generations do not add up): " \
            + json.dumps(report["self_check"])
        pathlib.Path(a.report or os.path.join(os.path.dirname(os.path.abspath(a.state)), "last-report.json")
                     ).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
        write_state(a.state, "error", msg, {})
        print(f"[german_parts] ERROR {msg}", file=sys.stderr)
        return 2
    ack = acknowledged(a.acknowledged)
    verdict = judge(report, ack, slack=a.ack_slack)
    for pat, n in verdict["match_counts"].items():
        print(f"[german_parts] acknowledged line {pat}: matches {n} alarmed generation(s)"
              + ("  STALE" if n == 0 else ""))
    report["judgement"] = verdict
    rp = a.report or os.path.join(os.path.dirname(os.path.abspath(a.state)), "last-report.json")
    pathlib.Path(rp).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    write_state(a.state, "alarm" if verdict["new_alarms"] else
                ("warn" if verdict["stale_lines"] or verdict["grown_with_parts"] else "ok"),
                verdict["summary"],
                {"alarms": verdict["alarms"], "new_alarms": verdict["new_alarms"],
                 "stale_lines": len(verdict["stale_lines"]),
                 "grown_with_parts": len(verdict["grown_with_parts"]),
                 "undersampled": verdict["undersampled"], "report": os.path.abspath(rp)})
    print("[german_parts] " + verdict["summary"])
    return 3 if verdict["new_alarms"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
