"""Consent-flag backfill: the WRITE engine (ADR-115, task 684).

Driver: `cyt consent backfill {twin,prod-sample,prod-plan}` (bottom of this file).

Two legs, one derivation (`cytir.consent.derive.derive_consent_flags`):

``apply_twin``   scrolls a TWIN once (payload fields only, no vectors), derives every
                 row and writes the four fields -- `search_ok`, `train_ok`,
                 `consent_checked_at`, `consent_basis` -- with Qdrant `set_payload` by
                 id, grouped by identical write-sets. Only class A and B rows are ever
                 written. A row that already has both flags (E) or that falls to D is
                 never touched. Idempotent: a re-run derives class E for every row it
                 wrote. Checkpointed: the saved scroll offset is only advanced after
                 every buffered id was written, so a resume loses nothing.
``plan_prod``    READ-ONLY. From a twin that has been applied, takes the rows this ADR
                 wrote (their `consent_basis`), reads the same ids from the prod
                 collection by id (never a scan), and counts what a prod write WOULD do
                 after `encode` (the `Projector`, coded `consent_basis`).
``apply_prod``   the same rows, written to prod through `encode`, only where the prod
                 row still has NO flag (an E row is never rewritten), batch by batch
                 with the caller's guard between batches.

CODED TWIN (task 855). Every twin row is read as its CANONICAL row (`cytir.index.row_decode`, the
twin's declared profile): an integer `provenance` / `consent_state` / `consent_basis` reads as its
value, so the family and the class are the ones a raw twin gives. A write to a twin whose profile
CODES (the twin after the aw_e10 flip) goes through that twin's `Projector` (`twin_encoder`), so a
coded twin never takes a raw string; a raw twin (aw_e8) is written raw, as before.

Never point a scroll at the prod serving box: `provenance`, `license`, `search_ok`,
`train_ok` and `consent_*` are not indexed there (prod-serving-box-fragile-scrolls).
Prod is only ever read and written BY ID. All Qdrant calls go through an injectable
`api(method, path, body)` so the tests run on an in-memory store.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
import pathlib
import random
import sys
import time
import urllib.request
from cytir.index import prod_guard as PG  # noqa: E402 -- no prod scroll, no prod replay (2026-10-07)
from cytir.index import payload_profile as PP  # noqa: E402 -- a coded twin is written coded (task 855)
from cytir.index import row_decode as RD  # noqa: E402 -- canonical stored rows, any profile (task 855)
import uuid
from typing import Callable, Optional

from .derive import (WRITTEN_BASES, DUMP_CONSENT_BASES, RULES_VERSION, DeriveContext,
                     derive_consent_flags, family_of_row, host_of)

SCHEMA = "aw.consent-backfill-apply.v1"
SCAN_FIELDS = ["url", "domain", "provenance", "license", "search_ok", "train_ok",
               "consent_state", "consent_basis", "consent_checked_at"]
WRITE_FIELDS = ("search_ok", "train_ok", "consent_checked_at", "consent_basis")
PER_URL_DOMAINS = frozenset({"arxiv.org", "pubmed.ncbi.nlm.nih.gov"})


def http_api(base: str, timeout: float = 600.0, tries: int = 5) -> Callable:
    """The one transport: JSON over HTTP, retried with backoff."""
    def api(method: str, path: str, body: Optional[dict] = None):
        PG.check(base, method, path)        # no prod scroll, no prod replay (2026-10-07)
        last = None
        for i in range(tries):
            try:
                data = None if body is None else json.dumps(body).encode()
                req = urllib.request.Request(base + path, data, {"content-type": "application/json"},
                                             method=method)
                return json.load(urllib.request.urlopen(req, timeout=timeout))
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(min(30, 2 ** i))
        raise RuntimeError(f"qdrant call failed after {tries} tries: {method} {base}{path}: {last!r}")
    return api


def count(api, collection: str) -> int:
    return int(api("POST", f"/collections/{collection}/points/count", {"exact": True})["result"]["count"])


def fetch_by_id(api, collection: str, ids: list, fields: list) -> dict:
    """{id: payload} for the ids that exist, AS STORED. A by-id read: safe on prod."""
    if not ids:
        return {}
    r = api("POST", f"/collections/{collection}/points",
            {"ids": ids, "with_payload": {"include": fields}, "with_vector": False})["result"]
    return {str(p["id"]): (p.get("payload") or {}) for p in r}


def twin_encoder(collection: str) -> Optional[Callable]:
    """The write encoding of a TWIN: None for a raw (identity) collection, else the collection's
    own `Projector.project` (a coded twin stores codes, task 855). An undeclared collection is
    refused, as for every writer: the form is never guessed (a twin setting moved before its
    declaration landed would otherwise take raw strings)."""
    pj = PP.Projector.for_collection(collection)
    return None if pj.identity else pj.project


def set_payload(api, collection: str, ids: list, payload: dict) -> None:
    api("POST", f"/collections/{collection}/points/payload?wait=true",
        {"payload": payload, "points": ids})


def _sig(writes: dict) -> tuple:
    return tuple(sorted(writes.items()))


class _Writer:
    """Buffers ids per identical write-set; flushes by id on a small worker pool; reads
    a sample of each flushed batch back. `drain()` waits for every in-flight write and
    re-raises the first failure, so a checkpoint never outruns a write."""

    def __init__(self, api, collection: str, encode: Optional[Callable], batch: int,
                 workers: int, readback: int, guard=None, pause_s: float = 0.0):
        self.api, self.collection, self.encode = api, collection, encode
        self.batch, self.readback, self.guard, self.pause_s = batch, readback, guard, pause_s
        self.buf: dict = collections.defaultdict(list)
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers))
        self.pending: list = []
        self.written = collections.Counter()      # rule -> ids written
        self.wrote_ids = 0
        self.rng = random.Random(684)

    def add(self, pid, writes: dict, rule: str):
        k = (_sig(writes), rule)
        self.buf[k].append(pid)
        if len(self.buf[k]) >= self.batch:
            self._flush(k)

    def _flush(self, k):
        ids = self.buf.pop(k, [])
        if not ids:
            return
        sig, rule = k
        writes = dict(sig)
        payload = self.encode(writes) if self.encode else writes
        self.pending.append(self.pool.submit(self._send, ids, payload, writes, rule))
        self._reap(keep=4 * max(1, self.pool._max_workers))

    def _send(self, ids, payload, writes, rule):
        if self.guard:
            self.guard.tick()
        set_payload(self.api, self.collection, ids, payload)
        if self.readback:
            sample = self.rng.sample(ids, min(self.readback, len(ids)))
            got = fetch_by_id(self.api, self.collection, sample, list(payload))
            for pid in sample:
                if got.get(str(pid)) != payload:
                    raise RuntimeError(f"readback mismatch id={pid} want={payload} have={got.get(str(pid))}")
        if self.pause_s:
            time.sleep(self.pause_s)
        return len(ids), rule

    def _collect(self, f):
        n, rule = f.result()
        self.written[rule] += n
        self.wrote_ids += n

    def _reap(self, keep: int = 0):
        while self.pending and (len(self.pending) > keep or self.pending[0].done()):
            self._collect(self.pending.pop(0))

    def drain(self):
        for k in list(self.buf):
            self._flush(k)
        self._reap(keep=0)

    def close(self):
        self.pool.shutdown(wait=True)


def apply_twin(api, collection: str, ctx: DeriveContext, *, page: int = 5000, batch: int = 5000,
               workers: int = 4, readback: int = 2, checkpoint_pages: int = 100,
               state: Optional[dict] = None, save_state: Callable = lambda s: None,
               limit: Optional[int] = None, dry: bool = False, log=print,
               encode: Optional[Callable] = None, guard=None, pause_s: float = 0.0,
               deadline: Optional[float] = None) -> dict:
    """Derive and write. `state` (a saved dict) resumes from its `offset`. `dry=True`
    derives and groups exactly the same but sends nothing (the conservation identity is
    then checked on the plan alone). Rows are read as canonical rows; `encode` defaults to
    the twin's own encoding (`twin_encoder`: None on a raw twin). A dry run projects each
    DISTINCT write set once with that encoding and reports `encode_refusals`: a value a coded
    twin has no code for refuses in the dry run, not after the real run wrote earlier batches."""
    t0 = time.time()
    decode = RD.reader(collection)
    probe = None
    probe_note = None
    if encode is None:
        if not dry:
            encode = twin_encoder(collection)
        else:
            try:
                probe = twin_encoder(collection)
            except PP.PayloadProfileError as exc:
                probe_note = f"not checked: {exc}"[:300]
    probed: set = set()
    encode_refusals: dict = {}
    state = dict(state or {})
    if state.get("rules_version") not in (None, RULES_VERSION):
        # A resume keeps the rules it started under: the saved offset and tally were
        # produced by another rule set, and mixing the two would stamp one collection by
        # two derivations with one checkpoint (task 684 review).
        raise RuntimeError(
            f"REFUSED: saved state was written under rules {state['rules_version']!r}, this "
            f"code is {RULES_VERSION!r}; start a new state file (or run the matching code)")
    start = count(api, collection)
    res = state.setdefault("tally", {})
    klass = collections.Counter(res.get("classes", {}))
    rules = collections.Counter({tuple(k.split("|", 1)): v for k, v in res.get("rules", {}).items()})
    fams = collections.defaultdict(collections.Counter)
    for fam, cc in res.get("families", {}).items():
        fams[fam].update(cc)
    rows = int(res.get("rows", 0))
    off = state.get("offset")
    writer = None if dry else _Writer(api, collection, encode, batch, workers, readback, guard, pause_s)
    planned = collections.Counter(res.get("planned_by_rule", {}))
    d_domains = collections.Counter(res.get("d_domains", {}))
    memo: dict = {}
    pages = 0
    session_planned = 0
    last_log = time.time()
    state.update(schema=SCHEMA, rules_version=RULES_VERSION, collection=collection,
                 as_of=ctx.as_of, points_at_start=state.get("points_at_start", start),
                 dry=dry, started_at=state.get("started_at", t0))

    def checkpoint(final=False):
        if writer:
            writer.drain()
        res.update(rows=rows, classes=dict(klass),
                   rules={"|".join(k): v for k, v in rules.items()},
                   families={f: dict(c) for f, c in fams.items()},
                   planned_by_rule=dict(planned), d_domains=dict(d_domains.most_common(300)),
                   written_by_rule=dict(writer.written) if writer else {})
        state.update(offset=off, tally=res, saved_at=time.time())
        save_state(state)

    try:
        while True:
            body = {"limit": page, "with_payload": {"include": SCAN_FIELDS}, "with_vector": False}
            if off is not None:
                body["offset"] = off
            r = api("POST", f"/collections/{collection}/points/scroll", body)["result"]
            for pt in r["points"]:
                p = decode(pt.get("payload"))
                dom = str(p.get("domain") or "")
                if dom in PER_URL_DOMAINS:
                    d = derive_consent_flags(p, ctx)
                    fam = family_of_row(p, ctx) or "(unmapped)"
                else:
                    k = (p.get("provenance"), dom, host_of(p.get("url")), p.get("license"),  # canonical row
                         p.get("search_ok"), p.get("train_ok"), p.get("consent_state"),  # canonical row
                         p.get("consent_basis"))  # canonical row
                    hit = memo.get(k)
                    if hit is None:
                        if len(memo) > 2_000_000:
                            memo.clear()
                        hit = memo[k] = (derive_consent_flags(p, ctx), family_of_row(p, ctx) or "(unmapped)")
                    d, fam = hit
                rows += 1
                klass[d.klass] += 1
                rules[(d.klass, d.rule)] += 1
                fams[fam][d.klass] += 1
                if d.klass == "D":
                    d_domains[dom] += 1
                w = d.writes()
                if w:
                    planned[d.rule] += 1
                    session_planned += 1
                    if writer:
                        writer.add(pt["id"], w, d.rule)
                    elif probe is not None and _sig(w) not in probed:
                        probed.add(_sig(w))
                        try:
                            probe(w)
                        except PP.PayloadProfileError as exc:
                            encode_refusals[repr(_sig(w))[:300]] = str(exc)[:300]
            off = r.get("next_page_offset")
            pages += 1
            if off is None or (limit and rows >= limit):
                break
            if deadline is not None and time.monotonic() > deadline:
                log(f"window over at offset {off}; checkpointing and stopping")
                break
            if pages % checkpoint_pages == 0:
                checkpoint()
                now = time.time()
                log(f"rows={rows} written={writer.wrote_ids if writer else 0} "
                    f"rate={rows / max(1, now - t0):.0f}/s offset={off}")
            elif time.time() - last_log >= 60:
                last_log = time.time()
                log(f"rows={rows} written={writer.wrote_ids if writer else 0} offset={off}")
        finished = off is None
        checkpoint(final=True)
    finally:
        if writer:
            writer.close()
    end = count(api, collection)
    n_ab = klass["A"] + klass["B"]
    if not dry and not limit:
        assert writer.wrote_ids == session_planned, \
            f"written {writer.wrote_ids} != planned {session_planned}"
    assert rows == sum(klass.values()), "class totals do not sum to rows scanned"
    lo, hi = min(state["points_at_start"], end), max(state["points_at_start"], end)
    out = {"schema": SCHEMA, "collection": collection, "as_of": ctx.as_of, "dry": dry,
           "points_at_start": state["points_at_start"], "rows_scanned": rows, "points_at_end": end,
           "scan_complete": bool(finished and not limit and lo <= rows <= hi),
           "classes": dict(klass), "ab_rows": n_ab,
           "planned_by_rule": dict(planned), "d_domains_top": dict(d_domains.most_common(40)),
           "written_by_rule": dict(writer.written) if writer else {},
           "encode_refusals": (encode_refusals if probe_note is None
                               else {"<twin encoding>": probe_note}),
           "rules": {"|".join(k): v for k, v in rules.items()},
           "families": {f: dict(c) for f, c in fams.items()},
           "seconds": round(time.time() - t0, 1)}
    state["result"] = out
    save_state(state)
    return out


# ------------------------------------------------------------------ the prod leg
def written_by_adr115(p: dict, as_of: str) -> bool:
    """True for a twin row this ADR wrote: its basis is one of the five written bases, or
    it is an r5/r0 row that kept the dump basis and carries the apply's own stamp."""
    b = p.get("consent_basis")  # canonical row
    if b in WRITTEN_BASES:
        return True
    return b in DUMP_CONSENT_BASES and p.get("consent_checked_at") == as_of


def _carry(p: dict) -> dict:
    return {k: p[k] for k in WRITE_FIELDS if k in p}


def _twin_written_pages(twin_api, twin: str, as_of: str, page: int, limit: Optional[int] = None,
                        start_offset=None):
    """Yield (rows this ADR wrote on this page, offset AFTER the page)."""
    off = start_offset
    seen = 0
    decode = RD.reader(twin)
    while True:
        body = {"limit": page, "with_payload": {"include": list(WRITE_FIELDS)}, "with_vector": False}
        if off is not None:
            body["offset"] = off
        r = twin_api("POST", f"/collections/{twin}/points/scroll", body)["result"]
        rows = [(str(pt["id"]), decode(pt.get("payload"))) for pt in r["points"]]
        got = [(pid, _carry(p)) for pid, p in rows if written_by_adr115(p, as_of)]
        seen += len(r["points"])
        off = r.get("next_page_offset")
        yield got, off
        if off is None or (limit and seen >= limit):
            return


def _classify_prod(prod_row: Optional[dict], want: dict, encode: Optional[Callable]) -> str:
    """absent | write | same | conflict | partial. Only `write` is ever written."""
    if prod_row is None:
        return "absent"
    s, t = prod_row.get("search_ok"), prod_row.get("train_ok")
    sb, tb = isinstance(s, bool), isinstance(t, bool)
    if sb and tb:
        enc = encode(want) if encode else want
        return "same" if all(prod_row.get(k) == enc.get(k) for k in ("search_ok", "train_ok")) else "conflict"
    if sb or tb:
        return "partial"
    return "write"


def plan_prod(twin_api, twin: str, prod_api, prod: str, as_of: str, encode: Callable, *,
              page: int = 2000, limit: Optional[int] = None, log=print,
              state: Optional[dict] = None, save_state: Callable = lambda s: None,
              decode: Optional[Callable] = None, deadline: Optional[float] = None) -> dict:
    """Walk the applied twin; for the rows this ADR wrote, read the prod rows by id and
    classify. Plans only: it never writes prod and never reads prod by scroll or by filter."""
    t0 = time.time()
    state = dict(state or {})
    tally = collections.Counter(state.get("tally", {}))
    twin_n, prod_n = count(twin_api, twin), count(prod_api, prod)
    decoded_bad = 0
    examples: dict = {}
    finished = False
    for got, off in _twin_written_pages(twin_api, twin, as_of, page, limit, state.get("offset")):
        ids = [i for i, _ in got]
        have = fetch_by_id(prod_api, prod, ids, ["search_ok", "train_ok", "consent_basis", "consent_checked_at"])
        for pid, want in got:
            tally["twin_written_rows"] += 1
            c = _classify_prod(have.get(pid), want, encode)
            tally[c] += 1
            examples.setdefault(c, pid)
            if c == "write":
                if decode is not None:
                    # the projected row must decode to the twin's raw writes
                    back = decode(encode(want))
                    if any(back.get(k) != want.get(k) for k in WRITE_FIELDS):
                        decoded_bad += 1
        if tally["twin_written_rows"] % 200000 < len(got):
            log(f"twin_written={tally['twin_written_rows']} {dict(tally)}")
        state.update(offset=off, tally=dict(tally))
        save_state(state)
        if off is None:
            finished = True
        if deadline is not None and time.monotonic() > deadline and off is not None:
            log(f"window over at offset {off}; stopping cleanly (resume with the same state file)")
            break
    prod_after = count(prod_api, prod)
    out = {"schema": SCHEMA, "mode": "plan", "twin": twin, "prod": prod,
           "as_of": as_of, "twin_points": twin_n, "prod_points_before": prod_n,
           "prod_points_after": prod_after, "classes": dict(tally), "decode_mismatch": decoded_bad,
           "example_ids": examples, "written": 0,
           "finished": finished, "offset": state.get("offset"),
           "seconds": round(time.time() - t0, 1)}
    return out


def sample_twin_ids(api, collection: str, n: int, seed: int) -> list:
    """Random ids by random uuid offsets, limit 1 (the roundtrip probe's method)."""
    rng = random.Random(seed)
    out: list = []
    guard = 0
    while len(out) < n and guard < n * 5:
        guard += 1
        off = str(uuid.UUID(int=rng.getrandbits(128)))
        r = api("POST", f"/collections/{collection}/points/scroll",
                {"limit": 1, "offset": off, "with_payload": False, "with_vector": False})["result"]
        for pt in r["points"]:
            if str(pt["id"]) not in out:
                out.append(str(pt["id"]))
    return out[:n]


def sample_prod_projection(twin_api, twin: str, prod_api, prod: str, as_of: str,
                           encode: Callable, decode: Callable, n: int, seed: int) -> dict:
    """READ-ONLY dry-run of the prod projection on a random sample of twin rows: for each
    row this ADR wrote on the twin, what prod holds, what the write would be, and whether
    the projected write decodes back to the twin's raw values."""
    ids = sample_twin_ids(twin_api, twin, n, seed)
    tw = RD.reader(twin).decode_map(fetch_by_id(twin_api, twin, ids, SCAN_FIELDS))
    pr = fetch_by_id(prod_api, prod, ids, SCAN_FIELDS)
    cls = collections.Counter()
    bad = []
    bases = collections.Counter()
    for pid in ids:
        t = tw.get(pid)
        if t is None:
            cls["twin_absent"] += 1
            continue
        if not written_by_adr115(t, as_of):
            cls["twin_not_written_by_adr115"] += 1
            continue
        want = _carry(t)
        enc = encode(want)
        if decode(enc) != want:
            bad.append(pid)
        c = _classify_prod(pr.get(pid), want, encode)
        cls[c] += 1
        bases[t.get("consent_basis")] += 1  # canonical row
    return {"sampled": len(ids), "classes": dict(cls), "bases": dict(bases),
            "decode_mismatch": len(bad), "decode_mismatch_ids": bad[:5], "seed": seed}


# ------------------------------------------------------------------- the driver (task 684, ported 1241)
# `cyt consent backfill {twin,prod-sample,prod-plan}`. Run on the compute box (next to the stores).
# Nothing here scans prod. There is NO prod-apply verb: it imported a module task 683 deleted and
# could not run. A prod write of this backfill, if it is ever needed again, goes through the twin-first
# door: `cytir.index.twin_first.ProductionWriterLock`.
TASK = "684"
PURPOSE = "consent-backfill-adr115"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _save(path: str):
    def save(state: dict) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=1, default=str)
        os.replace(tmp, path)
    return save


def _ctx(a) -> DeriveContext:
    from . import arxiv_licence  # noqa: PLC0415
    from .derive import load_cache_index, load_sealed_probes  # noqa: PLC0415
    contracts = json.load(open(a.contracts))
    return DeriveContext(families=contracts["families"],
                         cache_index=load_cache_index(os.path.expanduser(a.cache_dir)),
                         as_of=a.as_of, arxiv_licences=arxiv_licence.load_map(a.arxiv_licences),
                         sealed_probes=load_sealed_probes(contracts["families"], str(pathlib.Path(a.contracts).resolve().parents[1])))


def resolve_alias(api, alias: str) -> str:
    for e in api("GET", "/aliases")["result"]["aliases"]:
        if e["alias_name"] == alias:
            return e["collection_name"]
    raise SystemExit(f"alias {alias!r} not found")


def _projector(prod_collection: str):
    pj = PP.Projector.for_collection(prod_collection)
    if pj.identity:
        return (lambda w: dict(w)), (lambda p: dict(p)), pj
    return pj.project, (lambda p: PP.decode_payload(p, pj.profile, pj.tables)), pj


def main(argv=None, prog: str | None = None) -> int:
    from cytir.core import paths as PATHS  # noqa: PLC0415
    from cytir.core import twin as TWIN  # noqa: PLC0415
    ap = argparse.ArgumentParser(prog=prog, description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="verb", required=True)

    def common(p):
        p.add_argument("--twin-url", default=TWIN.url())
        p.add_argument("--collection", default=TWIN.collection(), help="the twin collection")
        p.add_argument("--contracts", default=str(PATHS.data_path("source-family-contracts.v1.json")))
        p.add_argument("--cache-dir", default="~/aw-data/consent")
        p.add_argument("--arxiv-licences", required=True)
        p.add_argument("--as-of", required=True, help="UTC stamp written as consent_checked_at on instrument rows")
        p.add_argument("--state", required=True)
        p.add_argument("--page", type=int, default=5000)

    t = sub.add_parser("twin", help="derive + write the four consent fields on the TWIN")
    common(t)
    t.add_argument("--dry", action="store_true")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--batch", type=int, default=5000)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--stop-timer", default="aw-promotion-nightly.timer")
    t.add_argument("--wait-service", default="aw-promotion-nightly.service")
    t.add_argument("--wait-s", type=float, default=7200)
    t.add_argument("--max-minutes", type=float, default=None,
                   help="window length once the lock is held; stops at a checkpoint, resume with --resume")
    for name, what in (("prod-sample", "READ-ONLY dry run of the prod projection on a twin sample"),
                       ("prod-plan", "READ-ONLY: the same for every twin row from this ADR, by id")):
        p = sub.add_parser(name, help=what)
        common(p)
        p.add_argument("--prod-url", required=True)
        p.add_argument("--prod-alias", default="aw_prod")
        p.add_argument("--limit", type=int, default=None, help="twin rows scanned (smoke)")
        p.add_argument("--out", required=True)
        if name == "prod-sample":
            p.add_argument("--n", type=int, default=2000)
            p.add_argument("--seed", type=int, default=684)
    a = ap.parse_args(argv)

    if a.verb == "twin":
        if not a.dry:
            # task 756 / review 776: no receipt-less twin write while an aw_e10 build holds the twin
            from cytir.index import twin_write_hold as TWH  # noqa: PLC0415
            try:
                TWH.refuse_if_held(a.collection, tool="cyt consent backfill twin")
            except TWH.TwinWriteHeld as exc:
                raise SystemExit(str(exc)) from exc
        return _run_twin(a)
    return _run_prod(a)


def _refuse_prod_twin(url: str, collection: str) -> None:
    """The twin door is TWIN-ONLY (review 1294): refuse a prod endpoint or a live collection name
    before any HTTP call. A prod write goes through `cytir.index.twin_first.ProductionWriterLock`."""
    from cytir.index import live_target as LT  # noqa: PLC0415
    if PG.is_prod(url, collection) or collection in LT.live_collection_names():
        raise SystemExit(f"REFUSED: `cyt consent backfill twin` writes a TWIN only; {collection!r} at "
                         f"{url!r} is a live/prod target. Use the twin-first door "
                         "(cytir.index.twin_first.ProductionWriterLock).")


def _run_twin(a) -> int:
    _refuse_prod_twin(a.twin_url, a.collection)      # first: before any HTTP call
    api = http_api(a.twin_url)
    ctx = _ctx(a)
    save = _save(a.state)
    state = json.load(open(a.state)) if (a.resume and os.path.exists(a.state)) else None
    if state and state.get("as_of") != a.as_of:
        raise SystemExit("REFUSED: --as-of differs from the saved state; a resume keeps one stamp")
    log = lambda m: print(m, flush=True)  # noqa: E731
    if a.dry:
        out = apply_twin(api, a.collection, ctx, dry=True, page=a.page, limit=a.limit,
                         state=state, save_state=save, log=log)
        print(json.dumps(out, indent=1))
        return 0 if out["scan_complete"] or a.limit else 3
    from cytir.core.stopped_timer import StoppedTimer  # noqa: PLC0415
    from cytir.ops import write_lock as WL  # noqa: PLC0415
    timer = StoppedTimer(a.stop_timer, wait_unit=a.wait_service, wait_s=a.wait_s) if a.stop_timer else None
    got_lock = False
    try:
        if timer:
            timer.__enter__()
        WL.acquire("TWIN-WRITE-LOCK", TASK, PURPOSE)
        got_lock = True
        log(f"twin lock taken {_now()}")
        out = apply_twin(api, a.collection, ctx, page=a.page, batch=a.batch, workers=a.workers,
                         limit=a.limit, state=state, save_state=save, log=log,
                         deadline=(time.monotonic() + a.max_minutes * 60) if a.max_minutes else None)
    finally:
        if got_lock:
            WL.release("TWIN-WRITE-LOCK", TASK)
        if timer:
            timer.__exit__(None, None, None)
    print(json.dumps(out, indent=1))
    return 0 if out["scan_complete"] or a.limit else 3


def _run_prod(a) -> int:
    """prod-sample / prod-plan: read only, by id."""
    twin_api, prod_api = http_api(a.twin_url), http_api(a.prod_url)
    prod = resolve_alias(prod_api, a.prod_alias)
    encode, decode, pj = _projector(prod)
    log = lambda m: print(m, flush=True)  # noqa: E731
    if a.verb == "prod-sample":
        out = sample_prod_projection(twin_api, a.collection, prod_api, prod, a.as_of, encode,
                                     decode, a.n, a.seed)
        out.update(prod_collection=prod, profile=pj.profile, tables_sha256=PP.file_sha256(PP.TABLES_PATH))
    else:
        out = plan_prod(twin_api, a.collection, prod_api, prod, a.as_of, encode, decode=decode,
                        limit=a.limit, log=log)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(prog="python -m cytir.consent.apply"))
