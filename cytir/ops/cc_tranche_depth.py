"""Depth continuation of finished CC tranches (task 2194; follow-up of task 2174).

A host of tranches 7-20 that hit its page cap holds more captures in the crawl. No verb continued such a host:
`discovery_ingest.process_domain` takes its depth from the roster row `max_pages`, and a host in `state.json`
`done` is skipped. This module adds the missing step AROUND `discovery_ingest` (that module is not touched):

  roster  seeded random sample of capped hosts (cap 60, so the step is 60 -> 120), consent re-checked at seal
  fetch   `discovery_ingest fetch` with the columnar resolver narrowed to the NEW BAND only
  gate    the pre-landing numbers of the first hosts; STOP below the floors

The band. For a fixed crawl the selection of one host is deterministic: `dedup_rank(resolved[:3 * D])[:D]`
(what `process_domain` does at depth D). The continuation fetches exactly `select(D_new) - select(D_old)`;
a URL of the old selection is never resolved again for fetch, so it is never re-fetched or re-staged. A roster
row carries `band_from` (old depth) and `band_to` (new depth); its `max_pages` is the band size, so
`process_domain` keeps the whole band. A host with no new URL yields an empty resolve and ends `empty`.

The run is a tranche of the documented recipe: its queue is `tranches/tranche21_never_queued.json`, its out dir
`tranches/t21_out` (own `state.json`, resumable, the shared egress bucket), sealed by the same seal watchdog.

    python -m cytir.ops.cc_tranche_depth roster --tranches-dir D --out Q [--n 2000 --seed 2194 --report R]
    python -m cytir.ops.cc_tranche_depth fetch <discovery_ingest fetch arguments>
    python -m cytir.ops.cc_tranche_depth gate --queue Q --out-dir O [--first 200] [--twin URL --collection C]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DEPTH_FROM = 60
DEPTH_TO = 120
SEED = 2194
SLICE_HOSTS = 2000
FIRST_HOSTS = 200
# Gate floors (task 2194): stop below either. 2174 measured 4.115 rows per page and a landed share near 0.93.
MIN_ROWS_PER_PAGE = 2.0
MIN_LANDED_SHARE = 0.8
FIRST_TRANCHE, LAST_TRANCHE = 7, 20


def select(records: list[dict], depth: int) -> list[dict]:
    """The records `process_domain` fetches at `depth` from a resolved list (ordered shallowest first)."""
    from cytir.acquire.discovery_ingest import dedup_rank
    return dedup_rank(records[:3 * depth], "")[:depth]


def band(records: list[dict], old_depth: int, new_depth: int) -> list[dict]:
    """`select(new) - select(old)`, by URL, in fetch order. Empty when the host holds nothing beyond `old_depth`."""
    old = {r["url"] for r in select(records, old_depth)}
    return [r for r in select(records, new_depth) if r["url"] not in old]


# --- roster ------------------------------------------------------------------------------------------------

def _consent_sets(tranche_dir: Path, extra_declined: set[str]) -> tuple[set[str], set[str], bool]:
    gp = tranche_dir / "gated" / "gate_summary.json"
    if not gp.is_file():
        return set(), set(), False
    g = json.loads(gp.read_text())
    declined = {h["host"] for h in g.get("declined_hosts", [])} | extra_declined
    unknown = {h["host"] for h in g.get("unknown_hosts", [])}
    return declined, unknown, True


def build_roster(tranches_dir: Path, *, n: int = SLICE_HOSTS, seed: int = SEED,
                 declined_register: set[str] = frozenset(), exclude_hosts: set[str] | None = None) -> tuple[list[dict], dict]:
    """Seeded random `n` hosts from the cap-60 hosts that hit the cap, in tranches 7-20. Returns (rows, report).

    A host whose consent is declined or unknown at seal (tranche gate summary, or the declined register) is
    skipped and counted; a tranche with no gate summary has all its hosts skipped (counted `no_gate_summary`).
    `exclude_hosts` (e.g. the hosts of an earlier slice) is removed from the pool before the seeded draw and
    counted as `excluded`. No new host: every row is a host of a finished tranche. No consent is changed.
    """
    from cytir.acquire.prep_discovery_queue import norm_host   # the canonical host key the queue and state use
    pool, skipped, excluded = [], collections.Counter(), 0
    exclude_hosts = {norm_host(h) for h in (exclude_hosts or ())}
    for t in range(FIRST_TRANCHE, LAST_TRANCHE + 1):
        out = tranches_dir / f"t{t}_out"
        queue = tranches_dir / f"tranche{t}_never_queued.json"
        if not (out / "state.json").is_file() or not queue.is_file():
            skipped["tranche_missing"] += 1
            continue
        done = json.loads((out / "state.json").read_text()).get("done", {})
        rows = {r["domain"]: r for r in json.loads(queue.read_text())}
        declined, unknown, have_gate = _consent_sets(out, set(declined_register))
        for dom in sorted(done):
            v = done[dom]
            row = rows.get(dom)
            if v.get("status") != "ok" or row is None:
                continue
            cap = row.get("max_pages") or DEPTH_FROM
            if cap != DEPTH_FROM or v.get("pages", 0) < cap:
                skipped["not_cap60_capped"] += 1
                continue
            if not have_gate:
                skipped["no_gate_summary"] += 1
            elif dom in declined or dom in declined_register:
                skipped["consent_declined"] += 1
            elif dom in unknown:
                skipped["consent_unknown"] += 1
            elif norm_host(dom) in exclude_hosts:
                excluded += 1
            else:
                pool.append((t, dom, row))
    chosen = random.Random(seed).sample(pool, min(n, len(pool)))   # sample() order is the roster order
    # keep the seeded-random order: the first N hosts of the queue are then a random subsample (the gate reads them)
    out_rows = []
    for t, dom, row in chosen:
        r = {k: row[k] for k in ("domain", "class", "class_basis", "n_endorser_regs") if k in row}
        r.update(max_pages=DEPTH_TO - DEPTH_FROM, band_from=DEPTH_FROM, band_to=DEPTH_TO, src_tranche=t)
        out_rows.append(r)
    report = {"seed": seed, "pool_cap60_consent_ok": len(pool), "hosts": len(out_rows),
              "skipped": dict(skipped), "per_tranche": dict(sorted(collections.Counter(r["src_tranche"] for r in out_rows).items())),
              "depth": [DEPTH_FROM, DEPTH_TO]}
    if exclude_hosts:   # only when asked: the no-flag report stays byte-identical
        report["excluded_hosts_given"] = len(exclude_hosts)
        report["excluded"] = excluded
    return out_rows, report


# --- fetch -------------------------------------------------------------------------------------------------

@contextmanager
def band_resolver(queue: list[dict]):
    """Narrow `cc_columnar.resolve_domains` (as `discovery_ingest.cmd_fetch` calls it) to each host's band."""
    from cytir.acquire import cc_columnar as ccx
    depth = {q["domain"]: (int(q.get("band_from", 0)), int(q["band_to"])) for q in queue if "band_to" in q}
    orig = ccx.resolve_domains

    def narrowed(crawl_id, domains, con=None, per_domain=None, **kw):
        want = max((depth[d][1] for d in domains if d in depth), default=0)
        got = orig(crawl_id, domains, con=con, per_domain=max(per_domain or 0, want * 3) or None, **kw)
        out = {}
        for d, recs in got.items():
            lo, hi = depth.get(d, (0, 0))
            out[d] = band(recs, lo, hi) if hi else recs
        return out

    ccx.resolve_domains = narrowed
    try:
        yield
    finally:
        ccx.resolve_domains = orig


def _opt(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None


def mark_probe(argv: list[str], roster_size: int) -> None:
    """A `--limit-domains` run is a probe: its `FETCH done.` line is not a finished tranche (task 2216).

    `--out` is the passages FILE; the marker goes in its parent, the tranche dir the watchdog reads
    (`<dir>/.probe`). Written BEFORE fetching. A full run (no `--limit-domains`) that resumes over an
    existing `--state` file removes a stale marker; a fresh start has none to remove."""
    out_s = _opt(argv, "--out")
    if not out_s:
        return
    d = Path(out_s).parent
    marker = d / ".probe"
    limit = int(_opt(argv, "--limit-domains") or 0)
    if limit > 0:
        d.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        marker.write_text(f"limit={limit} roster={roster_size} time={stamp}\n")
    elif marker.exists():
        state_s = _opt(argv, "--state")
        if state_s and Path(state_s).exists():
            marker.unlink()


def run_fetch(argv: list[str]) -> int:
    from cytir.acquire import discovery_ingest as di
    if "--queue" not in argv:
        print("REFUSED: fetch needs --queue", file=sys.stderr)
        return 2
    queue = json.loads(Path(argv[argv.index("--queue") + 1]).read_text())
    bad = [q.get("domain") for q in queue if "band_to" not in q or int(q.get("band_from", 0)) >= int(q["band_to"])]
    if bad:
        print(f"REFUSED: {len(bad)} queue row(s) carry no band (band_from < band_to), e.g. {bad[:3]}", file=sys.stderr)
        return 2
    mark_probe(argv, len(queue))
    old = sys.argv
    sys.argv = ["discovery_ingest", "fetch", *argv]
    try:
        with band_resolver(queue):
            di.main()
    finally:
        sys.argv = old
    return 0


# --- gate --------------------------------------------------------------------------------------------------

def gate(queue: list[dict], state: dict, passages: Path, *, first: int = FIRST_HOSTS, served_urls=None) -> dict:
    """The numbers of the first `first` hosts of the queue that are done, against the floors.

    rows_per_page = rows / fetched pages. landed_share = rows that are distinct by digest, not served on the
    twin (`served_urls`), and from a host consented at roster seal / rows. The consent and junk gates of the
    seal chain run later; this share is their upper bound, taken BEFORE anything is staged.
    """
    done = state.get("done", {})
    hosts = [q["domain"] for q in queue if q["domain"] in done][:first]
    pages = sum(done[h].get("pages", 0) for h in hosts)
    rows = sum(done[h].get("rows", 0) for h in hosts)
    keep = set(hosts)
    seen, urls, digests, row_urls = collections.Counter(), {}, collections.Counter(), []
    with passages.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("domain") not in keep:
                continue
            urls.setdefault(r["url"], None)
            row_urls.append(r["url"])
            seen[r["domain"]] += 1
            digests[r.get("digest") or r.get("point_id") or r["url"]] += 1
    served: set[str] = set()
    if served_urls is not None:
        batch = list(urls)
        for i in range(0, len(batch), 500):
            served |= served_urls(batch[i:i + 500])
    n_rows = sum(seen.values())
    dup_digest = sum(c - 1 for c in digests.values())
    served_rows = sum(1 for u in row_urls if u in served)
    landed = max(n_rows - dup_digest - served_rows, 0)
    rpp = rows / pages if pages else 0.0
    share = landed / n_rows if n_rows else 0.0
    verdict = "GO" if (len(hosts) >= min(first, len(queue)) and rpp >= MIN_ROWS_PER_PAGE and share >= MIN_LANDED_SHARE) else "STOP"
    reason = None
    if served_urls is None:
        verdict, reason = "STOP", "served-url check not run (pass --twin)"
    return {"hosts_done": len(hosts), "pages": pages, "rows": rows, "rows_per_page": round(rpp, 3), "reason": reason,
            "rows_read": n_rows, "dup_digest_rows": dup_digest, "served_rows": served_rows,
            "served_checked": served_urls is not None, "landed_share": round(share, 4),
            "floors": {"rows_per_page": MIN_ROWS_PER_PAGE, "landed_share": MIN_LANDED_SHARE},
            "estimate_2174": {"rows_per_page": 4.115, "landed_share": 0.93}, "verdict": verdict}


# --- cli ---------------------------------------------------------------------------------------------------

def main(argv=None, prog: str = "cc_tranche_depth") -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["fetch"]:
        return run_fetch(argv[1:])
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("roster")
    r.add_argument("--tranches-dir", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--report", type=Path)
    r.add_argument("--n", type=int, default=SLICE_HOSTS)
    r.add_argument("--seed", type=int, default=SEED)
    r.add_argument("--declined-register", type=Path, help="the shipped consent-declined register (consent_declined_hosts.json, cytir.core.paths.data_path)")
    r.add_argument("--exclude-roster", type=Path, action="append", default=[],
                   help="an earlier roster JSON (same format this verb writes); its hosts are excluded (repeatable)")
    g = sub.add_parser("gate")
    g.add_argument("--queue", type=Path, required=True)
    g.add_argument("--out-dir", type=Path, required=True)
    g.add_argument("--first", type=int, default=FIRST_HOSTS)
    g.add_argument("--twin")
    g.add_argument("--collection")
    g.add_argument("--report", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "roster":
        if a.out.exists():
            print(f"REFUSED: {a.out} exists", file=sys.stderr)
            return 2
        reg = set()
        if a.declined_register:
            reg = set(json.loads(a.declined_register.read_text()).get("hosts", {}) or [])
        excl = {r["domain"] for p in a.exclude_roster for r in json.loads(p.read_text())}
        rows, rep = build_roster(a.tranches_dir, n=a.n, seed=a.seed, declined_register=reg,
                                 exclude_hosts=excl or None)
        if a.exclude_roster:   # each exclude roster is named by path and sha256 in the report
            rep["exclude_rosters"] = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                      for p in a.exclude_roster]
        if not rows:
            print("REFUSED: nothing left to roster", file=sys.stderr)
            return 2
        a.out.write_text(json.dumps(rows, indent=0))
        if a.report:
            a.report.write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep))
        return 0
    served = None
    if a.twin:
        from cytir.ops.cc_tranche_delta import _twin_served_urls
        served = _twin_served_urls(a.twin, a.collection)
    res = gate(json.loads(a.queue.read_text()), json.loads((a.out_dir / "state.json").read_text()),
               a.out_dir / "passages.jsonl", first=a.first, served_urls=served)
    print(json.dumps(res, indent=1))
    if a.report:
        a.report.write_text(json.dumps(res, indent=1))
    return 0 if res["verdict"] == "GO" else 3


if __name__ == "__main__":
    raise SystemExit(main())
