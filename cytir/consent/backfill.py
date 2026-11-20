"""Consent-flag backfill: DRY-RUN reader (ADR-115). It never writes a payload.

    python -m cytir.consent.backfill derive --qdrant <twin-url> --collection <twin-collection> \\
        --contracts data/source-family-contracts.v1.json --cache-dir ~/aw-data/consent \\
        --arxiv-licences arxiv-licences.tsv --as-of 2026-09-30T00:00:00Z --out plan.json
    python -m cytir.consent.backfill arxiv-licences --raw-dir <arxiv_raw> --out <tsv>

`derive` scrolls a TWIN (payload fields only, no vectors), runs
`cytir.consent.derive.derive_consent_flags` on every row and reports, per class, per
rule and per family, what an apply WOULD write. `--apply` is refused here: the apply
runs after Tim merges ADR-115, twin first and then prod, through the payload writer
under the prod writer lock, as a separate task (deployment-doctrine).

Scan completeness is a printed identity (points at start / rows scanned / points at
end), and the class totals must sum to the rows scanned, or the run exits non-zero.
Never point this at the prod serving box (prod-serving-box-fragile-scrolls).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import urllib.request
from cytir.index import prod_guard as PG  # noqa: E402 -- no prod scroll, no prod replay (2026-10-07)
from cytir.index import row_decode as RD  # noqa: E402 -- canonical stored rows, any profile (task 855)

from . import arxiv_licence
from .derive import (RULES_VERSION, DeriveContext, derive_consent_flags, family_of_row, host_of,
                     load_sealed_probes,
                     load_cache_index)

SCHEMA = "aw.consent-backfill-plan.v1"
FIELDS = ["url", "domain", "provenance", "license", "search_ok", "train_ok",
          "consent_state", "consent_basis", "consent_checked_at"]
# Domains whose rows are derived from the row's URL (per-paper licence): not memoised.
PER_URL_DOMAINS = frozenset({"arxiv.org", "pubmed.ncbi.nlm.nih.gov"})


def _post(url: str, body: dict, timeout: float = 180.0, tries: int = 5):
    PG.check_url(url, "POST")               # no prod scroll, no prod replay (2026-10-07)
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, json.dumps(body).encode(),
                                         {"content-type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(30, 2 ** i))
    raise RuntimeError(f"qdrant call failed after {tries} tries: {url}: {last!r}")


def exact_count(qdrant: str, collection: str) -> int:
    r = _post(f"{qdrant}/collections/{collection}/points/count", {"exact": True}, timeout=900)
    return int(r["result"]["count"])


class Tally:
    """Every counter the PR table needs; one place, so the identity is checkable."""

    def __init__(self):
        self.rows = 0
        self.klass = collections.Counter()
        self.rule = collections.Counter()
        self.fam_class = collections.defaultdict(collections.Counter)
        self.fam_train_false = collections.Counter()
        self.fam_search_false = collections.Counter()
        self.d_reason = collections.Counter()
        self.d_domain = collections.Counter()
        self.carried = collections.Counter()
        self.www_apex_rows = 0
        self.cache_variant = collections.Counter()
        self.licence_bucket = collections.Counter()     # (rule, label, train) -> rows
        self.before_explicit = 0
        self.basis = collections.Counter()

    def add(self, row: dict, fam: str, d, n: int = 1):
        self.rows += n
        self.klass[d.klass] += n
        self.rule[(d.klass, d.rule)] += n
        self.fam_class[fam][d.klass] += n
        if d.klass == "E":
            self.before_explicit += n
        if d.klass in ("A", "B"):
            self.basis[d.consent_basis] += n
            if d.train_ok is False:
                self.fam_train_false[fam] += n
            if d.search_ok is False:
                self.fam_search_false[fam] += n
            if d.rule.startswith(("r1-arxiv", "r1-pubmed")):
                self.licence_bucket[(d.rule, d.detail, d.train_ok)] += n
            if d.klass == "B":
                self.cache_variant[d.detail] += n
                if d.detail.startswith("www-apex"):
                    self.www_apex_rows += n
        if d.klass == "D":
            self.d_reason[d.rule] += n
            self.d_domain[str(row.get("domain") or "")] += n
        if row.get("consent_state") == "carried":  # canonical row
            self.carried[(d.klass, d.rule, d.search_ok)] += n


def run_derive(qdrant: str, collection: str, ctx: DeriveContext, page: int = 5000,
               limit: int | None = None, log=print) -> dict:
    t0 = time.time()
    start = exact_count(qdrant, collection)
    log(f"start_count={start} at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    tally = Tally()
    memo: dict = {}
    off = None
    last_log = t0
    url = f"{qdrant}/collections/{collection}/points/scroll"
    decode = RD.reader(collection)          # a coded twin reads like a raw one (task 855)
    while True:
        body = {"limit": page, "with_payload": {"include": FIELDS}, "with_vector": False}
        if off is not None:
            body["offset"] = off
        res = _post(url, body)["result"]
        for pt in res["points"]:
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
                    hit = memo[k] = (derive_consent_flags(p, ctx), family_of_row(p, ctx) or "(unmapped)")
                d, fam = hit
            tally.add(p, fam, d)
        off = res.get("next_page_offset")
        now = time.time()
        if now - last_log >= 60:
            rate = tally.rows / (now - t0)
            log(f"rows={tally.rows} rate={rate:.0f}/s eta_s={(start - tally.rows) / rate:.0f} keys={len(memo)}")
            last_log = now
        if off is None or (limit and tally.rows >= limit):
            break
    end = exact_count(qdrant, collection)
    lo, hi = min(start, end), max(start, end)
    complete = limit is None and off is None and lo <= tally.rows <= hi
    assert tally.rows == sum(tally.klass.values()), "class totals do not sum to rows scanned"
    doc = {
        "schema": SCHEMA, "rules_version": RULES_VERSION,
        "collection": collection, "qdrant": qdrant, "as_of": ctx.as_of,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 1),
        "points_at_start": start, "rows_scanned": tally.rows, "points_at_end": end,
        "scan_complete": complete, "limit": limit, "applied": False,
        "arxiv_licence_records": len(ctx.arxiv_licences or {}),
        "cache_hosts": len(ctx.cache_index) - 1,
        "cache_unparseable": ctx.cache_index.get("__unparseable__", 0),
        "classes": dict(tally.klass),
        "rules": {f"{k[0]}:{k[1]}": v for k, v in sorted(tally.rule.items(), key=lambda kv: -kv[1])},
        "families": {f: dict(c) for f, c in sorted(tally.fam_class.items(),
                                                   key=lambda kv: -sum(kv[1].values()))},
        "train_false_by_family": dict(tally.fam_train_false.most_common()),
        "search_false_by_family": dict(tally.fam_search_false.most_common()),
        "d_reasons": dict(tally.d_reason.most_common()),
        "d_domains_top": dict(tally.d_domain.most_common(40)),
        "bases_written": dict(tally.basis.most_common()),
        "cache_variants": dict(tally.cache_variant.most_common()),
        "www_apex_rows": tally.www_apex_rows,
        "licence_buckets": [{"rule": k[0], "label": k[1], "train_ok": k[2], "rows": v}
                            for k, v in sorted(tally.licence_bucket.items(), key=lambda kv: -kv[1])],
        "carried": [{"class": k[0], "rule": k[1], "search_ok": k[2], "rows": v}
                    for k, v in sorted(tally.carried.items(), key=lambda kv: -kv[1])],
    }
    log(f"done rows={tally.rows} start={start} end={end} complete={complete}")
    return doc


def _pct(n, d) -> str:
    return f"{100.0 * n / d:.2f}%" if d else "n/a"


def render_md(doc: dict) -> str:
    n = doc["rows_scanned"]
    c = doc["classes"]
    out = [f"collection `{doc['collection']}`: start {doc['points_at_start']:,}, scanned {n:,}, "
           f"end {doc['points_at_end']:,}, complete={doc['scan_complete']}, applied={doc['applied']}", "",
           "| class | rows | share |", "|---|--:|--:|"]
    for k, label in (("E", "E already has both flags (untouched)"), ("A", "A from an instrument (licence / contract)"),
                     ("B", "B from the host's current cache verdict"), ("D", "D no flags written")):
        out.append(f"| {label} | {c.get(k, 0):,} | {_pct(c.get(k, 0), n)} |")
    out.append(f"| total | {n:,} | 100% |")
    out += ["", "| rule | rows |", "|---|--:|"]
    out += [f"| {k} | {v:,} |" for k, v in doc["rules"].items()]
    out += ["", "| family (mapped) | rows | E | A | B | D | train_ok=false written |", "|---|--:|--:|--:|--:|--:|--:|"]
    tf = doc["train_false_by_family"]
    for f, cc in list(doc["families"].items())[:60]:
        out.append(f"| {f} | {sum(cc.values()):,} | {cc.get('E', 0):,} | {cc.get('A', 0):,} | "
                   f"{cc.get('B', 0):,} | {cc.get('D', 0):,} | {tf.get(f, 0):,} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ADR-115 consent-flag backfill (dry-run reader)")
    sub = ap.add_subparsers(dest="verb", required=True)
    d = sub.add_parser("derive")
    d.add_argument("--qdrant", required=True)
    d.add_argument("--collection", required=True)
    d.add_argument("--contracts", required=True)
    d.add_argument("--cache-dir", required=True)
    d.add_argument("--arxiv-licences", required=True)
    d.add_argument("--as-of", required=True, help="UTC stamp written as consent_checked_at on instrument rows")
    d.add_argument("--out", required=True)
    d.add_argument("--md")
    d.add_argument("--page", type=int, default=5000)
    d.add_argument("--limit", type=int, default=None, help="smoke runs only; marks the run incomplete")
    d.add_argument("--apply", action="store_true",
                   help="REFUSED: the apply is a separate task after Tim's merge (payload writer, writer lock)")
    a = sub.add_parser("arxiv-licences")
    a.add_argument("--raw-dir", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.verb == "arxiv-licences":
        print(json.dumps(arxiv_licence.build_map(args.raw_dir, args.out), indent=1))
        return 0
    if args.apply:
        print("refusing: this tool only plans. The apply runs after ADR-115 is merged, twin first, "
              "then prod, through the payload writer under the prod writer lock.", file=sys.stderr)
        return 2
    contracts = json.load(open(args.contracts))
    ctx = DeriveContext(families=contracts["families"],
                        cache_index=load_cache_index(os.path.expanduser(args.cache_dir)),
                        as_of=args.as_of, arxiv_licences=arxiv_licence.load_map(args.arxiv_licences),
                        sealed_probes=load_sealed_probes(
                            contracts["families"], os.path.dirname(os.path.dirname(os.path.abspath(args.contracts)))))
    doc = run_derive(args.qdrant, args.collection, ctx, page=args.page, limit=args.limit,
                     log=lambda m: print(m, flush=True))
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    md = render_md(doc)
    if args.md:
        with open(args.md, "w") as f:
            f.write(md)
    print(md)
    return 0 if doc["scan_complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
