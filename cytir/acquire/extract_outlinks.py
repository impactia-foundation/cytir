"""ADR-020: multi-seed endorsement graph. For each curated seed domain, pull its
pages from Common Crawl (dumps-first, policy (a) — never live-scrape), extract
EXTERNAL outbound links, filter chrome/template links, and aggregate into an
endorsement graph + per-domain authority + discovery URLs.

  python -m cytir.acquire.extract_outlinks --seeds data/seeds_ch.json --out endorse_ch.json

Seed file: JSON array of {"domain","class","name","lang", ("trust": float)?}.

Error-vs-empty contract (2026-07-03, after the 2026-06-28 CDX outage)
---------------------------------------------------------------------
Every seed in the output's `edges` carries `"status": "ok"|"empty"|"error"`.
- ok:    CDX listed pages and >=1 fetched — targets are real endorsements.
- empty: CDX answered authoritatively (200-with-no-rows / 404) — a CONFIRMED
         zero; safe to treat as "this seed endorses nothing in this crawl".
- error: CDX was unreachable after retries — an OUTAGE, **not** a zero.
         Consumers (e.g. prep_discovery_queue.py) must never fold status=error
         seeds into "confirmed zero" semantics; the top-level `errors` list and
         `status_counts` flag a partial run, and this process exits nonzero
         (1 = some seeds errored, 2 = preflight found CDX down) so batch
         drivers can't mistake an outage for "no endorsements".
A preflight probe hits the CDX endpoint once before the seed loop and aborts
loudly if it is down (index.commoncrawl.org can be dead while
data.commoncrawl.org is fine — HTTP 000 from two networks on 2026-06-28).
`AW_CDX_BASE` env overrides the CDX host (outage simulation / testing).

Columnar resolver (2026-07-04, task 074 verdict → implemented)
----------------------------------------------------------------
`--resolver columnar` enumerates seed pages from the columnar cc-index
(Parquet over DuckDB) on the unblocked data.commoncrawl.org mirror instead of
the CDX HTTP API — the CDX host is no longer a failure mode at all. Same
error-vs-empty contract: the preflight is the parquet-manifest/footers fetch
(abort exit 2 if the mirror is down); a seed whose resolve fails after
retries is status=error (never empty); genuine no-captures stay `empty`.

FAMILY-wide semantics (CDX matchType=domain parity): `resolve_seed_family`
queries the surtkey range [`tld,dom)`, `tld,dom,￿`) — apex+www AND all
subdomains ("ch,ethz," keys sort after "ch,ethz)" because ',' > ')'). This
matters: measured on CC-MAIN-2026-25, ethz.ch/uzh.ch/zh.ch/nzz.ch have ZERO
apex+www captures — their CC presence is entirely subdomains, so
cc_columnar.resolve_domains (host-granular, right for the discovery fetch)
would have falsely emptied them here. One query per seed keeps the
per-seed page LIMIT exact. Ranking delta vs CDX (deliberate, task 074):
shortest-URL-first (landing pages) instead of CDX capture order.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client

# Unified extraction (ADR-020): the per-page link logic + BOILERPLATE filter +
# reg_domain now live in aw_common.process_html, shared with cc_ingest so CC is
# parsed once for content AND links. Re-exported here for back-compat (callers /
# tests may reference extract_outlinks.{BOILERPLATE,reg_domain}).
from cytir.extract.html_extract import BOILERPLATE, process_html, reg_domain
from cytir.core.user_agent import user_agent  # task 432: one home for the crawler identity

CDX_BASE = os.environ.get("AW_CDX_BASE", "https://index.commoncrawl.org")
CDX = CDX_BASE + "/{crawl}-index"
WARC = "https://data.commoncrawl.org/"
UA = {"User-Agent": user_agent()}
DEFAULT_CRAWL = "CC-MAIN-2026-25"

TRUST = {"federal-gov": 1.0, "cantonal": 0.9, "universities-research": 0.9,
         "health": 0.9, "reference-cultural": 0.85, "portals": 0.8}


class CdxError(RuntimeError):
    """CDX index lookup failed transiently (e.g. index.commoncrawl.org outage or
    429/5xx). DISTINCT from a genuine empty result — surfaced so an outage is
    never silently mistaken for 'this seed has no captures'. (2026-06-28: a CDX
    outage made 117/120 seeds look empty; that must never recur silently.)"""


def _retryable(status):
    return status == 0 or status == 429 or 500 <= status <= 599


def cdx_query(client, domain, crawl, limit, retries=5):
    """Return CDX records (possibly []) for a genuine empty/404, or raise
    CdxError after exhausting retries on transient failures."""
    last = None
    for attempt in range(retries):
        try:
            r = client.get(CDX.format(crawl=crawl),
                           params={"url": domain, "matchType": "domain", "output": "json",
                                   "filter": "status:200", "limit": str(limit)})
        except Exception as e:  # noqa: BLE001 — connection/timeout = transient
            last = f"conn:{type(e).__name__}"
        else:
            if r.status_code == 200:
                recs = []
                for line in r.text.splitlines():
                    try:
                        j = json.loads(line)
                        if "text/html" in j.get("mime", "text/html"):
                            recs.append(j)
                    except Exception:  # noqa: BLE001
                        pass
                return recs                       # genuine result (may be empty)
            if r.status_code == 404:
                return []                         # CC: domain has no captures here
            if not _retryable(r.status_code):
                raise CdxError(f"{domain}: CDX HTTP {r.status_code}")
            last = f"HTTP {r.status_code}"
        time.sleep(min(5 * 2 ** attempt, 60))     # 5,10,20,40,60s backoff
    raise CdxError(f"{domain}: CDX unreachable after {retries} tries ({last})")


def cdx_preflight(client, crawl, probe="hug.ch/*"):
    """One cheap probe against the CDX endpoint BEFORE looping all seeds.
    Returns normally if CDX answers (any authoritative status); raises
    CdxError if the host is unreachable / erroring — the 2026-06-28 outage
    mode, where looping 120 seeds would just produce 120 fake 'empty's."""
    for attempt in range(3):
        try:
            r = client.get(CDX.format(crawl=crawl),
                           params={"url": probe, "output": "json", "limit": "1"},
                           timeout=20)
        except Exception as e:  # noqa: BLE001
            last = f"conn:{type(e).__name__}"
        else:
            if not _retryable(r.status_code):
                return r.status_code              # 200/404/... = CDX is up
            last = f"HTTP {r.status_code}"
        time.sleep(5 * (attempt + 1))
    raise CdxError(f"preflight: CDX {CDX_BASE} unreachable ({last})")


def resolve_seed_family(ccx, con, crawl, dom, limit):
    """Columnar resolve of a seed's whole domain FAMILY (apex+www+subdomains,
    CDX matchType=domain parity) via the surtkey range [`tld,dom)`, `tld,dom,￿`).
    One query per seed → the page LIMIT is exactly per-seed (resolve_domains'
    per-file LIMIT would let co-located families starve each other). Returns
    fetch_html-ready records (shortest URLs first); raises on query failure —
    caller must surface that as status=error, never 'empty'."""
    base = ccx.domain_surtkey_prefix(dom)
    if not base:
        return []
    base = base[:-1]                          # "ch,ethz)" -> "ch,ethz"
    lo, hi = base + ")", base + ",￿"
    files = [fn for fn, mn, mx in ccx.file_bounds(crawl, con)
             if not (mx < lo or mn > hi)]
    if not files:
        return []
    flist = "['" + "','".join(files) + "']"
    q = f"""
    WITH hits AS (
      SELECT url_surtkey, url, warc_filename, warc_record_offset,
             warc_record_length,
             row_number() OVER (PARTITION BY url_surtkey
                                ORDER BY warc_record_length DESC) AS rn
      FROM read_parquet({flist})
      WHERE url_surtkey >= '{lo}' AND url_surtkey < '{hi}'
        AND fetch_status = 200 AND content_mime_type LIKE 'text/html%'
    )
    SELECT url, warc_filename, warc_record_offset, warc_record_length
    FROM hits WHERE rn = 1 ORDER BY length(url), url LIMIT {int(limit)}"""
    return [{"url": u, "filename": w, "offset": int(o), "length": int(ln)}
            for u, w, o, ln in con.execute(q).fetchall()]


def fetch_html(client, rec, retries=3):
    """Range-GET one WARC record body. Retries transient failures (conn error /
    429/5xx) with backoff — a mirror throttle window must not silently zero a
    whole seed's pages (2026-07-04: bag.admin.ch resolved 200 records but the
    unretried fetches all failed → seed looked falsely 'empty')."""
    off = int(rec["offset"]); ln = int(rec["length"])
    for attempt in range(retries):
        try:
            r = client.get(WARC + rec["filename"],
                           headers={**UA, "Range": f"bytes={off}-{off + ln - 1}"})
            if r.status_code in (200, 206):
                raw = gzip.GzipFile(fileobj=io.BytesIO(r.content)).read()
                # WARC record: headers \r\n\r\n http-headers \r\n\r\n body
                parts = raw.split(b"\r\n\r\n", 2)
                return parts[2].decode("utf-8", "ignore") if len(parts) == 3 else None
            if not _retryable(r.status_code):
                return None                    # authoritative miss (403/416/…)
        except Exception:  # noqa: BLE001 — conn/timeout/bad-gzip = transient
            pass
        time.sleep(3 * 2 ** attempt)           # 3,6,12s
    return None


def process_seed(client, seed, crawl, max_pages, tmpl_frac, fetch_sleep=0.4,
                 pre=None):
    """Returns (domain, kept_targets, n_pages, n_recs, status) where status is
    'ok' | 'empty' (genuinely no captures) | 'error' (index lookup failed, OR
    records resolved but ZERO fetched — a WARC-fetch outage, not an empty).

    `pre` selects the resolver: None → CDX HTTP lookup here; the string
    "error" → the columnar pre-resolve failed for this seed (surface as
    status=error, NEVER empty); a list → pre-resolved WARC records
    ({url, filename, offset, length}) from cc_columnar."""
    dom = seed["domain"]
    seed_reg = reg_domain(dom)
    if isinstance(pre, str) and pre == "error":
        return dom, {}, 0, 0, "error"
    if pre is None:
        try:
            recs = cdx_query(client, dom, crawl, max_pages)
        except CdxError:
            return dom, {}, 0, 0, "error"
    else:
        recs = pre
    if not recs:
        return dom, {}, 0, 0, "empty"
    # target_url -> set of source pages on this seed
    tgt_pages = collections.defaultdict(set)
    n_pages = 0
    for rec in recs:
        html = fetch_html(client, rec)
        time.sleep(fetch_sleep)              # politeness; keep total ≤5 req/s
        if not html:
            continue
        n_pages += 1
        page = rec.get("url", f"https://{dom}/")
        # ONE-pass extractor (ADR-020): links only here — content extraction is
        # cc_ingest's job; this is the endorsement-graph caller of process_html.
        _, links = process_html(page, html,
                                {"content": False, "links": True, "seed_reg": seed_reg})
        for link in links:
            tgt_pages[link].add(page)
    if n_pages == 0:
        # records EXIST in the crawl but none fetched — that is a WARC-fetch
        # outage/throttle, never a confirmed zero (same silent-swallow family
        # as the CDX bug; observed 2026-07-04 on bag.admin.ch, 0/200 fetched).
        return dom, {}, 0, len(recs), "error"
    # template filter: drop links present on > tmpl_frac of sampled pages
    kept = {u: len(p) for u, p in tgt_pages.items()
            if len(p) / n_pages <= tmpl_frac}
    return dom, kept, n_pages, len(recs), "ok"


def main(argv: list[str] | None = None, prog: str | None = None):
    ap = argparse.ArgumentParser(prog=prog or "extract_outlinks")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crawl", default=DEFAULT_CRAWL)
    ap.add_argument("--max-pages", type=int, default=400, help="CC pages sampled per seed")
    ap.add_argument("--template-frac", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=2,
                    help="keep low: CC asks ≤5 req/s TOTAL across workers (ADR-014)")
    ap.add_argument("--fetch-sleep", type=float, default=0.4,
                    help="sleep after each WARC range-GET (politeness pacing; "
                         "0.5 with --workers 1 keeps the run ≤2 req/s)")
    ap.add_argument("--resolver", choices=("cdx", "columnar"), default="cdx",
                    help="columnar = family-wide surtkey-range queries over "
                         "the Parquet cc-index on data.commoncrawl.org (no "
                         "CDX host at all; task 074 verdict). See module "
                         "docstring.")
    args = ap.parse_args(argv)
    seeds = json.load(open(args.seeds))
    seeds = [s for s in seeds if s.get("resolves", True)]
    # aw_common._parse_soup degrades to None (=> 0 links/page, looks 'ok') if
    # bs4 is missing — same silent-swallow family as the CDX bug. Fail loudly.
    import bs4  # noqa: F401

    client = make_client(HttpSpec(timeout=40, headers={"user-agent": UA["User-Agent"]},
                                  follow_redirects=True))

    resolved: dict[str, list | None] = {}
    if args.resolver == "columnar":
        from cytir.acquire import cc_columnar as ccx
        try:                                # preflight = manifest + footers
            con = ccx._con(threads=1)
            bounds = ccx.file_bounds(args.crawl, con)
        except Exception as e:  # noqa: BLE001
            print(f"ABORT: columnar preflight failed — {e!r}\n"
                  f"data.commoncrawl.org parquet index unreachable. No output "
                  f"written; nothing is 'empty'. Re-run when it recovers.",
                  file=sys.stderr, flush=True)
            sys.exit(2)
        print(f"columnar preflight ok ({len(bounds)} parquet index files for "
              f"{args.crawl})", flush=True)
        doms = [s["domain"] for s in seeds]
        for i, d in enumerate(doms, 1):
            recs = None
            for attempt in range(3):
                try:
                    recs = resolve_seed_family(ccx, con, args.crawl, d,
                                               args.max_pages)
                    break
                except Exception as e:  # noqa: BLE001 — transient parquet/HTTP
                    print(f"  [columnar] {d} attempt {attempt + 1} failed "
                          f"({e!r})", file=sys.stderr, flush=True)
                    time.sleep(20 * (attempt + 1))
            resolved[d] = recs              # None = resolve error, NOT empty
            if i % 10 == 0 or i == len(doms):
                print(f"  [columnar] resolved {i}/{len(doms)} seeds", flush=True)
    else:
        try:                                # abort loudly BEFORE looping seeds
            st = cdx_preflight(client, args.crawl)
        except CdxError as e:
            print(f"ABORT: CDX preflight failed — {e}\n"
                  f"index.commoncrawl.org is down (it fails independently of "
                  f"data.commoncrawl.org). No output written; nothing is 'empty'. "
                  f"Re-run when it recovers.", file=sys.stderr, flush=True)
            sys.exit(2)
        print(f"CDX preflight ok (HTTP {st} from {CDX_BASE})", flush=True)
    print(f"{len(seeds)} seeds; sampling ≤{args.max_pages} CC pages each "
          f"(resolver={args.resolver})", flush=True)

    edges = {}           # seed -> {target_url: page_count}
    dom_auth = collections.defaultdict(float)
    url_auth = collections.defaultdict(float)

    def work(s):
        if args.resolver == "columnar":
            pre = resolved.get(s["domain"])
            pre = "error" if pre is None else pre
        else:
            pre = None
        return s, process_seed(client, s, args.crawl, args.max_pages,
                               args.template_frac, fetch_sleep=args.fetch_sleep,
                               pre=pre)

    status_ct = collections.Counter()
    errors = []
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for s, (dom, kept, n_pages, n_recs, status) in ex.map(work, seeds):
            done += 1
            status_ct[status] += 1
            if status == "error":
                errors.append(dom)
            w = s.get("trust", TRUST.get(s.get("class"), 0.8))
            edges[dom] = {"n_pages": n_pages, "n_recs": n_recs,
                          "class": s.get("class"),
                          "status": status, "targets": kept}
            for u, cnt in kept.items():
                strength = min(cnt, 5)        # cap a single seed's vote
                url_auth[u] += w * strength
                dom_auth[reg_domain(urlparse(u).netloc)] += w * strength
            print(f"  [{done}/{len(seeds)}] {dom}: {status} {n_pages}/{n_recs} "
                  f"pages fetched, {len(kept)} endorsed targets", flush=True)

    top_dom = dict(sorted(dom_auth.items(), key=lambda x: -x[1])[:5000])
    top_url = dict(sorted(url_auth.items(), key=lambda x: -x[1])[:20000])
    json.dump({"crawl": args.crawl, "n_seeds": len(seeds),
               "status_counts": dict(status_ct), "errors": errors,
               "domain_authority": top_dom, "url_authority": top_url,
               "edges": edges}, open(args.out, "w"))
    print("\nTop endorsed domains:")
    for d, a in list(top_dom.items())[:25]:
        print(f"  {a:8.1f}  {d}")
    print(f"\nseed status: ok={status_ct['ok']} empty={status_ct['empty']} "
          f"error={status_ct['error']} (of {len(seeds)})")
    print(f"wrote {args.out}")
    if errors:
        print(f"⚠ {len(errors)} seeds ERRORED (index lookup failed — CDX host "
              f"or columnar parquet read) — NOT 'empty'. Output is INCOMPLETE; "
              f"re-run when the index recovers (only these need retry). "
              f"e.g. {errors[:5]}", file=sys.stderr)
        sys.exit(1)                      # batch drivers must see partial != done


if __name__ == "__main__":
    main()
