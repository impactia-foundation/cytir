"""GitHub phase-1 admission roster (task 986, RULING 44): `cyt ops github-admission roster`.

Rule (specs/github-admission-and-recrawl.md section 2, option (b) phase 1 = the stars >= 5,000 side):
a repo is a CANDIDATE when stars >= 5000, not archived, pushed within 12 months. It is then decided:

  admit      SPDX licence is permissive (U2/ADR-049) AND the docs tree holds >= MIN_DOCS_FILES doc files
  link-only  licence none / NOASSERTION / unknown (decision 4) or copyleft (tagged, not dropped): the pointer
             README only, no snippet, until a licence file is read
  refuse     one reason: archived | stale | fork | no-docs-tree | docs-tree-too-small | tree-truncated | tree-unavailable

Every candidate reaches exactly one terminal; the summary prints in = admit + link-only + refuse.

Enumeration is the GitHub search API through `gh api` (never Brave/Google/Bing). The API caps a query at 1,000
results, so bands are split by stars, then by creation-date window, until each is < 1,000. Calls are spaced
(search >= 2.2 s: the 30/min search limit; core >= 0.5 s). A secondary-rate-limit response STOPS the run
(`RateLimitStop`); the per-repo enrichment file is resumable. Nothing here touches prod or the twin.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from cytir.acquire.github_docs_tree import PERMISSIVE, select_docs_tree

SCHEMA = "aw.github-admission-roster.v1"
MIN_STARS = 5000
ACTIVE_MONTHS = 12
MIN_DOCS_FILES = 20            # task 232: >= ~20 doc pages is the eligibility floor
SEARCH_DELAY = 2.2             # 30 search calls/min
CORE_DELAY = 0.5
MEDIAN_FULL_TREE_ROWS = 1600   # task 232 median passages per full-tree repo
EMBED_ROWS_PER_S = 57.8        # task 910 mean
RESULT_CAP = 1000
GQL_BATCH = 50


class RateLimitStop(RuntimeError):
    """A secondary rate limit (or exhausted primary limit we refuse to wait out): stop and report."""


class Gh:
    """`gh api` with spacing. `run(args) -> (status, headers, body)` is injectable for tests."""

    def __init__(self, run=None, sleep=time.sleep, clock=time.monotonic):
        self._run, self._sleep, self._clock = run or self._gh, sleep, clock
        self._last = {"search": 0.0, "core": 0.0}
        self.calls = Counter()

    @staticmethod
    def _gh(args: list[str]):
        p = subprocess.run(["gh", "api", "-i", *args], capture_output=True, text=True, timeout=120)
        head, _, body = p.stdout.partition("\r\n\r\n") if "\r\n\r\n" in p.stdout else p.stdout.partition("\n\n")
        lines = head.splitlines()
        status = int(lines[0].split()[1]) if lines and lines[0].startswith("HTTP") else 0
        hdr = {k.strip().lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:])}
        return status, hdr, body

    def call(self, args: list[str], kind: str = "core"):
        delay = SEARCH_DELAY if kind == "search" else CORE_DELAY
        wait = self._last[kind] + delay - self._clock()
        if wait > 0:
            self._sleep(wait)
        status, hdr, body = self._run(args)
        self._last[kind] = self._clock()
        self.calls[kind] += 1
        low = (body or "")[:600].lower()
        if status in (403, 429) and ("secondary rate limit" in low or "abuse" in low or "retry-after" in hdr):
            raise RateLimitStop(f"secondary rate limit on {args[:2]}: {body[:200]}")
        if status in (403, 429) and hdr.get("x-ratelimit-remaining") == "0":
            reset = int(hdr.get("x-ratelimit-reset", "0"))
            pause = max(0, reset - int(time.time())) + 2
            if kind == "core" and pause <= 3900:     # primary limit: wait for the hourly reset, then retry once
                self._sleep(pause)
                status, hdr, body = self._run(args)
            else:
                raise RateLimitStop(f"primary limit exhausted ({kind}); reset in {pause}s")
        return status, hdr, body

    def json(self, args, kind="core"):
        status, hdr, body = self.call(args, kind)
        try:
            return status, json.loads(body) if body.strip() else None
        except json.JSONDecodeError:
            return status, None


# ---- enumeration --------------------------------------------------------------------------------------------

def _q(lo, hi, c_from, c_to, pushed_since):
    stars = f"{lo}..{hi}" if hi is not None else f">={lo}"
    return f"stars:{stars} archived:false pushed:>={pushed_since} created:{c_from}..{c_to}"


def _page(gh: Gh, q: str, page: int):
    st, d = gh.json(["-X", "GET", "search/repositories", "-f", f"q={q}", "-f", "sort=stars", "-f", "order=desc",
                     "-F", "per_page=100", "-F", f"page={page}"], "search")
    if st != 200 or not isinstance(d, dict):
        raise RuntimeError(f"search status {st} for q={q!r} page={page}")
    return d


INITIAL_BANDS = [(5000, 5499), (5500, 5999), (6000, 6999), (7000, 8999), (9000, 11999), (12000, 19999),
                 (20000, None)]


def enumerate_candidates(gh: Gh, *, now: dt.date, min_stars: int = MIN_STARS, months: int = ACTIVE_MONTHS,
                         bands=None, log=lambda m: None):
    """Search bands split until each is < 1,000. Returns (items by lowercase full_name, band log)."""
    since = (now - dt.timedelta(days=round(months * 30.44))).isoformat()
    items: dict[str, dict] = {}
    bands_log: list[dict] = []
    first_day = dt.date(2007, 1, 1)

    def walk(lo, hi, c_from, c_to):
        q = _q(lo, hi, c_from.isoformat(), c_to.isoformat(), since)
        d = _page(gh, q, 1)
        total = int(d.get("total_count", 0))
        if d.get("incomplete_results"):
            log(f"incomplete_results for {q}")
        if total > RESULT_CAP:
            if hi is None or (hi is not None and hi > lo):      # split by stars first
                top = hi if hi is not None else lo * 2
                mid = (lo + top) // 2
                if hi is None:
                    walk(lo, top - 1, c_from, c_to)
                    walk(top, None, c_from, c_to)
                else:
                    walk(lo, mid, c_from, c_to)
                    walk(mid + 1, hi, c_from, c_to)
                return
            span = (c_to - c_from).days
            if span >= 1:                                           # one star value: split by creation date
                mid = c_from + dt.timedelta(days=span // 2)
                walk(lo, hi, c_from, mid)
                walk(lo, hi, mid + dt.timedelta(days=1), c_to)
                return
            bands_log.append({"q": q, "total": total, "warning": "unsplittable; first 1000 only"})
        pages = (min(total, RESULT_CAP) + 99) // 100
        got = list(d.get("items") or [])
        for p in range(2, pages + 1):
            got.extend(_page(gh, q, p).get("items") or [])
        for it in got:
            items.setdefault(str(it["full_name"]).lower(), it)
        bands_log.append({"q": q, "total": total, "fetched": len(got)})
        log(f"band {lo}..{hi} {c_from}..{c_to}: total={total} fetched={len(got)}")

    for lo, hi in (bands or INITIAL_BANDS):
        if hi is not None and hi < min_stars:
            continue
        walk(max(lo, min_stars), hi, first_day, now)
    return items, bands_log


# ---- enrichment ---------------------------------------------------------------------------------------------

def licence_class(spdx: str | None) -> str:
    s = (spdx or "").upper()
    if s in ("", "NONE", "NOASSERTION", "OTHER"):
        return "unknown"
    if s in PERMISSIVE:
        return "permissive"
    return "copyleft"       # GPL/AGPL/LGPL/EPL/CC-BY-NC...: tagged, not dropped; pointer-only like the acquirer


def docs_tree_stats(nodes: list[dict]) -> dict:
    """Doc files in the docs tree (dirs docs/doc/documentation) and the full-tree selection size, uncapped."""
    sel = select_docs_tree(nodes, True, 10**9)
    in_dirs = [n for n in sel if "/" in str(n["path"])]
    return {"docs_files": len(in_dirs), "tree_files": len(sel),
            "tree_bytes": sum(int(n.get("size") or 0) for n in sel)}


def enrich_tree(gh: Gh, full_name: str, branch: str) -> dict:
    st, d = gh.json([f"repos/{full_name}/git/trees/{quote(branch, safe='')}?recursive=1"])
    if st != 200 or not isinstance(d, dict):
        return {"tree_status": st}
    out = {"tree_status": 200, "tree_truncated": d.get("truncated") is True}
    out.update(docs_tree_stats([n for n in d.get("tree", []) if isinstance(n, dict)]))
    return out


def head_shas(gh: Gh, names: list[str]) -> dict[str, str | None]:
    """Default-branch commit SHAs, GQL_BATCH repos per call."""
    out: dict[str, str | None] = {}
    for i in range(0, len(names), GQL_BATCH):
        chunk = names[i:i + GQL_BATCH]
        frag = " ".join(
            f'r{j}: repository(owner:{json.dumps(n.split("/")[0])}, name:{json.dumps(n.split("/")[1])})'
            ' { defaultBranchRef { target { oid } } }' for j, n in enumerate(chunk))
        st, d = gh.json(["graphql", "-f", f"query=query {{ {frag} }}"])
        data = (d or {}).get("data") or {}
        for j, n in enumerate(chunk):
            ref = ((data.get(f"r{j}") or {}).get("defaultBranchRef") or {}).get("target") or {}
            out[n] = ref.get("oid")
    return out


def decide(row: dict, *, now: dt.date, months: int = ACTIVE_MONTHS, min_docs: int = MIN_DOCS_FILES) -> tuple[str, str]:
    """(decision, reason). Order: facts that need no tree first, so a refused repo costs no tree call."""
    if row.get("archived"):
        return "refuse", "archived"
    pushed = (row.get("pushed_at") or "")[:10]
    if not pushed or pushed < (now - dt.timedelta(days=round(months * 30.44))).isoformat():
        return "refuse", "stale"
    if row.get("fork"):
        return "refuse", "fork"
    lc = licence_class(row.get("license"))
    if lc == "unknown":
        return "link-only", "licence-unknown"
    if lc == "copyleft":
        return "link-only", "copyleft-tagged"
    if row.get("tree_status") != 200:
        return "refuse", "tree-unavailable"
    if row.get("tree_truncated"):
        return "refuse", "tree-truncated"
    n = int(row.get("docs_files") or 0)
    if n == 0:
        return "refuse", "no-docs-tree"
    if n < min_docs:
        return "refuse", "docs-tree-too-small"
    return "admit", "permissive+docs-tree"


def needs_tree(row: dict, *, now: dt.date) -> bool:
    d, r = decide({**row, "tree_status": 200, "docs_files": MIN_DOCS_FILES}, now=now)
    return d == "admit"


def base_row(it: dict) -> dict:
    lic = (it.get("license") or {}).get("spdx_id") if isinstance(it.get("license"), dict) else None
    return {"repo": it["full_name"], "stars": it.get("stargazers_count"), "pushed_at": it.get("pushed_at"),
            "archived": bool(it.get("archived")), "fork": bool(it.get("fork")), "license": lic,
            "default_branch": it.get("default_branch")}


def estimate_rows(rows: list[dict]) -> dict:
    """Estimated rows per decision: admit at the task-232 median, link-only 1 pointer row."""
    adm = [r for r in rows if r["decision"] == "admit"]
    new = [r for r in adm if not r.get("in_index")]
    return {"admit_repos": len(adm), "admit_repos_net_new": len(new),
            "rows_median_1600": len(adm) * MEDIAN_FULL_TREE_ROWS,
            "rows_counted_files": sum(int(r.get("tree_files") or 0) for r in adm),
            "link_only_rows": sum(1 for r in rows if r["decision"] == "link-only"),
            "gpu_hours_median_1600": round(len(adm) * MEDIAN_FULL_TREE_ROWS / EMBED_ROWS_PER_S / 3600, 1),
            "gpu_hours_counted_files": round(sum(int(r.get("tree_files") or 0) for r in adm)
                                             / EMBED_ROWS_PER_S / 3600, 1)}


def summarize(rows: list[dict]) -> dict:
    dec = Counter(r["decision"] for r in rows)
    reasons = Counter(f'{r["decision"]}:{r["reason"]}' for r in rows)
    hist_edges = [0, 1, 5, 20, 50, 100, 200, 400, 10**9]
    hist = Counter()
    for r in rows:
        if r["decision"] in ("admit", "refuse") and r.get("docs_files") is not None:
            n = int(r["docs_files"])
            for a, b in zip(hist_edges, hist_edges[1:]):
                if a <= n < b:
                    hist[f"{a}-{b - 1}" if b < 10**9 else f"{a}+"] += 1
                    break
    return {"schema": SCHEMA, "candidates": len(rows), "decisions": dict(dec),
            "invariant_in_equals_sum": len(rows) == sum(dec.values()), "reasons": dict(reasons),
            "licence_histogram": dict(Counter((r.get("license") or "none") for r in rows).most_common()),
            "docs_files_histogram": dict(hist), "in_index": sum(1 for r in rows if r.get("in_index")),
            "estimate": estimate_rows(rows)}


# ---- verb ---------------------------------------------------------------------------------------------------

def load_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["repo"].lower()] = r
    return out


def roster(args, gh: Gh | None = None, now: dt.date | None = None) -> int:
    gh = gh or Gh()
    now = now or dt.datetime.now(dt.timezone.utc).date()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    indexed = {ln.strip().lower() for ln in Path(args.index_repos).read_text().splitlines() if ln.strip()}
    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
    cache = out / "enriched.partial.jsonl"
    cands_file = out / "candidates.json"
    if cands_file.exists() and not args.refresh:
        items = json.loads(cands_file.read_text())["items"]
        log(f"reusing {len(items)} candidates from {cands_file.name}")
    else:
        items, bands = enumerate_candidates(gh, now=now, min_stars=args.min_stars, log=log)
        cands_file.write_text(json.dumps({"now": now.isoformat(), "bands": bands, "items": items}))
    names = sorted(items)
    if args.max_repos:
        names = names[:args.max_repos]
    done = load_jsonl(cache)
    todo = [n for n in names if n not in done]
    log(f"{len(names)} candidates, {len(done)} enriched, {len(todo)} to do")
    try:
        with cache.open("a") as fh:
            for n in todo:
                row = base_row(items[n])
                if needs_tree(row, now=now):
                    row.update(enrich_tree(gh, row["repo"], row["default_branch"] or "main"))
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                done[n] = row
    except RateLimitStop as exc:
        log(f"[STOP] {exc}; {len(done)} of {len(names)} enriched; rerun resumes")
        return 3
    rows = [done[n] for n in names]
    shas = head_shas(gh, [r["repo"] for r in rows if needs_tree(r, now=now)])
    for r in rows:
        r["default_branch_sha"] = shas.get(r["repo"])
        r["decision"], r["reason"] = decide(r, now=now)
        r["in_index"] = r["repo"].lower() in indexed
    (out / "phase1-roster.v1.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    summary = summarize(rows)
    summary.update({"as_of": now.isoformat(), "api_calls": dict(gh.calls)})
    (out / "phase1-summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


# ---- licence-file read (task 995, RULING 44 decision 4) -----------------------------------------------------

LICENCE_READ_SCHEMA = "aw.github-admission-licence-read.v1"
LICENCE_AFTER_PERMISSIVE = frozenset({"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "MPL-2.0",
                                      "CC0-1.0", "Unlicense", "0BSD", "Zlib", "PSF-2.0"})


def match_licence_text(text: str) -> str:
    """Match the licence header (first 400 chars). The BSD clause count reads the full text for 'Neither the name'."""
    head = (text or "")[:400].upper()
    full = (text or "").upper()
    if "MIT LICENSE" in head:
        return "MIT"
    if "APACHE LICENSE" in head and "VERSION 2.0" in head:
        return "Apache-2.0"
    if "BSD 3-CLAUSE" in head:
        return "BSD-3-Clause"
    if "BSD 2-CLAUSE" in head:
        return "BSD-2-Clause"
    if "REDISTRIBUTION AND USE IN SOURCE AND BINARY FORMS" in head:
        return "BSD-3-Clause" if "NEITHER THE NAME" in full else "BSD-2-Clause"
    if "ISC LICENSE" in head:
        return "ISC"
    if "MOZILLA PUBLIC LICENSE" in head and "2.0" in head:
        return "MPL-2.0"
    if "GNU GENERAL PUBLIC LICENSE" in head:
        return "GPL"
    if "GNU AFFERO" in head:
        return "AGPL"
    if "GNU LESSER" in head:
        return "LGPL"
    if "CC0" in head:
        return "CC0-1.0"
    if "UNLICENSE" in head:
        return "Unlicense"
    return "unrecognised"


def licence_decision(licence_after: str, docs_files: int | None, licence_read: str = "ok",
                     *, min_docs: int = MIN_DOCS_FILES) -> tuple[str, str]:
    """(decision_after, reason) for a licence-read row. Only a permissive licence with >= min_docs doc files admits."""
    if licence_read == "none":
        return "link-only", "no_licence_file"
    if licence_after not in LICENCE_AFTER_PERMISSIVE:
        return "link-only", "licence_not_permissive"
    if docs_files is None:
        return "link-only", "tree_unavailable"
    if docs_files < min_docs:
        return "link-only", "docs_tree_small"
    return "admit", "permissive+docs-tree"


def read_licence(gh: Gh, row: dict) -> dict:
    """One roster link-only row: GET /repos/{o}/{r}/license, text-match if NOASSERTION, tree only if permissive."""
    o = row["repo"]
    st, d = gh.json([f"repos/{o}/license"])
    out = {"owner_repo": o, "licence_before": row.get("license") or "none",
           "in_index": bool(row.get("in_index")), "default_branch": row.get("default_branch"), "docs_files": None}
    if st == 404:
        out.update(licence_read="none", spdx_id=None, licence_key=None, matched_by="none", licence_after="none")
    elif st == 200 and isinstance(d, dict):
        lic = d.get("license") or {}
        spdx = lic.get("spdx_id")
        out.update(licence_read="ok", spdx_id=spdx, licence_key=lic.get("key"))
        if spdx == "NOASSERTION":
            text = base64.b64decode(d.get("content") or "").decode("utf-8", "replace")
            out.update(licence_after=match_licence_text(text), matched_by="text")
        else:
            out.update(licence_after=spdx or "none", matched_by="api" if spdx else "none")
    else:
        raise RuntimeError(f"licence endpoint status {st}")          # never record an error as an empty licence
    if out["licence_after"] in LICENCE_AFTER_PERMISSIVE and row.get("default_branch"):
        t = enrich_tree(gh, o, row["default_branch"])
        out["tree_status"] = t.get("tree_status")
        out["docs_files"] = t.get("docs_files")
    out["decision_after"], out["reason"] = licence_decision(out["licence_after"], out["docs_files"],
                                                            out["licence_read"])
    return out


def licence_summary(rows: list[dict]) -> dict:
    return {"schema": LICENCE_READ_SCHEMA + ".summary", "rows": len(rows),
            "by_licence_after": dict(Counter(r["licence_after"] for r in rows)),
            "by_matched_by": dict(Counter(r["matched_by"] for r in rows)),
            "by_decision_after": dict(Counter(r["decision_after"] for r in rows)),
            "admit_in_index_false": sum(1 for r in rows if r["decision_after"] == "admit" and not r["in_index"])}


def licence_read_verb(args, gh: Gh | None = None) -> int:
    """Resumable: rows already in --out are skipped; each row is appended and flushed as it is read."""
    roster_rows = load_jsonl(Path(args.roster)).values()
    targets = [r for r in roster_rows if r.get("decision") == "link-only" and licence_class(r.get("license")) == "unknown"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(ln)["owner_repo"] for ln in out.read_text().splitlines() if ln.strip()} if out.exists() else set()
    gh = gh or Gh()
    stopped = None
    with out.open("a") as fh:
        for r in targets:
            if r["repo"] in done:
                continue
            try:
                rec = read_licence(gh, r)
            except (RateLimitStop, RuntimeError) as e:
                stopped = f"{r['repo']}: {e}"
                break
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
    rows = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    if args.summary:
        Path(args.summary).write_text(json.dumps(licence_summary(rows), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"targets": len(targets), "rows_in_out": len(rows), "calls": dict(gh.calls)}))
    if stopped:
        print(f"STOPPED at {stopped}")
        return 2
    return 0



def main(argv=None, prog="cyt ops github-admission") -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("roster", help="enumerate, enrich and decide the phase-1 candidates")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--index-repos", required=True, help="file: one owner/repo per line already in the index")
    r.add_argument("--min-stars", type=int, default=MIN_STARS)
    r.add_argument("--max-repos", type=int, default=0, help="debug: only the first N candidates")
    r.add_argument("--refresh", action="store_true", help="re-run the search enumeration")
    lr = sub.add_parser("licence-read", help="read the licence file of link-only roster rows (task 995)")
    lr.add_argument("--roster", required=True)
    lr.add_argument("--out", required=True, help="jsonl; resumable: rows already present are skipped")
    lr.add_argument("--summary", default=None, help="optional summary json path")
    args = ap.parse_args(argv)
    if args.verb == "licence-read":
        return licence_read_verb(args)
    return roster(args)


if __name__ == "__main__":
    raise SystemExit(main())
