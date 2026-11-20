"""Package-registry census (task 987): top-N packages per registry -> source repo URL.

Turns the "registry side 3-5k repos" estimate of the GitHub admission rule into counts.
Registries' own public APIs or published dumps only; polite (>= 0.3 s between calls to a host,
`Retry-After` honoured); the crawler identity comes from `cytir.core.user_agent`.

CLI:
  python3 -m cytir.ops.registry_census run --registry npm --top 5000 --out rows-npm.jsonl
  python3 -m cytir.ops.registry_census merge --rows rows-*.jsonl --json out.json --csv out.csv
  python3 -m cytir.ops.registry_census stars --json out.json --out stars.json [--sample 600]

Registries without a public download/dependents ranking (Go, Maven) are recorded as
`not_available` with the reason; they are never guessed.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.parse
import httpx
from datetime import datetime, timezone
from typing import Callable, Iterable

from cytir.consent.block import ConsentBlocked, ConsentRegistryError
from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client

MIN_GAP_S = 0.3
REGISTRIES = ("npm", "pypi", "crates", "rubygems", "go", "maven")
NOT_AVAILABLE = {
    "go": "no public ranked list: pkg.go.dev has no top list and deps.dev serves per-package "
          "dependents only (the ranked data is BigQuery-only)",
    "maven": "no public download/dependents ranking: search.maven.org has none; deps.dev serves "
             "per-package dependents only (the ranked data is BigQuery-only)",
}
NPM_TOP_URL = "https://raw.githubusercontent.com/wooorm/npm-high-impact/main/lib/top.js"
PYPI_TOP_URL = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json"
CRATES_URL = "https://crates.io/api/v1/crates?sort=downloads&per_page=100"
BESTGEMS_URL = "https://bestgems.org/total?page={page}"

_HOSTS_GIT = ("github.com", "gitlab.com", "bitbucket.org", "codeberg.org", "sr.ht", "git.sr.ht")


# ----------------------------------------------------------------------------- normaliser
def normalise_repo_url(raw: str | None) -> dict | None:
    """Return {host, repo, url} or None.  GitHub -> url `github.com/<owner>/<repo>` (lowercase).

    Other hosts keep their host; their url is `<host>/<path>` with the same cleaning.  A bare
    `owner/repo` (npm shorthand) means GitHub.  `*.github.io` pages are not repositories."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    s = re.sub(r"^git\+", "", s)
    m = re.match(r"^([a-z]+):(?!//)(.+)$", s)  # npm shorthand github:o/r, gitlab:o/r
    if m and m.group(1) in ("github", "gitlab", "bitbucket"):
        s = f"https://{m.group(1)}.{'com' if m.group(1) != 'bitbucket' else 'org'}/{m.group(2)}"
    m = re.match(r"^(?:[\w.-]+@)([\w.-]+):(?!//)(.+)$", s)  # scp-like git@host:o/r
    if m:
        s = f"https://{m.group(1)}/{m.group(2)}"
    if re.match(r"^[\w.-]+/[\w.-]+$", s) and "." not in s.split("/")[0]:
        s = f"https://github.com/{s}"
    s = re.sub(r"^(git|ssh|git\+ssh|git\+https|git\+http|http|https)://", "https://", s)
    s = re.sub(r"^https://[^/@]+@", "https://", s)  # userinfo
    if not s.startswith("https://"):
        s = "https://" + s
    p = urllib.parse.urlsplit(s)
    host = (p.hostname or "").lower()
    host = host[4:] if host.startswith("www.") else host
    if not host or "." not in host:
        return None
    segs = [x for x in p.path.split("/") if x]
    if host.endswith(".github.io") or host == "github.io":
        return {"host": host, "repo": None, "url": None}
    if host == "github.com" or host in _HOSTS_GIT and host != "gitlab.com":
        segs = segs[:2]
    else:  # gitlab subgroups: cut at "/-/" or tree/blob markers
        for i, x in enumerate(segs):
            if x == "-" or (i >= 2 and x in ("tree", "blob", "issues", "wiki")):
                segs = segs[:i]
                break
    if len(segs) < 2:
        return {"host": host, "repo": None, "url": None}
    segs[-1] = re.sub(r"\.git$", "", segs[-1])
    path = "/".join(segs).lower()
    if host == "github.com" and path.split("/")[0] in ("sponsors", "orgs", "topics", "features"):
        return {"host": host, "repo": None, "url": None}
    return {"host": host, "repo": path, "url": f"{host}/{path}"}


# ----------------------------------------------------------------------------- HTTP
_last: dict[str, float] = {}


def _ua() -> str:
    from cytir.core import user_agent
    return user_agent.user_agent("registry-census")


def http_get(url: str, *, retries: int = 4) -> str | None:
    """GET with per-host spacing and 429/5xx back-off.  None on 404/410 or exhausted retries."""
    host = urllib.parse.urlsplit(url).hostname or ""
    for attempt in range(retries):
        wait = MIN_GAP_S - (time.monotonic() - _last.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        _last[host] = time.monotonic()
        try:
            with make_client(HttpSpec(timeout=40.0, headers={
                    "user-agent": _ua(), "accept": "application/json, text/*"})) as client:
                r = client.get(url)
            if r.status_code < 400:
                return r.content.decode("utf-8", "replace")
            if r.status_code in (404, 410):
                return None
            if r.status_code == 429 or r.status_code >= 500:
                ra = r.headers.get("Retry-After")
                time.sleep(min(float(ra), 120) if ra and ra.isdigit() else 5 * (attempt + 1))
                continue
            return None
        except (ConsentBlocked, ConsentRegistryError):
            raise                      # fail closed: a declined host is never retried
        except (httpx.TransportError, OSError):
            time.sleep(3 * (attempt + 1))
    return None


# ----------------------------------------------------------------------------- parsers (pure)
def parse_npm_top(js_text: str) -> list[str]:
    return re.findall(r"^\s+'([^']+)',?\s*$", js_text, flags=re.M)


def parse_npm_repo(json_text: str) -> str | None:
    d = json.loads(json_text)
    r = d.get("repository")
    if isinstance(r, dict):
        r = r.get("url")
    if not r:
        r = d.get("homepage")
    return r if isinstance(r, str) else None


def parse_pypi_top(json_text: str) -> list[tuple[str, int]]:
    d = json.loads(json_text)
    rows = d.get("rows") or d.get("data") or []
    return [(r["project"], int(r["download_count"])) for r in rows]


def parse_pypi_repo(json_text: str) -> tuple[str | None, str]:
    """Return (raw repo url, source field).  Prefer a GitHub URL among project_urls, by key
    priority; then home_page; then any other recognised repo host."""
    info = json.loads(json_text).get("info") or {}
    urls = info.get("project_urls") or {}
    prio = ("source", "repository", "code", "github", "git", "sourcecode", "source code")
    ordered = sorted(urls.items(), key=lambda kv: (
        0 if kv[0].lower() in prio else 1 if any(k in kv[0].lower() for k in prio) else 2))
    cands = [(v, f"project_urls.{k}") for k, v in ordered] + [(info.get("home_page"), "home_page")]
    cands = [(v, f) for v, f in cands if v]
    for v, f in cands:
        n = normalise_repo_url(v)
        if n and n["host"] == "github.com" and n["repo"]:
            return v, f
    for v, f in cands:
        n = normalise_repo_url(v)
        if n and n["repo"] and n["host"] in _HOSTS_GIT:
            return v, f
    return None, ""


def parse_crates_page(json_text: str) -> tuple[list[dict], str | None]:
    d = json.loads(json_text)
    rows = [{"package": c["id"], "metric_value": int(c.get("downloads") or 0),
             "repo": c.get("repository") or c.get("homepage")} for c in d.get("crates", [])]
    nxt = (d.get("meta") or {}).get("next_page")
    return rows, nxt


def parse_bestgems_page(html: str) -> list[tuple[str, int, int]]:
    out = []
    for m in re.finditer(r'<tr><td class="right">([\d,]+)</td><td class="right">([\d,]+)</td><td><a href="/gems/([^"]+)">', html):
        out.append((m.group(3), int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))))
    return out


def parse_gem_repo(json_text: str) -> tuple[str | None, str]:
    d = json.loads(json_text)
    for f in ("source_code_uri", "homepage_uri", "bug_tracker_uri", "documentation_uri"):
        n = normalise_repo_url(d.get(f))
        if n and n["repo"] and n["host"] in _HOSTS_GIT:
            return d[f], f
    return None, ""


# ----------------------------------------------------------------------------- fetchers
def _row(registry, package, rank, metric, value, raw, source):
    n = normalise_repo_url(raw)
    if n and n["host"] not in _HOSTS_GIT:  # a vanity/homepage host is not a repository
        n = {"host": n["host"], "repo": None, "url": None}
    return {"registry": registry, "package": package, "rank": rank, "metric": metric,
            "metric_value": value, "repo_url": n["url"] if n else None,
            "repo_host": n["host"] if n else None, "repo_url_source": source if n and n["url"] else ""}


def fetch_npm(top: int, get: Callable = http_get) -> Iterable[dict]:
    names = parse_npm_top(get(NPM_TOP_URL) or "")[:top]
    for i, name in enumerate(names, 1):
        t = get(f"https://registry.npmjs.org/{urllib.parse.quote(name, safe='@')}/latest")
        raw = parse_npm_repo(t) if t else None
        yield _row("npm", name, i, "high_impact_rank", i, raw, "repository.url|homepage")


def fetch_pypi(top: int, get: Callable = http_get) -> Iterable[dict]:
    rows = parse_pypi_top(get(PYPI_TOP_URL) or '{"rows":[]}')[:top]
    for i, (name, dl) in enumerate(rows, 1):
        t = get(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json")
        raw, src = parse_pypi_repo(t) if t else (None, "")
        yield _row("pypi", name, i, "downloads_30d", dl, raw, src)


def fetch_crates(top: int, get: Callable = http_get) -> Iterable[dict]:
    url, i = CRATES_URL, 0
    while url and i < top:
        rows, nxt = parse_crates_page(get(url) or '{"crates":[]}')
        if not rows:
            return
        for r in rows:
            i += 1
            if i > top:
                return
            yield _row("crates", r["package"], i, "downloads_total", r["metric_value"], r["repo"], "repository|homepage")
        url = ("https://crates.io/api/v1/crates" + nxt) if nxt else None


def fetch_rubygems(top: int, get: Callable = http_get, skip: frozenset = frozenset()) -> Iterable[dict]:
    ranked, page = [], 1
    while len(ranked) < top:
        got = parse_bestgems_page(get(BESTGEMS_URL.format(page=page)) or "")
        if not got:
            break
        ranked += got
        page += 1
    for name, rank, dl in ranked[:top]:
        if name in skip:
            continue
        t = get(f"https://rubygems.org/api/v1/gems/{urllib.parse.quote(name)}.json")
        raw, src = parse_gem_repo(t) if t else (None, "")
        yield _row("rubygems", name, rank, "downloads_total", dl, raw, src)


FETCHERS = {"npm": fetch_npm, "pypi": fetch_pypi, "crates": fetch_crates, "rubygems": fetch_rubygems}
BASIS = {
    "npm": "wooorm/npm-high-impact top list (ordered by npm downloads), repository via registry /latest",
    "pypi": "hugovk top-pypi-packages 30-day downloads (ClickHouse), repo via pypi.org JSON",
    "crates": "crates.io API sort=downloads (all-time), repository field",
    "rubygems": "bestgems.org total-downloads ranking, repo via rubygems.org gem JSON",
}


# ----------------------------------------------------------------------------- merge / stars
def merge(row_files: list[str], json_out: str, csv_out: str, now: str | None = None) -> dict:
    rows = []
    for pat in row_files:
        for fn in sorted(glob.glob(pat)):
            with open(fn) as f:
                rows += [json.loads(x) for x in f if x.strip()]
    now = now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    per: dict[str, dict] = {}
    repos: dict[str, dict] = {}
    for r in rows:
        p = per.setdefault(r["registry"], {"n": 0, "resolved": 0, "github": 0})
        p["n"] += 1
        if r["repo_url"]:
            p["resolved"] += 1
            e = repos.setdefault(r["repo_url"], {"repo_url": r["repo_url"], "host": r["repo_host"], "registries": {}})
            e["registries"].setdefault(r["registry"], []).append(r["package"])
            if r["repo_host"] == "github.com":
                p["github"] += 1
    for reg in NOT_AVAILABLE:
        per[reg] = {"n": 0, "resolved": 0, "github": 0, "not_available": NOT_AVAILABLE[reg]}
    for reg, p in per.items():
        p["basis"] = BASIS.get(reg, "")
        p["resolution_rate"] = round(p["resolved"] / p["n"], 4) if p["n"] else None
    doc = {"provenance": {"fetched_at": now, "registries": per,
                          "note": "repo list is distinct normalised urls; github.com repos are lowercase owner/repo"},
           "repos": sorted(repos.values(), key=lambda e: e["repo_url"])}
    with open(json_out, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["registry", "package", "rank", "metric", "metric_value",
                                          "repo_url", "repo_host", "repo_url_source"])
        w.writeheader()
        w.writerows(rows)
    return doc


def gh_stars(slug: str) -> int | None:
    time.sleep(0.5)
    r = subprocess.run(["gh", "api", f"repos/{slug}", "--jq", ".stargazers_count"],
                       capture_output=True, text=True, timeout=60)
    return int(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip().isdigit() else None


def stars(json_path: str, out: str, sample: int, seed: int = 987, cap: int = 3000) -> dict:
    doc = json.load(open(json_path))
    gh = [e["repo_url"].split("/", 1)[1] for e in doc["repos"] if e["host"] == "github.com"]
    sampled = len(gh) > cap
    pick = sorted(random.Random(seed).sample(gh, sample)) if sampled else gh
    res = {s: gh_stars(s) for s in pick}
    ok = [v for v in res.values() if v is not None]
    n = len(ok)
    out_doc = {"distinct_github": len(gh), "sampled": sampled, "queried": len(pick), "answered": n,
               "stars": res}
    for t in (2000, 5000):
        k = sum(1 for v in ok if v >= t)
        p = k / n if n else 0.0
        out_doc[f"share_ge_{t}"] = round(p, 4)
        out_doc[f"share_ge_{t}_se"] = round((p * (1 - p) / n) ** 0.5, 4) if n else None
        out_doc[f"est_count_ge_{t}"] = round(p * len(gh))
    json.dump(out_doc, open(out, "w"), indent=1, sort_keys=True)
    return out_doc


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog or "registry_census")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--registry", required=True, choices=REGISTRIES)
    r.add_argument("--top", type=int, default=5000)
    r.add_argument("--out", required=True)
    r.add_argument("--resume", action="store_true", help="rubygems only: skip packages already in --out")
    m = sub.add_parser("merge")
    m.add_argument("--rows", nargs="+", required=True)
    m.add_argument("--json", required=True)
    m.add_argument("--csv", required=True)
    s = sub.add_parser("stars")
    s.add_argument("--json", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--sample", type=int, default=600)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        if a.registry in NOT_AVAILABLE:
            print(f"{a.registry}: not_available -- {NOT_AVAILABLE[a.registry]}")
            open(a.out, "w").close()
            return 0
        n = ok = 0
        done = frozenset()
        if a.resume and a.registry == "rubygems" and os.path.exists(a.out):
            done = frozenset(json.loads(x)["package"] for x in open(a.out) if x.strip())
        with open(a.out, "a" if done else "w") as f:
            fetch = FETCHERS[a.registry]
            for row in (fetch(a.top, skip=done) if done else fetch(a.top)):
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
                n += 1
                ok += bool(row["repo_url"])
        print(f"{a.registry}: {n} packages, {ok} resolved")
    elif a.cmd == "merge":
        d = merge(a.rows, a.json, a.csv)
        print(json.dumps({k: [v["n"], v["resolved"]] for k, v in d["provenance"]["registries"].items()}), len(d["repos"]))
    else:
        d = stars(a.json, a.out, a.sample)
        print({k: v for k, v in d.items() if k != "stars"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
