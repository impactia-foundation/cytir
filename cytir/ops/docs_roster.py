#!/usr/bin/env python3
"""The docs-layer roster: authoritative technical-documentation projects, derived from
package registries -- never a hand-list (task 694, 2026-10-02).

WHY THIS EXISTS
---------------
PORTFOLIO section 4.4 targets 15M points of canonical technical documentation and the index
serves about 2.2M. The wave plan (task 691 item 10) needs a roster of 400-800 projects before a
single Common Crawl byte is fetched. The roster is the work: it says which hosts are canonical
docs, why we believe it (registry evidence), what licence class the project states, what the
publisher's live consent says, what the twin already serves, what Common Crawl holds and what a
fetch is worth.

STAGES (each is a subcommand; each stage writes its own columns and prints its predicate)
  fetch    registries -> sealed inputs (jsonl + sha256 manifest). Network. Run on the build box.
           ecosyste.ms package registry API (open data, per-registry top-N by downloads or
           dependents), Homebrew analytics, Wikidata SPARQL. No search engine API is used or allowed
           (memory aw-no-competitor-search-apis).
  build    sealed inputs -> ranked roster rows. Pure and offline.
  twin     join the twin's indexed `domain` facet (host, family, docs-only) onto each row.
  cc       count status-200 HTML URLs per host in the LOCAL Common Crawl columnar mirror.
  consent  resolve each host through the one resolver (cytir.consent.resolver). Heimdall only.
  rank     estimate net-new points, mark eligibility with a REASON per row, write the roster.
  queue    write a discovery_ingest queue for the top-N eligible hosts (a tranche).
  summary  print the roster summary.

EVERY ROW REACHES ONE TERMINAL (corpus-doctrine 2). A candidate is `eligible` or carries exactly one
`ineligible_reason`. `summary` prints in = eligible + sum(reasons).

WHAT THIS DOES NOT DECIDE
-------------------------
* Licence. The roster carries the registry's SPDX string for the PACKAGE (code licence) and says so
  in `licence_basis`. The project's DOCS licence statement is not read here. Non-permissive classes
  are `ineligible_reason = licence_class_*` and are listed for a POLICY ruling, never edited into
  POLICY/PORTFOLIO. The tranche admission basis is ADR-111 / the `cc_authority_frontier_w7`
  declaration; this module only refuses classes that basis does not cover.
* Consent. Our own 403 / timeout / outage is `held`, never `declined` (corpus-doctrine 3). CCBot
  disallow prices train_ok only.
* Estimates are estimates. `est_points` = min(cc_urls, cap) x chunks/page, where chunks/page is
  MEASURED from a sealed docs tranche (`--calibration`, pooled rows per URL), printed with the predicate.
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

# Started by path, sys.path[0] is this file's own directory, not the repo root: put the root first
# so `import cytir` resolves.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cytir.core import twin as TWIN

SCHEMA = "aw.docs-roster.v1"
ECOSYSTE_MS = "https://packages.ecosyste.ms/api/v1/registries/{registry}/packages"
WDQS = "https://query.wikidata.org/sparql"
BREW_ANALYTICS = "https://formulae.brew.sh/api/analytics/install-on-request/365d.json"
BREW_FORMULA = "https://formulae.brew.sh/api/formula/{name}.json"

# registry -> (top-N, sort key). Downloads where the registry reports them, dependents otherwise.
REGISTRIES = {
    "pypi.org": (2000, "downloads"),
    "npmjs.org": (2000, "downloads"),
    "crates.io": (1000, "downloads"),
    "rubygems.org": (800, "downloads"),
    "nuget.org": (800, "downloads"),
    "packagist.org": (800, "downloads"),
    "pub.dev": (500, "downloads"),
    "hex.pm": (400, "downloads"),
    "cran.r-project.org": (400, "downloads"),
    "repo1.maven.org": (1000, "dependent_repos_count"),
    "proxy.golang.org": (1000, "dependent_repos_count"),
    "registry.terraform.io": (300, "downloads"),
}

# Hosts that are infrastructure or aggregators, not a project's own documentation. A host listed
# here is kept in the roster with reason `platform_aggregator` so the exclusion is visible.
PLATFORM_HOSTS = frozenset("""
github.com gitlab.com bitbucket.org sourceforge.net codeberg.org pypi.org npmjs.com npmjs.org
crates.io docs.rs pkg.go.dev rubydoc.info javadoc.io hexdocs.pm hex.pm readthedocs.org
readthedocs.io github.io gitlab.io medium.com wordpress.com blogspot.com youtube.com twitter.com
x.com linkedin.com facebook.com discord.com discord.gg gitter.im t.me reddit.com
stackoverflow.com mvnrepository.com search.maven.org central.sonatype.com packagist.org
nuget.org rubygems.org cran.r-project.org pub.dev hub.docker.com anaconda.org libraries.io
opencollective.com patreon.com archive.org web.archive.org wikipedia.org en.wikipedia.org
pythonhosted.org npm.im bit.ly goo.gl registry.terraform.io gist.github.com
""".split())

# Vendor documentation under a proprietary or mixed licence (task 232 section 2A: EXCLUDE
# AWS/GCP/Azure cloud docs). The package's code licence says nothing about these pages.
VENDOR_PROPRIETARY_DOCS = frozenset("""
docs.aws.amazon.com learn.microsoft.com docs.microsoft.com cloud.google.com
developer.apple.com docs.oracle.com developer.android.com firebase.google.com
developers.google.com docs.snowflake.com docs.databricks.com
""".split())

# Hosts that host many projects as path prefixes. The scope of a row on them is the prefix.
SHARED_PATH_HOSTS = frozenset("""
learn.microsoft.com docs.aws.amazon.com cloud.google.com
developers.google.com docs.github.com
""".split())

# A tenant of these hosting clouds is a project's own docs host (the apex is not).
DOCS_CLOUD_SUFFIXES = (".readthedocs.io", ".github.io", ".gitlab.io", ".gitbook.io",
                       ".netlify.app", ".vercel.app", ".pages.dev", ".docs.page")

PERMISSIVE = {
    "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "bsd-3-clause-clear", "bsd-4-clause",
    "isc", "mpl-2.0", "0bsd", "cc0-1.0", "unlicense", "zlib", "psf-2.0", "python-2.0",
    "bsl-1.0", "cc-by-4.0", "cc-by-3.0", "cc-by-sa-4.0", "cc-by-sa-3.0", "blueoak-1.0.0",
    "artistic-2.0", "postgresql", "x11", "ncsa", "unicode-3.0", "wtfpl", "hpnd", "apache",
    "bsd", "mit-0", "afl-3.0", "ofl-1.1", "curl", "libpng-2.0", "json", "openssl",
}
COPYLEFT_PREFIXES = ("gpl", "lgpl", "agpl", "eupl", "cecill", "osl-", "sspl", "rpl", "epl",
                     "cddl", "gfdl", "mspl", "cpl", "ms-rl")

WD_OFFICIAL_SITE_IS_DOCS = frozenset({"programming language", "technical standard",
                                     "communication protocol"})

_FILE_TAIL = re.compile(r"\.(html?|php|aspx?|md|rst|txt)$", re.I)
_VERSIONISH = re.compile(r"^(v?\d+(\.\d+)*|latest|stable|main|master|dev|current|en|docs?)$", re.I)
_DOCSY_HOST = re.compile(r"^(docs?|documentation|developer|developers|dev|devdocs|manual|man|"
                         r"reference|ref|help|handbook|guide|learn|api)[.\-]", re.I)
_DOCS_SEG = re.compile(r"^(docs?|documentation|guide|guides|manual|reference|api|learn|handbook|"
                       r"tutorial|tutorials|help|python|java|js|php|ruby|go|rust|dotnet)$", re.I)
_DOCSY_PATH = re.compile(r"/(docs?|documentation|manual|reference|guide|handbook|api)(/|$)", re.I)


# --- sealed inputs -------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _get(url: str, ua: str, accept: str = "application/json", timeout: int = 60,
         tries: int = 4):
    from cytir.consent.block import ConsentBlocked, ConsentRegistryError
    from cytir.sources.family import HttpSpec
    from cytir.sources.http import make_client
    last = None
    # redirects followed, as urlopen did; a >= 400 answer is an error (retried), as HTTPError was
    spec = HttpSpec(timeout=float(timeout), headers={"user-agent": ua, "accept": accept})
    for n in range(tries):
        try:
            with make_client(spec) as client:
                r = client.get(url)
                r.raise_for_status()
                return r.json()
        except (ConsentBlocked, ConsentRegistryError):
            raise                      # fail closed: a declined host is never retried
        except Exception as e:  # noqa: BLE001 -- retried, then raised: an error is not an empty
            last = e
            time.sleep(2 ** n)
    raise RuntimeError(f"GET {url} failed after {tries} tries: {last!r}")


def _packages_row(p: dict, registry: str, rank: int) -> dict:
    md = p.get("metadata") if isinstance(p.get("metadata"), dict) else {}
    rm = p.get("repo_metadata") if isinstance(p.get("repo_metadata"), dict) else {}
    return {
        "registry": registry, "rank": rank, "name": p.get("name"),
        "downloads": p.get("downloads"), "downloads_period": p.get("downloads_period"),
        "dependent_packages_count": p.get("dependent_packages_count"),
        "dependent_repos_count": p.get("dependent_repos_count"),
        "homepage": p.get("homepage"), "documentation_url": p.get("documentation_url"),
        "metadata_documentation": md.get("documentation"),
        "repository_url": p.get("repository_url"), "repo_homepage": rm.get("homepage"),
        "licenses": p.get("licenses"),
    }


def fetch_registries(out_dir: Path, ua: str, registries: dict | None = None,
                     sleep: float = 0.4) -> dict:
    """Pull top-N per registry into sealed jsonl files + a sha256 manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": SCHEMA + ".inputs", "fetched_at": _now(), "sources": {}}
    for registry, (top_n, sort) in (registries or REGISTRIES).items():
        rows, page = [], 1
        while len(rows) < top_n:
            url = (ECOSYSTE_MS.format(registry=registry)
                   + f"?sort={sort}&order=desc&per_page=100&page={page}")
            batch = _get(url, ua)
            if not batch:
                break
            rows.extend(_packages_row(p, registry, len(rows) + i + 1)
                        for i, p in enumerate(batch))
            page += 1
            time.sleep(sleep)
        rows = rows[:top_n]
        path = out_dir / f"registry-{registry}.jsonl"
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        manifest["sources"][f"registry:{registry}"] = {
            "file": path.name, "rows": len(rows), "sort": sort, "sha256": sha256_file(path),
            "api": ECOSYSTE_MS.format(registry=registry)}
        print(f"fetch {registry}: {len(rows)} rows (sort={sort})", flush=True)
    (out_dir / "inputs.manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def fetch_brew(out_dir: Path, ua: str, top_n: int = 400, sleep: float = 0.25) -> dict:
    data = _get(BREW_ANALYTICS, ua)
    rows = []
    for item in data["items"][:top_n]:
        name = item["formula"]
        try:
            meta = _get(BREW_FORMULA.format(name=urllib.parse.quote(name)), ua, tries=2)
        except RuntimeError:
            continue
        rows.append({"registry": "homebrew", "rank": int(item["number"]), "name": name,
                     "downloads": int(item["count"].replace(",", "")),
                     "downloads_period": "365d", "homepage": meta.get("homepage"),
                     "documentation_url": None, "licenses": meta.get("license")})
        time.sleep(sleep)
    path = out_dir / "registry-homebrew.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return {"file": path.name, "rows": len(rows), "sha256": sha256_file(path), "api": BREW_ANALYTICS}


WDQS_CLASSES = {  # QIDs printed with every row's class label so a wrong QID is visible
    "programming language": "Q9143", "software framework": "Q271680",
    "technical standard": "Q317623", "communication protocol": "Q132364",
    "database management system": "Q8513",
}


def fetch_wikidata(out_dir: Path, ua: str, min_sitelinks: int = 20, limit: int = 1500) -> dict:
    rows = []
    for label, qid in WDQS_CLASSES.items():
        q = f"""SELECT ?item ?itemLabel ?site ?doc ?sl
          (GROUP_CONCAT(DISTINCT ?licLabel; separator=" | ") AS ?lics) WHERE {{
          ?item wdt:P31 wd:{qid} . ?item wdt:P856 ?site . OPTIONAL {{ ?item wdt:P2078 ?doc }}
          OPTIONAL {{ ?item wdt:P275 ?lic . ?lic rdfs:label ?licLabel . FILTER(LANG(?licLabel) = "en") }}
          ?item wikibase:sitelinks ?sl . FILTER(?sl >= {min_sitelinks})
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }}
          GROUP BY ?item ?itemLabel ?site ?doc ?sl LIMIT {limit}"""
        url = WDQS + "?format=json&query=" + urllib.parse.quote(q)
        data = _get(url, ua, accept="application/sparql-results+json", timeout=120)
        for b in data["results"]["bindings"]:
            rows.append({"registry": "wikidata", "wd_class": label, "wd_qid": qid,
                         "name": b["itemLabel"]["value"], "item": b["item"]["value"],
                         "homepage": b["site"]["value"],
                         "documentation_url": (b.get("doc") or {}).get("value"),
                         "licenses": (b.get("lics") or {}).get("value") or None,
                         "licenses_basis": "wikidata P275 (license) label",
                         "sitelinks": int(b["sl"]["value"])})
        print(f"fetch wikidata {label}: cumulative {len(rows)}", flush=True)
        time.sleep(2)
    path = out_dir / "registry-wikidata.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return {"file": path.name, "rows": len(rows), "sha256": sha256_file(path), "api": WDQS}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --- normalisation -------------------------------------------------------------------------

def fold_host(host: str) -> str:
    host = host.lower().strip().strip(".")
    host = host.split(":")[0]
    return re.sub(r"^www\d*\.", "", host)


def registrable(host: str) -> str:
    """Heuristic eTLD+1 (two labels, three under a known second-level suffix)."""
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net", "ac", "gov", "edu"} \
            and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def docs_target(url: str | None):
    """URL -> (host, path_prefix) or None. Prefix is `/` unless the host is a shared one."""
    if not url or not isinstance(url, str):
        return None
    url = url.strip()
    if "://" not in url:
        if re.match(r"^[a-z][a-z0-9+.-]*:(?!\d)", url, re.I):      # mailto:, javascript:, ...
            return None
        url = "https://" + url
    sp = urllib.parse.urlsplit(url)
    if sp.scheme not in ("http", "https") or not sp.hostname:
        return None
    host = fold_host(sp.hostname)
    if "." not in host:
        return None
    prefix = "/"
    if host in SHARED_PATH_HOSTS:
        segs = [s for s in _FILE_TAIL.sub("", sp.path).split("/") if s]
        keep = []
        for s in segs[:2]:
            if _VERSIONISH.match(s) and keep:
                break
            keep.append(s)
        prefix = "/" + "/".join(keep) if keep else "/"
    return host, prefix


def is_platform(host: str) -> bool:
    return host in PLATFORM_HOSTS


def docsy(host: str, url: str) -> bool:
    """A homepage counts as docs evidence only if the URL itself says documentation."""
    path = urllib.parse.urlsplit(url if "://" in url else "https://" + url).path
    return bool(_DOCSY_HOST.match(host) or _DOCSY_PATH.search(path)
                or host.endswith(DOCS_CLOUD_SUFFIXES))


_LABEL_SPDX = (          # Wikidata P275 English labels -> SPDX-ish; first match wins
    ("mit license", "MIT"), ("apache license", "Apache-2.0"), ("bsd", "BSD-3-Clause"),
    ("isc license", "ISC"), ("mozilla public license", "MPL-2.0"),
    ("python software foundation license", "PSF-2.0"), ("zlib", "Zlib"),
    ("creative commons zero", "CC0-1.0"), ("cc0", "CC0-1.0"),
    ("creative commons attribution-sharealike", "CC-BY-SA-4.0"),
    ("creative commons attribution", "CC-BY-4.0"), ("artistic license", "Artistic-2.0"),
    ("unlicense", "Unlicense"), ("boost software license", "BSL-1.0"),
    ("postgresql license", "PostgreSQL"), ("gnu affero", "AGPL-3.0"),
    ("gnu lesser", "LGPL-3.0"), ("gnu general public", "GPL-3.0"),
    ("eclipse public", "EPL-2.0"), ("proprietary", "proprietary"),
    ("all rights reserved", "proprietary"), ("freeware", "proprietary"),
)


def label_to_spdx(text: str | None) -> str | None:
    """A ` | `-joined list of licence LABELS -> a SPDX OR-expression; unknown labels are kept raw
    (and classify as `other`), never guessed."""
    if not text:
        return None
    out = []
    for lab in (x.strip() for x in text.split("|")):
        low = lab.lower()
        out.append(next((spdx for key, spdx in _LABEL_SPDX if key in low), lab))
    return " OR ".join(dict.fromkeys(out))


def licence_class(spdx: str | None) -> str:
    if not spdx:
        return "unstated"
    toks = [t for t in re.split(r"[\s()/,]+|\bOR\b|\bAND\b|\bWITH\b", str(spdx).lower()) if t]
    toks = [t for t in toks if t not in ("or", "and", "with", "or-later", "only")]
    if not toks:
        return "unstated"
    if any(t in PERMISSIVE for t in toks):          # a dual licence offering a permissive arm
        return "permissive"
    if any(t.startswith(COPYLEFT_PREFIXES) for t in toks):
        return "copyleft"
    if any(t in ("proprietary", "commercial", "unlicensed", "see license in license") for t in toks):
        return "proprietary"
    return "other"


# --- build ---------------------------------------------------------------------------------

def load_inputs(in_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(in_dir.glob("registry-*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{in_dir}: no registry-*.jsonl inputs; an empty roster is not a roster")
    wd = sorted((r for r in rows if r["registry"] == "wikidata"),
                key=lambda r: (-int(r.get("sitelinks") or 0), r["name"]))
    for i, r in enumerate(wd, 1):              # Wikidata has no usage count: sitelinks order is its rank
        r["rank"] = i
    return rows


def candidate_urls(r: dict):
    """(field, url) in evidence order. A homepage is docs evidence only when it looks like docs."""
    for field in ("documentation_url", "metadata_documentation"):
        if r.get(field):
            yield field, r[field], True
    if r.get("registry") == "wikidata" and r.get("wd_class") in WD_OFFICIAL_SITE_IS_DOCS \
            and r.get("homepage"):
        # the official website (P856) of a language, standard or protocol IS its canonical
        # reference; for a framework or DBMS it is only evidence when it looks like docs (below)
        yield "wikidata_P856", r["homepage"], True
    for field in ("homepage", "repo_homepage"):
        v = r.get(field)
        t = docs_target(v)
        if t and docsy(t[0], v):
            yield field, v, False


def _usage_score(r: dict) -> float:
    """Per-registry rank decay: downloads are not comparable across ecosystems, ranks are."""
    return 1.0 / (r["rank"] + 20)


def scope_of(host: str, prefix: str, url: str, wd_official: bool) -> tuple[str, str]:
    """How much of a host is the project's documentation?
    host         docs-dominant: a docs-labelled host, a docs-cloud tenant, or the official site of a
                 language / standard / protocol (Wikidata). Fetched whole.
    path_prefix  documentation lives under a path of a host that serves other things (vendor
                 marketing, blogs): the vendor-content boundary of POLICY 4.4 forbids fetching
                 the whole host at the documentation trust class.
    site_root    the evidence URL is the root of a generic host: single-project site or vendor
                 site, the registry cannot say which. Not fetched until classified."""
    if prefix != "/":                                   # SHARED_PATH_HOSTS fixed it already
        return "path_prefix", prefix
    if _DOCSY_HOST.match(host) or host.endswith(DOCS_CLOUD_SUFFIXES) or wd_official:
        return "host", "/"
    path = urllib.parse.urlsplit(url if "://" in url else "https://" + url).path
    segs = [x for x in path.split("/") if x]
    if segs and _DOCS_SEG.match(segs[0]):
        return "path_prefix", "/" + segs[0]
    return "site_root", "/"


def build_roster(rows: list[dict], top_n: int = 800) -> list[dict]:
    """Group candidates by (host, prefix). Rank by summed per-registry rank decay."""
    groups: dict[tuple, dict] = {}
    seen_inputs = 0
    for r in rows:
        seen_inputs += 1
        got = next(((f, u, explicit) for f, u, explicit in candidate_urls(r)
                    if docs_target(u) is not None), None)
        if not got:
            continue
        field, url, explicit = got
        host, prefix = docs_target(url)
        g = groups.setdefault((host, prefix), {
            "host": host, "path_prefix": prefix, "projects": [], "first_url": url, "evidence": [],
            "registries": set(), "score": 0.0, "usage": 0, "wd_official": False})
        g["wd_official"] = g["wd_official"] or field == "wikidata_P856"
        g["score"] += _usage_score(r)
        g["registries"].add(r["registry"])
        dl = r.get("downloads")
        g["usage"] += int(dl) if isinstance(dl, (int, float)) else 0
        g["projects"].append({"name": r["name"], "registry": r["registry"], "rank": r["rank"],
                              "licenses": r.get("licenses"), "score": _usage_score(r),
                              "repository_url": r.get("repository_url")})
        g["evidence"].append({"registry": r["registry"], "package": r["name"], "field": field,
                              "url": url, "explicit_docs_field": explicit})
    roster = []
    for (host, prefix), g in groups.items():
        scope, prefix = scope_of(host, prefix, g["first_url"], g["wd_official"])
        g["projects"].sort(key=lambda p: -p["score"])
        top = g["projects"][0]
        lic_text = top["licenses"]
        if top["registry"] == "wikidata":
            lic_text = label_to_spdx(lic_text)
        lic_class = "proprietary_vendor" if host in VENDOR_PROPRIETARY_DOCS \
            else licence_class(lic_text)
        roster.append({
            "schema": SCHEMA, "project": top["name"], "host": host, "path_prefix": prefix,
            "scope": scope,
            "n_packages": len(g["projects"]), "other_projects": [p["name"] for p in g["projects"][1:4]],
            "registries": sorted(g["registries"]), "usage_score": round(g["score"], 6),
            "downloads_sum": g["usage"],
            "evidence": g["evidence"][:5],
            "docs_evidence_explicit": any(e["explicit_docs_field"] for e in g["evidence"]),
            "licence_spdx": lic_text, "licence_class": lic_class,
            "repository_url": top.get("repository_url"),
            "licence_basis": ("vendor-proprietary-docs (task 232 2A exclusion)"
                              if lic_class == "proprietary_vendor" else
                              "Wikidata P275 label of the item (the project's own licence "
                              "statement as Wikidata records it); docs licence NOT read"
                              if top["registry"] == "wikidata" else
                              "registry SPDX of the top package (CODE licence); the project's "
                              "docs-licence statement is NOT read at roster time"),
            "platform": is_platform(host)})
    roster.sort(key=lambda r: (-r["usage_score"], r["host"]))
    for i, r in enumerate(roster, 1):
        r["usage_rank"] = i
    # the roster keeps platform rows (visible exclusions) but they do not count against top_n
    kept, n = [], 0
    for r in roster:
        if r["platform"]:
            kept.append(r)
        elif n < top_n:
            kept.append(r)
            n += 1
    return kept


# --- enrichment ----------------------------------------------------------------------------

def twin_facets(twin_url: str, collection: str) -> tuple[dict, dict]:
    """(all domain counts, docs-only domain counts). Facet, never a scroll."""
    def facet(flt):
        body = {"key": "domain", "limit": 400_000}
        if flt:
            body["filter"] = flt
        req = urllib.request.Request(
            f"{twin_url}/collections/{collection}/facet", data=json.dumps(body).encode(),
            headers={"content-type": "application/json"})
        hits = json.load(IH.urlopen(req, timeout=900))["result"]["hits"]
        return {h["value"]: h["count"] for h in hits}
    docs = facet({"must": [{"key": "source_type", "match": {"value": "docs"}}]})
    return facet(None), docs


def join_twin(roster: list[dict], all_counts: dict, docs_counts: dict) -> None:
    by_reg = defaultdict(int)
    for d, c in all_counts.items():
        by_reg[registrable(d)] += c
    for r in roster:
        h = r["host"]
        r["served_points"] = int(all_counts.get(h, 0))
        r["served_docs_points"] = int(docs_counts.get(h, 0))
        r["served_registrable"] = registrable(h)
        r["served_registrable_points"] = int(by_reg.get(registrable(h), 0))
        r["twin_join"] = "domain==host(www-folded), indexed facet (approximate, not exact=true); registrable = suffix-sum heuristic"


def consent_class(rec: dict | None) -> str:
    """The live resolver's receipt -> one named state. Our failures are `held_*`, never declined."""
    if not rec:
        return "unresolved"
    robots = (rec.get("signals") or {}).get("robots") or {}
    if rec.get("status") == "unknown":
        return "held_unresolved"
    if rec.get("search_ok") is False:
        if robots.get("http_status") in (401, 403) or not robots.get("fetched", True):
            return "held_unreadable"                  # 403 is our vantage's fact (ADR-103)
        return "declined_search"
    if rec.get("train_ok") is False:
        return "search_only"                          # CCBot / train opt-out: train_ok=false only
    return "ok"


# --- estimates and eligibility -------------------------------------------------------------

def chunks_per_page(passages_jsonl: Path) -> dict:
    """MEASURED chunks/page from a sealed docs tranche: rows / distinct urls, plus per-host median."""
    per_url: dict[str, int] = Counter()
    per_host_rows: dict[str, int] = Counter()
    per_host_urls: dict[str, set] = defaultdict(set)
    with open(passages_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            u = d.get("url") or d.get("canonical_url")
            if not u:
                continue
            per_url[u] += 1
            h = fold_host(urllib.parse.urlsplit(u).hostname or "")
            per_host_rows[h] += 1
            per_host_urls[h].add(u)
    ratios = sorted(per_host_rows[h] / len(per_host_urls[h]) for h in per_host_rows)
    if not ratios:
        raise SystemExit(f"{passages_jsonl}: no url-bearing rows; cannot calibrate")
    rows = sum(per_url.values())
    return {"file": str(passages_jsonl), "rows": rows, "urls": len(per_url),
            "hosts": len(ratios), "pooled": rows / len(per_url),
            "host_median": ratios[len(ratios) // 2],
            "host_p25": ratios[len(ratios) // 4], "host_p75": ratios[3 * len(ratios) // 4]}


HOST_SCOPE_MAX_URLS = 20_000      # above this a "docs host" is more likely a platform (scratch.mit.edu)


def rank_rows(roster: list[dict], cal: dict, cap: int, min_cc_urls: int = 20) -> None:
    """Fill est_points / est_net_new / eligible / ineligible_reason (exactly one terminal)."""
    cpp = cal["pooled"]       # page-weighted: the estimate is a SUM over hosts, so a page-weighted ratio
    for r in roster:
        urls = r.get("cc_urls")
        if urls is None:
            r["est_points"] = r["est_net_new"] = None
        else:
            r["est_points"] = int(min(urls, cap) * cpp)
            served = max(r.get("served_docs_points", 0), r.get("served_points", 0))
            r["est_net_new"] = max(0, r["est_points"] - served)
        r["cap_pages"], r["chunks_per_page"] = cap, round(cpp, 2)
        reason = None
        if r.get("tranche"):
            reason = f"in_{r['tranche']}"          # already launched: not eligible a second time
        elif r["platform"]:
            reason = "platform_aggregator"
        elif r["licence_class"] != "permissive":
            reason = f"licence_class_{r['licence_class']}"
        elif r.get("consent_state") in (None, "unresolved"):
            reason = "consent_not_resolved"
        elif r["consent_state"].startswith("held"):
            reason = f"consent_{r['consent_state']}"
        elif r["consent_state"] == "declined_search":
            reason = "consent_declined_search"
        elif urls is None:
            reason = "cc_not_counted"
        elif urls < min_cc_urls:
            reason = "cc_urls_below_floor"
        elif r["scope"] == "path_prefix":
            reason = "path_prefix_scope_needs_prefix_fetch"
        elif r["scope"] == "site_root":
            reason = "site_root_scope_unclassified"
        elif urls > HOST_SCOPE_MAX_URLS:
            reason = "host_too_large_to_assume_docs_only"
        elif not r["est_net_new"]:
            reason = "already_served_to_estimate"
        r["ineligible_reason"] = reason
        r["eligible"] = reason is None


def queue_entry(r: dict, cap: int) -> dict:
    """A discovery_ingest queue item. `self-signal` is the affirmative classification of
    POLICY 4.4: the registry names this host as the project's documentation URL."""
    return {"domain": r["host"], "reg": registrable(r["host"]), "class": "tech",
            "score": r["usage_score"], "authority": 0.75, "swiss": False,
            "class_basis": "self-signal", "n_endorser_regs": 0, "cap": cap,
            "why": f"docs-roster:{r['project']}", "docs_roster_rank": r["usage_rank"],
            "licence_spdx": r["licence_spdx"], "licence_class": r["licence_class"]}


def summarise(roster: list[dict]) -> str:
    n = len(roster)
    reasons = Counter(r.get("ineligible_reason") or "ELIGIBLE" for r in roster)
    eligible = [r for r in roster if r.get("eligible")]
    lines = [f"docs roster: {n} rows | eligible {len(eligible)} | "
             + " ".join(f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])),
             f"conservation: in={n} eligible={len(eligible)} "
             f"ineligible={sum(v for k, v in reasons.items() if k != 'ELIGIBLE')} "
             f"-> {'OK' if len(eligible) + sum(v for k, v in reasons.items() if k != 'ELIGIBLE') == n else 'BROKEN'}"]
    tot = sum(r["est_net_new"] or 0 for r in eligible)
    lines.append(f"est net-new points, eligible rows: {tot:,}")
    lines.append("licence classes: " + str(dict(Counter(r["licence_class"] for r in roster))))
    lines.append("consent states: " + str(dict(Counter(r.get("consent_state") for r in roster))))
    lines.append("scopes: " + str(dict(Counter(r["scope"] for r in roster))))
    lines.append("registries: " + str(dict(Counter(g for r in roster for g in r.get("registries", [])))))
    return "\n".join(lines)


# --- CLI -----------------------------------------------------------------------------------

def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def _ua() -> str:
    from cytir.core import user_agent
    return user_agent.user_agent()


def cc_counts(hosts_prefixes: list[tuple[str, str]], crawl: str, threads: int = 4) -> dict:
    """Distinct status-200 HTML URLs (no query string) per host[/prefix] in the LOCAL mirror.
    Uses cc_columnar's own surtkey ranges; one query per parquet file touched."""
    from cytir.acquire import cc_columnar as ccx
    if not os.environ.get("AW_CC_REQUIRE_LOCAL_INDEX") or os.environ["AW_CC_REQUIRE_LOCAL_INDEX"] == "0":
        raise SystemExit("cc stage reads the local mirror only: set CYTIR_CC_REQUIRE_LOCAL_INDEX=1 "
                         "and AW_CC_LOCAL_INDEX_DIR (no remote parquet scan from this stage)")
    con = ccx._con(threads)
    pre2key: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for host, prefix in hosts_prefixes:
        p = ccx.domain_surtkey_prefix(host)
        if p:
            pre2key[p].append((host, prefix))
    bounds = ccx.file_bounds(crawl, con)
    by_file: dict[str, list[str]] = defaultdict(list)
    for p in pre2key:
        lo, hi = p + "/", p + "/￿"
        for fn, mn, mx in bounds:
            if not (mx < lo or mn > hi):
                by_file[fn].append(p)
    remote = [fn for fn in by_file if not (fn.startswith("/") or fn.startswith("file://"))]
    if remote:
        raise SystemExit(f"{len(remote)} needed parquet file(s) not in the local mirror, e.g. "
                         f"{remote[0]}; refusing a remote read (deferred, not empty)")
    counts: dict[tuple, int] = defaultdict(int)
    digest = ccx._nonempty_digest_predicate(column="p.content_digest")
    for n, (fn, prefixes) in enumerate(by_file.items(), 1):
        ors = " OR ".join(f"(p.url_surtkey >= '{pp}/' AND p.url_surtkey < '{pp}/￿')"
                          for pp in prefixes)
        rows = con.execute(f"""
            SELECT p.url_surtkey, any_value(p.url_path) FROM read_parquet('{fn}') p
            WHERE ({ors}) AND p.fetch_status = 200 AND p.content_mime_type LIKE 'text/html%'
              AND ({digest}) AND position('?' in p.url) = 0
            GROUP BY p.url_surtkey""").fetchall()
        for sk, path in rows:
            pre = sk.split(")", 1)[0] + ")"
            for host, prefix in pre2key.get(pre, []):
                if prefix == "/" or (path or "").startswith(prefix):
                    counts[(host, prefix)] += 1
        print(f"cc file {n}/{len(by_file)}: {len(rows)} urls", flush=True)
    return {k: counts.get(k, 0) for k in hosts_prefixes}


def main(argv=None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--out", required=True)
    f.add_argument("--only-wikidata", action="store_true",
                   help="re-pull Wikidata into an existing inputs dir, keep the other sealed inputs")
    f.add_argument("--no-brew", action="store_true"); f.add_argument("--no-wikidata", action="store_true")
    b = sub.add_parser("build"); b.add_argument("--inputs", required=True)
    b.add_argument("--out", required=True); b.add_argument("--top", type=int, default=800)
    t = sub.add_parser("twin"); t.add_argument("--roster", required=True)
    t.add_argument("--twin", default=TWIN.url()); t.add_argument("--collection", default=TWIN.collection())
    c = sub.add_parser("cc"); c.add_argument("--roster", required=True)
    c.add_argument("--crawl", default="CC-MAIN-2026-39")
    k = sub.add_parser("consent"); k.add_argument("--roster", required=True)
    k.add_argument("--cache-dir", required=True)
    k.add_argument("--concurrency", type=int, default=4,
                   help="distinct hosts in flight; <=4 is the ratified polite bound")
    r = sub.add_parser("rank"); r.add_argument("--roster", required=True)
    r.add_argument("--calibration", required=True, help="passages.jsonl of a sealed docs tranche")
    r.add_argument("--cap", type=int, default=1000)
    r.add_argument("--queue", action="append", default=[],
                   help="a launched tranche queue (repeatable); its hosts leave the eligible set")
    q = sub.add_parser("queue"); q.add_argument("--roster", required=True)
    q.add_argument("--out", required=True); q.add_argument("--top", type=int, default=50)
    q.add_argument("--cap", type=int, default=1000)
    s = sub.add_parser("summary"); s.add_argument("--roster", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "fetch":
        out = Path(a.out)
        if a.only_wikidata:
            man = json.loads((out / "inputs.manifest.json").read_text())
            man["sources"]["registry:wikidata"] = fetch_wikidata(out, _ua())
            (out / "inputs.manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
            print(json.dumps({k: v["rows"] for k, v in man["sources"].items()}, indent=1))
            return 0
        man = fetch_registries(out, _ua())
        if not a.no_brew:
            man["sources"]["registry:homebrew"] = fetch_brew(out, _ua())
        if not a.no_wikidata:
            man["sources"]["registry:wikidata"] = fetch_wikidata(out, _ua())
        (out / "inputs.manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
        print(json.dumps({k: v["rows"] for k, v in man["sources"].items()}, indent=1))
    elif a.cmd == "build":
        rows = load_inputs(Path(a.inputs))
        roster = build_roster(rows, a.top)
        _write(Path(a.out), roster)
        print(f"build: {len(rows)} registry rows -> {len(roster)} roster rows "
              f"({sum(1 for x in roster if x['platform'])} platform rows kept visible)")
    elif a.cmd == "twin":
        roster = _read(Path(a.roster))
        allc, docsc = twin_facets(a.twin, a.collection)
        join_twin(roster, allc, docsc)
        _write(Path(a.roster), roster)
        print(f"twin: {len(allc)} domains, {sum(allc.values()):,} points; docs-only "
              f"{len(docsc)} domains {sum(docsc.values()):,} points; rows with served>0: "
              f"{sum(1 for x in roster if x['served_points'])}")
    elif a.cmd == "cc":
        roster = _read(Path(a.roster))
        res = cc_counts([(x["host"], x["path_prefix"]) for x in roster], a.crawl)
        for x in roster:
            x["cc_urls"] = res[(x["host"], x["path_prefix"])]
            x["cc_crawl"] = a.crawl
            x["cc_predicate"] = "distinct url_surtkey, fetch_status=200, text/html, digest, no '?', local mirror"
        _write(Path(a.roster), roster)
        print(f"cc: {sum(1 for x in roster if x['cc_urls'])} of {len(roster)} hosts have captures; "
              f"{sum(x['cc_urls'] for x in roster):,} urls")
    elif a.cmd == "consent":
        from cytir.consent import resolver
        roster = _read(Path(a.roster))
        hosts = sorted({x["host"] for x in roster if not x["platform"]})
        recs = resolver.resolve_hosts(hosts, cache_dir=a.cache_dir, concurrency=a.concurrency)
        for x in roster:
            rec = recs.get(x["host"])
            x["consent_state"] = None if x["platform"] else consent_class(rec)
            x["consent_checked_at"] = (rec or {}).get("checked_at")
            x["consent_search_ok"] = (rec or {}).get("search_ok")
            x["consent_train_ok"] = (rec or {}).get("train_ok")
            x["consent_reasons"] = ((rec or {}).get("reasons") or [])[:2]
        _write(Path(a.roster), roster)
        print("consent:", dict(Counter(x["consent_state"] for x in roster)))
    elif a.cmd == "rank":
        roster = _read(Path(a.roster))
        cal = chunks_per_page(Path(a.calibration))
        for qpath in a.queue:                           # tranche name = the queue file's stem
            qhosts = {x["domain"] for x in json.loads(Path(qpath).read_text())}
            for x in roster:
                if x["host"] in qhosts:
                    x["tranche"] = Path(qpath).stem
        rank_rows(roster, cal, a.cap)
        _write(Path(a.roster), roster)
        print("calibration:", json.dumps(cal))
        print(summarise(roster))
    elif a.cmd == "queue":
        roster = _read(Path(a.roster))
        el = sorted((x for x in roster if x["eligible"]), key=lambda x: -x["est_net_new"])[: a.top]
        Path(a.out).write_text(json.dumps([queue_entry(x, a.cap) for x in el], indent=1))
        print(f"queue: {len(el)} hosts, est net-new {sum(x['est_net_new'] for x in el):,} -> {a.out}")
    elif a.cmd == "summary":
        print(summarise(_read(Path(a.roster))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
