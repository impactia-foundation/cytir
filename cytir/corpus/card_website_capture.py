#!/usr/bin/env python3
"""Build resumable Common-Crawl-first website snapshots for entity cards.

This is a master-side source stage. It never embeds or writes Qdrant. Input
rows must already carry an entity-bound ``site_association_receipt``; URL/domain
equality is never treated as association evidence. One site snapshot is built per
registrable domain and then joined to every independently validated card.

Common Crawl is preferred: a pinned crawl's exact URL variants are resolved
first, followed by deterministic domain fallback. Exact archived response
bytes are range-fetched, content-addressed in L0, then decoded/extracted. A
direct own-UA crawl is eligible only after a genuine CC miss or when live
CC-mode consent makes the archived copy inadmissible. Lookup/fetch errors are
retryable errors, never clean misses.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import time
import urllib.parse
import urllib.robotparser

import httpx

import cytir.core.common as AC
from cytir.extract import html_extract as _html_extract
from cytir.acquire import cc_columnar as CC
from cytir.consent import policy as CP
import cytir.consent.resolver as CR
import cytir.corpus.entity_card_website as ECW
from cytir.acquire import l0_store as l0_store
from cytir.acquire.artifact_io import (atomic_publish, canonical_json,
                         publish_jsonl_or_verify, sha256_file)
from cytir.acquire.crawl import page_links
from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client


VERSION = "entity-card-website-snapshot.v3"
SELECTION_VERSION = "cc-exact-then-domain-newest-complete-v1"
SITE_SCHEMA = "aw.entity-card.website-site-snapshot.v1"
RUN_SCHEMA = "aw.entity-card.website-snapshot-run.v1"
MANIFEST_SCHEMA = "aw.entity-card.website-snapshot-manifest.v1"
L1_MARKERS = {
    "about": ("about", "company", "uber-uns", "ueber-uns", "a-propos",
              "qui-sommes", "chi-siamo"),
    "services": ("service", "services", "leistungen", "dienstleistungen",
                 "prestations", "servizi"),
    "products": ("product", "products", "produkte", "produits", "prodotti"),
}


def _warc_transport():
    """Lazy heavy ingest import; CLI/config checks stay dependency-light."""
    from cytir.acquire import discovery_ingest
    return discovery_ingest


class RetryableSiteError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: pathlib.Path) -> str:
    return sha256_file(path)


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def atomic_state(path: pathlib.Path, value: dict) -> None:
    """Replace mutable retry state; terminal checkpoints use atomic_publish."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _host(url: str) -> str:
    return (urllib.parse.urlsplit(url).hostname or "").lower()


def _domain(url: str) -> str:
    return _html_extract.reg_domain(_host(url))


def exact_url_variants(url: str) -> list[str]:
    """Finite declared variants; no unrelated path/query guessing."""
    clean = AC.strip_tracking_url(url).split("#", 1)[0]
    parsed = urllib.parse.urlsplit(clean)
    if not parsed.hostname:
        return []
    path = parsed.path or "/"
    hosts = [parsed.hostname.lower()]
    if hosts[0].startswith("www."):
        hosts.append(hosts[0][4:])
    else:
        hosts.append("www." + hosts[0])
    variants = []
    for scheme in (parsed.scheme if parsed.scheme in {"http", "https"} else "https",
                   "http", "https"):
        for host in hosts:
            variants.append(urllib.parse.urlunsplit(
                (scheme, host, path, parsed.query, "")))
    return list(dict.fromkeys(variants))


def _capture_rank(record: dict) -> tuple:
    return (str(record.get("fetch_time") or ""), int(record.get("length") or 0),
            str(record.get("warc_filename") or ""),
            -int(record.get("offset") or 0))


def choose_exact(records: dict[str, dict], variants: list[str]) -> dict | None:
    candidates = [(record, -variants.index(key)) for key, record in records.items()
                  if key in variants]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (_capture_rank(item[0]), item[1]))[0]


def choose_domain_homepage(records: list[dict], domain: str) -> dict | None:
    admitted = [row for row in records if _domain(row.get("url") or "") == domain]
    if not admitted:
        return None

    def rank(row: dict) -> tuple:
        parsed = urllib.parse.urlsplit(row["url"])
        path = parsed.path or "/"
        depth = len([part for part in path.split("/") if part])
        home = 0 if path in {"", "/", "/index.html", "/index.htm"} else 1
        return (home, depth, len(path), 0 if parsed.scheme == "https" else 1,
                row["url"])

    return min(admitted, key=rank)


def l1_rule(url: str) -> str | None:
    path = urllib.parse.unquote(urllib.parse.urlsplit(url).path).casefold()
    folded = path.replace("ü", "u").replace("à", "a").replace("é", "e")
    for rule, markers in L1_MARKERS.items():
        if any(marker in folded for marker in markers):
            return rule
    return None


def l1_candidates(base_url: str, html: str, maximum: int) -> list[tuple[str, str]]:
    if maximum <= 0:
        return []
    host = _host(base_url)

    def link_key(value: str, keep_query: bool = False) -> str:
        cleaned = AC.strip_tracking_url(value).split("#", 1)[0]
        if not keep_query:
            parsed = urllib.parse.urlsplit(cleaned)
            cleaned = urllib.parse.urlunsplit(
                (parsed.scheme, parsed.netloc, parsed.path or "/", "", ""))
        return cleaned

    by_rule: dict[str, list[str]] = collections.defaultdict(list)
    for url in page_links(base_url, html, host, link_key):
        rule = l1_rule(url)
        if rule and _domain(url) == _domain(base_url):
            by_rule[rule].append(url)
    chosen = []
    for rule in ("about", "services", "products"):
        if by_rule[rule]:
            chosen.append((rule, sorted(set(by_rule[rule]))[0]))
        if len(chosen) >= maximum:
            break
    return chosen


def normalized_surface(html: str, url: str, depth: int) -> tuple[str, str] | None:
    markdown, mode = _html_extract.extract_content_markdown(html, url)
    if not markdown:
        return None
    # Bound by the shared Arctic-token budget, not by characters. The former
    # 900/1,200-char + 256-word caps were a different unit from the ratified
    # budget and clamped the surface *before* the receipt hash, so the budget
    # could never bind downstream (task 393 §2). Per-source capture keeps the
    # full budget available; the total allowance is enforced at composition,
    # where homepage and depth-1 pages are spent against one shared budget.
    content = ECW.normalized_content_tokens(markdown, ECW.WEBSITE_TOKEN_BUDGET)
    return (content, mode) if content else None


def path_allowed(policy: dict, url: str, agent: str) -> bool:
    body = policy.get("_robots_body") or ""
    if not body:
        return True
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(body.splitlines())
    return parser.can_fetch(agent, url)


class Services:
    """Network/L0 boundary, injectable for fully offline tests."""

    def __init__(self, *, l0_root: pathlib.Path, timeout: float, delay: float,
                 enable_index: bool = True):
        self.l0_root = l0_root
        self.delay = delay
        self.consent_client = CR._client()
        self.cc_con = CC._con() if enable_index else None
        self.cc_route = ("common-crawl-s3-warc-range" if CC.use_s3()
                         else "common-crawl-https-warc-range")
        self.cc_client = make_client(HttpSpec(timeout=timeout, follow_redirects=True))
        self.direct_client = make_client(HttpSpec(
            timeout=timeout, follow_redirects=False,
            headers={"Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5"}))
        self.last_direct_request: dict[str, float] = {}
        self.candidate_cache: dict[str, tuple[dict | None, str]] = {}
        self.candidate_errors: dict[str, str] = {}

    def close(self) -> None:
        self.consent_client.close()
        if self.cc_con is not None:
            self.cc_con.close()
        self.cc_client.close()
        self.direct_client.close()

    def consent(self, url: str, *, direct: bool) -> dict:
        # Fetch workers may be restarted after a transient consent outage.
        # They must share the resolver's persisted receipt cache instead of
        # re-probing every completed host on each resume.  Unknown receipts are
        # intentionally not cached by ``resolve_cached_host`` and stay held.
        policy = CR.resolve_cached_host(self.consent_client, _host(url),
                                        direct_crawl=direct)
        expected = "direct" if direct else "cc"
        if policy.get("mode") != expected:
            raise RetryableSiteError("consent_mode", "resolver mode drift")
        return policy

    def resolve_exact(self, crawl: str, urls: list[str], *, complete: bool) -> dict:
        if self.cc_con is None:
            raise RuntimeError("columnar resolution is disabled in fetch-only mode")
        return CC.resolve(crawl, urls, con=self.cc_con,
                          require_complete=complete)

    def resolve_domains(self, crawl: str, domains: list[str], *,
                        complete: bool) -> dict[str, list[dict]]:
        if self.cc_con is None:
            raise RuntimeError("columnar resolution is disabled in fetch-only mode")
        return CC.resolve_domains(crawl, domains, con=self.cc_con, per_domain=100,
                                  require_complete=complete)

    def fetch_cc(self, record: dict) -> bytes | None:
        return _warc_transport().fetch_warc_response_body(self.cc_client, record)

    def decode_cc(self, body: bytes) -> str:
        return _warc_transport()._decode_http_body(b"", body)

    def store_l0(self, url: str, fetched_at: str, body: bytes,
                 extra: dict) -> dict:
        return l0_store.write(url, fetched_at, body, root=str(self.l0_root),
                              extra=extra)

    def prepare_candidates(self, crawl: str, seeds: dict[str, dict],
                           batch_sites: int) -> None:
        """Resolve at most four columnar queries per SITE BATCH, not per site."""
        domains = sorted(seeds)
        for start in range(0, len(domains), batch_sites):
            batch = domains[start:start + batch_sites]
            variants = {domain: exact_url_variants(seeds[domain]["url"])
                        for domain in batch}
            try:
                exact = self.resolve_exact(
                    crawl, [url for domain in batch for url in variants[domain]],
                    complete=True)
                missing = []
                for domain in batch:
                    record = choose_exact(exact, variants[domain])
                    if record is None:
                        missing.append(domain)
                    else:
                        self.candidate_cache[domain] = (record, "exact")
                keys = list(dict.fromkeys(
                    key for domain in missing
                    for key in (_host(seeds[domain]["url"]), domain)))
                resolved = self.resolve_domains(crawl, keys, complete=True)
                unresolved = []
                for domain in missing:
                    rows = []
                    for key in dict.fromkeys((_host(seeds[domain]["url"]), domain)):
                        rows.extend(resolved.get(key, []))
                    record = choose_domain_homepage(rows, domain)
                    if record is None:
                        unresolved.append(domain)
                    else:
                        self.candidate_cache[domain] = (record, "domain")
                if not unresolved:
                    continue
                any_exact = self.resolve_exact(
                    crawl, [url for domain in unresolved for url in variants[domain]],
                    complete=False)
                any_keys = list(dict.fromkeys(
                    key for domain in unresolved
                    for key in (_host(seeds[domain]["url"]), domain)))
                any_domains = self.resolve_domains(
                    crawl, any_keys, complete=False)
                for domain in unresolved:
                    has_exact = any(url in any_exact for url in variants[domain])
                    has_domain = any(any_domains.get(key) for key in
                                     dict.fromkeys((_host(seeds[domain]["url"]), domain)))
                    self.candidate_cache[domain] = (
                        None, "unusable" if has_exact or has_domain else "miss")
            except Exception as exc:  # one batch error is retryable for its sites
                for domain in batch:
                    self.candidate_errors[domain] = str(exc)

    def direct_get(self, url: str, domain: str) -> dict:
        requested = url
        for _hop in range(7):
            if _domain(url) != domain:
                raise ValueError("direct redirect left verified registrable domain")
            policy = self.consent(url, direct=True)
            if policy.get("status") == "unknown":
                raise RetryableSiteError("direct_consent", "direct consent unresolved")
            try:
                CP.require_resolved_search_index_receipt(policy)
            except ValueError as exc:
                raise ValueError(f"direct consent declined: {exc}") from exc
            if not path_allowed(policy, url, CR.ROBOT_AGENT):
                raise ValueError("direct URL-specific robots rule disallows path")
            host = _host(url)
            pause = self.delay - (
                time.monotonic() - self.last_direct_request.get(host, 0.0))
            if pause > 0:
                time.sleep(pause)
            fetched_at = now_iso()
            try:
                response = self.direct_client.get(url)
            except httpx.HTTPError as exc:
                raise RetryableSiteError(
                    "direct_fetch", f"direct HTTP error: {type(exc).__name__}") from exc
            self.last_direct_request[host] = time.monotonic()
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location:
                    raise ValueError("direct redirect lacks Location")
                url = urllib.parse.urljoin(url, location)
                continue
            if response.status_code != 200:
                raise ValueError(f"direct HTTP {response.status_code}")
            if "html" not in response.headers.get("content-type", "").lower():
                raise ValueError("direct response is not HTML")
            header = ("Content-Type: " + response.headers.get("content-type", "")) \
                .encode("latin-1", "replace")
            return {
                "requested_url": requested, "url": str(response.url),
                "raw": response.content,
                "html": _warc_transport()._decode_http_body(header, response.content),
                "fetched_at": fetched_at, "retrieved_at": now_iso(),
                "policy": policy,
            }
        raise ValueError("direct redirect limit exceeded")


def admitted_policy(policy: dict) -> bool:
    if policy.get("status") == "unknown":
        raise RetryableSiteError("consent", "live consent unresolved")
    try:
        CP.require_resolved_search_index_receipt(policy)
    except ValueError:
        return False
    return True


def _policy_core(policy: dict) -> dict:
    return {key: policy[key] for key in
            ("status", "search_ok", "train_ok", "checked_at", "mode")}


def cc_acquisition(record: dict, *, crawl: str, manifest_sha: str,
                   retrieved_at: str, route: str) -> dict:
    return {
        "schema": ECW.ACQUISITION_RECEIPT_SCHEMA,
        "route": route, "crawl": crawl,
        "fetch_time": record["fetch_time"], "url": record["url"],
        "warc_filename": record["warc_filename"], "offset": record["offset"],
        "length": record["length"], "content_digest": record["digest"],
        "content_truncated": record["content_truncated"],
        "index_manifest_sha256": manifest_sha,
        "selection_version": SELECTION_VERSION, "retrieved_at": retrieved_at,
    }


def direct_acquisition(page: dict) -> dict:
    return {
        "schema": ECW.ACQUISITION_RECEIPT_SCHEMA,
        "route": "direct-crawl-fallback", "url": page["url"],
        "retrieved_at": page["retrieved_at"], "fetch_time": page["fetched_at"],
        "user_agent": CR.UA_STR,
        "robots_checked_at": page["policy"]["checked_at"],
    }


def archive_page(services: Services, *, raw: bytes, html: str, url: str,
                 fetched_at: str, policy: dict, acquisition: dict,
                 depth: int, rule: str | None = None) -> dict:
    l0 = services.store_l0(url, fetched_at, raw, {"acquisition": acquisition})
    surface = normalized_surface(html, url, depth)
    if surface is None:
        return {"status": "EMPTY", "url": url, "fetched_at": fetched_at,
                "raw_sha256": sha256_bytes(raw),
                "raw_artifact_pointer": "l0://" + l0["path"],
                "policy": _policy_core(policy), "acquisition": acquisition,
                "depth": depth, "selection_rule": rule}
    content, mode = surface
    return {
        "status": "ADMITTED", "url": url, "fetched_at": fetched_at,
        "content": content, "extraction_mode": mode,
        "raw_sha256": sha256_bytes(raw),
        "raw_artifact_pointer": "l0://" + l0["path"],
        "policy": _policy_core(policy), "acquisition": acquisition,
        "depth": depth, "selection_rule": rule,
    }


def _resolve_domain_records(services: Services, crawl: str, seed_url: str,
                            complete: bool) -> list[dict]:
    host, domain = _host(seed_url), _domain(seed_url)
    keys = list(dict.fromkeys([host, domain]))
    resolved = services.resolve_domains(crawl, keys, complete=complete)
    rows, seen = [], set()
    for key in keys:
        for row in resolved.get(key, []):
            identity = (row["warc_filename"], row["offset"], row["length"])
            if identity not in seen:
                seen.add(identity)
                rows.append(row)
    return rows


def _cc_candidate(services: Services, crawl: str, seed_url: str) -> tuple[dict | None, str]:
    domain = _domain(seed_url)
    if domain in getattr(services, "candidate_errors", {}):
        raise RuntimeError(services.candidate_errors[domain])
    if domain in getattr(services, "candidate_cache", {}):
        return services.candidate_cache[domain]
    variants = exact_url_variants(seed_url)
    exact = services.resolve_exact(crawl, variants, complete=True)
    selected = choose_exact(exact, variants)
    if selected is not None:
        return selected, "exact"
    domain_rows = _resolve_domain_records(services, crawl, seed_url, True)
    selected = choose_domain_homepage(domain_rows, _domain(seed_url))
    if selected is not None:
        return selected, "domain"
    # A complete miss is eligible for direct fallback. A capture filtered only
    # because it is truncated is explicitly unusable, not a miss.
    any_exact = services.resolve_exact(crawl, variants, complete=False)
    any_domain = _resolve_domain_records(services, crawl, seed_url, False)
    if any_exact or any_domain:
        return None, "unusable"
    return None, "miss"


def _cc_page(services: Services, record: dict, *, crawl: str,
             manifest_sha: str, policy: dict, depth: int,
             rule: str | None = None) -> dict:
    retrieved_at = now_iso()
    raw = services.fetch_cc(record)
    if raw is None:
        raise RetryableSiteError("cc_fetch", "WARC range fetch/parse failed")
    acquisition = cc_acquisition(
        record, crawl=crawl, manifest_sha=manifest_sha,
        retrieved_at=retrieved_at, route=services.cc_route)
    html = services.decode_cc(raw)
    page = archive_page(
        services, raw=raw, html=html, url=record["url"],
        fetched_at=record["fetch_time"], policy=policy,
        acquisition=acquisition, depth=depth, rule=rule)
    page["_html"] = html
    return page


def cc_snapshot(services: Services, *, domain: str, seed: dict, crawl: str,
                manifest_sha: str, max_l1: int) -> dict:
    try:
        record, selection = _cc_candidate(services, crawl, seed["url"])
    except Exception as exc:
        if isinstance(exc, RetryableSiteError):
            raise
        raise RetryableSiteError("cc_lookup", str(exc)) from exc
    if record is None:
        return {"status": "CC_UNUSABLE" if selection == "unusable" else "CC_MISS",
                "cc_selection": selection}
    policy = services.consent(record["url"], direct=False)
    if not admitted_policy(policy):
        return {"status": "CC_CONSENT_DECLINED", "cc_selection": selection,
                "policy": _policy_core(policy)}
    if not path_allowed(policy, record["url"], "CCBot"):
        return {"status": "CC_CONSENT_DECLINED", "cc_selection": selection,
                "policy": _policy_core(policy),
                "reason": "CCBot URL-specific robots rule disallows capture path"}
    homepage = _cc_page(
        services, record, crawl=crawl, manifest_sha=manifest_sha,
        policy=policy, depth=0)
    if homepage["status"] != "ADMITTED":
        homepage.pop("_html", None)
        return {"status": "CC_EMPTY", "cc_selection": selection,
                "homepage": homepage}
    candidates = l1_candidates(homepage["url"], homepage.pop("_html"), max_l1)
    l1_pages, complete = [], True
    if candidates:
        urls = [url for _rule, url in candidates]
        try:
            resolved = services.resolve_exact(crawl, urls, complete=True)
        except Exception as exc:
            raise RetryableSiteError("cc_l1_lookup", str(exc)) from exc
        for rule, url in candidates:
            record_l1 = resolved.get(url)
            if record_l1 is None:
                complete = False
                continue
            l1_policy = services.consent(record_l1["url"], direct=False)
            if not admitted_policy(l1_policy):
                complete = False
                continue
            if not path_allowed(l1_policy, record_l1["url"], "CCBot"):
                complete = False
                continue
            page = _cc_page(
                services, record_l1, crawl=crawl, manifest_sha=manifest_sha,
                policy=l1_policy, depth=1, rule=rule)
            page.pop("_html", None)
            if page["status"] == "ADMITTED":
                l1_pages.append(page)
            else:
                complete = False
    return {"status": "ADMITTED", "route": services.cc_route,
            "cc_selection": selection, "homepage": homepage,
            "l1_pages": l1_pages, "l1_refresh_complete": complete,
            "l1_candidates": len(candidates)}


def direct_snapshot(services: Services, *, domain: str, seed: dict,
                    max_l1: int, reason: str) -> dict:
    try:
        raw_page = services.direct_get(seed["url"], domain)
    except RetryableSiteError:
        raise
    except ValueError as exc:
        return {"status": "DIRECT_DECLINED", "direct_reason": str(exc),
                "fallback_reason": reason}
    acquisition = direct_acquisition(raw_page)
    homepage = archive_page(
        services, raw=raw_page["raw"], html=raw_page["html"],
        url=raw_page["url"], fetched_at=raw_page["fetched_at"],
        policy=raw_page["policy"], acquisition=acquisition, depth=0)
    if homepage["status"] != "ADMITTED":
        return {"status": "DIRECT_EMPTY", "homepage": homepage,
                "fallback_reason": reason}
    candidates = l1_candidates(raw_page["url"], raw_page["html"], max_l1)
    pages, complete = [], True
    for rule, url in candidates:
        try:
            page_raw = services.direct_get(url, domain)
        except RetryableSiteError:
            raise
        except ValueError:
            complete = False
            continue
        page = archive_page(
            services, raw=page_raw["raw"], html=page_raw["html"],
            url=page_raw["url"], fetched_at=page_raw["fetched_at"],
            policy=page_raw["policy"], acquisition=direct_acquisition(page_raw),
            depth=1, rule=rule)
        if page["status"] == "ADMITTED":
            pages.append(page)
        else:
            complete = False
    return {"status": "ADMITTED", "route": "direct-crawl-fallback",
            "fallback_reason": reason, "homepage": homepage,
            "l1_pages": pages, "l1_refresh_complete": complete,
            "l1_candidates": len(candidates)}


def process_site(services: Services, *, domain: str, seed: dict, crawl: str,
                 manifest_sha: str, max_l1: int,
                 direct_fallback: bool) -> dict:
    snapshot = cc_snapshot(
        services, domain=domain, seed=seed, crawl=crawl,
        manifest_sha=manifest_sha, max_l1=max_l1)
    if snapshot["status"] in {"CC_MISS", "CC_CONSENT_DECLINED"}:
        if direct_fallback:
            return direct_snapshot(
                services, domain=domain, seed=seed, max_l1=max_l1,
                reason=snapshot["status"].lower())
        snapshot["status"] += "_DIRECT_DISABLED"
    return snapshot


def _license(seed: dict, page: dict) -> tuple[str, dict]:
    evidence = seed.get("website_license_evidence")
    if evidence is None:
        value = "all-rights-reserved"
        evidence = {
            "schema": ECW.LICENSE_EVIDENCE_SCHEMA,
            "basis": "default-copyright", "source_url": page["url"],
            "checked_at": page["fetched_at"],
        }
    else:
        value = str(seed.get("website_license") or "").strip()
        if not value:
            raise ValueError("reviewed website license evidence lacks license value")
    return value, ECW.validate_license_evidence(page["url"], value, evidence)


def build_receipt(seed: dict, page: dict, association: dict) -> dict:
    license_value, license_evidence = _license(seed, page)
    policy = page["policy"]
    receipt = {
        "schema": ECW.WEBSITE_RECEIPT_SCHEMA, "source_url": page["url"],
        "status": policy["status"], "search_ok": policy["search_ok"],
        "train_ok": (policy["train_ok"] is True
                     and not ECW.license_is_restrictive(license_value)),
        "consent_mode": policy["mode"], "license": license_value,
        "attribution": str(seed.get("website_attribution") or seed.get("name")
                           or seed.get("title") or seed["domain"]),
        "checked_at": policy["checked_at"], "fetched_at": page["fetched_at"],
        "content_sha256": sha256_bytes(page["content"].encode("utf-8")),
        "raw_sha256": page["raw_sha256"],
        "raw_artifact_pointer": page["raw_artifact_pointer"],
        "association_receipt": association, "license_evidence": license_evidence,
        "acquisition_receipt": page["acquisition"],
    }
    if page["depth"] == 1:
        receipt.update(depth=1, selection_rule=page["selection_rule"],
                       token_count=len(page["content"].split()))
    return ECW.validate_receipt(seed, page["content"], receipt)


def load_inventory(path: pathlib.Path, shard_count: int,
                   shard_index: int, limit: int) -> dict[str, list[tuple[int, dict, dict, dict]]]:
    groups: dict[str, list[tuple[int, dict, dict, dict]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            seed = row.get("seed", row)
            if not isinstance(seed, dict) or not seed.get("url") or not seed.get("domain"):
                raise ValueError(f"line {line_no}: seed URL/domain required")
            domain = _domain(seed["url"])
            if not domain or domain != _html_extract.reg_domain(str(seed["domain"]).lower()):
                raise ValueError(f"line {line_no}: seed URL/domain disagree")
            association = ECW.validate_association_receipt(
                seed, seed.get("site_association_receipt"))
            shard = int(hashlib.sha256(domain.encode()).hexdigest()[:16], 16)
            if shard % shard_count != shard_index:
                continue
            if limit and domain not in groups and len(groups) >= limit:
                continue
            groups.setdefault(domain, []).append((line_no, row, seed, association))
    return groups


def checkpoint_path(work: pathlib.Path, domain: str) -> pathlib.Path:
    return work / "sites" / (sha256_bytes(domain.encode()) + ".json")


def retry_path(work: pathlib.Path, domain: str) -> pathlib.Path:
    return work / "retry" / (sha256_bytes(domain.encode()) + ".json")


def representative_seed(rows: list[tuple[int, dict, dict, dict]]) -> dict:
    """Stable site seed; input line order never changes the shared snapshot."""
    return min(
        (row[2] for row in rows),
        key=lambda seed: (AC.canonical_url(seed["url"]), ECW.entity_key(seed)))


def _joined_rows(groups, work: pathlib.Path):
    for domain in sorted(groups):
        snapshot = json.loads(checkpoint_path(work, domain).read_text(encoding="utf-8"))
        if snapshot["status"] != "ADMITTED":
            continue
        for line_no, row, seed, association in sorted(groups[domain], key=lambda item: item[0]):
            enriched_seed = dict(seed)
            homepage = snapshot["homepage"]
            enriched_seed.update(
                website_content=homepage["content"],
                website_receipt=build_receipt(seed, homepage, association))
            if snapshot["l1_refresh_complete"]:
                enriched_seed["l1_pages"] = [
                    {"content": page["content"],
                     "receipt": build_receipt(seed, page, association),
                     "artifact_pointer": page["raw_artifact_pointer"]}
                    for page in snapshot["l1_pages"]
                ]
            enriched = dict(row)
            if "seed" in row:
                enriched["seed"] = enriched_seed
            else:
                enriched = enriched_seed
            yield line_no, enriched


def _audit_rows(groups, work: pathlib.Path):
    for domain in sorted(groups):
        snapshot = json.loads(checkpoint_path(work, domain).read_text(encoding="utf-8"))
        compact = {key: value for key, value in snapshot.items()
                   if key not in {"homepage", "l1_pages"}}
        if snapshot.get("homepage"):
            compact["homepage_provenance"] = {
                key: value for key, value in snapshot["homepage"].items()
                if key not in {"content", "_html"}}
        if snapshot.get("l1_pages"):
            compact["l1_provenance"] = [
                {key: value for key, value in page.items()
                 if key not in {"content", "_html"}}
                for page in snapshot["l1_pages"]]
        compact["domain"] = domain
        yield compact


def main(argv: list[str] | None = None, prog: str = "cyt corpus card-website") -> None:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="input", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--audit-out", required=True, type=pathlib.Path)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--work-root", required=True, type=pathlib.Path)
    parser.add_argument("--l0-root", required=True, type=pathlib.Path)
    parser.add_argument("--wave", required=True)
    parser.add_argument("--crawl", required=True)
    parser.add_argument(
        "--cc-transport", required=True,
        choices=("signed-s3", "https-mirror"),
        help="Fail-loud Common Crawl index and WARC transport for this wave")
    parser.add_argument("--cc-index-manifest-sha256", required=True)
    parser.add_argument(
        "--source-git-sha", required=True,
        help="40-character source commit represented by the immutable release")
    parser.add_argument(
        "--runtime-lock-sha256", required=True,
        help="SHA-256 of the immutable runtime dependency lock used by this wave")
    parser.add_argument("--max-l1", type=int, default=3)
    parser.add_argument("--direct-fallback", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0,
                        help="registrable-site limit, never a card-row limit")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resolve-batch-sites", type=int, default=250)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        raise SystemExit("ABORT input does not exist")
    if (not 1 <= args.shard_count or not 0 <= args.shard_index < args.shard_count
            or not 0 <= args.max_l1 <= 3 or args.timeout <= 0 or args.delay < 0
            or args.resolve_batch_sites <= 0):
        raise SystemExit("ABORT invalid shard/max-l1/network parameters")
    for label, value in (
            ("CC index manifest", args.cc_index_manifest_sha256),
            ("runtime lock", args.runtime_lock_sha256)):
        if (len(value) != 64 or
                not all(character in "0123456789abcdefABCDEF"
                        for character in value)):
            raise SystemExit(f"ABORT {label} SHA-256 is invalid")
    if (len(args.source_git_sha) != 40 or
            not all(character in "0123456789abcdefABCDEF"
                    for character in args.source_git_sha)):
        raise SystemExit("ABORT source git SHA is invalid")

    try:
        groups = load_inventory(
            args.input, args.shard_count, args.shard_index, args.limit)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ABORT {exc}") from exc
    work = (args.work_root / f"wave={args.wave}" /
            f"shard={args.shard_index:04d}-of-{args.shard_count:04d}")
    identity = {
        "schema": RUN_SCHEMA, "version": VERSION,
        "input_sha256": file_sha(args.input), "wave": args.wave,
        "crawl": args.crawl,
        "cc_transport": args.cc_transport,
        "cc_index_manifest_sha256": args.cc_index_manifest_sha256.lower(),
        "runtime_lock_sha256": args.runtime_lock_sha256.lower(),
        "selection_version": SELECTION_VERSION, "max_l1": args.max_l1,
        "direct_fallback": args.direct_fallback,
        "limit": args.limit,
        "resolve_batch_sites": args.resolve_batch_sites,
        "l0_root": str(args.l0_root.resolve()),
        "shard_count": args.shard_count, "shard_index": args.shard_index,
        "git_sha": args.source_git_sha.lower(), "started_at": now_iso(),
    }
    identity_path = work / "_run.identity.json"
    if args.manifest.exists():
        raise SystemExit("COMPLETE immutable manifest already exists")
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in identity.items()
                      if key != "started_at"}
        prior = {key: value for key, value in existing.items()
                 if key != "started_at"}
        if prior != comparable:
            raise SystemExit("ABORT partial run identity differs; use a new wave")
        identity = existing
    else:
        atomic_publish(identity_path,
                       (json.dumps(identity, indent=2, sort_keys=True) + "\n").encode())

    os.environ["AW_CC_S3"] = "1" if args.cc_transport == "signed-s3" else "0"
    CC._reset_s3_decision()
    services = Services(l0_root=args.l0_root, timeout=args.timeout, delay=args.delay)
    expected_route = ("common-crawl-s3-warc-range"
                      if args.cc_transport == "signed-s3"
                      else "common-crawl-https-warc-range")
    if services.cc_route != expected_route:
        services.close()
        raise SystemExit(
            f"ABORT requested CC transport {args.cc_transport!r} unavailable; "
            f"resolved route was {services.cc_route!r}")
    retryable = []
    try:
        services.prepare_candidates(
            args.crawl,
            {domain: representative_seed(rows) for domain, rows in groups.items()
             if not checkpoint_path(work, domain).exists()},
            args.resolve_batch_sites)
        for ordinal, domain in enumerate(sorted(groups), 1):
            terminal = checkpoint_path(work, domain)
            if terminal.exists():
                checkpoint = json.loads(terminal.read_text(encoding="utf-8"))
                if checkpoint.get("domain") != domain:
                    raise ValueError("terminal site checkpoint domain drift")
                continue
            try:
                snapshot = process_site(
                    services, domain=domain, seed=representative_seed(groups[domain]),
                    crawl=args.crawl,
                    manifest_sha=args.cc_index_manifest_sha256.lower(),
                    max_l1=args.max_l1,
                    direct_fallback=args.direct_fallback)
            except RetryableSiteError as exc:
                state = {"domain": domain, "status": "RETRYABLE_ERROR",
                         "stage": exc.stage, "error": str(exc),
                         "observed_at": now_iso()}
                atomic_state(retry_path(work, domain), state)
                retryable.append(state)
                continue
            snapshot.update(schema=SITE_SCHEMA, domain=domain,
                            completed_at=now_iso())
            atomic_publish(
                terminal,
                (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode())
            retry_path(work, domain).unlink(missing_ok=True)
            if ordinal % 100 == 0:
                print(f"{ordinal}/{len(groups)} site checkpoints", flush=True)
    finally:
        services.close()
    if retryable:
        stages = dict(collections.Counter(row["stage"] for row in retryable))
        raise SystemExit(f"INCOMPLETE retryable_sites={len(retryable)} stages={stages}")

    output_sha, admitted_cards = publish_jsonl_or_verify(
        args.out, (row for _line, row in _joined_rows(groups, work)))
    audit_sha, _audit_rows_count = publish_jsonl_or_verify(
        args.audit_out, _audit_rows(groups, work))
    checkpoints = [json.loads(checkpoint_path(work, domain).read_text(encoding="utf-8"))
                   for domain in sorted(groups)]
    statuses = dict(collections.Counter(row["status"] for row in checkpoints))
    manifest = {
        "schema": MANIFEST_SCHEMA, "version": VERSION,
        "created_at": identity["started_at"], "git_sha": identity["git_sha"],
        "input": str(args.input), "input_sha256": identity["input_sha256"],
        "wave": args.wave, "crawl": args.crawl,
        "cc_transport": identity["cc_transport"],
        "cc_index_manifest_sha256": identity["cc_index_manifest_sha256"],
        "runtime_lock_sha256": identity["runtime_lock_sha256"],
        "selection_version": SELECTION_VERSION,
        "shard_count": args.shard_count, "shard_index": args.shard_index,
        "selected_sites": len(groups),
        "selected_card_rows": sum(len(rows) for rows in groups.values()),
        "admitted_card_rows": admitted_cards, "site_statuses": statuses,
        "output": str(args.out), "output_sha256": output_sha,
        "audit": str(args.audit_out), "audit_sha256": audit_sha,
        "site_checkpoints": {
            checkpoint_path(work, domain).name: file_sha(checkpoint_path(work, domain))
            for domain in sorted(groups)},
        "l0_root": identity["l0_root"],
        "policy": "Common-Crawl-first exact+domain; live CC-mode consent; "
                  "direct own-UA/path-robots fallback only after genuine CC miss "
                  "or CC-copy consent decline; entity-bound association joins",
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    atomic_publish(args.manifest, manifest_bytes)  # terminal publication, always last
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
