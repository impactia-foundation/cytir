#!/usr/bin/env python3
"""Live per-host CONSENT resolution — the ADR-009 / spec §4.2 two-flag resolver.

Given a list of hosts, re-resolve consent LIVE per host (a CC snapshot carries
no usable consent — spec principle 3 / CLAUDE.md: "Consent re-resolved live,
per source, two flags search_ok/train_ok, never inherited from Common Crawl").
Emits one record per host:

    {host, status, search_ok, train_ok, reasons, signals, checked_at}

status ∈ "ok" (resolved) | "unknown" (transient error — RETRYABLE, never
treated as consent OR as refusal; re-run re-resolves it).

Signal chain implemented (spec §4.2 precedence — AIPREF → robots.txt → TDMRep;
"stricter signal wins on conflict"):

  1. ``https://<host>/robots.txt``  — bot-token rules for our token
     (``authoritative-web``), the wildcard ``*``, ``CCBot`` (our copy's
     provenance), and the training-only token family (``GPTBot``,
     ``anthropic-ai``/``ClaudeBot``, ``cohere-ai``, ``Bytespider`` and any
     ``*-Extended`` token, per the spec's "honour training-only tokens").
  2. AIPREF ``Content-Usage`` — the WG has no RFC yet (ADR-009: "track before
     coding"); the mechanism the existing chain (fetch_hn_cc.py) implements is
     robots.txt-embedded declarations, so we parse best-effort
     ``Content-Usage:`` lines in robots.txt (``search=n``, ``ai=n``,
     ``train-ai=n``, ``all=n``) plus the widespread ``noai`` marker.
  3. W3C TDMRep — ``tdm-reservation: 1`` inside robots.txt (the fetch_hn_cc
     sniff) AND ``https://<host>/.well-known/tdmrep.json`` (any policy entry
     with ``tdm-reservation: 1``).
  4. ``ai.txt`` — NOT fetched at this stage (spec lists it last in precedence;
     the existing fetch_hn_cc chain does not implement it, and we keep the
     politeness budget at ≤2 requests/host). Tracked as a follow-up.

RUBRIC — how signals map to the two flags. The spec defines the flags
(`search_ok` = may we store minimal retrieval representations, including
dense/sparse vectors, terms, and a short snippet, and surface the result;
`train_ok` = may content enter training datasets, model training, or training
export) but is silent on some mappings; where
silent we take the CONSERVATIVE reading, documented here:

  * ``User-agent: * Disallow: /`` (root disallowed for the wildcard)
      → search_ok=False, train_ok=False. (Plain full opt-out.)
  * Our own token (``authoritative-web``) disallowed on ``/``
      → search_ok=False, train_ok=False.
  * **CCBot fully disallowed → train_ok=False ONLY; search_ok unaffected.**
    Common Crawl *is* the canonical LLM-training corpus, so a CCBot block is
    a TRAINING signal — the same reading ADR-024 amendment 2 (Tim,
    2026-07-15) already gave GPTBot / ClaudeBot / anthropic-ai /
    Google-Extended. Ruled for CCBot itself by Tim on 2026-09-03 (ADR-024
    amendment 5): *"A cc-bot decline should mean no_train, not no_search."*
    This holds in BOTH provenance modes — the ``direct_crawl`` argument does
    not change the CCBot outcome (see ``apply_robots_flags``). Superseded
    reading (before 2026-09-03): search_ok=False AND train_ok=False whenever
    ``direct_crawl=False``, on the ground that a CC-derived copy of a
    CCBot-refusing source had no consent chain to us.
  * **TDM reservation (tdmrep.json or robots ``tdm-reservation: 1``)
      → train_ok=False AND search_ok=False.** Strictly TDMRep reserves
    text-and-data-mining (which would only force train_ok=False), but the
    spec calls it a "hard opt-out, EU-legally load-bearing" and the existing
    fetch_hn_cc chain DROPS such URLs entirely (returns "blocked"), i.e. the
    spec's intent treats a TDM reservation as a full opt-out. We keep that
    reading.
  * ``noai`` in robots.txt → train_ok=False AND search_ok=False (same
    fetch_hn_cc hard-opt-out precedent as TDM).
  * Training-only token disallowed (GPTBot / *-Extended / anthropic-ai /
    ClaudeBot / cohere-ai / Bytespider on ``/``) → train_ok=False only;
    search_ok unaffected (the spec's "training-only tokens" case — the source
    permits retrieval indexing but not AI training; retrieval embeddings remain
    allowed while training-data use/export is prohibited).
  * AIPREF Content-Usage: ``search=n`` → search_ok=False; ``ai=n`` /
    ``train-ai=n`` / ``genai=n`` → train_ok=False; ``all=n`` → both False.
  * robots.txt 401/403 → search_ok=False, train_ok=False (access-controlled,
    per fetch_hn_cc's "BLOCK").
  * robots.txt 404 / empty 200 → no robots signal (open); flags stay True
    unless TDMRep flips them.
  * Transient errors (network / timeout / 5xx) on EITHER probe
      → status="unknown" (retryable). An outage is NEVER treated as consent
    (fetch_hn_cc: "we never silently treat an outage as consent") — and,
    conservatively, never as refusal either: unknown hosts are HELD, not
    merged and not carded.

Host granularity: consent is resolved per HOST (subdomain), not per
registrable domain — robots.txt and tdmrep.json are host-scoped. Flags are
evaluated at the root path ``/`` (host-level gate). The fetched robots body is
cached so a later per-URL gate can re-evaluate specific paths.

Politeness: ≤2 concurrent hosts, ~0.5 s spacing per host, 10 s timeouts,
2 attempts on transient failures. Errors never crash the batch.

Cache: ``~/aw-data/consent/<host>.json`` with ``checked_at``; re-resolved when
older than 30 days or when status=="unknown" (retryable).

USAGE
  python consent_resolver.py --hosts a.ch,b.ch [--out results.jsonl]
  python consent_resolver.py --hosts-file hosts.txt --cache-dir ~/aw-data/consent
"""
from __future__ import annotations

import argparse
import json
import os
import ipaddress
import math
import pathlib
import re
import sys
import threading
import time
import urllib.robotparser
from typing import Any
from urllib.parse import urlsplit
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import datetime, timezone

import httpx

from cytir.core.user_agent import user_agent   # task 432: one home for the crawler identity

# The identity we present when ASKING PERMISSION. It must be the same identity we
# later crawl with -- a consent verdict recorded under one requester and acted on
# by another describes a crawler that does not exist (the defect fixed in
# cytir.news.freshness_ingest / cytir.news.weekly_delta). Sourced from operator.json so a
# self-hoster asks as themselves; for AW deployments it renders byte-identically
# to the literal this line replaced.
UA_STR = user_agent()
ROBOT_AGENT = "authoritative-web"          # the token robots rules bind us by
CACHE_DIR = os.environ.get("AW_CONSENT_CACHE_DIR") or os.path.expanduser("~/aw-data/consent")
# ADR-024 amendment 3 (RATIFIED Tim 2026-08-19, PR #61; option (c)). Path is data,
# not a literal: same relative layout in the repo and inside a pinned release.
CONSENT_UNRESOLVED_REGISTER = os.environ.get(
    "AW_CONSENT_UNRESOLVED_REGISTER",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "consent-unresolved-register.yaml"))
MAX_AGE_DAYS = 30
TIMEOUT = 10.0
HOST_DELAY = 0.5
CONCURRENCY = 2
ROBOTS_BODY_CAP = 65536                    # cached body truncation


def _positive_seconds(name: str, default: float) -> float:
    """Read a wall-clock budget from the environment. Never yields 'unbounded'."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number of seconds, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0 seconds, got {value!r}: a policy "
                         "probe has no unbounded mode")
    return value


# --- wall-clock bounds on a policy probe (task 485) --------------------------
# A policy endpoint that will not answer is an INSTRUMENT failure: it resolves
# to held_unknown, never to a verdict, and it must do so in bounded time.
#
# Until task 485 the only bound was httpx's PER-OPERATION timeout, and the
# deadline machinery below it was opt-in through AW_CONSENT_ENDPOINT_BUDGET —
# unset in every deployed sweep unit, so no total bound existed at all. Two
# shapes escape a per-operation timeout: a redirect chain (httpx follows up to
# twenty hops, each hop getting a fresh connect+read budget) and a server that
# trickles bytes just under the read timeout, which never trips it. Either one
# holds a worker for as long as the origin cares to hold it; with concurrency=2
# two such hosts stop a whole 100-host batch, and because a batch is only
# written out at its end the unit stays ACTIVE while recording nothing. That is
# the Wave28 TDMRep livelock class (ADR-024 amendment 3) seen from the
# transport side.
#
# The budgets are wall-clock, always on, and enforced hop by hop and chunk by
# chunk. Verdict semantics are untouched: every abort returns "no answer", and
# no answer has always meant held_unknown.
# 70s is the honest worst case of one policy endpoint that answers slowly but
# legally: two HTTPX attempts at TIMEOUT each, the pauses between them, and the
# curl fallback, for BOTH schemes when the caller has not pinned one (~63s).
# 150s covers the two probes of a host plus their pacing. The budgets exist to
# cut off what has no honest worst case at all -- a redirect chain and a
# trickled body -- not to hurry a slow publisher into held_unknown.
POLICY_ENDPOINT_BUDGET = _positive_seconds("AW_CONSENT_ENDPOINT_BUDGET", 70.0)
HOST_BUDGET = _positive_seconds("AW_CONSENT_HOST_BUDGET", 150.0)
MAX_REDIRECTS = 5
BODY_BYTE_CAP = ROBOTS_BODY_CAP * 4        # read cap on a policy body
BUDGET_EXHAUSTED = "policy endpoint budget exhausted"

# Training-only bot tokens (spec §4.2: "honour training-only tokens like
# *-Extended"). Any UA token ending in "-extended" is also honoured.
TRAIN_TOKENS = ("gptbot", "anthropic-ai", "claudebot", "claude-web",
                "cohere-ai", "bytespider", "meta-externalagent")

_UA_LINE = re.compile(r"(?im)^[ \t]*user-agent[ \t]*:[ \t]*(\S+)")
_CONTENT_USAGE = re.compile(r"(?im)^[ \t]*content-usage[ \t]*:[ \t]*(.+)$")
_TDM_ROBOTS = re.compile(r"(?im)^[ \t]*tdm-reservation[ \t]*:[ \t]*1\b")
_NOAI = re.compile(r"(?i)\bnoai\b")

# --- robots.txt body shape (task 466) ----------------------------------------
# RFC 9309 §2.2 names the fields a robots.txt line may carry; §2.2.1 says a
# crawler with no matching group may access everything, so a file that carries
# only non-group records (a bare ``Sitemap:``) is a WELL-FORMED allow-all, not
# an absence of policy. Both facts are needed to tell "the publisher published
# no rules for us" from "we failed to parse what the publisher published".
_GROUP_FIELDS = frozenset({
    "user-agent", "allow", "disallow", "crawl-delay", "noindex",
    "content-signal", "content-usage", "tdm-reservation",
})
_NON_GROUP_FIELDS = frozenset({
    "sitemap", "host", "clean-param", "request-rate", "visit-time",
})
_ROBOTS_FIELDS = _GROUP_FIELDS | _NON_GROUP_FIELDS
_FIELD_LINE = re.compile(
    r"(?i)^[ \t]*(" + "|".join(sorted(_ROBOTS_FIELDS)) + r")[ \t]*:")
_FIELD_ANYWHERE = re.compile(
    r"(?i)\b(" + "|".join(sorted(_ROBOTS_FIELDS)) + r")[ \t]*:")
# A CMS that prepends its own banner to robots.txt (measured live on
# asvhg-foot.com: ``/** Load from cache **/User-agent: *``) writes a real
# directive behind a short junk prefix. Recover it, but only a SHORT one, and
# never inside an HTML body where the same token is prose, not policy.
JUNK_PREFIX_CAP = 200
_HTML_HEAD = re.compile(r"(?is)^\s*(<!doctype\s+html|<html\b|<\?xml|<!--)")
# W3C TDMRep expresses the same reservation in HTML: the value is the policy,
# so ``content="0"`` is an explicit NON-reservation and must never read as one
# (task 433 S4; ADR-096's direction that an explicit machine-readable statement
# is read for what it says).
_TDM_META = re.compile(
    r"""(?is)<meta\b[^>]*\bname[ \t]*=[ \t]*["']?tdm-reservation["']?[^>]*>""")
_META_CONTENT = re.compile(r"""(?is)\bcontent[ \t]*=[ \t]*["']?([^"'\s>]*)""")


def looks_like_html(body: str) -> bool:
    """True when a body served AS robots.txt is in fact a markup document."""
    return bool(_HTML_HEAD.match(body or ""))


def sanitize_robots_body(body: str) -> tuple[str, dict]:
    """Normalise a robots.txt body before parsing; never changes its meaning.

    Two defects this repairs, both measured on live receipts (task 466):
      * a UTF-8 BOM ahead of the first field name;
      * a short junk prefix on a line whose remainder IS a directive.

    Recovery is deliberately bounded — non-markup bodies only, prefix ≤
    ``JUNK_PREFIX_CAP`` — because the alternative reading of a stray
    ``User-agent:`` inside prose is to invent policy the publisher never wrote.
    Recovering a group heading can only make MORE of the publisher's rules
    bind, never fewer.
    """
    stats = {"bom": False, "recovered_lines": 0,
             "html": looks_like_html(body or "")}
    text = body or ""
    if text.startswith("\ufeff"):
        text, stats["bom"] = text[1:], True
    if stats["html"]:
        return text, stats
    lines = []
    for raw in text.splitlines():
        line = raw
        head = line.split("#", 1)[0]
        if head.strip() and not _FIELD_LINE.match(head):
            match = _FIELD_ANYWHERE.search(head[:JUNK_PREFIX_CAP + 32])
            if match is not None and 0 < match.start() <= JUNK_PREFIX_CAP:
                line = line[match.start():]
                stats["recovered_lines"] += 1
        lines.append(line)
    if not stats["recovered_lines"]:
        return text, stats           # nothing to repair: hand back the bytes
    return "\n".join(lines), stats


def robots_body_shape(body: str) -> dict:
    """Describe a robots.txt body: is it robots syntax, and does it say anything?

    ``group_lines``/``non_group_lines`` count RFC 9309 records; a body with
    ``non_group_lines`` but no groups (the ``Sitemap:``-only file that
    spark.apache.org, numpy.org and rubyonrails.org publish) is a valid
    allow-all, which is exactly the case the retracted evidence gate misread as
    "no policy directives".
    """
    text, stats = sanitize_robots_body(body)
    shape = {"html": stats["html"], "bom": stats["bom"],
             "recovered_lines": stats["recovered_lines"],
             "group_lines": 0, "non_group_lines": 0, "unparsed_lines": 0,
             "fields": []}
    fields: set[str] = set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = _FIELD_LINE.match(line)
        if match is None:
            shape["unparsed_lines"] += 1
            continue
        field = match.group(1).lower()
        fields.add(field)
        if field in _GROUP_FIELDS:
            shape["group_lines"] += 1
        else:
            shape["non_group_lines"] += 1
    shape["fields"] = sorted(fields)
    shape["directive_lines"] = shape["group_lines"] + shape["non_group_lines"]
    return shape


def tdm_reservation_in_html(body: str) -> dict:
    """Value-SENSITIVE W3C TDMRep ``<meta name="tdm-reservation">`` reading.

    ``content="1"`` reserves; ``content="0"`` explicitly does NOT (TDMRep §2.1).
    A value-blind match would turn a publisher's explicit "you may mine this"
    into a decline, which is the opposite of reading what they wrote.
    """
    values = []
    for tag in _TDM_META.findall(body or ""):
        content = _META_CONTENT.search(tag)
        values.append((content.group(1).strip() if content else "").lower())
    return {"present": bool(values), "values": values,
            "reserved": any(value == "1" for value in values)}


def _content_signals(body: str) -> dict:
    """Cloudflare **Content Signals Policy** (task 187; first seen live on
    ohchr.org): per-UA-group ``Content-Signal: search=yes,ai-train=no,...``
    lines in robots.txt. Distinct from the AIPREF ``Content-Usage`` draft —
    values are yes/no words and the line is group-scoped like Allow/Disallow.
    Signals defined by the policy: ``search`` (building a search index and
    returning hyperlinks + short excerpts), ``ai-input`` (RAG/grounding),
    ``ai-train`` (training/fine-tuning models), ``use``.

    Returns the signal dict of the group that binds us: a group naming our
    token wins; otherwise the union of all ``*`` groups (sites often carry a
    Cloudflare-managed ``*`` group AND their own ``*`` group)."""
    groups: list[tuple[list[str], dict]] = []
    agents: list[str] = []
    sigs: dict = {}
    last_was_ua = False
    for raw in body.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"(?i)([a-z-]+)[ \t]*:[ \t]*(.*)", line)
        if not m:
            continue
        key, val = m.group(1).lower(), m.group(2).strip()
        if key == "user-agent":
            if not last_was_ua and agents:
                groups.append((agents, sigs))
                agents, sigs = [], {}
            agents.append(val.lower())
            last_was_ua = True
        else:
            last_was_ua = False
            if key == "content-signal":
                for part in val.split(","):
                    if "=" in part:
                        k, _, v = part.partition("=")
                        sigs[k.strip().lower()] = v.strip().lower()
    if agents:
        groups.append((agents, sigs))
    for ags, sg in groups:            # a group naming our token wins over *
        if any(a != "*" and (a in ROBOT_AGENT or a in "authoritativewebbot")
               for a in ags):
            return sg
    merged: dict = {}
    for ags, sg in groups:
        if "*" in ags:
            merged.update(sg)
    return merged


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _client() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT, follow_redirects=True,
                        max_redirects=MAX_REDIRECTS,
                        headers={"User-Agent": UA_STR},
                        limits=httpx.Limits(max_connections=CONCURRENCY * 2))


class BudgetExhausted(httpx.HTTPError):
    """The policy endpoint did not answer inside its wall-clock budget.

    Subclasses ``httpx.HTTPError`` on purpose: every caller already treats a
    transport error as "no answer, hold the record", so an abort raised here
    inherits the ratified semantics instead of inventing a new class.
    """


def _is_public_host(hostname: str) -> bool:
    """False for a literal address we must not chase a policy endpoint to.

    A redirect into RFC 1918 / loopback / link-local space is never a
    publisher's policy — it is a misconfigured origin (or a captive gateway)
    pointing us at our own network, and following it once cost the Wave28
    sweep a worker for as long as the internal address held the connection
    open (documents-dds-ny.un.org). Only literal addresses are decided here;
    a NAME is left to DNS, where the connect timeout still bounds us.
    """
    try:
        ip = ipaddress.ip_address(hostname.strip("[]"))
    except ValueError:
        return True
    return ip.is_global


def _fetch_bounded(client: httpx.Client, url: str, deadline: float) -> httpx.Response:
    """One policy-endpoint GET, bounded by a wall-clock ``deadline``.

    Redirects are followed HERE rather than by httpx, because httpx's redirect
    limit bounds the number of hops while its timeouts bound each operation —
    the product of the two is not a bound on anything. The body is streamed and
    capped for the same reason: a trickle under the read timeout never trips
    it. Raises ``httpx.HTTPError`` (``BudgetExhausted`` for our own aborts).

    Task 502 defect 2: two live examples (gc-handball.ch, germany-tourism.de)
    showed a card-row receipt recording ``http_status: 301`` with an empty
    body. Reproduced from the sealed task-426 receipts (pre-task-485 code, plain
    ``httpx`` auto-redirect): the origin answered a 3xx with NO usable
    ``Location`` — httpx (and this loop, identically) cannot follow nothing, so
    it correctly returns the bare 3xx as terminal. That was never a bug; the
    bug is that nothing downstream could tell "redirect stalled" apart from "no
    policy published" (fixed in ``resolve_host``) and that the hops actually
    walked before a terminal answer were discarded (fixed here: returned in
    ``response.extensions["aw_redirect_hops"]``). Separately, a Location that
    ``httpx.URL.join`` cannot parse at all (``httpx.InvalidURL`` — malformed
    port, a stray control character) does NOT subclass ``httpx.HTTPError`` and
    would otherwise escape every caller's ``except httpx.HTTPError`` and crash
    the resolver; it is folded into ``BudgetExhausted`` like every other
    unfollowable target below.
    """
    seen = {url}
    hops: list[dict[str, Any]] = []
    for hop in range(MAX_REDIRECTS + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BudgetExhausted(f"{BUDGET_EXHAUSTED} after {hop} redirect(s)")
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            raise BudgetExhausted(
                f"policy endpoint redirected to unsupported scheme {parts.scheme!r}")
        if not _is_public_host(parts.hostname or ""):
            raise BudgetExhausted(
                f"policy endpoint redirected to non-public address {parts.hostname!r}")
        timeout = max(0.1, min(TIMEOUT, remaining))
        with client.stream("GET", url, follow_redirects=False,
                           timeout=timeout) as response:
            location = response.headers.get("location")
            hop_status = response.status_code
            if not (300 <= hop_status < 400 and location):
                body = bytearray()
                # iter_bytes() with NO chunk size on purpose: any size makes
                # httpx buffer until the chunk is full, and a body trickled a
                # few bytes at a time never fills one — the deadline check
                # below would then never get to run, which is the whole hole
                # this function exists to close.
                for chunk in response.iter_bytes():
                    body += chunk
                    if len(body) >= BODY_BYTE_CAP:
                        break
                    if time.monotonic() >= deadline:
                        raise BudgetExhausted(
                            f"{BUDGET_EXHAUSTED} while reading the policy body")
                # Rebuild rather than return the streamed response: the body is
                # already decoded and capped, so the framing headers that
                # described the wire bytes would now be lying about it.
                # Task 791: carry the RAW header bytes, never the decoded str
                # pairs. ``Headers(...)`` re-encodes str values as ASCII, so one
                # non-ASCII header value ("404 Categoría no encontrada",
                # "“SAMEORIGIN”", a CSP with a raw umlaut host) raised
                # UnicodeEncodeError and the whole host was held "resolver
                # crash". The bytes are what the origin sent; policy parsing
                # decodes them exactly as before.
                headers = [(k, v) for k, v in response.headers.raw
                           if k.lower() not in (b"content-encoding",
                                                b"content-length",
                                                b"transfer-encoding")]
                return httpx.Response(status_code=hop_status,
                                      headers=headers, content=bytes(body),
                                      request=response.request,
                                      extensions={"aw_redirect_hops": hops})
        try:
            next_url = str(httpx.URL(url).join(location))
        except httpx.InvalidURL as exc:
            raise BudgetExhausted(
                f"policy endpoint redirected to an unparsable target "
                f"({exc})") from exc
        hops.append({"url": url, "status": hop_status, "location": location})
        url = next_url
        if url in seen:
            raise BudgetExhausted(f"policy endpoint redirect loop at {url}")
        seen.add(url)
    raise BudgetExhausted(
        f"policy endpoint exceeded {MAX_REDIRECTS} redirects")


class _CurlResp:
    """Minimal response shim for the curl fallback (status_code + text)."""

    def __init__(self, status_code: int, text: str):
        self.status_code, self.text = status_code, text


def _curl_get(url: str, timeout: float | None = None):
    """Fallback GET via curl subprocess (task 177): several hosts —
    clinicaltrials.gov, www.fda.gov — TLS-fingerprint-403 python httpx/requests
    while serving the SAME honest UA a 200 via plain curl. A 403 produced by
    that artifact is NOT an access-control signal, and treating it as one
    false-declines open public-domain hosts. Semantics preserved: if curl
    ALSO gets 401/403, the host stays access-controlled/declined."""
    import subprocess
    try:
        bounded = TIMEOUT if timeout is None else max(1.0, min(TIMEOUT, timeout))
        out = subprocess.run(
            ["curl", "-sS", "-A", UA_STR, "-L", "--max-time", str(int(bounded)),
             "-w", "\n__AW_STATUS__ %{http_code}", url],
            capture_output=True, text=True, timeout=bounded + 2)
        body, _, meta = out.stdout.rpartition("\n__AW_STATUS__ ")
        return _CurlResp(int(meta.strip() or 0), body)
    except Exception:  # noqa: BLE001
        return None


def _get(client: httpx.Client, url: str, *, deadline: float | None = None):
    """GET with one retry on transient failure. Returns (response|None, err|None).
    On 401/403 or after both transient HTTPX attempts fail, re-checks once via
    curl with the same UA. A definitive curl status replaces the Python
    transport result; two independent transient failures remain unknown.

    ``deadline`` is a ``time.monotonic()`` instant covering everything below —
    both attempts, the pause between them, and the curl fallback. A caller that
    supplies none gets ``POLICY_ENDPOINT_BUDGET``; there is deliberately no
    unbounded mode any more (task 485). Running out of budget returns "no
    answer", which is the same held_unknown a reset or a timeout has always
    produced — the bound changes how long we wait, never what we conclude."""
    if deadline is None:
        deadline = time.monotonic() + POLICY_ENDPOINT_BUDGET
    last = None
    for attempt in range(2):
        if deadline - time.monotonic() <= 0:
            return None, last or BUDGET_EXHAUSTED
        try:
            r = _fetch_bounded(client, url, deadline)
            if r.status_code >= 500:
                last = f"HTTP {r.status_code}"
            elif r.status_code in (401, 403):
                cr = _curl_get(url, timeout=deadline - time.monotonic())
                if cr is not None and 0 < cr.status_code < 500:
                    return cr, None
                return r, None
            else:
                return r, None
        except BudgetExhausted as e:
            last = str(e)
        except httpx.HTTPError as e:
            last = f"{type(e).__name__}"
        sleep_for = min(0.5 * (attempt + 1),
                        max(0.0, deadline - time.monotonic()))
        if sleep_for:
            time.sleep(sleep_for)
    # TLS fingerprints, HTTP/2 negotiation, and proxy behavior can make a
    # reachable policy endpoint fail only through HTTPX.  The same-UA curl
    # fallback is a transport retry, not a weaker consent rule: only a
    # definitive <500 response resolves the probe.
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None, last or BUDGET_EXHAUSTED
    cr = _curl_get(url, timeout=remaining)
    if cr is not None and 0 < cr.status_code < 500:
        return cr, None
    return None, last


def _get_policy_endpoint(client: httpx.Client, host: str, path: str,
                         *, origin_scheme: str | None = None,
                         deadline: float | None = None):
    """Fetch a host-scoped policy endpoint with the same bounded transport
    fallback used for robots.txt.

    A number of small sites serve a usable HTTP policy endpoint while their
    HTTPS virtual host has an expired/mismatched certificate or resets before
    returning a response.  Treating that transport failure as an opt-in would
    be wrong; this helper merely tries the other scheme.  A 5xx, timeout, or
    connection error on *both* schemes remains ``unknown`` and therefore held.
    The caller records the scheme, so the terminal receipt is not ambiguous
    about which policy representation was evaluated.

    One ``POLICY_ENDPOINT_BUDGET`` covers this whole call, both schemes
    included, so trying the sibling scheme cannot buy a slow host a second full
    budget; the caller's ``deadline`` (the per-host budget) caps it further.
    """
    own = time.monotonic() + POLICY_ENDPOINT_BUDGET
    endpoint_deadline = own if deadline is None else min(own, deadline)

    def get(url: str):
        return _get(client, url, deadline=endpoint_deadline)
    if origin_scheme is not None:
        if origin_scheme not in {"http", "https"}:
            raise ValueError(f"unsupported consent origin scheme: {origin_scheme!r}")
        response, error = get(f"{origin_scheme}://{host}{path}")
        return response, error, origin_scheme if response is not None else None

    https_url = f"https://{host}{path}"
    response, error = get(https_url)
    if response is not None:
        return response, error, "https"
    http_url = f"http://{host}{path}"
    response, http_error = get(http_url)
    if response is not None:
        return response, http_error, "http"
    return None, http_error or error, None


def _parse_content_usage(body: str) -> dict:
    """Best-effort AIPREF Content-Usage parse (draft, no RFC — ADR-009).
    Returns {key: 'y'|'n'} over keys like search / ai / train-ai / all."""
    prefs: dict[str, str] = {}
    for m in _CONTENT_USAGE.finditer(body):
        for part in m.group(1).split(","):
            if "=" in part:
                k, _, v = part.partition("=")
                prefs[k.strip().lower()] = v.strip().lower()[:1]
    return prefs


def _robots_signals(raw_body: str) -> dict:
    """Extract all robots-derived signals from a robots.txt body.

    The body is sanitised first (task 466): a BOM or a CMS banner glued to the
    front of line 1 used to hide the publisher's whole ``User-agent`` group
    from every parser here — ``ua_tokens_seen`` came back empty and the group's
    own Disallow rules never bound us.
    """
    body, sanitation = sanitize_robots_body(raw_body)
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(body.splitlines())

    def allowed_root(agent: str) -> bool:
        try:
            return rp.can_fetch(agent, "/")
        except Exception:  # noqa: BLE001 — malformed robots → conservative
            return False

    tokens = {t.lower() for t in _UA_LINE.findall(body)}
    extended = sorted(t for t in tokens if t.endswith("-extended"))
    # only count a training token as an explicit opt-out when the file
    # actually addresses it (a "* Disallow:/" already flips search_ok anyway)
    train_disallowed = sorted(
        t for t in set(list(TRAIN_TOKENS) + extended) & tokens
        if not allowed_root(t))
    cs = _content_signals(body)
    return {
        "our_agent_root": allowed_root(ROBOT_AGENT),
        "wildcard_root": allowed_root("*"),
        "ccbot_root": allowed_root("CCBot"),
        "train_tokens_disallowed": train_disallowed,
        "noai": bool(_NOAI.search(body)),
        "tdm_reservation_in_robots": bool(_TDM_ROBOTS.search(body)),
        "content_usage": _parse_content_usage(body),
        "content_signal": cs,
        "content_signal_search_grant": cs.get("search") == "yes",
        "ua_tokens_seen": sorted(tokens)[:40],
        "body_shape": robots_body_shape(raw_body),
        "lines_recovered": sanitation["recovered_lines"],
    }


def _tdmrep_reserved(payload) -> bool:
    """Any tdm-reservation:1 entry ⇒ reserved (conservative: we don't try to
    prove our URLs fall outside a partial location scope)."""
    try:
        entries = payload if isinstance(payload, list) else \
            payload.get("policies", payload.get("rules", [payload]))
        for e in entries:
            if isinstance(e, dict) and str(e.get("tdm-reservation")) == "1":
                return True
    except Exception:  # noqa: BLE001
        return False
    return False


def apply_robots_flags(rec: dict, s: dict, direct_crawl: bool = False) -> dict:
    """Apply the RUBRIC above to one parsed robots signal set, in place.

    Extracted verbatim from ``resolve_host`` (task 466) so the same rules can
    be re-derived OFFLINE from a sealed receipt's stored body. One copy of
    the rubric, one place to change it.

    ``direct_crawl`` is kept because callers pass it and it still selects the
    receipt's ``mode`` / cache key, but since ADR-024 amendment 5 (Tim,
    2026-09-03) it no longer changes any flag this function sets: a CCBot
    disallow is a training signal in both modes. It is accepted here so the
    call sites do not have to change and so the parameter can go on carrying
    provenance if a future rule needs it.
    """
    if not s["wildcard_root"]:
        rec["search_ok"] = rec["train_ok"] = False
        rec["reasons"].append("robots: User-agent:* disallows / (full opt-out)")
    if not s["our_agent_root"]:
        rec["search_ok"] = rec["train_ok"] = False
        rec["reasons"].append(f"robots: our token '{ROBOT_AGENT}' disallowed on /")
    if not s["ccbot_root"]:
        # ADR-024 amendment 5 (Tim, 2026-09-03): "A cc-bot decline should mean
        # no_train, not no_search." Common Crawl is the canonical LLM-training
        # corpus, so refusing CCBot targets training data, not search indexing
        # — the reading amendment 2 already applied to every other AI-bot
        # token. Unconditional on `direct_crawl`: the provenance mode does not
        # change what the publisher asked for. The old CC-mode branch set both
        # flags False and was the single largest source of declines (69% of
        # all declines, ADR-073).
        rec["train_ok"] = False
        rec["reasons"].append(
            "robots: CCBot disallowed → train_ok=False only (Tim 2026-09-03, "
            "ADR-024 am.5); search_ok unaffected")
    if s["tdm_reservation_in_robots"]:
        rec["search_ok"] = rec["train_ok"] = False
        rec["reasons"].append("robots: tdm-reservation: 1 (TDMRep hard opt-out)")
    if s["noai"]:
        rec["search_ok"] = rec["train_ok"] = False
        rec["reasons"].append("robots: noai marker (hard opt-out per fetch_hn_cc precedent)")
    if s["train_tokens_disallowed"]:
        rec["train_ok"] = False
        rec["reasons"].append(
            "robots: training-only tokens disallowed: "
            + ",".join(s["train_tokens_disallowed"])
            + " (train_ok=False; excluded from training use/export)")
    cu = s["content_usage"]
    if cu.get("all") == "n":
        rec["search_ok"] = rec["train_ok"] = False
        rec["reasons"].append("AIPREF Content-Usage: all=n")
    if cu.get("search") == "n":
        rec["search_ok"] = False
        rec["reasons"].append("AIPREF Content-Usage: search=n")
    if any(cu.get(k) == "n" for k in ("ai", "train-ai", "genai")):
        rec["train_ok"] = False
        rec["reasons"].append("AIPREF Content-Usage: ai/train-ai=n")
    cs = s["content_signal"]
    if cs.get("search") == "no":
        rec["search_ok"] = False
        rec["reasons"].append("Content-Signal: search=no")
    if cs.get("ai-train") == "no":
        rec["train_ok"] = False
        rec["reasons"].append(
            "Content-Signal: ai-train=no (train_ok=False; excluded from "
            "any training-data use/export)")
    if s["content_signal_search_grant"]:
        rec["reasons"].append(
            "Content-Signal: search=yes — EXPLICIT grant for building a "
            "search index and returning hyperlinks + short excerpts")
    return rec


def resolve_host(client: httpx.Client, host: str,
                 direct_crawl: bool = False,
                 origin_scheme: str | None = None,
                 deadline: float | None = None) -> dict:
    """Resolve the two consent flags for one host. Never raises, never waits
    without end: both probes share one ``HOST_BUDGET`` wall clock (task 485).

    ``direct_crawl=True`` (task 187) records that the wave's copy comes from
    OUR OWN polite fetch under our own UA token, not from Common Crawl. It
    sets ``mode`` and the cache key (``<host>@direct.json``) so a receipt
    always describes the fetch we actually made. Since ADR-024 amendment 5
    (Tim, 2026-09-03) it no longer changes any FLAG: a CCBot disallow stamps
    ``train_ok=False`` and leaves ``search_ok`` alone in both modes. All
    other signals (wildcard/our-token/TDM/noai/training-tokens/
    Content-Signal) apply unchanged and always did."""
    if deadline is None:
        deadline = time.monotonic() + HOST_BUDGET
    rec = {"host": host, "status": "ok", "search_ok": True, "train_ok": True,
           "mode": "direct" if direct_crawl else "cc",
           "reasons": [], "signals": {}, "checked_at": _now_iso()}
    if origin_scheme is not None:
        rec["origin_scheme"] = origin_scheme
        rec["origin"] = f"{origin_scheme}://{host}"
    robots_body = None

    # --- probe 1: robots.txt --------------------------------------------------
    r, err, robots_transport = _get_policy_endpoint(
        client, host, "/robots.txt", origin_scheme=origin_scheme,
        deadline=deadline)
    if r is None:
        rec.update(status="unknown")
        rec["reasons"].append(f"robots.txt unreachable ({err}) — held, retryable")
        # `unreadable_class` was already recorded on the tdmrep leg just below
        # (task 466) but never on this one -- an asymmetry that hid exactly
        # the case task 542 fixes: a deterministic malformed-redirect target
        # on the robots.txt leg looked identical, on this receipt, to a
        # genuine timeout. One classifier, both legs (engineering-doctrine 1).
        rec["signals"]["robots"] = {"fetched": False, "error": err,
                                    "unreadable_class": classify_unreadable(err)}
        return rec

    sig: dict = {"fetched": True, "http_status": r.status_code,
                 "transport": robots_transport}
    # `_fetch_bounded` returns the redirect chain it actually walked (task 502
    # defect 2), on ANY terminal shape -- not only the stalled-redirect one --
    # so a reader can always see how many hops preceded the answer.
    hops = (getattr(r, "extensions", None) or {}).get("aw_redirect_hops")
    if hops:
        sig["hops"] = hops
    if r.status_code in (401, 403):
        rec["search_ok"] = rec["train_ok"] = False
        rec["reasons"].append(f"robots.txt HTTP {r.status_code} (access-controlled)")
    elif r.status_code == 200 and r.text.strip():
        robots_body = r.text[: ROBOTS_BODY_CAP * 4]
        s = _robots_signals(robots_body)
        sig.update(s)
        apply_robots_flags(rec, s, direct_crawl=direct_crawl)
    elif 300 <= r.status_code < 400:
        # `_fetch_bounded` only ever returns a 3xx here when the terminal hop
        # carried no usable Location -- there is nothing left to follow, and
        # nothing was ever read as a policy (task 502 defect 2: two live
        # examples, gc-handball.ch/germany-tourism.de, recorded exactly this
        # shape). Named distinctly from a genuine no-policy answer so a reader
        # does not mistake a stalled redirect for an open host; the URL-level
        # gate (`direct_url_allowed`) already fails closed on it regardless.
        sig["note"] = "redirect terminal carried no further Location (unreadable, not open)"
    else:
        # 404/410/other 4xx or empty 200 → no robots published → open
        sig["note"] = "no robots rules published (open)"
    rec["signals"]["robots"] = sig

    # --- probe 2: /.well-known/tdmrep.json -------------------------------------
    time.sleep(0.3)
    t, terr, tdmrep_transport = _get_policy_endpoint(
        client, host, "/.well-known/tdmrep.json", origin_scheme=origin_scheme,
        deadline=deadline)
    tsig: dict = {}
    if t is None:
        # transient failure on the TDMRep probe: a reservation there would be a
        # hard opt-out, so an unreadable answer leaves consent UNRESOLVED.
        rec.update(status="unknown")
        rec["reasons"].append(f"tdmrep.json unreachable ({terr}) — held, retryable")
        tsig = {"fetched": False, "error": terr,
                "unreadable_class": classify_unreadable(terr)}
    else:
        tsig = {"fetched": True, "http_status": t.status_code,
                "transport": tdmrep_transport}
        thops = (getattr(t, "extensions", None) or {}).get("aw_redirect_hops")
        if thops:
            tsig["hops"] = thops
        if t.status_code == 200:
            try:
                payload = t.json()
            except Exception:  # noqa: BLE001 — SPA catch-all HTML etc.
                payload = None
                tsig["note"] = "200 but not JSON (ignored)"
            if payload is not None and _tdmrep_reserved(payload):
                rec["search_ok"] = rec["train_ok"] = False
                rec["reasons"].append(
                    ".well-known/tdmrep.json: tdm-reservation=1 (hard opt-out)")
                tsig["reservation"] = 1
    rec["signals"]["tdmrep"] = tsig

    # ADR-024 amendment 3: promote the unresolved-but-registered class to
    # search-only. Applied HERE, at the single point where consent is decided,
    # so every caller inherits one rule rather than re-implementing it.
    if unresolved_search_ok(rec):
        rec.update(status="ok", search_ok=True, train_ok=False)
        rec["reasons"].append(
            "ADR-024 amdt-3 (ratified 2026-08-19): tdmrep.json gave no readable "
            "policy answer while robots is readable and permissive; registered in "
            "data/consent-unresolved-register.yaml with a sealed witness — admitted "
            "SEARCH ONLY, train_ok=false (never in any training dump/export)")

    if rec["status"] == "ok" and rec["search_ok"] and rec["train_ok"]:
        rec["reasons"].append("no opt-out signal found")
    rec["_robots_body"] = (robots_body or "")[:ROBOTS_BODY_CAP]
    return rec


# --- cache -------------------------------------------------------------------

def _cache_path(cache_dir: str, host: str,
                direct_crawl: bool = False,
                origin_scheme: str | None = None) -> pathlib.Path:
    safe = host.lower().strip().strip(".")
    # direct-crawl records live under a distinct key so a receipt always names
    # the provenance of the copy it was resolved for; a direct record is never
    # served to a CC-provenance gate (and vice versa). Until ADR-024
    # amendment 5 the two modes also produced DIFFERENT FLAGS on a CCBot
    # disallow; they no longer do, and the split key is now provenance
    # bookkeeping alone.
    if origin_scheme is not None:
        if origin_scheme not in {"http", "https"}:
            raise ValueError(f"unsupported consent origin scheme: {origin_scheme!r}")
        mode = "direct" if direct_crawl else "cc"
        return pathlib.Path(cache_dir) / f"{safe}@{origin_scheme}@{mode}.json"
    return pathlib.Path(cache_dir) / (f"{safe}@direct.json" if direct_crawl
                                      else f"{safe}.json")


def _load_cached(cache_dir: str, host: str, max_age_days: int,
                 direct_crawl: bool = False,
                 origin_scheme: str | None = None):
    p = _cache_path(cache_dir, host, direct_crawl, origin_scheme)
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None
    if rec.get("status") == "unknown":
        return None                      # retryable → re-resolve
    try:
        age = (datetime.now(timezone.utc)
               - datetime.strptime(rec["checked_at"], "%Y-%m-%dT%H:%M:%SZ")
               .replace(tzinfo=timezone.utc)).days
    except Exception:  # noqa: BLE001
        return None
    return rec if age <= max_age_days else None


def _save_cache(cache_dir: str, rec: dict):
    p = _cache_path(cache_dir, rec["host"],
                    direct_crawl=rec.get("mode") == "direct",
                    origin_scheme=rec.get("origin_scheme"))
    p.parent.mkdir(parents=True, exist_ok=True)
    # A capture wave can have multiple worker processes resolving different
    # hosts into the shared cache.  A fixed ``.tmp`` name makes those otherwise
    # independent writes race and can publish the wrong host's receipt.
    tmp = p.with_name(f".{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "w") as f:
        json.dump(rec, f, ensure_ascii=False)
    os.replace(tmp, p)


def resolve_cached_host(client: httpx.Client, host: str, *,
                        cache_dir: str = CACHE_DIR,
                        max_age_days: int = MAX_AGE_DAYS,
                        direct_crawl: bool = False,
                        origin_scheme: str | None = None) -> dict:
    """Resolve one host through the same persistent cache as batch callers.

    Consumers need the cached robots body for their later URL-specific gate,
    so this intentionally returns the full receipt (unlike ``resolve_hosts``).
    Unknown receipts are never reused: ``_load_cached`` deliberately returns
    ``None`` for them and the host is resolved again.
    """
    host = host.lower().strip().strip(".")
    cached = _load_cached(cache_dir, host, max_age_days,
                          direct_crawl=direct_crawl,
                          origin_scheme=origin_scheme)
    if cached is not None:
        return cached
    receipt = resolve_host(client, host, direct_crawl=direct_crawl,
                           origin_scheme=origin_scheme)
    _save_cache(cache_dir, receipt)
    return receipt


def robots_unreadable_class(receipt: dict) -> str:
    """Name why a robots.txt behind a search-declined verdict was never
    actually READ as a policy, or ``"readable"`` when it was.

    Task 502 defect 1/ADR-103: ``resolve_host``/``apply_robots_flags`` can flip
    ``search_ok`` to False on evidence that was never a publisher's readable
    policy -- an access-controlled probe (401/403), or a redirect chain that
    ended without a Location to follow (defect 2, ``"redirect terminal carried
    no further Location"`` above). Those are ADR-073's retry posture (a second
    vantage + the reason-tagged retry queue), not a decline: the publisher said
    nothing we could read. A genuine ``User-agent:*``/our-token/TDM/``noai``
    disallow, parsed from an actually-fetched body, stays a decline.

    THE single place this distinction is made, so ``direct_terminal_contract``
    and any future census both read it the same way instead of re-deriving it
    from ``reasons`` strings per script (the task-502 anatomy classifier did
    exactly that, by hand, over 7,925 rows).
    """
    sig = (receipt.get("signals") or {}).get("robots") or {}
    if not sig.get("fetched"):
        return "unfetched"                     # no answer at all; not a verdict
    status = sig.get("http_status")
    if status in (401, 403):
        return "access_controlled"
    if receipt.get("_robots_body"):
        return "readable"                      # a body was actually parsed
    if status in (200, 404, 410):
        return "readable"                      # explicit, definitive no-policy answer
    if isinstance(status, int) and 300 <= status < 400:
        return "redirect_no_location"
    return "unreadable_other"


def robots_unreadable(receipt: dict) -> bool:
    """True unless the robots.txt evidence behind a verdict was readable."""
    return robots_unreadable_class(receipt) != "readable"


def unresolved_terminal_reason(receipt: dict) -> str:
    """Name why a ``status="unknown"`` receipt never resolved.

    ``"consent_unreadable"`` for a deterministic, permanently-broken policy-
    endpoint redirect on either leg (``classify_unreadable`` ==
    ``"malformed_redirect_target"``) -- the origin's own redirect chain is
    broken and a retry cannot fix it, so this is not a "come back later"
    posture the way a timeout or a reset is. ``"consent_unresolved"`` for
    everything else: a genuine instrument failure that a later, witnessed
    probe can still resolve (ADR-024 amendment 3, task 466: "definitively
    unreadable" means a 5xx and nothing else).

    THE single place ``direct_terminal_contract.consent_terminal`` (and any
    future census) makes this distinction, mirroring ``robots_unreadable``'s
    role for the ``status="ok"``/``search_ok=False`` case. Does not change,
    and cannot change, whether the host is admitted or declined -- a
    ``status="unknown"`` receipt stays HELD under either reason; amendment
    3's 5xx-only promotion boundary lives entirely in
    ``unresolved_search_ok`` and is untouched here.
    """
    for leg in ("robots", "tdmrep"):
        sig = (receipt.get("signals") or {}).get(leg) or {}
        if sig.get("unreadable_class") == "malformed_redirect_target":
            return "consent_unreadable"
    return "consent_unresolved"


def direct_url_allowed(receipt: dict, url: str) -> bool:
    """Apply a resolved direct-mode receipt to one exact URL path.

    Host-level consent is necessary but not sufficient for an origin fetch:
    a permissive root can still publish a path-specific ``Disallow``.  This is
    the shared mirror for direct acquisition callers; it deliberately checks
    both our declared token and the wildcard group against the exact URL.
    Missing, cross-host, non-direct, unresolved, or search-declined evidence
    fails closed.
    """
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold().strip(".")
    try:
        port = parsed.port
    except ValueError:
        return False
    display_host = f"[{host}]" if ":" in host else host
    authority = display_host if port is None else f"{display_host}:{port}"
    if (parsed.scheme.casefold() not in {"http", "https"} or not host
            or authority != str(receipt.get("host") or "").casefold().strip(".")
            or (receipt.get("origin_scheme") is not None
                and receipt.get("origin_scheme") != parsed.scheme.casefold())
            or receipt.get("mode") != "direct"
            or receipt.get("status") != "ok"
            or receipt.get("search_ok") is not True):
        return False
    body = receipt.get("_robots_body") or ""
    if not body:
        signal = (receipt.get("signals") or {}).get("robots") or {}
        # An empty body is permissive only when the endpoint gave one of the
        # policy's explicit readable/no-policy answers.  In particular, a
        # rate-limit page (429) or an arbitrary 4xx must never become consent
        # merely because it carried no robots rules.
        return (signal.get("fetched") is True
                and signal.get("http_status") in (200, 404, 410))
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(str(body).splitlines())
    try:
        return (parser.can_fetch(ROBOT_AGENT, url)
                and parser.can_fetch("*", url))
    except Exception:  # noqa: BLE001 -- malformed policy fails closed
        return False


def _batch_budget(host_count: int, concurrency: int) -> float:
    """Wall clock after which a batch stops waiting for its stragglers.

    Every worker is already bounded by ``HOST_BUDGET``, so this is the length
    of the worst honest batch plus slack — deliberately loose, because firing
    it means something we did not foresee, and a backstop that fires early
    would turn slow-but-working hosts into held_unknown.
    """
    per_worker = math.ceil(host_count / max(1, concurrency))
    return (HOST_BUDGET + HOST_DELAY + 5.0) * per_worker + 60.0


def resolve_hosts(hosts, cache_dir: str = CACHE_DIR,
                  max_age_days: int = MAX_AGE_DAYS,
                  concurrency: int = CONCURRENCY,
                  progress: bool = True,
                  direct_crawl: bool = False,
                  origin_scheme: str | None = None) -> dict[str, dict]:
    """Resolve consent for many hosts (cached, polite). Returns {host: record}.
    The cached record includes `_robots_body`; the returned records do NOT
    (stripped — use the cache file for per-URL re-evaluation)."""
    hosts = sorted({h.lower().strip().strip(".") for h in hosts if h and "." in h})
    out: dict[str, dict] = {}
    todo = []
    for h in hosts:
        c = _load_cached(cache_dir, h, max_age_days,
                         direct_crawl=direct_crawl,
                         origin_scheme=origin_scheme)
        if c is not None:
            c.pop("_robots_body", None)
            c["from_cache"] = True
            out[h] = c
        else:
            todo.append(h)
    if progress:
        print(f"consent: {len(hosts)} hosts | cached {len(out)} | resolving {len(todo)}",
              flush=True)
    if not todo:
        return out

    lock = threading.Lock()
    client = _client()

    def held(host: str, reason: str) -> dict:
        rec = {"host": host, "status": "unknown", "search_ok": True,
               "train_ok": True, "reasons": [reason],
               "mode": "direct" if direct_crawl else "cc",
               "signals": {}, "checked_at": _now_iso()}
        if origin_scheme is not None:
            rec["origin_scheme"] = origin_scheme
            rec["origin"] = f"{origin_scheme}://{host}"
        return rec

    def work(host):
        time.sleep(HOST_DELAY)           # ~0.5 s/host pacing per worker
        try:
            rec = resolve_host(client, host, direct_crawl=direct_crawl,
                               origin_scheme=origin_scheme)
        except Exception as e:  # noqa: BLE001 — never crash the batch
            rec = held(host, f"resolver crash: {e!r}")
        with lock:
            _save_cache(cache_dir, rec)
        slim = dict(rec)
        slim.pop("_robots_body", None)
        return slim

    # Backstop, not the mechanism: resolve_host is bounded by HOST_BUDGET, so a
    # worker cannot outlive it. What this guards is the failure mode task 485
    # was called in to explain — one host holding as_completed() open while the
    # batch records nothing and the unit still reports ACTIVE. If it ever fires,
    # the hosts we never heard back about are held_unknown like any other
    # non-answer, with a receipt saying so.
    batch_budget = _batch_budget(len(todo), concurrency)
    ex = ThreadPoolExecutor(max_workers=max(1, concurrency))
    timed_out = False
    try:
        futs = {ex.submit(work, h): h for h in todo}
        try:
            for i, fut in enumerate(as_completed(futs, timeout=batch_budget), 1):
                rec = fut.result()
                out[rec["host"]] = rec
                if progress:
                    verdict = rec["status"] if rec["status"] != "ok" else \
                        ("ok" if rec["search_ok"] and rec["train_ok"] else "declined")
                    print(f"  [{i}/{len(todo)}] {rec['host']}: {verdict}"
                          f" ({'; '.join(rec['reasons'][:2])})", flush=True)
        except FuturesTimeout:
            timed_out = True
            for fut, host in futs.items():
                if host in out:
                    continue
                fut.cancel()
                rec = held(host, f"resolver did not answer within the "
                                 f"{batch_budget:.0f}s batch budget — held, retryable")
                with lock:
                    _save_cache(cache_dir, rec)
                out[host] = rec
                if progress:
                    print(f"  [--/{len(todo)}] {host}: unknown "
                          f"(batch budget exhausted)", flush=True)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
        if not timed_out:
            # A worker that outlived the batch budget still holds this client;
            # closing it under that thread would turn a bounded stall into an
            # exception storm. The process is short-lived, so leaking it once is
            # the cheaper honesty.
            client.close()
    return out


# Claude-family tokens: a block on ANY of these is a decline, never
# unclear-class (ADR-024 amendment 2026-07-13 — "a wildcard AI block /
# ClaudeBot block / tdm-reservation still = decline, unchanged").
CLAUDE_TOKENS = ("anthropic-ai", "claudebot", "claude-web")


def unclear_search_tokens(rec: dict) -> list[str]:
    """The vendor-specific training-token blocks behind a host's
    train_ok=False verdict -- the one list `unclear_search_ok` and the
    consent-unclear register writer (`consent_gate.py`) both read, so they
    can never drift apart on what counts as a token (engineering-doctrine 1,
    one truth one home).

    Includes ``"ccbot"`` when ``_robots_signals``' ``ccbot_root`` is False.
    ADR-024 amendment 2 named CCBot in the same breath as gptbot/
    google-extended -- a vendor-specific AI-training block that should log
    the host to the unclear register with its token -- but amendment 5 (Tim,
    2026-09-03, PR #86) gave CCBot its OWN flag-setting branch in
    ``apply_robots_flags`` (the ``ccbot_root`` check below), which never
    touches ``train_tokens_disallowed``. A CCBot-only disallow therefore
    produced ``train_ok=False`` with an EMPTY token list, which
    `unclear_search_ok` read as "train_ok flipped by something else
    entirely" and never registered — leaving amendment 2's own request
    unmet for exactly the token amendment 5 was about. decisions.md's
    amendment-5 write-up named this and left it "open for Tim"; this
    resolves it, narrowly: CCBot counts as a token for THIS test only. It
    changes nothing about which flags a CCBot disallow sets — that stays
    amendment 5's train_ok=False-only rule, untouched.
    """
    rb = rec.get("signals", {}).get("robots", {}) or {}
    toks = list(rb.get("train_tokens_disallowed") or [])
    if rb.get("ccbot_root") is False and "ccbot" not in toks:
        toks.append("ccbot")
    return toks


def unclear_search_ok(rec: dict) -> bool:
    """ADR-024 amendment (Tim, 2026-07-13) 'unclear consent' class: robots
    blocks ONLY vendor-specific AI-TRAINING tokens (google-extended, gptbot,
    ccbot, …), none of them Claude-family, with no wildcard/general AI
    block, no tdm/noai, and no explicit AI/search signal either way. Such
    hosts are included FOR SEARCH (train_ok=False stamped) and MUST be
    tracked in data/consent-unclear-register.yaml (the gate appends, never
    silently)."""
    if rec.get("status") != "ok":
        return False    # unknown/held hosts are never admitted here
    if not rec.get("search_ok") or rec.get("train_ok"):
        return False    # search_ok=False = hard decline; both-ok needs nothing
    toks = unclear_search_tokens(rec)
    if not toks:
        return False    # train_ok flipped by something else entirely
    if any(t in CLAUDE_TOKENS for t in toks):
        return False    # Claude-family block = decline, unchanged
    rb = rec.get("signals", {}).get("robots", {})
    cu = rb.get("content_usage") or {}
    if any(cu.get(k) == "n" for k in ("ai", "train-ai", "genai")):
        return False    # explicit AIPREF opt-out — not "unclear"
    cs = rb.get("content_signal") or {}
    if cs.get("ai-train") == "no":
        return False    # explicit Content-Signal opt-out — not "unclear"
    return True


_UNRESOLVED_REGISTER_CACHE: dict = {}


def load_unresolved_register(path: str | None = None) -> dict:
    """Load the ADR-024 amendment-3 consent-UNRESOLVED register (host -> entry).

    Absent file = feature off (empty mapping). A present but unparseable or
    malformed register is a LOUD failure: silently resolving it to "no hosts"
    would turn a policy artifact into a no-op and admit nothing while looking
    healthy, which is exactly the silent-fallback class this project bans.
    """
    path = path or CONSENT_UNRESOLVED_REGISTER
    cached = _UNRESOLVED_REGISTER_CACHE.get(path)
    if cached is not None:
        return cached
    if not os.path.exists(path):
        _UNRESOLVED_REGISTER_CACHE[path] = {}
        return {}
    import yaml  # lazy: keeps yaml off the hot import path
    with open(path) as handle:
        doc = yaml.safe_load(handle) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"consent-unresolved register malformed (not a mapping): {path}")
    out: dict = {}
    for entry in doc.get("hosts") or []:
        if not isinstance(entry, dict) or not entry.get("host"):
            raise ValueError(f"consent-unresolved register entry lacks host: {path}")
        # The ratified guardrail is not optional: no witness, no admission.
        for key in ("witness", "witness_sha256", "egress_liveness",
                    "egress_liveness_sha256"):
            if not entry.get(key):
                raise ValueError(
                    f"consent-unresolved register: {entry['host']} lacks {key} — "
                    "ADR-024 amdt-3 admission requires a sealed per-host witness "
                    "carrying both probe statuses and that attempt's egress proof")
        if entry.get("train_ok") is not False:
            raise ValueError(
                f"consent-unresolved register: {entry['host']} must stamp train_ok=false")
        out[entry["host"]] = entry
    _UNRESOLVED_REGISTER_CACHE[path] = out
    return out


_HTTP_ERROR = re.compile(r"^HTTP (\d{3})$")


_MALFORMED_REDIRECT_PREFIX = "policy endpoint redirected to"


def classify_unreadable(error: str | None) -> str:
    """Name the terminal class behind an unreadable policy probe.

    ``_get`` returns either ``"HTTP <status>"`` (the origin answered, badly),
    an httpx exception class name (nothing answered), or the budget message.
    The distinction is the whole of ADR-024 amendment 3: only a 5xx is the
    ORIGIN malfunctioning on an absent path. A timeout or a reset is OUR
    instrument failing, and an instrument failure is not a publisher verdict.
    """
    match = _HTTP_ERROR.match(str(error or "").strip())
    if match is not None:
        status = int(match.group(1))
        return "http_5xx" if 500 <= status <= 599 else f"http_{status // 100}xx"
    if error is None or not str(error).strip():
        return "unrecorded"
    if str(error).strip().startswith(_MALFORMED_REDIRECT_PREFIX):
        # `_fetch_bounded`'s redirect loop raises exactly this prefix for a
        # Location it will NEVER be able to follow -- an unparsable target
        # (``httpx.InvalidURL``, e.g. a malformed port), a redirect to an
        # unsupported scheme, or a redirect to a non-public address. All
        # three are BudgetExhausted subclasses today (folded in so they can
        # never crash the resolver -- task 502 defect 2/PR #88), but unlike a
        # timeout or a reset they are DETERMINISTIC: the origin will answer
        # with the identical broken redirect every time, so `_get`'s
        # retry+sleep loop before this classification runs is pure wasted
        # budget (task 509's deferred note names this exactly). Named
        # separately so a caller can tell "this policy endpoint will never
        # resolve on its own" from "ask again later" -- it is still not a
        # 5xx, so ADR-024 amendment 3's 5xx-only promotion boundary
        # (`unresolved_search_ok`) is completely unaffected: this class never
        # promotes a host toward admission, it only names a HELD reason.
        return "malformed_redirect_target"
    if str(error).strip().startswith(BUDGET_EXHAUSTED):
        # The budget messages say WHERE the clock ran out (a redirect hop, the
        # body read); the class stays one thing, because "we ran out of time"
        # is one instrument failure however it is phrased.
        return "budget_exhausted"
    return "transport_error"


def tdmrep_unreadable_class(rec: dict) -> str:
    """The terminal class of a record's TDMRep leg: ``readable`` or a class.

    Prefers the structured ``unreadable_class`` recorded since task 466; falls
    back to the stored error string for receipts sealed before it existed —
    reading a historical field, not guessing a missing one.
    """
    tdmrep = (rec.get("signals") or {}).get("tdmrep") or {}
    if tdmrep.get("fetched"):
        return "readable"
    recorded = tdmrep.get("unreadable_class")
    if isinstance(recorded, str) and recorded:
        return recorded
    return classify_unreadable(tdmrep.get("error"))


def explicit_search_refusal(rec: dict) -> bool:
    """True when the host refused us FOR SEARCH, whatever the other leg did.

    A record can carry an explicit refusal and still be ``status="unknown"``
    because the second probe never answered. Reporting that as "held, unknown"
    hides a decision the publisher actually made — the euipo case (403 on both
    probes), which ADR-024 amendment 3 classes as a decline.

    A training-only refusal (``train_ok=False`` with ``search_ok`` intact) is
    NOT this: that host permits search indexing and is handled by the
    unclear-consent register, not by a decline.
    """
    if rec.get("search_ok") is False:
        return True
    robots = (rec.get("signals") or {}).get("robots") or {}
    return robots.get("http_status") in (401, 403)


def unresolved_search_ok(rec: dict, register: dict | None = None) -> bool:
    """ADR-024 amendment 3 (RATIFIED Tim 2026-08-19, PR #61; option (c)).

    An UNREADABLE policy endpoint is not an ABSENT one — but it is not a
    refusal either. Sub-class (a): robots.txt is readable and permissive, while
    /.well-known/tdmrep.json returns no readable policy answer (a host that
    answers 5xx on an absent well-known path can never resolve, where a 404
    would resolve cleanly as "no reservation" — a livelock, established over
    four zero-progress Wave28 rechecks against proven-live egress).

    The rule is general: it keys off the record's own signals plus registry
    membership. Registry membership is DATA; no hostname belongs in this file.
    Explicit refusals are never overridden.

    Amended by Tim's ruling of 2026-08-28 (task 466, Q3(b)): "definitively
    unreadable" means a **5xx** and nothing else. Any other non-answer — a
    timeout, a connection reset, an exhausted budget — is an instrument
    failure on our side and leaves the record held_unknown, awaiting a
    witnessed probe.
    """
    host = rec.get("host")
    entry = (load_unresolved_register() if register is None else register).get(host)
    if not entry:
        return False
    if rec.get("status") != "unknown":
        return False        # only the held/unresolved class is in scope
    if rec.get("search_ok") is not True:
        return False        # robots already declined us — never overridden
    robots = (rec.get("signals") or {}).get("robots") or {}
    if not robots.get("fetched"):
        return False        # robots itself unreadable => sub-class (b), not (a)
    if robots.get("http_status") not in (200, 404, 410):
        return False        # a status without a readable answer is not permission
    # "Definitively unreadable" is 5xx and nothing else (Tim's ruling
    # 2026-08-28, Q3(b); task 466). The ratified text of amendment 3 names the
    # 5xx livelock — "a host that answers 5xx on an absent well-known path can
    # never resolve" — while this code used to admit ANY non-answer, so a
    # ConnectError or a ReadTimeout could be promoted to search-only. Those are
    # our instrument failing, not the origin malfunctioning: they go back to
    # held_unknown pending a witnessed probe.
    if tdmrep_unreadable_class(rec) != "http_5xx":
        return False
    return True


def rederive_sealed_verdict(receipt: dict) -> dict:
    """Re-read a SEALED receipt's stored robots body with today's parser.

    Offline only — no network, no origin contact. The sealed flags are the
    ratified resolver's verdict and stay authoritative; this re-read can only
    make the verdict STRICTER, never looser, so a parser fix can never
    manufacture consent out of a receipt that was recorded as a refusal. Two
    ways it tightens:

      * a directive group the old parser lost behind a junk prefix now binds;
      * an HTML body served as robots.txt carries an explicit TDMRep
        ``<meta name="tdm-reservation" content="1">`` (value-sensitive: a
        ``content="0"`` says the opposite and is not a refusal).

    Returns ``{search_ok, train_ok, changed, added_reasons, body_shape,
    truncated}``; ``changed`` is True only when this re-read tightened.
    """
    sealed_search = receipt.get("search_ok") is True
    sealed_train = receipt.get("train_ok") is True
    body = receipt.get("_robots_body") or ""
    robots = (receipt.get("signals") or {}).get("robots") or {}
    out = {"search_ok": sealed_search, "train_ok": sealed_train,
           "changed": False, "added_reasons": [],
           "body_shape": robots_body_shape(body),
           "truncated": len(body) >= ROBOTS_BODY_CAP,
           "tdm_meta": tdm_reservation_in_html(body)}
    if not body or not robots.get("fetched") or robots.get("http_status") != 200:
        return out
    probe = {"search_ok": True, "train_ok": True, "reasons": []}
    apply_robots_flags(probe, _robots_signals(body),
                       direct_crawl=receipt.get("mode") == "direct")
    if out["tdm_meta"]["reserved"]:
        probe["search_ok"] = probe["train_ok"] = False
        probe["reasons"].append(
            "robots body is HTML carrying <meta name=\"tdm-reservation\" "
            "content=\"1\"> (W3C TDMRep hard opt-out, value-sensitive)")
    for flag in ("search_ok", "train_ok"):
        if out[flag] and probe[flag] is False:
            out[flag] = False
            out["changed"] = True
    if out["changed"]:
        out["added_reasons"] = list(probe["reasons"])
    return out


def bucket(rec: dict, accept_search_grant: bool = False,
           unclear_ok: bool = False) -> str:
    """ok → mergeable; declined → excluded+card; unknown → held (retryable).
    NOTE: this is the historical strict default used by legacy merge callers.
    Principal collection indexing must call ``consent_gate.gate_bucket`` with
    ``search_indexing=True``: search_ok gates dense/sparse retrieval material,
    while train_ok is retained as a training-data/model/export exclusion.

    EXCEPTION 1 (opt-in, task 187 / orchestrator directive on ohchr.org):
    ``accept_search_grant=True`` returns 'ok_search_only' when the host
    carries an EXPLICIT Cloudflare Content-Signal ``search=yes`` grant with
    ``ai-train=no``. The policy's own definition of *search* is "building a
    search index and providing search results (hyperlinks and short
    excerpts)" — which is precisely AW's retrieval embedding + ≤25-word
    snippet and nothing more. Such rows merge with ``train_ok=False``
    stamped in the payload, and MUST be excluded from any training-data
    use or export.

    EXCEPTION 2 (opt-in, ADR-024 amendment 2026-07-13): ``unclear_ok=True``
    returns 'ok_search_only' for the unclear-consent class (see
    unclear_search_ok) — the CALLER must record the host in
    data/consent-unclear-register.yaml (consent_gate.py does this; never
    enable this flag without the register wiring).

    Default False/False keeps the strict both-flags rule."""
    if rec.get("status") == "unknown":
        return "unknown"
    if rec.get("search_ok") and rec.get("train_ok"):
        return "ok"
    if (accept_search_grant and rec.get("search_ok")
            and not rec.get("train_ok")
            and rec.get("signals", {}).get("robots", {})
                   .get("content_signal_search_grant")):
        return "ok_search_only"
    if unclear_ok and unclear_search_ok(rec):
        return "ok_search_only"
    return "declined"


def search_only_declines(records: dict) -> list[str]:
    """Hosts that `bucket()` calls 'declined' while `search_ok` is still True.

    WHY THIS EXISTS (task 629). `bucket()`'s strict default folds
    "train_ok=False, search_ok=True" into 'declined', because that is the right
    answer for its legacy merge callers. It is the WRONG answer for search
    admission, and the module already says so in `bucket`'s own docstring
    ("Principal collection indexing must call consent_gate.gate_bucket with
    search_indexing=True"). But the CLI summary prints only the bucket counts,
    so a reader sees `declined: 4` for four hosts that are perfectly admissible
    for search under ADR-024 amendment 5 (a CCBot or AI-training-token disallow
    sets train_ok=False ONLY and never touches search_ok — Tim, 2026-09-03).

    That misreading nearly refused three Pool A families in task 629, whose
    origins disallow Bytespider (github.com) and CCBot plus the training-token
    set (rsi.ch, srf.ch, rts.ch). The verdict follows the FLAGS, not the bucket.
    This function names the affected hosts so the summary can say so out loud.
    """
    return sorted(
        host for host, rec in records.items()
        if bucket(rec) == "declined" and rec.get("search_ok"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hosts", default="", help="comma-separated hosts")
    ap.add_argument("--hosts-file", default="", help="one host per line")
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--max-age-days", type=int, default=MAX_AGE_DAYS)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--out", default="", help="write results jsonl here")
    args = ap.parse_args()

    hosts = [h for h in args.hosts.split(",") if h.strip()]
    if args.hosts_file:
        hosts += [l.strip() for l in open(os.path.expanduser(args.hosts_file))
                  if l.strip() and not l.startswith("#")]
    if not hosts:
        ap.error("no hosts given (--hosts or --hosts-file)")

    res = resolve_hosts(hosts, cache_dir=os.path.expanduser(args.cache_dir),
                        max_age_days=args.max_age_days,
                        concurrency=args.concurrency)
    counts = {"ok": 0, "declined": 0, "unknown": 0}
    for rec in res.values():
        counts[bucket(rec)] += 1
    if args.out:
        with open(os.path.expanduser(args.out), "w") as f:
            for h in sorted(res):
                f.write(json.dumps(res[h], ensure_ascii=False) + "\n")
        print(f"wrote {len(res)} records -> {args.out}")
    print(f"CONSENT SUMMARY: {counts} of {len(res)} hosts")
    search_only = search_only_declines(res)
    if search_only:
        # Not a warning about the hosts -- a warning about the label above.
        print(f"WARNING: {len(search_only)} of the {counts['declined']} "
              f"'declined' hosts are TRAIN-ONLY declines and remain "
              f"search_ok=True: {', '.join(search_only)}")
        print("         `bucket()`'s strict default folds train_ok=False into "
              "'declined'. For SEARCH admission read the flags, or call "
              "consent_gate.gate_bucket(..., search_indexing=True). A CCBot or "
              "AI-training-token disallow never removes search_ok "
              "(ADR-024 am.5).")


if __name__ == "__main__":
    sys.exit(main())
