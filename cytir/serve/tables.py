"""Curated tables and the weights fingerprint, split out of `retrieval.py` (S10 step 4, task 1095).

The parasite-domain blocklist, the hub-host and authority-host tables, and `weights_fingerprint`.
Each table loads once at import and fails safe to empty (a bad table never breaks search); the
loaded-or-not state is echoed by `loaded_artifacts` (`GET /config`, spec risk R5). Bodies are
verbatim from `retrieval.py`, which imports every name back. This module must not import
`retrieval`: the learned heads, which stay there until step 5, are passed in.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

from . import settings as _settings
from .geo import _COUNTRY_MAP, _COUNTRY_MAP_PATH, _GAZETTEER, _GAZ_PATH
from .settings import SETTINGS, _weights_path
from .signals import _domain

_S = SETTINGS   # typed config, loaded once at import (S10 step 2, task 1071)

# --------------------------------------------------------------------------- #
#  Parasite-domain suppression (task 125, roadmap #4).
#  A residual junk class sits ABOVE recovered authoritative golds at rank 1-3:
#  scraper/paste/mirror/PR sites that are NEVER the authoritative source for any
#  target-domain query, yet were mis-stamped with high trust_tier (0.75-0.85,
#  several as source_type=official) so trust 2.5*tier + title 2.5 floats them to
#  the top (e.g. ch-law-001: snippet.host + aic.lv above the fedlex gold — 123
#  §6). This is a RANKING demotion, not deletion (ADR-032): the content stays
#  indexed, it just can't outrank a real source. The blocklist is CONFIG-DRIVEN
#  (parasite_domains.json, hot-editable, no code change to add a host) and split
#  into full-host demotions + URL-pattern demotions (pkg.go.dev ?tab=importedby/
#  licenses/versions junk tabs — the bare doc page pkg.go.dev/strings#Split is a
#  real gold and is untouched). Every listed host was verified to NEVER be a
#  gold in the 348-query eval, so the penalty can only demote wrong answers —
#  it is applied STAGE-2 ONLY (unlike 123's cantonal flood, a handful of junk
#  URLs never push a gold below the stage-1 margin-40 pool; the gold stays
#  reachable and the stage-2 penalty reorders it above the parasite). Weight
#  <= 0, default 0.0 = fully inert (byte-identical scoring); rollback = 0.0.
# --------------------------------------------------------------------------- #
def _load_parasites():
    path = _S.runtime.parasite_domains
    try:
        with open(path) as f:
            cfg = json.load(f)
        hosts = frozenset(h.lower().lstrip(".") for h in cfg.get("domains", []))
        pats = [re.compile(p, re.I) for p in cfg.get("url_patterns", [])]
        return hosts, pats
    except Exception:  # noqa: BLE001
        return frozenset(), []


_PARASITE_HOSTS, _PARASITE_RES = _load_parasites()


def _parasite(url: str) -> float:
    """1.0 when the URL is a known never-authoritative parasite (blocklisted
    host or any parent suffix, OR a junk URL pattern), else 0.0. Host match is
    suffix-aware so `foo.aic.lv` is caught by `aic.lv`, but a listed host never
    matches a longer unrelated host."""
    if not url:
        return 0.0
    if _PARASITE_HOSTS:
        h = _domain(url)
        if h.startswith("www."):
            h = h[4:]
        parts = h.split(".")
        for i in range(len(parts) - 1):
            if ".".join(parts[i:]) in _PARASITE_HOSTS:
                return 1.0
    for rx in _PARASITE_RES:
        if rx.search(url):
            return 1.0
    return 0.0


def weights_fingerprint() -> dict:
    """Which fusion-weights file is actually loaded, and its content hash.

    Task 522: a weights promotion is a versioned-artifact swap
    (engineering-doctrine "Learned ranking artifacts ... are versioned data
    with provenance -- a candidate artifact never sits one filename typo from
    the serving path"), so /health has to be able to say which file and which
    sha256 it loaded, not just "ok" (deployment-doctrine "No Invisible
    State"). Read fresh every call rather than cached at import time, so the
    answer is the file currently on disk, not the one present when the
    process started.
    """
    path = _weights_path()
    try:
        raw = pathlib.Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
    except Exception:  # noqa: BLE001
        digest = None
    return {"path": path, "sha256": digest}


# --------------------------------------------------------------------------- #
#  HUB-FLOOR (task 282 / specs/eval-general-relevance-flood.md; ADR-025 stage-2).
#  The RELEVANCE-RIDING-HUB flood (class C), the residual the trust head (ADR-036)
#  and per-page BM25 tf-saturation (ADR-037) BOTH structurally miss. A large,
#  uniformly-high-trust AUTHORITY hub (canada.ca 241k / laws-lois 148k /
#  federalreserve 52k / eur-lex 38k …) owns a page with genuine-but-GENERIC
#  dense/BM25 relevance to a generic query token and rides `rel_core` + the flat
#  trust floor to #1 on a general-intent query. It is per-HOST document-frequency
#  driven ("lottery tickets" — a 241k-page domain holds many spurious generic-token
#  matches), ORTHOGONAL to the per-PAGE tf that ADR-037 bounds, and rides rel_core
#  not trust (task 228 measured: stripping the whole trust_term leaves canada.ca #1
#  on `ta-msft-rev` — so the trust head is inert here).
#
#  The abandoned hub-gate (210/228, AW_HUBGATE) keyed on dense-ABSENCE + strips
#  TRUST — both wrong for a dense-present rel-rider (it inverted, +13 named-gov
#  flood). This is a DIFFERENT lever: a bounded ADDITIVE penalty on the candidate
#  TOTAL (so it can move a rel-rider trust-stripping cannot), gated on THREE
#  signals the trust head and the country head each lack:
#    (i)   INTENT     — reuse the ADR-036 head's general-intent beta (rh_beta >=
#                       MINBETA). On authority-lookup intent (rh_beta->0: statutes,
#                       specs, bb-027 UN-population) the floor NEVER fires -> the
#                       legal/authority hold is preserved BY CONSTRUCTION.
#    (ii)  HUB-NESS   — a serving-side host-df table (hub_hosts.json; ADR-054: ship
#                       the POLICY behind a flag, not a 47.8M-record payload write).
#                       Built read-only from the `domain` facet by
#                       the former build_hub_table.py (git-blob:42bc09f1b6262591cdeb14d2b4d303fccfe765fd, deleted); wikipedia(reference)/pubmed,
#                       arxiv(academic)/INTL-tech EXCLUDED by class (wikipedia is
#                       OFTEN the right answer — spec §4 Q4).
#    (iii) DISTINCTIVENESS — the title-centrality tokenizer (ADR-031): a hub whose
#                       TITLE is NOT central to the query matched on a GENERIC token
#                       (canada.ca-CRA "Canada Revenue Agency" for "Revenues 2024 of
#                       Microsoft" — matches "revenue", not "Microsoft"). A genuine
#                       hub authority (laws-lois for "Canada Labour Code termination",
#                       MDN for "Array.prototype.map") matches the DISTINCTIVE tokens
#                       in-title -> high centrality -> EXCLUDED. THIS is what protects
#                       the authority hold; validated on a sample before tuning.
#  DEFERS to the country levers (foreign_gov / off_country / country_scope head)
#  when the query is geo-scoped AND the hub is stamped off-country (§2.4 — no double
#  penalty; the country head just merged). Positive-only rule preserved (ADR-031):
#  it demotes ONLY a low-centrality hub match under general intent, never a general
#  low-centrality page. Flag-gated AW_HUBFLOOR (default OFF, byte-identical when
#  off); per-knob env-revertible. rollback = AW_HUBFLOOR=0 (or AW_HUBFLOOR_W=0.0).
# --------------------------------------------------------------------------- #
_HUBFLOOR = _S.trust.hubfloor


_HUBFLOOR_TABLE = _S.runtime.hubfloor_table


def _load_hub_hosts():
    """{registrable-normalized domain -> hubness in [0,1]} from the committed
    serving table. Empty (=> lever inert) when the flag is off or the table is
    missing/bad — a bad table never breaks search (fail-safe, ADR-054 discipline)."""
    if not _HUBFLOOR:
        return {}
    try:
        with open(_HUBFLOOR_TABLE) as f:
            cfg = json.load(f)
        out = {}
        for dom, v in (cfg.get("hosts") or {}).items():
            d = dom.lower().lstrip(".")
            if d.startswith("www."):
                d = d[4:]
            out[d] = float(v.get("hubness", 1.0)) if isinstance(v, dict) else float(v)
        return out
    except Exception as e:  # noqa: BLE001 — a bad hub table must not break search
        import logging
        logging.getLogger("aw_server.retrieval").warning(
            "hub table load failed (%s: %s) — hubfloor disabled", type(e).__name__, e)
        return {}


_HUB_HOSTS = _load_hub_hosts()


def _hubness(pl) -> float:
    """Per-host hubness from the serving hub table, keyed on the `domain` facet
    (the exact field the table was built from) with a URL-host fallback. A sub-host
    of a listed hub inherits it (suffix walk). 0.0 when the host is not a hub."""
    dom = (pl.get("domain") or _domain(pl.get("url") or "")).lower()
    if dom.startswith("www."):
        dom = dom[4:]
    h = _HUB_HOSTS.get(dom)
    if h is not None:
        return h
    parts = dom.split(".")
    for i in range(1, len(parts) - 1):
        h = _HUB_HOSTS.get(".".join(parts[i:]))
        if h is not None:
            return h
    return 0.0


# --------------------------------------------------------------------------- #
#  AUTHORITY (ADR-019 product-1 / prd-link-graph.md — the 2026-07-20 ship
#  amendment). A per-domain ENDORSEMENT/citation signal wired as a bounded,
#  flag-gated ADDITIVE fusion boost. Designed in ADR-019 but never wired in
#  (trust_tier carried authority alone); the signal is already computed — the
#  endorsement graph (endorse_all.json, post-242 641k edges) gives every target
#  HOST a class-weighted citation score. build_authority_table.py folds
#  it into cytir/serve/authority_hosts.json (host -> authority in [0,1]),
#  applying the ADR-019 prerequisites AT BUILD TIME: archive/proxy wrappers
#  UNWRAPPED (credit flows to the real target, not web.archive.org — the #1 raw
#  citation), and the SOURCE-CLASS taxonomy (primary-artifact=full weight,
#  publisher=half+threshold, media/infra=EXCLUDED signal-only).
#
#  COMPLEMENTS, does not double-count, trust_tier + AW_HUBFLOOR:
#    * trust_tier is the CURATED authority prior (a tier per source_type); this is
#      the EARNED citation signal, an orthogonal axis. Bounded + additive (a small
#      lift), so it sharpens ordering WITHIN a trust band rather than restacking it.
#    * AW_HUBFLOOR DEMOTES a large gov hub that rides generic relevance onto a
#      general-intent query (low title-centrality). Authority BOOSTS a genuinely-
#      endorsed source. They could fight on exactly the gov-hub/general-intent
#      candidate hubfloor targets — so authority DEFERS: no boost is added to a
#      candidate the hub-floor is penalising (hubfloored) nor to one a COUNTRY lever
#      demoted (geo_penalized). Where hubfloor is silent (authority-lookup intent,
#      or a hub that genuinely matches in-title) the boost applies — that is the
#      authority-ranking lift on the legal/names/authoritative golds.
#  Flag-gated AW_AUTHORITY (default OFF, byte-identical when off); rollback =
#  AW_AUTHORITY=0 (or AW_AUTHORITY_W=0.0). Per-knob env-revertible.
# --------------------------------------------------------------------------- #
_AUTHORITY = _S.trust.authority


_AUTHORITY_TABLE = _S.runtime.authority_table


def _load_authority_hosts():
    """{registrable-normalized host -> authority in [0,1]} from the committed
    serving table. Empty (=> lever inert) when the flag is off or the table is
    missing/bad — a bad table never breaks search (fail-safe, ADR-054 discipline)."""
    if not _AUTHORITY:
        return {}
    try:
        with open(_AUTHORITY_TABLE) as f:
            cfg = json.load(f)
        out = {}
        for dom, v in (cfg.get("hosts") or {}).items():
            d = dom.lower().lstrip(".")
            if d.startswith("www."):
                d = d[4:]
            a = float(v.get("authority", 0.0)) if isinstance(v, dict) else float(v)
            if a > 0:
                out[d] = a
        return out
    except Exception as e:  # noqa: BLE001 — a bad authority table must not break search
        import logging
        logging.getLogger("aw_server.retrieval").warning(
            "authority table load failed (%s: %s) — authority disabled",
            type(e).__name__, e)
        return {}


_AUTH_HOSTS = _load_authority_hosts()


def _authority(pl) -> float:
    """Per-host endorsement authority from the serving table, keyed on the
    `domain` facet (the field the table was built from) with a URL-host fallback.
    A sub-host of a listed host inherits it (suffix walk). 0.0 when not endorsed."""
    dom = (pl.get("domain") or _domain(pl.get("url") or "")).lower()
    if dom.startswith("www."):
        dom = dom[4:]
    a = _AUTH_HOSTS.get(dom)
    if a is not None:
        return a
    parts = dom.split(".")
    for i in range(1, len(parts) - 1):
        a = _AUTH_HOSTS.get(".".join(parts[i:]))
        if a is not None:
            return a
    return 0.0


def loaded_artifacts(heads: dict) -> dict:
    """{name: (path, loaded)} for every table and head this process read (`GET /config`). A
    table that failed to load disables its feature without an error (spec risk R5), so the
    echo says which ones actually loaded. `heads` is the learned heads' own entries
    (`heads._head_artifacts()`), spliced in at their place in the listing: the heads live
    in `retrieval` until step 5 and this module may not import it."""
    _present = _settings.present
    return {
        "fusion_weights": (_weights_path(), _S.weights_file_loaded),
        **heads,
        "hub_hosts": (_HUBFLOOR_TABLE, _present(_HUB_HOSTS)),
        "authority_hosts": (_AUTHORITY_TABLE, _present(_AUTH_HOSTS)),
        "parasite_domains": (_S.runtime.parasite_domains, _present(_PARASITE_HOSTS)),
        "gazetteer": (_GAZ_PATH, _present(_GAZETTEER)),
        "country_map": (_COUNTRY_MAP_PATH or None, _present(_COUNTRY_MAP)),
    }
