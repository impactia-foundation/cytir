#!/usr/bin/env python3
"""First-refresh / weekly-cadence composer for the refresh-cadence new-refs
lane's feed-discovered packs (task 661 F5 follow-up, 2026-09-28).

WHY THIS EXISTS
----------------
`cytir/sources/refresh_direct/new_refs.py` DISCOVERS candidate URLs (via each
pack's registered `ADAPTER_REGISTRY[...].discover()`) but never fetches
article bodies, embeds, or writes anywhere (its own docstring). Once
Tim ratified `sources/us/github-releases-tracked.yaml`,
`sources/us/huggingface-orgs-tracked.yaml` and
`sources/us/vendor-ai-blogs-tracked.yaml` (#383/#387), someone has to turn
those discovered URLs into indexable records -- fetch, consent-gate,
extract, embed, and enqueue -- on the SAME ADR-110 promotion-queue
mechanical path the news lane already uses. This module is that missing
"compose" step, generic across any pack registered in
`cytir.sources.refresh_direct.new_refs.ADAPTER_REGISTRY` whose route is a plain
exact-URL feed (`source_owned_incremental_feed` / `documented_source_api`
with a `query`) -- the next such pack to onboard adds one `PACK_META`
entry here and nothing else changes, same discipline as
`ADAPTER_REGISTRY` itself.

SHAPE: mirrors `hn_discussion_daily.py`'s own stage -> pack -> merge ->
tombstone -> enqueue pipeline almost exactly (same `news_daily_pack_and_
queue`/`news_backlog_composer` helpers, same promotion-queue pack format)
-- but the "stage" step is different in kind: HN discussion shells out to
a dedicated `hn_discussion_stage.sh`; this module fetches + extracts +
consent-gates the discovered URLs itself (there is no per-family shell
script to reuse for a brand-new feed family) and then stages through the
SAME `cytir.index.stage_pack_b1` embedder every other b1-native arm
(chmedia/swissinfo/hn/hn_discussion) uses.

CONSENT (ADR-111/113, general-web minimum-record basis): every fetched
row is passed through `consent_gate.py gate --direct-crawl
--search-indexing`, THE live robots/AIPREF/TDMRep resolver every other
direct-fetch lane uses (consent_resolver.py) -- never a bespoke or
inherited verdict. A declined host's rows never reach the staging
collection. `--direct-crawl` matches these packs' genuinely direct
(non-CC) fetch; `--search-indexing` reads search_ok (not train_ok) as the
admission gate, per ADR-024 amendment 5 -- a CCBot/AI-training-token-only
decline still search-admits.

EXTRACTION: `aw_common.process_html`, the SAME one-pass extractor
`ingest_tech_blogs.py` uses for exactly this class of content
(startup/tech/company blogs) -- title/snippet/language/chunking are all
its existing, already-tested machinery. This module supplies only the
per-pack `passage_common` fields (`tier`, `source_type`, `trust_tier`)
`process_html` cannot infer from the page itself.

Never applies anything, never takes a lock -- writes only a promotion-
queue pack (manifest.json + news-points.*.jsonl.gz + tombstones.jsonl +
READY); ADR-110's drain lands it like any other pack.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import email.utils as _email_utils
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit

from cytir.core import twin as TWIN  # noqa: E402
from cytir.core.paths import REPO_ROOT, data_path, lazy_setting_path  # noqa: E402
import cytir.core.common as AC  # noqa: E402
from cytir.extract import html_extract as _html_extract  # noqa: E402
from cytir.acquire import fetch_process as FP  # noqa: E402
import cytir.sources.hf_model_cards.ingest as HFC  # noqa: E402
import cytir.news.compose as C  # noqa: E402
from cytir.news import daily as D  # noqa: E402
import cytir.sources.refresh_direct.docs_sitemap as DSA  # noqa: E402
# the docs-row chunk selection + Markdown row builder live in the package (task 1157, S7); re-exported, the lane calls them
from cytir.chunk.markdown_rows import (DOCS_SOURCE_TYPE, _DOCS_SELECTION, _SELECTOR_ARTIFACT,  # noqa: E402,F401
                                    build_rows_for_markdown, docs_chunk_selection, require_docs_chunker)
from cytir.chunk.markdown_rows import stamp_docs_row as _stamp_docs_row  # noqa: E402  (task 1169; the clock stays here)
from cytir.index import supersede as US  # noqa: E402
from cytir.sources.refresh_direct import new_refs as RLN  # noqa: E402
from cytir.sources.refresh_direct import github_repo_docs as GRD  # noqa: E402  (task 2630: head-sha skip state)
from cytir.news import hn_outlinks as HO  # noqa: E402
from cytir.core.user_agent import user_agent  # noqa: E402

R = D.R  # same pinned release the news/hn_discussion arms run from
W = lazy_setting_path("refresh_direct_work")  # resolves on first use, never at import
QUEUE_DIR = D.QUEUE_DIR
TWIN_URL, TWIN_COLL = D.TWIN_URL, D.TWIN_COLL
MATERIALIZE_SCRIPT = R / "cytir/sources/families/news.py"

PACK_PREFIX = "refresh-direct-"
FETCH_INTERVAL_S = 1.0  # per-process politeness: ~1 request/s (HF 429s at 3.5/s, task 693)
RATE_LIMIT_TRIES = 5
# A run where more than this share of fetched candidates errored (with at
# least FETCH_ERROR_MIN of them) is a partial failure, not a clean run.
FETCH_ERROR_SHARE = 0.2
FETCH_ERROR_MIN = 3
STATE_PATH = W / "state.json"
NEXT_PATH = W / "NEXT.md"

# One entry per pack this composer knows how to run. `label` feeds the
# staging collection name (news_stage_contract.news_stage_collection_name)
# and the pack_id suffix; `tier`/`source_type`/`trust_tier` seed
# aw_common.process_html's passage_common (the fields it cannot infer from
# the page itself -- see aw_common.TIER_TRUST/TIER_SOURCE: tier B (0.82) is
# the closest coarse band to these packs' own 0.80-0.85 ratified trust_tier,
# so tier="B"/source_type="reference" for all three, same as any other
# curated third-party-authored source at this trust band).
DOCS_DELTA_SWITCH = "AW_DOCS_DELTA"  # master: arms every docs delta pack (the task-972 hand procedure)
# Task 2622: each docs pack also has its OWN switch, so one can run unattended (a drop-in sets it) while
# the other stays hand-armed. Either the pack's own name or the master set to "1" arms a pack.
DOCS_DELTA_OPENCODE_SWITCH = "AW_DOCS_DELTA_OPENCODE"
DOCS_DELTA_GH_REPO_DOCS_SWITCH = "AW_DOCS_DELTA_GH_REPO_DOCS"
DOCS_DELTA_NOTE = ("the docs delta packs run only when armed (task 972; per-pack switches task 2622): "
                   "a pack's own switch, or AW_DOCS_DELTA=1 for all; the unit's drop-in arms only opencode")
PACK_META = {
    "us-github-releases-tracked": {
        "label": "gh-releases", "tier": "B", "source_type": "reference",
        "trust_tier": 0.85,
    },
    "us-huggingface-orgs-tracked": {
        "label": "hf-models", "tier": "B", "source_type": "reference",
        "trust_tier": 0.85,
        # Task 693: ONE way to ingest HF cards -- the task-120 README shape
        # (`ingest_hf_model_cards.card_records`: repo-id title, structured
        # summary + README chunks, tier C / trust 0.75, same as the rows already
        # in the index). The HTML-page route gave "<id> · Hugging Face –
        # Huggingface" titles. tier/trust above are NOT used on this route.
        "route": "hf_readme",
    },
    # Task 697 F1 / ruling 23.1: the pages Hacker News stories link to. ONE discovery source of
    # this lane, behind AW_HN_OUTLINKS=1 (OFF by default until Tim merges the ADR-098 Ruling 2
    # amendment). Not a cadence-table pack: its candidates come from the HN story window, not
    # from a ratified roster; each HOST is gated by live consent before any request.
    HO.PACK_ID: {**HO.META, "route": "hn_outlinks", "gated_by": HO.SWITCH_ENV},
    "us-vendor-ai-blogs-tracked": {
        "label": "vendor-blogs", "tier": "B", "source_type": "reference",
        "trust_tier": 0.80,
    },
    # Task 972: the first DOCS delta packs. `docs_html`: a docs host's sitemap, fetched page by page
    # (route fields copied from data/cov351_routes.yaml `opencode_ai_docs`: tier A, trust 0.75,
    # docs, licence `docs-index-only-unstated`). `docs_markdown`: a repo's docs at a pinned commit,
    # already Markdown. Both skip a URL whose chunks equal the twin's (`DSA.unchanged_against_twin`)
    # and stamp `published` = lastmod / last commit date ONLY when known (never the crawl date);
    # `run_one` merges docs packs with `require_published=False`.
    "us-opencode-ai-docs": {
        "label": "opencode-ai", "tier": "A", "source_type": "docs", "trust_tier": 0.75,
        "route": "docs_html", "license": "docs-index-only-unstated",
        "gated_by": DOCS_DELTA_OPENCODE_SWITCH, "gate_note": DOCS_DELTA_NOTE,
    },
    "us-github-repo-docs-tracked": {
        "label": "gh-repo-docs", "tier": "A", "source_type": "docs", "trust_tier": 0.75,
        "route": "docs_markdown",
        "gated_by": DOCS_DELTA_GH_REPO_DOCS_SWITCH, "gate_note": DOCS_DELTA_NOTE,
    },
}


def pack_armed(pack_id: str, env=None) -> bool:
    """True when the pack has no switch, or its own switch (or, for a docs pack, the master) is "1"."""
    env = os.environ if env is None else env
    gate = PACK_META[pack_id].get("gated_by")
    if not gate:
        return True
    names = [gate] + ([DOCS_DELTA_SWITCH] if PACK_META[pack_id].get("route", "").startswith("docs_") else [])
    return any(env.get(n) == "1" for n in names)


def default_pack_ids(env=None) -> list[str]:
    """Every pack, minus the ones whose switch is not set (the HN-linked-pages route is OFF
    unless AW_HN_OUTLINKS=1). Naming such a pack with --pack-id while the switch is off is
    refused in `main`."""
    env = os.environ if env is None else env
    return sorted(p for p in PACK_META if pack_armed(p, env))


def cadence_row(pack_id: str, cadence_path: pathlib.Path) -> dict:
    rows = RLN.load_cadence_rows(cadence_path)
    for row in rows:
        if row["id"] == pack_id:
            return row
    raise SystemExit(f"{pack_id} is not in {cadence_path} -- regenerate "
                     "data/refresh-cadence.v1.json (`cyt source "
                     "refresh-cadence`) after adding a pack")


def discover_candidates(pack_id: str, row: dict, adapter_opts: dict | None = None) -> dict:
    """Reuse the SAME adapter `new_refs.py` would dispatch to
    (family key first, then pack id -- identical lookup, task 661 F5), and
    the SAME `pack_path` fallback those adapters already implement for
    reading their tracked-item list off the real pack yaml. Never a second
    discovery path."""
    impl = (RLN.ADAPTER_REGISTRY.get(row["adapter"])
            or RLN.ADAPTER_REGISTRY.get(pack_id))
    if impl is None:
        raise SystemExit(f"{pack_id}: no adapter registered "
                         f"(adapter={row['adapter']!r}) -- add one to "
                         "new_refs.ADAPTER_REGISTRY first")
    return impl({**row, "adapter_opts": adapter_opts} if adapter_opts else row)


def hn_discover(*, since_days: int, twin_url: str, collection: str,
                min_points: int = HO.DEFAULT_MIN_POINTS, now_i: int | None = None,
                client=None, algolia=None, resolve=None, bucket=None, indexed=None,
                paywalled=None, declined=None) -> dict:
    """Everything before the first page request of the HN-linked-pages route: the story window
    (Algolia, the enumeration the discussion frontier already uses), the host screen, the dedup
    against the twin by url, and the LIVE consent verdict of every remaining host (the build box
    vantage). Returns a discovery-shaped dict plus `hn`: {outcomes {url: terminal}, hosts
    {host: bucket}, counts}. Only candidates on an admitted host reach `candidates`. The keyword
    arguments are the seams the tests use; a real run passes none."""
    from cytir.sources.family import HttpSpec  # noqa: PLC0415
    from cytir.sources.hn.freshness_ingest import algolia_window  # noqa: PLC0415
    from cytir.sources.http import make_client  # noqa: PLC0415
    now_i = int(time.time()) if now_i is None else now_i
    own_client = client is None
    client = client or make_client(HttpSpec(timeout=5.0, follow_redirects=False))
    try:
        hits, enumeration = (algolia or algolia_window)(
            client, now_i - since_days * 86400, now_i, "story")
    finally:
        if own_client:
            client.close()
    cands = HO.story_candidates(hits, min_points=min_points)
    if paywalled is None:
        from cytir.corpus.source_cards import PAYWALLED as paywalled  # noqa: PLC0415
    if declined is None:
        from cytir.consent import block as _block  # noqa: PLC0415
        declined = _block.is_blocked
    kept, dropped = HO.screen(cands, paywalled=paywalled, declined=declined)
    outcomes = {u: t for t, urls in dropped.items() for u in urls}
    if indexed is None:
        from cytir.core.qdrant import qdrant_client  # noqa: PLC0415
        tw = qdrant_client(twin_url, timeout=120)
        indexed = lambda urls: HO.indexed_urls(tw, collection, urls)  # noqa: E731
    have = indexed([c["url"] for c in kept])
    for c in kept:
        if c["url"] in have:
            outcomes[c["url"]] = "indexed_already"
    kept = [c for c in kept if c["url"] not in have]
    if resolve is None or bucket is None:
        from cytir.acquire import sitemap_ingest as SI  # noqa: PLC0415
        from cytir.consent import resolver as cr  # noqa: PLC0415
        resolve = lambda hosts: cr.resolve_hosts(  # noqa: E731
            hosts, cache_dir=cr.CACHE_DIR, progress=False, direct_crawl=True)
        bucket = SI._search_index_bucket
    verdicts = HO.host_verdicts([HO.host_of(c["url"]) for c in kept], resolve, bucket)
    admitted = []
    for c in kept:
        v = verdicts[HO.host_of(c["url"])]
        if v in HO.FETCHABLE_BUCKETS:
            admitted.append(c)
        else:
            outcomes[c["url"]] = "consent_declined" if v == "declined" else "consent_unknown"
    return {"candidates": admitted, "errors": {}, "orgs_checked": 1, "orgs_errored": 0,
            "hn": {"outcomes": outcomes, "hosts": verdicts, "stories_in_window": len(hits),
                   "enumeration": enumeration}}


def hn_fetcher(preserve_root: pathlib.Path):
    """The sealed, polite fetcher of the route (live wiring)."""
    import httpx  # noqa: PLC0415 -- httpx.Timeout only; the client is the guarded factory's
    from cytir.acquire import sitemap_ingest as SI  # noqa: PLC0415
    from cytir.consent import resolver as cr  # noqa: PLC0415
    from cytir.sources.family import HttpSpec  # noqa: PLC0415
    from cytir.sources.http import make_client  # noqa: PLC0415
    ua = user_agent()
    client = make_client(HttpSpec(
        timeout=httpx.Timeout(connect=15.0, read=30.0, write=30.0, pool=30.0),
        follow_redirects=False))
    return HO.LinkedFetcher(
        client, preserve_root=preserve_root, ua=ua,
        robots=lambda host: SI.load_robots(host, cr.CACHE_DIR), robots_ok=SI.robots_ok)


def discovery_refusal(pack_id: str, discovery: dict) -> str | None:
    """A named refusal when discovery FAILED, never a quiet zero (task 693,
    corpus-doctrine #3/#8: an API outage or a rate limit is a fact about us,
    and "0 candidates" from a dead channel looks identical to a genuine
    quiet day). A genuine 0 is `orgs_errored == 0` (or no such key, for the
    adapters that have no per-item errors) with `candidates_found == 0`.
    A PARTIAL failure (some orgs errored, others answered) is not refused
    -- the survivors still land -- but `main()` ends non-zero and the state
    file records it, so the watchdog sees it."""
    errored = int(discovery.get("orgs_errored") or 0)
    if errored and not discovery.get("candidates"):
        return (f"{pack_id}: discovery FAILED on {errored} of "
                f"{discovery.get('orgs_checked')} source(s) and found 0 "
                f"candidates -- an error is not an empty: "
                f"{json.dumps(discovery.get('errors') or {})[:600]}")
    return None


def build_rows_for_candidate(url: str, html: str, *, meta: dict) -> list[dict]:
    """Pure (given already-fetched `html`): extract passages via the SAME
    one-pass extractor `ingest_tech_blogs.py` uses, and turn each into a
    plain b1-native row (the shape `aw_b1_projection.project_record`
    reads: `url`, `text`, plus whatever `Passage.payload()` carries).
    A page that extracts to zero passages (too short, boilerplate,
    listing-shell, wrong language -- `aw_common`'s own gates) returns []
    -- a real, named empty, not an error."""
    domain = urlsplit(url).netloc.lower()
    passage_common = {
        "domain": domain, "tier": meta["tier"],
        "source_type": meta["source_type"], "trust_tier": meta["trust_tier"],
    }
    docs_sel = docs_chunk_selection(meta)
    passages, _links = _html_extract.process_html(
        url, html, {"content": True, "links": False,
                    "passage_common": passage_common,
                    "chunk_selection_for": (lambda md: docs_sel) if docs_sel else None})
    rows = []
    for p in passages:
        row = p.payload()
        row["text"] = p.text
        row["chunk_n_words"] = len(p.text.split())   # task 984: the composer's content_words_by_url
        if meta.get("published") and not row.get("published"):
            row["published"] = meta["published"]
        rows.append(row)
    return rows


def stamp_docs_row(row: dict, *, meta: dict, published: str | None, license_: str | None) -> dict:
    """The lane's clock over the package's stamp (task 1169): `last_seen` is today's date, read HERE."""
    return _stamp_docs_row(row, meta=meta, published=published, license_=license_,
                           today=_dt.date.today().isoformat())


def parse_published(value) -> _dt.datetime | None:
    """Tolerant parser across the three adapters' own date shapes: Atom
    `<updated>` (ISO 8601, github_releases_adapter), the HF Hub API's
    `lastModified`/`createdAt` (ISO 8601 with milliseconds,
    hf_org_models_adapter), and RSS 2.0 `<pubDate>` (RFC 822,
    vendor_blog_rss_adapter). Returns None (never raises) for anything
    that doesn't parse -- an unparseable/missing date is a real, named
    absence, not a crash."""
    if not value:
        return None
    try:
        dt = _email_utils.parsedate_to_datetime(value)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_dt.timezone.utc)
            return dt
    except (TypeError, ValueError):
        pass
    try:
        return _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def filter_recent_candidates(candidates: list[dict], *, since_days: int,
                             now: _dt.datetime | None = None) -> dict:
    """Cap the fetch to items published within the last `since_days` --
    without this, a first run against a long-lived RSS feed (openai.com's
    carries 1230 historical items) re-fetches the publisher's entire
    archive on every run, not just what's new since the last one (found
    live, task 661, 2026-09-28: 1230 openai.com candidates in one run,
    the great majority years old). A candidate with NO parseable date is
    KEPT (never silently dropped for missing metadata) -- only a
    confidently-old date excludes it. Returns
    `{"kept": [...], "dropped_old": N, "dropped_old_oldest_seen": iso|None}`."""
    now = now or _dt.datetime.now(_dt.timezone.utc)
    cutoff = now - _dt.timedelta(days=since_days)
    kept, dropped = [], []
    for cand in candidates:
        published = parse_published(cand.get("published"))
        if published is not None and published < cutoff:
            dropped.append(published)
            continue
        kept.append(cand)
    return {"kept": kept, "dropped_old": len(dropped),
           "dropped_old_oldest_seen":
               min(dropped).isoformat() if dropped else None}


class ChallengedError(Exception):
    """A page answered with an access-control challenge (WAF/bot-
    mitigation, e.g. a 403 with a Cloudflare managed-challenge response),
    not an ordinary fetch failure. Caught by `run_stage()` and routed to
    `build_feed_item_row` instead of `errors` -- see that function's
    docstring for why this is never retried past a plain GET."""


def plain_get(url: str, *, ua: str, httpx_module, sleep=time.sleep) -> str:
    """ONE plain GET, honest UA, no retry and no fallback past it
    (coordinator ruling, task 661, 2026-09-28, reversing an earlier
    same-day headless-browser-render fallback): a 403 challenge is the
    site's own access control, not a technical obstacle to route around
    -- consent-probe-vantage's standing rule ("a WAF or challenge
    response means UNREADABLE, recorded as such, never circumvented")
    applies to the CONTENT fetch exactly as it already does to a consent
    probe. openai.com/index/* articles return 403 with a Cloudflare
    `cf-mitigated: challenge` header (confirmed live, UA-independent --
    not a consent decline; openai.com's own robots.txt already resolves
    search_ok=true) -- raises `ChallengedError` so the caller falls back
    to the feed's own openly-published item metadata instead of the
    page body. Any OTHER failure (404, 5xx, network error, a non-403
    4xx) raises normally and is recorded as a plain fetch error."""
    # 429/503 are the host asking US to slow down -- a rate signal, not a
    # verdict about the page and not a challenge. Back off (honouring
    # Retry-After, capped) and retry; only a still-throttled final answer is
    # recorded as a fetch error. Task 693: the first 45-day HF catch-up at
    # ~3.5 req/s lost 144 of 244 pages to 429 with no retry.
    for attempt in range(RATE_LIMIT_TRIES):
        r = httpx_module.get(url, timeout=20.0, follow_redirects=True,
                             headers={"User-Agent": ua})
        if r.status_code not in (429, 503) or attempt == RATE_LIMIT_TRIES - 1:
            break
        try:
            wait = float(r.headers.get("retry-after", ""))
        except (TypeError, ValueError, AttributeError):
            wait = 5.0 * (3 ** attempt)
        sleep(min(max(wait, 1.0), 120.0))
    if r.status_code == 403:
        raise ChallengedError(f"403 (access-control challenge) for {url}")
    r.raise_for_status()
    return r.text


def build_feed_item_row(cand: dict, *, meta: dict) -> dict | None:
    """The minimum record for a page behind a challenge: ONLY what the
    source itself already published openly in its own feed -- title,
    description/summary, link, pubDate. Never the challenged page body
    (never fetched for this row at all). `source_extract: "feed_item"`
    names the row's own provenance (aw_b1_projection.BASE_PASSTHROUGH);
    `consent_state`/`consent_basis`/`consent_source`/`consent_checked_at`
    are stamped directly on the row (aw_b1_projection.project_record
    reads a row's own consent fields over the CLI-global default --
    `record_consent` in that module) so this ONE row can carry a
    different consent story than the rest of the same `--in` file, which
    is uniformly `resolved`/`robots-live-resolve`.
    Returns None (a real, named empty, not an error) when there's no
    text at all to build a record from (no title, no description).

    `lang`: found live, task 661 (2026-09-28) -- omitting it entirely
    tripped cytir.index.stage_pack_b1's own payload-carry guard (lang coverage
    70.4% < the required 99%, once ~30% of a run's rows were feed-item
    fallbacks with no `lang`). Detected from the SAME title+description
    text via `fetch_process.detect_lang` -- the one real language
    detector every other row in this repo is scored by (not a second
    detector for this path), never guessed from the URL alone."""
    title = (cand.get("title") or "").strip()
    description = (cand.get("description") or "").strip()
    text = " — ".join(part for part in (title, description) if part)
    if not text:
        return None
    domain = urlsplit(cand["url"]).netloc.lower()
    lang, _confident = FP.detect_lang(text, cand["url"], meta.get("lang_default", "en"))
    return {
        "url": cand["url"], "text": text, "title": title or None,
        "domain": domain, "tier": meta["tier"], "lang": lang,
        "source_type": meta["source_type"], "trust_tier": meta["trust_tier"],
        "published": cand.get("published"),
        "source_extract": "feed_item",
        "consent_state": "unreadable (challenge)",
        "consent_basis": "robots-live-resolve",
        "consent_source": "page fetch returned an access-control "
                          "challenge (403); indexed from the source's "
                          "own openly-published feed item only",
        "consent_checked_at": _dt.datetime.now(_dt.timezone.utc)
                              .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def hf_card_rows(cand: dict, *, get=None) -> tuple[str, object]:
    """HF README route for one candidate (task 693). Returns
    `("rows", [record, ...])` or `("derivative", base_repo_id)`. Raises on a
    detail/README fetch failure (the caller records it in `errors`).

    A derivative (declared base_model owned by a DIFFERENT org: a
    quantization, finetune or re-upload) is SKIPPED, not admitted with a
    lower trust: AW's grounding target is the authoritative card, a derivative
    carries the base model's name and text and outranks it (measured on prod
    2026-10-02: nvidia/GLM-5.3-NVFP4 above zai-org/GLM-5.3). Skipping is
    recorded and counted in the pack manifest, never silent."""
    get = get or HFC._get
    mid = cand["model_id"]
    status, body, _ = get(f"{HFC.API}/{mid}")
    if status != 200 or not body:
        raise RuntimeError(f"HF detail status {status} for {mid}")
    detail = json.loads(body)
    if detail.get("private") or detail.get("disabled"):
        return "rows", []
    base = HFC.derivative_base(mid, detail)
    if base:
        return "derivative", base
    gated = bool(detail.get("gated"))
    readme = None
    if not gated:
        rstatus, rbody, _ = get(HFC.RAW.format(id=mid), accept_json=False)
        if rstatus == 200 and rbody:
            readme = rbody
        elif rstatus not in (401, 403, 404):
            raise RuntimeError(f"HF README status {rstatus} for {mid}")
    seen: set = set()
    rows = []
    for rec, _kind in HFC.card_records(
            mid, detail, readme, row={}, gated=gated, seen_pid=seen,
            discovery={"source": "huggingface-hub-api", "publisher": cand.get("org"),
                       "selection": "refresh_lane_direct_compose"}):
        # `published` = the card's own lastModified (merge_rows drops undated rows)
        rec["published"] = detail.get("lastModified") or cand.get("published")
        rows.append(rec)
    return "rows", rows


def derivative_tombstones(client, collection: str, urls: list[str], *,
                          marker_rows: list | None = None) -> list[dict]:
    """Every point currently under a skipped-derivative URL, as tombstones,
    EXCEPT marker rows (`supersede.is_card`: type `entity_card` or carrying
    `entity_kind`/`qid`): a `takedown` sits outside the drain's stored-row
    pair check, and nothing but this filter keeps a derivative-url scroll
    from deleting an entity row that shares the url (task 698 follow-up;
    twin 2026-10-04: 2 of 35,128 huggingface.co rows carry a marker, neither
    a model repo). Skipped ids are appended to `marker_rows` (counted in the
    manifest) -- never silently dropped.

    `url` is an indexed payload field (exact match), so this is a cheap
    read-only scroll per URL. Reason `takedown`: promote_queue's
    TOMBSTONE_REASONS has no "policy-excluded", `superseded` requires the url
    in the pack's own set (an orphan otherwise) and `aged_out` requires an old
    `published`; `takedown` is the only reason with no such precondition."""
    from qdrant_client import models as qm  # noqa: PLC0415
    if urls:
        US.guard_scroll(client, collection)     # prod gate: a prod scroll is refused (no reason given)
    out = []
    for url in sorted(set(urls)):
        offset = None
        while True:
            pts, offset = client.scroll(
                collection, scroll_filter=qm.Filter(must=[qm.FieldCondition(
                    key="url", match=qm.MatchValue(value=url))]),
                limit=500, offset=offset,
                with_payload=["source_type", "entity_kind", "qid"],
                with_vectors=False)
            for p in pts:
                pl = p.payload or {}
                if US.is_card(pl.get("source_type"), pl.get("entity_kind"), pl.get("qid")):
                    if marker_rows is not None:
                        marker_rows.append({"id": str(p.id), "url": url})
                    continue
                out.append({"id": str(p.id), "url": url, "reason": "takedown"})
            if offset is None:
                break
    return out


def run_stage(pack_id: str, *, cadence_path: pathlib.Path,
             out_log: pathlib.Path, mrl_dim: int = 256,
             fetch=None, staging_url: str = "http://localhost:6398",
             since_days: int = 14, label_suffix: str = "",
             egress_iface: str = D.DEFAULT_EGRESS_IFACE,
             twin_url: str = TWIN_URL, collection: str = TWIN.collection(),
             hn_fetcher_factory=None, hn_kwargs: dict | None = None,
             twin_client=None, adapter_opts: dict | None = None) -> dict:
    """Discover -> filter to recent -> fetch -> extract -> consent-gate ->
    embed+stage. Returns `{"staged": N, "collection": name, "candidates":
    M, "errors": {...}}`. Never raises on an individual candidate's
    fetch/extract failure (named in `errors`, never silently dropped) --
    only a REPO-LEVEL problem (bad pack, no adapter, staging write
    failure) raises."""
    meta = PACK_META[pack_id]
    hn_route = meta.get("route") == "hn_outlinks"
    if hn_route:
        if os.environ.get(HO.SWITCH_ENV) != "1":
            raise SystemExit(f"REFUSED: {pack_id}: {HO.SWITCH_ENV} is not 1")
        discovery = hn_hit = hn_discover(since_days=since_days, twin_url=twin_url,
                                         collection=collection, **(hn_kwargs or {}))
        hn_outcomes = hn_hit["hn"]["outcomes"]
    else:
        row = cadence_row(pack_id, cadence_path)
        discovery = (discover_candidates(pack_id, row, adapter_opts) if adapter_opts
                     else discover_candidates(pack_id, row))
    refusal = discovery_refusal(pack_id, discovery)
    if refusal:
        raise SystemExit(f"REFUSED: {refusal}")
    discovery_errors = dict(discovery.get("errors") or {})
    unchanged_repos = int(discovery.get("orgs_unchanged") or 0)   # task 2630: repos skipped on an equal head sha
    heads_lookup_failed = int(discovery.get("heads_lookup_failed") or 0)   # task 2718: fetched without a head check
    all_candidates = discovery["candidates"]
    # the HN route's window is the Algolia story window itself (every candidate is already inside it)
    recency = ({"kept": all_candidates, "dropped_old": 0, "dropped_old_oldest_seen": None}
               if hn_route else
               filter_recent_candidates(all_candidates, since_days=since_days))
    candidates = recency["kept"]
    if recency["dropped_old"]:
        print(f"[refresh-direct stage] {pack_id}: {recency['dropped_old']} "
             f"of {len(all_candidates)} candidate(s) older than "
             f"{since_days}d dropped (oldest seen: "
             f"{recency['dropped_old_oldest_seen']})", file=sys.stderr)

    hn_fetch = None
    if hn_route:
        W.mkdir(parents=True, exist_ok=True)
        hn_fetch = (hn_fetcher_factory or hn_fetcher)(W / "hn-linked-sealed")
    elif fetch is None:
        from cytir.sources.family import HttpSpec  # noqa: PLC0415
        from cytir.sources.http import make_client  # noqa: PLC0415
        ua = user_agent()
        plain_client = make_client(HttpSpec(timeout=20.0, follow_redirects=True))
        def fetch(url: str) -> str:
            return plain_get(url, ua=ua, httpx_module=plain_client)

    W.mkdir(parents=True, exist_ok=True)
    passages_path = W / f"passages-{pack_id}.jsonl"
    errors: dict[str, str] = {}
    derivatives: dict[str, str] = {}
    n_rows = 0
    n_challenged = 0
    hf_route = meta.get("route") == "hf_readme"
    docs_route = meta.get("route") in ("docs_html", "docs_markdown")
    n_unchanged = 0
    unchanged_urls: list[str] = []
    if docs_route and twin_client is None:
        from cytir.core.qdrant import qdrant_client  # noqa: PLC0415
        twin_client = qdrant_client(twin_url, timeout=120)
    with passages_path.open("w") as out:
        for cand in candidates:
            url = cand["url"]
            if not hn_route and meta.get("route") != "docs_markdown":
                # <= 4 req/s, same politeness as ingest_hf_model_cards; a docs route's own delay wins
                time.sleep(max(FETCH_INTERVAL_S, float(cand.get("delay_s") or 0)))
            try:
                if hn_route:
                    # the host gate and per-host spacing live in the fetcher; a terminal other
                    # than rows/title_only is counted, never an error of the whole pack
                    terminal, rows, note = HO.process_candidate(
                        cand, hn_fetch,
                        build_rows=lambda u, h, m: build_rows_for_candidate(u, h, meta=m),
                        detect_lang=lambda t, u: FP.detect_lang(t, u, "en")[0])
                    hn_outcomes[url] = terminal
                    if terminal == "fetch_error":
                        errors[url] = note
                elif hf_route:
                    kind, value = hf_card_rows(cand)
                    if kind == "derivative":
                        derivatives[url] = value
                        continue
                    rows = value
                elif docs_route:
                    if meta["route"] == "docs_html":
                        rows = build_rows_for_candidate(url, fetch(url), meta=meta)
                        stamp = cand.get("lastmod")
                    else:
                        rows = build_rows_for_markdown(
                            url, cand["markdown"], title=cand.get("title"),
                            meta={**meta, "trust_tier": cand.get("trust_tier") or meta["trust_tier"]})
                        stamp = cand.get("committed_at")   # file's last commit date; none today
                    stamp = (stamp or "")[:10] or None
                    rows = [stamp_docs_row(r, meta=meta, published=stamp,
                                           license_=cand.get("license") or meta.get("license"))
                            for r in rows]
                    if rows and DSA.unchanged_against_twin(
                            rows, DSA.twin_digests(twin_client, collection, url)):
                        n_unchanged += 1
                        unchanged_urls.append(url)
                        continue
                else:
                    html = fetch(url)
                    rows = build_rows_for_candidate(
                        url, html, meta={**meta, "published": cand.get("published")})
            except ChallengedError:
                # The site's own access control, not a technical fetch
                # failure -- never retried, never routed around (see
                # plain_get's docstring). Index only what the source
                # already published openly in its own feed.
                n_challenged += 1
                row = build_feed_item_row(cand, meta=meta)
                rows = [row] if row is not None else []
            except Exception as exc:  # noqa: BLE001 -- one candidate must
                # not take the whole pack down; named, never swallowed.
                errors[url] = f"{type(exc).__name__}: {exc}"
                continue
            for r in rows:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_rows += 1

    hn_info = None
    if hn_route:
        hn_info = {"accounting": HO.accounting(hn_outcomes),
                   "hosts_consent": {b: sum(1 for v in hn_hit["hn"]["hosts"].values() if v == b)
                                     for b in sorted(set(hn_hit["hn"]["hosts"].values()))},
                   "stories_in_window": hn_hit["hn"]["stories_in_window"]}
        print(f"[refresh-direct stage] {pack_id}: {json.dumps(hn_info, sort_keys=True)}",
              file=sys.stderr)
    if candidates and n_rows == 0 and n_challenged == 0 and len(errors) == len(candidates):
        raise SystemExit(
            f"REFUSED: {pack_id}: all {len(candidates)} fetched candidate(s) "
            f"errored and 0 rows extracted -- a dead channel, not a quiet "
            f"day: {json.dumps(dict(list(errors.items())[:5]))[:600]}")

    gated_dir = W / f"gated-{pack_id}"
    # A previous run's `passages_ok.jsonl` must never be staged as this run's:
    # with 0 passages the gate writes nothing ("nothing fetched yet"), and the
    # old file was re-staged. Live 2026-10-03: `refresh-direct-gh-releases-
    # 20261003` reported `candidates: 0` (20 of 20 older than 3 d) yet staged
    # 54 rows, the 2026-09-28 rows again (same sha256 as the 10-02 pack).
    shutil.rmtree(gated_dir, ignore_errors=True)
    with out_log.open("w") as fh:
        # `sys.executable`, not a hardcoded interpreter: cytir.consent.gate has
        # no special dependency (pure stdlib + consent_resolver, itself
        # pure stdlib), so it runs correctly under whatever interpreter is
        # ALREADY running this orchestrator -- which is required to be the
        # langgate interpreter anyway, for `build_rows_for_candidate`'s own
        # in-process fastText call (see deploy/scripts/refresh-lane-direct-
        # compose-run.sh's header).
        result = subprocess.run(
            [sys.executable, "-m", "cytir.consent.gate", "gate",
             "--passages", str(passages_path), "--out-dir", str(gated_dir),
             "--direct-crawl", "--search-indexing"],
            cwd=str(W),
            # `-m` puts the cwd first; this checkout's root goes on PYTHONPATH so its own `aw` runs
            env={**os.environ, "PYTHONPATH": os.pathsep.join(
                p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if p)},
            stdout=fh, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise SystemExit(f"{pack_id}: consent_gate.py FAILED "
                         f"(exit={result.returncode}) -- see {out_log}")

    gated_path = gated_dir / "passages_ok.jsonl"
    collection = D.NSC.news_stage_collection_name(
        _dt.date.today().isoformat(), meta["label"] + label_suffix, mrl_dim)
    if not gated_path.is_file() or not gated_path.stat().st_size:
        print(f"[refresh-direct stage] {pack_id}: 0 rows cleared consent "
             f"gate ({n_rows} extracted, {n_challenged} challenged -> "
             f"feed-item, {len(errors)} fetch/extract errors) -- "
             "nothing to stage", file=sys.stderr)
        return {"staged": 0, "collection": collection,
               "candidates": len(candidates), "challenged": n_challenged,
               "candidates_discovered": len(all_candidates), "errors": errors,
               "discovery_errors": discovery_errors,
               "derivatives": derivatives, "hn": hn_info,
               "unchanged": n_unchanged, "unchanged_repos": unchanged_repos,
               "repo_heads": discovery.get("repo_heads") or {}}

    before = D.iface_bytes(egress_iface)
    t0 = time.monotonic()
    stage_log = out_log.with_suffix(".b1stage.log")
    stage_env = {**os.environ, "AW_FOLD_ACCENTS": "1"}
    with stage_log.open("w") as fh:
        result = subprocess.run(
            [D._STABLE_ENV["AW_QPY"], "-m", "cytir.index.stage_pack_b1",
             "--in", str(gated_path), "--collection", collection,
             "--qdrant-url", staging_url, "--mrl-dim", str(mrl_dim),
             "--consent-state", "resolved",
             "--consent-basis", "robots-live-resolve",
             "--consent-source", "consent_gate.py direct-crawl receipt",
             "--consent-checked-at", _dt.date.today().isoformat(),
             # RULING 39: docs rows were cut at chunk@4; without this flag cytir.index.stage_pack_b1 restamps
             # every row with its build default (chunk 1) and the r4 cut shipped a false epoch.
             *(["--preserve-rule-epochs"] if docs_chunk_selection(meta) else [])],
            # `-m` puts the cwd first; this checkout's root goes on PYTHONPATH so its own `aw` runs
            cwd=str(W), env={**stage_env, "PYTHONPATH": os.pathsep.join(
                p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if p)},
            stdout=fh, stderr=subprocess.STDOUT, text=True)
    elapsed = max(time.monotonic() - t0, 1e-6)
    after = D.iface_bytes(egress_iface)
    mbps = (after - before) / 1e6 / elapsed if after >= before else None
    if result.returncode:
        raise SystemExit(f"{pack_id}: cytir.index.stage_pack_b1 FAILED "
                         f"(exit={result.returncode}) -- see {stage_log}")

    from cytir.core.qdrant import qdrant_client  # noqa: PLC0415
    client = qdrant_client(staging_url, timeout=120)
    staged = client.get_collection(collection).points_count
    print(f"[refresh-direct stage] {pack_id}: "
         f"candidates_discovered={len(all_candidates)} "
         f"candidates_fetched={len(candidates)} extracted_rows={n_rows} "
         f"challenged={n_challenged} unchanged={n_unchanged} unchanged_repos={unchanged_repos} "
         f"heads_lookup_failed={heads_lookup_failed} "
         f"staged={staged} errors={len(errors)} "
         f"elapsed={elapsed:.0f}s "
         f"egress={'n/a' if mbps is None else f'{mbps:.3f} MB/s'}",
         file=sys.stderr)
    return {"staged": staged, "collection": collection,
           "candidates": len(candidates), "challenged": n_challenged,
           "candidates_discovered": len(all_candidates), "errors": errors,
               "discovery_errors": discovery_errors,
               "derivatives": derivatives, "hn": hn_info,
               "unchanged": n_unchanged, "unchanged_repos": unchanged_repos,
               "repo_heads": discovery.get("repo_heads") or {}}


def run_pack(pack_id: str, staged_collection: str, *, out_log: pathlib.Path,
            collection: str = TWIN.collection()) -> dict:
    """Read-only pack step, identical call shape to
    `hn_discussion_daily.run_pack` -- same `materialize_news_pack.py`,
    same twin, no lock, no twin/prod write."""
    today = _dt.date.today().isoformat()
    aged_ids_path = W / f"empty-aged-ids-{pack_id}.txt"
    if not aged_ids_path.exists():
        aged_ids_path.write_text("")
    out_dir = W / f"pack-{pack_id}"
    with out_log.open("w") as fh:
        result = subprocess.run(
            [D._STABLE_ENV["AW_QPY"], str(MATERIALIZE_SCRIPT),
             "--staging-url", D._STABLE_ENV.get(
                 "AW_STAGING_URL", "http://localhost:6398"),
             "--staging-collection", staged_collection,
             "--twin-url", TWIN_URL, "--collection", collection,
             "--aged-ids", str(aged_ids_path), "--cutoff", today,
             "--out-dir", str(out_dir)],
            cwd=str(W), stdout=fh, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise SystemExit(f"{pack_id}: pack build FAILED (exit="
                         f"{result.returncode}) -- see {out_log}")
    summary = D.parse_pack_summary(out_log.read_text())
    art = pathlib.Path(summary["artifact"])
    if not art.is_file():
        art = W / summary["artifact"]
    if not art.is_file():
        raise SystemExit(f"pack summary names {art}, which does not exist")
    return {"artifact": art, "rows": summary["rows"]}


SINCE_CAP_DAYS = 30


def effective_since_days(pack_id: str, floor: int, *, state: dict | None = None,
                         now: _dt.datetime | None = None) -> int:
    """The fetch window: `--since-days` is a FLOOR; the window grows to cover
    the time since this pack's last SUCCESSFUL run (state.json) plus one day
    of margin, capped at SINCE_CAP_DAYS. One missed day (a stopped timer, a
    failed run) is then harmless: 09-29..10-01 vendor-feed items were never
    packed when the window was a fixed 3 days. No state, or an unreadable
    one, gives the floor."""
    if state is None:
        try:
            state = json.loads(STATE_PATH.read_text())
        except (OSError, ValueError):
            return floor
    last = ((state.get("packs") or {}).get(pack_id) or {}).get("last_success")
    if not last:
        return floor
    try:
        then = _dt.datetime.strptime(last, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=_dt.timezone.utc)
    except ValueError:
        return floor
    now = now or _dt.datetime.now(_dt.timezone.utc)
    gap = (now - then).total_seconds() / 86400
    return max(floor, min(SINCE_CAP_DAYS, int(gap) + 2))


def free_pack_suffix(pack_id: str, base: str, today: str,
                     queue_dir: pathlib.Path | None = None) -> str:
    """The pack-id suffix for this run: `base`, or `-r2`, `-r3`... when the
    queue already holds a folder with that id (any state, DONE or not). A
    second same-day run used to reuse the id; the drain skips a DONE folder,
    so that run's rows were lost with no record (10-02, BAAI/AREX-2).
    The suffix also names the staging collection, so it is minted before
    staging. Never overwrites an existing queue folder."""
    queue_dir = queue_dir or QUEUE_DIR
    day = today.replace("-", "")
    label = f"{PACK_PREFIX}{PACK_META[pack_id]['label']}"
    if not (queue_dir / f"{label}{base}-{day}").exists():
        return base
    n = 2
    while (queue_dir / f"{label}-r{n}-{day}").exists():
        n += 1
    return f"-r{n}"


def adapter_opts_from_args(args) -> dict | None:
    """Batch knobs for the roster-driven github_repo_docs pack (task 996); None when none is set."""
    opts = {"repo_offset": getattr(args, "repo_offset", 0) or 0,
            "repo_limit": getattr(args, "repo_limit", None),
            "repo_in_index": {"any": None, "false": False, "true": True}[
                getattr(args, "repo_in_index", "any") or "any"],
            "preserve_root": getattr(args, "preserve_root", None),
            "workers": getattr(args, "workers", 1) or 1}
    return opts if (opts["repo_offset"] or opts["repo_limit"] is not None
                    or opts["repo_in_index"] is not None or opts["preserve_root"]
                    or opts["workers"] != 1) else None


def run_one(pack_id: str, args) -> dict:
    today = _dt.date.today().isoformat()
    base_suffix = getattr(args, "pack_suffix", "") or ""
    suffix = free_pack_suffix(pack_id, base_suffix, today)
    if suffix != base_suffix:
        print(f"[refresh-direct] {pack_id}: the queue already holds "
              f"{PACK_PREFIX}{PACK_META[pack_id]['label']}{base_suffix}-"
              f"{today.replace('-', '')}; this re-run uses suffix {suffix}",
              file=sys.stderr)
    stage_out_log = W / f"stage-{pack_id}-{today}.out"
    pack_out_log = W / f"pack-{pack_id}-{today}.out"

    staged = run_stage(pack_id, cadence_path=pathlib.Path(args.cadence_table),
                       out_log=stage_out_log, mrl_dim=args.mrl_dim,
                       staging_url=args.staging_url,
                       since_days=effective_since_days(
                           pack_id, args.since_days),
                       label_suffix=suffix,
                       egress_iface=args.egress_iface,
                       twin_url=args.twin_url, collection=args.collection,
                       adapter_opts=adapter_opts_from_args(args))
    if staged["staged"] == 0:
        print(f"[refresh-direct] {pack_id}: 0 points staged this run "
             "(a real quiet run, not necessarily an error -- see the "
             "candidates/errors counts above)", file=sys.stderr)
        return {"pack_id": pack_id, "staged": 0, "packed": 0, "queued": False,
                "candidates": staged["candidates"],
                "candidates_discovered": staged["candidates_discovered"],
                "fetch_errors": len(staged["errors"]),
                "derivatives_skipped": len(staged.get("derivatives") or {}),
                **({"unchanged_repos": staged["unchanged_repos"]} if "unchanged_repos" in staged else {}),
                "discovery_errors": staged["discovery_errors"],
                **({"hn": staged["hn"]} if staged.get("hn") else {})}

    packed = run_pack(pack_id, staged["collection"], out_log=pack_out_log,
                      collection=args.collection)
    print(f"[refresh-direct pack] {pack_id}: rows={packed['rows']} "
         f"artifact={packed['artifact'].name}", file=sys.stderr)

    rows = C.read_pack_points(str(packed["artifact"].parent))
    merged = C.merge_rows(rows, require_published=PACK_META[pack_id].get("route") not in ("docs_html", "docs_markdown"))
    kept_urls = {(r.get("payload") or {}).get("url", "") for r in merged["kept"]}
    print(f"[refresh-direct merge] {pack_id}: kept={len(merged['kept'])} "
         f"urls={len(kept_urls)} undated_dropped={len(merged['undated'])}",
         file=sys.stderr)

    tombstones: list = []
    pack_full_id = (f"{PACK_PREFIX}{PACK_META[pack_id]['label']}"
                    f"{suffix}-{today.replace('-', '')}")
    derivatives = staged.get("derivatives") or {}
    derivative_tombs: list = []
    derivative_marker_rows: list = []
    if not args.no_tombstones:
        from cytir.core.qdrant import qdrant_client  # noqa: PLC0415
        client = qdrant_client(args.twin_url, timeout=120)
        C.assert_dense_matches_collection(
            merged["kept"], client=client, collection=args.collection,
            log=lambda msg, **kw: print(msg, file=sys.stderr, **kw))
        tombstones = C.compute_superseded_tombstones(
            client, args.collection, merged["kept"])
        already_claimed = C.other_queued_tombstone_ids(
            str(QUEUE_DIR), pack_full_id)
        tombstones = C.dedupe_tombstones_against_queue(
            tombstones, already_claimed,
            log=lambda msg, **kw: print(msg, file=sys.stderr, **kw))
        C.assert_no_cross_pack_tombstone_collision(tombstones, already_claimed)
        if derivatives:
            have = {t["id"] for t in tombstones}
            derivative_tombs = [t for t in derivative_tombstones(
                client, args.collection, list(derivatives),
                marker_rows=derivative_marker_rows)
                if t["id"] not in have and t["id"] not in already_claimed]
            tombstones = tombstones + derivative_tombs
            print(f"[refresh-direct derivatives] {len(derivatives)} derivative "
                  f"url(s) skipped; {len(derivative_tombs)} already-landed "
                  f"point(s) tombstoned; {len(derivative_marker_rows)} marker "
                  "row(s) left alone", file=sys.stderr)

    stage_only = getattr(args, "stage_only", None)
    out_dir = (pathlib.Path(stage_only) if stage_only else QUEUE_DIR) / pack_full_id
    manifest = C.write_pack(
        str(out_dir), kept=merged["kept"], tombstones=tombstones,
        pack_id=pack_full_id, family="news", collection=args.collection,
        days=[today],
        extra_manifest={
            "arm": "refresh_lane_direct", "source_pack_id": pack_id,
            "candidates": staged["candidates"],
            "challenged_feed_item_only": staged["challenged"],
            "fetch_errors": staged["errors"],
            "undated_dropped": len(merged["undated"]),
            "unchanged_skipped": staged.get("unchanged", 0),
            "urls": len(kept_urls), "staged_points": staged["staged"],
            "derivatives_skipped": derivatives,
            "derivative_tombstones": len(derivative_tombs),
            "derivative_marker_rows_skipped": len(derivative_marker_rows),
            **({"hn_outlinks": staged["hn"]} if staged.get("hn") else {}),
        })
    if stage_only:
        # a STAGED pack has no READY: the retired `l1_backfill_compose.enqueue_staged_pack` (task 1315) moved
        # it into the queue and wrote READY last, refusing while another content pack was open
        (out_dir / "READY").unlink()
    if not stage_only and staged.get("repo_heads"):
        # task 2630: advance the head-sha skip state only now that the pack is QUEUED; a repo with
        # a fetch/extract error (its URL is a key of `errors`) stays unrecorded and is retried
        bad = {k for k in staged["repo_heads"]
               if any(u.startswith(f"https://github.com/{k}/blob/") for u in staged["errors"])}
        from cytir.core import paths as _paths  # noqa: PLC0415
        try:   # the pack is already queued: a heads write failure must not turn it into a crash
            GRD.record_heads(GRD.heads_path(adapter_opts_from_args(args)),
                             {k: v for k, v in staged["repo_heads"].items() if k not in bad},
                             pack_id=pack_full_id)
        except (OSError, _paths.SettingRefused) as exc:
            print(f"[refresh-direct] heads not recorded: {exc}", file=sys.stderr)
    print(json.dumps(manifest, indent=1))
    return {"pack_id": pack_id, "staged": staged["staged"],
           "packed": len(merged["kept"]), "queued": not stage_only, "staged_dir": str(out_dir),
           "queue_pack_id": pack_full_id,
           "candidates": staged["candidates"],
           "candidates_discovered": staged["candidates_discovered"],
           "fetch_errors": len(staged["errors"]),
           "derivatives_skipped": len(derivatives),
           "derivative_tombstones": len(derivative_tombs),
           "pack_suffix": suffix,
           **({"unchanged_repos": staged["unchanged_repos"]} if "unchanged_repos" in staged else {}),
           "discovery_errors": staged["discovery_errors"],
                **({"hn": staged["hn"]} if staged.get("hn") else {})}


def main(argv=None, prog=None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    ap.add_argument("--pack-id", action="append", default=None,
                    help="repeatable; default: every pack in PACK_META")
    ap.add_argument("--cadence-table",
                    default=str(data_path("refresh-cadence.v1.json")))
    ap.add_argument("--collection", default=TWIN.collection())
    ap.add_argument("--twin-url", default=TWIN_URL)
    ap.add_argument("--staging-url", default="http://localhost:6398")
    ap.add_argument("--mrl-dim", type=int, default=256)
    ap.add_argument("--since-days", type=int, default=14,
                    help="only fetch candidates published within this many "
                         "days (a long-lived feed's full history is a "
                         "first-run flood, not a weekly refresh)")
    ap.add_argument("--pack-suffix", default="",
                    help="e.g. -r2: distinguishes a SAME-DAY re-run's staging "
                         "collection and queue pack id from the first run's")
    ap.add_argument("--repo-offset", type=int, default=0,
                    help="github_repo_docs only: skip the first N repos of the merged route list "
                         "(routes, then repo_list order; after --repo-in-index)")
    ap.add_argument("--repo-limit", type=int, default=None,
                    help="github_repo_docs only: take at most M repos from the offset")
    ap.add_argument("--repo-in-index", choices=("any", "false", "true"), default="any",
                    help="github_repo_docs only: keep list rows by their `in_index` flag "
                         "(false = net-new repos); applied before offset/limit")
    ap.add_argument("--preserve-root", default=None, metavar="DIR",
                    help="github_repo_docs only: reuse this fetch root; a repo with a valid receipt "
                         "at its pinned SHA is not fetched again")
    ap.add_argument("--workers", type=int, default=1,
                    help="github_repo_docs only: acquirer --workers for the batch")
    ap.add_argument("--no-tombstones", action="store_true")
    ap.add_argument("--stage-only", default=None, metavar="DIR",
                    help="write the pack under DIR WITHOUT a READY marker instead of into the "
                         "promotion queue (task 972: one content pack at a time); enqueue it later "
                         "by hand (the l1_backfill_compose enqueue helper is retired, task 1315)")
    ap.add_argument("--egress-iface", default=D.DEFAULT_EGRESS_IFACE)
    args = ap.parse_args(argv)

    # Same defended-against class as hn_discussion_stage.sh/
    # news_refresh_stage.sh: this module's `build_rows_for_candidate`
    # calls aw_common.process_html in-process, which runs fastText
    # language detection -- degraded (URL-heuristic-only) mode is fine for
    # offline tests but must never silently ship into a real index.
    if os.environ.get("AW_LANGGATE_OPTIONAL") == "1":
        raise SystemExit("ABORT: AW_LANGGATE_OPTIONAL=1 is forbidden for "
                         "refresh-lane-direct indexing")
    require_docs_chunker()  # task 976: docs rows need the tokenizer; refuse, never cut under another ruler

    W.mkdir(parents=True, exist_ok=True)
    pack_ids = args.pack_id or default_pack_ids()
    unknown = [p for p in pack_ids if p not in PACK_META]
    if unknown:
        ap.error(f"unregistered pack id(s): {unknown} -- add to PACK_META first")
    switched_off = [p for p in pack_ids if not pack_armed(p)]
    if switched_off:
        ap.error(f"{switched_off}: switched off (set "
                 f"{PACK_META[switched_off[0]]['gated_by']}=1; "
                 f"{PACK_META[switched_off[0]].get('gate_note') or 'it stays off until Tim merges the ADR-098 Ruling 2 amendment'})")

    results = []
    for pack_id in pack_ids:
        try:
            results.append(run_one(pack_id, args))
        except SystemExit as exc:  # a REFUSED/FAILED pack must not skip the others
            results.append({"pack_id": pack_id, "failed": str(exc.code)})
            print(f"[refresh-direct] {pack_id}: {exc.code}", file=sys.stderr)
    write_state(results)
    print(json.dumps({"schema": "aw.refresh-lane-direct-compose-run.v1",
                      "results": results}, indent=1))
    bad = [r["pack_id"] for r in results
           if r.get("failed") or r.get("discovery_errors")
           or partial_fetch_failure(r)]
    if bad:
        print(f"[refresh-direct] EXIT 1: failed or partially-failed discovery "
              f"for {bad} -- see {STATE_PATH}", file=sys.stderr)
        return 1
    return 0


def partial_fetch_failure(result: dict) -> bool:
    n, bad = int(result.get("candidates") or 0), int(result.get("fetch_errors") or 0)
    return bad >= FETCH_ERROR_MIN and bad > FETCH_ERROR_SHARE * n


def write_state(results: list[dict], *, state_path: pathlib.Path | None = None,
                next_path: pathlib.Path | None = None,
                now: _dt.datetime | None = None) -> dict:
    """The lane's state file + NEXT.md (task 693). `last_success` per pack
    advances only on a run with no failure and no discovery error, so a
    pack that keeps refusing goes stale and the monitor's `refresh_direct`
    check (deploy/scripts/aw-monitor-check.sh) alarms on its age. NEXT.md is
    regenerated from the state, never hand-edited, so it cannot rot."""
    state_path = state_path or STATE_PATH
    next_path = next_path or NEXT_PATH
    now = now or _dt.datetime.now(_dt.timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        state = {}
    state.setdefault("schema", "aw.refresh-lane-direct-state.v1")
    packs = state.setdefault("packs", {})
    for r in results:
        entry = packs.setdefault(r["pack_id"], {})
        ok = (not r.get("failed") and not r.get("discovery_errors")
              and not partial_fetch_failure(r))
        entry["last_run"] = stamp
        entry["last_result"] = r
        if ok:
            entry["last_success"] = stamp
        if ok and r.get("queued"):
            entry["last_queued"] = {"at": stamp, "pack": r.get("queue_pack_id"),
                                    "points": r.get("packed")}
    state["updated"] = stamp
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    tmp.replace(state_path)
    lines = ["# refresh-lane direct compose -- NEXT (generated; do not edit)", "",
             f"Updated {stamp}. Daily unit: aw-refresh-lane-direct-compose.timer "
             "(its unit file is under deploy/). State: state.json beside this file. Never "
             "applies twin or prod; ADR-110's drain lands the queued packs.", "",
             "| pack | last run | last success | last queued | last result |",
             "|---|---|---|---|---|"]
    for pid, e in sorted(packs.items()):
        res = e.get("last_result", {})
        what = res.get("failed") or (
            f"cand {res.get('candidates')}/{res.get('candidates_discovered')}, "
            f"packed {res.get('packed')}, fetch_err {res.get('fetch_errors')}, "
            f"disc_err {len(res.get('discovery_errors') or {})}")
        lq = e.get("last_queued") or {}
        lines.append(f"| {pid} | {e.get('last_run')} | {e.get('last_success')} | "
                     f"{lq.get('pack')} ({lq.get('points')} pts) | {str(what)[:160]} |")
    lines += ["", "Next: the timer fires daily 11:30 UTC. A `REFUSED` result means "
              "discovery or fetch hit an error with 0 usable output -- fix the "
              "cause, never read it as a quiet day. The monitor check "
              "`refresh_direct` alarms when any pack's last_success is older "
              "than 36 h."]
    next_path.write_text("\n".join(lines) + "\n")
    return state


if __name__ == "__main__":
    raise SystemExit(main())
