"""Build cytir/serve/authority_hosts.json — the ADR-019 product-1
per-domain AUTHORITY signal, as a serving-side host->authority table.

ADR-019 product-1 ("authority signal — ship first, no new pages"): a per-domain
endorsement/citation score, log-normalized to a bounded [0,1] feature, wired into
the live fusion as a flag-gated additive boost (AW_AUTHORITY, retrieval.py stage-2).
The signal is the endorsement graph magnitude (how many independent, class-weighted
sources cite a host) — the same `endorse_score` prep_discovery_queue.py ranks the
discovery queue by, re-aggregated at HOST granularity over the WHOLE target set
(NOT deduped against the live index — serving needs authority for the hosts that
ARE in the index, i.e. the endorsement TARGETS).

Two ADR-019 prerequisites, applied HERE at build time (not at serving):

  * UNWRAP archive/proxy wrappers (Rule 1 — "the big one"). web.archive.org is the
    #1 raw citation in EN/DE/FR; credit MUST flow to the REAL target, not the
    wrapper. `web.archive.org/web/<ts>/<realurl>` -> realurl; same for archive.today
    / archive.ph / archive.wikiwix.com / webcache.googleusercontent.com. Without
    this ~40% of the top signal is link-rot noise (ADR-019).

  * SOURCE-CLASS TAXONOMY (Rule 2) decides how much authority a host earns:
      primary-artifact (gov/edu/health/ref/tech) -> DEEP authority, full weight
      publisher       (news)                     -> pointer+THRESHOLD, half weight,
                                                    only above PUBLISHER_MIN_SCORE
      media / low-value (youtube/reddit/social/CDN/...) -> signal-only, EXCLUDED
    A media host must NOT get the same authority weight as a primary source.

Reuses prep_discovery_queue's exact aggregation (SEED_TRUST class weights,
SEED_VOTE_CAP, ENDORSER_HOST_CAP), classifier, and noise denylist — so the table
is consistent with the deployed discovery queue, only re-scoped to serving.

Output schema (mirrors hub_hosts.json for the serving loader):
  {"_comment": ..., "meta": {...}, "hosts": {host: {authority, class, raw, scale}}}
where `authority` in [0,1] is the FINAL bounded feature (class_scale * lognorm),
the value retrieval._authority() reads and AW_AUTHORITY_W scales.

READ-ONLY offline build (endorse_all.json only, no network, no twin/prod writes).

  cyt ops build-authority-table \
      --endorse ~/aw-data/endorse_all.json \
      --out cytir/serve/authority_hosts.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from urllib.parse import unquote, urlparse

APP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cytir.core.serve_layout import SERVER_REL  # noqa: E402
from cytir.acquire.prep_discovery_queue import (  # reuse the EXACT deployed logic
    ENDORSER_HOST_CAP,
    SEED_TRUST,
    SEED_TRUST_DEFAULT,
    SEED_VOTE_CAP,
    classify,
    is_noise,
    norm_host,
    reg_domain,
)

# ADR-019 Rule-1 archive/proxy wrappers — unwrap to the REAL cited target so the
# authority credit flows to the source, not the wrapper (web.archive.org is the
# #1 raw citation in EN/DE/FR). Each entry: (host-substring, unwrap-fn(url)->realurl).
_WEB_ARCHIVE_RE = re.compile(r"^https?://[^/]*web\.archive\.org/web/[0-9]+[a-z_]*/(.+)$", re.I)
_ARCHIVE_TODAY_RE = re.compile(r"^https?://[^/]*archive\.(?:today|ph|is|li|vn|fo|md)/"
                               r"(?:[0-9]{4,}|newest|[A-Za-z0-9]{5})?/?(https?://.+)$", re.I)
_WIKIWIX_RE = re.compile(r"[?&]url=(https?[^&]+)", re.I)
_GCACHE_RE = re.compile(r"https?://[^/]*webcache\.googleusercontent\.com/search\?q=cache:"
                        r"(?:[^:]+:)?(https?://.+)$", re.I)


def unwrap(url: str, _depth: int = 0) -> str:
    """Resolve an archive/proxy wrapper URL to the REAL cited target (ADR-019
    Rule 1). Idempotent + recursive (nested wraps), bounded. Non-wrapped URLs
    pass through unchanged."""
    if _depth > 3 or not url:
        return url
    u = url.strip()
    m = _WEB_ARCHIVE_RE.match(u)
    if m:
        return unwrap(m.group(1), _depth + 1)
    m = _ARCHIVE_TODAY_RE.match(u)
    if m:
        return unwrap(m.group(1), _depth + 1)
    low = u.lower()
    if "webcache.googleusercontent.com" in low:
        m = _GCACHE_RE.match(u)
        if m:
            return unwrap(m.group(1), _depth + 1)
    if "wikiwix.com" in low or "archive.wikiwix" in low:
        m = _WIKIWIX_RE.search(u)
        if m:
            return unwrap(unquote(m.group(1)), _depth + 1)
    return url


# Rule-2 taxonomy: prep_discovery_queue.classify() emits gov/edu/health/ref/tech/
# news/other/commercial. Map to the ADR-019 source-class + a per-class authority
# SCALE (a media host must not weigh like a primary source):
#   primary-artifact -> 1.0   (gov/edu/health/ref/tech: deep, always)
#   publisher        -> 0.5   (news: pointer + threshold, see PUBLISHER_MIN_SCORE)
#   media/other      -> EXCLUDED (signal-only / unclassified: no ranking authority)
# Declared per-host class labels that REPLACE the graph-derived class in the output (task 1065, ruling 59).
# Applied after the taxonomy filter, so source_class, scale and authority are those of the graph class: only the
# display `class` changes. `corporate` is not in CLASS_TAXONOMY on purpose (a corporate site earns authority only
# through the class the graph gave it); a host the filter drops stays dropped.
HOST_CLASS_OVERRIDE = {"roche.ch": "corporate"}

CLASS_TAXONOMY = {
    "gov": ("primary-artifact", 1.0),
    "edu": ("primary-artifact", 1.0),
    "health": ("primary-artifact", 1.0),
    "ref": ("primary-artifact", 1.0),
    "tech": ("primary-artifact", 1.0),
    "news": ("publisher", 0.5),
    # other / commercial -> not in the table (excluded); see build()
}
# Publisher (news) hosts earn authority only above this endorse_score — ADR-019
# Rule-2 "publisher = pointer + threshold": a once-cited outlet is not an authority.
PUBLISHER_MIN_SCORE = 15.0

# Mega-platform APEXES that are aggregation hubs, not document authorities at apex
# granularity (ADR-019 flags github needs sub-classing: repo=ingest, gist/issue=
# pointer). Excluded so the apex can't ride authority onto a general query — the
# hubfloor lever only covers the gov class, so these need an explicit carve.
EXCLUDE_HOSTS = frozenset({
    "github.com", "gitlab.com", "bitbucket.org",
    # archive/cache wrapper APEXES: unwrap() re-homes /web/<ts>/ links to the real
    # target, but the bare wrapper homepage (https://web.archive.org/) survives as
    # its own low-value target — it is a wrapper, not an authority (ADR-019 Rule 1).
    "web.archive.org", "archive.org", "archive.today", "archive.ph", "archive.is",
    "archive.wikiwix.com", "webcache.googleusercontent.com",
})

# ADR-019 Rule-2 "media / low-value = signal-only": leaks the prep noise denylist
# (is_noise) does NOT catch but which classify() mis-grants a primary class via
# graph-provenance (a single edu/health seed linking them). Substring-matched on
# the host. Two groups:
#   (a) media / streaming / social / retail (ADR-019's youtube/imdb/discogs/allmusic
#       examples + observed leaks: soundcloud/twitch/imdb/unsplash/hatena/amazon.*)
#   (b) Wikimedia INFRASTRUCTURE (ADR-019 Phase-1: filter toolforge/wmflabs/wmcloud/
#       geohack/iabot + the MediaWiki-software/foundation/meta hosts). The CONTENT
#       hubs wikidata.org / commons.wikimedia.org / *.wikipedia.org are KEPT.
MEDIA_INFRA_SUB = (
    # (a) media / streaming / social / stock-media / retail
    "soundcloud.com", "twitch.tv", "vimeo.com", "dailymotion.com", "deezer.com",
    "genius.com", "last.fm", "imdb.com", "discogs.com", "allmusic.com",
    "metacritic.com", "unsplash.com", "giphy.com", "imgur.com", "500px.com",
    "hatena.ne.jp", "mastodonshare.com", "getpocket.com",
    "amazon.co.jp", "amazon.ca", "amazon.es", "amazon.in", "amazon.cn",
    "amazon.nl", "amazon.se", "amazon.com.br", "amazon.com.mx", "amazon.co",
    # (b) Wikimedia infra / tool hosts (NOT the content projects above)
    "wmflabs.org", "wmcloud.org", "toolforge.org", "mediawiki.org",
    "meta.wikimedia.org", "foundation.wikimedia.org", "species.wikimedia.org",
    "wikimediafoundation.org", "iabot", "geohack",
)


def _media_infra(h: str) -> bool:
    d = (h or "").lower()
    return any(s in d for s in MEDIA_INFRA_SUB)


def host_authority_scores(endorse: dict) -> tuple[dict, dict]:
    """Re-aggregate the edge graph at HOST granularity WITH archive-unwrap.

    Mirrors prep_discovery_queue.host_scores_from_edges (same SEED_TRUST class
    weights, SEED_VOTE_CAP per-URL, ENDORSER_HOST_CAP per-endorser-host) so scores
    are consistent with the deployed queue — but the target host is taken AFTER
    unwrapping the archive/proxy wrapper, and NO live-index dedup (serving needs
    authority for indexed targets). Returns (host_score, host_classes)."""
    host_score: dict[str, float] = defaultdict(float)
    host_classes: dict[str, set] = defaultdict(set)
    n_unwrapped = 0
    for _seed, info in (endorse.get("edges") or {}).items():
        sc = info.get("class")
        w = SEED_TRUST.get(sc, SEED_TRUST_DEFAULT)
        seed_host: dict[str, float] = defaultdict(float)   # this endorser only
        for tgt, cnt in (info.get("targets") or {}).items():
            real = unwrap(tgt)
            if real != tgt:
                n_unwrapped += 1
            try:
                netloc = urlparse(real if "//" in real else "http://" + real).netloc
                h = norm_host(netloc)
            except Exception:  # noqa: BLE001
                continue
            if not h or "." not in h:
                continue
            try:
                strength = min(float(cnt), SEED_VOTE_CAP)
            except (TypeError, ValueError):
                strength = 1.0
            seed_host[h] += strength
            if sc:
                host_classes[h].add(sc)
        for h, s in seed_host.items():                     # ENDORSER_HOST_CAP
            host_score[h] += w * min(s, ENDORSER_HOST_CAP)
    host_score["__n_unwrapped__"] = n_unwrapped
    return host_score, host_classes


def build(endorse: dict, min_score: float = 3.0):
    host_score, host_classes = host_authority_scores(endorse)
    n_unwrapped = int(host_score.pop("__n_unwrapped__", 0))

    # Pass 1 — filter to the eligible authority hosts (class + noise/media + apex),
    # collecting (host, raw, source_class, scale). A host earns authority only via
    # the primary-artifact / publisher classes; media/infra/other are signal-only.
    cand = []
    cls_ct = Counter()
    dropped_noise = dropped_media = dropped_class = dropped_excl = dropped_lowscore = 0
    for h, s in host_score.items():
        if h in EXCLUDE_HOSTS or reg_domain(h) in EXCLUDE_HOSTS:
            dropped_excl += 1
            continue
        if is_noise(h, "host"):          # prep noise denylist -> signal-only
            dropped_noise += 1
            continue
        if _media_infra(h):              # ADR-019 Rule-2 media/infra -> signal-only
            dropped_media += 1
            continue
        cls, _why = classify(h, host_classes.get(h, set()))
        tax = CLASS_TAXONOMY.get(cls)
        if tax is None:                  # other / commercial -> excluded
            dropped_class += 1
            continue
        source_class, scale = tax
        if source_class == "publisher" and s < PUBLISHER_MIN_SCORE:
            dropped_class += 1           # ADR-019 publisher threshold
            continue
        if s < min_score:                # long faint tail: one stray edge, not authority
            dropped_lowscore += 1
            continue
        cand.append((h, s, cls, source_class, scale))

    # log-normalization denominator: log1p(MAX raw). Citation counts are extremely
    # fat-tailed (p50=0.8, p99=31, max~2700 in the post-242 graph) — a percentile
    # denom saturates the whole authoritative head at 1.0. log1p(s)/log1p(max) is
    # the ADR-019 "log-scaled" feature with real spread across [0,1]: the top hub
    # -> 1.0, a strong national authority (~raw 400) -> ~0.75, a modestly-cited
    # source (~raw 30) -> ~0.44, a barely-cited one (~raw 3) -> ~0.18.
    max_raw = max((s for _h, s, *_ in cand), default=1.0)
    denom = math.log1p(max_raw) or 1.0

    hosts = {}
    for h, s, cls, source_class, scale in cand:
        lognorm = min(1.0, math.log1p(s) / denom)
        authority = round(scale * lognorm, 4)
        if authority <= 0.0:
            continue
        cls = HOST_CLASS_OVERRIDE.get(h, cls)
        hosts[h] = {"authority": authority, "class": cls,
                    "source_class": source_class,
                    "raw": round(s, 2), "scale": scale}
        cls_ct[cls] += 1

    hosts = dict(sorted(hosts.items(), key=lambda kv: -kv[1]["authority"]))
    meta = {
        "n_hosts": len(hosts), "n_unwrapped_targets": n_unwrapped,
        "min_score": min_score, "max_raw": round(max_raw, 2),
        "log_denom": round(denom, 4),
        "publisher_min_score": PUBLISHER_MIN_SCORE,
        "by_class": dict(cls_ct),
        "dropped": {"noise": dropped_noise, "media_infra": dropped_media,
                    "class_other": dropped_class, "platform_apex": dropped_excl,
                    "low_score": dropped_lowscore},
        "source": "endorse_all.json edge graph (post-242, 641k edges), "
                  "host-granular, archive-unwrapped, no live-index dedup",
    }
    return hosts, meta


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--endorse", default=os.path.expanduser("~/aw-data/endorse_all.json"))
    ap.add_argument("--out", default=os.path.join(APP, SERVER_REL,
                                                  "authority_hosts.json"))
    ap.add_argument("--min-score", type=float, default=3.0,
                    help="drop hosts below this raw endorsement score (the faint "
                         "one-stray-edge tail is not authority; 0 = keep all)")
    args = ap.parse_args(argv)

    endorse = json.load(open(args.endorse))
    err_seeds = endorse.get("errors") or [
        d for d, i in endorse.get("edges", {}).items() if i.get("status") == "error"]
    if err_seeds:
        print(f"⚠ endorse file has {len(err_seeds)} errored seeds (CDX outage) — "
              f"authority for their targets is UNDER-counted. e.g. {err_seeds[:3]}",
              file=sys.stderr)

    hosts, meta = build(endorse, args.min_score)
    out = {
        "_comment": (
            "ADR-019 product-1 AUTHORITY table. host -> bounded authority feature "
            "in [0,1] (= source_class scale x log-normalized endorsement score). "
            "Built read-only from endorse_all.json by aw ops build-authority-table (restored 2026-10-23 from the retired pipeline builder): "  # rename-codemod: keep (persisted: _comment written into the authority table)
            "archive/proxy wrappers UNWRAPPED (credit to the real target, ADR-019 "
            "Rule 1); source-class taxonomy applied (primary-artifact=full, "
            "publisher=half+threshold, media=EXCLUDED, ADR-019 Rule 2). Read by "
            "retrieval._authority(); scaled by AW_AUTHORITY_W; DEFERS to AW_HUBFLOOR "
            "(no boost where the hub-floor penalises). Flag-gated AW_AUTHORITY, "
            "default OFF (byte-identical when off)."),
        "meta": meta,
        "hosts": hosts,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
    from cytir.ops._provenance import write_provenance
    write_provenance(args.out, "build-authority-table", {"endorse": args.endorse})

    print(f"wrote {args.out}")
    print(f"  hosts: {meta['n_hosts']}  by_class: {meta['by_class']}")
    print(f"  unwrapped archive/proxy targets: {meta['n_unwrapped_targets']}")
    print(f"  log denom = log1p(max_raw={meta['max_raw']}) = {meta['log_denom']}  "
          f"dropped: {meta['dropped']}")
    print("  top 25 by authority:")
    for h, v in list(hosts.items())[:25]:
        print(f"    {v['authority']:.3f}  [{v['class']:6}] {h}  (raw={v['raw']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
