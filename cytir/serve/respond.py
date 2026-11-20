"""S10 step 6 (task 1104): result shaping. The canonical-URL dedup key (task 103) lives here.

Moved out of `retrieval.py` unchanged. Reads the frozen `SETTINGS` (`AW_DEDUP_V2`) at import and
nothing else; it imports no server module but `settings` (so `qdrant_io` and `retrieval` can both use it).
"""
from __future__ import annotations

import urllib.parse

from .settings import SETTINGS as _S
# S10 step 8b (task 1125): stage 2, levers and respond moved out of retrieval.search()
from .trace import _LAST_QUERY_PLAN_TRACE, _QHEAD_COUNTRY_TRACE
from .fusion import _RELFLOOR

# --------------------------------------------------------------------------- #
#  Canonical-URL dedup (task 103, orthogonal to the trust change; ALWAYS on).
#  The same page is indexed many times under different tracking/referrer query
#  params (`?ref=truth11.com`, `?utm_source=..`, `?form=MG0AV3`, ...) — all
#  score IDENTICALLY, so the per-URL dedup (which keyed on the EXACT url) let
#  every variant through: `sdgs.un.org/goals` filled 6 of the top-10 for
#  "united nations sustainable development goals". We collapse candidates to ONE
#  per CANONICAL key = (scheme, host, path-no-trailing-slash, non-tracking
#  query, fragment). The fragment is KEPT so distinct statute provisions
#  (fedlex `…/fr#art_4` vs `#art_349_b`) stay as separate results; only the
#  tracking-only ?ref/?utm/?form variants (same host+path, empty fragment)
#  collapse. Functional params (`?lang=en`, `?id=97592`) are preserved.
#  The root fix (strip at ingest so pages aren't indexed 6x) is tracked
#  separately (#31); this is the query-time safety net.
# --------------------------------------------------------------------------- #
_TRACKING_PARAMS = frozenset({
    "ref", "ref_src", "ref_url", "referrer", "source", "form",
    "fbclid", "gclid", "dclid", "gclsrc", "gbraid", "wbraid", "msclkid",
    "yclid", "twclid", "igshid", "mc_eid", "mc_cid", "mkt_tok",
    "vero_id", "oly_enc_id", "oly_anon_id", "_ga", "_gl", "_hsenc", "_hsmi",
    "spm", "scm", "trk", "li_fat_id", "s_cid", "ceid", "cmpid", "campaign_id",
    "__s", "piwik_campaign", "pk_campaign", "pk_kwd", "sr_share",
})

# --------------------------------------------------------------------------- #
#  AW_DEDUP_V2 (task 233 / diagnosed task 229) — extended canonical-key.
#  Flag-gated (default OFF) until the regression gate passes. The task-103 key
#  above missed the whole url-dedup class the probes flagged: locale/account
#  selectors (`?authuser=N&hl=xx` — ai.google.dev, up to 89 variants of one
#  page), email/campaign trackers (`?asrc=`, `?_fb_noscript=1`), and the
#  http/https + `www.` split (novartis). V2 strips those AND normalizes
#  scheme->https + drops a leading `www.` so those variants collapse to one key.
#
#  CRITICAL SAFETY (task 229 diag): content-bearing params are NEVER stripped —
#  `pid` is statcan.gc.ca's data-TABLE id (`…tv.action?pid=NNN` = distinct
#  tables; stripping it false-collapsed 10,043 statcan points), `page` is
#  pagination, `id`/`lang`/`release`/`node` select distinct documents. The set
#  below is closed to known tracking/locale/session/analytics keys only.
# --------------------------------------------------------------------------- #
_DEDUP_V2 = _S.output.dedup_v2

_TRACKING_PARAMS_V2 = _TRACKING_PARAMS | frozenset({
    "authuser", "hl",            # Google-family locale/account selectors
    "_fb_noscript",              # Meta no-JS marker
    "asrc",                      # TechTarget/computerweekly email-source
    "cjevent", "_bhlid",         # affiliate / campaign click ids
    "s_kwcid", "nvid",           # nvidia / search campaign ids
    "_bt", "_bk", "_bm", "_bn", "_bg",   # bing paid-search tuple
    # task 246: universally-tracking ad/click params (Google Ads gad_* cluster,
    # major-ad-network click ids, email/analytics campaign ids). These carry NO
    # content selection — only the ad-attribution source of the visit — so a page
    # reached via `?gad_source=1` is byte-identical to the same page reached
    # organically and MUST collapse to one canonical key. Most of these already
    # live in the base _TRACKING_PARAMS (stripped in every mode); listing them
    # again here is a harmless frozenset union that documents the ad-param class
    # as a unit and adds the two genuinely-new Google-Ads keys (gad_source,
    # gad_campaignid). Content params (pid/page/id — the statcan landmine) are
    # deliberately NOT here; they select distinct documents (see block comment).
    "gad_source", "gad_campaignid",           # Google Ads (new)
    "gclid", "dclid", "wbraid", "gbraid",     # Google/DoubleClick click ids
    "fbclid", "msclkid", "yclid",             # Meta / Microsoft / Yandex click ids
    "mc_cid", "mc_eid",                       # Mailchimp campaign / recipient
    "_hsenc", "_hsmi",                        # HubSpot email encoding / message id
    "mkt_tok", "igshid",                      # Marketo token / Instagram share id
    "vero_id", "oly_enc_id",                  # Vero / Omeda email ids
})
_TRACKING_PREFIXES_V2 = ("click_", "intern_", "__cft__", "__tn__", "npn_")


def _is_tracking_param(key: str) -> bool:
    k = (key or "").lower()
    if k in _TRACKING_PARAMS or k.startswith("utm_"):
        return True
    if _DEDUP_V2:
        if k in _TRACKING_PARAMS_V2:
            return True
        if any(k.startswith(p) for p in _TRACKING_PREFIXES_V2):
            return True
    return False


def _canonical_key(url: str, anchor=None) -> str:
    """Dedup key that collapses tracking/referrer variants of the SAME page but
    keeps genuinely-distinct pages (functional params, distinct fragments/
    provisions) apart. Falls back to the raw url if parsing fails.

    The fragment comes from the URL ITSELF (`…/491/fr#art_4`) — fedlex statute
    provisions carry it inline, so distinct articles stay as separate results.
    We deliberately do NOT fold in the `anchor` PAYLOAD field: that is the
    chunk's section-heading text, which differs between chunks of the SAME page
    and would (wrongly) stop same-page duplicates from collapsing (measured: it
    dropped eval_names MRR 0.433->0.364 by splitting multi-chunk pages).

    AW_DEDUP_V2 (default off) additionally normalizes scheme->https and drops a
    leading `www.` so http/https/www variants of one page share a key."""
    try:
        p = urllib.parse.urlsplit(url or "")
        keep = [(k, v) for k, v in urllib.parse.parse_qsl(p.query, keep_blank_values=True)
                if not _is_tracking_param(k)]
        keep.sort()
        query = urllib.parse.urlencode(keep)
        path = p.path.rstrip("/")
        netloc = p.netloc.lower()
        scheme = p.scheme.lower()
        if _DEDUP_V2:
            if netloc.startswith("www."):
                netloc = netloc[4:]
            scheme = "https"       # http/https of the same page are one page
        return f"{scheme}://{netloc}{path}?{query}#{p.fragment}"
    except Exception:  # noqa: BLE001
        return url or ""


# --- S10 step 8b (task 1125): constants that travelled with their readers ---

#  ZERO-SIGNAL df-guard (task 246) — the DESTRUCTIVE half of AW_RELFLOOR, still
#  gated on the same flag. Root cause (task 240 Part 2, probe-007 "who is
#  aquiffoo"): a query whose only DISTINCTIVE term is ABSENT from the index
#  (df=0) matches purely on stopword-class tokens ("who"/"is"), which long
#  statute pages BM25-saturate to an IDENTICAL ceiling → 8 results tied at the
#  same fused score, floated by identical trust — a junk flood presented as
#  confident answers. This is NOT thresholdable per-relevance (task 207 parked
#  the general floor for that reason: the legit low-relevance-high-authority
#  band is nested inside the flood band). But a query with NO distinctive signal
#  AT ALL is separable on a QUERY-SIDE test that never touches the legit band:
#    (a) the per-token sub-legs ran (2..MAXQ tokens, so df is known exactly);
#    (b) >=1 query token is genuinely ABSENT (df=0 → empty sub-leg);
#    (c) NO query token is present-and-distinctive (rare_toks empty — every
#        present token is a full-window common/stopword term); AND
#    (d) the dense leg gives NO strong semantic match either (best raw cosine
#        < _RELFLOOR_DENSE_MAX) — this is the guard that spares a legit query
#        answerable via dense/paraphrase when a side term happens to be absent.
#  Only when ALL hold is the query ZERO-SIGNAL: no lexical AND no semantic
#  handle exists, so every result is a stopword coincidence. We then trim to
#  _RELFLOOR_ZS_K (default 1 → "few", 0 → "none") and stamp `zero_signal` on the
#  advisory confidence marker. A legit rare-but-PRESENT query has a distinctive
#  df>0 token (rare_toks non-empty) → condition (c) fails → never trimmed.
#  Flag-off (AW_RELFLOOR=0) => zero_signal is always False => byte-identical.
_RELFLOOR_ZS_K = 1  # fixed: was env AW_RELFLOOR_ZS_K (S10 D5: no product setter)


def finish(*, country, k, relmeta, scored, zero_signal):
    """Response assembly: dedup by canonical URL and shape the k items (S10 step 8b)."""

    # dedup by CANONICAL url (task 103) — collapse tracking/referrer variants of
    # the same page (keep the highest-scoring chunk), keep distinct pages/
    # provisions. Return k DISTINCT authoritative sources.
    # task 246: on a ZERO-SIGNAL query trim to _RELFLOOR_ZS_K (few/none) instead
    # of a k-long junk flood — every candidate is a stopword coincidence.
    eff_k = min(k, _RELFLOOR_ZS_K) if zero_signal else k
    if eff_k <= 0:            # AW_RELFLOOR_ZS_K=0 on a zero-signal query => none
        return []
    out, seen = [], set()
    for fused, pid, pl in scored:
        url = pl.get("url")
        key = _canonical_key(url)
        if key in seen:
            continue
        seen.add(key)
        if _QHEAD_COUNTRY_TRACE:
            # Keep the entire deduplicated final ordering, not merely top-k:
            # rank changes beneath the requested k are still diagnostic gate
            # evidence and each entry remains small/JSON-safe.
            _LAST_QUERY_PLAN_TRACE["final_ranked"].append({
                "id": str(pid), "url": url, "score": round(float(fused), 8)})
        item = {
            "url": url,
            "anchor": pl.get("anchor"),
            "title": pl.get("title"),
            "snippet": pl.get("snippet"),
            "source_type": pl.get("source_type"),
            "lang": pl.get("lang"),
            "trust_tier": pl.get("trust_tier"),
            "last_seen": pl.get("last_seen"),
            # task 625: surface the genuine, never-inferred `published` date
            # captured at ingest (see the identifier-lookup path above for the
            # full rationale). Additive key.
            "published": pl.get("published"),
            "score": round(float(fused), 4),
        }
        if country is not None:  # only then: default response stays byte-identical
            item["country"] = pl.get("country")
        if pl.get("source_type") == "entity_card":
            # entity cards carry identity metadata the UI renders as a card
            for f in ("entity_kind", "register_id", "register_scheme", "qid",
                      "canton", "reachable"):
                if pl.get(f) is not None:
                    item[f] = pl.get(f)
        if _RELFLOOR and id(pl) in relmeta:   # task 207 advisory marker
            cm = relmeta[id(pl)]
            if zero_signal:                   # task 246 low-confidence stamp
                cm = dict(cm, zero_signal=True, low_relevance=True)
            item["confidence"] = cm
        if len(out) < eff_k:
            out.append(item)
        if len(out) >= eff_k and not _QHEAD_COUNTRY_TRACE:
            break
    return out
