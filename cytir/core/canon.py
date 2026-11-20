"""cytir.core.canon — URL canonicalization (task 646 S2, task 643 §6.1).

MOVED HERE from the retired `aw_common` module (the "canon" slice of task 646 S2's
"canon/tokenize/fold" migration item; task 643 §6.2). the retired `aw_common` module
now imports these names from here instead of defining them, so every
existing `aw_common.canonical_url(...)` / `from aw_common import
canonical_url` call site keeps working unchanged — `aw_common.canonical_url
is cytir.core.canon.canonical_url` (pinned by the canon shim test).

`canonical_url` is an IDENTITY INPUT (`cytir.core.ids.mint_point_id` mints the
point id over it) — this module, not `aw_common`, is now the one place that
fact lives. `tokenize`/`fold` (the BM25 write/query mirror contract with its
own independent copy in `cytir/serve/schema.py`) is a separate, higher-
risk unit and is deliberately NOT moved in this PR — see aw_common.py's
top-of-file note.
"""
from __future__ import annotations

import os
import re
import threading
import urllib.parse
from typing import Optional

# URL canonicalization — agents want the *stable* URL, and dedup needs one
# canonical form. CC captures version-pinned / localized mirrors of the same
# page (docs.python.org/3.13/… for the /3/ alias; /es/ /ja/ … mirrors).
_PY_VER = re.compile(r"^/(?:3\.\d+(?:\.\d+)?|2\.\d+)(/|$)")
_PY_LOCALE = re.compile(r"^/(es|ja|ko|fr|zh-cn|zh-tw|pt-br|pl|tr|id|uk|de|it)/")


def normalize_register_id(value: object) -> str:
    """Return the punctuation-insensitive form used for register-ID joins.

    Display payloads retain each registry's canonical punctuation (for example
    ``CHE-123.456.789``), while official exports may use ``CHE123456789``.
    Identity comparisons and joins use this shared form; stored/display values
    must not be rewritten merely to make them byte-equal.
    """
    return "".join(character for character in str(value or "").upper()
                   if character.isalnum())


def canonicalize_url(url: str) -> Optional[str]:
    """Map a captured URL to the stable form to store/return. None ⇒ drop the
    record (a non-English doc mirror we don't want)."""
    try:
        p = urllib.parse.urlsplit((url or "").strip())
    except ValueError:
        return url
    host, path, query = (p.netloc or "").lower(), p.path or "/", ""
    if host in ("docs.python.org", "www.docs.python.org"):
        host = "docs.python.org"
        if _PY_LOCALE.match(path):
            return None                       # drop non-English mirrors
        path = _PY_VER.sub(r"/3\1", path)     # /3.13/… → /3/…  (stable alias)
    elif "aza" in path:                       # bger decision id lives in query
        query = p.query
    path = path.rstrip("/") or "/"
    return urllib.parse.urlunsplit((p.scheme or "https", host, path, query, ""))


# --- tracking-param dedup canonicalization (task #31, index-hygiene sweep) ---
# CC captures the same page under many referral/campaign query variants
# (sdgs.un.org/goals?ref=truth11.com, ?ref=ghost.org, …): identical content,
# duplicate points that flood results. `canonical_url` maps those variants to a
# single grouping key so ingest can dedup and the sweep can collapse existing
# dups. DESIGN: blacklist-strip *known* tracking keys only (never whitelist —
# an unrecognized param is assumed functional and kept, e.g. geo.admin
# ?E=&N=&zoom=, fedlex ?…, ?lang= ?id= ?page=). www.<host> == <host>. Trailing
# slash normalized. The #fragment/anchor is KEPT in the key — distinct-anchor
# passages of one page (fedlex …/de#art_218) are separate records, not dups.
_TRACKING_KEYS = frozenset({
    "ref", "ref_src", "ref_url", "referrer", "referer", "source", "src",
    "utm_id", "fbclid", "gclid", "gclsrc", "dclid", "gbraid", "wbraid",
    "msclkid", "yclid", "twclid", "igshid", "igsh", "mc_cid", "mc_eid",
    "mkt_tok", "_hsenc", "_hsmi", "vero_id", "vero_conv", "oly_anon_id",
    "oly_enc_id", "_openstat", "wickedid", "s_kwcid", "scid", "cvid",
    "guccounter", "guce_referrer", "guce_referrer_sig", "__twitter_impression",
    "spm", "trk", "trkCampaign", "cmpid", "campaign_id", "ncid",
    # GA cross-domain linker params (task 128: ~152 pure dups in prod). NOT
    # "sid=" — functional on some hosts (imodules/utah.edu, 128 §3).
    "_gl", "_ga", "_gac",
    # Google account index (task 767: 65 % of ai.google.dev rows on the twin were `?authuser=N`
    # copies of the same pages). It picks WHICH signed-in account's chrome the page shows, never
    # the page's content; no host is known to vary content by it. Generic, not host-specific.
    "authuser",
})
_TRACKING_PREFIXES = ("utm_", "hsa_", "pk_", "piwik_", "mtm_", "matomo_")

# --- canon EPOCHS (ADR-067; task 756 after review 773 M1) ---------------------------------------
# `canonical_url` is an ID INPUT (`cytir.core.ids.mint_point_id` mints over it), so the tracking set,
# the prefixes, the `;jsessionid` rule and the www fold are part of the id rule. Every change to any
# of them is a new canon EPOCH, frozen here with its number, never a silent list append: a stored
# point id is proved with the epoch that minted it (`canonical_url_at`), a new id is minted with the
# current one. Epoch 1 = the set every point minted before #704 (2026-10-08 14:00Z) used; epoch 2 =
# epoch 1 + `authuser` (#704). `tests/test_aw_canon_epochs.py` pins both sets.
CANON_EPOCHS: dict[int, frozenset] = {
    1: frozenset(_TRACKING_KEYS - {"authuser"}),
    2: _TRACKING_KEYS,
}
CANON_EPOCH = max(CANON_EPOCHS)
# Java-servlet session tokens ride in the PATH (`…/page.html;jsessionid=ABC`),
# not the query string — task 111: 14,987/14,998 statistik.arbeitsagentur.de
# capture URLs were session variants of just 82 real pages, eating the whole
# per-domain fetch budget and storing duplicate points.
# `[\w-]*` before `=`: ros.ie-style suffixed keys (;jsessionidABC12=…) evade a
# literal match (task 128).
_PATH_SESSION = re.compile(r";jsessionid[\w-]*=[^;/?#]*", re.IGNORECASE)


def _is_tracking_key(k: str, keys: frozenset | None = None) -> bool:
    kl = k.lower()
    return kl in (_TRACKING_KEYS if keys is None else keys) or kl.startswith(_TRACKING_PREFIXES)


def strip_tracking_url(url: str, *, epoch: int | None = None) -> str:
    """INGEST-safe clean: strip tracking/referral query params + normalize the
    trailing slash, but KEEP host (incl. www) and scheme exactly as captured and
    KEEP the #fragment + functional params. This is what we STORE — it kills the
    `?ref=`/`?utm_*` duplicate variants (identical stored url ⇒ identical
    point_id ⇒ Qdrant dedups on upsert) without changing which host/scheme the
    agent fetches. Aggressive www/http folding is reserved for canonical_url
    (grouping key / offline sweep), where we pick the cleanest real variant."""
    try:
        p = urllib.parse.urlsplit((url or "").strip())
    except ValueError:
        return url or ""
    path = _PATH_SESSION.sub("", (p.path or "/")).rstrip("/") or "/"
    keys = None if epoch is None else CANON_EPOCHS[epoch]
    kept = [(k, v) for k, v in urllib.parse.parse_qsl(p.query, keep_blank_values=True)
            if not _is_tracking_key(k, keys)]
    query = urllib.parse.urlencode(kept, doseq=True) if kept else ""
    return urllib.parse.urlunsplit((p.scheme, p.netloc, path, query, p.fragment or ""))


# --- credential-URL scrub (task 643 §5.1: 91 gitleaks `jwt` hits, all
# `?accessToken=eyJ…`, sitting in captured/indexed URLs) -----------------
# `canonical_url` is an IDENTITY INPUT: `cytir.core.ids.mint_point_id` mints
# the point id over its output, so changing what it returns re-keys every
# point whose URL carries one of these params. That is a rebuild-class
# change, not a bugfix, so the scrub stays OFF by default (task 646 S0) —
# this PR only lands the capability and a hit count; turning it on is a
# separate, deliberate decision with a rebuild attached. Mirror this flag
# wherever a URL is re-derived at query time (cytir.core.ids, servers) so the
# write and read sides never disagree about which credential shape to strip
# (engineering-doctrine #6, mirror contracts).
_CREDENTIAL_PARAM_NAMES = frozenset({
    "accesstoken", "access_token", "id_token", "auth_token", "authtoken",
    "api_key", "apikey", "api-key", "secret", "client_secret", "token",
    "sig", "signature", "x-amz-signature", "x-amz-security-token",
    "x-amz-credential", "session_token", "sessiontoken",
})
# JWTs are base64url header.payload.signature and always start `eyJ` (the
# base64 of `{"`); this is what the 91 gitleaks `jwt` hits matched.
_JWT_LIKE_RE = re.compile(r"^eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*$")

_SCRUB_CREDENTIAL_URL_PARAMS = (os.environ.get(
    "AW_SCRUB_CREDENTIAL_URL_PARAMS", "").strip().lower()
    in ("1", "true", "yes", "on"))
_CREDENTIAL_URL_HITS = 0
_CREDENTIAL_URL_HITS_LOCK = threading.Lock()


def _is_credential_param(key: str, value: str) -> bool:
    """A query param that looks like a bearer credential, not a functional
    identifier (task 643 §5.1)."""
    if key.lower() in _CREDENTIAL_PARAM_NAMES:
        return True
    return bool(_JWT_LIKE_RE.match(value or ""))


def credential_url_hit_count() -> int:
    """Census counter: how many `canonical_url` calls have stripped at least
    one credential-shaped param, since process start or the last reset.
    Counts even when the scrub is off, so a census run can measure exposure
    before flipping `AW_SCRUB_CREDENTIAL_URL_PARAMS` on."""
    return _CREDENTIAL_URL_HITS


def reset_credential_url_hit_count() -> None:
    """Test/census hook: zero the counter without touching the flag."""
    global _CREDENTIAL_URL_HITS
    with _CREDENTIAL_URL_HITS_LOCK:
        _CREDENTIAL_URL_HITS = 0


def _scrub_credential_params(query: str) -> str:
    """Return `query` with credential-shaped params removed, and count the
    hit. Called unconditionally by `canonical_url` (so the census counter is
    accurate even with the scrub off); only the RETURNED query string is
    gated on `_SCRUB_CREDENTIAL_URL_PARAMS`."""
    global _CREDENTIAL_URL_HITS
    if not query:
        return query
    pairs = urllib.parse.parse_qsl(query, keep_blank_values=True)
    kept, hit = [], False
    for k, v in pairs:
        if _is_credential_param(k, v):
            hit = True
        else:
            kept.append((k, v))
    if not hit:
        return query
    with _CREDENTIAL_URL_HITS_LOCK:
        _CREDENTIAL_URL_HITS += 1
    if not _SCRUB_CREDENTIAL_URL_PARAMS:
        return query          # flag off: count it, change nothing
    return urllib.parse.urlencode(kept, doseq=True) if kept else ""


def redact_credential_query_values(url: str) -> tuple[str, int]:
    """Replace credential-shaped query VALUES with `REDACTED`, keeping the
    param name, the rest of the query string, and the scheme/host/path/
    fragment intact (task 651: sealed gate-evidence artifacts must not carry
    a live bearer token, but the fix there is DIFFERENT from `canonical_url`'s
    scrub -- that one DROPS the param entirely, for a dedup grouping key, and
    also drops unrelated tracking params; this one only replaces the value,
    so a JSON evidence file stays byte-shaped and every OTHER field is
    untouched).

    Uses the SAME detection `canonical_url`'s scrub uses (`_is_credential_param`:
    a known credential param name, or a JWT-shaped value under any name), so
    the write path and the evidence-redaction path never disagree about what
    counts as a credential (engineering-doctrine: one source of truth).

    Returns `(url, 0)` unchanged -- not re-parsed or re-encoded -- when there
    is nothing to redact, so a file with no credential params is byte-
    identical after a pass over it.
    """
    if not url or "?" not in url:
        return url, 0
    try:
        p = urllib.parse.urlsplit(url)
    except ValueError:
        return url, 0
    if not p.query:
        return url, 0
    pairs = urllib.parse.parse_qsl(p.query, keep_blank_values=True)
    hits = 0
    kept = []
    for k, v in pairs:
        if v == "REDACTED":
            kept.append((k, v))  # already redacted -- idempotent, not a new hit
        elif _is_credential_param(k, v):
            kept.append((k, "REDACTED"))
            hits += 1
        else:
            kept.append((k, v))
    if not hits:
        return url, 0
    new_query = urllib.parse.urlencode(kept, doseq=True)
    return (urllib.parse.urlunsplit(
        (p.scheme, p.netloc, p.path, new_query, p.fragment)), hits)


def canonical_url_at(url: str, epoch: int) -> str:
    """`canonical_url` under canon epoch `epoch` (`CANON_EPOCHS`): the key a point minted in that
    epoch was minted over. Used to PROVE a stored id; new ids are minted with `canonical_url`."""
    if epoch not in CANON_EPOCHS:
        raise KeyError(f"canon epoch {epoch!r} is not declared (CANON_EPOCHS: {sorted(CANON_EPOCHS)})")
    return canonical_url(url, _epoch=epoch)


def canonical_url(url: str, *, _epoch: int | None = None) -> str:
    """Grouping key for duplicate collapse: strip tracking/referral query params,
    fold www., force https, normalize trailing slash; keep functional params and
    #fragment. Always returns a string (never drops). Use for dedup grouping,
    NOT as the stored/returned URL (that stays the cleanest *real* variant).

    Also runs the credential-URL census (`_scrub_credential_params`) — counts
    unconditionally, but only changes the returned query string when
    `AW_SCRUB_CREDENTIAL_URL_PARAMS` is truthy (off by default; task 646 S0)."""
    try:
        p = urllib.parse.urlsplit((strip_tracking_url(url, epoch=_epoch)))
    except ValueError:
        return url or ""
    host = (p.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    scheme = "https" if p.scheme in ("http", "https", "") else p.scheme
    query = _scrub_credential_params(p.query)
    return urllib.parse.urlunsplit((scheme, host, p.path or "/", query, p.fragment or ""))


def https_frontier_url(url: str, keep_query: bool = False,
                       alias: dict[str, str] | None = None) -> str:
    """Crawl-frontier link key (task 2166; was three private `canon` copies): force https, lower the
    host and map it through `alias`, keep the path (empty -> "/", trailing slash KEPT), drop the
    query unless `keep_query`, drop the fragment. Returns "" on an unparseable URL. NOT
    `canonical_url`: it keeps the trailing slash and does not fold `www.`, so frontier keys and
    seeds written by the packs stay as they were. Never used to mint a point id."""
    try:
        p = urllib.parse.urlsplit(url)
    except ValueError:
        return ""
    host = (p.netloc or "").lower()
    if alias:
        host = alias.get(host, host)
    return urllib.parse.urlunsplit(("https", host, p.path or "/", p.query if keep_query else "", ""))


def document_url_key(url: str | None) -> str | None:
    """THE "same document URL" key for tombstone-vs-pack comparisons (task
    698). `canonical_url` of the url, or None when there is no document to
    name: falsy url, or one with no host (`" "`, `"/"`, `"https://"`,
    `"http://www."` all fold to `https:///` and would match any other
    host-less url). Callers refuse / skip on None and otherwise compare keys
    with `==`, never raw strings. What it folds (beyond scheme and `www.`):
    trailing slash, host case, query re-encoding, `;jsessionid`, tracking
    keys. What it does NOT fold: port, path case, percent-escape case,
    fragment, `www2.`."""
    if not url or not str(url).strip():
        return None
    key = canonical_url(url)
    try:
        if not urllib.parse.urlsplit(key).netloc:
            return None
    except ValueError:
        return None
    return key


_WWW_RE = re.compile(r"^www\d*\.")


def fold_www(host: str) -> str:
    """Fold a leading `www.` / `www<N>.` off a host. The ONE home for the rule.

    Two folds existed in this file and they disagree. `canonical_url` folds a
    LITERAL `www.` (it is a URL grouping key, and `aw_serving_metadata.
    serving_domain` takes the host from it); `_WWW_RE` folds `^www\\d*\\.` and
    was reachable only from the outlink sink. A caller that wanted "the host as
    production's `domain` field spells it" had to pick one, and
    `promote_sealed_pack` had picked a third — its own hand-rolled `www.` strip.

    Measured on the aw_e8 twin, 2026-09-23, before choosing: every one of the
    13,861 rows whose url is `www150.` or `www23.statcan.gc.ca` carries
    `domain = statcan.gc.ca`, from two different ingest lanes; over a 200,704-row
    sample across 4,707 domains, 12 rows carry a `www<N>.` domain
    (`www4.ti.ch` 11, `www9.org` 1) and 380 carry a `www.` one (the nine hosts
    `aw_serving_metadata`'s docstring already names). So the `^www\\d*\\.` fold
    is what production's `domain` actually is, at 99.8%, and the residue is a
    known pre-existing inconsistency rather than the convention.

    `canonical_url` is deliberately NOT changed to match: it is an identity
    input (`cytir.core.ids.mint_point_id` mints over it), and re-folding it
    would re-key every point id in the collection. The divergence is real and
    is named here so the next reader does not have to re-derive it.
    """
    return _WWW_RE.sub("", (host or "").lower().lstrip("."))
