"""cytir.core.ids — I1 identity + I2 rule-epoch stamps for the aw_b1 build.

MOVED HERE from the retired `aw_b1_identity` module in task 646 S2 (task 643
§6.1/§6.2): the former `aw_b1_identity` shim is deleted
(`from cytir.core.ids import *`) so the ~60 existing `import aw_b1_identity`
call sites keep working untouched. New code imports `cytir.core.ids` directly.

Normative sources: `specs/identity.md` (I1) and `specs/epochs.md` (I2). This is
the *code* the architecture-native builder (`build_aw_b1.py`) mints against; it
deliberately does NOT touch the live `aw_common.Passage.__post_init__` scheme
(which conflates `sha1(text)` INTO the id — the F1 bug I1 exists to kill).

Two questions, never mixed (identity.md §1):
    point_id  = uuid5(AW_NAMESPACE, canonical_url + "#" + anchor_component)
    digest    = sha256(normalize(chunk_text))
`point_id` answers "is this the same record?"; `digest` answers "did its
content change?".

The id-input string is UNIFORM across tiers: `canonical_url + "#" + anchor`.
The anchor component is tier-dependent:
    Tier A : the source-native, non-positional anchor (e.g. fedlex eId `art_336`)
    Tier B : the digest itself (content-addressed anchor-of-last-resort)
    Tier C : "" (empty) — entity cards are url-only

IMPORTANT (verified against identity.md §6 worked examples, computed exactly):
the separator "#" is ALWAYS present, even for Tier C. Worked example #4
(digitec.ch) reproduces `uuid5(NS, "https://digitec.ch/#")`, NOT
`uuid5(NS, "https://digitec.ch/")` — so Tier C is `canonical_url + "#" + ""`,
i.e. a trailing "#". identity.md §2's prose formula `uuid5(NS, canonical_url)`
for Tier C is INCONSISTENT with its own §6 worked value; the worked value is the
acceptance test, so we implement the trailing-"#" form. See task 306 for the
flag. Examples #1 (Tier A, `art_336`) and #4 (Tier C) reproduce byte-exactly;
#2/#3/#5 (Tier B) depend on the real chunk digest and are validated by
construction (the spec shows only truncated illustrative digests).
"""
from __future__ import annotations

import hashlib
import re
import uuid

# --- I1: the constitutional namespace (identity.md §1.1) --------------------
# Generated once (uuid.uuid4(), 2026-07-22) and FROZEN. Never regenerate —
# regenerating re-ids every record in the corpus (an ADR-level event).
AW_NAMESPACE = uuid.UUID("9addc040-52b1-432b-b072-e3f89057688a")

# --- I2: the rule-epoch stamps this build mints (epochs.md §2) ---------------
# aw_b1 pulls in the BUILT/HELD epochs the rebuild was budgeted for:
#   tok=2      (accent-fold, AW_FOLD_ACCENTS=1)     — epochs.md §2 "flips at aw_b1"
#   country=2  (ported 261 P17-primary stack)       — epochs.md §2 country row
# everything else ships at its LIVE epoch (=1).
EPOCHS = {
    "extract": 1,   # unified extractor, trafilatura precision+recall (LIVE)
    "chunk": 1,     # fp.chunk paragraph windows / per-article law splitters (LIVE)
    "embed": 1,     # arctic-embed-l-v2 MRL-128 int8, config-B (LIVE; dense reused)
    "tok": 2,       # BM25-sat + accent-fold (BUILT -> flips at aw_b1)
    "country": 2,   # 261 P17-primary stack (BUILT/gated -> applied at aw_b1)
    "trust": 1,     # static per-source-type tier constants (LIVE)
    "card": 1,      # single-channel card recipe baseline (LIVE)
    # prov_id mint rule (ADR-117 am.1, task 1361): 1 = path-bound (rows minted before task 771),
    # 2 = portable, checkout-independent (771 onward). A payload with no `prov_id` epoch reads as 1.
    "prov_id": 2,
}

# What a payload with no entry for a key reads as. Only keys added after rows were minted appear here.
ABSENT_EPOCH = {"prov_id": 1}


def rule_epoch(payload, key: str):
    """The epoch `payload["rule_epochs"][key]`, or the declared ABSENT_EPOCH default (None if none).

    Asymmetry: `prov_id` reads 1 when absent (conservative: absent means "not known portable", not
    "known path-bound"); keys without an `ABSENT_EPOCH` entry read None.
    """
    eps = (payload or {}).get("rule_epochs") or {}
    return eps.get(key, ABSENT_EPOCH.get(key))

# rule_legend strings for the MANIFEST (epochs.md / manifest-schema.md).
# The `extract` legend records the pinned extractor libs (epochs.md §3 F12).
RULE_LEGEND = {
    "extract": "unified-extractor@1",   # + lib pins injected at build (see build_aw_b1)
    "chunk": "fp.chunk-paragraph-window@1",
    "embed": "arctic-l-v2-mrl128-int8@1",
    "tok": "bm25-sat+fold@2",
    "country": "p17-primary-261@2",
    "trust": "static-tier-constants@1",
    "card": "single-channel@1",
    "prov_id": "portable-prov-id@2",
}

# --- tier classification ----------------------------------------------------
# A positional anchor is NOT a stable identity (identity.md §4): wiki_dump emits
# "chunk-%d" which shifts on any insertion above it. Such anchors are demoted to
# Tier B (digest-as-anchor). Matches "chunk-5", "chunk5", "chunk_5".
_POSITIONAL_ANCHOR = re.compile(r"^chunk[-_]?\d+$", re.IGNORECASE)
_WS = re.compile(r"\s+")


def _clean_anchor(anchor) -> str | None:
    """Normalize a raw anchor field: parquet/qdrant may carry the literal string
    'None', empty, or a real anchor. Returns a real anchor string or None."""
    if anchor is None:
        return None
    a = str(anchor).strip()
    if not a or a == "None":
        return None
    return a


def is_positional_anchor(anchor) -> bool:
    a = _clean_anchor(anchor)
    return bool(a and _POSITIONAL_ANCHOR.match(a))


# --- source-native anchor, the ONE predicate (task 575) ---------------------
# A "source-native anchor" is the fragment THE SOURCE gave the passage: a fedlex
# Akoma Ntoso eId (`art_336`), a legifrance article id. It is the property
# ADR-090 §4 scopes `chunk.law` by, and the property the `chunk.reference`
# selector's P3 refuses on.
#
# A TIER-RESOLVED IDENTITY COMPONENT IS NOT ONE. `classify_tier` returns the
# native anchor for Tier A, the CONTENT DIGEST for Tier B and "" for Tier C, and
# a record that has been through it therefore carries a non-empty component
# whenever it carries text at all. Reading that component as "this record has a
# native anchor" is reading `bool(digest)` -- true of every row that exists.
#
# MEASURED, task 575, four stages agreeing:
#   * the ADR-093 L1-b markdown records carry `anchor: None`;
#   * the corpus store's own `anchor` column is empty on 100% of the rows of all
#     25 domains the 33 named golds sit on (53,990,262 rows cross-tabbed);
#   * the staged pack rows carry `anchor` non-empty ZERO times in 1.8M rows over
#     62 shards, while `identity_anchor_component == content_digest` on 100% of
#     the rows that carry it;
#   * `aw_e8` payloads carry no `anchor` key at all (390 of 390 probed).
# And yet `build_pages_565.py`'s `bool(rec.anchor or rec.identity_anchor_component)`
# marked 100.000% of the 211,464 url-keyed pages as anchored and 0.000% of the
# 19,762 sha256-keyed ones -- a split that tracks WHICH STAGING ERA WROTE THE
# FIELD (`identity_anchor_component` vs `anchor_component`), not any anchor.
# That is why this predicate lives here, in one place, and why the three callers
# that each spelled it differently now import it (engineering-doctrine 1/6).
#
#: Fields that may carry a SOURCE-NATIVE anchor, as the source gave it.
NATIVE_ANCHOR_FIELDS = ("anchor", "native_anchor")

#: Fields that carry a TIER-RESOLVED component. NEVER evidence of a native
#: anchor: for Tier B this is the content digest and for Tier C it is "".
RESOLVED_COMPONENT_FIELDS = ("anchor_component", "identity_anchor_component")


def source_native_anchor(record, *, title=None, force_tier_b: bool = False):
    """The record's source-native anchor, or None.

    Applies exactly the guards `classify_tier` applies before it returns Tier A
    -- non-empty, not the literal "None", not positional, not equal to the
    record's title, and not force-demoted -- so this function and `classify_tier`
    cannot disagree about what a native anchor is. It is deliberately a
    RE-STATEMENT over a record rather than a second rule: `classify_tier` needs a
    digest it can fall back to, and a caller asking "is this anchored?" has no
    digest and no business inventing one.

    `record` is any mapping (an L1 markdown record, a staged pack row, a qdrant
    payload). A tier-resolved component is ignored by construction -- it is not
    read at all.
    """
    if not hasattr(record, "get"):
        raise TypeError(
            "source_native_anchor() takes a record mapping, not "
            f"{type(record).__name__}. Pass the record; the anchor field name "
            "is this module's business, not the caller's.")
    a = None
    for field in NATIVE_ANCHOR_FIELDS:
        a = _clean_anchor(record.get(field))
        if a:
            break
    if not a or force_tier_b:
        return None
    t = title if title is not None else record.get("title")
    if t and a.strip().lower() == str(t).strip().lower():
        return None                      # a page label, never a chunk anchor
    if _POSITIONAL_ANCHOR.match(a):
        return None                      # "chunk-5" shifts on any insertion
    return a


def has_source_native_anchor(record, *, title=None,
                             force_tier_b: bool = False) -> bool:
    """True iff `record` carries a source-native anchor (see above)."""
    return source_native_anchor(record, title=title,
                                force_tier_b=force_tier_b) is not None


def normalize_digest_text(text: str) -> str:
    """identity.md §5: collapse all whitespace runs to a single ASCII space and
    strip. NO case-fold, NO accent-fold, NO NFKD — digest answers 'did the
    content change', a retrieval-independent question."""
    return _WS.sub(" ", text or "").strip()


def content_digest(text: str) -> str:
    """sha256 of the normalized chunk text (identity.md §5). 64-hex, no
    truncation. Upgrades the legacy sha1(text) — a NEW field, no migration."""
    return hashlib.sha256(normalize_digest_text(text).encode("utf-8")).hexdigest()


def classify_tier(source_type: str | None, native_anchor, has_text: bool,
                  digest: str | None, title: str | None = None,
                  force_tier_b: bool = False) -> tuple[str, str]:
    """Return (tier, anchor_component) per identity.md §2.

    - Tier C: entity_card -> url-only, anchor_component = "" (trailing "#").
    - Tier A: a source-native, NON-positional, PER-CHUNK anchor exists.
    - Tier B: everything else -> anchor_component = digest (content-addressed).

    `digest` must be supplied for Tier B/C (it is always computed for the record).

    Two guards against a PAGE-LEVEL anchor masquerading as a per-chunk Tier-A
    anchor (identity.md §2/§3 — the "constant anchor collapses all chunks of a
    multi-chunk page to one record" data-loss trap):
      * `force_tier_b`: caller knows this source is anchor-less prose (e.g. the
        swissinfo/ch-media news packs, which put the ARTICLE TITLE into `anchor`
        for every chunk of the article — a per-page label, not a per-chunk anchor;
        architecture.md §3 classifies flat news as Tier B).
      * anchor == title: a title is a page identity, never a chunk anchor. If the
        anchor equals the record's title it is a page-level label -> Tier B.
    """
    st = (source_type or "").strip().lower()
    if st == "entity_card":
        return "C", ""
    a = _clean_anchor(native_anchor)
    title_eq = bool(a and title and a.strip().lower() == str(title).strip().lower())
    if a and not force_tier_b and not title_eq and not _POSITIONAL_ANCHOR.match(a):
        return "A", a
    # Tier B: digest-as-anchor, PER CHUNK (identity.md §2). Requires a digest.
    if not digest:
        raise ValueError("Tier B classification requires a content digest")
    return "B", digest


def mint_point_id(canonical_url: str, anchor_component: str) -> str:
    """point_id = uuid5(AW_NAMESPACE, canonical_url + "#" + anchor_component).

    `anchor_component` is the tier-resolved component from classify_tier():
    a real anchor (A), the digest (B), or "" (C, yielding a trailing "#").
    """
    return str(uuid.uuid5(AW_NAMESPACE, f"{canonical_url}#{anchor_component}"))


def record_point_id(r) -> str:
    """Stable Qdrant point id for a GENERIC record (no aw_b1 identity).

    Moved here from the pipeline-era `index_bakeoff.point_id` (task 682, S6): the
    `uuid5(` mint belongs in `cytir/core` (test_core_ratchet), and the generic
    embed stage (`cytir.embed.collection_stage`) needs it without importing
    `pipeline/`. A record's own `point_id` (or `url`) that is already a UUID is
    kept; anything else is hashed with uuid5 in the URL namespace; a record with
    neither gets a random uuid4 (the bake-off's historical behaviour, kept so
    the moved function is byte-identical). aw_b1 rows never come through here:
    they carry `mint_point_id`'s `AW_NAMESPACE` ids.
    """
    pid = r.get("point_id") or r.get("url") or str(uuid.uuid4())
    try:
        uuid.UUID(str(pid))
        return str(pid)
    except (ValueError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, str(pid)))


def salted_anchor_component(anchor_component: str, salt: int) -> str:
    """Task 325 (id-collision repair): an EXPLICIT, PERSISTED disambiguation salt.

    When two records of one canonical_url resolve to the same anchor_component
    (degenerate Tier-B fallbacks, identical snippets, page-level KV texts), the
    later record's id must not silently overwrite the earlier one. The builder
    demotes the later record down the honest fallback chain first; if the id
    STILL collides, it appends this salt: component + "|s<k>" (k>=1).

    The salt is stamped into the payload (`id_salt`) so the id re-mints from
    stored fields forever after — position in the build stream matters only at
    FIRST assignment, then the salt travels with the record. `content_digest`
    is NOT salted (it keeps answering "did the content change"). salt=0 is the
    unsalted identity (returns the component unchanged)."""
    if not salt:
        return anchor_component
    return f"{anchor_component}|s{int(salt)}"


#: Separator for a law article's sub-chunk ordinal. Deliberately NOT the
#: collision salt's `|s<k>`: a sub-ordinal is a declared boundary of a policy
#: (`law-article-sub-pack-v1`), a salt is an emergency disambiguation, and a
#: reader who cannot tell them apart cannot tell a designed split from a bug.
LAW_SUB_SEPARATOR = "~s"


def law_sub_anchor_component(anchor_component: str, ordinal: int,
                             n_sub: int) -> str:
    """Task 573 / E573-1: the identity component of one sub-chunk of an article.

    An article that fits in one chunk keeps its component EXACTLY as
    `chunk.law` mints it today (`n_sub == 1` returns the component unchanged),
    so a short article's point_id is byte-identical across the two arms and
    across the policy change -- the property that makes "short articles
    unchanged" a fact about ids and not only about text.

    An article that is split gets a NEW component for EVERY sub-chunk,
    including the first. The alternative -- letting sub-chunk 1 keep the bare
    anchor -- would mint one id that is stable while its text changed, which is
    exactly the content-bound-identity violation `content_digest` exists to
    prevent, and would let an arm-A row and an arm-B row share an id with
    different bodies inside one measurement.

    `ordinal` is 1-based and appears in the component, so the component says
    which slice it is without consulting anything else.
    """
    if not isinstance(n_sub, int) or n_sub < 1:
        raise ValueError(f"n_sub must be >= 1, got {n_sub!r}")
    if not isinstance(ordinal, int) or not 1 <= ordinal <= n_sub:
        raise ValueError(f"ordinal {ordinal!r} outside 1..{n_sub}")
    if n_sub == 1:
        return anchor_component
    return f"{anchor_component}{LAW_SUB_SEPARATOR}{ordinal}"


def tier_b_fallback_digest_input(snippet, title, url) -> tuple[str, str]:
    """Digest input for a Tier-B record whose per-chunk TEXT is unavailable
    (corpus-join miss, or ambiguous anchor-less multi-chunk join that snippet
    matching could not resolve — task 319 / preflight GAP A33).

    Returns (digest_input, quality_tag). The ONLY legal fallback order is:
      snippet -> "fallback_snippet"   (per-chunk-unique for distinct chunks:
                                       the stored snippet is an extractive
                                       PREFIX of the chunk's own text, so
                                       distinct chunks mint distinct ids)
      title   -> "degenerate_title"   (page-level — all snippet-less chunks of
                                       the URL COLLAPSE to one id; the builder
                                       must count this class loudly)
      url     -> "degenerate_url"     (same collapse; last resort, total)

    Incident (task 318, gate FAIL): the aw_b1 build joined anchor-less
    multi-chunk pages through a first-write-wins KV keyed by (url, anchor=∅),
    so every chunk of e.g. anthropic.com/constitution (218 chunks) got the SAME
    text -> same digest -> same Tier-B id -> one surviving point (414 -> 15 on
    anthropic.com; ~1.8M collisions corpus-wide). identity.md §2 requires the
    digest-as-anchor substitution to be PER-CHUNK; this helper is the single
    home for what "per-chunk" means when the chunk text is missing.
    """
    s = (snippet or "").strip()
    if s:
        return s, "fallback_snippet"
    t = (title or "").strip()
    if t:
        return t, "degenerate_title"
    return url or "", "degenerate_url"


# --- self-test: the two byte-reproducible worked examples (identity.md §6) ---
# #1 Tier A fedlex CO Art.336 ; #4 Tier C digitec. #2/#3/#5 are Tier B and depend
# on the real chunk digest (spec shows truncated illustrative text) — validated
# by construction in build_aw_b1's slice check, not here.
_WORKED = [
    # (canonical_url, anchor_component, expected_point_id)
    ("https://fedlex.admin.ch/eli/cc/27/317_321_377/fr#art_336", "art_336",
     "6fa396f2-72d6-5182-b052-9a4470206545"),
    ("https://digitec.ch/", "",
     "b5a74ce4-8c7c-5afe-8601-a18ca13e7372"),
]


def self_test() -> None:
    for cu, anchor, exp in _WORKED:
        got = mint_point_id(cu, anchor)
        assert got == exp, f"id mismatch for {cu!r}#{anchor!r}: {got} != {exp}"
    assert str(AW_NAMESPACE) == "9addc040-52b1-432b-b072-e3f89057688a"
    # digest is sha256, 64 hex
    d = content_digest("  Genève   art. 336\n\n")
    assert len(d) == 64 and d == content_digest("Genève art. 336")
    # A33 regression (task 319): two anchor-less chunks of the SAME url with
    # DIFFERENT texts must mint DISTINCT Tier-B ids; identical texts dedup.
    cu = "https://anthropic.com/constitution"
    ids = set()
    for text in ("Chunk one prose about the constitution.",
                 "Chunk two, different prose entirely."):
        dg = content_digest(text)
        tier, comp = classify_tier("content", None, True, dg)
        assert tier == "B" and comp == dg
        ids.add(mint_point_id(cu, comp))
    assert len(ids) == 2, "A33: distinct chunks must mint distinct ids"
    dg = content_digest("Chunk one prose about the constitution.")
    assert mint_point_id(cu, dg) in ids  # identical text -> same id (dedup)
    # A33 fallback: distinct snippets mint distinct ids when text is missing
    fb = set()
    for snip in ("First chunk snippet…", "Second chunk snippet…"):
        di, tag = tier_b_fallback_digest_input(snip, "Page Title", cu)
        assert tag == "fallback_snippet"
        fb.add(mint_point_id(cu, content_digest(di)))
    assert len(fb) == 2, "A33: distinct snippets must mint distinct ids"
    di, tag = tier_b_fallback_digest_input(None, "Page Title", cu)
    assert tag == "degenerate_title"
    # 325: persisted salt — distinct ids for same component; salt=0 = identity
    dg0 = content_digest("Page Title")
    assert salted_anchor_component(dg0, 0) == dg0
    s1 = mint_point_id(cu, salted_anchor_component(dg0, 1))
    s2 = mint_point_id(cu, salted_anchor_component(dg0, 2))
    assert len({mint_point_id(cu, dg0), s1, s2}) == 3, \
        "325: salted components must mint distinct ids"
    print("aw_b1_identity self_test: OK (worked examples #1, #4 exact; "
          "A33 per-chunk distinctness + fallback order)")
