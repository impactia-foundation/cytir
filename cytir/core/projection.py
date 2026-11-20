#!/usr/bin/env python3
# Moved from cytir.core.projection (task 677, task 643 S5); the old path is an alias shim.
"""Pure aw_b1 file-record projection shared by staging and offline workers.

This module owns the document-side projection contract: identity validation,
dense/BM25 text composition, sparse tokenization/saturation, and the compact
serving payload.  It deliberately has no Qdrant or embedder dependency.
"""
from __future__ import annotations

import math
import os
from typing import Any, Iterable

# tok@2 is an aw_b1 invariant.  Set it before importing the tokenizer module;
# callers cannot accidentally build a different sparse epoch by import order.
os.environ.setdefault("AW_FOLD_ACCENTS", "1")

from cytir.core import ids as ID  # noqa: E402
from cytir.core import license as LIC  # noqa: E402
from cytir.core import common as AC  # noqa: E402
from cytir.core import bm25_tokenizer as _tokenize
from cytir.core import country as CT  # noqa: E402
from cytir.core import bm25_saturate as SAT  # noqa: E402
from cytir.core import search_result_url as SRU  # noqa: E402

from cytir.core import bm25_tokenizer as _TOK  # noqa: E402
_TOK._FOLD_ACCENTS = True

BASE_PASSTHROUGH = (
    "title", "lang", "domain", "source_type", "trust_tier", "site_name",
    "last_seen", "country", "provenance", "nature", "theme", "published",
    "search_ok", "train_ok",
    "status", "source_version", "source_version_url", "source_anchor_evidence",
    "base_anchor",
    # task 1391 / ADR-124: master-only (stripped by common.MASTER_ONLY_KEYS in the dump)
    "family_id",
    # Entity/register fields are compact structural retrieval metadata, not
    # readable source content.  Keep this shared projection aligned with the
    # full-rebuild whitelist in build_aw_b1.project_record(): a source-scoped
    # card re-emit must not silently erase lookup identity or registered-state
    # evidence merely because it takes the normal pack stage rather than a
    # whole-collection rebuild (Task 384 / the earlier canton-drop incident).
    "register_id", "register_scheme", "lei", "names", "entity_kind",
    "category", "retail_kind", "qid", "disc_class", "agent_policy", "canton",
    "founded", "reg_status", "reg_status_delete_date", "dissolved",
    "resolve_confidence",
    # task 661 F5 (2026-09-28, coordinator ruling): a row built from a
    # source's own openly-published feed metadata (title/description/
    # link/pubDate) rather than a fetched page -- the record when a page
    # is behind a WAF/bot challenge (consent-probe-vantage: a challenge
    # is the site's own access control, recorded as unreadable, never
    # circumvented). "feed_item" is the only value in use today;
    # absent for every ordinary page-fetched row.
    "source_extract",
    # task 1040 (RULING 55, Tim 2026-10-22): a windowed fedlex article keeps its BARE url
    # (`#art_N`) on every window, so the window's place is a payload field, not a url suffix:
    # `{"index": K, "count": N}` (1-based), absent on a one-row article. Locating metadata only.
    "window",
)

# These are locating/provenance fields specific to the chunk@2 artifact.  Raw
# source-policy evidence and master attribution do not belong in this list.
CHUNK2_PASSTHROUGH = (
    "canonical_url", "identity_tier", "identity_anchor_component", "id_salt",
    "chunk_rule", "chunk_ordinal", "chunk_header_path", "chunk_n_words",
    "chunk_unit", "extraction_rule", "consent_search_ok", "consent_train_ok",
)

CONSENT_KEYS = (
    "consent_state", "consent_basis", "consent_source", "consent_checked_at",
)

# Content is represented only by the dense/sparse vectors.  These names are
# deliberately class-wide rather than a Zefix exception: a source-specific
# readable body (people, a website scrape, a product list) must never acquire
# serving semantics merely because its producer gave it a convenient field.
SERVING_CONTENT_FIELDS = frozenset({
    "text", "dense_text", "sparse_text", "register_people", "website_content",
    "website_body", "body_text", "content", "people",
})

# Entity cards are one logical dense chunk assembled from independently
# refreshable evidence events.  Keeping the event boundaries on the sparse
# build surface lets BM25 normalize each assertion by its own length instead
# of letting an unrelated long assertion dilute a short exact-name match.
# The separator is build-only: ``sparse_text`` never enters the serving
# payload, and ordinary unsegmented records retain their byte-for-byte path.
SPARSE_SEGMENT_SEPARATOR = "\n\u241e\n"


class ProjectionError(ValueError):
    """An input cannot be represented truthfully as an aw_b1 point."""


def saturated_bm25(text: str, lang: str | None, *,
                   decompound: bool = True) -> tuple[list[int], list[float]]:
    """Canonical tok@2 sparse vector, without importing a Qdrant type.

    A normal text surface follows the historical whole-record saturation
    path.  A surface carrying explicit evidence-segment boundaries saturates
    each segment independently and keeps the maximum posting weight per term.
    Max (rather than sum) prevents duplicate source assertions from becoming
    a ranking boost while preserving the strongest local evidence for a term.

    `decompound=False`: see `cytir.core.common.bm25_tokens` (the rebuild's proof
    of surface only; never a writer's posting).
    """
    segments = text.split(SPARSE_SEGMENT_SEPARATOR)
    if len(segments) == 1:
        tf = _tokenize.bm25_tokens(text, lang, decompound=decompound)
        if not tf:
            return [], []
        indices = list(tf)
        values = [tf[index] for index in indices]
        return SAT.saturate_sparse(indices, values)

    merged: dict[int, float] = {}
    order: list[int] = []
    for segment in segments:
        tf = _tokenize.bm25_tokens(segment, lang, decompound=decompound)
        if not tf:
            continue
        indices = list(tf)
        _indices, values = SAT.saturate_sparse(
            indices, [tf[index] for index in indices])
        for index, value in zip(_indices, values):
            if index not in merged:
                order.append(index)
                merged[index] = value
            else:
                merged[index] = max(merged[index], value)
    return order, [merged[index] for index in order]


def _validate_precomputed_identity(
    record: dict[str, Any], *, canonical_url: str, digest: str, tier: str,
    anchor_component: str, point_id: str,
) -> None:
    expected = {
        "canonical_url": canonical_url,
        "content_digest": digest,
        "identity_tier": tier,
        "identity_anchor_component": anchor_component,
        "point_id": point_id,
        "id_salt": 0,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ProjectionError(
                f"precomputed identity drift at {field}: {record.get(field)!r} != {value!r}")


def project_record(
    record: dict[str, Any], *, consent: dict[str, Any] | None = None,
    attribution: str | None = None, license_version: str | None = None,
    force_tier_b: bool = False, validate_precomputed_identity: bool = False,
    preserve_rule_epochs: bool = False,
    extra_passthrough: Iterable[str] = (), require_license: bool = False,
    require_embedding_consent: bool = False,
    family_license: str | None = None,
) -> tuple[str, str, str, dict[str, Any]] | None:
    """Return ``(id, dense_text, sparse_text, payload)`` for one source row.

    The default is byte-contract compatible with ``stage_pack_b1``.  The
    chunk@2 worker opts into the stronger precomputed-identity, policy, and
    upstream-epoch checks rather than growing a second projection algorithm.
    """
    url = (record.get("url") or "").strip()
    text = record.get("text") or ""
    if not url or not text:
        return None
    if SRU.is_search_result_url(url):
        # Task 2009: a site-search result page is no authoritative source. Counted
        # (`search_result_reject_count`); the caller's own skip counter sees the None.
        SRU.note_search_result_reject()
        return None

    digest = ID.content_digest(text)
    tier, anchor_component = ID.classify_tier(
        record.get("source_type"), record.get("anchor"), True, digest,
        title=record.get("title"), force_tier_b=force_tier_b,
    )
    canonical_url = AC.canonical_url(url)
    point_id = ID.mint_point_id(canonical_url, anchor_component)
    if validate_precomputed_identity:
        _validate_precomputed_identity(
            record, canonical_url=canonical_url, digest=digest, tier=tier,
            anchor_component=anchor_component, point_id=point_id,
        )

    dense_raw, sparse_raw = AC.record_text_channels(record)
    dense_text = AC.compose_index_text({**record, "text": dense_raw})
    sparse_text = AC.compose_index_text({**record, "text": sparse_raw})

    domain = (record.get("domain") or "").lower()
    country = CT.country_for_record(
        domain, record.get("source_type"), record.get("register_id"), record.get("lei"))
    if country == CT.KEEP_EXISTING:
        country = record.get("country")
    license_value = LIC.resolve_license(
        record.get("license"), domain, domain, record.get("source_type"),
        record.get("provenance"),
    )
    if not license_value and family_license:
        # Task 641 (2026-09-23): a FAMILY-LEVEL basis (ADR-111's opt-out +
        # minimum-record reading for cc_authority_frontier_w7, a multi-
        # origin family no single host rule can cover), applied ONLY as a
        # last resort -- a real per-row `license` or a real host-table
        # entry (LIC.resolve_license above) both still win. Never a second,
        # earlier check: this is the exact same `not license_value` gate
        # `require_license` below reads.
        license_value = family_license
    if require_license and not license_value:
        raise ProjectionError(f"{url}: unresolved license")
    if require_embedding_consent:
        # Collection embedding is search indexing, not model training.
        # train_ok remains provenance in the payload but is not an admission
        # gate (build_aw_b1 ConsentGate uses fresh search_ok for the same reason).
        if record.get("consent_search_ok") is not True:
            raise ProjectionError(f"{url}: indexing requires explicit search_ok=true")
        if "consent_train_ok" in record and not isinstance(record["consent_train_ok"], bool):
            raise ProjectionError(f"{url}: consent_train_ok must be boolean when present")
        if not record.get("consent_checked_at"):
            raise ProjectionError(f"{url}: embedding consent lacks checked_at evidence")

    extra = tuple(extra_passthrough)
    forbidden = sorted(set(extra) & SERVING_CONTENT_FIELDS)
    if forbidden:
        raise ProjectionError(
            "content fields cannot enter serving payload: " + ", ".join(forbidden))
    fields = (*BASE_PASSTHROUGH, *extra)
    payload = {key: record[key] for key in fields if record.get(key) is not None}
    if "trust_tier" in payload:
        try:
            trust = float(payload["trust_tier"])
        except (TypeError, ValueError) as exc:
            raise ProjectionError(
                f"invalid trust_tier for {url}: {payload['trust_tier']!r}") from exc
        if not math.isfinite(trust):
            raise ProjectionError(f"non-finite trust_tier for {url}: {payload['trust_tier']!r}")
        payload["trust_tier"] = trust

    record_consent = {key: record[key] for key in CONSENT_KEYS if record.get(key)}
    epochs = dict(ID.EPOCHS)
    if preserve_rule_epochs:
        upstream = record.get("rule_epochs")
        if not isinstance(upstream, dict) or not all(
            isinstance(key, str) and isinstance(value, int)
            for key, value in upstream.items()
        ):
            raise ProjectionError(f"{url}: rule_epochs must be a string-to-int object")
        # An old row keeps the prov_id it was minted with (ADR-117): no entry reads as epoch 1 (conservative; ADR-117 am.1).
        epochs.pop("prov_id", None)  # absent, not 1: a stamped 1 would be a new enum value with no code
        epochs.update(upstream)
        # These are the projection/embedding contracts, never upstream claims.
        epochs.update({"embed": ID.EPOCHS["embed"], "tok": ID.EPOCHS["tok"],
                       "country": ID.EPOCHS["country"]})

    payload.update({
        "url": url,
        "snippet": AC.extractive_snippet(record.get("snippet") or text, max_words=25),
        "tier": tier,
        "country": country or record.get("country"),
        "license": license_value,
        "content_digest": digest,
        "point_id": point_id,
        "rule_epochs": epochs,
        "build_id": "aw_b1",
        **(consent or {}),
        **record_consent,
    })
    if preserve_rule_epochs:
        # task 407 DEFECT FIX. This read `epochs.get("chunk") == 2` and wrote a
        # hardcoded `{"chunk": 2}`, so a chunk@3 row (the 19.2M-row epoch-3
        # wiki slice) and a chunk@4 row (Tim's 2026-08-13 go-forward policy)
        # silently lost their `stamps` field entirely — the branch could only
        # ever fire for the one epoch that existed when it was written. Carry
        # whatever the upstream row actually declares, including the interned
        # `prov_id` that names the exact policy: epoch 3 and epoch 4 differ in
        # a single policy field (`token_hard_max.exact` 2000 -> 2560) and are
        # otherwise byte-identical, so the epoch int alone cannot separate
        # them after the fact.
        upstream_stamps = record.get("stamps")
        chunk_epoch = epochs.get("chunk")
        if isinstance(upstream_stamps, dict) and upstream_stamps:
            if upstream_stamps.get("chunk") not in (None, chunk_epoch):
                raise ProjectionError(
                    f"{url}: stamps.chunk={upstream_stamps.get('chunk')!r} "
                    f"contradicts rule_epochs.chunk={chunk_epoch!r}")
            payload["stamps"] = {**upstream_stamps, "chunk": chunk_epoch}
        elif chunk_epoch != ID.EPOCHS["chunk"]:
            # A non-baseline epoch with no upstream stamp: record the epoch we
            # can prove and say plainly that the policy is unnamed, rather
            # than inventing the policy that happens to be current.
            payload["stamps"] = {"chunk": chunk_epoch, "prov_id": None}
    if tier == "A" and record.get("anchor"):
        payload["anchor"] = record["anchor"]

    master: dict[str, str] = {}
    if record.get("attribution") or attribution:
        master["attribution"] = record.get("attribution") or attribution
    if record.get("license_version") or license_version:
        master["license_version"] = record.get("license_version") or license_version
    if master:
        payload["_master"] = master

    # Source receipts are traceability evidence, not rankable/readable content.
    # They are intentionally master-only and omitted from compact dumps.
    if record.get("bundle_digest") is not None:
        digest_value = record["bundle_digest"]
        if not isinstance(digest_value, str) or len(digest_value) != 64:
            raise ProjectionError(f"{url}: invalid bundle_digest")
        payload["bundle_digest"] = digest_value

    compact = {
        key: value for key, value in payload.items() if value is not None
    }
    leaked = sorted(set(compact) & SERVING_CONTENT_FIELDS)
    if leaked:
        raise ProjectionError("content leaked into serving payload: " + ", ".join(leaked))
    return point_id, dense_text, sparse_text, compact
