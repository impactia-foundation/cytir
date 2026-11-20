"""Pure downstream-use policy for resolved consent receipts.

The live resolver owns signal collection.  This module owns the deliberately
small classification step shared by the corpus gate and source-derived card
builders.  Keeping it dependency-free prevents search-index admission from
forking between those two principal paths.
"""
from __future__ import annotations


WEBSITE_RECEIPT_SCHEMA = "aw.entity-card.website-receipt.v1"


def gate_bucket(rec: dict, *, search_indexing: bool = False,
                accept_search_grant: bool = False,
                unclear_ok: bool = False, legacy_bucket=None) -> str:
    """Classify one resolved receipt for the declared downstream use.

    A collection embedding is a non-readable search-index representation, not
    model training.  An explicit search-index caller therefore admits a live
    resolved ``search_ok=true`` receipt while retaining ``train_ok`` as the
    separate training/export restriction.  Historical callers keep the older
    resolver bucket through ``legacy_bucket``.
    """
    if search_indexing:
        if rec.get("status") == "unknown":
            return "unknown"
        if rec.get("status") != "ok" or rec.get("search_ok") is not True:
            return "declined"
        return "ok" if rec.get("train_ok") is True else "ok_search_only"
    if rec.get("status") == "unknown":
        return "unknown"
    if legacy_bucket is None:
        raise ValueError("non-search-index classification requires legacy_bucket")
    return legacy_bucket(rec, accept_search_grant=accept_search_grant,
                         unclear_ok=unclear_ok)


def require_resolved_search_index_receipt(rec: dict) -> str:
    """Return its admitted bucket or fail loudly on an incomplete receipt."""
    if not isinstance(rec, dict):
        raise ValueError("search-index receipt must be an object")
    for key in ("status", "search_ok", "train_ok", "checked_at"):
        if key not in rec or rec[key] in (None, ""):
            raise ValueError(f"search-index receipt lacks {key}")
    if not isinstance(rec["search_ok"], bool) or not isinstance(rec["train_ok"], bool):
        raise ValueError("search-index receipt flags must be boolean")
    bucket = gate_bucket(rec, search_indexing=True)
    if bucket not in {"ok", "ok_search_only"}:
        raise ValueError(
            "search-index receipt requires live resolved status=ok and search_ok=true")
    return bucket
