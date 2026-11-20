"""Shared Common Crawl content-digest normalization and matching."""
from __future__ import annotations

import base64
import binascii
import hashlib


# SHA-1 of the zero-byte payload.  Common Crawl index rows may spell digests
# either as bare base32 or ``sha1:<base32>``; empty HTML cannot yield a parent
# and must remain eligible for an older-crawl lookup (Task 426, 2026-08-24).
EMPTY_SHA1_BASE32 = base64.b32encode(
    hashlib.sha1(b"").digest()).decode("ascii").rstrip("=")


def normalize_sha1(value: str) -> str | None:
    digest = str(value or "").strip()
    if ":" in digest:
        algorithm, digest = digest.split(":", 1)
        if algorithm.casefold() != "sha1":
            return None
    digest = digest.rstrip("=").upper()
    try:
        decoded = base64.b32decode(
            digest + "=" * (-len(digest) % 8), casefold=True)
    except (binascii.Error, ValueError):
        return None
    return digest if len(decoded) == hashlib.sha1().digest_size else None


def matches_sha1(body: bytes, expected: str) -> bool:
    observed = base64.b32encode(
        hashlib.sha1(body).digest()).decode("ascii").rstrip("=")
    return normalize_sha1(expected) == observed


def is_empty_sha1(value: str) -> bool:
    return normalize_sha1(value) == EMPTY_SHA1_BASE32
