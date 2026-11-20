"""The one home for JSON digest canonicalizers (task 1181, S3).

Sealed hashes are computed over these bytes. Never re-implement them locally
(engineering-doctrine: a mirror contract). Two shapes live here:

* ``canonical_line_bytes`` -- compact sorted JSON plus a trailing newline.
* ``canonical_sha256``     -- sha256 hex of compact sorted JSON, no newline.

Both keep the json default ``ensure_ascii=True``. The ``ensure_ascii=False``
family (task 1191) has its own named functions: ``canonical_json_utf8`` (str),
``canonical_utf8_bytes`` (utf-8 bytes) and ``canonical_utf8_sha256`` (hex). Strict
variants (``allow_nan=False``, the noop point bytes) and ``indent`` forms are
different bytes and stay with their owners.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_line_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def canonical_sha256(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def canonical_json_utf8(value: Any) -> str:
    """Compact sorted JSON, non-ASCII kept as is (``ensure_ascii=False``)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_utf8_bytes(value: Any) -> bytes:
    return canonical_json_utf8(value).encode("utf-8")


def canonical_utf8_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_utf8_bytes(value)).hexdigest()
