"""cytir.core.qdrant — the one legal `QdrantClient(...)` construction site for
NEW code (task 650, migration S8).

The core ratchet test (task 646 S3) fails CI on a new
`QdrantClient(` call site anywhere outside `cytir/core` — the 255 existing call
sites are frozen into `data/core-ratchet-baseline.v1.json` as pre-existing
debt the ratchet does not require fixing today ("Replace the forks only on
the prod-reachable set (class A) — a separate, larger effort this test does
not do"). This module exists so a brand-new caller has a legal home instead
of becoming the 256th baseline entry: `cytir/distribute/dump.py`,
`cytir/distribute/apply.py` and `cytir/distribute/license_check.py` (task 650) are
its first three callers.

Deliberately thin — a connection-pooling or retry policy is future work if
one is ever needed. Two functions, split by what the caller has:
`qdrant_connect(url, *, timeout, **kwargs)` for a remote server, with an
explicit timeout and no default (task 1195); `qdrant_client(location, **kwargs)`
for location dispatch (`:memory:`, a URL, or a local path).
"""
from __future__ import annotations

from typing import Any


def qdrant_client(location: str, **kwargs: Any):
    """`location` is a URL (`http(s)://...`), `:memory:`, or a local
    filesystem path — the last two run with no live Qdrant server at all,
    which is what tests and the CI release rehearsal (task 650 step 4) use.

    `QdrantClient`'s own `location=` positional does NOT dispatch a plain
    string to local on-disk mode — only `:memory:` gets special-cased;
    anything else is treated as a remote URL and fails with an opaque DNS
    error instead of a clear one (verified: `QdrantClient("/tmp/x")` raises
    `ResponseHandlingException: ... name resolution` from inside
    `qdrant_client.qdrant_client.QdrantClient.__init__`, not a local-store
    open). Local-on-disk needs the separate `path=` keyword. This function
    is the one place that distinction is made, so no caller has to
    reproduce it."""
    from qdrant_client import QdrantClient

    if location == ":memory:" or location.startswith(("http://", "https://")):
        return QdrantClient(location, **kwargs)
    return QdrantClient(path=location, **kwargs)


def qdrant_connect(url: str, *, timeout: int, **kwargs: Any):
    """The remote-server constructor: `QdrantClient(url=url, timeout=timeout,
    **kwargs)` and nothing else (task 1195, S3). `timeout` is required and has
    no default, so a 900 s apply site can never silently become a 300 s one;
    extra keywords (e.g. `check_compatibility=False`) pass through unchanged."""
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, timeout=timeout, **kwargs)
