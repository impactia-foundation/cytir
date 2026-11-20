"""Search operators (task 2640): the typed form of the Qdrant filter an agent can ask for.

`site:` / `-site:` match the HOST and every sub-host (`site:admin.ch` = `admin.ch` and `*.admin.ch`),
which needs the list of hosts the served collection holds. That list (and the `source_type`
vocabulary, which is open: the corpus grows new types) comes from ONE Qdrant facet call per
collection, cached in this module and refreshed at most hourly on a request. With no operator
nothing here runs: the filter is None and serving is byte-identical.
"""
from __future__ import annotations

import threading
import time

from qdrant_client import models as qm

from . import qdrant_io
from .query import Operators, _norm_host  # noqa: F401  (Operators lives in query: pure, no Qdrant)
from .schema import COLLECTION

REFRESH_S = 3600.0        # a loaded table is re-read at most hourly
RETRY_S = 60.0            # after a failed read, do not hit the facet again for a minute
FACET_LIMIT = 1_000_000   # twin 2026-11-10: 411,094 distinct hosts, 1.05 s, 15.7 MB
MAX_HOSTS = 2000          # one operator may expand to at most this many hosts


class OperatorError(ValueError):
    """A bad operator value; the API turns it into a 400."""


class _Table:
    def __init__(self):
        self.hosts: frozenset = frozenset()
        self.sources: frozenset = frozenset()
        self.loaded_at: float | None = None
        self.failed_at: float | None = None
        self.lock = threading.Lock()


_T = _Table()


def _facet(key: str, limit: int) -> frozenset:
    r = qdrant_io.client().facet(collection_name=COLLECTION, key=key, limit=limit, exact=False)
    return frozenset(h.value for h in r.hits)


def _ensure_table(now: float | None = None) -> bool:
    """True when a table is available (possibly stale). A refresh runs on the request that finds the
    table old, under a lock; a failure keeps the old table and backs off for RETRY_S."""
    now = time.monotonic() if now is None else now
    t = _T
    fresh = t.loaded_at is not None and now - t.loaded_at < REFRESH_S
    backing_off = t.failed_at is not None and now - t.failed_at < RETRY_S
    if fresh or backing_off:
        return t.loaded_at is not None
    with t.lock:
        # re-check under the lock: requests that waited here must not each re-run the facet
        # calls after the first one failed (or loaded) while they queued
        if t.loaded_at is not None and now - t.loaded_at < REFRESH_S:
            return True
        if t.failed_at is not None and now - t.failed_at < RETRY_S:
            return t.loaded_at is not None
        try:
            t.hosts, t.sources = _facet("domain", FACET_LIMIT), _facet("source_type", 500)
            t.loaded_at, t.failed_at = now, None
        except Exception:  # noqa: BLE001 — the operator degrades to exact host, never 500s
            t.failed_at = now
    return t.loaded_at is not None


def expand_host(domain: str, hosts) -> list[str]:
    """`domain` and every sub-host of it that the index holds."""
    suffix = "." + domain
    out = sorted(h for h in hosts if h == domain or h.endswith(suffix))
    return out or [domain]


def _host_cond(domains, table_ok: bool):
    hosts = set()
    for d in domains:
        got = expand_host(d, _T.hosts) if table_ok else [d]
        if len(got) > MAX_HOSTS:
            raise OperatorError(f"site:{d} matches {len(got)} hosts (limit {MAX_HOSTS}); "
                                "use a narrower domain")
        hosts.update(got)
    vals = sorted(hosts)
    if len(vals) == 1:
        return qm.FieldCondition(key="domain", match=qm.MatchValue(value=vals[0]))
    return qm.FieldCondition(key="domain", match=qm.MatchAny(any=vals))


def build(ops: Operators):
    """(Qdrant filter or None, echo of what was applied). Raises OperatorError on an unknown
    `source`. lang/country are NOT in the filter: they travel through the existing parameters."""
    echo: dict = {}
    notes: list = []
    must, must_not = [], []
    if ops.site or ops.exclude_site or ops.source:
        table_ok = _ensure_table()
        if (ops.site or ops.exclude_site) and not table_ok:
            notes.append("host table unavailable: site: matched the exact host only, not sub-hosts")
        if ops.source:
            if table_ok:
                unknown = [s for s in ops.source if s not in _T.sources]
                if unknown:
                    # never a 400: `import os source:main` is a normal query. The caller puts
                    # the token back into the query text (echo["unknown_source"]).
                    notes += [f"source:{s} is not a known source_type; kept as text (valid: official, reference, docs, ...)" for s in unknown]
                    echo["unknown_source"] = unknown
            else:
                notes.append("source vocabulary unavailable: source: was not validated")
            known = [s for s in ops.source if s not in echo.get("unknown_source", ())]
            if known:
                must.append(qm.FieldCondition(
                    key="source_type",
                    match=qm.MatchValue(value=known[0]) if len(known) == 1
                    else qm.MatchAny(any=known)))
                echo["source"] = known
        if ops.site:
            must.append(_host_cond(ops.site, table_ok))
            echo["site"] = list(ops.site)
        if ops.exclude_site:
            must_not.append(_host_cond(ops.exclude_site, table_ok))
            echo["exclude_site"] = list(ops.exclude_site)
    if notes:
        echo["note"] = "; ".join(notes)
    if not (must or must_not):
        return None, echo
    return qm.Filter(must=must or None, must_not=must_not or None), echo


def and_filter(base, extra):
    """`base` AND `extra`; either may be None. Nested so base's own `should` stays an OR."""
    if extra is None:
        return base
    if base is None:
        return extra
    return qm.Filter(must=[base, extra])


def host_predicate(ops: Operators):
    """host -> bool for the same `site:` / `-site:` sets, for results that are not Qdrant-filtered
    (the authorities sidecar). None when the operators name no host."""
    if not (ops.site or ops.exclude_site):
        return None
    ok = _ensure_table()

    def hosts(domains):
        out = set()
        for d in domains:
            out.update(expand_host(d, _T.hosts) if ok else [d])
        return out
    inc, exc = hosts(ops.site), hosts(ops.exclude_site)
    return lambda host: (not inc or host in inc) and host not in exc
