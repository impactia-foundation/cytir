#!/usr/bin/env python3
"""Guard for `consent_resolver.search_only_declines` (task 629).

The bug this exists for is a READING bug, not a scoring bug. `bucket()`'s strict
default folds `train_ok=False, search_ok=True` into `'declined'` — correct for its
legacy merge callers, wrong for search admission — and the CLI printed only the
bucket counts. In task 629 that read `declined: 4` for four Pool A origins that
are perfectly admissible for search under ADR-024 amendment 5, and nearly refused
three families on the strength of a label.

So these tests assert two things that must not drift apart: that the strict
bucket still says `declined` (nothing about scoring changed), and that the
reporting helper names exactly those hosts so a reader cannot trip on it.

**On the import dance below, because it is load-bearing and looks like clutter.**
`consent_resolver` computes `UA_STR = user_agent()` at module scope and *refuses
to import* without a configured crawler identity — correct behaviour
(`user_agent.UnidentifiedCrawler`: an unidentified crawl is not honest), and not
what these tests are about. But `UA_STR` is bound on FIRST import, process-wide,
so a test module that leaves `AW_OPERATOR_*` set changes what every later module's
first import sees. That is exactly how this file's first draft turned
`test_consent_identity.py::test_consent_bulk_uses_the_same_identity_not_a_second_copy`
into a collection-order-dependent failure. So the identity is set, the import is
taken, and anything this module added to the environment is removed again.
"""
from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

_ADDED = [name for name in ("AW_OPERATOR_URL", "AW_OPERATOR_CONTACT")
          if name not in os.environ]
os.environ.setdefault("AW_OPERATOR_URL", "https://websear.ch")
os.environ.setdefault("AW_OPERATOR_CONTACT", "bender@impactia.org")
try:
    import cytir.consent.resolver as cr  # noqa: E402
finally:
    for _name in _ADDED:
        os.environ.pop(_name, None)
# What this module itself still holds after its import. Other modules' collection-time
# setdefault() calls (test_hn_outlinks.py and friends) can leave the names set; that is
# not this module's footprint, so it is not asserted here.
_LEFT_BEHIND = [name for name in _ADDED if name in os.environ]


def rec(**kw):
    base = {"status": "ok", "search_ok": True, "train_ok": True,
            "signals": {"robots": {}}}
    base.update(kw)
    return base


#: The real 2026-09-22 Pool A readings (task 629 §13.2): github.com disallows
#: Bytespider; rsi/srf/rts disallow CCBot plus the training-token set; swissinfo
#: is fully open. Sealed in the 629 pool-a consent artifact.
POOL_A = {
    "github.com": rec(train_ok=False),
    "www.rsi.ch": rec(train_ok=False),
    "www.srf.ch": rec(train_ok=False),
    "www.rts.ch": rec(train_ok=False),
    "www.swissinfo.ch": rec(),
    "openalex.org": rec(),
    "www150.statcan.gc.ca": rec(),
}


def test_the_strict_bucket_still_calls_a_train_only_decline_declined():
    """Nothing about scoring changed; only the reporting."""
    assert cr.bucket(POOL_A["github.com"]) == "declined"
    assert cr.bucket(POOL_A["www.swissinfo.ch"]) == "ok"


def test_search_only_declines_names_exactly_the_train_only_hosts():
    assert cr.search_only_declines(POOL_A) == [
        "github.com", "www.rsi.ch", "www.rts.ch", "www.srf.ch"]


def test_a_real_decline_is_not_reported_as_search_only():
    """A host that withheld SEARCH must never appear in this list — that would
    invert the warning into a licence to index something we may not."""
    records = dict(POOL_A)
    records["declined.example"] = rec(search_ok=False, train_ok=False)
    assert "declined.example" not in cr.search_only_declines(records)


def test_an_unknown_host_is_not_reported_as_search_only():
    """`unknown` is held/retryable, a different disposition from either."""
    records = dict(POOL_A)
    records["held.example"] = rec(status="unknown", search_ok=None,
                                  train_ok=None)
    assert cr.bucket(records["held.example"]) == "unknown"
    assert "held.example" not in cr.search_only_declines(records)


def test_empty_when_every_host_is_fully_open():
    assert cr.search_only_declines({"a.example": rec()}) == []


def test_the_helper_is_stable_under_ordering():
    """Callers print it; a set's iteration order would make the line unstable."""
    forward = cr.search_only_declines(POOL_A)
    backward = cr.search_only_declines(dict(reversed(list(POOL_A.items()))))
    assert forward == backward == sorted(forward)


def test_this_module_leaves_no_operator_identity_behind():
    """The coupling this module was rewritten to remove, asserted rather than
    described: if its import leaks AW_OPERATOR_*, a later module's first import of
    consent_resolver silently binds a different UA_STR."""
    assert _LEFT_BEHIND == [], (
        f"{_LEFT_BEHIND} leaked out of this module's import; a later first import "
        "of consent_resolver would bind a UA this module chose")
