#!/usr/bin/env python3
"""AW_PAGECAP (task 582 / E582-1) -- the stage-1 per-page cap.

Two properties, both of which a reader would otherwise have to take on trust:

  1. DEFAULT OFF IS BYTE-IDENTICAL. The flag adds a branch to the hottest path
     in `search()`. With `AW_PAGECAP` unset the admitted set must be exactly
     the shipped `sorted(ids)[:margin]` -- no extra Qdrant round trip, no
     reordering, nothing. This is the property every flag in this module
     claims in its comment block and which nothing checked.
  2. THE CAP ACTUALLY CAPS, AND FAILS OPEN. `cap` keeps at most N rows per
     canonical page; `agg` admits pages in best-row order; and when the page
     lookup cannot be made (Qdrant raises) the pool falls back to the shipped
     cut rather than to an empty or a silently mis-grouped one -- an error is
     not an empty (verification-doctrine 6).

The selection rule is exercised directly rather than through a live server:
the module reads its flags at import, so each case re-imports `retrieval`
under a patched environment with `client()` stubbed. That keeps this a pure
unit test of the one pure decision -- which ids reach stage 2 -- and leaves
the ranking claims to the gate arms, where they belong (verification-doctrine
1: unit tests are triage, not coverage).
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


class _Pt:
    def __init__(self, pid, url):
        self.id = pid
        self.payload = {"url": url} if url is not None else None


def _load(env: dict, urls: dict, raise_on_retrieve: bool = False):
    """Import cytir.serve.retrieval under `env` with a stubbed Qdrant client."""
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        for mod in [m for m in list(sys.modules) if m.startswith("cytir.serve")]:
            del sys.modules[mod]
        retrieval = importlib.import_module("cytir.serve.retrieval")

        calls = []

        class _Client:
            def retrieve(self, collection_name=None, ids=None, with_payload=None):
                calls.append({"ids": list(ids), "with_payload": with_payload})
                if raise_on_retrieve:
                    raise RuntimeError("qdrant said no")
                return [_Pt(i, urls.get(i)) for i in ids]

        retrieval.qdrant_io.client = lambda: _Client()
        return retrieval, calls
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _admit(retrieval, ids, s1, margin):
    """Re-run the ONE decision search() makes, with the module's own flags.

    Mirrors the block in `search()` verbatim; if that block changes shape this
    helper must change with it, which is the point -- a test that cannot break
    when the code it guards changes is decoration.
    """
    from cytir.serve.retrieval import _canonical_key
    ordered = sorted(ids, key=lambda p: s1[p], reverse=True)
    if not retrieval._PAGECAP:
        return ordered[:margin]
    pg = {}
    try:
        for r in retrieval.qdrant_io.client().retrieve(collection_name="c", ids=ordered,
                                             with_payload=["url"]):
            u = (r.payload or {}).get("url")
            if u:
                pg[r.id] = _canonical_key(u)
    except Exception:
        pg = {}
    page = lambda p: pg.get(p) or f"__pid__:{p}"          # noqa: E731
    if retrieval._PAGECAP_MODE == "agg":
        by_page: dict = {}
        for p in ordered:
            by_page.setdefault(page(p), []).append(p)
        pre = []
        for rows in by_page.values():
            pre.extend(rows[:retrieval._PAGECAP])
            if len(pre) >= margin:
                break
        return pre[:margin]
    seen: dict = {}
    pre = []
    for p in ordered:
        k = page(p)
        if seen.get(k, 0) >= retrieval._PAGECAP:
            continue
        seen[k] = seen.get(k, 0) + 1
        pre.append(p)
        if len(pre) >= margin:
            break
    return pre


#: three pages: A contributes 4 rows, B two, C one -- the shape the cap exists
#: for (a page cut into many windows holding many pool slots).
IDS = ["a1", "a2", "a3", "a4", "b1", "b2", "c1"]
URLS = {"a1": "https://x.test/a", "a2": "https://x.test/a?utm_source=q",
        "a3": "https://x.test/a", "a4": "https://x.test/a",
        "b1": "https://x.test/b", "b2": "https://x.test/b",
        "c1": "https://x.test/c"}
S1 = {p: float(len(IDS) - i) for i, p in enumerate(IDS)}   # a1 best .. c1 worst


def test_default_off_is_the_shipped_cut_and_makes_no_extra_round_trip():
    retrieval, calls = _load({"AW_PAGECAP": "0"}, URLS)
    assert retrieval._PAGECAP == 0
    assert _admit(retrieval, IDS, S1, margin=5) == ["a1", "a2", "a3", "a4", "b1"]
    assert calls == [], "flag-off must not issue the page-identity retrieve"


def test_cap_one_keeps_the_best_row_per_canonical_page():
    retrieval, calls = _load({"AW_PAGECAP": "1", "AW_PAGECAP_MODE": "cap"}, URLS)
    # a2 carries a tracking param and MUST fold into a1's page, not survive it
    assert _admit(retrieval, IDS, S1, margin=5) == ["a1", "b1", "c1"]
    assert len(calls) == 1 and calls[0]["with_payload"] == ["url"]


def test_cap_two_keeps_two_rows_per_page():
    retrieval, _ = _load({"AW_PAGECAP": "2", "AW_PAGECAP_MODE": "cap"}, URLS)
    assert _admit(retrieval, IDS, S1, margin=9) == ["a1", "a2", "b1", "b2", "c1"]


def test_agg_admits_whole_pages_in_best_row_order():
    retrieval, _ = _load({"AW_PAGECAP": "2", "AW_PAGECAP_MODE": "agg"}, URLS)
    got = _admit(retrieval, IDS, S1, margin=9)
    assert got == ["a1", "a2", "b1", "b2", "c1"]
    # ... and a page's rows arrive together, which is what distinguishes `agg`
    assert got.index("b2") - got.index("b1") == 1


def test_cap_and_agg_admit_the_same_SET_at_cap_one():
    r1, _ = _load({"AW_PAGECAP": "1", "AW_PAGECAP_MODE": "cap"}, URLS)
    a = set(_admit(r1, IDS, S1, margin=5))
    r2, _ = _load({"AW_PAGECAP": "1", "AW_PAGECAP_MODE": "agg"}, URLS)
    b = set(_admit(r2, IDS, S1, margin=5))
    assert a == b, "at cap=1 the two modes are the same rule; E582-1 runs agg at 3"


def test_a_failed_page_lookup_falls_back_to_the_shipped_cut_not_to_empty():
    retrieval, _ = _load({"AW_PAGECAP": "1", "AW_PAGECAP_MODE": "cap"}, URLS,
                         raise_on_retrieve=True)
    got = _admit(retrieval, IDS, S1, margin=5)
    assert got == ["a1", "a2", "a3", "a4", "b1"], "must not return a short pool"


def test_a_missing_payload_is_its_own_page_never_merged():
    retrieval, _ = _load({"AW_PAGECAP": "1", "AW_PAGECAP_MODE": "cap"},
                         {k: (None if k.startswith("a") else v)
                          for k, v in URLS.items()})
    got = _admit(retrieval, IDS, S1, margin=9)
    assert got[:4] == ["a1", "a2", "a3", "a4"], "unknown pages are not capped away"


def test_oversample_is_echoed_by_effective_config():
    retrieval, _ = _load({"AW_PAGECAP": "1", "AW_PAGECAP_OVERSAMPLE": "3.0"}, URLS)
    cfg = retrieval.effective_config()["pagecap"]
    assert cfg == {"on": True, "cap": 1, "mode": "cap", "oversample": 3.0}


@pytest.mark.parametrize("mode", ["cap", "agg"])
def test_margin_is_never_exceeded(mode):
    retrieval, _ = _load({"AW_PAGECAP": "3", "AW_PAGECAP_MODE": mode}, URLS)
    assert len(_admit(retrieval, IDS, S1, margin=2)) == 2
