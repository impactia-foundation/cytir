#!/usr/bin/env python3
"""Offline unit tests for the HOST-granularity discovery prep (2026-07-03 fix).

The bug: dedup vs the live index worked at REGISTRABLE-domain granularity, so
fedlex.admin.ch being indexed hid every admin.ch agency host (estv/bag/seco/…)
from discovery. These tests pin the fixed contract:

  * host-level candidates re-derived from the endorsement edge graph,
  * exact-host live dedup (fedlex.admin.ch excluded, estv.admin.ch eligible),
  * pattern-class noise rules (personal-publishing clouds always dropped;
    dev-docs cloud TENANTS eligible at host granularity, apex dropped),
  * registrable-level fallback for candidates without edge provenance,
  * legacy `registrable` mode still reproduces the old behaviour,
  * cc_columnar.match_prefix maps host surtkeys to the right queue key
    (the resolve_domains mis-keying fix).

Run: python3 -m pytest tests/test_aw_prep_discovery_queue.py
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cytir.acquire.prep_discovery_queue import (build_queue, class_basis, classify,  # noqa: E402
                                             host_scores_from_edges, is_noise, norm_host)

# --- fixtures ----------------------------------------------------------------
# One federal-gov seed (trust 1.0) endorsing: an indexed host (fedlex), two
# agency hosts of the same registrable, a dev-docs cloud tenant, a blog cloud,
# and a single-owner Swiss site. Plus a portals seed (trust 0.8) for scoring.
ENDORSE = {
    "errors": [],
    "edges": {
        "seed-a.ch": {"class": "federal-gov", "status": "ok", "n_pages": 10,
                      "targets": {
                          "https://www.fedlex.admin.ch/eli/cc/1999/404": 3,
                          "https://www.estv.admin.ch/estv/de/home/mwst.html": 4,
                          "https://www.bag.admin.ch/bag/de/home.html": 2,
                          "https://leaflet.github.io/reference.html": 9,
                          "https://foo.blogspot.com/post": 5,
                          "https://github.io/": 2,
                          "https://www.snb.ch/en/stats": 1,
                      }},
        "seed-b.ch": {"class": "portals", "status": "ok", "n_pages": 5,
                      "targets": {
                          "https://estv.admin.ch:443/vat": 2,
                          "https://www.snb.ch/de/rates": 5,
                      }},
    },
}
# registrable-level candidate list (the legacy artifact): note admin.ch hosts
# were ALREADY collapsed/hidden upstream; a no-edge candidate rides along.
CANDS = [
    {"domain": "snb.ch", "endorse_score": 5.0},
    {"domain": "estv.admin.ch", "endorse_score": 6.0},   # collapsed reg legacy
    {"domain": "hn-only-source.org", "endorse_score": 7.0},  # no edge provenance
    {"domain": "admin.chhttps:", "endorse_score": 1.0},      # malformed junk
]
LIVE = {"result": {"hits": [{"value": "fedlex.admin.ch", "count": 1},
                            {"value": "docs.python.org", "count": 1}]}}


def _run(granularity="host"):
    from cytir.acquire.prep_discovery_queue import load_live_hosts  # noqa: F401 (shape parity)
    hosts = {norm_host(h["value"]) for h in LIVE["result"]["hits"]}
    regs = set()
    for h in hosts:
        from cytir.acquire.prep_discovery_queue import reg_domain
        regs.add(reg_domain(h))
        regs.add(h)
    kept, noise, indexed = build_queue(CANDS, ENDORSE, hosts, regs, granularity)
    return {k["domain"]: k for k in kept}, dict(noise), set(indexed)


# --- tests --------------------------------------------------------------------
def test_norm_host():
    assert norm_host("WWW.ESTV.Admin.CH") == "estv.admin.ch"
    assert norm_host("www2.example.com") == "example.com"
    assert norm_host("estv.admin.ch:443") == "estv.admin.ch"
    assert norm_host("news.admin.ch.") == "news.admin.ch"
    assert norm_host("wwwx.example.com") == "wwwx.example.com"  # not www\d*.


def test_host_scores_reaggregate_edges():
    score, hcls, rcls = host_scores_from_edges(ENDORSE)
    # federal-gov trust 1.0, vote min(4,5) + portals 0.8 * min(2,5)
    assert abs(score["estv.admin.ch"] - (1.0 * 4 + 0.8 * 2)) < 1e-9
    assert hcls["estv.admin.ch"] == {"federal-gov", "portals"}
    assert "admin.ch" in rcls and "federal-gov" in rcls["admin.ch"]
    # www folded: both snb targets land on one host
    assert abs(score["snb.ch"] - (1.0 * 1 + 0.8 * 5)) < 1e-9


def test_host_dedup_carveout_agency_layer():
    """THE bug fix: indexed host dedups exactly; sibling agency hosts surface."""
    kept, noise, indexed = _run("host")
    assert "fedlex.admin.ch" in indexed          # already live -> still excluded
    assert "estv.admin.ch" in kept               # sibling host now eligible
    assert "bag.admin.ch" in kept
    assert kept["estv.admin.ch"]["class"] == "gov"
    assert kept["estv.admin.ch"]["swiss"] is True
    assert kept["estv.admin.ch"]["reg"] == "admin.ch"
    # sensible rank: gov + swiss boost -> authority 1.5, score = es * 1.5
    assert abs(kept["estv.admin.ch"]["score"] - round(5.6 * 1.5, 2)) < 1e-6


def test_legacy_registrable_mode_reproduces_old_hiding():
    kept, noise, indexed = _run("registrable")
    assert "estv.admin.ch" in indexed            # reg admin.ch live -> hidden
    assert not any(k.endswith("admin.ch") for k in kept)


def test_platform_pattern_classes():
    kept, noise, indexed = _run("host")
    # dev-docs cloud TENANT eligible at host granularity...
    assert "leaflet.github.io" in kept
    # ...but the bare platform apex is noise
    assert noise.get("github.io") == "github.io"
    # personal-publishing clouds dropped at any granularity
    assert "foo.blogspot.com" in noise
    # and in legacy mode the tenant is dropped too (old blanket rule)
    kept_r, noise_r, _ = _run("registrable")
    assert "leaflet.github.io" not in kept_r
    assert is_noise("leaflet.github.io", "registrable") == "github.io"
    assert is_noise("leaflet.github.io", "host") is None


def test_reg_fallback_for_no_edge_candidates():
    kept, noise, indexed = _run("host")
    # candidate with no edge provenance folds in at reg granularity, own score
    assert "hn-only-source.org" in kept
    assert kept["hn-only-source.org"]["endorse_score"] == 7.0
    # edge-covered registrables do NOT get a duplicate reg fallback row:
    # snb.ch appears once, with the edge-derived score (not the file's 5.0)
    assert abs(kept["snb.ch"]["endorse_score"] - 5.0) < 0.01  # 1*1 + 0.8*5 = 5.0
    # malformed junk is dropped as non-domain
    assert "admin.chhttps:" not in kept and "admin.chhttps:" in dict(noise)


def test_docs_python_live_does_not_hide_siblings():
    """Single-owner registrables: an indexed host no longer shadows siblings."""
    endorse = {"edges": {"s.ch": {"class": "portals", "targets": {
        "https://docs.python.org/3/library/json.html": 3,
        "https://www.python.org/downloads/": 2}}}}
    kept, _, indexed = build_queue([], endorse, {"docs.python.org"},
                                   {"python.org", "docs.python.org"}, "host")
    doms = {k["domain"] for k in kept}
    assert "docs.python.org" in indexed and "python.org" in doms


def test_classify_no_longer_demotes_by_denylist_task440():
    """SUPERSEDES test_classify_commercial_guard_task203.

    Task 203 capped the fallback with a hardcoded 90-host COMMERCIAL_DENY set,
    and this test used to assert `classify("heathrow.com", edu_seed) ==
    "commercial"` -- i.e. it pinned a host by name. That is forbidden doctrine
    (ADR-043 A1 §7; Tim 2026-07-28 "no host special-cased") and it did not work:
    the list caught 503 of the 90,598 hosts holding a high-trust class purely by
    inheritance, and none of task 438's six measured gov-tier commercial hosts.

    classify() no longer demotes anything. It reports which branch fired, and
    trust_model prices an endorsed class from the vendor-content floor. So the
    commercial host and the genuine authority now get the SAME class from the
    same evidence -- which is honest, because the link graph genuinely cannot
    tell them apart -- and neither one gets mission-tier trust for it.
    """
    edu_seed = {"universities-research"}
    for host in ("heathrow.com", "numpy.org", "some-random-seo.de"):
        cls, why = classify(host, edu_seed)
        assert cls == "edu", (host, cls)
        assert class_basis(why) == "endorsed", (host, why)
    # a credential in the hostname still short-circuits to branch 1
    assert classify("estv.admin.ch", edu_seed) == ("gov", "gov TLD/keyword")
    assert class_basis(classify("mit.edu", edu_seed)[1]) == "self-signal"
    # no endorsement at all -> unclassified
    assert classify("heathrow.com", set())[0] == "other"


def test_build_queue_emits_basis_and_corroboration_task440():
    """The queue row must carry what discovery_ingest needs to price trust.
    Pre-439 it carried `endorse_score` and `by_classes` to disk and the ingester
    read neither."""
    endorse = {"edges": {
        "uni-a.example": {"class": "universities-research",
                          "targets": {"http://cited.example/p": 3}},
        "uni-b.example": {"class": "universities-research",
                          "targets": {"http://cited.example/q": 2}},
        "uni-c.example": {"class": "universities-research",
                          "targets": {"http://lonely.example/p": 9}},
    }}
    kept, _, _ = build_queue([], endorse, set(), set())
    rows = {r["domain"]: r for r in kept}
    assert rows["cited.example"]["class_basis"] == "endorsed"
    assert rows["cited.example"]["n_endorser_regs"] == 2
    assert rows["lonely.example"]["n_endorser_regs"] == 1


def test_match_prefix_host_mapping():
    from cytir.acquire.cc_columnar import match_prefix
    prefixes = sorted(["ch,admin)", "ch,admin,estv)", "ch,snb)"])
    assert match_prefix("ch,admin,estv)/vat/rates", prefixes) == "ch,admin,estv)"
    assert match_prefix("ch,admin)/gov/start", prefixes) == "ch,admin)"
    assert match_prefix("ch,admin,bag)/home", prefixes) is None   # not queued
    assert match_prefix("ch,snb)/en", prefixes) == "ch,snb)"
    assert match_prefix("de,example)/x", prefixes) is None


def test_the_package_module_uses_the_package_reg_domain_and_no_checkout_path():
    """Task 825: the module lives in `cytir/acquire`; it imports `cytir.core.common.reg_domain` (no
    `sys.path` insert, no fixed checkout path, no fallback copy)."""
    import cytir.acquire.prep_discovery_queue as m
    import cytir.core.common as common
    assert m.reg_domain is common.reg_domain
    code = [ln for ln in pathlib.Path(m.__file__).read_text(encoding="utf-8").splitlines()
            if not ln.lstrip().startswith("#")]
    assert not [ln for ln in code if "sys.path" in ln or "/home/" in ln]


# --- golden over one REAL Common Crawl record (task 825, pinned BEFORE the move) ------------------
GOLDEN = json.loads((ROOT / "tests/fixtures/discovery_prep/golden.json").read_text())


def _real_seed_run():
    """`process_seed` on the real admin.ch record (tests/fixtures/cc_real_record), no network."""
    import os
    os.environ.setdefault("AW_ALLOW_UNIDENTIFIED_CRAWL", "1")
    from cytir.acquire import extract_outlinks as eo
    fix = ROOT / "tests/fixtures/cc_real_record"
    rec = json.loads((fix / "record.json").read_text())
    member = (fix / "member.gz").read_bytes()

    class Resp:
        status_code = 206
        content = member

    class Client:
        def get(self, url, headers=None, **kw):
            return Resp()

    pre = [{k: rec[k] for k in ("url", "filename", "offset", "length")}]
    return eo, eo.process_seed(Client(), {"domain": "admin.ch", "class": "federal-gov"},
                               "CC-MAIN-2026-25", 5, 1.0, fetch_sleep=0, pre=pre)


def test_outlink_step_on_the_real_record_equals_the_golden():
    eo, (dom, kept, n_pages, n_recs, status) = _real_seed_run()
    g = GOLDEN["process_seed"]
    assert (dom, n_pages, n_recs, status) == (g["domain"], g["n_pages"], g["n_recs"], g["status"])
    assert status == "ok" and kept                       # smoke: >= 1 endorsed target
    assert dict(sorted(kept.items())) == g["kept"]
    assert eo.TRUST == GOLDEN["TRUST"]


def test_classifier_and_queue_on_the_real_record_equal_the_golden():
    from urllib.parse import urlparse
    from cytir.acquire import prep_discovery_queue as pq
    _, (dom, kept, n_pages, n_recs, status) = _real_seed_run()
    hosts = sorted({norm_host(urlparse(u).netloc) for u in kept})
    classes = {h: list(classify(h, {"federal-gov"})) for h in hosts}
    assert classes == GOLDEN["classify_federal_gov"] and classes   # >= 1 classified row
    endorse = {"errors": [], "edges": {"admin.ch": {"class": "federal-gov", "status": status,
                                                     "n_pages": n_pages, "n_recs": n_recs,
                                                     "targets": kept}}}
    for g in ("host", "registrable"):
        k, noise, idx = build_queue(GOLDEN["cands"], endorse, {"fedlex.admin.ch"},
                                    {"admin.ch"} if g == "registrable" else set(), g)
        want = GOLDEN["build_queue"][g]
        # equal scores tie and `build_queue` orders by -score only (a set iteration): compare
        # in a total order, never touch build_queue in a move PR
        order = lambda rows: sorted(rows, key=lambda r: (-r["score"], r["domain"]))  # noqa: E731
        assert order(k) == order(want["kept"]) and k and want["kept"]
        assert sorted(map(list, dict(noise).items())) == want["noise"]
        assert sorted(idx) == want["indexed"]
    assert pq.SEED_TRUST == GOLDEN["SEED_TRUST"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok   {fn.__name__}")
    print(f"all {len(fns)} prep_discovery_queue tests passed")
