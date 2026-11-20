#!/usr/bin/env python3
"""Tests for the source runner (cytir.sources.runner) and the operator doc (task 434)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from tests.helpers.repo_files import repo_file_or_skip
import cytir.sources.pack_loader as PL  # noqa: E402
from cytir.sources import runner as SR  # noqa: E402


# --- the invariant that matters most ----------------------------------------

def test_every_ratified_route_has_an_engine():
    """No route may be silently unhandled.

    If policy gains a route and nothing maps it, a pack declaring it would fail
    at runtime instead of at review. This test is the tripwire.
    """
    policy = json.loads((REPO / "data" / "acquisition-policy.v1.json").read_text())
    routes = {r["route"] for r in policy["route_precedence"]}
    handled = set(SR.PACK_FETCH_MODES) | SR.EXTRACTOR_ROUTES
    assert routes == handled, (
        f"unhandled: {sorted(routes - handled)}; "
        f"handled-but-not-in-policy: {sorted(handled - routes)}")


def test_engine_sets_are_disjoint():
    """A route dispatched two ways would be a coin flip."""
    assert not (set(SR.PACK_FETCH_MODES) & SR.EXTRACTOR_ROUTES)


def test_route_params_covers_every_route():
    """Every ratified route needs a declared param contract."""
    policy = json.loads((REPO / "data" / "acquisition-policy.v1.json").read_text())
    routes = {r["route"] for r in policy["route_precedence"]}
    assert routes == set(PL.ROUTE_PARAMS)


# --- fail loud ---------------------------------------------------------------

def _plan(route, **kw):
    base = dict(
        pack_id="p", pack_path="", route_id="r", route=route, params={},
        select=None, lang=None, delay_s=None, cap=None, extractor=None,
        engine="default", config_ref=None,
        reason_receipt=None, no_fallback=False, route_license=None, route_trust=None, adapter="dump", country="CH", layer="news",
        trust_tier=0.9, tier=None, license={}, consent_basis="",
        consent_train_ok_override=None, preservation_profiles=[],
        freshness="static", status="staged", host=None)
    base.update(kw)
    return PL.AcquisitionPlan(**base)


def test_extractor_route_without_extractor_fails_loud():
    """A missing extractor must raise, not quietly no-op into a green run."""
    class Args:
        extractor_args = None
        dry_run = False
    with pytest.raises(SR.RunError, match="needs an"):
        SR.run_via_extractor(_plan("documented_source_api"), args=Args())


def test_extractor_pointing_at_nothing_fails_loud():
    class Args:
        extractor_args = None
        dry_run = False
    with pytest.raises(SR.RunError, match="does not exist"):
        SR.run_via_extractor(
            _plan("documented_source_api", extractor="nope/nope_xyz.py"),
            args=Args())


def test_extractor_environment_binds_the_exact_route(monkeypatch, tmp_path):
    """Family adapters receive route identity/config without bespoke CLIs."""
    captured = {}

    class Proc:
        returncode = 0

    def fake_run(cmd, **kwargs):
        captured.update(kwargs["env"])
        return Proc()

    monkeypatch.setattr(SR.subprocess, "run", fake_run)

    class Args:
        extractor_args = None
        dry_run = False
        out_dir = str(tmp_path)
        limit = 7
        budget = 19
        no_l0 = True

    ref = {"path": "data/cov351_routes.yaml",
           "pack": "regdocs1", "source": "finma_docs2"}
    p = _plan("documented_source_api", pack_id="ch-regulator-docs",
              route_id="www.finma.ch", extractor="cytir/sources/cov/adapter.py",
              engine="extractor", config_ref=ref)
    SR.run_via_extractor(p, args=Args())
    assert captured["AW_SOURCE_PACK_ID"] == "ch-regulator-docs"
    assert captured["AW_SOURCE_ROUTE_ID"] == "www.finma.ch"
    assert json.loads(captured["AW_SOURCE_CONFIG_REF"]) == ref
    assert captured["AW_SOURCE_LIMIT"] == "7"
    assert captured["AW_SOURCE_NO_L0"] == "1"


def test_explicit_extractor_engine_overrides_route_default(monkeypatch, tmp_path):
    """A rich sitemap family may deliberately reuse its proven extractor."""
    called = []
    monkeypatch.setattr(SR, "run_via_extractor",
                        lambda plan, **kw: called.append(plan.route_id) or {})

    class Args:
        out_dir = str(tmp_path)
        limit = 0
        budget = 10
        no_l0 = True
        skip_consent = True
        dry_run = True
        extractor_args = None

    p = _plan("source_owned_sitemap_fetch", host="x.ch",
              params={"urls": ["https://x.ch/sitemap.xml"]},
              extractor="cytir/sources/cov/adapter.py", engine="extractor",
              config_ref={"path": "data/cov351_routes.yaml",
                          "pack": "regdocs1", "source": "finma_docs2"})
    SR.run_pack_plans({"id": "p", "_path": "", "adapter": "sitemap"},
                      [p], Args())
    assert called == ["r"]


# --- consent gating ----------------------------------------------------------

def test_blocked_host_skips_its_plan(monkeypatch, tmp_path):
    """A host that does not clear consent must not be fetched, and the receipt
    must say so rather than silently omitting it."""
    monkeypatch.setattr(SR, "resolve_consent",
                        lambda host, **kw: {"host": host, "bucket": "declined",
                                            "status": "ok", "search_ok": False,
                                            "train_ok": False, "checked_at": "x",
                                            "reasons": ["robots disallow"]})

    class Args:
        out_dir = str(tmp_path)
        limit = 0
        budget = 10
        no_l0 = True
        skip_consent = False
        dry_run = True
        extractor_args = None

    plans = [_plan("source_owned_sitemap_fetch", host="x.ch",
                   params={"urls": ["https://x.ch/s.xml"]})]
    receipt = SR.run_pack_plans({"id": "p", "_path": "", "adapter": "sitemap"},
                                plans, Args())
    assert receipt["consent"][0]["bucket"] == "declined"
    assert receipt["plans"][0]["skipped"] is True
    assert receipt["results"] == {"ran": 0}


def test_ok_search_only_still_runs(monkeypatch, tmp_path):
    """search_ok + train_ok=false is a GRANT for indexing, not a refusal.
    iiep.unesco.org is exactly this case and must not be dropped."""
    monkeypatch.setattr(SR, "resolve_consent",
                        lambda host, **kw: {"host": host, "bucket": "ok_search_only",
                                            "status": "ok", "search_ok": True,
                                            "train_ok": False, "checked_at": "x",
                                            "reasons": []})

    class Args:
        out_dir = str(tmp_path)
        limit = 0
        budget = 10
        no_l0 = True
        skip_consent = False
        dry_run = True
        extractor_args = None

    plans = [_plan("source_owned_sitemap_fetch", host="x.ch",
                   params={"urls": ["https://x.ch/s.xml"]})]
    receipt = SR.run_pack_plans({"id": "p", "_path": "", "adapter": "sitemap"},
                                plans, Args())
    assert receipt["plans"][0]["skipped"] is False


def test_hostless_plan_still_gets_a_consent_check():
    """A dump/api route addressed by URL must not slip past the gate."""
    p = _plan("source_owned_bulk_dump",
              params={"url_template": "https://fedlex.data.admin.ch/filestore/x",
                      "index_endpoint": "https://fedlex.data.admin.ch/sparqlendpoint",
                      "format": "xml"})
    assert p.effective_hosts() == ["fedlex.data.admin.ch"]


def test_secondary_origins_are_part_of_effective_hosts():
    p = _plan(
        "source_owned_sitemap_fetch", host="www.suva.ch",
        params={"urls": ["https://www.suva.ch/sitemap/de-CH.xml"],
                "ok_hosts": ["www.suva.ch",
                             "suva-p-001.sitecorecontenthub.cloud"]})
    assert p.effective_hosts() == [
        "www.suva.ch", "suva-p-001.sitecorecontenthub.cloud"]


@pytest.mark.parametrize("route, expected_direct", [
    ("source_owned_incremental_feed", True),
    ("source_owned_bulk_dump", True),
    ("documented_source_api", True),
    ("common_crawl", False),
    ("source_owned_sitemap_fetch", True),
    ("direct_origin_crawl", True),
])
def test_consent_mode_follows_acquisition_provenance(
        monkeypatch, tmp_path, route, expected_direct):
    calls = []

    def resolve(host, *, direct_crawl, **_kw):
        calls.append((host, direct_crawl))
        return {"host": host, "mode": "direct" if direct_crawl else "cc",
                "bucket": "ok", "status": "ok", "search_ok": True,
                "train_ok": True, "checked_at": "x", "reasons": []}

    monkeypatch.setattr(SR, "resolve_consent", resolve)
    monkeypatch.setattr(SR, "run_via_pack_fetch", lambda *_a, **_kw: {})
    monkeypatch.setattr(SR, "run_via_extractor", lambda *_a, **_kw: {})

    class Args:
        out_dir = str(tmp_path)
        limit = 0
        budget = 10
        no_l0 = True
        skip_consent = False
        dry_run = True
        extractor_args = None

    p = _plan(route, host="origin.test", extractor="cytir/sources/families/fedlex.py")
    SR.run_pack_plans({"id": "p", "_path": "", "adapter": "dump"},
                      [p], Args())
    assert calls == [("origin.test", expected_direct)]


def test_same_host_in_cc_and_direct_lanes_gets_two_receipts(
        monkeypatch, tmp_path):
    calls = []

    def resolve(host, *, direct_crawl, **_kw):
        calls.append((host, direct_crawl))
        return {"host": host, "mode": "direct" if direct_crawl else "cc",
                "bucket": "ok", "status": "ok", "search_ok": True,
                "train_ok": True, "checked_at": "x", "reasons": []}

    monkeypatch.setattr(SR, "resolve_consent", resolve)
    monkeypatch.setattr(SR, "run_via_extractor", lambda *_a, **_kw: {})

    class Args:
        out_dir = str(tmp_path)
        limit = 0
        budget = 10
        no_l0 = True
        skip_consent = False
        dry_run = True
        extractor_args = None

    ps = [
        _plan("common_crawl", route_id="cc", host="shared.test",
              extractor="cytir/sources/families/fedlex.py"),
        _plan("documented_source_api", route_id="api", host="shared.test",
              extractor="cytir/sources/families/fedlex.py"),
    ]
    receipt = SR.run_pack_plans(
        {"id": "p", "_path": "", "adapter": "api"}, ps, Args())
    assert calls == [("shared.test", False), ("shared.test", True)]
    assert [r["mode"] for r in receipt["consent"]] == ["cc", "direct"]


def test_declined_secondary_origin_holds_the_whole_plan(monkeypatch, tmp_path):
    def resolve(host, *, direct_crawl, **_kw):
        allowed = host == "primary.test"
        return {"host": host, "mode": "direct", "bucket": "ok" if allowed else "declined",
                "status": "ok", "search_ok": allowed, "train_ok": allowed,
                "checked_at": "x", "reasons": []}

    monkeypatch.setattr(SR, "resolve_consent", resolve)

    class Args:
        out_dir = str(tmp_path)
        limit = 0
        budget = 10
        no_l0 = True
        skip_consent = False
        dry_run = True
        extractor_args = None

    p = _plan("source_owned_sitemap_fetch", host="primary.test",
              params={"urls": ["https://primary.test/sitemap.xml"],
                      "ok_hosts": ["secondary.test"]})
    receipt = SR.run_pack_plans(
        {"id": "p", "_path": "", "adapter": "sitemap"}, [p], Args())
    assert receipt["plans"][0]["skipped"] is True
    assert receipt["plans"][0]["hosts"] == ["primary.test", "secondary.test"]
    assert receipt["results"] == {"ran": 0}


# --- the migrated packs ------------------------------------------------------

def test_fedlex_pack_compiles_to_one_dump_plan():
    plans = PL.plans(PL.load_pack(repo_file_or_skip("sources/ch/fedlex.yaml")))
    assert len(plans) == 1
    assert plans[0].route == "source_owned_bulk_dump"
    assert plans[0].extractor == "cytir/sources/families/fedlex.py"
    assert plans[0].effective_hosts() == ["fedlex.data.admin.ch"]


def test_education_pack_compiles_only_staged_routes():
    """Four of six routes are deferred; a deferred route is a decision, not work."""
    doc = PL.load_pack(REPO / "sources/themes/education/education-research.yaml")
    plans = PL.plans(doc)
    assert {p.route_id for p in plans} == {"ies.ed.gov", "www.iiep.unesco.org"}
    assert all(p.route == "source_owned_sitemap_fetch" for p in plans)


def test_education_predicates_match_the_original_regexes():
    """Guards the migration: these are IES_ALLOW / IIEP_ALLOW behaviour."""
    doc = PL.load_pack(REPO / "sources/themes/education/education-research.yaml")
    by = {p.route_id: p for p in PL.plans(doc)}
    ies = by["ies.ed.gov"].predicate()
    assert ies("https://ies.ed.gov/pubs/2024/report")
    assert not ies("https://ies.ed.gov/pubs")             # _deep: landing page
    assert not ies("https://ies.ed.gov/pubs/search")      # _clean: apparatus
    assert not ies("https://ies.ed.gov/about/team")       # not in IES_ALLOW

    iiep = by["www.iiep.unesco.org"]
    assert iiep.predicate()("https://www.iiep.unesco.org/en/publication/x")
    assert not iiep.predicate()("https://www.iiep.unesco.org/es/publication/x")
    assert iiep.lang_fn()("https://www.iiep.unesco.org/fr/publication/x") == "fr"
    assert iiep.lang_fn()("https://www.iiep.unesco.org/en/publication/x") == "en"


# --- doc rot -----------------------------------------------------------------

def test_operator_doc_example_pack_is_valid(tmp_path):
    """The example in ADDING-A-SOURCE.md must actually validate and compile.

    A doc whose copy-paste example is rejected is worse than no doc.
    """
    doc = (REPO / "ADDING-A-SOURCE.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```yaml\n(.*?)```", doc, re.S)
    assert blocks, "no yaml example found in ADDING-A-SOURCE.md"
    example = max(blocks, key=len)
    path = tmp_path / "example.yaml"
    path.write_text(example)
    loaded = PL.load_pack(path)
    plans = PL.plans(loaded)
    assert len(plans) == 1
    assert plans[0].route == "source_owned_sitemap_fetch"
    pred = plans[0].predicate()
    assert pred("https://www.example.ch/publications/2026/report")
    assert not pred("https://www.example.ch/publications/search")


def test_operator_doc_names_only_real_routes():
    """Every route named in the doc's table must exist in policy."""
    doc = (REPO / "ADDING-A-SOURCE.md").read_text(encoding="utf-8")
    policy = json.loads((REPO / "data" / "acquisition-policy.v1.json").read_text())
    routes = {r["route"] for r in policy["route_precedence"]}
    named = set(re.findall(r"`(source_owned_\w+|documented_source_api|common_crawl|direct_origin_crawl)`", doc))
    assert named <= routes, f"doc names non-existent routes: {sorted(named - routes)}"


# --- the two incoming lanes' real shapes (2026-08-20) ------------------------
# These are not hypotheticals. They are the acquisition shapes the UK/US/IT
# government lane and the company-register lane reported, encoded as tests so
# the contract cannot regress out from under them before they arrive.

def _minimal(**over):
    d = dict(
        id="lane-test", name="T", country="IT", layer="law-government",
        adapter="dump", acquisition_policy="aw-acquisition-v1",
        outlink_policy="capture-and-receipt",
        preservation_profiles=["anchored_law"], provenance="authored",
        trust_tier=1.0, freshness="daily",
        license={"default": "CC-BY-4.0"},
        consent={"basis": "official-no-copyright"}, status="planned",
        admission={"channel": "official-dump", "evidence": ["lane report"],
                   "decision": "auto-admit", "policy": "POLICY.md@v1 §4.1",
                   "date": "2026-08-20"})
    d.update(over)
    return d


def _write(tmp_path, doc):
    p = tmp_path / "pack.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


def test_bulk_dump_over_git_transport(tmp_path):
    """IT gov lane: the Italian Senate ships 1.71 GB of Akoma Ntoso via git clone.

    A dump by every meaningful definition whose transport happens to be git.
    """
    doc = _minimal(acquire={
        "route": "source_owned_bulk_dump",
        "params": {"url": "https://github.com/SenatoDellaRepubblica/AkomaNtoso",
                   "format": "xml", "transport": "git", "ref": "main",
                   "subpath": "atti"}})
    plan = PL.plans(PL.load_pack(_write(tmp_path, doc)))[0]
    assert plan.params["transport"] == "git"
    assert plan.effective_hosts() == ["github.com"]


def test_api_dialects_the_lanes_need(tmp_path):
    """Normattiva = OpenAPI; ISTAT = SDMX REST."""
    for dialect, endpoint in (("openapi", "https://www.normattiva.it/api"),
                              ("sdmx", "https://esploradati.istat.it/SDMXWS/rest")):
        doc = _minimal(adapter="api", acquire={
            "route": "documented_source_api",
            "params": {"endpoint": endpoint, "dialect": dialect,
                       "paginate": {"style": "page"}}})
        plan = PL.plans(PL.load_pack(_write(tmp_path, doc)))[0]
        assert plan.params["dialect"] == dialect


def test_date_enumerated_url_pattern_without_seeds(tmp_path):
    """IT gov lane: Gazzetta Ufficiale is date-based path enumeration.

    Closer to a crawl than a sitemap, and expressible only as a pattern. The
    `urllist` mode in ingest_cov_docs.py already did exactly this for NORMLEX.
    """
    doc = _minimal(adapter="dump", acquire={
        "route": "direct_origin_crawl",
        "params": {"url_pattern": "https://www.gazzettaufficiale.it/eli/gu/{date}/sg",
                   "date_from": "2020-01-01", "date_to": "2026-08-20",
                   "date_format": "%Y/%m/%d"},
        "reason_receipt": {
            "higher_priority_routes_checked": ["source_owned_bulk_dump",
                                               "documented_source_api"],
            "reason_selected_route_is_needed": "no dump or API; ELI paths are enumerable by date"}})
    plan = PL.plans(PL.load_pack(_write(tmp_path, doc)))[0]
    assert plan.route == "direct_origin_crawl"
    assert "seeds" not in plan.params


def test_periodic_single_file_snapshots(tmp_path):
    """Register lane: SIRENE monthly parquet, Canada daily CSV — the simplest shape."""
    for fmt, url in (("parquet", "https://files.data.gouv.fr/insee-sirene/{yyyymm}.parquet"),
                     ("csv", "https://ised-isde.canada.ca/cc/open-data/{yyyymmdd}.csv")):
        doc = _minimal(country="FR", layer="commercial", adapter="registry",
                       preservation_profiles=["entity_registry"],
                       license={"default": "open-licence-2.0"},
                       consent={"basis": "live-resolve-open"},
                       acquire={"route": "source_owned_bulk_dump",
                                "params": {"url_template": url, "format": fmt}})
        plan = PL.plans(PL.load_pack(_write(tmp_path, doc)))[0]
        assert plan.params["format"] == fmt


# --- no_fallback: a commitment the tool can actually check -------------------

def test_no_fallback_pack_rejects_a_crawl_route_at_validation(tmp_path):
    """Register lane HARD requirement: the Canadian pack commits to never
    crawling the HTML detail pages. That must be enforceable, not conventional."""
    doc = _minimal(country="CA", layer="commercial", adapter="registry",
                   preservation_profiles=["entity_registry"],
                   consent={"basis": "live-resolve-open"},
                   acquire={"route": "direct_origin_crawl", "no_fallback": True,
                            "params": {"seeds": ["https://ised-isde.canada.ca/"]}})
    with pytest.raises(PL.PackError, match="no_fallback"):
        PL.load_pack(_write(tmp_path, doc))


def test_no_fallback_allows_the_bulk_route(tmp_path):
    doc = _minimal(country="CA", layer="commercial", adapter="registry",
                   preservation_profiles=["entity_registry"],
                   consent={"basis": "live-resolve-open"},
                   acquire={"route": "source_owned_bulk_dump", "no_fallback": True,
                            "params": {"url": "https://ised-isde.canada.ca/open-data.csv",
                                       "format": "csv"}})
    plan = PL.plans(PL.load_pack(_write(tmp_path, doc)))[0]
    assert plan.no_fallback is True


def test_no_fallback_is_enforced_at_run_time_too(tmp_path):
    """Second, independent check: a hand-built plan must not bypass the promise.
    A commitment enforced in only one place is a commitment with a bypass."""
    class Args:
        out_dir = str(tmp_path)
        limit = 0
        budget = 10
        no_l0 = True
        skip_consent = True
        dry_run = True
        extractor_args = None

    plans = [_plan("direct_origin_crawl", host="x.ca", no_fallback=True,
                   params={"seeds": ["https://x.ca/"]})]
    with pytest.raises(SR.RunError, match="no_fallback"):
        SR.run_pack_plans({"id": "p", "_path": "", "adapter": "registry"},
                          plans, Args())


def test_no_fallback_inherits_from_pack_level_to_routes(tmp_path):
    """Declared once at pack level, it must bind every route — otherwise the
    commitment is only as good as the reader's diligence."""
    doc = _minimal(country="CA", layer="commercial", adapter="registry",
                   preservation_profiles=["entity_registry"],
                   consent={"basis": "live-resolve-open"},
                   acquire={"route": "source_owned_bulk_dump", "no_fallback": True,
                            "params": {"url": "https://x.ca/d.csv", "format": "csv"}},
                   routes=[{"host": "x.ca", "status": "planned"}])
    plan = PL.plans(PL.load_pack(_write(tmp_path, doc)))[0]
    assert plan.no_fallback is True


def test_ch_media_is_one_pack_with_two_shapes():
    """ADR-097's headline claim, under test.

    ch-media reaches srf.ch and rsi.ch through Common Crawl and rts.ch through
    its sitemap. If a future change ever collapses acquisition shape back to a
    per-pack field, this fails — which is the point.
    """
    doc = PL.load_pack(repo_file_or_skip("sources/ch/ch-media.yaml"))
    got = {p.route_id: p.route for p in PL.plans(doc)}
    assert got == {"srf.ch": "common_crawl",
                   "rsi.ch": "common_crawl",
                   "rts.ch": "source_owned_sitemap_fetch"}
    # and the path scope that used to be prose is now addressable data
    srf = next(p for p in PL.plans(doc) if p.route_id == "srf.ch")
    assert srf.params["path_prefixes"] == ["/news", "/sport", "/kultur",
                                           "/wissen", "/meteo"]


def test_cc_route_without_extractor_fails_loud_not_silently():
    """ch-media declares no extractor, so its CC routes cannot run yet.
    That must be an error naming the gap, not a quiet no-op."""
    class Args:
        extractor_args = None
        dry_run = False
    doc = PL.load_pack(repo_file_or_skip("sources/ch/ch-media.yaml"))
    cc = next(p for p in PL.plans(doc) if p.route == "common_crawl")
    with pytest.raises(SR.RunError, match="needs an"):
        SR.run_via_extractor(cc, args=Args())
