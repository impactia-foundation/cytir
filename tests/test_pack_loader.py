#!/usr/bin/env python3
"""Tests for cytir/sources/pack_loader.py — the pack -> plan binding (task 434)."""
from __future__ import annotations

import copy
import json
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cytir.sources.pack_loader as PL  # noqa: E402


# --- the closed predicate vocabulary ----------------------------------------

def test_select_empty_accepts_everything():
    pred = PL.compile_select(None)
    assert pred("https://x.ch/anything")


def test_select_path_prefix():
    pred = PL.compile_select({"path_prefix": ["/news", "/sport"]})
    assert pred("https://srf.ch/news/a")
    assert pred("https://srf.ch/sport/b")
    assert not pred("https://srf.ch/kultur/c")


def test_select_deny_beats_allow():
    """Deny is checked first: an explicit exclusion must win."""
    pred = PL.compile_select({"path_regex": r"^/blog/",
                              "path_deny_regex": r"/(404|search)(/|$)"})
    assert pred("https://x.ch/blog/2026/post")
    assert not pred("https://x.ch/blog/search")


def test_select_segment1_in():
    pred = PL.compile_select({"segment1_in": ["data", "statistics"]})
    assert pred("https://x.ch/data/tables")
    assert not pred("https://x.ch/about/team")


def test_select_root_path_does_not_crash():
    for sel in ({"path_prefix": ["/a"]}, {"segment1_in": ["a"]},
                {"path_regex": "^/a"}):
        assert PL.compile_select(sel)("https://x.ch") is False


def test_lang_const_and_by_segment():
    assert PL.compile_lang({"const": "de"})("https://x.ch/fr/a") == "de"
    fn = PL.compile_lang({"by_path_segment": {"de": "de", "fr": "fr"},
                          "default": "de"})
    assert fn("https://x.ch/fr/a") == "fr"
    assert fn("https://x.ch/zz/a") == "de"


# --- pack compilation --------------------------------------------------------

BASE = yaml.safe_load(textwrap.dedent("""
    id: test-pack
    name: Test
    country: CH
    layer: news
    adapter: cc
    acquisition_policy: aw-acquisition-v1
    outlink_policy: capture-and-receipt
    preservation_profiles: [news_article]
    provenance: authored
    trust_tier: 0.8
    freshness: daily
    license: {default: public-domain}
    consent: {basis: live-resolve-open}
    status: staged
    admission:
      channel: official-dump
      evidence: ["task 434"]
      decision: auto-admit
      policy: "POLICY.md@v1 §4.7"
      date: "2026-08-20"
"""))


def _pack(**over):
    d = copy.deepcopy(BASE)
    d.update(over)
    return d


def test_pack_without_acquire_yields_no_plans():
    """The honest pre-migration state: a pack is documentation until it declares acquire."""
    assert PL.plans(_pack()) == []


def test_route_level_acquire_overrides_pack_level():
    """THE core rule (task 434): shape belongs to the route, not the pack.

    ch-media is the real case -- srf/rsi by common_crawl, rts by sitemap.
    """
    doc = _pack(
        acquire={"route": "common_crawl", "params": {"path_prefixes": ["/news"]}},
        routes=[
            {"host": "srf.ch", "status": "staged"},
            {"host": "rts.ch", "status": "staged",
             "acquire": {"route": "source_owned_sitemap_fetch",
                         "params": {"urls": ["https://rts.ch/sitemap.xml"]}}},
        ])
    ps = {p.route_id: p for p in PL.plans(doc)}
    assert ps["srf.ch"].route == "common_crawl"          # inherits pack default
    assert ps["rts.ch"].route == "source_owned_sitemap_fetch"  # overrides it
    assert ps["rts.ch"].params["urls"] == ["https://rts.ch/sitemap.xml"]


def test_non_runnable_routes_are_skipped():
    """A declined/not-viable route is a decision on record, not a work item."""
    doc = _pack(
        acquire={"route": "common_crawl", "params": {}},
        routes=[{"host": "a.ch", "status": "staged"},
                {"host": "b.ch", "status": "declined"},
                {"host": "c.ch", "status": "not-viable"},
                {"host": "d.ch", "status": "deferred"}])
    assert [p.route_id for p in PL.plans(doc)] == ["a.ch"]


def test_governance_fields_carry_into_every_plan():
    doc = _pack(acquire={"route": "common_crawl", "params": {}},
                routes=[{"host": "a.ch", "status": "staged"}])
    p = PL.plans(doc)[0]
    assert p.trust_tier == 0.8
    assert p.consent_basis == "live-resolve-open"
    assert p.preservation_profiles == ["news_article"]
    assert p.license["default"] == "public-domain"


def test_to_host_cfg_projects_into_pack_fetch_shape():
    """The bridge: YAML -> the HOSTS dict run_pack already accepts."""
    doc = _pack(adapter="sitemap",
                routes=[{"host": "www.eeoc.gov", "status": "staged",
                         "acquire": {"route": "source_owned_sitemap_fetch",
                                     "params": {"urls": ["https://www.eeoc.gov/sitemap.xml"]},
                                     "select": {"path_prefix": ["/laws"]},
                                     "lang": {"const": "en"},
                                     "delay_s": 2.0, "cap": 3600}}])
    cfg = PL.plans(doc)[0].to_host_cfg("sitemap")
    assert cfg["host"] == "www.eeoc.gov"
    assert cfg["sitemaps"] == ["https://www.eeoc.gov/sitemap.xml"]
    assert cfg["delay"] == 2.0 and cfg["cap"] == 3600
    assert cfg["trust"] == 0.8
    assert callable(cfg["include"]) and callable(cfg["lang"])
    assert cfg["include"]("https://www.eeoc.gov/laws/x")
    assert not cfg["include"]("https://www.eeoc.gov/about")
    assert cfg["lang"]("https://www.eeoc.gov/laws/x") == "en"


# --- validation is real, not decorative -------------------------------------

def test_bad_route_enum_is_rejected(tmp_path):
    doc = _pack(acquire={"route": "totally_made_up", "params": {}})
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(PL.PackError):
        PL.load_pack(p)


def test_missing_required_param_is_rejected(tmp_path):
    """sitemap route without `urls` cannot run; fail at load, not mid-crawl."""
    doc = _pack(acquire={"route": "source_owned_sitemap_fetch", "params": {}})
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(PL.PackError, match="requires"):
        PL.load_pack(p)


def test_unknown_param_key_is_rejected(tmp_path):
    """A typo'd param must fail loudly rather than be silently ignored."""
    doc = _pack(acquire={"route": "source_owned_sitemap_fetch",
                         "params": {"urls": ["https://x.ch/s.xml"], "delay": 2}})
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(PL.PackError, match="unknown key"):
        PL.load_pack(p)


def test_config_ref_requires_extractor_engine(tmp_path):
    doc = _pack(acquire={
        "route": "source_owned_sitemap_fetch",
        "params": {"urls": ["https://x.ch/s.xml"]},
        "config_ref": {"path": "data/cov351_routes.yaml",
                       "pack": "regdocs1", "source": "finma_docs2"}})
    path = tmp_path / "config-ref.yaml"
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(PL.PackError, match="engine=extractor"):
        PL.load_pack(path)


def test_config_ref_path_must_exist(tmp_path):
    doc = _pack(acquire={
        "route": "source_owned_sitemap_fetch", "engine": "extractor",
        "params": {"urls": ["https://x.ch/s.xml"]},
        "config_ref": {"path": "configs/not-real.yaml",
                       "pack": "p", "source": "s"}})
    path = tmp_path / "config-ref.yaml"
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(PL.PackError, match="does not resolve"):
        PL.load_pack(path)


def test_config_ref_path_cannot_escape_repository(tmp_path):
    doc = _pack(acquire={
        "route": "source_owned_sitemap_fetch", "engine": "extractor",
        "params": {"urls": ["https://x.ch/s.xml"]},
        "config_ref": {"path": "../../etc/passwd", "pack": "p", "source": "s"}})
    path = tmp_path / "config-ref.yaml"
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(PL.PackError, match="under the repository"):
        PL.load_pack(path)


def test_declared_rank_must_match_policy(tmp_path):
    """Redundant-by-design: a silent policy re-rank fails pack validation."""
    doc = _pack(acquire={"route": "common_crawl", "rank": 1, "params": {}})
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(PL.PackError, match="ranks"):
        PL.load_pack(p)


def test_direct_crawl_without_receipt_warns_but_does_not_fail(tmp_path, capsys):
    """Orchestrator ruling 2026-08-20: warn-only first. 16 scripts owe a receipt;
    a validator that fails everything on day one gets disabled."""
    doc = _pack(acquire={"route": "direct_origin_crawl",
                         "params": {"seeds": ["https://x.ch/"]}})
    p = tmp_path / "dc.yaml"
    p.write_text(yaml.safe_dump(doc))
    loaded = PL.load_pack(p)          # must NOT raise
    assert loaded["id"] == "test-pack"
    assert "reason_receipt" in capsys.readouterr().err


def test_all_real_packs_still_load():
    """The 16 shipped packs must survive the schema extension."""
    n = 0
    for path in sorted(PL.sources_root().rglob("*.yaml")):
        if "_schema" in path.parts:
            continue
        PL.load_pack(path)
        n += 1
    assert n >= 16


def test_schema_profile_enum_matches_registry():
    """Task 437: the pack schema's preservation_profiles enum must equal the
    registry, so the editor-facing copy cannot silently drift.

    Validation itself now loads the enum from the registry at runtime
    (doctrine 1 -- the registry is the one home). This test guards the schema
    file's hand-copied restatement that editor tooling reads: a profile added
    to data/source-preservation-profiles.v1.json without updating the schema
    fails here, loud, instead of going stale for weeks.
    """
    schema = json.loads(
        (PL.sources_root() / "_schema" / "pack.schema.json").read_text())
    registry = json.loads(
        (PL.REPO_ROOT / "data" / "source-preservation-profiles.v1.json")
        .read_text())
    schema_enum = schema["properties"]["preservation_profiles"]["items"]["enum"]
    assert sorted(schema_enum) == sorted(registry["profiles"])


# --- ADR-039: a record must never carry a placeholder licence ---------------

def test_placeholder_license_is_refused_not_stamped():
    """Regression. The first cut of the education migration would have stamped
    the literal string "per-route" as the licence on every record, because the
    pack declares `license.default: per-route` and keeps the real per-host
    mapping in prose `notes:`. It looked fine and was an ADR-039 violation.
    """
    doc = _pack(license={"default": "per-route"},
                routes=[{"host": "a.ch", "status": "staged",
                         "acquire": {"route": "source_owned_sitemap_fetch",
                                     "params": {"urls": ["https://a.ch/s.xml"]}}}])
    plan = PL.plans(doc)[0]
    with pytest.raises(PL.PackError, match="placeholder"):
        plan.effective_license()


def test_route_license_overrides_pack_default():
    doc = _pack(license={"default": "per-route"},
                routes=[{"host": "a.ch", "status": "staged",
                         "license": "us-gov-public-domain",
                         "acquire": {"route": "source_owned_sitemap_fetch",
                                     "params": {"urls": ["https://a.ch/s.xml"]}}}])
    plan = PL.plans(doc)[0]
    assert plan.effective_license() == "us-gov-public-domain"
    assert plan.to_host_cfg("sitemap")["license"] == "us-gov-public-domain"


def test_real_pack_default_license_is_used_when_route_is_silent():
    doc = _pack(license={"default": "public-domain"},
                routes=[{"host": "a.ch", "status": "staged",
                         "acquire": {"route": "source_owned_sitemap_fetch",
                                     "params": {"urls": ["https://a.ch/s.xml"]}}}])
    assert PL.plans(doc)[0].effective_license() == "public-domain"


def test_migrated_education_routes_carry_real_licences():
    doc = PL.load_pack(PL.sources_root() / "themes/education/education-research.yaml")
    got = {p.route_id: p.effective_license() for p in PL.plans(doc)}
    assert got == {"ies.ed.gov": "us-gov-public-domain",
                   "www.iiep.unesco.org": "unesco-cc-by-sa-igo"}


@pytest.mark.parametrize("pack, expected", [
    ("themes/energy-climate/energy-climate-intl.yaml",
     {"www.ipcc.ch": ("ipcc-reports-terms", 0.95),
      "www.iea.org": ("iea-terms", 0.90),
      "energy.ec.europa.eu": ("eu-reuse-2011-833", 0.90),
      "www.umweltbundesamt.de": ("uba-terms", 0.90),
      "www.eia.gov": ("us-gov-public-domain", 0.90)}),
    ("themes/health/biomed-authority-core.yaml",
     {"www.cochrane.org": ("cochrane-terms", 0.95),
      "www.who.int": ("who-web-terms", 0.95),
      "www.ecdc.europa.eu": ("eu-reuse-2011-833", 0.90),
      "medlineplus.gov": ("us-gov-public-domain", 0.90),
      "www.santepubliquefrance.fr": ("etalab-2.0", 0.90),
      "www.rki.de": ("rki-terms", 0.90)}),
])
def test_newly_migrated_routes_preserve_license_and_effective_trust(pack, expected):
    plans = PL.plans(PL.load_pack(PL.sources_root() / pack))
    got = {p.route_id: (p.effective_license(), p.effective_trust()) for p in plans}
    assert got == expected


def test_ipcc_bfs_preserves_content_and_hub_predicates():
    plans = PL.plans(PL.load_pack(
        PL.sources_root() / "themes/energy-climate/energy-climate-intl.yaml"))
    ipcc = next(p for p in plans if p.route_id == "www.ipcc.ch")
    cfg = ipcc.to_host_cfg("bfs")
    assert cfg["allow"]("https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-1/")
    assert not cfg["allow"]("https://www.ipcc.ch/reports/")
    assert cfg["hub"]("https://www.ipcc.ch/report/ar6/wg1/")
    assert not cfg["hub"]("https://www.ipcc.ch/about/")
    assert cfg["hub_depth"] == 4


# --- ADR-098 (PROPOSED): per-route trust ------------------------------------

def test_route_trust_overrides_pack_trust():
    doc = _pack(trust_tier=0.95,
                routes=[{"host": "ipcc.ch", "status": "staged",
                         "acquire": {"route": "source_owned_sitemap_fetch",
                                     "params": {"urls": ["https://ipcc.ch/s.xml"]}}},
                        {"host": "iea.org", "status": "staged", "trust_tier": 0.90,
                         "acquire": {"route": "source_owned_sitemap_fetch",
                                     "params": {"urls": ["https://iea.org/s.xml"]}}}])
    got = {p.route_id: p.effective_trust() for p in PL.plans(doc)}
    assert got == {"ipcc.ch": 0.95, "iea.org": 0.90}


def test_route_trust_reaches_the_host_cfg():
    """The override must survive into what the fetch layer actually stamps."""
    doc = _pack(trust_tier=0.95,
                routes=[{"host": "iea.org", "status": "staged", "trust_tier": 0.90,
                         "acquire": {"route": "source_owned_sitemap_fetch",
                                     "params": {"urls": ["https://iea.org/s.xml"]}}}])
    assert PL.plans(doc)[0].to_host_cfg("sitemap")["trust"] == 0.90


def test_pack_trust_still_applies_when_route_is_silent():
    doc = _pack(trust_tier=0.75,
                routes=[{"host": "a.ch", "status": "staged",
                         "acquire": {"route": "source_owned_sitemap_fetch",
                                     "params": {"urls": ["https://a.ch/s.xml"]}}}])
    assert PL.plans(doc)[0].effective_trust() == 0.75
