"""The four authority packs on the family Protocol (task 1393). No network.

Pins: the four families are registered and conformant; consent and licence are read from each
contract row (biomed declares `instrument`; the other three stay `unknown`, never inferred); a
dry-run `discover` over a synthetic sitemap per host (served by an in-memory transport) emits
exactly the units each host's pack route selects, each carrying its own host for the live consent
gate; every record's trust, tier and licence slug come from the unit's own route; the energy
pack's BFS route is declared PARTIAL and reported, not fetched.
"""
from __future__ import annotations

import pathlib

import httpx
import pytest
import yaml

from cytir.sources import contract as contract_mod
from cytir.sources import registry
from cytir.sources.families.authority_packs import (BiomedAuthorityCoreFamily, EducationResearchFamily,
                                                 EnergyClimateIntlFamily, UrbanGovernanceFamily)
from cytir.sources.family import RunContext, SourceFamily

FAMILIES = {"biomed_authority_core": BiomedAuthorityCoreFamily,
            "education_research": EducationResearchFamily,
            "energy_climate_intl": EnergyClimateIntlFamily,
            "urban_governance": UrbanGovernanceFamily}
BASIS = {"biomed_authority_core": "instrument", "education_research": "unknown",
         "energy_climate_intl": "unknown", "urban_governance": "unknown"}
PATHS = ["reports/a/b", "analysis/x/y", "evidence/CD1_x", "publications/m/a", "docs/article/x",
         "en/publikationen/a", "ncee/a/b", "en/publication/x", "some-slug", "search/q/z",
         "a/b/c", "print/a/b"]

# The public cut ships no `sources/*.yaml`, so the tests build a small synthetic pack per family
# (same `id`, own hosts, the fields `discover` reads) and point the loader at it. The assertion
# stays real: the pack's own route predicates decide what is selected from each host's sitemap.
PACKS = {"biomed-authority-core": ["www.cochrane.org", "www.who.int"],
         "education-research": ["ies.ed.gov", "www.iiep.unesco.org"],
         "energy-climate-intl": ["www.iea.org", "energy.ec.europa.eu"],
         "urban-governance": ["www.citiesalliance.org", "iclei.org"]}
SELECT = {"path_regex": "(?i)^/(reports|analysis|evidence|publications|docs|en|ncee)(/|$)",
          "min_path_segments": 2}


def _real_pack(pack_id):
    hits = sorted((pathlib.Path(__file__).resolve().parents[1] / "sources" / "themes").glob(f"*/{pack_id}.yaml"))
    assert len(hits) == 1, pack_id
    return hits[0]


def _pack_doc(pack_id, hosts):
    def route(host, extra=None):
        acq = {"route": "source_owned_sitemap_fetch", "params": {"urls": [f"https://{host}/sitemap.xml"]},
               "select": SELECT, "lang": {"const": "en"}, "delay_s": 0, "cap": 3}
        return {"host": host, "lang": ["en"], "license": f"{host}-terms", "status": "staged",
                "acquire": acq, **(extra or {})}
    routes = [route(h) for h in hosts]
    if pack_id == "energy-climate-intl":      # the BFS route the adapter reports and never fetches
        routes.insert(0, {"host": "www.ipcc.ch", "lang": ["en"], "license": "ipcc-terms",
                          "status": "staged", "acquire": {"route": "direct_origin_crawl",
                                                          "params": {"seeds": ["https://www.ipcc.ch/reports/"], "hub_path_regex": "^/report"}}})
    doc = yaml.safe_load(_real_pack(pack_id).read_text(encoding="utf-8"))
    doc["routes"] = routes      # the shipped pack's schema-valid header, synthetic routes
    doc.pop("deferred", None)
    return doc



@pytest.fixture(autouse=True)
def _synthetic_packs(tmp_path, monkeypatch):
    root = tmp_path / "sources"
    root.mkdir()
    for pid, hosts in PACKS.items():
        (root / f"{pid}.yaml").write_text(yaml.safe_dump(_pack_doc(pid, hosts)), encoding="utf-8")
    ext = tmp_path / "cytir" / "sources" / "families" / "authority_packs.py"   # a pack's extractor pointer must resolve
    ext.parent.mkdir(parents=True)
    ext.write_text("", encoding="utf-8")
    monkeypatch.setenv("AW_SOURCES_DIR", str(root))


def _row(fid):
    return contract_mod.by_id(contract_mod.load())[fid]


def _sitemaps(family):
    """sitemap URL -> body; every route's sitemap is answered, and a foreign host's URL is
    listed in each (it must be counted, never taken)."""
    body = {}
    for plan in family.sitemap_routes():
        locs = [f"https://{plan.host}/{p}" for p in PATHS] + ["https://other.example/x/y/z"]
        xml = "<urlset>" + "".join(f"<loc>{u}</loc>" for u in locs) + "</urlset>"
        for sm in plan.params["urls"]:
            body[sm] = xml
    return body


def _ctx(family, tmp_path, fid):
    bodies = _sitemaps(family)

    def handler(request):
        text = bodies.get(str(request.url))
        return httpx.Response(200 if text else 404, content=(text or "").encode())
    return RunContext(out_dir=tmp_path / "out", today="2026-10-30", skip_consent=True,
                      contract=_row(fid), log=lambda m: None,
                      http=httpx.Client(transport=httpx.MockTransport(handler)))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)


@pytest.mark.parametrize("fid", sorted(FAMILIES))
def test_registered_and_conformant(fid):
    assert fid in registry.registered()
    adapter, row, spec = registry.admit(fid)
    assert isinstance(adapter, SourceFamily) and row["family_id"] == fid
    assert spec.declared == row["chunk_policy"]["policy_id"]
    assert getattr(adapter, "PARTIAL", ()) == ()


@pytest.mark.parametrize("fid", sorted(FAMILIES))
def test_consent_and_licence_are_the_rows(fid):
    row = _row(fid)
    assert row["licence"]["basis"] == BASIS[fid]
    assert row["consent"]["search_ok"] is None        # live per origin, never frozen in the row
    assert row["licence"]["inferred_from_hosting"] is False


@pytest.mark.parametrize("fid", sorted(FAMILIES))
def test_discover_dry_run_selects_each_hosts_own_urls(fid, tmp_path):
    fam = FAMILIES[fid]()
    ctx = _ctx(fam, tmp_path, fid)
    units = list(fam.discover(ctx))
    assert units, "an empty walk is a wrong pack, not an empty source"
    for plan in fam.sitemap_routes():
        urls = [f"https://{plan.host}/{p}" for p in PATHS]
        want = [u for u in urls if plan.predicate()(u)][:int(plan.cap)]
        got = [u for u in units if u.meta["host"] == plan.host]
        assert [u.url for u in got] == want
        for u in got:
            assert u.consent_hosts == (plan.host,)
            assert u.meta["lang"] == (plan.lang_fn()(u.url) or "")
            common = fam.passage_common(ctx, u)
            assert common["trust_tier"] == plan.effective_trust()
            assert common["license"] == plan.effective_license()
            assert common["tier"] == (plan.tier or "A") and common["source_type"] == "official"
    assert not any("other.example" in u.url for u in units)


def test_energy_declared_runner_route_resolves_in_the_runner():
    from cytir.sources import runner
    fam = EnergyClimateIntlFamily()
    assert fam.RUNNER_ROUTES == ("direct_origin_crawl",)
    assert all(r in runner.PACK_FETCH_MODES for r in fam.RUNNER_ROUTES)
    assert runner.PACK_FETCH_MODES["direct_origin_crawl"] == "bfs"


def test_energy_reports_the_bfs_route_it_does_not_fetch(tmp_path):
    fam = EnergyClimateIntlFamily()
    ctx = _ctx(fam, tmp_path, "energy_climate_intl")
    units = list(fam.discover(ctx))
    assert ctx.notes["runner_routes"] == {"www.ipcc.ch": "direct_origin_crawl"}
    assert not any("ipcc.ch" in u.url for u in units)
