"""The four leading-authority packs of task 399 on the `SourceFamily` Protocol (task 1393, S7).

Biomedical (`biomed_authority_core`), education research (`education_research`), energy and
climate (`energy_climate_intl`) and urban governance (`urban_governance`): one contract row per
pack, several hosts per row. The legacy driver (`cytir.acquire.authority_packs`, the old
`ingest_authority_packs.py`) kept one hand-written HOSTS dict per pack and ran it through
`cytir.acquire.crawl.run_pack`. This module is the Protocol side of the same four packs.

DELEGATED, not copied. A sitemap host of a pack is exactly the dialect of `SitemapFamily`
(`cytir.sources.families.sitemap`): read the pack route's sitemaps, keep what its compiled `select`
keeps, take each page politely, hand the bytes to the shared web tail. `AuthorityPackFamily`
subclasses it and adds the one thing the single-host base lacks: it walks EVERY sitemap route of
its pack (`SitemapFamily.discover_host`, once per route) and each unit carries its host, so the
trust, tier and licence slug of the right route reach the row (`plan_for`). What a host selects,
its language, cap, delay, trust, tier and licence slug come from the pack YAML under `sources/`;
the characterization test `test_authority_packs_selection` proves them equal to the legacy HOSTS dicts.

CONSENT is live per origin (ruling 23.10): every unit names its host in `consent_hosts`, so the
framework resolves robots, TDMRep and AIPREF for that host at stage time. `train_ok` therefore
differs per host (iiep, citiesalliance) with no pack-wide flag. The two flags are read from the
row; nothing here writes them.

LICENCE: the row's `licence` is the contract's. Biomed declares `instrument`; education, energy
and urban declare `unknown` (task 737: the packs stamp one licence SLUG per host, not an instrument
read from a terms page). The slug on a record is the pack route's `license`, unchanged. This module
does not infer a basis from it.

RUNNER ROUTES: the energy pack's `www.ipcc.ch` route is a `direct_origin_crawl` (a BFS from report
hubs, no sitemap exists). The BFS engine is `cytir.sources.runner` -> `pack_fetch.run_pack(mode="bfs")`
and the family declares the route in `RUNNER_ROUTES` instead of copying it. `discover` lists it in
the run notes (`runner_routes`) and does not fetch it; a route that is neither sitemap nor declared
there is refused.
"""
from __future__ import annotations

from cytir.sources import pack_loader as PL
from cytir.sources.families.sitemap import SITEMAP_ROUTE, SitemapFamily
from cytir.sources.family import RunContext, Unit
from cytir.sources.registry import register_family


class AuthorityPackFamily(SitemapFamily):
    """Base of the four packs: every sitemap route of `pack_id`, one family."""

    host = ""          # several hosts: each unit carries its own (`meta["host"]`)
    #: pack routes this family does NOT walk in `discover`: the source runner runs them with its one
    #: engine (`cytir.sources.runner.PACK_FETCH_MODES`). Each must resolve there (test_aw_family_authority_packs).
    RUNNER_ROUTES: tuple[str, ...] = ()

    def routes(self) -> list[PL.AcquisitionPlan]:
        cached = self.__dict__.get("_routes")
        if cached is None:
            doc = PL.load_pack(PL.find_pack(self.pack_id))
            cached = self._routes = PL.plans(doc)
        return cached

    def sitemap_routes(self) -> list[PL.AcquisitionPlan]:
        mine = [p for p in self.routes() if p.route == SITEMAP_ROUTE]
        if not mine:
            raise LookupError(f"{self.family_id}: pack {self.pack_id!r} has no {SITEMAP_ROUTE} route")
        return mine

    def plan(self):
        raise LookupError(f"{self.family_id}: a pack family has several routes; use plan_for(unit)")

    def plan_for(self, unit: Unit) -> PL.AcquisitionPlan:
        host = unit.meta["host"]
        for p in self.sitemap_routes():
            if p.host == host:
                return p
        raise LookupError(f"{self.family_id}: no {SITEMAP_ROUTE} route for {host!r} in {self.pack_id!r}")

    def discover(self, ctx: RunContext):
        units: list[Unit] = []
        per_host: dict[str, dict] = {}
        for plan in self.sitemap_routes():
            got, notes = self.discover_host(ctx, plan, plan.host)
            units.extend(got)
            per_host[plan.host] = notes
        other = {p.host: p.route for p in self.routes() if p.route != SITEMAP_ROUTE}
        undeclared = {h: r for h, r in other.items() if r not in self.RUNNER_ROUTES}
        if undeclared:
            raise LookupError(f"{self.family_id}: routes {undeclared} are neither sitemap nor in RUNNER_ROUTES")
        ctx.notes.update({"pack": self.pack_id, "hosts": per_host, "units": len(units),
                          "runner_routes": other})
        return units


@register_family
class BiomedAuthorityCoreFamily(AuthorityPackFamily):
    """Health authority core (POLICY 4.6): cochrane, who, ecdc, medlineplus, spf, rki."""

    family_id = "biomed_authority_core"
    pack_id = "biomed-authority-core"


@register_family
class EducationResearchFamily(AuthorityPackFamily):
    """Education research: ies.ed.gov, www.iiep.unesco.org."""

    family_id = "education_research"
    pack_id = "education-research"


@register_family
class EnergyClimateIntlFamily(AuthorityPackFamily):
    """Energy and climate, international: iea, energy.ec.europa.eu, uba, eia (+ ipcc, PARTIAL)."""

    family_id = "energy_climate_intl"
    pack_id = "energy-climate-intl"
    #: the ipcc.ch route is a BFS (`direct_origin_crawl`): run by the source runner's
    #: `pack_fetch.run_pack(mode="bfs")`, the one BFS engine; `discover` reports it in `runner_routes`
    RUNNER_ROUTES = ("direct_origin_crawl",)


@register_family
class UrbanGovernanceFamily(AuthorityPackFamily):
    """Urban governance: www.citiesalliance.org, iclei.org."""

    family_id = "urban_governance"
    pack_id = "urban-governance"
