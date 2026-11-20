"""Sitemap families (task 683, S7): a site declared in a pack, acquired by its own sitemap.

`SitemapFamily` is the whole dialect of the `source_owned_sitemap_fetch` route (acquisition
rank 5): read the pack's sitemaps, keep the URLs the pack's compiled `select` keeps, take each
page politely, hand the bytes to the shared web tail (`cytir.sources.families.web`). What it reads
comes from the PACK (`sources/**/*.yaml`, compiled by `cytir.sources.pack_loader`): the sitemap
URLs, `select`, `lang`, `delay_s`, `cap`, trust, tier and licence. A new sitemap site is
therefore:

    1. a route in a pack YAML (already how every site is described);
    2. a row in `data/source-family-contracts.v1.json` (a POLICY PR);
    3. FIVE LINES below: a subclass naming the pack, the host and the contract `family_id`.

Exemplar: `scratch_elcom_docs_exact`, the ElCom (federal electricity regulator) HTML pages of
`sources/ch/regulator-docs.yaml`. Its declared name is a task-era one (`scratch_*`); the
proposed real name is in task 683's table. The PDF harvest of that pack's `engine: extractor`
route (`pdf_include`, `cytir/sources/cov/engine.py`, Docling) is NOT ported: a sitemap URL that is a PDF
is discovered, then EXCLUDED with `unsupported_kind:pdf`, so it is counted and never silently
dropped. The selection (which sitemap URLs are in scope, and their language) is proved equal to
`cytir.sources.cov.engine.enum_sitemap`'s in the family sitemap test.
"""
from __future__ import annotations

import re
import time
from urllib.parse import urlsplit

from cytir.sources import pack_loader as PL
from cytir.sources.families import pdf_replay
from cytir.sources.families.web import WebPageMixin
from cytir.sources.family import (CHUNK_DECLARED, HttpSpec, CHUNK_DISPATCHER, Captured, RunContext, Unit,
                               UnitExcluded)
from cytir.sources.registry import register_family

SITEMAP_ROUTE = "source_owned_sitemap_fetch"
PDF = re.compile(r"(?i)\.pdf(\?|$)")         # the same test `cytir.sources.cov.engine.enum_sitemap` uses


class DiscoveryError(RuntimeError):
    """A declared sitemap could not be read. An error is not an empty inventory."""


def _opt_list(value) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


class SitemapFamily(WebPageMixin):
    """Base of every sitemap family. Subclasses set `family_id`, `pack_id`, `host`."""

    route = SITEMAP_ROUTE
    direct_crawl = True
    chunk_policy_id = CHUNK_DISPATCHER
    http_spec = HttpSpec(timeout=60)   # task 1556: the runner builds and closes the client
    pack_id: str
    host: str
    #: True = this family also runs the PDF replay route (`cytir.sources.families.pdf_replay`): the
    #: same adapter reads a sealed L0 listing instead of the sitemap when `l0_listing` is given.
    replay_l0 = False

    def chunk_policy_for(self, row: dict) -> str:
        """`registry.admit` asks the adapter which policy it chunks under, given its contract row:
        the declared one when the row names a policy (`WebPageMixin.preflight` then checks that the
        dispatcher's selection is that policy's sha256), else the dispatcher's pinned selection
        (a HELD row behaves exactly as before)."""
        from cytir.sources import contract as contract_mod

        pid = (row.get("chunk_policy") or {}).get("policy_id")
        return CHUNK_DISPATCHER if pid in (None, contract_mod.UNKNOWN) else CHUNK_DECLARED

    # -- the pack's plan for this host ------------------------------------------------------
    def plan(self) -> PL.AcquisitionPlan:
        cached = self.__dict__.get("_plan")
        if cached is not None:
            return cached
        doc = PL.load_pack(PL.find_pack(self.pack_id))
        mine = [p for p in PL.plans(doc) if p.host == self.host and p.route == SITEMAP_ROUTE]
        if len(mine) != 1:
            raise LookupError(
                f"{self.family_id}: pack {self.pack_id!r} has {len(mine)} "
                f"{SITEMAP_ROUTE} route(s) for {self.host!r}; a sitemap family names exactly one")
        self._plan = mine[0]
        return mine[0]

    # -- SourceFamily -----------------------------------------------------------------------
    def preflight(self, ctx: RunContext) -> None:
        if pdf_replay.active(ctx):
            pdf_replay.preflight(self, ctx)
        super().preflight(ctx)

    def plan_for(self, unit: Unit) -> PL.AcquisitionPlan:
        """The pack route that produced `unit`. One host per family: always `self.plan()`; a
        multi-host family (`authority_packs`) overrides this to read the unit's host."""
        return self.plan()

    def discover(self, ctx: RunContext):
        if pdf_replay.active(ctx):
            return pdf_replay.discover(self, ctx)
        units, notes = self.discover_host(ctx, self.plan(), self.host)
        ctx.notes.update(notes)
        return units

    def discover_host(self, ctx: RunContext, plan: PL.AcquisitionPlan, host: str):
        """One pack route's sitemaps -> (units, notes). The whole discovery dialect of a
        sitemap route; a family of several hosts calls it once per route."""
        from cytir.acquire import crawl

        client = ctx.http
        delay = float(plan.delay_s if plan.delay_s is not None else 1.0)
        inventory: list[str] = []
        for sm in plan.params["urls"]:
            try:
                urls = crawl.sitemap_urls(client, sm, delay=delay)
            except crawl.SitemapError as exc:      # an error is not an empty inventory
                raise DiscoveryError(f"{self.family_id}: {exc}") from exc
            if not urls:
                raise DiscoveryError(
                    f"{self.family_id}: sitemap {sm} lists no URLs (a read error raises before "
                    "this; an empty sitemap is not a clean inventory either)")
            inventory.extend(urls)
        # the same canonical form `pack_fetch`/`cytir.sources.cov.engine` key their inventories on:
        # https, host lowered and aliased, query dropped; other hosts are not this route's
        canon = crawl.make_canon()
        seen: dict[str, None] = {}
        foreign = 0
        for u in inventory:
            cu = canon(u)
            if not cu:
                continue
            if urlsplit(cu).netloc != host:
                foreign += 1
                continue
            seen.setdefault(cu)
        inventory_canon = list(seen)
        select, lang = plan.predicate(), plan.lang_fn()
        selected = [u for u in inventory_canon if select(u)]
        only = set(_opt_list(ctx.options.get("url")))
        if only:
            selected = [u for u in selected if u in only]
        cap = int(plan.cap) if plan.cap else None
        taken = selected[:cap] if cap else selected
        notes = {
            "pack": self.pack_id, "host": host, "sitemaps": list(plan.params["urls"]),
            "inventory_locs": len(inventory), "inventory_canonical": len(inventory_canon),
            "other_host_locs": foreign, "selected": len(selected), "cap": cap,
            "units": len(taken), "restricted_to_urls": len(only)}
        units = [Unit(key=u, url=u, meta={"lang": lang(u) or "", "delay_s": delay, "host": host},
                      consent_hosts=(urlsplit(u).hostname or host,))
                 for u in taken]
        return units, notes

    def acquire(self, unit: Unit, ctx: RunContext):
        if pdf_replay.active(ctx):
            return pdf_replay.acquire(self, unit, ctx)
        if PDF.search(unit.url):
            # the PDF harvest is the `engine: extractor` route (cytir.sources.cov.engine, Docling): not
            # this adapter's dialect. Accounted for, not silently dropped.
            raise UnitExcluded("unsupported_kind:pdf")
        client = ctx.http
        time.sleep(float(unit.meta.get("delay_s") or 0))       # the pack's own politeness
        r = client.get(unit.url)
        if r.status_code in (404, 410):
            raise UnitExcluded(f"origin_gone:{r.status_code}")      # the origin's own answer
        if r.status_code != 200 or not r.content:
            raise RuntimeError(f"http {r.status_code}")             # ours to retry: quarantined
        return [Captured(unit, r.content, {
            "status": r.status_code, "content_type": r.headers.get("content-type"),
            "final_url": str(getattr(r, "url", unit.url))})]

    def extract(self, captured: Captured, ctx: RunContext):
        if pdf_replay.active(ctx):
            return pdf_replay.extract(self, captured, ctx)
        return super().extract(captured, ctx)

    def chunk(self, doc, ctx: RunContext):
        if pdf_replay.active(ctx):
            return pdf_replay.chunk(self, doc, ctx)
        return super().chunk(doc, ctx)

    def passage_base(self, ctx: RunContext, unit: Unit) -> dict:
        """The row fields every route of this host shares; the licence is each route's own."""
        from cytir.core.common import fold_www

        plan = self.plan_for(unit)
        common = {"domain": fold_www(urlsplit(unit.url).netloc), "tier": plan.tier or "A",
                  "source_type": "official", "trust_tier": plan.effective_trust(),
                  "last_seen": ctx.today}
        return common

    def passage_common(self, ctx: RunContext, unit: Unit) -> dict:
        common = self.passage_base(ctx, unit)
        licence = self.plan_for(unit).effective_license()
        if licence:
            common["license"] = licence
        return common


@register_family
class ElcomDocsFamily(SitemapFamily):
    """ElCom, the Swiss federal electricity commission (www.elcom.admin.ch)."""

    family_id = "scratch_elcom_docs_exact"
    replay_l0 = True          # the PDF replay route (task 782)
    pack_id = "ch-regulator-docs"
    host = "www.elcom.admin.ch"


# The other sitemap routes of `ch-regulator-docs` that already have a declared contract row.
# Each is the five-line subclass the module docstring promises; the hosts of the same pack with
# NO declared row (blv, anmeldestelle, oak-bv, suva, swissgrid, enisa) wait for a POLICY PR.
# Selection equivalence with `cytir.sources.cov.engine.enum_sitemap` per route:
# the family sitemap routes test.

@register_family
class BavDocsFamily(SitemapFamily):
    """BAV, the Swiss federal transport office (www.bav.admin.ch)."""

    family_id = "scratch_bav_docs_exact"
    replay_l0 = True          # the PDF replay route (task 782)
    pack_id = "ch-regulator-docs"
    host = "www.bav.admin.ch"


@register_family
class BazlDocsFamily(SitemapFamily):
    """BAZL, the Swiss federal civil aviation office (www.bazl.admin.ch)."""

    family_id = "scratch_bazl_docs_exact"
    replay_l0 = True          # the PDF replay route (task 782)
    pack_id = "ch-regulator-docs"
    host = "www.bazl.admin.ch"


@register_family
class WekoDocsFamily(SitemapFamily):
    """WEKO, the Swiss competition commission (www.weko.admin.ch)."""

    family_id = "scratch_weko_docs_exact"
    replay_l0 = True          # the PDF replay route (task 782)
    pack_id = "ch-regulator-docs"
    host = "www.weko.admin.ch"


@register_family
class AstraHtmlFamily(SitemapFamily):
    """ASTRA, the Swiss federal roads office (www.astra.admin.ch): its HTML pages."""

    family_id = "astra_html_exact"
    pack_id = "ch-regulator-docs"
    host = "www.astra.admin.ch"


@register_family
class SwissmedicHtmlFamily(SitemapFamily):
    """Swissmedic, the Swiss therapeutic products agency (www.swissmedic.ch): its HTML pages."""

    family_id = "swissmedic_html_exact"
    pack_id = "ch-regulator-docs"
    host = "www.swissmedic.ch"
