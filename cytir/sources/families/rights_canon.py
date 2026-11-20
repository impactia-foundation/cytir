"""`scratch_rights_canon_exact` -- the fundamental-rights canon pack (task 187), on the Protocol (task 1408).

The pack crawled seven court / UN / EU / national-rights-body hosts politely, read ECtHR leading
case law from the HUDOC JSON query API (one landing record per judgment) and took the EU Charter of
Fundamental Rights article by article from EUR-Lex. It is a FROZEN build (654 sealed rows in the
assembly; no live route runs it). The adapter ports the stages those sealed rows need, so a
stranger can replay the pack from a driver directory; the record SHAPE lives once, in the moved
driver `cytir.sources.rights_canon.crawl` (`page_records`, `hudoc_record`, `charter_records`), which
the resumable crawl CLI also calls.

STAGE MAP
  discover   adapter   sitemap hosts: the host inventory (`host_inventory`, the driver's own,
                       `<root>/inv/<host>.txt` when present, else built from the sitemaps) cut at the
                       host `cap`; BFS hosts: their seed pages; HUDOC: one unit per driver query;
                       Charter: one unit per language
  acquire    adapter   page: robots (merged groups, longest match) then one polite GET; HUDOC: the
                       query paged through `_hudoc_get`; Charter: the legal-content HTML
  extract    moved     page -> html Doc; HUDOC -> `hudoc_record`; Charter -> `charter_records`
  chunk      moved     page -> `page_records` (extraction gates, citation-list filter, head-cap 50,
                       title clean); HUDOC and Charter rows are already one record each
  seal       framework
  l1         PARTIAL   not written
  PARTIAL    `discover:bfs_expansion`: the driver follows page links from the seeds (`cncdh`,
             `bverfg`); the adapter lists the seeds only. Run the crawl CLI for the link walk.

CONSENT AND LICENCE are read from the contract row, never written here: the row holds `unknown`
for both and the run notes carry the values as they stand. Facts the code shows (for Tim, not
declarations): per-record licences are stamped in the rows (`ohchr`: "UN (c) ... Content-Signal:
search=yes, ai-train=no -> train_ok=false", `bverfg`: "§ 5 UrhG gemeinfrei", HUDOC: "pointer+snippet
only"); the driver gates ran with `--direct-crawl --accept-search-grant`.

Options (`--opt KEY=VALUE`): `root` (required; the driver out-dir, holds `inv/`), `hosts` (comma
list of host keys; default all), `api` (`0` skips HUDOC and Charter), `max_chunks` (default 50).
"""
from __future__ import annotations

import json
import os
import urllib.parse

from cytir.sources.families.web import WebPageMixin
from cytir.sources.family import CHUNK_DISPATCHER, HttpSpec, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family
from cytir.sources.rights_canon import crawl as C

HUDOC_HOST = "hudoc.echr.coe.int"
EURLEX_HOST = "eur-lex.europa.eu"


def contract_terms(ctx: RunContext) -> dict:
    """Consent and licence exactly as the row declares them (today: `unknown`)."""
    row = ctx.contract or {}
    consent, licence = row.get("consent", {}), row.get("licence", {})
    return {"search_ok": consent.get("search_ok"), "train_ok": consent.get("train_ok"),
            "consent_resolution": consent.get("resolution"),
            "licence_basis": licence.get("basis"), "licence_spdx": licence.get("spdx")}


@register_family
class RightsCanonFamily(WebPageMixin):
    family_id = "scratch_rights_canon_exact"
    route = "source_owned_sitemap_fetch"
    direct_crawl = True
    chunk_policy_id = CHUNK_DISPATCHER      # the row declares none (all `unknown`)
    http_spec = HttpSpec(timeout=45, user_agent_kind="pack_fetch")   # task 1556: the runner builds and closes the client
    require_records = True
    PARTIAL = ("discover:bfs_expansion", "l1")

    def __init__(self) -> None:
        self._robots: dict[str, object] = {}
        self._next_ok: dict[str, float] = {}
        self._seen_hudoc: set[tuple[str, str]] = set()

    # -- helpers ----------------------------------------------------------------------------
    def _cfg(self, key: str) -> dict:
        cfg = C.HOSTS[key]
        return dict(cfg, eff_delay=cfg["delay"])

    # -- SourceFamily -----------------------------------------------------------------------
    def discover(self, ctx: RunContext):
        ctx.notes["contract_terms"] = contract_terms(ctx)
        root = ctx.options.get("root")
        if not root or not os.path.isdir(root):
            raise ValueError("scratch_rights_canon_exact needs --opt root=<driver out-dir>")
        want = [h for h in (ctx.options.get("hosts") or "").split(",") if h] or list(C.HOSTS)
        unknown = [h for h in want if h not in C.HOSTS]
        if unknown:
            raise ValueError(f"scratch_rights_canon_exact: unknown host key(s) {unknown}")
        units, counts = [], {}
        for key in want:
            cfg = self._cfg(key)
            if cfg["mode"] == "sitemap":
                urls = C.host_inventory(ctx.http, key, cfg, os.path.join(root, "inv"))
                urls = urls[:cfg["cap"]]
            else:
                urls = []
            urls += [C.canon(s) for s in cfg.get("seeds", [])]
            seen = set()
            for u in urls:
                if u in seen or C.BINARY_RE.search(u):
                    continue
                seen.add(u)
                units.append(Unit(key=f"page:{key}:{u}", url=u,
                                  meta={"kind": "page", "host_key": key},
                                  consent_hosts=(cfg["host"],)))
            counts[key] = len(seen)
        if any(C.HOSTS[k]["mode"] == "bfs" for k in want):
            ctx.notes["bfs_expansion"] = "not_run: seeds only (PARTIAL discover:bfs_expansion)"
        if ctx.options.get("api", "1") != "0":
            for i, (lang, q) in enumerate(C.HUDOC_QUERIES):
                units.append(Unit(key=f"hudoc:{lang}:{i}", url=C.HUDOC_API,
                                  meta={"kind": "hudoc", "lang": lang, "query": q},
                                  consent_hosts=(HUDOC_HOST,)))
            for L in C.CHARTER_LANGS:
                units.append(Unit(key=f"charter:{L}", url=C.charter_url(L),
                                  meta={"kind": "charter", "L": L},
                                  consent_hosts=(EURLEX_HOST,)))
        if not units:
            raise RuntimeError("scratch_rights_canon_exact: nothing discovered; an empty walk "
                               "is a wrong root, not an empty source")
        ctx.notes.update({"units": len(units), "page_units_by_host": counts})
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        kind = unit.meta["kind"]
        client = ctx.http
        if kind == "hudoc":
            pages, start, total = [], 0, None
            while total is None or start < total:
                data = C._hudoc_get(client, unit.meta["query"], start, C.HUDOC_PAGE)
                if data is None:
                    raise RuntimeError(f"hudoc query failed: {unit.meta['query']} @ {start}")
                total = int(data.get("resultcount", 0))
                if not data.get("results"):
                    break
                pages.append(data)
                start += C.HUDOC_PAGE
            body = json.dumps(pages, sort_keys=True, ensure_ascii=False).encode("utf-8")
            return [Captured(unit, body, {"content_type": "application/json", "pages": len(pages)})]
        if kind == "charter":
            r = client.get(unit.url)
            if r.status_code != 200:
                raise RuntimeError(f"eurlex charter {unit.meta['L']}: HTTP {r.status_code}")
            return [Captured(unit, r.content, {"status": 200,
                                               "content_type": r.headers.get("content-type")})]
        return self._acquire_page(unit, ctx, client)

    def _acquire_page(self, unit: Unit, ctx: RunContext, client):
        import time
        key = unit.meta["host_key"]
        cfg = self._cfg(key)
        if key not in self._robots:
            rb = C.fetch_robots(client, cfg["host"])
            if rb is None:
                raise RuntimeError(f"{cfg['host']}: robots unavailable or a script shell; "
                                   "host skipped (the driver's rule)")
            self._robots[key] = rb
            cfg["eff_delay"] = max(cfg["delay"], rb.delay or 0.0)
            ctx.notes.setdefault("eff_delay_s", {})[key] = cfg["eff_delay"]
        rb = self._robots[key]
        eff = max(cfg["delay"], rb.delay or 0.0)
        pq = urllib.parse.urlsplit(unit.url)
        if not rb.allowed(pq.path + (("?" + pq.query) if pq.query else "")):
            raise UnitExcluded("robots_disallowed")
        allow = cfg.get("allow")
        if allow is not None and not allow(unit.url):
            raise UnitExcluded("not_scope")
        wait = self._next_ok.get(key, 0.0) - time.time()
        if wait > 0:
            time.sleep(wait)
        self._next_ok[key] = time.time() + eff
        r = client.get(unit.url)
        final_host = urllib.parse.urlsplit(str(r.url)).netloc.lower()
        ctype = r.headers.get("content-type", "").lower()
        if r.status_code in (404, 410):
            raise UnitExcluded(f"origin_gone:{r.status_code}")
        if r.status_code != 200 or "html" not in ctype or \
                C.ALIAS.get(final_host, final_host) not in cfg.get("ok_hosts", {cfg["host"]}):
            raise RuntimeError(f"http {r.status_code} {ctype!r} host {final_host}")
        return [Captured(unit, r.content, {"status": 200, "content_type": r.headers.get("content-type"),
                                           "final_url": str(r.url).split("#")[0]})]

    def extract(self, cap: Captured, ctx: RunContext):
        meta = cap.unit.meta
        if meta["kind"] == "page":
            html = self.page_html(cap)
            return [Doc(url=cap.meta.get("final_url") or cap.unit.url, text=html, title="",
                        extra={"host_key": meta["host_key"]})]
        if meta["kind"] == "hudoc":
            from cytir.acquire.fetch_process import detect_lang
            docs = []
            for page in json.loads(cap.body):
                for row in page.get("results", []):
                    c = row.get("columns", {})
                    itemid = c.get("itemid", "")
                    if not itemid or (meta["lang"], itemid) in self._seen_hudoc:
                        continue
                    self._seen_hudoc.add((meta["lang"], itemid))
                    rec = C.hudoc_record(c, meta["lang"], itemid, detect_lang, ctx.today)
                    if rec is None:
                        ctx.notes["hudoc_langdrop"] = ctx.notes.get("hudoc_langdrop", 0) + 1
                        continue
                    docs.append(Doc(url=rec["url"], text=rec["text"], title=rec["title"],
                                    lang=rec["lang"], extra={"row": rec}))
            if not docs:
                raise UnitExcluded("no_new_hudoc_rows")
            return docs
        secs = C._charter_sections(cap.body.decode("utf-8", "replace"))
        if len(secs) < 40:                       # the Charter has 54 articles
            raise RuntimeError(f"eurlex charter {meta['L']}: only {len(secs)} sections parsed "
                               "-- layout changed, refusing to ship junk")
        return [Doc(url=r["url"], text=r["text"], title=r["title"], lang=r["lang"],
                    anchor=r["anchor"], extra={"row": r})
                for r in C.charter_records(meta["L"], secs, ctx.today)]

    def chunk(self, doc: Doc, ctx: RunContext):
        if "row" in doc.extra:
            return [doc.extra["row"]]
        cfg = C.HOSTS[doc.extra["host_key"]]
        rows, counts = C.page_records(doc.url, doc.text, cfg, ctx.today,
                                      int(ctx.options.get("max_chunks") or 50))
        for k, v in counts["apparatus"].items():
            ap = ctx.notes.setdefault("apparatus", {})
            ap[k] = ap.get(k, 0) + v
        return rows
