"""`scratch_rights_canon_exact` on the family Protocol (task 1408).

No network: every origin is a `httpx.MockTransport` over hand-written fixtures (a French rights
body page, a HUDOC result page, a 45-article Charter page). Pins: registered and conformant
(PARTIAL named); a `plan` dry run lists the inventory units; consent and licence come from the row
as they stand (`unknown`); robots and scope refusals are EXCLUDED, not dropped.

BYTE PARITY with the pre-move driver. The reference digests come from running the ORIGINAL driver
(the file as of commit 0dffefb85, before this task moved it) over exactly these fixtures with the
same offline harness (fake clock, `LAST_SEEN` fixed to 2026-10-30), and hashing `passages.jsonl`.
The moved driver's `main()` and the adapter's stages must produce the same bytes.
"""
from __future__ import annotations

import hashlib
import json
import sys

import httpx
import pytest

from cytir.sources import contract as contract_mod
from cytir.sources import registry
from cytir.sources.family import RunContext, SourceFamily, UnitExcluded
from cytir.sources.run_family import run_family

# ---- reference digests of the ORIGINAL driver's passages.jsonl -------------------------------
REF_ALL = ("42406657e435cb863258d9c265178d325c9ed5415bed7c874128333b73431909", 145)   # cncdh+hudoc+eurlex
REF_API = ("65dc77f46830cbfa7e5c2f99e09ad50eb8ce72848db3bbefc92e6fc221588d9f", 139)   # hudoc+eurlex
REF_DDD = ("f8bb516768e71fd43320b8eafc59f8736dd96a6e76e553c6b56c996b6d931f01", 6)     # ddd, 2 pages

DDD_URLS = ["https://www.defenseurdesdroits.fr/a", "https://www.defenseurdesdroits.fr/b"]


def _par(i):
    return ("La Commission nationale consultative des droits de l'homme rappelle que la dignité de "
            "chaque personne est inviolable et que l'État doit garantir l'égalité devant la loi, la "
            "liberté d'expression et la protection contre toute forme de discrimination (avis %d). "
            "Elle recommande aux pouvoirs publics de renforcer l'accès au droit et à la justice pour "
            "toutes les personnes vulnérables, notamment les mineurs isolés et les demandeurs d'asile." % i)


def _page(title, n, links=()):
    body = "".join("<p>%s</p>" % _par(i) for i in range(n))
    lk = "".join('<a href="%s">x</a>' % link for link in links)
    return ("<html lang='fr'><head><title>%s | CNCDH</title></head><body><main><h1>%s</h1>%s%s</main>"
            "</body></html>" % (title, title, body, lk))


CHARTER = "<html><body>" + "".join(
    "<div class='eli-subdivision' id='art_%d'>Article %d — Everyone has the right to respect for "
    "dignity and liberty under the Charter provision number %d of the Union.</div>" % (i, i, i)
    for i in range(1, 46)) + "</body></html>"


def _hrow(i):
    return {"columns": {"itemid": "001-%d" % i, "docname": "CASE OF X v. Y %d" % i,
                        "appno": "1234/7%d;55/8%d" % (i, i), "importance": "1",
                        "kpdate": "1976-12-07T00:00:00", "article": "10;P1-1",
                        "conclusion": "Violation of Article 10", "respondent": "UK;FRA",
                        "doctypebranch": "GRANDCHAMBER", "languageisocode": "ENG"}}


def _handler(robots="User-agent: *\nAllow: /\n"):
    def handler(req):
        u = str(req.url)
        if u.endswith("/robots.txt"):
            return httpx.Response(200, text=robots)
        if "hudoc.echr.coe.int/app/query/results" in u:
            return httpx.Response(200, json={"resultcount": 2, "results": [_hrow(1), _hrow(2)]})
        if "eur-lex.europa.eu" in u:
            return httpx.Response(200, text=CHARTER, headers={"content-type": "text/html"})
        p = req.url.path
        h = {"content-type": "text/html; charset=utf-8"}
        if p == "/":
            return httpx.Response(200, text=_page("Accueil", 1, ["/droits-a", "/droits-b", "/doc.pdf"]),
                                  headers=h)
        if p in ("/droits-a", "/droits-b", "/a", "/b"):
            return httpx.Response(200, text=_page("Droits " + p[-1], 4), headers=h)
        return httpx.Response(404, text="nope")
    return handler


def _client(**kw):
    return httpx.Client(transport=httpx.MockTransport(_handler(**kw)), follow_redirects=True)


class _Clock:
    t = 1000.0

    @classmethod
    def time(cls):
        return cls.t

    @classmethod
    def sleep(cls, s):
        cls.t += s


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("AW_ALLOW_UNIDENTIFIED_CRAWL", "1")
    monkeypatch.setenv("AW_LANGGATE_OPTIONAL", "1")       # no 126 MB lid model in CI
    monkeypatch.setattr("time.sleep", lambda s: None)


def _driver_digest(tmp_path, monkeypatch, only, inv=None, extra=()):
    from cytir.sources.rights_canon import crawl as C
    out = tmp_path / "drv"
    for key, urls in (inv or {}).items():
        (out / "inv").mkdir(parents=True, exist_ok=True)
        (out / "inv" / f"{key}.txt").write_text("\n".join(urls))
    monkeypatch.setattr(C, "LAST_SEEN", "2026-10-30")
    monkeypatch.setattr(C, "time", _Clock)
    monkeypatch.setattr(httpx, "HTTPTransport", lambda *a, **k: httpx.MockTransport(_handler()))
    monkeypatch.setattr(sys, "argv", ["x", "--out-dir", str(out), "--only", only, *extra])
    C.main()
    data = (out / "passages.jsonl").read_bytes()
    return hashlib.sha256(data).hexdigest(), data.count(b"\n")


def _ctx(root, tmp_path, **opts):
    row = contract_mod.by_id(contract_mod.load())["scratch_rights_canon_exact"]
    return RunContext(out_dir=tmp_path / "out", today="2026-10-30", skip_consent=True,
                      http=_client(), options={"root": str(root), **opts}, contract=row,
                      log=lambda m: None)


def _stage_all(fam, ctx):
    rows = []
    for unit in fam.discover(ctx):
        try:
            caps = fam.acquire(unit, ctx)
            for cap in caps:
                for doc in fam.extract(cap, ctx):
                    rows += fam.chunk(doc, ctx)
        except UnitExcluded:
            pass
    return rows


def _digest(rows):
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    return hashlib.sha256(data).hexdigest(), data.count(b"\n")


@pytest.fixture
def root(tmp_path):
    (tmp_path / "root" / "inv").mkdir(parents=True)
    (tmp_path / "root" / "inv" / "ddd.txt").write_text("\n".join(DDD_URLS))
    return tmp_path / "root"


def test_registered_and_conformant():
    assert "scratch_rights_canon_exact" in registry.registered()
    adapter, row, _spec = registry.admit("scratch_rights_canon_exact")
    assert isinstance(adapter, SourceFamily)
    assert row["family_id"] == "scratch_rights_canon_exact"
    assert adapter.PARTIAL == ("discover:bfs_expansion", "l1")


def test_plan_dry_run_lists_inventory_and_api_units(root, tmp_path):
    plan = run_family("scratch_rights_canon_exact", out_dir=tmp_path / "o", today="2026-10-30",
                      skip_consent=True, options={"root": str(root), "hosts": "ddd"},
                      dry_run=True, log=lambda m: None)
    assert plan["units"] == [f"page:ddd:{u}" for u in DDD_URLS] + [
        "hudoc:en:0", "hudoc:en:1", "hudoc:en:2", "hudoc:fr:3", "hudoc:fr:4", "hudoc:fr:5",
        "charter:EN", "charter:FR", "charter:DE"]


def test_terms_come_from_the_row_unknown_stays_unknown(root, tmp_path):
    from cytir.sources.families.rights_canon import RightsCanonFamily
    fam, ctx = RightsCanonFamily(), _ctx(root, tmp_path, hosts="ddd", api="0")
    fam.discover(ctx)
    terms, row = ctx.notes["contract_terms"], ctx.contract
    assert terms["search_ok"] == row["consent"]["search_ok"]
    assert terms["licence_basis"] == row["licence"]["basis"]
    bfs_ctx = _ctx(root, tmp_path, hosts="cncdh", api="0")
    fam.discover(bfs_ctx)
    assert bfs_ctx.notes["bfs_expansion"] == "not_run: seeds only (PARTIAL discover:bfs_expansion)"


def test_moved_driver_main_is_byte_identical_to_the_pre_move_driver(tmp_path, monkeypatch):
    assert _driver_digest(tmp_path, monkeypatch, "cncdh,hudoc,eurlex") == REF_ALL


def test_adapter_sitemap_pages_are_byte_identical_to_the_pre_move_driver(root, tmp_path):
    from cytir.sources.families.rights_canon import RightsCanonFamily
    rows = _stage_all(RightsCanonFamily(), _ctx(root, tmp_path, hosts="ddd", api="0"))
    assert _digest(rows) == REF_DDD


def test_adapter_hudoc_and_charter_are_byte_identical_to_the_pre_move_driver(root, tmp_path):
    from cytir.sources.families.rights_canon import RightsCanonFamily
    rows = _stage_all(RightsCanonFamily(), _ctx(root, tmp_path, hosts="ddd"))
    api = [r for r in rows if r.get("provenance") == "rights_canon"
           and r["domain"] in ("hudoc.echr.coe.int", "eur-lex.europa.eu")]
    assert _digest(api) == REF_API


def test_robots_disallow_is_an_exclusion_not_a_silent_drop(root, tmp_path):
    from cytir.sources.families.rights_canon import RightsCanonFamily
    fam, ctx = RightsCanonFamily(), _ctx(root, tmp_path, hosts="ddd", api="0")
    ctx.http = _client(robots="User-agent: *\nDisallow: /a\n")
    first, second = list(fam.discover(ctx))
    with pytest.raises(UnitExcluded) as e:
        fam.acquire(first, ctx)
    assert e.value.reason == "robots_disallowed"
    assert fam.acquire(second, ctx)[0].meta["status"] == 200


def test_a_charter_layout_change_is_refused_not_shipped(root, tmp_path, monkeypatch):
    from cytir.sources.families.rights_canon import RightsCanonFamily
    fam, ctx = RightsCanonFamily(), _ctx(root, tmp_path, hosts="ddd")
    unit = next(u for u in fam.discover(ctx) if u.key == "charter:EN")
    cap = fam.acquire(unit, ctx)[0]
    cap.body = b"<html><body><p>no articles</p></body></html>"
    with pytest.raises(RuntimeError, match="layout changed"):
        fam.extract(cap, ctx)


def test_crawl_import_reads_no_identity(tmp_path, monkeypatch):
    import importlib
    from cytir.core import user_agent
    from cytir.sources.rights_canon import crawl as C
    for k in ("AW_OPERATOR_URL", "AW_OPERATOR_CONTACT", "AW_ALLOW_UNIDENTIFIED_CRAWL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(user_agent, "operator",
                        lambda *a, **k: calls.append(1) or ("https://example.org", "ops@example.org", False))
    importlib.reload(C)
    assert calls == []
    assert C.headers()["User-Agent"].startswith("AuthoritativeWebBot")
    assert calls == [1]
