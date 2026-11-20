"""Five more sitemap families, each proved selection-equivalent (task 683, S7 part 2).

`bav`, `bazl`, `weko`, `astra` and `swissmedic` (HTML) are five-line subclasses of
`SitemapFamily` (see the end of `cytir/sources/families/sitemap.py`). What the port changes is the
selection decision: before, the route's `include`/`exclude` regexes in `data/cov351_routes.yaml`
applied by `cytir.sources.cov.engine.enum_sitemap`; after, the pack's compiled `select`/`lang` applied by
the family's `discover`. For each route both run over the SAME inventory and must agree on every
URL and its language (the method of `test_aw_family_sitemap.py`, which proves ElCom).

The inventory is SYNTHETIC (no new network recording was approved for these hosts): the route's
own sitemap URLs serve a generated list of plausible paths in every language plus an adversarial
tail (other host, `http://`, query string, duplicate, excluded hubs, PDFs in and out of scope). The
test asserts the inventory is not trivial: each route keeps some URLs and drops others, so an
equivalence over an inventory the predicate ignores cannot pass.

`cytir.sources.cov.engine` (was `ingest_cov_docs`) is named by pack `igo-core`, so it is NOT re-pointed
here: the family is added beside it.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

from tests.helpers.repo_files import repo_file_or_skip

httpx = pytest.importorskip("httpx")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

TODAY = "2026-09-30"
#: (family id, cov351 pack, cov351 source, host)
ROUTES = [
    ("scratch_bav_docs_exact", "regdocs3", "bav_docs", "www.bav.admin.ch"),
    ("scratch_bazl_docs_exact", "regdocs3", "bazl_docs", "www.bazl.admin.ch"),
    ("scratch_weko_docs_exact", "regdocs2", "weko_docs", "www.weko.admin.ch"),
    ("astra_html_exact", "regdocs3", "astra_docs", "www.astra.admin.ch"),
    ("swissmedic_html_exact", "regdocs1", "swissmedic_guidance", "www.swissmedic.ch"),
]
LANGS = ("de", "fr", "it", "en")
PATHS = ("home/richtlinien.html", "home/zulassung/leitfaden.html", "richtlinien", "aufgaben",
         "kontakt", "impressum", "newsletter-abonnieren", "suche", "news/mitteilungen/archiv.html",
         "404.html", "sitemap.html", "ueber-uns/karriere", "offene-stellen", "dokumente/a.pdf",
         "download/x.pdf")


def _xml(urls):
    return ('<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            + "".join(f"<url><loc>{u}</loc></url>" for u in urls) + "</urlset>")


def _inventory(host):
    urls = []
    for lang in LANGS:
        for path in PATHS:
            urls += [f"https://{host}/{lang}/{path}",
                     f"https://{host}/{host.split('.')[1]}/{lang}/{path}",   # swissmedic-style
                     f"https://{host}/swissmedic/{lang}/{path}"]
        urls += [f"https://{host}/dam/{lang}/sd-web/abc/weisung.pdf",
                 f"https://{host}/{lang}/home/x/y/deep-page.html"]
    urls += [f"https://{host}/", f"http://{host}/de/richtlinien",
             f"https://{host.upper()}/de/richtlinien?utm_source=x",
             f"https://{host}/de/richtlinien",                       # duplicate
             "https://elsewhere.example.org/de/richtlinien"]         # another host
    return urls


def _client(sitemaps, inventory):
    from cytir.acquire.tape import Tape
    body = _xml(inventory)
    return Tape([{"method": "GET", "url": u, "status": 200, "content_type": "application/xml",
                  "body": body} for u in sitemaps]).client()


@pytest.fixture
def regdocs_pack():
    """The pack is withheld from the public cut: the tests that read it skip there."""
    return repo_file_or_skip("sources/ch/regulator-docs.yaml")


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setenv("AW_LANGGATE_OPTIONAL", "1")
    monkeypatch.setenv("AW_LID_MODEL", "/nonexistent/lid.176.bin")
    monkeypatch.setenv("AW_CHUNKER", "legacy")


@pytest.mark.parametrize("family,pack,source,host", ROUTES, ids=[r[0] for r in ROUTES])
def test_selection_equals_the_old_enumeration_on_the_same_inventory(tmp_path, family, pack,
                                                                     source, host, regdocs_pack):
    from cytir.sources.cov import engine as COV
    from cytir.sources import registry
    from cytir.sources.family import RunContext

    cfg = COV.load_routes(str(REPO / "data" / "cov351_routes.yaml"), pack)["sources"][source]
    inventory = _inventory(host)
    old = {row["url"]: row["lang"] for row in
           COV.enum_sitemap(_client(cfg["sitemaps"], inventory), cfg, None, lambda m: None)}

    adapter, _row, spec = registry.admit(family)
    plan = adapter.plan()
    assert list(plan.params["urls"]) == list(cfg["sitemaps"])        # same sitemaps, pack vs cov351
    ctx = RunContext(out_dir=tmp_path, today=TODAY, chunk_spec=spec,
                     http=_client(plan.params["urls"], inventory))
    new = {u.url: u.meta["lang"] for u in adapter.discover(ctx)}

    assert old == new, sorted(set(old) ^ set(new))[:10]
    # the inventory really exercised the rule: some kept, some dropped, each alike
    canon_all = {u for u in inventory if u.startswith(f"https://{host}/")}
    assert 5 <= len(new) < len(canon_all), (len(new), len(canon_all))
    assert not [u for u in new if "elsewhere.example.org" in u]


@pytest.mark.parametrize("family,pack,source,host", ROUTES, ids=[r[0] for r in ROUTES])
def test_each_family_is_admitted_and_reads_the_same_contract_row(family, pack, source, host):
    from cytir.sources import registry
    adapter, row, spec = registry.admit(family)
    assert adapter.host == host and row["family_id"] == family
    # task 852: once the row declares a policy the adapter answers `declared` (chunk_policy_for), and
    # `spec.declared` is that policy id; a HELD row keeps the dispatcher's pinned selection.
    declared = row["chunk_policy"]["policy_id"]
    assert spec.declared == ("dispatcher" if declared == "unknown" else declared)
