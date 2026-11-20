"""The four authority packs (task 1393): the pack YAML selects what the legacy HOSTS dicts select.

The Protocol adapter (`cytir.sources.families.authority_packs`) reads each host's selection, language,
cap, delay, trust, tier and licence slug from `sources/**/*.yaml`; the legacy driver
(`ingest_authority_packs`, an alias of `cytir.acquire.authority_packs`) carries them as hand-written
lambdas. They must agree on every URL of an adversarial pool and on every scalar. No network.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO),):
    if p not in sys.path:
        sys.path.insert(0, p)

from cytir.acquire import authority_packs as legacy  # noqa: E402
from cytir.sources import pack_loader as PL  # noqa: E402

PACK_IDS = {"energy": "energy-climate-intl", "biomed": "biomed-authority-core",
            "education": "education-research", "urban": "urban-governance"}

SEGS = ["reports", "report/ar6/wg1/spm", "sr15/chapter", "sr15", "analysis", "analysis/x/y",
        "commentaries/a", "policies/a/b", "articles/a", "data-and-statistics/charts/a/b",
        "topics/a", "resources/x", "en/publikationen/a", "en/topics/a/b", "themen/klima-energie/a",
        "publikationen/a", "en/node/12", "evidence/CD123_x", "evidence", "news/a/b",
        "publications/m/a", "bulletin/volumes/1/2", "csr/don/2020/x", "es/news/a", "ar/news/b",
        "zh/x/y", "spanish/a/b", "health-topics/a/b", "en/medicines/a", "en/news/a",
        "en/publications-data/a", "docs/article/x", "regions-et-territoires/a/b",
        "DE/Themen/a", "EN/Topics/b", "DE/Aktuelles/x/y", "ncee/a/b", "wwc/x", "pubs/a",
        "en/publication/x", "fr/articles/a", "es/publication/x", "en/research/x/y",
        "some-article-slug", "news", "author/jane", "wp-content/u/x.png", "search/x/y",
        "login/a", "print/a/b", "contact", "newsletter/a/b", "a/b/c/d", "guidance/a/b"]
HOSTS = sorted({cfg["host"] for pack in legacy.PACKS.values() for cfg in pack.values()})
URLS = [f"https://{h}/{s}" for h in HOSTS for s in SEGS]


def _pairs():
    for key, pack_id in PACK_IDS.items():
        by_host = {p.host: p for p in PL.plans(PL.load_pack(PL.find_pack(pack_id)))}
        assert set(by_host) == {c["host"] for c in legacy.PACKS[key].values()}, key
        for name, cfg in legacy.PACKS[key].items():
            yield pytest.param(cfg, by_host[cfg["host"]], id=f"{key}-{name}")


@pytest.mark.parametrize("cfg,plan", list(_pairs()))
def test_pack_yaml_equals_legacy_host(cfg, plan):
    assert plan.cap == cfg["cap"] and plan.delay_s == cfg["delay"]
    assert plan.effective_trust() == cfg["trust"] and plan.tier == cfg["tier"]
    assert plan.effective_license() == cfg["license"]
    legacy_select = cfg["include"] if cfg["mode"] == "sitemap" else cfg["allow"]
    select, lang = plan.predicate(), plan.lang_fn()
    if cfg["mode"] == "sitemap":
        assert list(plan.params["urls"]) == cfg["sitemaps"]
        assert plan.route == "source_owned_sitemap_fetch"
    else:
        assert plan.route == "direct_origin_crawl" and list(plan.params["seeds"]) == cfg["seeds"]
    for u in (x for x in URLS if x.split("/")[2] == cfg["host"]):
        assert bool(select(u)) == bool(legacy_select(u)), u
        assert (lang(u) or "") == cfg["lang"](u), u
