"""URL admission rule (task 2713): the 13 junk rows of the served index, as fixtures."""
from __future__ import annotations

import json

import pytest

from cytir.ops import spam_tombstones as ST
from cytir.sources import family_stage as FS
from cytir.sources import url_admission as UA

KO = "%EC%B9%B4%EC%A7%80%EB%85%B8+%EC%82%AC%EC%9D%B4%ED%8A%B8"   # stands in for the two spam queries
SEARCH_PAGES = [
    "https://docs.github.com/en/search?query=rest+api+deployments",
    "https://docs.github.com/en/search?query=custom+domain",
    f"https://docs.github.com/en/search?query={KO}",
    f"https://docs.github.com/pt/search?query={KO}",
]
HUB_DUPLICATES = [
    ("https://docs.github.com/en/rest?latest", "https://docs.github.com/en/rest"),
    ("https://docs.github.com/en/rest?apiVersion=2026-03-10", "https://docs.github.com/en/rest"),
    ("https://docs.github.com/de/rest?ref=daveabrock.com", "https://docs.github.com/de/rest"),
]
FEDLEX_CHROME = [
    "https://fedlex.admin.ch/page-not-found",
    "https://fedlex.admin.ch/",
    "https://www.fedlex.admin.ch/de/cc",
    "https://www.fedlex.admin.ch/de/fga",
]
#: the served index holds `page-not-found` and the root twice each (two chunk rows): six rows, four urls
FEDLEX_ROWS = FEDLEX_CHROME + ["https://fedlex.admin.ch/page-not-found", "https://fedlex.admin.ch/"]


@pytest.mark.parametrize("url", SEARCH_PAGES)
def test_search_result_pages_are_refused(url):
    assert UA.refusal(url) == UA.REASON_SEARCH


def test_search_rule_is_any_host_and_needs_a_query_parameter():
    assert UA.refusal("https://example.org/docs/search?q=x") == UA.REASON_SEARCH
    assert UA.refusal("https://example.org/docs/search") is None
    assert UA.refusal("https://example.org/docs/search?page=2") is None
    assert UA.refusal("https://example.org/search/howto?query=x") is None


@pytest.mark.parametrize("url", FEDLEX_CHROME)
def test_fedlex_site_chrome_is_refused(url):
    assert UA.refusal(url) == UA.REASON_CHROME


def test_fedlex_eli_urls_are_admitted():
    for u in ("https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de",
              "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_336_c",
              "https://fedlex.admin.ch/eli/cc/2015/493/fr"):
        assert UA.refusal(u) is None
    assert UA.refusal("https://example.org/page-not-found") is None


@pytest.mark.parametrize("variant,canonical", HUB_DUPLICATES)
def test_version_and_referral_params_fold_into_the_page(variant, canonical):
    assert UA.fold_url(variant) == canonical
    assert UA.fold_url(canonical) == canonical
    assert UA.refusal(variant) is None


def test_fold_keeps_functional_params():
    assert UA.fold_url("https://docs.github.com/a?id=7&latest&utm_source=x") == "https://docs.github.com/a?id=7"


def test_version_fold_is_scoped_to_known_docs_hosts():
    assert UA.fold_url("https://example.org/a?latest") == UA.strip_tracking_url("https://example.org/a?latest")


def test_fedlex_data_filestore_urls_are_admitted():
    assert UA.refusal("https://fedlex.data.admin.ch/filestore/fedlex.data.admin.ch/eli/cc/27/317_321_377/"
                      "20250101/de/xml/cc-27-317_321_377-20250101-de-xml-1.xml") is None


def _b(url, pid):
    return (json.dumps({"url": url, "point_id": pid, "text": "t", "license": "cc-by"}) + "\n").encode()


def test_stage_door_drops_every_junk_url_and_folds_hub_variants():
    junk = SEARCH_PAGES + FEDLEX_ROWS
    buf = [_b(u, f"j{i}") for i, u in enumerate(junk)]
    buf += [_b(v, f"v{i}") for i, (v, _c) in enumerate(HUB_DUPLICATES)]
    buf += [_b("https://docs.github.com/en/rest", "c1")]
    kept, refused = FS.drop_refused_urls(buf)
    assert sorted(r["point_id"] for r in refused) == sorted(f"j{i}" for i in range(len(junk)))
    assert {r["reason"] for r in refused} == {UA.REASON_SEARCH, UA.REASON_CHROME}
    keep, dropped = FS.drop_url_variants(kept)
    assert [json.loads(b)["point_id"] for b in keep] == ["v2", "c1"]      # de/rest has no canonical row
    assert sorted(dropped) == ["v0", "v1"]


def test_resolve_labels_the_thirteen_rows(monkeypatch):
    rows = {}
    n = 0
    for u in SEARCH_PAGES + [v for v, _ in HUB_DUPLICATES] + FEDLEX_ROWS + [
            "https://docs.github.com/en/rest", "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de"]:
        rows[f"id{n}"] = {"url": u, "domain": "x"}
        n += 1

    def fake_scroll(url, collection, must):
        return [{"id": i, **p} for i, p in rows.items()]
    monkeypatch.setattr(ST, "_scroll", fake_scroll)
    out = ST.resolve("http://twin", "c", urls=[], domains=["docs.github.com"])
    assert len(out) == 13
    by = {}
    for r in out:
        by[r["reason"]] = by.get(r["reason"], 0) + 1
    assert by == {UA.REASON_SEARCH: 4, UA.REASON_TRACKING: 3, UA.REASON_CHROME: 6}


def test_resolve_leaves_out_a_variant_that_is_the_only_copy(monkeypatch):
    rows = {"a": {"url": "https://docs.github.com/en/rest?latest", "domain": "x"},
            "b": {"url": "https://docs.github.com/en/pages?latest", "domain": "x"},
            "c": {"url": "https://docs.github.com/en/pages", "domain": "x"}}

    def fake_scroll(url, collection, must):
        k = must[0]
        if k["key"] == "domain":
            return [{"id": i, **p} for i, p in rows.items()]
        return [{"id": i, **p} for i, p in rows.items() if p["url"] == k["match"]["value"]]
    monkeypatch.setattr(ST, "_scroll", fake_scroll)
    warns = []
    out = ST.resolve("http://t", "c", urls=[], domains=["docs.github.com"], warn=warns.append)
    assert [r["point_id"] for r in out] == ["b"] and len(warns) == 1
    out = ST.resolve("http://t", "c", urls=[], domains=["docs.github.com"], allow_orphan_variants=True)
    assert [r["point_id"] for r in out] == ["a", "b"]
