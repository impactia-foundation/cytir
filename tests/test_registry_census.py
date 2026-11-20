"""Tests for cytir/ops/registry_census.py (task 987): sealed URL samples through the normaliser, each
registry parser on a saved response, the fetchers end to end with an injected getter (no network),
and the merge output shape."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cytir.ops import registry_census as C

FX = Path(__file__).parent / "fixtures" / "registry_census"


def _fx(name):
    return (FX / name).read_text()


@pytest.mark.parametrize("raw,want", [
    ("git+https://github.com/react/react.git", "github.com/react/react"),
    ("git@github.com:Foo/Bar.git", "github.com/foo/bar"),
    ("github:a/b", "github.com/a/b"),
    ("a/b", "github.com/a/b"),
    ("https://github.com/o/r/tree/main/pkg", "github.com/o/r"),
    ("ssh://git@github.com/o/r.git", "github.com/o/r"),
    ("git://github.com/o/r#readme", "github.com/o/r"),
    ("https://www.github.com/O/R/", "github.com/o/r"),
    ("https://gitlab.com/g/sub/r/-/tree/x", "gitlab.com/g/sub/r"),
    ("https://github.com/o", None),
    ("https://o.github.io/r", None),
    ("", None),
    (None, None),
])
def test_normalise(raw, want):
    n = C.normalise_repo_url(raw)
    assert (n["url"] if n else None) == want


def test_npm_parsers():
    assert C.parse_npm_top(_fx("npm_top.js"))[:2] == ["semver", "minimatch"]
    assert C.normalise_repo_url(C.parse_npm_repo(_fx("npm_latest_react.json")))["url"] == "github.com/react/react"


def test_pypi_parsers():
    assert C.parse_pypi_top(_fx("pypi_top.json"))[0][0] == "boto3"
    raw, src = C.parse_pypi_repo(_fx("pypi_requests.json"))
    assert raw == "https://github.com/psf/requests" and src == "project_urls.Source"


def test_crates_parser():
    rows, nxt = C.parse_crates_page(_fx("crates_page.json"))
    assert rows[0]["package"] == "hashbrown" and rows[0]["repo"].endswith("rust-lang/hashbrown")
    assert nxt.startswith("?sort=downloads")


def test_rubygems_parsers():
    assert C.parse_bestgems_page(_fx("bestgems_page.html"))[0] == ("bundler", 1, 3845795181)
    assert C.parse_bestgems_page(_fx("bestgems_page.html"))[-1] == ("rswag-api", 1000, 52735292)
    raw, src = C.parse_gem_repo(_fx("gem_rails.json"))
    assert C.normalise_repo_url(raw)["url"] == "github.com/rails/rails" and src == "source_code_uri"


def test_fetchers_with_injected_getter():
    def get(url):
        if url == C.NPM_TOP_URL:
            return _fx("npm_top.js")
        if "registry.npmjs.org" in url:
            return _fx("npm_latest_react.json")
        if url == C.PYPI_TOP_URL:
            return _fx("pypi_top.json")
        if "pypi.org" in url:
            return _fx("pypi_requests.json")
        if "crates.io" in url:
            return _fx("crates_page.json")
        if "bestgems" in url:
            return _fx("bestgems_page.html") if url.endswith("page=1") else ""
        return _fx("gem_rails.json")
    npm = list(C.fetch_npm(2, get))
    assert [r["package"] for r in npm] == ["semver", "minimatch"] and npm[0]["repo_url"] == "github.com/react/react"
    assert list(C.fetch_pypi(1, get))[0]["repo_url"] == "github.com/psf/requests"
    cr = list(C.fetch_crates(2, get))
    assert [r["rank"] for r in cr] == [1, 2] and cr[1]["repo_url"] == "github.com/dtolnay/syn"
    assert list(C.fetch_rubygems(1, get))[0]["repo_url"] == "github.com/rails/rails"


def test_vanity_host_is_not_a_repo():
    assert C._row("npm", "x", 1, "m", 1, "https://example.com/a/b", "s")["repo_url"] is None


def test_merge_overlap_and_not_available(tmp_path):
    rows = [
        C._row("npm", "a", 1, "m", 9, "github:o/r", "s"),
        C._row("pypi", "b", 1, "m", 9, "https://github.com/O/R.git", "s"),
        C._row("pypi", "c", 2, "m", 8, None, ""),
    ]
    rf = tmp_path / "rows-x.jsonl"
    rf.write_text("\n".join(json.dumps(r) for r in rows))
    d = C.merge([str(rf)], str(tmp_path / "o.json"), str(tmp_path / "o.csv"), now="2026-01-01T00:00:00Z")
    assert len(d["repos"]) == 1 and set(d["repos"][0]["registries"]) == {"npm", "pypi"}
    assert d["provenance"]["registries"]["pypi"]["resolution_rate"] == 0.5
    assert "not_available" in d["provenance"]["registries"]["go"]
    assert (tmp_path / "o.csv").read_text().startswith("registry,package")
