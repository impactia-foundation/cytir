"""The docs-repo and tech-docs renderers of `cytir.acquire.github_docs_tree` (task 1237).

`tests/fixtures/docs_repos/` is a miniature of the drivers' inputs: PyTorch (rst with autodoc, a skipped
community page, an index, a too-short page), TensorFlow (a notebook), Ollama (mdx with a sitemap that drops
one page), react.dev (frontmatter, MDX, JSX) and the Rust Book (mdBook directive, SUMMARY.md). The
`expected_*.jsonl` files are what the legacy `ingest_docs_repo.py` and `ingest_techdocs.py` wrote for
them before they were folded into the module; the rows must stay byte-identical. No network.
"""
from __future__ import annotations

import json
import pathlib

import httpx
import pytest

from cytir.acquire import github_docs_tree as G

FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "docs_repos"


def _want(name):
    return [json.loads(x) for x in (FIX / name).read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("repo,passages,skipped,empty,off_sitemap", [
    ("pytorch", 1, 2, 1, 0), ("tensorflow", 1, 0, 0, 0), ("ollama", 1, 0, 0, 1)])
def test_docs_repo_rows_equal_the_legacy_driver(repo, passages, skipped, empty, off_sitemap):
    rows, stats = G.docs_repo_rows(repo, str(FIX / "frameworks"))
    assert rows == _want(f"expected_docs_repo_{repo}.jsonl")
    assert (stats["passages"], stats["skipped"], stats["empty"], stats["off_sitemap"]) == (
        passages, skipped, empty, off_sitemap)


def test_the_rst_identifiers_survive_for_symbol_queries():
    rows, _ = G.docs_repo_rows("pytorch", str(FIX / "frameworks"))
    assert "torch.autograd.Function" in rows[0]["text"]
    assert rows[0]["url"] == "https://pytorch.org/docs/stable/notes/autograd.html"


@pytest.mark.parametrize("source,n", [("react", 2), ("rust", 1)])
def test_techdocs_rows_equal_the_legacy_driver(source, n):
    rows, stats = G.techdocs_rows(source, str(FIX / "techdocs" / source), last_seen="2026-10-01")
    assert rows == _want(f"expected_techdocs_{source}.jsonl")
    assert stats["passages"] == n
    assert {r["provenance"] for r in rows} == {"tech_gapfill_react_rust"}


def test_verify_urls_splits_200_from_the_rest():
    def handler(req):
        return httpx.Response(200 if req.url.path.endswith("ok") else 404)
    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        ok, bad = G.verify_urls(["https://x.dev/ok", "https://x.dev/gone", "https://x.dev/ok"], c,
                                sleep=lambda s: None)
    assert ok == {"https://x.dev/ok": 200} and bad == {"https://x.dev/gone": 404}
