"""task 661 F5, 2026-09-26: the GitHub-releases new-refs discovery adapter.

Fixture `tests/data/refresh_direct/github-releases-qdrant-qdrant.atom` is a REAL
feed (fetched live from github.com/qdrant/qdrant/releases.atom,
2026-09-26, 10 entries) -- not a hand-shortened one. GitHub's real feed
carries no `<published>` element (only `<updated>`); a fixture built from
the spec alone, not a real response, would have missed that and shipped
an adapter that silently stamped `published: None` on every real row.
"""
from __future__ import annotations

import pathlib

import pytest

from cytir.sources.refresh_direct import github_releases as G

FIXTURE = pathlib.Path(__file__).parent / "data" / "refresh_direct" / "github-releases-qdrant-qdrant.atom"


def test_feed_url_builds_the_real_atom_endpoint():
    assert G.feed_url("qdrant/qdrant") == "https://github.com/qdrant/qdrant/releases.atom"


def test_feed_url_refuses_a_malformed_repo_slug():
    with pytest.raises(ValueError):
        G.feed_url("not-a-valid-slug")


def test_parse_releases_atom_reads_the_real_feed():
    releases = G.parse_releases_atom(FIXTURE.read_text(), "qdrant/qdrant")
    assert len(releases) == 10
    first = releases[0]
    assert first["repo"] == "qdrant/qdrant"
    assert first["url"] == "https://github.com/qdrant/qdrant/releases/tag/v1.19.1"
    assert first["title"] == "v1.19.1"
    # GitHub's real feed has no <published> -- this must come from <updated>,
    # never silently None.
    assert first["published"] == "2026-09-04T08:42:33Z"


def test_parse_releases_atom_skips_an_entry_with_no_link():
    xml = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><title>no link here</title><updated>2026-01-01T00:00:00Z</updated></entry>
  <entry>
    <title>has a link</title>
    <updated>2026-01-02T00:00:00Z</updated>
    <link rel="alternate" href="https://github.com/x/y/releases/tag/v1"/>
  </entry>
</feed>"""
    releases = G.parse_releases_atom(xml, "x/y")
    assert len(releases) == 1
    assert releases[0]["title"] == "has a link"


def test_discover_refuses_when_no_repos_are_configured():
    with pytest.raises(ValueError):
        G.discover({"acquire": {"params": {"repos": []}}})


def test_discover_uses_the_injected_fetch_never_the_network():
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return FIXTURE.read_text()

    result = G.discover(
        {"acquire": {"params": {"repos": ["qdrant/qdrant", "owner/repo2"]}}},
        fetch=fake_fetch)
    assert calls == ["https://github.com/qdrant/qdrant/releases.atom",
                     "https://github.com/owner/repo2/releases.atom"]
    assert result["repos_checked"] == 2
    assert result["repos_errored"] == 0
    assert result["candidates_found"] == 20  # 10 releases x 2 repos (same fixture)


def test_discover_names_a_failed_repos_error_without_dropping_the_others():
    def flaky_fetch(url):
        if "bad-repo" in url:
            raise RuntimeError("HTTP 404")
        return FIXTURE.read_text()

    result = G.discover(
        {"acquire": {"params": {"repos": ["qdrant/qdrant", "owner/bad-repo"]}}},
        fetch=flaky_fetch)
    assert result["repos_checked"] == 2
    assert result["repos_errored"] == 1
    assert "owner/bad-repo" in result["errors"]
    assert "404" in result["errors"]["owner/bad-repo"]
    assert result["candidates_found"] == 10  # the good repo's releases still counted


def test_repos_from_pack_path_reads_the_real_pack():
    """task 661 PR #383 fix: the pack's root `acquire.params` cannot hold a
    flat repo list (schema-invalid), so the tracked repos moved into
    `routes[].host`. This proves the real committed pack still yields the
    same two-repo list the adapter used to read from `acquire.params.repos`."""
    repos = G.repos_from_pack_path("sources/us/github-releases-tracked.yaml")
    assert repos == ["qdrant/qdrant", "huggingface/text-embeddings-inference"]


def test_discover_falls_back_to_pack_path_when_acquire_is_absent():
    """The real `new_refs.py` path: cadence-table rows never
    carry `acquire` (only `pack_path`) -- this is the exact shape `run()`
    hands to a pack-id-keyed adapter."""
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return FIXTURE.read_text()

    result = G.discover(
        {"id": "us-github-releases-tracked",
         "pack_path": "sources/us/github-releases-tracked.yaml"},
        fetch=fake_fetch)
    assert calls == ["https://github.com/qdrant/qdrant/releases.atom",
                     "https://github.com/huggingface/text-embeddings-inference/releases.atom"]
    assert result["repos_checked"] == 2


def test_discover_refuses_when_pack_path_has_no_tracked_repos(tmp_path):
    empty_pack = tmp_path / "empty.yaml"
    empty_pack.write_text("id: x\nroutes: []\n")
    # an absolute pack_path overrides REPO_ROOT in the `/` join (pathlib
    # semantics), so this works without needing the fixture under REPO_ROOT.
    with pytest.raises(ValueError):
        G.discover({"pack_path": str(empty_pack)})
