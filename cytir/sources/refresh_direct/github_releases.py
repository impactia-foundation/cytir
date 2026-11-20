#!/usr/bin/env python3
"""GitHub releases discovery adapter for the refresh-cadence new-refs lane
(task 661 F5, 2026-09-26).

The `new_refs.py` docstring's own scope: this DISCOVERS
candidate URLs from a source-owned feed (never scrapes where a feed
exists, per `data/acquisition-policy.v1.json` route precedence) -- it
does not fetch article bodies, embed, or write anywhere. GitHub publishes
a public, unauthenticated Atom feed per repository
(`https://github.com/<owner>/<repo>/releases.atom`), which is exactly the
"source-owned discovery route" the lane's design calls for; no GitHub API
token, no scraping the releases HTML page.

Registered as `ADAPTER_REGISTRY["github_releases"]` in
`new_refs.py` (one line, once the pack itself is admitted --
see `sources/us/github-releases-tracked.yaml`, status `planned`, awaiting
the POLICY PR this task also raises).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
ATOM_NS = "{http://www.w3.org/2005/Atom}"
FEED_URL_TEMPLATE = "https://github.com/{repo}/releases.atom"
GITHUB_HOST_PREFIX = "github.com/"


def repos_from_pack_path(pack_path: str) -> list[str]:
    """Read the tracked-repo list from the pack yaml's own `routes[].host`
    (task 661, PR #383 fix): the pack schema's root `acquire.params` has a
    fixed `documented_source_api` shape (`endpoint` singular) with no room
    for a flat list, so the tracked repos live as one `routes[]` entry
    each, `host: "github.com/<owner>/<repo>"` -- see
    `sources/us/github-releases-tracked.yaml`. `pack_path` is the field
    `cyt source refresh-cadence` already stamps on every
    cadence-table row (repo-root-relative), so this needs no new plumbing
    through `new_refs.py`."""
    import yaml  # noqa: PLC0415 -- only needed on this path, not in unit tests
    doc = yaml.safe_load((REPO_ROOT / pack_path).read_text(encoding="utf-8"))
    repos = []
    for route in (doc or {}).get("routes") or []:
        host = (route or {}).get("host", "")
        if host.startswith(GITHUB_HOST_PREFIX):
            repos.append(host[len(GITHUB_HOST_PREFIX):])
    return repos


def feed_url(repo: str) -> str:
    """`repo` is "owner/name", e.g. "qdrant/qdrant"."""
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError(f"not an 'owner/name' repo slug: {repo!r}")
    return FEED_URL_TEMPLATE.format(repo=repo)


def parse_releases_atom(xml_text: str, repo: str) -> list[dict[str, Any]]:
    """Pure: no I/O. Each Atom `<entry>` is one GitHub release. Returns
    `[{"url", "title", "published", "repo"}, ...]`, newest first (GitHub's
    own feed order). GitHub's real feed (verified live, 2026-09-26, e.g.
    github.com/qdrant/qdrant/releases.atom) carries NO `<published>`
    element per entry, only `<updated>` -- this stamps `published` from
    `<updated>` rather than assuming the generic Atom field a naive read
    of the spec would expect, so a caller never silently gets `None` for
    every real GitHub release."""
    root = ET.fromstring(xml_text)
    out = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        link_el = entry.find(f"{ATOM_NS}link")
        title_el = entry.find(f"{ATOM_NS}title")
        updated_el = entry.find(f"{ATOM_NS}updated")
        url = link_el.get("href") if link_el is not None else None
        if not url:
            continue
        out.append({
            "url": url,
            "title": (title_el.text or "").strip() if title_el is not None else "",
            "published": updated_el.text if updated_el is not None else None,
            "repo": repo,
        })
    return out


def discover(source_row: dict, *, fetch=None) -> dict:
    """The `ADAPTER_REGISTRY` entry point: `callable(source_row) -> dict`.
    `fetch` is injected (defaults to a real `httpx` GET) so tests never
    touch the network -- pass a fake that returns fixture Atom XML.
    The tracked-repo list comes from `source_row["acquire"]["params"]["repos"]`
    when the caller provides it directly (unit tests), else -- the real
    `new_refs.py` path, whose cadence-table rows never carry
    `acquire` -- from `source_row["pack_path"]`'s own `routes[]` (see
    `repos_from_pack_path`). An empty/missing list either way is a
    refusal, never a silent zero-candidate "check"."""
    repos = ((source_row.get("acquire") or {}).get("params") or {}).get("repos") or []
    if not repos and source_row.get("pack_path"):
        repos = repos_from_pack_path(source_row["pack_path"])
    if not repos:
        raise ValueError(
            "github_releases adapter: no tracked repos found in "
            "source_row.acquire.params.repos nor in the pack at "
            f"source_row['pack_path']={source_row.get('pack_path')!r} -- "
            "nothing to track (a real gap, not a quiet day)")

    if fetch is None:
        from cytir.sources.family import HttpSpec  # noqa: PLC0415
        from cytir.sources.http import make_client  # noqa: PLC0415
        def fetch(url: str) -> str:
            with make_client(HttpSpec(timeout=15.0, follow_redirects=True)) as client:
                r = client.get(url)
            r.raise_for_status()
            return r.text

    candidates: list[dict] = []
    errors: dict[str, str] = {}
    for repo in repos:
        url = feed_url(repo)
        try:
            xml_text = fetch(url)
            candidates.extend(parse_releases_atom(xml_text, repo))
        except Exception as exc:  # noqa: BLE001 -- one repo's feed failing
            # must not silently drop the others; it's named, not swallowed.
            errors[repo] = f"{type(exc).__name__}: {exc}"

    return {
        "adapter": "github_releases",
        "repos_checked": len(repos),
        "repos_errored": len(errors),
        "errors": errors,
        "candidates_found": len(candidates),
        "candidates": candidates,
    }
