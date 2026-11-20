#!/usr/bin/env python3
"""Hugging Face org model-card discovery adapter for the refresh-cadence
new-refs lane (task 661 F5, 2026-09-26).

Same shape and scope as `cytir/sources/refresh_direct/github_releases.py`: DISCOVERS candidate
model-card URLs from a source-owned, unauthenticated API route (the Hub
API, `https://huggingface.co/api/models?author=<org>&sort=lastModified`)
-- never scrapes the model-card HTML page, per
`data/acquisition-policy.v1.json`'s route precedence. Does not fetch card
bodies, embed, or write anywhere. `tests/data/refresh_direct/hf-models-snowflake.json`
is a REAL response (fetched live 2026-09-26, `author=Snowflake`, 5
results) -- confirmed fields: `id` (the "owner/model-name" slug, no
separate canonical URL in the response -- the model page IS
`https://huggingface.co/<id>`), `lastModified`, `createdAt`, `private`.
No `published` field at all; `lastModified` is the closest freshness
signal and is what this adapter reports as `published`, same convention
`cytir/sources/refresh_direct/github_releases.py` uses when a feed has no dedicated field.

Registered as `ADAPTER_REGISTRY["hf_org_models"]` in
`new_refs.py`, activates once
`sources/us/huggingface-orgs-tracked.yaml` lands as `ingested` (PR #383).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
API_URL_TEMPLATE = (
    "https://huggingface.co/api/models"
    "?author={org}&sort=lastModified&direction=-1&limit={limit}"
)
MODEL_PAGE_TEMPLATE = "https://huggingface.co/{model_id}"


def api_url(org: str, *, limit: int = 100) -> str:
    if not org or "/" in org:
        raise ValueError(f"not a valid HF org/user name: {org!r}")
    return API_URL_TEMPLATE.format(org=org, limit=limit)


def orgs_from_pack_path(pack_path: str) -> list[str]:
    """Read the tracked-org list from the pack yaml's own
    `routes[].acquire.params.query.author` (task 661, PR #383 fix): the
    pack schema's root `acquire.params` has a fixed `documented_source_api`
    shape (`endpoint` singular, `query` a single dict) with no room for a
    flat list, so the tracked orgs live as one `routes[]` entry each,
    each carrying its own `query.author` -- see
    `sources/us/huggingface-orgs-tracked.yaml`. `pack_path` is the field
    `cyt source refresh-cadence` already stamps on every
    cadence-table row (repo-root-relative), so this needs no new plumbing
    through `new_refs.py`."""
    import yaml  # noqa: PLC0415 -- only needed on this path, not in unit tests
    doc = yaml.safe_load((REPO_ROOT / pack_path).read_text(encoding="utf-8"))
    orgs = []
    for route in (doc or {}).get("routes") or []:
        query = ((route or {}).get("acquire") or {}).get("params", {}).get("query") or {}
        author = query.get("author")
        if author:
            orgs.append(author)
    return orgs


def parse_models_response(models: list[dict], org: str) -> list[dict[str, Any]]:
    """Pure: no I/O. `models` is the Hub API's own parsed JSON list.
    Returns `[{"url", "model_id", "published", "org", "private"}, ...]`,
    skipping private models (never a candidate to fetch/index) -- the
    API can return them for the authenticated caller's own org, but this
    adapter runs unauthenticated and treats a `private: true` row as
    evidence the caller should not have received it, not as a real
    public candidate."""
    out = []
    for m in models:
        model_id = m.get("id") or m.get("modelId")
        if not model_id:
            continue
        if m.get("private"):
            continue
        out.append({
            "url": MODEL_PAGE_TEMPLATE.format(model_id=model_id),
            "model_id": model_id,
            "published": m.get("lastModified") or m.get("createdAt"),
            "org": org,
        })
    return out


def discover(source_row: dict, *, fetch=None) -> dict:
    """The `ADAPTER_REGISTRY` entry point: `callable(source_row) -> dict`.
    `fetch` is injected (defaults to a real `httpx` GET returning parsed
    JSON) so tests never touch the network.
    The tracked-org list comes from `source_row["acquire"]["params"]["orgs"]`
    when the caller provides it directly (unit tests), else -- the real
    `new_refs.py` path, whose cadence-table rows never carry
    `acquire` -- from `source_row["pack_path"]`'s own `routes[]` (see
    `orgs_from_pack_path`). An empty/missing list either way is a
    refusal, never a silent zero-candidate "check" (same discipline as
    the GitHub adapter)."""
    orgs = ((source_row.get("acquire") or {}).get("params") or {}).get("orgs") or []
    if not orgs and source_row.get("pack_path"):
        orgs = orgs_from_pack_path(source_row["pack_path"])
    if not orgs:
        raise ValueError(
            "hf_org_models adapter: no tracked orgs found in "
            "source_row.acquire.params.orgs nor in the pack at "
            f"source_row['pack_path']={source_row.get('pack_path')!r} -- "
            "nothing to track (a real gap, not a quiet day)")

    if fetch is None:
        from cytir.sources.family import HttpSpec  # noqa: PLC0415
        from cytir.sources.http import make_client  # noqa: PLC0415
        def fetch(url: str) -> list[dict]:
            with make_client(HttpSpec(timeout=15.0, follow_redirects=True)) as client:
                r = client.get(url)
            r.raise_for_status()
            return r.json()

    candidates: list[dict] = []
    errors: dict[str, str] = {}
    for org in orgs:
        url = api_url(org)
        try:
            models = fetch(url)
            candidates.extend(parse_models_response(models, org))
        except Exception as exc:  # noqa: BLE001 -- one org's API call failing
            # must not silently drop the others; it's named, not swallowed.
            errors[org] = f"{type(exc).__name__}: {exc}"

    return {
        "adapter": "hf_org_models",
        "orgs_checked": len(orgs),
        "orgs_errored": len(errors),
        "errors": errors,
        "candidates_found": len(candidates),
        "candidates": candidates,
    }
