"""task 661 F5, 2026-09-26: the HF org model-cards new-refs discovery
adapter.

Fixture `tests/data/refresh_direct/hf-models-snowflake.json` is a REAL response
(fetched live from huggingface.co/api/models?author=Snowflake,
2026-09-26, 5 results) -- not hand-built. The real response has no
`published` field at all; a fixture invented from guessing the schema
would have missed that.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from cytir.sources.refresh_direct import hf_org_models as H

FIXTURE = pathlib.Path(__file__).parent / "data" / "refresh_direct" / "hf-models-snowflake.json"


def _load_fixture():
    return json.loads(FIXTURE.read_text())


def test_api_url_builds_the_real_hub_endpoint():
    assert H.api_url("Snowflake") == (
        "https://huggingface.co/api/models"
        "?author=Snowflake&sort=lastModified&direction=-1&limit=100")


def test_api_url_refuses_a_malformed_org():
    with pytest.raises(ValueError):
        H.api_url("not/a/valid/org")
    with pytest.raises(ValueError):
        H.api_url("")


def test_parse_models_response_reads_the_real_fixture():
    models = _load_fixture()
    candidates = H.parse_models_response(models, "Snowflake")
    assert len(candidates) == 5
    first = candidates[0]
    assert first["model_id"] == "Snowflake/Arctic-LSTM-Speculator-Qwen3-32B-longcontext"
    assert first["url"] == (
        "https://huggingface.co/Snowflake/Arctic-LSTM-Speculator-Qwen3-32B-longcontext")
    assert first["org"] == "Snowflake"
    # the real response has NO "published" field -- must fall back to
    # lastModified, never silently None.
    assert first["published"] == "2026-06-30T01:31:30.000Z"


def test_parse_models_response_skips_private_models():
    models = [
        {"id": "org/public-model", "lastModified": "2026-01-01T00:00:00.000Z"},
        {"id": "org/private-model", "lastModified": "2026-01-02T00:00:00.000Z",
         "private": True},
    ]
    candidates = H.parse_models_response(models, "org")
    assert len(candidates) == 1
    assert candidates[0]["model_id"] == "org/public-model"


def test_parse_models_response_falls_back_to_created_at_when_no_last_modified():
    models = [{"id": "org/model", "createdAt": "2026-03-01T00:00:00.000Z"}]
    candidates = H.parse_models_response(models, "org")
    assert candidates[0]["published"] == "2026-03-01T00:00:00.000Z"


def test_discover_refuses_when_no_orgs_are_configured():
    with pytest.raises(ValueError):
        H.discover({"acquire": {"params": {"orgs": []}}})


def test_discover_uses_the_injected_fetch_never_the_network():
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return _load_fixture()

    result = H.discover(
        {"acquire": {"params": {"orgs": ["Snowflake", "BAAI"]}}}, fetch=fake_fetch)
    assert calls == [H.api_url("Snowflake"), H.api_url("BAAI")]
    assert result["orgs_checked"] == 2
    assert result["orgs_errored"] == 0
    assert result["candidates_found"] == 10  # 5 models x 2 orgs (same fixture)


def test_discover_names_a_failed_orgs_error_without_dropping_the_others():
    def flaky_fetch(url):
        if "BadOrg" in url:
            raise RuntimeError("HTTP 429")
        return _load_fixture()

    result = H.discover(
        {"acquire": {"params": {"orgs": ["Snowflake", "BadOrg"]}}}, fetch=flaky_fetch)
    assert result["orgs_checked"] == 2
    assert result["orgs_errored"] == 1
    assert "BadOrg" in result["errors"]
    assert "429" in result["errors"]["BadOrg"]
    assert result["candidates_found"] == 5  # the good org's models still counted


def test_orgs_from_pack_path_reads_the_real_pack():
    """task 661 PR #383 fix: the pack's root `acquire.params` cannot hold a
    flat org list (schema-invalid), so the tracked orgs moved into
    `routes[].acquire.params.query.author`. This proves the real
    committed pack still yields the same four-org list the adapter used
    to read from `acquire.params.orgs`. Task 693 widened the list to the
    major open-weight publishers; the original four stay, in order, first,
    and zai-org (the GLM-5.3 trigger) is tracked."""
    orgs = H.orgs_from_pack_path("sources/us/huggingface-orgs-tracked.yaml")
    assert orgs[:4] == ["Snowflake", "BAAI", "Qwen", "qdrant"]
    assert "zai-org" in orgs and len(orgs) == len(set(orgs))


def test_discover_falls_back_to_pack_path_when_acquire_is_absent():
    """The real `new_refs.py` path: cadence-table rows never
    carry `acquire` (only `pack_path`) -- this is the exact shape `run()`
    hands to a pack-id-keyed adapter."""
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return _load_fixture()

    result = H.discover(
        {"id": "us-huggingface-orgs-tracked",
         "pack_path": "sources/us/huggingface-orgs-tracked.yaml"},
        fetch=fake_fetch)
    assert calls[:4] == [H.api_url("Snowflake"), H.api_url("BAAI"),
                         H.api_url("Qwen"), H.api_url("qdrant")]
    assert H.api_url("zai-org") in calls
    assert result["orgs_checked"] == len(calls)


def test_discover_refuses_when_pack_path_has_no_tracked_orgs(tmp_path):
    empty_pack = tmp_path / "empty.yaml"
    empty_pack.write_text("id: x\nroutes: []\n")
    with pytest.raises(ValueError):
        H.discover({"pack_path": str(empty_pack)})
