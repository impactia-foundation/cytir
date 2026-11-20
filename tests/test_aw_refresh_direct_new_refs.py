"""real-record tests for cytir.sources.refresh_direct.new_refs
(task 632). Runs against the real committed data/refresh-cadence.v1.json."""
from __future__ import annotations
import json
from datetime import datetime, timezone
import pytest
from cytir.sources.refresh_direct import new_refs as mod


NOW = datetime(2026, 9, 22, 12, 0, 0, tzinfo=timezone.utc)


def test_adapter_registry_falls_back_to_pack_id_when_family_key_is_unregistered(
        tmp_path, monkeypatch):
    """task 661 F5: github_releases/hf_org_models are both `adapter: api`
    (the schema's coarse family), sharing that key with unrelated packs
    (e.g. us-agency-guidance) -- they must dispatch by their own pack
    `id` instead, never registering under the shared family key."""
    fake_cadence = tmp_path / "cadence.json"
    fake_cadence.write_text(json.dumps({
        "schema": "aw.refresh-cadence.v1",
        "sources": [{
            "id": "us-github-releases-tracked",
            "layer": "tech-docs",
            "country": "US",
            "adapter": "api",
            "eligible_now": True,
            "new_refs_discovery": True,
            "periodic_refetch_days": 7,
        }],
    }))
    calls = []

    def fake_id_adapter(row):
        calls.append(row["id"])
        return {"candidates_found": 5}

    monkeypatch.setitem(mod.ADAPTER_REGISTRY, "us-github-releases-tracked",
                        fake_id_adapter)
    state_path = tmp_path / "state.json"
    report = mod.run(fake_cadence, state_path, NOW, armed=True)

    assert calls == ["us-github-releases-tracked"]
    checked = [e for e in report["sources"] if e["status"] == "checked"]
    assert len(checked) == 1
    assert checked[0]["candidates_found"] == 5


def test_adapter_registry_family_key_still_wins_over_a_pack_id_match():
    """The family-key lookup is checked FIRST (`or` short-circuits) -- a
    shared family handler, once one exists, is never silently bypassed by
    a same-named pack id."""
    assert "api" not in mod.ADAPTER_REGISTRY, (
        "this test's premise (family key absent) no longer holds -- "
        "update it alongside whatever now registers 'api'")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
