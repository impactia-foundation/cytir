from __future__ import annotations

import json
from pathlib import Path

import pytest

from cytir.index import seal_source_profile_compliance as subject


REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "data/source-preservation-profiles.v1.json"


def _checks(profile_id: str) -> dict:
    profile = json.loads(REGISTRY.read_text())["profiles"][profile_id]
    return {
        stage: [{"requirement": requirement, "status": "PASS",
                 "verification_method": "fixture evidence"}
                for requirement in profile.get(stage, {}).get("requirements", [])]
        for stage in profile["required_stages"]
    }


def test_seal_validates_all_profile_requirements(tmp_path: Path):
    checks = tmp_path / "checks.json"
    checks.write_text(json.dumps(_checks("academic_dump")))
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"status":"PASS"}\n')
    receipt = tmp_path / "profile/receipt.json"
    envelope = subject.seal(
        artifact_root=tmp_path, registry_path=REGISTRY,
        profile_id="academic_dump", source_id="pubmed",
        checks_path=checks, evidence_path=evidence, receipt_path=receipt)
    assert envelope["receipt"]["uri"] == "artifact://profile/receipt.json"


def test_missing_requirement_is_rejected(tmp_path: Path):
    value = _checks("anchored_law")
    value["markdown"].pop()
    checks = tmp_path / "checks.json"
    checks.write_text(json.dumps(value))
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"status":"PASS"}\n')
    with pytest.raises(ValueError, match="preserve every profile requirement"):
        subject.seal(
            artifact_root=tmp_path, registry_path=REGISTRY,
            profile_id="anchored_law", source_id="law",
            checks_path=checks, evidence_path=evidence,
            receipt_path=tmp_path / "receipt.json")
