from pathlib import Path

import cytir.sources.stackoverflow.audit_profile as subject
from cytir.index.seal_content_fidelity_audit import audit_adapter


def test_stackoverflow_audit_has_stable_source_profile_and_no_content_cap():
    assert subject.SOURCE_ID == "stackoverflow-qa-score-cells-v1"
    assert subject.PROFILE == "community_qa"
    assert len(subject.EXPECTED_CELLS) == 4
    assert audit_adapter(Path(subject.__file__)) == []


def test_community_profile_methods_have_one_per_requirement():
    import json
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "data/source-preservation-profiles.v1.json").read_text())[
        "profiles"][subject.PROFILE]
    assert [len(profile.get(stage, {}).get("requirements", []))
            for stage in profile["required_stages"]] == [4, 4, 4, 3, 0]
