import json
from pathlib import Path


REGISTRY = Path(__file__).resolve().parents[1] / "data/source-preservation-profiles.v1.json"


def test_long_record_profiles_require_record_local_token_safe_chunks():
    profiles = json.loads(REGISTRY.read_text())["profiles"]
    for profile_id in ("academic_dump", "structured_data_feed"):
        chunks = profiles[profile_id]["chunks"]
        assert chunks["boundary"] == "within-source-record"
        assert "never merge different source records" in chunks["requirements"]
        assert "every embedding text within pinned model token maximum" in chunks[
            "requirements"]
