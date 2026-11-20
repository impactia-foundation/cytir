import hashlib
import json
from pathlib import Path

from cytir.ops.audits import news_article_profile as A


def test_news_policy_semantics_are_profile_owned(tmp_path: Path):
    data = tmp_path / "data"; data.mkdir()
    policy = {
        "policy_id": A.NEWS_POLICY_ID,
        "algorithm": "struct-markdown-adjacent-section-pack-v1",
        "token_hard_max": {"exact": 2560},
    }
    path = data / Path(A.NEWS_POLICY_PATH).name
    path.write_text(json.dumps(policy))
    pack = {"chunk_policy": {"path": A.NEWS_POLICY_PATH,
                             "policy_id": A.NEWS_POLICY_ID,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
    A._validate_news_policy(pack, data / "source-preservation-profiles.v1.json")


def test_news_policy_rejects_wrong_sha(tmp_path: Path):
    data = tmp_path / "data"; data.mkdir()
    path = data / Path(A.NEWS_POLICY_PATH).name
    path.write_text(json.dumps({"policy_id": A.NEWS_POLICY_ID,
                                "algorithm": "struct-markdown-adjacent-section-pack-v1",
                                "token_hard_max": {"exact": 2560}}))
    try:
        A._validate_news_policy({"chunk_policy": {"path": A.NEWS_POLICY_PATH,
            "policy_id": A.NEWS_POLICY_ID, "sha256": "0"*64}},
            data / "source-preservation-profiles.v1.json")
    except ValueError as exc:
        assert "SHA drifted" in str(exc)
    else:
        raise AssertionError("wrong policy SHA accepted")
