from pathlib import Path

from cytir.chunk.policy import load_chunk_policy


def test_news_article_policy_is_strict_loader_compatible():
    path = Path(__file__).resolve().parents[1] / "data" / (
        "chunk-policy-news-article-768w-o0-adjacent-sections-tail-1152-tok2560.v1.json")
    policy = load_chunk_policy(path)
    assert policy.policy_id == "news-article-768w-o0-adjacent-sections-tail-1152-tok2560"
    assert policy.rule_legend.endswith(f"@{policy.epoch}")
