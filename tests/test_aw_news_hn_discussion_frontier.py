"""Task 601: HN-discussion frontier discovery (points-only, self-posts kept)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cytir.news import hn_discussion_frontier as subject


class Response:
    def __init__(self, value):
        self.value = value
        self.status_code = 200
        self.text = str(value)

    def json(self):
        return self.value


class OneSliceAlgoliaClient:
    """One un-capped slice covering the requested window."""

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def get(self, _url, *, params, **_kwargs):
        self.calls.append(params)
        return Response({"nbHits": len(self.hits), "nbPages": 1, "hits": self.hits})


def test_admits_self_posts_that_the_linked_page_arm_would_drop():
    """Ask HN (no `url`) is a real discussion candidate; the news arm's
    `qualifying_stories` would silently drop it -- this module must not."""
    client = OneSliceAlgoliaClient([
        {"objectID": "1", "points": 50, "created_at_i": 900, "title": "Ask HN: x"},
        {"objectID": "2", "points": 41, "created_at_i": 800, "url": "https://a.test"},
        {"objectID": "3", "points": 39, "created_at_i": 700, "url": "https://b.test"},
    ])

    item_ids, receipt = subject.discover(
        client, since_days=7, min_points=40, now_i=1000)

    assert item_ids == [1, 2]  # 39 points excluded, self-post (1) admitted
    assert receipt["counts"] == {"stories_in_window": 3, "admitted": 2}
    assert receipt["completeness"] == "time-window-complete"


def test_dedupes_by_object_id_not_by_url():
    """Two DIFFERENT HN threads about the same URL are two distinct
    discussions -- unlike the linked-page arm, row identity here is the
    thread's own item id, never the external URL."""
    client = OneSliceAlgoliaClient([
        {"objectID": "10", "points": 100, "created_at_i": 900, "url": "https://same.test"},
        {"objectID": "11", "points": 100, "created_at_i": 800, "url": "https://same.test"},
    ])
    item_ids, _ = subject.discover(client, since_days=7, min_points=40, now_i=1000)
    assert item_ids == [10, 11]


def test_zero_admitted_stories_refuses_rather_than_writing_an_empty_frontier(tmp_path):
    client = OneSliceAlgoliaClient([
        {"objectID": "1", "points": 5, "created_at_i": 900},
    ])
    item_ids, receipt = subject.discover(client, since_days=7, min_points=40, now_i=1000)
    assert item_ids == []
    with pytest.raises(subject.HNSourceError):
        subject.write_frontier(item_ids, tmp_path / "out.jsonl", receipt,
                               tmp_path / "receipt.json")


def test_write_frontier_is_the_shape_hn_discussion_ingest_reads(tmp_path):
    client = OneSliceAlgoliaClient([
        {"objectID": "5", "points": 50, "created_at_i": 900},
        {"objectID": "3", "points": 50, "created_at_i": 800},
    ])
    item_ids, receipt = subject.discover(client, since_days=7, min_points=40, now_i=1000)
    out, receipt_path = tmp_path / "frontier.jsonl", tmp_path / "receipt.json"

    subject.write_frontier(item_ids, out, receipt, receipt_path)

    from cytir.sources.hn import discussion_ingest as ING
    ids, _ = ING.load_item_ids(out)
    assert ids == [3, 5]
    sealed_receipt = json.loads(receipt_path.read_text())
    assert sealed_receipt["frontier_sha256"] == __import__("hashlib").sha256(
        out.read_bytes()).hexdigest()


def test_rerun_is_deterministic_and_immutability_safe(tmp_path):
    client = OneSliceAlgoliaClient([{"objectID": "7", "points": 50, "created_at_i": 900}])
    item_ids, receipt = subject.discover(client, since_days=7, min_points=40, now_i=1000)
    out, receipt_path = tmp_path / "frontier.jsonl", tmp_path / "receipt.json"
    subject.write_frontier(item_ids, out, receipt, receipt_path)
    first = out.read_bytes()
    # A second discovery run (fresh receipt, same admitted set) must not
    # collide with the immutable artifact it already wrote.
    subject.write_frontier(item_ids, out, receipt, receipt_path)
    assert out.read_bytes() == first
