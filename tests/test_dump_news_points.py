from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import pytest

# Bender intentionally lacks the Heimdall Qdrant dependency. Keep pure
# contract tests runnable there with the tiny model surface used by the tool.
_STANDIN = False
try:
    import qdrant_client  # noqa: F401
except ModuleNotFoundError:
    _STANDIN = True
    models = types.ModuleType("qdrant_client.models")

    class MatchValue:
        def __init__(self, value): self.value = value

    class FieldCondition:
        def __init__(self, key, match): self.key, self.match = key, match

    class Filter:
        def __init__(self, must): self.must = must

    models.MatchValue = MatchValue
    models.FieldCondition = FieldCondition
    models.Filter = Filter
    package = types.ModuleType("qdrant_client")
    package.QdrantClient = object
    package.models = models
    sys.modules["qdrant_client"] = package
    sys.modules["qdrant_client.models"] = models

from cytir.news import dump_points as subject

# Task 1534: the stand-in is captured by dump_points at import; it must not outlive
# this file, or later files see a stub instead of the real package (or ImportError).
if _STANDIN:
    sys.modules.pop("qdrant_client", None)
    sys.modules.pop("qdrant_client.models", None)


class Client:
    def __init__(self, points, schema):
        self.points = points
        self.schema = schema
        self.retrieve_ids = []

    def get_collection(self, collection):
        return SimpleNamespace(payload_schema=self.schema)

    def scroll(self, collection, scroll_filter, **kwargs):
        if scroll_filter is None:
            return self.points, None
        condition = scroll_filter.must[0]
        return ([p for p in self.points
                 if (p.payload or {}).get(condition.key) == condition.match.value], None)

    def retrieve(self, collection, ids, **kwargs):
        self.retrieve_ids.extend(ids)
        wanted = set(ids)
        return [point for point in self.points if point.id in wanted]


def point(point_id, disc_class="news", source_type="reference"):
    return SimpleNamespace(
        id=point_id,
        payload={"disc_class": disc_class, "source_type": source_type,
                 "url": f"https://example/{point_id}", "published": "2026-08-01"})


def registry(tmp_path: Path, points) -> tuple[Path, str]:
    path = tmp_path / "candidates.jsonl"
    path.write_text("".join(json.dumps(
        {"id": p.id, "payload": p.payload}, sort_keys=True) + "\n" for p in points))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_unindexed_path_rereads_registry_and_unions_staged_candidates(tmp_path):
    live_old = point("old")
    deleted = point("deleted")
    newly_staged = point("new")
    path, sha = registry(tmp_path, [live_old, deleted])
    client = Client([live_old], {"source_type": {}})
    current, candidates, receipt = subject.collect_news_points(
        client, "aw_e3", "news", 1, path, sha, [newly_staged])
    assert [row["id"] for row in current] == ["old"]
    assert [row["id"] for row in candidates] == ["deleted", "new", "old"]
    assert client.retrieve_ids == ["deleted", "new", "old"]
    assert receipt["mode"] == "candidate-registry-reread"
    assert receipt["staged_candidates_added"] == 1


def test_indexed_disc_class_uses_direct_current_scan_but_advances_registry(tmp_path):
    path, sha = registry(tmp_path, [point("old")])
    client = Client([point("canonical"), point("gov", "gov")], {"disc_class": {}})
    current, candidates, receipt = subject.collect_news_points(
        client, "aw_e3", "news", 10, path, sha, [point("new")])
    assert [row["id"] for row in current] == ["canonical"]
    assert [row["id"] for row in candidates] == ["new", "old"]
    assert receipt["mode"] == "indexed-disc-class"
    assert client.retrieve_ids == []


def test_registry_hash_and_rows_are_strict(tmp_path):
    path, sha = registry(tmp_path, [point("a")])
    with pytest.raises(subject.DumpError, match="SHA-256 mismatch"):
        subject._load_candidates(path, "0" * 64, "news")
    path.write_text('{"id":"a","payload":{"disc_class":"gov"}}\n')
    bad_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(subject.DumpError, match="is not disc_class=news"):
        subject._load_candidates(path, bad_sha, "news")


def test_manifested_staging_arm_rejects_non_news(tmp_path):
    path, sha = registry(tmp_path, [point("a")])
    client = Client([point("a")], {})
    with pytest.raises(subject.DumpError, match="non-NEWS"):
        subject.collect_news_points(
            client, "aw_e3", "news", 10, path, sha, [point("bad", "gov")])


def test_atomic_outputs_bind_both_hashes(tmp_path):
    current = tmp_path / "current.jsonl"
    candidates = tmp_path / "next.jsonl"
    receipt_out = tmp_path / "receipt.json"
    rows = [{"id": "a", "payload": {"disc_class": "news"}}]
    subject.atomic_write(rows, current, rows, candidates,
                         {"schema": "test"}, receipt_out)
    receipt = json.loads(receipt_out.read_text())
    assert receipt["current_output_sha256"] == hashlib.sha256(current.read_bytes()).hexdigest()
    assert receipt["successor_candidates_sha256"] == hashlib.sha256(candidates.read_bytes()).hexdigest()
