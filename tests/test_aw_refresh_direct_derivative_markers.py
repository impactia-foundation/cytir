#!/usr/bin/env python3
"""`derivative_tombstones()` emits `takedown` tombstones, which the drain's
stored-row pair check does not cover; it must therefore never target a marker
row (`supersede.is_card`) on its own. Twin 2026-10-04: 2 of 35,128
huggingface.co rows carry a marker (a `docs` page with `entity_kind`, an
`entity_card`), neither a model repo; the filter keeps it that way by
construction rather than by data."""
from __future__ import annotations


import pytest

pytest.importorskip("qdrant_client")

from cytir.sources.refresh_direct import direct_compose as R  # noqa: E402
from qdrant_client.http import models as qm  # noqa: E402

from cytir.core.qdrant import qdrant_client  # noqa: E402

URL = "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard"


def _client(rows):
    c = qdrant_client(":memory:")
    c.create_collection("c", vectors_config={"dense": qm.VectorParams(size=3, distance=qm.Distance.COSINE)})
    c.upsert("c", points=[qm.PointStruct(id=i, vector={"dense": [0.1, 0.2, 0.3]}, payload=p)
                          for i, p in rows])
    return c


def test_marker_rows_are_skipped_and_counted_plain_rows_are_tombstoned():
    c = _client([
        ("00000000-0000-0000-0000-000000000001", {"url": URL, "source_type": "reference"}),
        ("00000000-0000-0000-0000-000000000002",
         {"url": URL, "source_type": "entity_card", "entity_kind": "reference"}),
        ("00000000-0000-0000-0000-000000000003",
         {"url": URL, "source_type": "reference", "qid": "Q1"}),
    ])
    skipped: list = []
    tombs = R.derivative_tombstones(c, "c", [URL], marker_rows=skipped)
    assert [t["id"] for t in tombs] == ["00000000-0000-0000-0000-000000000001"]
    assert all(t["reason"] == "takedown" for t in tombs)
    assert sorted(s["id"] for s in skipped) == [
        "00000000-0000-0000-0000-000000000002", "00000000-0000-0000-0000-000000000003"]


def test_without_a_sink_the_marker_rows_are_still_never_tombstoned():
    c = _client([("00000000-0000-0000-0000-000000000009",
                  {"url": URL, "source_type": "entity_card"})])
    assert R.derivative_tombstones(c, "c", [URL]) == []
