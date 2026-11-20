"""Executable card route: event waves -> parquet -> recipe -> stage -> rebuild."""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
from collections import defaultdict
from types import SimpleNamespace

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("qdrant_client")

import cytir.core.projection as projection
from cytir.index import b1_slice as rebuild  # noqa: F401
from cytir.index import b1_projection as BP
import cytir.corpus.card_recipe as recipe
import cytir.corpus.entity_card_website as website
from cytir.corpus import facet_bundle as facets
from cytir.index import stage_pack_b1 as stage


APP = pathlib.Path(__file__).resolve().parents[1]
ENTITY = "CHE341859062"


def _event(value, source, *, facet="prose", source_id=None, license="CC-BY-4.0"):
    return facets.make_event(
        entity=ENTITY, facet=facet, value=value, lang="en", source=source,
        evidence=value, source_url="https://source.example/",
        source_id=source_id or source, effective_at="2026-08-01",
        retrieved_at="2026-08-01", extractor_version="initial-v1",
        license=license, consent="search_ok", attribution=source, task="initial",
        selection={"depth": 0, "rule": "canonical homepage"}
        if source == "homepage" else None)


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "cytir.cli", "corpus", *map(str, args)], cwd=APP, check=True,
        text=True, capture_output=True)


def test_real_card_event_route_and_full_rebuild_sparse_regeneration(tmp_path):
    # task 530: the dry-run subprocess builds a website card through
    # entity_card_website.py, which refuses to approximate the token budget
    # when the pinned Arctic tokenizer is unavailable (fail loud) — the
    # subprocess shares this interpreter's environment, so skip here too.
    pytest.importorskip("transformers")
    events_root = tmp_path / "facet_events"
    first_wave = events_root / "000-initial.jsonl"
    homepage_v1 = _event(
        "Original generic homepage statement.", "homepage",
        source_id="https://www.vector8.example/en")
    register = _event("Registered software and technology consulting purpose.",
                      "register_field", source_id=ENTITY, license="OGD")
    person = _event("Marc Maret (President of the board)", "register_field",
                    facet="person", source_id=ENTITY, license="OGD")
    alias = _event("Vector Eight", "wikidata_facet", facet="alias",
                   source_id="Q1", license="CC0")
    facets.publish_event_wave(
        [homepage_v1, register, person, alias], first_wave, task="initial")

    current_v1 = tmp_path / "current-v1.parquet"
    _run("facet-compact", "--events-root", events_root,
         "--out", current_v1)
    assert {row["event_id"] for row in facets.read_parquet(current_v1)} == {
        homepage_v1["event_id"], register["event_id"], person["event_id"],
        alias["event_id"],
    }

    homepage_v2_text = (
        "Expert AI for the Enterprise. AI strategy, build and integration services.")
    receipt = {
        "schema": recipe.WEBSITE_RECEIPT_SCHEMA,
        "source_url": "https://www.vector8.example/en",
        "status": "ok", "search_ok": True, "train_ok": False,
        "consent_mode": "direct",
        "license": "All rights reserved", "attribution": "Vector8 AG",
        "checked_at": "2026-08-02T09:00:00Z",
        "fetched_at": "2026-08-03T10:00:00Z",
        "content_sha256": hashlib.sha256(homepage_v2_text.encode()).hexdigest(),
        "raw_sha256": "a" * 64,
        "raw_artifact_pointer": "master://website/vector8.html",
        "association_receipt": {
            "schema": website.ASSOCIATION_RECEIPT_SCHEMA,
            "entity": "CHE-341.859.062", "domain": "vector8.example",
            "status": "admitted", "evidence_class": "uid_on_page",
            "source": "zefix_ccjoin",
            "observed_at": "2026-08-01T09:00:00Z",
            "evidence_sha256": "e" * 64,
            "source_artifact_sha256": "d" * 64,
            "artifact_pointer": "master://associations/vector8.json",
            "resolve_confidence": 1.0,
            "resolve_method": "fixture_uid_on_page",
        },
        "license_evidence": {
            "schema": website.LICENSE_EVIDENCE_SCHEMA,
            "basis": "default-copyright",
            "source_url": "https://www.vector8.example/en",
            "checked_at": "2026-08-01T10:00:00Z",
        },
        "acquisition_receipt": {
            "schema": website.ACQUISITION_RECEIPT_SCHEMA,
            "route": "direct-crawl-fallback",
            "url": "https://www.vector8.example/en",
            "retrieved_at": "2026-08-03T10:00:01Z",
            "fetch_time": "2026-08-03T10:00:00Z",
            "user_agent": "authoritative-web/1.0",
            "robots_checked_at": "2026-08-02T09:00:00Z",
        },
    }
    source_input = tmp_path / "refresh-input.jsonl"
    source_input.write_text(json.dumps({
        "seed": {
            "name": "Vector8 AG", "domain": "vector8.example",
            "url": "https://www.vector8.example/en", "country": "CH",
            "lang": "en", "kind": "org", "cats": [],
            "register_id": "CHE-341.859.062", "register_scheme": "CHE-UID",
            "provenance": "zefix_ccjoin", "effective_at": "2026-08-03",
            "retrieved_at": "2026-08-03", "website_content": homepage_v2_text,
            "website_receipt": receipt,
        },
        "zweck": None, "entity": None, "qid": None,
    }) + "\n", encoding="utf-8")
    cards_out = tmp_path / "cards.jsonl"
    second_wave = events_root / "001-homepage-refresh.jsonl"
    _run("card-content-dry-run",
         "--in", source_input, "--out", cards_out, "--events-out", second_wave,
         "--current-parquet", current_v1, "--events-root", events_root,
         "--phase", "route-test")
    card = json.loads(cards_out.read_text(encoding="utf-8"))
    changes = facets.load_event_wave(second_wave)
    retract = next(row for row in changes if row["op"] == "retract")
    replacement = next(row for row in changes
                       if row["op"] == "assert" and row["source"] == "homepage")
    assert retract["tombstone_of"] == homepage_v1["event_id"]
    assert replacement["supersedes"] == homepage_v1["event_id"]
    assert replacement["source_receipt"] == receipt
    assert replacement["artifact_pointer"] == receipt["raw_artifact_pointer"]
    assert homepage_v2_text in card["dense_text"]
    assert "Registered software" in card["dense_text"]
    assert "Marc Maret" in card["dense_text"]
    assert "Vector Eight" in card["sparse_text"]
    assert "Original generic" not in card["dense_text"]

    current_v2 = tmp_path / "current-v2.parquet"
    _run("facet-compact", "--events-root", events_root,
         "--out", current_v2)
    current_rows = facets.read_parquet(current_v2)
    values = {row["value"] for row in current_rows}
    assert homepage_v1["value"] not in values
    assert {register["value"], person["value"], alias["value"], homepage_v2_text} <= values

    args = SimpleNamespace(attribution=None, license_version=None)
    consent = {
        "consent_state": "dump", "consent_basis": "route-test",
        "consent_source": "fixture", "consent_checked_at": "2026-08-03",
    }
    staged = stage.project(card, args, consent)
    assert staged is not None
    _pid, _dense_surface, sparse_surface, serving_payload = staged
    assert serving_payload["bundle_digest"] == card["bundle_digest"]
    assert not ({"text", "dense_text", "sparse_text", "register_people",
                 "website_content"} & set(serving_payload))

    stale_source_payload = dict(serving_payload, bundle_digest="b" * 64)
    source = SimpleNamespace(
        id="old-card", vector={"dense": [0.1] * 128,
                               "bm25": {"indices": [999], "values": [9.0]}},
        payload=stale_source_payload)
    corpus_row = {
        "texts": [card["text"]], "dense_text": card["dense_text"],
        "sparse_text": card["sparse_text"], "bundle_digest": card["bundle_digest"],
    }
    gate = SimpleNamespace(
        verdict=lambda _host, _row, _payload=None: (True, "ok"))
    point, meta = BP.project_record(
        source, "base", corpus_row, {}, {}, gate, defaultdict(int))
    assert meta == {} and point is not None
    sparse = point["vector"]["bm25"]
    expected_indices, expected_values = projection.saturated_bm25(sparse_surface, "en")
    assert list(sparse.indices) == expected_indices
    assert list(sparse.values) == pytest.approx(expected_values)
    assert 999 not in sparse.indices
    assert point["tok_epoch"] == 2
    assert point["payload"]["rule_epochs"]["tok"] == 2
    assert point["payload"]["bundle_digest"] == card["bundle_digest"]
    assert point["dense_reembed_text"] is not None


def test_full_rebuild_refuses_card_without_reconstructable_text():
    source = SimpleNamespace(
        id="old-card", vector={"dense": [0.1] * 128,
                               "bm25": {"indices": [999], "values": [9.0]}},
        payload={
            "url": "https://example.test/", "title": "Example AG", "lang": "en",
            "domain": "example.test", "source_type": "entity_card",
            "license": "CC0-1.0", "bundle_digest": "a" * 64,
        })
    gate = SimpleNamespace(
        verdict=lambda _host, _row, _payload=None: (True, "ok"))
    with pytest.raises(ValueError, match="lacks corpus text"):
        BP.project_record(
            source, "base", None, {}, {}, gate, defaultdict(int))


def test_full_rebuild_refuses_card_channels_without_bundle_lineage():
    source = SimpleNamespace(
        id="old-card", vector={"dense": [0.1] * 128}, payload={
            "url": "https://example.test/", "title": "Example AG", "lang": "en",
            "domain": "example.test", "source_type": "entity_card",
            "license": "CC0-1.0", "bundle_digest": "a" * 64,
        })
    gate = SimpleNamespace(
        verdict=lambda _host, _row, _payload=None: (True, "ok"))
    with pytest.raises(ValueError, match="channels lack bundle_digest"):
        BP.project_record(
            source, "base", {"texts": ["new"], "dense_text": "new",
                             "sparse_text": "new"},
            {}, {}, gate, defaultdict(int))


def test_full_rebuild_refuses_card_with_only_an_ordinary_corpus_body():
    source = SimpleNamespace(
        id="old-card", vector={"dense": [0.1] * 128}, payload={
            "url": "https://example.test/", "title": "Example AG", "lang": "en",
            "domain": "example.test", "source_type": "entity_card",
            "license": "CC0-1.0", "bundle_digest": "a" * 64,
        })
    gate = SimpleNamespace(
        verdict=lambda _host, _row, _payload=None: (True, "ok"))
    with pytest.raises(ValueError, match="canonical paired corpus channels"):
        BP.project_record(
            source, "base", {"texts": ["ordinary body only"]},
            {}, {}, gate, defaultdict(int))
