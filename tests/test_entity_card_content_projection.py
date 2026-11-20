"""Contract tests for source-neutral entity-card content projection."""
from __future__ import annotations

import hashlib
import os
import sys

import pytest


import cytir.core.projection as projection  # noqa: E402
import cytir.corpus.card_recipe as recipe  # noqa: E402
import cytir.corpus.entity_card_website as website  # noqa: E402
from cytir.corpus import facet_bundle as fb  # noqa: E402


def _association(*, entity="CHE-341.859.062") -> dict:
    return {
        "schema": website.ASSOCIATION_RECEIPT_SCHEMA,
        "entity": entity, "domain": "vector8.example",
        "status": "admitted", "evidence_class": "uid_on_page",
        "source": "zefix_ccjoin", "observed_at": "2026-08-01T09:00:00Z",
        "evidence_sha256": "e" * 64,
        "source_artifact_sha256": "d" * 64,
        "artifact_pointer": "master://associations/vector8.json",
        "resolve_confidence": 1.0, "resolve_method": "fixture_uid_on_page",
    }


def _license_evidence(source_url: str, license: str) -> dict:
    restrictive = website.license_is_restrictive(license)
    evidence = {
        "schema": website.LICENSE_EVIDENCE_SCHEMA,
        "basis": "default-copyright" if restrictive else "reviewed-license",
        "source_url": source_url, "checked_at": "2026-08-01T10:00:00Z",
    }
    if not restrictive:
        evidence.update(evidence_sha256="f" * 64,
                        artifact_pointer="master://licenses/vector8.html")
    return evidence


def _receipt(text: str, *, source_url="https://www.vector8.example/en",
             license="CC-BY-4.0", train_ok=True, entity="CHE-341.859.062") -> dict:
    return {
        "schema": recipe.WEBSITE_RECEIPT_SCHEMA,
        "source_url": source_url, "status": "ok", "search_ok": True,
        "train_ok": train_ok, "consent_mode": "direct", "license": license,
        "attribution": "Vector8 AG",
        "checked_at": "2026-08-02T09:00:00Z",
        "fetched_at": "2026-08-03T10:00:00Z",
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "raw_sha256": "a" * 64,
        "raw_artifact_pointer": "master://website/vector8.html",
        "association_receipt": _association(entity=entity),
        "license_evidence": _license_evidence(source_url, license),
        "acquisition_receipt": {
            "schema": website.ACQUISITION_RECEIPT_SCHEMA,
            "route": "direct-crawl-fallback", "url": source_url,
            "retrieved_at": "2026-08-03T10:00:01Z",
            "fetch_time": "2026-08-03T10:00:00Z",
            "user_agent": "authoritative-web/1.0",
            "robots_checked_at": "2026-08-02T09:00:00Z",
        },
    }


def _seed(**extra) -> dict:
    return {
        "name": "Vector8 AG", "domain": "vector8.example",
        "url": "https://www.vector8.example/en", "country": "CH",
        "lang": "en", "kind": "org", "cats": [],
        "register_id": "CHE-341.859.062", "register_scheme": "CHE-UID",
        "provenance": "zefix_ccjoin", "effective_at": "2026-08-03",
        "retrieved_at": "2026-08-03", **extra,
    }


def _source(value: str, source: str, *, facet="prose", source_id=None,
            date="2026-08-01", license="CC-BY-4.0") -> dict:
    return fb.make_event(
        entity="CHE341859062", facet=facet, value=value, lang="en", source=source,
        evidence=value, source_url="https://source.example/",
        source_id=source_id or source, effective_at=date, retrieved_at=date,
        extractor_version="test-v1", license=license, consent="search_ok",
        attribution=source, task="test")


def test_url_only_zefix_ccjoin_does_not_claim_website_body():
    # task 530: build_two_channel's website channel refuses to
    # approximate the token budget when the pinned Arctic tokenizer is
    # unavailable (fail loud, entity_card_website.py:_website_tokenizer)
    # -- deterministic without `transformers` installed / network access.
    pytest.importorskip("transformers")
    rec, audit = recipe.build_two_channel(
        _seed(), None, ent=None, qid=None,
        zweck="Software development and technology consulting.", phase="test")
    assert audit["description_source"] == "register_field"
    assert "website_content" not in rec
    assert "Software" in rec["dense_text"]


def test_consented_website_and_register_text_share_universal_dense_sparse_projection():
    # task 530: build_two_channel's website channel refuses to
    # approximate the token budget when the pinned Arctic tokenizer is
    # unavailable (fail loud, entity_card_website.py:_website_tokenizer)
    # -- deterministic without `transformers` installed / network access.
    pytest.importorskip("transformers")
    body = "Expert AI for the Enterprise. AI strategy, build and integration services."
    rec, audit = recipe.build_two_channel(
        _seed(website_content=body, website_receipt=_receipt(body)), None,
        ent=None, qid=None,
        zweck="Software development and technology consulting.", phase="test")
    assert audit["description_source"] == "homepage+register_field"
    assert "AI strategy" in rec["dense_text"]
    assert "Software development" in rec["dense_text"]
    assert {"strategy", "enterprise", "technology"} <= set(rec["sparse_text"].lower().split())
    assert len(rec["bundle_digest"]) == 64
    assert any(event["source"] == "homepage" for event in audit["facet_events"])


def test_all_rights_reserved_is_search_indexed_but_never_training_admitted():
    # task 530: build_two_channel's website channel refuses to
    # approximate the token budget when the pinned Arctic tokenizer is
    # unavailable (fail loud, entity_card_website.py:_website_tokenizer)
    # -- deterministic without `transformers` installed / network access.
    pytest.importorskip("transformers")
    body = "Expert AI for the Enterprise."
    receipt = _receipt(body, license="All rights reserved", train_ok=False)
    rec, audit = recipe.build_two_channel(
        _seed(website_content=body, website_receipt=receipt), None,
        ent=None, qid=None, zweck="Software consulting.", phase="test")
    assert body in rec["dense_text"] and "Enterprise" in rec["sparse_text"]
    assert rec["search_ok"] is True and rec["train_ok"] is False
    assert rec["consent_checked_at"] == receipt["checked_at"]
    assert "Composite source licenses/statuses:" in rec["license"]
    homepage = next(row for row in audit["current_bundle"]
                    if row["source"] == "homepage")
    assert homepage["license"] == "All rights reserved"
    assert homepage["source_receipt"]["train_ok"] is False
    _point_id, _dense, _sparse, payload = projection.project_record(rec)
    assert payload["search_ok"] is True and payload["train_ok"] is False
    assert "source_receipt" not in payload and "website_content" not in payload


def test_restrictive_status_cannot_claim_training_permission():
    # task 530: build_two_channel's website channel refuses to
    # approximate the token budget when the pinned Arctic tokenizer is
    # unavailable (fail loud, entity_card_website.py:_website_tokenizer)
    # -- deterministic without `transformers` installed / network access.
    pytest.importorskip("transformers")
    body = "Expert AI for the Enterprise."
    receipt = _receipt(body, license="All rights reserved", train_ok=True)
    with pytest.raises(ValueError, match="train_ok=false"):
        recipe.build_two_channel(
            _seed(website_content=body, website_receipt=receipt), None,
            ent=None, qid=None, zweck="Software consulting.", phase="test")


def test_unknown_license_and_association_or_hash_drift_fail_closed():
    # task 530: build_two_channel's website channel refuses to
    # approximate the token budget when the pinned Arctic tokenizer is
    # unavailable (fail loud, entity_card_website.py:_website_tokenizer)
    # -- deterministic without `transformers` installed / network access.
    pytest.importorskip("transformers")
    body = "Expert AI for the Enterprise."
    unknown = _receipt(body)
    unknown.update(license="unknown", license_evidence={
        "schema": website.LICENSE_EVIDENCE_SCHEMA,
        "basis": "default-copyright", "source_url": unknown["source_url"],
        "checked_at": "2026-08-01T10:00:00Z",
    })
    with pytest.raises(ValueError, match="identified license"):
        recipe.admitted_website_content(
            _seed(website_content=body, website_receipt=unknown))
    wrong_entity = _receipt(body, entity="CHE-000.000.000")
    with pytest.raises(ValueError, match="card entity"):
        recipe.admitted_website_content(
            _seed(website_content=body, website_receipt=wrong_entity))
    nonhex = {**_receipt(body), "raw_sha256": "z" * 64}
    with pytest.raises(ValueError, match="SHA-256"):
        recipe.admitted_website_content(
            _seed(website_content=body, website_receipt=nonhex))


def test_partial_homepage_refresh_preserves_register_and_wikidata_assertions():
    # task 530: build_two_channel's website channel refuses to
    # approximate the token budget when the pinned Arctic tokenizer is
    # unavailable (fail loud, entity_card_website.py:_website_tokenizer)
    # -- deterministic without `transformers` installed / network access.
    pytest.importorskip("transformers")
    homepage = _source("Old homepage", "homepage", source_id="https://www.vector8.example/en")
    register = _source("Registered software purpose", "register_field",
                       source_id="CHE341859062", license="OGD")
    alias = _source("Vector Eight", "wikidata_facet", facet="alias", source_id="Q1",
                    license="CC0")
    body = "New homepage AI consulting statement."
    rec, audit = recipe.build_two_channel(
        _seed(website_content=body, website_receipt=_receipt(body),
              effective_at="2026-08-03"), None, ent=None, qid=None,
        zweck=None, phase="refresh", current_bundle=[homepage, register, alias])
    assert "Registered software purpose" in rec["dense_text"]
    assert "Vector Eight" in rec["sparse_text"]
    assert "New homepage" in rec["dense_text"] and "Old homepage" not in rec["dense_text"]
    retracts = [row for row in audit["facet_events"] if row["op"] == "retract"]
    assert [row["tombstone_of"] for row in retracts] == [homepage["event_id"]]
    assert {row["source"] for row in audit["facet_events"]} == {"homepage"}
    assert register in audit["current_bundle"] and alias in audit["current_bundle"]


def test_website_retraction_keeps_immutable_receipt_and_raw_tombstone_pointer():
    # task 530: build_two_channel's website channel refuses to
    # approximate the token budget when the pinned Arctic tokenizer is
    # unavailable (fail loud, entity_card_website.py:_website_tokenizer)
    # -- deterministic without `transformers` installed / network access.
    pytest.importorskip("transformers")
    old_text = "Old website statement"
    old_receipt = _receipt(old_text)
    old = fb.make_event(
        entity="CHE341859062", facet="prose", value=old_text, lang="en",
        source="homepage", evidence=old_text,
        source_url=old_receipt["source_url"], source_id=old_receipt["source_url"],
        effective_at="2026-08-01T00:00:00Z",
        retrieved_at=old_receipt["fetched_at"], extractor_version="website-v1",
        license=old_receipt["license"], consent="resolved-search-index",
        attribution=old_receipt["attribution"], task="old",
        artifact_pointer=old_receipt["raw_artifact_pointer"],
        selection={"depth": 0, "rule": "canonical homepage"},
        source_receipt=old_receipt)
    rec, audit = recipe.build_two_channel(
        _seed(refreshed_sources=[["homepage", "prose"]]), None,
        ent=None, qid=None, zweck=None, phase="remove", current_bundle=[old])
    retract = next(row for row in audit["facet_events"] if row["op"] == "retract")
    assert retract["tombstone_of"] == old["event_id"]
    assert retract["artifact_pointer"] == old_receipt["raw_artifact_pointer"]
    assert retract["source_receipt"] == old_receipt
    assert old_text not in rec["dense_text"]


def test_explicit_empty_homepage_refresh_emits_tombstone_and_template_fallback():
    # task 530: build_two_channel's website channel refuses to
    # approximate the token budget when the pinned Arctic tokenizer is
    # unavailable (fail loud, entity_card_website.py:_website_tokenizer)
    # -- deterministic without `transformers` installed / network access.
    pytest.importorskip("transformers")
    homepage = _source("Old homepage", "homepage",
                       source_id="https://www.vector8.example/en")
    rec, audit = recipe.build_two_channel(
        _seed(refreshed_sources=[["homepage", "prose"]]), None,
        ent=None, qid=None, zweck=None, phase="remove", current_bundle=[homepage])
    retract = next(row for row in audit["facet_events"] if row["op"] == "retract")
    assert retract["tombstone_of"] == homepage["event_id"]
    assert "Old homepage" not in rec["dense_text"]
    assert any(row["source"] == "aw_template" for row in audit["current_bundle"])


def test_register_people_are_eventized_content_not_a_payload_contract():
    # task 530: build_two_channel's website channel refuses to
    # approximate the token budget when the pinned Arctic tokenizer is
    # unavailable (fail loud, entity_card_website.py:_website_tokenizer)
    # -- deterministic without `transformers` installed / network access.
    pytest.importorskip("transformers")
    rec, audit = recipe.build_two_channel(
        _seed(register_people={"marc|maret": "Marc Maret (President of the board)"}),
        None, ent=None, qid=None, zweck="Software consulting.", phase="test")
    assert "Marc Maret" in rec["dense_text"]
    assert "president" in rec["sparse_text"].lower()
    assert any(row["facet"] == "person" for row in audit["current_bundle"])
    _pid, _dense, _sparse, payload = projection.project_record(rec)
    assert "register_people" not in payload
    assert payload["bundle_digest"] == rec["bundle_digest"]


def test_event_segment_bm25_prevents_unrelated_content_from_diluting_person_name():
    # task 530: build_two_channel's website channel refuses to
    # approximate the token budget when the pinned Arctic tokenizer is
    # unavailable (fail loud, entity_card_website.py:_website_tokenizer)
    # -- deterministic without `transformers` installed / network access.
    pytest.importorskip("transformers")
    person = "Marc Maret (President of the board)"
    long_purpose = " ".join(
        f"enterprise software consulting integration service capability{i}"
        for i in range(35))
    rec, _audit = recipe.build_two_channel(
        _seed(register_people={"marc|maret": person}), None,
        ent=None, qid=None, zweck=long_purpose, phase="test")
    _pid, _dense, sparse_surface, _payload = projection.project_record(rec)
    assert projection.SPARSE_SEGMENT_SEPARATOR in sparse_surface

    segmented_indices, segmented_values = projection.saturated_bm25(
        sparse_surface, "en")
    segmented = dict(zip(segmented_indices, segmented_values))
    # The previous card path flattened only hooks + complete dense content;
    # evidence assertions were not repeated as independent surfaces.
    legacy_surface = recipe.drop_stopwords(" ".join(
        sparse_surface.split(projection.SPARSE_SEGMENT_SEPARATOR)[:2]))
    flattened_indices, flattened_values = projection.saturated_bm25(
        legacy_surface, "en")
    flattened = dict(zip(flattened_indices, flattened_values))
    marc_hash = next(iter(projection.AC.bm25_tokens("Marc", "en")))

    # The complete logical chunk is still indexed, but the exact register
    # assertion keeps local length normalization instead of inheriting the
    # length of an unrelated purpose paragraph.
    assert set(flattened) <= set(segmented)
    assert segmented[marc_hash] > flattened[marc_hash]
    assert segmented[marc_hash] > 1.6


def test_l1_requires_complete_receipt_and_exact_bounded_token_count():
    # task 530: build_two_channel's website channel refuses to
    # approximate the token budget when the pinned Arctic tokenizer is
    # unavailable (fail loud, entity_card_website.py:_website_tokenizer)
    # -- deterministic without `transformers` installed / network access.
    pytest.importorskip("transformers")
    text = "Enterprise AI services"
    page_receipt = {
        **_receipt(text, source_url="https://www.vector8.example/services"),
        "depth": 1, "selection_rule": "services", "token_count": 3,
    }
    rec, audit = recipe.build_two_channel(
        _seed(l1_pages=[{"content": text, "receipt": page_receipt}]), None,
        ent=None, qid=None, zweck="Software consulting.", phase="test")
    assert text in rec["dense_text"]
    l1 = next(row for row in audit["current_bundle"] if row["source"] == "homepage_l1")
    assert l1["selection"] == {"depth": 1, "rule": "services", "token_count": 3}
    bad = dict(page_receipt, token_count=2)
    with pytest.raises(ValueError, match="token_count"):
        recipe.build_two_channel(
            _seed(l1_pages=[{"content": text, "receipt": bad}]), None,
            ent=None, qid=None, zweck="Software consulting.", phase="test")


def test_projection_strips_content_and_keeps_only_bundle_digest():
    body = "Expert AI for the Enterprise."
    record = {
        "url": "https://www.vector8.example/en", "text": body,
        "dense_text": body, "sparse_text": body,
        "title": "Vector8 AG", "lang": "en", "domain": "vector8.example",
        "source_type": "entity_card", "country": "CH", "license": "CC-BY-4.0",
        "register_people": ["Marc Maret"], "website_content": body,
        "bundle_digest": "a" * 64,
    }
    _pid, _dense, _sparse, payload = projection.project_record(record)
    assert "register_people" not in payload
    assert "website_content" not in payload
    assert payload["bundle_digest"] == "a" * 64
    with pytest.raises(projection.ProjectionError, match="content fields"):
        projection.project_record(record, extra_passthrough=("register_people",))


def test_projection_retains_entity_lookup_and_register_state_metadata():
    """A normal source-scoped card stage must match rebuild metadata shape."""
    record = {
        "url": "https://www.vector8.example/en", "text": "Registered purpose.",
        "dense_text": "Registered purpose.", "sparse_text": "Registered purpose.",
        "title": "Vector8 AG", "lang": "de", "domain": "vector8.example",
        "source_type": "entity_card", "country": "CH", "license": "OGD",
        "register_id": "CHE-341.859.062", "register_scheme": "CHE-UID",
        "names": "Vector8 AG | Vector8 Ltd", "entity_kind": "org",
        "founded": "2024-05-13", "reg_status": "EXISTIEREND",
        "dissolved": False, "resolve_confidence": 1.0,
        "bundle_digest": "b" * 64,
    }
    _pid, _dense, _sparse, payload = projection.project_record(record)
    assert {key: payload[key] for key in (
        "register_id", "register_scheme", "names", "entity_kind", "founded",
        "reg_status", "dissolved", "resolve_confidence",
    )} == {
        "register_id": "CHE-341.859.062", "register_scheme": "CHE-UID",
        "names": "Vector8 AG | Vector8 Ltd", "entity_kind": "org",
        "founded": "2024-05-13", "reg_status": "EXISTIEREND",
        "dissolved": False, "resolve_confidence": 1.0,
    }


def test_source_scoped_card_reemit_preserves_identity_and_register_metadata():
    # task 530: build_two_channel's website channel refuses to
    # approximate the token budget when the pinned Arctic tokenizer is
    # unavailable (fail loud, entity_card_website.py:_website_tokenizer)
    # -- deterministic without `transformers` installed / network access.
    pytest.importorskip("transformers")
    rec, _audit = recipe.build_two_channel(
        _seed(
            names="Vector8 AG | Vector8 Ltd | CHE-341.859.062",
            tier="C", resolve_confidence=1.0,
            register_facets={
                "published": "2025-12-02", "founded": "2024-05-13",
                "reg_status": "EXISTIEREND", "dissolved": False,
            },
        ),
        {"point_id": "2a5bc752-ce61-58e7-9453-d63302b86e06",
         "url": "https://www.vector8.example/en", "trust_tier": 0.85},
        ent=None, qid=None, zweck="Software consulting.", phase="test")
    assert rec["names"] == "Vector8 AG | Vector8 Ltd | CHE-341.859.062"
    assert rec["point_id"] == "2a5bc752-ce61-58e7-9453-d63302b86e06"
    assert rec["trust_tier"] == 0.85
    assert {key: rec[key] for key in ("published", "founded", "reg_status", "dissolved")} == {
        "published": "2025-12-02", "founded": "2024-05-13",
        "reg_status": "EXISTIEREND", "dissolved": False,
    }
