"""Focused tests for the source-neutral entity-card website receipt."""
from __future__ import annotations

import pytest

import cytir.corpus.entity_card_website as website


def _cc(**extra) -> dict:
    return {
        "schema": website.ACQUISITION_RECEIPT_SCHEMA,
        "route": "common-crawl-s3-warc-range",
        "url": "https://example.ch/about",
        "retrieved_at": "2026-08-04T12:00:00Z",
        "crawl": "CC-MAIN-2026-30",
        "fetch_time": "2026-07-22T08:00:00Z",
        "warc_filename": "crawl-data/CC-MAIN-2026-30/segments/x/warc/x.warc.gz",
        "offset": 42, "length": 4096, "content_digest": "sha1:ABC",
        "content_truncated": False, "index_manifest_sha256": "a" * 64,
        "selection_version": "entity-card-cc-newest-v1", **extra,
    }


@pytest.mark.parametrize("route", [
    "common-crawl-s3-warc-range",
    "common-crawl-https-warc-range",
])
def test_common_crawl_receipt_preserves_deterministic_warc_locator(route):
    receipt = _cc(route=route)
    assert website.validate_acquisition_receipt(
        "https://www.example.ch/about", "2026-07-22T08:00:00Z", "cc",
        receipt) == receipt


@pytest.mark.parametrize("change,match", [
    ({"fetch_time": "2026-07-21T00:00:00Z"}, "capture time"),
    ({"content_truncated": True}, "truncated"),
    ({"index_manifest_sha256": "z" * 64}, "manifest digest"),
    ({"length": 0}, "byte-range"),
])
def test_invalid_common_crawl_locator_fails_closed(change, match):
    with pytest.raises(ValueError, match=match):
        website.validate_acquisition_receipt(
            "https://example.ch/about", "2026-07-22T08:00:00Z", "cc",
            _cc(**change))


def test_acquisition_route_cannot_cross_the_verified_site():
    with pytest.raises(ValueError, match="crosses source domain"):
        website.validate_acquisition_receipt(
            "https://other.example/about", "2026-07-22T08:00:00Z", "cc", _cc())


def test_common_crawl_capture_cannot_use_direct_mode_consent():
    with pytest.raises(ValueError, match="CC-mode"):
        website.validate_acquisition_receipt(
            "https://example.ch/about", "2026-07-22T08:00:00Z",
            "direct", _cc())


def test_cc_only_site_association_is_traceable_but_held_from_join():
    seed = {"register_id": "CHE-1", "domain": "example.ch",
            "url": "https://example.ch/"}
    receipt = {
        "schema": website.ASSOCIATION_RECEIPT_SCHEMA,
        "entity": "CHE-1", "domain": "example.ch", "status": "held",
        "evidence_class": "cc_heuristic", "source": "task275_zefix_ccjoin",
        "observed_at": "2026-07-14", "evidence_sha256": "e" * 64,
        "source_artifact_sha256": "d" * 64,
        "artifact_pointer": "master://depth275/cards#line=1",
        "resolve_confidence": 0.9, "resolve_method": "task275_cc_heuristic:schema",
    }
    assert website.validate_association_receipt(
        seed, receipt, require_admitted=False)["status"] == "held"
    with pytest.raises(ValueError, match="held"):
        website.validate_association_receipt(seed, receipt)
