#!/usr/bin/env python3
"""Tests for the task-465 rii dump-basis backfill (ADR-076) and its licence entry.

Runs cold: no Qdrant, no network, no corpus. The 38,159-row source is not in
the repo, so the identity/conservation machinery is exercised on synthetic rows
built by the same rules, and the SEALED artifact in `data/assembly/` is checked
as-is against the constants and against the consent gate's own source.
"""
from __future__ import annotations

import gzip
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import cytir.core.ids as ID              # noqa: E402
import cytir.core.license as LIC              # noqa: E402
from cytir.ops import rii_dump_basis_backfill as R      # noqa: E402

MANIFEST_PATH = os.path.join(ROOT, "data", "assembly",
                             "rii-dump-basis-backfill.v1.json")


from tests.helpers.repo_files import repo_file_or_skip

repo_file_or_skip("data/assembly/rii-dump-basis-backfill.v1.json")  # the public cut withholds this file; skip there (task 2382)


@pytest.fixture(scope="module")
def manifest() -> dict:
    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# 1. the licence entry
# ---------------------------------------------------------------------------

def test_rii_resolves_to_the_urhg_5_official_work_value():
    assert LIC.resolve_license(None, "rechtsprechung-im-internet.de",
                               "www.rechtsprechung-im-internet.de",
                               "official", "rii_courts") == \
        "public-domain-official-work-UrhG-5"


def test_rii_licence_resolves_from_the_bare_host_and_the_www_host():
    for host in ("rechtsprechung-im-internet.de",
                 "www.rechtsprechung-im-internet.de"):
        assert LIC.resolve_license(None, None, host, "official", None) == \
            "public-domain-official-work-UrhG-5"


def test_rii_was_a_licensing_exception_before_this_entry():
    """The gap this closes: without the entry the record has no licence at all,
    which ADR-039 makes a counted MANIFEST exception, and the dump basis needs
    a non-blank licence to admit."""
    without = dict(LIC._HOST_LICENSE)
    without.pop("rechtsprechung-im-internet.de")
    original = LIC._HOST_LICENSE
    LIC._HOST_LICENSE = without
    try:
        assert LIC.resolve_license(None, "rechtsprechung-im-internet.de",
                                   "www.rechtsprechung-im-internet.de",
                                   "official", None) is None
    finally:
        LIC._HOST_LICENSE = original


def test_an_ingester_supplied_licence_still_wins():
    assert LIC.resolve_license("CC0-1.0", "rechtsprechung-im-internet.de",
                               "www.rechtsprechung-im-internet.de",
                               "official", None) == "CC0-1.0"


def test_the_entry_does_not_leak_onto_neighbouring_german_law_hosts():
    assert LIC.resolve_license(None, None, "www.gesetze-im-internet.de",
                               "official", None) == "GermanGov-Terms"
    assert LIC.resolve_license(None, None, "gesetze-rechtsprechung.sh.juris.de",
                               "official", None) is None


# ---------------------------------------------------------------------------
# 2. the stamp, against the consent gate's OWN constants
# ---------------------------------------------------------------------------

def test_gate_constants_are_read_from_the_builder_source():
    gate = R.gate_constants()
    assert gate["DUMP_CONSENT_STATE"] == "dump"
    assert "official-dump-license" in gate["DUMP_CONSENT_BASES"]


def test_stamp_satisfies_every_field_the_gate_requires():
    assert R.check_stamp(R.stamp("2026-08-28")) == []


def test_stamp_matches_the_real_dump_basis_predicate_when_importable():
    """If the build environment can import the builder gate, the real predicate must
    agree with `check_stamp` — the AST reading is a stand-in, never a fork."""
    try:
        from cytir.consent import build_gate as B
    except Exception:  # pragma: no cover - qdrant_client absent on this box
        pytest.skip("builder gate not importable here")
    assert B.dump_basis_evidence(R.stamp("2026-08-28")) is not None


@pytest.mark.parametrize("field,value", [
    ("consent_state", "grace"),
    ("consent_state", "granted"),
    ("consent_basis", "robots-live-resolve"),
    ("consent_basis", "official dump license"),
    ("consent_source", ""),
    ("license", ""),
    ("consent_checked_at", ""),
])
def test_a_stamp_that_would_launder_a_decline_is_refused(field, value):
    bad = R.stamp("2026-08-28")
    bad[field] = value
    assert R.check_stamp(bad), f"{field}={value!r} must not pass"


@pytest.mark.parametrize("value", [None, True, "false", 0])
def test_train_ok_must_be_exactly_false(value):
    """`tdm-reservation: 1` on every rii response makes this mandatory, and
    task 457 measured that nothing on these points records it today."""
    bad = R.stamp("2026-08-28")
    bad["train_ok"] = value
    problems = R.check_stamp(bad)
    assert any("train_ok" in problem for problem in problems)


def test_search_ok_must_be_true_because_that_is_all_the_dump_basis_admits():
    bad = R.stamp("2026-08-28")
    bad["search_ok"] = False
    assert any("search_ok" in problem for problem in R.check_stamp(bad))


def test_stamp_licence_must_agree_with_what_the_build_would_resolve():
    bad = R.stamp("2026-08-28")
    bad["license"] = "GermanGov-Terms"
    assert any("aw_b1_license resolves" in problem for problem in R.check_stamp(bad))


def test_the_expired_grace_fields_are_cleared_not_overwritten():
    assert set(R.CLEAR_FIELDS) == {"consent_reason", "consent_grace_until"}


# ---------------------------------------------------------------------------
# 3. identity + conservation, on synthetic rows
# ---------------------------------------------------------------------------

def _row(n: int) -> dict:
    return {
        "url": ("https://www.rechtsprechung-im-internet.de/jportal/"
                f"?quelle=jlink&docid=SYNTH{n:06d}&psml=bsjrsprod.psml&max=true"),
        "title": f"BGH 1. Zivilsenat, Urteil vom 2020-01-{n % 28 + 1:02d}",
        "text": f"BGH 1. Zivilsenat, Urteil vom 2020-01-01\nLeitsatz: synthetic {n}",
        "domain": "rechtsprechung-im-internet.de",
        "source_type": "official",
        "point_id": f"legacy-{n:06d}",
    }


def test_identity_is_tier_b_and_reminted_from_url_plus_text_digest():
    row = _row(1)
    minted = R.mint(row)
    assert minted["identity_tier"] == "B"
    expected = ID.mint_point_id(minted["canonical_url"],
                                ID.content_digest(row["text"]))
    assert minted["point_id"] == expected
    assert minted["legacy_point_id"] == "legacy-000001"


def test_changing_one_character_of_the_retained_text_changes_the_identity():
    """Why the embed queue may not re-derive the text: the id embeds its digest."""
    a, b = _row(1), _row(1)
    b["text"] += "."
    assert R.mint(a)["point_id"] != R.mint(b)["point_id"]


def test_a_row_without_text_cannot_be_minted():
    row = _row(1)
    row["text"] = "   "
    with pytest.raises(ValueError):
        R.mint(row)


def _write_source(tmp_path, rows) -> str:
    path = os.path.join(str(tmp_path), "rii.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def test_derive_refuses_a_source_whose_digest_is_not_the_pinned_one(tmp_path):
    path = _write_source(tmp_path, [_row(n) for n in range(5)])
    with pytest.raises(SystemExit) as excinfo:
        R.derive(path)
    assert "source digest" in str(excinfo.value)


def test_derive_refuses_a_foreign_domain(tmp_path, monkeypatch):
    rows = [_row(0), _row(1)]
    rows[1]["domain"] = "example.com"
    path = _write_source(tmp_path, rows)
    monkeypatch.setattr(R, "SOURCE_SHA256", R.sha256_file(path))
    with pytest.raises(SystemExit) as excinfo:
        R.derive(path)
    assert "foreign domain" in str(excinfo.value)


def test_conservation_reports_a_short_population_instead_of_sealing_it(
        tmp_path, monkeypatch):
    path = _write_source(tmp_path, [_row(n) for n in range(4)])
    monkeypatch.setattr(R, "SOURCE_SHA256", R.sha256_file(path))
    monkeypatch.setattr(R, "EXPECTED_POINTS", 5)
    _, conservation = R.derive(path)
    assert conservation["ok"] is False
    assert any("rows_read 4" in problem for problem in conservation["problems"])


def test_conservation_catches_a_duplicate_identity(tmp_path, monkeypatch):
    rows = [_row(0), _row(1), _row(1)]
    rows[2]["point_id"] = "legacy-000002"
    path = _write_source(tmp_path, rows)
    monkeypatch.setattr(R, "SOURCE_SHA256", R.sha256_file(path))
    monkeypatch.setattr(R, "EXPECTED_POINTS", 3)
    _, conservation = R.derive(path)
    assert conservation["ok"] is False
    assert conservation["id_collisions"] == 1


def test_sealing_is_deterministic_and_sorted(tmp_path, monkeypatch):
    rows = [_row(n) for n in range(6)]
    path = _write_source(tmp_path, rows)
    monkeypatch.setattr(R, "SOURCE_SHA256", R.sha256_file(path))
    monkeypatch.setattr(R, "EXPECTED_POINTS", 6)
    minted, _ = R.derive(path)
    first = R.seal_points(minted, os.path.join(str(tmp_path), "a.jsonl.gz"))
    second = R.seal_points(minted, os.path.join(str(tmp_path), "b.jsonl.gz"))
    assert first["sha256"] == second["sha256"]
    with gzip.open(os.path.join(str(tmp_path), "a.jsonl.gz"), "rt") as handle:
        ids = [json.loads(line)["point_id"] for line in handle]
    assert ids == sorted(ids) and len(ids) == 6


# ---------------------------------------------------------------------------
# 4. the sealed artifact in the repo
# ---------------------------------------------------------------------------

def test_manifest_stamp_still_passes_the_gate_check(manifest):
    assert R.check_stamp(manifest["stamp"]) == []


def test_manifest_is_bound_to_the_pinned_source_and_count(manifest):
    assert manifest["source"]["sha256"] == R.SOURCE_SHA256
    assert manifest["source"]["rows"] == R.EXPECTED_POINTS
    assert manifest["points"]["count"] == R.EXPECTED_POINTS
    assert manifest["conservation"]["ok"] is True
    assert manifest["conservation"]["id_collisions"] == 0


def test_manifest_membership_binding_is_a_full_measurement_not_a_sample(manifest):
    binding = manifest["membership_binding"]
    assert binding["scope"] == "all_claims"
    assert binding["claimed"] == R.EXPECTED_POINTS
    assert binding["unmatched"] == 0
    assert binding["points_sha256"] == manifest["points"]["sha256"]


def test_manifest_licence_reading_quotes_the_instrument(manifest):
    reading = manifest["licence_reading"]
    assert "amtlich verfaßte Leitsätze zu Entscheidungen" in reading["quote_abs_1"]
    assert "andere amtliche Werke" in reading["quote_abs_2"]
    assert reading["quote_source"]["url"].endswith("/urhg/___5.html")
    assert len(reading["grounds"]) >= 3


def test_manifest_licence_reading_is_measured_over_every_record(manifest):
    composition = manifest["licence_reading"]["retained_text_composition"]
    assert composition["header_court_kind_date_az"] == R.EXPECTED_POINTS
    # the two documentation elements are NAMED with their counts, not smoothed
    assert composition["Titelzeile"] > 0
    assert composition["Orientierungssatz_sonstosatz"] > 0


def test_manifest_does_not_rest_on_task_132s_unwitnessed_line(manifest):
    assert "free reuse" not in json.dumps(manifest["licence_reading"]["grounds"])
    assert "not_used" in manifest["licence_reading"]


def test_manifest_records_the_non_refreshable_ruling(manifest):
    policy = manifest["refresh_policy"]
    assert policy["state"] == "non-refreshable"
    assert policy["ruled_by"] == "Tim"
    assert policy["ruled_on"] == "2026-08-28"


def test_manifest_corrects_the_common_crawl_origin_claim(manifest):
    corrected = manifest["supersedes_consent_receipt"]["corrected_reason"]
    assert "came via Common Crawl" in corrected["wrong"]
    assert "no scraping" in corrected["why_wrong"]
    assert len(corrected["reasons_that_stand"]) == 2
    assert manifest["supersedes_consent_receipt"]["sha256"] == R.DECLINE_RECEIPT_SHA256


def test_manifest_names_the_expired_grace_window_it_replaces(manifest):
    assert manifest["expired_grace"]["consent_grace_until"] == "2026-08-24"
    assert manifest["clear_fields"] == list(R.CLEAR_FIELDS)


def test_manifest_assembly_pointer_is_concrete(manifest):
    consumption = manifest["assembly_consumption"]
    entry = consumption["add_set_entry"]
    assert entry["set_id"] == R.SET_ID
    assert entry["claims"]["sha256"] == manifest["points"]["sha256"]
    assert entry["claims"]["granularity"] == "point_id"
    assert entry["claims"]["count"] == R.EXPECTED_POINTS
    assert consumption["embed_queue"]["input_sha256"] == R.SOURCE_SHA256
    assert set("G1 G3 G5 G7 G8".split()) <= set(consumption["guards"])


def test_manifest_master_only_fields_are_marked_adr_039(manifest):
    assert "master-only" in manifest["master_only"]["note"]
    assert manifest["master_only"]["attribution"]


# ---------------------------------------------------------------------------
# 5. the verifier fails closed
# ---------------------------------------------------------------------------

class _Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _sealed_pair(tmp_path, manifest):
    """A points file matching the committed manifest is not available offline,
    so build a small consistent pair and re-point the manifest at it."""
    rows = [_row(n) for n in range(3)]
    source = _write_source(tmp_path, rows)
    return rows, source


def test_verify_rejects_a_manifest_whose_points_file_was_tampered_with(
        tmp_path, manifest, monkeypatch):
    rows, source = _sealed_pair(tmp_path, manifest)
    monkeypatch.setattr(R, "SOURCE_SHA256", R.sha256_file(source))
    monkeypatch.setattr(R, "EXPECTED_POINTS", 3)
    minted, _ = R.derive(source)
    points = os.path.join(str(tmp_path), "points.jsonl.gz")
    sealed = R.seal_points(minted, points)

    local = json.loads(json.dumps(manifest))
    local["points"]["sha256"] = sealed["sha256"]
    local["points"]["count"] = 3
    local["source"]["sha256"] = R.SOURCE_SHA256
    local["source"]["rows"] = 3
    local["membership_binding"]["points_sha256"] = sealed["sha256"]
    local["membership_binding"]["claimed"] = 3
    local["assembly_consumption"]["add_set_entry"]["claims"]["sha256"] = sealed["sha256"]
    local["licence_reading"]["retained_text_composition"][
        "header_court_kind_date_az"] = 3
    path = os.path.join(str(tmp_path), "m.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(local, handle)

    args = _Args(manifest=path, points=points, rii_jsonl=source)
    assert R.cmd_verify(args) == 0, "the consistent pair must verify first"

    R.seal_points(minted[:2], points)          # drop one point, keep the manifest
    assert R.cmd_verify(args) == 1


def test_verify_rejects_a_manifest_with_no_membership_binding(tmp_path, manifest):
    local = json.loads(json.dumps(manifest))
    local["membership_binding"] = None
    path = os.path.join(str(tmp_path), "m.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(local, handle)
    missing = os.path.join(str(tmp_path), "absent.jsonl.gz")
    assert R.cmd_verify(_Args(manifest=path, points=missing, rii_jsonl="")) == 1


def test_verify_rejects_a_sampled_membership_binding(tmp_path, manifest):
    local = json.loads(json.dumps(manifest))
    local["membership_binding"]["scope"] = "sample"
    path = os.path.join(str(tmp_path), "m.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(local, handle)
    missing = os.path.join(str(tmp_path), "absent.jsonl.gz")
    assert R.cmd_verify(_Args(manifest=path, points=missing, rii_jsonl="")) == 1
