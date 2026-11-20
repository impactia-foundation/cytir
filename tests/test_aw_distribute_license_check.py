# Task 650 (migration S8): lives in tests/ with the rest of the package's tests (task 696 slice 2).
"""Tests for cytir.distribute.license_check (task 650, S8).

`assert_export_license_coverage`/`CoverageTally` are the ADR-039 caller
task 645 found missing (the old `assert_license` script had none); ADR-112
point 2 requires this to gate on PRESENCE only, never on the license
string's content ("unknown" ships).
"""
from __future__ import annotations

import pytest

from cytir.distribute.license_check import (
    CoverageTally,
    LicenseCoverageRefusal,
    assert_export_license_coverage,
    has_license,
)


def test_has_license():
    assert has_license({"license": "CC-BY-4.0"}) is True
    assert has_license({"license": "unknown"}) is True  # ADR-112 point 2
    assert has_license({"license": ""}) is False
    assert has_license({"license": None}) is False
    assert has_license({}) is False


def test_coverage_full_passes_at_default_threshold():
    rows = [{"license": "CC0-1.0", "source_type": "a"}] * 10
    report = assert_export_license_coverage(rows)
    assert report["pct"] == 1.0
    assert report["total"] == 10


def test_coverage_below_threshold_refuses():
    rows = ([{"license": "CC0-1.0", "source_type": "a"}] * 8 +
            [{"source_type": "a"}] * 2)  # 80% coverage
    with pytest.raises(LicenseCoverageRefusal):
        assert_export_license_coverage(rows, threshold=0.95)


def test_coverage_unknown_value_still_counts_as_covered():
    rows = [{"license": "unknown", "source_type": "entity_card"}] * 5
    report = assert_export_license_coverage(rows, threshold=1.0)
    assert report["pct"] == 1.0


def test_coverage_per_family_breakdown():
    rows = ([{"license": "MIT", "source_type": "a"}] * 5 +
            [{"source_type": "b"}] * 5)
    tally = CoverageTally()
    for r in rows:
        tally.add(r)
    report = tally.report()
    assert report["families"]["a"]["pct"] == 1.0
    assert report["families"]["b"]["pct"] == 0.0


def test_empty_tally_reports_full_coverage():
    assert CoverageTally().report()["pct"] == 1.0
