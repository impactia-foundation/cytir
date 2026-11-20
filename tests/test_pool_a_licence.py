#!/usr/bin/env python3
"""Guard for task 629 P8 -- Pool A's per-record licence, read from the
instrument (memory `licence-basis-from-the-instrument`).

Four things this guards, and why each one is here:

1. **Every non-github GO-family origin resolves a non-empty licence** through
   `aw_b1_license.resolve_license()`. Each of those resolutions is grounded in
   a host-table entry this task added, and each entry cites the instrument it
   was transcribed from in a comment (never inferred from hosting).

2. **`github.com` carries no host rule.** A GitHub-hosted file's licence is
   the repository's, not the host's (ADR-039 §U2's per-record contract), so a
   blanket `github.com -> <some licence>` entry would misrepresent every repo
   whose real licence differs. `resolve_license(None, "github.com", ...)`
   must return `None`.

3. **A `NOASSERTION` `existing_license` is rejected, not passed through.**
   `github_agent_resources`' acquisition receipts captured exactly this value
   for 100% of a 3962-receipt sample (task 629 P8 census on the build box) --
   `NOASSERTION` is GitHub's/the roster's "no assertion made", never a grant.
   Before this task, `resolve_license()`'s truthiness check would have let it
   through as though it were an SPDX id.

4. **The licence artefact's origins match the sealed consent artefact's
   origin roster exactly.** the sealed Pool A consent artefact's
   `family_origin_roster` is P7's sealed, rows-derived roster; P8's licence
   artefact must cover the identical (family, origin) set with no drift
   between the two preconditions' bookkeeping.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import cytir.core.license as lic  # noqa: E402
from tests.helpers.repo_files import repo_file_or_skip  # noqa: E402

CONSENT_ARTIFACT = ROOT / "tasks" / "629-artifacts" / "pool-a-consent.v1.json"
LICENCE_ARTIFACT = ROOT / "tasks" / "629-artifacts" / "pool-a-licence.v1.json"

# The 8 families P7 marked GO, and their origins, as captured in the sealed
# consent artefact's `family_origin_roster` (task 629 §13.4). Hand-transcribed
# here (not re-read from the artefact) for guard #1/#2/#3 below, so a change to
# either file is caught by a diff against this literal roster, not silently
# absorbed by re-deriving from whichever file changed.
GO_FAMILY_ORIGINS = {
    "technical_manufacturer_catalog": ["sunfounder.com", "upesy.com", "toybox.com"],
    "news_current_articles": [
        "www.swissinfo.ch", "www.rsi.ch", "www.srf.ch", "www.rts.ch",
    ],
    "canada_official_news": ["www.canada.ca"],
    "service_public_fiche": [
        "www.service-public.gouv.fr", "entreprendre.service-public.gouv.fr",
    ],
    "statcan_wds": ["www150.statcan.gc.ca"],
    "rfc_editor_net_new": ["www.rfc-editor.org"],
    # github.com origins are handled separately: they do NOT resolve via a
    # host rule, so they are excluded from the "non-empty licence" guard and
    # covered instead by their own NOASSERTION-specific assertions below.
    "github_agent_resources": ["github.com"],
    "context7_pinned_docs": ["github.com"],
}

GITHUB_FAMILIES = {"github_agent_resources", "context7_pinned_docs"}


def _host_pair(origin: str) -> tuple[str, str]:
    # aw_b1_license.resolve_license takes (domain, host); pass the same value
    # for both, matching how the P7 origin roster stores a single host string.
    return origin, origin


def test_every_non_github_go_origin_resolves_a_licence():
    for family, origins in GO_FAMILY_ORIGINS.items():
        if family in GITHUB_FAMILIES:
            continue
        for origin in origins:
            domain, host = _host_pair(origin)
            resolved = lic.resolve_license(None, domain, host, "unknown")
            assert resolved, (
                f"{family} / {origin} resolved no licence "
                f"(resolve_license returned {resolved!r})"
            )
            assert isinstance(resolved, str) and resolved.strip() == resolved


def test_github_origins_do_not_resolve_via_a_host_rule():
    # No existing_license on the record at all (the honest state for both
    # github_agent_resources, whose input roster carries "license": null, and
    # context7_pinned_docs, whose receipts carry no license field at all) ->
    # must fall all the way through to None. A host rule for github.com would
    # incorrectly grant every repo the same licence regardless of its own.
    assert lic.resolve_license(None, "github.com", "github.com", "unknown") is None
    assert lic.resolve_license("", "github.com", "github.com", "unknown") is None
    assert "github.com" not in lic._HOST_LICENSE
    # the suffix-match loop must not accidentally catch it via some other key
    for suffix in lic._HOST_LICENSE:
        assert not ("github.com" == suffix or "github.com".endswith("." + suffix)), (
            f"github.com unexpectedly matches host-table entry {suffix!r}"
        )


def test_noassertion_is_rejected_not_passed_through():
    for value in ("NOASSERTION", "noassertion", "NoAssertion", "None", "null", "", None):
        resolved = lic.resolve_license(value, "github.com", "github.com", "unknown")
        assert resolved is None, f"existing_license={value!r} incorrectly resolved to {resolved!r}"
    # A real SPDX id on the record must still pass through untouched -- the
    # rejection is specific to no-grant sentinels, not to every existing value.
    assert lic.resolve_license("MIT", "github.com", "github.com", "unknown") == "MIT"
    assert lic.resolve_license("Apache-2.0", "github.com", "github.com", "unknown") == "Apache-2.0"


def test_github_families_currently_resolve_zero_rows():
    # Documents the P8 finding precisely: as currently staged, NEITHER github
    # origin has a usable per-record licence (github_agent_resources' roster
    # never carried one; context7_pinned_docs' receipts never captured the
    # field at all), so both resolve to a licensing exception today. This is
    # not a partial-exclusion guard -- it is the honest, verified state, and
    # it must not silently start passing (e.g. via an accidental host rule)
    # without a deliberate, instrument-grounded change.
    for family in GITHUB_FAMILIES:
        for origin in GO_FAMILY_ORIGINS[family]:
            assert origin == "github.com"
            assert lic.resolve_license(None, origin, origin, "unknown") is None


def test_licence_artifact_matches_go_family_origin_roster():
    data = json.loads(repo_file_or_skip(LICENCE_ARTIFACT.relative_to(ROOT)).read_text(encoding="utf-8"))
    assert data["schema"] == "aw.pool-a-licence.v1"
    families = data["families"]
    assert set(families) == set(GO_FAMILY_ORIGINS), (
        set(families).symmetric_difference(GO_FAMILY_ORIGINS)
    )
    for family, expected_origins in GO_FAMILY_ORIGINS.items():
        actual_origins = set(families[family]["origins"])
        assert actual_origins == set(expected_origins), (
            family, actual_origins.symmetric_difference(expected_origins)
        )


def test_licence_artifact_origins_match_sealed_consent_roster():
    consent = json.loads(repo_file_or_skip(CONSENT_ARTIFACT.relative_to(ROOT)).read_text(encoding="utf-8"))
    licence = json.loads(repo_file_or_skip(LICENCE_ARTIFACT.relative_to(ROOT)).read_text(encoding="utf-8"))
    consent_roster = consent["family_origin_roster"]
    go_families = set(GO_FAMILY_ORIGINS)  # the P7 §13.4 GO set, held above
    # The consent roster also carries HELD/REFUSED families (stackoverflow_qa,
    # eric_official_pdf, openalex_academic_dump, hn_firebase_pointer) that P8
    # is not scoped to license -- restrict the cross-check to the GO set.
    for family in go_families:
        assert family in consent_roster, f"{family} missing from sealed consent roster"
        consent_origins = set(consent_roster[family])
        licence_origins = set(licence["families"][family]["origins"])
        assert consent_origins == licence_origins, (
            family, consent_origins.symmetric_difference(licence_origins)
        )


def test_licence_artifact_flags_the_canada_ca_correction():
    licence = json.loads(repo_file_or_skip(LICENCE_ARTIFACT.relative_to(ROOT)).read_text(encoding="utf-8"))
    corrections = {c["host"]: c for c in licence["corrections_to_existing_table"]}
    assert "canada.ca" in corrections
    assert corrections["canada.ca"]["old_value"] == "OGL-Canada-2.0"
    # the live host table must actually reflect the corrected value, not just
    # the artefact's record of the correction. Task 636 wrapped every
    # _HOST_LICENSE value in an _Entry(license, citation) NamedTuple (a
    # citation is now structurally required, not just a comment) -- read
    # `.license` rather than comparing the entry itself.
    assert lic._HOST_LICENSE["canada.ca"].license == corrections["canada.ca"]["new_value"]
    assert lic._HOST_LICENSE["canada.ca"].license != "OGL-Canada-2.0"
    assert lic._HOST_LICENSE["canada.ca"].citation, (
        "canada.ca's entry must cite the instrument it was read from"
    )


def test_licence_artifact_flags_both_github_exceptions_with_zero_surviving():
    licence = json.loads(repo_file_or_skip(LICENCE_ARTIFACT.relative_to(ROOT)).read_text(encoding="utf-8"))
    exceptions = {e["family"]: e for e in licence["licensing_exceptions"]}
    for family in GITHUB_FAMILIES:
        assert family in exceptions
        assert exceptions[family]["rows_surviving"] == 0
        assert exceptions[family]["rows"] > 0
