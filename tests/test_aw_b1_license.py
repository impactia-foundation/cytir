"""Guard for `aw_b1_license.py` (task 636).

Two defects this closes, both real, both measured on the build twin
(aw_e8) 2026-09-22:

1. `canada.ca` carried `OGL-Canada-2.0` INHERITED from its
   `laws-lois.justice.gc.ca` / `bankofcanada.ca` siblings — never read from
   canada.ca's own terms, which forbid commercial redistribution without
   permission. 178,417 live rows over-claimed a permissive grant that does
   not exist for the general site (memory `licence-basis-from-the-instrument`).
   The real Open Government Licence – Canada only covers `open.canada.ca`.

2. SPDX's `NOASSERTION` sentinel ("no assertion was made about the license")
   was being trusted as a real grant by `resolve_license`'s "trust the
   ingester" rule, because it is a non-empty, non-"None" string. It means the
   opposite of a grant.

Every `_HOST_LICENSE` entry is required to carry a citation to the declaring
instrument (a live terms-page quote+URL+date, or a statutory citation) —
never a bare license string with no evidence behind it. That is the
structural fix: a table entry with no citation is exactly how the canada.ca
inheritance slipped in unnoticed.
"""
from __future__ import annotations

import re

import cytir.core.license as subject


def test_every_host_entry_cites_an_instrument():
    """No `_HOST_LICENSE` entry may exist without a citation to what grants it.

    A citation must be non-trivial (not just the license string repeated) and
    must look like evidence: either it names a URL (http/https) or it names a
    statutory instrument (a section-sign, "Art.", or a named Order/Act/Licence).
    A bare, uncited assertion is exactly the shape of the canada.ca defect.
    """
    looks_like_url = re.compile(r"https?://")
    looks_like_statute = re.compile(
        r"§|Art\.\s|UrhG|URG|LDA|SI/\d|\bOrder\b|\bAct\b|\bLicence\b|\bLicense\b"
    )
    missing = []
    thin = []
    for host, entry in subject._HOST_LICENSE.items():
        citation = (entry.citation or "").strip()
        if not citation:
            missing.append(host)
            continue
        if len(citation) < 20:
            thin.append(host)
            continue
        if not (looks_like_url.search(citation) or looks_like_statute.search(citation)):
            thin.append(host)
    assert not missing, f"hosts with NO citation at all: {missing}"
    assert not thin, f"hosts with a citation too thin to be evidence: {thin}"


def test_every_citation_carries_a_fetch_or_determination_date():
    """A citation with no date can silently go stale with nobody noticing."""
    has_date = re.compile(r"20\d\d-\d\d-\d\d|task[s]?[\s/]\d{3}")
    undated = [
        host for host, entry in subject._HOST_LICENSE.items()
        if not has_date.search(entry.citation)
    ]
    assert not undated, f"hosts whose citation carries no date/task pointer: {undated}"


def test_canada_ca_no_longer_inherits_the_open_government_licence():
    """The task 629/636 defect, pinned: bare canada.ca must NOT resolve to OGL.

    The corrected value is `CanadaGov-Terms-NC-Attribution` (task 629 P8,
    the pool-a-licence.v1.json task evidence) — canada.ca's own terms
    page grants non-commercial reproduction with attribution, a real but
    narrower grant than the Open Government Licence, so the value names that
    grant rather than reusing the generic no-grant fair-use pointer.
    """
    for host in ("canada.ca", "www.canada.ca", "health-infobase.canada.ca",
                 "canada.ca.anything.canada.ca"):
        got = subject.resolve_license(None, "canada.ca", host, "official")
        assert got != "OGL-Canada-2.0", (
            f"{host} resolved to OGL-Canada-2.0 — the inherited-licence "
            "defect task 629/636 fixed is back"
        )
        assert got == "CanadaGov-Terms-NC-Attribution"


def test_open_canada_ca_keeps_the_real_ogl_grant():
    """open.canada.ca is the actual OGL-Canada-2.0 catalogue and must resolve
    to it independently of the generic canada.ca entry."""
    got = subject.resolve_license(None, "canada.ca", "open.canada.ca", "official")
    assert got == "OGL-Canada-2.0"


def test_laws_lois_resolves_to_the_reproduction_of_federal_law_order():
    """Federal statutes/regs/case law carry SI/97-5, not the OGL — a
    broader grant (no attribution clause) than OGL-Canada-2.0, and the value
    ADR-039's own text uses as its worked example of a gov-specific licence
    string."""
    got = subject.resolve_license(
        None, "laws-lois.justice.gc.ca", "laws-lois.justice.gc.ca", "official")
    assert got == "SI/97-5"


def test_bank_of_canada_gets_its_own_terms_not_the_ogl():
    got = subject.resolve_license(
        None, "bankofcanada.ca", "bankofcanada.ca", "central-bank")
    assert got == "BankOfCanada-Terms"


def test_admin_ch_no_longer_claims_reuse_without_consent():
    """Bare admin.ch was sharing swisstopo's old permissive value with no
    citation; its own terms require written consent to reproduce."""
    for host in ("admin.ch", "bag.admin.ch", "www.admin.ch"):
        got = subject.resolve_license(None, "admin.ch", host, "official")
        assert got == "copyright-fairuse-snippet-pointer"


def test_swisstopo_keeps_its_own_verified_open_data_grant():
    """swisstopo's 2021 open-geodata policy is real and must not be
    downgraded by the admin.ch fix above — it needs its own key/value."""
    got = subject.resolve_license(
        None, "admin.ch", "swisstopo.admin.ch", "official")
    assert got == "swisstopo-OGD-2021"


def test_fedlex_is_unaffected_by_the_admin_ch_downgrade():
    got = subject.resolve_license(
        None, "admin.ch", "fedlex.admin.ch", "official")
    assert got == "public-domain"   # RULING 53 (task 1040)


def test_openlegaldata_resolves_to_the_declared_urhg5_basis():
    """Task 641 (2026-09-23): 34,323 real OLD rows refused with
    `unresolved license` because this table had no `de.openlegaldata.io`
    entry -- the basis itself was already declared and ratified
    (sources/de/openlegaldata.yaml, task 460), just never encoded here."""
    got = subject.resolve_license(
        None, "de.openlegaldata.io", "de.openlegaldata.io", "official")
    assert got == "public-domain-official-work-UrhG-5"


def test_openlegaldata_never_resolves_to_its_own_odbl_sibling_value():
    """The source file is explicit: ODbL covers ONLY OpenLegalData's own
    database compilation/metadata and must never be stamped on a record."""
    got = subject.resolve_license(
        None, "de.openlegaldata.io", "de.openlegaldata.io", "official")
    assert got != "ODbL"


def test_openlegaldata_shares_the_urhg5_family_with_its_sibling_host():
    """Same statutory basis as rechtsprechung-im-internet.de -- two hosts,
    one instrument (§ 5 Abs. 1 UrhG), matching the source file's own note
    that these are 'the same statutory family'."""
    old = subject.resolve_license(
        None, "de.openlegaldata.io", "de.openlegaldata.io", "official")
    rii = subject.resolve_license(
        None, "rechtsprechung-im-internet.de",
        "rechtsprechung-im-internet.de", "official")
    assert old == rii == "public-domain-official-work-UrhG-5"


def test_noassertion_is_not_trusted_as_a_grant():
    """SPDX NOASSERTION means 'no assertion made', not 'permitted'. Task 629
    found github_agent_resources captured it verbatim from GitHub's
    repo-license API; resolve_license must fall through past it, exactly as
    it would an empty or 'None' value."""
    for sentinel in ("NOASSERTION", "noassertion", "NoAssertion", "None", "", "unknown"):
        got = subject.resolve_license(sentinel, "github.com", "github.com", "reference")
        # falls through to the host table; github.com has no entry, so None.
        assert got is None, f"{sentinel!r} was trusted as a grant: got {got!r}"


def test_a_real_existing_license_is_still_trusted():
    """The NOASSERTION fix must not become a blanket distrust of ingester
    values — a genuine SPDX string still wins immediately."""
    got = subject.resolve_license("CC-BY-SA-3.0", "stackoverflow.com",
                                   "stackoverflow.com", "qa")
    assert got == "CC-BY-SA-3.0"


def test_longest_suffix_wins_over_a_shorter_generic_entry():
    """A subdomain of a specific, longer key (open.canada.ca) must never be
    shadowed by a shorter, more generic key (canada.ca) that also matches by
    suffix -- the exact ordering bug that let health-infobase.canada.ca (a
    canada.ca subdomain) inherit OGL-Canada-2.0 pre-fix."""
    got = subject.resolve_license(
        None, "canada.ca", "sub.open.canada.ca", "official")
    assert got == "OGL-Canada-2.0"


def test_statcan_stays_on_its_own_open_licence_not_ogl_canada():
    got = subject.resolve_license(
        None, "statcan.gc.ca", "www150.statcan.gc.ca", "official")
    assert got == "StatCan-OGL"


def test_entity_card_source_type_fallback_unaffected():
    got = subject.resolve_license(None, "unknown-domain.example",
                                   "unknown-domain.example", "entity_card")
    assert got == "CC0-1.0"


def test_unknown_host_and_type_is_a_licensing_exception_not_a_guess():
    got = subject.resolve_license(None, "unknown-domain.example",
                                   "unknown-domain.example", "reference")
    assert got is None
