"""Task 448 — per-publisher `country` / `license` stamps for the news lane.

The 2026-08-27 news pack (delta-20260827T174001Z, 6,357 rows) carried
`country=INTL` on every row — including the four Swiss publishers — and no
`license` at all on 6,349 of them. Root cause: the news lane types its passages
`source_type="reference"`, and both durable resolvers fell through:

  * `aw_country.country_for_record` rule (2): `reference` -> INTL, applied
    before any national signal. Correct for PubMed/arXiv, wrong for SRF.
  * `aw_b1_license.resolve_license`: no host entry for the four publishers and
    no `reference` source_type rule -> None -> ADR-039 licensing exception.

Both stamps are transcriptions of already-ratified source packs, never new
judgements (see the comments at each table). These tests pin the transcription
per publisher, and pin the two things that must NOT change: HN's linked
third-party pages stay INTL/unlicensed, and a record that already carries a
licence keeps it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import cytir.core.license as LIC
from cytir.core import country as CT
import cytir.news.enrich_tier1 as subject

REPO_ROOT = Path(__file__).resolve().parent.parent

# publisher -> (declared country, declared licence, declaring pack)
DECLARED = {
    "swissinfo.ch": ("CH", "permissive-or-fairuse-snippet",
                     "sources/ch/swissinfo.yaml"),
    "srf.ch": ("CH", "copyright-fairuse-snippet-pointer",
               "sources/ch/ch-media.yaml"),
    "rts.ch": ("CH", "copyright-fairuse-snippet-pointer",
               "sources/ch/ch-media.yaml"),
    "rsi.ch": ("CH", "copyright-fairuse-snippet-pointer",
               "sources/ch/ch-media.yaml"),
}


# --- country -----------------------------------------------------------------

from tests.helpers.repo_files import repo_file_or_skip

repo_file_or_skip("sources/ch/swissinfo.yaml")  # the public cut withholds this file; skip there (task 2382)
repo_file_or_skip("sources/ch/ch-media.yaml")  # the public cut withholds this file; skip there (task 2382)


@pytest.mark.parametrize("domain", sorted(DECLARED))
def test_swiss_publisher_news_row_is_stamped_ch_not_intl(domain):
    """`reference` no longer sends a declared publisher to INTL."""
    assert CT.country_for_record(domain, "reference") == DECLARED[domain][0]


@pytest.mark.parametrize("domain", sorted(DECLARED))
def test_publisher_country_survives_a_www_host_and_a_subdomain(domain):
    for host in (f"www.{domain}", f"https://www.{domain}/news/x", domain):
        assert CT.country_for_record(host, "reference") == "CH"


@pytest.mark.parametrize("source_type", ["reference", "news", "docs", None,
                                         "official", "entity_card"])
def test_publisher_declaration_holds_for_every_source_type(source_type):
    """A curated jurisdiction is a fact about the publisher, not about how one
    lane happened to type the passage."""
    assert CT.country_for_record("srf.ch", source_type) == "CH"


def test_hn_linked_third_party_pages_keep_their_previous_intl_stamp():
    for domain in ("theguardian.com", "nytimes.com", "github.com", "proton.me",
                   "news.ycombinator.com"):
        assert CT.country_for_record(domain, "reference") == "INTL"


def test_undeclared_domains_are_untouched_by_the_publisher_table():
    assert CT.country_for_record("arxiv.org", "reference") == "INTL"
    assert CT.country_for_record("example.ch", "reference") == "INTL"
    assert CT.country_for_record("example.ch", "gov") == "CH"
    assert CT.country_for_record("example.com", "docs") is None


# --- licence -----------------------------------------------------------------

@pytest.mark.parametrize("domain", sorted(DECLARED))
def test_publisher_license_matches_the_ratified_pack(domain):
    _, expected, _ = DECLARED[domain]
    assert LIC.resolve_license(None, domain, domain, "reference") == expected
    assert LIC.resolve_license(None, f"www.{domain}", f"www.{domain}",
                               "reference") == expected


@pytest.mark.parametrize("domain", sorted(DECLARED))
def test_declared_license_is_transcribed_from_its_pack_yaml_not_invented(domain):
    """The value in code MUST equal the `license.default` of the pack that
    declares it. If someone edits one side only, this fails."""
    _, expected, pack = DECLARED[domain]
    text = (REPO_ROOT / pack).read_text(encoding="utf-8")
    # `license:` block -> `default:` line; the packs write it plain or quoted.
    block = text.split("\nlicense:", 1)[1]
    default = block.split("default:", 1)[1].splitlines()[0].strip().strip('"\'')
    assert default == expected, f"{pack} declares {default!r}, code stamps {expected!r}"


def test_an_ingester_supplied_license_still_wins():
    assert LIC.resolve_license("CC-BY-4.0", "srf.ch", "srf.ch",
                               "reference") == "CC-BY-4.0"


def test_hn_linked_third_party_pages_get_no_invented_license():
    """No established instrument -> licensing EXCEPTION, never a guess."""
    for domain in ("theguardian.com", "nytimes.com", "proton.me",
                   "news.ycombinator.com"):
        assert LIC.resolve_license(None, domain, domain, "reference") is None


# --- the news lane's enrich step ---------------------------------------------

def _enrich(record, host_meta=None):
    return subject.enrich_record(dict(record), host_meta or {})


@pytest.mark.parametrize("domain", sorted(DECLARED))
def test_enrich_tier1_stamps_the_publisher_country(domain):
    row = _enrich({"domain": domain, "source_type": "reference",
                   "url": f"https://www.{domain}/a", "title": "t"})
    assert row["country"] == "CH"


def test_enrich_tier1_no_longer_writes_the_keep_existing_sentinel():
    """Pre-448 this compared against the literal "KEEP_EXISTING" while the real
    sentinel is "__KEEP_EXISTING__", so the sentinel string itself was written
    into the country facet of every registered entity_card."""
    row = _enrich({"domain": "example.com", "source_type": "entity_card",
                   "register_id": "CHE-123.456.789", "country": "CH",
                   "url": "https://example.com/", "title": "t"})
    assert row["country"] == "CH"
    assert CT.KEEP_EXISTING not in json.dumps(row)


def test_enrich_tier1_host_meta_license_is_still_honoured():
    """Task 638: honoured only WITH a citation now -- see
    tests/test_aw_news_enrich_tier1.py::test_enrich_file_drops_a_license_with_no_citation
    for the no-citation case this used to (wrongly) also pass."""
    row = _enrich({"domain": "huggingface.co", "source_type": "reference",
                   "url": "https://huggingface.co/x", "title": "t"},
                  {"huggingface.co": {
                      "license": "Apache-2.0",
                      "license_citation": "repo LICENSE file, captured via "
                      "GitHub's repo-license API at fetch time.",
                  }})
    assert row["license"] == "Apache-2.0"
