"""Task 558 -- pure-function tests for the DOAB/OAPEN monograph family (moved to tests/, task 1237).

Sealed fixtures, no network (verification-doctrine 1: unit tests are for
pure functions). The five records below were captured ONCE from the operator build box on
2026-09-09 via the real DOAB REST API (``/rest/items/<uuid>?expand=metadata``)
and the real per-title DOAB HTML page (``rel="license"`` badge) -- abstract
text stripped (never needed: this family embeds no book prose). Four are
real, complete MIT Press OA titles; the fifth (``Boosting``) is real too and
is the negative fixture: DOAB's own record for it carries no
``oapen.identifier.doi``, which is exactly the shape that must exclude a
title rather than crash or fabricate a URL.
"""
from __future__ import annotations

import json

from cytir.sources.families import doab_monographs as M

MIT_PRESS = "The MIT Press"
TODAY = "2026-09-09"

# -- sealed fixtures (task 558, captured from the operator build box 2026-09-09) ---------
DEMOCRATIZING_INNOVATION = {
    "title": "Democratizing Innovation",
    "authors": ["Hippel, Eric von"],
    "subjects": ["thema EDItEUR::K Economics, Finance, Business and "
                "Management::KJ Business and Management::KJH "
                "Entrepreneurship / Start-ups"],
    "language": "English", "dc_type": "book",
    "doi": "10.7551/mitpress/2333.001.0001",
    "handle_uri": "https://directory.doabooks.org/handle/20.500.12854/77862",
    "publisher_name": MIT_PRESS, "publisher_identifier": "1480",
}
DEMOCRATIZING_INNOVATION_BADGE_URL = "http://creativecommons.org/licenses/by-nc-nd/4.0"
DEMOCRATIZING_INNOVATION_URL = "https://direct.mit.edu/books/book/2821/Democratizing-Innovation"

DISTRIBUTIONAL_RL = {
    "title": "Distributional Reinforcement Learning",
    "authors": ["Bellemare, Marc G.", "Dabney, Will", "Rowland, Mark"],
    "subjects": [
        "thema EDItEUR::P Mathematics and Science::PB Mathematics::PBT "
        "Probability and statistics",
        "thema EDItEUR::U Computing and Information Technology::UY Computer "
        "science::UYQ Artificial intelligence::UYQM Machine learning",
    ],
    "language": "English", "dc_type": "book",
    "doi": "10.7551/mitpress/14207.001.0001",
    "handle_uri": "https://directory.doabooks.org/handle/20.500.12854/111581",
    "publisher_name": MIT_PRESS, "publisher_identifier": "1480",
}
DISTRIBUTIONAL_RL_BADGE_URL = "https://creativecommons.org/licenses/BY-NC-ND/4.0/"
DISTRIBUTIONAL_RL_URL = ("https://direct.mit.edu/books/book/5590/"
                         "Distributional-Reinforcement-Learning")

ARTIFICIAL_COMMUNICATION = {
    "title": "Artificial Communication",
    "authors": ["Esposito, Elena"],
    "subjects": [
        "thema EDItEUR::J Society and Social Sciences::JB Society and "
        "culture: general::JBC Cultural and media studies::JBCT Media "
        "studies::JBCT3 Media studies: advertising and society",
        "thema EDItEUR::U Computing and Information Technology::UY Computer "
        "science::UYQ Artificial intelligence::UYQM Machine learning",
    ],
    "language": "English", "dc_type": "book",
    "doi": "10.7551/mitpress/14189.001.0001",
    "handle_uri": "https://directory.doabooks.org/handle/20.500.12854/84607",
    "publisher_name": MIT_PRESS, "publisher_identifier": "1480",
}
ARTIFICIAL_COMMUNICATION_BADGE_URL = "https://creativecommons.org/licenses/by-nc-nd/4.0/"
ARTIFICIAL_COMMUNICATION_URL = ("https://direct.mit.edu/books/book/5338/"
                                "Artificial-CommunicationHow-Algorithms-Produce")

GRADIENT_EXPECTATIONS = {
    "title": "Gradient Expectations",
    "authors": ["Downing, Keith L."],
    "subjects": [
        "thema EDItEUR::U Computing and Information Technology::UY Computer "
        "science::UYQ Artificial intelligence::UYQN Neural networks and "
        "fuzzy systems",
    ],
    "language": "English", "dc_type": "book",
    "doi": "10.7551/mitpress/14723.001.0001",
    "handle_uri": "https://directory.doabooks.org/handle/20.500.12854/111599",
    "publisher_name": MIT_PRESS, "publisher_identifier": "1480",
}
GRADIENT_EXPECTATIONS_BADGE_URL = "https://creativecommons.org/licenses/BY-NC-ND/4.0/"
GRADIENT_EXPECTATIONS_URL = ("https://direct.mit.edu/books/book/5608/"
                             "Gradient-ExpectationsStructure-Origins-and")

# The negative fixture: real DOAB metadata, no DOI.
BOOSTING_NO_DOI = {
    "title": "Boosting",
    "authors": ["Schapire, Robert E.", "Freund, Yoav"],
    "subjects": ["thema EDItEUR::U Computing and Information Technology::UM "
                "Computer programming / software engineering::UMB "
                "Algorithms and data structures"],
    "language": "English", "dc_type": "book",
    "doi": None,
    "handle_uri": "https://directory.doabooks.org/handle/20.500.12854/77893",
    "publisher_name": MIT_PRESS, "publisher_identifier": "1480",
}

OK_CONSENT = {"search_ok": True, "train_ok": True, "basis": "b", "source": "s",
             "checked_at": TODAY}
HELD_CONSENT = {"search_ok": False, "train_ok": False, "basis": "b",
                "source": "s", "checked_at": TODAY}


# ---------------------------------------------------------------------------
# extract_license_badge / normalize_cc_license
# ---------------------------------------------------------------------------
def test_extract_license_badge_reads_rel_license_not_the_doab_catalogue_link():
    # The exact two-anchor shape a real DOAB item page carries (task 556
    # S2.2): a rel="license" anchor for the BOOK's own instrument, and a
    # plain link (no rel attribute) for DOAB's own compiled-record right.
    html = (
        '<div><a title="x" alt="x" '
        'href="https://creativecommons.org/licenses/BY-NC-ND/4.0/" '
        'rel="license"><img/></a></div>'
        '<span>this item\'s license is described as '
        '<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>'
        '</span>'
    )
    assert (M.extract_license_badge(html)
           == "https://creativecommons.org/licenses/BY-NC-ND/4.0/")


def test_extract_license_badge_absent_returns_none():
    html = "<html><body>no license anchor here</body></html>"
    assert M.extract_license_badge(html) is None


def test_normalize_cc_license_case_insensitive_and_versioned():
    assert M.normalize_cc_license(
        "http://creativecommons.org/licenses/by-nc-nd/4.0") == "CC-BY-NC-ND-4.0"
    assert M.normalize_cc_license(
        "https://creativecommons.org/licenses/BY-NC-ND/4.0/") == "CC-BY-NC-ND-4.0"
    assert M.normalize_cc_license(
        "https://creativecommons.org/licenses/by/4.0/") == "CC-BY-4.0"


def test_normalize_cc_license_unreadable_is_none_not_a_guess():
    assert M.normalize_cc_license(None) is None
    assert M.normalize_cc_license("https://example.com/not-a-license") is None


# ---------------------------------------------------------------------------
# parse_item_metadata
# ---------------------------------------------------------------------------
def test_parse_item_metadata_flattens_real_rest_response_shape():
    raw = [
        {"key": "dc.title", "value": "Distributional Reinforcement Learning"},
        {"key": "dc.contributor.author", "value": "Bellemare, Marc G."},
        {"key": "dc.contributor.author", "value": "Dabney, Will"},
        {"key": "dc.subject.classification",
         "value": "thema EDItEUR::U Computing and Information "
                  "Technology::UY Computer science::UYQ Artificial "
                  "intelligence::UYQM Machine learning"},
        {"key": "dc.language", "value": "English"},
        {"key": "dc.type", "value": "book"},
        {"key": "oapen.identifier.doi", "value": "10.7551/mitpress/14207.001.0001"},
        {"key": "dc.identifier.uri",
         "value": "https://directory.doabooks.org/handle/20.500.12854/111581"},
        {"key": "publisher.name", "value": "The MIT Press"},
        {"key": "publisher.identifier", "value": "1480"},
        # abstract must never survive into the parsed dict at all
        {"key": "dc.description.abstract", "value": "The first comprehensive guide..."},
    ]
    meta = M.parse_item_metadata(raw)
    assert meta["title"] == "Distributional Reinforcement Learning"
    assert meta["authors"] == ["Bellemare, Marc G.", "Dabney, Will"]
    assert meta["doi"] == "10.7551/mitpress/14207.001.0001"
    assert meta["publisher_name"] == "The MIT Press"
    assert "abstract" not in meta
    assert not any("comprehensive guide" in str(v) for v in meta.values())


# ---------------------------------------------------------------------------
# build_record -- the 4 real admits + the 1 real no-DOI exclusion
# ---------------------------------------------------------------------------
def _spdx(url):
    return M.normalize_cc_license(url)


def _chunk_with_built_url(monkeypatch, url):
    from types import SimpleNamespace
    rec = {"url": url, "text": "Ein Absatz mit genug Text.", "title": "T",
           "source_type": "web", "domain": "x.ch"}
    monkeypatch.setattr(M, "build_record", lambda *a, **k: dict(rec))
    monkeypatch.setattr(M.DoabMonographsFamily, "_publisher", staticmethod(lambda ctx: "p"))
    doc = SimpleNamespace(url=url, extra={"meta": {}, "spdx": "CC-BY-4.0"})
    return M.DoabMonographsFamily().chunk(doc, SimpleNamespace(today="2026-10-31"))


def test_chunk_excludes_a_search_result_url_with_a_counted_reason(monkeypatch):
    # task 2048: project_record returns None for a site-search URL; chunk must not index None.
    # task 3015: the unit is excluded with a reason, not closed as chunk_empty.
    import pytest
    from cytir.sources.family import UnitExcluded
    with pytest.raises(UnitExcluded, match="search_result_url"):
        _chunk_with_built_url(monkeypatch, "https://x.ch/?s=a")


def test_chunk_projects_a_normal_url_into_a_row_with_point_id(monkeypatch):
    out = _chunk_with_built_url(monkeypatch, "https://example.ch/artikel")
    assert len(out) == 1 and out[0]["point_id"]


def test_build_record_admits_distributional_rl_no_abstract_no_prose():
    rec = M.build_record(
        DISTRIBUTIONAL_RL, _spdx(DISTRIBUTIONAL_RL_BADGE_URL), DISTRIBUTIONAL_RL_URL,
        target_publisher=MIT_PRESS, consent=OK_CONSENT, today=TODAY)
    assert rec is not None
    assert rec["url"] == DISTRIBUTIONAL_RL_URL
    assert rec["title"] == "Distributional Reinforcement Learning"
    assert rec["license"] == "CC-BY-NC-ND-4.0"
    assert rec["source_type"] == "oa-monograph"
    assert rec["tier"] == "B"
    assert rec["trust_tier"] == M.TIER_TRUST["B"]
    assert rec["domain"] == "direct.mit.edu"
    assert rec["doab_id"] == "20.500.12854/111581"
    # the representation contract: no book prose/abstract anywhere in the
    # embedded surfaces, only register metadata.
    for surface in (rec["text"], rec["dense_text"], rec["sparse_text"]):
        assert "comprehensive guide" not in surface
        assert "Bellemare" in surface or "Reinforcement" in surface
    # dense surface = title + authors + subject (the declared composition);
    # sparse surface deliberately drops subject (title + authors +
    # publisher + doab_id, per the family's representation declaration).
    assert "Machine learning" in rec["dense_text"]
    assert "Machine learning" not in rec["sparse_text"]
    assert rec["sparse_text"].count("Distributional Reinforcement Learning") == 1
    assert rec["publisher"] in rec["sparse_text"]
    assert rec["doab_id"] in rec["sparse_text"]
    # ADR-039 master-only fields never claim to be the served license
    assert rec["_master"]["doab_catalogue_license"] == "CC-BY-4.0"
    assert rec["_master"]["doab_catalogue_license"] != rec["license"]


def test_build_record_admits_the_other_three_real_titles():
    for meta, badge, url in (
        (DEMOCRATIZING_INNOVATION, DEMOCRATIZING_INNOVATION_BADGE_URL,
         DEMOCRATIZING_INNOVATION_URL),
        (ARTIFICIAL_COMMUNICATION, ARTIFICIAL_COMMUNICATION_BADGE_URL,
         ARTIFICIAL_COMMUNICATION_URL),
        (GRADIENT_EXPECTATIONS, GRADIENT_EXPECTATIONS_BADGE_URL,
         GRADIENT_EXPECTATIONS_URL),
    ):
        rec = M.build_record(meta, _spdx(badge), url, target_publisher=MIT_PRESS,
                             consent=OK_CONSENT, today=TODAY)
        assert rec is not None, meta["title"]
        assert rec["license"] == "CC-BY-NC-ND-4.0"
        assert rec["url"] == url


def test_build_record_excludes_title_with_no_doi_and_therefore_no_url():
    # This is the real shape: DOAB's own record for "Boosting" has no
    # oapen.identifier.doi, so resolve_doi_url (exercised separately) can
    # mint no URL. build_record must refuse a record with no URL rather
    # than invent one.
    rec = M.build_record(BOOSTING_NO_DOI, "CC-BY-NC-ND-4.0", None,
                         target_publisher=MIT_PRESS, consent=OK_CONSENT, today=TODAY)
    assert rec is None


def test_build_record_excludes_unreadable_license_fail_closed():
    rec = M.build_record(DISTRIBUTIONAL_RL, None, DISTRIBUTIONAL_RL_URL,
                         target_publisher=MIT_PRESS, consent=OK_CONSENT, today=TODAY)
    assert rec is None


def test_build_record_excludes_wrong_publisher():
    other = dict(DISTRIBUTIONAL_RL, publisher_name="Some Other Press")
    rec = M.build_record(other, "CC-BY-NC-ND-4.0", DISTRIBUTIONAL_RL_URL,
                         target_publisher=MIT_PRESS, consent=OK_CONSENT, today=TODAY)
    assert rec is None


def test_build_record_excludes_when_consent_not_ok():
    rec = M.build_record(DISTRIBUTIONAL_RL, "CC-BY-NC-ND-4.0", DISTRIBUTIONAL_RL_URL,
                         target_publisher=MIT_PRESS, consent=HELD_CONSENT, today=TODAY)
    assert rec is None


def test_build_record_with_consent_none_skips_the_gate_but_stamps_nothing():
    # `consent=None` is the "not yet resolved" caller state (metadata-filter
    # pass, before the live consent call) -- must not itself gate admission,
    # and must not fabricate search_ok/train_ok fields.
    rec = M.build_record(DISTRIBUTIONAL_RL, "CC-BY-NC-ND-4.0", DISTRIBUTIONAL_RL_URL,
                         target_publisher=MIT_PRESS, consent=None, today=TODAY)
    assert rec is not None
    assert "search_ok" not in rec
    assert "train_ok" not in rec


# ---------------------------------------------------------------------------
# simplify_subject
# ---------------------------------------------------------------------------
def test_simplify_subject_strips_thema_path_and_code():
    assert M.simplify_subject(
        "thema EDItEUR::U Computing and Information Technology::UY Computer "
        "science::UYQ Artificial intelligence::UYQM Machine learning"
    ) == "Machine learning"


# ---------------------------------------------------------------------------
# has_title_and_authors -- the real defect found on the full MIT Press run:
# 62 of 379 metadata-admitted candidates silently vanished before staging
# (edited volumes carry dc.contributor.editor, not .author). Now checked in
# run()'s own counted `excluded["no_title_or_authors"]`, with the SAME
# function build_record uses, so the two can never drift apart again.
# ---------------------------------------------------------------------------
def test_has_title_and_authors_true_for_a_real_admitted_title():
    assert M.has_title_and_authors(DISTRIBUTIONAL_RL) is True


def test_has_title_and_authors_false_for_an_edited_volume_with_no_author():
    edited_volume = dict(DISTRIBUTIONAL_RL, authors=[])
    assert M.has_title_and_authors(edited_volume) is False


def test_has_title_and_authors_false_for_authors_that_are_blank_strings():
    sparse_authors = dict(DISTRIBUTIONAL_RL, authors=["  ", ""])
    assert M.has_title_and_authors(sparse_authors) is False


def test_has_title_and_authors_false_for_a_blank_title():
    blank_title = dict(DISTRIBUTIONAL_RL, title="   ")
    assert M.has_title_and_authors(blank_title) is False


# ---------------------------------------------------------------------------
# resolved_host_is_admissible -- the OTHER real defect found on the full MIT
# Press run: 9 of 317 metadata-admitted DOIs did not resolve to
# direct.mit.edu at all (real URLs captured on that run).
# ---------------------------------------------------------------------------
def test_resolved_host_is_admissible_for_the_real_pointer_host():
    assert M.resolved_host_is_admissible(
        DISTRIBUTIONAL_RL_URL, ("direct.mit.edu",)) is True


def test_resolved_host_is_admissible_false_for_the_crossref_chooser_interstitial():
    # Real URL from the task 558 run: a DOI that resolves to Crossref's own
    # content-negotiation chooser page instead of the book, for 8 of 317
    # candidates. Not a landing page an agent could use -- must be excluded,
    # not staged as a working pointer that isn't one.
    chooser_url = "https://chooser.crossref.org/?doi=10.7551%2Fmitpress%2F11008.001.0001"
    assert M.resolved_host_is_admissible(chooser_url, ("direct.mit.edu",)) is False


def test_resolved_host_is_admissible_false_for_a_third_party_microsite():
    # Real URL from the task 558 run: one MIT Press DOI resolves to the
    # book's own dedicated microsite, not direct.mit.edu -- a real landing
    # page, but an unvetted, non-allowlisted host.
    microsite_url = "https://islamic-pasts-futures.org/"
    assert M.resolved_host_is_admissible(microsite_url, ("direct.mit.edu",)) is False


def test_resolved_host_is_admissible_false_for_no_url():
    assert M.resolved_host_is_admissible(None, ("direct.mit.edu",)) is False


def test_resolved_host_is_admissible_is_case_insensitive():
    assert M.resolved_host_is_admissible(
        "https://DIRECT.MIT.EDU/books/book/1", ("direct.mit.edu",)) is True


# ---------------------------------------------------------------------------
# doab_search_page / enumerate_all_uuids -- the offset/start pagination bug
# ---------------------------------------------------------------------------
def _fixture_search_response(uuids):
    return [{"uuid": u, "handle": f"h/{u}", "name": None} for u in uuids]


def test_search_page_uses_offset_not_start():
    """Regression test for the real API quirk measured 2026-09-09: DOAB's
    ``/rest/search`` silently IGNORES a ``start=`` parameter and returns page
    1 again, but honours ``offset=``. A stub fetch stands in for the DOAB
    server: it returns the FULL set on offset=0 and an EMPTY page on any
    other offset, so a caller that (re-)introduces ``start=`` instead of
    ``offset=`` gets the same page forever and this test loops/duplicates
    rather than terminating cleanly.
    """
    all_uuids = [f"uuid-{i}" for i in range(5)]

    def stub_fetch(url):
        assert "start=" not in url, "must paginate on offset=, not start="
        if "offset=0" in url:
            body = json.dumps(_fixture_search_response(all_uuids)).encode()
            headers = {"x-total-count": str(len(all_uuids))}
        else:
            body = json.dumps([]).encode()
            headers = {"x-total-count": str(len(all_uuids))}
        return 200, body, url, headers

    got, total = M.enumerate_all_uuids(stub_fetch, 'publisher:"The MIT Press"',
                                       page_size=100)
    assert total == 5
    assert got == all_uuids


def test_enumerate_all_uuids_pages_across_multiple_offsets():
    page1 = [f"uuid-{i}" for i in range(3)]
    page2 = [f"uuid-{i}" for i in range(3, 5)]

    def stub_fetch(url):
        headers = {"x-total-count": "5"}
        if "offset=0" in url:
            return 200, json.dumps(_fixture_search_response(page1)).encode(), url, headers
        if "offset=3" in url:
            return 200, json.dumps(_fixture_search_response(page2)).encode(), url, headers
        return 200, json.dumps([]).encode(), url, headers

    got, total = M.enumerate_all_uuids(stub_fetch, 'publisher:"The MIT Press"',
                                       page_size=3)
    assert total == 5
    assert got == page1 + page2


def test_doab_search_page_raises_on_non_200():
    def stub_fetch(url):
        return 500, b"", url, {}

    try:
        M.doab_search_page(stub_fetch, "q", 0, 10)
        assert False, "expected DoabError"
    except M.DoabError:
        pass


# ---------------------------------------------------------------------------
# resolve_doi_url -- reads only the redirect header, never the body
# ---------------------------------------------------------------------------
def test_resolve_doi_url_follows_to_publisher_url_even_on_terminal_403():
    calls = []

    def stub_fetch(url):
        calls.append(url)
        # simulate urllib having already internally followed doi.org's 302
        # and landed on the Cloudflare-challenged 403 -- final_url is the
        # real publisher URL despite the non-2xx status.
        return 403, b"<html>Just a moment...</html>", DISTRIBUTIONAL_RL_URL, {}

    url = M.resolve_doi_url(stub_fetch, DISTRIBUTIONAL_RL["doi"])
    assert url == DISTRIBUTIONAL_RL_URL
    assert calls == [f"https://doi.org/{DISTRIBUTIONAL_RL['doi']}"]


def test_resolve_doi_url_with_no_doi_returns_none_without_a_request():
    def stub_fetch(url):
        raise AssertionError("must not fetch with no doi")

    assert M.resolve_doi_url(stub_fetch, None) is None


def test_resolve_doi_url_connection_failure_returns_none():
    def stub_fetch(url):
        return 0, b"", url, {}   # make_fetch's own encoding of a URLError

    assert M.resolve_doi_url(stub_fetch, "10.7551/mitpress/14207.001.0001") is None
