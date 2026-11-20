#!/usr/bin/env python3
"""[test] ADR-106 / task 586 — authored prose is never a page's content.

The records here are REAL and still served, not invented. ``data/refresh_manifest.json``
is read from the repo: its 199 ``text`` bodies were written by a model on
2026-07-08/09 (tasks 106, 107b, 148) and production serves 370 rows of them at
the manifest's URLs (task 586's authored-rows artifact for aw_e3, an
exhaustive per-URL scroll of the twin). The candidate ``aw_e8`` carries 0 of
827 rows at those same URLs as authored text — which is why the flip removes
the population and why the held-out h@1 gap closes when it goes.

Nothing here fetches, and nothing reads a mirror: ``tests/conftest.py``'s
no-network fixture would fail the suite if anything tried.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
from cytir.corpus import authored_text_guard as ATG  # noqa: E402
from tests.helpers.repo_files import repo_file_or_skip  # noqa: E402

MANIFEST = os.path.join(APP, "data", "refresh_manifest.json")
POLICY = os.path.join(APP, "data", "authored-text-origins.v1.json")

#: the exact origin both writer lanes pass to persist_l1_records
LANE_ORIGIN = "freshness-manifest:curated text (data/refresh_manifest.json)"


@pytest.fixture(scope="module")
def policy():
    return ATG.load_policy(POLICY)


@pytest.fixture(scope="module")
def manifest():
    repo_file_or_skip("data/refresh_manifest.json")  # not shipped in the cut
    with open(MANIFEST, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------- the policy fails closed
def test_missing_policy_is_a_refusal_not_an_empty_list(tmp_path):
    """A guard that defaults to "nothing is authored" passes everything."""
    with pytest.raises(ATG.AuthoredTextError) as e:
        ATG.load_policy(str(tmp_path / "does-not-exist.json"))
    assert "Refusing to run" in str(e.value)


def test_policy_naming_no_origins_is_a_refusal(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"schema": ATG.POLICY_SCHEMA,
                             "authored_l1_origins": [],
                             "authored_l1_origin_prefixes": []}))
    with pytest.raises(ATG.AuthoredTextError):
        ATG.load_policy(str(p))


def test_unknown_origin_is_treated_as_authored(policy):
    """Unknown is not fetched. Every row that caused this predates the field."""
    assert ATG.is_authored_origin(None, policy) is True
    assert ATG.is_authored_origin("", policy) is True
    assert ATG.is_authored_origin("cc-warc:raw-html", policy) is False


def test_the_lanes_own_origin_string_is_covered(policy):
    """If the lane's literal is edited, this test is the thing that notices."""
    assert ATG.is_authored_origin(LANE_ORIGIN, policy) is True


# ------------------------------------------- the real record: the manifest
def test_the_real_manifest_body_is_refused_as_an_embedding_input(manifest, policy):
    entry = dict(manifest["sources"][0])
    entry["text_origin"] = LANE_ORIGIN
    with pytest.raises(ATG.AuthoredTextError) as e:
        ATG.assert_chunk_embeddable([entry], label="test", policy=policy)
    assert entry["url"] in str(e.value)


def test_a_fetched_row_at_the_same_url_is_admitted(manifest, policy):
    """The guard bans the ORIGIN, not the URL: a real fetch of the same page
    must still be indexable, or the fix would delete the source family."""
    entry = dict(manifest["sources"][0])
    entry["text_origin"] = "direct-fetch:html"
    ATG.assert_chunk_embeddable([entry], label="test", policy=policy)


def test_one_authored_member_refuses_the_whole_batch(manifest, policy):
    good = dict(manifest["sources"][1], text_origin="direct-fetch:html")
    bad = dict(manifest["sources"][2], text_origin=LANE_ORIGIN)
    with pytest.raises(ATG.AuthoredTextError):
        ATG.assert_chunk_embeddable([good, bad], label="test", policy=policy)


def test_manifest_roster_drops_every_indexable_body(manifest, policy):
    roster = ATG.manifest_roster(manifest, manifest_path=MANIFEST, policy=policy)
    assert len(roster) == len(manifest["sources"]) == 199
    for row in roster:
        assert "text" not in row and "snippet" not in row, \
            "a present-but-empty body is an invitation to fall back to it"
        assert row["url"]
    # the refresh signal survives -- the manifest keeps being a roster
    assert any(r.get("change_signal") for r in roster)


# --------------------------------------------------------- the stale guard
def test_expired_dated_claim_fails_the_refresh(policy):
    body = ("Latest Linux kernel is the 7.1 stable series (7.1.3, ~6 July "
            "2026); the current LTS is 6.18.38.")
    with pytest.raises(ATG.AuthoredTextError) as e:
        ATG.assert_claims_fresh(body, now=dt.date(2026, 9, 16),
                                label="kernel.org", policy=policy)
    assert "72 days old" in str(e.value)


def test_a_recent_claim_passes(policy):
    body = "Latest release 1.9.1, 2026-09-10."
    v = ATG.assert_claims_fresh(body, now=dt.date(2026, 9, 16),
                                label="t", policy=policy)
    assert v["age_days"] == 6 and v["verdict"] == "FRESH"


def test_an_undated_body_is_not_called_stale(policy):
    """No date is not evidence of staleness; it is evidence of no claim."""
    v = ATG.assert_claims_fresh("A page about pensions.",
                                now=dt.date(2026, 9, 16), label="t",
                                policy=policy)
    assert v["age_days"] is None


def test_month_only_claims_date_to_the_month_end(policy):
    """"June 2026" must not go stale on 1 July: err towards fresh."""
    assert ATG.claim_age_days("as of June 2026", now=dt.date(2026, 7, 1)) == 1


@pytest.mark.parametrize("body,expect", [
    ("publié le 30 juin 2026", dt.date(2026, 6, 30)),
    ("Stand: 1. Juli 2026", dt.date(2026, 7, 1)),
    ("aggiornato a maggio 2026", dt.date(2026, 5, 31)),
    ("released June 13, 2026", dt.date(2026, 6, 13)),
    ("2026-04-02", dt.date(2026, 4, 2)),
])
def test_dates_are_read_in_all_four_indexed_languages(body, expect):
    """An English-only matcher would read a French statistic as undated and
    pass it forever -- and 3 of the 4 indexed languages are not English."""
    claims = ATG.dated_claims(body)
    assert claims and claims[-1]["date"] == expect


def test_the_newest_claim_governs():
    body = "Released in 2019-01-01. The current version, 2026-08-01, adds X."
    assert ATG.claim_age_days(body, now=dt.date(2026, 8, 15)) == 14


def test_the_real_manifest_is_measurably_stale(manifest, policy):
    """The measurement, not an anecdote: as of the day this landed, most of the
    manifest's bodies had already expired against their own 30-day bar."""
    fields = ("text", "snippet")
    now = dt.date(2026, 9, 16)
    expired = sum(
        1 for e in manifest["sources"]
        if (ATG.claim_age_days(" ".join(str(e.get(f) or "") for f in fields),
                               now=now) or 0)
        > policy["stale_claim_max_age_days"])
    assert expired == 118, (
        "118 of 199 manifest bodies carried an expired dated claim on "
        "2026-09-16; if this number moved, the manifest was edited and the "
        "rows production serves were not")


# ------------------------------------------------------------- L1 refusals
def test_l1_batch_with_an_authored_origin_is_refused(policy):
    recs = [{"url": "https://react.dev/versions", "l1_origin": LANE_ORIGIN}]
    with pytest.raises(ATG.AuthoredTextError) as e:
        ATG.assert_l1_records_fetched(recs, label="t", policy=policy)
    assert "replays L1 as content" in str(e.value) or "rebuild" in str(e.value)


def test_l1_accounting_reports_its_input_count(policy):
    recs = [{"url": "https://a/", "l1_origin": "direct-fetch:html"},
            {"url": "https://b/", "l1_origin": "cc-warc:raw-html"}]
    v = ATG.assert_l1_records_fetched(recs, label="t", policy=policy)
    assert v["checked"] == 2 and v["admitted"] == 2 and v["refused"] == 0


def test_fetched_text_never_falls_back_to_the_authored_record(policy):
    url = "https://react.dev/versions"
    l1 = [{"url": url, "l1_origin": LANE_ORIGIN, "markdown": "authored prose"}]
    assert ATG.fetched_text_for(url, l1, policy) is None, \
        "None means do not write the row; it never means use the manifest body"
    l1.append({"url": url, "l1_origin": "direct-fetch:html",
               "markdown": "the real page"})
    assert ATG.fetched_text_for(url, l1, policy) == "the real page"


# ------------------------------- the writer paths, checked on the real code
def test_persist_l1_records_refuses_the_lanes_authored_origin(tmp_path):
    """The ADR-105 fix filed the lane's own prose so its guard would pass.
    That door is now shut at the filing module, not only at the caller."""
    from cytir.corpus import persist_l1_records as PL1
    with pytest.raises(PL1.L1PersistError) as e:
        PL1.write_l1_records(
            [{"url": "https://react.dev/versions", "text": "authored prose"}],
            release="test", source="test", origin=LANE_ORIGIN,
            root=str(tmp_path))
    assert "authored" in str(e.value)


def test_build_points_calls_the_guard_before_the_embedder(monkeypatch):
    """A guard with a test and no caller is decoration (verification-doctrine
    12). This asserts the CALL, on the real function, and that it happens
    BEFORE any text reaches the embedder."""
    from cytir.news import freshness_ingest as FI
    order = []

    def fake_guard(chunk, **kw):
        order.append("guard")
        raise ATG.AuthoredTextError("refused")

    monkeypatch.setattr(FI.ATG, "assert_chunk_embeddable", fake_guard)
    monkeypatch.setattr(FI, "embed_tei",
                        lambda *a, **k: order.append("embed") or [[0.0]])
    with pytest.raises(ATG.AuthoredTextError):
        FI.build_points(None, [{"url": "https://x/", "text": "t"}],
                        "http://tei", "2026-09-16")
    assert order == ["guard"], "the embedder ran before, or despite, the guard"


def test_no_writer_path_embeds_the_manifest_body_unguarded():
    """The class, not the instance: any module that reads the manifest's
    indexable fields AND builds an embedding input must route through the
    guard. A new lane that copies the old one fails here."""
    import re
    policy = ATG.load_policy(POLICY)
    fields = set(policy["authored_manifests"][0]["text_fields"])
    reads_body = re.compile(
        r"""\[\s*["'](%s)["']\s*\]|\.get\(\s*["'](%s)["']"""
        % ("|".join(fields), "|".join(fields)))
    embeds = re.compile(r"embed_tei|bm25_tokens|build_points\(")
    offenders = []
    for name in sorted(os.listdir(HERE)):
        if not name.endswith(".py") or name.startswith("test_"):
            continue
        if name in ("authored_text_guard.py",):
            continue
        src = open(os.path.join(HERE, name), encoding="utf-8").read()
        if "refresh_manifest" not in src:
            continue
        if reads_body.search(src) and embeds.search(src):
            if "authored_text_guard" not in src and "ATG." not in src:
                offenders.append(name)
    assert not offenders, (
        f"{offenders} read the manifest's authored body into an embedding "
        f"path without the ADR-106 guard")
