"""ADR-115: per-record consent-flag derivation, one test per ruling clause.

Rows are REAL twin payloads (sampled from `aw_e8` on the build box, 2026-09-30, fields
trimmed to the ones the function reads). The contract table is the real
`data/source-family-contracts.v1.json`. Cache records are real cache files, minus the
robots body.

Run: python3 -m pytest tests/test_consent_derive_adr115.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

from cytir.consent import arxiv_licence, backfill
from cytir.consent.derive import (DeriveContext, cache_lookup, derive_consent_flags,
                               paper_licence_train_ok)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
from consent_adr115_rows import (AS_OF, ARXIV, DILA, FAMILIES, HN, IAIS_CARRIED, LODA_DUMP,  # noqa: E402,F401
                                 PUBMED, RII, SE, WEKO_CARRIED, WIKI, ctx, rec)


def test_dump_bases_match_the_gate():
    src = (REPO / "cytir" / "consent" / "build_gate.py").read_text()
    from cytir.consent.derive import DUMP_CONSENT_BASES
    assert 'DUMP_CONSENT_BASES = frozenset({"official-dump-license"})' in src
    assert DUMP_CONSENT_BASES == {"official-dump-license"}


# --- (1) dump-route families: the licence instrument ------------------------------

def test_r1_wikipedia_is_true_true_from_its_licence():
    d = derive_consent_flags(WIKI, ctx())
    assert (d.klass, d.rule, d.search_ok, d.train_ok) == ("A", "r1-wikipedia", True, True)
    assert d.writes() == {"search_ok": True, "train_ok": True, "consent_checked_at": AS_OF,
                          "consent_basis": "dump-licence"}


def test_r1_wikipedia_host_verdict_is_never_the_evidence():
    # a cache record that says search_ok=false must not move a Wikipedia row
    d = derive_consent_flags(WIKI, ctx({"en.wikipedia.org": rec(False, False)}))
    assert (d.search_ok, d.train_ok) == (True, True)


def test_r1_wikipedia_without_the_licence_is_unknown_not_a_guess():
    # without the CC-BY-SA licence the row is not claimed as a dump row; with no host
    # record either, nothing is written.
    d = derive_consent_flags({**WIKI, "license": "MIT"}, ctx())
    assert d.klass == "D" and d.writes() == {}


@pytest.mark.parametrize("lic,train,label", [
    ("http://creativecommons.org/licenses/by/4.0/", True, "cc-by"),
    ("http://creativecommons.org/licenses/by/3.0/", True, "cc-by"),
    ("http://creativecommons.org/licenses/by-sa/4.0/", True, "cc-by-sa"),
    ("http://creativecommons.org/publicdomain/zero/1.0/", True, "cc0"),
    ("http://creativecommons.org/publicdomain/mark/1.0/", True, "public-domain-mark"),
    ("http://creativecommons.org/licenses/publicdomain/", True, "public-domain"),
    ("http://creativecommons.org/licenses/by-nc-sa/3.0/", False, "cc-by-nc-sa"),
    ("http://creativecommons.org/licenses/by-nc-sa/4.0/", False, "cc-by-nc-sa"),
    ("http://creativecommons.org/licenses/by-nc-nd/4.0/", False, "cc-by-nc-nd"),
    ("http://creativecommons.org/licenses/by-nc/4.0/", False, "cc-by-nc"),
    ("http://creativecommons.org/licenses/by-nd/4.0/", False, "cc-by-nd"),
    ("http://arxiv.org/licenses/nonexclusive-distrib/1.0/", False, "arxiv-nonexclusive"),
    ("", False, "missing"),
    (None, False, "missing"),
    ("http://example.org/some-new-licence/1.0/", False, "unrecognised"),
])
def test_arxiv_paper_licence_table_fails_closed(lic, train, label):
    assert paper_licence_train_ok(lic) == (train, label)


def test_r1_arxiv_search_true_train_follows_the_paper_licence():
    by = {"2503.21080": "http://creativecommons.org/licenses/by/4.0/"}
    d = derive_consent_flags(ARXIV, ctx(arxiv=by))
    assert (d.rule, d.search_ok, d.train_ok, d.detail) == ("r1-arxiv", True, True, "cc-by")
    nc = {"2503.21080": "http://creativecommons.org/licenses/by-nc-nd/4.0/"}
    d = derive_consent_flags(ARXIV, ctx(arxiv=nc))
    assert (d.search_ok, d.train_ok) == (True, False)


def test_r1_arxiv_missing_licence_fails_closed_on_train_only():
    d = derive_consent_flags(ARXIV, ctx(arxiv={}))          # the map has no record of this paper
    assert (d.klass, d.search_ok, d.train_ok, d.detail) == ("A", True, False, "missing")
    d = derive_consent_flags(ARXIV, ctx(arxiv=None))        # no map at all: same, never true
    assert d.train_ok is False


def test_r1_arxiv_metadata_licence_string_is_not_the_paper_licence():
    # the row says arXiv-metadata-CC0; that licenses the metadata snapshot, not the paper
    d = derive_consent_flags(ARXIV, ctx(arxiv={"2503.21080": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"}))
    assert d.train_ok is False


def test_r1_arxiv_oai_page_parses_licences_and_old_papers_have_none():
    page = """<OAI-PMH><ListRecords>
    <record><header><identifier>oai:arXiv.org:astro-ph/0008137</identifier></header><metadata>
      <arXiv><id>astro-ph/0008137</id><created>2000-08-09</created><title>ADONIS</title></arXiv></metadata></record>
    <record><header><identifier>oai:arXiv.org:2503.21080</identifier></header><metadata>
      <arXiv><id>2503.21080</id><title>EmoDebt</title>
      <license>http://creativecommons.org/licenses/by/4.0/</license></arXiv></metadata></record>
    <record><header status="deleted"><identifier>oai:arXiv.org:0000.0000</identifier></header></record>
    </ListRecords></OAI-PMH>"""
    assert list(arxiv_licence.parse_page(page)) == [
        ("astro-ph/0008137", ""), ("2503.21080", "http://creativecommons.org/licenses/by/4.0/")]


def test_r1_pubmed_is_search_true_train_false_when_no_pmc_record():
    d = derive_consent_flags(PUBMED, ctx())
    assert (d.rule, d.search_ok, d.train_ok, d.detail) == ("r1-pubmed", True, False, "no-pmc-oa-record")


def test_r1_pubmed_open_access_under_a_permitting_licence_may_train():
    ok = {"19815838": "https://creativecommons.org/licenses/by/4.0/"}
    assert derive_consent_flags(PUBMED, ctx(pmc=ok)).train_ok is True
    nc = {"19815838": "https://creativecommons.org/licenses/by-nc/4.0/"}
    assert derive_consent_flags(PUBMED, ctx(pmc=nc)).train_ok is False


# --- (2) Stack Exchange ----------------------------------------------------------------

def test_r2_stack_exchange_cc_by_sa_is_true_true():
    d = derive_consent_flags(SE, ctx())
    assert (d.rule, d.search_ok, d.train_ok) == ("r2-stackexchange", True, True)


# --- (3) carried rows: the host's CURRENT verdict + its checked_at -----------------------------

def test_r3_carried_row_takes_the_hosts_current_verdict_and_its_checked_at():
    d = derive_consent_flags(WEKO_CARRIED, ctx({"www.weko.admin.ch": rec(True, True)}))
    assert d.klass == "B" and d.consent_checked_at == "2026-07-19T09:30:38Z"
    assert d.consent_basis == "host-cache" and (d.search_ok, d.train_ok) == (True, True)


def test_r3_carried_row_whose_host_now_declines_gets_the_current_false_flags():
    d = derive_consent_flags(WEKO_CARRIED, ctx({"www.weko.admin.ch": rec(False, False)}))
    assert (d.klass, d.search_ok, d.train_ok) == ("B", False, False)   # retention is ADR-103's, not ours


def test_r3_a_served_non_carried_row_whose_host_declines_is_not_ruled_and_stays_d():
    row = {**WEKO_CARRIED, "consent_state": "", "consent_basis": ""}
    d = derive_consent_flags(row, ctx({"www.weko.admin.ch": rec(False, False)}))
    assert d.klass == "D" and d.writes() == {}


def test_r3_no_record_no_flags():
    d = derive_consent_flags(IAIS_CARRIED, ctx())
    assert d.klass == "D" and d.writes() == {}


def test_r3_unusable_cache_records_are_no_record():
    for bad in (rec(True, True, status="unknown"), rec(True, True, at=None), rec(None, True), rec(True, None)):
        assert derive_consent_flags(IAIS_CARRIED, ctx({"www.iais.org": bad})).klass == "D"


# --- (4) www -> apex only ---------------------------------------------------------------------

def test_r4_a_www_record_stands_for_the_apex_when_the_apex_has_none():
    row = {"url": "https://weko.admin.ch/x", "domain": "weko.admin.ch", "provenance": "weko_docs"}
    d = derive_consent_flags(row, ctx({"www.weko.admin.ch": rec(True, False)}))
    assert (d.klass, d.consent_basis, d.detail) == ("B", "host-cache-www-apex", "www-apex:cc")
    assert d.train_ok is False


def test_r4_the_apex_record_wins_over_the_www_record():
    row = {"url": "https://weko.admin.ch/x", "domain": "weko.admin.ch", "provenance": "weko_docs"}
    d = derive_consent_flags(row, ctx({"weko.admin.ch": rec(True, True, "2026-09-01T00:00:00Z"),
                                        "www.weko.admin.ch": rec(False, False)}))
    assert (d.search_ok, d.consent_basis, d.consent_checked_at) == (True, "host-cache", "2026-09-01T00:00:00Z")


def test_r4_the_reverse_direction_is_not_approved():
    row = {"url": "https://www.weko.admin.ch/x", "domain": "weko.admin.ch", "provenance": "weko_docs"}
    d = derive_consent_flags(row, ctx({"weko.admin.ch": rec(True, True)}))
    assert d.klass == "D" and cache_lookup({"weko.admin.ch": rec(True, True)}, "www.weko.admin.ch")[1] == "miss"


def test_r4_a_parent_domain_record_is_never_used():
    row = {"url": "https://sub.weko.admin.ch/x", "domain": "sub.weko.admin.ch", "provenance": "weko_docs"}
    assert derive_consent_flags(row, ctx({"weko.admin.ch": rec(True, True)})).klass == "D"


# --- (5) official dumps: the family basis, never the host verdict -----------------------------------

@pytest.mark.parametrize("row", DILA + [RII], ids=lambda r: r["provenance"])
def test_r5_official_dump_rows_ignore_a_host_that_declines(row):
    host = "www.legifrance.gouv.fr" if "legifrance" in row["url"] else "www.rechtsprechung-im-internet.de"
    d = derive_consent_flags(row, ctx({host: rec(False, False)}))
    assert (d.klass, d.rule, d.search_ok, d.train_ok) == ("A", "r5-official-dump", True, True)
    assert d.consent_basis == "official-dump-licence-family"


def test_r5_a_dump_row_keeps_its_adr076_basis_string():
    d = derive_consent_flags(LODA_DUMP, ctx({"www.legifrance.gouv.fr": rec(False, False)}))
    assert d.consent_basis == "official-dump-license" and d.search_ok is True


def test_r5_a_row_without_the_dump_licence_does_not_claim_the_dump_basis():
    d = derive_consent_flags({**DILA[0], "license": ""}, ctx({"www.legifrance.gouv.fr": rec(False, False)}))
    assert d.klass == "D"


# --- E, one-flag, retired contracts, determinism --------------------------------------------------

def test_hn_rows_keep_their_official_api_flags():
    d = derive_consent_flags(HN, ctx())
    assert d.klass == "E" and d.writes() == {}


def test_a_retired_contract_never_stamps_a_row():
    # hn_firebase_pointer declares false/false and is RETIRED (a refused 406-row pack);
    # the ADR-108 HN rows are a different family. The contract must not reach any row.
    hn_contract = next(f for f in FAMILIES if f["family_id"] == "hn_firebase_pointer")
    assert hn_contract["status"] == "retired"
    row = {"url": "https://example.org/x", "domain": "example.org", "provenance": "hn_firebase_pointer"}
    d = derive_consent_flags(row, ctx())
    assert d.klass == "D" and d.writes() == {}


def test_a_one_flag_row_is_never_completed():
    d = derive_consent_flags({**WIKI, "search_ok": True}, ctx())
    assert d.klass == "D" and d.writes() == {}


def test_derivation_is_pure_and_repeatable():
    c = ctx({"www.weko.admin.ch": rec(True, True)})
    assert derive_consent_flags(WEKO_CARRIED, c) == derive_consent_flags(dict(WEKO_CARRIED), c)


# --- the dry-run reader ---------------------------------------------------------------------------

class FakeQdrant:
    def __init__(self, rows):
        self.rows = rows

    def __call__(self, url, body, timeout=0, tries=0):
        if url.endswith("/points/count"):
            return {"result": {"count": len(self.rows)}}
        off = body.get("offset") or 0
        page = self.rows[off:off + body["limit"]]
        nxt = off + body["limit"]
        return {"result": {"points": [{"payload": r} for r in page],
                           "next_page_offset": nxt if nxt < len(self.rows) else None}}


def test_dry_run_conserves_rows_and_memo_equals_direct(monkeypatch):
    rows = [WIKI, SE, ARXIV, PUBMED, HN, WEKO_CARRIED, IAIS_CARRIED, LODA_DUMP, RII, *DILA, dict(WIKI), dict(SE)]
    monkeypatch.setattr(backfill, "_post", FakeQdrant(rows))
    c = ctx({"www.weko.admin.ch": rec(True, True)}, arxiv={"2503.21080": "http://creativecommons.org/licenses/by/4.0/"})
    doc = backfill.run_derive("http://x", "t", c, page=4, log=lambda m: None)
    assert doc["scan_complete"] and doc["rows_scanned"] == len(rows) and doc["applied"] is False
    assert sum(doc["classes"].values()) == len(rows)
    direct = {}
    for r in rows:
        k = derive_consent_flags(r, c).klass
        direct[k] = direct.get(k, 0) + 1
    assert doc["classes"] == direct
    assert doc["classes"]["D"] == 1 and doc["classes"]["E"] == 1      # IAIS stays D; HN is E


def test_dry_run_refuses_apply(tmp_path, capsys):
    rc = backfill.main(["derive", "--qdrant", "http://x", "--collection", "t", "--contracts", "c",
                        "--cache-dir", "d", "--arxiv-licences", "a", "--as-of", AS_OF,
                        "--out", str(tmp_path / "o"), "--apply"])
    assert rc == 2 and "refusing" in capsys.readouterr().err
