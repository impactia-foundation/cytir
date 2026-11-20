"""The builder consent gate in its package home (`cytir.consent.build_gate`).

Values are copied from the builder's dump-basis tests; the full matrix stays
there. These three cases pin the move: a gov-host decision, a dump-basis
evidence record, and a refusal."""
from __future__ import annotations

from cytir.consent import build_gate as BG

HOST = "www.legifrance.gouv.fr"

DUMP_PAYLOAD = {
    "url": "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000051962161",
    "domain": "legifrance.gouv.fr",
    "license": "Etalab-2.0",
    "consent_state": "dump",
    "consent_basis": "official-dump-license",
    "consent_source": "dila-opendata echanges.dila.gouv.fr (Licence Ouverte v2.0)",
    "consent_checked_at": "2026-07-25",
}
DECLINED_REC = {"status": "ok", "search_ok": False, "train_ok": False,
                "reason": "declined"}


def test_gov_host_decision():
    assert BG._is_gov_host("admin.ch")
    assert BG._is_gov_host("www.admin.ch")
    assert not BG._is_gov_host("notadmin.ch")
    assert BG._host_of({"url": "https://WWW.Admin.ch/x"}) == "www.admin.ch"


def test_dump_basis_evidence_record():
    assert BG.DUMP_CONSENT_STATE == "dump"
    assert BG.DUMP_CONSENT_BASES == frozenset({"official-dump-license"})
    assert BG.dump_basis_evidence(DUMP_PAYLOAD) == {
        "consent_basis": "official-dump-license",
        "consent_source": DUMP_PAYLOAD["consent_source"],
        "license": "Etalab-2.0"}


def test_refusal_and_admission_on_a_host_decline():
    assert BG.dump_basis_evidence({**DUMP_PAYLOAD, "consent_basis": "trust-me"}) is None
    gate = BG.ConsentGate(mode="cache", grace_days=30, cache_path="")
    gate.host_cache = {HOST: DECLINED_REC}
    admit, _ = gate.verdict(HOST, None, payload=DUMP_PAYLOAD)
    assert admit is True
    refused, reason = gate.verdict(HOST, None, payload={"url": "https://x"})
    assert refused is False and "declined" in reason
