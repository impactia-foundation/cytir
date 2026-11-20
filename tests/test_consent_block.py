#!/usr/bin/env python3
"""Tests for the consent-block guard (task 433).

The guard exists because two ingesters recorded a consent decision in prose and
enforced it nowhere. These tests assert the enforcement, the fail-closed
behaviour, and -- as acceptance tests -- the two originally-reported instances.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cytir.consent.block import (  # noqa: E402
    BLOCKING_STATES, ConsentBlocked, ConsentRegistryError, assert_crawlable,
    is_blocked, load_registry, lookup, registry_path,
)

REPO = os.path.normpath(os.path.join(HERE, ".."))


# --------------------------------------------------------------- the registry
def test_registry_loads_and_is_wellformed():
    hosts = load_registry()
    assert hosts, "registry must not be empty"
    for host, entry in hosts.items():
        assert host == host.lower().strip(), f"{host} must be normalised"
        assert entry["state"] in ("declined", "unknown", "ok"), host
        # every block must carry its evidence and its exit condition
        if entry["state"] in BLOCKING_STATES:
            for field in ("decided_on", "basis", "recorded_in",
                          "unblock_condition"):
                assert entry.get(field), f"{host} missing {field}"


def test_registry_is_valid_json_on_disk():
    with open(registry_path(), encoding="utf-8") as fh:
        json.load(fh)


# ------------------------------------------------- the reported instances (P9)
def test_prestations_vd_is_blocked():
    """Finding 1: the (deleted, task 1552) ingest_vd.py docstring marked this host CONSENT-BLOCKED."""
    assert is_blocked("prestations.vd.ch")
    with pytest.raises(ConsentBlocked) as e:
        assert_crawlable("prestations.vd.ch")
    msg = str(e.value)
    assert "CONSENT-BLOCKED" in msg
    assert "2026-07-09" in msg
    assert "un-block condition" in msg


@pytest.mark.parametrize("host", [
    "gesetze.berlin.de",            # BE
    "landesrecht-mv.de",            # MV
    "landesrecht.rlp.de",           # RP
    "recht.saarland.de",            # SL
    "landesrecht.thueringen.de",    # TH
])
def test_declined_laender_are_blocked(host):
    assert is_blocked(host)
    with pytest.raises(ConsentBlocked):
        assert_crawlable(host)


@pytest.mark.parametrize("host", [
    "landesrecht-hamburg.de",       # HH
    "landesrecht.hessen.de",        # HE
])
def test_unknown_consent_is_also_blocked(host):
    """Consent that could not be established is not consent."""
    assert lookup(host)[1]["state"] == "unknown"
    with pytest.raises(ConsentBlocked):
        assert_crawlable(host)


@pytest.mark.parametrize("host", [
    # SH remains admitted: its consent is genuinely UNKNOWN (its cached body is
    # juris.de's robots, reached via a cross-host redirect, so there is no
    # own-host evidence either way) but nobody has authorised blocking it.
    # Task 433 Part 6 flags it as the one live crawl path with no valid
    # evidence. If it is ruled on, add it to the registry.
    "gesetze-rechtsprechung.sh.juris.de",   # SH
    "fedlex.admin.ch",
])
def test_admitted_hosts_are_not_blocked(host):
    assert not is_blocked(host)
    assert_crawlable(host)          # must not raise


@pytest.mark.parametrize("host", [
    "landesrecht.sachsen-anhalt.de",        # the key the ingester crawls
    "www.landesrecht.sachsen-anhalt.de",    # the key carrying the robots
])
def test_st_is_blocked_under_both_host_keys(host):
    """ST: authorised 2026-08-19 on Part-6 evidence.

    The www key carries a valid robots document that declines (named-bot
    allowlist then `User-agent: * / Disallow: /`); the bare key the ingester
    uses has no valid own-host evidence at all. Registered under the bare host;
    _candidates() must map the www form onto it so neither key can crawl.
    """
    assert is_blocked(host)
    with pytest.raises(ConsentBlocked):
        assert_crawlable(host)


def test_bw_host_is_blocked_by_the_later_exact_host_verdict():
    """The 2026-07-09 OK for BW was parsed from an HTML page served as 200.

    The exact host the ingester crawls has a later cached verdict of
    search_ok=false/train_ok=false (operator-box probe, 2026-07-14) backed by the
    portal's real robots, which allowlists named engines and disallows
    everyone else at root. Enforce the later, evidence-backed verdict.
    """
    assert is_blocked("www.landesrecht-bw.de")
    with pytest.raises(ConsentBlocked):
        assert_crawlable("www.landesrecht-bw.de")


# ------------------------------------------------------------- normalisation
@pytest.mark.parametrize("variant", [
    "prestations.vd.ch",
    "PRESTATIONS.VD.CH",
    "https://prestations.vd.ch/pub/blv-publication",
    "prestations.vd.ch:443",
    "prestations.vd.ch.",
    " prestations.vd.ch ",
    "http://user@prestations.vd.ch/x?y=1",
])
def test_blocked_host_variants_all_match(variant):
    assert is_blocked(variant), f"{variant!r} must resolve to the blocked host"


def test_subdomain_of_blocked_domain_is_blocked():
    """A decline at a domain covers hosts beneath it."""
    assert is_blocked("www.landesrecht.rlp.de")
    assert is_blocked("api.gesetze.berlin.de")


def test_unrelated_host_sharing_a_suffix_is_not_blocked():
    assert not is_blocked("vd.ch")            # www.vd.ch is a different, open host
    assert not is_blocked("berlin.de")


def test_empty_host_is_not_matched():
    assert not is_blocked("")


# ------------------------------------------------------------- fail closed
def test_missing_registry_raises_rather_than_allowing(tmp_path):
    missing = str(tmp_path / "nope.json")
    with pytest.raises(ConsentRegistryError):
        assert_crawlable("prestations.vd.ch", path=missing)


def test_malformed_registry_raises_rather_than_allowing(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConsentRegistryError):
        assert_crawlable("example.com", path=str(bad))


def test_registry_without_hosts_key_raises(tmp_path):
    bad = tmp_path / "empty.json"
    bad.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(ConsentRegistryError):
        assert_crawlable("example.com", path=str(bad))


def test_guard_makes_no_network_calls(monkeypatch):
    """The guard must never resolve consent by asking the host."""
    import socket

    def boom(*a, **k):
        raise AssertionError("consent_block attempted a network call")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    assert is_blocked("prestations.vd.ch")
    assert not is_blocked("fedlex.admin.ch")


def test_no_env_bypass_exists(monkeypatch):
    """Un-blocking is a registry edit via PR, not a runtime flag."""
    for var in ("AW_CONSENT_FORCE", "AW_IGNORE_CONSENT", "AW_CONSENT_BYPASS",
                "FORCE", "AW_SKIP_CONSENT"):
        monkeypatch.setenv(var, "1")
    with pytest.raises(ConsentBlocked):
        assert_crawlable("prestations.vd.ch")


# ------------------------------------------- the ingesters actually enforce it
def test_package_fetch_choke_point_refuses_vd_before_any_client_use():
    """task 1552: the Vaud guard now lives in the package. resolve_consent runs
    consent_block.assert_crawlable before any client is touched."""
    from cytir.sources import runner
    class _NoNetwork:
        def __getattr__(self, name):
            raise AssertionError("client used before consent refusal")
    with pytest.raises(ConsentBlocked):
        runner.resolve_consent("prestations.vd.ch", direct_crawl=True, client=_NoNetwork())


def test_laender_portal_refuses_declined_host():
    import cytir.sources.families.de_laender_juris as m
    with pytest.raises(ConsentBlocked):
        m.Portal("gesetze.berlin.de", "bsbe")


def test_only_sh_remains_crawlable_of_the_three_laender():
    """BW and ST must now refuse; SH stays until someone rules on it."""
    import cytir.sources.families.de_laender_juris as m
    assert_crawlable(m.LANDER["SH"]["host"])            # must not raise
    for code in ("BW", "ST"):
        with pytest.raises(ConsentBlocked):
            m.Portal(m.LANDER[code]["host"], m.LANDER[code]["cid"])


def test_laender_cli_rejects_declined_code_loudly():
    r = subprocess.run(
        [sys.executable, "-m", "cytir.sources.families.de_laender_juris",
         "--land", "BE", "--out", "/tmp/aw_t433_laender_should_not_exist.jsonl"],
        cwd=os.path.join(HERE, ".."), capture_output=True, text=True, timeout=60)
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert "consent-DECLINED" in out or "non-admitted" in out
    assert not os.path.exists("/tmp/aw_t433_laender_should_not_exist.jsonl")


# ------------------------------------------------------------------------
# A consent claim in a comment must carry: what was observed, when, from what
# evidence, and what the code then did. These guard against the claim being
# re-truncated back to a bare verdict -- the bare verdict in ingest_vd.py was
# read as a confession of knowing misconduct and was relayed as such before the
# task-433 forensics established the ordering.
# ------------------------------------------------------------------------

def _doc(name):
    """Module docstring with whitespace collapsed, so re-wrapping a comment
    (which is harmless) does not fail these tests, while deleting the content
    (which is not harmless) still does."""
    path = (os.path.join(HERE, "..", "cytir", "sources", "families", "de_laender_juris.py")
            if name == "ingest_de_laender_juris.py" else os.path.join(HERE, name))  # moved: task 1414
    raw = open(path, encoding="utf-8").read().split('"""')[1]
    return re.sub(r"\s+", " ", raw)


def test_laender_docstring_carries_evidence_and_ordering():
    d = _doc("ingest_de_laender_juris.py")
    assert "19:27" in d and "19:50" in d, "consent-before-ingest ordering"
    assert "6.5 HOURS LATER" in d, "the recon-before-consent exposure"
    assert "no opt-out signal found" in d, "how BW's verdict was manufactured"
    assert "tdm-reservation" in d, "HH is a decline we could not read"
    # the bare, unsafe claim must not have been restored
    assert "consent-OK (fresh consent_resolver 2026-07-09) on THREE" not in d


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
