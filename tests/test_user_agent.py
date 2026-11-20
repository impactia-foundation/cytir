"""Task 432 item 4 — the crawler must identify as whoever is running it.

~220 files hardcoded a User-Agent naming websear.ch and bender@impactia.org, so
a stranger running this pipeline identified as us to every host they crawled.
Site operators apply allow/deny rules, rate limits and abuse reports to that
token; our own per-host consent evidence is polluted by traffic we never sent.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cytir.core.user_agent as ua  # noqa: E402

AW_URL = "https://websear.ch"
AW_CONTACT = "bender@impactia.org"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.delenv("AW_OPERATOR_URL", raising=False)
    monkeypatch.delenv("AW_OPERATOR_CONTACT", raising=False)
    monkeypatch.setattr(ua, "CONFIG_FILE", tmp_path / "operator.json")
    monkeypatch.setattr(ua, "_warned", False)
    yield


def test_aw_identity_is_byte_identical_to_the_current_live_string(monkeypatch):
    """The no-op guarantee for AW's own traffic. This exact string is what
    consent_resolver.UA_STR sends today and what 28 files hardcode."""
    monkeypatch.setenv("AW_OPERATOR_URL", AW_URL)
    monkeypatch.setenv("AW_OPERATOR_CONTACT", AW_CONTACT)
    assert ua.user_agent() == "authoritative-web/0.1 (+https://websear.ch; bender@impactia.org)"


def test_unconfigured_refuses_by_default(monkeypatch):
    """Fail closed. A placeholder on the wire is still a wrong answer to
    "who is asking?", and a lane that starts unconfigured would write consent
    evidence describing a crawler that does not exist."""
    monkeypatch.delenv(ua.ALLOW_UNIDENTIFIED_ENV, raising=False)
    with pytest.raises(ua.UnidentifiedCrawler) as exc:
        ua.user_agent()
    msg = str(exc.value)
    assert "AW_OPERATOR_URL" in msg and "operator.json" in msg
    assert ua.ALLOW_UNIDENTIFIED_ENV in msg


def test_refusal_names_what_it_declined_to_send(monkeypatch):
    monkeypatch.delenv(ua.ALLOW_UNIDENTIFIED_ENV, raising=False)
    with pytest.raises(ua.UnidentifiedCrawler) as exc:
        ua.user_agent()
    assert "UNCONFIGURED" in str(exc.value)


def test_unconfigured_does_not_impersonate_anyone(monkeypatch):
    monkeypatch.setenv(ua.ALLOW_UNIDENTIFIED_ENV, "1")
    got = ua.user_agent()
    assert "websear.ch" not in got
    assert "impactia.org" not in got
    assert "UNCONFIGURED" in got


def test_unconfigured_warns_loudly_once(monkeypatch, capsys):
    monkeypatch.setenv(ua.ALLOW_UNIDENTIFIED_ENV, "1")
    ua.user_agent()
    ua.user_agent()
    err = capsys.readouterr().err
    assert "AW CRAWLER IDENTITY IS NOT CONFIGURED" in err
    assert err.count("AW CRAWLER IDENTITY IS NOT CONFIGURED") == 1


def test_never_impersonates_a_browser(monkeypatch):
    monkeypatch.setenv("AW_OPERATOR_URL", AW_URL)
    monkeypatch.setenv("AW_OPERATOR_CONTACT", AW_CONTACT)
    got = ua.user_agent()
    for lie in ("Mozilla", "AppleWebKit", "Chrome", "Safari", "Gecko"):
        assert lie not in got


def test_ua_is_answerable(monkeypatch):
    """Product, a URL a human can open, an address a human can write to."""
    monkeypatch.setenv("AW_OPERATOR_URL", "https://example.org/aw")
    monkeypatch.setenv("AW_OPERATOR_CONTACT", "ops@example.org")
    got = ua.user_agent()
    assert got.startswith("authoritative-web/")
    assert "+https://example.org/aw" in got
    assert "ops@example.org" in got


def test_config_file_supplies_identity(monkeypatch, tmp_path):
    cfg = tmp_path / "operator.json"
    cfg.write_text(json.dumps({"url": "https://mine.example", "contact": "me@example"}))
    monkeypatch.setattr(ua, "CONFIG_FILE", cfg)
    assert ua.user_agent() == "authoritative-web/0.1 (+https://mine.example; me@example)"
    assert ua.is_configured()


def test_env_beats_config_file(monkeypatch, tmp_path):
    cfg = tmp_path / "operator.json"
    cfg.write_text(json.dumps({"url": "https://file.example", "contact": "file@example"}))
    monkeypatch.setattr(ua, "CONFIG_FILE", cfg)
    monkeypatch.setenv("AW_OPERATOR_URL", "https://env.example")
    monkeypatch.setenv("AW_OPERATOR_CONTACT", "env@example")
    assert "env.example" in ua.user_agent()


def test_deliberate_unidentified_crawl_is_still_non_impersonating(monkeypatch):
    """The escape hatch must not become a licence to lie."""
    monkeypatch.setenv(ua.ALLOW_UNIDENTIFIED_ENV, "1")
    got = ua.user_agent()
    for lie in ("Mozilla", "AppleWebKit", "Chrome", "Safari", "Gecko"):
        assert lie not in got


def test_malformed_config_refuses_rather_than_falling_back(monkeypatch, tmp_path):
    """A silent fallback here means you do not know who you are crawling as."""
    cfg = tmp_path / "operator.json"
    cfg.write_text("{not json")
    monkeypatch.setattr(ua, "CONFIG_FILE", cfg)
    with pytest.raises(RuntimeError, match="could not be read as JSON"):
        ua.user_agent()


def test_half_configured_is_not_configured(monkeypatch):
    """A URL with no contact is not an answerable identity."""
    monkeypatch.setenv("AW_OPERATOR_URL", "https://mine.example")
    assert not ua.is_configured()
    with pytest.raises(RuntimeError, match="not configured"):
        ua.assert_configured("test")


def test_component_is_appended_never_substituted(monkeypatch):
    monkeypatch.setenv("AW_OPERATOR_URL", AW_URL)
    monkeypatch.setenv("AW_OPERATOR_CONTACT", AW_CONTACT)
    got = ua.user_agent("consent")
    assert got.endswith(" consent")
    assert AW_URL in got and AW_CONTACT in got


def test_cli_prints_the_same_string_for_shell_callers(monkeypatch):
    """~58 shell scripts cannot import Python. They call this, so there is still
    exactly one implementation."""
    env = {"AW_OPERATOR_URL": AW_URL, "AW_OPERATOR_CONTACT": AW_CONTACT,
           "PATH": "/usr/bin:/bin", "PYTHONPATH": str(HERE.parent)}
    out = subprocess.run([sys.executable, "-m", "cytir.core.user_agent"],
                         capture_output=True, text=True, env=env, timeout=60, cwd="/")
    assert out.stdout.strip() == "authoritative-web/0.1 (+https://websear.ch; bender@impactia.org)"


def test_example_config_carries_no_aw_identity():
    example = json.loads((HERE.parent / "operator.example.json").read_text())
    blob = json.dumps(example)
    assert "websear.ch" not in blob
    assert "impactia.org" not in blob
