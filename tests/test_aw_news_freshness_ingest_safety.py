"""Safety contracts shared by daily and weekly legacy freshness jobs."""
from __future__ import annotations

import pytest

from cytir.news import freshness_ingest as fi


@pytest.fixture(autouse=True)
def _operator_identity(monkeypatch):
    monkeypatch.setenv("AW_OPERATOR_URL", "https://websear.ch")
    monkeypatch.setenv("AW_OPERATOR_CONTACT", "ops@example.test")


def test_unreachable_robots_is_held_not_default_allowed(monkeypatch):
    # The pre-2026-08-03 implementation returned True here and silently wrote
    # an embedding on a transient robots/WAF failure.
    monkeypatch.setattr(fi, "_robots", lambda _scheme: None)
    assert fi.consent_ok("https://example.test/current") is False


def test_robots_disallow_is_held():
    class Deny:
        def can_fetch(self, _ua, _url):
            return False
    fi._robots_cache.clear()
    # Avoid network and preserve the exact URL-level can_fetch boundary.
    original = fi._robots
    try:
        fi._robots = lambda _scheme: Deny()
        assert fi.consent_ok("https://example.test/current") is False
    finally:
        fi._robots = original
