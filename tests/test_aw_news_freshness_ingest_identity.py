"""Task 432 — freshness_ingest must crawl as the identity it asked permission for.

THE DEFECT (fixed here): robots.txt was evaluated as `AuthoritativeWebBot`, but
the page was then fetched as `Mozilla/5.0 (compatible; AuthoritativeWebBot/1.0)`.
Two different identities: one to ask, another to act.

That is not UA hygiene, it is consent evidence that does not describe the request
we then make. Python's robotparser resolves an agent to the token BEFORE the
first slash, so the two strings are not interchangeable — the string we
transmitted resolved to the agent "mozilla" and did NOT match a rule a site
operator writes to control `AuthoritativeWebBot`. The operator could neither
write an effective rule against us nor recognise us in their logs.

`test_transmitted_ua_matches_the_rule_written_for_us` is the regression that
matters: it fails on the old code and passes on the new.
"""
from __future__ import annotations

import urllib.robotparser

import pytest

from cytir.news import freshness_ingest as fi  # noqa: E402


@pytest.fixture(autouse=True)
def _operator_identity(monkeypatch):
    monkeypatch.setenv("AW_OPERATOR_URL", "https://websear.ch")
    monkeypatch.setenv("AW_OPERATOR_CONTACT", "ops@example.test")

ROBOTS_FOR_US = """User-agent: AuthoritativeWebBot
Disallow: /private

User-agent: *
Allow: /
"""


def _capture(monkeypatch):
    """Record the User-Agent of every outbound request (the guarded client, task 1621)."""
    import httpx
    seen: list[str] = []

    def _handler(request):
        seen.append(request.headers.get("user-agent"))
        body = ROBOTS_FOR_US if request.url.path.endswith("/robots.txt") else "<html><p>x</p></html>"
        return httpx.Response(200, content=body.encode())

    inner = httpx.MockTransport(_handler)
    monkeypatch.setattr(httpx, "HTTPTransport", lambda *a, **k: inner)
    monkeypatch.setattr(fi, "_robots_cache", {})
    return seen


def test_robots_fetch_and_content_fetch_send_the_same_identity(monkeypatch):
    """The core of the fix: one identity, asked and acted on."""
    seen = _capture(monkeypatch)
    fi._robots("https://example.test")
    fi.refetch_text("https://example.test/page")
    assert len(seen) == 2, seen
    assert seen[0] == seen[1], f"split identity: robots={seen[0]!r} fetch={seen[1]!r}"
    assert seen[0] == fi._ua()


def test_transmitted_ua_matches_the_rule_written_for_us(monkeypatch):
    """The regression that prices the defect.

    A site operator writes `User-agent: AuthoritativeWebBot / Disallow: /private`.
    The UA we actually transmit must be caught by that rule. The old fetch UA
    resolved to the agent token "mozilla" and sailed straight past it.
    """
    seen = _capture(monkeypatch)
    fi.refetch_text("https://example.test/private")
    transmitted = seen[-1]

    rp = urllib.robotparser.RobotFileParser()
    rp.parse(ROBOTS_FOR_US.splitlines())
    assert rp.can_fetch(transmitted, "https://example.test/private") is False, (
        f"the UA we send ({transmitted!r}) is not matched by a robots rule "
        f"written for us — the operator cannot control or identify us")


def test_consent_evaluation_is_unchanged(monkeypatch):
    """The token before the slash did not move, so every robots decision this
    script has ever made is reproduced exactly."""
    _capture(monkeypatch)
    assert fi.consent_ok("https://example.test/private") is False
    assert fi.consent_ok("https://example.test/public") is True


def test_identity_does_not_impersonate_a_browser():
    for lie in ("Mozilla", "AppleWebKit", "Chrome/", "Safari/", "Gecko", "Firefox/"):
        assert lie not in fi._ua()


def test_identity_is_answerable():
    """A UA a host can act on: product, a URL to read, an address to write to."""
    assert fi._ua().startswith("AuthoritativeWebBot")
    assert "+https://" in fi._ua()
    assert "@" in fi._ua()


def test_no_second_ua_literal_survives_in_the_source():
    """A future edit must not reintroduce a second identity."""
    src = open(fi.__file__, encoding="utf-8").read()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert '"user-agent": _ua()' in code
    for lie in ("Mozilla/5.0", "AppleWebKit"):
        assert lie not in code, f"{lie} reintroduced outside a comment"


# --- weekly_delta shares the same identity (task 432) ------------------------

def test_weekly_delta_fetches_as_the_agent_it_gates_with():
    """weekly_delta.py gates embeds on freshness_ingest.consent_ok() but used to
    fetch with a LOCAL UA of its own whose robotparser token was `mozilla`. Ask
    as one crawler, fetch as another — the same defect, in a second file."""
    from cytir.news import weekly_delta as wd
    assert wd.fi is fi


def test_weekly_delta_transmitted_ua_matches_the_rule_written_for_us():
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(ROBOTS_FOR_US.splitlines())
    assert rp.can_fetch(fi._ua(), "https://example.test/private") is False


def test_weekly_delta_does_not_impersonate_a_browser():
    for lie in ("Mozilla", "AppleWebKit", "Chrome/", "Safari/", "Gecko"):
        assert lie not in fi._ua()
