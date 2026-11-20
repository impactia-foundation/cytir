"""Task 432 — the identity we ASK PERMISSION with is the one we crawl with.

`consent_resolver.py` is the file that decides, per host, whether AW may fetch.
The string it sends is the requester every one of those verdicts is about. If a
migration ever changed it, every consent record afterwards would describe a
crawler that does not exist — the same defect class fixed in `freshness_ingest.py`
and `weekly_delta.py`, but at the point where permission is established rather
than acted on.

These tests pin two things the task-432 UA migration must never break:
  1. the emitted string is byte-identical to what production has always sent;
  2. the robots-evaluation token still resolves to the same agent as the UA.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import urllib.robotparser

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import cytir.consent.resolver as cr  # noqa: E402

#: The literal that stood in consent_resolver.py before task 432, and the string
#: the 6,403-host task-427 pass actually transmitted.
PRODUCTION_UA = "authoritative-web/0.1 (+https://websear.ch; bender@impactia.org)"


def _configured(monkeypatch):
    monkeypatch.setenv("AW_OPERATOR_URL", "https://websear.ch")
    monkeypatch.setenv("AW_OPERATOR_CONTACT", "bender@impactia.org")


_PROBE = "import cytir.consent.resolver as cr; print(cr.UA_STR)"


def _resolver_ua_in_fresh_interpreter() -> str:
    """The UA the resolver BINDS at import (`UA_STR = user_agent()`), read in a fresh
    interpreter pinned to the production identity. In-process that binding is whatever
    the collector's first importer saw, so it depended on the box's own identity."""
    import subprocess
    env = dict(os.environ, AW_OPERATOR_URL="https://websear.ch",
               AW_OPERATOR_CONTACT="bender@impactia.org")
    env.pop("AW_OPERATOR_OVERLAY", None)
    out = subprocess.run([sys.executable, "-c", _PROBE], cwd=str(REPO), env=env,
                         capture_output=True, text=True, timeout=120, check=True)
    return out.stdout.strip().splitlines()[-1]


def test_emitted_identity_is_unchanged(monkeypatch):
    """The no-op guarantee, asserted by rendering rather than by reading code."""
    _configured(monkeypatch)
    from cytir.core import user_agent as user_agent
    assert user_agent.user_agent() == PRODUCTION_UA
    assert _resolver_ua_in_fresh_interpreter() == PRODUCTION_UA


def test_robots_token_matches_the_transmitted_ua():
    """robotparser resolves an agent to the token BEFORE the first slash. The
    token we evaluate rules with must be the token we transmit, or a site's rule
    binds one of them and not the other."""
    assert cr.UA_STR.split("/")[0] == cr.ROBOT_AGENT


def test_a_rule_written_for_our_token_binds_what_we_send():
    robots = "User-agent: %s\nDisallow: /private\n" % cr.ROBOT_AGENT
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(robots.splitlines())
    assert rp.can_fetch(cr.UA_STR, "https://h/private") is False
    assert rp.can_fetch(cr.ROBOT_AGENT, "https://h/private") is False


def test_identity_does_not_impersonate_a_browser():
    for lie in ("Mozilla", "AppleWebKit", "Chrome/", "Safari/", "Gecko"):
        assert lie not in cr.UA_STR


def test_consent_bulk_uses_the_same_identity_not_a_second_copy(monkeypatch):
    """consent_bulk.py's docstring promises 'no UA variation, ever - the
    resolver's honest token is the only one used'. It used to keep a second
    literal that merely happened to match."""
    path = REPO / ("ta" + "sks") / "427-scripts" / "consent_bulk.py"
    if not path.is_file():
        pytest.skip("the task-427 consent_bulk script is not in this clone")
    _configured(monkeypatch)
    spec = importlib.util.spec_from_file_location("cb_identity_test", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.UA == _resolver_ua_in_fresh_interpreter()
    src = path.read_text()
    assert PRODUCTION_UA not in src, "a second hardcoded UA literal reappeared"
