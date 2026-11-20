"""ctgov_psy CurlClient: the guarded manual redirect loop (task 1634)."""
import json
import types

import pytest

from cytir.consent.block import ConsentBlocked
from cytir.core import user_agent

START = "https://cleared.example/a"


@pytest.fixture(autouse=True)
def registry(tmp_path, monkeypatch):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"hosts": {"declined.example": {"state": "declined"}}}))
    monkeypatch.setenv("AW_CONSENT_REGISTRY", str(path))
    monkeypatch.setattr(user_agent, "pack_fetch_user_agent", lambda: "test-pack-agent")


def script(monkeypatch, replies):
    """Replace subprocess.run with a scripted sequence of (code, redirect, ctype, body)."""
    runs = []

    def fake(argv, **kw):
        runs.append(argv)
        code, loc, ctype, body = replies[len(runs) - 1]
        return types.SimpleNamespace(stdout=f"{body}\n__AW_META__ {code}\t{loc}\t{ctype}")
    from cytir.sources.ctgov_psy import harvest
    monkeypatch.setattr(harvest.subprocess, "run", fake)
    return runs, harvest


def test_redirect_to_declined_host_blocked_before_second_run(monkeypatch):
    runs, h = script(monkeypatch, [(302, "https://declined.example/y", "", "")])
    with pytest.raises(ConsentBlocked):
        h.CurlClient().get(START)
    assert len(runs) == 1


def test_redirect_to_cleared_host_returns_second_body(monkeypatch):
    runs, h = script(monkeypatch, [(302, "/b", "", ""), (200, "", "text/plain", "second")])
    r = h.CurlClient().get(START)
    assert len(runs) == 2 and runs[1][-1] == "https://cleared.example/b"
    assert (r.status_code, r.text, r.ctype) == (200, "second", "text/plain")
    assert "-L" not in runs[0]


def test_eleven_hops_raise_with_last_status(monkeypatch):
    _, h = script(monkeypatch, [(302, "/n", "", "")] * 12)
    with pytest.raises(RuntimeError, match="HTTP 302"):
        h.CurlClient().get(START)


def test_redirect_without_location_raises(monkeypatch):
    _, h = script(monkeypatch, [(301, "", "", "")])
    with pytest.raises(RuntimeError, match="HTTP 301"):
        h.CurlClient().get(START)


def test_plain_200_unchanged_shape(monkeypatch):
    runs, h = script(monkeypatch, [(200, "", "application/json", '{"a": 1}')])
    r = h.CurlClient().get(START)
    assert len(runs) == 1
    assert (r.status_code, r.text, r.url, r.headers) == (
        200, '{"a": 1}', START, {"content-type": "application/json"})
    assert r.json() == {"a": 1}


@pytest.mark.parametrize("code", [302, 303, 307])
def test_post_redirect_reissues_post_with_same_body(monkeypatch, code):
    # curl -X POST -L kept POST and the body on every hop; we keep parity.
    runs, h = script(monkeypatch, [(code, "/p", "", ""), (200, "", "", "ok")])
    h.CurlClient().post_json(START, {"a": 1})
    assert len(runs) == 2
    assert runs[1][:-1] == runs[0][:-1]
    assert runs[1][runs[1].index("-X") + 1] == "POST"
    assert runs[1][runs[1].index("-d") + 1] == json.dumps({"a": 1})
    assert runs[1][-1] == "https://cleared.example/p"
