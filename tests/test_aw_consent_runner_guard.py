"""Consent choke point, steps 2+3 (task 1600): run_family builds the guarded client; an injected
client is wrapped; `--skip-consent` does not switch the transport guard off. Fake transports only."""
from __future__ import annotations

import json
import pathlib
import sys

import httpx
import pytest

HERE = pathlib.Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cytir.consent.block import ConsentBlocked  # noqa: E402
from cytir.consent.guard import is_guarded  # noqa: E402
from cytir.core import user_agent  # noqa: E402
from cytir.sources import registry, run_family, run_windows  # noqa: E402
from cytir.sources.family import Captured, HttpSpec  # noqa: E402
import test_aw_run_family_http_spec as H  # noqa: E402

OUTCOME: list = []


class Inner(httpx.BaseTransport):
    def __init__(self):
        self.calls = 0

    def handle_request(self, request):
        self.calls += 1
        return httpx.Response(200, text="ok")


class Fetcher(H.Toy):
    family_id = "toy_consent_guard_for_tests"
    http_spec = HttpSpec()
    host = "declined.example"

    def acquire(self, unit, ctx):
        H.SEEN.append(ctx.http)
        try:
            OUTCOME.append(ctx.http.get(f"https://{self.host}/x").status_code)
        except ConsentBlocked as e:
            OUTCOME.append(e)
        return [Captured(unit, b"some text", {"status": 200})]


@pytest.fixture(autouse=True)
def registry_file(tmp_path, monkeypatch):
    path = tmp_path / "reg.json"
    path.write_text(json.dumps({"hosts": {"declined.example": {"state": "declined"}}}))
    monkeypatch.setenv("AW_CONSENT_REGISTRY", str(path))
    monkeypatch.setattr(user_agent, "user_agent", lambda component=None: "test-agent")
    monkeypatch.setattr(user_agent, "pack_fetch_user_agent", lambda: "test-pack-agent")
    OUTCOME.clear()


def test_factory_client_refuses_declined_host_before_the_socket(monkeypatch, tmp_path):   # spec 1
    inner = Inner()
    monkeypatch.setattr(httpx, "HTTPTransport", lambda *a, **k: inner)
    H._run(monkeypatch, tmp_path, Fetcher)
    (client,) = H.SEEN
    assert is_guarded(client)
    assert isinstance(OUTCOME[0], ConsentBlocked) and inner.calls == 0


def test_factory_client_is_not_double_wrapped(monkeypatch, tmp_path):                     # no-double-wrap
    H._run(monkeypatch, tmp_path, Fetcher)
    (client,) = H.SEEN
    inner = client._transport._inner
    assert run_family.ensure_guarded(client) is client
    assert client._transport._inner is inner          # no second layer


def test_injected_plain_client_is_wrapped(monkeypatch, tmp_path):                         # spec 9
    inner = Inner()
    mine = httpx.Client(transport=inner)
    H._run(monkeypatch, tmp_path, Fetcher, http=mine)
    assert H.SEEN == [mine] and is_guarded(mine) and not mine.is_closed
    assert isinstance(OUTCOME[0], ConsentBlocked) and inner.calls == 0
    mine.close()


def test_mock_transport_tape_is_not_wrapped(monkeypatch, tmp_path):                      # spec 9
    mine = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    H._run(monkeypatch, tmp_path, Fetcher, http=mine)
    assert not is_guarded(mine) and OUTCOME == [200]
    mine.close()


def test_skip_consent_does_not_disable_the_transport_guard(monkeypatch, tmp_path):        # spec 3
    inner = Inner()
    mine = httpx.Client(transport=inner)
    H._run(monkeypatch, tmp_path, Fetcher, http=mine, skip_consent=True)
    assert isinstance(OUTCOME[0], ConsentBlocked) and inner.calls == 0
    mine.close()


def _family_specs():
    out = []
    for fid in registry.registered():
        spec = getattr(registry._FACTORIES[fid](), "http_spec", None)
        if spec is None:
            out.append(pytest.param(fid, None, marks=pytest.mark.skip(reason="family has no HttpSpec")))
        elif spec.kind == "curl":
            out.append(pytest.param(fid, spec, marks=pytest.mark.skip(
                reason="curl kind: covered by the CurlClient test (guard lands with the step-4 curl lane)")))
        else:
            out.append(pytest.param(fid, spec))
    return out


@pytest.mark.parametrize("family_id,spec", _family_specs())
def test_every_registered_httpx_family_refuses_a_declined_host(monkeypatch, family_id, spec):  # spec 1
    inner = Inner()
    monkeypatch.setattr(httpx, "HTTPTransport", lambda *a, **k: inner)
    client = run_family.build_http_client(spec)
    try:
        assert is_guarded(client)
        with pytest.raises(ConsentBlocked):
            client.get("https://declined.example/x")
        assert inner.calls == 0
    finally:
        client.close()


def test_curl_client_refuses_declined_host_before_exec(monkeypatch):                      # spec 8
    from cytir.sources.ctgov_psy import harvest
    execs = []
    monkeypatch.setattr(harvest.subprocess, "run", lambda *a, **k: execs.append(a))
    with pytest.raises(ConsentBlocked):
        harvest.CurlClient().get("https://declined.example/x")
    assert execs == []


def test_run_windows_plan_guards_a_plain_client(monkeypatch, tmp_path):
    inner = Inner()
    mine = httpx.Client(transport=inner)
    seen = []

    class Planner(Fetcher):
        family_id = "toy_consent_guard_plan_for_tests"

        def plan_windows(self, ctx, cap=0):
            seen.append(ctx.http)
            return [{"id": "w1"}]

    monkeypatch.setattr(registry, "admit", lambda fid, strict=False: (Planner(), {}, None))
    wins = run_windows.plan(Planner.family_id, tmp_path, {}, "2026-10-30", 1, http=mine)
    assert wins == [{"id": "w1"}] and seen == [mine] and is_guarded(mine)
    mine.close()
