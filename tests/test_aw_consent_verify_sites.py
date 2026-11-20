"""Consent choke point, the VERIFY sites (task 1631): each listed site's origin was read and classed.

EXTERNAL (a third-party host, including the api.anthropic.com judge call) goes through `make_client`, so
a registry-declined host is refused before any socket. INTERNAL (Qdrant, the AW API: a configured service
named by a CLI arg or setting, loopback by default) stays on urllib and is allow-listed below with its reason.
PUBLIC_SELF (AW's own public endpoint, websear.ch, behind an exact-host allow-list and https) is a third, pinned mechanism.
"""
import ast
import json
import re
from pathlib import Path

import httpx
import pytest

from cytir.consent.block import ConsentBlocked
from cytir.core import user_agent

APP = Path(__file__).resolve().parent.parent


class Inner(httpx.BaseTransport):
    def __init__(self):
        self.calls = 0

    def handle_request(self, request):
        self.calls += 1
        raise AssertionError("the guard let a declined request reach the inner transport")


@pytest.fixture(autouse=True)
def world(tmp_path, monkeypatch):
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"hosts": {"api.anthropic.com": {"state": "declined"}}}))
    monkeypatch.setenv("AW_CONSENT_REGISTRY", str(reg))
    monkeypatch.setattr(user_agent, "user_agent", lambda component=None: "test-agent")
    inner = Inner()
    monkeypatch.setattr(httpx, "HTTPTransport", lambda *a, **k: inner)
    return inner


def _eval_general_run(monkeypatch):
    from cytir.ops import eval_general_run as m
    monkeypatch.setattr("time.sleep", lambda s: pytest.fail("a declined host was retried"))
    with pytest.raises(ConsentBlocked):
        m._call("some-model", "q", "sk-test")


SITES = [("ops.eval_general_run", _eval_general_run)]


@pytest.mark.parametrize("module,site", SITES, ids=[s[0] for s in SITES])
def test_site_refuses_declined_host_with_no_inner_call(module, site, world, monkeypatch):
    site(monkeypatch)
    assert world.calls == 0


def test_anthropic_call_keeps_its_auth_headers(monkeypatch):
    seen = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200, json={"usage": {"input_tokens": 1, "output_tokens": 2},
                                         "content": [{"type": "text", "text": "ok"}]})
    from cytir.ops import eval_general_run as m
    from cytir.sources import http as http_mod
    real = http_mod.make_client
    monkeypatch.setattr(m, "make_client",
                        lambda spec=None, **k: real(spec, transport=httpx.MockTransport(handler)))
    # the registry in `world` declines api.anthropic.com; lift it for this one allowed-host check
    reg = Path(__import__("os").environ["AW_CONSENT_REGISTRY"])
    reg.write_text(json.dumps({"hosts": {}}))
    assert m._call("some-model", "q", "sk-test") == "ok"
    assert seen["x-api-key"] == "sk-test" and seen["anthropic-version"] == "2023-06-01"
    assert seen["content-type"] == "application/json"


# --- census ------------------------------------------------------------------------------------
BANNED = re.compile(r"urllib\.request\.(urlopen|Request)")
SUFFIX = " — step 5 make_internal_client (PR #1351) refuses a non-configured host and closes the any-host-via-CLI gap"

#: INTERNAL sites left on urllib, keyed (file, line fragment) -> reason (the configured service, default host).
INTERNAL = {
    ("cytir/ops/rii_dump_basis_backfill.py", "urllib.request.Request("): "Qdrant, --qdrant-url (retrieve-by-id)" + SUFFIX,
    ("cytir/ops/restore_pack.py", "urllib.request.Request("): "Qdrant, --qdrant-url" + SUFFIX,
    ("cytir/ops/landing_check.py", "urllib.request.Request("): "Qdrant, --<stage>-url" + SUFFIX,
    ("cytir/corpus/store.py", "urllib.request.Request("): "Qdrant scroll, qdrant_url argument (prod refused by the prod guard)" + SUFFIX,
    ("cytir/sources/families/news_tier2_apply.py", "urllib.request.Request("): "Qdrant scroll, --qdrant-url" + SUFFIX,
    ("cytir/ops/family_census.py", "urllib.request.Request("): "Qdrant, --collection-url (no default)" + SUFFIX,
}

#: INTERNAL sites keyed (file, enclosing function): every urllib call inside that function is the allowed one.
INTERNAL_FUNC = {
    ("cytir/ops/docs_roster.py", "twin_facets"): "twin Qdrant facet, twin_url argument" + SUFFIX,
}

#: EXTERNAL doc fetch `docs_roster._get` (lane F, PR #1352 migrates it). Tolerated ONLY while it still uses
#: urllib: the census then xfails visibly; once #1352 lands the call is gone and the xfail disappears.
LANE_F_FUNC = ("cytir/ops/docs_roster.py", "_get")

#: PUBLIC_SELF sites keyed (file, enclosing function): urllib to AW's OWN public endpoint, never a third-party
#: host (those go through make_client). The function pins an exact-host allow-list and https (task 2041;
#: mechanism added by the orchestrator, task 2054). The host set is pinned below so a widening is a diff here.
PUBLIC_SELF = {
    ("cytir/probe/run.py", "_public_urlopen"): "websear.ch probe door: PUBLIC_PROBE_HOSTS exact host, https only",
}
PUBLIC_SELF_HOSTS = {"websear.ch"}

FILES = [APP / f for f in (
    "cytir/ops/rii_dump_basis_backfill.py", "cytir/ops/restore_pack.py", "cytir/ops/landing_check.py",
    "cytir/probe/run.py", "cytir/corpus/store.py", "cytir/sources/families/news_tier2_apply.py",
    "cytir/ops/docs_roster.py", "cytir/gates/slice_gate.py", "cytir/index/readiness.py",
    "cytir/ops/family_census.py", "cytir/ops/eval_general_run.py")]


def _functions_at(path: Path) -> dict:
    """line number -> names of every function whose body spans that line (nested ones included)."""
    out: dict = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in range(node.lineno, node.end_lineno + 1):
                out.setdefault(n, set()).add(node.name)
    return out


def test_public_self_host_set_is_pinned():
    from cytir.probe import run
    assert set(run.PUBLIC_PROBE_HOSTS) == PUBLIC_SELF_HOSTS


def test_no_unclassified_urllib_call_remains():
    offenders, used, lane_f = [], set(), []
    for f in FILES:
        rel = str(f.relative_to(APP))
        funcs = _functions_at(f)
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if not BANNED.search(line) or line.lstrip().startswith("#"):
                continue
            names = funcs.get(i, set())
            fn_key = next(((rel, n) for n in names if (rel, n) in INTERNAL_FUNC or (rel, n) in PUBLIC_SELF), None)
            key = next((k for k in INTERNAL if k[0] == rel and k[1] in line), None)
            if fn_key:
                used.add(fn_key)
            elif key:
                used.add(key)
            elif (rel, "_get") == LANE_F_FUNC and "_get" in names:
                lane_f.append(f"{rel}:{i}")
            else:
                offenders.append(f"{rel}:{i}: {line.strip()[:90]}")
    assert not offenders, "\n".join(offenders)
    expected = set(INTERNAL) | set(INTERNAL_FUNC) | set(PUBLIC_SELF)
    assert used == expected, f"stale allow-list rows: {sorted(expected - used, key=str)}"
    if lane_f:
        pytest.xfail(reason="docs_roster._get migrates in PR #1352 (lane F)")


def test_variable_origin_internal_sites_name_a_service_not_a_free_url():
    assert 'os.environ.get("AW_PROBE_PROD_API_URL", "http://localhost:8090")' in (APP / "cytir/probe/run.py").read_text()
    assert "--collection-url" in (APP / "cytir/ops/family_census.py").read_text()
    assert "--qdrant-url" in (APP / "cytir/ops/restore_pack.py").read_text()
