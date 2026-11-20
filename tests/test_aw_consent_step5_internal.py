"""Consent choke point step 5b (task 1673): the remaining INTERNAL sites use the one internal door.

Census: none of the migrated modules builds a raw client or calls `urlopen`. Behaviour: the shared
door (`cytir.core.internal_http`, the one name) refuses a host that is
neither loopback nor a configured service, keeps the `urlopen` shape (`Request` method/data/headers,
`with ... as r`, `HTTPError` on non-2xx, `URLError` on transport failure), and a TEI call through
`cytir.serve.models` reaches the door.
"""
import ast
import pathlib
import urllib.error
import urllib.request

import httpx
import pytest

from cytir.sources import http as H
from cytir.core import internal_http as IH

APP = pathlib.Path(__file__).resolve().parents[1]
MIGRATED = (
    "bench/scope", "corpus/store", "gates/content_pack_arm",
    "gates/gate_payload_contract", "gates/news_daily", "gates/prod_legs", "gates/run_arm",
    "gates/serve_parity", "gates/slice_gate", "news/coverage_backfill", "news/freshness_ingest",
    "news/weekly_delta", "ops/docs_roster", "ops/family_census", "ops/german_parts_tripwire",
    "ops/landing_check", "ops/news_census", "ops/restore_pack",
    "ops/rii_dump_basis_backfill", "ops/serve", "probe/run", "retune/emit", "retune/guard",
    "retune/preflight", "serve/mcp_server", "serve/models", "sources/families/news_tier2_apply",
    "sources/family_stage",
)
#: PUBLIC_SELF: the one raw call allowed in a MIGRATED module, keyed (file, enclosing function): urllib to AW's
#: own public endpoint. Mirrors tests/test_aw_consent_verify_sites.py (task 2054 ruling C / task 2068).
PUBLIC_SELF = {
    ("cytir/probe/run.py", "_public_urlopen"): "AW's own public endpoint (websear.ch), task 2054 ruling C / task 2068",
}
RAW = {"Client", "AsyncClient", "get", "post", "urlopen", "Session", "CurlClient"}
RAW_BASES = {"httpx", "requests", "urllib", "request"}


def _raw_calls(path: pathlib.Path) -> list[str]:
    hits = []
    tree = ast.parse(path.read_text())
    rel = str(path.relative_to(APP))
    exempt_lines: set[int] = set()
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and (rel, fn.name) in PUBLIC_SELF:
            exempt_lines.update(range(fn.lineno, (fn.end_lineno or fn.lineno) + 1))
    for node in ast.walk(tree):
        refs = []
        if isinstance(node, ast.Call):
            refs.append(node.func)
        elif isinstance(node, ast.arguments):  # `opener=urllib.request.urlopen` defaults
            refs.extend(d for d in node.defaults + node.kw_defaults if d is not None)
        for f in refs:
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "IH":
                continue  # the door itself
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if isinstance(f, ast.Attribute):
                raw = name in {"urlopen", "Client", "AsyncClient"} or (
                    name in RAW and isinstance(f.value, ast.Name) and f.value.id in RAW_BASES)
            else:
                raw = name in {"urlopen", "CurlClient"}
            if raw and getattr(f, "lineno", 0) in exempt_lines:
                continue  # PUBLIC_SELF: exact (file, function) key only
            if raw:
                hits.append(f"{path.name}:{getattr(f, 'lineno', 0)}:{name}")
    return hits


def test_public_self_key_set_is_pinned():
    assert set(PUBLIC_SELF) == {("cytir/probe/run.py", "_public_urlopen")}


@pytest.mark.parametrize("mod", MIGRATED)
def test_migrated_module_builds_no_raw_client(mod):
    path = APP / "cytir" / f"{mod}.py"
    assert _raw_calls(path) == []
    assert "internal_http" in path.read_text(), f"{mod} no longer reaches its service through the door"


def test_the_shared_door_builds_only_the_internal_client():
    path = APP / "cytir" / "core" / "internal_http.py"
    assert _raw_calls(path) == []
    assert "make_internal_client" in path.read_text()


def test_the_old_door_names_are_gone():
    assert not (APP / "cytir" / "sources" / "internal_http.py").exists()
    assert not (APP / "cytir" / "index" / "internal_http.py").exists()


def _through(monkeypatch, handler):
    real = H.make_internal_client
    monkeypatch.setattr(IH, "make_internal_client",
                        lambda base, timeout=10.0: real(base, timeout=timeout, transport=httpx.MockTransport(handler)))


def _configure(monkeypatch, **knobs):
    from cytir.serve import settings
    real = settings.current
    monkeypatch.setattr(settings, "current", lambda name, env=None: knobs.get(name) or real(name, env))


def test_urlopen_refuses_a_host_that_is_not_a_service(monkeypatch):
    _through(monkeypatch, lambda req: pytest.fail("a request reached the transport"))
    with pytest.raises(H.InternalHostRefused):
        IH.urlopen("https://example.org/collections/x")
    req = urllib.request.Request("http://some-origin.invalid:6333/x", data=b"{}", method="POST")
    with pytest.raises(H.InternalHostRefused):
        IH.urlopen(req, timeout=5)


def test_urlopen_keeps_request_method_data_headers_and_the_with_shape(monkeypatch):
    _configure(monkeypatch, AW_QDRANT_URL="http://qdrant.internal:6333")
    seen = {}

    def handler(req):
        seen.update(method=req.method, url=str(req.url), body=req.content, ctype=req.headers.get("content-type"))
        return httpx.Response(200, json={"result": {"count": 3}})

    _through(monkeypatch, handler)
    req = urllib.request.Request("http://qdrant.internal:6333/collections/c/points/count",
                                 data=b'{"exact": true}', headers={"Content-Type": "application/json"})
    with IH.urlopen(req, timeout=9) as r:
        assert __import__("json").load(r)["result"]["count"] == 3
    assert seen["method"] == "POST" and seen["body"] == b'{"exact": true}' and seen["ctype"] == "application/json"
    gone = urllib.request.Request("http://qdrant.internal:6333/collections/c", method="DELETE")
    IH.urlopen(gone).read()
    assert seen["method"] == "DELETE"


def test_urlopen_error_shapes_match_urllib(monkeypatch):
    _configure(monkeypatch, AW_QDRANT_URL="http://qdrant.internal:6333")
    _through(monkeypatch, lambda req: httpx.Response(503, text="busy"))
    with pytest.raises(urllib.error.HTTPError) as ei:
        IH.urlopen("http://qdrant.internal:6333/x")
    assert ei.value.code == 503 and ei.value.read() == b"busy"

    def boom(req):
        raise httpx.ConnectError("down")

    _through(monkeypatch, boom)
    with pytest.raises(urllib.error.URLError):
        IH.urlopen("http://qdrant.internal:6333/x")


def test_the_serve_models_tei_call_goes_through_the_door(monkeypatch):
    from cytir.serve import models
    _configure(monkeypatch, AW_TEI_URL="http://tei.internal:8080")
    seen = {}

    def handler(req):
        seen.update(url=str(req.url), body=req.content)
        return httpx.Response(200, json=[[1.0, 0.0]])

    _through(monkeypatch, handler)
    out = IH.fetch_json("http://tei.internal:8080/embed", body={"inputs": ["q"]}, timeout=5)
    assert out == [[1.0, 0.0]] and seen["url"].endswith("/embed")
    assert "internal_http" in pathlib.Path(models.__file__).read_text()


def test_the_health_probe_shape_returns_the_status(monkeypatch):
    _configure(monkeypatch, AW_TEI_URL="http://tei.internal:8080")
    _through(monkeypatch, lambda req: httpx.Response(200, json={}))
    assert IH.send("http://tei.internal:8080/health", timeout=15)[0] == 200
