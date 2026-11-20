"""Consent choke point step 5 (task 1648): the `cytir/index/` INTERNAL sites use `make_internal_client`.

Census: none of the migrated modules builds a raw client or calls `urlopen`. Behaviour: the one
index door (`cytir.core.internal_http`) refuses a host that is not loopback or a configured service,
and a configured host passes through with the operator User-Agent and the JSON content type.
"""
import ast
import pathlib
import urllib.error

import httpx
import pytest

from cytir.core import internal_http as IH
from cytir.sources import http as H

APP = pathlib.Path(__file__).resolve().parents[1]
MIGRATED = ("apply", "box_health_guard", "content_pack_preservation", "id_digest", "idf_table",
            "layout", "point_delete", "prod_guard", "promote_sealed_pack", "readiness",
            "seal_collection_membership", "upsert")
RAW = {"Client", "AsyncClient", "get", "post", "urlopen", "Session", "CurlClient"}
RAW_BASES = {"httpx", "requests", "urllib", "request"}


def _raw_calls(path: pathlib.Path) -> list[str]:
    hits = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name in RAW and (not isinstance(f, ast.Attribute) or name in {"urlopen", "Client", "AsyncClient"}
                                or (isinstance(f.value, ast.Name) and f.value.id in RAW_BASES)):
                hits.append(f"{path.name}:{node.lineno}:{name}")
    return hits


@pytest.mark.parametrize("mod", MIGRATED)
def test_migrated_module_builds_no_raw_client(mod):
    path = APP / "cytir" / "index" / f"{mod}.py"
    assert _raw_calls(path) == []
    assert "internal_http" in path.read_text(), f"{mod} no longer reaches Qdrant through the one door"


def test_the_door_itself_builds_only_the_internal_client():
    path = APP / "cytir" / "core" / "internal_http.py"
    assert _raw_calls(path) == []
    assert "make_internal_client" in path.read_text()


def _through(monkeypatch, handler):
    real = H.make_internal_client
    monkeypatch.setattr(IH, "make_internal_client",
                        lambda base, timeout=10.0: real(base, timeout=timeout, transport=httpx.MockTransport(handler)))


def test_a_non_configured_host_is_refused_before_any_request(monkeypatch):
    _through(monkeypatch, lambda req: pytest.fail("a request reached the transport"))
    with pytest.raises(H.InternalHostRefused):
        IH.send("https://example.org/collections/x")
    with pytest.raises(H.InternalHostRefused):
        IH.fetch_json("http://some-origin.invalid:6333/telemetry", body={"a": 1})


def test_a_configured_host_passes_with_the_operator_headers(monkeypatch):
    from cytir.serve import settings
    real = settings.current
    monkeypatch.setattr(settings, "current", lambda name, env=None: "http://qdrant.internal:6333"
                        if name == "AW_QDRANT_URL" else real(name, env))
    seen = {}

    def handler(req):
        seen.update(method=req.method, url=str(req.url), ua=req.headers["user-agent"],
                    ctype=req.headers.get("content-type"), body=req.content)
        return httpx.Response(200, json={"status": "ok", "result": {"n": 1}})

    _through(monkeypatch, handler)
    out = IH.fetch_json("http://qdrant.internal:6333/collections/c/points/count", body={"exact": True}, timeout=7)
    assert out["result"] == {"n": 1}
    assert seen["method"] == "POST" and seen["url"].endswith("/collections/c/points/count")
    assert seen["ctype"] == "application/json" and seen["body"] == b'{"exact": true}'
    assert seen["ua"] and not seen["ua"].lower().startswith("python-urllib")


def test_loopback_get_has_no_body_and_a_non_2xx_keeps_the_urllib_error_shape(monkeypatch):
    _through(monkeypatch, lambda req: httpx.Response(503, content=b'{"status":{"error":"x"}}'))
    assert IH.send("http://127.0.0.1:6333/telemetry")[0] == 503
    with pytest.raises(urllib.error.HTTPError) as exc:
        IH.fetch("http://127.0.0.1:6333/telemetry")
    assert exc.value.code == 503 and b'"error"' in exc.value.read()


def test_a_timeout_keeps_the_urllib_timeout_shape(monkeypatch):
    def boom(req):
        raise httpx.ReadTimeout("slow", request=req)
    _through(monkeypatch, boom)
    with pytest.raises(urllib.error.URLError) as exc:
        IH.fetch("http://localhost:6333/x")
    assert isinstance(exc.value.reason, TimeoutError)
