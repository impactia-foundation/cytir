"""Task 2663: the MCP `find_authority` tool mirrors GET /find_authority (one implementation, the API route)."""
import asyncio
import io
import json
import urllib.error

import pytest

pytest.importorskip("mcp", reason="mcp is in requirements-ci.txt")

from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from cytir.serve import mcp_server  # noqa: E402


def _tools():
    return {t.name: t for t in asyncio.run(mcp_server.mcp.list_tools())}


def _schema(t):
    return getattr(t, "input_schema", None) or t.inputSchema


def test_registered_with_the_http_route_arguments():
    tools = _tools()
    assert set(tools) == {"search", "find_authority"}
    sch = _schema(tools["find_authority"])
    assert set(sch["properties"]) == {"query", "k", "country"} and sch["required"] == ["query"]


def test_description_length_and_required_phrases():
    d = _tools()["find_authority"].description.split("\n\n")[0]
    assert len(d) <= 700
    for needle in ("authority card", "navigate.search_url", "`search`", "weak_match", "note"):
        assert needle in d


def test_calls_the_http_route_and_passes_the_response_through(monkeypatch):
    seen = {}
    doc = {"query": "x", "count": 1, "authorities": [{"confidence": 0.9, "weak_match": False,
           "fetchable": True, "note": "n", "declined_note": "d"}]}

    def fake(url, timeout=0):
        seen["url"] = url
        return doc
    monkeypatch.setattr(mcp_server.IH, "fetch_json", fake)
    assert mcp_server.find_authority("Art 336", k=5, country="CH") is doc
    assert seen["url"] == f"{mcp_server.API}/find_authority?q=Art+336&k=5&country=CH"
    src = open(mcp_server.__file__, encoding="utf-8").read()
    assert "find_authorities(" not in src   # no second implementation: the route is the one door


def test_errors_are_tool_errors(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.HTTPError("u", 503, "x", {}, io.BytesIO(
            json.dumps({"error": "authority lookup unavailable"}).encode()))
    monkeypatch.setattr(mcp_server.IH, "fetch_json", boom)
    with pytest.raises(ToolError, match="find_authority API HTTP 503: authority lookup unavailable.*retry"):
        mcp_server.find_authority("x")

    def down(*a, **k):
        raise urllib.error.URLError("refused")
    monkeypatch.setattr(mcp_server.IH, "fetch_json", down)
    with pytest.raises(ToolError, match="unreachable.*retry"):
        mcp_server.find_authority("x")
