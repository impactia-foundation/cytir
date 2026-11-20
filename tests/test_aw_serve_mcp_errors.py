"""Task 2655: the MCP `search` tool names the API's HTTP status and error text (probe 2651 fu-4)."""
import io
import json
import urllib.error

import pytest

pytest.importorskip("mcp", reason="mcp is in requirements-ci.txt")

from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from cytir.serve import mcp_server  # noqa: E402


def _http(status, payload):
    body = json.dumps(payload).encode()
    return urllib.error.HTTPError("http://x/search", status, f"HTTP {status}", {}, io.BytesIO(body))


def _raise(exc):
    def f(*a, **k):
        raise exc
    return f


def test_operator_only_query_400_carries_the_api_error(monkeypatch):
    err = "query has no terms besides the operators"
    monkeypatch.setattr(mcp_server.IH, "fetch_json",
                        _raise(_http(400, {"error": err, "status": "bad_request"})))
    with pytest.raises(ToolError) as ei:
        mcp_server.search("site:admin.ch")
    msg = str(ei.value)
    assert "HTTP 400" in msg and err in msg and "retry" not in msg.lower()


def test_503_carries_detail_and_a_retry_hint(monkeypatch):
    monkeypatch.setattr(mcp_server.IH, "fetch_json", _raise(_http(
        503, {"error": "search backend unavailable — try again shortly", "status": "degraded"})))
    with pytest.raises(ToolError) as ei:
        mcp_server.search("Art. 336 CO")
    msg = str(ei.value)
    assert "HTTP 503" in msg and "search backend unavailable" in msg and "retry" in msg.lower()


def test_non_json_body_and_empty_body():
    msg = mcp_server.api_error_message(502, b"<html>bad gateway</html>")
    assert msg.startswith("search API HTTP 502: no detail") and "html" not in msg   # status only (review 2660)
    assert "HTTP 400: no detail" in mcp_server.api_error_message(400, b"")


def test_transport_failure_is_a_transient_tool_error(monkeypatch):
    monkeypatch.setattr(mcp_server.IH, "fetch_json", _raise(urllib.error.URLError("refused")))
    with pytest.raises(ToolError, match="unreachable.*retry"):
        mcp_server.search("x")


def test_timeout_says_backend_timed_out_retry_once(monkeypatch):
    monkeypatch.setattr(mcp_server.IH, "fetch_json",
                        _raise(urllib.error.URLError(TimeoutError("read timed out"))))
    with pytest.raises(ToolError, match="search backend timed out. Retry once"):
        mcp_server.search("x")
    with pytest.raises(ToolError, match="find_authority backend timed out. Retry once"):
        mcp_server.find_authority("x")


def test_server_carries_the_cytir_brand():
    assert mcp_server.mcp.name == "cytir-websearch"
    assert "Cytir" in mcp_server.MCP_INSTRUCTIONS and "authoritative-web" not in mcp_server.MCP_INSTRUCTIONS
