"""The MCP server module must import and build its ASGI app under the pinned `mcp`.

Task 1149: `mcp==2.2.0` was pinned while `cytir.serve.mcp_server` still imported the 1.x
`FastMCP`; no CI job installed `mcp`, so aw-mcp crash-looped on its first rebuild.
"""
import asyncio

import pytest

pytest.importorskip("mcp", reason="mcp is in the dev extra and requirements-ci.txt; "
                                  "a CI job without it would hide an import break")


def test_mcp_server_imports_and_builds_app_without_binding_a_port():
    from cytir.serve import mcp_server

    assert mcp_server.MCP_PATH == "/mcp"
    names = [t.name for t in asyncio.run(mcp_server.mcp.list_tools())]
    assert "search" in names
    app = mcp_server.mcp.streamable_http_app(streamable_http_path=mcp_server.MCP_PATH)
    assert any(getattr(r, "path", None) == "/mcp" for r in app.routes)
