"""`cytir-reader`: stdio MCP server with one tool, `read` (task 2710). Run: python -m cytir.reader.mcp

Runs on the client's machine. It is not the Cytir search server (`cytir.serve.mcp_server`) and holds no index."""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import config
from . import read as R
from .fetch import FetchError, FetchRefused

MCP_NAME = "cytir-reader"
mcp = MCPServer(MCP_NAME, title="Cytir reader: read a source URL as text",
                instructions="Read the page at a URL (from a Cytir `search` result) as text. Handles JavaScript pages "
                             "with an installed browser. Not a search tool: search-engine URLs are refused.")


@mcp.tool()
def read(url: str, read_url: str | None = None, max_chars: int = config.DEFAULT_MAX_CHARS) -> dict:
    """Read the page at `url` as text. Pass a search result's `read_url` when it has one. A `#fragment` returns only
    that element (e.g. one article of a law). Returns url, final_url, source_used (url|read_url|rendered), renderer,
    text, truncated, elapsed_ms, notes. If `notes` says the page is a JavaScript shell, no browser was available.

    Args:
        url: the source URL to read (http or https; search-engine result pages are refused).
        read_url: optional twin URL that needs no JavaScript, from the search result.
        max_chars: cap on returned text (default 20000); `truncated` says when it cut.
    """
    try:
        return R.read(url, read_url=read_url, max_chars=max_chars)
    except (FetchRefused, FetchError, ValueError) as e:
        raise ToolError(str(e)) from e


def main() -> None:
    """Serve the reader over stdio (`cyt read --mcp` or `python -m cytir.reader.mcp`)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
