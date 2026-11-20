"""`cyt read <url> [--read-url U] [--max-chars N] [--renderer R]`: print the reader's JSON (task 2710)."""
from __future__ import annotations

import argparse
import json
import sys

from . import config
from . import read as R
from .fetch import FetchError, FetchRefused
from .render import find_renderer


def main(argv: list[str] | None = None, prog: str = "cyt read") -> int:
    p = argparse.ArgumentParser(prog=prog, description="Read a source URL as text (local reader; JavaScript pages "
                                "are rendered with an installed browser).")
    p.add_argument("url", nargs="?")
    p.add_argument("--mcp", action="store_true", help="serve the reader as the cytir-reader stdio MCP server")
    p.add_argument("--read-url", default=None, help="twin URL that needs no JavaScript (from a search result)")
    p.add_argument("--max-chars", type=int, default=config.DEFAULT_MAX_CHARS)
    p.add_argument("--renderer", default=None,
                   help="auto|installed_browser|obscura|moli|lightpanda|none (default: $CYTIR_READER_RENDERER or auto)")
    a = p.parse_args(argv)
    if a.mcp:
        from cytir.reader.mcp import main as mcp_main  # lazy: plain `cyt read` imports no MCP SDK
        mcp_main()
        return 0
    if not a.url:
        p.error("url is required (or pass --mcp)")
    try:
        out = R.read(a.url, read_url=a.read_url, max_chars=a.max_chars, renderer=find_renderer(a.renderer))
    except (FetchRefused, FetchError) as e:
        print(json.dumps({"url": a.url, "error": str(e)}, ensure_ascii=False))
        return 1
    json.dump(out, sys.stdout, ensure_ascii=False)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
