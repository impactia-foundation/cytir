"""Cytir MCP server (`cytir-websearch`; Streamable HTTP). Thin: the `search` tool calls the
search-api over localhost, so models load only once (in aw-api).

Run: python -m cytir.serve.mcp_server   (serves /mcp on :8081)
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import json
import urllib.error
import urllib.parse
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .settings import SETTINGS as _S

API = _S.runtime.api_url

# mcp 2.x moved host/port/path off the constructor onto run() (task 1149).
MCP_PATH = "/mcp"
MCP_NAME = "cytir-websearch"
MCP_TITLE = "Cytir: open web search for agents"
MCP_INSTRUCTIONS = (
    "Cytir is open web search for agents. It finds the URL of the authoritative source for a fact "
    "(Swiss law on fedlex, official statistics, government, docs); you then fetch and read that URL. "
    "Use `search` for the page that states a fact, `find_authority` for WHICH source to look in. "
    "If a call fails with a transient error, retry once."
)
mcp = MCPServer(MCP_NAME, title=MCP_TITLE, instructions=MCP_INSTRUCTIONS)


def api_error_message(status: int, body: bytes | str, label: str = "search") -> str:
    """The text an agent reads when the search API answers non-2xx (task 2655).

    A bare `Error executing tool search` tells the agent nothing: a 400 means "fix the query", a
    5xx means "retry". The text carries the HTTP status and the API's own `error`/`detail`."""
    detail = ""
    try:
        doc = json.loads(body)
        if isinstance(doc, dict):
            detail = str(doc.get("error") or doc.get("detail") or "")
    except (ValueError, TypeError):
        pass   # a non-JSON body (proxy HTML) is never echoed to the agent: status only (review 2660)
    msg = f"{label} API HTTP {status}: {detail or 'no detail'}"
    if status >= 500:
        msg += ". Transient: retry in a few seconds."
    return msg


def transport_error_message(e: urllib.error.URLError, label: str = "search") -> str:
    """A timeout reads as "backend timed out, retry once"; any other transport failure names the reason."""
    if isinstance(e.reason, TimeoutError):
        return f"{label} backend timed out. Retry once; if it fails again, wait a minute."
    return f"{label} API unreachable: {e.reason}. Transient: retry in a few seconds."


@mcp.tool()
def search(query: str, k: int = 10, lang: str | None = None,
           country: str | None = None, site: list[str] | None = None,
           exclude_site: list[str] | None = None, source: str | None = None) -> dict:
    """Find the authoritative source URL to GROUND a fact (Swiss law on fedlex, official statistics, government). Returns ranked URLs to FETCH AND READ, with title, short snippet, source_type, trust_tier, lang. Snippet = triage only. Snippets choose the URL; fetch `url` (or `read_url` when present, a no-JavaScript copy) to read the source.

    Args:
        query: the claim or question. May hold operators: site:HOST, -site:HOST, source:TYPE, lang:XX, country:XX.
        k: results (10).
        lang: en|de|fr|it filter.
        country: CH|FR|DE|CA|US soft BOOST, not a filter (echoed as `boosts`).
        site / exclude_site: host lists, like site: / -site:.
        source: same as source:.

    site: matches the host and all sub-hosts (admin.ch covers fedlex.admin.ch), max 2000 hosts. A query of only operators is rejected.
    source:<type> e.g. official, reference, docs, news, entity-self (the list grows; an unknown value is searched as text and `filters.note` names valid ones).
    Examples: "Art. 336 CO site:fedlex.admin.ch"; "minimum wage -site:wikipedia.org source:official".
    """
    params = [("q", query), ("k", k)]
    if lang:
        params.append(("lang", lang))
    if country:
        params.append(("country", country))
    params += [("site", h) for h in site or []]
    params += [("exclude_site", h) for h in exclude_site or []]
    if source:
        params.append(("source", source))
    try:
        return IH.fetch_json(f"{API}/search?" + urllib.parse.urlencode(params), timeout=30)
    except urllib.error.HTTPError as e:
        body = e.fp.read() if e.fp is not None else b""
        raise ToolError(api_error_message(e.code, body)) from e
    except urllib.error.URLError as e:
        raise ToolError(transport_error_message(e, "search")) from e


@mcp.tool()
def find_authority(query: str, k: int = 3, country: str | None = None) -> dict:
    """Find WHICH SOURCE is authoritative for a topic (an authority card), not a page. Each card names a source (fedlex, a registry, a docs site) and `navigate.search_url`, a URL pattern to query that source directly. Use it when the question is "where do I look?"; use `search` when you need the page that states the fact. If a card has `weak_match` or a `note`, no good authority is known: fall back to `search`. Other fields (confidence, fetchable, declined_note) pass through unchanged.

    Args:
        query: the topic or question.
        k: cards (3, max 10).
        country: CH|FR|DE|CA|US|EU. HARD filter: only cards whose jurisdiction equals country are returned. EU is mapped to INTL (EU law is carded as INTL). Pass it only when the question names or implies that jurisdiction. Omit it for international standards, software vendors, and any topic with no country. If the filter yields no card, the server retries once without it and the response carries `filter_relaxed: \"country\"`: those cards are NOT from the requested jurisdiction; check their `jurisdiction` before relying on them.
    """
    params = [("q", query), ("k", k)]
    if country:
        params.append(("country", country))
    try:
        return IH.fetch_json(f"{API}/find_authority?" + urllib.parse.urlencode(params), timeout=30)
    except urllib.error.HTTPError as e:
        body = e.fp.read() if e.fp is not None else b""
        raise ToolError(api_error_message(e.code, body, "find_authority")) from e
    except urllib.error.URLError as e:
        raise ToolError(transport_error_message(e, "find_authority")) from e


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0",
            port=_S.runtime.mcp_port, streamable_http_path=MCP_PATH)
