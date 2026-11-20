"""Shared identity and outcome contract for consent-bound direct terminals."""
from __future__ import annotations

import hashlib
import ipaddress
import socket
from typing import Any
from urllib.parse import urlsplit

from cytir.consent import resolver as CR


MODE = "direct"
OUTCOMES = (
    "complete", "redirect", "declined", "held", "non_textual",
    "oversized", "http_terminal", "transport_error", "retry_queued",
)
# Terminal-outcome schema version (task 502 defect 1 / ADR-103). v6 splits the
# two "robots.txt could not be READ as a policy" cases -- access-controlled
# (401/403) and a redirect chain that ended with no Location to follow -- out
# of "declined" into ("held", "consent_unreadable"): ADR-073's own text calls
# an unreadable origin a RETRY posture ("re-probing from a second vantage +
# reason-tagged retry queue"), not a refusal. A readable robots.txt that
# actually disallows (a parsed body, or an explicit 200/404/410 no-policy
# answer) is unchanged and stays "declined". Nothing here changes WHICH urls
# get fetched -- ``direct_url_allowed``'s bool is untouched -- only how the
# reason for a fetch-refusal is named.
#
# Forward-only: no sealed ledger is rewritten. A caller re-deriving a v5
# terminal against ALREADY-SEALED v5 evidence (e.g. replaying a task-426
# rollout) must pass ``schema_version=SCHEMA_VERSION_V5`` to get the old,
# undivided "declined" reading back -- the sealed receipt's own ``status``
# was written under v5 and a v6 re-derivation would disagree with it for
# exactly the rows this defect describes, which is a false verification
# failure, not a real one. New acquisition/materialization runs use the v6
# default.
#
# v7 (task 523) takes the same reading one step further, to the HTTP side. A
# response that ASKS US TO COME BACK LATER is not a disposition of the URL: it
# is the state of the conversation at that instant, usually of our own making.
# The measured case is `commission.europa.eu`, where 1,935 of the 2,202 points
# the 1024 rebuild lost carry the terminal reason `http_429` -- OUR rate limit,
# sealed as if the European Commission had answered about the document. That is
# an error recorded as an empty (verification doctrine 6). At v7 those codes
# resolve to ("retry_queued", "http_<code>"), an outcome that carries no
# judgement about the URL and belongs in a retry queue; every other non-2xx
# code keeps "http_terminal" unchanged. Forward-only, exactly as v6: replaying
# a sealed v5/v6 ledger must pass that version to re-derive its own rows.
#
# v8 (task 542) is the deferred item task 509 named and left for its own PR:
# a `status="unknown"` receipt whose robots/tdmrep leg failed on a
# deterministic, unfollowable redirect (`httpx.InvalidURL` on an unparsable
# Location, folded into `BudgetExhausted` since PR #88 so it cannot crash the
# resolver -- but classified identically to a genuine timeout ever since) now
# resolves to ("held", "consent_unreadable") instead of the generic
# ("held", "consent_unresolved"). `CR.unresolved_terminal_reason` makes the
# call, from `CR.classify_unreadable`'s new "malformed_redirect_target" class
# (`consent_resolver.py`). This stays firmly inside ADR-024 amendment 3's
# 5xx-only boundary: the receipt is STILL "held" either way -- amendment 3
# governs promotion out of held into admission, and nothing here is a 5xx or
# touches `unresolved_search_ok`. It only tells apart "will never resolve on
# its own, stop retrying" from "ask again later", the same way v6 told apart
# "never read as a policy" from "read and declines" within the OK/declined
# side. Forward-only, exactly as v6/v7: replaying a sealed v5/v6/v7 ledger
# must pass that version to re-derive its own rows.
SCHEMA_VERSION_V5 = "adr092-direct-terminal.v5"
SCHEMA_VERSION_V6 = "adr092-direct-terminal.v6"
SCHEMA_VERSION_V7 = "adr092-direct-terminal.v7"
SCHEMA_VERSION_V8 = "adr092-direct-terminal.v8"
SCHEMA_VERSIONS = (SCHEMA_VERSION_V5, SCHEMA_VERSION_V6, SCHEMA_VERSION_V7,
                   SCHEMA_VERSION_V8)
CURRENT_SCHEMA_VERSION = SCHEMA_VERSION_V8

# HTTP status codes whose meaning is "not now", never "not this URL".
#   408 Request Timeout, 425 Too Early  -- the request never landed
#   429 Too Many Requests               -- our own rate limit, per RFC 6585
#   502/503/504                         -- an upstream/gateway that is not up
# 500 is deliberately ABSENT: a bare Internal Server Error is not a documented
# come-back-later signal, and widening this set beyond the evidence would turn
# real dead ends into an unbounded retry queue.
RETRY_LATER_HTTP_STATUS = frozenset({408, 425, 429, 502, 503, 504})


def http_outcome(code: int, *, has_body: bool = True,
                 schema_version: str = CURRENT_SCHEMA_VERSION
                 ) -> tuple[str, str]:
    """Outcome for a fetch that reached the origin and got a non-success answer.

    Returns ``(status, reason)``. At v7+ a come-back-later status is
    ``("retry_queued", "http_<code>")``; at v5/v6, and for every other code, it
    is the historical ``("http_terminal", ...)`` so sealed ledgers re-derive
    byte-identically. ``has_body`` distinguishes the 2xx-with-empty-body case
    the acquisition lane also routes here.

    ``schema_version in (V7, V8)``, not ``== V7``: v8 (task 542) only adds a
    NEW distinction on the status="unknown" side of ``consent_terminal`` and
    must carry v7's http-side behavior forward unchanged, or a caller taking
    the default ``CURRENT_SCHEMA_VERSION`` would silently regress every
    429/502/503/504 straight back to ``http_terminal`` the moment v8 became
    current -- exactly the class of defect this file's own forward-only
    versioning exists to prevent.
    """
    if schema_version not in SCHEMA_VERSIONS:
        raise DirectTerminalContractError(
            f"unknown terminal schema version: {schema_version!r}")
    if 200 <= code < 300 and not has_body:
        return "http_terminal", "empty_success_body"
    if (schema_version in (SCHEMA_VERSION_V7, SCHEMA_VERSION_V8)
            and code in RETRY_LATER_HTTP_STATUS):
        return "retry_queued", f"http_{code}"
    return "http_terminal", f"http_{code}"
TEXTUAL_MEDIA_TYPES = frozenset({
    "text/html", "application/xhtml+xml", "application/xml", "text/xml",
    "application/json", "text/json", "text/plain", "text/markdown",
})


class DirectTerminalContractError(ValueError):
    """Direct-terminal identity, consent, or outcome evidence is invalid."""


def origin_parts(url: str) -> tuple[str, str, str, str]:
    """Return exact normalized origin, scheme, DNS/IP host, and authority."""
    identity = classify_url(url)
    if identity["fetchable"] is not True:
        raise DirectTerminalContractError(
            f"URL is not fetchable ({identity['url_shape']}): {url}")
    return (identity["origin"], identity["origin_scheme"], identity["host"],
            identity["authority"])


def classify_url(url: str) -> dict[str, Any]:
    """Classify an exact URL without turning unsafe shapes into requests.

    A credential-bearing HTTP URL still has an observable RFC origin, which
    is useful for honest census accounting, but it is never fetchable and is
    never assigned to that origin's worker.
    """
    if not isinstance(url, str) or not url:
        raise DirectTerminalContractError("URL must be a non-empty string")
    if url != url.strip() or any(ord(value) < 0x20 or ord(value) == 0x7f
                                 for value in url):
        return {"origin": None, "origin_scheme": None, "host": None,
                "authority": None, "fetchable": False,
                "url_shape": "whitespace_or_control"}
    try:
        parsed = urlsplit(url)
        raw_host = parsed.hostname
    except ValueError:
        return {"origin": None, "origin_scheme": None, "host": None,
                "authority": None, "fetchable": False,
                "url_shape": "invalid_authority"}
    scheme = parsed.scheme.casefold()
    host = (raw_host or "").casefold().strip(".")
    shape = "fetchable"
    if not scheme:
        shape = "missing_scheme"
    elif scheme not in {"http", "https"}:
        shape = "unsupported_scheme"
    elif not host:
        shape = "missing_host"
    try:
        port = parsed.port
    except ValueError:
        port = None
        shape = "invalid_authority"
    authority = origin = None
    if scheme in {"http", "https"} and host and shape != "invalid_authority":
        display_host = f"[{host}]" if ":" in host else host
        authority = display_host if port is None else f"{display_host}:{port}"
        origin = f"{scheme}://{authority}"
        if parsed.username is not None or parsed.password is not None:
            shape = "userinfo"
    return {"origin": origin, "origin_scheme": scheme or None,
            "host": host or None, "authority": authority,
            "fetchable": shape == "fetchable", "url_shape": shape}


def validate_frontier_row(row: Any) -> dict[str, Any]:
    """Validate and normalize one exact-URL/point-count residual row."""
    if not isinstance(row, dict):
        raise DirectTerminalContractError("frontier row is not an object")
    url = row.get("url")
    points = row.get("point_count")
    ordinal = row.get("ordinal")
    if (not isinstance(url, str) or not url
            or type(points) is not int or points <= 0
            or type(ordinal) is not int or ordinal < 0):
        raise DirectTerminalContractError("frontier row fields are invalid")
    identity = classify_url(url)
    expected = identity
    for key, value in expected.items():
        if key in row and row[key] != value:
            raise DirectTerminalContractError(
                f"frontier row {key} identity drift for {url}")
    return {**row, **expected, "url": url, "point_count": points,
            "ordinal": ordinal}


def partition_for_origin(origin: str, partition_count: int) -> int:
    if type(partition_count) is not int or not 1 <= partition_count <= 1024:
        raise DirectTerminalContractError("partition_count must be 1..1024")
    value = int.from_bytes(hashlib.sha256(origin.encode("utf-8")).digest()[:8],
                           "big")
    return value % partition_count


def partition_for_frontier_row(row: Any, partition_count: int) -> int:
    """Keep fetchable origins together; isolate no-network rows by exact URL."""
    value = validate_frontier_row(row)
    if value["fetchable"] is True:
        return partition_for_origin(value["origin"], partition_count)
    digest = hashlib.sha256(
        f"no-network\0{value['url']}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % partition_count


def invalid_url_reason(row: Any) -> str:
    value = validate_frontier_row(row)
    if value["fetchable"] is True:
        raise DirectTerminalContractError("fetchable URL has no invalid reason")
    return "invalid_url_" + value["url_shape"]


def consent_key(origin: str) -> str:
    return hashlib.sha256(f"{origin}\0{MODE}".encode("utf-8")).hexdigest()


def terminal_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def validate_consent(receipt: Any, origin: str) -> dict[str, Any]:
    """Require an exact origin/mode direct-consent result without rebadging."""
    if not isinstance(receipt, dict):
        raise DirectTerminalContractError("consent receipt is not an object")
    expected_origin, scheme, _host, authority = origin_parts(origin)
    if (expected_origin != origin or receipt.get("origin") != origin
            or receipt.get("origin_scheme") != scheme
            or receipt.get("host") != authority
            or receipt.get("mode") != MODE
            or receipt.get("status") not in {"ok", "unknown"}
            or type(receipt.get("search_ok")) is not bool
            or type(receipt.get("train_ok")) is not bool
            or not isinstance(receipt.get("checked_at"), str)
            or not receipt["checked_at"]):
        raise DirectTerminalContractError(
            f"direct consent identity/mode/origin drift: {origin}")
    return receipt


def consent_terminal(receipt: dict[str, Any], url: str, *,
                     schema_version: str = CURRENT_SCHEMA_VERSION
                     ) -> tuple[str, str] | None:
    """Return the fail-closed terminal outcome, or None when fetch is allowed.

    ``schema_version`` selects the reason taxonomy (see ``SCHEMA_VERSION_V6``
    above); it never changes what ``direct_url_allowed`` decides, only which
    reason string names a search-ok=False/path-declined verdict (v6+) or a
    status="unknown" verdict (v8+). Pass ``SCHEMA_VERSION_V5`` only to
    re-derive a terminal against evidence sealed before the v6 split
    existed, or an earlier version to re-derive one sealed before v8's.
    """
    if schema_version not in SCHEMA_VERSIONS:
        raise DirectTerminalContractError(
            f"unknown direct-terminal schema version: {schema_version!r}")
    origin, _scheme, _host, _authority = origin_parts(url)
    validate_consent(receipt, origin)
    split_unreadable = schema_version != SCHEMA_VERSION_V5
    split_unresolved_reason = schema_version == SCHEMA_VERSION_V8
    if receipt["status"] != "ok":
        if (split_unresolved_reason
                and CR.unresolved_terminal_reason(receipt) == "consent_unreadable"):
            return "held", "consent_unreadable"
        return "held", "consent_unresolved"
    if receipt["search_ok"] is not True:
        if split_unreadable and CR.robots_unreadable(receipt):
            return "held", "consent_unreadable"
        return "declined", "origin_search_declined"
    if not CR.direct_url_allowed(receipt, url):
        if split_unreadable and CR.robots_unreadable(receipt):
            return "held", "consent_unreadable"
        return "declined", "path_robots_declined"
    return None


def media_type(headers: list[list[str]]) -> str:
    values = [value for key, value in headers
              if key.casefold() == "content-type"]
    if not values:
        return ""
    return values[-1].split(";", 1)[0].strip().casefold()


def is_textual(value: str) -> bool:
    return value in TEXTUAL_MEDIA_TYPES or value.startswith("text/")


def exclusively_public_host(host: str) -> bool:
    """Fail closed unless every current address for the exact host is public."""
    try:
        literal = ipaddress.ip_address(host)
        return literal.is_global
    except ValueError:
        pass
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(
            host, 443, type=socket.SOCK_STREAM)}
        return bool(addresses) and all(
            ipaddress.ip_address(value).is_global for value in addresses)
    except (OSError, ValueError):
        return False
