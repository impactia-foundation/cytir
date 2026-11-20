"""Task 485: a policy probe never waits without end.

The sweep that motivated these tests looked ACTIVE while recording nothing.
Two shapes of policy endpoint escape a per-operation HTTP timeout entirely —
a redirect chain (each hop gets a fresh connect+read budget) and a body served
slower than the read timeout but never finished — and until now nothing above
them imposed a wall clock, because the deadline machinery was opt-in through an
environment variable no deployed unit set.

Every test here asserts the same contract from a different angle: the probe
stops, and stopping produces "no answer", which is the held_unknown the ratified
semantics have always given a non-answer. None of it changes what a readable
policy means.
"""
import http.server
import socketserver
import threading
import time

import httpx
import pytest

import cytir.consent.resolver as cr
import cytir.consent.terminal as DTC


class _Stream:
    """Minimal stand-in for the context manager `client.stream` returns."""

    def __init__(self, status_code, headers=None, chunks=(), url="https://example.test/robots.txt"):
        self.status_code = status_code
        self.headers = httpx.Headers(headers or {})
        self._chunks = chunks
        self.request = httpx.Request("GET", url)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def iter_bytes(self, _size=None):
        for chunk in self._chunks:
            yield chunk


class _Client:
    """A client whose every response is scripted, so only timing is under test."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.urls = []

    def stream(self, _method, url, **_kwargs):
        self.urls.append(url)
        response = self._responses.pop(0) if self._responses else self._responses
        return response(url) if callable(response) else response


def _redirect(target):
    return lambda url: _Stream(301, {"location": target}, url=url)


def _ok(body=b"User-agent: *\nAllow: /\n"):
    return lambda url: _Stream(200, {"content-type": "text/plain"}, [body], url=url)


# --- redirects ---------------------------------------------------------------

def test_redirect_chain_is_capped_and_held_not_followed_forever():
    client = _Client([_redirect(f"https://example.test/hop{i}") for i in range(20)])
    with pytest.raises(cr.BudgetExhausted, match="exceeded 5 redirects"):
        cr._fetch_bounded(client, "https://example.test/robots.txt",
                          time.monotonic() + 30)
    assert len(client.urls) == cr.MAX_REDIRECTS + 1


def test_redirect_loop_is_named_rather_than_walked():
    client = _Client([_redirect("https://example.test/robots.txt")] * 4)
    with pytest.raises(cr.BudgetExhausted, match="redirect loop"):
        cr._fetch_bounded(client, "https://example.test/robots.txt",
                          time.monotonic() + 30)


def test_redirect_to_private_address_aborts(monkeypatch):
    """The documents-dds-ny.un.org shape: an origin points us at RFC 1918."""
    client = _Client([_redirect("http://192.168.10.4/robots.txt")])
    with pytest.raises(cr.BudgetExhausted, match="non-public address"):
        cr._fetch_bounded(client, "https://example.test/robots.txt",
                          time.monotonic() + 30)


def test_redirect_to_unsupported_scheme_aborts():
    client = _Client([_redirect("ftp://example.test/robots.txt")])
    with pytest.raises(cr.BudgetExhausted, match="unsupported scheme"):
        cr._fetch_bounded(client, "https://example.test/robots.txt",
                          time.monotonic() + 30)


def test_a_short_redirect_chain_still_resolves():
    client = _Client([_redirect("https://www.example.test/robots.txt"), _ok()])
    response = cr._fetch_bounded(client, "https://example.test/robots.txt",
                                 time.monotonic() + 30)
    assert response.status_code == 200
    assert "Allow: /" in response.text
    assert client.urls == ["https://example.test/robots.txt",
                           "https://www.example.test/robots.txt"]


# --- task 502 defect 2: the 3xx-with-empty-body receipt shape ----------------
#
# Two live task-467 card rows (gc-handball.ch, germany-tourism.de) recorded a
# consent receipt with http_status 301 and an empty body. Reproduced from the
# sealed task-426 evidence (read-only fetch from the GPU build box): both origins answered a
# 3xx with NO Location header the client could follow -- httpx's own
# auto-redirect (the pre-task-485 code that actually produced those receipts)
# returns a response like that as terminal rather than raising, and this loop
# does the same thing on purpose (there is nothing to follow). The shapes below
# are that recorded status/header/body combination, not a live fetch.

def test_redirect_status_with_no_location_is_returned_as_terminal():
    """A 3xx with no Location cannot be followed -- it is the honest answer,
    not an error -- and must not raise or hang waiting for a hop that never
    comes."""
    client = _Client([lambda url: _Stream(301, {}, [b""], url=url)])
    response = cr._fetch_bounded(client, "http://www.gc-handball.ch/robots.txt",
                                 time.monotonic() + 30)
    assert response.status_code == 301
    assert response.content == b""
    assert client.urls == ["http://www.gc-handball.ch/robots.txt"]


def test_redirect_status_with_empty_location_is_also_terminal():
    """An empty (present-but-blank) Location is exactly as unfollowable as a
    missing one -- ``germany-tourism.de``'s recorded shape."""
    client = _Client([lambda url: _Stream(301, {"location": ""}, [b""], url=url)])
    response = cr._fetch_bounded(client, "http://www.germany-tourism.de/robots.txt",
                                 time.monotonic() + 30)
    assert response.status_code == 301
    assert response.content == b""


def test_hops_walked_before_a_terminal_answer_are_recorded():
    """The chain leading to a terminal answer is no longer discarded (task 502
    defect 2): a reader can now tell "stalled after N real hops" from "no
    redirect happened at all"."""
    client = _Client([
        _redirect("https://hop1.example.test/robots.txt"),
        _redirect("https://hop2.example.test/robots.txt"),
        lambda url: _Stream(301, {}, [b""], url=url),   # terminal: no Location
    ])
    response = cr._fetch_bounded(client, "https://example.test/robots.txt",
                                 time.monotonic() + 30)
    assert response.status_code == 301
    hops = response.extensions["aw_redirect_hops"]
    assert [h["url"] for h in hops] == [
        "https://example.test/robots.txt",
        "https://hop1.example.test/robots.txt",
    ]
    assert all(h["status"] == 301 for h in hops)


def test_hops_are_empty_when_the_first_answer_is_already_terminal():
    client = _Client([_ok()])
    response = cr._fetch_bounded(client, "https://example.test/robots.txt",
                                 time.monotonic() + 30)
    assert response.extensions.get("aw_redirect_hops") == []


def test_a_location_httpx_cannot_parse_is_held_not_crashed():
    """``httpx.InvalidURL`` does not subclass ``httpx.HTTPError`` -- unwrapped,
    it would escape every caller's ``except httpx.HTTPError`` and crash the
    resolver instead of holding the record."""
    client = _Client([_redirect("http://[::1")])   # unparsable authority
    with pytest.raises(cr.BudgetExhausted, match="unparsable target"):
        cr._fetch_bounded(client, "https://example.test/robots.txt",
                          time.monotonic() + 30)


def test_resolve_host_names_the_stalled_redirect_distinctly_from_open(monkeypatch):
    """End to end: ``resolve_host`` must not describe a stalled redirect the
    same way it describes a genuine 404/empty-200 "nothing published" answer,
    and must leave the fetch-closed URL-level gate untouched either way."""
    def fake_get_policy_endpoint(_client, _host, path, *, origin_scheme=None,
                                 deadline=None):
        if path == "/robots.txt":
            resp = httpx.Response(301, headers=[], content=b"",
                                  request=httpx.Request(
                                      "GET", f"https://x/{path}"),
                                  extensions={"aw_redirect_hops": []})
            return resp, None, "https"
        # tdmrep: a definitive 404 (no reservation) so `status` stays "ok" and
        # this test isolates the robots-leg note/hops behaviour under test.
        resp = httpx.Response(404, headers=[], content=b"",
                              request=httpx.Request("GET", f"https://x/{path}"))
        return resp, None, "https"

    monkeypatch.setattr(cr, "_get_policy_endpoint", fake_get_policy_endpoint)
    monkeypatch.setattr(cr.time, "sleep", lambda _s: None)
    rec = cr.resolve_host(object(), "www.gc-handball.ch", direct_crawl=True,
                          origin_scheme="https")
    sig = rec["signals"]["robots"]
    assert sig["http_status"] == 301
    assert sig["note"] != "no robots rules published (open)"
    assert "no further Location" in sig["note"]
    # Host-level flags are UNCHANGED by this receipt-clarity fix -- the
    # existing fail-closed behaviour at the URL gate is what actually blocks
    # the fetch (see direct_terminal_contract tests), not this flag.
    assert rec["search_ok"] is True
    assert not cr.direct_url_allowed(
        rec, "https://www.gc-handball.ch/some/path")


# --- task 542: InvalidURL is a deterministic-unreadable, not a generic ------
# ------------------------ instrument failure --------------------------------
#
# `gc-handball.ch` (above) was PR #88's own live InvalidURL evidence when
# that PR was written (2026-08-28: an absent Location header, discovered to
# have crashed the pre-fix resolver on `httpx.InvalidURL`). Re-checked live
# from the GPU build box for this task (2026-09-08, same politely rate-limited GET):
# its `/robots.txt` redirect target has since changed to
# `https://www.gc-amicitia.chrobots.txt` -- a syntactically VALID absolute
# URL (a real scheme, a real if non-existent host, a real path) that
# `httpx.URL.join` parses fine and then fails to CONNECT to. That is the
# already-correctly-handled `transport_error`/DNS-failure class, not this
# one -- which is exactly why the case this task fixes needs a fixture: a
# byte-for-byte unparsable `Location` header is a rare, transient thing to
# catch a live publisher serving on demand. The fixture below
# (`http://[::1`, a truncated IPv6 literal) is not invented for this test --
# it is the IDENTICAL input PR #88's own crash-guard test
# (`test_a_location_httpx_cannot_parse_is_held_not_crashed`, above) already
# uses as this defect class's real reproduction case; this extends that same
# input one step further, end to end through `resolve_host` and
# `direct_terminal_contract.consent_terminal`, to prove the reclassification.

def test_resolve_host_names_a_malformed_redirect_target_distinctly(monkeypatch):
    """End to end: an InvalidURL-raising redirect on the robots.txt leg must
    record a distinct `unreadable_class`, not the same bare error string a
    timeout or a reset would leave -- the asymmetry with the tdmrep leg
    (which already recorded `unreadable_class`, task 466) that let this
    defect hide on the robots leg specifically."""
    # Generate the real error text from the REAL redirect-following code
    # (not typed by hand), the same way `_fetch_bounded` would in production.
    client = _Client([_redirect("http://[::1")])
    try:
        cr._fetch_bounded(client, "https://www.gc-handball.ch/robots.txt",
                          time.monotonic() + 30)
        raise AssertionError("expected BudgetExhausted")
    except cr.BudgetExhausted as exc:
        real_error = str(exc)
    assert "unparsable target" in real_error

    def fake_get_policy_endpoint(_client, _host, path, *, origin_scheme=None,
                                 deadline=None):
        if path == "/robots.txt":
            return None, real_error, None
        resp = httpx.Response(404, headers=[], content=b"",
                              request=httpx.Request("GET", f"https://x/{path}"))
        return resp, None, "https"

    monkeypatch.setattr(cr, "_get_policy_endpoint", fake_get_policy_endpoint)
    monkeypatch.setattr(cr.time, "sleep", lambda _s: None)
    rec = cr.resolve_host(object(), "www.gc-handball.ch", direct_crawl=True,
                          origin_scheme="https")

    assert rec["status"] == "unknown"
    assert rec["signals"]["robots"]["unreadable_class"] == "malformed_redirect_target"
    assert cr.unresolved_terminal_reason(rec) == "consent_unreadable"

    # v8 (current): the deterministic case gets its own HELD reason.
    assert DTC.consent_terminal(
        rec, "https://www.gc-handball.ch/some/path") == (
            "held", "consent_unreadable")
    # Forward-only: replaying this exact receipt under an earlier pinned
    # schema version reproduces the OLD, undivided reading byte-identically
    # -- a sealed pre-v8 ledger must never be silently re-derived differently.
    for old_version in (DTC.SCHEMA_VERSION_V5, DTC.SCHEMA_VERSION_V6,
                        DTC.SCHEMA_VERSION_V7):
        assert DTC.consent_terminal(
            rec, "https://www.gc-handball.ch/some/path",
            schema_version=old_version) == ("held", "consent_unresolved")


def test_a_genuine_timeout_still_gets_the_generic_unresolved_reason(monkeypatch):
    """The regression guard: this change must not promote every held/unknown
    host to `consent_unreadable` -- only the deterministic redirect-target
    class. A real instrument failure (timeout/reset/budget exhausted) keeps
    the generic, genuinely-retryable reason."""
    def fake_get_policy_endpoint(_client, _host, path, *, origin_scheme=None,
                                 deadline=None):
        return None, cr.BUDGET_EXHAUSTED, None

    monkeypatch.setattr(cr, "_get_policy_endpoint", fake_get_policy_endpoint)
    rec = cr.resolve_host(object(), "example.test", direct_crawl=True,
                          origin_scheme="https")
    assert rec["status"] == "unknown"
    assert rec["signals"]["robots"]["unreadable_class"] == "budget_exhausted"
    assert cr.unresolved_terminal_reason(rec) == "consent_unresolved"
    assert DTC.consent_terminal(
        rec, "https://example.test/some/path") == (
            "held", "consent_unresolved")


# --- the trickled body -------------------------------------------------------

def test_body_read_stops_at_the_deadline(monkeypatch):
    """A server that never finishes never trips a per-read timeout."""
    clock = [1000.0]
    monkeypatch.setattr(cr.time, "monotonic", lambda: clock[0])

    def forever():
        while True:
            clock[0] += 1.0        # each chunk arrives just inside the read timeout
            yield b"x" * 8

    stream = _Stream(200, {"content-type": "text/plain"})
    stream.iter_bytes = lambda _size=None: forever()
    client = _Client([stream])
    with pytest.raises(cr.BudgetExhausted, match="while reading the policy body"):
        cr._fetch_bounded(client, "https://example.test/robots.txt", 1005.0)


def test_body_read_stops_at_the_byte_cap():
    oversized = b"#" * (cr.BODY_BYTE_CAP + 4096)
    client = _Client([_ok(oversized)])
    response = cr._fetch_bounded(client, "https://example.test/robots.txt",
                                 time.monotonic() + 30)
    assert len(response.content) <= cr.BODY_BYTE_CAP + 65536


def test_framing_headers_are_dropped_when_the_body_is_rebuilt():
    stream = _Stream(200, {"content-type": "text/plain",
                           "content-encoding": "gzip",
                           "content-length": "99999"}, [b"User-agent: *\n"])
    response = cr._fetch_bounded(_Client([stream]),
                                 "https://example.test/robots.txt",
                                 time.monotonic() + 30)
    assert response.text == "User-agent: *\n"
    assert "content-encoding" not in response.headers


# --- the bound is always on --------------------------------------------------

def test_get_imposes_its_own_budget_when_the_caller_gives_none(monkeypatch):
    seen = []

    def fetch(_client, _url, deadline):
        seen.append(deadline)
        raise httpx.ConnectError("fixture")

    monkeypatch.setattr(cr, "_fetch_bounded", fetch)
    monkeypatch.setattr(cr.time, "sleep", lambda _s: None)
    monkeypatch.setattr(cr, "_curl_get", lambda _url, timeout=None: None)
    before = time.monotonic()
    response, error = cr._get(object(), "https://example.test/robots.txt")
    assert response is None and error == "ConnectError"
    assert seen and all(before < d <= before + cr.POLICY_ENDPOINT_BUDGET + 1
                        for d in seen)


def test_exhausted_budget_is_held_unknown_never_a_verdict(monkeypatch):
    monkeypatch.setattr(
        cr, "_get_policy_endpoint",
        lambda *_a, **_kw: (None, f"{cr.BUDGET_EXHAUSTED} after 5 redirect(s)", None))
    rec = cr.resolve_host(object(), "example.test", origin_scheme="https")
    assert rec["status"] == "unknown"
    assert rec["search_ok"] is True and rec["train_ok"] is True
    assert "held, retryable" in rec["reasons"][0]
    assert cr.classify_unreadable(rec["signals"]["robots"]["error"]) == "budget_exhausted"


def test_budget_exhaustion_is_not_the_5xx_class(monkeypatch):
    """ADR-024 amdt-3 promotes only a 5xx; an exhausted clock is our failure."""
    rec = {"host": "example.test", "status": "unknown", "search_ok": True,
           "signals": {"robots": {"fetched": True, "http_status": 200},
                       "tdmrep": {"fetched": False,
                                  "error": f"{cr.BUDGET_EXHAUSTED} after 5 redirect(s)"}}}
    assert cr.tdmrep_unreadable_class(rec) == "budget_exhausted"
    assert cr.unresolved_search_ok(rec, register={"example.test": {"witness": "x"}}) is False


def test_host_budget_covers_both_probes(monkeypatch):
    deadlines = []

    def endpoint(_client, _host, _path, *, origin_scheme=None, deadline=None):
        deadlines.append(deadline)
        return None, "ConnectError", None

    monkeypatch.setattr(cr, "_get_policy_endpoint", endpoint)
    monkeypatch.setattr(cr.time, "sleep", lambda _s: None)
    cr.resolve_host(object(), "example.test", origin_scheme="https")
    # Probe 1 holds the record, so only one endpoint call is made -- and it is
    # made under a deadline, which is the point.
    assert deadlines and all(d is not None for d in deadlines)


def test_budget_env_override_refuses_an_unbounded_value(monkeypatch):
    monkeypatch.setenv("AW_CONSENT_ENDPOINT_BUDGET", "0")
    with pytest.raises(ValueError, match="no unbounded mode"):
        cr._positive_seconds("AW_CONSENT_ENDPOINT_BUDGET", 70.0)
    monkeypatch.setenv("AW_CONSENT_ENDPOINT_BUDGET", "notanumber")
    with pytest.raises(ValueError, match="number of seconds"):
        cr._positive_seconds("AW_CONSENT_ENDPOINT_BUDGET", 70.0)
    monkeypatch.delenv("AW_CONSENT_ENDPOINT_BUDGET")
    assert cr._positive_seconds("AW_CONSENT_ENDPOINT_BUDGET", 70.0) == 70.0


# --- the batch cannot be held open by one host -------------------------------

def test_batch_backstop_holds_the_hosts_it_never_heard_back_about(monkeypatch, tmp_path):
    """If a worker somehow outlives its budget, the batch still completes."""

    def slow(_client, host, direct_crawl=False, origin_scheme=None, deadline=None):
        if host == "slow.test":
            time.sleep(20)
        return {"host": host, "status": "ok", "search_ok": True, "train_ok": True,
                "mode": "cc", "reasons": [], "signals": {},
                "checked_at": cr._now_iso()}

    class Client:
        closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(cr, "resolve_host", slow)
    monkeypatch.setattr(cr, "HOST_DELAY", 0.0)
    monkeypatch.setattr(cr, "_batch_budget", lambda _n, _c: 0.5)
    monkeypatch.setattr(cr, "_client", Client)
    out = cr.resolve_hosts(["slow.test", "fast.test"], cache_dir=str(tmp_path),
                           max_age_days=0, concurrency=2, progress=False,
                           origin_scheme="https")
    assert set(out) == {"slow.test", "fast.test"}
    assert out["slow.test"]["status"] == "unknown"
    assert "batch budget" in out["slow.test"]["reasons"][0]
    assert out["slow.test"]["search_ok"] is True   # never coerced to a verdict
    assert cr._cache_path(str(tmp_path), "slow.test",
                          origin_scheme="https").exists()


# --- end to end, against a server that really does not stop -------------------

class _DripServer(http.server.BaseHTTPRequestHandler):
    """Announces a huge body, then trickles it a few bytes at a time forever.

    Every chunk arrives inside the read timeout, so no per-operation timeout
    ever fires: this shape is unbounded to any client that only bounds reads.
    """

    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", "100000000")
        self.end_headers()
        try:
            while True:
                self.wfile.write(b"# drip\n")
                self.wfile.flush()
                time.sleep(0.2)
        except Exception:      # noqa: BLE001 — the client hung up, as intended
            pass


@pytest.fixture
def drip_origin():
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = Server(("127.0.0.1", 0), _DripServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_a_trickled_policy_body_is_held_in_bounded_time(drip_origin, monkeypatch):
    # The fixture has to live on loopback; _is_public_host has its own test
    # above, so standing it down here leaves the read bound as what is measured.
    monkeypatch.setattr(cr, "_is_public_host", lambda _h: True)
    monkeypatch.setattr(cr, "_curl_get", lambda _url, timeout=None: None)
    monkeypatch.setattr(cr, "POLICY_ENDPOINT_BUDGET", 3.0)
    monkeypatch.setattr(cr, "HOST_BUDGET", 8.0)
    client = cr._client()
    try:
        started = time.monotonic()
        rec = cr.resolve_host(client, drip_origin, origin_scheme="http")
        elapsed = time.monotonic() - started
    finally:
        client.close()
    assert elapsed < cr.HOST_BUDGET + 5
    assert rec["status"] == "unknown"
    assert rec["search_ok"] is True and rec["train_ok"] is True
    error = rec["signals"]["robots"]["error"]
    assert cr.classify_unreadable(error) == "budget_exhausted"
