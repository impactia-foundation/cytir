"""Round 4, F1 (HIGH) of #553: no Qdrant-backed test writes to an endpoint that is not provably a
throwaway. Needs no Qdrant: a recording STUB server plays the endpoint, so the test can count what
the harness sends. The stub answers the two listings (`GET /aliases`, `GET /collections`) from a
planted state and records every request; it never changes that state.

The planted "live" instance is the reviewer's scenario: a tunnel to prod on a port the old port list
did not know, holding the alias `aw_prod` and the collection `aw_e9`.
"""
from __future__ import annotations

import http.server
import json
import pathlib
import sys
import threading

import pytest

pytestmark = pytest.mark.guard

HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(HERE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import drain_transaction_harness as H  # noqa: E402
import throwaway_qdrant as TQ  # noqa: E402


class Stub:
    """A fake Qdrant that records every request and answers the listings from a planted state."""

    def __init__(self, aliases=None, collections=(), listing_status=200):
        self.aliases = dict(aliases or {})
        self.collections = list(collections)
        self.listing_status = listing_status
        self.requests: list[tuple[str, str]] = []
        stub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _answer(self):
                stub.requests.append((self.command, self.path))
                n = int(self.headers.get("content-length") or 0)
                if n:
                    self.rfile.read(n)
                if self.command == "GET" and stub.listing_status != 200:
                    self.send_response(stub.listing_status)
                    self.end_headers()
                    return
                if self.command == "GET" and self.path == "/aliases":
                    res = {"aliases": [{"alias_name": a, "collection_name": c}
                                       for a, c in stub.aliases.items()]}
                elif self.command == "GET" and self.path == "/collections":
                    res = {"collections": [{"name": c} for c in stub.collections]}
                else:
                    res = True
                body = json.dumps({"result": res, "status": "ok"}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_PUT = do_POST = do_DELETE = _answer

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def writes(self):
        return [r for r in self.requests if r[0] != "GET"]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub():
    made: list[Stub] = []

    def make(**kw):
        s = Stub(**kw)
        made.append(s)
        return s
    yield make
    for s in made:
        s.close()
    TQ._PROVEN.clear()


@pytest.fixture(autouse=True)
def _clean_proofs():
    TQ._PROVEN.clear()
    yield
    TQ._PROVEN.clear()


PLANTED_LIVE = [
    pytest.param(dict(aliases={"aw_prod": "aw_e9"}, collections=["aw_e9", "aw_e8", "source_cards"]),
                 id="prod-alias-and-live-collections"),
    pytest.param(dict(collections=["aw_e9"]), id="only-aw_e9"),
    pytest.param(dict(collections=["aw_e8"]), id="only-aw_e8"),
    pytest.param(dict(collections=["source_cards"]), id="only-source_cards"),
    pytest.param(dict(aliases={"aw_prod": "x"}, collections=["x"]), id="only-the-alias"),
    pytest.param(dict(aliases={"some_other_alias": "x"}, collections=["x"]), id="any-alias"),
    pytest.param(dict(collections=["customer_data"]), id="non-empty-and-unmarked"),
    pytest.param(dict(collections=[TQ.SENTINEL, "aw_e9"]), id="marked-but-holds-a-live-name"),
    pytest.param(dict(listing_status=500), id="listing-fails"),
]


@pytest.mark.parametrize("planted", PLANTED_LIVE)
def test_a_planted_live_instance_is_refused_without_one_write(stub, planted, monkeypatch):
    """The reviewer's tunnel on an UNLISTED port: it is refused by what it holds, and the stub
    saw only GETs. Through the harness entry point, as a test would call it."""
    s = stub(**planted)
    assert s.server.server_address[1] not in TQ.LIVE_PORTS      # the port list does not catch it
    monkeypatch.setenv("AW_TEST_QDRANT_URL", s.url)
    with pytest.raises(TQ.ThrowawayRefused):
        H.qdrant_url()
    assert s.writes() == [], s.requests


@pytest.mark.parametrize("planted", PLANTED_LIVE)
def test_the_harness_writes_and_deletes_refuse_an_unproven_endpoint_with_no_request(stub, planted):
    """The second belt: even a test that skips `qdrant_url()` and calls a helper directly cannot
    delete `aw_e9` or `aw_e8`/`source_cards`, nor move an alias: nothing is sent at all."""
    s = stub(**planted)
    for call in (lambda: H.reset_collection(s.url, "aw_e9"),
                 lambda: H.clear_live(s.url),
                 lambda: H.set_alias(s.url, "aw_prod", "aw_e9"),
                 lambda: H.api(s.url, "/collections/aw_e9", method="DELETE"),
                 lambda: H.api(s.url, "/collections/x", {"vectors": {}})):
        with pytest.raises(TQ.ThrowawayRefused):
            call()
    assert s.requests == [], s.requests
    # a refusal is not swallowed by drop_alias's "a missing alias is fine"
    with pytest.raises(TQ.ThrowawayRefused):
        H.drop_alias(s.url, "aw_prod")
    assert s.requests == []


def test_an_empty_instance_is_marked_by_exactly_one_put(stub, monkeypatch):
    s = stub()
    monkeypatch.setenv("AW_TEST_QDRANT_URL", s.url)
    assert H.qdrant_url() == s.url
    assert s.writes() == [("PUT", f"/collections/{TQ.SENTINEL}")], s.requests
    n = len(s.requests)
    assert H.qdrant_url() == s.url and len(s.requests) == n       # proven once per session


def test_a_marked_instance_that_holds_no_live_name_is_accepted_without_a_write(stub, monkeypatch):
    s = stub(collections=[TQ.SENTINEL, "awtest_x"])
    monkeypatch.setenv("AW_TEST_QDRANT_URL", s.url)
    assert H.qdrant_url() == s.url
    assert s.writes() == []


def test_the_port_list_is_still_a_second_belt(monkeypatch):
    monkeypatch.setenv("AW_TEST_QDRANT_URL", "http://127.0.0.1:6334")
    with pytest.raises(AssertionError, match="live Qdrant"):
        TQ.require_throwaway()
    monkeypatch.setenv("AW_TEST_QDRANT_URL", "http://prod.example:9999")
    with pytest.raises(AssertionError, match="only a local"):
        TQ.require_throwaway()


def test_the_cards_env_override_cannot_unlist_the_default_cards_collection(stub, monkeypatch):
    s = stub(collections=["source_cards"])
    monkeypatch.setenv("AW_CARDS_COLLECTION", "cards_other")
    with pytest.raises(TQ.ThrowawayRefused):
        TQ.prove_throwaway(s.url)
    assert s.writes() == []


def test_every_qdrant_backed_test_module_takes_its_url_from_the_shared_gate():
    """No test module may read `AW_TEST_QDRANT_URL` itself (a private copy of the guard is how the
    port list survived): the env var is read in `tests/throwaway_qdrant.py` only."""
    app = HERE.parent
    offenders = []
    for d in (app / "pipeline", app / "tests"):
        for p in sorted(d.glob("*.py")):
            if p.name in ("throwaway_qdrant.py", pathlib.Path(__file__).name):
                continue
            text = p.read_text()
            reads = "os.environ.get(\"AW_TEST_QDRANT_URL\"" in text or "environ['AW_TEST_QDRANT_URL']" in text
            if reads and "throwaway_qdrant" not in text:
                offenders.append(p.name)
    assert offenders == [], offenders


# --------------------------------------------------------------------------------------------
# Round 5, N1 of #559: no test hands a LITERAL endpoint to a client or to `cyt index` / drain code.
# A test that spelled `http://127.0.0.1` reached :6333, the default Qdrant port: on a stranger's box
# that is their data, on a CI runner it can be a legacy store. The URL a test uses comes from
# `throwaway_qdrant.require_throwaway()` (via `H.qdrant_url()`); a literal is a second way to spell it.
# --------------------------------------------------------------------------------------------
import ast  # noqa: E402
import re  # noqa: E402

_ENDPOINT = re.compile(r"^(?:https?://)?(?P<host>[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*|\[[0-9A-Fa-f:.]+\])"
                       r"(?::(?P<port>\d{1,5}))?/?$")
_ADDRESS = re.compile(r"^(localhost|0\.0\.0\.0|\d+\.\d+\.\d+\.\d+|\[[0-9A-Fa-f:.]+\])$")
#: Hosts that cannot resolve (RFC 2606 / 6761): a request to one never leaves the process.
_UNRESOLVABLE = re.compile(r"(\.invalid|\.example|\.test|(^|\.)example\.(com|org|net))$")
#: Ports no Qdrant listens on (tcpmux, discard): a literal that names one is a refused connection.
_DEAD_PORTS = {1, 9}

#: (file name, literal) -> why this literal reaches no live port. Each was read, and the modules were run with every outbound connect refused
#: (nothing connected: see the PR body).
ALLOWED_LITERALS = {
    # consent step 5 (tasks 1648, 1673): the internal door is exercised through httpx.MockTransport; no socket opens
    ("test_aw_consent_step5_index.py", "http://qdrant.internal:6333"): "MockTransport behind the internal door; nothing connects",
    ("test_aw_consent_step5_internal.py", "http://qdrant.internal:6333"): "MockTransport behind the internal door; nothing connects",
    ("test_aw_consent_step5_internal.py", "http://tei.internal:8080"): "MockTransport behind the internal door; nothing connects",
    ("test_aw_internal_client.py", "http://qdrant.internal:6333"): "MockTransport; nothing connects",
    ("test_aw_internal_client.py", "http://localhost:6333"): "a pure host check of the internal client; nothing connects",
    ("test_serve_model_pin_read_side.py", "http://tei:80"): "a monkeypatched TEI setting in a pure read-side test; no client connects (moved from server/, S10 step 9)",
    ("test_serve_s10_pure_modules.py", "http://x:6333"): "a monkeypatched setting read by a stubbed client factory; nothing connects (moved from server/, S10 step 9)",
    ("test_aw_distribute_dump.py", "http://127.0.0.1:16333"):
        "assert_qdrant_url_is_not_prod() is a pure string check; it never opens a connection",
    ("test_aw_distribute_dump.py", "http://localhost:16333"): "same pure check",
    ("test_aw_distribute_dump.py", "http://127.0.0.1:9999"): "same pure check",
    ("test_aw_server_rehearsal.py", "http://127.0.0.1:6486"):
        "check_scratch() is a pure string check that refuses the live twin's port; it never opens a connection (task 2339)",
    ("test_aw_server_rehearsal.py", "http://localhost:6486"): "same pure check",
    ("test_aw_server_rehearsal.py", "http://127.0.0.1:6499"): "same pure check",
    ("test_drain_build_refusals.py", "http://127.0.0.1"):
        "LT.normalize_endpoint() is a pure function (the port-less spelling IS the thing under test)",
    ("test_drain_build_refusals.py", "http://[::1]"): "same pure function",
    ("test_drain_build_refusals.py", "http://127.0.0.1:6334/"): "same pure function",
    ("test_content_pack_preservation.py", "http://twin:6444/"):
        "urlopen is replaced by a fake in the same test",
    ("test_discovery_ingest.py", "http://localhost:6457"):
        "argparse.Namespace field of a test double; the embed call is faked",
    ("test_news_content_pack_intent.py", "http://localhost:6398"):
        "recorded in the receipt; the staging read is the injected fixture_extractor/scroll",
    ("test_news_content_pack_intent.py", "http://127.0.0.1:6333"):
        "recorded in the receipt; the scroll is injected",
    ("test_news_daily_gate.py", "http://127.0.0.1:6457"):
        "CL627.build() records a fingerprint dict; it takes no connection",
    ("test_seal_collection_membership.py", "http://127.0.0.1:6444"):
        "recorded in the receipt; the scroll is injected",
    ("test_throwaway_qdrant_guard.py", "http://127.0.0.1"):
        "the live-port probe: the root conftest refuses the connect itself (marker live_port_probe), "
        "and the test asserts every attempt was refused",
    ("test_throwaway_qdrant_guard.py", "http://127.0.0.1:6334"):
        "setenv for a refusal test: require_throwaway() refuses on the port before any request",
    ("test_upsert_supersede_point_delete_routing.py", "http://twin:6457"):
        "the shared delete primitive is monkeypatched in the same test",
}


def literal_endpoint_hits(source: str) -> list[tuple[int, str, str]]:
    """(line, callee, literal) for every string literal that spells an endpoint -- a loopback or any
    address host, or `http(s)://host:port` -- and is an argument (at any depth: lists, dicts, tuples) of
    a call. A name bound once to such a literal in the same module counts too. Dead ports and
    unresolvable hosts are exempt. f-strings built from a throwaway's port are not literals."""
    tree = ast.parse(source)
    bound: dict[str, list[str]] = {}
    for n in ast.walk(tree):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)):
            bound.setdefault(n.targets[0].id, []).append(n.value.value)

    def endpoint(v: str) -> bool:
        m = _ENDPOINT.match(v)
        if not m:
            return False
        host, port = m.group("host"), m.group("port")
        if _UNRESOLVABLE.search(host) or (port and int(port) in _DEAD_PORTS):
            return False
        scheme = v.startswith(("http://", "https://"))
        # a bare address with no scheme and no port is a bind address (`HTTPServer(("127.0.0.1", 0))`)
        return bool(_ADDRESS.match(host)) and (scheme or bool(port)) or (scheme and bool(port))

    hits: set[tuple[int, str, str]] = set()
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        callee = ast.unparse(call.func)
        for arg in [*call.args, *(k.value for k in call.keywords)]:
            for sub in ast.walk(arg):
                vals = []
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    vals = [sub.value]
                elif isinstance(sub, ast.Name) and sub.id in bound:
                    vals = bound[sub.id]
                hits.update((call.lineno, callee, v) for v in vals if endpoint(v))
    return sorted(hits)


def _scanned_files():
    app = HERE.parent
    out = [app / "conftest.py"]
    for d in (app / "pipeline", app / "tests"):
        out += [p for p in sorted(d.rglob("*.py"))
                if p.name.startswith("test_") or "harness" in p.name or p.name in ("conftest.py", "throwaway_qdrant.py")]
    return out


def test_the_scan_catches_the_literal_that_reached_port_6333():
    """The N1 test, as the reviewer found it, and the spellings around it."""
    for src in ('B.run(url="http://127.0.0.1", collection="c")',
                'build_run("http://127.0.0.1", tmp, stages)',
                'QdrantClient("localhost:6333")',
                'Q.drain(q, twin_url="http://10.1.2.3:6334", prod_url=u)',
                'U = "http://127.0.0.1:6457"\nQ.Single(U)',
                'f(urls=["http://[::1]:6333"])'):
        assert literal_endpoint_hits(src), src
    for src in ('B.run(url=H.qdrant_url())', 'Q.Single("http://127.0.0.1:9")',
                'Q.Single("http://prod.invalid:1")', 'f(f"http://127.0.0.1:{port}")',
                'g("https://www.fedlex.admin.ch/eli/x")'):
        assert literal_endpoint_hits(src) == [], src


def test_no_test_or_harness_hands_a_literal_endpoint_to_a_client_or_to_drain_code():
    """The class behind N1 (and behind #553's F1): every URL a test passes comes from the proven
    throwaway. A literal is allowed only when it is in `ALLOWED_LITERALS`, with the reason.

    What this CANNOT see (a limit, not a pass): an f-string or a concatenation that spells a literal
    host with a non-throwaway port; a literal in a dict that is not an argument of a call; an endpoint
    held in an environment variable set by a shell script; non-Python harnesses."""
    offenders, used = [], set()
    scanned = _scanned_files()
    shipped = {p.name for p in scanned}      # the cut ships fewer test modules than the private tree
    for p in scanned:
        for line, callee, v in literal_endpoint_hits(p.read_text()):
            if (p.name, v) in ALLOWED_LITERALS:
                used.add((p.name, v))
                continue
            offenders.append(f"{p.relative_to(HERE.parent)}:{line}: {callee}(... {v!r})")
    assert offenders == [], "a literal Qdrant endpoint; use H.qdrant_url():\n" + "\n".join(offenders)
    # an entry for a module this tree does not carry cannot match anything here (it is checked where
    # the module lives); an entry for a module that IS here and matches nothing is stale
    stale = sorted(k for k in set(ALLOWED_LITERALS) - used if k[0] in shipped)
    assert stale == [], f"allowlist entries that no longer match anything (delete them): {stale}"


@pytest.mark.live_port_probe
def test_a_connect_to_a_live_qdrant_port_is_refused_before_it_is_sent(_no_connect_to_a_live_qdrant_port):
    """The belt under the literal scan: the ORIGINAL N1 test derived its port-less URL from the
    throwaway's (no literal in sight), and the client then went to :6333. The root conftest refuses
    the connect itself, by any spelling. Nothing is sent: the refusal comes before the connect."""
    import socket
    from cytir.core.qdrant import qdrant_client
    attempts = _no_connect_to_a_live_qdrant_port
    with pytest.raises(AssertionError, match="live Qdrant port"):
        socket.create_connection(("127.0.0.1", 6333), timeout=1)
    assert attempts == [("127.0.0.1", 6333)]
    # the client's own default for a port-less URL (the original N1 path)
    try:
        qdrant_client("http://127.0.0.1", timeout=2, check_compatibility=False).get_collections()
    except Exception:       # noqa: BLE001 -- the guard's refusal may be wrapped by the client
        pass
    assert all(a[1] == 6333 for a in attempts) and len(attempts) >= 2
