"""Consent choke point step 4, lane C (task 1615): the acquire core, the exact-URL doors, fedlex and
de_laender_juris fetch through `make_client`, so a declined host is refused before any socket.

One mechanism for all eight sites: `httpx.HTTPTransport` is replaced by a factory that returns ONE
inner transport which fails if called. `make_client` (and the two ingesters that pass their own
`HTTPTransport(limits=...)`) therefore wrap that inner in the guard, and every site is built
through its real entry callable. The four sites buried deep in a long command (discovery_ingest,
sitemap_ingest, cc_preserved, direct_preserved) are reached through the real entry with a spy on
the module's `make_client` that hands back the client it built and stops the run there.
"""
import json
import re
from pathlib import Path

import httpx
import pytest

from cytir.consent.block import ConsentBlocked
from cytir.core import user_agent
from cytir.sources import http as http_mod

APP = Path(__file__).resolve().parent.parent
DECLINED = "https://declined.example/x"


class Inner(httpx.BaseTransport):
    def __init__(self):
        self.calls = 0

    def handle_request(self, request):
        self.calls += 1
        raise AssertionError("the guard let a declined request reach the inner transport")


class Reached(Exception):
    def __init__(self, client):
        self.client = client


@pytest.fixture(autouse=True)
def world(tmp_path, monkeypatch):
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"hosts": {h: {"state": "declined"} for h in (
        "declined.example", "data.commoncrawl.org", "fedlex.data.admin.ch")}}))
    monkeypatch.setenv("AW_CONSENT_REGISTRY", str(reg))
    monkeypatch.setattr(user_agent, "user_agent", lambda component=None: "test-agent")
    monkeypatch.setattr(user_agent, "pack_fetch_user_agent", lambda: "test-pack-agent")
    inner = Inner()
    monkeypatch.setattr(httpx, "HTTPTransport", lambda *a, **k: inner)
    return inner


def _spy(monkeypatch, module):
    real = module.make_client

    def spy(*a, **k):
        raise Reached(real(*a, **k))
    monkeypatch.setattr(module, "make_client", spy)


def _reach(call):
    with pytest.raises(Reached) as exc:
        call()
    return exc.value.client


# --- the eight sites ---------------------------------------------------------------------------
def _cc_columnar(monkeypatch, tmp_path):
    from cytir.acquire import cc_columnar as m
    monkeypatch.setattr(m, "_CLIENT", None)
    monkeypatch.setattr(m, "use_s3", lambda: False)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(ConsentBlocked):
        m._read_manifest_lines("CC-MAIN-2025-00")


def _fetch_process(monkeypatch, tmp_path):
    from cytir.acquire import fetch_process as m
    monkeypatch.setattr(m, "_CLIENT", None)
    with pytest.raises(ConsentBlocked):
        m._http().get(DECLINED)
    assert m.fetch_url(DECLINED) is None          # fetch_url fails closed (it logs, returns None)


def _discovery_ingest(monkeypatch, tmp_path):
    from cytir.acquire import discovery_ingest as m
    queue = tmp_path / "q.json"
    queue.write_text(json.dumps([{"domain": "declined.example"}]))
    monkeypatch.setattr("sys.argv", [
        "x", "fetch", "--queue", str(queue), "--out", str(tmp_path / "out" / "p.jsonl"),
        "--state", str(tmp_path / "state.json"), "--skip-smoke-check",
        "--min-free-gb", "0"])  # task 1776: tmp out dir must not depend on host free disk
    monkeypatch.setenv("AW_EGRESS_STATE", str(tmp_path / "egress.json"))
    _spy(monkeypatch, m)
    client = _reach(m.main)
    with pytest.raises(ConsentBlocked):
        client.get(DECLINED)


def _sitemap_ingest(monkeypatch, tmp_path):
    from cytir.acquire import sitemap_ingest as m
    queue = tmp_path / "q.json"
    queue.write_text(json.dumps([{"domain": "declined.example", "wave": "1"}]))
    monkeypatch.setattr("sys.argv", [
        "x", "fetch", "--queue", str(queue), "--wave", "1", "--out", str(tmp_path / "out" / "p.jsonl"),
        "--raw-dir", str(tmp_path / "raw"), "--state", str(tmp_path / "state.json")])
    _spy(monkeypatch, m)
    client = _reach(m.main)
    with pytest.raises(ConsentBlocked):
        client.get(DECLINED)


def _cc_preserved(monkeypatch, tmp_path):
    from cytir.sources.exact_url import cc_preserved as m
    root = tmp_path / "root"
    (root / "plan").mkdir(parents=True)
    (root / "plan" / "RECEIPT.json").write_text(json.dumps({
        "schema": "aw.exact-url-cc-plan-receipt.v1", "status": "complete",
        "plan": {"sha256": "s"}, "frontier": {"urls": 0}}))
    monkeypatch.setattr(m, "_load_plan_partition", lambda *a, **k: ([], 0, "s"))
    _spy(monkeypatch, m)
    client = _reach(lambda: m.acquire(root=root, artifact_root=tmp_path))
    with pytest.raises(ConsentBlocked):
        client.get(DECLINED)


def _direct_preserved(monkeypatch, tmp_path):
    from cytir.sources.exact_url import direct_preserved as m
    monkeypatch.setattr(m, "_frontier", lambda cc_root, artifact_root: [])
    monkeypatch.setattr(m, "user_agent", lambda: "test-agent")
    root = tmp_path / "art" / "root"
    _spy(monkeypatch, m)
    client = _reach(lambda: m.acquire(cc_root=tmp_path, root=root, artifact_root=tmp_path / "art"))
    with pytest.raises(ConsentBlocked):
        client.get(DECLINED)


def _de_laender_juris(monkeypatch, tmp_path):
    from cytir.sources.families import de_laender_juris as m
    # Portal asserts the host is crawlable at construction; use a cleared host for the portal
    # object, then prove its own client is guarded against the declined URL.
    monkeypatch.setattr(m, "assert_crawlable", lambda host, *a, **k: None)
    portal = m.Portal("open.example", "cid")
    assert portal.c.cookies.get("r3autologin") is not None
    with pytest.raises(ConsentBlocked):
        portal.c.get(DECLINED)


def _fedlex(monkeypatch, tmp_path):
    from cytir.sources.families import fedlex as m
    monkeypatch.setattr(m, "H", None)
    with pytest.raises(ConsentBlocked):
        m.resolve_date("cc/1999/404", "2026-01-01")      # default client, fedlex SPARQL host
    with pytest.raises(ConsentBlocked):
        m._h().get(DECLINED)


SITES = [
    ("cytir.acquire.cc_columnar", _cc_columnar),
    ("cytir.acquire.discovery_ingest", _discovery_ingest),
    ("cytir.acquire.fetch_process", _fetch_process),
    ("cytir.acquire.sitemap_ingest", _sitemap_ingest),
    ("cytir.sources.exact_url.cc_preserved", _cc_preserved),
    ("cytir.sources.exact_url.direct_preserved", _direct_preserved),
    ("cytir.sources.families.de_laender_juris", _de_laender_juris),
    ("cytir.sources.families.fedlex", _fedlex),
]


@pytest.mark.parametrize("module,site", SITES, ids=[s[0].rsplit(".", 1)[1] for s in SITES])
def test_site_refuses_declined_host_with_no_inner_call(module, site, world, monkeypatch, tmp_path):
    site(monkeypatch, tmp_path)
    assert world.calls == 0


# --- census ------------------------------------------------------------------------------------
BANNED = re.compile(r"httpx\.Client\(|httpx\.get\(|httpx\.AsyncClient\(|urllib\.request")


def test_no_unguarded_http_client_construction_remains():
    files = [APP / "cytir/acquire" / f"{n}.py" for n in
             ("cc_columnar", "discovery_ingest", "fetch_process", "sitemap_ingest")]
    files += sorted((APP / "cytir/sources/exact_url").glob("*.py"))
    files += [APP / "cytir/sources/families" / f"{n}.py" for n in ("de_laender_juris", "fedlex")]
    offenders = []
    for f in files:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if BANNED.search(line) and not line.lstrip().startswith("#"):
                offenders.append(f"{f.relative_to(APP)}:{i}: {line.strip()[:90]}")
    assert not offenders, "\n".join(offenders)


def test_factory_is_the_only_route(world):
    # a construction through the factory is guarded and its inner is never reached
    client = http_mod.make_client()
    with pytest.raises(ConsentBlocked):
        client.get(DECLINED)
    assert world.calls == 0
