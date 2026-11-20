"""Consent choke point step 4, lane F (task 1622): the model download and the two ops rosters fetch
through `make_client`, so a declined host is refused before any socket.

One mechanism: `httpx.HTTPTransport` is replaced by a factory returning ONE inner transport that
fails if called; every site is run through its real entry callable against a declined host.
`bench/scout_adapter.py` (deleted, task 2546) lost its `httpx.AsyncClient` path; the census proves it and that the
other files hold no direct client.
"""
import json
import re
from pathlib import Path

import httpx
import pytest

from cytir.consent.block import ConsentBlocked
from cytir.core import user_agent

APP = Path(__file__).resolve().parent.parent
DECLINED = "https://declined.example/x"


class Inner(httpx.BaseTransport):
    def __init__(self):
        self.calls = 0
        self.routes = {}

    def handle_request(self, request):
        self.calls += 1
        if str(request.url) in self.routes:
            return self.routes[str(request.url)]
        raise AssertionError("the guard let a declined request reach the inner transport")


@pytest.fixture
def world(tmp_path, monkeypatch):
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"hosts": {"declined.example": {"state": "declined"}}}))
    monkeypatch.setenv("AW_CONSENT_REGISTRY", str(reg))
    monkeypatch.setattr(user_agent, "user_agent", lambda component=None: "test-agent")
    monkeypatch.setattr(user_agent, "pack_fetch_user_agent", lambda: "test-pack-agent")
    monkeypatch.setattr("time.sleep", lambda s: None)
    inner = Inner()
    monkeypatch.setattr(httpx, "HTTPTransport", lambda *a, **k: inner)
    return inner


def _fetch_models(tmp_path):
    from cytir.embed import fetch_models as m

    class Asset:
        name, url, size_bytes, sha256 = "x", DECLINED, 1, "0"

    with pytest.raises(ConsentBlocked):
        m._download(Asset(), tmp_path / "m" / "x.bin")
    assert not (tmp_path / "m" / "x.bin").exists()


def _docs_roster(tmp_path):
    from cytir.ops import docs_roster as m
    with pytest.raises(ConsentBlocked):
        m._get(DECLINED, "ua", tries=3)


def _registry_census(tmp_path):
    from cytir.ops import registry_census as m
    with pytest.raises(ConsentBlocked):
        m.http_get(DECLINED)


SITES = [("embed.fetch_models", _fetch_models), ("ops.docs_roster", _docs_roster),
         ("ops.registry_census", _registry_census)]


@pytest.mark.parametrize("name,call", SITES, ids=[n for n, _ in SITES])
def test_declined_host_is_refused_before_any_socket(name, call, world, tmp_path):
    call(tmp_path)
    assert world.calls == 0


CENSUS_FILES = ["cytir/embed/fetch_models.py", "cytir/ops/docs_roster.py",
                "cytir/ops/registry_census.py"]
# own-infrastructure call, not an origin host: the twin Qdrant facet (docs_roster.twin_facets)
# stays on urllib, outside this lane
OWN_INFRA = {"cytir/ops/docs_roster.py": ("urllib.request.Request(", "urllib.request.urlopen(req, timeout=900)")}
BANNED = re.compile(r"httpx\.Client\(|httpx\.get\(|httpx\.AsyncClient\(|urllib\.request")


@pytest.mark.parametrize("rel", CENSUS_FILES)
def test_no_direct_client_left(rel):
    found = []
    for n, line in enumerate((APP / rel).read_text().splitlines(), 1):
        code = line.split("#")[0]
        if not BANNED.search(code) or code.strip().startswith("import urllib.request"):
            continue
        if any(a in code for a in OWN_INFRA.get(rel, ())):
            continue
        found.append((n, line.strip()))
    assert not found, found
