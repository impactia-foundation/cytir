"""Consent choke point, step 4, lane B (task 1614): five httpx sites through `make_client`, and
the ctgov_psy curl client behind `assert_crawlable`. Fake transports only; no socket, no exec.

One mechanism for every httpx site: `httpx.HTTPTransport` is replaced so the factory's inner
transport is a transport that FAILS if called. A site that still built its own client would
bypass it, so the census below backs this up.
"""
import argparse
import json
import re
from pathlib import Path

import httpx
import pytest

from cytir.consent import guard
from cytir.consent.block import ConsentBlocked
from cytir.core import user_agent

DECLINED = "https://declined.example/x"


class FailingInner(httpx.BaseTransport):
    def __init__(self):
        self.calls = 0
        self.refused = []

    def handle_request(self, request):
        self.calls += 1
        raise AssertionError(f"inner transport reached for {request.url}")


@pytest.fixture(autouse=True)
def registry(tmp_path, monkeypatch):
    path = tmp_path / "reg.json"
    path.write_text(json.dumps({"hosts": {"declined.example": {"state": "declined"}}}))
    monkeypatch.setenv("AW_CONSENT_REGISTRY", str(path))
    monkeypatch.setattr(user_agent, "user_agent", lambda component=None: "test-agent")
    monkeypatch.setattr(user_agent, "pack_fetch_user_agent", lambda: "test-pack-agent")
    monkeypatch.setattr(user_agent, "operator", lambda: ("https://op.example", "a@op.example", True))
    monkeypatch.setattr("time.sleep", lambda s: None)


@pytest.fixture
def inner(monkeypatch):
    mine = FailingInner()
    monkeypatch.setattr(httpx, "HTTPTransport", lambda *a, **k: mine)
    real = guard.ConsentGuardTransport.handle_request

    def spy(self, request):
        try:
            return real(self, request)
        except ConsentBlocked:
            mine.refused.append(str(request.url))
            raise
    monkeypatch.setattr(guard.ConsentGuardTransport, "handle_request", spy)
    return mine


def _entscheidsuche(tmp_path, mp):
    from cytir.sources.entscheidsuche import discover
    mp.setattr(discover, "SITEMAP_INDEX", DECLINED)
    return discover.cmd_toc, dict(args=argparse.Namespace(spiders=""))


def _gesetze(tmp_path, mp):
    from cytir.sources.gesetze import ingest
    mp.setattr(ingest, "_TLOCAL", None)
    return (lambda url: ingest.fetch_law_zip(ingest._client(), url)), dict(url=DECLINED)


def _rights_canon(tmp_path, mp):
    from cytir.sources.rights_canon import crawl
    mp.setattr(crawl, "HOSTS", {"k": dict(host="declined.example", mode="sitemap", delay=0)})
    args = argparse.Namespace(out_dir=str(tmp_path / "rc"), max_chunks=1, limit=1, only="k",
                              budget=1, skip_api=True)
    mp.setattr("argparse.ArgumentParser.parse_args", lambda self, *a, **k: args)
    return crawl.main, {}


def _bfs_dam(tmp_path, mp):
    from cytir.sources.bfs_dam import enumerate as en
    mp.setattr(en, "API", DECLINED)
    args = argparse.Namespace(raw=str(tmp_path / "raw.jsonl"), state=str(tmp_path / "s.json"),
                              sleep=0.0, start_skip=0, end_skip=0, limit=0)
    return en.cmd_enumerate, dict(args=args)


def _technical_manufacturer(tmp_path, mp):
    from cytir.sources.technical_manufacturer import catalog
    fetcher = catalog.Fetcher(root=tmp_path / "a" / "r", artifact_root=tmp_path / "a", rate=1000.0)
    return (lambda url: fetcher.client.get(url)), dict(url=DECLINED)


# (site, builder -> (entry callable, kwargs), True when the module's own handler swallows errors)
SITES = [
    ("entscheidsuche.discover.cmd_toc", _entscheidsuche, False),
    ("gesetze.ingest.fetch_law_zip", _gesetze, True),
    ("rights_canon.crawl.main", _rights_canon, True),
    ("bfs_dam.enumerate.cmd_enumerate", _bfs_dam, False),
    ("technical_manufacturer.catalog.Fetcher.client", _technical_manufacturer, False),
]


@pytest.mark.parametrize("name,build,swallowed", SITES, ids=[s[0] for s in SITES])
def test_site_client_is_guarded_and_inner_transport_never_reached(
        name, build, swallowed, inner, tmp_path, monkeypatch):                      # spec test 2
    entry, kwargs = build(tmp_path, monkeypatch)
    if swallowed:       # the module's own retry or skip handler catches Exception; the guard still fired
        entry(**kwargs)
    else:
        with pytest.raises(ConsentBlocked):
            entry(**kwargs)
    assert inner.refused, "the guard never refused: the site is not behind make_client"
    assert inner.calls == 0


def test_curl_client_refuses_declined_host_before_exec(monkeypatch):                # spec test 8
    from cytir.sources.ctgov_psy import harvest
    execs = []
    monkeypatch.setattr(harvest.subprocess, "run", lambda *a, **k: execs.append(a))
    with pytest.raises(ConsentBlocked):
        harvest.CurlClient().get(DECLINED, params={"q": "1"})
    with pytest.raises(ConsentBlocked):
        harvest.CurlClient().post_json(DECLINED, {"a": 1})
    assert execs == []


LANE_DIRS = ("ctgov_psy", "gesetze", "entscheidsuche", "rights_canon", "bfs_dam",
             "technical_manufacturer")
BANNED = re.compile(r"httpx\.Client\(|httpx\.get\(|httpx\.AsyncClient\(|urllib\.request")


def test_no_lane_b_module_builds_its_own_http_client():
    root = Path(__file__).resolve().parents[1] / "cytir" / "sources"
    offenders = []
    for d in LANE_DIRS:
        for f in sorted((root / d).rglob("*.py")):
            for n, line in enumerate(f.read_text().splitlines(), 1):
                if BANNED.search(line) and not line.lstrip().startswith("#"):
                    offenders.append(f"{f.relative_to(root)}:{n}: {line.strip()}")
    assert offenders == [], "\n".join(offenders)
