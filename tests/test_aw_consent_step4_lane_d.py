"""Consent choke point step 4, lane D (task 1620): the crawl core, `extract_outlinks` and the six
extractor fetch sites go through `make_client`, so a declined host is refused before any socket.

One mechanism for all seven rows: `httpx.HTTPTransport` is replaced by a factory that returns ONE
inner transport which fails if called, so `make_client` wraps it in the guard. The four entry points
that build their client inside a long command (the crawl core, `extract_outlinks`, `bfs_pxweb`,
`swisstopo`) are reached through the real entry with a spy on the module's `make_client`
that hands back the client it built and stops the run there. `cantonal_lexwork._get` and the two
retailer-card fetchers are called directly.

Outcome per row: `ConsentBlocked` is RAISED, except `wp_summary`, whose caller swallows every
per-URL error and returns "" (recorded as the empty result, inner transport untouched).
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
WIKI_HOSTS = ("www.wikidata.org", "en.wikipedia.org")
LEX_HOST = "declined.example"


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
        "declined.example", *WIKI_HOSTS)}}))
    monkeypatch.setenv("AW_CONSENT_REGISTRY", str(reg))
    monkeypatch.setattr(user_agent, "user_agent", lambda component=None: "test-agent")
    monkeypatch.setattr(user_agent, "pack_fetch_user_agent", lambda: "test-pack-agent")
    # crawl binds pack_fetch_user_agent at import (from-import); patch that binding too,
    # so the row passes regardless of which module imported it first.
    from cytir.acquire import crawl as _crawl_mod
    monkeypatch.setattr(_crawl_mod, "pack_fetch_user_agent", lambda: "test-pack-agent")
    inner = Inner()
    monkeypatch.setattr(httpx, "HTTPTransport", lambda *a, **k: inner)
    return inner


def _reach(module, monkeypatch, call):
    real = module.make_client

    def spy(*a, **k):
        raise Reached(real(*a, **k))
    monkeypatch.setattr(module, "make_client", spy)
    with pytest.raises(Reached) as exc:
        call()
    with pytest.raises(ConsentBlocked):
        exc.value.client.get(DECLINED)
    return exc.value.client


# --- the seven rows ----------------------------------------------------------------------------
def _crawl(monkeypatch, tmp_path):
    import argparse
    from cytir.acquire import crawl as m
    ap = argparse.ArgumentParser()
    m.add_standard_args(ap, out_dir=str(tmp_path / "out"))
    args = ap.parse_args(["--no-l0", "--l1-root", str(tmp_path / "l1")])
    client = _reach(m, monkeypatch, lambda: m.run_pack(
        pack="lane-d", hosts={}, args=args, canon=lambda u: u))
    assert client.headers["user-agent"] == "test-pack-agent"


def _extract_outlinks(monkeypatch, tmp_path):
    from cytir.acquire import extract_outlinks as m
    seeds = tmp_path / "seeds.json"
    seeds.write_text("[]")
    monkeypatch.setattr("sys.argv", ["x", "--seeds", str(seeds), "--out", str(tmp_path / "o.json")])
    _reach(m, monkeypatch, m.main)


def _bfs_pxweb(monkeypatch, tmp_path):
    from cytir.extract import bfs_pxweb as m
    monkeypatch.setattr("sys.argv", ["x", "--out", str(tmp_path / "o.jsonl")])
    client = _reach(m, monkeypatch, m.main)
    assert client.headers["user-agent"].endswith("bfs-pxweb-ingest")


def _swisstopo(monkeypatch, tmp_path):
    from cytir.extract import swisstopo as m
    monkeypatch.setattr("sys.argv", ["x", "--out", str(tmp_path / "o.jsonl")])
    client = _reach(m, monkeypatch, m.main)
    assert client.headers["user-agent"].endswith("swisstopo-ingest")


def _cantonal_lexwork(monkeypatch, tmp_path):
    from cytir.extract import cantonal_lexwork as m
    monkeypatch.setattr(m, "_CLIENT", None)
    monkeypatch.setattr(m, "_throttle", lambda host: None)
    monkeypatch.setattr("time.sleep", lambda s: pytest.fail("a refusal must not be retried"))
    with pytest.raises(ConsentBlocked):
        m._get(LEX_HOST, "/api/de/texts_of_law/lightweight_index")
    assert m._http().headers["accept"] == "application/json"


def _wd_api(monkeypatch, tmp_path):
    from cytir.extract import retailer_card_channels as m
    monkeypatch.setattr(m, "_CLIENT", None)
    with pytest.raises(ConsentBlocked):
        m.wd_api({"action": "wbgetentities", "ids": "Q1"})


def _wp_summary(monkeypatch, tmp_path):
    from cytir.extract import retailer_card_channels as m
    monkeypatch.setattr(m, "_CLIENT", None)
    assert m.wp_summary("Migros", "en") == ""       # swallowed per-URL error: empty result
    with pytest.raises(ConsentBlocked):
        m._http().get("https://en.wikipedia.org/api/rest_v1/page/summary/x")


SITES = [
    ("cytir.acquire.crawl", _crawl),
    ("cytir.acquire.extract_outlinks", _extract_outlinks),
    ("cytir.extract.bfs_pxweb", _bfs_pxweb),
    # pubmed: extract module has no HTTP entry since task 1753; the adapter path reads local files
    # (tests/test_aw_family_pubmed.py:65) and the make_client refusal is tests/test_aw_consent_runner_guard.py:60
    ("cytir.extract.swisstopo", _swisstopo),
    ("cytir.extract.cantonal_lexwork", _cantonal_lexwork),
    ("cytir.extract.retailer_card_channels.wd_api", _wd_api),
    ("cytir.extract.retailer_card_channels.wp_summary", _wp_summary),
]


@pytest.mark.parametrize("module,site", SITES, ids=[s[0].rsplit(".", 1)[1] for s in SITES])
def test_site_refuses_declined_host_with_no_inner_call(module, site, world, monkeypatch, tmp_path):
    site(monkeypatch, tmp_path)
    assert world.calls == 0


# --- census ------------------------------------------------------------------------------------
BANNED = re.compile(r"httpx\.Client\(|httpx\.get\(|httpx\.AsyncClient\(|urllib\.request")


def test_no_unguarded_http_client_construction_remains():
    files = [APP / "cytir/acquire" / f"{n}.py" for n in ("crawl", "extract_outlinks")]
    files += [APP / "cytir/extract" / f"{n}.py" for n in (
        "bfs_pxweb", "cantonal_lexwork", "pubmed", "retailer_card_channels", "swisstopo")]
    offenders = []
    for f in files:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if BANNED.search(line) and not line.lstrip().startswith("#"):
                offenders.append(f"{f.relative_to(APP)}:{i}: {line.strip()[:90]}")
    assert not offenders, "\n".join(offenders)
