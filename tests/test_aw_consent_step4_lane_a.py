"""Consent request choke point, step 4 lane A (task 1613): eight fetch sites go through make_client.

Mechanism for all eight: `httpx.HTTPTransport` is replaced by a counting inner transport, so the
guard that `make_client` wraps around it is the only thing between the call and the "socket".
Each site is entered through its real entry point against a declined host; the inner transport
must never be reached. No socket is opened.
"""
import argparse
import json
import re
from pathlib import Path

import httpx
import pytest

from cytir.consent.block import ConsentBlocked
from cytir.core import user_agent

DECLINED = "https://declined.example/x"
APP = Path(__file__).resolve().parents[1]


class Inner(httpx.BaseTransport):
    def __init__(self):
        self.calls = 0

    def handle_request(self, request):
        self.calls += 1
        raise AssertionError("inner transport reached for " + str(request.url))


@pytest.fixture(autouse=True)
def world(tmp_path, monkeypatch):
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"hosts": {"declined.example": {"state": "declined"}}}))
    monkeypatch.setenv("AW_CONSENT_REGISTRY", str(reg))
    monkeypatch.setattr(user_agent, "user_agent", lambda component=None: "test-agent")
    monkeypatch.setattr(user_agent, "pack_fetch_user_agent", lambda: "test-pack-agent")
    inner = Inner()
    monkeypatch.setattr(httpx, "HTTPTransport", lambda *a, **k: inner)
    return inner


# --- the eight sites, each through its real entry point ------------------------------------

def _docs_sitemap(mp):
    from cytir.sources.refresh_direct import docs_sitemap
    docs_sitemap.default_fetch_bytes(DECLINED)


def _discover_errors(module, params, mp, patch=None):
    if patch:
        mp.setattr(module, patch, lambda *_a, **_k: DECLINED)
    out = module.discover({"acquire": {"params": params}})
    # these adapters name a per-item failure in `errors` instead of raising (unchanged)
    errs = " ".join(map(str, out["errors"].values()))
    assert "ConsentBlocked" in errs
    raise ConsentBlocked(errs)


def _github_releases(mp):
    from cytir.sources.refresh_direct import github_releases as m
    _discover_errors(m, {"repos": ["o/r"]}, mp, "feed_url")


def _hf_org_models(mp):
    from cytir.sources.refresh_direct import hf_org_models as m
    _discover_errors(m, {"orgs": ["o"]}, mp, "api_url")


def _vendor_blog_rss(mp):
    from cytir.sources.refresh_direct import vendor_blog_rss as m
    _discover_errors(m, {"blogs": [{"label": "b", "url": DECLINED}]}, mp)


def _finma(mp, tmp_path):
    from cytir.sources.finma import circulars as m
    # fetch_robots swallows errors by design; probe through the client stage_extract built
    mp.setattr(m, "fetch_robots", lambda client, host: client.get(DECLINED))
    m.stage_extract(argparse.Namespace(out_dir=str(tmp_path / "out")))


def _github_repo_docs(mp):
    from cytir.sources.github_repo_docs import acquire as m
    settings = m.Settings(root=Path("."), max_files=1, retries=0, timeout=1.0)
    m.GitHubClient("", settings).get(DECLINED)


def _hf_model_cards(mp):
    from cytir.sources.hf_model_cards import ingest as m
    m._get(DECLINED, tries=1)


def _hf_model_cards_probe(mp):
    # the net-down parking probe is the second urllib site
    from cytir.sources.hf_model_cards import ingest as m
    mp.setattr(m, "API", DECLINED)
    mp.setattr(m.time, "sleep", lambda s: None)

    def boom(*_a, **_k):
        raise OSError("Network is unreachable")
    mp.setattr(m, "_client", _flaky_client(m, boom))
    m._get("https://open.example/x", tries=1)


def _flaky_client(m, boom):
    real = m._client
    state = {"n": 0}

    def factory(timeout=60):
        state["n"] += 1
        if state["n"] == 1:           # first request: simulate the outage
            class Down:
                def __enter__(self): return self
                def __exit__(self, *a): return False
                def get(self, *_a, **_k): boom()
            return Down()
        return real(timeout=timeout)   # probe: real guarded client
    return factory


SITES = [
    ("refresh_direct.docs_sitemap", lambda mp, tp: _docs_sitemap(mp)),
    ("refresh_direct.github_releases", lambda mp, tp: _github_releases(mp)),
    ("refresh_direct.hf_org_models", lambda mp, tp: _hf_org_models(mp)),
    ("refresh_direct.vendor_blog_rss", lambda mp, tp: _vendor_blog_rss(mp)),
    ("finma.circulars", _finma),
    ("github_repo_docs.acquire", lambda mp, tp: _github_repo_docs(mp)),
    ("hf_model_cards.ingest:_get", lambda mp, tp: _hf_model_cards(mp)),
    ("hf_model_cards.ingest:probe", lambda mp, tp: _hf_model_cards_probe(mp)),
]


@pytest.mark.parametrize("name,run", SITES, ids=[s[0] for s in SITES])
def test_site_refuses_declined_host(name, run, world, monkeypatch, tmp_path):
    with pytest.raises(ConsentBlocked):
        run(monkeypatch, tmp_path)
    assert world.calls == 0


# --- census: no direct client left in the migrated trees -----------------------------------

BANNED = re.compile(r"httpx\.Client\(|httpx\.get\(|httpx\.AsyncClient\(|urllib\.request")
TREES = ("refresh_direct", "finma", "github_repo_docs", "hf_model_cards")


def test_no_unguarded_client_in_lane_a_trees():
    offenders = []
    for tree in TREES:
        root = APP / "cytir" / "sources" / tree
        assert root.is_dir(), tree
        for path in sorted(root.rglob("*.py")):
            if BANNED.search(path.read_text()):
                offenders.append(str(path.relative_to(APP)))
    assert not offenders, offenders
