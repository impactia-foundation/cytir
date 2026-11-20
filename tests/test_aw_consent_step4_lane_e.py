"""Consent choke point step 4, lane E (task 1621): the news fetchers and the entity-card Wikidata /
LINDAS lookups fetch through `make_client`, so a declined host is refused before any socket.

One mechanism for all seven sites: `httpx.HTTPTransport` is replaced by a factory that returns ONE
inner transport which fails if called, so `make_client` wraps that inner in the guard. Each site is
driven through its real entry callable. How the refusal surfaces is per site and stated in its row:
`raised` (ConsentBlocked reaches the caller) or `recorded` (the site's own fail-soft contract turns
it into its existing "no result" value; the guard still stopped the call, `inner.calls == 0`).
The three `_live` builders do their consent resolution first, so they are reached by a spy on
`make_client` that hands back the client they built.
"""
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

    def handle_request(self, request):
        self.calls += 1
        raise AssertionError("the guard let a declined request reach the inner transport")


class Reached(Exception):
    def __init__(self, client):
        self.client = client


@pytest.fixture(autouse=True)
def world(tmp_path, monkeypatch):
    import json
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"hosts": {h: {"state": "declined"} for h in (
        "declined.example", "www.wikidata.org", "ld.admin.ch")}}))
    monkeypatch.setenv("AW_CONSENT_REGISTRY", str(reg))
    monkeypatch.setattr(user_agent, "user_agent", lambda component=None: "test-agent")
    monkeypatch.setattr(user_agent, "pack_fetch_user_agent", lambda: "test-pack-agent")
    inner = Inner()
    monkeypatch.setattr(httpx, "HTTPTransport", lambda *a, **k: inner)
    return inner


def _spy_factory(monkeypatch):
    from cytir.sources import http as http_mod
    real = http_mod.make_client

    def spy(*a, **k):
        raise Reached(real(*a, **k))
    monkeypatch.setattr(http_mod, "make_client", spy)


def _reach(call):
    with pytest.raises(Reached) as exc:
        call()
    return exc.value.client


def _stub_consent_resolution(monkeypatch):
    """The `_live` builders resolve consent and robots for their publisher first; stub both."""
    from cytir.acquire import sitemap_ingest as SI
    from cytir.consent import resolver as cr
    monkeypatch.setattr(cr, "resolve_hosts", lambda hosts, **k: {h: {} for h in hosts})
    monkeypatch.setattr(SI, "_search_index_bucket", lambda c: "ok")
    monkeypatch.setattr(SI, "load_robots", lambda host, cache: (None, 0.0))


# --- the seven sites ---------------------------------------------------------------------------
def _freshness_robots(monkeypatch, tmp_path):                      # recorded: robots None -> held
    from cytir.news import freshness_ingest as fi
    monkeypatch.setattr(fi, "_robots_cache", {})
    assert fi._robots("https://declined.example") is None
    assert fi.consent_ok(DECLINED) is False                       # UNRESOLVED -> held, never default-allow
    with pytest.raises(ConsentBlocked):
        fi.guarded_get(DECLINED, 10)                              # raised: the shared fetch helper
    assert fi.refetch_text(DECLINED) is None                      # recorded: keep seed


def _weekly_delta(monkeypatch, tmp_path):                         # recorded: status 0 ("unknown")
    from cytir.news import weekly_delta as wd
    r = wd.http_get(DECLINED)
    assert (r.status, r.body) == (0, b"")


def _page_date_refetch(monkeypatch, tmp_path):                    # raised
    from types import SimpleNamespace
    from cytir.news import page_date_refetch as m
    _stub_consent_resolution(monkeypatch)
    _spy_factory(monkeypatch)
    client = _reach(lambda: m._live(SimpleNamespace(domain="declined.example", cache_dir=str(tmp_path))))
    with pytest.raises(ConsentBlocked):
        client.get(DECLINED)


def _rts_section_discovery(monkeypatch, tmp_path):                # raised
    from types import SimpleNamespace
    from cytir.news import rts_section_discovery as m
    _stub_consent_resolution(monkeypatch)
    _spy_factory(monkeypatch)
    client = _reach(lambda: m._live(SimpleNamespace(cache_dir=str(tmp_path))))
    with pytest.raises(ConsentBlocked):
        client.get(DECLINED)


def _coverage_backfill(monkeypatch, tmp_path):                    # raised
    from cytir.news import coverage_backfill as m
    _stub_consent_resolution(monkeypatch)
    _spy_factory(monkeypatch)
    client = _reach(lambda: m._live_swissinfo_fetch(str(tmp_path), host="declined.example"))
    with pytest.raises(ConsentBlocked):
        client.get(DECLINED)


def _card_recipe(monkeypatch, tmp_path):                          # raised, and not retried
    from cytir.corpus import card_recipe as m
    monkeypatch.setattr("time.sleep", lambda s: pytest.fail("a declined host was retried"))
    with pytest.raises(ConsentBlocked):
        m._sparql("SELECT * WHERE {}")


SITES = [
    ("news.freshness_ingest", _freshness_robots),
    ("news.weekly_delta", _weekly_delta),
    ("news.page_date_refetch", _page_date_refetch),
    ("news.rts_section_discovery", _rts_section_discovery),
    ("news.coverage_backfill", _coverage_backfill),
    ("corpus.card_recipe", _card_recipe),
]


@pytest.mark.parametrize("module,site", SITES, ids=[s[0] for s in SITES])
def test_site_refuses_declined_host_with_no_inner_call(module, site, world, monkeypatch, tmp_path):
    site(monkeypatch, tmp_path)
    assert world.calls == 0


# --- census ------------------------------------------------------------------------------------
BANNED = re.compile(r"httpx\.Client\(|httpx\.get\(|httpx\.post\(|httpx\.AsyncClient\(|urllib\.request")

#: INTERNAL sites left in place (step 5 moves them to the internal factory). (file, line fragment) -> reason.
INTERNAL = {
    ("cytir/news/coverage_backfill.py", "import urllib.request"): "staging Qdrant DELETE, AW_STAGING_URL default localhost:6398",
    ("cytir/news/coverage_backfill.py", "urllib.request.Request(f\"{STAGING_URL}"): "staging Qdrant, localhost:6398",
}

FILES = ([APP / "cytir/news" / f"{n}.py" for n in (
    "freshness_ingest", "page_date_refetch", "rts_section_discovery", "weekly_delta", "coverage_backfill")]
    + [APP / "cytir/corpus" / f"{n}.py" for n in ("card_recipe",)])


def test_no_unguarded_http_client_construction_remains():
    offenders, used = [], set()
    for f in FILES:
        rel = str(f.relative_to(APP))
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if not BANNED.search(line) or line.lstrip().startswith("#"):
                continue
            key = next((k for k in INTERNAL if k[0] == rel and k[1] in line), None)
            if key:
                used.add(key)
            else:
                offenders.append(f"{rel}:{i}: {line.strip()[:90]}")
    assert not offenders, "\n".join(offenders)
    assert used == set(INTERNAL), f"stale allow-list rows: {sorted(set(INTERNAL) - used)}"


def test_internal_allow_list_hosts_are_local():
    src = (APP / "cytir/news/coverage_backfill.py").read_text()
    assert 'STAGING_URL = os.environ.get("AW_STAGING_URL", "http://localhost:6398")' in src
    for name in ("freshness_ingest", "weekly_delta"):
        text = (APP / "cytir/news" / f"{name}.py").read_text()
        assert '"--tei-url", default="http://localhost:8088"' in text
        assert '"--health-url", default="http://localhost:8090/health"' in text
