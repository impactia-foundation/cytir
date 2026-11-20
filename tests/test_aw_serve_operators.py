"""task 2640: search operators `site:` `-site:` `source:` `lang:` `country:` + named params.
No Qdrant, no network: the facet call and the legs' Qdrant client are stubbed."""
import ast
import inspect
import pathlib
from types import SimpleNamespace as NS

import pytest
from qdrant_client import models as qm

from cytir.serve import legs, operators
from cytir.serve.operators import Operators, OperatorError
from cytir.serve.query import parse_operators

HOSTS = frozenset({"admin.ch", "fedlex.admin.ch", "bfs.admin.ch", "wikipedia.org", "en.wikipedia.org",
                   "notadmin.ch", "example.com"})
SOURCES = frozenset({"official", "reference", "docs", "entity_card"})


@pytest.fixture(autouse=True)
def _current_serve_modules():
    """Other tests drop every `cytir.serve*` entry from sys.modules and re-import; names bound at this
    file's collection then point at dead copies (a different `Operators` class, a different `_T`).
    Rebind them to the live modules at the start of each test."""
    import importlib
    g = globals()
    g["operators"] = importlib.import_module("cytir.serve.operators")
    g["legs"] = importlib.import_module("cytir.serve.legs")
    g["Operators"] = g["operators"].Operators
    g["OperatorError"] = g["operators"].OperatorError
    g["parse_operators"] = importlib.import_module("cytir.serve.query").parse_operators


@pytest.fixture
def table(monkeypatch):
    t = operators._Table()
    t.hosts, t.sources, t.loaded_at = HOSTS, SOURCES, 10**12   # fresh for the whole test
    monkeypatch.setattr(operators, "_T", t)
    monkeypatch.setattr(operators.time, "monotonic", lambda: 10**12 + 1)
    return t


# ---- parser ---------------------------------------------------------------------------------
def test_each_operator():
    q, o = parse_operators("art 336 site:fedlex.admin.ch")
    assert (q, o.site) == ("art 336", ("fedlex.admin.ch",))
    assert parse_operators("x -site:wikipedia.org")[1].exclude_site == ("wikipedia.org",)
    assert parse_operators("x source:Official")[1].source == ("official",)
    assert parse_operators("x lang:DE")[1].lang == "de"
    assert parse_operators("x country:ch")[1].country == "CH"


def test_case_insensitive_name_and_www_strip():
    q, o = parse_operators("hello SITE:WWW.Admin.CH -Site:https://www.x.org/path")
    assert q == "hello" and o.site == ("admin.ch",) and o.exclude_site == ("x.org",)


def test_mixed_and_repeated():
    q, o = parse_operators("a site:a.ch b site:b.ch -site:c.ch -site:d.ch source:docs source:official lang:fr")
    assert q == "a b"
    assert o.site == ("a.ch", "b.ch") and o.exclude_site == ("c.ch", "d.ch")
    assert o.source == ("docs", "official") and o.lang == "fr"


@pytest.mark.parametrize("q", ["re: hello", "see https://x.org/a:b", "site: alone", "foo:bar", "site:", "-site:",
                               "plain   spaced  query"])
def test_non_operators_keep_their_text(q):
    assert parse_operators(q) == (q, Operators())


def test_unknown_prefix_kept_next_to_operator():
    assert parse_operators("re: x site:a.ch")[0] == "re: x"


# ---- host table + filter construction ---------------------------------------------------------
def test_site_matches_host_and_sub_hosts_only(table):
    f, echo = operators.build(Operators(site=("admin.ch",)))
    cond = f.must[0]
    assert cond.key == "domain"
    assert sorted(cond.match.any) == ["admin.ch", "bfs.admin.ch", "fedlex.admin.ch"]  # not notadmin.ch
    assert echo == {"site": ["admin.ch"]} and f.must_not is None


def test_single_host_is_match_value_and_unknown_host_stays_exact(table):
    f, _ = operators.build(Operators(site=("example.com",)))
    assert f.must[0].match.value == "example.com"
    f, _ = operators.build(Operators(site=("nope.test",)))
    assert f.must[0].match.value == "nope.test"


def test_several_sites_or_and_exclusions_must_not(table):
    f, echo = operators.build(Operators(site=("admin.ch", "example.com"),
                                        exclude_site=("wikipedia.org",), source=("official",)))
    assert sorted(f.must[1].match.any) == ["admin.ch", "bfs.admin.ch", "example.com", "fedlex.admin.ch"]
    assert f.must[0].key == "source_type" and f.must[0].match.value == "official"
    assert sorted(f.must_not[0].match.any) == ["en.wikipedia.org", "wikipedia.org"]
    assert echo["exclude_site"] == ["wikipedia.org"]


def test_unknown_source_is_kept_as_text_never_an_error(table):
    f, echo = operators.build(Operators(source=("main", "official")))
    assert echo["unknown_source"] == ["main"] and echo["source"] == ["official"]
    assert "source:main is not a known source_type; kept as text" in echo["note"]
    assert f.must[0].match.value == "official"
    f, echo = operators.build(Operators(source=("main",)))
    assert f is None and echo["unknown_source"] == ["main"]


def test_too_many_hosts_is_refused(table, monkeypatch):
    monkeypatch.setattr(operators, "MAX_HOSTS", 2)
    with pytest.raises(OperatorError, match="narrower"):
        operators.build(Operators(site=("admin.ch",)))


def _boom(*a):
    raise RuntimeError("down")


def test_facet_failure_falls_back_to_exact_host_with_a_note(monkeypatch):
    monkeypatch.setattr(operators, "_T", operators._Table())
    monkeypatch.setattr(operators, "_facet", _boom)
    f, echo = operators.build(Operators(site=("admin.ch",), source=("anything",)))
    assert f.must[1].match.value == "admin.ch"
    assert "exact host only" in echo["note"] and "not validated" in echo["note"]


def test_table_loads_once_then_refreshes_hourly_and_backs_off(monkeypatch):
    calls = []

    def fake(key, limit):
        calls.append(key)
        return HOSTS if key == "domain" else SOURCES
    monkeypatch.setattr(operators, "_T", operators._Table())
    monkeypatch.setattr(operators, "_facet", fake)
    assert operators._ensure_table(100.0) and calls == ["domain", "source_type"]
    operators._ensure_table(200.0)
    assert len(calls) == 2
    operators._ensure_table(100.0 + operators.REFRESH_S + 1)
    assert len(calls) == 4
    # a failed refresh keeps the old table and does not retry inside RETRY_S
    monkeypatch.setattr(operators, "_facet", _boom)
    t0 = 100.0 + 2 * operators.REFRESH_S + 5
    assert operators._ensure_table(t0) is True
    monkeypatch.setattr(operators, "_facet", fake)
    operators._ensure_table(t0 + 1)
    assert len(calls) == 4


def test_two_sequential_calls_during_an_outage_make_one_facet_attempt(monkeypatch):
    calls = []

    def failing(*a):
        calls.append(a)
        raise RuntimeError("down")
    monkeypatch.setattr(operators, "_T", operators._Table())
    monkeypatch.setattr(operators, "_facet", failing)
    assert operators._ensure_table(100.0) is False
    assert operators._ensure_table(101.0) is False
    assert len(calls) == 1          # the second call is inside RETRY_S, checked under the lock too


def test_back_off_is_rechecked_under_the_lock(monkeypatch):
    """A waiter that got past the unlocked check while the first caller failed must not re-run."""
    t = operators._Table()
    monkeypatch.setattr(operators, "_T", t)
    calls = []
    monkeypatch.setattr(operators, "_facet", lambda *a: calls.append(a) or HOSTS)
    t.lock.acquire()
    import threading
    th = threading.Thread(target=operators._ensure_table, args=(100.0,))
    th.start()
    t.failed_at = 99.5              # the "first caller" failed while the waiter queued
    t.lock.release()
    th.join(5)
    assert calls == []


def test_and_filter_nests_so_a_should_stays_or():
    base = qm.Filter(should=[qm.FieldCondition(key="lang", match=qm.MatchValue(value="de"))])
    extra = qm.Filter(must=[qm.FieldCondition(key="domain", match=qm.MatchValue(value="a.ch"))])
    assert operators.and_filter(base, None) is base and operators.and_filter(None, extra) is extra
    assert operators.and_filter(None, None) is None
    assert operators.and_filter(base, extra).must == [base, extra]


# ---- the filter reaches every leg ---------------------------------------------------------------
def _run(monkeypatch, op_filter, lang="de"):
    seen = []

    class Client:
        def query_batch_points(self, collection_name, requests):
            seen.extend(requests)
            return [NS(points=[]) for _ in requests]

    monkeypatch.setattr(legs.qdrant_io, "client", lambda: Client())
    monkeypatch.setattr(legs, "_BATCH_LEGS", True)
    monkeypatch.setattr(legs, "_PER_TOKEN", True)
    monkeypatch.setattr(legs, "_TWO_STAGE", True)
    monkeypatch.setattr(legs, "_AUTHLEG", True)
    plan = NS(country_legs_enabled=True, country="CH", country_dense_limit=5, country_bm25_limit=5)
    legs.run_legs(q="alpha beta gamma", lang=lang, dense=[0.1, 0.2], lexical=None, candidates=10,
                  country=None, query_plan=plan, scoped=False, q_country_soft=None, experimental=False,
                  target_bm25_language=None, target_bm25_quota=0, target_bm25_weight=0.0,
                  translated_bm25_query=None, translated_bm25_quota=0, translated_bm25_weight=0.0,
                  op_filter=op_filter)
    return seen


def _conds(f):
    """Every FieldCondition reachable in a filter, nested included."""
    out = []
    for part in (f.must or []) + (f.should or []) + (f.must_not or []):
        out += _conds(part) if isinstance(part, qm.Filter) else [part]
    return out


def _holds(f, target):
    """`target` is a (possibly nested) must-member of `f`."""
    return any(m == target or (isinstance(m, qm.Filter) and _holds(m, target)) for m in (f.must or []))


def test_operator_filter_reaches_every_leg_and_sub_leg(monkeypatch):
    opf = qm.Filter(must=[qm.FieldCondition(key="domain", match=qm.MatchValue(value="a.ch"))],
                    must_not=[qm.FieldCondition(key="domain", match=qm.MatchValue(value="b.ch"))])
    reqs = _run(monkeypatch, opf)
    assert len(reqs) >= 5      # dense, bm25, authority, country x2, + per-token sub-legs
    for r in reqs:
        assert r.filter is not None and _holds(r.filter, opf), r.using


def test_no_operator_leaves_the_legs_filters_without_a_domain_condition(monkeypatch):
    reqs = _run(monkeypatch, None)
    assert len(reqs) >= 5
    for r in reqs:
        assert r.filter is None or not any(c.key == "domain" for c in _conds(r.filter))


# ---- API ------------------------------------------------------------------------------------------
def _app():
    # Imported lazily: test_serve_privacy_route re-imports `cytir.serve.app` against its own settings
    # at ITS collection time and a module-level import here would change what it finds.
    import importlib
    return importlib.import_module("cytir.serve.app")


def _call(monkeypatch, q, *, lang=None, country=None, site=None, exclude_site=None, source=None):
    app = _app()
    got = {}

    def fake_search(q_, k=10, lang=None, country=None, **kw):
        got.update(q=q_, lang=lang, country=country, **kw)
        return [{"url": "https://x"}]
    monkeypatch.setattr(app.retrieval, "search", fake_search)
    resp = app.search(q=q, k=10, lang=lang, country=country, jurisdiction=None, authorities=0,
                      site=site, exclude_site=exclude_site, source=source)
    return resp, got


def test_api_without_operator_is_byte_identical(monkeypatch):
    resp, got = _call(monkeypatch, "  Art. 336   CO ", lang="de")
    assert resp == {"query": "  Art. 336   CO ", "lang": "de", "count": 1, "results": [{"url": "https://x"}]}
    assert got == {"q": "  Art. 336   CO ", "lang": "de", "country": None}   # no op_filter kwarg at all


def test_api_operators_and_params_merge(monkeypatch, table):
    resp, got = _call(monkeypatch, "kuendigung site:admin.ch lang:fr", lang="de",
                      site=["example.com"], exclude_site=["wikipedia.org"], source=["official"])
    assert resp["query"] == "kuendigung" and got["lang"] == "fr"
    f = resp["filters"]
    assert f["site"] == ["admin.ch", "example.com"] and f["exclude_site"] == ["wikipedia.org"]
    assert f["source"] == ["official"] and f["lang"] == "fr"
    assert "overrides lang=de" in f["note"]
    assert isinstance(got["op_filter"], qm.Filter)


def test_api_lang_only_operator_adds_no_qdrant_filter(monkeypatch):
    resp, got = _call(monkeypatch, "x country:ch")
    assert got["country"] == "CH" and "op_filter" not in got and "filters" not in resp and resp["boosts"] == {"country": "CH"}


def test_api_unknown_source_stays_in_the_query_with_a_note(monkeypatch, table):
    resp, got = _call(monkeypatch, "import os source:main")
    assert got["q"] == "import os source:main" and "op_filter" not in got
    assert resp["query"] == "import os source:main"
    assert resp["filters"] == {"note": "source:main is not a known source_type; kept as text (valid: official, reference, docs, ...)"}


def test_api_operators_only_query_is_400(monkeypatch, table):
    resp, _ = _call(monkeypatch, "site:admin.ch")
    assert resp.status_code == 400


# ---- MCP description -----------------------------------------------------------------------------
def test_mcp_description_names_every_operator_and_stays_short():
    src = pathlib.Path(_app().__file__).with_name("mcp_server.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "search")
    doc = inspect.cleandoc(ast.get_docstring(fn, clean=False))
    assert len(doc) < 1100
    for needle in ("site:", "-site:", "source:", "lang:", "country:", "exclude_site", "sub-host",
                   "site:fedlex.admin.ch", "-site:wikipedia.org source:official"):
        assert needle in doc, needle
    assert [a.arg for a in fn.args.args][-3:] == ["site", "exclude_site", "source"]


# ---- authorities sidecar + identifier fast path ----------------------------------------------------
def test_host_predicate_both_directions(table):
    inc = operators.host_predicate(Operators(site=("admin.ch",)))
    assert inc("fedlex.admin.ch") and inc("admin.ch") and not inc("notadmin.ch") and not inc("")
    exc = operators.host_predicate(Operators(exclude_site=("wikipedia.org",)))
    assert not exc("en.wikipedia.org") and exc("admin.ch")
    assert operators.host_predicate(Operators(source=("docs",))) is None


def test_sidecar_honours_site_and_exclude_site(monkeypatch, table):
    app = _app()
    seen = {}

    def fake_find(q, k=3, country=None, floor=None, host_ok=None):
        seen["ok"] = host_ok
        return [c for c in [{"domain": "fedlex.admin.ch"}, {"domain": "wikipedia.org"}]
                if host_ok is None or host_ok(c["domain"])]
    monkeypatch.setattr(app.authority, "find_authorities", fake_find)
    monkeypatch.setattr(app.retrieval, "search", lambda *a, **k: [])
    r = app.search(q="x site:admin.ch", k=10, lang=None, country=None, jurisdiction=None, authorities=1,
                   site=None, exclude_site=None, source=None)
    assert r["authorities"] == [{"domain": "fedlex.admin.ch"}]
    r = app.search(q="x -site:admin.ch", k=10, lang=None, country=None, jurisdiction=None, authorities=1,
                   site=None, exclude_site=None, source=None)
    assert r["authorities"] == [{"domain": "wikipedia.org"}]
    r = app.search(q="x", k=10, lang=None, country=None, jurisdiction=None, authorities=1,
                   site=None, exclude_site=None, source=None)
    assert seen["ok"] is None and len(r["authorities"]) == 2


def test_identifier_lookup_carries_the_operator_filter(monkeypatch):
    from cytir.serve import qdrant_io
    got = {}

    class C:
        def scroll(self, collection_name, scroll_filter, limit, with_payload):
            got["f"] = scroll_filter
            return [], None
    monkeypatch.setattr(qdrant_io, "client", lambda: C())
    monkeypatch.setattr(qdrant_io, "indexed_fields", lambda: frozenset({"register_id"}))
    opf = qm.Filter(must=[qm.FieldCondition(key="domain", match=qm.MatchValue(value="a.ch"))])
    qdrant_io._identifier_lookup("register_id", "CHE-1", 5)
    assert len(got["f"].must) == 1 and got["f"].must[0].key == "register_id"
    qdrant_io._identifier_lookup("register_id", "CHE-1", 5, opf)
    assert opf in got["f"].must


def test_named_country_param_echoes_boosts_like_the_operator(monkeypatch):
    resp, _ = _call(monkeypatch, "Art. 336 CO", country="ch")
    assert resp["boosts"] == {"country": "CH"} and resp["country"] == "CH" and "filters" not in resp
