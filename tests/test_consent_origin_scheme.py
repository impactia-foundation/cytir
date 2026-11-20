import cytir.consent.resolver as cr


class Response:
    status_code = 404
    text = ""

    def json(self):
        raise ValueError("not JSON")


def test_strict_origin_scheme_does_not_probe_sibling_scheme(monkeypatch):
    urls = []

    def fake_get(_client, url, **_kwargs):
        urls.append(url)
        return Response(), None

    monkeypatch.setattr(cr, "_get", fake_get)
    monkeypatch.setattr(cr.time, "sleep", lambda _seconds: None)
    receipt = cr.resolve_host(object(), "example.test", direct_crawl=True,
                              origin_scheme="http")
    assert urls == [
        "http://example.test/robots.txt",
        "http://example.test/.well-known/tdmrep.json",
    ]
    assert receipt["origin"] == "http://example.test"
    assert receipt["origin_scheme"] == "http"
    assert receipt["mode"] == "direct"


def test_origin_and_mode_have_disjoint_cache_keys(tmp_path):
    paths = {
        cr._cache_path(str(tmp_path), "example.test", False, "http"),
        cr._cache_path(str(tmp_path), "example.test", False, "https"),
        cr._cache_path(str(tmp_path), "example.test", True, "http"),
        cr._cache_path(str(tmp_path), "example.test", True, "https"),
    }
    assert len(paths) == 4


def test_nondefault_port_is_preserved_in_origin_and_cache_key(monkeypatch, tmp_path):
    urls = []

    def fake_get(_client, url, **_kwargs):
        urls.append(url)
        return Response(), None

    monkeypatch.setattr(cr, "_get", fake_get)
    monkeypatch.setattr(cr.time, "sleep", lambda _seconds: None)
    receipt = cr.resolve_host(object(), "example.test:8443",
                              direct_crawl=True, origin_scheme="https")
    assert receipt["host"] == "example.test:8443"
    assert receipt["origin"] == "https://example.test:8443"
    assert urls[0] == "https://example.test:8443/robots.txt"
    assert cr._cache_path(str(tmp_path), receipt["host"], True, "https").name == \
        "example.test:8443@https@direct.json"


def test_direct_url_gate_binds_nondefault_port():
    receipt = {
        "host": "example.test:8443", "origin_scheme": "https",
        "mode": "direct", "status": "ok", "search_ok": True,
        "_robots_body": "User-agent: *\nAllow: /\n",
    }
    assert cr.direct_url_allowed(receipt, "https://example.test:8443/a") is True
    assert cr.direct_url_allowed(receipt, "https://example.test/a") is False
    assert cr.direct_url_allowed(receipt, "https://example.test:9443/a") is False


def test_legacy_cache_key_is_unchanged(tmp_path):
    assert cr._cache_path(str(tmp_path), "example.test") == tmp_path / "example.test.json"
    assert cr._cache_path(str(tmp_path), "example.test", True) == \
        tmp_path / "example.test@direct.json"
