import cytir.consent.resolver as C


def _receipt(body: str = "") -> dict:
    return {
        "host": "example.test", "mode": "direct", "status": "ok",
        "search_ok": True, "train_ok": False, "_robots_body": body,
        "signals": {"robots": {"fetched": True, "http_status": 200}},
    }


def test_direct_url_allowed_honours_exact_path_and_both_groups():
    receipt = _receipt(
        "User-agent: authoritative-web\nDisallow: /private\n"
        "User-agent: *\nDisallow: /wildcard\n")
    assert C.direct_url_allowed(receipt, "https://example.test/public")
    assert not C.direct_url_allowed(receipt, "https://example.test/private/a")
    assert not C.direct_url_allowed(receipt, "https://example.test/wildcard/a")


def test_direct_url_allowed_fails_closed_on_identity_or_mode_drift():
    assert not C.direct_url_allowed(_receipt(), "https://other.test/public")
    receipt = _receipt(); receipt["mode"] = "cc"
    assert not C.direct_url_allowed(receipt, "https://example.test/public")
    receipt = _receipt(); receipt["status"] = "unknown"
    assert not C.direct_url_allowed(receipt, "https://example.test/public")
    receipt = _receipt(); receipt["origin_scheme"] = "http"
    assert not C.direct_url_allowed(receipt, "https://example.test/public")
    assert C.direct_url_allowed(receipt, "http://example.test/public")


def test_direct_url_allowed_accepts_readable_empty_policy_only():
    assert C.direct_url_allowed(_receipt(), "https://example.test/public")
    for status in (404, 410):
        receipt = _receipt(); receipt["signals"]["robots"]["http_status"] = status
        assert C.direct_url_allowed(receipt, "https://example.test/public")
    for status in (0, 204, 400, 429, 500):
        receipt = _receipt(); receipt["signals"]["robots"]["http_status"] = status
        assert not C.direct_url_allowed(receipt, "https://example.test/public")
    receipt = _receipt(); receipt["signals"]["robots"]["fetched"] = False
    assert not C.direct_url_allowed(receipt, "https://example.test/public")


def test_direct_url_allowed_binds_exact_authority_including_port():
    receipt = _receipt(); receipt["host"] = "example.test:8443"
    assert C.direct_url_allowed(receipt, "https://example.test:8443/public")
    assert not C.direct_url_allowed(receipt, "https://example.test/public")
    assert not C.direct_url_allowed(receipt, "https://example.test:9443/public")
