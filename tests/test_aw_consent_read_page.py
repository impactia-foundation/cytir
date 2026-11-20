"""Task 2804: `cyt consent read-page` records (no network: transport mocked)."""
import hashlib
import json

import httpx

from cytir.consent import read_page, resolver

READABLE = "<html><body><h1>Terms</h1><p>" + ("Content may be reused with attribution. " * 30) + "</p></body></html>"
SHELL = "<html><head></head><body><div id=app></div><noscript>Enable JavaScript</noscript></body></html>"


def _run(tmp_path, monkeypatch, status, body, url="https://example.org/terms"):
    def handler(request):
        return httpx.Response(status, content=body.encode(), headers={"content-type": "text/html"})
    monkeypatch.setattr(resolver, "_curl_get", lambda *a, **k: None)
    client = httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": resolver.UA_STR})
    return read_page.read_page(client, url, tmp_path), url


def test_readable_page(tmp_path, monkeypatch):
    rec, url = _run(tmp_path, monkeypatch, 200, READABLE)
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    saved = json.loads((tmp_path / f"{key}.json").read_text())
    assert saved["blocked"] == "" and saved["http_status"] == 200
    assert saved["sha256"] == hashlib.sha256(READABLE.encode()).hexdigest()
    assert (tmp_path / f"{key}.body").read_bytes() == READABLE.encode()
    assert "reused with attribution" in saved["text_excerpt"] and len(saved["text_excerpt"]) <= 4000
    assert read_page.first_sentence(rec["_text"], "attribution").startswith("Content may be reused")


def test_js_shell_unreadable(tmp_path, monkeypatch):
    rec, _ = _run(tmp_path, monkeypatch, 200, SHELL)
    assert rec["blocked"] == "js_shell"


def test_412_unreadable(tmp_path, monkeypatch):
    rec, _ = _run(tmp_path, monkeypatch, 412, "nope")
    assert rec["blocked"] == "http_412" and rec["http_status"] == 412


def test_cli_exit_zero_on_unreadable(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(resolver, "_get", lambda c, u, **k: (None, "ConnectError"))
    assert read_page.main(["https://example.org/x", "--out", str(tmp_path)]) == 0
    assert "unreadable (no_response)" in capsys.readouterr().out


def _mock_client(monkeypatch, status=200, body=READABLE):
    monkeypatch.setattr(resolver, "_curl_get", lambda *a, **k: None)
    monkeypatch.setattr(resolver, "_client", lambda: httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(status, content=body.encode()))))


def test_cli_routing_end_to_end(tmp_path, monkeypatch, capsys):
    from cytir import cli
    _mock_client(monkeypatch)
    rc = cli._consent_main(["read-page", "https://example.org/t", "--out", str(tmp_path),
                            "--quote", "attribution"])
    out = capsys.readouterr().out
    assert rc == 0 and "readable" in out and "quote: Content may be reused" in out


def test_cli_bad_regex_exits_2_before_fetch(tmp_path, monkeypatch, capsys):
    from cytir import cli
    monkeypatch.setattr(resolver, "_client", lambda: (_ for _ in ()).throw(AssertionError("fetched")))
    rc = cli._consent_main(["read-page", "https://example.org/t", "--out", str(tmp_path), "--quote", "("])
    assert rc == 2 and "invalid --quote" in capsys.readouterr().err


def test_body_capped_and_flagged(tmp_path, monkeypatch):
    big = "<html><body>" + "x" * (resolver.BODY_BYTE_CAP + 10) + "</body></html>"
    rec, _ = _run(tmp_path, monkeypatch, 200, big)
    assert rec["truncated"] is True and rec["bytes"] == resolver.BODY_BYTE_CAP


def test_challenge_page_200_unreadable(tmp_path, monkeypatch):
    body = ("<html><body><p>Due to unusually high traffic, we need to verify that requests "
            "are coming from real users.</p></body></html>")
    rec, _ = _run(tmp_path, monkeypatch, 200, body)
    assert rec["blocked"] == "challenge"


def test_long_legal_page_mentioning_javascript_readable(tmp_path, monkeypatch):
    body = READABLE.replace("</p>", " Enable JavaScript and cookies. " + "More terms apply. " * 60 + "</p>")
    rec, _ = _run(tmp_path, monkeypatch, 200, body)
    assert rec["blocked"] == ""


def test_text_plain_body_readable(tmp_path, monkeypatch):
    body = "Licence: CC-BY 4.0. " * 40
    def handler(request):
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/plain"})
    monkeypatch.setattr(resolver, "_curl_get", lambda *a, **k: None)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    rec = read_page.read_page(client, "https://example.org/llms.txt", tmp_path)
    assert rec["blocked"] == "" and rec["text_source"] == "plain"
    assert rec["text_excerpt"].startswith("Licence: CC-BY") and len(rec["text_excerpt"]) <= 4000


def test_vantage_lowercase(tmp_path, monkeypatch):
    monkeypatch.setattr(read_page.socket, "gethostname", lambda: "Probe-Host")
    rec, _ = _run(tmp_path, monkeypatch, 200, READABLE)
    assert rec["vantage"] == "probe-host" and rec["text_source"] == "html"


def test_short_challenge_with_noscript_is_js_shell(tmp_path, monkeypatch):
    body = ("<html><body><p>Verification Required</p>"
            "<noscript>Enable JavaScript</noscript></body></html>")
    rec, _ = _run(tmp_path, monkeypatch, 200, body)
    assert rec["blocked"] == "js_shell"


def test_challenge_phrase_only_in_script_uses_raw_body(tmp_path, monkeypatch):
    body = ("<html><head><script>var m='Checking your browser before accessing';</script>"
            "<!-- Just a moment --></head><body></body></html>")
    rec, _ = _run(tmp_path, monkeypatch, 200, body)
    assert rec["text_excerpt"].strip() == "" and rec["blocked"] == "challenge"
