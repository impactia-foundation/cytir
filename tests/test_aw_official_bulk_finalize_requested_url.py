"""Finalizers write requested_url through redact_url; public constant URLs stay byte-identical."""
import cytir.sources.official_bulk.finalize_companies_house as ch
import cytir.sources.official_bulk.finalize_gleif as gleif
import cytir.sources.official_bulk.finalize_sec as sec
from cytir.sources.official_bulk.http_evidence import redact_url


def test_finalizers_use_redact_url():
    assert sec.redact_url is redact_url and gleif.redact_url is redact_url
    assert ch.redact_url is redact_url


def test_public_constant_urls_are_byte_identical():
    assert redact_url(sec.URL) == sec.URL
    assert redact_url("https://leidata.gleif.org/api/v2/golden-copies/publishes/lei2/current") == \
        "https://leidata.gleif.org/api/v2/golden-copies/publishes/lei2/current"


def test_credentialed_url_is_redacted():
    assert redact_url("https://u:p@h.test/x") == "https://<redacted>@h.test/x"
