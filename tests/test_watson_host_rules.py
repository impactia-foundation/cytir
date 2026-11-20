"""watson.ch host rules (Board admission 2026-10-03, POLICY §4.7, pack `sources/ch/watson.yaml`):
licence and country are stamped by host, from the pack's own declarations, like the SRG publishers."""
import pathlib
import re

import pytest

from cytir.core import country as C
from cytir.core import license as L

PACK = pathlib.Path(__file__).resolve().parents[1] / "sources" / "ch" / "watson.yaml"


def _pack_value(key_path):
    text = PACK.read_text(encoding="utf-8")
    m = re.search(rf"^{key_path}:\s*(\S+)", text, re.M)
    return m.group(1) if m else None


@pytest.mark.parametrize("host", ["watson.ch", "www.watson.ch", "news.watson.ch"])
def test_watson_rows_get_the_pack_licence(host):
    got = L.resolve_license(None, host, host, "news_article")
    assert got == "copyright-fairuse-snippet-pointer"
    assert got == L.resolve_license(None, "srf.ch", "www.srf.ch", "news_article"), "same posture as the SRG publishers"


def test_the_licence_is_the_packs_license_default():
    text = PACK.read_text(encoding="utf-8")
    assert re.search(r"license:\s*\n\s+default:\s*copyright-fairuse-snippet-pointer", text)
    assert "sources/ch/watson.yaml" in L._HOST_LICENSE["watson.ch"].citation


def test_a_lookalike_host_does_not_inherit_it():
    assert L.resolve_license(None, "notwatson.ch", "notwatson.ch", "news_article") is None
    assert L.resolve_license(None, "watson.ch.evil.example", "watson.ch.evil.example", "news_article") is None


def test_an_existing_licence_on_the_record_still_wins():
    assert L.resolve_license("CC0-1.0", "watson.ch", "www.watson.ch", "entity_card") == "CC0-1.0"


def test_watson_is_stamped_swiss_as_its_pack_declares():
    assert re.search(r"^country:\s*CH\b", PACK.read_text(encoding="utf-8"), re.M)
    for d in ("watson.ch", "www.watson.ch"):
        assert C.country_for_record(d, "news_article") == "CH"
    assert C.country_for_record("notwatson.ch", "news_article") is None, "a lookalike host is not declared"
