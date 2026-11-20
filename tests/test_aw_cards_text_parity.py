"""Parity pins for `cytir.extract.retailer_card_channels` (task 1438, S7).

Every expected value below was computed ONCE by running the ORIGINAL driver module (blob
fa234d7f9435a803510f80ea7fe9ebe65ed3246c at origin/main) on the same fixed inputs. The test imports
only the package; table digests pin CAT_TERMS / CAT_SELLS / CAT_BRANDS content.
"""
from __future__ import annotations

import hashlib
import inspect
import json

import cytir.extract.retailer_card_channels as m

PINS = {'brands': ['Nike',
            'Adidas',
            'Puma',
            'Levis',
            'Tommy',
            'Hilfiger',
            'Calvin',
            'Klein',
            'Vans',
            'Converse',
            'Lacoste',
            'Hugo',
            'Boss',
            'Zara',
            'New',
            'Balance',
            'Reebok'],
 'bsm': [True, False, False],
 'dl': ['digitec', 'my shop', '', ''],
 'fs1': "The Quick brown fox, d'un l'été der und die Hund. Online-Shop für Möbel!",
 'fs2': "The Quick brown fox, d'un l'été der und die Hund.…",
 'fs3': '',
 'fs4': 'One.',
 'hx': [True, False],
 'ns': '6ba7b811-9dad-11d1-80b4-00c04fd430c8',
 'qids': ['Q1'],
 'reg': ['digitec.ch', 'amazon.com', 'amazon.de', '', 'a.b'],
 'rq': [['Q3', 'site-exact'], [None, 'unresolved']],
 'sells': 'a wide general assortment across many product categories; a broad third-party online '
          'marketplace spanning most product categories; fashion, clothing, shoes/footwear and '
          'accessories',
 'sld': ['amazon', 'exlibris', 'amazon', 'x', ''],
 'stop1': "commerce Mode Möbel für Schweiz Digitec l'été",
 'stop2': '',
 'stop3': '',
 'stop_sha': '49e916776eab8145d5cbec3cbb945fcafcd8f44b115644a5dcacfa3bb50b05ff',
 'strs': ['https://a.ch', '+2020'],
 'tables': {'CAT_BRANDS': '8325f35b97afa6ee98b910ad9faec15e5fcb41bb9c621d4ec2867e1a7c3c35eb',
            'CAT_SELLS': '1101bb548535bb5c4a1be5494e00f5aa1fce02cc5f8d56010f9ebfc25b1b854e',
            'CAT_TERMS': '2a8687b51d5805728a936977b038d95b02c70cb33661d92deb2bcb741eb4038a'},
 'ua': 'authoritative-web/1.0 (+https://websear.ch/bot; bender@impactia.org)'}

BIG = {'base': 'online shop Onlineshop Onlinehandel Versandhandel e-commerce ecommerce commerce boutique '
         'ligne vente ligne achats shopping acheter kaufen comprare buy détaillant Detailhändler '
         'Einzelhandel commerçant magasin Geschäft negozio',
 'cl': {'CA': 'en', 'CH': 'de', 'DE': 'de', 'FR': 'fr', 'GB': 'en', 'US': 'en'},
 'geo': {'CA': 'Canada canadian canadien',
         'CH': 'Schweiz Suisse Svizzera Switzerland swiss suisse svizzero helvétique',
         'DE': 'Deutschland Germany allemand deutsch',
         'FR': 'France français francais',
         'GB': 'UK United Kingdom britannique',
         'GLOBAL': 'international mondial global weltweit',
         'US': 'USA United States américain american'},
 'p31': ['Q11460',
         'Q13415637',
         'Q13417005',
         'Q161380',
         'Q1631129',
         'Q18043413',
         'Q180846',
         'Q194189',
         'Q219577',
         'Q2401749',
         'Q270791',
         'Q3277302',
         'Q431289',
         'Q4539',
         'Q4830453',
         'Q507619',
         'Q6881511',
         'Q7551384',
         'Q783794',
         'Q891723']}

CATS = ["general", "marketplace", "fashion", "nope"]
T = ("The Quick brown fox, d'un l'été der und die Hund. Online-Shop für Möbel! "
     "Second sentence here? Third one. " + "word " * 80)


def _mk(v):
    return {"claims": {"P": v}}


def _digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def test_text_helpers():
    assert m.drop_stopwords("Le commerce de la Mode et des Möbel für the Schweiz of Digitec digitec l'été") == PINS["stop1"]
    assert m.drop_stopwords("") == PINS["stop2"] and m.drop_stopwords(None) == PINS["stop3"]
    assert m.first_sentences(T) == PINS["fs1"]
    assert m.first_sentences(T, 3, 60) == PINS["fs2"]
    assert m.first_sentences("  ") == PINS["fs3"]
    assert m.first_sentences("One. Two. Three.", 1) == PINS["fs4"]
    assert hashlib.sha256(json.dumps(sorted(m._STOP)).encode()).hexdigest() == PINS["stop_sha"]


def test_host_helpers():
    assert [m.domain_label(x) for x in ["digitec.ch", "my-shop.co.uk", "", None]] == PINS["dl"]
    assert [m.registrable(x) for x in ["https://www.digitec.ch/de/x", "WWW.Amazon.com", "amazon.de", None, "//www.a.b/c"]] == PINS["reg"]
    assert [m._sld(x) for x in ["amazon.co.uk", "exlibris.com.cn", "https://www.amazon.com/x", "x", ""]] == PINS["sld"]
    assert [m.host_exact("amazon.com", ["https://www.amazon.com/", "x.ch"]), m.host_exact("a.ch", ["b.ch"])] == PINS["hx"]
    assert [m.brand_sld_match("amazon", ["amazon.de"]), m.brand_sld_match("", ["a.b"]),
            m.brand_sld_match("ama", ["amazon.de"])] == PINS["bsm"]


def test_claims_and_terms():
    assert m.claims_qids(_mk([{"mainsnak": {"datavalue": {"value": {"id": "Q1"}}}}, {"mainsnak": {}},
                              {"mainsnak": {"datavalue": {"value": "s"}}}]), "P") == PINS["qids"]
    assert m.claims_strings(_mk([{"mainsnak": {"datavalue": {"value": "https://a.ch"}}},
                                 {"mainsnak": {"datavalue": {"value": {"time": "+2020"}}}},
                                 {"mainsnak": {"datavalue": {"value": {"id": "Q"}}}}, {}]), "P") == PINS["strs"]
    assert m.sells_phrase(CATS) == PINS["sells"]
    assert m.brands_for(CATS) == PINS["brands"]
    for k, want in PINS["tables"].items():
        assert _digest(getattr(m, k)) == want, k
    assert not hasattr(m, "UA") and str(m.NS) == PINS["ns"]
    assert "_http()" in inspect.getsource(m.wd_api) and "_http()" in inspect.getsource(m.wp_summary)  # identity = the factory's operator UA (task 1620)
    assert m.GEO_TERMS == BIG["geo"] and m.BASE_COMMERCE == BIG["base"]
    assert m.COUNTRY_LANG == BIG["cl"] and sorted(m.RETAIL_P31) == BIG["p31"]


def test_resolve_qid_offline(monkeypatch):
    def wd(params):
        if params["action"] == "wbsearchentities":
            return {"search": [{"id": "Q2"}, {"id": "Q3"}]}
        i = params["ids"]
        site = "https://www.shop.ch/" if i == "Q3" else "https://other.com"
        return {"entities": {i: {"claims": {
            "P856": [{"mainsnak": {"datavalue": {"value": site}}}],
            "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q4830453"}}}}]}}}}

    monkeypatch.setattr(m, "wd_api", wd)
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    got = [m.resolve_qid({"domain": "shop.ch", "country": "CH", "search": "Shop", "brand": "shop", "pin": "Q9"}),
           m.resolve_qid({"domain": "none.ch", "country": "FR", "search": "x"})]
    assert [list(g) for g in got] == [list(g) for g in PINS["rq"]]
