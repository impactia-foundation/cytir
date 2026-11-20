"""Card channel primitives: the live helpers of the retired task-299 retailer-card driver.

ADR-060 two-channel recipe pieces shared by `cytir.corpus.card_recipe`, `build351_phase3_wikidata` (deleted; last at git-blob:e334582b365f28bc9d1be605771ab8ab310111cb:build351_phase3_wikidata.py) and
`representation_contract`: stopword-filtered sparse text (`drop_stopwords`), lead sentences,
host/domain normalisation (`registrable`), the Wikidata/Wikipedia fetchers (`wd_api`,
`wp_summary`, ADR-040: no Brave/Google/Bing), claim readers, the category term tables and the
validated QID resolver. Moved byte-for-byte from the retired task-299 retailer-cards driver
(blob fa234d7f9435a803510f80ea7fe9ebe65ed3246c); the driver half (SEEDS, build_card, main) was dead
and is gone. The retired driver's file is now a forwarding shim (task 1438, S7).
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import uuid

from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client

NS = uuid.NAMESPACE_URL

RETAIL_P31 = {
    "Q4830453", "Q6881511", "Q783794", "Q891723", "Q219577", "Q18043413",
    "Q13415637", "Q13417005", "Q1631129", "Q431289", "Q507619", "Q194189",
    "Q3277302", "Q180846", "Q161380", "Q4539", "Q270791",
    "Q2401749", "Q7551384", "Q11460",
}

CAT_TERMS = {
    "electronics": "electronics électronique Elektronik elettronica informatique Informatik "
        "computer computers ordinateur ordinateurs PC laptop laptops notebook notebooks "
        "smartphone smartphones téléphone Handy Mobiltelefon télévision Fernseher televisore "
        "appareils photo caméra Kamera fotocamera gaming console Spielkonsole imprimante printer "
        "Drucker Haushaltsgeräte électroménager appliances Unterhaltungselektronik hardware audio hifi",
    "marketplace": "marketplace Marktplatz marché annonces classifieds petites annonces Kleinanzeigen "
        "enchères auctions Auktionen occasion gebraucht seconde main vendeurs revendeurs",
    "groceries": "supermarché Supermarkt supermercato groceries alimentation Lebensmittel courses "
        "Onlinesupermarkt épicerie food nourriture Getränke boissons produits frais Frischeprodukte",
    "furniture": "meubles Möbel furniture mobili ameublement Einrichtung décoration Wohnen intérieur "
        "cuisine Küche literie matelas Matratzen canapé Sofa",
    "fashion": "mode Mode fashion moda vêtements Kleidung Bekleidung abbigliamento habits chaussures "
        "Schuhe shoes scarpe accessoires Accessoires prêt-à-porter",
    "sporting": "sport Sport articles sport Sportartikel Sportgeschäft matériel sport équipement sportif "
        "fitness outdoor vélo Velo Fahrrad randonnée camping",
    "diy": "bricolage Baumarkt DIY heimwerken jardinage Garten jardin outils Werkzeug matériaux "
        "construction quincaillerie",
    "department": "grand magasin Warenhaus department store Kaufhaus beauté maison parfumerie",
    "books": "livres Bücher books libri médias musique films DVD papeterie",
    "general": "assortiment vaste choix produits Vielfalt Sortiment gamme",
}
# Readable product-RANGE per category (for the DENSE "what it sells" prose — the
# Zweck principle: describe the retailer's SCOPE, not a product catalog; ADR-039).
CAT_SELLS = {
    "electronics": "electronics, computers, laptops, desktops, printers, monitors, TVs, "
        "smartphones, tablets, cameras, home appliances and gaming",
    "marketplace": "a broad third-party online marketplace spanning most product categories",
    "groceries": "groceries, fresh food, beverages and everyday household goods",
    "furniture": "furniture, home furnishings, kitchens, decoration and mattresses",
    "fashion": "fashion, clothing, shoes/footwear and accessories",
    "sporting": "sports equipment, sportswear, outdoor, fitness and bicycles",
    "diy": "DIY, hardware, tools, building materials and garden",
    "department": "a department-store range: fashion, home, furniture, electronics and beauty",
    "books": "books, media, music, films and games",
    "general": "a wide general assortment across many product categories",
}
# Notable BRANDS carried at CATEGORY level (own general knowledge, ADR-040-clean —
# NOT scraped from the retailer; describes what the category stocks so a
# brand+category query like "imprimante epson" matches). Category-scope, not a claim
# about a specific SKU. Only categories where brand-matching is meaningful.
CAT_BRANDS = {
    "electronics": "Epson HP Canon Brother Apple iPhone iPad MacBook Samsung Galaxy Lenovo Dell "
        "Asus Acer Microsoft Surface Sony LG Philips Bose JBL Sonos Nintendo Playstation Xbox "
        "Bosch Siemens Miele Logitech Xiaomi Huawei Garmin GoPro Dyson Nespresso",
    "fashion": "Nike Adidas Puma Levis Tommy Hilfiger Calvin Klein Vans Converse Lacoste "
        "Hugo Boss Zara New Balance Reebok",
    "sporting": "Nike Adidas Puma Salomon Mammut Scott Garmin Odlo Asics On-Running Wilson",
    "diy": "Bosch Makita Gardena Kärcher Einhell Hilti Metabo",
    "department": "Nike Adidas Levis Samsung Apple Bosch Lego",
}


def sells_phrase(cats):
    return "; ".join(dict.fromkeys(CAT_SELLS.get(c, "") for c in cats if CAT_SELLS.get(c)))


def brands_for(cats):
    toks = []
    for c in cats:
        for b in CAT_BRANDS.get(c, "").split():
            if b not in toks:
                toks.append(b)
    return toks


BASE_COMMERCE = ("online shop Onlineshop Onlinehandel Versandhandel e-commerce ecommerce commerce "
                 "boutique ligne vente ligne achats shopping acheter kaufen comprare buy "
                 "détaillant Detailhändler Einzelhandel commerçant magasin Geschäft negozio")
GEO_TERMS = {
    "CH": "Schweiz Suisse Svizzera Switzerland swiss suisse svizzero helvétique",
    "DE": "Deutschland Germany allemand deutsch",
    "FR": "France français francais",
    "CA": "Canada canadian canadien",
    "US": "USA United States américain american",
    "GB": "UK United Kingdom britannique",
    "GLOBAL": "international mondial global weltweit",
}


_STOP = set("""
de des du la le les un une et ou aux en pour par sur avec dans se sa son ses ne
der die das den dem und oder fur von mit im am zum zur ein eine einen auf aus sich
di del della dei delle il lo gli una uno per con su tra fra al alla allo alle
the and or of to in on for with at by an as is are be it that this
d l e a o y su da ma si no com net org www
""".split())
_WORD = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)


def drop_stopwords(text):
    out, seen = [], set()
    for w in _WORD.findall(text or ""):
        wl = w.lower()
        if len(wl) < 2 or wl in _STOP or wl in seen:
            continue
        seen.add(wl)
        out.append(w)
    return " ".join(out)


_SENT = re.compile(r"(?<=[.!?])\s+")


def first_sentences(text, n=2, cap=340):
    text = (text or "").strip()
    if not text:
        return ""
    out = " ".join(_SENT.split(text)[:n]).strip()
    if len(out) > cap:
        out = out[:cap].rsplit(" ", 1)[0] + "…"
    return out


def domain_label(dom):
    return (dom or "").split(".")[0].replace("-", " ")


def registrable(host):
    host = (host or "").lower()
    m = re.search(r"//([^/]+)", host)
    if m:
        host = m.group(1)
    return host.replace("www.", "")


_CLIENT = None


def _http():
    """The module's lazy, consent-guarded client; identity is the operator User-Agent as before."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = make_client(HttpSpec(timeout=45, follow_redirects=True))
    return _CLIENT


def wd_api(params):
    url = "https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode(params)
    r = _http().get(url, timeout=45)
    r.raise_for_status()
    return json.loads(r.content)


def wp_summary(title, lang):
    t = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{t}"
    try:
        r = _http().get(url, timeout=30)
        r.raise_for_status()
        d = json.loads(r.content)
        return (d.get("extract") or "").strip()
    except Exception:
        return ""


def claims_qids(ent, prop):
    out = []
    for c in ent.get("claims", {}).get(prop, []):
        try:
            dv = c["mainsnak"]["datavalue"]["value"]
            if isinstance(dv, dict) and "id" in dv:
                out.append(dv["id"])
        except Exception:
            pass
    return out


def claims_strings(ent, prop):
    out = []
    for c in ent.get("claims", {}).get(prop, []):
        try:
            dv = c["mainsnak"]["datavalue"]["value"]
            if isinstance(dv, str):
                out.append(dv)
            elif isinstance(dv, dict) and "time" in dv:
                out.append(dv["time"])
        except Exception:
            pass
    return out


def _p856_p31(qid):
    ent = wd_api({"action": "wbgetentities", "ids": qid, "props": "claims",
                  "format": "json"}).get("entities", {}).get(qid, {})
    return claims_strings(ent, "P856"), claims_qids(ent, "P31")


COUNTRY_LANG = {"CH": "de", "FR": "fr", "DE": "de", "CA": "en", "US": "en", "GB": "en"}


_SLD2 = {"co", "com", "org", "net", "gov", "ac", "edu"}   # second-level public suffixes


def _sld(host):
    """The registrable second-level label: amazon.com->amazon, exlibris.com.cn->
    exlibris, amazon.co.uk->amazon. EXACT token, never a substring."""
    parts = registrable(host).split(".")
    if len(parts) >= 3 and parts[-2] in _SLD2:
        return parts[-3]
    if len(parts) >= 2:
        return parts[-2]
    return parts[0] if parts else ""


def host_exact(card_domain, sites):
    return any(registrable(s) == card_domain for s in sites)


def brand_sld_match(brand, sites):
    return bool(brand) and any(brand == _sld(s) for s in sites)


def resolve_qid(seed):
    brand = seed.get("brand") or domain_label(seed["domain"]).replace(" ", "")
    cands = []
    if seed.get("pin"):
        cands.append(seed["pin"])
    langs = ["en"]
    nl = COUNTRY_LANG.get(seed["country"])
    if nl and nl not in langs:
        langs.append(nl)
    for lg in langs:
        d = wd_api({"action": "wbsearchentities", "search": seed["search"],
                    "language": lg, "format": "json", "limit": 7, "type": "item"})
        cands += [h["id"] for h in d.get("search", []) if h["id"] not in cands]
    # EXACT-HOST is the only unambiguous signal (two firms can share a brand name:
    # 'exlibris' = Migros bookstore exlibris.ch AND the Israeli library-software
    # Ex Libris on exlibris.com.cn — brand/SLD matching cannot tell them apart).
    # Brand/SLD matching is allowed ONLY for a flagged country storefront of a
    # global chain (amazon.de -> Amazon@amazon.com). NO generic p31-retail fallback
    # (it grabbed the wrong same-named company); unresolved -> enrich-from-base,
    # which is correct for name collisions. NO pin-soft (a bad pin gave Siam
    # Commercial Bank as MediaMarkt).
    storefront = seed.get("storefront", False)
    store_hit = None
    for qid in cands[:14]:
        try:
            sites, p31 = _p856_p31(qid)
        except Exception:
            continue
        time.sleep(0.15)
        if host_exact(seed["domain"], sites):
            return qid, "site-exact"
        if (store_hit is None and storefront and brand_sld_match(brand, sites)
                and any(q in RETAIL_P31 for q in p31)):
            store_hit = qid
    if store_hit:
        return store_hit, "storefront-brand"
    return None, "unresolved"

