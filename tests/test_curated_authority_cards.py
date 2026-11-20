"""Task 2659: the curated authority cards (register-first, ADR-033/041) parse, are complete, carry a
tier consistent with the allowlist, are unique, and win a merge on host collision."""
import json
import pathlib

import yaml

from cytir.corpus import source_cards as sc

APP = pathlib.Path(__file__).resolve().parents[1]
CANTONS = {"ZH", "BE", "LU", "UR", "SZ", "OW", "NW", "GL", "ZG", "FR", "SO", "BS", "BL", "SH", "AR",
           "AI", "SG", "GR", "AG", "TG", "TI", "VD", "VS", "NE", "GE", "JU"}
MUST = {"fedlex.admin.ch", "bger.ch", "bj.admin.ch", "estv.admin.ch", "seco.admin.ch", "bag.admin.ch",
        "bfs.admin.ch", "zefix.ch", "shab.ch", "lexfind.ch", "docs.python.org", "pypi.org"}


def test_curated_file_parses_and_is_complete():
    cards = sc.load_curated()
    assert len(cards) >= 50
    for c in cards:
        assert set(c["name"]) == set(sc.CURATED_LANGS) and all(c["name"].values())
        assert c["tier"] in sc.TIER_TRUST and c["authoritative_for"] and c["description"]


def test_hosts_unique_and_named_set_present():
    hosts = [c["domain"] for c in sc.load_curated()]
    assert len(hosts) == len(set(hosts))
    assert MUST <= set(hosts)


def test_all_26_cantons_have_one_card():
    regions = [c["jurisdiction"].partition("-")[2] for c in sc.load_curated()
               if c["jurisdiction"].startswith("CH-")]
    assert sorted(regions) == sorted(CANTONS)


def test_tier_matches_allowlist():
    allow = {a["domain"]: a["tier"]
             for a in yaml.safe_load(open(APP / "data" / "allowlist.yaml"))["allowlist"]}
    for c in sc.load_curated():
        if c["domain"] in allow:
            assert c["tier"] == allow[c["domain"]], c["domain"]


def test_disposition_from_consent_registry_not_file(tmp_path, monkeypatch):
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"hosts": {"fedlex.admin.ch": {
        "state": "declined", "basis": "test", "search_ok": False, "train_ok": False}}}))
    monkeypatch.setenv("AW_CONSENT_REGISTRY", str(reg))
    by = {c["domain"]: sc.curated_card(c) for c in sc.load_curated()}
    assert by["fedlex.admin.ch"]["disposition"] == "consent-declined"
    assert by["bger.ch"]["disposition"] == "card"
    assert all("disposition" not in c for c in sc.load_curated())


def _disc(domain, **kw):
    return dict(domain=domain, set="swiss-discovery", disposition="card", trust_tier=0.85,
                authoritative_for=["x"], **kw)


def test_merge_prefers_curated_and_keeps_stricter_consent():
    cur = [sc.curated_card(c) for c in sc.load_curated()]
    declined = _disc("www.rsn.ne.ch", consent={"search_ok": False}, why="declined")
    declined["disposition"] = "consent-declined"
    base = [_disc("bger.ch"), _disc("other.ch"), declined]
    out = {c["domain"]: c for c in sc.merge_cards(base, cur)}
    assert out["bger.ch"]["source"] == "curated" and out["bger.ch"]["trust_tier"] == 1.0
    assert out["other.ch"]["set"] == "swiss-discovery"
    assert out["rsn.ne.ch"]["source"] == "curated"
    assert out["rsn.ne.ch"]["disposition"] == "consent-declined" and out["rsn.ne.ch"]["consent"]
    assert len(out) == len(cur) + 1                       # one non-colliding discovery card
    assert len(sc.merge_cards(base, cur)) == len(out)


def test_card_text_unchanged_without_names_and_names_embedded_for_curated():
    plain = {"domain": "a.ch", "authoritative_for": ["t"], "jurisdiction": "CH", "description": "d"}
    assert sc.card_text(plain) == "a.ch — authoritative source for: t. Jurisdiction: CH. d"
    c = sc.curated_card(next(x for x in sc.load_curated() if x["domain"] == "bger.ch"))
    assert "Tribunal federal" in sc.card_text(c) and "Bundesgericht" in sc.card_text(c)


# Three REAL discovery cards from the served snapshot (2,759 cards, task 2641), with their card_text as
# the pre-2659 formula produced it, frozen as literals: the embedding text of a card without `names`
# must not change, or promoting the merged set would silently re-embed them differently.
_FROZEN = [
    ({'domain': 'schuldenfragen.ch', 'authoritative_for': ['government', 'public administration', 'official records', 'regulation', 'public services', 'statistics'], 'jurisdiction': 'CH', 'description': ''},
     'schuldenfragen.ch — authoritative source for: government, public administration, official records, regulation, public services, statistics. Jurisdiction: CH.'),
    ({'domain': 'nestle.ch', 'authoritative_for': ['university', 'research', 'academic', 'higher education', 'science', 'publications'], 'jurisdiction': 'CH', 'description': ''},
     'nestle.ch — authoritative source for: university, research, academic, higher education, science, publications. Jurisdiction: CH.'),
    ({'domain': 'bfs.admin.ch', 'authoritative_for': ['Swiss official statistics', 'population statistics Switzerland', 'Bevölkerung Schweiz Statistik', 'consumer price index Switzerland (LIK)', 'inflation Switzerland', 'unemployment and labour market statistics', 'census data Switzerland', 'cantonal and communal statistics', 'GDP and economic statistics Switzerland', 'health, education and migration statistics', 'statistical publications and press releases Switzerland'], 'jurisdiction': 'CH', 'description': 'National statistical office: population, economy, prices/inflation, work, health, education, mobility statistics, publications and press releases in DE/FR/IT/EN. Portal pages are an AEM/Vue JS shell (no static text in crawl captures) — query its own search, statistics portal or APIs.'},
     'bfs.admin.ch — authoritative source for: Swiss official statistics, population statistics Switzerland, Bevölkerung Schweiz Statistik, consumer price index Switzerland (LIK), inflation Switzerland, unemployment and labour market statistics, census data Switzerland, cantonal and communal statistics, GDP and economic statistics Switzerland, health, education and migration statistics, statistical publications and press releases Switzerland. Jurisdiction: CH. National statistical office: population, economy, prices/inflation, work, health, education, mobility statistics, publications and press releases in DE/FR/IT/EN. Portal pages are an AEM/Vue JS shell (no static text in crawl captures) — query its own search, statistics portal or APIs.'),
]


def test_card_text_frozen_literals_for_real_discovery_cards():
    for card, text in _FROZEN:
        assert sc.card_text(card) == text


def test_four_wafblocked_hosts_are_carded_as_card_not_declined():
    by = {c["domain"]: sc.curated_card(c) for c in sc.load_curated()}
    for h in ("admin.ch", "legifrance.gouv.fr", "eur-lex.europa.eu", "bundesverwaltungsgericht.ch"):
        assert by[h]["disposition"] == "card", h


def test_home_override_and_registry_declines_for_three_hosts():
    by = {c["domain"]: sc.curated_card(c) for c in sc.load_curated()}
    assert by["gr-lex.gr.ch"]["navigation"]["home"] == "https://www.gr-lex.gr.ch/"
    assert by["rechtsbuch.tg.ch"]["navigation"]["home"] == "https://www.rechtsbuch.tg.ch/"
    assert by["bger.ch"]["navigation"]["home"] == "https://bger.ch/"
    for h in ("shab.ch", "rsn.ne.ch"):
        assert by[h]["disposition"] == "consent-declined", h
    ris = by["ris.bka.gv.at"]          # train-only decline: carried on the card, never blocks
    assert ris["disposition"] == "card" and ris["consent"] == {"search_ok": True, "train_ok": False, "checked_at": "2026-07-05T16:37:42Z"}


def test_train_only_discovery_decline_does_not_override_curated_card():
    cur = [sc.curated_card(c) for c in sc.load_curated()]
    d = _disc("ris.bka.gv.at", consent={"search_ok": True, "train_ok": False}, why="train-only")
    d["disposition"] = "consent-declined"
    out = {c["domain"]: c for c in sc.merge_cards([d], cur)}
    assert out["ris.bka.gv.at"]["disposition"] == "card" and out["ris.bka.gv.at"]["consent"]["train_ok"] is False
