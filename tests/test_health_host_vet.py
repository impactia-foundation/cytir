"""Sealed-sample test for cytir.ops.health_host_vet (task 991). No network."""
import json

from cytir.ops import health_host_vet as v

RULES = v.load_rules()

CASES = {
    "ouh.nhs.uk": "official", "bag.admin.ch": "official", "tools.rki.de": "official",
    "sanidad.gob.es": "official", "gesundheit.gv.at": "official", "digital.gob.es": "gov_nonhealth",
    "covid19.who.int": "igo", "shop.un.org": "igo", "faostat.fao.org": "igo",
    "cks.nice.org.uk": "society", "training.cochrane.org": "society", "leitlinien.dgk.org": "society",
    "jama.ama-assn.org": "society", "aerzte-gesellschaft.example.org": "society",
    "mayoclinic.org": "academic_medical", "neurologie.insel.ch": "academic_medical",
    "pure.mpg.de": "academic_medical", "medicine.yale.edu": "academic_medical",
    "cs.uni-salzburg.at": "academic_other", "math.example.edu": "academic_other",
    "nejm.org": "publisher", "link.springer.com": "publisher", "journals.plos.org": "publisher",
    "klinik-am-see.de": "hospital_provider", "apotheke-beispiel.ch": "hospital_provider",
    "spital-beispiel.ch": "hospital_provider",
    "webmd.com": "commercial_wellness", "focus-gesundheit.de": "commercial_wellness",
    "vitamin-shop.de": "commercial_wellness", "fitness-blog.org": "commercial_wellness",
    "happinet-phantom.com": "unknown", "9news.com": "unknown", "polisen.se": "unknown",
    "cancer-cure-news.com": "unknown", "uni-fun.com": "unknown", "random.example.org": "unknown",
    "drugs.com": "commercial_wellness", "who.int": "igo", "hse.ie": "official", "aap.org": "society",
}


def test_sealed_sample_categories():
    got = {h: v.classify(h, RULES)[0] for h in CASES}
    assert got == CASES


def test_every_decision_has_a_basis():
    for h in CASES:
        cat, basis = v.classify(h, RULES)
        assert cat in v.CATEGORIES and basis


def test_commercial_tld_never_reaches_a_trusted_pattern_category():
    cat, basis = v.classify("heart-society-medical.com", RULES)
    assert cat in ("unknown", "commercial_wellness") and cat not in v.ADMIT_SET


def test_vet_rows_and_summary(tmp_path):
    roster = [{"domain": h, "class": "health", "n_src": 6 if i % 2 else 2, "n_endorser_regs": 3}
              for i, h in enumerate(CASES)] + [{"domain": "x.dev", "class": "tech", "n_src": 9}]
    rows = v.vet_rows(roster, RULES)
    assert len(rows) == len(CASES)
    s = v.summarize(rows)
    assert s["total_hosts"] == len(CASES)
    assert sum(c["hosts"] for c in s["categories"].values()) == len(CASES)
    assert s["categories"]["igo"]["est_rows"] == s["categories"]["igo"]["hosts"] * 145
    p = tmp_path / "r.json"
    p.write_text(json.dumps(roster))
    assert v.main(["--roster", str(p), "--out", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "summary.json").exists()


def test_country_split():
    assert v.country("x.admin.ch", RULES) == "ch" and v.country("x.who.int", RULES) == "intl"
    assert v.country("x.nhs.uk", RULES) == "uk" and v.country("x.co.jp", RULES) == "other"
