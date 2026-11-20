"""`cytir.ops.spam_rows` R7 (task 704): the injected anchor `Beste Online Casino Deutschland`.

Fixtures are the snippets read from the twin on 2026-10-03: deutsch-plus.de tag pages whose row is
the menu led by the anchor, one spam sentence, and genuine rows of the same host that must stay.
"""
from __future__ import annotations

from cytir.ops import spam_rows as S

MENU = ("Beste Online Casino Deutschland Über uns Kontakt Was Wir tun Projekte Kampagnen "
        "Aktuelles Events Was Du tun kannst Kampagnen Fördermitglied werden Spenden Artikel schreiben Newsletter")
SENTENCE = ("Sie möchten ohne Limit spielen? Dann sollten Sie das beste Online Casino Deutschland "
            "mit internationaler Lizenz wählen.")
GENUINE = [
    ("Datenschutz | DeutschPlus e.V.",
     "Wir klären Sie darüber auf, dass die Bereitstellung personenbezogener Daten zum Teil gesetzlich vorgeschrieben ist."),
    ("Artikel schreiben",
     "Mit unserem Magazin bringen wir neue Perspektiven in die Einwanderungsgesellschaft ein."),
    ("Veranstaltung",
     "DeutschPlus stellt Ergebnisse des Kooperationsprojekts Vielfaltscheck Diakonie und Kirchen vor."),
]


def test_menu_row_with_injected_anchor_is_r7():
    why = S.row_reason("Gleichberechtigung | DeutschPlus e.V.", MENU, "https://www.deutsch-plus.de/tag/gleichberechtigung")
    assert why == "R7"


def test_spam_sentence_is_r7():
    assert S.row_reason("Aktuelles", SENTENCE, "https://www.deutsch-plus.de/was-wir-tun/aktuelles") == "R7"


def test_the_lexicon_alone_never_flagged_it():
    assert len(S.lexicon_hits(MENU)) == 1  # one hit: R1 (>= 3 distinct, >= 4 hits) cannot fire


def test_genuine_rows_of_the_same_host_stay():
    for title, text in GENUINE:
        assert S.row_reason(title, text, "https://www.deutsch-plus.de/x") is None


def test_an_article_about_online_casinos_in_general_stays():
    assert S.row_reason("Glücksspielstaatsvertrag", "Der Staatsvertrag regelt Online Casinos in Deutschland neu.") is None


def test_r7_flags_the_row_only_not_the_page():
    rows = [("a", "https://x.example/p", "R7", 1), ("b", "https://x.example/p", None, 0)]
    assert S.page_flags(rows) == {"a": "R7"}
