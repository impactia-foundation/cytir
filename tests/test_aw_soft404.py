"""The framework soft-404 detector (task 683, S7 part 2).

Real witnesses: the two ElCom error pages that answer HTTP 200 (sealed tape from task 683). The
framework, not each adapter, must exclude them, and must not exclude a long page that merely
discusses errors. No network.
"""
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tests.helpers.repo_files import repo_file_or_skip
from cytir.sources import soft404  # noqa: E402

TAPE = HERE / "fixtures" / "sitemap_elcom" / "tape.json"

CASES_EXCLUDED = [
    ("Ein Fehler ist aufgetreten", "Die aufgerufene Seite wurde leider nicht gefunden."),
    ("Fehler 500: Internal Server Error", "Der Server scheint momentan nicht verfuegbar zu sein."),
    ("Page not found", "Sorry."),
    ("Erreur 404", "La page demandee est introuvable."),
    ("Errore 404", "Nulla."),
    ("Startseite", "Diese Seite existiert nicht mehr. Bitte nutzen Sie die Suche."),
    ("Home", "The page you were looking for could not be found."),
    ("Accueil", "La page que vous recherchez n'existe pas."),
    ("Home", "La pagina richiesta non è stata trovata."),
    # three error pages found in a 199,494-point sample of the served index (task 683)
    ("Errore 500: Internal Server Error", "Il server sembra non essere disponibile al momento."),
    ("Errore 503 - Servizio temporaneamente non disponibile", "A causa di un errore interno."),
    ("Si è verificato un errore", "La pagina desiderata non è disponibile. È possibile che"),
    ("Pagina defecta", "Unsere Website wurde neu strukturiert. Die aufgerufene Seite wurde nicht gefunden."),
]

CASES_ADMITTED = [
    # a real page
    ("Aufgaben", "Die ElCom ist die unabhaengige staatliche Regulierungsbehoerde. " * 5),
    # a LONG page that discusses the error: the length guard admits it
    ("Page not found: how redirects work", "A 404 means the page was not found. " * 100),
    ("Fehler 404 in der Praxis", "Seite nicht gefunden, was nun? " * 100),
    # short page whose title only contains a number
    ("Art. 404 OR", "Der Auftrag kann von jedem Teil jederzeit widerrufen werden."),
    ("", ""),
]


@pytest.fixture
def regdocs_pack():
    """The pack is withheld from the public cut: the tests that read it skip there."""
    return repo_file_or_skip("sources/ch/regulator-docs.yaml")


@pytest.mark.parametrize("title,text", CASES_EXCLUDED)
def test_error_pages_are_detected(title, text):
    assert soft404.detect(title, text), (title, text)


@pytest.mark.parametrize("title,text", CASES_ADMITTED)
def test_real_and_long_pages_are_not(title, text):
    assert soft404.detect(title, text) is None


def test_the_two_real_elcom_pages_are_detected_through_the_framework(tmp_path, monkeypatch, word_only_chunking_allowed, regdocs_pack):
    """/de/404 and /de/500 are HTTP 200 on the real site: excluded, bytes still sealed."""
    sys.path.insert(0, str(HERE))
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setenv("AW_LANGGATE_OPTIONAL", "1")
    monkeypatch.setenv("AW_LID_MODEL", "/nonexistent/lid.176.bin")
    import test_aw_family_sitemap as T

    T.select_declared_policy(monkeypatch)    # task 852: the row declares the generic epoch-4 policy
    try:
        manifest, tape = T._run(tmp_path)
    finally:
        T.restore_chunker_defaults()
    by_unit = {u["unit"]: u for u in manifest["units"]}
    for code in ("404", "500"):
        u = by_unit[f"https://www.elcom.admin.ch/de/{code}"]
        assert u["terminal"] == "excluded" and u["reason"].startswith("soft_404:")
        assert u["records"] == 0
    admitted = [u for u in manifest["units"] if u["terminal"] == "admitted"]
    assert len(admitted) == 2
    # the bytes of an excluded page stay sealed (corpus-doctrine #6)
    sealed = {json.loads(line)["url"] for line in
              (tmp_path / "l0" / "index.jsonl").read_text().splitlines()}
    assert "https://www.elcom.admin.ch/de/404" in sealed
    # and no record of an error page reached the pack
    rows = [json.loads(line) for line in (tmp_path / "pack" / "records.jsonl").read_text().splitlines()]
    assert not [r for r in rows if r["url"].endswith(("/de/404", "/de/500"))]


def test_a_bad_schema_is_refused(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"schema": "x"}))
    with pytest.raises(ValueError):
        soft404.detect("a", "b", path=p)
