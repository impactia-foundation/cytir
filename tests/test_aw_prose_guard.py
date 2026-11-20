"""cytir.core.prose_guard -- task 767, defect B (binary PDF streams stored as text).

SEALED REAL ROWS: `fixtures/prose_guard/real_rows.jsonl` holds snippets read off the twin on
2026-10-08: 36 binary rows from weko.admin.ch / bazl.admin.ch (all three reasons) and 149 clean
controls (German/French legal text, markdown tables, Word-bullet text with private-use glyphs,
code, maths, encyclopaedia prose). The guard must flag every binary row and no control.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from cytir.core import common, prose_guard as PG

ROWS = [json.loads(line) for line in
        (pathlib.Path(__file__).parent / "fixtures" / "prose_guard" / "real_rows.jsonl")
        .read_text(encoding="utf-8").splitlines()]
BIN = [r for r in ROWS if r["kind"].startswith("bin:")]
CTL = [r for r in ROWS if r["kind"].startswith("ctl:")]
MISS = [r for r in ROWS if r["kind"].startswith("miss:")]


def test_fixture_has_the_three_binary_reasons_and_real_controls():
    assert {r["kind"] for r in BIN} == {"bin:pdf_structure", "bin:pdf_syntax", "bin:byte_noise"}
    assert len(BIN) >= 30 and len(CTL) >= 100
    assert {"ctl:pua", "ctl:table", "ctl:legal-de", "ctl:code"} <= {r["kind"] for r in CTL}


@pytest.mark.parametrize("row", BIN, ids=[r["id"][:8] for r in BIN])
def test_every_real_binary_row_is_refused_with_its_reason(row):
    why = PG.binary_content(row["snippet"])
    assert why == "binary_content:" + row["kind"].split(":")[1], row["snippet"][:80]


def test_zero_false_positives_on_the_controls():
    flagged = [(r["kind"], r["snippet"][:80]) for r in CTL if PG.binary_content(r["snippet"])]
    assert flagged == []


def test_controls_are_prose_and_the_garbage_is_labelled_as_a_known_miss():
    # task 777 (M5): six rows once labelled `ctl:census-fp` are binary bytes the guard cannot yet
    # tell from prose (no PDF syntax, noise below 4 %). They are KNOWN MISSES, not controls: the
    # zero-false-positive test must never assert that garbage passes.
    assert len(MISS) == 6 and all(r["kind"] == "miss:garbage" for r in MISS)
    assert sum(r["kind"] == "ctl:fp773" for r in CTL) >= 40
    assert len({r["domain"] for r in CTL if r["kind"] == "ctl:fp773"}) >= 30


@pytest.mark.parametrize("text", [
    "Gebühren: Reisepass £75.50, Visum £130, Zusatzgebühr £12 (Stand 2026) — £6.40 pro Woche.",
    "Mean difference 4.2 ± 0.3, 5.1 ± 0.2, 6.0 ± 0.4, 3.9 ± 0.1 and 7.2 ± 0.5 (n = 12, p < 0.05)",
    "Die Norm ‖x‖ ≤ ‖y‖ gilt, ebenso ‖a+b‖ ≤ ‖a‖ + ‖b‖ und ‖Ax‖ ≤ ‖A‖‖x‖ für alle x.",
    "Texto: ¿Qué hora es? ¡Qué bien! ¿Dónde vives? ¡Hola! ¿Cómo estás? ¡Adiós! ¿Y tú? ¡Claro!",
    "Fussnote¹ siehe Anhang² und Tabelle³; ¹ Gesetz, ¹ Verordnung, ¹ Bericht, ¹ Merkblatt ¹ Anhang",
    "Price ¥1,200, ¥3,400, ¥5,600 and ¥7,800; the fee is ¥900 per night plus ¥150 tax, ¥100 extra",
    "\x1b[31mERROR\x1b[0m disk full \x1b[32mOK\x1b[0m retry \x1b[33mWARN\x1b[0m slow \x1b[0m",
    "If \u200e$\u200ef$ \u200eis a \u200econcave function \u200ethen \u200e$\u200ef$ \u200eis increasing",
    "Andalucia est un film\u2063\u2063\u2063 \u2063\u2063français\u2063\u2063, \u2063 réalisé par Alain Gomis.",
    "Boxmeer[ 51°41′55″N 5°56′00″E] Oeijenbraak Someren[ 51°25′20″N 5°40′05″E] Oekel Zundert[ 51°28′N]",
    "Regular season ― ― ― Kansas City ― ― List of players ‡ ± 897 ‡ 508 443 231 881 ‡ 868 ‡ 289",
])
def test_task777_prose_symbols_are_not_byte_noise(text):
    assert PG.binary_content(text) is None


@pytest.mark.parametrize("text", [
    "Liste von Kartoffelsorten Name R 93 102 R 93 106 R 94 110 R 95 112 R 96 118 R Sorte",
    "# iosnoop UID PID D BLOCK SIZE COMM 100 15795 R 100 15796 R 100 15797 R 7 4 R 8 5 R",
    "Technique PDF1: see the 12 3 R and 14 5 R table rows 21 8 R 2 1 R of the report",
])
def test_task777_bare_n_n_r_is_not_pdf_syntax(text):
    assert PG.binary_content(text) is None


def test_task777_a_reference_array_still_needs_no_marker_when_generation_is_zero():
    assert PG.binary_content("[1569 0 R 1570 0 R 1571 0 R 1572 0 R 1573 0 R 1574 0 R]") \
        == "binary_content:pdf_structure"
    assert PG.binary_content("12 0 obj 5 7 R 6 8 R endobj") == "binary_content:pdf_structure"


@pytest.mark.parametrize("text", [
    # UTF-8 read as Latin-1/cp1252 (the exempted symbols sit beside non-ASCII letters)
    "DemnÃ¤chst in RiedenÂ **Ø¨Ù‡ÛŒÙ†Ù‡ Ø³Ø§Ø²ÛŒ Ø³Ø§ÛŒØª ÙˆØ±Ø²Ø´ÛŒ Ø§Ù†ØªØ®Ø§Ø¨ Ø§Ø³Øª",
    "â€¢ â¢ steuerfreie SonntagszuschlÃ¤ge â€¢ â¢ NachtzuschlÃ¤ge â€¢ â¢ Weiterbildung â€¢ â¢ Urlaub",
    # EUC-JP read as Latin-1
    "| 0054 | CASIO | ±©Â¼EMC¥»¥ó¥¿¡¼¡¡¥·¡¼¥ë¥É¥ë¡¼¥à2 | C-10224 | ¥Ø¡¼¥¿¡¼¥¹¥È¥µ¥¤¥È¡¡¥Æ¥¹¥È",
])
def test_task777_mojibake_is_still_noise(text):
    assert PG.binary_content(text) == "binary_content:byte_noise"


def test_pua_bullets_are_prose_not_noise():
    # WEKO decisions carry Word bullets as U+F02D; they sit well below the byte-noise share
    pua = [r for r in CTL if r["kind"] == "ctl:pua"]
    assert pua and all(PG.noise_share(r["snippet"]) < PG.NOISE_SHARE_MAX for r in pua)


@pytest.mark.parametrize("text", [
    "Art. 336 OR: Die Kündigung ist missbräuchlich, wenn sie ausgesprochen wird: a. aus einem Grund.",
    "Les « conditions » de l'article 12 § 3 sont remplies (2 °C, 5 µm, ½ heure).",
    "Un PDF contient des objets : il suffit qu'un lecteur voie un endobj isolé dans le texte.",
    "cd /usr/local && ls /var/log /etc /opt && cat /proc/cpuinfo",
    "| Jahr | Wert |\n|------|------|\n| 2024 | 12,5 |",
    "",
    None,
])
def test_prose_code_tables_and_empty_pass(text):
    assert PG.binary_content(text) is None


def test_pure_syntax_and_high_noise_are_refused():
    assert PG.binary_content("1135 0 obj <>/Font<>> endobj 1191 0 obj [1 0 R 2 0 R] endobj") \
        == "binary_content:pdf_structure"
    assert PG.binary_content("\x00\x01\x02" * 30) == "binary_content:byte_noise"
    assert PG.binary_content("�" * 50 + " text") == "binary_content:byte_noise"


def test_extraction_junk_carries_the_guard_so_every_ingest_path_gets_it():
    bad = next(r for r in BIN if r["kind"] == "bin:byte_noise")["snippet"]
    assert common.extraction_junk(bad) == "binary_content"
    # `low_quality` was blind to this (Latin-1 mojibake is alphabetic): it now asks the prose test
    assert common.low_quality(bad) is True
    assert common.low_quality("Die Wettbewerbskommission prüft den Zusammenschluss. " * 4) is False


# --- ROOT CAUSE: the HTML extractor fed a real PDF's bytes (task 767) -------------------------
PDF = pathlib.Path(__file__).parent / "fixtures" / "prose_guard" / "weko_bekanntmachung_homologation.pdf"


@pytest.mark.parametrize("enc,kw", [("latin-1", {}), ("cp1252", {"errors": "replace"}),
                                    ("utf-8", {"errors": "replace"})])
def test_the_html_extractor_refuses_the_bytes_of_a_real_pdf(enc, kw):
    pytest.importorskip("trafilatura")
    raw = PDF.read_bytes()
    assert raw.startswith(b"%PDF-")                       # a real WEKO decision (Bekanntmachung 1997)
    md, mode = common.extract_content_markdown(raw.decode(enc, **kw),
                                               "https://www.weko.admin.ch/dam/de/sd-web/x.pdf")
    assert (md, mode) == (None, None)


def test_before_the_fix_trafilatura_returned_the_garbage_the_index_holds():
    trafilatura = pytest.importorskip("trafilatura")
    text = PDF.read_bytes().decode("latin-1")
    md = trafilatura.extract(text, url="https://www.weko.admin.ch/x.pdf", output_format="markdown",
                             include_tables=True, favor_precision=True)
    assert md and md.startswith("%PDF-1.7") and "obj\n<>\nstream" in md   # the stored rows' shape
    assert PG.binary_content(md[:600]) is not None and common.low_quality(md)   # both gates now see it


def test_html_that_mentions_pdf_syntax_is_still_extracted():
    pytest.importorskip("trafilatura")
    html = ("<html><body><article><p>" + "Die Wettbewerbskommission hat entschieden. " * 12
            + "Ein PDF enthält Objekte wie 12 0 obj und endobj.</p></article></body></html>")
    md, mode = common.extract_content_markdown(html, "https://x.ch/a")
    assert md and "Wettbewerbskommission" in md
