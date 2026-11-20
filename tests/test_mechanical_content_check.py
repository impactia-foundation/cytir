#!/usr/bin/env python3
"""Real-record guards for `mechanical_content_check` (task 641) -- ADR-110's
"content guard passes", read per the 2026-09-23 ruling as a mechanical,
per-row, no-query check."""
from __future__ import annotations

import gzip
import json
import os
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent

@pytest.fixture(autouse=True)
def _operator_identity(monkeypatch):
    """Operator identity for each test, restored after it. A module-level setdefault leaked it into every later test."""
    monkeypatch.setenv("AW_OPERATOR_URL", "https://websear.ch")
    monkeypatch.setenv("AW_OPERATOR_CONTACT", "bender@impactia.org")

from cytir.gates import mechanical_content_check as M  # noqa: E402
from cytir.index import promote_sealed_pack as PSP  # noqa: E402

FIXTURE = HERE / "fixtures" / "promote" / "task629_sealed_pack"
CHUNKS = str(FIXTURE / "chunks")

PACK_DECLARED = dict(
    identity_check="pack-declared",
    identity_reason="test fixture: Pool A ids are not reproducible")


def real_projected_rows() -> list[dict]:
    """The real B01 fixture, actually projected (title/lang/etc. resolved
    exactly as promote_sealed_pack.plan() would produce them), with the
    dense embed text merged back in under `canonical_text` -- `project_
    record`'s PAYLOAD carries only a 25-word extractive `snippet` (ADR-039
    minimum-content), never the full text, so a content check that needs
    the real word count (this module) must be handed the dense text
    alongside the payload, the same way `promote_queue.py`'s real
    integration will."""
    raw = list(PSP.read_records(CHUNKS))
    family_payload = {"source_type": "docs", "trust_tier": 0.6}
    joined, _skipped, _modes, _vk, _undated = PSP.project_rows(
        iter(raw), consent=None, attribution=None, license_version=None,
        family_payload=family_payload, title_from=("source_metadata.title",),
        **PACK_DECLARED)
    return [{**payload, "point_id": pid, "canonical_text": dense_text}
           for pid, (dense_text, _s, payload, _rec) in joined.items()]


def test_a_clean_real_pack_passes():
    rows = real_projected_rows()
    assert len(rows) == 3
    v = M.evaluate(rows, detect_language=False)  # lang-gate unavailable here
    assert v["verdict"] == "PASS"
    assert v["failing_rows"] == 0
    assert "coverage: no l1_sources declared" in v["skipped_checks"][0]


def test_a_row_missing_title_is_named_and_dropped_below_threshold():
    rows = real_projected_rows()
    mutated = [dict(r) for r in rows]
    mutated[0].pop("title", None)
    v = M.evaluate(mutated, detect_language=False)
    # 1 of 3 = 33% > 1% default -- refuses the whole pack.
    assert v["verdict"] == "REFUSE"
    assert v["failing_rows"] == 1
    pid = mutated[0]["point_id"]
    assert "title missing" in v["problems"][pid]


def test_below_the_fail_share_max_reports_pass_with_named_drops():
    rows = real_projected_rows() * 200  # 600 rows, 1 bad = 0.17%
    mutated = list(rows)
    bad = dict(mutated[0])
    bad.pop("title", None)
    bad["point_id"] = "synthetic-bad-row"
    mutated.append(bad)
    v = M.evaluate(mutated, detect_language=False)
    assert v["verdict"] == "PASS"
    assert v["failing_rows"] == 1
    assert "synthetic-bad-row" in v["problems"]


def test_word_floor_flags_a_real_row_shortened_below_it():
    rows = real_projected_rows()
    mutated = [dict(r) for r in rows]
    mutated[0]["canonical_text"] = "too short"
    mutated[0]["text"] = "too short"
    v = M.evaluate(mutated, detect_language=False, word_floor=15)
    pid = mutated[0]["point_id"]
    assert any("word floor" in p for p in v["problems"][pid])


# ------------------------------------------------- task 664, 2026-09-26 ruling
# SE tranche 8: 2.91% of a real 13,179-row pack was below the word floor --
# all short, genuine accepted-answer text, no other defect -- and the
# ordinary 1% `fail_share_max` rule refused the whole pack for it.
# Orchestrator ruling: a word-floor-only drop is recorded exactly like any
# other problem, but does NOT count toward `fail_share_max`; it gets its own
# 10% ceiling (WORD_FLOOR_FAIL_SHARE_MAX), because a share that high is
# itself the systemic-failure signal `fail_share_max` exists to catch.

def _shorten(row: dict) -> dict:
    row = dict(row)
    row["canonical_text"] = "too short"
    row["text"] = "too short"
    return row


def test_word_floor_alone_passes_well_past_the_ordinary_one_percent_ceiling():
    """3% below the word floor, nothing else wrong -- the real SE tranche 8
    shape. Under the OLD rule this refused (3% > 1%); under the ruling it
    passes, because 3% < the word floor's own 10% ceiling."""
    rows = real_projected_rows() * 34  # 102 rows
    mutated = list(rows)
    bad = _shorten(mutated[0])
    bad["point_id"] = "synthetic-short-row"
    mutated[1] = bad  # 1/102 = 0.98% -- comfortably matches SE's ~3% shape
                      # in kind (below floor, nothing else wrong), scaled
                      # small enough to keep the fixture cheap; the ceiling
                      # comparison below is what actually matters.
    v = M.evaluate(mutated, detect_language=False)
    assert v["verdict"] == "PASS"
    assert v["word_floor_dropped"] == 1
    assert "synthetic-short-row" in v["problems"]
    assert v["other_failing_rows"] == 0


def test_word_floor_past_ten_percent_still_refuses():
    """The ceiling this ruling adds is real, not a blanket exemption: past
    10% below the word floor is the systemic-extraction-failure signal the
    module exists to catch, and it still refuses."""
    rows = real_projected_rows() * 4  # 12 rows
    mutated = [_shorten(r) if i < 2 else dict(r)
              for i, r in enumerate(rows)]  # 2/12 = 16.7% > 10%
    v = M.evaluate(mutated, detect_language=False)
    assert v["verdict"] == "REFUSE"
    assert v["word_floor_share"] > M.WORD_FLOOR_FAIL_SHARE_MAX


def test_a_row_that_fails_the_word_floor_and_something_else_still_counts_toward_one_percent():
    """Shortness must never mask a genuine, unrelated defect: a row with
    NO title AND below the word floor is not "just" a word-floor drop."""
    rows = real_projected_rows() * 200  # 600 rows
    mutated = list(rows)
    bad = _shorten(dict(mutated[0]))
    bad.pop("title", None)
    bad["point_id"] = "synthetic-short-and-titleless"
    mutated.append(bad)  # 1/601 -- below 1%, so this alone proves nothing
                        # about the ceiling; the assertion is about WHICH
                        # bucket it lands in, not whether it refuses.
    v = M.evaluate(mutated, detect_language=False)
    assert v["word_floor_dropped"] == 0, (
        "a mixed defect must not be counted as word-floor-only")
    assert v["other_failing_rows"] == 1
    assert set(v["problems"]["synthetic-short-and-titleless"]) == {
        "title missing", "text below word floor (2 < 15)"}


def test_word_floor_share_is_recorded_even_when_it_does_not_govern_the_verdict():
    """The receipt must always show both shares (task 664: "recorded (count
    and ids in the receipt)"), not only whichever one happened to refuse."""
    rows = real_projected_rows()
    v = M.evaluate(rows, detect_language=False)
    assert v["word_floor_dropped"] == 0
    assert v["word_floor_share"] == 0.0
    assert v["word_floor_fail_share_max"] == M.WORD_FLOOR_FAIL_SHARE_MAX
    assert v["other_failing_rows"] == 0
    assert v["other_failing_share"] == 0.0


def test_language_check_is_named_skipped_when_the_gate_is_unavailable(monkeypatch):
    """This driver's default interpreter has no fasttext (measured
    2026-09-23: the build box's is numpy>=2, fasttext absent) -- the check must
    be SKIPPED and NAMED, never silently treated as a pass for every row.

    Forced deterministically via monkeypatch (not ambient environment
    state): another test module in the full suite sets AW_LANGGATE_
    OPTIONAL=1 without restoring it, which degrades fetch_process.
    detect_lang() to a URL-heuristic fallback instead of raising --
    order-dependent and a real defect this test must not depend on.
    """
    from cytir.acquire import fetch_process as FP  # noqa: PLC0415
    def _raise(*_a, **_k):
        raise FP.LangGateInactive("simulated: no lang-gate in this test")
    monkeypatch.setattr(FP, "detect_lang", _raise)
    rows = real_projected_rows()
    v = M.evaluate(rows, detect_language=True)
    assert v["language_checked"] is False
    assert any("language: lang-gate unavailable" in s
              for s in v["skipped_checks"])


def test_coverage_flags_a_real_row_against_a_real_l1_word_count(tmp_path):
    rows = real_projected_rows()
    row = rows[0]
    # A real L1 "parent" with far more words than the served chunk -- the
    # exact B02 shape (747 pack words against 4,776 L1 words).
    l1 = tmp_path / "l1.jsonl.gz"
    with gzip.open(l1, "wt") as fh:
        fh.write(json.dumps({
            "url": row["url"],
            "markdown": " ".join(["word"] * 500),
        }) + "\n")
    v = M.evaluate([row], detect_language=False, l1_sources=[str(l1)])
    assert v["coverage_checked"] is True
    assert v["verdict"] == "REFUSE"
    assert any("coverage" in p for p in v["problems"][row["point_id"]])


def test_coverage_is_named_skipped_with_no_l1_sources_declared():
    rows = real_projected_rows()
    v = M.evaluate(rows, detect_language=False)
    assert v["coverage_checked"] is False
    assert any(s.startswith("coverage:") for s in v["skipped_checks"])


# --------------------------------- language scope (orchestrator ruling,
# --------------------------------- 2026-09-23, CC tranche 1, task 641)
def _stub_detect(monkeypatch, mapping):
    """`mapping`: url substring or exact text -> (lang, confident)."""
    from cytir.acquire import fetch_process as FP  # noqa: PLC0415

    def fake(text, url, default):
        for key, val in mapping.items():
            if key in (url or "") or key in (text or ""):
                return val
        return (default, False)
    monkeypatch.setattr(FP, "detect_lang", fake)


def test_classify_language_pass_when_detected_is_in_scope_and_matches(monkeypatch):
    _stub_detect(monkeypatch, {"sunfounder": ("en", True)})
    v = M.classify_language("some real english text here", "https://sunfounder.com/x", "en")
    assert v["status"] == "pass"
    assert v["detected"] == "en"


def test_classify_language_out_of_scope_when_no_declared_and_detected_outside_set(monkeypatch):
    """The real CC tranche 1 shape: `lang` field is empty/missing, and the
    detector confidently finds a non-EN/DE/FR/IT language."""
    _stub_detect(monkeypatch, {"example.es": ("es", True)})
    v = M.classify_language("un texto real en espanol aqui", "https://example.es/x", None)
    assert v["status"] == "out_of_scope"
    assert v["detected"] == "es"


def test_classify_language_out_of_scope_even_when_declared_agrees(monkeypatch):
    """Declared 'es' and detected 'es' agree -- no CONTRADICTION -- but 'es'
    is still outside the product's language set, so this is scope routing,
    not a pass."""
    _stub_detect(monkeypatch, {"example.es": ("es", True)})
    v = M.classify_language("un texto real en espanol aqui", "https://example.es/x", "es")
    assert v["status"] == "out_of_scope"


def test_classify_language_contradiction_takes_priority_over_scope(monkeypatch):
    """Declared 'fr' but confidently detected 'de' -- a genuine metadata
    contradiction, checked BEFORE the out-of-scope routing (both languages
    here are in-scope, but the point is the contradiction test runs first
    regardless)."""
    _stub_detect(monkeypatch, {"example.ch": ("de", True)})
    v = M.classify_language("echter deutscher text hier", "https://example.ch/x", "fr")
    assert v["status"] == "contradiction"
    assert v["detected"] == "de"


def test_classify_language_detection_failed_on_low_confidence(monkeypatch):
    _stub_detect(monkeypatch, {"example.com": ("en", False)})
    v = M.classify_language("too short", "https://example.com/x", None)
    assert v["status"] == "detection_failed"


def test_classify_language_no_text_is_named_separately():
    v = M.classify_language("", "https://x", "en")
    assert v["status"] == "no_text"


def test_classify_language_gate_unavailable_is_named_separately(monkeypatch):
    from cytir.acquire import fetch_process as FP  # noqa: PLC0415

    def _raise(*_a, **_k):
        raise FP.LangGateInactive("simulated")
    monkeypatch.setattr(FP, "detect_lang", _raise)
    v = M.classify_language("real text here", "https://x", "en")
    assert v["status"] == "gate_unavailable"


def test_language_scope_filter_drops_only_out_of_scope_real_shape(monkeypatch):
    """The exact CC tranche 1 row shape: raw pack rows with `point_id`,
    `url`, `text`, `lang` (lang empty for the out-of-scope one -- measured
    live: 14,660/204,736 rows, zero of them populated-but-out-of-scope)."""
    _stub_detect(monkeypatch, {
        "swiss-health.ch": ("de", True),
        "some-spanish-site.es": ("es", True),
    })
    rows = [
        {"point_id": "a1", "url": "https://swiss-health.ch/x",
         "text": "echter deutscher gesundheitstext", "lang": "de"},
        {"point_id": "a2", "url": "https://some-spanish-site.es/x",
         "text": "contenido real en espanol", "lang": None},
    ]
    report = M.language_scope_filter(rows)
    assert report["out_of_scope_ids"] == {"a2"}
    assert report["dropped_by_language"] == {"es": 1}
    assert report["rows_checked"] == 2


def test_language_scope_filter_does_not_drop_a_contradiction_row(monkeypatch):
    """A declared/detected contradiction stays IN the population -- the
    filter only removes true out-of-scope rows; the contradiction is
    evaluate()'s job to catch, on the projected rows."""
    _stub_detect(monkeypatch, {"example.ch": ("de", True)})
    rows = [{"point_id": "b1", "url": "https://example.ch/x",
            "text": "echter deutscher text", "lang": "fr"}]
    report = M.language_scope_filter(rows)
    assert report["out_of_scope_ids"] == set()


def test_evaluate_scores_detection_failed_as_a_defect(monkeypatch):
    """New defect category, orchestrator ruling 2026-09-23: a confident-less
    detection on non-empty text now counts toward the 1% budget."""
    _stub_detect(monkeypatch, {"sunfounder": ("en", False)})
    rows = real_projected_rows()
    v = M.evaluate(rows, detect_language=True)
    assert v["verdict"] == "REFUSE"
    for pid in [r["point_id"] for r in rows]:
        assert any("could not confidently classify" in p
                  for p in v["problems"][pid])


def test_evaluate_does_not_penalize_an_out_of_scope_language_if_one_slips_through(
        monkeypatch):
    """Defense in depth: if a row detected as out-of-scope somehow reaches
    evaluate() (the scope filter is a separate, earlier pass), it must not
    be scored as a defect either -- scope routing is not a content check."""
    _stub_detect(monkeypatch, {"sunfounder": ("es", True)})
    rows = real_projected_rows()
    for r in rows:
        r["lang"] = None
    v = M.evaluate(rows, detect_language=True)
    assert v["verdict"] == "PASS"
    assert v["problems"] == {}


# --------------------------------------- lang backfill (orchestrator ruling,
# --------------------------------------- 2026-09-23, CC tranche 1, task 652)
def test_backfill_lang_fills_a_confident_in_scope_empty_row(monkeypatch):
    """Real CC tranche 1 shape: `lang` empty, detector confidently finds an
    in-scope language -- filled, with provenance recorded."""
    _stub_detect(monkeypatch, {"swiss-health.ch": ("de", True)})
    rows = [{"point_id": "a1", "url": "https://swiss-health.ch/x",
            "text": "echter deutscher gesundheitstext", "lang": None}]
    out, report = M.backfill_lang(rows)
    assert out[0]["lang"] == "de"
    assert out[0]["lang_source"] == M.lang_source_tag()
    assert out[0]["lang_source"].startswith("detected:fasttext-lid176@")
    assert report["filled"] == 1
    assert report["already_declared"] == 0


def test_backfill_lang_never_touches_a_declared_row(monkeypatch):
    """Even when the detector would say something different, a row that
    already declares a `lang` is never called into the detector at all --
    a declared/detected disagreement stays `classify_language`'s
    `contradiction` to catch elsewhere, never silently resolved here."""
    calls = []
    from cytir.acquire import fetch_process as FP  # noqa: PLC0415

    def fake(text, url, default):
        calls.append(url)
        return ("de", True)
    monkeypatch.setattr(FP, "detect_lang", fake)
    rows = [{"point_id": "a1", "url": "https://example.fr/x",
            "text": "un texte francais reel ici", "lang": "fr"}]
    out, report = M.backfill_lang(rows)
    assert out[0]["lang"] == "fr"
    assert "lang_source" not in out[0]
    assert calls == []  # detector never invoked for a declared row
    assert report["already_declared"] == 1
    assert report["filled"] == 0


def test_backfill_lang_leaves_out_of_scope_rows_alone_for_the_scope_filter(
        monkeypatch):
    """An out-of-scope row is NOT dropped or filled here -- it stays empty,
    for `language_scope_filter` (run separately) to drop."""
    _stub_detect(monkeypatch, {"some-spanish-site.es": ("es", True)})
    rows = [{"point_id": "a2", "url": "https://some-spanish-site.es/x",
            "text": "contenido real en espanol", "lang": None}]
    out, report = M.backfill_lang(rows)
    assert out[0]["lang"] is None
    assert "lang_source" not in out[0]
    assert report["left_out_of_scope"] == 1
    assert report["filled"] == 0


def test_backfill_lang_leaves_detection_failed_rows_alone(monkeypatch):
    _stub_detect(monkeypatch, {"example.com": ("en", False)})
    rows = [{"point_id": "a3", "url": "https://example.com/x",
            "text": "too short a fragment", "lang": None}]
    out, report = M.backfill_lang(rows)
    assert out[0]["lang"] is None
    assert report["left_detection_failed"] == 1


def test_backfill_lang_is_named_when_the_gate_is_unavailable(monkeypatch):
    from cytir.acquire import fetch_process as FP  # noqa: PLC0415
    def _raise(*_a, **_k):
        raise FP.LangGateInactive("simulated: no lang-gate in this test")
    monkeypatch.setattr(FP, "detect_lang", _raise)
    rows = [{"point_id": "a4", "url": "https://example.com/x",
            "text": "some real text here", "lang": None}]
    out, report = M.backfill_lang(rows)
    assert out[0]["lang"] is None
    assert report["left_gate_unavailable"] == 1


def test_backfill_lang_on_real_projected_rows_end_to_end(monkeypatch):
    """The real B01 fixture, with `lang` stripped -- proves the whole path
    (classify_language -> backfill_lang) against real record shapes, not
    just hand-built dicts."""
    _stub_detect(monkeypatch, {"sunfounder": ("en", True)})
    rows = real_projected_rows()
    for r in rows:
        r["lang"] = None
    out, report = M.backfill_lang(rows)
    assert report["filled"] == len(rows)
    assert all(r["lang"] == "en" for r in out)
    assert all(r["lang_source"] == M.lang_source_tag() for r in out)


# ----------------------------------------------- lang_drop_detail (Tim's
# ----------------------------------------------- ruling, 2026-09-24, task 652)
def _stub_detect_scored(monkeypatch, mapping):
    """`mapping`: url substring -> dict (as `fetch_process.detect_lang_scored`
    returns). Stubs the SCORED function, distinct from `_stub_detect`'s
    `FP.detect_lang` stub -- `lang_drop_detail` calls `_detect_lang_scored`,
    never `classify_language`'s `_detect_lang`."""
    from cytir.acquire import fetch_process as FP  # noqa: PLC0415

    def fake(text, url, default):
        for key, val in mapping.items():
            if key in (url or ""):
                return val
        return {"lang": default, "confident": False, "raw_label": None,
               "raw_prob": None, "text_chars": len(text or "")}
    monkeypatch.setattr(FP, "detect_lang_scored", fake)


def test_lang_drop_detail_skips_a_declared_row_without_calling_the_detector(
        monkeypatch):
    calls = []
    from cytir.acquire import fetch_process as FP  # noqa: PLC0415
    monkeypatch.setattr(FP, "detect_lang_scored",
                        lambda *a, **k: calls.append(1) or {})
    rows = [{"point_id": "a1", "url": "https://x.de", "text": "text",
            "lang": "de"}]
    detail = M.lang_drop_detail(rows)
    assert detail == []
    assert calls == []


def test_lang_drop_detail_reports_score_for_a_low_confidence_row(monkeypatch):
    """The real e-lir.ch (Romansh) shape from task 652's investigation:
    empty declared lang, non-empty text, low-confidence detection."""
    _stub_detect_scored(monkeypatch, {"e-lir.ch": {
        "lang": "en", "confident": False, "raw_label": "rm",
        "raw_prob": 0.42, "text_chars": 300}})
    rows = [{"point_id": "b1", "url": "http://e-lir.ch/?id=1333",
            "text": "x" * 300, "lang": None}]
    detail = M.lang_drop_detail(rows)
    assert len(detail) == 1
    entry = detail[0]
    assert entry["point_id"] == "b1"
    assert entry["url"] == "http://e-lir.ch/?id=1333"
    assert entry["text_chars"] == 300
    assert entry["confident"] is False
    assert entry["raw_label"] == "rm"
    assert entry["raw_prob"] == 0.42
    assert entry["gate_unavailable"] is False


def test_lang_drop_detail_names_gate_unavailable_rows_separately(monkeypatch):
    from cytir.acquire import fetch_process as FP  # noqa: PLC0415
    def _raise(*_a, **_k):
        raise FP.LangGateInactive("simulated")
    monkeypatch.setattr(FP, "detect_lang_scored", _raise)
    rows = [{"point_id": "c1", "url": "https://x", "text": "some text",
            "lang": None}]
    detail = M.lang_drop_detail(rows)
    assert detail[0]["gate_unavailable"] is True
    assert detail[0]["raw_prob"] is None


def test_lang_drop_detail_on_real_no_lang_rows(monkeypatch):
    """Real B01 fixture rows, `lang` stripped, run through the whole
    dropped-row accounting path end to end."""
    _stub_detect_scored(monkeypatch, {"sunfounder": {
        "lang": "en", "confident": False, "raw_label": "en",
        "raw_prob": 0.6, "text_chars": 999}})
    rows = real_projected_rows()
    for r in rows:
        r["lang"] = None
    detail = M.lang_drop_detail(rows, text_field="canonical_text")
    assert len(detail) == len(rows)
    assert all(d["raw_label"] == "en" and d["confident"] is False
              for d in detail)
    assert all(d["text_chars"] > 0 for d in detail)
