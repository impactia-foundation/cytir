#!/usr/bin/env python3
# Moved from the retired `mechanical_content_check` module (task 677, task 643 S5); the old path is an alias shim.
"""`mechanical-content-v1` -- ADR-110's "content guard passes", read per the
orchestrator's ruling (2026-09-23): the gold-query `cytir/gates/content_guard.py`
arms are RANKING measurement and ADR-110 moved that to the weekly battery.
The per-pack gate is a MECHANICAL, no-query, per-row content check.

Composed from EXISTING validators, per the ruling ("write nothing new unless
a check does not exist"):

* title present            -- `payload_carry_contract._conforms('title', ...)`
* text present + word floor -- presence via the same `_conforms`; the FLOOR
  itself is new (no reusable word-floor function existed).
* language matches          -- `fetch_process.detect_lang` (fasttext lid.176,
  the project's one content-based language gate, task 166/236/238). Requires
  the dedicated lang-gate venv (numpy<2 + fasttext); where that is not the
  running interpreter, this sub-check is SKIPPED and NAMED, never silently
  passed (the main venv on the build box is numpy>=2 and cannot run it -- measured
  2026-09-23, `<langgate-venv>` is the venv that can).
* no stub/boilerplate (coverage vs L1) -- the ratio
  task 629's `measure_pack_l1_coverage.py` used to find B02's 39
  fragments (mean 0.248, 37/39 below 0.50): pack words for a url / L1 words
  for that url. OPT-IN per pack (the manifest names an L1 source to compare
  against) because reading every row's L1 parent at 1M+-row scale is real
  I/O this module will not do unasked; absent a declared source this
  sub-check is SKIPPED and NAMED.
* digest verified against sealed L1     -- ALREADY enforced, unchanged, by
  `promote_sealed_pack.plan()`'s `persisted_inputs.assert_pages_persisted`
  call (this module does not re-run it; pass `check_sha=True` into that
  call for the mechanical profile, which is the one-line change that turns
  it on -- see `promote_queue.py`).
* no dup of an already-served url       -- ALREADY enforced, unchanged, by
  `promote_sealed_pack.plan()`'s `already_served_urls` refusal. Not
  re-implemented here.
* published_date present (news only)    -- ALREADY enforced, unchanged, by
  `promote_sealed_pack.plan()`'s `require_published`/`published_from`.

WHAT THIS MODULE ACTUALLY DECIDES
----------------------------------
Only the checks with no existing whole-pack enforcement elsewhere:
title, text+floor, language, coverage, and (task 767) `binary_content`: the text is
prose, not a binary file read as text (`cytir.core.prose_guard`; task 777: a counted per-row HOLD-OUT -- `held_out`/`held_out_rows`/`held_out_by_reason` -- dropped by the drain, never part of `fail_share_max`, so a guard false positive cannot refuse a pack). `evaluate()` scores every row against
each APPLICABLE check (language/coverage skip, named, when their inputs are
unavailable) and returns, per row, which checks it failed. Every failing row
is named and meant to be DROPPED by the caller before `build_points()`,
never silently promoted -- but TWO separate shares govern whether the whole
pack refuses:

> `fail_share_max` (default 0.01) governs every failure EXCEPT a row whose
ONLY problem is the word floor; `WORD_FLOOR_FAIL_SHARE_MAX` (0.10) governs
those alone. Orchestrator ruling, task 664, 2026-09-26 (SE tranche 8: 2.91%
of a real pack was below the word floor, all short-but-genuine accepted-
answer text, no other defect -- the 1% rule refused the whole pack for a
natural characteristic of the family's own content, not a systemic
extraction failure). A row that fails the word floor AND something else
still counts toward `fail_share_max` -- shortness never masks a genuine
defect. Both shares are recorded (`word_floor_share`/`other_failing_share`)
regardless of which one governs; the pack refuses if EITHER exceeds its own
ceiling.
"""
from __future__ import annotations

import gzip
import json
import pathlib
from collections import Counter
from typing import Iterable

ROOT = pathlib.Path(__file__).resolve().parents[2]

from cytir.core.prose_guard import binary_content  # noqa: E402
from cytir.index import payload_carry as PC  # noqa: E402

SCHEMA = "aw.mechanical-content-check.v1"

#: Not measured per family yet -- a conservative floor, overridable per
#: manifest (`word_floor`). Below this a row is functionally a stub even if
#: `text` is technically non-empty.
DEFAULT_WORD_FLOOR = 15
#: The coverage ratio below which task 629 found fragments (B02: 37 of 39
#: excluded rows were below this; the family's clean rows averaged 0.964).
DEFAULT_COVERAGE_FLOOR = 0.50
DEFAULT_FAIL_SHARE_MAX = 0.01
#: Orchestrator ruling, task 664, 2026-09-26 (SE tranche 8: 2.91% of a
#: 13,179-row pack was BELOW the word floor -- all short, genuine accepted-
#: answer text, no other defect -- and the 1% rule refused the whole pack
#: for it). A row below the word floor is dropped and recorded exactly like
#: any other mechanical-content problem, but it does NOT count toward the
#: 1% `fail_share_max` share: a short-but-real answer is a natural
#: characteristic of a Q&A family's own content, not the systemic
#: extraction failure `fail_share_max` exists to catch (a stub page, a
#: paywall wall, a fetch that silently returned nothing). Word-floor drops
#: get their OWN, separate ceiling: a pack refuses on this reason alone
#: only past 10% -- a share that high is itself the systemic-failure
#: signal (bulk truncation, a chunker regression), not natural variance.
#: NOT manifest-overridable (orchestrator's explicit ruling: no per-pack
#: `fail_share_max`-style escape hatch for this one) -- unlike every other
#: threshold in this module, which stays a deliberate, named per-pack
#: override.
WORD_FLOOR_FAIL_SHARE_MAX = 0.10


class LangGateUnavailable(RuntimeError):
    """The fasttext lang-gate could not be loaded in THIS interpreter."""


def _word_count(text: str | None) -> int:
    return len((text or "").split())


def _detect_lang(text: str, url: str, default: str) -> tuple[str, bool] | None:
    """Wraps `fetch_process.detect_lang`; returns None (not a guess) when the
    lang-gate cannot run in this interpreter, rather than crashing the whole
    evaluate() or silently trusting a URL-heuristic default as "detected"."""
    try:
        from cytir.acquire import fetch_process as FP  # noqa: PLC0415
    except Exception:  # noqa: BLE001 -- any import-time failure, not just ImportError
        return None
    try:
        return FP.detect_lang(text, url, default)
    except Exception:  # noqa: BLE001 -- FP.LangGateInactive or a runtime failure
        return None


def _detect_lang_scored(text: str, url: str, default: str) -> dict | None:
    """Wraps `fetch_process.detect_lang_scored` the same way `_detect_lang`
    wraps `detect_lang` -- returns None (not a guess) when the gate cannot
    run here. Kept SEPARATE from `_detect_lang`/`classify_language`'s own
    call (rather than switching that call to the scored variant) so every
    existing test that monkeypatches `fetch_process.detect_lang` keeps
    exercising exactly the path it patches; only `lang_drop_detail` (task
    652: which score a dropped row actually got) uses this."""
    try:
        from cytir.acquire import fetch_process as FP  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    try:
        return FP.detect_lang_scored(text, url, default)
    except Exception:  # noqa: BLE001
        return None


#: The product's language set (CLAUDE.md: "Multilingual (EN/DE/FR/IT)").
#: Orchestrator's ruling, 2026-09-23 (CC tranche 1, task 641): for a
#: MULTI-LANGUAGE family, "language matches" means the DETECTED language is
#: in this set -- not that it equals a declared per-row `lang`. A row
#: detected outside it is out of the product's scope, not a content defect.
IN_SCOPE_LANGS = frozenset({"en", "de", "fr", "it"})


def classify_language(text: str, url: str, declared: str | None, *,
                      detect_language: bool = True) -> dict:
    """One row's language verdict -- never raises, every branch is a named
    status, because two different callers need to tell them apart: the
    pre-`plan()` scope filter (drop `out_of_scope`, touch nothing else) and
    `evaluate()`'s defect scoring (`contradiction`/`detection_failed` count
    toward the 1% budget, `out_of_scope`/`pass` do not).

    The two DEFECT statuses, per the ruling, are narrow by design:
    `contradiction` -- a CONFIDENT detection that disagrees with a present
    declared `lang` (the row's own metadata is wrong, regardless of whether
    either language is in scope). `detection_failed` -- non-empty text the
    detector could not confidently classify at all (`fetch_process.
    detect_lang`'s own `confident` flag, set False rather than raised when
    its lid.176 top-1 probability falls under threshold -- that function's
    own docstring calls this "never drops on a low-confidence guess" for
    ITS purposes; this mechanical check's job is different, and the
    orchestrator ruled it a defect here).
    """
    if not text or not text.strip():
        return {"status": "no_text", "detected": None, "confident": None,
               "declared": declared}
    if not detect_language:
        return {"status": "skipped", "detected": None, "confident": None,
               "declared": declared}
    result = _detect_lang(text, url or "", declared or "en")
    if result is None:
        return {"status": "gate_unavailable", "detected": None,
               "confident": None, "declared": declared}
    det_lang, confident = result
    if not confident:
        return {"status": "detection_failed", "detected": det_lang,
               "confident": False, "declared": declared}
    if declared and det_lang != declared:
        return {"status": "contradiction", "detected": det_lang,
               "confident": True, "declared": declared}
    if det_lang not in IN_SCOPE_LANGS:
        return {"status": "out_of_scope", "detected": det_lang,
               "confident": True, "declared": declared}
    return {"status": "pass", "detected": det_lang, "confident": True,
           "declared": declared}


def language_scope_filter(rows: Iterable[dict], *, id_field: str = "point_id",
                         url_field: str = "url", text_field: str = "text",
                         lang_field: str = "lang",
                         detect_language: bool = True) -> dict:
    """Pre-`plan()` pass over a pack's RAW (pre-projection) rows: which ids
    are OUT OF SCOPE for the product's language set and must be excluded
    BEFORE a population-level gate -- `payload_carry_contract`'s lang
    PRESENCE floor among them -- ever scores them. A row missing its
    declared `lang` is exactly the shape a source classifier leaves when it
    would not confidently label a non-target-language page (measured live,
    CC tranche 1, 2026-09-23: 14,660 of 204,736 rows with an empty `lang`
    field, zero rows with a populated-but-out-of-scope one).

    Deliberately narrow: only drops `out_of_scope`. `contradiction` and
    `detection_failed` rows are left IN the population -- they surface as
    defects in `evaluate()` on the PROJECTED rows this filter's caller
    passes through `plan()` next, which is where the 1% budget is judged.
    """
    out_of_scope_ids: set[str] = set()
    dropped_by_language: dict[str, int] = {}
    checked = 0
    gate_unavailable = 0
    for row in rows:
        rid = str(row.get(id_field) or row.get(url_field) or "")
        verdict = classify_language(
            row.get(text_field) or "", row.get(url_field) or "",
            row.get(lang_field), detect_language=detect_language)
        if verdict["status"] == "gate_unavailable":
            gate_unavailable += 1
            continue
        if verdict["status"] in ("no_text", "skipped"):
            continue
        checked += 1
        if verdict["status"] == "out_of_scope":
            out_of_scope_ids.add(rid)
            dropped_by_language[verdict["detected"]] = (
                dropped_by_language.get(verdict["detected"], 0) + 1)
    return {
        "rows_checked": checked,
        "out_of_scope_ids": out_of_scope_ids,
        "dropped_by_language": dropped_by_language,
        "gate_unavailable_rows": gate_unavailable,
        "in_scope_langs": sorted(IN_SCOPE_LANGS),
    }


#: Detector name pinned alongside the sha256 in `model_assets.LID_176` --
#: this is the ONE identity string a `lang_source` provenance value may cite,
#: so a later reader can tell "this label came from the tracked detector"
#: from "this label was declared by the source" without re-deriving it.
_DETECTOR_NAME = "fasttext-lid176"


def lang_source_tag() -> str:
    """`detected:<detector>@<version>` -- orchestrator ruling 2026-09-23 (CC
    tranche 1, task 652, reading ADR-110's content check). `<version>` is the
    pinned model's sha256 prefix (`model_assets.LID_176.sha256`), the same
    identity `fetch_process.detect_lang` verifies before it will predict --
    never a package version number nothing else in the repo tracks."""
    from cytir.embed.model_assets import LID_176  # noqa: PLC0415
    return f"detected:{_DETECTOR_NAME}@{LID_176.sha256[:12]}"


def backfill_lang(rows: Iterable[dict], *, url_field: str = "url",
                  text_field: str = "canonical_text",
                  text_fallback_field: str = "text",
                  lang_field: str = "lang",
                  lang_source_field: str = "lang_source") -> tuple[list[dict], dict]:
    """Fill an EMPTY `lang` field from the tracked language detector, when
    detection is confident and the result is in the product's language set
    (orchestrator ruling 2026-09-23, CC tranche 1, task 652, reading
    ADR-110's content check).

    A row's own DECLARED `lang` is never touched -- this function does not
    even call the detector for a row that already carries one, so a
    disagreement between a declared label and the detector (the
    `contradiction` status `classify_language` names) can never be
    silently resolved by this path. A filled row records its provenance in
    `lang_source_field` (`lang_source_tag()`) so nothing downstream can
    mistake a detected label for a declared one.

    Out-of-scope rows are NOT dropped here -- that stays
    `language_scope_filter`'s job, run separately over the same population
    (task 641/#327); this function only ever adds a `lang`/`lang_source`
    pair, never removes a row, so the two passes can be composed or run
    independently without one's behaviour depending on the other having
    already run.

    Mutates and returns the SAME row dicts (matches
    `normalize_sealed_pack_shape.merge`'s in-memory shape); `report` counts
    every branch by name so a caller's receipt can state what happened to
    every one of the empty-`lang` rows, not just how many were filled.
    """
    rows = list(rows)
    filled = 0
    left_out_of_scope = 0
    left_detection_failed = 0
    left_gate_unavailable = 0
    left_no_text = 0
    already_declared = 0
    tag = lang_source_tag()
    for row in rows:
        if row.get(lang_field):
            already_declared += 1
            continue
        text = row.get(text_field) or row.get(text_fallback_field) or ""
        verdict = classify_language(text, row.get(url_field) or "", None,
                                    detect_language=True)
        status = verdict["status"]
        if status == "pass":
            row[lang_field] = verdict["detected"]
            row[lang_source_field] = tag
            filled += 1
        elif status == "out_of_scope":
            left_out_of_scope += 1
        elif status == "gate_unavailable":
            left_gate_unavailable += 1
        elif status == "detection_failed":
            left_detection_failed += 1
        else:  # "no_text" -- nothing to detect from
            left_no_text += 1
    report = {
        "rows": len(rows),
        "already_declared": already_declared,
        "filled": filled,
        "left_out_of_scope": left_out_of_scope,
        "left_detection_failed": left_detection_failed,
        "left_gate_unavailable": left_gate_unavailable,
        "left_no_text": left_no_text,
        "lang_source_tag": tag,
    }
    return rows, report


def lang_drop_detail(rows: Iterable[dict], *, url_field: str = "url",
                     text_field: str = "text", lang_field: str = "lang",
                     id_field: str = "point_id") -> list[dict]:
    """Per-row detail for every row with an EMPTY `lang` -- id, url, text
    length and the detector's raw score, even for a row the detector could
    not confidently classify (`_detect_lang_scored`, which -- unlike
    `classify_language` -- never collapses a low-confidence result into
    just `confident: False`).

    Built for the CC tranche 1 lang-presence waiver (Tim, 2026-09-24:
    "record them (ids, urls, text length, detector score)" in the
    promotion receipt) and reused, unchanged, for the investigation into
    WHY detection failed -- one implementation, not a receipt copy and an
    investigation copy that can silently drift apart. A row that already
    declares a `lang` is skipped without ever calling the detector, same
    discipline as `backfill_lang`.
    """
    out: list[dict] = []
    for row in rows:
        if row.get(lang_field):
            continue
        text = row.get(text_field) or ""
        pid = str(row.get(id_field) or row.get(url_field) or "")
        entry = {"point_id": pid, "url": row.get(url_field),
                "text_chars": len(text)}
        scored = _detect_lang_scored(text, row.get(url_field) or "", "en")
        if scored is None:
            entry.update(detected=None, confident=None, raw_label=None,
                        raw_prob=None, gate_unavailable=True)
        else:
            entry.update(detected=scored["lang"], confident=scored["confident"],
                        raw_label=scored["raw_label"], raw_prob=scored["raw_prob"],
                        gate_unavailable=False)
        out.append(entry)
    return out


def _l1_word_counts(l1_path: str) -> dict[str, int]:
    """`url -> word count` from one L1 markdown shard (the same field-read
    `measure_pack_l1_coverage.py` used: `markdown` or `text`, keyed by `url`
    or `canonical_url`)."""
    out: dict[str, int] = {}
    opener = gzip.open if l1_path.endswith(".gz") else open
    with opener(l1_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row.get("url") or row.get("canonical_url")
            if key:
                out[key] = _word_count(row.get("markdown") or row.get("text"))
    return out


def evaluate(rows: Iterable[dict], *, url_field: str = "url",
            text_field: str = "canonical_text",
            text_fallback_field: str = "text",
            lang_field: str = "lang",
            word_floor: int = DEFAULT_WORD_FLOOR,
            coverage_floor: float = DEFAULT_COVERAGE_FLOOR,
            fail_share_max: float = DEFAULT_FAIL_SHARE_MAX,
            l1_sources: list[str] | None = None,
            detect_language: bool = True) -> dict:
    """Score every row; return the verdict + per-row problems.

    `rows` are the PROJECTED payload dicts (what `promote_sealed_pack.
    project_rows` produces), not the raw pack rows -- so `url`/`lang`/
    `title` read the same fields the rest of the pipeline already resolved.
    """
    rows = list(rows)
    n = len(rows)
    skipped_checks: list[str] = []

    coverage: dict[str, int] = {}
    if l1_sources:
        for src in l1_sources:
            coverage.update(_l1_word_counts(src))
    else:
        skipped_checks.append(
            "coverage: no l1_sources declared in the manifest -- SKIPPED, "
            "not scored as a pass")

    lang_checked_any = False
    problems: dict[str, list[str]] = {}
    #: Rows whose ONLY problem is the word floor (task 664, 2026-09-26 --
    #: see WORD_FLOOR_FAIL_SHARE_MAX). A row that ALSO fails some other
    #: check is not "only" the word floor and stays in the ordinary
    #: `fail_share_max` bucket -- shortness never masks a genuine defect.
    word_floor_only: dict[str, list[str]] = {}
    #: Rows that are a binary file read as text (task 777; ADR-094: quarantine the row, never
    #: abort the pack). A counted per-row hold-out: dropped by the drain like a failing row, but
    #: outside `fail_share_max` -- a guard false positive must not refuse a whole family pack.
    held_out: dict[str, str] = {}
    for row in rows:
        pid = str(row.get("point_id") or row.get("url") or id(row))
        fails: list[str] = []
        is_word_floor_fail = False
        text = row.get(text_field) or row.get(text_fallback_field) or ""
        binary = binary_content(text) if text.strip() else None   # task 767: binary read as text
        if binary:
            held_out[pid] = binary
            continue
        if not PC._conforms("title", row):
            fails.append("title missing")
        if not text.strip():
            fails.append("text missing")
        else:
            wc = _word_count(text)
            if wc < word_floor:
                fails.append(f"text below word floor ({wc} < {word_floor})")
                is_word_floor_fail = True
        if detect_language and text.strip():
            declared = row.get(lang_field)
            verdict = classify_language(text, row.get(url_field) or "",
                                        declared, detect_language=True)
            if verdict["status"] == "gate_unavailable":
                if "language: lang-gate unavailable in this interpreter " \
                   "-- SKIPPED, not scored as a pass" not in skipped_checks:
                    skipped_checks.append(
                        "language: lang-gate unavailable in this "
                        "interpreter -- SKIPPED, not scored as a pass")
            else:
                lang_checked_any = True
                if verdict["status"] == "contradiction":
                    fails.append(
                        f"language mismatch (declared {declared!r}, "
                        f"detected {verdict['detected']!r})")
                elif verdict["status"] == "detection_failed":
                    fails.append(
                        "language detection could not confidently classify "
                        "non-empty text (orchestrator ruling 2026-09-23)")
                # "pass" and "out_of_scope" are both non-defects here --
                # out-of-scope rows are expected to already be gone by this
                # point (language_scope_filter runs before plan()); one
                # slipping through is scope routing, not a content defect.
        if coverage:
            u = row.get(url_field)
            l1w = coverage.get(u)
            if l1w:
                ratio = _word_count(text) / l1w
                if ratio < coverage_floor:
                    fails.append(
                        f"coverage {ratio:.3f} < {coverage_floor} vs L1 "
                        f"({l1w} words)")
        if fails:
            problems[pid] = fails
            if is_word_floor_fail and len(fails) == 1:
                # ONLY the word floor -- a genuinely short-but-real row, not
                # a mixed defect. Still recorded and dropped via `problems`
                # (unchanged); just excluded from the ordinary
                # `fail_share_max` bucket (task 664, WORD_FLOOR_FAIL_SHARE_MAX).
                word_floor_only[pid] = fails

    failing = len(problems)
    word_floor_dropped = len(word_floor_only)
    other_failing = failing - word_floor_dropped
    share = (failing / n) if n else 0.0
    other_share = (other_failing / n) if n else 0.0
    word_floor_share = (word_floor_dropped / n) if n else 0.0
    held_by_reason = dict(Counter(held_out.values()))
    verdict = ("REFUSE" if (other_share > fail_share_max
                           or word_floor_share > WORD_FLOOR_FAIL_SHARE_MAX
                           or (n and len(held_out) == n))   # every row is binary: nothing to apply
              else "PASS")
    out = {
        "schema": SCHEMA,
        "rows_scored": n,
        "failing_rows": failing,
        "failing_share": round(share, 4),
        "fail_share_max": fail_share_max,
        "other_failing_rows": other_failing,
        "other_failing_share": round(other_share, 4),
        "word_floor_dropped": word_floor_dropped,
        "word_floor_share": round(word_floor_share, 4),
        "word_floor_fail_share_max": WORD_FLOOR_FAIL_SHARE_MAX,
        "verdict": verdict,
        "problems": problems,
        "skipped_checks": skipped_checks,
        "language_checked": lang_checked_any,
        "coverage_checked": bool(coverage),
        "checks_composed_from": {
            "title": "payload_carry_contract._conforms",
            "text_presence": "payload_carry_contract._conforms (word floor is new)",
            "language": "fetch_process.detect_lang",
            "coverage": "the task-629 pack-L1 coverage measure's ratio "
                       "(pack words / L1 words per url)",
            "digest_vs_l1": "promote_sealed_pack.plan() -> persisted_inputs."
                            "assert_pages_persisted(check_sha=True) -- not "
                            "re-run here",
            "dup_of_served_url": "promote_sealed_pack.plan() -> "
                                 "already_served_urls -- not re-run here",
            "published_date": "promote_sealed_pack.plan() -> require_published"
                              "/published_from -- not re-run here",
        },
    }
    if held_out:        # task 777: absent when nothing is held, so a clean pack's receipt is unchanged
        out.update(held_out_rows=len(held_out), held_out_by_reason=held_by_reason,
                   held_out=held_out)
    return out
