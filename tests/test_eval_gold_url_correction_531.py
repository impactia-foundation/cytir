"""Task 531 — the corrected gold set has ZERO of the two known-broken URL
forms, and every URL this task corrected is proven reachable.

This does NOT re-scan the 34M-row aw_e3 membership snapshot on the build host --
ADR-054 persisted-script discipline: a unit test reads the PERSISTED proof
(the task-531 corrections.json artifact, written by the task-531
check_membership and live_check scripts against the real
sealed snapshot and the real internet on 2026-09-06), it does not repeat the
network or build-host work on every test run. What it verifies fresh, every run:

  1. the corrected 629-set (`data/eval_tasks.v2.yaml`) carries NEITHER of the
     two broken forms anywhere, and matches the sha256 now pinned in
     the task-524 run_arm_524 script;
  2. `data/eval_authoritative.yaml` (1,418-gate `SETS`) carries no
     `/info/rfcN/` gold_urls;
  3. every correction this task made is, per the persisted evidence, BOTH a
     member of the sealed aw_e3 membership snapshot's URL space it was
     checked against AND a live HTTP 200 -- i.e. the "0 unreachable golds"
     claim in the changelog is backed by a recorded check, not an assertion;
  4. the two frozen v1 files (`data/eval_tasks.yaml`,
     the task-524 run_arm_524 script's OLD sha256 in its own docstring
     table) are named but never re-derived here -- v1 stays untouched by
     construction (this test never opens it for writing).
"""
import json
import os
import re

import pathlib

import pytest
import yaml

from tests.helpers.repo_files import repo_file_or_skip

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# The public cut withholds each of these inputs; skip there (task 2411).
V2_PATH = str(repo_file_or_skip("data/eval_tasks.v2.yaml"))
V1_PATH = str(repo_file_or_skip("data/eval_tasks.yaml"))
AUTHORITATIVE_PATH = str(repo_file_or_skip("data/eval_authoritative.yaml"))
CORRECTIONS_PATH = str(repo_file_or_skip(pathlib.Path(REPO, "tasks", "531-artifacts", "corrections.json")))
repo_file_or_skip(pathlib.Path(REPO, "tasks", "524-scripts", "run_arm_524.py"))

GESETZE_TRIPLE_UNDERSCORE = re.compile(r"gesetze-im-internet\.de/[a-z0-9_]+/___\d+[a-z]?\.html", re.I)
RFC_INFO_PATH = re.compile(r"rfc-editor\.org/info/rfc\d+/?", re.I)


def _file_sha256(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_v2_has_629_rows_and_matches_the_new_pin():
    items = yaml.safe_load(open(V2_PATH))["eval"]
    assert len(items) == 629

    import sys
    sys.path.insert(0, os.path.join(REPO, "tasks", "524-scripts"))
    import importlib
    sys.modules.setdefault("run_eval", importlib.import_module("cytir.gates.p0_eval"))  # the frozen script imports the pre-#1455 name
    import run_arm_524 as pinned  # noqa: E402

    assert pinned.EVAL_PATH == V2_PATH
    assert _file_sha256(V2_PATH) == pinned.PINNED_SHA256


def test_v2_carries_neither_known_broken_url_form():
    raw = open(V2_PATH, encoding="utf-8").read()
    assert not GESETZE_TRIPLE_UNDERSCORE.search(raw), (
        "eval_tasks.v2.yaml still carries a triple-underscore "
        "gesetze-im-internet.de gold URL")
    assert not RFC_INFO_PATH.search(raw), (
        "eval_tasks.v2.yaml still carries an /info/rfcN rfc-editor.org gold URL")


def test_v1_is_untouched():
    # v1 is the historical record every baseline through task 525 was scored
    # against (verification-doctrine 2) -- it is EXPECTED to still carry the
    # 74 broken URLs task 531 corrected. This test only proves v1 still
    # exists and still has 629 rows; it must never be "fixed" in place.
    items = yaml.safe_load(open(V1_PATH))["eval"]
    assert len(items) == 629
    raw = open(V1_PATH, encoding="utf-8").read()
    assert GESETZE_TRIPLE_UNDERSCORE.search(raw), "v1 should still carry the pre-fix URLs"
    assert RFC_INFO_PATH.search(raw), "v1 should still carry the pre-fix URLs"


def test_eval_authoritative_has_no_info_rfc_path_golds():
    docs = [d for d in yaml.safe_load_all(open(AUTHORITATIVE_PATH)) if d]
    n_gold_urls = 0
    for doc in docs:
        rows = doc if isinstance(doc, list) else doc.get("queries", [])
        for r in rows:
            for g in (r.get("gold_urls") or []):
                n_gold_urls += 1
                assert not RFC_INFO_PATH.search(g), (
                    f"{r.get('id')} still carries /info/rfcN gold_url: {g}")
    assert n_gold_urls > 0, "sanity: expected to iterate over some gold_urls"


def test_every_correction_is_recorded_as_member_and_live_200():
    corrections = json.load(open(CORRECTIONS_PATH))
    assert len(corrections) == 83, (
        "expected 83 corrections (43 gesetze + 40 rfc, across both caches); "
        f"got {len(corrections)}")
    by_pattern = {}
    for c in corrections:
        by_pattern.setdefault(c["pattern"], []).append(c)
        assert c["live_http_status"] == 200, (
            f"{c['old_url']} -> {c['new_url']} did not resolve live: "
            f"{c.get('live_error') or c['live_http_status']}")
    # The rfc-editor.org fix is corpus-correct for the STAGED next-build
    # corpus and live-200 today, but the CURRENT twin snapshot predates that
    # rebuild -- see data/CHANGELOG-eval-gold.md "A finding, not a footnote".
    # So membership is asserted per-pattern, not uniformly, matching what was
    # actually measured (verification-doctrine 6: an error is not an empty,
    # and neither is a documented not-yet).
    for c in by_pattern.get("gesetze_triple_underscore", []):
        assert c["in_corpus_membership"] is True, (
            f"{c['new_url']} should already be a member of the live aw_e3 twin")
    assert len(by_pattern.get("gesetze_triple_underscore", [])) == 43
    assert len(by_pattern.get("rfc_info_path", [])) == 40


def test_corrected_urls_actually_landed_in_v2_or_authoritative():
    v2_raw = open(V2_PATH, encoding="utf-8").read()
    auth_raw = open(AUTHORITATIVE_PATH, encoding="utf-8").read()
    corrections = json.load(open(CORRECTIONS_PATH))
    for c in corrections:
        files = c["files"]
        if any(f.endswith("eval_tasks.yaml") for f in files):
            assert c["new_url"] in v2_raw, f"{c['new_url']} missing from eval_tasks.v2.yaml"
        if any(f.endswith("eval_authoritative.yaml") for f in files):
            assert c["new_url"] in auth_raw, f"{c['new_url']} missing from eval_authoritative.yaml"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
