"""Task 542 — closes PR #94/task 531's "found, not fixed" item: 7
gesetze-im-internet.de Grundgesetz golds (`de35`-`de41`) carried a
capitalized `gg/Art_N.html` path while the corpus and live site use
lowercase `gg/art_N.html`. `run_eval.norm` (the p0_eval module) does not fold path
case, so these missed under any ranking/content change until now.

Same discipline as `tests/test_eval_gold_url_correction_531.py`
(ADR-054 persisted-script discipline): this does NOT re-hit the build host or
re-scan the aw_e3 membership snapshot on every test run. It reads the
PERSISTED evidence this task's scripts wrote against the real internet and
the real sealed snapshot on 2026-09-08
(the task-542 live_check_results.json and membership-check-summary.json
artifacts), and verifies fresh,
every run:

  1. `data/eval_tasks.v2.yaml` carries the corrected lowercase form for all
     7 ids and carries NONE of the old capitalized forms anywhere;
  2. v2's sha256 matches the pin now in the task-524 run_arm_524 script
     (the one place that pins the 629-set's sha for the twin-control
     runner);
  3. `data/eval_tasks.yaml` (v1) is untouched -- still 629 rows, still
     carries the old capitalized forms (verification-doctrine 2: the
     yardstick is immutable);
  4. every one of the 7 corrections is recorded, in the persisted
     evidence, as both live-200 (new form) / 404 (old form) and a member
     of the sealed aw_e3 snapshot (new form) / not a member (old form).
"""
import hashlib
import json
import os
import pathlib
import sys

import pytest
import yaml

from tests.helpers.repo_files import repo_file_or_skip

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# The public cut withholds each of these inputs; skip there (task 2411).
V2_PATH = str(repo_file_or_skip("data/eval_tasks.v2.yaml"))
V1_PATH = str(repo_file_or_skip("data/eval_tasks.yaml"))
LIVE_CHECK_PATH = str(repo_file_or_skip(pathlib.Path(REPO, "tasks", "542-artifacts", "live_check_results.json")))
MEMBERSHIP_PATH = str(repo_file_or_skip(pathlib.Path(REPO, "tasks", "542-artifacts", "membership-check-summary.json")))
repo_file_or_skip(pathlib.Path(REPO, "tasks", "524-scripts", "run_arm_524.py"))

CORRECTED_IDS = ["de35", "de36", "de37", "de38", "de39", "de40", "de41"]

OLD_URLS = {
    "de35": "https://www.gesetze-im-internet.de/gg/Art_1.html",
    "de36": "https://www.gesetze-im-internet.de/gg/Art_2.html",
    "de37": "https://www.gesetze-im-internet.de/gg/Art_3.html",
    "de38": "https://www.gesetze-im-internet.de/gg/Art_5.html",
    "de39": "https://www.gesetze-im-internet.de/gg/Art_12.html",
    "de40": "https://www.gesetze-im-internet.de/gg/Art_14.html",
    "de41": "https://www.gesetze-im-internet.de/gg/Art_20.html",
}
NEW_URLS = {
    "de35": "https://www.gesetze-im-internet.de/gg/art_1.html",
    "de36": "https://www.gesetze-im-internet.de/gg/art_2.html",
    "de37": "https://www.gesetze-im-internet.de/gg/art_3.html",
    "de38": "https://www.gesetze-im-internet.de/gg/art_5.html",
    "de39": "https://www.gesetze-im-internet.de/gg/art_12.html",
    "de40": "https://www.gesetze-im-internet.de/gg/art_14.html",
    "de41": "https://www.gesetze-im-internet.de/gg/art_20.html",
}


def _file_sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _v2_rows():
    return yaml.safe_load(open(V2_PATH))["eval"]


def test_v2_has_629_rows_and_matches_the_new_pin():
    items = _v2_rows()
    assert len(items) == 629

    sys.path.insert(0, os.path.join(REPO, "tasks", "524-scripts"))
    # The frozen task-524 script does `from run_eval import norm, hit_rank`; that module
    # is `cytir/gates/p0_eval.py` since the S10 port (task 1874). Exploration scripts are
    # never edited, so the test supplies the old name (landing batch, 2026-10-31).
    import importlib
    sys.modules.setdefault("run_eval", importlib.import_module("cytir.gates.p0_eval"))
    import run_arm_524 as pinned  # noqa: E402

    assert pinned.EVAL_PATH == V2_PATH
    assert _file_sha256(V2_PATH) == pinned.PINNED_SHA256


def test_v2_carries_the_corrected_lowercase_forms():
    by_id = {r["id"]: r for r in _v2_rows()}
    for gid in CORRECTED_IDS:
        assert gid in by_id, f"{gid} missing from eval_tasks.v2.yaml"
        assert by_id[gid]["target_url"] == NEW_URLS[gid], (
            f"{gid} target_url is {by_id[gid]['target_url']!r}, "
            f"expected {NEW_URLS[gid]!r}")


def test_v2_carries_none_of_the_old_capitalized_forms():
    raw = open(V2_PATH, encoding="utf-8").read()
    for gid, old in OLD_URLS.items():
        assert old not in raw, (
            f"eval_tasks.v2.yaml still carries the old capitalized form "
            f"for {gid}: {old}")


def test_v1_is_untouched():
    # v1 is the historical record every baseline through task 531 was scored
    # against -- it is EXPECTED to still carry the pre-fix capitalized URLs.
    # This must never be "fixed" in place (verification-doctrine 2).
    items = yaml.safe_load(open(V1_PATH))["eval"]
    assert len(items) == 629
    raw = open(V1_PATH, encoding="utf-8").read()
    for gid, old in OLD_URLS.items():
        assert old in raw, f"v1 should still carry the pre-fix URL for {gid}"


def test_every_correction_is_recorded_as_live_200_and_corpus_member():
    live = json.load(open(LIVE_CHECK_PATH))
    membership = json.load(open(MEMBERSHIP_PATH))

    live_by_id = {r["id"]: r for r in live["results"]}
    mem_by_id = {r["id"]: r for r in membership["rows"]}

    assert set(live_by_id) == set(CORRECTED_IDS)
    assert set(mem_by_id) == set(CORRECTED_IDS)

    for gid in CORRECTED_IDS:
        lr = live_by_id[gid]
        assert lr["old_url"] == OLD_URLS[gid]
        assert lr["new_url"] == NEW_URLS[gid]
        assert lr["new_url_result"]["status"] == 200, (
            f"{gid}: new form did not resolve live 200: {lr['new_url_result']}")
        assert lr["old_url_result"]["status"] != 200, (
            f"{gid}: old form unexpectedly resolves live 200 -- re-check the bug premise")

        mr = mem_by_id[gid]
        assert mr["new_url"] == NEW_URLS[gid]
        assert mr["new_in_corpus"] is True, (
            f"{gid}: new form is not a member of the sealed aw_e3 snapshot")
        assert mr["old_in_corpus"] is False, (
            f"{gid}: old form unexpectedly IS a member of the sealed aw_e3 snapshot")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
