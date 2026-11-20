"""`producer_ids` vocabulary (task 1396 ruling): two statuses, every key present, a real code home."""
from __future__ import annotations

import copy

import pytest

from cytir.sources import contract as contract_mod


def _row() -> dict:
    row = copy.deepcopy(contract_mod.by_id(contract_mod.load())["finma_circulars"])
    row["producer_ids"] = {"old/path.py": {
        "status": "retired-path", "meaning": "a retired path", "declared": "task 1396",
        "date": "2026-10-30", "code_home": "cytir/sources/contract.py"}}
    return row


def _producer_problems(row: dict) -> list[str]:
    return [m for m in contract_mod.validate_family(row) if "producer_ids" in m]


def _real_scripts_home() -> str | None:
    """A non-aw code home that exists in the real contract file (None until such a row lands)."""
    for row in contract_mod.by_id(contract_mod.load()).values():
        for entry in (row.get("producer_ids") or {}).values():
            home = (entry.get("code_home") or "").split()[0:1]
            if home and home[0] != "none" and not home[0].startswith("cytir/") \
                    and (contract_mod.REPO / home[0]).is_file():
                return home[0]
    return None


def test_a_good_block_passes():
    assert _producer_problems(_row()) == []


def test_the_vocabulary_is_closed():
    assert contract_mod.PRODUCER_ID_STATUSES == {"frozen-label", "retired-path"}
    row = _row()
    row["producer_ids"]["old/path.py"]["status"] = "gone"
    assert any("status 'gone'" in m for m in _producer_problems(row))


def test_a_missing_key_or_missing_code_home_fails():
    row = _row()
    del row["producer_ids"]["old/path.py"]["meaning"]
    assert any("lacks ['meaning']" in m for m in _producer_problems(row))
    row = _row()
    row["producer_ids"]["old/path.py"]["code_home"] = "cytir/nope.py"
    assert any("code_home" in m for m in _producer_problems(row))
    row["producer_ids"]["old/path.py"]["code_home"] = "elsewhere/contract.py"
    assert any("code_home" in m for m in _producer_problems(row))


def test_none_with_a_note_passes_for_frozen_label():
    row = _row()
    row["producer_ids"]["old/path.py"]["status"] = "frozen-label"
    row["producer_ids"]["old/path.py"]["code_home"] = "none (frozen label, no live producer)"
    assert _producer_problems(row) == []


def test_none_with_a_note_passes_for_retired_path():
    row = _row()
    row["producer_ids"]["old/path.py"]["code_home"] = "none (producer removed)"
    assert _producer_problems(row) == []


def test_bare_none_fails_for_both_statuses():
    for status in ("frozen-label", "retired-path"):
        row = _row()
        row["producer_ids"]["old/path.py"]["status"] = status
        row["producer_ids"]["old/path.py"]["code_home"] = "none"
        assert any("needs a note" in m for m in _producer_problems(row)), status


def test_a_non_aw_non_scripts_path_fails_with_the_reworded_message():
    row = _row()
    row["producer_ids"]["old/path.py"]["code_home"] = "elsewhere/contract.py"
    assert any("under cytir/ or" in m and "-scripts/" in m for m in _producer_problems(row))


def test_an_existing_scripts_file_passes_for_frozen_label():
    home = _real_scripts_home()
    if home is None:
        pytest.skip("no frozen-label row with a tasks-scripts code home on main yet (task 1250)")
    row = _row()
    row["producer_ids"]["old/path.py"]["status"] = "frozen-label"
    row["producer_ids"]["old/path.py"]["code_home"] = home
    assert _producer_problems(row) == []


def test_the_real_contract_file_has_no_producer_id_problems():
    report = contract_mod.validate(contract_mod.load())
    assert [m for m in report["problems"] if "producer_ids" in m] == []


PRIVATE = "ta" + "sks"            # split literal: the cut ships no path into the private tree
LABEL = "tasks/417-scripts/x.py"


def _problems_in(tree, home: str) -> list[str]:
    row = _row()
    row["producer_ids"]["old/path.py"]["status"] = "frozen-label"
    row["producer_ids"]["old/path.py"]["code_home"] = home
    return [m for m in contract_mod.validate_family(row, repo=tree) if "producer_ids" in m]


def test_tasks_scripts_label_must_exist_when_the_tree_has_tasks(tmp_path):
    (tmp_path / PRIVATE).mkdir()          # task 1787: the private tree
    assert _problems_in(tmp_path, LABEL)
    (tmp_path / PRIVATE / "417-scripts").mkdir()
    (tmp_path / LABEL).write_text("")
    assert _problems_in(tmp_path, LABEL) == []


def test_tasks_scripts_label_is_accepted_when_tasks_is_absent(tmp_path):
    assert _problems_in(tmp_path, LABEL) == []   # the public cut


def test_a_missing_aw_file_is_a_problem_with_or_without_tasks(tmp_path):
    assert _problems_in(tmp_path, "cytir/nope.py")
    (tmp_path / PRIVATE).mkdir()
    assert _problems_in(tmp_path, "cytir/nope.py")
