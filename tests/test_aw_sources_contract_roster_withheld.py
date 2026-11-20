"""A public cut may lack the roster file; master may not (task 2229)."""
import json

import pytest

from cytir.sources.contract import (CUT_MARKER, FamilyContractError, RosterWithheld,
                                 format_report, roster, validate)

_DOC = {"roster_source": {"uri": "data/assembly/roster.json"}, "families": []}


def test_missing_roster_in_master_fails(tmp_path):
    with pytest.raises(FamilyContractError) as ei:
        roster(_DOC, tmp_path)
    assert not isinstance(ei.value, RosterWithheld)
    rep = validate(_DOC, repo=tmp_path)
    assert rep["verdict"] == "FAIL"
    assert rep["roster_withheld"] is None


def test_missing_roster_in_cut_is_withheld(tmp_path):
    (tmp_path / CUT_MARKER).write_text("{}")
    with pytest.raises(RosterWithheld):
        roster(_DOC, tmp_path)
    rep = validate(_DOC, repo=tmp_path)
    assert rep["verdict"] == "PASS"
    assert rep["coverage"] is None
    lines = []
    format_report(rep, log=lines.append)
    assert any(x.startswith("  roster: withheld-in-cut (data/assembly/roster.json)") for x in lines)


def test_present_roster_in_cut_is_still_checked(tmp_path):
    (tmp_path / CUT_MARKER).write_text("{}")
    (tmp_path / "data/assembly").mkdir(parents=True)
    (tmp_path / "data/assembly/roster.json").write_text(json.dumps({"sets": [{"set_id": "a"}]}))
    rep = validate(_DOC, repo=tmp_path)
    assert rep["verdict"] == "FAIL"
    assert rep["coverage"]["undeclared"] == ["a"]
