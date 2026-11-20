"""Task 2285: the per-build `data/assembly/` records are not shipped; each reader meets the absent file with ONE named reason."""
from __future__ import annotations

import pytest

from cytir.index import assemble_inputs as AI
from cytir.ops import build_assembly_plan as B


def test_loaders_note_the_absent_record_and_use_the_empty_default(tmp_path, capsys):
    assert AI.load_pack_route_extra_sources(tmp_path / "nope.json") == []
    assert AI.load_pack_route_exclusions(tmp_path / "nope2.json") == {}
    err = capsys.readouterr().err
    assert "nope.json not found -> no extras" in err
    assert "nope2.json not found -> no exclusions" in err
    assert "not shipped" in err


def test_plan_builder_names_the_missing_input_and_exits(tmp_path):
    with pytest.raises(SystemExit) as e:
        B.require_input(tmp_path, "data/assembly/fold-report.v1.json")
    msg = str(e.value)
    assert "fold-report.v1.json not found under" in msg and "--root" in msg
    with pytest.raises(SystemExit):
        B.binding(tmp_path, "data/assembly/claim-manifests.v1.json")


def test_plan_builder_passes_a_present_input(tmp_path):
    d = tmp_path / "data" / "assembly"
    d.mkdir(parents=True)
    (d / "fold-report.v1.json").write_text("{}")
    assert B.require_input(tmp_path, "data/assembly/fold-report.v1.json").is_file()
    assert B.binding(tmp_path, "data/assembly/fold-report.v1.json")["bytes"] == 2
