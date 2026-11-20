"""The assembly binder's guards and plan (task 2099, slice A2), cases copied from their old tests."""
from __future__ import annotations

import json

import pytest

from cytir.index import assemble_plan as AP


def test_cmd_plan_refuses_an_undeclared_gz_record(tmp_path, monkeypatch):
    # the cut does not ship the sealed-populations registry; stand one in
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"artifact_root": {}, "populations": []}))
    monkeypatch.setattr(AP, "data_path", lambda *a: reg)
    plan = {"records": {"mystery_dump": {"uri": "data/assembly/whatever.jsonl.gz"}},
            "apply": {"identity_collapse": {"uri": "x", "count": 0}},
            "build_id": "t", "target": {}, "disposition": {}, "membership": {}}
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan))

    class A:
        work = str(tmp_path / "w")
        artifact_root = str(tmp_path)
        sets = None
    A.plan = str(p)
    with pytest.raises(SystemExit) as e:
        AP.cmd_plan(A)
    assert "role" in str(e.value)


def test_excluded_marker_is_checked_before_the_stage_receipt(tmp_path, monkeypatch):
    for name, excluded in (("kept", False), ("marked", True)):
        d = tmp_path / "vectors" / "release=r" / f"source={name}"
        (d / "rollup-evidence" / "stage").mkdir(parents=True)
        (d / "rollup-evidence" / "stage" / "STAGE-RECEIPT.json").write_text(
            json.dumps({"artifacts": [], "records": 0}))
        if excluded:
            (d / "EXCLUDED.json").write_text(json.dumps(
                {"schema": "aw.duplicate-vector-rollup-exclusion.v1",
                 "admission": False, "status": "excluded"}))
    monkeypatch.setattr(AP, "load_pack_route_extra_sources", lambda: [])
    monkeypatch.setattr(AP, "load_pack_stage_overrides", lambda: {})
    monkeypatch.setattr(AP, "load_pack_route_exclusions", lambda *a, **k: {})
    by_set: dict = {}
    pop = {"set_id": "task417_families", "mode": "replacement",
           "vectors": "artifact://vectors/release=r/source=kept/x"}
    AP._plan_pack_sources(pop, tmp_path, by_set)
    assert by_set["task417_families"]["sources"] == 2
    assert by_set["task417_families"]["unresolved"] == 2


def test_cmd_guards_refuses_an_unsealed_lane_outside_the_known_set(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"pending_lanes": [{"lane_id": "Z-surprise"}]}))
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"guards": [{"id": "G1", "status": "WAITING"}]}))

    class R:
        stdout = ""
        stderr = ""
    monkeypatch.setattr(AP.subprocess, "run", lambda *a, **k: R)

    class A:
        pass
    a = A()
    a.report, a.work, a.plan, a.require_green = str(report), None, str(plan), False
    with pytest.raises(SystemExit) as e:
        AP.cmd_guards(a)
    assert "Z-surprise" in str(e.value)
