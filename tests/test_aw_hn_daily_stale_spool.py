"""HN daily: a spool dir without a manifest is moved aside before a retry,
and STATUS names the holes below the high-water mark (task 1347)."""
from __future__ import annotations

import json

import pytest

from cytir.news import hn_daily as H


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "W", tmp_path)
    monkeypatch.setattr(H.D, "iface_bytes", lambda _iface: 0)
    monkeypatch.setattr(H, "day_env", lambda day: {})
    monkeypatch.setattr(H, "R", tmp_path)
    return tmp_path


def _fake_run(calls, spool):
    def run(cmd, **kw):
        calls.append(cmd)
        spool.mkdir(parents=True, exist_ok=True)
        (spool / "stage_manifest.json").write_text(
            json.dumps({"points_count": 7, "collection": "c"}))
        return type("R", (), {"returncode": 0})()
    return run


def test_half_staged_spool_moved_aside_then_staged(work, monkeypatch):
    spool = work / "stage-2026-10-25"
    spool.mkdir()
    (spool / "frontier.jsonl").write_text("{}\n")
    calls: list = []
    monkeypatch.setattr(H.subprocess, "run", _fake_run(calls, spool))
    out = H.run_stage("2026-10-24", out_log=work / "o.log")
    assert len(calls) == 1
    assert out["staged"] == 7
    aside = [p for p in work.iterdir() if ".failed-" in p.name]
    assert len(aside) == 1 and aside[0].name.startswith("stage-2026-10-25.failed-")
    assert (aside[0] / "frontier.jsonl").is_file()
    assert not (spool / "frontier.jsonl").exists()


def test_staged_spool_with_manifest_is_reused(work, monkeypatch):
    spool = work / "stage-2026-10-25"
    spool.mkdir()
    (spool / "stage_manifest.json").write_text(json.dumps({"points_count": 3}))
    monkeypatch.setattr(H.subprocess, "run", lambda *a, **k: pytest.fail("ran"))
    out = H.run_stage("2026-10-24", out_log=work / "o.log")
    assert out["staged"] == 3
    assert not [p for p in work.iterdir() if ".failed-" in p.name]


def test_landed_holes_names_the_gap(tmp_path):
    for day, status in (("20261022", "LANDED"), ("20261024", "LANDED"),
                        ("20261023", "FAILED")):
        d = tmp_path / f"{H.PACK_PREFIX}{day}"
        d.mkdir()
        (d / "receipt.json").write_text(json.dumps({"status": status}))
    holes = H.landed_holes("2026-10-24", queue_dir=tmp_path, window_days=4)
    assert holes == ["2026-10-21", "2026-10-23"]
