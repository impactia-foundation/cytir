"""`statcan_wds` on the Protocol: rows identical to the legacy driver's (task 1237).

`tests/fixtures/statcan_wds/` is a small WDS catalogue (an active cube, an archived one, a one-word
title, a missing French title, a cube without product id, a duplicate cube, a codeset-less subject) with
four release days. `expected_rows.jsonl` is what the legacy `ingest_statcan_wds.py parse` wrote for it
before the driver was folded into the family. The live path runs over `httpx.MockTransport`; no network.
"""
from __future__ import annotations

import json
import pathlib

import httpx

from cytir.sources import run_family

FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "statcan_wds"
OK = {"bucket": "ok", "search_ok": True, "train_ok": True, "status": "ok"}
TODAY = "2026-10-01"      # the four release days of the fixture are 09-30 .. 09-27


def _run(tmp_path, options, consent=lambda h, d: {**OK, "host": h}, http=None):
    return run_family.run_family("statcan_wds", out_dir=tmp_path / "o", today=TODAY, log=lambda m: None,
                                 consent=consent, options=options, http=http)


def _want():
    return sorted((json.loads(x) for x in (FIX / "expected_rows.jsonl").read_text().splitlines()),
                  key=lambda r: r["point_id"])


def _mock(day_answers=None):
    daily = {d["date"]: d["productIds"]
             for d in map(json.loads, (FIX / "daily.jsonl").read_text().splitlines())}

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path.endswith("getAllCubesListLite"):
            return httpx.Response(200, json=json.loads((FIX / "cubes.json").read_text()))
        if path.endswith("getCodeSets"):
            return httpx.Response(200, json=json.loads((FIX / "codesets.json").read_text()))
        day = path.rsplit("/", 1)[-1]
        if day_answers and day in day_answers:
            return day_answers[day]
        return httpx.Response(200, json={"object": [{"productId": p} for p in daily.get(day, [])]})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_replayed_rows_are_identical_to_the_legacy_driver(tmp_path):
    _run(tmp_path, {"raw": str(FIX)})
    _, recs = run_family.load_pack(tmp_path / "o" / "pack")
    recs = [{k: v for k, v in r.items() if k != 'family_id'} for r in recs]   # ADR-124: the door's stamp is not the driver's
    assert sorted(recs, key=lambda r: r["point_id"]) == _want()


def test_the_live_route_builds_the_same_rows(tmp_path, monkeypatch):
    monkeypatch.setattr("cytir.sources.api_fetch.time.sleep", lambda s: None)
    r = _run(tmp_path, {"daily_days": "4", "max_req_s": "1000"}, http=_mock())
    _, recs = run_family.load_pack(tmp_path / "o" / "pack")
    recs = [{k: v for k, v in r.items() if k != 'family_id'} for r in recs]   # ADR-124: the door's stamp is not the driver's
    assert sorted(recs, key=lambda x: x["point_id"]) == _want()
    assert r["terminals"]["units_admitted"] == 1


def test_row_drops_are_counted_with_the_drivers_own_reasons(tmp_path):
    _run(tmp_path, {"raw": str(FIX)})
    notes = json.loads((tmp_path / "o" / "pack" / "manifest.json").read_text())
    blob = json.dumps(notes)
    for reason in ("archived", "no_title", "thin_title", "no_pid", "dup", "daily_empty",
                   "daily_no_titles"):
        assert reason in blob, reason


def test_a_declined_host_leaves_the_catalogue_excluded(tmp_path):
    t = _run(tmp_path, {"raw": str(FIX)},
             consent=lambda h, d: {"host": h, "bucket": "declined"})["terminals"]
    assert t["units_admitted"] == 0
    assert t["excluded_reasons"] == {"consent_declined:www150.statcan.gc.ca": 1}


def test_the_card_states_the_declared_class():
    rows = _want()
    assert {r["source_type"] for r in rows} == {"gov-stat"}
    assert {r["trust_tier"] for r in rows} == {0.95}
    assert {r["country"] for r in rows} == {"CA"}


def test_a_not_released_day_is_skipped_with_a_reason_not_a_failure(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("cytir.sources.api_fetch.time.sleep", lambda s: None)
    http = _mock({d: httpx.Response(409, json={"message": "The product is not released yet"})
                  for d in ("2026-09-30", "2026-09-29")})
    full = _run(tmp_path / "a", {"daily_days": "4", "max_req_s": "1000"}, http=_mock())
    with caplog.at_level("WARNING"):
        r = _run(tmp_path / "b", {"daily_days": "4", "max_req_s": "1000"}, http=http)
    assert r["terminals"]["units_admitted"] == full["terminals"]["units_admitted"] == 1
    assert "not released yet" in caplog.text


def test_any_other_409_or_5xx_still_fails_the_run(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr("cytir.sources.api_fetch.time.sleep", lambda s: None)
    for bad in (httpx.Response(409, json={"message": "conflict"}), httpx.Response(400, json={}),
                httpx.Response(500, json={})):
        http = _mock({d: bad for d in ("2026-09-30",)})
        r = None
        try:
            r = _run(tmp_path / str(bad.status_code), {"daily_days": "4", "max_req_s": "1000"}, http=http)
        except Exception:  # noqa: BLE001
            continue
        assert r["terminals"]["units_admitted"] == 0


def test_every_record_carries_the_host_table_licence_and_nothing_master_only(tmp_path):
    """ADR-039: `license` comes from cytir.core.license for the WDS host; attribution and license_version
    are master-only and stay unset (the table declares no owner for the host)."""
    from cytir.core import common as C
    from cytir.core import license as LIC

    want = LIC.resolve_license(None, "statcan.gc.ca", "www150.statcan.gc.ca", "gov-stat", None)
    assert want
    _run(tmp_path, {"raw": str(FIX)})
    _, recs = run_family.load_pack(tmp_path / "o" / "pack")
    assert recs and all(r["license"] == want for r in recs)
    assert all(not r.get("attribution") and not r.get("license_version") for r in recs)
    assert all(C.compact_payload(r).get("license") == want for r in recs)
