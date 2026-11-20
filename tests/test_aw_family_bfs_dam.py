"""`bfs_dam_exact` on the Protocol: rows identical to the legacy driver's (task 1101, S7).

`data/samples/bfs-dam/assets_raw.jsonl` is a sealed eight-line catalogue cut (four kept, four
dropped for the four named reasons). `expected_rows.jsonl` is what the legacy `ingest_bfs_dam.py
parse` wrote for it BEFORE the builder moved into the package (task 1101): the family must produce
the same rows, byte for byte (ids, text, provenance), and the same drop reasons. No network.
"""
from __future__ import annotations

import json
import pathlib

from cytir.sources import run_family

REPO = pathlib.Path(__file__).resolve().parent.parent
SAMPLE = REPO / "data" / "samples" / "bfs-dam"
OK = {"bucket": "ok", "search_ok": True, "train_ok": True, "status": "ok"}


def _run(tmp_path):
    return run_family.run_family(
        "bfs_dam_exact", out_dir=tmp_path / "o", today="2026-10-23", log=lambda m: None,
        consent=lambda h, d: {**OK, "host": h}, options={"root": str(SAMPLE)})


def _rows(tmp_path):
    _, recs = run_family.load_pack(tmp_path / "o" / "pack")
    recs = [{k: v for k, v in r.items() if k != 'family_id'} for r in recs]   # ADR-124: the door's stamp is not the driver's
    return recs


def test_rows_are_byte_identical_to_the_legacy_driver(tmp_path):
    _run(tmp_path)
    want = [json.dumps(json.loads(x), ensure_ascii=False)
            for x in (SAMPLE / "expected_rows.jsonl").read_text(encoding="utf-8").splitlines()]
    got = [json.dumps(r, ensure_ascii=False) for r in _rows(tmp_path)]
    assert got == want


def test_every_drop_has_the_builders_own_reason_and_the_ledger_balances(tmp_path):
    t = _run(tmp_path)["terminals"]
    assert t["units_in"] == t["units_admitted"] + t["units_excluded"] + t["units_quarantined"] == 8
    assert t["excluded_reasons"] == {"bad_lang": 1, "future_embargo": 1, "no_damid": 1,
                                     "thin_no_prose": 1}


def test_a_declined_api_host_leaves_every_unit_excluded_with_its_reason(tmp_path):
    m = run_family.run_family(
        "bfs_dam_exact", out_dir=tmp_path / "d", today="2026-10-23", log=lambda m: None,
        consent=lambda h, d: {"host": h, "bucket": "declined"}, options={"root": str(SAMPLE)})
    t = m["terminals"]
    assert t["units_admitted"] == 0 and t["records"] == 0
    assert t["excluded_reasons"] == {"consent_declined:dam-api.bfs.admin.ch": 8}


# --- the `enumerate` phase moved into the package (task 1487) ------------------------------------
# `tests/data/bfs_dam_enumerate/expected_raw.jsonl` is what the legacy driver (blob
# dc96da9073ee8b1a6891fa36c7ef757c494ba2f4 on origin/main) wrote for `listing.json` through the
# same fake transport, recorded before the move. No network.
ENUM = REPO / "tests" / "data" / "bfs_dam_enumerate"


def _enumerate(monkeypatch, tmp_path, *extra, responses=None):
    import httpx

    from cytir.sources.bfs_dam import enumerate as E
    pages = json.loads((ENUM / "listing.json").read_text())
    seen = []
    queue = list(responses or [])

    def handler(req):
        seen.append(req.url.params["skip"])
        if queue:
            return queue.pop(0)
        return httpx.Response(200, json=pages.get(req.url.params["skip"], {"total": 60, "data": []}))

    monkeypatch.setattr(httpx, "HTTPTransport", lambda *a, **k: httpx.MockTransport(handler))
    monkeypatch.setattr(E.time, "sleep", lambda s: None)
    raw, state = tmp_path / "raw.jsonl", tmp_path / "st.json"
    monkeypatch.setattr("sys.argv", ["x", "enumerate", "--raw", str(raw), "--state", str(state), *extra])
    E.main()
    return raw, state, seen


def test_enumerate_matches_recorded_legacy_driver(monkeypatch, tmp_path):
    raw, state, seen = _enumerate(monkeypatch, tmp_path)
    assert raw.read_text() == (ENUM / "expected_raw.jsonl").read_text()
    assert seen == ["0", "20", "40"]
    st = json.loads(state.read_text())
    assert (st["next_skip"], st["count"], st["total"]) == (60, 8, 60)


def test_enumerate_resumes_and_retries_throttle(monkeypatch, tmp_path):
    import httpx
    (tmp_path / "st.json").write_text(json.dumps({"next_skip": 40, "count": 6, "total": 60}))
    raw, state, seen = _enumerate(monkeypatch, tmp_path, responses=[httpx.Response(429)])
    assert seen == ["40", "40"]                       # one 429, then the page
    assert len(raw.read_text().splitlines()) == 2
    assert json.loads(state.read_text())["count"] == 8


def test_enumerate_output_feeds_the_family(monkeypatch, tmp_path):
    raw, _, _ = _enumerate(monkeypatch, tmp_path)
    out = run_family.run_family(
        "bfs_dam_exact", out_dir=tmp_path / "f", today="2026-10-23", log=lambda m: None,
        consent=lambda h, d: {**OK, "host": h}, options={"root": str(raw)})
    assert out is not None
