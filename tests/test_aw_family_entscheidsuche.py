"""`scratch_entscheidsuche_exact` on the Protocol: rows identical to the legacy driver's (task 1101).

`data/samples/entscheidsuche/` is a sealed five-line queue over four decision JSONs. The decisions
are served by an `httpx.MockTransport` (no network). `expected_rows.jsonl` is what the legacy
`ingest_entscheidsuche.py` composed for them before its builder moved into the package.
"""
from __future__ import annotations

import json
import pathlib

import httpx

from cytir.sources import run_family

REPO = pathlib.Path(__file__).resolve().parent.parent
SAMPLE = REPO / "data" / "samples" / "entscheidsuche"
OK = {"bucket": "ok", "search_ok": True, "train_ok": True, "status": "ok"}


def _http():
    def handler(request: httpx.Request) -> httpx.Response:
        f = SAMPLE / "decisions" / request.url.path.rsplit("/", 1)[-1]
        if not f.is_file():
            return httpx.Response(404)
        return httpx.Response(200, content=f.read_bytes())
    return httpx.Client(transport=httpx.MockTransport(handler))


def _run(tmp_path, consent=lambda h, d: {**OK, "host": h}):
    return run_family.run_family(
        "scratch_entscheidsuche_exact", out_dir=tmp_path / "o", today="2026-10-23",
        log=lambda m: None, http=_http(), consent=consent,
        options={"root": str(SAMPLE), "rate": "1000"})


def test_rows_are_byte_identical_to_the_legacy_driver(tmp_path):
    _run(tmp_path)
    want = [json.dumps(json.loads(x), ensure_ascii=False)
            for x in (SAMPLE / "expected_rows.jsonl").read_text(encoding="utf-8").splitlines()]
    _, recs = run_family.load_pack(tmp_path / "o" / "pack")
    recs = [{k: v for k, v in r.items() if k != 'family_id'} for r in recs]   # ADR-124: the door's stamp is not the driver's
    # the pack orders its rows by point id; the driver wrote queue order: compare as sets of lines
    assert sorted(json.dumps(r, ensure_ascii=False) for r in recs) == sorted(want)


def test_each_drop_has_a_named_terminal_and_the_ledger_balances(tmp_path):
    t = _run(tmp_path)["terminals"]
    assert t["units_in"] == 5 == t["units_admitted"] + t["units_excluded"] + t["units_quarantined"]
    assert t["units_admitted"] == 3
    assert t["excluded_reasons"] == {"http_404": 1, "no_headnote_or_text_quality": 1}


def test_a_declined_host_fetches_nothing(tmp_path):
    t = _run(tmp_path, consent=lambda h, d: {"host": h, "bucket": "declined"})["terminals"]
    assert t["units_admitted"] == 0
    assert t["excluded_reasons"] == {"consent_declined:entscheidsuche.ch": 5}


# --- task 1486: the `toc` (discovery) phase, recording-pinned to the legacy driver -------------

import hashlib  # noqa: E402

from cytir.sources.entscheidsuche import discover  # noqa: E402

_BASE = "https://entscheidsuche.ch"
_INDEX = ["CH_BGE_1", "CH_BGer_1", "CH_BGer_2", "ZH_Obergericht_1", "XX_Unlisted_1"]
_PARTS = {
    "CH_BGE_1": ["CH_BGE_001_BGE-100-Ia-1_1954-01-27", "CH_BGE_001_BGE-100-Ia-1_1954",
                 "CH_BGE_001_BGE-100-Ia-1_nodate", "CH_BGE_001_BGE-150-II-100_2024-03-12"],
    "CH_BGer_1": ["CH_BGer_001_1C-1-2020_2020-01-02", "CH_BGer_001_1C-2-2024_2024-02-03"],
    "CH_BGer_2": ["CH_BGer_001_1C-3-2025_2025-05-06"],
    "ZH_Obergericht_1": ["ZH_Obergericht_001_2025-01-01", "ZH_Obergericht_001_2026-01-15"],
    "XX_Unlisted_1": ["XX_Unlisted_001_2026-01-01"],
}
# sha256 of the sorted queue lines the ORIGINAL driver wrote for this fixture (blob
# ceaa49e87cdbab3815dfe3cfae86a42e70988ae9, origin/main before task 1486): the recording pin.
_TOC_PIN = "dba51ef1d433c05f0a99fa67d62f653c9efb3d0d9ab89e1b5e74df156176ea9f"


def _sitemap_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/sitemaps/sitemapindex.xml":
            return httpx.Response(200, text="".join(
                f"<loc>{_BASE}/docs/Sitemaps/{p}.xml</loc>" for p in _INDEX))
        part = path.rsplit("/", 1)[-1][:-4]
        return httpx.Response(200, text="".join(
            f"<loc>{_BASE}/view/{v}</loc>" for v in _PARTS[part]))
    return httpx.MockTransport(handler)


def _toc_lines(tmp_path, monkeypatch, **kw):
    monkeypatch.setattr(httpx, "HTTPTransport", lambda *a, **k: _sitemap_transport())
    out = tmp_path / "queue.jsonl"
    args = type("A", (), {"out": str(out), "all_years": False, "spiders": "", **kw})()
    discover.cmd_toc(args)
    return sorted(out.read_text(encoding="utf-8").splitlines())


def test_toc_queue_matches_the_legacy_driver_recording(tmp_path, monkeypatch):
    lines = _toc_lines(tmp_path, monkeypatch)
    assert len(lines) == 5  # BGE variants collapse; slice drops BGer 2020, ZH 2025
    assert hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest() == _TOC_PIN


def test_toc_all_years_and_spider_override(tmp_path, monkeypatch):
    assert len(_toc_lines(tmp_path, monkeypatch, all_years=True, spiders="CH_BGer")) == 3


# --- task 1486 review: cmd_fetch appends the family rows to --out; --state is warned, not used ---

def _fake_fetch(monkeypatch):
    from cytir.sources import run_family as rf
    rows = [{"point_id": "a", "n": 1}, {"point_id": "b", "n": 2}]
    monkeypatch.setattr(rf, "run_family", lambda *a, **kw: None)
    monkeypatch.setattr(rf, "load_pack", lambda path: (None, rows))


def test_cmd_fetch_appends_rows_and_warns_on_state(tmp_path, monkeypatch, capsys):
    import argparse
    _fake_fetch(monkeypatch)
    out = tmp_path / "es.jsonl"
    args = argparse.Namespace(queue=str(tmp_path / "q.jsonl"), out=str(out), state=str(tmp_path / "some.json"), rate=4.0)
    discover.cmd_fetch(args)
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert lines == [{"point_id": "a", "n": 1}, {"point_id": "b", "n": 2}]
    err = capsys.readouterr().err
    assert "--state is accepted for CLI compatibility and ignored" in err
