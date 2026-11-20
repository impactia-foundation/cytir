"""`scratch_ch_cantonal_law_exact` on the Protocol: rows identical to the legacy drivers' (task 1108, S7).

`data/samples/ch-cantonal-law/` holds sealed tapes per portal (shaped like the origin responses,
written for the test) and `expected_<portal>.jsonl`, what the UNMODIFIED legacy drivers wrote for
them before the builders moved. The family must produce the same rows, byte for byte. No network.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

from cytir.sources import run_family

REPO = pathlib.Path(__file__).resolve().parent.parent
SAMPLE = REPO / "data" / "samples" / "ch-cantonal-law"
OK = {"bucket": "ok", "search_ok": True, "train_ok": True, "status": "ok"}
FID = "scratch_ch_cantonal_law_exact"


def _run(tmp_path, portals, consent=None):
    return run_family.run_family(
        FID, out_dir=tmp_path / "o", today="2026-10-23", log=lambda m: None,
        consent=consent or (lambda h, d: {**OK, "host": h}),
        options={"root": str(SAMPLE), "portals": portals})


# --- cantonal_cc (task 1174): the driver's rows, recorded with the real language gate -------------------------
CC_REC = json.loads((SAMPLE / "cantonal_cc-recording.json").read_text(encoding="utf-8"))
_h = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()   # noqa: E731


@pytest.fixture()
def recorded_gate(monkeypatch):
    """The language gate answers from the recording; a question it never saw fails loudly."""
    from cytir.acquire import fetch_process as fp

    def detect_lang(text, url, default):
        key = json.dumps([_h(text), url, default])
        if key not in CC_REC["gate"]:
            raise AssertionError(f"language gate asked a question the recording never saw: {url}")
        return tuple(CC_REC["gate"][key])
    monkeypatch.setattr(fp, "detect_lang", detect_lang)




def _want(portal):
    return sorted(json.dumps(json.loads(x), ensure_ascii=False)
                  for x in (SAMPLE / f"expected_{portal}.jsonl").read_text(encoding="utf-8").splitlines())


@pytest.mark.parametrize("portal", ["lexwork", "zhlex"])
def test_rows_are_byte_identical_to_the_legacy_driver(tmp_path, portal):
    _run(tmp_path, portal)
    _, recs = run_family.load_pack(tmp_path / "o" / "pack")
    recs = [{k: v for k, v in r.items() if k != 'family_id'} for r in recs]   # ADR-124: the door's stamp is not the driver's
    assert sorted(json.dumps(r, ensure_ascii=False) for r in recs) == _want(portal)


def test_lexwork_drops_name_their_reason_and_the_ledger_balances(tmp_path):
    t = _run(tmp_path, "lexwork")["terminals"]
    assert t["units_in"] == t["units_admitted"] + t["units_excluded"] + t["units_quarantined"] == 3
    assert t["excluded_reasons"] == {"no_xhtml": 1}


def test_zhlex_drops_name_their_reason(tmp_path):
    t = _run(tmp_path, "zhlex")["terminals"]
    assert t["units_in"] == 4 and t["units_admitted"] == 2
    assert t["excluded_reasons"] == {"no_title_or_link": 1, "repealed": 1}


def test_all_three_portals_in_one_run(tmp_path, recorded_gate):
    m = _run(tmp_path, "")
    assert m["terminals"]["units_in"] == 17 and m["terminals"]["records"] == 8 + 8


def test_the_declined_portal_is_refused_with_its_reason(tmp_path):
    with pytest.raises(ValueError, match="consent-declined"):
        _run(tmp_path, "vd")


def test_cantonal_cc_recording_declares_real_models_and_a_redacted_origin():
    prov = CC_REC["provenance"]
    assert prov["tokenizer_status"] == "ok" and prov["tokenizer_is_fast"] is True and prov["lid_sha256"]
    assert prov["tape_sha256"] == hashlib.sha256((SAMPLE / "cantonal_cc.jsonl").read_bytes()).hexdigest()
    assert len(prov["git_sha"]) == 40 and prov["today_frozen"] == "2026-10-23"
    assert all(str(prov[k]).startswith("redacted") for k in ("host", "checkout_app", "lid_model", "command"))


def test_cantonal_cc_rows_equal_the_driver_recording(tmp_path, recorded_gate):
    m = _run(tmp_path, "cantonal_cc")
    _, recs = run_family.load_pack(tmp_path / "o" / "pack")
    recs = [{k: v for k, v in r.items() if k != 'family_id'} for r in recs]   # ADR-124: the door's stamp is not the driver's
    key = lambda r: r["point_id"]   # noqa: E731
    assert len(CC_REC["rows"]) == 8
    assert json.dumps(sorted(recs, key=key), sort_keys=True, ensure_ascii=False) == \
        json.dumps(sorted(CC_REC["rows"], key=key), sort_keys=True, ensure_ascii=False)
    t = m["terminals"]
    assert t["units_in"] == 10 and t["units_admitted"] == 8        # the 404 shell and the link list yield no row
    assert {r["canton"] for r in recs} == {"GE", "NE", "JU", "SZ", "TI"} and {r["last_seen"] for r in recs} == {"2026-10-23"}


def test_a_declined_host_leaves_every_unit_excluded_and_adds_no_consent_field(tmp_path):
    m = _run(tmp_path, "zhlex", consent=lambda h, d: {"host": h, "bucket": "declined"})
    t = m["terminals"]
    assert t["units_admitted"] == 0 and t["records"] == 0
    assert t["excluded_reasons"] == {"consent_declined:www.zh.ch": 4}


def test_the_contract_row_declares_vd_as_a_consent_declined_origin():
    """Ruling 60 D10: vd is a declared decline (a value of the consent_state enum), not a missing dialect."""
    rows = json.loads((REPO / "data" / "source-family-contracts.v1.json").read_text(encoding="utf-8"))["families"]
    row = next(r for r in rows if r["family_id"] == FID)
    (vd,) = [o for o in row["consent"]["declared_declined_origins"] if o["portal"] == "vd"]
    enum = json.loads((REPO / "data" / "payload-enum-tables.v1.json").read_text(encoding="utf-8"))
    states = {json.loads(v) for t in enum.values() if isinstance(t, dict) and "consent_state" in t
              for v in t["consent_state"].values()}
    assert vd["consent_state"] in states
    assert (vd["search_ok"], vd["train_ok"]) == (False, False) and vd["host"].endswith(".vd.ch")
    assert "2026-10-25" in vd["evidence"]


# --- lexwork LIVE path (task 1488): no tape, a fake `_get`, no network ----------------------------------------
from cytir.extract import cantonal_lexwork as LW   # noqa: E402

#: sha256 over the sorted row JSON that the main-branch driver (`process_law`, blob 080e520a8e149e3c905580ff56abd77db8315292)
#: produced for the sample's BS law 170.100 (5 rows) behind a fake `_get`.
LIVE_PIN = "9e81099bfa5433a4950d55e9df46047c656bb34594a7e277e55f7aba548e364b"
_LAW = json.loads((SAMPLE / "lexwork.jsonl").read_text(encoding="utf-8").splitlines()[0])


class _Resp:
    def __init__(self, body, status=200):
        self._b, self.status_code, self.headers = body, status, {}

    def json(self):
        return self._b

    def raise_for_status(self):
        import httpx
        if self.status_code >= 400:     # what the package's real client (httpx) raises
            raise httpx.HTTPStatusError(f"{self.status_code}", request=httpx.Request("GET", "https://x/"),
                                        response=httpx.Response(self.status_code))


@pytest.fixture()
def fake_origin(monkeypatch):
    calls = []
    index = {"1": [{"systematic_number": "170.100", "title": "Gemeindegesetz", "abrogated": False},
                   {"systematic_number": "999.9", "title": "Weg", "abrogated": False},
                   {"systematic_number": "170.200", "title": "Alt", "abrogated": True}]}

    def get(host, path, tries=5):
        calls.append((host, path))
        if path.endswith("/lightweight_index"):
            return _Resp(index)
        if path.endswith("/170.100"):
            return _Resp(_LAW["response"])
        _Resp({}, 404).raise_for_status()
    monkeypatch.setattr(LW, "_get", get)
    return calls


def _live(tmp_path, consent=None, **opts):
    return run_family.run_family(
        FID, out_dir=tmp_path / "o", today="2026-10-23", log=lambda m: None,
        consent=consent or (lambda h, d: {**OK, "host": h}),
        options={"portals": "lexwork", "cantons": "BS", **opts})


def test_live_rows_equal_the_driver_recording(tmp_path, fake_origin):
    m = _live(tmp_path)
    _, recs = run_family.load_pack(tmp_path / "o" / "pack")
    recs = [{k: v for k, v in r.items() if k != "family_id"} for r in recs]   # ADR-124: the door's stamp is not the driver's
    rows = "\n".join(sorted(json.dumps(r, ensure_ascii=False) for r in recs))
    assert hashlib.sha256(rows.encode()).hexdigest() == LIVE_PIN
    assert m["terminals"]["excluded_reasons"] == {"not_found": 1}      # 999.9 404s: named, not skipped
    assert all(h == "www.gesetzessammlung.bs.ch" for h, _ in fake_origin)


def test_live_refuses_a_host_with_undeclared_consent(tmp_path, fake_origin):
    with pytest.raises(RuntimeError, match="www.gesetzessammlung.bs.ch"):
        _live(tmp_path, consent=lambda h, d: {"host": h, "bucket": "unknown"})
    assert fake_origin == []                                           # not one request left the box


def test_live_refuses_skip_consent(tmp_path, fake_origin):
    with pytest.raises(ValueError, match="skip-consent"):
        run_family.run_family(FID, out_dir=tmp_path / "o", today="2026-10-23", skip_consent=True,
                              log=lambda m: None, options={"portals": "lexwork", "cantons": "BS"})
    assert fake_origin == []


def test_live_done_file_skips_a_done_law_and_checkpoints_new_ones(tmp_path, fake_origin):
    done = tmp_path / "x.done"
    done.write_text("BS|de|999.9\n")
    _live(tmp_path, done_file=str(done))
    assert not any(p.endswith("/999.9") for _, p in fake_origin)
    assert done.read_text().splitlines() == ["BS|de|999.9", "BS|de|170.100"]


# --- task 1726: the per-page extraction deadline is an explicit step ---------------------------------------------

@pytest.fixture(autouse=True)
def _in_process_extractor(monkeypatch):
    """The drivers' pool spawns workers that cannot see a patched gate; run the same extractor in-process."""
    from cytir.core import common as C
    from cytir.sources.families import ch_cantonal_law as c
    monkeypatch.setattr(c, "_extract_in_pool",
                        lambda url, html, meta: (*C.process_html(url, html, meta), {}, False))


def test_the_page_step_uses_the_drivers_helper_and_cap():
    import inspect
    from cytir.acquire import discovery_ingest as di
    from cytir.sources.families import ch_cantonal_law as c
    src = inspect.getsource(c._extract_page)
    assert "di._HTML_CAP" in src and "_extract_in_pool" in src
    assert di._EXTRACT_TIMEOUT > 0


def test_a_page_past_the_deadline_is_recorded_timed_out_and_the_run_continues(tmp_path, recorded_gate, monkeypatch):
    from cytir.core import common as C
    from cytir.sources.families import ch_cantonal_law as c
    seen = []

    def stub(url, html, meta):
        seen.append(url)
        if len(seen) == 1:          # the first page "sleeps past the timeout": the helper reports timed_out
            return [], set(), {}, True
        return (*C.process_html(url, html, meta), {}, False)
    monkeypatch.setattr(c, "_extract_in_pool", stub)
    m = _run(tmp_path, "cantonal_cc")
    t = m["terminals"]
    assert t["excluded_reasons"].get("extract_timeout") == 1
    assert t["units_admitted"] == 7 and len(seen) >= 2


# --- task 1730: per-host circuit breaker after consecutive extraction timeouts -----------------------------------

def _breaker_run(monkeypatch, plan):
    """Drive `_extract_page` on pages (host, timed_out) in order; returns (outcomes, stub-called urls)."""
    from cytir.core import common as C
    from cytir.sources.family import Doc, RunContext, UnitExcluded
    from cytir.sources.families import ch_cantonal_law as c
    called = []
    pages = [(f"https://{h}/p{i}", t) for i, (h, t) in enumerate(plan)]
    timed = dict(pages)

    def stub(url, html, meta):
        called.append(url)
        return ([], set(), {}, True) if timed[url] else (*C.process_html(url, html, meta), {}, False)
    monkeypatch.setattr(c, "_extract_in_pool", stub)
    ctx = RunContext(out_dir=None, today="2026-10-30")
    out = []
    for url, _ in pages:
        doc = Doc(url=url, text="<html><body><p>x</p></body></html>", lang="de",
                  extra={"passage_common": {}, "digest": None})
        try:
            c._extract_page(doc, ctx)
            out.append("ok")
        except UnitExcluded as e:
            out.append(e.reason)
    return out, called


def test_the_threshold_and_reason_are_the_drivers_own():
    from cytir.acquire import discovery_ingest as di
    from cytir.sources.families import ch_cantonal_law as c
    assert c._HOST_TIMEOUT_BREAKER == di._DOMAIN_TIMEOUT_BREAKER == 5
    assert c.BREAKER_REASON == "circuit_breaker_early_stop"


def test_a_host_that_times_out_n_times_is_cut_off_and_the_stub_is_not_called_again(monkeypatch):
    from cytir.sources.families import ch_cantonal_law as c
    n = c._HOST_TIMEOUT_BREAKER
    out, called = _breaker_run(monkeypatch, [("a.ch", True)] * (n + 3))
    assert out == ["extract_timeout"] * n + [c.BREAKER_REASON] * 3
    assert len(called) == n


def test_another_host_is_unaffected(monkeypatch):
    from cytir.sources.families import ch_cantonal_law as c
    n = c._HOST_TIMEOUT_BREAKER
    out, called = _breaker_run(monkeypatch, [("a.ch", True)] * n + [("b.ch", False), ("a.ch", True)])
    assert out[n] == "ok" and out[n + 1] == c.BREAKER_REASON
    assert len(called) == n + 1


def test_a_success_between_timeouts_resets_the_counter(monkeypatch):
    from cytir.sources.families import ch_cantonal_law as c
    n = c._HOST_TIMEOUT_BREAKER
    plan = [("a.ch", True)] * (n - 1) + [("a.ch", False)] + [("a.ch", True)] * (n - 1) + [("a.ch", False)]
    out, called = _breaker_run(monkeypatch, plan)
    assert c.BREAKER_REASON not in out and len(called) == len(plan)


# --- task 1737: the breaker through run_family's ledger; `scratch` never reaches the manifest ---------------------

def test_scratch_is_a_declared_field_and_nothing_serialises_it():
    import dataclasses
    from cytir.sources import family, run_family as rf
    assert "scratch" in {f.name for f in dataclasses.fields(family.RunContext)}
    assert "scratch" not in pathlib.Path(rf.__file__).read_text(encoding="utf-8")


def test_the_breaker_runs_through_the_runner_ledger_and_the_manifest_has_no_scratch(tmp_path, recorded_gate, monkeypatch):
    from cytir.core import common as C
    from cytir.sources.families import ch_cantonal_law as c
    n = 1                               # the sample has 2 pages per host: N=1 -> N timeouts + 1 early stop
    monkeypatch.setattr(c, "_HOST_TIMEOUT_BREAKER", n)
    calls = []

    def stub(url, html, meta):          # every page of rsn.ne.ch times out; the other hosts extract normally
        calls.append(url)
        if "rsn.ne.ch" in url:
            return [], set(), {}, True
        return (*C.process_html(url, html, meta), {}, False)
    monkeypatch.setattr(c, "_extract_in_pool", stub)
    m = _run(tmp_path, "cantonal_cc")
    r = m["terminals"]["excluded_reasons"]
    assert r.get("extract_timeout") == n and r.get(c.BREAKER_REASON) == 1
    assert sum("rsn.ne.ch" in u for u in calls) == n          # the cut-off page never reached the extractor
    assert m["terminals"]["units_admitted"] == 7              # the run continued past the cut-off host
    assert '"scratch"' not in json.dumps(m) and "cantonal_timeouts" not in json.dumps(m)


def test_live_acquire_needs_no_requests_module(tmp_path, fake_origin, monkeypatch):
    """Task 1838: `requests` is not a declared dependency; the live path must not import it (CI venv lacks it)."""
    monkeypatch.setitem(sys.modules, "requests", None)       # `import requests` now raises ImportError
    m = _live(tmp_path)
    assert m["terminals"]["excluded_reasons"] == {"not_found": 1}
