"""`scratch_de_laender_juris_exact` on the Protocol: rows identical to the pre-move driver's (task 1414).

`tests/fixtures/de_laender_juris/` holds synthetic recherche3 law frames (SH, BW): a plain law, a law whose
title already holds its abbreviation, an amendment act, a repeal act, an announcement, a one-word title, a
frame with no date. `expected_rows.jsonl` is what the PRE-MOVE driver wrote for them: the blob
old driver script at origin/main f1b85fe99 (git blob 81da7e1d), loaded with
`importlib` as a plain module, only `Portal` replaced by a stub that returns the frames, then its own
`build_rows` run for BW and SH. The digest below pins that file. The live route runs over
`httpx.MockTransport`; no network, no live fixture.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import httpx
import pytest

from cytir.sources import contract, registry, run_family
from cytir.sources.families import de_laender_juris as M

FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "de_laender_juris"
PINNED_SHA256 = "ef103ed9ba496216f4976bbcbda66b3db3b6b3208c0fbdba1d9dc73ec7957012"
FID = "scratch_de_laender_juris_exact"
OK = {"bucket": "ok", "search_ok": True, "train_ok": True, "status": "ok"}
TODAY = "2026-10-30"


def _basis_undeclared() -> bool:
    """True while the family row's consent basis is undeclared (task 1624, memo 10)."""
    row = contract.by_id(contract.load()).get(FID, {})
    return ((row.get("consent") or {}).get("basis", "unknown")) == "unknown"


_BASIS_UNDECLARED = _basis_undeclared()
_SKIP_UNDECLARED = pytest.mark.skipif(
    _BASIS_UNDECLARED,
    reason="scratch_de_laender_juris_exact declares no consent basis; run_family refuses undeclared consent (task 1406); skipped until Tim declares the basis (task 1624, memo 10)",
)
LOGS: list = []


@pytest.fixture(autouse=True)
def _identity(monkeypatch):
    LOGS.clear()
    monkeypatch.setenv("AW_OPERATOR_URL", "https://example.invalid/operator")
    monkeypatch.setenv("AW_OPERATOR_CONTACT", "crawler@example.invalid")
    monkeypatch.setattr(M, "_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(M.time, "sleep", lambda s: None)


def _run(tmp_path, options, consent=lambda h, d: {**OK, "host": h}, http=None):
    return run_family.run_family(FID, out_dir=tmp_path / "o", today=TODAY, log=LOGS.append,
                                 consent=consent, options=options, http=http)


def _want():
    return [json.loads(x) for x in (FIX / "expected_rows.jsonl").read_text(encoding="utf-8").splitlines()]


def _got(tmp_path):
    _, recs = run_family.load_pack(tmp_path / "o" / "pack")
    return {r["url"]: r for r in recs}


def test_the_expected_rows_are_the_pinned_premove_output():
    assert hashlib.sha256((FIX / "expected_rows.jsonl").read_bytes()).hexdigest() == PINNED_SHA256


@_SKIP_UNDECLARED
def test_replayed_rows_carry_every_field_of_the_premove_driver_rows(tmp_path):
    r = _run(tmp_path, {"raw": str(FIX), "land": "BW,SH"})
    got = _got(tmp_path)
    want = _want()
    assert len(got) == len(want) == 5
    for w in want:
        g = got[w["url"]]
        assert {k: g.get(k) for k in w} == w, w["url"]
    assert r["terminals"]["units_admitted"] == 2


@_SKIP_UNDECLARED
def test_the_live_route_builds_the_same_rows(tmp_path):
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.url.path, req.headers.get("user-agent"), req.headers.get("x-csrf-token")))
        ep = req.url.path.rsplit("/", 1)[-1]
        if ep == "init":
            return httpx.Response(200, json={"csrfToken": "tok", "user": {"login": "Buergerservice"}})
        start = json.loads(req.content)["searchTasks"]["RESULT_LIST"]["start"]
        frames = json.loads((FIX / "SH.json").read_text(encoding="utf-8")) if start == 1 else []
        return httpx.Response(200, json={"resultList": frames})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    r = _run(tmp_path, {"land": "SH"}, http=http)
    got = _got(tmp_path)
    want = [w for w in _want() if w["land"] == "SH"]
    assert len(got) == len(want) == 4
    for w in want:
        assert {k: got[w["url"]].get(k) for k in w} == w
    assert r["terminals"]["units_admitted"] == 1
    assert seen[0][2] is None and all(s[2] == "tok" for s in seen[1:])      # csrf only after init
    assert all(s[1].startswith("authoritative-web") and "Mozilla" not in s[1] for s in seen)


def test_no_user_agent_is_read_at_import():
    src = pathlib.Path(M.__file__).read_text(encoding="utf-8")
    assert "Mozilla/5.0" not in src
    assert "user_agent" in src and "def _user_agent" in src


@_SKIP_UNDECLARED
def test_an_unidentified_crawler_fails_closed_per_request(tmp_path, monkeypatch):
    monkeypatch.delenv("AW_OPERATOR_URL")
    monkeypatch.delenv("AW_OPERATOR_CONTACT")
    monkeypatch.delenv("AW_ALLOW_UNIDENTIFIED_CRAWL", raising=False)
    monkeypatch.setattr("cytir.core.user_agent._load_file", lambda: {})
    monkeypatch.setattr("cytir.core.user_agent.operator", lambda: ("", "", False))
    http = httpx.Client(transport=httpx.MockTransport(
        lambda req: pytest.fail("no request may leave an unidentified crawler")))
    with pytest.raises(run_family.RunError, match="every one of 1 unit"):
        _run(tmp_path, {"land": "SH"}, http=http)
    assert "AW_ALLOW_UNIDENTIFIED_CRAWL" in "\n".join(LOGS)


@_SKIP_UNDECLARED
def test_row_drops_are_counted_with_their_reasons(tmp_path):
    _run(tmp_path, {"raw": str(FIX), "land": "BW,SH"})
    notes = json.loads((tmp_path / "o" / "pack" / "manifest.json").read_text())
    blob = json.dumps(notes)
    assert '"dropped_amend_repeal": 3' in blob
    assert '"no_title": 1' in blob


@_SKIP_UNDECLARED
def test_a_declined_host_leaves_the_land_excluded(tmp_path):
    t = _run(tmp_path, {"raw": str(FIX), "land": "SH"},
             consent=lambda h, d: {"host": h, "bucket": "declined"})["terminals"]
    assert t["units_admitted"] == 0
    assert t["excluded_reasons"] == {"consent_declined:gesetze-rechtsprechung.sh.juris.de": 1}


@_SKIP_UNDECLARED
def test_a_registry_blocked_land_is_excluded_on_the_live_route_not_crawled(tmp_path):
    hosts = []

    def handler(req):
        hosts.append(req.url.host)
        if req.url.path.endswith("init"):
            return httpx.Response(200, json={"csrfToken": "t"})
        start = json.loads(req.content)["searchTasks"]["RESULT_LIST"]["start"]
        return httpx.Response(200, json={"resultList": json.loads(
            (FIX / "SH.json").read_text(encoding="utf-8")) if start == 1 else []})
    http = httpx.Client(transport=httpx.MockTransport(handler))
    t = _run(tmp_path, {"land": "BW,SH"}, http=http)["terminals"]
    assert t["units_admitted"] == 1
    assert t["excluded_reasons"] == {"consent_registry_blocked:www.landesrecht-bw.de": 1}
    assert set(hosts) == {"gesetze-rechtsprechung.sh.juris.de"}       # the blocked host got no request


@_SKIP_UNDECLARED
def test_an_empty_enumeration_is_a_quarantine_not_a_clean_empty(tmp_path):
    def handler(req):
        if req.url.path.endswith("init"):
            return httpx.Response(200, json={"csrfToken": "t"})
        return httpx.Response(200, json={"resultList": []})
    http = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(run_family.RunError, match="every one of 1 unit"):
        _run(tmp_path, {"land": "SH"}, http=http)
    assert "no law frames" in "\n".join(LOGS)


def test_a_non_admitted_land_code_is_refused_loudly(tmp_path):
    with pytest.raises(SystemExit, match="non-admitted"):
        _run(tmp_path, {"raw": str(FIX), "land": "BE"})


@_SKIP_UNDECLARED
def test_the_card_states_the_declared_class(tmp_path):
    _run(tmp_path, {"raw": str(FIX), "land": "BW,SH"})
    rows = list(_got(tmp_path).values())
    assert {r["source_type"] for r in rows} == {"official"}
    assert {r["trust_tier"] for r in rows} == {1.0}
    assert {(r["country"], r["provenance"]) for r in rows} == {("DE", "de_laender_juris")}


def test_the_family_is_registered_and_its_licence_and_consent_come_from_the_row():
    adapter, row, _spec = registry.admit(FID)
    assert adapter.family_id == FID and adapter.direct_crawl is True
    assert row["consent"]["search_ok"] is None and row["licence"]["spdx"] is None   # left unknown
    assert not getattr(adapter, "PARTIAL", ())
