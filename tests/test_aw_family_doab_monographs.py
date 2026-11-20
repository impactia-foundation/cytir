"""`doab_monographs_mitpress_p0` on the Protocol (task 1237): rows identical to the legacy driver's.

`tests/fixtures/doab_monographs/world.json` is a miniature DOAB register: two admissible MIT Press titles
(one German), and one title each for: no DOI, an edited volume with no author, no licence badge, an
unreadable badge, a DOI that lands on a Crossref chooser, and another publisher. `expected_rows.json` is
what the legacy `ingest_doab_monographs.py run()` staged for it (with the framework's consent verdict
standing in for the per-row consent fields it used to stamp). The network is an `httpx.MockTransport`.
"""
from __future__ import annotations

import json
import pathlib

import httpx
import pytest

from cytir.core import projection
from cytir.sources import run_family
from cytir.sources.families import doab_monographs as M

FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "doab_monographs"
WORLD = json.loads((FIX / "world.json").read_text(encoding="utf-8"))
OK = {"bucket": "ok", "search_ok": True, "train_ok": True, "status": "ok"}


def _transport(world=WORLD, fail_pages=False):
    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if "/rest/search" in url:
            off, lim = int(req.url.params["offset"]), int(req.url.params["limit"])
            us = world["search"][off:off + lim]
            return httpx.Response(200, json=[{"uuid": u} for u in us],
                                  headers={"x-total-count": str(len(world["search"]))})
        if "/rest/items/" in url:
            u = req.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json={"metadata": world["items"][u]})
        if req.url.host == "doi.org":
            final = world["doi"].get(req.url.path[1:])
            return (httpx.Response(302, headers={"location": final}) if final
                    else httpx.Response(404))
        if url in world["pages"]:
            if fail_pages:
                raise httpx.ConnectError("down")
            return httpx.Response(200, text=world["pages"][url])
        if req.url.host in ("direct.mit.edu", "chooser.crossref.org"):
            return httpx.Response(403, text="challenge")     # the gated landing page: never read
        return httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(handler))


def _run(tmp_path, consent=lambda h, d: {**OK, "host": h}, http=None, **opt):
    return run_family.run_family(
        "doab_monographs_mitpress_p0", out_dir=tmp_path / "o", today="2026-10-01",
        log=lambda m: None, consent=consent, http=http or _transport(),
        options={"max_req_s": "1000", **opt})


def test_rows_are_identical_to_the_legacy_driver(tmp_path):
    _run(tmp_path)
    _, recs = run_family.load_pack(tmp_path / "o" / "pack")
    recs = [{k: v for k, v in r.items() if k != 'family_id'} for r in recs]   # ADR-124: the door's stamp is not the driver's
    want = json.loads((FIX / "expected_rows.json").read_text(encoding="utf-8"))
    key = lambda r: r["url"]  # noqa: E731
    ids = {r["url"]: r.pop("point_id") for r in recs}      # the framework's id; the legacy row had none
    assert sorted(recs, key=key) == sorted(want, key=key)
    for r in recs:                                           # ... and it is the one the b1 projection mints
        assert projection.project_record(r)[0] == ids[r["url"]]


def test_every_unit_has_the_legacy_terminal_and_reason(tmp_path):
    t = _run(tmp_path)["terminals"]
    assert t["units_in"] == 8 and t["units_admitted"] == 2 and t["units_quarantined"] == 0
    assert t["excluded_reasons"] == {
        "no_doi": 1, "no_title_or_authors": 1, "no_license_badge": 1, "no_license_spdx": 1,
        "resolved_off_pointer_host": 1, "wrong_publisher_or_missing_fields": 1}


def test_a_declined_pointer_host_reads_no_item_at_all(tmp_path):
    t = _run(tmp_path, consent=lambda h, d: {"host": h, "bucket": "declined"})["terminals"]
    assert t["units_admitted"] == 0
    assert t["excluded_reasons"] == {"consent_declined:direct.mit.edu": 8}


def test_an_unreachable_item_page_is_a_failure_not_a_missing_badge(monkeypatch):
    monkeypatch.setattr(M.time, "sleep", lambda s: None)
    from cytir.sources.family import RunContext, Unit
    ctx = RunContext(out_dir=pathlib.Path("."), today="2026-10-01", http=_transport(fail_pages=True),
                     options={"max_req_s": "1000"})
    fam = M.DoabMonographsFamily()
    with pytest.raises(M.DoabError):          # the run quarantines the unit; it is not `no_license_badge`
        fam.acquire(Unit(key="u-ok1", url="x"), ctx)


def test_the_publisher_option_picks_the_other_press(tmp_path):
    with pytest.raises(run_family.RunError) as e:      # one title, no landing URL in the fixture register
        _run(tmp_path, publisher="Another Press", pointer_hosts="example.org")
    assert "'no_url': 1" in str(e.value)
