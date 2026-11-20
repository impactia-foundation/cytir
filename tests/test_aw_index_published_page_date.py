"""`published` re-derived from the sealed originals (task 676, plan.md ruling 18).

Real pages: `tests/fixtures/payload_profile_reconcile/page-date-rows-fixture.jsonl.gz` holds 13 pages read
from the sealed originals of the 2026-09-28 daily stage (7 swissinfo, 2 RSI, 4 SRF),
each with the `published` the pipeline stamped on its row at the time (the sitemap
`lastmod` for swissinfo and RSI). They are laid out as a work dir the way the stage
writes it (`stage-<day>/news_documents.jsonl` plus `pages/original/.../<sha>.html.gz`)
and a fake twin holds two stored points per url, so the roster, the latest-capture
rule, the canonical-url join, the terminal tables and the explicit-`after` write all
run for real.

What each group would have caught:
* the join is the url, not the staging point id (measured live: 123,400 of 123,400
  staging ids were absent from the twin, which a run keyed on them reported as
  `absent` and never rewrote);
* every url and every stored row reaches exactly one terminal;
* the write sets `after` only where the store still holds the planned `before`.
"""
from __future__ import annotations

import copy
import json
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent

from cytir.index import payload_profile_reconcile as REC  # noqa: E402
import cytir.news.published_page_date as PPD  # noqa: E402
from cytir.index import payload_profile as PP  # noqa: E402

from published_page_date_world import (CHUNKS, TWIN, TWIN_URL, FakeTwin, _pages,  # noqa: E402,F401
                                      _staging_pid, _write_spool)


@pytest.fixture
def world(tmp_path):
    pages = _pages()
    work = tmp_path / "work"
    _write_spool(work, "2026-09-28", pages)
    return work, pages


def _plan(work, twin, **kw):
    return PPD.plan_from_pages(work, TWIN_URL, TWIN, api=twin.api, **kw)


def test_the_staging_ids_of_the_documents_are_not_twin_ids(world):
    work, pages = world
    twin = FakeTwin(TWIN, pages)
    staging = {_staging_pid(r["url"]) for r in pages}
    assert staging.isdisjoint(twin.points), "the live measurement: 0 of 123,400 matched"
    assert _plan(work, twin)["selected_rows"] > 0, "the url join still finds every row"


def test_the_roster_is_globbed_from_the_work_dir_and_printed_with_the_plan(world):
    work, pages = world
    _write_spool(work, "2026-09-29", pages[:2], fetched_at="2026-09-29T07:00:00Z")
    plan = _plan(work, FakeTwin(TWIN, pages))
    assert plan["roster"] == ["stage-2026-09-28", "stage-2026-09-29"]
    assert plan["roster_size"] == 2


def test_every_url_and_every_stored_row_reaches_exactly_one_terminal(world):
    work, pages = world
    plan = _plan(work, FakeTwin(TWIN, pages))
    u, r = plan["url_terminals"], plan["row_terminals"]
    assert u == {"urls": 13, "dated": 11, "page_none": 2}
    assert r == {"rows": 26, "changed": 14, "same": 8, "page_none": 4}
    assert plan["url_terminals_by_domain"]["swissinfo.ch"] == {"urls": 7, "dated": 7}
    assert plan["row_terminals_by_domain"]["rsi.ch"] == {"rows": 4, "changed": 4}
    assert plan["row_terminals_by_domain"]["srf.ch"] == {"rows": 8, "same": 4, "page_none": 4}
    assert plan["selected_rows"] == 14 and plan["mode"] == "explicit"


def test_every_stored_point_of_a_url_gets_the_pages_date_and_the_direction_is_reported(world):
    work, pages = world
    plan = _plan(work, FakeTwin(TWIN, pages))
    want = {r["url"]: r for r in pages}
    assert len(plan["rows"]) == 14
    per_url = {}
    for row in plan["rows"]:
        src = want[row["url"]]
        assert row["before"] == src["row_published"] and row["after"] == src["page_date"]
        per_url.setdefault(row["url"], set()).add(row["id"])
    assert all(len(ids) == CHUNKS for ids in per_url.values())
    assert plan["direction_of_change"] == {"lastmod_later": 10, "lastmod_earlier": 4}


def test_the_join_is_by_canonical_url_so_another_spelling_is_still_found(world):
    work, pages = world
    target = next(r for r in pages if r["kind"] == "pos")
    spelled = {target["url"]: target["url"].replace("https://www.", "http://")}
    assert spelled[target["url"]] != target["url"]
    plan = _plan(work, FakeTwin(TWIN, pages, spelled=spelled))
    assert {r["url"] for r in plan["rows"]} >= {spelled[target["url"]]}
    assert plan["row_terminals"]["changed"] == 14


def test_the_latest_capture_of_a_url_wins(world):
    work, pages = world
    target = next(r for r in pages if r["kind"] == "pos")
    other = next(r for r in pages if r["kind"] == "same" and r["page_date"] != target["page_date"])
    # an EARLIER capture of the same url with a different page body (another sha, the
    # date signals of a different real page) must not decide
    early = dict(target, original_sha256="ab" * 32, html_signals=other["html_signals"])
    _write_spool(work, "2026-09-01", [early], fetched_at="2026-09-01T07:00:00Z")
    plan = _plan(work, FakeTwin(TWIN, pages))
    assert plan["url_terminals"]["urls"] == 13
    assert plan["url_terminals"].get("recaptured") == 1
    assert plan["distinct_originals_parsed"] == 13, "the earlier body was never parsed"
    rows = [r for r in plan["rows"] if r["url"] == target["url"]]
    assert {r["after"] for r in rows} == {target["page_date"]} != {other["page_date"]}
    assert {r["original_sha256"] for r in rows} == {target["original_sha256"]}


def test_absent_and_unsealed_urls_have_their_own_terminal(tmp_path):
    pages = _pages()
    gone, lost = pages[0]["url"], pages[1]["url"]
    work = tmp_path / "work"
    _write_spool(work, "2026-09-28", pages, drop_original={lost})
    plan = _plan(work, FakeTwin(TWIN, pages, absent={gone}))
    u, r = plan["url_terminals"], plan["row_terminals"]
    assert u["absent_on_twin"] == 1 and u["original_missing"] == 1
    assert r["original_missing"] == CHUNKS and r["rows"] == 12 * CHUNKS


def test_a_row_with_no_published_is_filled_from_the_page(tmp_path):
    pages = _pages()
    target = dict(next(r for r in pages if r["kind"] == "pos"), row_published=None)
    work = tmp_path / "work"
    _write_spool(work, "2026-09-28", [target])
    plan = _plan(work, FakeTwin(TWIN, [target]))
    assert plan["row_terminals"]["filled"] == CHUNKS
    assert {r["before"] for r in plan["rows"]} == {None}
    assert {r["after"] for r in plan["rows"]} == {target["page_date"]}


def test_a_smoke_slice_plans_only_its_urls(world):
    work, pages = world
    plan = _plan(work, FakeTwin(TWIN, pages), limit_urls=5)
    assert plan["url_terminals"]["urls"] == 5


# ------------------------------------------------------------------------ apply
def _apply(rows, store, **kw):
    return REC.apply_published_leg(rows, TWIN_URL, TWIN, pause_s=0, api=store.api,
                                   fetch=store.fetch, explicit=True, **kw)


def test_apply_sets_the_page_date_only_where_the_store_holds_the_planned_value(world):
    work, pages = world
    store = FakeTwin(TWIN, pages)
    plan = _plan(work, store)
    before = copy.deepcopy(store.points)
    out = _apply(plan["rows"], store, batch=5)
    assert out == {"planned": 14, "rewritten": 14, "already": 0, "absent": 0,
                   "unreadable_kept": 0, "moved": 0, "points_count": 26}
    for row in plan["rows"]:
        assert store.points[row["id"]]["published"] == row["after"]
    for pid, pl in store.points.items():
        assert {k: v for k, v in pl.items() if k != "published"} == \
            {k: v for k, v in before[pid].items() if k != "published"}
    assert _apply(plan["rows"], store)["already"] == 14, "idempotent"


def test_a_row_that_changed_since_the_plan_is_counted_moved_and_left_alone(world):
    work, pages = world
    store = FakeTwin(TWIN, pages)
    plan = _plan(work, store)
    touched = plan["rows"][0]
    store.points[touched["id"]]["published"] = "2026-01-01"
    out = _apply(plan["rows"], store)
    assert out["moved"] == 1 and out["rewritten"] == 13
    assert store.points[touched["id"]]["published"] == "2026-01-01"
    assert out["planned"] == out["rewritten"] + out["already"] + out["absent"] + out["moved"]


def test_a_write_that_did_not_land_fails_the_readback(world):
    work, pages = world
    store = FakeTwin(TWIN, pages, ignore_writes=True)
    plan = _plan(work, store)
    with pytest.raises(PP.PayloadProfileError, match="readback mismatch"):
        _apply(plan["rows"], store)


def test_verify_rows_counts_equal_before_and_other(world):
    work, pages = world
    store = FakeTwin(TWIN, pages)
    plan = _plan(work, store)
    v = REC.verify_rows(TWIN_URL, TWIN, plan["rows"], n=100, seed=1, fetch=store.fetch)
    assert v["still_before"] == 14 and v.get("equal_after", 0) == 0
    _apply(plan["rows"], store)
    v = REC.verify_rows(TWIN_URL, TWIN, plan["rows"], n=100, seed=1, fetch=store.fetch)
    assert v["equal_after"] == 14 and v.get("still_before", 0) == 0


def test_the_verb_is_registered_and_takes_the_work_dir(monkeypatch, tmp_path, capsys, world):
    work, pages = world
    twin = FakeTwin(TWIN, pages)
    prod = FakeTwin("aw_e9", pages)
    stores = {"http://twin": twin, "http://prod": prod}
    monkeypatch.setattr(REC.PSP, "_api", lambda url, path, body=None, method=None:
                        stores[url].api(url, path, body, method))
    monkeypatch.setattr(REC.PD, "fetch_payload_fields",
                        lambda url, coll, ids, fields, batch=1000:
                        stores[url].fetch(url, coll, ids, fields))
    from cytir.index import promote_queue as Q
    monkeypatch.setattr(Q, "resolve_prod_alias", lambda url, alias="aw_prod": "aw_e9")
    rc = REC.main(["published-from-pages", "--twin-url", "http://twin",
                   "--twin-collection", TWIN, "--prod-url", "http://prod",
                   "--work", str(work), "--plan-out", str(tmp_path / "plan.json")])
    assert rc == 0
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["selected_rows"] == 14 and plan["roster"] == ["stage-2026-09-28"]
    assert '"still_before": 14' in capsys.readouterr().out
    assert twin.ops == [] and prod.ops == [], "a dry run writes nothing"
