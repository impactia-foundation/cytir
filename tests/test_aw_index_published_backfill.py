"""The `published` backfill (task 676): id selector, twin leg, prod leg.

Real rows, no live calls. `tests/fixtures/payload_profile_reconcile/published-rows-fixture.jsonl` holds 74
rows read from the twin on 2026-09-30: openai (44 RFC-822), Hugging Face ISO
datetimes, Swiss dd.mm.yyyy values, ISO dates (HN, swissinfo, blog.google) and bare
years (PubMed). They are loaded into a fake Qdrant that answers the five calls the
tool makes (facet, filtered scroll, retrieve by id, batched set_payload, collection
info) with Qdrant's own semantics, so the selector, the proof against the facet, the
payload-only write and the conservation identity run for real.

What each group would have caught:
* the selector is exact in both directions (a scroll that loses a row refuses);
* only `published` is written: every other key of every row is unchanged, no id moves,
  points_count is unchanged, the write is read back;
* a second run finds nothing to do; a row already fixed, absent or unreadable has its
  own terminal and planned == rewritten + already + absent + unreadable_kept;
* the promotion timer and the writer lock come back even when a leg fails (the health guard
  and the timer have their own test files).
"""
from __future__ import annotations

import copy
import json
import pathlib
import types

import pytest

HERE = pathlib.Path(__file__).resolve().parent

from cytir.index import payload_profile_reconcile as REC  # noqa: E402
from cytir.core.stopped_timer import StoppedTimer  # noqa: E402
from cytir.core.published import classify_published  # noqa: E402
from cytir.index import payload_profile as PP  # noqa: E402

FIXTURE = HERE / "fixtures" / "payload_profile_reconcile" / "published-rows-fixture.jsonl"
TWIN_URL, TWIN = "http://twin", "aw_e8"


def _rows():
    return [json.loads(line) for line in FIXTURE.read_text().splitlines()]


class FakeQdrant:
    """The subset of Qdrant the tool calls, with its semantics (exact keyword match,
    `match any`, ordered scroll with a next_page_offset, set_payload merge)."""

    def __init__(self, collection, rows, *, drop_from_scroll=(), ignore_writes=False,
                 page=10, absent=()):
        self.collection = collection
        self.points = {r["id"]: copy.deepcopy(r["payload"]) for r in rows
                       if r["id"] not in absent}
        self.drop, self.ignore_writes, self.page = set(drop_from_scroll), ignore_writes, page
        self.ops = []

    def api(self, url, path, body=None, method=None):
        base = f"/collections/{self.collection}"
        if path == base:
            return {"points_count": len(self.points)}
        if path == base + "/facet":
            assert body["key"] == "published" and body["exact"] is True
            hits: dict = {}
            for pl in self.points.values():
                if pl.get("published") is not None:
                    hits[pl["published"]] = hits.get(pl["published"], 0) + 1
            return {"hits": [{"value": v, "count": n} for v, n in sorted(hits.items())]}
        if path == base + "/points/scroll":
            (cond,) = body["filter"]["must"]
            assert cond["key"] == "published" and "any" in cond["match"]
            ids = sorted(i for i, pl in self.points.items()
                         if pl.get("published") in cond["match"]["any"] and i not in self.drop)
            start = ids.index(body["offset"]) if body.get("offset") in ids else 0
            chunk = ids[start:start + self.page]
            nxt = ids[start + self.page] if start + self.page < len(ids) else None
            return {"points": [{"id": i, "payload": {"published": self.points[i]["published"]}}
                               for i in chunk], "next_page_offset": nxt}
        if path == base + "/points/batch?wait=true":
            for op in body["operations"]:
                assert list(op) == ["set_payload"], "payload-only: no vector, no overwrite"
                spec = op["set_payload"]
                assert list(spec["payload"]) == ["published"], "only `published` is written"
                self.ops.append(op)
                if not self.ignore_writes:
                    for pid in spec["points"]:
                        self.points[pid].update(spec["payload"])
            return {}
        raise AssertionError(path)

    def fetch(self, url, collection, ids, fields, batch=1000):
        out = {}
        for i in ids:
            if i in self.points:
                pl = self.points[i]
                out[i] = copy.deepcopy(pl) if fields is True else {
                    k: pl[k] for k in fields if k in pl}
        return out


def _twin(**kw):
    return FakeQdrant(TWIN, _rows(), **kw)


def _plan(store):
    return REC.plan_published(TWIN_URL, TWIN, api=store.api, fetch=store.fetch)


def test_the_fixture_is_the_real_mix_of_formats():
    classes = {}
    for r in _rows():
        c = classify_published(r["payload"].get("published"))
        classes[c] = classes.get(c, 0) + 1
    assert classes == {"rfc822": 44, "iso_date": 18, "dotted_dmy": 6,
                       "iso_datetime": 3, "iso_year": 3}


def test_classify_facet_selects_exactly_the_non_contract_classes():
    store = _twin()
    cls = REC.classify_facet(REC.facet_published(TWIN_URL, TWIN, api=store.api))
    by_class = {}
    for value, (klass, n) in cls["rewrite"].items():
        by_class[klass] = by_class.get(klass, 0) + n
    assert by_class == {"rfc822": 44, "dotted_dmy": 6, "iso_datetime": 3}
    assert cls["classes"]["iso_date"]["rows"] == 18 and cls["classes"]["iso_year"]["rows"] == 3
    assert cls["unreadable"] == {}


def test_the_plan_lists_every_row_with_its_before_and_after():
    plan = _plan(_twin())
    assert plan["selected_rows"] == plan["expected_rows"] == 53
    assert plan["rows_by_class"] == {"rfc822": 44, "dotted_dmy": 6, "iso_datetime": 3}
    assert plan["facet_rows"] == 74 and len(plan["ids_sha256"]) == 64
    for row in plan["rows"]:
        assert classify_published(row["before"]) in {"rfc822", "dotted_dmy", "iso_datetime"}
        assert classify_published(row["after"]) == "iso_date"
    pairs = {(r["before"], r["after"]) for r in plan["rows"]}
    assert ("17.03.2015", "2015-03-17") in pairs
    assert any(b.startswith("Wed, 16 Sep 2026") and a == "2026-09-16" for b, a in pairs)
    assert plan["sample_checked"] == 53 and plan["sample_only_published_moves"] == 53


def test_the_selection_is_proved_against_the_facet_so_a_lost_row_refuses():
    victim = next(r["id"] for r in _rows()
                  if classify_published(r["payload"]["published"]) == "dotted_dmy")
    with pytest.raises(PP.PayloadProfileError, match="selection is not exact"):
        _plan(_twin(drop_from_scroll={victim}))


def test_the_plan_refuses_a_projection_that_moves_more_than_published(monkeypatch):
    real = PP.Projector.project

    def greedy(self, payload):
        out = real(self, payload)
        out["snippet"] = "changed"
        return out
    monkeypatch.setattr(PP.Projector, "project", greedy)
    with pytest.raises(PP.PayloadProfileError, match="more than `published`"):
        _plan(_twin())


def test_apply_writes_only_published_and_nothing_else_moves():
    store = _twin()
    before = copy.deepcopy(store.points)
    plan = _plan(store)
    out = REC.apply_published_leg(plan["rows"], TWIN_URL, TWIN, pause_s=0, batch=7,
                                  api=store.api, fetch=store.fetch)
    assert out == {"planned": 53, "rewritten": 53, "already": 0, "absent": 0,
                   "unreadable_kept": 0, "points_count": 74}
    assert set(store.points) == set(before), "no id added or removed"
    moved = 0
    for pid, pl in store.points.items():
        rest_now = {k: v for k, v in pl.items() if k != "published"}
        rest_then = {k: v for k, v in before[pid].items() if k != "published"}
        assert rest_now == rest_then, pid
        moved += pl.get("published") != before[pid].get("published")
    assert moved == 53
    assert all(classify_published(pl["published"]) in {"iso_date", "iso_year"}
               for pl in store.points.values())


def test_a_second_run_finds_nothing_to_do_and_the_identity_still_balances():
    store = _twin()
    plan = _plan(store)
    REC.apply_published_leg(plan["rows"], TWIN_URL, TWIN, pause_s=0, api=store.api,
                            fetch=store.fetch)
    n_ops = len(store.ops)
    again = REC.apply_published_leg(plan["rows"], TWIN_URL, TWIN, pause_s=0,
                                    api=store.api, fetch=store.fetch)
    assert again["rewritten"] == 0 and again["already"] == 53 and len(store.ops) == n_ops


def test_the_prod_leg_reads_each_rows_current_value_and_accounts_for_absent_ids():
    plan = _plan(_twin())
    gone = {plan["rows"][0]["id"], plan["rows"][1]["id"]}
    prod = FakeQdrant("aw_e9", _rows(), absent=gone)
    # one row already fixed on prod since the plan was cut
    fixed = plan["rows"][2]
    prod.points[fixed["id"]]["published"] = fixed["after"]
    out = REC.apply_published_leg(plan["rows"], "http://prod", "aw_e9", pause_s=0,
                                  api=prod.api, fetch=prod.fetch)
    assert out["absent"] == 2 and out["already"] == 1 and out["rewritten"] == 50
    assert out["planned"] == 53 == sum(out[k] for k in
                                       ("rewritten", "already", "absent", "unreadable_kept"))


def test_an_unreadable_value_is_kept_as_written_and_counted():
    rows = _rows()
    rows[0]["payload"]["published"] = "sometime last spring"
    store = FakeQdrant(TWIN, rows)
    plan_rows = [{"id": rows[0]["id"], "before": "sometime last spring", "after": None}]
    out = REC.apply_published_leg(plan_rows, TWIN_URL, TWIN, pause_s=0, api=store.api,
                                  fetch=store.fetch)
    assert out["unreadable_kept"] == 1 and out["rewritten"] == 0
    assert store.points[rows[0]["id"]]["published"] == "sometime last spring"
    assert REC.classify_facet(REC.facet_published(TWIN_URL, TWIN, api=store.api)
                              )["unreadable"] == {"sometime last spring": 1}


def test_a_write_that_did_not_land_fails_the_readback():
    store = _twin(ignore_writes=True)
    plan = _plan(store)
    with pytest.raises(PP.PayloadProfileError, match="readback mismatch"):
        REC.apply_published_leg(plan["rows"], TWIN_URL, TWIN, pause_s=0, api=store.api,
                                fetch=store.fetch)


def test_a_moving_point_count_aborts():
    store = _twin()
    plan = _plan(store)
    real = store.api
    calls = {"n": 0}

    def api(url, path, body=None, method=None):
        if path == f"/collections/{TWIN}":
            calls["n"] += 1
            return {"points_count": 74 if calls["n"] == 1 else 75}
        return real(url, path, body, method)
    with pytest.raises(PP.PayloadProfileError, match="points_count moved"):
        REC.apply_published_leg(plan["rows"], TWIN_URL, TWIN, pause_s=0, api=api,
                                fetch=store.fetch)


def test_verify_published_counts_format_classes_by_facet_and_by_id():
    store = _twin()
    facet = REC.verify_published(TWIN_URL, TWIN, api=store.api)
    assert facet["non_contract_rows"] == 53 and facet["rows_with_published"] == 74
    plan = _plan(store)
    REC.apply_published_leg(plan["rows"], TWIN_URL, TWIN, pause_s=0, api=store.api,
                            fetch=store.fetch)
    assert REC.verify_published(TWIN_URL, TWIN, api=store.api)["non_contract_rows"] == 0
    ids = [r["id"] for r in plan["rows"]][:20] + ["00000000-0000-0000-0000-000000000000"]
    by_id = REC.verify_published(TWIN_URL, TWIN, ids=ids, api=store.api, fetch=store.fetch)
    assert by_id["classes"] == {"iso_date": 20, "absent": 1} and by_id["non_contract"] == 0


# ------------------------------------------------------------------ guard, timer, lock
# ------------------------------------------------------------------ lock, CLI
def _timer_run(active, calls):
    def run(*cmd):
        calls.append(cmd)
        out = "active\n" if (cmd[1] == "is-active" and active) else (
            "inactive\n" if cmd[1] == "is-active" else "")
        return types.SimpleNamespace(stdout=out, stderr="", returncode=0)
    return run



def _cli_world(monkeypatch, tmp_path, *, prod_ignores_writes=False):
    """main(["published", ...]) with the twin and prod both fake, the lock, the alias
    resolver, the guard and the timer stubbed."""
    twin = _twin()
    prod = FakeQdrant("aw_e9", _rows(), ignore_writes=prod_ignores_writes)
    stores = {"http://twin": twin, "http://prod": prod}
    order = []

    def api(url, path, body=None, method=None):
        if path.endswith("/points/batch?wait=true"):   # task 2786: set_payload_batch sends no explicit method
            order.append(url)
        return stores[url].api(url, path, body, method)

    def fetch(url, coll, ids, fields, batch=1000):
        return stores[url].fetch(url, coll, ids, fields)
    monkeypatch.setattr(REC.PSP, "_api", api)
    monkeypatch.setattr(REC.PD, "fetch_payload_fields", fetch)
    from cytir.index import promote_queue as Q
    from cytir.index import twin_first as TFA
    monkeypatch.setattr(Q, "resolve_prod_alias", lambda url, alias="aw_prod": "aw_e9")
    monkeypatch.setattr(Q, "_revalidate_prod_collection", lambda *a, **k: None)
    monkeypatch.setattr(Q, "prod_lock_host", lambda: "prodbox")   # no operator overlay needed
    events = []

    class Lock:
        def __init__(self, **kw):
            events.append("lock-new")

        def acquire(self):
            events.append("lock-acquire")

        def release(self):
            events.append("lock-release")
    monkeypatch.setattr(TFA, "ProductionWriterLock", Lock)
    monkeypatch.setattr(REC.BoxHealthGuard, "start", lambda self: {"mem_available_gb": 20})
    monkeypatch.setattr(REC.BoxHealthGuard, "tick", lambda self: None)
    calls = []
    monkeypatch.setattr(REC, "StoppedTimer", lambda unit, **kw: _RealTimer(unit, calls))
    return twin, prod, order, events, calls


class _RealTimer(StoppedTimer):
    def __init__(self, unit, calls):
        super().__init__(unit, run=_timer_run(True, calls))


def _argv(tmp_path, *extra):
    return ["published", "--twin-url", "http://twin", "--twin-collection", TWIN,
            "--guard-container", "qdrant-test", "--guard-search-url", "http://prod.invalid/s",
            "--guard-ssh-host", "prodbox",
            "--prod-url", "http://prod", "--plan-out", str(tmp_path / "plan.json"),
            "--state-out", str(tmp_path / "state.json"), "--pause-s", "0", *extra]


def test_cli_dry_run_writes_the_plan_and_no_store(monkeypatch, tmp_path, capsys):
    twin, prod, order, events, calls = _cli_world(monkeypatch, tmp_path)
    assert REC.main(_argv(tmp_path)) == 0
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["selected_rows"] == 53 and len(plan["rows"]) == 53
    assert order == [] and events == [] and calls == []
    assert "prod_probe_before" in capsys.readouterr().out


def test_cli_apply_is_twin_first_then_prod_under_the_lock_with_the_timer_off(
        monkeypatch, tmp_path):
    twin, prod, order, events, calls = _cli_world(monkeypatch, tmp_path)
    rc = REC.main(_argv(tmp_path, "--apply", "--stop-timer", "aw-promotion-nightly.timer"))
    assert rc == 0
    assert order and order[0] == "http://twin" and order[-1] == "http://prod"
    assert order.index("http://prod") > max(i for i, u in enumerate(order) if u == "http://twin")
    assert events == ["lock-new", "lock-acquire", "lock-release"]
    assert [c[1] for c in calls] == ["is-active", "stop", "start"]
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["legs"]["twin"]["rewritten"] == 53 and state["legs"]["prod"]["rewritten"] == 53
    assert state["twin_after"]["non_contract_rows"] == 0
    assert state["prod_probe_after"]["non_contract"] == 0 and state["timer_restored"] is True


def test_cli_a_failing_prod_leg_still_releases_the_lock_and_restarts_the_timer(
        monkeypatch, tmp_path):
    twin, prod, order, events, calls = _cli_world(monkeypatch, tmp_path,
                                                  prod_ignores_writes=True)
    with pytest.raises(PP.PayloadProfileError, match="readback mismatch"):
        REC.main(_argv(tmp_path, "--apply", "--stop-timer", "aw-promotion-nightly.timer"))
    assert events[-1] == "lock-release"
    assert [c[1] for c in calls] == ["is-active", "stop", "start"]
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["legs"]["twin"]["rewritten"] == 53 and "prod" not in state["legs"]
    assert "readback mismatch" in state["stopped"]
    # the twin leg is not undone and a rerun resumes on prod alone
    assert all(classify_published(pl["published"]) in {"iso_date", "iso_year"}
               for pl in twin.points.values())


def test_cli_apply_refuses_without_the_guard_targets_before_any_read(monkeypatch, tmp_path):
    """Task 1247: no built-in container or search URL; with neither flag nor setting, --apply
    refuses before the plan reads a store."""
    from cytir.core import paths as P
    twin, prod, order, events, calls = _cli_world(monkeypatch, tmp_path)
    monkeypatch.setattr(P, "_cached", P.load_settings(env={}, overlay=None))
    argv = [x for x in _argv(tmp_path, "--apply") if x not in (
        "--guard-container", "qdrant-test", "--guard-search-url", "http://prod.invalid/s")]
    with pytest.raises(SystemExit, match="prod_qdrant_container.*prod_health_search_url"):
        REC.main(argv)
    assert order == [] and events == [] and calls == []


def test_cli_guard_targets_come_from_the_operator_settings(monkeypatch, tmp_path):
    from cytir.core import paths as P
    _cli_world(monkeypatch, tmp_path)
    monkeypatch.setattr(P, "_cached", P.load_settings(env={
        "AW_PROD_LOCK_HOST": "box", "AW_PROD_QDRANT_CONTAINER": "q1",
        "AW_PROD_HEALTH_SEARCH_URL": "http://h/s"}, overlay=None))
    a = types.SimpleNamespace(guard_ssh_host="", guard_container="", guard_search_url="")
    g = REC._health_guard(a)
    assert (g.ssh_host, g.container, g.search_url) == ("box", "q1", "http://h/s")
    a.guard_container = "flag-wins"
    assert REC._health_guard(a).container == "flag-wins"


def test_cli_refuses_a_plan_for_another_collection(monkeypatch, tmp_path):
    _cli_world(monkeypatch, tmp_path)
    (tmp_path / "other.json").write_text(json.dumps(
        {"schema": REC.PUBLISHED_SCHEMA, "collection": "aw_x", "rows": []}))
    with pytest.raises(SystemExit, match="not a plan for"):
        REC.main(_argv(tmp_path, "--plan-in", str(tmp_path / "other.json")))
