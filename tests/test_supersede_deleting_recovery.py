"""Task 2364 (F1 + F2 of the #1753 review): a url-group supersede state left at `deleting` is decided
by a prod lookup BY ID, not refused forever. Real loda rows (tests/fixtures), the in-memory qdrant
of `test_aw_url_group_supersede` (it fails on a scroll of prod and on any unstubbed call)."""
from __future__ import annotations

import json

import pytest

import test_aw_url_group_supersede as T
from cytir.index import change_log as CL
from cytir.index import point_delete as PD
from cytir.index import promote_queue as Q

settings = T.settings
_stage_root = T._stage_root
SHA = "a" * 64
STATE = "loda-refresh-url-group-supersede.json"


@pytest.fixture
def world(monkeypatch, tmp_path, settings):
    """The twin leg ran, the prod delete died: state `deleting`, prod's export and `to_delete`
    (3 ids) recorded, prod still holding all three. Returns (cluster, manifest, root, state)."""
    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", T.FakeLock)
    monkeypatch.setattr(Q, "_revalidate_prod_collection", lambda *a, **k: None)

    class RB:
        from cytir.ops.write_lock import LOCKS
        assert_locks_free = staticmethod(lambda task: {})
        acquire = staticmethod(lambda *a, **k: None)
        release = staticmethod(lambda *a, **k: None)
    monkeypatch.setattr(Q, "_load_run_batch", lambda: RB)
    rows = T.served(*T.OLD_A[:2], T.NEW_A)
    c, j, man, sp = T.live_with_prod(monkeypatch, tmp_path, rows)
    real_delete = T.die_on_prod_delete(monkeypatch)
    with pytest.raises(RuntimeError, match="prod went away"):
        T.run_live(sp, man, tmp_path)
    monkeypatch.setattr(PD, "delete_points_verified", real_delete)
    root = tmp_path / "artifacts"
    state = json.loads((root / STATE).read_text())
    assert state["stage"] == "deleting" and len(state["prod"]["to_delete"]) == 3
    return c, man, root, state


def settle(man, root, *, apply, sha=SHA):
    return Q.settle_supersede_deleting(
        man, sha, root, twin_url="http://twin", prod_url="http://prod", prod_collection="aw_e8",
        prod_alias="aw_prod", batch=10, task="2364-test", apply=apply)


def stage(root):
    return json.loads((root / STATE).read_text())["stage"]


def prod_deletes(c):
    return [x for x in c.calls if x == ("http://prod", "delete")]


def test_a_none_of_to_delete_present_marks_applied_with_no_delete(world):
    c, man, root, state = world
    for i in state["prod"]["to_delete"]:
        c.colls["http://prod"].pop(i)
    before = len(prod_deletes(c))
    rec = settle(man, root, apply=True)
    assert rec["decision"] == "marked-applied" and rec["present_on_prod"] == 0
    assert len(prod_deletes(c)) == before and stage(root) == "applied"
    assert settle(man, root, apply=True) is None            # the next drain is clean
    # and the change log's tombstones are the ids the earlier delete removed
    assert [t["id"] for t in Q._supersede_tombstones_on_resume(man, SHA, root)] \
        == sorted(state["prod"]["to_delete"])


def test_b_some_present_deletes_exactly_those_ids_and_receipts_it(world, monkeypatch):
    c, man, root, state = world
    gone, *left = state["prod"]["to_delete"]
    c.colls["http://prod"].pop(gone)
    seen = []
    real = PD.delete_points_verified

    def spy(url, coll, ids, **k):
        if url == "http://prod":
            seen.append(sorted(ids))
        return real(url, coll, ids, **k)
    monkeypatch.setattr(PD, "delete_points_verified", spy)
    rec = settle(man, root, apply=True)
    assert seen == [sorted(left)]
    assert rec["decision"] == "deleted-present" and rec["deleted"] == 2 and rec["present_on_prod"] == 2
    assert rec["state_after"] == "applied" and stage(root) == "applied"
    assert not set(state["prod"]["to_delete"]) & set(c.ids("http://prod"))
    saved = json.loads((root / STATE).read_text())
    assert saved["prod"]["to_delete"] == sorted(state["prod"]["to_delete"])    # the full set is kept


def test_c_dry_run_reads_only_writes_nothing(world):
    c, man, root, state = world
    raw = (root / STATE).read_bytes()
    calls = list(c.calls)
    rec = settle(man, root, apply=False)
    assert rec["decision"] == "would-delete" and rec["present_on_prod"] == 3
    assert (root / STATE).read_bytes() == raw and c.calls == calls
    for i in state["prod"]["to_delete"]:
        c.colls["http://prod"].pop(i)
    assert settle(man, root, apply=False)["decision"] == "would-mark-applied"
    assert stage(root) == "deleting"


def test_d_unreadable_state_is_refused_and_nothing_is_written(world):
    c, man, root, state = world
    (root / STATE).write_text("{not json")
    calls = list(c.calls)
    with pytest.raises(CL.ChangeLogError, match="unreadable"):
        settle(man, root, apply=True)
    assert (root / STATE).read_text() == "{not json" and c.calls == calls


def test_e_state_before_prod_started_checks_all_ids_by_id(world):
    c, man, root, state = world
    st = json.loads((root / STATE).read_text())
    st["prod"] = None
    (root / STATE).write_text(json.dumps(st))
    for i in st["ids"]:
        c.colls["http://prod"].pop(i, None)
    rec = settle(man, root, apply=True)
    assert rec["decision"] == "marked-applied" and rec["checked"] == len(st["ids"])
    assert stage(root) == "applied"


def test_f_not_a_supersede_pack_and_a_reseal(world):
    c, man, root, state = world
    assert settle({"pack_id": "x"}, root, apply=True) is None
    with pytest.raises(CL.ChangeLogError):      # a reseal of a pack that began deleting still refuses
        settle(man, root, apply=True, sha="b" * 64)


def _twin_landed_prod_died(tmp_path, monkeypatch, settings):
    """First drain: the twin leg lands (state `deleting`), the prod upsert dies. The second drain then
    sees the pack as already landed on the twin (`dry_plan` raises, as it does for a non-supersede pack)."""
    urls = T.real_pack_urls(monkeypatch)
    T.give_the_pack_a_family(monkeypatch)
    old = T.old_row_at(sorted(urls.values())[0])
    c = T.make_cluster(monkeypatch, [old])
    queue, contract = T.drain_fixture(tmp_path, monkeypatch, settings, supersede="url-group")
    fails = [True]
    T.wire_real_drain(monkeypatch, c, prod_upsert_fails=fails)
    with pytest.raises(RuntimeError, match="prod went away"):
        T.run_drain(queue, contract, tmp_path)
    assert old["id"] in c.ids("http://prod") and old["id"] not in c.ids("http://twin")
    fails[0] = False
    monkeypatch.setattr(Q.PSP, "already_present",
                        lambda url, coll, ids, **k: [i for i in ids if i in c.colls[url]])
    landed = frozenset(urls)

    def already_landed(*a, **k):
        raise Q.PSP.AlreadyLandedError("twin holds the pack", ids=landed, present=landed)
    monkeypatch.setattr(Q, "dry_plan", already_landed)
    return c, urls, old, queue, contract


@T.needs_pack
def test_g_the_prod_only_resume_still_refuses_a_supersede_pack_and_deletes_nothing(tmp_path, monkeypatch, settings):
    """F2 as found: for a pack that declares `allow_supersede` the prod-only resume refuses in
    `apply_side` before any write (so it cannot reach DONE with old rows on prod). Pinned here; the
    helper is wired after that leg anyway, so a later change to the refusal cannot reopen the hole."""
    c, urls, old, queue, contract = _twin_landed_prod_died(tmp_path, monkeypatch, settings)
    rec = T.run_drain(queue, contract, tmp_path)[0]
    assert rec["status"] == "REFUSED" and "reconcile by hand" in rec["reason"]
    assert old["id"] in c.ids("http://prod") and not (queue / "loda-refresh" / "DONE").exists()
    assert stage(tmp_path / "artifacts") == "deleting"


@T.needs_pack
def test_h_the_already_held_exit_settles_deleting_before_done(tmp_path, monkeypatch, settings):
    """F1 at the drain: both legs hold the pack's rows, the old row is still on prod, state `deleting`.
    The exit used to refuse on every drain; it now deletes exactly that row and reaches DONE."""
    c, urls, old, queue, contract = _twin_landed_prod_died(tmp_path, monkeypatch, settings)
    for i in urls:                       # prod got the pack's rows, the delete never ran
        c.colls["http://prod"][i] = dict(c.colls["http://twin"][i])
    rec = T.run_drain(queue, contract, tmp_path)[0]
    assert rec["status"] == "LANDED", rec
    assert rec["supersede_recovery"]["decision"] == "deleted-present"
    assert rec["supersede_recovery"]["present_on_prod"] == 1
    assert old["id"] not in c.ids("http://prod") and set(urls) <= c.ids("http://prod")
    assert (queue / "loda-refresh" / "DONE").exists()
    assert stage(tmp_path / "artifacts") == "applied"
    assert rec["supersede_recovery"]["present_on_twin"] == 0


# ------------------------------------------------------------------ review round 2 (PR 1768)
@pytest.mark.parametrize("answer", [None, {}], ids=["none", "empty-dict"])
def test_1_a_lookup_without_a_result_list_refuses_and_keeps_deleting(world, monkeypatch, answer):
    """A 200 with no `result` is not 'prod holds none': it must not mark the state applied."""
    c, man, root, state = world
    real = c.api

    def api(url, path, body=None, method=None):
        if url == "http://prod" and path.endswith("/points") and body and "ids" in body:
            return answer
        return real(url, path, body, method)
    monkeypatch.setattr(Q.PSP, "_api", api)
    with pytest.raises(Q.QueueError, match="not a list"):
        settle(man, root, apply=True)
    assert stage(root) == "deleting" and len(c.ids("http://prod")) == 4


def test_1b_a_lookup_that_returns_an_id_not_asked_for_refuses(world, monkeypatch):
    c, man, root, state = world
    real = c.api

    def api(url, path, body=None, method=None):
        if url == "http://prod" and path.endswith("/points") and body and "ids" in body:
            return [{"id": "00000000-0000-5000-8000-0000000000ff"}]
        return real(url, path, body, method)
    monkeypatch.setattr(Q.PSP, "_api", api)
    with pytest.raises(Q.QueueError, match="not asked for"):
        settle(man, root, apply=True)
    assert stage(root) == "deleting"


def test_2_the_twin_still_holding_a_superseded_row_refuses_before_any_prod_delete(world):
    c, man, root, state = world
    back = state["ids"][0]
    c.colls["http://twin"][back] = dict(c.colls["http://prod"][back])    # the twin delete died halfway
    state["prod"] = None
    (root / STATE).write_text(json.dumps(state))
    calls = list(c.calls)
    with pytest.raises(Q.QueueError, match="twin delete is unfinished"):
        settle(man, root, apply=True)
    assert c.calls == calls and stage(root) == "deleting" and len(c.ids("http://prod")) == 4


def _both_hold_the_pack(tmp_path, monkeypatch, settings):
    c, urls, old, queue, contract = _twin_landed_prod_died(tmp_path, monkeypatch, settings)
    for i in urls:
        c.colls["http://prod"][i] = dict(c.colls["http://twin"][i])
    return c, urls, old, queue, contract


@T.needs_pack
def test_3a_a_refusal_in_the_exit_is_a_receipt_and_the_drain_goes_on(tmp_path, monkeypatch, settings):
    c, urls, old, queue, contract = _both_hold_the_pack(tmp_path, monkeypatch, settings)
    c.colls["http://twin"][old["id"]] = dict(c.colls["http://prod"][old["id"]])   # twin delete unfinished
    rec = T.run_drain(queue, contract, tmp_path)[0]          # returns: the drain did not crash
    assert rec["status"] == "REFUSED" and rec["stage"] == "supersede-settle"
    saved = json.loads((queue / "loda-refresh" / "receipt.json").read_text())
    assert saved["status"] == "REFUSED" and "twin delete is unfinished" in saved["reason"]
    assert not (queue / "loda-refresh" / "DONE").exists() and stage(tmp_path / "artifacts") == "deleting"


@T.needs_pack
def test_3b_a_failure_after_a_delete_is_an_aborted_receipt_and_stops_the_drain(tmp_path, monkeypatch, settings):
    c, urls, old, queue, contract = _both_hold_the_pack(tmp_path, monkeypatch, settings)

    def die(url, coll, ids, **k):
        raise RuntimeError("prod went away mid-delete")
    monkeypatch.setattr(PD, "delete_points_verified", die)
    with pytest.raises(RuntimeError, match="mid-delete"):
        T.run_drain(queue, contract, tmp_path)
    saved = json.loads((queue / "loda-refresh" / "receipt.json").read_text())
    assert saved["status"] == "ABORTED" and saved["stage"] == "supersede-settle"
    assert not (queue / "loda-refresh" / "DONE").exists() and stage(tmp_path / "artifacts") == "deleting"


@T.needs_pack
def test_4_the_dry_run_shows_the_decision_and_writes_nothing(tmp_path, monkeypatch, settings):
    c, urls, old, queue, contract = _both_hold_the_pack(tmp_path, monkeypatch, settings)
    calls = list(c.calls)
    rec = T.run_drain(queue, contract, tmp_path, apply=False)[0]
    assert rec["status"] == "PROVED-DRY-RUN", rec
    assert rec["would_supersede_recovery"]["decision"] == "would-delete"
    assert "delete the 1 superseded row" in rec["would"]
    assert c.calls == calls and old["id"] in c.ids("http://prod")
    assert stage(tmp_path / "artifacts") == "deleting" and not (queue / "loda-refresh" / "DONE").exists()


@T.needs_pack
def test_5_an_alias_move_at_the_exit_is_an_aborted_receipt_then_the_drain_stops(tmp_path, monkeypatch, settings):
    """`ApplyAbort` is a SystemExit. The alias re-read raising it must still leave a receipt."""
    c, urls, old, queue, contract = _both_hold_the_pack(tmp_path, monkeypatch, settings)

    def moved(*a, **k):
        raise Q.TFA.ApplyAbort("aw_prod moved during the drain")
    monkeypatch.setattr(Q, "_revalidate_prod_collection", moved)
    with pytest.raises(Q.TFA.ApplyAbort):
        T.run_drain(queue, contract, tmp_path)
    saved = json.loads((queue / "loda-refresh" / "receipt.json").read_text())
    assert saved["status"] == "ABORTED" and saved["stage"] == "supersede-settle"
    assert "aw_prod moved" in saved["reason"] and "marking DONE" not in saved["note"]
    assert not (queue / "loda-refresh" / "DONE").exists() and stage(tmp_path / "artifacts") == "deleting"


# ------------------------------------------------------------------ task 2929: pins before the settle moves to drain/resume.py
def _recording(monkeypatch):
    """The settle's protocol as events: the lock object it builds, the runner calls, the alias re-checks."""
    import os
    ev = []

    class RB:
        from cytir.ops.write_lock import LOCKS

        @staticmethod
        def assert_locks_free(task):
            ev.append(("free", task))
            return {}

        @staticmethod
        def acquire(name, task, why, *, pid=None):
            ev.append(("acquire", name, task, why, pid == os.getpid()))

        @staticmethod
        def release(name, task):
            ev.append(("release", name, task))

    class Lock:
        def __init__(self, **kw):
            ev.append(("lock", kw))

        def acquire(self):
            ev.append("prod-lock-acquire")

        def release(self):
            ev.append("prod-lock-release")
    monkeypatch.setattr(Q, "_load_run_batch", lambda: RB)
    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", Lock)
    monkeypatch.setattr(Q, "prod_lock_host", lambda: "a-host")
    monkeypatch.setattr(Q, "_revalidate_prod_collection", lambda *a, **k: ev.append(("alias", a)))
    return ev


def _spy_prod_delete(monkeypatch, ev, boom=False):
    real = PD.delete_points_verified

    def spy(url, coll, ids, **k):
        if url == "http://prod":
            ev.append(("delete", coll, sorted(ids), k.get("batch")))
            if boom:
                raise RuntimeError("prod went away again")
        return real(url, coll, ids, **k)
    monkeypatch.setattr(PD, "delete_points_verified", spy)


def _settle_apply(man, root, wa=None, *, apply=True):
    return Q.settle_supersede_deleting(man, SHA, root, twin_url="http://twin", prod_url="http://prod",
                                       prod_collection="aw_e8", prod_alias="aw_prod", batch=10, task="2364-test",
                                       apply=apply, write_attempt=wa)


def test_2929_the_settle_that_deletes_takes_the_locks_in_order_and_rechecks_the_alias_inside_them(world, monkeypatch):
    c, man, root, state = world
    gone, *left = state["prod"]["to_delete"]
    c.colls["http://prod"].pop(gone)
    ev = _recording(monkeypatch)
    _spy_prod_delete(monkeypatch, ev)
    wa = Q.WriteAttempt()
    rec = _settle_apply(man, root, wa)
    alias = ("alias", (man["pack_id"], "http://prod", "aw_prod", "aw_e8"))
    why = f"{man['pack_id']}-supersede-settle"
    assert ev == [alias,
                  ("lock", {"host": "a-host", "lock_path": Q.PROD_LOCK_PATH,
                            "purpose": f"task2364-test-{man['pack_id']}-supersede-settle"}),
                  ("free", "2364-test"),
                  ("acquire", "TWIN-WRITE-LOCK", "2364-test", why, True),
                  ("acquire", "PROD-WRITE-LOCK", "2364-test", why, True),
                  "prod-lock-acquire", alias, ("delete", "aw_e8", sorted(left), 10), "prod-lock-release",
                  ("release", "PROD-WRITE-LOCK", "2364-test"), ("release", "TWIN-WRITE-LOCK", "2364-test")]
    assert wa.started is True and rec["decision"] == "deleted-present"
    assert rec["checked"] == 3 and rec["present_on_twin"] == 0 and rec["prod_collection"] == "aw_e8"
    assert rec["state_path"] == str(root / STATE) and rec["state_after"] == "applied"


def test_2929_a_settle_that_dies_in_the_delete_gives_the_locks_back_in_reverse_and_marks_the_attempt(world, monkeypatch):
    c, man, root, state = world
    ev = _recording(monkeypatch)
    _spy_prod_delete(monkeypatch, ev, boom=True)
    wa = Q.WriteAttempt()
    with pytest.raises(RuntimeError, match="prod went away again"):
        _settle_apply(man, root, wa)
    assert ev[-3:] == ["prod-lock-release", ("release", "PROD-WRITE-LOCK", "2364-test"),
                       ("release", "TWIN-WRITE-LOCK", "2364-test")]
    assert wa.started is True and stage(root) == "deleting"


def test_2929_a_settle_that_reads_only_takes_no_lock_and_still_checks_the_alias_once(world, monkeypatch):
    c, man, root, state = world
    ev = _recording(monkeypatch)
    rec = _settle_apply(man, root, apply=False)
    assert rec["decision"] == "would-delete" and rec["present_on_twin"] == 0
    assert ev == [("alias", (man["pack_id"], "http://prod", "aw_prod", "aw_e8"))]


def test_2929_a_settle_with_nothing_left_on_prod_takes_no_lock_and_issues_no_delete(world, monkeypatch):
    c, man, root, state = world
    for i in state["prod"]["to_delete"]:
        c.colls["http://prod"].pop(i)
    ev = _recording(monkeypatch)
    _spy_prod_delete(monkeypatch, ev)
    wa = Q.WriteAttempt()
    rec = _settle_apply(man, root, wa)
    assert ev == [("alias", (man["pack_id"], "http://prod", "aw_prod", "aw_e8"))]
    assert rec["decision"] == "marked-applied" and rec["state_after"] == "applied" and wa.started is False
    assert rec["present_on_twin"] == 0 and "deleted" not in rec


# ---------------------------------------------------------------- task 2933: the settle with PG.is_prod TRUE
def _prod_is_http_prod(monkeypatch):
    """The serving box answers `PG.is_prod` true. The in-memory qdrant replaces `PSP._api`, the door that
    asks the prod guard (`PG.check`) before every call; this wrapper asks it first, as the real door does, so
    a green run proves the settle's calls are ones the guard lets through to prod (no scroll, no filter).
    Returns the list of (method, path) the guard let through to prod."""
    from cytir.index import prod_guard as PG
    through = []
    monkeypatch.setattr(PG, "is_prod", lambda u, c=None: u == "http://prod")
    monkeypatch.setattr(PG, "PROD_BATCH_SLEEP_S", 0.0)
    PG.reset()
    for mod in (Q.PSP, PD):
        fake = mod._api

        def api(u, path, body=None, method=None, *a, _fake=fake, **k):
            if PG.check(u, method or ("POST" if body is not None else "GET"), path):
                through.append((method or ("POST" if body is not None else "GET"), path.split("?")[0]))
            return _fake(u, path, body, method, *a, **k)
        monkeypatch.setattr(mod, "_api", api)
    return through


def test_2933_the_settle_on_a_prod_target_deletes_exactly_the_present_ids_like_a_non_prod_one(world, monkeypatch):
    """Today's behaviour: the prod guard changes the lookup pacing and the worker cap, nothing the settle decides.
    The decision, the receipt, the state and the delete are those of test_b (the guard answered false there)."""
    through = _prod_is_http_prod(monkeypatch)
    c, man, root, state = world
    gone, *left = state["prod"]["to_delete"]
    c.colls["http://prod"].pop(gone)
    seen = []
    real = PD.delete_points_verified

    def spy(url, coll, ids, **k):
        if url == "http://prod":
            seen.append(sorted(ids))
        return real(url, coll, ids, **k)
    monkeypatch.setattr(PD, "delete_points_verified", spy)
    rec = _settle_apply(man, root)
    assert through and not [x for x in through if x[1].endswith("/scroll")]     # the prod branch ran, no scroll
    assert seen == [sorted(left)]
    assert rec["decision"] == "deleted-present" and rec["deleted"] == 2 and rec["present_on_prod"] == 2
    assert rec["state_after"] == "applied" and stage(root) == "applied"
    assert not set(state["prod"]["to_delete"]) & set(c.ids("http://prod"))


def test_2933_the_settle_on_a_prod_target_with_nothing_left_marks_applied_and_deletes_nothing(world, monkeypatch):
    through = _prod_is_http_prod(monkeypatch)
    c, man, root, state = world
    for i in state["prod"]["to_delete"]:
        c.colls["http://prod"].pop(i)
    before = len(prod_deletes(c))
    rec = _settle_apply(man, root)
    assert through == [("POST", "/collections/aw_e8/points")]       # one id lookup, nothing else, reached prod
    assert rec["decision"] == "marked-applied" and rec["present_on_prod"] == 0
    assert len(prod_deletes(c)) == before and stage(root) == "applied"
