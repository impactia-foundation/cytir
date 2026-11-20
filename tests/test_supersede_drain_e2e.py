"""Task 2423: end-to-end drain tests of the `supersede: url-group` path with the change log ON, and
call-site tests for the "already held" DONE exits. Tests only; the in-memory qdrant and wiring are
those of `test_aw_url_group_supersede` / `test_supersede_deleting_recovery`."""
from __future__ import annotations

import hashlib
import json

import pytest

import test_aw_url_group_supersede as T
import test_supersede_deleting_recovery as R
from cytir.core import paths
from cytir.index import change_log as CL
from cytir.index import promote_queue as Q

settings = T.settings


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    d = tmp_path / "changelog"
    monkeypatch.setenv("AW_CHANGE_LOG_DIR", str(d))
    paths.reset_settings_cache()
    yield d
    monkeypatch.delenv("AW_CHANGE_LOG_DIR", raising=False)
    paths.reset_settings_cache()


def _segment(log_dir):
    segs = sorted(log_dir.glob("*.jsonl"))
    assert len(segs) == 1, segs
    return CL.read_segment(segs[0])


@T.needs_pack
def test_e2e_url_group_drain_deletes_old_on_both_legs_and_writes_the_segment(tmp_path, monkeypatch, settings, log_dir):
    urls = T.real_pack_urls(monkeypatch)
    T.give_the_pack_a_family(monkeypatch)
    old = T.old_row_at(sorted(urls.values())[0])
    c = T.make_cluster(monkeypatch, [old])
    queue, contract = T.drain_fixture(tmp_path, monkeypatch, settings, supersede="url-group")
    T.wire_real_drain(monkeypatch, c, prod_upsert_fails=[False])
    c.colls["http://prod"][old["id"]] = dict(c.colls["http://twin"][old["id"]])   # prod holds the old row too
    rec = T.run_drain(queue, contract, tmp_path)[0]
    assert rec["status"] == "LANDED", rec
    for url in ("http://twin", "http://prod"):
        assert old["id"] not in c.ids(url) and set(urls) <= c.ids(url)
    assert (queue / "loda-refresh" / "DONE").exists()
    assert json.loads((tmp_path / "artifacts" / R.STATE).read_text())["stage"] == "applied"
    _, body = _segment(log_dir)
    adds = {x["id"] for x in body if x["op"] != "tombstone"}
    tombs = [x for x in body if x["op"] == "tombstone"]
    assert adds == set(urls)
    assert [(x["id"], x["reason"]) for x in tombs] == [(old["id"], "superseded")]


@T.needs_pack
def test_f2_prod_leg_dies_at_deleting_the_resume_deletes_old_rows_and_logs_them(tmp_path, monkeypatch, settings, log_dir):
    """Kill after the twin leg (state `deleting`); the next drain resumes. #1768 promises the prod
    delete runs before DONE: prod has no old row, the state is `applied`, the segment tombstones it."""
    urls = T.real_pack_urls(monkeypatch)
    T.give_the_pack_a_family(monkeypatch)
    old = T.old_row_at(sorted(urls.values())[0])
    c = T.make_cluster(monkeypatch, [old])
    queue, contract = T.drain_fixture(tmp_path, monkeypatch, settings, supersede="url-group")
    fails = [True]
    T.wire_real_drain(monkeypatch, c, prod_upsert_fails=fails)
    with pytest.raises(RuntimeError, match="prod went away"):
        T.run_drain(queue, contract, tmp_path)
    assert R.stage(tmp_path / "artifacts") == "deleting" and old["id"] in c.ids("http://prod")
    assert not list(log_dir.glob("*.jsonl")) and not (queue / "loda-refresh" / "DONE").exists()
    fails[0] = False
    rec = T.run_drain(queue, contract, tmp_path)[0]
    assert rec["status"] == "LANDED", rec
    assert old["id"] not in c.ids("http://prod") and set(urls) <= c.ids("http://prod")
    assert R.stage(tmp_path / "artifacts") == "applied" and (queue / "loda-refresh" / "DONE").exists()
    _, body = _segment(log_dir)
    assert [x["id"] for x in body if x["op"] == "tombstone"] == [old["id"]]


@T.needs_pack
def test_site_already_held_passes_the_membership_ids_and_the_state_tombstones(tmp_path, monkeypatch, settings, log_dir):
    c, urls, old, queue, contract = R._both_hold_the_pack(tmp_path, monkeypatch, settings)
    seen = []
    real = Q._log_resumed_landing

    def spy(record, ids, **k):
        seen.append((set(ids), list(k["tombstones"])))
        return real(record, ids, **k)
    monkeypatch.setattr(Q, "_log_resumed_landing", spy)
    rec = T.run_drain(queue, contract, tmp_path)[0]
    assert rec["status"] == "LANDED"
    assert seen == [(set(urls), [{"id": old["id"], "reason": "superseded"}])]
    _, body = _segment(log_dir)
    assert {x["id"] for x in body if x["op"] == "add"} == set(urls)


@T.needs_pack
def test_site_already_held_refuses_not_done_when_the_segment_write_fails(tmp_path, monkeypatch, settings, log_dir):
    c, urls, old, queue, contract = R._both_hold_the_pack(tmp_path, monkeypatch, settings)

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(CL, "record_landing", boom)
    rec = T.run_drain(queue, contract, tmp_path)[0]
    assert rec["status"] == "REFUSED" and rec["stage"] == "change-log" and "disk full" in rec["reason"]
    assert not (queue / "loda-refresh" / "DONE").exists()
    saved = json.loads((queue / "loda-refresh" / "receipt.json").read_text())
    assert saved["status"] == "REFUSED"


def _drop_one_from_both(c, urls, tmp_path):
    """The twin leg's gate dropped one pack id: neither leg holds it, the twin-written artifact omits it."""
    dropped = sorted(urls)[-1]
    for u in ("http://twin", "http://prod"):
        c.colls[u].pop(dropped, None)
    kept = sorted(set(urls) - {dropped})
    (tmp_path / "artifacts" / "loda-refresh-twin-written-ids.json").write_text(json.dumps(kept))
    return dropped, set(kept)


@T.needs_pack
def test_site_landed_with_twin_drops_passes_exc_ids_minus_prod_missing(tmp_path, monkeypatch, settings, log_dir):
    c, urls, old, queue, contract = R._both_hold_the_pack(tmp_path, monkeypatch, settings)
    dropped, kept = _drop_one_from_both(c, urls, tmp_path)
    assert len(urls) > 1 and dropped not in kept
    seen = []
    real = Q._log_resumed_landing

    def spy(record, ids, **k):
        seen.append((set(ids), list(k["tombstones"])))
        return real(record, ids, **k)
    monkeypatch.setattr(Q, "_log_resumed_landing", spy)
    rec = T.run_drain(queue, contract, tmp_path)[0]
    assert rec["status"] == "LANDED" and "landed_with_twin_drops" in rec, rec
    assert seen == [(kept, [{"id": old["id"], "reason": "superseded"}])]
    _, body = _segment(log_dir)
    assert {x["id"] for x in body if x["op"] == "add"} == kept


@T.needs_pack
def test_site_landed_with_twin_drops_refuses_not_done_when_the_segment_write_fails(tmp_path, monkeypatch, settings, log_dir):
    c, urls, old, queue, contract = R._both_hold_the_pack(tmp_path, monkeypatch, settings)
    _drop_one_from_both(c, urls, tmp_path)

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(CL, "record_landing", boom)
    rec = T.run_drain(queue, contract, tmp_path)[0]
    assert rec["status"] == "REFUSED" and rec["stage"] == "change-log", rec
    assert not (queue / "loda-refresh" / "DONE").exists()


@T.needs_pack
def test_site_resume_prod_only_passes_the_prod_written_ids_artifact(tmp_path, monkeypatch, settings, log_dir):
    """A plain (no supersede) sealed pack: twin landed, prod died; the prod-only resume logs exactly the
    ids of the pack's `prod-written-ids` artifact, with no tombstone."""
    urls = T.real_pack_urls(monkeypatch)
    T.give_the_pack_a_family(monkeypatch)
    c = T.make_cluster(monkeypatch, [])
    queue, contract = T.drain_fixture(tmp_path, monkeypatch, settings)
    pack = queue / "loda-refresh"
    man = json.loads((pack / "manifest.json").read_text())
    del man["allow_supersede"]                  # an ordinary pack: the prod-only resume is open to it
    (pack / "manifest.json").write_text(json.dumps(man, indent=1))
    (pack / "READY").write_text(hashlib.sha256((pack / "manifest.json").read_bytes()).hexdigest() + "\n")
    fails = [True]
    T.wire_real_drain(monkeypatch, c, prod_upsert_fails=fails)
    with pytest.raises(RuntimeError, match="prod went away"):
        T.run_drain(queue, contract, tmp_path)
    fails[0] = False
    monkeypatch.setattr(Q.PSP, "already_present",
                        lambda url, coll, ids, **k: [i for i in ids if i in c.colls[url]])
    landed = frozenset(urls)

    def already_landed(*a, **k):
        raise Q.PSP.AlreadyLandedError("twin holds the pack", ids=landed, present=landed)
    monkeypatch.setattr(Q, "dry_plan", already_landed)
    seen = []
    real = Q._log_resumed_landing

    def spy(record, ids, **k):
        seen.append((set(ids), list(k["tombstones"])))
        return real(record, ids, **k)
    monkeypatch.setattr(Q, "_log_resumed_landing", spy)
    rec = T.run_drain(queue, contract, tmp_path)[0]
    assert rec["status"] == "LANDED" and rec.get("resumed_prod_only"), rec
    written = set(CL.read_written_ids(CL.written_ids_path(tmp_path / "artifacts", "loda-refresh", "prod")))
    assert written and seen == [(written, [])]
    _, body = _segment(log_dir)
    assert {x["id"] for x in body} == written and all(x["op"] == "add" for x in body)


@T.needs_pack
def test_reverify_before_write_failure_stops_the_prod_write_and_the_pack_does_not_land(tmp_path, monkeypatch, settings, log_dir):
    """Item pin (task 2908): `UGS.reverify_before_write` runs under the lock right before the prod
    write. When it raises, prod is untouched (old row kept, no new ids), the pack does not land
    (no DONE marker, no change-log segment, state stays `deleting`). Today's twin state: the twin
    leg already finished (new ids written, old row deleted) before the prod leg starts."""
    urls = T.real_pack_urls(monkeypatch)
    T.give_the_pack_a_family(monkeypatch)
    old = T.old_row_at(sorted(urls.values())[0])
    c = T.make_cluster(monkeypatch, [old])
    queue, contract = T.drain_fixture(tmp_path, monkeypatch, settings, supersede="url-group")
    T.wire_real_drain(monkeypatch, c, prod_upsert_fails=[False])
    c.colls["http://prod"][old["id"]] = dict(c.colls["http://twin"][old["id"]])
    prod_before = set(c.ids("http://prod"))
    calls = []

    def boom(*a, **kw):
        calls.append((a, kw))
        raise RuntimeError("export no longer intact")

    monkeypatch.setattr(Q.UGS, "reverify_before_write", boom)
    with pytest.raises(RuntimeError, match="export no longer intact"):
        T.run_drain(queue, contract, tmp_path)
    assert len(calls) == 1 and calls[0][1].get("twin_url") == "http://twin"
    assert set(c.ids("http://prod")) == prod_before and old["id"] in c.ids("http://prod")
    assert not (queue / "loda-refresh" / "DONE").exists()
    assert not list(log_dir.glob("*.jsonl"))
    assert R.stage(tmp_path / "artifacts") != "applied"
    # twin: written and superseded before the prod leg began
    assert set(urls) <= c.ids("http://twin") and old["id"] not in c.ids("http://twin")


# ---------------------------------------------------------------- task 2926: pins before the resume code moves
def _abort_at_reverify(tmp_path, monkeypatch, settings):
    """Pass 1 of a sealed url-group pack: the twin leg lands, `UGS.reverify_before_write` raises right before
    the prod write. Returns the world and a switch: `gate["fail"] = False` lifts the patch for pass 2."""
    urls = T.real_pack_urls(monkeypatch)
    T.give_the_pack_a_family(monkeypatch)
    old = T.old_row_at(sorted(urls.values())[0])
    c = T.make_cluster(monkeypatch, [old])
    queue, contract = T.drain_fixture(tmp_path, monkeypatch, settings, supersede="url-group")
    T.wire_real_drain(monkeypatch, c, prod_upsert_fails=[False])
    c.colls["http://prod"][old["id"]] = dict(c.colls["http://twin"][old["id"]])
    gate = {"fail": True}
    real = Q.UGS.reverify_before_write

    def switch(*a, **k):
        if gate["fail"]:
            raise RuntimeError("export no longer intact")
        return real(*a, **k)
    monkeypatch.setattr(Q.UGS, "reverify_before_write", switch)
    with pytest.raises(RuntimeError, match="export no longer intact"):
        T.run_drain(queue, contract, tmp_path)
    return c, urls, old, queue, contract, gate


def _receipt(queue, pack="loda-refresh"):
    return json.loads((queue / pack / "receipt.json").read_text())


@T.needs_pack
def test_p1_two_pass_url_group_pack_aborts_between_the_legs_and_pass_two_lands_by_replan(
        tmp_path, monkeypatch, settings, log_dir):
    """P1 (trace 2912, step 13 prerequisite): pass 1 dies at the re-verify (ABORTED, state `deleting`, the twin is
    ahead of prod); pass 2, patch lifted, LANDS by re-plan: old row gone on both sides, new ids on both, DONE,
    state `applied`, exactly one change-log segment."""
    c, urls, old, queue, contract, gate = _abort_at_reverify(tmp_path, monkeypatch, settings)
    assert _receipt(queue)["status"] == "ABORTED"
    assert R.stage(tmp_path / "artifacts") == "deleting"
    assert set(urls) <= c.ids("http://twin") and old["id"] not in c.ids("http://twin")      # twin ahead
    assert not (set(urls) & c.ids("http://prod")) and old["id"] in c.ids("http://prod")
    assert not (queue / "loda-refresh" / "DONE").exists() and not list(log_dir.glob("*.jsonl"))
    gate["fail"] = False
    rec = T.run_drain(queue, contract, tmp_path)[0]
    assert rec["status"] == "LANDED", rec
    assert rec["url_group_supersede"]["resumed_from_state"] == "deleting"          # the state is reused by the re-plan
    assert not rec.get("resumed_prod_only")
    for url in ("http://twin", "http://prod"):
        assert old["id"] not in c.ids(url) and set(urls) <= c.ids(url)
    assert (queue / "loda-refresh" / "DONE").exists()
    assert R.stage(tmp_path / "artifacts") == "applied"
    _, body = _segment(log_dir)                                                    # exactly one segment (helper asserts)
    assert {x["id"] for x in body if x["op"] != "tombstone"} == set(urls)
    assert [(x["id"], x["reason"]) for x in body if x["op"] == "tombstone"] == [(old["id"], "superseded")]


@T.needs_pack
def test_p2_a_deleting_url_group_state_never_reaches_a_prod_only_resume(tmp_path, monkeypatch, settings, log_dir):
    """P2: the heal of a `deleting` url-group state is the re-plan. Neither prod-only resume is called in pass 2
    (nor in pass 1); the resume module must not be handed this case."""
    called = []
    monkeypatch.setattr(Q, "resume_prod_only", lambda **k: called.append("resume_prod_only"))
    monkeypatch.setattr(Q, "resume_content_pack_prod_only", lambda **k: called.append("resume_content_pack"))
    c, urls, old, queue, contract, gate = _abort_at_reverify(tmp_path, monkeypatch, settings)
    gate["fail"] = False
    rec = T.run_drain(queue, contract, tmp_path)[0]
    assert rec["status"] == "LANDED" and (queue / "loda-refresh" / "DONE").exists()
    assert called == []


def _healthy_second_pack(queue, tmp_path, name="zz-healthy"):
    """A plain sealed pack queued behind `loda-refresh`: the real pack's rows with new point ids (the manifest
    declares `identity_check: pack-declared`, so ids need not be reproducible). Returns its new ids."""
    import gzip
    import shutil
    import uuid
    src = T.SEALED
    stage = tmp_path / "stage2"
    shutil.copytree(src / "chunks", stage / "chunks")
    vec = tmp_path / "vectors2"
    shutil.copytree(src / "vectors", vec)
    remap = {}

    def new_id(old):
        remap.setdefault(old, str(uuid.UUID(hashlib.sha256(("task2926/" + old).encode()).hexdigest()[:32])))
        return remap[old]
    shard = stage / "chunks" / "shard-0000.texts.jsonl.gz"
    with gzip.open(shard, "rt", encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh]
    for r in rows:
        r["point_id"] = new_id(r["point_id"])
    with gzip.open(shard, "wt", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    rid = vec / "shard-0000.row_ids.jsonl"
    lines = [json.loads(x) for x in rid.read_text().splitlines()]
    for r in lines:
        r["point_id"] = new_id(r["point_id"])
    rid.write_text("".join(json.dumps(r) + "\n" for r in lines))
    first = json.loads((queue / "loda-refresh" / "manifest.json").read_text())
    man = {k: v for k, v in first.items() if k not in ("allow_supersede", UGS_FIELD)}
    man.update(pack_id=name, pack_chunks=str(stage / "chunks"), vectors=str(vec))
    pack = queue / name
    pack.mkdir()
    (pack / "manifest.json").write_text(json.dumps(man, indent=1))
    (pack / "READY").write_text(hashlib.sha256((pack / "manifest.json").read_bytes()).hexdigest() + "\n")
    return set(remap.values())


UGS_FIELD = Q.UGS.FIELD


@T.needs_pack
def test_p3a_a_persistent_reverify_refusal_of_a_url_group_pack_refuses_and_the_pass_continues(
        tmp_path, monkeypatch, settings, log_dir):
    """P3a (trace 2912, policy 1): the url-group pack at state `deleting` whose export re-verify refuses for good
    is REFUSED `supersede_check` (receipt kept, no DONE, twin still ahead); the drain does NOT stop: the healthy pack
    queued behind it lands in the same pass."""
    c, urls, old, queue, contract, gate = _abort_at_reverify(tmp_path, monkeypatch, settings)
    healthy = _healthy_second_pack(queue, tmp_path)
    assert healthy and not (healthy & set(urls))

    def refuse(*a, **k):
        raise Q.QueueError("export file damaged")
    monkeypatch.setattr(Q.UGS, "reverify_export", refuse)
    prod_before = set(c.ids("http://prod"))
    res = T.run_drain(queue, contract, tmp_path)
    by = {r["pack_id"]: r for r in res}
    assert by["loda-refresh"]["status"] == "REFUSED" and by["loda-refresh"]["stage"] == "supersede_check"
    assert "export file damaged" in by["loda-refresh"]["reason"]
    assert by["zz-healthy"]["status"] == "LANDED", by["zz-healthy"]
    assert healthy <= c.ids("http://twin") and healthy <= c.ids("http://prod")
    assert (queue / "zz-healthy" / "DONE").exists() and not (queue / "loda-refresh" / "DONE").exists()
    assert R.stage(tmp_path / "artifacts") == "deleting"
    assert old["id"] in c.ids("http://prod") and (set(c.ids("http://prod")) - healthy) == prod_before


# ---------------------------------------------------------------- task 2933: every settle exit that deletes on prod is logged
def _settled_and_logged(c, old, rec, log_dir, kept):
    """The settle deleted the old row on prod; the change-log segment of the same pass carries that delete
    as a tombstone, next to the pack's adds."""
    assert old["id"] not in c.ids("http://prod")
    assert rec["supersede_recovery"]["decision"] == "deleted-present"
    _, body = _segment(log_dir)
    assert [x["id"] for x in body if x["op"] == "tombstone"] == [old["id"]]
    assert {x["id"] for x in body if x["op"] == "add"} == kept


@T.needs_pack
def test_2933_the_already_held_exit_that_deletes_on_prod_logs_the_delete(tmp_path, monkeypatch, settings, log_dir):
    c, urls, old, queue, contract = R._both_hold_the_pack(tmp_path, monkeypatch, settings)
    assert old["id"] in c.ids("http://prod")
    rec = T.run_drain(queue, contract, tmp_path)[0]
    assert rec["status"] == "LANDED"
    _settled_and_logged(c, old, rec, log_dir, set(urls))


@T.needs_pack
def test_2933_the_twin_drops_exit_that_deletes_on_prod_logs_the_delete(tmp_path, monkeypatch, settings, log_dir):
    c, urls, old, queue, contract = R._both_hold_the_pack(tmp_path, monkeypatch, settings)
    dropped, kept = _drop_one_from_both(c, urls, tmp_path)
    assert old["id"] in c.ids("http://prod")
    rec = T.run_drain(queue, contract, tmp_path)[0]
    assert rec["status"] == "LANDED" and "landed_with_twin_drops" in rec
    _settled_and_logged(c, old, rec, log_dir, kept)


# ---- task 2938 ---------------------------------------------------------------------------------------

@T.needs_pack
def test_2938_a_blocked_gate_with_apply_writes_no_supersede_export(tmp_path, monkeypatch, settings, log_dir):
    """S18: `--apply` with the gate contract NOT on `mechanical` is a proof, not a landing. The supersede
    plan is read-only then: the export dir stays empty, nothing is written to either leg, no DONE, no segment."""
    urls = T.real_pack_urls(monkeypatch)
    T.give_the_pack_a_family(monkeypatch)
    old = T.old_row_at(sorted(urls.values())[0])
    c = T.make_cluster(monkeypatch, [old])
    queue, contract = T.drain_fixture(tmp_path, monkeypatch, settings, supersede="url-group")
    contract.write_text(json.dumps({"schema": Q.GATE_CONTRACT_SCHEMA, "adr": "ADR-110", "pr": 302,
                                    "profile": "battery-required"}))
    monkeypatch.setattr(Q, "resolve_prod_alias", lambda *a, **k: "aw_e8")
    monkeypatch.setattr(Q.MCC, "evaluate", lambda rows, **k: {
        "verdict": "PASS", "problems": {}, "other_failing_share": 0, "fail_share_max": 1,
        "word_floor_share": 0, "word_floor_fail_share_max": 1, "rows_scored": len(rows)})
    monkeypatch.setattr(Q, "live_apply", lambda **k: pytest.fail("must not write"))
    rec = T.run_drain(queue, contract, tmp_path, apply=True)[0]
    assert rec["status"] == "PROVED-BLOCKED" and rec["blocked_on"]
    assert rec["url_group_supersede"]["export"]["would_write"]
    assert not list(settings.iterdir()), "a blocked run wrote the supersede export"
    assert c.ids("http://twin") == {old["id"]} and not list(log_dir.glob("*.jsonl"))
    assert not (queue / "loda-refresh" / "DONE").exists()


@T.needs_pack
def test_2938_a_prod_receipt_that_disagrees_with_the_planned_deletes_is_aborted_before_done(
        tmp_path, monkeypatch, settings, log_dir):
    """S24: with the change log on, the planned prod deletes must equal the prod receipt's `superseded` count.
    The plan names one id more than prod deletes: no DONE, no segment (the log must never under-record), an
    ABORTED receipt that says so, and the error leaves the drain (it stops). (The extra id is a row prod does not hold.)"""
    urls = T.real_pack_urls(monkeypatch)
    T.give_the_pack_a_family(monkeypatch)
    old = T.old_row_at(sorted(urls.values())[0])
    c = T.make_cluster(monkeypatch, [old])
    queue, contract = T.drain_fixture(tmp_path, monkeypatch, settings, supersede="url-group")
    T.wire_real_drain(monkeypatch, c, prod_upsert_fails=[False])
    c.colls["http://prod"][old["id"]] = dict(c.colls["http://twin"][old["id"]])
    real = Q.UGS.prepare_prod

    def inflated(*a, **k):
        prod = real(*a, **k)
        prod["to_delete"] = list(prod["to_delete"]) + ["00000000-0000-5000-8000-0000000000ee"]
        return prod
    monkeypatch.setattr(Q.UGS, "prepare_prod", inflated)
    with pytest.raises(CL.ChangeLogError, match="planned for prod but the prod receipt says 1 deleted"):
        T.run_drain(queue, contract, tmp_path)
    rec = _receipt(queue)
    assert rec["status"] == "ABORTED" and "planned for prod but the prod receipt" in rec["reason"], rec
    assert not (queue / "loda-refresh" / "DONE").exists() and not list(log_dir.glob("*.jsonl"))


def _deleting_after_a_dead_prod_leg(tmp_path, monkeypatch, settings):
    """Pass 1 of a url-group pack: the twin leg lands (twin holds the new rows, the old row is deleted there),
    the prod upsert dies. The state is `deleting`; prod holds the old row and none of the new ones."""
    urls = T.real_pack_urls(monkeypatch)
    T.give_the_pack_a_family(monkeypatch)
    old = T.old_row_at(sorted(urls.values())[0])
    c = T.make_cluster(monkeypatch, [old])
    queue, contract = T.drain_fixture(tmp_path, monkeypatch, settings, supersede="url-group")
    fails = [True]
    T.wire_real_drain(monkeypatch, c, prod_upsert_fails=fails)
    with pytest.raises(RuntimeError, match="prod went away"):
        T.run_drain(queue, contract, tmp_path)
    assert R.stage(tmp_path / "artifacts") == "deleting" and old["id"] in c.ids("http://prod")
    fails[0] = False
    monkeypatch.setattr(Q.PSP, "already_present",           # honest now: what each collection holds
                        lambda url, coll, ids, **k: [i for i in ids if i in c.colls[url]])
    return c, urls, old, queue, contract


@T.needs_pack
def test_2938_the_real_plan_never_hands_a_deleting_url_group_pack_to_the_prod_only_resume(
        tmp_path, monkeypatch, settings, log_dir):
    """L4/L5 premise, from the real plan: with `already_present` answering truthfully (the twin holds every new
    row, prod none) the REAL `dry_plan` of a pack that declares `allow_supersede` (a `supersede: url-group` pack must)
    does not raise AlreadyLandedError, so the sealed prod-only resume branch of `drain()` is not entered: the
    re-plan lands the pack, deletes the old row on prod and logs the tombstone. The branch is reachable only with
    a plan that refuses (the next test constructs that)."""
    c, urls, old, queue, contract = _deleting_after_a_dead_prod_leg(tmp_path, monkeypatch, settings)
    called = []
    monkeypatch.setattr(Q, "resume_prod_only", lambda **k: called.append(1))
    rec = T.run_drain(queue, contract, tmp_path)[0]
    assert called == [] and rec["status"] == "LANDED" and not rec.get("resumed_prod_only"), rec
    assert old["id"] not in c.ids("http://prod") and set(urls) <= c.ids("http://prod")
    _, body = _segment(log_dir)
    assert [x["id"] for x in body if x["op"] == "tombstone"] == [old["id"]]


def _third_exit_world(tmp_path, monkeypatch, settings, *, stub_resume, write_ids=True):
    """The `deleting` state of a url-group pack AND a plan that refuses AlreadyLanded (a plan the real code
    cannot produce for an `allow_supersede` pack, see the test above): the drain takes the sealed prod-only resume
    branch. `stub_resume`: replace `resume_prod_only` (which refuses a supersede pack, next test) by a stand-in that
    does what the real one does for any other pack: upsert the missing rows to prod and write the prod
    written-ids artifact. Everything after it (the settle, the log, DONE) is the real drain."""
    c, urls, old, queue, contract = _deleting_after_a_dead_prod_leg(tmp_path, monkeypatch, settings)
    landed = frozenset(urls)

    def already_landed(*a, **k):
        raise Q.PSP.AlreadyLandedError("twin holds the pack", ids=landed, present=landed)
    monkeypatch.setattr(Q, "dry_plan", already_landed)
    if stub_resume:
        def resume(*, pack_id, artifact_root, **k):
            for i in urls:
                c.colls["http://prod"][i] = dict(c.colls["http://twin"][i])
            if write_ids:
                CL.written_ids_path(artifact_root, pack_id, "prod").write_text(json.dumps(sorted(urls)))
            return {"prod_receipt": {}}
        monkeypatch.setattr(Q, "resume_prod_only", resume)
    return c, urls, old, queue, contract


@T.needs_pack
def test_2938_the_sealed_prod_only_resume_refuses_a_supersede_pack_and_settles_nothing(
        tmp_path, monkeypatch, settings, log_dir):
    """With the state `deleting` and a plan that refuses AlreadyLanded, the REAL `resume_prod_only` refuses an
    `allow_supersede` pack ("reconcile by hand"): a REFUSED receipt (no write issued, so no abort), no DONE, no
    segment, prod untouched, the state still `deleting`. So the settle after the resume cannot run for a url-group
    pack through the real queue: the exit below is defensive, and is pinned with a stand-in resume."""
    c, urls, old, queue, contract = _third_exit_world(tmp_path, monkeypatch, settings, stub_resume=False)
    rec = T.run_drain(queue, contract, tmp_path)[0]
    assert rec["status"] == "REFUSED" and rec["stage"] == "live-apply-preflight" and "reconcile by hand" in rec["reason"]
    assert old["id"] in c.ids("http://prod") and not (c.ids("http://prod") & set(urls))
    assert R.stage(tmp_path / "artifacts") == "deleting" and not (queue / "loda-refresh" / "DONE").exists()
    assert not list(log_dir.glob("*.jsonl"))


@T.needs_pack
def test_2938_third_settle_exit_the_prod_only_resume_settles_and_logs_the_tombstone_before_done(
        tmp_path, monkeypatch, settings, log_dir):
    """The prod-only resume exit of `drain()` (L4, L5), constructed directly. State `deleting`: after the resume the
    settle deletes the old row on prod and the receipt records it as `supersede_recovery` (also in the saved
    receipt); the segment carries the adds AND the tombstone of the old row and exists when DONE is first written;
    the state ends `applied`."""
    c, urls, old, queue, contract = _third_exit_world(tmp_path, monkeypatch, settings, stub_resume=True)
    order = []
    real_log = Q._log_resumed_landing
    monkeypatch.setattr(Q, "_log_resumed_landing", lambda *a, **k: order.append(
        ("log", (queue / "loda-refresh" / "DONE").exists())) or real_log(*a, **k))
    rec = T.run_drain(queue, contract, tmp_path)[0]
    assert rec["status"] == "LANDED" and rec["resumed_prod_only"] is True, rec
    assert rec["supersede_recovery"], rec
    assert old["id"] not in c.ids("http://prod") and set(urls) <= c.ids("http://prod")
    assert R.stage(tmp_path / "artifacts") == "applied" and (queue / "loda-refresh" / "DONE").exists()
    assert order == [("log", False)]                      # the segment was written before DONE existed
    _, body = _segment(log_dir)
    assert {x["id"] for x in body if x["op"] == "add"} == set(urls)
    assert [(x["id"], x["reason"]) for x in body if x["op"] == "tombstone"] == [(old["id"], "superseded")]
    saved = json.loads((queue / "loda-refresh" / "receipt.json").read_text())
    assert saved["supersede_recovery"] == rec["supersede_recovery"]


@T.needs_pack
def test_2938_third_settle_exit_without_a_change_log_dir_writes_done_and_no_segment(
        tmp_path, monkeypatch, settings):
    """Same exit, no `AW_CHANGE_LOG_DIR`: the old row is still deleted on prod, the settle is still recorded, DONE
    is written, and no segment exists anywhere."""
    monkeypatch.delenv("AW_CHANGE_LOG_DIR", raising=False)
    paths.reset_settings_cache()
    assert CL.configured_dir() is None
    c, urls, old, queue, contract = _third_exit_world(tmp_path, monkeypatch, settings, stub_resume=True,
                                                      write_ids=False)      # nothing to read: only the guard keeps it quiet
    rec = T.run_drain(queue, contract, tmp_path)[0]
    assert rec["status"] == "LANDED" and rec["resumed_prod_only"] is True and rec["supersede_recovery"], rec
    assert old["id"] not in c.ids("http://prod") and R.stage(tmp_path / "artifacts") == "applied"
    assert (queue / "loda-refresh" / "DONE").exists()
    assert not [f for f in tmp_path.rglob("*.jsonl") if not f.name.endswith("-ledger.jsonl")]   # no segment
