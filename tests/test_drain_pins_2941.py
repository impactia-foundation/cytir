"""Task 2941: the last pins before `drain()` and `main()` move (drain split step 14).

Each test names the mutation of `drain()` (id in the mutation table of the 2941 task doc) it turns red. Fakes only: no
service is contacted (the URLs below are never opened).
"""
from __future__ import annotations

import hashlib
import json

import pytest

from cytir.index import promote_queue as Q
from cytir.index import promote_sealed_pack as PSP
from cytir.index import sparse_rule as SR
from test_aw_sparse_rule import _capture, _point, fx, rule_env  # noqa: F401  (fixtures)
from test_promote_queue import (  # noqa: F401  (the autouse alias fixture must be imported to apply here)
    MANIFEST_BASE, _default_prod_alias_resolution, _drain_plain, _one_pack_queue, _stub_plan_deps, write_contract,
    write_pack)


def _quiet_plan_refusal(monkeypatch):
    """`dry_plan` refuses at once: a test that only needs the loop's first decisions stops the pack there."""
    monkeypatch.setattr(Q, "dry_plan", lambda *a, **k: (_ for _ in ()).throw(PSP.PromotionError("stop here")))


def test_2941_the_pass_over_line_counts_two_done_and_two_superseded_packs(tmp_path, monkeypatch, capsys):
    """I4/I5: the count is a sum, not a flag."""
    _stub_plan_deps(monkeypatch)
    (tmp_path / "queue").mkdir()
    for name, marker in (("a1", "DONE"), ("a2", "DONE"), ("b1", "SUPERSEDED"), ("b2", "SUPERSEDED")):
        (write_pack(tmp_path / "queue", {**MANIFEST_BASE, "pack_id": name}) / marker).write_text("x\n")
    assert _drain_plain(tmp_path) == []
    assert "2 DONE, 2 SUPERSEDED" in capsys.readouterr().out


def test_2941_the_skip_cache_result_names_the_reason_of_the_refusal_it_skipped(tmp_path, monkeypatch):
    """I11."""
    pack = _one_pack_queue(tmp_path, monkeypatch)
    sha = hashlib.sha256((pack / "manifest.json").read_bytes()).hexdigest()
    (pack / "receipt.json").write_text(json.dumps({
        "status": "REFUSED", "stage": "plan", "manifest_sha": sha, "transaction_key": None,
        "reason": "the planner said no", "checked_at": "2026-11-18T00:00:00Z"}))
    res = _drain_plain(tmp_path)
    assert res[0]["status"] == "SKIPPED-REFUSED-UNCHANGED" and res[0]["previous_reason"] == "the planner said no"


def test_2941_a_single_drain_never_refreshes_the_corpus_wide_persisted_index(tmp_path, monkeypatch, capsys):
    """I21: the twin-first drain refreshes it; the Single transaction says it skips it, and does."""
    _one_pack_queue(tmp_path, monkeypatch)
    _quiet_plan_refusal(monkeypatch)
    calls = []
    monkeypatch.setattr(Q, "refresh_persisted_index", lambda **k: calls.append(k) or {"rebuilt": False, "reason": "t"})
    single = Q.Single(url="http://t.invalid:3", collection="aw_build", state_dir=tmp_path / "state")
    monkeypatch.setattr(Q, "refuse_live_single", lambda *_a, **_k: tmp_path / "state")   # the live-target proof reads a service
    Q.drain(tmp_path / "queue", transaction=single, task="2941", apply=False, batch=10, pause_s=0,
            artifact_root=tmp_path / "artifacts", contract_path=write_contract(tmp_path, "mechanical"),
            refresh_index=True)
    assert calls == [] and "[persisted-index] skipped" in capsys.readouterr().out
    Q.drain(tmp_path / "queue", twin_url="http://x", prod_url="http://y", task="2941", apply=False, batch=10,
            pause_s=0, artifact_root=tmp_path / "artifacts", contract_path=write_contract(tmp_path, "mechanical"),
            refresh_index=True)
    assert len(calls) == 1


@pytest.mark.parametrize("kw", [{"twin_url": "http://x"}, {"prod_url": "http://y"}])
def test_2941_a_drain_with_only_one_of_the_two_urls_is_refused(tmp_path, kw):
    """I24."""
    (tmp_path / "queue").mkdir()
    with pytest.raises(Q.QueueError, match="twin_url and prod_url are required"):
        Q.drain(tmp_path / "queue", task="2941", apply=True, batch=10, pause_s=0,
                artifact_root=tmp_path / "artifacts", refresh_index=False, **kw)


def test_2941_an_unknown_transaction_object_is_refused(tmp_path):
    """I27."""
    (tmp_path / "queue").mkdir()
    with pytest.raises(Q.QueueError, match="unknown transaction"):
        Q.drain(tmp_path / "queue", transaction=object(), task="2941", apply=True, batch=10, pause_s=0,
                artifact_root=tmp_path / "artifacts", refresh_index=False)


# ---- the "already landed" branch of a sealed pack, driven by fakes (no service) ----------------------------
def _already_landed_world(monkeypatch, ids):
    """`dry_plan` says every id is on the twin already (`AlreadyLandedError`), as after a kill between the legs."""
    ids = frozenset(ids)

    def landed(*_a, **_k):
        raise PSP.AlreadyLandedError("already landed", ids=ids, present=ids)
    monkeypatch.setattr(Q, "dry_plan", landed)
    return ids


def _drain_with(tmp_path, **kw):
    base = dict(twin_url="http://twin.invalid", prod_url="http://prod.invalid", task="2941", apply=False, batch=10,
                pause_s=0, artifact_root=tmp_path / "artifacts", refresh_index=False)
    base["contract_path"] = kw.pop("contract_path", None) or write_contract(tmp_path, "mechanical")
    base.update(kw)
    return Q.drain(tmp_path / "queue", **base)


def _pending_content_resume(tmp_path, monkeypatch, *, stage="twin-write-complete"):
    """A tombstone-only content pack whose twin leg is recorded (`stage`); the prod writers are stubbed."""
    from cytir.index.drain import common as DC
    monkeypatch.setattr(DC, "_RESUME_RETRY_BACKOFF_S", (0, 0, 0))
    (tmp_path / "queue").mkdir()
    points = tmp_path / "points.jsonl.gz"
    points.write_bytes(__import__("gzip").compress(b""))
    manifest = {"schema": Q.MANIFEST_SCHEMA, "kind": "content_pack", "pack_id": "cp-pending", "family": "l1-backfill",
                "pack_points": str(points), "rows": 0, "tombstones_count": 1,
                "collection": "aw_e8"}
    pack = write_pack(tmp_path / "queue", manifest)
    sha = hashlib.sha256((pack / "manifest.json").read_bytes()).hexdigest()
    if stage is not None:
        Q.write_content_pack_twin_state(tmp_path / "artifacts", "cp-pending", stage=stage, manifest_sha=sha,
                                        pack_ids=[], tomb_ids=["t1"])
    return pack


def test_2941_the_prod_only_content_resume_is_retried_three_times_then_the_drain_stops(tmp_path, monkeypatch):
    """Item 4 (`_retry_resume` bound, T1/T2): a transport error that never goes away is tried four times (the first
    call and three retries), then the pack is ABORTED and the error leaves the drain; the pack is not DONE."""
    import urllib.error
    pack = _pending_content_resume(tmp_path, monkeypatch)
    calls = []

    def always_down(**k):
        calls.append(1)
        raise urllib.error.URLError("prod is down")
    monkeypatch.setattr(Q, "resume_content_pack_prod_only", always_down)
    with pytest.raises(urllib.error.URLError):
        _drain_with(tmp_path, apply=True)
    assert len(calls) == 4
    assert not (pack / "DONE").exists()
    assert json.loads((pack / "receipt.json").read_text())["status"] == "ABORTED"


def test_2941_the_retry_helper_waits_five_fifteen_thirty_seconds_and_gives_up_on_other_errors_at_once():
    """Item 4: the backoff schedule is the contract (a longer one would outlive the unit's timeout)."""
    import urllib.error
    slept, calls = [], []

    def down():
        calls.append(1)
        raise urllib.error.URLError("x")
    with pytest.raises(urllib.error.URLError):
        Q._retry_resume(down, pack_id="p", sleep=slept.append)
    assert slept == [5, 15, 30] and len(calls) == 4
    other = []

    def broken():
        other.append(1)
        raise ValueError("not transport")
    with pytest.raises(ValueError):
        Q._retry_resume(broken, pack_id="p", sleep=slept.append)
    assert len(other) == 1 and slept == [5, 15, 30]


def _tight_budget():
    """A 1000 s window of which 900 s are spent, 98.5 s margin: 1.5 s of room now. One op is estimated at one second."""
    import time
    return dict(budget_seconds=1000.0, started_at_epoch=time.time() - 900.0, stop_margin_s=98.5)


@pytest.mark.parametrize("stage,legs,fits", [("twin-write-complete", 1, True), ("twin-write-started", 2, False)])
def test_2941_a_pending_content_resume_is_estimated_at_one_leg_after_a_finished_twin_leg_and_two_otherwise(
        tmp_path, monkeypatch, stage, legs, fits):
    """B5/B6: with 1.5 s of room, one tombstone is one second for the prod-only resume (fits) and two seconds for a
    replay of both legs (does not fit: the drain stops, nothing is written, no result)."""
    pack = _pending_content_resume(tmp_path, monkeypatch, stage=stage)
    monkeypatch.setattr(Q, "ESTIMATED_CONTENT_PACK_OPS_PER_SECOND", 1.0)
    seen = []
    monkeypatch.setattr(Q, "resume_content_pack_prod_only", lambda **k: seen.append("prod-only") or {"prod_receipt": {}})
    monkeypatch.setattr(Q, "live_apply_content_pack", lambda **k: seen.append("replay") or {"prod_receipt": {}})
    res = _drain_with(tmp_path, apply=True, **_tight_budget())
    if fits:
        assert [r["status"] for r in res] == ["LANDED"] and seen == ["prod-only"] and (pack / "DONE").exists()
    else:
        assert res == [] and seen == [] and not (pack / "DONE").exists()


def test_2941_a_sealed_pack_that_does_not_fit_the_budget_stops_the_drain_once_and_runs_no_later_pack(
        tmp_path, monkeypatch, capsys):
    """B1/B2/B3: the budget check of a sealed pack is a STOP (one line, the packs behind it stay untouched), not a skip."""
    _stub_plan_deps(monkeypatch)
    planned = []
    monkeypatch.setattr(Q, "dry_plan", lambda m, *a, **k: planned.append(m["pack_id"]) or (_ for _ in ()).throw(
        PSP.PromotionError("stop here")))
    (tmp_path / "queue").mkdir()
    for pid in ("s1", "s2"):
        write_pack(tmp_path / "queue", {**MANIFEST_BASE, "pack_id": pid})
    res = _drain_with(tmp_path, apply=True, budget_seconds=100.0, started_at_epoch=__import__("time").time(),
                      stop_margin_s=200.0)
    out = capsys.readouterr().out
    assert res == [] and planned == [] and out.count("[drain] stopping before") == 1
    res = _drain_with(tmp_path, apply=True, budget_seconds=100.0, started_at_epoch=__import__("time").time(),
                      stop_margin_s=50.0)                 # the margin is part of the rule: 100 s left, 50 s asked
    assert planned == ["s1", "s2"] and [r["status"] for r in res] == ["REFUSED", "REFUSED"]


@pytest.mark.parametrize("how", ["default", "transaction"])
def test_2941_the_prod_alias_reaches_the_prod_collection_lookup(tmp_path, monkeypatch, how):
    """I30/I31: the alias the caller (or its TwinFirst transaction) names is the one resolved, not a literal."""
    _one_pack_queue(tmp_path, monkeypatch)
    _already_landed_world(monkeypatch, ["id1"])
    asked = []

    def resolve(url, alias):
        asked.append((url, alias))
        raise Q.QueueError("stop here")
    monkeypatch.setattr(Q, "resolve_prod_alias", resolve)
    if how == "default":
        res = _drain_with(tmp_path, prod_alias="alias_named_by_the_caller")
    else:
        res = _drain_with(tmp_path, twin_url=None, prod_url=None,
                          transaction=Q.TwinFirst("http://twin.invalid", "http://prod.invalid", "alias_of_the_transaction"))
    assert res[0]["status"] == "REFUSED"
    assert asked == [("http://prod.invalid", "alias_named_by_the_caller" if how == "default" else "alias_of_the_transaction")]


# ---- item 7: `LAST_READBACK` through a drain of two content packs on one collection ---------------------------
def test_2941_each_content_pack_of_a_drain_carries_its_own_sparse_rule_readback_and_the_dict_keeps_the_last(
        fx, rule_env, tmp_path, monkeypatch):
    """Item 7, today's behaviour (observed, not endorsed): two content packs land on the same collection in one
    drain with the sparse rule on. Each pack's own upsert sets `SR.LAST_READBACK[collection]`; the receipt of each pack
    carries the readback of ITS rows (`checked` 6 then 2); afterwards the process-wide dict still holds the last
    pack's value. A stale value cannot reach a receipt today: `apply_content_pack_side` always upserts, and the
    writer clears the value per leg. Red under a mutation of the copy in `verify.py` (`out["sparse_rule_readback"]`). The writer is a stand-in that
    runs the real `PSP.upsert` (the rule, the readback) and the real `verify_written_form` against a fake store."""
    puts, store = _capture(monkeypatch)
    SR.LAST_READBACK.clear()
    monkeypatch.setattr(Q, "resolve_prod_alias", lambda *_a, **_k: "aw_e9")
    monkeypatch.setattr(Q, "payload_profile_preflight_both", lambda *_a, **_k: {})
    monkeypatch.setattr(Q.PD, "fetch_rows", lambda url, coll, ids, batch=1000, vectors=None: {
        i: {"payload": store[(coll, i)]["payload"], "vector": {"dense": [0.0, 1.0]}} for i in ids})
    rows = {"cp-1": fx["rows"][:6], "cp-2": fx["rows"][6:8]}
    (tmp_path / "queue").mkdir()
    for pid in rows:
        points = tmp_path / f"{pid}.jsonl.gz"
        points.write_bytes(__import__("gzip").compress(b""))
        write_pack(tmp_path / "queue", {"schema": Q.MANIFEST_SCHEMA, "kind": "content_pack", "pack_id": pid,
                                        "family": "l1-backfill", "pack_points": str(points), "rows": 0,
                                        "tombstones_count": 1, "collection": "aw_e9"})
    monkeypatch.setattr(Q.GPG, "guarded", lambda fn, *_a, **_k: ({"checked": 0}, None))

    def dry(manifest, *_a, **_k):
        pts = [_point(r) for r in rows[manifest["pack_id"]]]
        return {"_points_data": pts, "_tombstones_data": [], "_held_out_cards": {"rows": [], "tombstone_ids": []}}
    monkeypatch.setattr(Q, "content_pack_dry_check", dry)

    def writer(*, pack_id, points, prod_collection, **_k):
        PSP.upsert("http://prod.invalid", prod_collection, points, batch=5)
        form = Q.verify_written_form("http://prod.invalid", prod_collection, points)
        return {"twin_receipt": {}, "prod_receipt": {"payload_form": form}}
    monkeypatch.setattr(Q, "live_apply_content_pack", writer)
    res = {r["pack_id"]: r for r in _drain_with(tmp_path, apply=True)}
    assert res["cp-1"]["status"] == res["cp-2"]["status"] == "LANDED"
    first = res["cp-1"]["prod"]["payload_form"]["sparse_rule_readback"]
    second = res["cp-2"]["prod"]["payload_form"]["sparse_rule_readback"]
    assert first["checked"] == 6 and second["checked"] == 2
    assert SR.LAST_READBACK["aw_e9"] == second
    SR.LAST_READBACK.clear()


# ---- sealed pack, already on the twin: the live-twin-check resume (no per-pack artifact) ----------------------
def _live_twin_check_world(tmp_path, monkeypatch, *, profile="mechanical"):
    """Two ids planned, both on the twin, only `a` on prod, and NO twin-written-ids artifact for the pack: the
    drain derives the resume from a live twin check (`twin_ids_override`). Returns the calls of the stubbed writers."""
    _one_pack_queue(tmp_path, monkeypatch)
    _already_landed_world(monkeypatch, ["a", "b"])
    seen = {"resume": [], "settle": [], "present": []}

    def present(url, collection, ids):
        seen["present"].append((url, sorted(ids)))
        return ["a"] if url == "http://prod.invalid" else list(ids)
    monkeypatch.setattr(PSP, "already_present", present)
    monkeypatch.setattr(Q, "resolve_prod_alias", lambda *_a, **_k: "aw_e8")
    monkeypatch.setattr(Q, "resume_prod_only", lambda **k: seen["resume"].append(k) or {"prod_receipt": {"written": 1}})
    monkeypatch.setattr(Q, "settle_supersede_deleting", lambda *a, **k: seen["settle"].append(k) or None)
    return seen, write_contract(tmp_path, profile)


def test_2941_a_resume_derived_from_a_live_twin_check_hands_those_ids_to_the_writer_and_says_so(tmp_path, monkeypatch):
    """Item 3 (S24/S25/S29/S30): the ids missing from prod are confirmed on the twin by a live check (no artifact),
    the writer gets exactly them as `twin_ids_override`, and the receipt carries the "duplicate of already-landed
    pack(s)" note and `resumed_prod_only`."""
    seen, contract = _live_twin_check_world(tmp_path, monkeypatch)
    res = _drain_with(tmp_path, apply=True, contract_path=contract)
    assert res[0]["status"] == "LANDED" and res[0]["resumed_prod_only"] is True
    assert "duplicate of already-landed pack" in res[0]["note"]
    assert [k["twin_ids_override"] for k in seen["resume"]] == [["b"]]
    assert ("http://twin.invalid", ["b"]) in seen["present"]
    assert len(seen["settle"]) == 1 and seen["settle"][0]["apply"] is True
    assert (tmp_path / "queue" / "test-pack-001" / "DONE").exists()


def test_2941_a_resume_with_its_own_twin_artifact_has_no_override_and_no_duplicate_note(tmp_path, monkeypatch):
    """Item 3, the other side: with the pack's own twin-written-ids artifact the writer gets no override, the receipt
    no note, and the twin is not asked."""
    seen, contract = _live_twin_check_world(tmp_path, monkeypatch)
    art = Q.twin_written_ids_path(tmp_path / "artifacts", "test-pack-001")
    art.parent.mkdir(parents=True)
    art.write_text(json.dumps(["a", "b"]))
    res = _drain_with(tmp_path, apply=True, contract_path=contract)
    assert res[0]["status"] == "LANDED" and "note" not in res[0]
    assert [k["twin_ids_override"] for k in seen["resume"]] == [None]
    assert all(u != "http://twin.invalid" for u, _ in seen["present"])


@pytest.mark.parametrize("profile,apply,status", [("mechanical", False, "PROVED-DRY-RUN"),
                                                  ("battery-required", True, "PROVED-BLOCKED")])
def test_2941_a_live_twin_check_resume_is_a_prod_write_and_waits_for_apply_and_an_open_gate(
        tmp_path, monkeypatch, profile, apply, status):
    """S26/S27/S28: no prod write in a dry run or under a blocked gate; the receipt says what it would do and from
    which source the twin ids come."""
    seen, contract = _live_twin_check_world(tmp_path, monkeypatch, profile=profile)
    res = _drain_with(tmp_path, apply=apply, contract_path=contract)
    assert res[0]["status"] == status and seen["resume"] == [] and seen["settle"] == []
    assert res[0]["would"] == "resume prod leg: 1 ids (twin ids from a live twin check)"
    assert not (tmp_path / "queue" / "test-pack-001" / "DONE").exists()
    assert bool(res[0]["blocked_on"]) == (status == "PROVED-BLOCKED")


def test_2941_ids_missing_from_prod_that_are_on_neither_side_are_refused_and_name_the_overlap(tmp_path, monkeypatch):
    """S19/S21/S22/S23: nothing on the twin for the missing id and no artifact: not a resume. REFUSED at
    live-apply-preflight, `already_on_twin` says how many ids of the pack the twin holds, how many of those prod
    holds too, and the first fifty."""
    seen, contract = _live_twin_check_world(tmp_path, monkeypatch)
    monkeypatch.setattr(PSP, "already_present", lambda url, c, ids: ["a"] if url == "http://prod.invalid" else [])
    res = _drain_with(tmp_path, apply=True, contract_path=contract)
    assert res[0]["status"] == "REFUSED" and res[0]["stage"] == "live-apply-preflight"
    assert "on NEITHER twin nor prod" in res[0]["reason"] and seen["resume"] == []
    assert res[0]["already_on_twin"] == {"count": 2, "also_on_prod": 1, "ids": ["a", "b"]}


# ---- sealed pack, fresh: what the drain hands to the supersede plan and to the writer ------------------------
def _fresh_sealed_world(tmp_path, monkeypatch, *, profile, supersede=True):
    """Three planned rows. `p1` fails a content check, `p3` is held out, `p2` is served under another id."""
    manifest = {**MANIFEST_BASE, "pack_id": "fresh-1"}
    if supersede:
        manifest[Q.UGS.FIELD] = Q.UGS.URL_GROUP
    (tmp_path / "queue").mkdir()
    write_pack(tmp_path / "queue", manifest)
    monkeypatch.setattr(Q, "resolve_prod_alias", lambda *_a, **_k: "aw_e8")
    monkeypatch.setattr(Q.GPG, "guarded", lambda fn, *_a, **_k: ({"checked": 0}, None))
    joined = {pid: ("u", "t", {"consent_probe": pid, "search_ok": True}) for pid in ("p1", "p2", "p3", "p4")}
    monkeypatch.setattr(Q, "dry_plan", lambda *a, **k: {
        "_joined": dict(joined), "_matrix": 1, "_id_index": 2, "rows": {"projected": 4},
        "identity_check": {"verdict": "PASS"}, "_already_served_pids": frozenset({"p2"})})
    monkeypatch.setattr(Q, "mechanical_content_status", lambda m, j: {
        "verdict": "PASS", "problems": {"p1": "too short"}, "held_out": {"p3": "binary"}, "rows_scored": 4})
    seen = {"plan": [], "apply": [], "preflight": []}
    monkeypatch.setattr(Q.UGS, "plan_for_drain", lambda *a, **k: seen["plan"].append(k) or {
        "summary": {"groups": 1}, "prod": {"to_delete": []}})
    monkeypatch.setattr(Q, "payload_profile_preflight_both",
                        lambda t, p, payloads: seen["preflight"].append(len(payloads)) or {})
    monkeypatch.setattr(Q, "live_apply", lambda **k: seen["apply"].append(k) or {
        "twin_receipt": {}, "prod_receipt": {"superseded": 0}})
    return seen, write_contract(tmp_path, profile)


def test_2941_a_blocked_sealed_pack_still_hands_the_served_rows_to_the_supersede_plan_and_writes_nothing(
        tmp_path, monkeypatch):
    """S14b (the `drop_ids` initialiser) and S55: under a blocked gate the plan sees the rows the twin serves under
    another id as dropped, its export is not written (`write` false), nothing reaches the writer."""
    seen, contract = _fresh_sealed_world(tmp_path, monkeypatch, profile="battery-required")
    res = _drain_with(tmp_path, apply=True, contract_path=contract)
    assert res[0]["status"] == "PROVED-BLOCKED" and res[0]["already_served_skipped"] == 1
    assert [k["drop_ids"] for k in seen["plan"]] == [frozenset({"p2"})] and seen["plan"][0]["write"] is False
    assert seen["apply"] == [] and res[0]["url_group_supersede"] == {"groups": 1}


def test_2941_an_applied_sealed_pack_drops_failed_held_out_and_served_rows_from_both_the_plan_and_the_writer(
        tmp_path, monkeypatch):
    """S52/S53/S61/S63/S64/S65/S66: the drop set is the union of the three; the supersede plan and the writer get
    it and the plan itself; the consent census and the profile check count the rows that stay; the receipt says how
    many were dropped."""
    seen, contract = _fresh_sealed_world(tmp_path, monkeypatch, profile="mechanical")
    res = _drain_with(tmp_path, apply=True, contract_path=contract, twin_parallel=3)
    assert res[0]["status"] == "LANDED"
    drop = frozenset({"p1", "p2", "p3"})
    assert seen["plan"][0]["drop_ids"] == drop and seen["plan"][0]["write"] is True
    k = seen["apply"][0]
    assert k["drop_ids"] == drop and k["twin_parallel"] == 3 and k["supersede"]["summary"] == {"groups": 1}
    assert res[0]["mechanical_content_dropped"] == 3 and seen["preflight"] == [1]
    assert res[0]["consent_flags"] == Q.consent_flag_census([{"consent_probe": "p4", "search_ok": True}])
    assert res[0]["mechanical_content"]["problem_sample"] == {"p1": "too short"}
    assert res[0]["mechanical_content"]["held_out_sample"] == {"p3": "binary"}
    assert res[0]["already_served_skipped"] == 1 and "_joined" not in res[0]["mechanical_plan"]


# ---- the budget receipts of the content-pack path --------------------------------------------------------------
def test_2941_a_pending_resume_that_can_never_fit_a_whole_budget_leaves_a_loud_receipt_and_stops_the_drain(
        tmp_path, monkeypatch):
    """B9/B10/B16: 2 estimated seconds against a 100 s budget with a 98.5 s margin never fit. The receipt on disk
    says BUDGET-EXCEEDS-TIMEOUT with the estimate, the budget and the resume stage; the drain stops, nothing is written."""
    import time
    pack = _pending_content_resume(tmp_path, monkeypatch, stage="twin-write-started")
    monkeypatch.setattr(Q, "ESTIMATED_CONTENT_PACK_OPS_PER_SECOND", 1.0)
    monkeypatch.setattr(Q, "live_apply_content_pack", lambda **k: pytest.fail("written"))
    res = _drain_with(tmp_path, apply=True, budget_seconds=100.0, started_at_epoch=time.time(), stop_margin_s=98.5)
    saved = json.loads((pack / "receipt.json").read_text())
    assert [r["status"] for r in res] == ["BUDGET-EXCEEDS-TIMEOUT"] == [saved["status"]]
    assert saved["estimated_seconds"] == 2.0 and saved["budget_seconds"] == 100.0
    assert saved["resume_stage"] == "twin-write-started" and not (pack / "DONE").exists()


def test_2941_a_fresh_content_pack_that_can_never_fit_a_budget_gets_a_receipt_and_the_drain_goes_on(
        tmp_path, monkeypatch):
    """B11/B12: no resume state: the pack is too big for any run (receipt on disk, with the estimate), and the
    pack behind it still gets its turn. A pack that fits a fresh window but not what is left is only SKIPPED-BUDGET
    (no receipt on disk, the estimate in the result: B13/B14)."""
    import time
    pack = _pending_content_resume(tmp_path, monkeypatch, stage=None)
    monkeypatch.setattr(Q, "ESTIMATED_CONTENT_PACK_OPS_PER_SECOND", 1.0)
    write_pack(tmp_path / "queue", {**MANIFEST_BASE, "pack_id": "zz-behind"})
    _stub_plan_deps(monkeypatch)
    monkeypatch.setattr(Q, "dry_plan", lambda *a, **k: (_ for _ in ()).throw(PSP.PromotionError("behind ran")))
    res = {r["pack_id"]: r for r in _drain_with(tmp_path, apply=True, budget_seconds=100.0,
                                                started_at_epoch=time.time(), stop_margin_s=98.5)}
    saved = json.loads((pack / "receipt.json").read_text())
    assert res["cp-pending"]["status"] == saved["status"] == "BUDGET-EXCEEDS-TIMEOUT"
    assert saved["estimated_seconds"] == 2.0 and saved["budget_seconds"] == 100.0
    assert res["zz-behind"]["status"] == "REFUSED" and "behind ran" in res["zz-behind"]["reason"]
    (pack / "receipt.json").unlink()
    res = {r["pack_id"]: r for r in _drain_with(tmp_path, apply=True, **_tight_budget())}
    assert res["cp-pending"]["status"] == "SKIPPED-BUDGET" and res["cp-pending"]["estimated_seconds"] == 2.0
    assert not (pack / "receipt.json").exists() and not (pack / "DONE").exists()


# ---- task 2948: the last sixteen survivors (ids in the mutation table of the 2941 task doc) --------------------------
def _fresh_content_world(tmp_path, monkeypatch, *, profile="mechanical"):
    """A fresh (no twin state) content pack whose dry check, preflight and writer are fakes."""
    pack = _pending_content_resume(tmp_path, monkeypatch, stage=None)
    seen = {"apply": []}
    monkeypatch.setattr(Q, "content_pack_dry_check", lambda *a, **k: {"_points_data": [], "_tombstones_data": [],
                                                                       "rows": 0})
    monkeypatch.setattr(Q.GPG, "guarded", lambda fn, *_a, **_k: ({"checked": 0}, None))
    monkeypatch.setattr(Q, "resolve_prod_alias", lambda *_a, **_k: "aw_e8")
    monkeypatch.setattr(Q, "payload_profile_preflight_both", lambda *a, **k: {})
    monkeypatch.setattr(Q, "live_apply_content_pack", lambda **k: seen["apply"].append(k) or {
        "twin_receipt": {}, "prod_receipt": {}})
    return pack, seen, write_contract(tmp_path, profile)


def test_2941_a_blocked_gate_on_a_fresh_content_pack_is_proved_blocked_with_the_reason_not_a_dry_run(
        tmp_path, monkeypatch):
    """C20: the status follows `blocked`, and a dry run under an open gate is a different status."""
    pack, seen, contract = _fresh_content_world(tmp_path, monkeypatch, profile="battery-required")
    res = _drain_with(tmp_path, apply=True, contract_path=contract)
    assert res[0]["status"] == "PROVED-BLOCKED" and res[0]["blocked_on"] and seen["apply"] == []
    (pack / "receipt.json").unlink()
    res = _drain_with(tmp_path, apply=False, contract_path=write_contract(tmp_path, "mechanical"))
    assert res[0]["status"] == "PROVED-DRY-RUN" and res[0]["blocked_on"] == [] and seen["apply"] == []


def test_2941_a_fresh_content_pack_refused_before_any_write_is_a_retried_preflight_refusal(tmp_path, monkeypatch):
    """C28: the stage is `live-apply-preflight` (never cached), not the cached `content_pack_check`."""
    pack, seen, contract = _fresh_content_world(tmp_path, monkeypatch)

    def refuse(**_k):
        raise RuntimeError("lock busy")
    monkeypatch.setattr(Q, "live_apply_content_pack", refuse)
    res = _drain_with(tmp_path, apply=True, contract_path=contract)
    assert res[0]["status"] == "REFUSED" and res[0]["stage"] == "live-apply-preflight"
    assert "lock busy" in res[0]["reason"] and not (pack / "DONE").exists()


@pytest.mark.parametrize("streak,loud", [(Q.READ_FAILURE_LOUD_AFTER - 1, False), (Q.READ_FAILURE_LOUD_AFTER, True)])
def test_2941_a_twin_read_failure_in_the_content_dry_check_is_a_retried_read_refusal_with_its_streak(
        tmp_path, monkeypatch, capsys, streak, loud):
    """C13/C14/C15: stage `content_pack_read` (not the cached content refusal), the streak is in the receipt, and
    the LOUD line appears from the threshold on, not before."""
    pack, _seen, contract = _fresh_content_world(tmp_path, monkeypatch)

    def down(*_a, **_k):
        raise Q.TransientReadError("twin read timed out")
    monkeypatch.setattr(Q, "content_pack_dry_check", down)
    monkeypatch.setattr(Q, "read_failure_streak", lambda *_a, **_k: streak)
    res = _drain_with(tmp_path, apply=True, contract_path=contract)
    assert res[0]["status"] == "REFUSED" and res[0]["stage"] == "content_pack_read"
    assert res[0]["consecutive_read_failures"] == streak
    assert ("LOUD" in capsys.readouterr().out) is loud


def test_2941_a_replayed_twin_leg_maps_every_tombstone_reason_before_any_write(tmp_path, monkeypatch):
    """C5: a state tombstone whose reason has no public mapping aborts the replay with nothing written."""
    pack = _pending_content_resume(tmp_path, monkeypatch, stage="twin-write-started")
    monkeypatch.setattr(Q, "load_content_pack_ids_from_state",
                        lambda *a, **k: ([], [{"id": "t1", "reason": "no-such-reason"}]))
    wrote = []
    monkeypatch.setattr(Q, "live_apply_content_pack", lambda **k: wrote.append(k) or {})
    with pytest.raises(Q.CL.ChangeLogError):
        _drain_with(tmp_path, apply=True)
    assert wrote == [] and not (pack / "DONE").exists()
    assert json.loads((pack / "receipt.json").read_text())["status"] == "ABORTED"


# ---- Single / sealed-pack receipts ------------------------------------------------------------------------------
def _single_drain(tmp_path, monkeypatch, **kw):
    monkeypatch.setattr(Q, "refuse_live_single", lambda *_a, **_k: tmp_path / "state")
    single = Q.Single(url="http://t.invalid:3", collection="aw_build", state_dir=tmp_path / "state")
    base = dict(task="2941", apply=False, batch=10, pause_s=0, artifact_root=tmp_path / "artifacts",
                contract_path=write_contract(tmp_path, "mechanical"), refresh_index=False)
    base.update(kw)
    return Q.drain(tmp_path / "queue", transaction=single, **base)


def test_2941_a_single_plan_is_given_a_persisted_check_and_a_twin_first_plan_is_not(tmp_path, monkeypatch):
    """S77."""
    _one_pack_queue(tmp_path, monkeypatch)
    kws = []
    monkeypatch.setattr(Q, "dry_plan", lambda *a, **k: kws.append(k) or (_ for _ in ()).throw(PSP.PromotionError("x")))
    monkeypatch.setattr(Q, "_single_persisted_check", lambda: "the-knob")
    _single_drain(tmp_path, monkeypatch)
    _drain_with(tmp_path)
    assert kws[0]["persisted_check"] == "the-knob" and kws[0]["collection"] == "aw_build"
    assert "persisted_check" not in kws[1]


def test_2941_a_single_drain_refuses_a_url_group_pack_before_it_plans_any_supersede(tmp_path, monkeypatch):
    """S59: a Single has no twin to read the old rows from."""
    seen, _contract = _fresh_sealed_world(tmp_path, monkeypatch, profile="mechanical")
    res = _single_drain(tmp_path, monkeypatch, apply=True)
    assert res[0]["status"] == "REFUSED" and res[0]["stage"] == "supersede_check"
    assert "twin-first transaction" in res[0]["reason"] and seen["plan"] == [] and seen["apply"] == []


def test_2941_the_plan_scratch_matrix_and_index_are_released_before_the_pack_is_applied(tmp_path, monkeypatch):
    """S78: `_matrix` and `_id_index` are large; the loop drops them from the plan dict it was handed."""
    seen, contract = _fresh_sealed_world(tmp_path, monkeypatch, profile="mechanical", supersede=False)
    plans = []
    inner = Q.dry_plan

    def keep(*a, **k):
        plans.append(inner(*a, **k))
        return plans[-1]
    monkeypatch.setattr(Q, "dry_plan", keep)
    _drain_with(tmp_path, apply=True, contract_path=contract)
    assert plans and "_matrix" not in plans[0] and "_id_index" not in plans[0] and "_joined" not in plans[0]


def _mc(**extra):
    base = {"verdict": "PASS", "problems": {}, "held_out": {}, "rows_scored": 100, "other_failing_share": 0.0,
            "fail_share_max": 0.05, "word_floor_share": 0.0, "word_floor_fail_share_max": 0.1,
            "other_failing_rows": 0, "word_floor_dropped": 0}
    base.update(extra)
    return base


def test_2941_the_receipt_samples_at_most_twenty_problem_rows_and_twenty_held_out_rows(tmp_path, monkeypatch):
    """S46/S48."""
    seen, contract = _fresh_sealed_world(tmp_path, monkeypatch, profile="mechanical", supersede=False)
    monkeypatch.setattr(Q, "mechanical_content_status", lambda m, j: _mc(
        problems={f"p{i}": "short" for i in range(30)}, held_out={f"h{i}": "binary" for i in range(25)}))
    res = _drain_with(tmp_path, apply=True, contract_path=contract)
    mc = res[0]["mechanical_content"]
    assert len(mc["problem_sample"]) == 20 and len(mc["held_out_sample"]) == 20
    assert "problems" not in mc and "held_out" not in mc


def test_2941_a_refusing_content_verdict_names_only_the_ceilings_that_fired(tmp_path, monkeypatch):
    """S49: the non-word-floor ceiling and the word-floor ceiling are independent reasons."""
    seen, contract = _fresh_sealed_world(tmp_path, monkeypatch, profile="mechanical", supersede=False)
    monkeypatch.setattr(Q, "mechanical_content_status", lambda m, j: _mc(
        verdict="REFUSE", other_failing_share=0.5, other_failing_rows=50))
    res = _drain_with(tmp_path, apply=True, contract_path=contract)
    why = res[0]["blocked_on"][0]
    assert res[0]["status"] == "PROVED-BLOCKED" and "non-word-floor check" in why and "word floor (max" not in why
    assert seen["apply"] == []


def test_2941_a_prod_only_resume_carries_the_prior_receipt_and_survives_a_failing_carry_over(tmp_path, monkeypatch):
    """S36/S37: the resume receipt is built from the receipt it replaces; a carry-over that raises falls back to the
    derive-only shape and the pack stays LANDED."""
    seen, contract = _live_twin_check_world(tmp_path, monkeypatch)
    pack = tmp_path / "queue" / "test-pack-001"
    (pack / "receipt.json").write_text(json.dumps({"status": "REFUSED", "gate": {"profile": "mechanical"},
                                                   "mechanical_plan": {"rows": {"projected": 7}}}))
    got = []
    real = Q.resume_receipt_contract
    monkeypatch.setattr(Q, "resume_receipt_contract", lambda prior, *a, **k: got.append(prior) or real(prior, *a, **k))
    res = _drain_with(tmp_path, apply=True, contract_path=contract)
    assert res[0]["status"] == "LANDED" and got and got[0] is not None
    assert got[0]["mechanical_plan"]["rows"] == {"projected": 7}
    (pack / "DONE").unlink()
    monkeypatch.setattr(Q, "resume_receipt_contract", lambda prior, *a, **k: (
        real(None, *a, **k) if prior is None else (_ for _ in ()).throw(ValueError("bad carry-over"))))
    res = _drain_with(tmp_path, apply=True, contract_path=contract)
    assert res[0]["status"] == "LANDED" and (pack / "DONE").exists()


def test_2941_a_pack_both_sides_already_hold_is_marked_done_with_the_note_and_no_write(tmp_path, monkeypatch):
    """S10: apply, prod holds every id: LANDED + note, DONE, no writer."""
    seen, contract = _live_twin_check_world(tmp_path, monkeypatch)
    monkeypatch.setattr(PSP, "already_present", lambda url, c, ids: list(ids))
    res = _drain_with(tmp_path, apply=True, contract_path=contract)
    assert res[0]["status"] == "LANDED" and "already had every id" in res[0]["note"] and seen["resume"] == []
    assert (tmp_path / "queue" / "test-pack-001" / "DONE").exists()


def test_2941_the_already_on_twin_receipt_lists_the_first_fifty_ids_of_a_larger_set(tmp_path, monkeypatch):
    """S23: 60 ids on the twin, prod holds none of them, and the pack has no artifact."""
    seen, contract = _live_twin_check_world(tmp_path, monkeypatch)
    ids = [f"i{n:03d}" for n in range(60)]
    _already_landed_world(monkeypatch, ids)
    monkeypatch.setattr(PSP, "already_present", lambda url, c, i: [] if url == "http://prod.invalid" else list(i)[:0])
    res = _drain_with(tmp_path, apply=True, contract_path=contract)
    a = res[0].get("already_on_twin")
    assert a is not None and a["count"] == 60 and a["ids"] == ids[:50]
