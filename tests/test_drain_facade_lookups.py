"""Pins of three lookups that must stay LIVE attribute lookups on the drain facade (task 2899).

`cytir.index.promote_queue` is the facade the drain's tests (and the operator's patches) replace names on.
A function that moved out of it, or that a later step moves, keeps honouring such a patch only while it
looks the name up on the facade at call time. A direct reference (a bound name, a local copy) passes
every other test and silently ignores the patch. Reviews 2895 and 2898 each found one such lookup that no
test pinned. No Qdrant: each lookup is replaced by a recording fake and the function under test must
call it.
"""
from __future__ import annotations

import pytest

from cytir.index import promote_queue as Q
from cytir.index import supersede as US
from cytir.index.drain import content_checks as CC


def test_revalidate_prod_collection_asks_the_facade_to_resolve_the_alias(monkeypatch):
    seen = []

    def fake(url, alias):
        seen.append((url, alias))
        return "X"
    monkeypatch.setattr(Q, "resolve_prod_alias", fake)
    Q._revalidate_prod_collection("pack-1", "http://prod.invalid", "an_alias", "X")      # passes
    with pytest.raises(Q.TFA.ApplyAbort, match="moved mid-drain"):
        Q._revalidate_prod_collection("pack-1", "http://prod.invalid", "an_alias", "Y")
    assert seen == [("http://prod.invalid", "an_alias")] * 2


def test_dry_plan_takes_its_plan_kwargs_from_the_facade(monkeypatch):
    seen = {}

    def fake_scope(manifest):
        seen["manifest"] = manifest
        return {"marker_from_fake": 1}, None, None

    def fake_plan(**kw):
        seen["plan_kwargs"] = kw
        return {}
    monkeypatch.setattr(Q, "plan_kwargs_with_scope", fake_scope)
    monkeypatch.setattr(Q.PSP, "plan", fake_plan)
    manifest = {"pack_chunks": "c", "vectors": "v", "mrl": 128, "collection": "aw_e8"}
    Q.dry_plan(manifest, "http://twin.invalid")
    assert seen["manifest"] is manifest
    assert seen["plan_kwargs"]["marker_from_fake"] == 1
    assert seen["plan_kwargs"]["collection"] == "aw_e8"


def test_check_generation_siblings_reads_the_served_group_from_the_facade(monkeypatch):
    url = "https://www.example.org/doc/1"
    stored = {"url": url, "provenance": "fam", "source_type": "official", "content_digest": "d"}
    gkey = US.payload_group_key(stored)
    tombs = [{"id": "old-1", "url": url}]
    seen = []

    def fake_members(target_url, collection, urls):
        seen.append((target_url, collection, list(urls)))
        return {gkey: [("survivor-1", dict(stored))]}
    monkeypatch.setattr(Q, "served_group_members", fake_members)
    problems = CC.check_generation_siblings(tombs, {"old-1": stored}, target_url="http://twin.invalid",
                                            collection="aw_e8", export_ids=set())
    assert problems == []
    assert seen == [("http://twin.invalid", "aw_e8", [url])]
    # the fake's answer decides: with no survivor in the group the same call names a problem
    monkeypatch.setattr(Q, "served_group_members", lambda *a, **k: {})
    again = CC.check_generation_siblings(tombs, {"old-1": stored}, target_url="http://twin.invalid",
                                         collection="aw_e8", export_ids=set())
    assert len(again) == 1 and "no surviving row" in again[0]


def test_check_canonical_duplicates_reads_the_survivors_from_the_facade(monkeypatch):
    canon = "https://www.example.org/doc/1"
    variant = canon + "?utm_source=x"
    stored = {"url": variant, "provenance": "fam", "source_type": "official", "content_digest": "d"}
    gkey = US.payload_group_key(stored)
    survivor = {"url": canon, "provenance": "fam", "source_type": "official", "content_digest": "d"}
    tombs = [{"id": "old-1", "url": variant}]
    seen = []

    def fake_members(target_url, collection, urls):
        seen.append((target_url, collection, list(urls)))
        return {gkey: [("survivor-1", dict(survivor))]}
    monkeypatch.setattr(Q, "served_group_members", fake_members)
    args = dict(target_url="http://twin.invalid", collection="aw_e8", export_ids=set())
    assert CC.check_canonical_duplicates(tombs, {"old-1": stored}, **args) == []
    assert seen == [("http://twin.invalid", "aw_e8", [canon])]
    # the fake's answer decides: with no survivor in the group the same call names a problem
    monkeypatch.setattr(Q, "served_group_members", lambda *a, **k: {})
    again = CC.check_canonical_duplicates(tombs, {"old-1": stored}, **args)
    assert len(again) == 1 and "no surviving row" in again[0]


# ---------------------------------------------------------------- task 2902: the sealed write path
class _Stop(Exception):
    pass


def _runb(seen, *, stop_at=None):
    class RB:
        LOCKS = ("L1",)

        @staticmethod
        def assert_locks_free(task):
            seen.append(("assert_locks_free", task))
            if stop_at == "free":
                raise _Stop

        @staticmethod
        def acquire(name, task, why, pid=None):
            seen.append(("acquire", name))

        @staticmethod
        def release(name, task):
            seen.append(("release", name))
    return RB


def test_live_apply_takes_the_runner_the_lock_host_and_apply_side_from_the_facade(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(Q, "_load_run_batch", lambda: _runb(seen, stop_at="free"))
    monkeypatch.setattr(Q, "prod_lock_host", lambda: seen.append("host") or "a-host")
    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", lambda **kw: seen.append(("lock", kw["host"])) or object())
    manifest = {"pack_id": "p-1", "collection": "aw_e8"}
    kw = dict(pack_id="p-1", manifest=manifest, twin_url="http://t.invalid", prod_url="http://p.invalid", task="t",
              batch=1, pause_s=0, artifact_root=tmp_path / "a", prod_collection="aw_e9")
    with pytest.raises(_Stop):
        Q.live_apply(**kw)
    assert seen == ["host", ("lock", "a-host"), ("assert_locks_free", "t")]
    # the legs: a fake `twin_first` runs the prod leg, which must use the facade's apply_side and
    # re-check the alias through the facade, before the write and again after it
    seen.clear()
    monkeypatch.setattr(Q, "_load_run_batch", lambda: _runb(seen))
    monkeypatch.setattr(Q, "_revalidate_prod_collection", lambda *a: seen.append(("revalidate", a[0])))
    monkeypatch.setattr(Q.LAYOUT, "fingerprint", lambda u, c: {"points_count": 0})

    def fake_apply_side(role, url, mf, coll, **k):
        seen.append(("apply_side", role, url, coll))
        return {"layout_after": {"points_count": 0}, "written": 0}, tmp_path / "ids.json"
    monkeypatch.setattr(Q, "apply_side", fake_apply_side)

    def fake_twin_first(**k):
        st = k["preflight"]()
        k["apply_target"]("prod", st)
        k["verify_target"]("prod", st)
    monkeypatch.setattr(Q.TFA, "twin_first", fake_twin_first)
    Q.live_apply(**kw)
    assert ("apply_side", "prod", "http://p.invalid", "aw_e9") in seen
    assert [x for x in seen if x[0] == "revalidate"] == [("revalidate", "p-1")] * 2


def test_single_apply_takes_its_helpers_from_the_facade(monkeypatch, tmp_path):
    import types
    seen = []
    monkeypatch.setattr(Q, "_load_run_batch", lambda: _runb(seen))
    monkeypatch.setattr(Q, "assert_single_target_not_live", lambda single: seen.append(("not_live", single.collection)))
    monkeypatch.setattr(Q, "_single_persisted_check", lambda: seen.append("persisted") or "FROM-FACADE")
    monkeypatch.setattr(Q.LAYOUT, "fingerprint", lambda u, c: {"points_count": 0})

    def fake_apply_side(role, url, mf, coll, **k):
        seen.append(("apply_side", role, k["persisted_check"]))
        return {"layout_after": {"points_count": 2}, "written": 2, "rows": {}}, tmp_path / "ids.json"
    monkeypatch.setattr(Q, "apply_side", fake_apply_side)
    single = types.SimpleNamespace(url="http://s.invalid", collection="c1", pause_s=0, parallel=1,
                                   payload_profile_of=None)
    out = Q.single_apply(pack_id="p-1", manifest={"pack_id": "p-1"}, single=single, task="t", batch=1,
                         artifact_root=tmp_path / "a")
    assert out["check5"]["status"] == "PASS"
    assert ("acquire", "SINGLE-WRITE-LOCK-c1") in seen and ("release", "SINGLE-WRITE-LOCK-c1") in seen
    assert ("not_live", "c1") in seen
    assert ("apply_side", "single", "FROM-FACADE") in seen


def test_rehearse_takes_the_runner_and_apply_side_from_the_facade(monkeypatch, tmp_path):
    import hashlib
    import json
    seen = []
    monkeypatch.setattr(Q, "_load_run_batch", lambda: _runb(seen))
    pack = tmp_path / "q" / "p-1"
    pack.mkdir(parents=True)
    manifest = {"schema": Q.MANIFEST_SCHEMA, "pack_id": "p-1", "family": "f", "collection": "aw_e8", "mrl": 128,
                "pack_chunks": "c", "vectors": "v", "identity_check": "pack-declared",
                "identity_reason": "test", "family_payload": {}, "title_from": []}
    (pack / "manifest.json").write_text(json.dumps(manifest))
    (pack / "READY").write_text(hashlib.sha256((pack / "manifest.json").read_bytes()).hexdigest() + "\n")
    monkeypatch.setattr(Q.PSP, "read_records", lambda *_a, **_k: iter([{}]))      # a "real" pack: one row
    monkeypatch.setattr(Q.LAYOUT, "fingerprint", lambda u, c: {"points_count": 0})
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps(["i1"]))

    def fake_apply_side(role, url, mf, coll, **k):
        seen.append(("apply_side", role))
        return {"written": 0, "write_seconds": 0.0}, ids
    monkeypatch.setattr(Q, "apply_side", fake_apply_side)
    monkeypatch.setattr(Q.RB, "assert_all_present", lambda *a: {"status": "PASS"})
    monkeypatch.setattr(Q.RB, "delete_ids", lambda *a: None)
    monkeypatch.setattr(Q.RB, "assert_rolled_back", lambda *a: {"status": "PASS"})
    out = Q.rehearse(pack, twin_url="http://t.invalid", task="t", batch=1, pause_s=0, artifact_root=tmp_path / "a")
    assert out["status"] == "REHEARSED-AND-ROLLED-BACK"
    assert ("apply_side", "twin") in seen and ("acquire", "TWIN-WRITE-LOCK") in seen


def test_apply_side_takes_its_plan_kwargs_from_the_facade(monkeypatch, tmp_path):
    seen = {}

    def fake_scope(manifest):
        seen["manifest"] = manifest
        return {"marker_from_fake": 1}, None, None

    def fake_plan(**kw):
        seen["plan_kwargs"] = kw
        raise _Stop
    monkeypatch.setattr(Q, "plan_kwargs_with_scope", fake_scope)
    monkeypatch.setattr(Q.PSP, "plan", fake_plan)
    manifest = {"pack_id": "p-1", "pack_chunks": "c", "vectors": "v", "mrl": 128, "collection": "aw_e8"}
    with pytest.raises(_Stop):
        Q.apply_side("twin", "http://t.invalid", manifest, "aw_e8", batch=1, pause_s=0, ids_dir=tmp_path)
    assert seen["manifest"] is manifest and seen["plan_kwargs"]["marker_from_fake"] == 1


# ---------------------------------------------------------------- task 2911: the content-pack write path
def _cp_kw(tmp_path):
    manifest = {"pack_id": "cp-1", "collection": "aw_e8"}
    return dict(pack_id="cp-1", manifest=manifest, twin_url="http://t.invalid", prod_url="http://p.invalid",
                task="t", batch=1, pause_s=0, artifact_root=tmp_path / "a", prod_collection="aw_e9",
                points=[{"id": "p1"}], tombstones=[])


def test_live_apply_content_pack_takes_the_runner_and_the_lock_host_from_the_facade(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(Q, "_load_run_batch", lambda: _runb(seen, stop_at="free"))
    monkeypatch.setattr(Q, "prod_lock_host", lambda: seen.append("host") or "a-host")
    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", lambda **kw: seen.append(("lock", kw["host"])) or object())
    with pytest.raises(_Stop):
        Q.live_apply_content_pack(**_cp_kw(tmp_path))
    assert seen == ["host", ("lock", "a-host"), ("assert_locks_free", "t")]


def test_live_apply_content_pack_legs_take_the_side_the_recheck_and_the_form_read_from_the_facade(
        monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(Q, "_load_run_batch", lambda: _runb(seen))
    monkeypatch.setattr(Q, "prod_lock_host", lambda: "a-host")
    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", lambda **kw: object())
    monkeypatch.setattr(Q, "_revalidate_prod_collection", lambda *a: seen.append(("revalidate", a[0])))
    monkeypatch.setattr(Q, "verify_written_form", lambda url, coll, pts, **k: seen.append(("form", url, coll)) or {})
    monkeypatch.setattr(Q.UGS, "reverify_generation_export", lambda *a, **k: None)
    monkeypatch.setattr(Q.LAYOUT, "fingerprint", lambda u, c: {"points_count": 0})

    def fake_side(role, url, coll, points, tombstones, **k):
        seen.append(("side", role, url, coll))
        return {"layout_before": {"points_count": 0}, "layout_after": {"points_count": 1}, "written": 1,
                "tombstoned": 0}
    monkeypatch.setattr(Q, "apply_content_pack_side", fake_side)

    def fake_twin_first(**k):
        st = k["preflight"]()
        k["apply_target"]("prod", st)
        seen.append("verify")
        k["verify_target"]("prod", st)
    monkeypatch.setattr(Q.TFA, "twin_first", fake_twin_first)
    monkeypatch.setattr(Q.PSP, "already_present",
                        lambda url, coll, ids, **k: list(ids) if "verify" in seen else [])
    Q.live_apply_content_pack(**_cp_kw(tmp_path))
    assert ("side", "prod", "http://p.invalid", "aw_e9") in seen
    assert [x for x in seen if x[0] == "revalidate"] == [("revalidate", "cp-1")] * 2     # before the write, after it
    assert ("form", "http://p.invalid", "aw_e9") in seen


# ---------------------------------------------------------------- task 2922: the payload-patch write path
def _pp_kw(tmp_path):
    manifest = {"pack_id": "pp-1", "collection": "aw_e8", "field": "snippet"}
    return dict(pack_id="pp-1", manifest=manifest, pack_dir=tmp_path, twin_url="http://t.invalid",
                prod_url="http://p.invalid", task="t", batch=1, pause_s=0, artifact_root=tmp_path / "a",
                prod_collection="aw_e9", twin_client=object(), prod_client=object())


def test_live_apply_payload_patch_takes_the_runner_and_the_lock_host_from_the_facade(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(Q, "_load_run_batch", lambda: _runb(seen, stop_at="free"))
    monkeypatch.setattr(Q, "prod_lock_host", lambda: seen.append("host") or "a-host")
    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", lambda **kw: seen.append(("lock", kw["host"])) or object())
    with pytest.raises(_Stop):
        Q.live_apply_payload_patch(**_pp_kw(tmp_path))
    assert seen == ["host", ("lock", "a-host"), ("assert_locks_free", "t")]


def test_live_apply_payload_patch_rechecks_the_alias_through_the_facade_at_all_three_places(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(Q, "_load_run_batch", lambda: _runb(seen))
    monkeypatch.setattr(Q, "prod_lock_host", lambda: "a-host")
    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", lambda **kw: object())
    monkeypatch.setattr(Q, "_revalidate_prod_collection", lambda *a: seen.append(("revalidate", a[0])))
    monkeypatch.setattr(Q.PPK, "load_rows", lambda manifest, pack_dir: [])
    monkeypatch.setattr(Q.PPK, "write_inverse_pack", lambda *a, **k: {})
    monkeypatch.setattr(Q.PPK, "write_leg_ids", lambda *a, **k: None)
    monkeypatch.setattr(Q.PG, "write_pause", lambda role, n, p: 0)

    def fake_apply_leg(role, client, coll, rows, field, **k):
        k["before_write"]()                       # the alias re-check right before the first write
        return {"points_before": 1, "points_after": 1, "verify_sample": 0, "selected": 0,
                **{t: 0 for t in Q.PPK.TERMS}}
    monkeypatch.setattr(Q.PPK, "apply_leg", fake_apply_leg)
    monkeypatch.setattr(Q.PPK, "verify_leg", lambda *a, **k: None)

    def fake_twin_first(**k):
        st = k["preflight"]()
        k["apply_target"]("prod", st)
        seen.append("verify")
        k["verify_target"]("prod", st)
    monkeypatch.setattr(Q.TFA, "twin_first", fake_twin_first)
    Q.live_apply_payload_patch(**_pp_kw(tmp_path))
    assert seen.index("verify") == 4              # [acquire, revalidate, revalidate, ...] precede the verify
    assert [x for x in seen if x[0] == "revalidate"] == [("revalidate", "pp-1")] * 3


def _pp_drain_call(tmp_path, *, manifest_over=None, apply=True, results=None):
    import json
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"id": "00000000-0000-4000-8000-000000000001", "expect": "o", "value": "n"}) + "\n")
    import hashlib
    manifest = {"pack_id": "pp-1", "collection": "aw_e8", "field": "snippet", "rows_file": "rows.jsonl",
                "count": 1, "sha256": hashlib.sha256(rows.read_bytes()).hexdigest(),
                "payload_patch_version": 1, "family": "f", **(manifest_over or {})}
    record = {"pack_id": "pp-1", "manifest_sha": "s"}
    gate = {"profile": "mechanical", "contract_path": "/x", "pr": 1}
    return record, lambda: Q._drain_payload_patch(
        pack_dir=tmp_path, manifest=manifest, record=record, gate=gate, apply=apply, twin_url="http://t.invalid",
        prod_url="http://p.invalid", task="t", batch=1, pause_s=0, artifact_root=tmp_path / "a",
        prod_collection=lambda: "aw_e9", prod_alias="aw_prod", results=[] if results is None else results)


def test_drain_payload_patch_writes_its_receipts_through_the_facade(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(Q, "write_pack_receipt", lambda pack_dir, record: seen.append(record["status"]))
    _, call = _pp_drain_call(tmp_path, manifest_over={"rows_file": "gone.jsonl"})
    call()                                                              # unreadable rows
    _, call = _pp_drain_call(tmp_path, apply=False)
    call()                                                              # dry run

    def landed(**k):
        ids = {"applied": ["i1"], "already_applied": []}
        leg = {"ids": ids, "selected": 1}
        return {"twin_receipt": dict(leg), "prod_receipt": dict(leg), "inverse_pack": {"dir": "x"}}
    monkeypatch.setattr(Q, "live_apply_payload_patch", landed)
    from cytir.index import change_log as CL
    monkeypatch.setattr(CL, "record_landing", lambda *a, **k: None)
    _, call = _pp_drain_call(tmp_path)
    call()                                                              # landed
    assert seen == ["REFUSED", "PROVED-DRY-RUN", "LANDED"]


def test_drain_payload_patch_takes_the_leg_and_the_write_attempt_from_the_facade(monkeypatch, tmp_path):
    made, got = [], []

    class FakeAttempt:
        started = False

        def __init__(self):
            made.append(self)
    monkeypatch.setattr(Q, "WriteAttempt", FakeAttempt)

    def stub(**k):
        got.append(k["write_attempt"])
        raise RuntimeError("stop here")
    monkeypatch.setattr(Q, "live_apply_payload_patch", stub)
    monkeypatch.setattr(Q, "write_pack_receipt", lambda *a: None)
    results = []
    record, call = _pp_drain_call(tmp_path, results=results)
    call()                                                              # plain failure: refused, drain goes on
    assert got == made and len(made) == 1 and results == [record]


# ---------------------------------------------------------------- task 2926: the prod-only resume functions
# `resume_prod_only` (also the RECONSTRUCTION resume: `twin_ids_override`) and `resume_content_pack_prod_only` move
# to `cytir/index/drain/resume.py` at step 13. They must keep reading `_load_run_batch`, `prod_lock_host` and `TFA`
# from the facade at call time: the drain goldens patch exactly these three. Each test REPLACES `Q.TFA` itself (not
# an attribute of the shared module object), so a bound `TFA` name in a moved module would see the real one.
import types


def _fake_tfa(seen, *, result=None):
    def lock(**kw):
        seen.append(("lock", kw["host"], kw["purpose"]))
        return "THE-LOCK"

    def resume(**kw):
        seen.append(("resume_after_verified_twin_gate", kw["lock"], kw["armed"]))
        return result if result is not None else {"from": "fake-tfa"}
    return types.SimpleNamespace(ProductionWriterLock=lock, resume_after_verified_twin_gate=resume,
                                 ApplyAbort=Q.TFA.ApplyAbort)


def _resume_kw(tmp_path):
    pts = tmp_path / "points.jsonl"
    pts.write_text("")
    manifest = {"pack_id": "p-1", "collection": "aw_e8", "pack_points": str(pts)}
    common = dict(pack_id="p-1", manifest=manifest, twin_url="http://t.invalid", prod_url="http://p.invalid",
                  task="t", batch=1, pause_s=0, artifact_root=tmp_path / "a", prod_collection="aw_e9")
    cp = dict(common, state_record={"stage": "twin-write-complete", "pack_ids": [], "tomb_ids": []})
    sealed = dict(common)
    return {"resume_prod_only": (Q.resume_prod_only, dict(sealed, twin_ids_override=["i1"])),     # reconstruction
            "resume_prod_only (artifact)": (Q.resume_prod_only, sealed),
            "resume_content_pack_prod_only": (Q.resume_content_pack_prod_only, cp)}


@pytest.mark.parametrize("which", ["resume_prod_only", "resume_prod_only (artifact)", "resume_content_pack_prod_only"])
def test_the_prod_only_resumes_take_the_runner_the_lock_host_and_tfa_from_the_facade(monkeypatch, tmp_path, which):
    fn, kw = _resume_kw(tmp_path)[which]
    seen = []
    monkeypatch.setattr(Q, "_load_run_batch", lambda: _runb(seen, stop_at="free"))
    monkeypatch.setattr(Q, "prod_lock_host", lambda: seen.append("host") or "a-host")
    monkeypatch.setattr(Q, "TFA", _fake_tfa(seen))
    if which == "resume_prod_only (artifact)":
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "p-1-twin-written-ids.json").write_text("[]")
    with pytest.raises(_Stop):
        fn(**kw)
    assert seen == ["host", ("lock", "a-host", "taskt-p-1-resume"), ("assert_locks_free", "t")]


@pytest.mark.parametrize("which", ["resume_prod_only", "resume_prod_only (artifact)", "resume_content_pack_prod_only"])
def test_the_prod_only_resumes_run_their_protocol_through_the_facades_tfa_under_the_facades_runner(
        monkeypatch, tmp_path, which):
    fn, kw = _resume_kw(tmp_path)[which]
    seen = []
    monkeypatch.setattr(Q, "_load_run_batch", lambda: _runb(seen))
    monkeypatch.setattr(Q, "prod_lock_host", lambda: "a-host")
    monkeypatch.setattr(Q, "TFA", _fake_tfa(seen, result={"from": "fake-tfa", "which": which}))
    if which == "resume_prod_only (artifact)":
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "p-1-twin-written-ids.json").write_text("[]")
    out = fn(**kw)
    assert out == {"from": "fake-tfa", "which": which}                  # the facade's TFA ran the protocol
    assert ("resume_after_verified_twin_gate", "THE-LOCK", True) in seen
    assert [x for x in seen if x[0] in ("acquire", "release")] == [("acquire", "L1"), ("release", "L1")]


# ---------------------------------------------------------------- task 2929: the names the moved resume code reads from the facade
# Step 13 moved the three functions to `cytir/index/drain/resume.py`. Besides `_load_run_batch`, `prod_lock_host` and `TFA`
# (pinned above for the two resumes) they read `apply_side` (sealed resume), `apply_content_pack_side` and `verify_written_form` (content resume) and
# `_revalidate_prod_collection` (every function) at call time. Tests rebind these on the facade; a name bound at import
# in `resume.py` would run the real one instead.
def _fake_tfa_running_the_legs(seen):
    def lock(**kw):
        return "THE-LOCK"

    def resume(**kw):
        st = {}
        kw["apply_prod"](st)
        kw["verify_mirrors"](st)
        seen.append(("legs ran", sorted(st)))
        return st
    return types.SimpleNamespace(ProductionWriterLock=lock, resume_after_verified_twin_gate=resume,
                                 ApplyAbort=Q.TFA.ApplyAbort)


def _rebind_the_collaborators(monkeypatch, tmp_path, seen):
    monkeypatch.setattr(Q, "_load_run_batch", lambda: _runb(seen))
    monkeypatch.setattr(Q, "prod_lock_host", lambda: "a-host")
    monkeypatch.setattr(Q, "TFA", _fake_tfa_running_the_legs(seen))
    monkeypatch.setattr(Q, "_revalidate_prod_collection", lambda *a, **k: seen.append(("alias", a)))
    monkeypatch.setattr(Q.LAYOUT, "fingerprint", lambda *a, **k: {"fp": 1})
    monkeypatch.setattr(Q.PSP, "already_present", lambda u, c, ids, **k: list(ids))
    ids_file = tmp_path / "a" / "p-1-prod-written-ids.json"
    ids_file.parent.mkdir(parents=True, exist_ok=True)
    ids_file.write_text('["i1"]')

    def apply_side(role, url, manifest, collection, **kw):
        seen.append(("apply_side", role, collection))
        return {"written": 1, "resume": {"present_before": 0}}, ids_file
    monkeypatch.setattr(Q, "apply_side", apply_side)
    monkeypatch.setattr(Q, "apply_content_pack_side", lambda role, url, coll, points, tombs, **kw: (
        seen.append(("apply_content_pack_side", role, coll)) or {"written": 0, "tombstoned": 0}))
    monkeypatch.setattr(Q, "verify_written_form", lambda *a, **k: seen.append(("form", a[1])) or {})


def test_the_sealed_resume_reads_apply_side_and_the_alias_check_from_the_facade(monkeypatch, tmp_path):
    seen = []
    _rebind_the_collaborators(monkeypatch, tmp_path, seen)
    fn, kw = _resume_kw(tmp_path)["resume_prod_only"]
    fn(**kw)
    assert ("apply_side", "prod", "aw_e9") in seen
    assert [x[0] for x in seen if x[0] == "alias"] == ["alias", "alias"]          # before the write, after the verify


@pytest.mark.usefixtures("no_sparse_rule")
def test_the_content_resume_reads_verify_written_form_and_the_alias_check_from_the_facade(monkeypatch, tmp_path):
    seen = []
    _rebind_the_collaborators(monkeypatch, tmp_path, seen)
    fn, kw = _resume_kw(tmp_path)["resume_content_pack_prod_only"]
    fn(**kw)
    assert ("form", "aw_e9") in seen and ("apply_content_pack_side", "prod", "aw_e9") in seen
    assert [x[0] for x in seen if x[0] == "alias"] == ["alias", "alias"]


def test_the_settle_reads_the_runner_the_lock_host_tfa_and_the_alias_check_from_the_facade(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(Q, "_load_run_batch", lambda: _runb(seen))
    monkeypatch.setattr(Q, "prod_lock_host", lambda: seen.append("host") or "a-host")
    monkeypatch.setattr(Q, "TFA", types.SimpleNamespace(
        ProductionWriterLock=lambda **kw: types.SimpleNamespace(
            acquire=lambda: seen.append("lock-acquire"), release=lambda: seen.append("lock-release")),
        ApplyAbort=Q.TFA.ApplyAbort))
    monkeypatch.setattr(Q, "_revalidate_prod_collection", lambda *a, **k: seen.append(("alias", a[0])))
    st = {"stage": "deleting", "ids": ["i1"], "urls": [], "export": {}, "prod": None, "summary": {}}
    monkeypatch.setattr(Q.UGS, "read_state", lambda *a, **k: st)
    monkeypatch.setattr(Q.UGS, "present_on", lambda url, coll, ids: list(ids) if "prod" in url else [])
    monkeypatch.setattr(Q.UGS, "delete_superseded", lambda *a, **k: seen.append("deleted") or 1)
    monkeypatch.setattr(Q.UGS, "mark_applied", lambda *a, **k: seen.append("marked"))
    out = Q.settle_supersede_deleting({"supersede": True, "pack_id": "p-1", "collection": "aw_e8"}, "s", tmp_path,
                                      twin_url="http://twin", prod_url="http://prod", prod_collection="aw_e9",
                                      prod_alias="aw_prod", batch=1, task="t", apply=True)
    assert out["decision"] == "deleted-present"
    assert seen == [("alias", "p-1"), "host", ("assert_locks_free", "t"), ("acquire", "L1"), "lock-acquire",
                    ("alias", "p-1"), "deleted", "lock-release", ("release", "L1"), "marked"]
