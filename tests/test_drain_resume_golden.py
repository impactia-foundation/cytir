"""Golden of the drain's RESUME write path, end to end on real Qdrant (task 2899; drain split step 13).

The other goldens pin a fresh landing (sealed, content pack) and one content-pack crash. This file
pins what a SECOND drain does after a first one died between the twin leg and the prod leg:

  G1  sealed pack, prod leg died before it wrote: the next drain resumes PROD ONLY
  G2  sealed pack, prod leg died after one batch (prod = the serving box): only the missing ids are sent
  G2n the same on a NON-prod target: refused by the already-served ceiling (characterization)
  G1k G1 with no receipt.json left by the first run (a kill)
  G2b content pack, big-pack chunk checkpoint: the batch the first attempt PUT is not sent again
  G3  content pack with a held-out card: the held-out file and the receipt fields survive a resume
  G4  sealed pack, the twin-written-ids artifact is gone: the resume rebuilds the id set from twin
  G5  sealed pack, the prod alias moved between the legs: the resume refuses and writes nothing

Each scenario runs the REAL `drain()` twice on one temp queue (first run: a prod upsert dies, like a
kill between the legs; second run: no fault). Nothing but the two writer locks and the persisted-input
index is stubbed (`drain_transaction_harness.install_fakes`). The write wrapper records and passes
through to Qdrant. HONESTY: the death is a `KeyboardInterrupt` from the wrapper, so the drain's
`except BaseException` handler runs; a SIGKILL would not run it.

Own alias name; the collections are the harness's (see the note at TW). Needs a throwaway Qdrant
(`AW_TEST_QDRANT_URL`); skips without one.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
import uuid

import pytest

pytestmark = pytest.mark.usefixtures("no_sparse_rule")

import drain_transaction_harness as H
from cytir.index import promote_queue as Q
from cytir.index import promote_sealed_pack as PSP

# The payload-profile declaration knows only the harness's collection names, so the collections are
# the harness's (`aw_e8` twin, `aw_e9` prod); the ALIAS is this file's own, and a third collection
# (never written) is the alias's other target for G5.
TW, PR, PR_MOVED, AL = H.TWIN, H.PROD, "aw_p2899b", "aw_prod2899"
FIX = H.HERE / "fixtures" / "drain_content_pack" / "points.jsonl"
PACK_ID = "resume-golden-batch-00001"


@pytest.fixture
def url():
    u = H.qdrant_url()
    yield u
    try:
        H.drop_alias(u, AL)
        H.clear_live(u)
        if PR_MOVED in {c["name"] for c in H.api(u, "/collections")["collections"]}:
            H.api(u, f"/collections/{PR_MOVED}", method="DELETE")
    except Exception as exc:  # noqa: BLE001 -- teardown must not turn a pass into an error
        print(f"teardown: could not clear {u}: {exc}", file=sys.stderr)


def _world(url: str) -> None:
    for n in (TW, PR, PR_MOVED):
        H.reset_collection(url, n)
    H.set_alias(url, AL, PR)


class Writes:
    """Records every point PUT (`PSP._api`, the one door every leg writes through, batches included)
    and passes through to the real Qdrant. `die_on_prod_call=N`: the N-th prod PUT (1-based) dies
    before it wrote."""

    def __init__(self, monkeypatch, die_on_prod_call: int | None = None):
        self.log: list = []
        self.die = die_on_prod_call
        self.prod_calls = 0
        real = PSP._api

        def role(coll):
            return {TW: "twin", PR: "prod", PR_MOVED: "moved"}.get(coll, coll)

        def api(u, path, body=None, method=None, *a, **k):
            m = re.match(r"/collections/([^/]+)/points\?", path)
            if m and method == "PUT" and body and "points" in body:
                who = role(m.group(1))
                if who == "prod":
                    self.prod_calls += 1
                    if self.die == self.prod_calls:
                        self.log.append(["put-died", who, len(body["points"])])
                        raise KeyboardInterrupt("simulated crash before this prod batch wrote")
                self.log.append(["put", who, sorted(str(p["id"]) for p in body["points"])])
            return real(u, path, body, method, *a, **k)
        monkeypatch.setattr(PSP, "_api", api)

    def sent(self, role: str) -> list[str]:
        return [i for w in self.log if w[0] == "put" and w[1] == role for i in w[2]]

    def roles(self) -> list[str]:
        return [f"{w[0]}:{w[1]}:{len(w[2]) if w[0] == 'put' else w[2]}" for w in self.log]


def _ids(url: str, coll: str) -> list[str]:
    return [str(p["id"]) for p in H.dump_points(url, coll)]


def _points(url: str, coll: str) -> list[tuple]:
    return [(str(p["id"]), json.dumps(p["payload"], sort_keys=True), json.dumps(p["vector"], sort_keys=True))
            for p in H.dump_points(url, coll)]


def _segments(log_dir: pathlib.Path) -> list:
    from cytir.index import change_log as CL
    return [CL.read_segment(s) for s in sorted(log_dir.glob("*.jsonl"))]


def _receipt(queue: pathlib.Path) -> dict:
    return json.loads(next(queue.glob("*/receipt.json")).read_text())


def _drain(queue, url, artifacts, pause_s=0, **kw):
    return Q.drain(queue, twin_url=url, prod_url=url, task="2899-golden", apply=True, batch=3,
                   pause_s=pause_s, artifact_root=artifacts, refresh_index=False, prod_alias=AL, **kw)


def _boot(tmp, url, monkeypatch, *, die_on_prod_call, queue=None):
    from cytir.core import paths
    _world(url)
    log_dir = tmp / "changelog"
    monkeypatch.setenv("AW_CHANGE_LOG_DIR", str(log_dir))
    paths.reset_settings_cache()
    queue = queue or H.sealed_pack(tmp, collection=TW)
    locks = H.install_fakes(monkeypatch, Q, PSP)
    w = Writes(monkeypatch, die_on_prod_call)
    return queue, tmp / "artifacts", log_dir, locks, w


def _crash_then_resume(tmp, url, monkeypatch, *, die_on_prod_call=1, queue=None, between=None, pause_s=0):
    queue, artifacts, log_dir, locks, w = _boot(tmp, url, monkeypatch, die_on_prod_call=die_on_prod_call,
                                                queue=queue)
    with pytest.raises(BaseException) as ei:
        _drain(queue, url, artifacts, pause_s)
    assert isinstance(ei.value, KeyboardInterrupt)
    first = {"locks": list(locks), "writes": list(w.log), "twin": _ids(url, TW), "prod": _ids(url, PR),
             "segments": len(_segments(log_dir)), "receipt": _receipt(queue),
             "files": sorted(p.name for p in artifacts.glob("*")),
             "done": (queue / SEALED / "DONE").exists()}
    del locks[:]
    w.log.clear()
    if between:
        between(url, artifacts)
    res = _drain(queue, url, artifacts, pause_s)
    return {"queue": queue, "artifacts": artifacts, "log_dir": log_dir, "locks": list(locks), "w": w,
            "first": first, "results": res}




SEALED = "dila-juri-v1-shard0"
T = "2899-golden"


def _lock_seq(purpose_suffix: str) -> list:
    """The exact order the drain took and gave back the writer locks of one transaction."""
    p = f"{SEALED}-{purpose_suffix}"
    return [("assert_locks_free", T), ("acquire", "TWIN-WRITE-LOCK", T, p),
            ("acquire", "PROD-WRITE-LOCK", T, p), "prod-lock-acquire", "prod-lock-release",
            ("release", "PROD-WRITE-LOCK", T), ("release", "TWIN-WRITE-LOCK", T)]


def _landed_ids(log_dir: pathlib.Path) -> list[tuple[str, str]]:
    segs = _segments(log_dir)
    assert len(segs) == 1, "exactly one change-log landing, never two"
    head, body = segs[0]
    assert head["n"] == len(body) and head["pack_id"] == SEALED
    return sorted((x["op"], x["id"]) for x in body)


def _assert_prod_equals_twin(url):
    t, p = H.dump_points(url, TW), H.dump_points(url, PR)
    assert [str(x["id"]) for x in t] == [str(x["id"]) for x in p] and t
    assert [x["vector"]["dense"] for x in t] == [x["vector"]["dense"] for x in p]
    assert [x["payload"]["url"] for x in t] == [x["payload"]["url"] for x in p]


def test_G1_sealed_pack_prod_only_resume(tmp_path, url, monkeypatch):
    r = _crash_then_resume(tmp_path, url, monkeypatch)
    f, w, rec = r["first"], r["w"], r["results"][0]
    # the interrupted first run: twin written (one PUT), prod died before it wrote, ABORTED, no landing
    assert [x[:2] for x in f["writes"]] == [["put", "twin"], ["put", "twin"], ["put-died", "prod"]]
    assert sorted(i for x in f["writes"] if x[0] == "put" for i in x[2]) == f["twin"]
    assert f["writes"][-1] == ["put-died", "prod", 3]       # batch=3: the first prod batch died
    assert len(f["twin"]) == 4 and f["prod"] == [] and f["segments"] == 0
    assert f["receipt"]["status"] == "ABORTED" and f["done"] is False
    assert f["locks"] == _lock_seq("promotion-queue")
    assert f"{SEALED}-twin-written-ids.json" in f["files"]
    # the resume: PROD ONLY (no twin PUT), the exact lock sequence, one landing
    assert w.sent("twin") == [] and w.sent("prod") == f["twin"]
    assert [x[:2] for x in w.log] == [["put", "prod"], ["put", "prod"]]      # two batches, both prod
    assert r["locks"] == _lock_seq("promotion-queue-resume")
    assert rec["status"] == "LANDED" and rec["resumed_prod_only"] is True and "twin" not in rec
    res = rec["prod"]["resume"]
    assert (res["present_before"], res["written"], res["present_after"], res["twin_written_ids"]) == (0, 4, 4, 4)
    assert res["resumed_partial"] is False and res["prod_written_ids_equal_twin"] is True
    assert res["twin_written_ids_source"].endswith(f"{SEALED}-twin-written-ids.json")
    assert (r["queue"] / SEALED / "DONE").exists()
    assert _landed_ids(r["log_dir"]) == [("add", i) for i in f["twin"]]
    assert _ids(url, TW) == f["twin"]
    _assert_prod_equals_twin(url)
    assert _receipt(r["queue"])["status"] == "LANDED"


def _prod_is_the_harness_prod(monkeypatch):
    """The serving box answers `PG.is_prod` true for its collection and alias. The harness URL is a
    throwaway, and the throwaway rule wins first in `prod_guard.is_prod`, so every golden but this one
    runs its prod leg as a NON-serving target. Here the harness prod collection and its alias count as
    prod, as on the serving box."""
    from cytir.index import prod_guard as PG
    orig = PG.is_prod
    monkeypatch.setattr(PG, "is_prod", lambda u, c=None: c in (PR, AL) or orig(u, c))


def _record_upserts(monkeypatch):
    """Record `(role-by-collection, pause_s)` of every `PSP.upsert` call and pass `pause_s=0` on, so
    nothing sleeps. Returns the list."""
    seen: list = []
    real = PSP.upsert

    def upsert(url, collection, points, batch, pause_s=0.0, *a, **k):
        seen.append((collection, pause_s))
        return real(url, collection, points, batch, 0.0, *a, **k)
    monkeypatch.setattr(PSP, "upsert", upsert)
    return seen


def test_G2_sealed_partial_prod_write_resume_sends_only_the_missing_rows(tmp_path, url, monkeypatch):
    """The prod leg died after one of two batches (3 of 4 rows on prod). With the harness prod treated
    as the serving box (`PG.is_prod` true), the resume skips the `already_served_urls` question (asked
    only off prod, since PR #660), classifies 3 rows present, sends ONLY the missing one, and LANDS.
    Also pins that the prod upsert gets the pause `PG.write_pause("prod", n, pause_s)` computes."""
    from cytir.index import prod_guard as PG
    _prod_is_the_harness_prod(monkeypatch)
    monkeypatch.setattr(PG, "write_pause", lambda role, rows, pause_s: pause_s + (7.0 if role == "prod" else 0.0))
    seen = _record_upserts(monkeypatch)
    r = _crash_then_resume(tmp_path, url, monkeypatch, die_on_prod_call=2, pause_s=0.25)
    f, w, rec = r["first"], r["w"], r["results"][0]
    assert [x[:2] for x in f["writes"]] == [["put", "twin"], ["put", "twin"], ["put", "prod"], ["put-died", "prod"]]
    assert len(f["prod"]) == 3 and len(f["twin"]) == 4
    assert rec["status"] == "LANDED" and rec["resumed_prod_only"] is True
    missing = sorted(set(f["twin"]) - set(f["prod"]))
    assert len(missing) == 1
    assert w.sent("twin") == [] and w.sent("prod") == missing        # the three already there: not re-sent
    res = rec["prod"]["resume"]
    assert (res["present_before"], res["written"], res["present_after"], res["resumed_partial"]) == (3, 1, 4, True)
    assert res["prod_written_ids_equal_twin"] is True and rec["prod"]["resumed_partial"] is True
    assert (r["queue"] / SEALED / "DONE").exists()
    assert _landed_ids(r["log_dir"]) == [("add", i) for i in f["twin"]]
    assert r["locks"] == _lock_seq("promotion-queue-resume")
    _assert_prod_equals_twin(url)
    # write pause: the prod upserts (got the configured 0.25 + the 7.0 the patched PG.write_pause adds for role prod
    prod_pauses = [p for c, p in seen if c == PR]
    assert prod_pauses and set(prod_pauses) == {7.25}


def test_G2n_non_prod_target_partial_sealed_resume_is_refused_by_the_already_served_ceiling(
        tmp_path, url, monkeypatch):
    """CHARACTERIZATION of the NON-prod branch only (`PG.is_prod` false, which is what the throwaway
    harness gives): since PR #660 the `already_served_urls` question is asked only off prod, so a
    partial sealed resume (3 of 4 rows already on the target = 75 %) is REFUSED at live-apply-preflight
    by the 5 % ceiling, before any write. On the serving box the question is not asked (see G2)."""
    r = _crash_then_resume(tmp_path, url, monkeypatch, die_on_prod_call=2)
    f, w, rec = r["first"], r["w"], r["results"][0]
    assert len(f["prod"]) == 3 and len(f["twin"]) == 4
    assert rec["status"] == "REFUSED" and rec["stage"] == "live-apply-preflight"
    assert "already_served_urls: 3 of 4" in rec["reason"] and "over the 5% ceiling" in rec["reason"]
    assert w.log == []
    assert _ids(url, PR) == f["prod"] and _ids(url, TW) == f["twin"]
    assert not (r["queue"] / SEALED / "DONE").exists() and _segments(r["log_dir"]) == []
    assert r["locks"] == _lock_seq("promotion-queue-resume")


def test_G1k_kill_leaves_no_receipt_and_the_resume_still_lands_prod_only(tmp_path, url, monkeypatch):
    """A SIGKILL runs no handler, so no `receipt.json` is written: delete it between the runs."""
    def no_receipt(url, artifacts):
        next(r.parent for r in tmp_path.rglob("receipt.json")).joinpath("receipt.json").unlink()
    r = _crash_then_resume(tmp_path, url, monkeypatch, between=no_receipt)
    f, w, rec = r["first"], r["w"], r["results"][0]
    assert not any(tmp_path.rglob("receipt.json")) or _receipt(r["queue"])["status"] == "LANDED"
    assert w.sent("twin") == [] and w.sent("prod") == f["twin"]
    assert [x[:2] for x in w.log] == [["put", "prod"], ["put", "prod"]]
    assert rec["status"] == "LANDED" and rec["resumed_prod_only"] is True
    assert r["locks"] == _lock_seq("promotion-queue-resume")
    assert (r["queue"] / SEALED / "DONE").exists()
    assert _landed_ids(r["log_dir"]) == [("add", i) for i in f["twin"]]
    _assert_prod_equals_twin(url)


def _drop_artifact(url, artifacts):
    (artifacts / f"{SEALED}-twin-written-ids.json").unlink()


def test_G4_twin_ids_artifact_missing_rebuilds_the_set_from_a_live_twin_check(tmp_path, url, monkeypatch):
    r = _crash_then_resume(tmp_path, url, monkeypatch, between=_drop_artifact)
    f, w, rec = r["first"], r["w"], r["results"][0]
    assert w.sent("twin") == [] and w.sent("prod") == f["twin"]
    assert rec["status"] == "LANDED" and rec["resumed_prod_only"] is True
    assert rec["note"].startswith("duplicate of already-landed pack(s)")
    res = rec["prod"]["resume"]
    assert res["twin_written_ids_source"].startswith("live twin check")
    assert (res["twin_written_ids"], res["written"], res["present_after"]) == (4, 4, 4)
    assert r["locks"] == _lock_seq("promotion-queue-resume")
    assert _landed_ids(r["log_dir"]) == [("add", i) for i in f["twin"]]
    _assert_prod_equals_twin(url)


def test_G4b_artifact_missing_and_an_id_on_neither_side_is_refused(tmp_path, url, monkeypatch):
    gone = []

    def lose_one(url, artifacts):
        _drop_artifact(url, artifacts)
        gone.append(_ids(url, TW)[0])
        H.api(url, f"/collections/{TW}/points/delete?wait=true", {"points": [gone[0]]}, method="POST")

    r = _crash_then_resume(tmp_path, url, monkeypatch, between=lose_one)
    rec = r["results"][0]
    assert rec["status"] == "REFUSED" and rec["stage"] == "live-apply-preflight"
    assert "NEITHER twin nor prod" in rec["reason"] and gone[0] in rec["reason"]
    assert r["w"].log == [] and _ids(url, PR) == []          # nothing written anywhere
    assert not (r["queue"] / SEALED / "DONE").exists() and _segments(r["log_dir"]) == []


def test_G5_alias_moved_between_the_legs_refuses_and_writes_nothing(tmp_path, url, monkeypatch):
    moved = {"done": False}
    queue, artifacts, log_dir, locks, w = _boot(tmp_path, url, monkeypatch, die_on_prod_call=1)
    with pytest.raises(KeyboardInterrupt):
        _drain(queue, url, artifacts)
    del locks[:]
    w.log.clear()
    real = PSP.already_present

    def ap(u, coll, ids, *a, **k):
        # the alias moves while the resume holds the prod lock, after drain() resolved it
        if not moved["done"] and ("acquire", "PROD-WRITE-LOCK", T, f"{SEALED}-promotion-queue-resume") in locks:
            moved["done"] = True
            H.set_alias(url, AL, PR_MOVED)
        return real(u, coll, ids, *a, **k)
    monkeypatch.setattr(PSP, "already_present", ap)
    rec = _drain(queue, url, artifacts)[0]
    assert moved["done"]
    assert rec["status"] == "REFUSED" and rec["stage"] == "live-apply-preflight"
    assert "moved mid-drain" in rec["reason"] and PR_MOVED in rec["reason"]
    assert w.log == []                                       # no PUT at all, to either collection
    assert _ids(url, PR) == [] and _ids(url, PR_MOVED) == []
    assert not (queue / SEALED / "DONE").exists() and _segments(log_dir) == []
    assert locks[-2:] == [("release", "PROD-WRITE-LOCK", T), ("release", "TWIN-WRITE-LOCK", T)]


def test_G6_fresh_sealed_drain_revalidates_the_alias_before_the_prod_write(tmp_path, url, monkeypatch):
    """Task 2902 pin: the alias moves while the twin leg writes. The prod leg must refuse BEFORE its
    first prod PUT (the re-check under the lock, right before the write), not only after it."""
    queue, artifacts, log_dir, locks, w = _boot(tmp_path, url, monkeypatch, die_on_prod_call=None)
    real = PSP.upsert

    def upsert(u, coll, points, batch, *a, **k):
        out = real(u, coll, points, batch, *a, **k)
        if coll == TW:
            H.set_alias(url, AL, PR_MOVED)
        return out
    monkeypatch.setattr(PSP, "upsert", upsert)
    with pytest.raises(Q.TFA.ApplyAbort, match="moved mid-drain"):
        _drain(queue, url, artifacts)
    assert w.sent("prod") == [] and w.sent("moved") == []        # not one PUT to prod, before or after
    assert _ids(url, PR) == [] and len(_ids(url, TW)) == 4
    assert not (queue / SEALED / "DONE").exists() and _segments(log_dir) == []


def test_G2c_sealed_prod_leg_keeps_a_chunk_record_and_the_resume_clears_it(tmp_path, url, monkeypatch):
    """Task 2902 pin: a big sealed pack's prod leg writes a chunk-level record of each PUT batch it
    sent (`<pack>-prod-write.chunks.jsonl`); the resume that follows deletes it."""
    from cytir.index import prod_guard as PG
    _prod_is_the_harness_prod(monkeypatch)
    monkeypatch.setattr(PG, "PROD_BIG_PACK_ROWS", 3)           # 4 rows is a "big" pack: the checkpoint is on
    monkeypatch.setattr(PG, "PROD_BIG_PACK_PAUSE_S", 0.0)
    r = _crash_then_resume(tmp_path, url, monkeypatch, die_on_prod_call=2)
    f, rec = r["first"], r["results"][0]
    ck = [n for n in f["files"] if n.endswith("-prod-write.chunks.jsonl")]
    assert len(ck) == 1, f["files"]                            # one PUT batch landed before the death
    assert not [p for p in r["artifacts"].glob("*-prod-write.chunks.jsonl")]       # the resume cleared it
    assert rec["status"] == "LANDED" and rec["resumed_prod_only"] is True
    _assert_prod_equals_twin(url)


def test_G2d_sealed_prod_leg_clears_its_chunk_record_when_every_batch_is_sent(tmp_path, url, monkeypatch):
    """Task 2902 pin: a fresh big sealed landing leaves no prod-write chunk record behind."""
    from cytir.index import prod_guard as PG
    _prod_is_the_harness_prod(monkeypatch)
    monkeypatch.setattr(PG, "PROD_BIG_PACK_ROWS", 3)
    monkeypatch.setattr(PG, "PROD_BIG_PACK_PAUSE_S", 0.0)
    queue, artifacts, log_dir, locks, w = _boot(tmp_path, url, monkeypatch, die_on_prod_call=None)
    rec = _drain(queue, url, artifacts)[0]
    assert rec["status"] == "LANDED" and len(w.sent("prod")) == 4
    assert not list(artifacts.glob("*-prod-write.chunks.jsonl"))
    _assert_prod_equals_twin(url)


def test_G7_a_stray_write_inside_the_locked_transaction_aborts_before_the_next_leg(tmp_path, url, monkeypatch):
    """Task 2902 pin: the per-leg check that the target grew by no more than this run wrote. Another
    writer adds one point to the twin while the twin leg runs; the transaction stops before prod."""
    queue, artifacts, log_dir, locks, w = _boot(tmp_path, url, monkeypatch, die_on_prod_call=None)
    real = PSP.upsert
    stray = {"id": str(uuid.UUID(bytes=hashlib.sha256(b"task2902-stray").digest()[:16])),
             "payload": {"url": "https://example.invalid/stray"},
             "vector": {"dense": [0.1] * H.MRL, "bm25": {"indices": [1], "values": [1.0]}}}

    def upsert(u, coll, points, batch, *a, **k):
        out = real(u, coll, points, batch, *a, **k)
        if coll == TW:
            H.api(url, f"/collections/{TW}/points?wait=true", {"points": [stray]})
        return out
    monkeypatch.setattr(PSP, "upsert", upsert)
    with pytest.raises(Q.QueueError, match="points_count grew"):
        _drain(queue, url, artifacts)
    assert w.sent("prod") == [] and _ids(url, PR) == []
    assert not (queue / SEALED / "DONE").exists() and _segments(log_dir) == []


# ---------------------------------------------------------------- content-pack scenarios (G2b, G3)
CP = "resume-golden-cp-00001"


def _fixture_rows() -> list[dict]:
    return [json.loads(l) for l in FIX.read_text().splitlines() if l.strip()]


def _clone(row: dict, n: int) -> dict:
    """A further pack row: a new id and a new document url, the rest of the real row as it is."""
    pid = str(uuid.UUID(bytes=hashlib.sha256(f"{row['id']}/{n}".encode()).digest()[:16]))
    pl = dict(row["payload"])
    pl["url"] = pl["url"] + f"00{n}"
    pl["point_id"] = pid
    return {"id": pid, "payload": pl, "vector": row["vector"]}


def _content_queue(tmp: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    queue = tmp / "queue"
    pack_dir = queue / CP
    pack_dir.mkdir(parents=True)
    pts = pack_dir / "points.jsonl"
    pts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = {"schema": Q.MANIFEST_SCHEMA, "kind": "content_pack", "pack_id": CP, "family": H.FAMILY,
                "collection": TW, "pack_points": str(pts),
                "pack_points_sha256": hashlib.sha256(pts.read_bytes()).hexdigest(), "rows": len(rows)}
    body = json.dumps(manifest, indent=1, sort_keys=True).encode()
    (pack_dir / "manifest.json").write_bytes(body)
    (pack_dir / "READY").write_text(hashlib.sha256(body).hexdigest() + "\n")
    return queue


def _cp_locks(purpose_suffix: str) -> list:
    p = f"{CP}-{purpose_suffix}"
    return [("assert_locks_free", T), ("acquire", "TWIN-WRITE-LOCK", T, p),
            ("acquire", "PROD-WRITE-LOCK", T, p), "prod-lock-acquire", "prod-lock-release",
            ("release", "PROD-WRITE-LOCK", T), ("release", "TWIN-WRITE-LOCK", T)]


def _cp_crash_then_resume(tmp, url, monkeypatch, rows, *, die_on_prod_call=1, seed=None):
    queue = _content_queue(tmp, rows)
    queue, artifacts, log_dir, locks, w = _boot(tmp, url, monkeypatch, die_on_prod_call=die_on_prod_call,
                                                queue=queue)
    if seed:
        seed(url)
    with pytest.raises(KeyboardInterrupt):
        _drain(queue, url, artifacts)
    first = {"locks": list(locks), "writes": list(w.log), "twin": _ids(url, TW), "prod": _ids(url, PR),
             "segments": len(_segments(log_dir)), "receipt": json.loads((queue / CP / "receipt.json").read_text()),
             "files": sorted(p.name for p in artifacts.glob("*")),
             "held_file": (queue / CP / "held-out-cards.jsonl").read_text()
             if (queue / CP / "held-out-cards.jsonl").exists() else None}
    del locks[:]
    w.log.clear()
    res = _drain(queue, url, artifacts)
    return {"queue": queue, "artifacts": artifacts, "log_dir": log_dir, "locks": list(locks), "w": w,
            "first": first, "results": res}


def test_G2b_content_pack_chunk_checkpoint_is_not_resent_on_resume(tmp_path, url, monkeypatch):
    from cytir.index import prod_guard as PG
    monkeypatch.setattr(PG, "PROD_BIG_PACK_ROWS", 3)          # 4 rows is a "big" pack: the checkpoint is on
    monkeypatch.setattr(PG, "PROD_BIG_PACK_PAUSE_S", 0.0)
    r = _cp_crash_then_resume(tmp_path, url, monkeypatch, _fixture_rows(), die_on_prod_call=2)
    f, w, rec = r["first"], r["w"], r["results"][0]
    assert [x[:2] for x in f["writes"]] == [["put", "twin"], ["put", "twin"], ["put", "prod"], ["put-died", "prod"]]
    sent_first_prod = f["writes"][2][2]
    assert len(sent_first_prod) == 3 and sorted(f["prod"]) == sent_first_prod
    ck = [n for n in f["files"] if n.endswith("-prod-write.chunks.jsonl")]
    assert len(ck) == 1, f["files"]                           # the first attempt's record of its one PUT batch
    # the resume: the checkpointed batch (3 ids) is NOT sent again, the missing one is; no twin PUT
    missing = sorted(set(f["twin"]) - set(f["prod"]))
    assert w.sent("twin") == [] and w.sent("prod") == missing and len(missing) == 1
    assert rec["status"] == "LANDED" and rec["note"] == "resumed prod-only from a twin-write-complete state record"
    assert rec["prod"]["written"] == 4                        # the receipt counts the skipped batch as written
    assert r["locks"] == _cp_locks("promotion-queue-resume")
    assert (r["queue"] / CP / "DONE").exists()
    _assert_prod_equals_twin(url)


def _stored_card(url, row_id):
    card = {"id": row_id, "payload": {"url": "https://example.invalid/the-card", "source_type": "entity_card",
                                      "snippet": "stored card"},
            "vector": {"dense": [0.1] * H.MRL, "bm25": {"indices": [1], "values": [1.0]}}}
    for coll in (TW, PR):
        H.api(url, f"/collections/{coll}/points?wait=true", {"points": [card]})


def test_G3_held_out_cards_survive_a_prod_only_resume(tmp_path, url, monkeypatch):
    base = _fixture_rows()
    rows = base + [_clone(base[0], 1), _clone(base[1], 2)]      # 6 rows: one held row is under the 20 % ceiling
    held_id = rows[0]["id"]

    def run(sub, die):
        d = tmp_path / sub
        d.mkdir()
        seed = lambda u: _stored_card(u, held_id)               # noqa: E731
        if die:
            return _cp_crash_then_resume(d, url, monkeypatch, rows, die_on_prod_call=1, seed=seed)
        queue = _content_queue(d, rows)
        queue, artifacts, log_dir, locks, w = _boot(d, url, monkeypatch, die_on_prod_call=None, queue=queue)
        seed(url)
        res = _drain(queue, url, artifacts)
        return {"queue": queue, "results": res, "w": w, "log_dir": log_dir}

    fresh = run("fresh", False)
    fresh_rec = fresh["results"][0]
    fresh_prod = [(str(p["id"]), p["payload"]) for p in H.dump_points(url, PR)]
    fresh_held = (fresh["queue"] / CP / "held-out-cards.jsonl").read_text()
    assert fresh_rec["status"] == "LANDED" and fresh_rec["held_out_rows"] == 1 and fresh_rec["held_out_cards"] == 1
    r = run("resumed", True)
    rec = r["results"][0]
    # the interrupted first run already wrote the sidecar and the twin leg; the held row went nowhere
    assert r["first"]["held_file"] == fresh_held and json.loads(fresh_held.splitlines()[0])["point"]["id"] == held_id
    assert len(r["first"]["twin"]) == 6 and r["first"]["prod"] == [held_id]   # 5 pack rows + the stored card
    # the resume: same held-out file, same receipt fields as a fresh run, prod equals the fresh run's prod
    assert rec["status"] == "LANDED" and rec["note"] == "resumed prod-only from a twin-write-complete state record"
    for k in ("held_out_cards", "held_out_rows", "held_out_by_reason", "held_out_cards_reasons",
              "held_out_cards_tombstones"):
        assert rec[k] == fresh_rec[k], k
    assert (r["queue"] / CP / "held-out-cards.jsonl").read_text() == fresh_held
    assert r["w"].sent("twin") == [] and held_id not in r["w"].sent("prod")
    got_prod = [(str(p["id"]), p["payload"]) for p in H.dump_points(url, PR)]
    assert got_prod == fresh_prod
    stored = next(p for p in H.dump_points(url, PR) if str(p["id"]) == held_id)
    assert stored["payload"]["source_type"] == "entity_card" and stored["payload"]["snippet"] == "stored card"
    assert [x for x in _ids(url, TW)] == [x for x in _ids(url, PR)]


def test_G4c_reconstruction_resume_with_part_of_the_pack_already_on_prod_refuses_with_a_collision_and_writes_nothing(
        tmp_path, url, monkeypatch):
    """Task 2926 (finding 2 of review 2900), TODAY'S behaviour. The prod leg died after one of two batches (3 of 4
    rows on prod) AND the twin-written-ids artifact is lost, so the resume RECONSTRUCTS the twin set from a live
    twin check (`twin_ids_override`). The override holds only the id prod misses; the three the pack already put on
    prod are then ids "prod holds that the twin leg did not write": `ResumeCollisionError`, REFUSED at
    live-apply-preflight, nothing written, no DONE, no segment. (A fix lane may turn this into a landing; it must
    change this test with it.) The same drain with the artifact kept lands (G2)."""
    _prod_is_the_harness_prod(monkeypatch)
    r = _crash_then_resume(tmp_path, url, monkeypatch, die_on_prod_call=2, between=_drop_artifact)
    f, w, rec = r["first"], r["w"], r["results"][0]
    assert len(f["prod"]) == 3 and len(f["twin"]) == 4
    assert rec["status"] == "REFUSED" and rec["stage"] == "live-apply-preflight", rec
    assert rec["reason"].startswith("ResumeCollisionError:"), rec["reason"]
    assert w.log == []                                                       # nothing written, either side
    assert _ids(url, PR) == f["prod"] and _ids(url, TW) == f["twin"]
    assert not (r["queue"] / SEALED / "DONE").exists() and _segments(r["log_dir"]) == []
    assert r["locks"] == _lock_seq("promotion-queue-resume")


# ================================================================ task 2929: pins before the resume code moves
# `resume_prod_only` (also the reconstruction resume) and `resume_content_pack_prod_only` move to
# `cytir/index/drain/resume.py` at step 13. These pins run the real drain (or the real function on the state
# a real first run left) and assert the protocol and the end state; each was shown red under a temporary
# edit of the function (the table is in the task doc).
def _ledger(artifacts, pack_id):
    rows = [json.loads(x) for x in (artifacts / f"{pack_id}-resume-ledger.jsonl").read_text().splitlines()]
    for r in rows:
        r.pop("ts")
    return rows


def _install_spies(monkeypatch):
    """What the resume asks of its collaborators: the runner's pid, the writer lock's arguments, the layout
    fingerprints it takes. The harness fakes stay in force; the spies wrap them."""
    from cytir.index import layout as LAYOUT
    seen = {"pids": [], "lock": [], "fp": []}
    rb = Q._load_run_batch()

    class RB:
        LOCKS = rb.LOCKS
        assert_locks_free = staticmethod(rb.assert_locks_free)
        release = staticmethod(rb.release)

        @staticmethod
        def acquire(name, task, why, *, pid=None):
            seen["pids"].append((name, pid))
            rb.acquire(name, task, why, pid=pid)
    monkeypatch.setattr(Q, "_load_run_batch", lambda: RB)
    real_lock = Q.TFA.ProductionWriterLock

    class Lock(real_lock):
        def __init__(self, **kw):
            seen["lock"].append(kw)
            super().__init__(**kw)
    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", Lock)
    real_fp = LAYOUT.fingerprint

    def fp(u, c, **k):
        out = real_fp(u, c, **k)
        seen["fp"].append((u, c, out))
        return out
    monkeypatch.setattr(LAYOUT, "fingerprint", fp)
    return seen


def _first_run_dies(tmp, url, monkeypatch, *, die=1):
    """The first drain: the twin leg lands, the die-th prod batch dies. Returns the world the resume starts in."""
    queue, artifacts, log_dir, locks, w = _boot(tmp, url, monkeypatch, die_on_prod_call=die)
    with pytest.raises(KeyboardInterrupt):
        _drain(queue, url, artifacts)
    del locks[:]
    w.log.clear()
    return queue, artifacts, log_dir, locks, w


def _resume_args(queue, artifacts, url, **over):
    kw = dict(pack_id=SEALED, manifest=json.loads((queue / SEALED / "manifest.json").read_text()),
              twin_url=url, prod_url=url, task=T, batch=3, pause_s=0, artifact_root=artifacts,
              prod_collection=PR, prod_alias=AL)
    kw.update(over)
    return kw


PRESERVATION_SCOPE = ("narrow: RESUME of a sealed pack's prod leg only; twin was already written and verified by a "
                      "prior run and is never rewritten here")
TWIN_GATE_NOTE = ("twin already verified by the ORIGINAL apply; re-confirmed present by this resume's own "
                  "preflight")
LEDGER_END = {"status": "TWIN-PROD-VERIFIED-OK",
              "recovery": "resumed from immutable failed predecessor after verified twin",
              "rollback": "artifact and target readback proofs retained; no destructive automatic rollback"}
LEDGER_STATUSES = ["RESUME-PREFLIGHT-OK", "PRESERVATION-VERIFIED", "RESUME-PREDECESSOR-BOUND", "TWIN-GATE-PASS",
                   "PROD-VERIFIED", "TWIN-PROD-VERIFIED-OK"]


def test_2929_sealed_resume_collaborators_ledger_and_receipt(tmp_path, url, monkeypatch):
    """The resume's arguments to its collaborators and what it records: the runner's pid, the writer lock's
    arguments, the one `apply_side` call (role, endpoint, collection, batch, pause, parallel, ids dir, the
    attempt, the twin's id set), the ledger line by line, the receipt (no private ids in it)."""
    seen, calls, settles = {}, [], []

    def hook(u, a):
        seen.update(_install_spies(monkeypatch))
        real_settle = Q.settle_supersede_deleting

        def settle(*args, **kw):
            settles.append((args, kw))
            return real_settle(*args, **kw)
        monkeypatch.setattr(Q, "settle_supersede_deleting", settle)
        real = Q.apply_side

        def spy(*args, **kw):
            calls.append((args, kw))
            return real(*args, **kw)
        monkeypatch.setattr(Q, "apply_side", spy)
    r = _crash_then_resume(tmp_path, url, monkeypatch, between=hook, pause_s=0.25)
    rec, ids = r["results"][0], r["first"]["twin"]
    assert rec["status"] == "LANDED" and rec["resumed_prod_only"] is True
    assert seen["pids"] == [("TWIN-WRITE-LOCK", os.getpid()), ("PROD-WRITE-LOCK", os.getpid())]
    assert seen["lock"] == [{"host": "lockhost", "lock_path": Q.PROD_LOCK_PATH,
                             "purpose": f"task{T}-{SEALED}-resume"}]
    assert {c for _, c, _ in seen["fp"]} == {PR} and len(seen["fp"]) >= 3       # the resume's own read is prod's
    (args, kw), = calls
    assert args[0] == "prod" and args[1] == url and args[3] == PR and args[2]["collection"] == TW
    assert {k: v for k, v in kw.items() if k != "write_attempt"} == {
        "batch": 3, "pause_s": 0.25, "parallel": 1, "ids_dir": r["artifacts"],
        "resume_twin_ids": frozenset(ids)}
    assert isinstance(kw["write_attempt"], Q.WriteAttempt)
    # the settle of a url-group supersede left at `deleting` runs once, as a WRITE (apply), under the same attempt
    (sargs, skw), = settles
    assert sargs[0]["pack_id"] == SEALED and sargs[2] == r["artifacts"]
    assert skw == {"twin_url": url, "prod_url": url, "prod_collection": PR, "prod_alias": AL, "batch": 3, "task": T,
                   "apply": True, "write_attempt": kw["write_attempt"]}
    art = r["artifacts"] / f"{SEALED}-twin-written-ids.json"
    resume = {"twin_written_ids": 4, "present_before": 0, "dropped_by_twin_decision": 0, "written": 4,
              "resumed_partial": False, "present_after": 4, "dropped_present_after": 0,
              "prod_written_ids_equal_twin": True, "twin_written_ids_source": str(art),
              "twin_written_ids_sha256": hashlib.sha256(art.read_bytes()).hexdigest()}
    led = _ledger(r["artifacts"], SEALED)
    assert [x["status"] for x in led] == LEDGER_STATUSES
    assert led[0] == {"status": "RESUME-PREFLIGHT-OK", "armed": True, "proof": {"twin_ids_count": 4}}
    assert led[1]["proof"] == {"status": "PASS", "scope": PRESERVATION_SCOPE, "pack_id": SEALED}
    assert led[2] == {"status": "RESUME-PREDECESSOR-BOUND", "proof": {"twin_ids_count": 4}}
    assert led[3]["proof"] == {"status": "PASS", "mechanical": True, "pack_id": SEALED, "note": TWIN_GATE_NOTE}
    assert led[4]["proof"] == {"status": "PASS", "role": "prod", "written": 4, "resume": resume}
    assert led[5] == LEDGER_END
    got = rec["prod"]
    assert got["resumed_partial"] is False and "_resume_dropped_ids" not in got
    assert {k: got["resume"][k] for k in resume} == resume
    assert (got["resume"]["projected"], got["resume"]["to_write"]) == (4, 4)       # the plan's own fields stay


def test_2929_sealed_reconstruction_resume_records_its_live_twin_source(tmp_path, url, monkeypatch):
    def hook(u, a):
        _drop_artifact(u, a)
    r = _crash_then_resume(tmp_path, url, monkeypatch, between=hook)
    rec, ids = r["results"][0], r["first"]["twin"]
    assert rec["status"] == "LANDED" and rec["resumed_prod_only"] is True
    assert rec["note"] == ("duplicate of already-landed pack(s) -- resumed via a live twin-vs-prod membership "
                           "diff, no per-pack artifact")
    led = _ledger(r["artifacts"], SEALED)
    assert [x["status"] for x in led] == LEDGER_STATUSES
    assert led[0]["proof"] == {"twin_ids_count": 4}
    resume = led[4]["proof"]["resume"]
    assert resume["twin_written_ids_source"] == "live twin check (twin_ids_override; no per-pack artifact)"
    assert resume["twin_written_ids_sha256"] == hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()
    assert rec["prod"]["resume"]["twin_written_ids_source"] == resume["twin_written_ids_source"]


def test_2929_the_resume_refuses_without_an_id_set_before_it_takes_a_lock(tmp_path, url, monkeypatch):
    queue, artifacts, log_dir, locks, w = _first_run_dies(tmp_path, url, monkeypatch)
    _drop_artifact(url, artifacts)
    with pytest.raises(Q.QueueError, match=r"twin-written-ids\.json .*does not exist -- refusing to guess"):
        Q.resume_prod_only(**_resume_args(queue, artifacts, url))
    assert locks == [("assert_locks_free", T)] and w.log == [] and _ids(url, PR) == []


def test_2929_the_override_wins_over_the_artifact_and_the_leg_writes_exactly_the_override(tmp_path, url, monkeypatch):
    queue, artifacts, log_dir, locks, w = _first_run_dies(tmp_path, url, monkeypatch)
    twin = _ids(url, TW)
    keep = twin[:3]
    state = Q.resume_prod_only(**_resume_args(queue, artifacts, url, twin_ids_override=list(keep)))
    assert _ids(url, PR) == sorted(keep) and w.sent("prod") == sorted(keep)
    assert sorted(state["resume_dropped_ids"]) == sorted(set(twin) - set(keep))
    assert "_resume_dropped_ids" not in state["prod_receipt"]
    assert state["prod_receipt"]["resume"]["twin_written_ids_source"].startswith("live twin check")
    assert state["prod_before"]["points_count"] == 0                      # the PROD collection, before the write
    assert pathlib.Path(state["prod_ids_path"]).parent == artifacts
    assert [x["status"] for x in _ledger(artifacts, SEALED)] == LEDGER_STATUSES
    assert _ledger(artifacts, SEALED)[0]["proof"] == {"twin_ids_count": 3}
    assert locks == _lock_seq("promotion-queue-resume")


def test_2929_a_twin_that_lost_an_id_fails_the_premise_before_any_prod_write_and_the_locks_come_back(
        tmp_path, url, monkeypatch):
    queue, artifacts, log_dir, locks, w = _first_run_dies(tmp_path, url, monkeypatch)
    gone = _ids(url, TW)[0]
    H.api(url, f"/collections/{TW}/points/delete?wait=true", {"points": [gone]}, method="POST")
    with pytest.raises(Q.QueueError, match=r"resume premise failed -- 1 of 4 id\(s\) the twin leg reported written"):
        Q.resume_prod_only(**_resume_args(queue, artifacts, url))
    assert w.log == [] and _ids(url, PR) == []
    assert locks == [("assert_locks_free", T),
                     ("acquire", "TWIN-WRITE-LOCK", T, f"{SEALED}-promotion-queue-resume"),
                     ("acquire", "PROD-WRITE-LOCK", T, f"{SEALED}-promotion-queue-resume"),
                     ("release", "PROD-WRITE-LOCK", T), ("release", "TWIN-WRITE-LOCK", T)]   # no prod lock taken
    assert not (artifacts / f"{SEALED}-resume-ledger.jsonl").exists()      # failed before its first ledger line


def _fault_after_the_prod_write(tmp_path, url, monkeypatch, *, match, on_apply=None, on_present=None):
    """The resume's prod write lands; then one of its own proofs is made to fail. The drain must stop (ABORTED, no
    DONE, no landing), the ledger records the failure, and both locks come back in reverse order."""
    queue, artifacts, log_dir, locks, w = _first_run_dies(tmp_path, url, monkeypatch)
    st = {"written": False, "fired": False}
    real_apply, real_present = Q.apply_side, PSP.already_present

    def apply(*a, **k):
        out = real_apply(*a, **k)
        st["written"] = True
        return on_apply(out) if on_apply else out

    def present(u, coll, ids, *a, **k):
        got = real_present(u, coll, ids, *a, **k)
        if st["written"] and coll == PR and on_present and not st["fired"]:
            st["fired"] = True
            return on_present(url, got, ids)
        return got
    monkeypatch.setattr(Q, "apply_side", apply)
    monkeypatch.setattr(PSP, "already_present", present)
    with pytest.raises(BaseException, match=match):
        _drain(queue, url, artifacts)
    rec = _receipt(queue)
    assert rec["status"] == "ABORTED" and re.search(match, rec["reason"])
    assert not (queue / SEALED / "DONE").exists() and _segments(log_dir) == []
    assert locks[-2:] == [("release", "PROD-WRITE-LOCK", T), ("release", "TWIN-WRITE-LOCK", T)]
    assert _ledger(artifacts, SEALED)[-1]["status"] == "FAILED"
    return queue, artifacts


def test_2929_a_prod_write_that_does_not_show_in_the_membership_proof_aborts(tmp_path, url, monkeypatch):
    _fault_after_the_prod_write(tmp_path, url, monkeypatch, match=r"1 id\(s\) missing after resume upsert",
                                on_present=lambda u, got, ids: sorted(got)[1:])


def test_2929_a_prod_written_ids_artifact_that_differs_from_the_twins_aborts(tmp_path, url, monkeypatch):
    def tamper(out):
        receipt, ids_path = out
        p = pathlib.Path(ids_path)
        p.write_text(json.dumps(json.loads(p.read_text())[1:]))
        return receipt, ids_path
    _fault_after_the_prod_write(tmp_path, url, monkeypatch, match=r"the prod written-ids artifact differs from the twin's",
                                on_apply=tamper)


def test_2929_an_id_the_twin_dropped_that_is_on_prod_aborts(tmp_path, url, monkeypatch):
    def drop(out):
        receipt, ids_path = out
        receipt["_resume_dropped_ids"] = [sorted(json.loads(pathlib.Path(ids_path).read_text()))[0]]
        return receipt, ids_path
    _fault_after_the_prod_write(tmp_path, url, monkeypatch, match=r"1 id\(s\) the twin leg dropped are on prod",
                                on_apply=drop)


def test_2929_resume_arithmetic_that_does_not_close_aborts(tmp_path, url, monkeypatch):
    def bump(out):
        receipt, ids_path = out
        receipt["written"] += 1
        return receipt, ids_path
    _fault_after_the_prod_write(tmp_path, url, monkeypatch, match=r"resume arithmetic does not close", on_apply=bump)


def test_2929_the_alias_moving_after_the_prod_write_is_the_last_check_before_done(tmp_path, url, monkeypatch):
    _fault_after_the_prod_write(tmp_path, url, monkeypatch, match=r"moved mid-drain",
                                on_present=lambda u, got, ids: (H.set_alias(u, AL, PR_MOVED), got)[1])


def _endpoint_recorder(monkeypatch):
    """`(url, collection)` of every request the resume sends through `PSP._api` and of every layout fingerprint."""
    from cytir.index import layout as LAYOUT
    sent = []
    real = PSP._api

    def api(u, path, body=None, method=None, *a, **k):
        m = re.match(r"/collections/([^/]+)", path)
        if m:
            sent.append((u, m.group(1)))
        return real(u, path, body, method, *a, **k)
    monkeypatch.setattr(PSP, "_api", api)
    real_fp = LAYOUT.fingerprint

    def fp(u, c, **k):
        sent.append((u, c))
        return real_fp(u, c, **k)
    monkeypatch.setattr(LAYOUT, "fingerprint", fp)
    return sent


def test_2929_each_leg_of_the_sealed_resume_talks_to_its_own_endpoint(tmp_path, url, monkeypatch):
    """Twin and prod are told apart by the url string (one throwaway server answers both): the twin is only
    read through `twin_url`, the prod collection only through `prod_url`."""
    queue, artifacts, log_dir, locks, w = _first_run_dies(tmp_path, url, monkeypatch)
    sent = _endpoint_recorder(monkeypatch)
    prod_url = url + "/"
    Q.resume_prod_only(**_resume_args(queue, artifacts, url, prod_url=prod_url))
    assert {(u, c) for u, c in sent} == {(url, TW), (prod_url, PR)}
    assert (prod_url, PR) in sent and sent.count((url, TW)) >= 2            # the preflight runs twice


def test_2929_the_alias_moved_before_the_sealed_resume_refuses_before_the_prod_write(tmp_path, url, monkeypatch):
    queue, artifacts, log_dir, locks, w = _first_run_dies(tmp_path, url, monkeypatch)
    H.set_alias(url, AL, PR_MOVED)
    with pytest.raises(BaseException, match="moved mid-drain"):
        Q.resume_prod_only(**_resume_args(queue, artifacts, url))
    assert w.log == [] and _ids(url, PR) == [] and _ids(url, PR_MOVED) == []
    assert _ledger(artifacts, SEALED)[-1]["status"] == "FAILED"


# ---------------------------------------------------------------- task 2933: the change-log entry comes BEFORE DONE
def _boom(*_a, **_k):
    from cytir.index import change_log as CL
    raise CL.ChangeLogError("log failed on purpose")


def _no_done_aborted(queue, pack, log_dir):
    """What a change-log failure leaves: no DONE, an ABORTED receipt, no segment."""
    assert not (queue / pack / "DONE").exists()
    rc = json.loads((queue / pack / "receipt.json").read_text())
    assert rc["status"] == "ABORTED" and "log failed on purpose" in rc["reason"]
    assert _segments(log_dir) == []


def _next_pass_finishes(monkeypatch, queue, pack, url, artifacts, log_dir, hook, real):
    """The log is healthy again: the next pass writes the one segment and then DONE."""
    monkeypatch.setattr(Q, hook, real)
    res = _drain(queue, url, artifacts)
    assert res[0]["status"] == "LANDED" and (queue / pack / "DONE").exists()
    assert len(_segments(log_dir)) == 1


@pytest.mark.parametrize("kind", ["sealed", "content"])
def test_2933_a_resume_that_cannot_log_writes_no_done_and_the_next_pass_retries(tmp_path, url, monkeypatch, kind):
    """Order: the resume writes the change-log entry BEFORE DONE. The log raises after the prod leg wrote:
    no DONE, an ABORTED receipt, the drain stops; the next pass (log healthy) writes the segment, then DONE."""
    from cytir.index import change_log as CL
    pack = SEALED if kind == "sealed" else CP
    queue = None if kind == "sealed" else _content_queue(tmp_path, _fixture_rows())
    queue, artifacts, log_dir, locks, w = _boot(tmp_path, url, monkeypatch, die_on_prod_call=1, queue=queue)
    with pytest.raises(KeyboardInterrupt):
        _drain(queue, url, artifacts)
    real = Q._log_resumed_landing
    monkeypatch.setattr(Q, "_log_resumed_landing", _boom)
    with pytest.raises(CL.ChangeLogError):
        _drain(queue, url, artifacts)
    _no_done_aborted(queue, pack, log_dir)
    assert _ids(url, PR) == _ids(url, TW) and len(_ids(url, PR)) == 4      # the prod leg DID write
    _next_pass_finishes(monkeypatch, queue, pack, url, artifacts, log_dir, "_log_resumed_landing", real)


@pytest.mark.parametrize("kind,hook", [("sealed", "_log_sealed_landing"), ("content", "_log_landing")])
def test_2933_a_fresh_landing_that_cannot_log_writes_no_done_and_the_next_pass_retries(
        tmp_path, url, monkeypatch, kind, hook):
    from cytir.index import change_log as CL
    pack = SEALED if kind == "sealed" else CP
    queue = None if kind == "sealed" else _content_queue(tmp_path, _fixture_rows())
    queue, artifacts, log_dir, locks, w = _boot(tmp_path, url, monkeypatch, die_on_prod_call=None, queue=queue)
    real = getattr(Q, hook)
    monkeypatch.setattr(Q, hook, _boom)
    with pytest.raises(CL.ChangeLogError):
        _drain(queue, url, artifacts)
    _no_done_aborted(queue, pack, log_dir)
    _next_pass_finishes(monkeypatch, queue, pack, url, artifacts, log_dir, hook, real)


# ---------------------------------------------------------------- task 2933: the content-pack resume with PG.is_prod TRUE
def test_2933_content_pack_resume_on_a_prod_target_lands_like_a_non_prod_one_and_counts_its_prod_calls(
        tmp_path, url, monkeypatch):
    """G2b ran the content-pack resume with `PG.is_prod` false. Here the harness prod counts as the serving
    box. Today's behaviour: the same resume (only the missing row is sent, the checkpointed batch is not, the
    same locks, DONE, one change-log landing); the one difference is the receipt's prod-call statistics,
    which count the calls made to a prod target (`calls` is 0 off prod, as G2b's run shows)."""
    from cytir.index import prod_guard as PG
    monkeypatch.setattr(PG, "PROD_BIG_PACK_ROWS", 3)
    monkeypatch.setattr(PG, "PROD_BIG_PACK_PAUSE_S", 0.0)
    monkeypatch.setattr(PG, "PROD_BATCH_SLEEP_S", 0.0)
    _prod_is_the_harness_prod(monkeypatch)
    r = _cp_crash_then_resume(tmp_path, url, monkeypatch, _fixture_rows(), die_on_prod_call=2)
    f, w, rec = r["first"], r["w"], r["results"][0]
    missing = sorted(set(f["twin"]) - set(f["prod"]))
    assert len(f["prod"]) == 3 and len(missing) == 1
    assert w.sent("twin") == [] and w.sent("prod") == missing
    assert rec["status"] == "LANDED" and rec["note"] == "resumed prod-only from a twin-write-complete state record"
    assert rec["prod"]["written"] == 4
    assert rec["prod"]["prod_calls"]["calls"] >= 1                # the prod branch ran: calls to prod are counted
    assert r["locks"] == _cp_locks("promotion-queue-resume")
    assert (r["queue"] / CP / "DONE").exists() and len(_segments(r["log_dir"])) == 1
    _assert_prod_equals_twin(url)


# ---------------------------------------------------------------- task 2933: a resume that starts without an artifact root
def test_2933_the_sealed_resume_creates_a_missing_artifact_root_and_lands(tmp_path, url, monkeypatch):
    """R09 of task 2929: the resume makes its artifact directory itself. The first run's artifacts are gone
    (the id set comes from the live twin, the `twin_ids_override` route, since the written-ids artifact is the
    one thing the resume refuses to guess): the leg lands, and the ledger and the prod written-ids file are
    in the directory it made."""
    import shutil
    queue, artifacts, log_dir, locks, w = _first_run_dies(tmp_path, url, monkeypatch)
    twin = _ids(url, TW)
    shutil.rmtree(artifacts)
    assert not artifacts.exists()
    state = Q.resume_prod_only(**_resume_args(queue, artifacts, url, twin_ids_override=list(twin)))
    assert _ids(url, PR) == twin and artifacts.is_dir()
    assert (artifacts / f"{SEALED}-resume-ledger.jsonl").is_file()
    assert pathlib.Path(state["prod_ids_path"]).parent == artifacts and pathlib.Path(state["prod_ids_path"]).is_file()


def test_2933_the_content_pack_resume_creates_a_missing_artifact_root_and_lands(tmp_path, url, monkeypatch):
    import shutil
    queue = _content_queue(tmp_path, _fixture_rows())
    queue, artifacts, log_dir, locks, w = _boot(tmp_path, url, monkeypatch, die_on_prod_call=1, queue=queue)
    with pytest.raises(KeyboardInterrupt):
        _drain(queue, url, artifacts)
    manifest = json.loads((queue / CP / "manifest.json").read_text())
    sha = hashlib.sha256((queue / CP / "manifest.json").read_bytes()).hexdigest()
    st = Q.read_content_pack_twin_state(artifacts, CP, manifest_sha=sha)
    assert st["stage"] == "twin-write-complete"
    shutil.rmtree(artifacts)
    state = Q.resume_content_pack_prod_only(
        pack_id=CP, manifest=manifest, twin_url=url, prod_url=url, task=T, batch=3, pause_s=0,
        artifact_root=artifacts, state_record=st, prod_collection=PR, prod_alias=AL)
    assert _ids(url, PR) == _ids(url, TW) and len(_ids(url, PR)) == 4
    assert artifacts.is_dir() and (artifacts / f"{CP}-resume-ledger.jsonl").is_file()
    assert state["prod_receipt"]["written"] == 4
