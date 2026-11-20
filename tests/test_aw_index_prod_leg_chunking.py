"""Task 786: the drain's prod leg survives a 1.4M-row pack on the 30 GB box.

Incident 2026-10-08, tranche 14 (1,409,761 rows): the twin leg landed, then the prod leg aborted on its
first prod call, `POST /collections/aw_e9/points -> HTTP 408`. That call is one id retrieve of
`already_present` (500 ids; the pack is 2,820 of them), and `_api` treated every HTTP status as final.

Pins: the lookup is chunked and bounded; a transient status (408, 429, 5xx) retries with exponential
backoff, bounded, and then raises the same clean error; a retry re-enters at the first unfinished chunk
or batch; a big pack's prod writes are paced and an ordinary pack's are not; the receipt records the cost;
(the AST hashes of the untouched functions are in the task doc: the public cut rewrites shipped source, so a pin cannot live here).
"""
from __future__ import annotations

import io
import json
import pathlib
import urllib.error
import urllib.request

import pytest

from cytir.index import chunk_checkpoint as CK
from cytir.index import prod_guard as PG
from cytir.index import promote_queue as PQ
from cytir.index import promote_sealed_pack as P

_H = "127.0.0.1"
PROD = "http://%s:%d" % (_H, PG.PROD_PORT)
TWIN = "http://%s:%d" % (_H, PG.PROD_PORT + 123)
ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("AW_PROD_READ_BUDGET", "unlimited")
    monkeypatch.delenv("AW_PROD_URL", raising=False)
    PG.reset()
    P.PROD_STATS.reset()
    sleeps = []
    monkeypatch.setattr(P.time, "sleep", sleeps.append)
    monkeypatch.setattr(PG, "PROD_BATCH_SLEEP_S", 0)
    return sleeps


class _Resp:
    def __init__(self, body: bytes):
        self._b = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._b


def _http_error(req, code):
    return urllib.error.HTTPError(req.full_url, code, "x", hdrs=None, fp=io.BytesIO(b"slow"))


class FakeProd:
    """urlopen stand-in. `fail(n_call, ids)` -> a status to answer instead of 200, or None."""

    def __init__(self, held=(), fail=lambda n, ids: None):
        self.held, self.fail, self.calls, self.sizes = set(held), fail, 0, []

    def __call__(self, req, timeout=300):
        self.calls += 1
        body = json.loads(req.data) if req.data else {}
        ids = body.get("ids") or [p["id"] for p in body.get("points", [])]
        code = self.fail(self.calls, ids)
        if code:
            raise _http_error(req, code)
        self.sizes.append(len(ids))
        if req.get_method() == "PUT":
            return _Resp(b'{"status": "ok", "result": {}}')
        return _Resp(json.dumps({"status": "ok",
                                 "result": [{"id": i} for i in ids if i in self.held]}).encode())


def _ids(n):
    return [f"id{i:07d}" for i in range(n)]


# ------------------------------------------------------------------ the arithmetic
def test_a_1_4m_id_pack_is_2820_requests_of_500_never_one(monkeypatch):
    seen = []

    def fake(url, path, body=None, method=None, **kw):
        seen.append(len(body["ids"]))
        return []

    monkeypatch.setattr(P, "_api", fake)
    ids = _ids(1_409_761)
    assert P.already_present(PROD, "aw_e9", ids) == []
    assert len(seen) == 2820 and max(seen) == 500 == P.ALREADY_PRESENT_BATCH
    assert sum(seen) == len(ids)


def test_the_lookup_batch_is_declared_and_capped(monkeypatch):
    monkeypatch.setattr(P, "_api", lambda *a, **k: [])
    assert P.ALREADY_PRESENT_BATCH_MAX == 5000
    P.already_present(PROD, "aw_e9", _ids(10), batch=5000)
    for bad in (0, 5001, 1_409_761):
        with pytest.raises(P.PromotionError):
            P.already_present(PROD, "aw_e9", _ids(10), batch=bad)


# ------------------------------------------------------------- bounded status retry
def test_a_408_on_the_first_call_is_retried_and_the_pack_goes_on(monkeypatch, _env):
    fake = FakeProd(held=_ids(30), fail=lambda n, ids: 408 if n == 1 else None)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    got = P.already_present(PROD, "aw_e9", _ids(1200), batch=500)
    assert len(got) == 30
    assert [s for s in _env if s] == [5], "one retry, first backoff step"
    snap = P.PROD_STATS.snapshot()
    assert snap["retries"] == 1 and snap["retried_statuses"] == {"408": 1}
    assert snap["lookup_chunks"] == 3 and snap["longest_call_s"] >= 0


def test_the_big_request_408s_and_the_chunks_succeed(monkeypatch):
    """The brief's fake: a request over 5,000 ids is answered 408 every time; chunked calls pass."""
    fake = FakeProd(held=_ids(50), fail=lambda n, ids: 408 if len(ids) > 5000 else None)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with pytest.raises(P.PromotionError):               # the old shape: one huge request
        P._api(PROD, "/collections/aw_e9/points", {"ids": _ids(20000)})
    assert len(P.already_present(PROD, "aw_e9", _ids(20000))) == 50


def test_the_retry_is_bounded_exponential_and_ends_in_the_clean_error(monkeypatch, _env):
    fake = FakeProd(fail=lambda n, ids: 408)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with pytest.raises(P.PromotionError) as e:
        P.already_present(PROD, "aw_e9", _ids(10))
    assert "HTTP 408" in str(e.value) and e.value.http_status == 408
    assert fake.calls == P._API_STATUS_RETRIES + 1          # never infinite
    assert [s for s in _env if s] == [5, 15, 45, 90]                          # exponential
    assert P.PROD_STATS.snapshot()["retries"] == P._API_STATUS_RETRIES


@pytest.mark.parametrize("code,retried", [(408, True), (429, True), (500, True), (502, True),
                                          (503, True), (504, True), (400, False), (404, False),
                                          (403, False)])
def test_only_transient_statuses_are_retried(monkeypatch, code, retried):
    fake = FakeProd(fail=lambda n, ids: code)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with pytest.raises(P.PromotionError):
        P._api(PROD, "/collections/aw_e9/points", {"ids": ["a"]})
    assert fake.calls == (P._API_STATUS_RETRIES + 1 if retried else 1)


# ---------------------------------------------------------------- chunk-level resume
def test_a_retry_after_an_abort_re_enters_at_the_first_unfinished_chunk(monkeypatch, tmp_path):
    ids = _ids(5000)
    held = set(ids[::7])
    path = tmp_path / "p-prod-lookup.chunks.jsonl"
    # attempt 1: chunk 5 (ids 2500..2999) answers 408 beyond every retry -> clean abort
    fake1 = FakeProd(held=held, fail=lambda n, a: 408 if a and a[0] == ids[2500] else None)
    monkeypatch.setattr(urllib.request, "urlopen", fake1)
    with pytest.raises(P.PromotionError):
        P.already_present(PROD, "aw_e9", ids, checkpoint=CK.ChunkCheckpoint(path))
    assert len(CK.ChunkCheckpoint(path)) == 5               # chunks 0..4 are recorded
    # attempt 2 (a new process: the checkpoint is re-read from disk) asks only chunks 5..9
    fake2 = FakeProd(held=held)
    monkeypatch.setattr(urllib.request, "urlopen", fake2)
    ck = CK.ChunkCheckpoint(path)
    got = P.already_present(PROD, "aw_e9", ids, checkpoint=ck)
    assert fake2.calls == 5 and ck.resumed == 5
    assert sorted(got) == sorted(held), "same answer as an uninterrupted run"
    assert P.PROD_STATS.snapshot()["lookup_chunks_resumed"] == 5


def test_the_upsert_resumes_at_the_first_unsent_batch(monkeypatch, tmp_path):
    pts = [{"id": i, "vector": {}, "payload": {}} for i in _ids(2000)]
    monkeypatch.setattr(P.SR.SparseRule, "for_collection", classmethod(lambda c, n: None))
    ident = type("I", (), {"project_points": lambda self, p: p})()
    monkeypatch.setattr(P.PP.Projector, "for_collection", classmethod(lambda c, n: ident))
    path = tmp_path / "w.chunks.jsonl"
    fake1 = FakeProd(fail=lambda n, a: 503 if a and a[0] == pts[1000]["id"] else None)
    monkeypatch.setattr(urllib.request, "urlopen", fake1)
    with pytest.raises(P.PromotionError):
        P.upsert(PROD, "aw_e9", pts, 500, checkpoint=CK.ChunkCheckpoint(path))
    fake2 = FakeProd()
    monkeypatch.setattr(urllib.request, "urlopen", fake2)
    assert P.upsert(PROD, "aw_e9", pts, 500, checkpoint=CK.ChunkCheckpoint(path)) == 2000
    assert fake2.calls == 2, "batches 0 and 1 were sent by attempt 1"


def test_a_checkpoint_is_keyed_by_collection_and_ids_and_tolerates_a_torn_line(tmp_path):
    path = tmp_path / "c.jsonl"
    ck = CK.ChunkCheckpoint(path)
    k = CK.chunk_key("aw_e9", ["a", "b"])
    ck.put(k, ["a"])
    with path.open("a") as f:
        f.write('{"k": "tor')                                # a kill mid-write
    again = CK.ChunkCheckpoint(path)
    assert again.get(k) == ["a"]
    assert again.get(CK.chunk_key("aw_e8", ["a", "b"])) is None     # another collection
    assert again.get(CK.chunk_key("aw_e9", ["b", "a"])) is None     # other ids
    again.clear()
    assert not path.exists()


def test_a_checkpoint_never_writes_in_a_dry_run(tmp_path):
    from cytir.index import dry_run as DRY
    with pytest.raises(DRY.DryRunWriteRefused):      # recorded at the call, raised when the guard exits
        with DRY.guard(True):
            CK.ChunkCheckpoint(tmp_path / "d.jsonl").put("k", [])
    assert not (tmp_path / "d.jsonl").exists()


# --------------------------------------------------------------------------- pacing
def test_a_big_pack_paces_its_prod_writes_and_nothing_else_changes():
    assert PG.write_pause("prod", PG.PROD_BIG_PACK_ROWS, 0.3) == PG.PROD_BIG_PACK_PAUSE_S
    assert PG.write_pause("prod", 1_409_761, 2.5) == 2.5, "never lowers a larger pause"
    assert PG.write_pause("prod", PG.PROD_BIG_PACK_ROWS - 1, 0.3) == 0.3
    assert PG.write_pause("twin", 5_000_000, 0.3) == 0.3, "the twin leg is unchanged"


def test_apply_content_pack_side_sleeps_longer_for_a_big_prod_pack(monkeypatch, _env):
    monkeypatch.setattr(PG, "PROD_BIG_PACK_ROWS", 4)
    seen = []

    def fake_upsert(url, coll, points, batch, pause_s=0.0, **kw):
        seen.append(pause_s)
        return len(points)

    monkeypatch.setattr(PQ.PSP, "upsert", fake_upsert)
    monkeypatch.setattr(PQ.LAYOUT, "fingerprint", lambda u, c: {"points_count": 0})
    pts = [{"id": str(i)} for i in range(6)]
    PQ.apply_content_pack_side("prod", PROD, "aw_e9", pts, [], batch=2, pause_s=0.3)
    PQ.apply_content_pack_side("twin", TWIN, "aw_e8", pts, [], batch=2, pause_s=0.3)
    PQ.apply_content_pack_side("prod", PROD, "aw_e9", pts[:3], [], batch=2, pause_s=0.3)
    assert seen == [PG.PROD_BIG_PACK_PAUSE_S, 0.3, 0.3]


# --------------------------------------------------------------------------- receipt
def test_the_receipt_of_an_ordinary_pack_is_unchanged_and_a_big_pack_records_its_cost():
    before = P.PROD_STATS.snapshot()
    P.PROD_STATS.note_call(0.2)
    assert PQ.prod_call_receipt(10_000, before) is None, "small pack, no retry: no new key"
    P.PROD_STATS.note_chunks(2820, 0)
    P.PROD_STATS.note_call(7.5)
    P.PROD_STATS.note_retry("408")
    r = PQ.prod_call_receipt(1_409_761, before)
    assert r["rows"] == 1_409_761 and r["lookup_ids_per_request"] == 500
    assert r["lookup_chunks"] == 2820 and r["retries"] == 1 and r["longest_call_s"] == 7.5
    assert r["retried_statuses"] == {"408": 1} and r["write_pause_s_floor"] == 1.0
    assert PQ.prod_call_receipt(10_000, before)["retries"] == 1, "a small pack that retried says so"


def test_twin_calls_are_not_counted_as_prod_calls(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", FakeProd())
    P.already_present(TWIN, "aw_e8", _ids(1000))
    assert P.PROD_STATS.snapshot()["calls"] == 0


# ----------------------------------------------- the transaction, end to end (fakes)
class _FakeLock:
    def __init__(self, *_a, **_k):
        pass

    def acquire(self):
        pass

    def release(self):
        pass


def _transaction_world(monkeypatch, *, fail_prod_upsert):
    monkeypatch.setattr(PQ, "prod_lock_host", lambda: "lockhost")
    """live_apply_content_pack over fakes of the Qdrant primitives; a held set per role."""
    class FakeRB:
        from cytir.ops.write_lock import LOCKS
        assert_locks_free = staticmethod(lambda task: {})
        acquire = staticmethod(lambda name, task, purpose, *, pid=None: None)
        release = staticmethod(lambda name, task: None)

    held = {"twin": set(), "prod": set()}
    log = {"lookup_kwargs": [], "upsert_kwargs": [], "prod_upserts": 0}

    def role(url):
        return "twin" if "twin" in url else "prod"

    def fake_present(url, coll, ids, batch=500, workers=1, **kw):
        if role(url) == "prod":
            log["lookup_kwargs"].append(sorted(kw))
        return [i for i in ids if i in held[role(url)]]

    def fake_upsert(url, coll, pts, batch, pause_s=0.0, **kw):
        if role(url) == "prod":
            log["upsert_kwargs"].append(sorted(kw))
            ck = kw.get("checkpoint")
            for i in range(0, len(pts), batch):
                chunk = pts[i:i + batch]
                key = CK.chunk_key(coll, [p["id"] for p in chunk])
                if ck is not None and ck.get(key) is not None:
                    continue
                if fail_prod_upsert and log["prod_upserts"] == 1:
                    raise P.PromotionError("POST -> HTTP 408 Request Timeout")
                log["prod_upserts"] += 1
                held["prod"] |= {p["id"] for p in chunk}
                if ck is not None:
                    ck.put(key, [])
            return len(pts)
        held["twin"] |= {p["id"] for p in pts}
        return len(pts)

    counts = {"n": 0}

    def fp(*_a, **_k):
        counts["n"] += 1
        return {"points_count": 0}

    monkeypatch.setattr(PQ, "_load_run_batch", lambda: FakeRB)
    monkeypatch.setattr(PQ.TFA, "ProductionWriterLock", _FakeLock)
    monkeypatch.setattr(PQ.PSP, "already_present", fake_present)
    monkeypatch.setattr(PQ.PSP, "upsert", fake_upsert)
    monkeypatch.setattr(PQ.LAYOUT, "fingerprint", fp)
    monkeypatch.setattr(PQ, "verify_written_form", lambda *a, **k: {"form": "ok"})
    monkeypatch.setattr(PQ, "_revalidate_prod_collection", lambda *a, **k: None)
    monkeypatch.setattr(PQ.UGS, "reverify_generation_export", lambda *a, **k: None)
    return held, log


def _run(tmp_path, n):
    pts = [{"id": i, "payload": {}} for i in _ids(n)]
    return PQ.live_apply_content_pack(
        pack_id="pk", manifest={"kind": "content_pack", "pack_id": "pk", "collection": "aw_e8"},
        twin_url="http://twin", prod_url="http://prod", task="786-test", batch=2, pause_s=0.1,
        artifact_root=tmp_path, points=pts, tombstones=[], prod_collection="aw_e9",
        manifest_sha="ab" * 32)


def test_a_small_pack_calls_and_receipts_exactly_as_before(tmp_path, monkeypatch):
    held, log = _transaction_world(monkeypatch, fail_prod_upsert=False)
    state = _run(tmp_path, 6)
    assert log["lookup_kwargs"] == [[], []] and log["upsert_kwargs"] == [[]]
    assert "prod_calls" not in state["prod_receipt"]
    assert not list(tmp_path.glob("*.chunks.jsonl"))


def test_a_big_pack_checkpoints_its_prod_leg_records_the_cost_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr(PG, "PROD_BIG_PACK_ROWS", 4)
    held, log = _transaction_world(monkeypatch, fail_prod_upsert=False)
    state = _run(tmp_path, 6)
    assert log["lookup_kwargs"] == [["checkpoint"], []] and log["upsert_kwargs"] == [["checkpoint"]]
    calls = state["prod_receipt"]["prod_calls"]
    assert calls["rows"] == 6 and calls["lookup_ids_per_request"] == 500
    assert {"retries", "longest_call_s", "lookup_chunks"} <= set(calls)
    assert not list(tmp_path.glob("*.chunks.jsonl")), "spent records are removed once prod verified"


def test_an_aborted_big_pack_leaves_its_chunk_records_for_the_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(PG, "PROD_BIG_PACK_ROWS", 4)
    held, log = _transaction_world(monkeypatch, fail_prod_upsert=True)
    with pytest.raises(P.PromotionError, match="408"):
        _run(tmp_path, 6)
    written = list(tmp_path.glob("pk-*-prod-write.chunks.jsonl"))
    assert len(written) == 1
    assert len(CK.ChunkCheckpoint(written[0])) == 1, "batch 0 was sent and is recorded"


# ---------------------------------- the sealed-pack path (the one tranche 14 actually took)
def _apply_side_world(monkeypatch):
    """apply_side over fakes: records the kwargs plan() and upsert() get."""
    got = {}

    def fake_plan(**kw):
        got["plan"] = sorted(k for k in kw if k == "lookup_checkpoint")
        return {"_joined": {"a": 1}, "_matrix": None, "_id_index": {"a": 0}}

    def fake_upsert(url, coll, pts, batch, pause_s=0.0, parallel=1, **kw):
        got["upsert"] = sorted(kw)
        got["pause_s"] = pause_s
        return len(pts)

    monkeypatch.setattr(PQ.PSP, "plan", fake_plan)
    monkeypatch.setattr(PQ.PSP, "build_points", lambda *a, **k: [{"id": str(i)} for i in range(20)])
    monkeypatch.setattr(PQ.PSP, "upsert", fake_upsert)
    monkeypatch.setattr(PQ.LAYOUT, "fingerprint", lambda *a, **k: {"points_count": 1})
    monkeypatch.setattr(PQ, "plan_kwargs_with_scope", lambda m: ({}, None, None))
    return got


def _manifest(rows):
    return {"pack_id": "sp", "pack_chunks": "c", "vectors": "v", "mrl": 128, "rows": rows}


def test_the_sealed_prod_leg_of_a_big_pack_resumes_by_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(PG, "PROD_BIG_PACK_ROWS", 10)
    got = _apply_side_world(monkeypatch)
    PQ.apply_side("prod", PROD, _manifest(50), "aw_e9", batch=1000, pause_s=0.3, ids_dir=tmp_path)
    assert got["plan"] == ["lookup_checkpoint"] and got["upsert"] == ["checkpoint"]
    assert got["pause_s"] == 1.0
    assert not list(tmp_path.glob("*.chunks.jsonl")), "spent once the batches are sent"


def test_the_sealed_twin_leg_and_a_small_pack_are_called_exactly_as_before(tmp_path, monkeypatch):
    monkeypatch.setattr(PG, "PROD_BIG_PACK_ROWS", 30)
    got = _apply_side_world(monkeypatch)
    PQ.apply_side("twin", TWIN, _manifest(50), "aw_e8", batch=1000, pause_s=0.3, ids_dir=tmp_path)
    assert got["plan"] == [] and got["upsert"] == []
    PQ.apply_side("prod", PROD, _manifest(5), "aw_e9", batch=1000, pause_s=0.3, ids_dir=tmp_path)
    assert got["plan"] == [] and got["upsert"] == [] and got["pause_s"] == 0.3


# ------------------------------------------------------------ task 787 review fixes
def test_a_lookup_record_is_spent_when_plan_refuses_after_the_lookup(tmp_path, monkeypatch):
    """must-fix 2: AlreadyLandedError after a complete lookup must not leave a record that later
    attempts would read INSTEAD of prod."""
    monkeypatch.setattr(PG, "PROD_BIG_PACK_ROWS", 10)
    monkeypatch.setattr(urllib.request, "urlopen", FakeProd(held=_ids(3)))

    def plan_refuses(**kw):
        P.already_present(PROD, "aw_e9", _ids(1200), batch=500, checkpoint=kw["lookup_checkpoint"])
        raise P.AlreadyLandedError("3 of 1200 ids are already in aw_e9",
                                   ids=frozenset(), present=frozenset())

    monkeypatch.setattr(PQ.PSP, "plan", plan_refuses)
    monkeypatch.setattr(PQ, "plan_kwargs_with_scope", lambda m: ({}, None, None))
    with pytest.raises(P.AlreadyLandedError):
        PQ.apply_side("prod", PROD, _manifest(50), "aw_e9", batch=1000, pause_s=0, ids_dir=tmp_path)
    assert not list(tmp_path.glob("*.chunks.jsonl"))


def test_a_lookup_record_is_kept_when_the_lookup_itself_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(PG, "PROD_BIG_PACK_ROWS", 10)
    monkeypatch.setattr(urllib.request, "urlopen",
                        FakeProd(fail=lambda n, a: 503 if a and a[0] == _ids(1200)[1000] else None))

    def plan_dies_in_lookup(**kw):
        P.already_present(PROD, "aw_e9", _ids(1200), batch=500, checkpoint=kw["lookup_checkpoint"])

    monkeypatch.setattr(PQ.PSP, "plan", plan_dies_in_lookup)
    monkeypatch.setattr(PQ, "plan_kwargs_with_scope", lambda m: ({}, None, None))
    with pytest.raises(P.PromotionError, match="503"):
        PQ.apply_side("prod", PROD, _manifest(50), "aw_e9", batch=1000, pause_s=0, ids_dir=tmp_path)
    (rec,) = tmp_path.glob("*-prod-lookup.chunks.jsonl")
    assert len(CK.ChunkCheckpoint(rec)) == 2, "chunks 0 and 1 answered; chunk 2 is the one to re-ask"


def test_the_leg_has_a_status_retry_budget_and_a_non_leg_call_has_none(monkeypatch, _env):
    monkeypatch.setattr(P, "LEG_STATUS_RETRY_BUDGET", 3)
    monkeypatch.setattr(urllib.request, "urlopen", FakeProd(fail=lambda n, a: 503 if n % 2 else None))
    with P.retry_leg():
        for _ in range(3):
            P._api(PROD, "/collections/aw_e9/points", {"ids": ["a"]})      # each call: 1 retry
        with P.retry_leg():                                                  # nested: the same budget
            with pytest.raises(P.PromotionError, match="budget is spent"):
                P._api(PROD, "/collections/aw_e9/points", {"ids": ["a"]})
    assert P._LEG["left"] is None and P._LEG["depth"] == 0
    for _ in range(10):                                                      # outside a leg: unlimited
        P._api(PROD, "/collections/aw_e9/points", {"ids": ["a"]})


def test_the_final_abort_of_a_prod_call_names_the_tunnel_a_twin_abort_does_not(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", FakeProd(fail=lambda n, a: 408))
    with pytest.raises(P.PromotionError, match=r"aw-prod-tunnel.*:6334") as e:
        P._api(PROD, "/collections/aw_e9/points", {"ids": ["a"]})
    assert e.value.http_status == 408
    with pytest.raises(P.PromotionError) as e2:
        P._api(TWIN, "/collections/aw_e8/points", {"ids": ["a"]})
    assert "aw-prod-tunnel" not in str(e2.value)


def test_the_receipt_counts_one_leg_not_the_process(monkeypatch):
    monkeypatch.setattr(PG, "PROD_BIG_PACK_ROWS", 1)
    P.PROD_STATS.note_retry("408")
    P.PROD_STATS.note_call(30.0)
    first = P.PROD_STATS.mark()
    P.PROD_STATS.note_retry("503")
    P.PROD_STATS.note_call(2.0)
    r = PQ.prod_call_receipt(5, first)
    assert r["retried_statuses"] == {"503": 1} and r["retries"] == 1 and r["longest_call_s"] == 2.0


def test_a_fail_closed_probe_does_not_wait_out_the_backoff(monkeypatch, _env):
    """CI on #718: the throwaway guard's listing probe got HTTP 500 and sat in the 155 s ladder."""
    from cytir.index import live_target as LT
    fake = FakeProd(fail=lambda n, a: 500)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with pytest.raises(LT.LiveTargetRefused):
        LT.read_holdings(TWIN)
    assert fake.calls == 1 and not [s for s in _env if s]
