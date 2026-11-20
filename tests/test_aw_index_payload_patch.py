"""Task 2748: the `payload_patch` pack kind. No Qdrant: a fake client holds the points in a dict.

Unit: the row state machine (four outcomes), conservation, the equal-count check, the inverse pack.
Drain: the pack travels `drain()` (twin first, then prod via the alias), with locks stubbed, and its
receipt is compared with the golden `fixtures/drain_transaction/payload_patch.golden.json`
(`AW_REGEN_GOLDEN=1` rewrites it, for a deliberate reviewed change only)."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import pathlib

import pytest

from cytir.index import change_log as CL
from cytir.index import payload_patch as PP
from cytir.index import point_delete as PD
from cytir.index import promote_queue as Q
from cytir.index import promote_sealed_pack as PSP

HERE = pathlib.Path(__file__).resolve().parent
GOLDEN = HERE / "fixtures" / "drain_transaction" / "payload_patch.golden.json"
U = {k: f"00000000-0000-4000-8000-00000000000{n}" for n, k in enumerate("abcde")}


class FakeClient:
    """Points in a dict. `vector_writes` counts any upsert or vector call (there is no such method:
    the leg's only write is `write`, a payload set)."""
    def __init__(self, points: dict[str, dict], *, count_drift: int = 0, lossy: bool = False):
        self.points = points
        self.writes: list[list[tuple[str, str]]] = []
        self.count_drift = count_drift
        self.lossy = lossy
        self.vector_writes = 0

    def upsert(self, *a, **k):
        self.vector_writes += 1

    def count(self, collection):
        return len(self.points) + (self.count_drift if self.writes else 0)

    def read(self, collection, ids, field):
        return {i: self.points[i].get(field) for i in ids if i in self.points}

    def write(self, collection, field, pairs):
        self.writes.append(list(pairs))
        if not self.lossy:
            for i, v in pairs:
                self.points[i][field] = v


def rows():
    # a: untouched (applied)  b: already new (already_applied)  c: moved by someone (precondition_failed)
    # d: absent (missing)
    return [{"id": U[k], "expect": f"old-{k}", "value": f"new-{k}"} for k in "abcd"]


def points(*, drop=()):
    pts = {U["a"]: {"snippet": "old-a"}, U["b"]: {"snippet": "new-b"}, U["c"]: {"snippet": "third"},
           U["e"]: {"snippet": "bystander"}}
    return {k: v for k, v in pts.items() if k not in drop}


def clean_rows():
    return [r for r in rows() if r["id"] in (U["a"], U["b"])]


def test_classification_is_read_only_and_has_four_outcomes():
    c = FakeClient(points())
    ids = PP.classify(c, "coll", rows(), "snippet")
    assert ids == {"applied": [U["a"]], "already_applied": [U["b"]], "precondition_failed": [U["c"]],
                   "missing": [U["d"]]}
    assert not c.writes


def test_a_moved_row_refuses_before_any_write():
    c = FakeClient(points())
    with pytest.raises(PP.PayloadPatchRefusal, match="moved since the producer"):
        PP.apply_leg("twin", c, "coll", rows(), "snippet", pause_s=0, expect_missing=1)
    assert not c.writes and c.points[U["c"]]["snippet"] == "third"


def test_missing_above_expect_missing_refuses_and_an_all_missing_pack_too():
    c = FakeClient(points())
    rs = clean_rows() + [{"id": U["d"], "expect": "o", "value": "n"}]
    with pytest.raises(PP.PayloadPatchRefusal, match="expect_missing is 0"):
        PP.apply_leg("twin", c, "coll", rs, "snippet", pause_s=0)
    PP.apply_leg("twin", FakeClient(points()), "coll", rs, "snippet", pause_s=0, expect_missing=1)
    allgone = FakeClient({U["e"]: {"snippet": "x"}})
    with pytest.raises(PP.PayloadPatchRefusal):
        PP.apply_leg("twin", allgone, "coll", rs, "snippet", pause_s=0)
    assert not allgone.writes


def test_clean_leg_conservation_and_equal_counts():
    c = FakeClient(points())
    r = PP.run_leg("twin", c, "coll", clean_rows(), "snippet", pause_s=0)
    assert (r["selected"], r["applied"], r["already_applied"], r["precondition_failed"], r["missing"]) == (2, 1, 1, 0, 0)
    assert PP.conserved(r) and r["points_before"] == r["points_after"] == 4
    assert c.points[U["a"]]["snippet"] == "new-a" and c.points[U["e"]]["snippet"] == "bystander"
    assert c.writes == [[(U["a"], "new-a")]] and c.vector_writes == 0
    assert r["verify_sample"] == 2


def test_prod_must_match_the_twin():
    twin = {"ids": {"applied": [U["a"]], "already_applied": [U["b"]]}}
    with pytest.raises(PP.PayloadPatchRefusal, match="on the twin"):
        PP.apply_leg("prod", FakeClient(points(drop=(U["b"],))), "coll", clean_rows(), "snippet",
                     pause_s=0, expect_missing=1, reference=twin)


def test_rerun_is_idempotent():
    c = FakeClient(points())
    PP.run_leg("twin", c, "coll", clean_rows(), "snippet", pause_s=0)
    again = PP.run_leg("twin", c, "coll", clean_rows(), "snippet", pause_s=0)
    assert again["applied"] == 0 and again["already_applied"] == 2 and len(c.writes) == 1


def test_batches_and_pauses():
    ids = [f"00000000-0000-4000-8000-{i:012d}" for i in range(2500)]
    pts = {i: {"snippet": "o"} for i in ids}
    rs = [{"id": i, "expect": "o", "value": "n"} for i in ids]
    slept: list[float] = []
    c = FakeClient(pts)
    r = PP.run_leg("prod", c, "x", rs, "snippet", batch=1000, pause_s=0.2, read_pause_s=0.5, sleep=slept.append)
    assert [len(w) for w in c.writes] == [1000, 1000, 500]
    assert slept.count(0.2) == 3 and slept.count(0.5) >= 3          # writes paced; retrieves paced too
    assert r["applied"] == 2500 and r["write_calls"] == 3 and r["verify_sample"] >= 200
    with pytest.raises(PP.PayloadPatchError):
        PP.classify(c, "x", rs, "snippet", batch=1001)


def test_count_moving_marks_the_leg_failed():
    with pytest.raises(PP.PayloadPatchError, match="points_count moved"):
        PP.run_leg("prod", FakeClient(points(), count_drift=1), "coll", clean_rows(), "snippet", pause_s=0)


def test_a_write_that_does_not_read_back_fails():
    with pytest.raises(PP.PayloadPatchError, match="do not read the new snippet"):
        PP.run_leg("twin", FakeClient(points(), lossy=True), "coll", clean_rows(), "snippet", pause_s=0)


def test_field_allow_list():
    with pytest.raises(PP.PayloadPatchError, match="not patchable"):
        PP.apply_leg("twin", FakeClient(points()), "coll", rows(), "url")


def test_client_request_bodies(monkeypatch):
    calls = []
    monkeypatch.setattr(PSP, "_api", lambda url, path, body=None, method=None: calls.append((path, body)) or (
        {"points_count": 7} if body is None else []))
    c = PP.QdrantPatchClient("http://q")
    assert c.count("coll") == 7
    c.write("coll", "snippet", [(U["a"], "new-a"), (U["b"], "new-b")])
    path, body = calls[-1]
    assert path == "/collections/coll/points/batch?wait=true"
    assert body == {"operations": [{"set_payload": {"payload": {"snippet": "new-a"}, "points": [U["a"]]}},
                                   {"set_payload": {"payload": {"snippet": "new-b"}, "points": [U["b"]]}}]}
    assert "vector" not in json.dumps(body) and "upsert" not in json.dumps(body)
    c.read("coll", [U["a"]], "snippet")
    path, body = calls[-1]
    assert path == "/collections/coll/points" and body["with_vector"] is False and body["with_payload"] == ["snippet"]


def _http_error(code):
    import urllib.error
    return urllib.error.HTTPError("http://q/x", code, "boom", {}, io.BytesIO(b"detail"))


def _flaky_fetch(monkeypatch, outcomes):
    """IH.fetch_json answers from `outcomes` (an exception is raised, a dict returned); sleeps are recorded."""
    calls, sleeps = [], []
    seq = list(outcomes)

    def fake(url, body=None, method=None, timeout=None):
        calls.append(url)
        out = seq.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out
    monkeypatch.setattr(PSP.IH, "fetch_json", fake)
    monkeypatch.setattr(PSP.time, "sleep", sleeps.append)
    return calls, sleeps


def test_client_retries_a_transient_503_then_succeeds(monkeypatch):
    calls, sleeps = _flaky_fetch(monkeypatch, [_http_error(503), TimeoutError(), {"status": "ok", "result": []}])
    c = PP.QdrantPatchClient("http://q")
    c.write("coll", "snippet", [(U["a"], "new-a")])
    assert len(calls) == 3 and sleeps == [PSP._API_STATUS_BACKOFF_S[0], PSP._API_TIMEOUT_BACKOFF_S[0]]


def test_client_ladder_is_bounded_and_a_4xx_is_never_retried(monkeypatch):
    calls, sleeps = _flaky_fetch(monkeypatch, [_http_error(503)] * (PSP._API_STATUS_RETRIES + 1))
    with pytest.raises(PSP.PromotionError, match="503"):
        PP.QdrantPatchClient("http://q").count("coll")
    assert len(calls) == PSP._API_STATUS_RETRIES + 1
    calls, sleeps = _flaky_fetch(monkeypatch, [_http_error(400), {"status": "ok", "result": {}}])
    with pytest.raises(PSP.PromotionError, match="400"):
        PP.QdrantPatchClient("http://q").read("coll", [U["a"]], "snippet")
    assert len(calls) == 1 and sleeps == []


def test_before_write_runs_after_classification_and_can_stop_the_write():
    c = FakeClient(points())
    order = []
    PP.apply_leg("prod", c, "coll", clean_rows(), "snippet", pause_s=0, before_write=lambda: order.append(len(c.writes)))
    assert order == [0] and len(c.writes) == 1
    c2 = FakeClient(points())

    def stop():
        raise RuntimeError("alias moved")
    with pytest.raises(RuntimeError):
        PP.apply_leg("prod", c2, "coll", clean_rows(), "snippet", pause_s=0, before_write=stop)
    assert c2.writes == []
    c3 = FakeClient(points())          # nothing to write: the hook is not called
    PP.apply_leg("prod", c3, "coll", [r for r in clean_rows() if r["id"] == U["b"]], "snippet", pause_s=0,
                 before_write=stop)


def test_both_writers_share_one_set_payload_primitive(monkeypatch):
    from cytir.ops import twin_patch_fields as TPF
    seen = []
    monkeypatch.setattr(PD, "set_payload_batch", lambda url, coll, ops, api=None: seen.append((coll, ops, api)))
    ops = [{"payload": {"snippet": "x"}, "points": [U["a"]]}]
    TPF.write("http://q", "coll", "multi", ops, [])
    PP.QdrantPatchClient("http://q").write("coll", "snippet", [(U["a"], "x")])
    assert [s[:2] for s in seen] == [("coll", ops)] * 2 and seen[0][2] is None and seen[1][2] is PSP._api


def _pack(tmp, rs=None, name="snip-pack-001", **over):
    rs = rows() if rs is None else rs
    body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rs).encode()
    d = pathlib.Path(tmp) / name
    d.mkdir()
    (d / "rows.jsonl").write_bytes(body)
    m = {"schema": Q.MANIFEST_SCHEMA, "kind": PP.KIND, "payload_patch_version": 1, "pack_id": name,
         "family": "snippet_fix", "collection": "aw_e8", "field": "snippet", "rows_file": "rows.jsonl",
         "count": len(rs), "sha256": hashlib.sha256(body).hexdigest(), **over}
    mb = (json.dumps(m, indent=1) + "\n").encode()
    (d / "manifest.json").write_bytes(mb)
    (d / "READY").write_text(hashlib.sha256(mb).hexdigest() + "\n")
    return d


def test_manifest_checks(tmp_path):
    d = _pack(tmp_path)
    m = Q.read_manifest(d)
    assert len(PP.load_rows(m, d)) == 4
    for bad, msg in (({"field": "url"}, "not patchable"), ({"count": 0}, "positive"),
                     ({"payload_patch_version": 2}, "payload_patch_version"), ({"sha256": None}, "missing"),
                     ({"expect_missing": -1}, "expect_missing")):
        with pytest.raises(PP.PayloadPatchError, match=msg):
            PP.check_manifest({**m, **bad})
    with pytest.raises(PP.PayloadPatchError, match="sha256"):
        PP.load_rows({**m, "sha256": "0" * 64}, d)
    with pytest.raises(PP.PayloadPatchError, match="manifest count"):
        PP.load_rows({**m, "count": 9}, d)


def test_ids_must_be_canonical_uuids(tmp_path):
    LOW = "abcdefab-0000-4000-8000-00000000000a"
    for bad in ("a", LOW.upper(), LOW.replace("-", "")):
        d = tmp_path / bad[:6]
        d.mkdir()
        rs = [{"id": bad, "expect": "o", "value": "n"}]
        body = (json.dumps(rs[0]) + "\n").encode()
        (d / "rows.jsonl").write_bytes(body)
        m = {"kind": PP.KIND, "payload_patch_version": 1, "pack_id": "p", "family": "f", "collection": "c",
             "field": "snippet", "rows_file": "rows.jsonl", "count": 1, "sha256": hashlib.sha256(body).hexdigest()}
        with pytest.raises(PP.PayloadPatchError, match="canonical lowercase UUID"):
            PP.load_rows(m, d)


def test_gz_rows_and_duplicate_ids(tmp_path):
    d = _pack(tmp_path)
    m = json.loads((d / "manifest.json").read_text())
    raw = (d / "rows.jsonl").read_bytes()
    (d / "rows.jsonl.gz").write_bytes(gzip.compress(raw))
    assert len(PP.load_rows({**m, "rows_file": "rows.jsonl.gz"}, d)) == 4   # sha of the DECOMPRESSED file
    dup = raw + raw.splitlines(True)[0]
    (d / "dup.jsonl").write_bytes(dup)
    with pytest.raises(PP.PayloadPatchError, match="duplicate"):
        PP.load_rows({**m, "rows_file": "dup.jsonl", "sha256": hashlib.sha256(dup).hexdigest(), "count": 5}, d)


def test_inverse_pack_swaps_and_round_trips(tmp_path):
    d = _pack(tmp_path)
    m = json.loads((d / "manifest.json").read_text())
    inv = PP.write_inverse_pack(tmp_path / "art", m, clean_rows())
    idir = pathlib.Path(inv["dir"])
    im = json.loads((idir / "manifest.json").read_text())
    assert im["inverse_of"] == "snip-pack-001" and not (idir / "READY").exists()
    irows = PP.load_rows(im, idir)
    assert [(r["id"], r["expect"], r["value"]) for r in irows][0] == (U["a"], "new-a", "old-a")
    c = FakeClient(points())
    PP.run_leg("twin", c, "coll", clean_rows(), "snippet", pause_s=0)
    back = PP.run_leg("twin", c, "coll", irows, "snippet", pause_s=0)
    assert c.points[U["a"]]["snippet"] == "old-a" and back["applied"] == 2   # a, and b (already new on entry)


# ------------------------------------------------------------------ through drain()
class _Lock:
    def __init__(self, *a, **k): pass
    def acquire(self): pass
    def release(self): pass


class _RunB:
    LOCKS = ("l1",)
    def __init__(self): self.log = []
    def assert_locks_free(self, task): pass
    def acquire(self, name, task, why, pid=None): self.log.append(("acquire", name))
    def release(self, name, task): self.log.append(("release", name))


def _make(monkeypatch, tmp_path, twin_pts, prod_pts, rs, **manifest_over):
    monkeypatch.setenv("AW_PROD_LOCK_HOST", "prod-lock-host")
    monkeypatch.setattr(PSP, "_api", lambda url, path, body=None, method=None: (
        {"aliases": [{"alias_name": "aw_prod", "collection_name": "aw_e9"}]} if path == "/aliases"
        else (_ for _ in ()).throw(AssertionError(path))))
    runb = _RunB()
    monkeypatch.setattr(Q, "_load_run_batch", lambda: runb)
    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", _Lock)
    monkeypatch.setattr(Q.PG, "PROD_BATCH_SLEEP_S", 0)
    twin, prod = FakeClient(twin_pts), FakeClient(prod_pts)
    order: list[str] = []
    real = PP.apply_leg

    def spy(role, client, *a, **k):
        order.append(role)
        return real(role, client, *a, **k)
    monkeypatch.setattr(PP, "apply_leg", spy)
    real_live = Q.live_apply_payload_patch
    monkeypatch.setattr(Q, "live_apply_payload_patch",
                        lambda **k: real_live(**k, twin_client=twin, prod_client=prod))
    qd = tmp_path / "queue"
    qd.mkdir()
    pack = _pack(qd, rs, **manifest_over)
    contract = tmp_path / "gate.json"
    contract.write_text(json.dumps({"schema": Q.GATE_CONTRACT_SCHEMA, "adr": "ADR-110", "pr": 302,
                                    "profile": "mechanical"}))
    return dict(qd=qd, pack=pack, twin=twin, prod=prod, order=order, runb=runb, contract=contract,
                art=tmp_path / "art")


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """The 2734 shape: the twin already holds the new values, prod holds the old ones."""
    new = {U["a"]: {"snippet": "new-a"}, U["b"]: {"snippet": "new-b"}, U["e"]: {"snippet": "bystander"}}
    old = {U["a"]: {"snippet": "old-a"}, U["b"]: {"snippet": "old-b"}, U["e"]: {"snippet": "bystander"}}
    h = _make(monkeypatch, tmp_path, new, old, clean_rows())
    return h


def _drain(h, apply=True, **kw):
    return Q.drain(h["qd"], twin_url="http://twin", prod_url="http://prod", task="2748-test", apply=apply,
                   batch=1000, pause_s=0, artifact_root=h["art"], contract_path=h["contract"],
                   refresh_index=False, **kw)


def test_drain_lands_twin_first_then_prod_with_receipts(harness):
    (res,) = _drain(harness)
    assert res["status"] == "LANDED" and harness["order"] == ["twin", "prod"]
    assert (harness["pack"] / "DONE").exists()
    assert (res["twin"]["applied"], res["twin"]["already_applied"]) == (0, 2)        # 2734: the twin is patched
    assert (res["prod"]["applied"], res["prod"]["already_applied"]) == (2, 0)
    for leg in ("twin", "prod"):
        r = res[leg]
        assert PP.conserved(r) and r["points_before"] == r["points_after"] and "ids" not in r
        assert json.loads(pathlib.Path(r["ids_artifact"]).read_text())["applied"] is not None
    assert harness["prod"].points[U["b"]]["snippet"] == "new-b"
    assert "needs_attention" not in res
    assert pathlib.Path(res["inverse_pack"]["dir"], "rows.jsonl").exists()
    assert ("acquire", "l1") in harness["runb"].log and ("release", "l1") in harness["runb"].log
    assert harness["twin"].vector_writes == harness["prod"].vector_writes == 0
    assert _drain(harness) == []                      # DONE: the pack is not drained twice


def test_landed_pack_has_a_change_log_segment_and_the_audit_is_clean(harness, tmp_path, monkeypatch):
    logs = tmp_path / "changelog"
    monkeypatch.setenv("AW_CHANGE_LOG_DIR", str(logs))
    (res,) = _drain(harness)
    seg = CL.segment_for(logs, "snip-pack-001")
    assert seg is not None
    header, lines = CL.read_segment(seg)
    assert {(x["id"], x["op"]) for x in lines} == {(U["a"], "replace"), (U["b"], "replace")}
    rep = CL.audit(harness["qd"], logs, since="2000-01-01T00:00:00Z")
    assert rep["done_checked"] == 1 and rep["done_without_segment"] == [] and rep["segment_without_done"] == []


def test_dry_run_writes_nothing(harness):
    (res,) = _drain(harness, apply=False)
    assert res["status"] == "PROVED-DRY-RUN" and harness["order"] == []
    assert not harness["twin"].writes and not (harness["pack"] / "DONE").exists()


def test_bad_sha_is_refused_before_any_write(harness):
    (harness["pack"] / "rows.jsonl").write_bytes(b'{"id":"a","expect":"x","value":"y"}\n')
    (res,) = _drain(harness)
    assert res["status"] == "REFUSED" and res["stage"] == "payload_patch_check"
    assert not harness["twin"].writes and not harness["prod"].writes


def _aborted(h):
    with pytest.raises(PP.PayloadPatchError):
        _drain(h)
    rec = json.loads((h["pack"] / "receipt.json").read_text())
    assert rec["status"] == "ABORTED" and not (h["pack"] / "DONE").exists()
    return rec


def test_twin_precondition_failed_stops_the_drain_before_any_write(tmp_path, monkeypatch):
    h = _make(monkeypatch, tmp_path, points(), points(), rows(), expect_missing=1)
    rec = _aborted(h)
    assert rec["writes_issued"] is False and not h["twin"].writes and not h["prod"].writes
    assert h["order"] == ["twin"]


def test_all_missing_pack_is_refused_on_the_twin_preread(tmp_path, monkeypatch):
    h = _make(monkeypatch, tmp_path, {U["e"]: {"snippet": "x"}}, {U["e"]: {"snippet": "x"}}, clean_rows())
    rec = _aborted(h)
    assert "expect_missing is 0" in rec["reason"] and not h["twin"].writes


def test_prod_mismatch_aborts_and_stops_the_drain(tmp_path, monkeypatch):
    new = {U["a"]: {"snippet": "new-a"}, U["b"]: {"snippet": "new-b"}}
    prod = {U["a"]: {"snippet": "old-a"}, U["b"]: {"snippet": "elsewhere"}}     # b moved on prod
    h = _make(monkeypatch, tmp_path, new, prod, clean_rows())
    rec = _aborted(h)
    assert h["order"] == ["twin", "prod"] and not h["prod"].writes and "moved since" in rec["reason"]


def _flip_after_prod_read(h, monkeypatch):
    target = {"aw_prod": "aw_e9"}
    monkeypatch.setattr(PSP, "_api", lambda url, path, body=None, method=None: {"aliases": [
        {"alias_name": "aw_prod", "collection_name": target["aw_prod"]}]})
    real_read = h["prod"].read

    def read_then_flip(collection, ids, field):      # the prod classification has read; now the alias moves
        out = real_read(collection, ids, field)
        target["aw_prod"] = "aw_e11"
        return out
    h["prod"].read = read_then_flip


def test_alias_flip_with_nothing_written_is_refused_and_the_drain_stops(tmp_path, monkeypatch):
    # The twin already holds the new values (a re-run): its leg writes nothing, so the flip is REFUSED.
    new = {U["a"]: {"snippet": "new-a"}, U["b"]: {"snippet": "new-b"}}
    old = {U["a"]: {"snippet": "old-a"}, U["b"]: {"snippet": "old-b"}}
    h = _make(monkeypatch, tmp_path, new, old, clean_rows())
    _flip_after_prod_read(h, monkeypatch)
    with pytest.raises(Q.TFA.ApplyAbort):             # task 2786: the drain stops, REFUSED stays the status
        _drain(h)
    rec = json.loads((h["pack"] / "receipt.json").read_text())
    assert rec["status"] == "REFUSED" and not (h["pack"] / "DONE").exists()
    assert h["order"] == ["twin", "prod"] and not h["twin"].writes and not h["prod"].writes
    assert "moved mid-drain" in json.dumps(rec)


def test_alias_flip_after_the_twin_wrote_is_aborted_and_the_drain_stops(tmp_path, monkeypatch):
    old = {U["a"]: {"snippet": "old-a"}, U["b"]: {"snippet": "old-b"}}
    h = _make(monkeypatch, tmp_path, dict(old), dict(old), clean_rows())     # the twin writes first
    _flip_after_prod_read(h, monkeypatch)
    with pytest.raises(Q.TFA.ApplyAbort):
        _drain(h)
    rec = json.loads((h["pack"] / "receipt.json").read_text())
    assert rec["status"] == "ABORTED" and rec["writes_issued"] is True and not (h["pack"] / "DONE").exists()
    assert h["twin"].writes and not h["prod"].writes and "moved mid-drain" in json.dumps(rec)


def test_keyboard_interrupt_with_nothing_written_propagates_and_writes_no_receipt(tmp_path, monkeypatch):
    # task 2800: an interrupt is not a REFUSED verdict. No write issued -> no receipt, pack stays READY.
    new = {U["a"]: {"snippet": "new-a"}, U["b"]: {"snippet": "new-b"}}
    old = {U["a"]: {"snippet": "old-a"}, U["b"]: {"snippet": "old-b"}}
    h = _make(monkeypatch, tmp_path, new, old, clean_rows())
    real_read = h["prod"].read

    def read_then_interrupt(collection, ids, field):
        real_read(collection, ids, field)
        raise KeyboardInterrupt
    h["prod"].read = read_then_interrupt
    with pytest.raises(KeyboardInterrupt):
        _drain(h)
    assert not (h["pack"] / "receipt.json").exists() and not (h["pack"] / "DONE").exists()
    assert (h["pack"] / "READY").exists() and not h["twin"].writes and not h["prod"].writes


def test_keyboard_interrupt_after_the_twin_wrote_writes_aborted_and_propagates(tmp_path, monkeypatch):
    old = {U["a"]: {"snippet": "old-a"}, U["b"]: {"snippet": "old-b"}}
    h = _make(monkeypatch, tmp_path, dict(old), dict(old), clean_rows())
    real_read = h["prod"].read

    def read_then_interrupt(collection, ids, field):
        real_read(collection, ids, field)
        raise KeyboardInterrupt
    h["prod"].read = read_then_interrupt
    with pytest.raises(KeyboardInterrupt):
        _drain(h)
    rec = json.loads((h["pack"] / "receipt.json").read_text())
    assert rec["status"] == "ABORTED" and rec["writes_issued"] is True and not (h["pack"] / "DONE").exists()


def _receipt_write_fails(monkeypatch, statuses):
    real = Q.write_pack_receipt

    def failing(pack_dir, record):
        if record.get("status") in statuses:
            raise OSError("disk full")
        real(pack_dir, record)
    monkeypatch.setattr(Q, "write_pack_receipt", failing)


def test_receipt_write_failure_in_the_aborted_branch_keeps_the_aborted_reason(tmp_path, monkeypatch, capsys):
    # task 2817: a write started (twin wrote), then the alias flips. The failing receipt write must not
    # replace the ApplyAbort the drain stops on, and the original reason is logged.
    old = {U["a"]: {"snippet": "old-a"}, U["b"]: {"snippet": "old-b"}}
    h = _make(monkeypatch, tmp_path, dict(old), dict(old), clean_rows())
    _flip_after_prod_read(h, monkeypatch)
    _receipt_write_fails(monkeypatch, {"ABORTED"})
    with pytest.raises(Q.TFA.ApplyAbort, match="moved mid-drain"):
        _drain(h)
    err = capsys.readouterr().err
    assert "ABORTED receipt write failed" in err and "moved mid-drain" in err
    assert h["twin"].writes and not (h["pack"] / "DONE").exists() and (h["pack"] / "READY").exists()


def test_receipt_write_failure_in_the_refused_branch_keeps_the_refused_reason(tmp_path, monkeypatch, capsys):
    new = {U["a"]: {"snippet": "new-a"}, U["b"]: {"snippet": "new-b"}}
    old = {U["a"]: {"snippet": "old-a"}, U["b"]: {"snippet": "old-b"}}
    h = _make(monkeypatch, tmp_path, new, old, clean_rows())
    _flip_after_prod_read(h, monkeypatch)
    _receipt_write_fails(monkeypatch, {"REFUSED"})
    with pytest.raises(Q.TFA.ApplyAbort, match="moved mid-drain"):
        _drain(h)
    err = capsys.readouterr().err
    assert "REFUSED receipt write failed" in err and "moved mid-drain" in err
    assert not (h["pack"] / "receipt.json").exists() and not (h["twin"].writes or h["prod"].writes)


def test_receipt_write_failure_on_a_classification_refusal_keeps_the_pack_error(tmp_path, monkeypatch, capsys):
    new = {U["a"]: {"snippet": "new-a"}, U["b"]: {"snippet": "new-b"}}
    prod = {U["a"]: {"snippet": "old-a"}, U["b"]: {"snippet": "elsewhere"}}
    h = _make(monkeypatch, tmp_path, new, prod, clean_rows())
    _receipt_write_fails(monkeypatch, {"ABORTED"})
    with pytest.raises(PP.PayloadPatchError, match="moved since"):
        _drain(h)
    assert "ABORTED receipt write failed" in capsys.readouterr().err
    assert not h["prod"].writes and not (h["pack"] / "DONE").exists() and (h["pack"] / "READY").exists()


def test_receipt_write_failure_on_a_plain_preflight_refusal_stops_the_drain(tmp_path, monkeypatch, capsys):
    # A normal error with no write is a plain-return verdict: nothing is in flight for the receipt error
    # to mask, so a failing receipt write must stop the drain (pack 2 must not run).
    new = {U["a"]: {"snippet": "new-a"}, U["b"]: {"snippet": "new-b"}}
    old = {U["a"]: {"snippet": "old-a"}, U["b"]: {"snippet": "old-b"}}
    h = _make(monkeypatch, tmp_path, new, old, clean_rows())
    p2 = _pack(h["qd"], clean_rows(), name="snip-pack-002")
    real_read = h["prod"].read

    def read_then_fail(collection, ids, field):
        real_read(collection, ids, field)
        raise RuntimeError("transient api error")
    h["prod"].read = read_then_fail
    _receipt_write_fails(monkeypatch, {"REFUSED"})
    with pytest.raises(RuntimeError, match="receipt write failed.*transient api error"):
        _drain(h)
    assert "REFUSED receipt write failed" in capsys.readouterr().err
    assert h["order"] == ["twin", "prod"] and not h["twin"].writes and not h["prod"].writes
    assert (p2 / "READY").exists() and not (p2 / "receipt.json").exists()


def test_receipt_write_failure_in_the_interrupt_branch_still_propagates_the_interrupt(tmp_path, monkeypatch, capsys):
    old = {U["a"]: {"snippet": "old-a"}, U["b"]: {"snippet": "old-b"}}
    h = _make(monkeypatch, tmp_path, dict(old), dict(old), clean_rows())
    real_read = h["prod"].read

    def read_then_interrupt(collection, ids, field):
        real_read(collection, ids, field)
        raise KeyboardInterrupt
    h["prod"].read = read_then_interrupt
    attempted = []
    real = Q.write_pack_receipt

    def failing(pack_dir, record):
        attempted.append(record.get("status"))
        raise OSError("disk full")
    monkeypatch.setattr(Q, "write_pack_receipt", failing)
    with pytest.raises(KeyboardInterrupt):
        _drain(h)
    assert attempted == ["ABORTED"] and "ABORTED receipt write failed" in capsys.readouterr().err
    assert h["twin"].writes and not (h["pack"] / "DONE").exists()


def test_a_flip_on_pack_one_leaves_pack_two_untouched(tmp_path, monkeypatch):
    new = {U["a"]: {"snippet": "new-a"}, U["b"]: {"snippet": "new-b"}}
    old = {U["a"]: {"snippet": "old-a"}, U["b"]: {"snippet": "old-b"}}
    h = _make(monkeypatch, tmp_path, new, old, clean_rows())
    p2 = _pack(h["qd"], clean_rows(), name="snip-pack-002")
    _flip_after_prod_read(h, monkeypatch)
    with pytest.raises(Q.TFA.ApplyAbort):
        _drain(h)
    assert h["order"] == ["twin", "prod"]              # pack 2 never reached a leg
    assert (p2 / "READY").exists() and not (p2 / "receipt.json").exists() and not (p2 / "DONE").exists()
    assert not h["twin"].writes


def test_prod_applied_plus_already_must_equal_the_twins(tmp_path, monkeypatch):
    new = {U["a"]: {"snippet": "new-a"}, U["b"]: {"snippet": "new-b"}}
    h = _make(monkeypatch, tmp_path, new, {U["a"]: {"snippet": "old-a"}}, clean_rows(), expect_missing=1)
    rec = _aborted(h)
    assert "on the twin" in rec["reason"] and not h["prod"].writes


def test_read_back_failure_aborts_through_drain(tmp_path, monkeypatch):
    new = {U["a"]: {"snippet": "new-a"}, U["b"]: {"snippet": "new-b"}}
    old = {U["a"]: {"snippet": "old-a"}, U["b"]: {"snippet": "old-b"}}
    h = _make(monkeypatch, tmp_path, new, old, clean_rows())
    h["prod"].lossy = True
    rec = _aborted(h)
    assert rec["writes_issued"] is True and "do not read the new snippet" in rec["reason"]


def test_a_single_transaction_refuses_this_kind(harness, tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "assert_single_target_not_live", lambda *a, **k: None)
    from cytir.index import promote_queue as PQ
    single = PQ.Single(url="http://single", collection="c")
    res = Q.drain(harness["qd"], task="2748-single", apply=True, batch=3, pause_s=0, artifact_root=harness["art"],
                  contract_path=harness["contract"], refresh_index=False, transaction=single)
    assert res[0]["status"] == "REFUSED" and res[0]["stage"] == "single-transaction"
    assert "payload_patch" in res[0]["reason"] and not harness["twin"].writes


def _normal(res: dict) -> dict:
    out = json.loads(json.dumps(res, sort_keys=True))
    for k in ("checked_at", "manifest_sha", "gate"):
        out.pop(k, None)
    out["inverse_pack"] = {k: v for k, v in out["inverse_pack"].items() if k != "dir"}
    for leg in ("twin", "prod"):
        out[leg]["ids_artifact"] = pathlib.Path(out[leg]["ids_artifact"]).name
    return out


def test_receipt_matches_the_golden(harness):
    got = _normal(_drain(harness)[0])
    if os.environ.get("AW_REGEN_GOLDEN") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
        pytest.skip("golden regenerated")
    assert got == json.loads(GOLDEN.read_text())



# ------------------------------------------------------------------ task 2922: pins of the write path
# (drain split step 12). Each pin runs the real `drain` / the real function and asserts one protocol step.
REAL_LIVE = Q.live_apply_payload_patch
PACK = "snip-pack-001"


def _new_old():
    new = {U["a"]: {"snippet": "new-a"}, U["b"]: {"snippet": "new-b"}, U["e"]: {"snippet": "bystander"}}
    old = {U["a"]: {"snippet": "old-a"}, U["b"]: {"snippet": "old-b"}, U["e"]: {"snippet": "bystander"}}
    return new, old


def _instrument(h, events):
    """Record every client call (role, kind, collection) into `events`."""
    for role in ("twin", "prod"):
        c = h[role]
        for kind in ("count", "read", "write"):
            real = getattr(c, kind)

            def wrapped(collection, *a, _real=real, _role=role, _kind=kind, **k):
                events.append((_role, _kind, collection))
                return _real(collection, *a, **k)
            setattr(c, kind, wrapped)


def _world(monkeypatch, tmp_path, twin=None, prod=None, rs=None, **over):
    new, old = _new_old()
    h = _make(monkeypatch, tmp_path, twin if twin is not None else new, prod if prod is not None else old,
              rs if rs is not None else clean_rows(), **over)
    h["events"] = []
    _instrument(h, h["events"])
    return h


def _go(h, **kw):
    args = dict(twin_url="http://twin", prod_url="http://prod", task="2748-test", apply=True, batch=1000,
                pause_s=0, artifact_root=h["art"], contract_path=h["contract"], refresh_index=False)
    args.update(kw)
    return Q.drain(h["qd"], **args)


def test_the_locks_are_built_and_taken_for_this_pack_before_the_first_write_and_released_in_reverse(
        tmp_path, monkeypatch):
    from cytir.index.drain.locks import PROD_LOCK_PATH
    h = _world(monkeypatch, tmp_path)
    ev, built = h["events"], []
    runb = h["runb"]
    runb.LOCKS = ("l1", "l2")
    runb.assert_locks_free = lambda task: ev.append(("lock", "free", task))
    runb.acquire = lambda name, task, why, pid=None: ev.append(("lock", "acquire", name, task, why, pid))
    runb.release = lambda name, task: ev.append(("lock", "release", name, task))

    class Spy(_Lock):
        def __init__(self, **k): built.append(k)
    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", Spy)
    assert _go(h)[0]["status"] == "LANDED"
    assert built == [{"host": "prod-lock-host", "lock_path": PROD_LOCK_PATH, "purpose": f"task2748-test-{PACK}"}]
    locks = [e for e in ev if e[0] == "lock"]
    why = f"{PACK}-promotion-queue"
    assert locks == [("lock", "free", "2748-test"),
                     ("lock", "acquire", "l1", "2748-test", why, os.getpid()),
                     ("lock", "acquire", "l2", "2748-test", why, os.getpid()),
                     ("lock", "release", "l2", "2748-test"), ("lock", "release", "l1", "2748-test")]
    writes = [i for i, e in enumerate(ev) if e[1] == "write"]
    last_acquire = max(i for i, e in enumerate(ev) if e[:2] == ("lock", "acquire"))
    assert last_acquire < min(writes) <= max(writes) < ev.index(("lock", "release", "l2", "2748-test"))


def test_the_locks_are_released_when_a_leg_aborts(tmp_path, monkeypatch):
    h = _world(monkeypatch, tmp_path, twin=points(), prod=points(), rs=rows(), expect_missing=1)
    runb = h["runb"]
    runb.LOCKS = ("l1", "l2")
    with pytest.raises(Exception):
        _go(h)
    assert runb.log == [("acquire", "l1"), ("acquire", "l2"), ("release", "l2"), ("release", "l1")]


def test_the_legs_get_the_write_pause_the_read_pause_the_batch_cap_and_run_inside_a_retry_leg(
        tmp_path, monkeypatch):
    h = _world(monkeypatch, tmp_path)
    asked, legs, verifies = [], [], []
    monkeypatch.setattr(Q.PG, "write_pause", lambda role, nrows, p: asked.append((role, nrows, p)) or 0.5)
    monkeypatch.setattr(Q.PG, "PROD_BATCH_SLEEP_S", 3.0)
    real_apply, real_verify = PP.apply_leg, PP.verify_leg

    def apply_spy(role, client, coll, rws, field, **k):
        legs.append((role, k["batch"], k["pause_s"], k["read_pause_s"], PSP._LEG["depth"]))
        return real_apply(role, client, coll, rws, field, **k)

    def verify_spy(client, coll, rws, receipt, **k):
        verifies.append((receipt["role"], k["read_pause_s"], PSP._LEG["depth"]))
        return real_verify(client, coll, rws, receipt, **k)
    monkeypatch.setattr(PP, "apply_leg", apply_spy)
    monkeypatch.setattr(PP, "verify_leg", verify_spy)
    assert _go(h, batch=5000, pause_s=0.25)[0]["status"] == "LANDED"
    assert asked == [("prod", 2, 0.25)] * 2        # the WRITE PAUSE rule: prod only (apply and verify read it)
    assert legs == [("twin", 1000, 0.25, 0.0, 1), ("prod", 1000, 0.5, 3.0, 1)]
    assert verifies == [("twin", 0.0, 0), ("prod", 3.0, 0)]
    assert PSP._LEG["depth"] == 0


def test_the_prod_alias_is_read_before_the_classification_before_the_first_write_and_after_the_verify(
        tmp_path, monkeypatch):
    h = _world(monkeypatch, tmp_path)
    ev = h["events"]
    real_api = PSP._api

    def api(url, path, body=None, method=None):
        if url == "http://prod":
            ev.append(("alias", path))
        return real_api(url, path, body, method)
    monkeypatch.setattr(PSP, "_api", api)
    assert _go(h)[0]["status"] == "LANDED"
    seq = [(e[0], e[1]) for e in ev if e[0] != "twin"]
    al = ("alias", "/aliases")
    # the drain's own read at the start; then: before the classification, classify (count, read), once more
    # right before the first write, the write, the verify (read, count), and last, the alias after the verify
    assert seq == [al, al, ("prod", "count"), ("prod", "read"), al, ("prod", "write"), ("prod", "read"),
                   ("prod", "count"), al]


def test_each_client_reads_and_writes_only_its_own_collection(tmp_path, monkeypatch):
    h = _world(monkeypatch, tmp_path)
    assert _go(h)[0]["status"] == "LANDED"
    by = {r: {e[2] for e in h["events"] if e[0] == r} for r in ("twin", "prod")}
    assert by == {"twin": {"aw_e8"}, "prod": {"aw_e9"}}


def test_the_clients_are_built_for_their_own_url_and_the_state_is_the_two_receipts(tmp_path, monkeypatch):
    h = _world(monkeypatch, tmp_path)
    built, sleeps, seen = [], [], {}
    mine = {"http://twin": h["twin"], "http://prod": h["prod"]}
    monkeypatch.setattr(PP, "QdrantPatchClient", lambda url: built.append(url) or mine[url])
    real_apply, real_verify = PP.apply_leg, PP.verify_leg

    def apply_spy(role, c, *a, **k):
        seen[role] = c
        sleeps.append(k["sleep"])
        return real_apply(role, c, *a, **k)

    def verify_spy(c, co, rs, rc, **k):
        sleeps.append(k["sleep"])
        return real_verify(c, co, rs, rc, **k)
    monkeypatch.setattr(PP, "apply_leg", apply_spy)
    monkeypatch.setattr(PP, "verify_leg", verify_spy)
    manifest = json.loads((h["pack"] / "manifest.json").read_text())

    def tick(s):
        return None
    state = REAL_LIVE(pack_id=PACK, manifest=manifest, pack_dir=h["pack"], twin_url="http://twin",
                      prod_url="http://prod", task="t", batch=7, pause_s=0, artifact_root=h["art"],
                      prod_collection="aw_e9", sleep=tick)
    assert built == ["http://twin", "http://prod"]
    assert seen == {"twin": h["twin"], "prod": h["prod"]}
    assert sleeps == [tick] * 4                                   # both legs' apply and verify get the sleep
    assert sorted(state) == ["inverse_pack", "prod_receipt", "rows", "twin_receipt"] and state["rows"] == 2
    assert h["prod"].points[U["a"]]["snippet"] == "new-a" and h["twin"].vector_writes == 0


def test_the_ledger_records_the_proofs_in_protocol_order(tmp_path, monkeypatch):
    h = _world(monkeypatch, tmp_path)
    assert _go(h)[0]["status"] == "LANDED"
    lines = [json.loads(x) for x in (h["art"] / f"{PACK}-ledger.jsonl").read_text().splitlines()]
    got = [{k: v for k, v in x.items() if k != "ts"} for x in lines]
    leg = {"missing": 0, "points_delta": 0, "precondition_failed": 0, "selected": 2, "status": "PASS",
           "verify_sample": 2}
    assert got == [
        {"armed": True, "status": "PREFLIGHT-OK"},
        {"status": "PRESERVATION-VERIFIED", "proof": {
            "pack_id": PACK, "status": "PASS",
            "scope": "narrow: payload-only set_payload of one allow-listed field; no ids added or removed, "
                     "no vector written; no release is published by this transaction"}},
        {"status": "TWIN-VERIFIED", "proof": {**leg, "role": "twin", "applied": 0, "already_applied": 2}},
        {"status": "TWIN-GATE-PASS", "proof": {"mechanical": True, "pack_id": PACK, "status": "PASS"}},
        {"status": "PROD-VERIFIED", "proof": {**leg, "role": "prod", "applied": 2, "already_applied": 0}},
        {"status": "TWIN-PROD-VERIFIED-OK", "rollback": "artifact and target readback proofs retained; "
                                                        "no destructive automatic rollback"},
    ]


def test_drain_hands_the_leg_the_run_arguments_and_the_receipt_on_disk_is_the_landed_one(tmp_path, monkeypatch):
    h = _world(monkeypatch, tmp_path)
    monkeypatch.setattr(PSP, "_api", lambda url, path, body=None, method=None: {"aliases": [
        {"alias_name": "aw_alt", "collection_name": "aw_e9"}]})
    inner, seen = Q.live_apply_payload_patch, []
    monkeypatch.setattr(Q, "live_apply_payload_patch", lambda **k: seen.append(k) or inner(**k))
    (res,) = _go(h, batch=5000, pause_s=0.25, prod_alias="aw_alt")
    (k,) = seen
    assert {n: k[n] for n in ("pack_id", "twin_url", "prod_url", "task", "batch", "pause_s", "artifact_root",
                              "prod_collection", "prod_alias", "pack_dir")} == {
        "pack_id": PACK, "twin_url": "http://twin", "prod_url": "http://prod", "task": "2748-test",
        "batch": 5000, "pause_s": 0.25, "artifact_root": h["art"], "prod_collection": "aw_e9",
        "prod_alias": "aw_alt", "pack_dir": h["pack"]}
    assert k["manifest"]["field"] == "snippet" and isinstance(k["write_attempt"], Q.WriteAttempt)
    on_disk = json.loads((h["pack"] / "receipt.json").read_text())
    assert on_disk["status"] == "LANDED" and on_disk["inverse_pack"] == res["inverse_pack"]
    assert (h["pack"] / "DONE").is_file()


def test_the_change_log_segment_is_written_before_done_and_holds_every_row_now_at_the_new_value(
        tmp_path, monkeypatch):
    new, _ = _new_old()
    prod = {U["a"]: {"snippet": "old-a"}, U["b"]: {"snippet": "new-b"}, U["e"]: {"snippet": "bystander"}}
    h = _world(monkeypatch, tmp_path, twin=new, prod=prod)
    logs = tmp_path / "changelog"
    monkeypatch.setenv("AW_CHANGE_LOG_DIR", str(logs))
    at_log = []
    real = CL.record_landing
    monkeypatch.setattr(CL, "record_landing",
                        lambda rec, plan, out: at_log.append((h["pack"] / "DONE").exists()) or real(rec, plan, out))
    (res,) = _go(h)
    assert res["prod"]["applied"] == 1 and res["prod"]["already_applied"] == 1
    assert at_log == [False]
    header, lines = CL.read_segment(CL.segment_for(logs, PACK))
    assert {(x["id"], x["op"]) for x in lines} == {(U["a"], "replace"), (U["b"], "replace")}
    assert header["pack_sha"] == res["manifest_sha"]


def test_a_rerun_after_an_abort_between_the_legs_lands_without_rewriting_the_twin(tmp_path, monkeypatch):
    h = _world(monkeypatch, tmp_path, twin={U["a"]: {"snippet": "old-a"}, U["b"]: {"snippet": "old-b"}},
               prod={U["a"]: {"snippet": "old-a"}, U["b"]: {"snippet": "old-b"}})
    _flip_after_prod_read(h, monkeypatch)
    with pytest.raises(Q.TFA.ApplyAbort):
        _go(h)
    twin_writes = len(h["twin"].writes)
    assert twin_writes and not h["prod"].writes
    monkeypatch.setattr(PSP, "_api", lambda url, path, body=None, method=None: {"aliases": [
        {"alias_name": "aw_prod", "collection_name": "aw_e9"}]})
    (res,) = _go(h)
    assert res["status"] == "LANDED" and (h["pack"] / "DONE").is_file()
    assert (res["twin"]["applied"], res["twin"]["already_applied"]) == (0, 2)
    assert len(h["twin"].writes) == twin_writes
    assert (res["prod"]["applied"], res["prod"]["already_applied"]) == (2, 0)
    assert h["prod"].points[U["a"]]["snippet"] == "new-a"


# ---- the per-pack driver, called directly: one result, one receipt, per outcome
def _direct(tmp_path, monkeypatch, *, gate_profile="mechanical", apply=True, live=None, manifest_over=None):
    h = _world(monkeypatch, tmp_path)
    manifest = {**json.loads((h["pack"] / "manifest.json").read_text()), **(manifest_over or {})}
    if live is not None:
        monkeypatch.setattr(Q, "live_apply_payload_patch", live)
    gate = {"profile": gate_profile, "contract_path": "/x/gate.json", "pr": 302}
    record = {"pack_id": PACK, "manifest_sha": "sha"}
    results: list = []

    def call():
        return Q._drain_payload_patch(
            pack_dir=h["pack"], manifest=manifest, record=record, gate=gate, apply=apply,
            twin_url="http://twin", prod_url="http://prod", task="t", batch=10, pause_s=0,
            artifact_root=h["art"], prod_collection=lambda: "aw_e9", prod_alias="aw_prod", results=results)
    return h, record, results, call


def _receipt(h):
    return json.loads((h["pack"] / "receipt.json").read_text())


def _no_write(**k):
    raise AssertionError("no write may be issued")


def _raises(exc, *, started):
    def live(**k):
        k["write_attempt"].started = started
        raise exc
    return live


def test_unreadable_rows_refuse_the_pack_with_a_receipt_and_one_result(tmp_path, monkeypatch):
    h, record, results, call = _direct(tmp_path, monkeypatch, manifest_over={"rows_file": "gone.jsonl"})
    assert call() is None
    assert results == [record] and record["status"] == "REFUSED" and record["stage"] == "payload_patch_check"
    assert _receipt(h)["status"] == "REFUSED" and not (h["pack"] / "DONE").exists()
    assert not h["twin"].writes


def test_a_dry_run_leaves_a_receipt_and_no_write(tmp_path, monkeypatch):
    h, record, results, call = _direct(tmp_path, monkeypatch, apply=False, live=_no_write)
    call()
    assert results == [record] and record["status"] == "PROVED-DRY-RUN" and record["blocked_on"] == []
    assert _receipt(h)["status"] == "PROVED-DRY-RUN" and not (h["pack"] / "DONE").exists()


def test_a_gate_that_is_not_mechanical_blocks_the_pack_even_with_apply(tmp_path, monkeypatch):
    h, record, results, call = _direct(tmp_path, monkeypatch, gate_profile="proposed", live=_no_write)
    call()
    assert results == [record] and record["status"] == "PROVED-BLOCKED"
    assert record["blocked_on"] == [
        "gate: /x/gate.json profile is 'proposed' (flips to 'mechanical' once PR #302 merges)"]
    assert _receipt(h)["status"] == "PROVED-BLOCKED"


def test_an_interrupt_after_a_write_records_aborted_and_propagates(tmp_path, monkeypatch):
    h, record, results, call = _direct(tmp_path, monkeypatch, live=_raises(KeyboardInterrupt(), started=True))
    with pytest.raises(KeyboardInterrupt):
        call()
    assert results == [record] and (record["status"], record["writes_issued"]) == ("ABORTED", True)
    assert _receipt(h)["status"] == "ABORTED"


def test_a_failure_after_a_write_records_aborted_with_the_result_and_propagates(tmp_path, monkeypatch):
    h, record, results, call = _direct(tmp_path, monkeypatch, live=_raises(RuntimeError("boom"), started=True))
    with pytest.raises(RuntimeError):
        call()
    assert results == [record] and (record["status"], record["writes_issued"]) == ("ABORTED", True)
    assert _receipt(h)["status"] == "ABORTED"


def test_a_pre_write_classification_refusal_is_aborted_with_no_writes_and_propagates(tmp_path, monkeypatch):
    h, record, results, call = _direct(tmp_path, monkeypatch,
                                       live=_raises(PP.PayloadPatchRefusal("moved"), started=False))
    with pytest.raises(PP.PayloadPatchRefusal):
        call()
    assert results == [record] and (record["status"], record["writes_issued"]) == ("ABORTED", False)


def test_a_plain_failure_before_any_write_is_refused_and_the_drain_goes_on(tmp_path, monkeypatch):
    h, record, results, call = _direct(tmp_path, monkeypatch, live=_raises(RuntimeError("busy"), started=False))
    assert call() is None
    assert results == [record] and (record["status"], record["stage"]) == ("REFUSED", "live-apply-preflight")
    assert _receipt(h)["status"] == "REFUSED" and not (h["pack"] / "DONE").exists()


def test_an_alias_move_before_the_first_write_is_refused_with_the_result_and_stops_the_drain(tmp_path, monkeypatch):
    h, record, results, call = _direct(tmp_path, monkeypatch,
                                       live=_raises(Q.TFA.ApplyAbort("moved mid-drain"), started=False))
    with pytest.raises(Q.TFA.ApplyAbort):
        call()
    assert results == [record] and record["status"] == "REFUSED" and _receipt(h)["status"] == "REFUSED"



def test_prod_patching_other_rows_than_the_twin_is_refused_by_id_set_with_the_twin_written(tmp_path, monkeypatch):
    twin = {U["a"]: {"snippet": "old-a"}, U["e"]: {"snippet": "bystander"}}
    prod = {U["b"]: {"snippet": "old-b"}, U["e"]: {"snippet": "bystander"}}
    h = _world(monkeypatch, tmp_path, twin=twin, prod=prod, expect_missing=1)
    logs = tmp_path / "changelog"
    monkeypatch.setenv("AW_CHANGE_LOG_DIR", str(logs))
    with pytest.raises(PP.PayloadPatchRefusal):
        _go(h)
    rec = json.loads((h["pack"] / "receipt.json").read_text())
    assert rec["status"] == "ABORTED" and rec["writes_issued"] is True and not (h["pack"] / "DONE").exists()
    assert h["twin"].points[U["a"]]["snippet"] == "new-a"
    assert "1 row(s) only on this leg" in rec["reason"] and "1 row(s) only on the twin" in rec["reason"]
    assert "new-" not in rec["reason"] and "old-" not in rec["reason"]
    assert not h["prod"].writes and CL.segment_for(logs, PACK) is None


def test_the_same_missing_row_on_both_legs_still_lands_and_logs_prods_ids(tmp_path, monkeypatch):
    twin = {U["a"]: {"snippet": "old-a"}, U["e"]: {"snippet": "bystander"}}
    prod = {U["a"]: {"snippet": "old-a"}, U["e"]: {"snippet": "bystander"}}
    h = _world(monkeypatch, tmp_path, twin=twin, prod=prod, expect_missing=1)
    logs = tmp_path / "changelog"
    monkeypatch.setenv("AW_CHANGE_LOG_DIR", str(logs))
    (res,) = _go(h)
    assert res["status"] == "LANDED" and (h["pack"] / "DONE").exists()
    assert h["twin"].points[U["a"]]["snippet"] == h["prod"].points[U["a"]]["snippet"] == "new-a"
    _, lines = CL.read_segment(CL.segment_for(logs, PACK))
    assert {(x["id"], x["op"]) for x in lines} == {(U["a"], "replace")}


def test_a_rerun_with_already_applied_on_one_leg_and_applied_on_the_other_lands(tmp_path, monkeypatch):
    twin = {U["a"]: {"snippet": "new-a"}, U["b"]: {"snippet": "new-b"}, U["e"]: {"snippet": "bystander"}}
    prod = {U["a"]: {"snippet": "old-a"}, U["b"]: {"snippet": "old-b"}, U["e"]: {"snippet": "bystander"}}
    h = _world(monkeypatch, tmp_path, twin=twin, prod=prod)
    (res,) = _go(h)
    assert res["status"] == "LANDED"
    assert res["twin"]["already_applied"] == 2 and res["prod"]["applied"] == 2


def test_mixed_classes_on_both_legs_land_whatever_the_classification_order(tmp_path, monkeypatch):
    twin = {U["a"]: {"snippet": "new-a"}, U["b"]: {"snippet": "old-b"}, U["e"]: {"snippet": "bystander"}}
    prod = {U["a"]: {"snippet": "old-a"}, U["b"]: {"snippet": "new-b"}, U["e"]: {"snippet": "bystander"}}
    h = _world(monkeypatch, tmp_path, twin=twin, prod=prod)
    (res,) = _go(h)
    assert res["status"] == "LANDED"
    for leg in ("twin", "prod"):
        assert res[leg]["applied"] == 1 and res[leg]["already_applied"] == 1


def test_2938_writes_issued_is_false_for_a_twin_leg_classification_refusal(tmp_path, monkeypatch):
    """`record["writes_issued"]` is `write_attempt.started`: True once ANY leg has begun a write. A twin-leg
    classification refusal (a moved row on the twin) happens before any write anywhere: False. (The module
    docstring says classification refusals have writes_issued=false: true for this one only, see the next test.)"""
    h = _make(monkeypatch, tmp_path, points(), points(), rows(), expect_missing=1)
    rec = _aborted(h)
    assert rec["writes_issued"] is False and not h["twin"].writes and not h["prod"].writes


def test_2938_writes_issued_is_true_for_a_prod_leg_classification_refusal_after_a_written_twin(tmp_path, monkeypatch):
    """A moved row on PROD (the twin already patched) is also a pre-write classification refusal, of the prod leg;
    the twin has written, so `writes_issued` is True: the twin holds the new values, prod is untouched, no DONE.
    (A twin that already held the new values would issue no write: the flag then stays False. It counts writes
    issued, not legs run.)"""
    _new, old = _new_old()
    moved = {**old, U["a"]: {"snippet": "third"}}                  # prod's row a is neither old nor new
    h = _world(monkeypatch, tmp_path, twin=old, prod=moved)        # the twin still needs the patch: it writes
    with pytest.raises(PP.PayloadPatchError):
        _go(h)
    rec = json.loads((h["pack"] / "receipt.json").read_text())
    assert rec["status"] == "ABORTED" and rec["writes_issued"] is True and not (h["pack"] / "DONE").exists()
    assert h["twin"].points[U["a"]]["snippet"] == "new-a"
    assert not h["prod"].writes and h["prod"].points[U["a"]]["snippet"] == "third"
