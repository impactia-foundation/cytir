"""Task 788: a sealed pack whose PROD leg died after a partial write resumes (F1 of task 786).

Before: the prod leg of a sealed pack died after k of n upsert batches (a tunnel stall, a unit
timeout, an OOM). Every later drain ran `resume_prod_only` -> `apply_side("prod")` -> `plan(prod)`,
which applied the first-time rule ("any id present refuses", `AlreadyLandedError`) to prod's
partial set and recorded `REFUSED live-apply-preflight` every hour. Only the manual task-664
reconcile closed it.

After: the twin-written ids ride into `plan()` (`resume_twin_ids`). Prod's present ids must be
twin-written ids (skipped, never re-sent); exactly the rest are written; an id prod holds that the
twin leg did not write refuses as a collision; a complete prod writes nothing and closes.

The transaction tests run the REAL `plan()` over the real 3-row sealed fixture (task 629's, the
same one `test_aw_url_group_supersede` reads; it is not in the public cut, so they skip there) and
the REAL `already_present`/`upsert`/`_api`, against a fake Qdrant at the `urlopen` level (no
network), so the #718 retries and the prod guard are the code that runs.
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import urllib.error
import urllib.request

import pytest

from cytir.index import prod_guard as PG
from cytir.index import promote_queue as PQ
from cytir.index import promote_sealed_pack as P

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEALED = ROOT / "tests" / "fixtures" / "promote" / "task629_sealed_pack"
_H = "127.0.0.1"
PROD = "http://%s:%d" % (_H, PG.PROD_PORT)
TWIN = "http://%s:%d" % (_H, PG.PROD_PORT + 123)
PACK = "resume-788"
needs_pack = pytest.mark.skipif(not list((SEALED / "vectors").glob("*.vectors.manifest.json")),
                                reason="the real sealed-pack fixture is not in this tree (the public cut)")

MANIFEST = {
    "schema": PQ.MANIFEST_SCHEMA,
    "pack_id": PACK,
    "family": "technical_manufacturer_catalog",
    "pack_chunks": str(SEALED / "chunks"),
    "vectors": str(SEALED / "vectors"),
    "mrl": 256,
    "collection": "aw_e8",
    "identity_check": "pack-declared",
    "identity_reason": "test: Pool A ids are not reproducible; module docstring",
    "family_payload": {"source_type": "docs", "trust_tier": 0.6},
    "title_from": ["source_metadata.title"],
}


def _ids() -> list[str]:
    """What the fixture projects (identity `pack-declared`: the pack's own ids), in the order a
    first-time apply writes them (projection order)."""
    return [str(r["point_id"]) for r in P.read_records(MANIFEST["pack_chunks"])]


# ----------------------------------------------------------------------------- the fake Qdrant
class _Resp:
    def __init__(self, obj):
        self._b = json.dumps(obj).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._b


class FakeQdrant:
    """`urlopen` stand-in for twin and prod (by port). `held[role]` is the id set a collection
    holds; a PUT adds to it; a by-id retrieve answers from it. `fail(role, method, n_role_call)`
    returns an HTTP status to answer instead (n counts that role's calls of that method)."""

    def __init__(self, *, twin=(), prod=(), fail=None):
        self.held = {"twin": set(twin), "prod": set(prod)}
        self.puts: dict[str, list[list[str]]] = {"twin": [], "prod": []}
        self.counts: dict[tuple[str, str], int] = {}
        self.fail = fail or (lambda role, method, n: None)

    def __call__(self, req, timeout=300):
        url = req.full_url
        role = "prod" if f":{PG.PROD_PORT}/" in url else "twin"
        if url.endswith("/aliases"):
            # the live twin carries its own `aw_prod` -> aw_e8 alias (task 790's replay resolved it)
            aliases = [{"alias_name": "aw_prod",
                        "collection_name": "aw_e9" if role == "prod" else "aw_e8"}]
            return _Resp({"status": "ok", "result": {"aliases": aliases}})
        method = req.get_method()
        n = self.counts[(role, method)] = self.counts.get((role, method), 0) + 1
        code = self.fail(role, method, n)
        if code:
            raise urllib.error.HTTPError(url, code, "x", hdrs=None, fp=io.BytesIO(b"fake"))
        body = json.loads(req.data) if req.data else {}
        if url.endswith("/points/count"):          # the twin plan's served-url count: none served
            assert role == "twin", "prod is never asked the url question"
            return _Resp({"status": "ok", "result": {"count": 0}})
        if method == "PUT":
            ids = [str(p["id"]) for p in body["points"]]
            self.puts[role].append(ids)
            self.held[role] |= set(ids)
            return _Resp({"status": "ok", "result": {}})
        return _Resp({"status": "ok",
                      "result": [{"id": i} for i in body["ids"] if i in self.held[role]]})

    def prod_put_ids(self) -> list[str]:
        return [i for batch in self.puts["prod"] for i in batch]


class _FakeLock:
    def __init__(self, *_a, **_k):
        pass

    def acquire(self):
        pass

    def release(self):
        pass


class _FakeRB:
    from cytir.ops.write_lock import LOCKS
    assert_locks_free = staticmethod(lambda task: {})
    acquire = staticmethod(lambda name, task, purpose, *, pid=None: None)
    release = staticmethod(lambda name, task: None)


@pytest.fixture(autouse=True)
def world(monkeypatch):
    """Everything but Qdrant is real or a no-op; returns the recorded sleeps."""
    monkeypatch.setenv("AW_PROD_READ_BUDGET", "unlimited")
    monkeypatch.delenv("AW_PROD_URL", raising=False)
    PG.reset()
    P.PROD_STATS.reset()
    sleeps: list[float] = []
    monkeypatch.setattr(P.time, "sleep", sleeps.append)
    monkeypatch.setattr(PG, "PROD_BATCH_SLEEP_S", 0)
    monkeypatch.setattr(P, "target_width", lambda *_a, **_k: 256)
    monkeypatch.setattr(P.PI, "assert_pages_persisted", lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(P.PC, "assert_scoreable", lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(P.LAYOUT, "fingerprint", lambda *_a, **_k: {"points_count": 0})
    monkeypatch.setattr(P.SR.SparseRule, "for_collection", classmethod(lambda c, n: None))
    ident = type("I", (), {"project_points": lambda self, p: p})()
    monkeypatch.setattr(P.PP.Projector, "for_collection", classmethod(lambda c, n: ident))
    monkeypatch.setattr(PQ, "_revalidate_prod_collection", lambda *a, **k: None)
    monkeypatch.setattr(PQ.TFA, "ProductionWriterLock", _FakeLock)
    monkeypatch.setattr(PQ, "_load_run_batch", lambda: _FakeRB)
    monkeypatch.setattr(PQ, "prod_lock_host", lambda: "lockhost")
    return sleeps


def _install(monkeypatch, fake: FakeQdrant) -> FakeQdrant:
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


def _twin_leg(tmp_path, manifest=MANIFEST, drop_ids=frozenset()) -> pathlib.Path:
    """The twin leg as the drain runs it: `apply_side("twin")` writes twin and leaves the per-pack
    twin-written-ids artifact the resume reads. `drop_ids`: the drain's drop set (mechanical-content
    problems, prose/binary hold-outs, already-served urls)."""
    _, ids_path = PQ.apply_side("twin", TWIN, manifest, "aw_e8", batch=1, pause_s=0,
                                ids_dir=tmp_path, drop_ids=drop_ids)
    assert ids_path == PQ.twin_written_ids_path(tmp_path, manifest["pack_id"])
    return ids_path


def _prod_leg_dies_after(tmp_path, fake: FakeQdrant, k: int, manifest=MANIFEST,
                         drop_ids=frozenset()):
    """The first prod leg (as `live_apply` runs it, with the same drop set), batch=1: PUT k+1
    answers 503 on every retry, so k batches landed."""
    fake.fail = lambda role, method, n: 503 if role == "prod" and method == "PUT" and n > k else None
    with pytest.raises(P.PromotionError, match="HTTP 503"):
        PQ.apply_side("prod", PROD, manifest, "aw_e9", batch=1, pause_s=0, ids_dir=tmp_path,
                      drop_ids=drop_ids)
    fake.fail = lambda *a: None
    fake.puts["prod"].clear()
    P.PROD_STATS.reset()


def _resume(tmp_path, manifest=MANIFEST, write_attempt=None, **kw):
    wa = write_attempt if write_attempt is not None else PQ.WriteAttempt()
    state = PQ.resume_prod_only(pack_id=manifest["pack_id"], manifest=manifest, twin_url=TWIN,
                                prod_url=PROD, task="788-test", batch=1, pause_s=0,
                                artifact_root=tmp_path, prod_collection="aw_e9",
                                write_attempt=wa, **kw)
    return state, wa


def _drain_once(tmp_path, monkeypatch, artifacts, fake: FakeQdrant, *, apply: bool):
    """One drain of the queued pack. `dry_plan` against twin raises `AlreadyLandedError` over the
    WHOLE projection, as the real one does for a landed pack (the twin holds what it wrote)."""
    ids = _ids()

    def dry_plan_on_twin(_manifest, _url, **_k):
        present = [i for i in ids if i in fake.held["twin"]]
        raise P.AlreadyLandedError(f"{len(present)} of {len(ids)} ids are already in aw_e8",
                                   ids=frozenset(ids), present=frozenset(present))
    monkeypatch.setattr(PQ, "dry_plan", dry_plan_on_twin)
    queue = tmp_path / "queue"
    pack_dir = queue / PACK
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "manifest.json").write_text(json.dumps(MANIFEST, indent=1))
    (pack_dir / "READY").write_text(
        hashlib.sha256((pack_dir / "manifest.json").read_bytes()).hexdigest() + "\n")
    contract = tmp_path / "gate-contract.json"
    contract.write_text(json.dumps({"schema": PQ.GATE_CONTRACT_SCHEMA, "adr": "ADR-110",
                                    "pr": 302, "profile": "mechanical"}))
    # a dry drain is a replay: the twin stands for both sides (prod_guard refuses prod), as task 790 ran it
    (rec,) = PQ.drain(queue, twin_url=TWIN, prod_url=(PROD if apply else TWIN), task="788-test", apply=apply,
                      batch=1, pause_s=0, artifact_root=artifacts, contract_path=contract,
                      refresh_index=False)
    return pack_dir, rec


def _ledger(tmp_path) -> list[str]:
    path = tmp_path / f"{PACK}-resume-ledger.jsonl"
    return [json.loads(line)["status"] for line in path.read_text().splitlines()]


# ------------------------------------------------- (1) the defect, and a partial prod resumes
@needs_pack
@pytest.mark.parametrize("k", [0, 1, 2])
def test_a_prod_leg_that_died_after_k_of_n_batches_resumes_with_exactly_n_minus_k(
        tmp_path, monkeypatch, k):
    fake = _install(monkeypatch, FakeQdrant())
    ids = _ids()
    twin_file = _twin_leg(tmp_path)
    assert fake.held["twin"] == set(ids)
    _prod_leg_dies_after(tmp_path, fake, k)
    landed = set(ids[:k])
    assert fake.held["prod"] == landed

    if k:
        # The defect (main before task 788): the first-time rule re-asked of a partial prod refuses.
        with pytest.raises(P.AlreadyLandedError) as e:
            PQ.apply_side("prod", PROD, MANIFEST, "aw_e9", batch=1, pause_s=0, ids_dir=tmp_path)
        assert not isinstance(e.value, P.ResumeCollisionError)
        assert fake.puts["prod"] == []

    state, wa = _resume(tmp_path)
    put = fake.prod_put_ids()
    assert sorted(put) == sorted(ids[k:]), "exactly the n-k missing ids"
    assert not set(put) & landed, "never overwrites an id prod already holds"
    assert fake.held["prod"] == set(ids)
    assert fake.puts["twin"] == [[i] for i in ids], "twin was written once, by its own leg"
    r = state["prod_receipt"]
    assert r["written"] == len(ids) - k
    assert r["resumed_partial"] is (k > 0)
    assert r["resume"]["resumed_partial"] is (k > 0)
    assert r["resume"]["present_before"] == k
    assert r["resume"]["present_after"] == len(ids)
    assert r["resume"]["twin_written_ids"] == len(ids)
    assert r["resume"]["twin_written_ids_sha256"] == hashlib.sha256(twin_file.read_bytes()).hexdigest()
    assert r["resume"]["twin_written_ids_source"] == str(twin_file)
    assert r["rows"]["projected"] == len(ids), "the plan still measured the whole pack"
    assert wa.started is True
    assert _ledger(tmp_path)[-1] == "TWIN-PROD-VERIFIED-OK"


@needs_pack
def test_the_drain_lands_a_partially_written_sealed_pack_instead_of_refusing(tmp_path, monkeypatch):
    """The drain's own path: twin landed, prod died after 1 of 3, the next drain resumes and
    marks DONE (before task 788: REFUSED live-apply-preflight, every run)."""
    fake = _install(monkeypatch, FakeQdrant())
    ids = _ids()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _twin_leg(artifacts)
    _prod_leg_dies_after(artifacts, fake, 1)

    pack_dir, rec = _drain_once(tmp_path, monkeypatch, artifacts, fake, apply=True)

    assert rec["status"] == "LANDED", rec
    assert rec["resumed_prod_only"] is True
    assert rec["prod"]["resumed_partial"] is True
    assert rec["prod"]["resume"]["present_before"] == 1 and rec["prod"]["written"] == 2
    assert (pack_dir / "DONE").exists()
    assert sorted(fake.prod_put_ids()) == sorted(ids[1:])
    # task 801: the receipt keeps the twin-first shape the readers need, next to `prod` + the resume record
    from cytir.ops import cc_tranche_delta as D
    assert rec["mechanical_plan"]["rows"]["distinct_urls"] == rec["prod"]["rows"]["distinct_urls"] > 0
    assert D._distinct_urls(json.loads((pack_dir / "receipt.json").read_text())) \
        == rec["prod"]["rows"]["distinct_urls"]


# ---------------------------------------------------------- (2) a complete prod leg: no-op DONE
@needs_pack
def test_a_resume_after_a_complete_prod_leg_is_a_no_op_that_closes(tmp_path, monkeypatch):
    fake = _install(monkeypatch, FakeQdrant())
    ids = _ids()
    _twin_leg(tmp_path)
    fake.held["prod"] = set(ids)

    state, wa = _resume(tmp_path)

    assert fake.puts["prod"] == [], "nothing sent"
    assert wa.started is False, "no write started: a failure after this is not a mid-write ABORT"
    r = state["prod_receipt"]
    assert r["written"] == 0 and r["resume"]["present_before"] == len(ids)
    assert r["resume"]["present_after"] == len(ids) and r["resumed_partial"] is True
    assert _ledger(tmp_path)[-2:] == ["PROD-VERIFIED", "TWIN-PROD-VERIFIED-OK"]
    # and again: idempotent
    state2, _ = _resume(tmp_path)
    assert state2["prod_receipt"]["written"] == 0 and fake.puts["prod"] == []


# ------------------------------------------- (3) an id on prod the twin leg did not write: refuse
@needs_pack
def test_a_prod_id_outside_the_twin_written_ids_is_a_collision_and_refuses(tmp_path, monkeypatch):
    fake = _install(monkeypatch, FakeQdrant())
    ids = _ids()
    # the twin leg wrote only two of the three (the third was dropped by the content check)
    PQ.apply_side("twin", TWIN, MANIFEST, "aw_e8", batch=1, pause_s=0, ids_dir=tmp_path,
                  drop_ids=frozenset({ids[2]}))
    assert sorted(json.loads(PQ.twin_written_ids_path(tmp_path, PACK).read_text())) == sorted(ids[:2])
    fake.held["prod"] = {ids[0], ids[2]}       # ids[2] is NOT ours: some other write put it there

    with pytest.raises(P.ResumeCollisionError) as e:
        _resume(tmp_path)
    assert e.value.present == frozenset({ids[2]})
    assert "collision" in str(e.value)
    assert fake.puts["prod"] == [], "nothing written next to a collision"
    assert _ledger(tmp_path)[-1] == "FAILED"


# ------------------------- (6) the write set is the twin's, never the pack's (tranche 14, 2026-10-09)
@needs_pack
@pytest.mark.parametrize("landed", [0, 1])
def test_a_resume_writes_the_twins_set_not_the_pack_when_the_twin_leg_held_rows_out(
        tmp_path, monkeypatch, landed):
    """Pack of n rows; the drain held k out (here k=1: the binary-content row). Twin wrote n-k; the
    first prod leg died after `landed` batches; the resume writes n-k-landed and NEVER the held-out
    row, and the prod artifact equals the twin's. Main's resume wrote the whole pack: 213 gate-
    rejected rows reached prod only."""
    fake = _install(monkeypatch, FakeQdrant())
    ids = _ids()
    held_out = frozenset({ids[2]})
    twin_file = _twin_leg(tmp_path, drop_ids=held_out)
    assert sorted(json.loads(twin_file.read_text())) == sorted(ids[:2])
    _prod_leg_dies_after(tmp_path, fake, landed, drop_ids=held_out)

    state, _ = _resume(tmp_path)

    assert sorted(fake.prod_put_ids()) == sorted(ids[landed:2]), "n - k - landed, never a held-out row"
    assert ids[2] not in fake.held["prod"]
    assert fake.held["prod"] == fake.held["twin"] == set(ids[:2])
    prod_file = PQ.twin_written_ids_path(tmp_path, PACK).with_name(f"{PACK}-prod-written-ids.json")
    assert set(json.loads(prod_file.read_text())) == set(json.loads(twin_file.read_text()))
    r = state["prod_receipt"]["resume"]
    assert r["dropped_by_twin_decision"] == 1 and r["written"] == 2 - landed
    assert r["prod_written_ids_equal_twin"] is True and r["projected"] == len(ids)


@needs_pack
def test_a_prod_artifact_that_differs_from_the_twins_fails_the_leg_and_never_closes(
        tmp_path, monkeypatch):
    """The guard behind the rule: if a regression made the resume write the pack (the tranche-14
    shape), the leg refuses BEFORE any write (review 797 M2); the ledger ends FAILED, not OK."""
    fake = _install(monkeypatch, FakeQdrant())
    ids = _ids()
    _twin_leg(tmp_path, drop_ids=frozenset({ids[2]}))
    real = P.resume_partition

    def writes_the_pack(written_ids, present, twin_ids, *, collection):     # main's old behaviour
        got = real(written_ids, present, twin_ids, collection=collection)
        got["_to_write"] = [i for i in written_ids if i not in set(present)]
        return got
    monkeypatch.setattr(P, "resume_partition", writes_the_pack)

    wa = PQ.WriteAttempt()
    with pytest.raises(PQ.QueueError, match="would make prod differ from the twin's .*1 not twin-written"):
        _resume(tmp_path, write_attempt=wa)
    assert ids[2] not in fake.held["prod"] and fake.puts["prod"] == [], "refused before any write"
    assert wa.started is False, "no write started: the drain records REFUSED, not ABORTED"
    assert _ledger(tmp_path)[-1] == "FAILED"
    assert "PROD-VERIFIED" not in _ledger(tmp_path)


@needs_pack
def test_a_dropped_id_that_reaches_prod_anyway_fails_the_leg(tmp_path, monkeypatch):
    """Review 797 S1: the artifact says what the leg MEANT. After the write one by-id check asks
    prod that no id the twin leg dropped is there (here something else writes it mid-leg)."""
    fake = _install(monkeypatch, FakeQdrant())
    ids = _ids()
    _twin_leg(tmp_path, drop_ids=frozenset({ids[2]}))
    fake.fail = lambda role, method, n: (fake.held["prod"].add(ids[2])
                                         if role == "prod" and method == "PUT" else None)
    with pytest.raises(PQ.QueueError, match="1 id\\(s\\) the twin leg dropped are on prod"):
        _resume(tmp_path)
    assert _ledger(tmp_path)[-1] == "FAILED"


@needs_pack
def test_a_resume_refused_inside_plan_leaves_no_stale_batch_record(tmp_path, monkeypatch):
    """Review 797 S3: the first prod leg's batch record is deleted when a resume starts, so a
    refusal inside plan() (here a collision) cannot leave it for a later fresh apply."""
    monkeypatch.setattr(PG, "PROD_BIG_PACK_ROWS", 3)
    manifest = {**MANIFEST, "rows": 3}
    fake = _install(monkeypatch, FakeQdrant())
    ids = _ids()
    _twin_leg(tmp_path, manifest, drop_ids=frozenset({ids[2]}))
    _prod_leg_dies_after(tmp_path, fake, 1, manifest, drop_ids=frozenset({ids[2]}))
    assert list(tmp_path.glob(f"{PACK}-*-prod-write.chunks.jsonl"))
    fake.held["prod"].add(ids[2])                         # a collision: plan() refuses
    with pytest.raises(P.ResumeCollisionError):
        _resume(tmp_path, manifest)
    assert not list(tmp_path.glob("*.chunks.jsonl"))


@needs_pack
def test_a_dropped_id_on_prod_is_never_landed_by_the_drain(tmp_path, monkeypatch):
    """Review 797 M1: twin = {ids0} (ids1, ids2 dropped), prod = {ids0, ids1}. Prod lacks no twin id,
    but holds a dropped one: a divergence, never "landed, nothing to do". The live drain refuses
    (the resume sees a collision) and writes no DONE."""
    fake = _install(monkeypatch, FakeQdrant())
    ids = _ids()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _twin_leg(artifacts, drop_ids=frozenset({ids[1], ids[2]}))
    fake.held["prod"] = {ids[0], ids[1]}
    fake.counts.clear()

    pack_dir, rec = _drain_once(tmp_path, monkeypatch, artifacts, fake, apply=True)

    assert rec["status"] == "REFUSED", rec
    assert "landed_with_twin_drops" not in rec and "collision" in rec["reason"]
    assert not (pack_dir / "DONE").exists()
    assert ("prod", "PUT") not in fake.counts
    assert PQ.landed_with_twin_drops(ids, [ids[2]], PQ.twin_written_ids_path(artifacts, PACK)) is None


@needs_pack
@pytest.mark.parametrize("apply", [False, True])
def test_a_landed_pack_with_twin_drops_says_landed_not_resume(tmp_path, monkeypatch, apply):
    """Task 790's replay: a re-plan of a LANDED pack whose twin leg dropped rows saw them "missing
    from prod" and read like a defect. With the pack's own artifact the drain says the real state
    (landed, N dropped by the twin leg's gate, nothing to do): PROVED-DRY-RUN dry, DONE live."""
    fake = _install(monkeypatch, FakeQdrant())
    ids = _ids()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    held_out = frozenset({ids[2]})
    _twin_leg(artifacts, drop_ids=held_out)
    fake.held["prod"] = set(ids[:2])                    # the landed state: prod == twin
    fake.counts.clear()

    pack_dir, rec = _drain_once(tmp_path, monkeypatch, artifacts, fake, apply=apply)

    note = f"pack landed; 1 ids were dropped by the twin leg's gate (see {PACK}-twin-written-ids.json); nothing to do"
    assert rec["landed_with_twin_drops"]["dropped_by_twin_decision"] == 1
    if apply:
        assert rec["status"] == "LANDED" and note in rec["note"]
        assert (pack_dir / "DONE").exists()
    else:
        assert rec["status"] == "PROVED-DRY-RUN" and rec["would"] == f"mark DONE: {note}"
        assert not (pack_dir / "DONE").exists()
    assert fake.puts["prod"] == [] and ("prod", "PUT") not in fake.counts, "nothing written"


@needs_pack
def test_a_resume_refuses_twin_ids_the_pack_does_not_project(tmp_path, monkeypatch):
    fake = _install(monkeypatch, FakeQdrant())
    ids = _ids()
    fake.held["twin"] = set(ids) | {"not-in-this-pack"}
    with pytest.raises(P.PromotionError, match="not projected by this pack"):
        _resume(tmp_path, twin_ids_override=ids + ["not-in-this-pack"])
    assert fake.puts["prod"] == []


def test_resume_partition_is_the_rule():
    w = ["a", "b", "c", "d"]
    got = P.resume_partition(w, ["b"], ["a", "b", "c"], collection="aw_e9")
    assert got["_to_write"] == ["a", "c"] and got["present_before"] == 1
    assert got["dropped_by_twin_decision"] == 1 and got["to_write"] == 2
    assert P.resume_partition(w, [], ["a"], collection="x")["_to_write"] == ["a"]
    assert P.resume_partition(w, ["a"], ["a"], collection="x")["_to_write"] == []
    with pytest.raises(P.ResumeCollisionError):
        P.resume_partition(w, ["d"], ["a", "b"], collection="x")
    with pytest.raises(P.PromotionError, match="not projected"):
        P.resume_partition(w, [], ["a", "z"], collection="x")


@needs_pack
def test_a_first_time_apply_keeps_the_all_or_nothing_rule(tmp_path, monkeypatch):
    """The remint-b1 guard (task 664) is untouched: without the resume set, one present id refuses."""
    fake = _install(monkeypatch, FakeQdrant(prod={_ids()[1]}))
    with pytest.raises(P.AlreadyLandedError, match="1 of 3 ids"):
        PQ.apply_side("prod", PROD, MANIFEST, "aw_e9", batch=1, pause_s=0, ids_dir=tmp_path)
    assert fake.puts["prod"] == []


def test_the_resume_set_is_for_the_prod_resume_only(tmp_path, monkeypatch):
    fake = _install(monkeypatch, FakeQdrant())
    ids = frozenset({"a", "b"})
    for kw in ({"role": "twin"}, {"drop_ids": frozenset({"x"})},
               {"precomputed": ({}, None, {}, {})}):
        args = {"role": "prod", **kw}
        role = args.pop("role")
        with pytest.raises(PQ.QueueError, match="prod leg's resume only"):
            PQ.apply_side(role, PROD, MANIFEST, "aw_e9", batch=1, pause_s=0, ids_dir=tmp_path,
                          resume_twin_ids=ids, **args)
    assert fake.counts == {}, "refused before any call"


@needs_pack
def test_a_resume_never_supersedes(tmp_path, monkeypatch):
    fake = _install(monkeypatch, FakeQdrant())
    ids = frozenset(_ids())
    with pytest.raises(P.PromotionError, match="never overwrites"):
        PQ.apply_side("prod", PROD, {**MANIFEST, "allow_supersede": "x"}, "aw_e9", batch=1,
                      pause_s=0, ids_dir=tmp_path, resume_twin_ids=ids)
    assert ("prod", "POST") not in fake.counts, "refused before the prod lookup"


# ------------------------------------------------- (4) the #718 lookup and upsert retries apply
@needs_pack
def test_the_resume_retries_a_transient_lookup_and_put_status(tmp_path, monkeypatch, world):
    fake = _install(monkeypatch, FakeQdrant())
    ids = _ids()
    _twin_leg(tmp_path)
    _prod_leg_dies_after(tmp_path, fake, 1)
    world.clear()
    # prod's 1st retrieve (the resume's plan lookup) answers 408 once; its 1st PUT answers 503 once
    fake.counts.clear()
    fake.fail = lambda role, method, n: ((408 if method == "POST" else 503)
                                         if role == "prod" and n == 1 else None)

    state, _ = _resume(tmp_path)

    assert sorted(fake.prod_put_ids()) == sorted(ids[1:])
    assert [s for s in world if s] == [5, 5], "one backoff step per retried call"
    snap = P.PROD_STATS.snapshot()
    assert snap["retries"] == 2 and snap["retried_statuses"] == {"408": 1, "503": 1}
    assert state["prod_receipt"]["prod_calls"]["retries"] == 2, "a leg that retried says so"


# ---------------------------------------- (5) a resume that dies mid-way stays resumable (twice)
@needs_pack
def test_a_resume_that_dies_mid_write_is_resumed_by_the_next_one(tmp_path, monkeypatch):
    fake = _install(monkeypatch, FakeQdrant())
    ids = _ids()
    _twin_leg(tmp_path)
    _prod_leg_dies_after(tmp_path, fake, 1)
    # resume 1: its first PUT lands, its second answers 503 past every retry
    fake.counts.clear()
    fake.fail = lambda role, method, n: 503 if role == "prod" and method == "PUT" and n > 1 else None
    with pytest.raises(P.PromotionError, match="HTTP 503"):
        _resume(tmp_path)
    assert fake.held["prod"] == set(ids[:2])
    assert _ledger(tmp_path)[-1] == "FAILED"
    # resume 2: asks prod afresh and writes only the last id
    fake.fail = lambda *a: None
    fake.puts["prod"].clear()
    state, _ = _resume(tmp_path)
    assert fake.prod_put_ids() == [ids[2]]
    assert state["prod_receipt"]["resume"]["present_before"] == 2
    assert fake.held["prod"] == set(ids)
    assert _ledger(tmp_path)[-1] == "TWIN-PROD-VERIFIED-OK"


@needs_pack
def test_a_big_pack_resume_never_reads_the_first_attempts_batch_record(tmp_path, monkeypatch):
    """The first prod leg of a big pack recorded the batches it sent. A resume asks prod afresh
    (what prod lacks is written), keeps no batch record of its own, and spends the stale one."""
    monkeypatch.setattr(PG, "PROD_BIG_PACK_ROWS", 3)
    manifest = {**MANIFEST, "rows": 3}
    fake = _install(monkeypatch, FakeQdrant())
    ids = _ids()
    _twin_leg(tmp_path, manifest)
    _prod_leg_dies_after(tmp_path, fake, 1, manifest)
    (stale,) = tmp_path.glob(f"{PACK}-*-prod-write.chunks.jsonl")
    # prod lost the batch the record says was sent (a qdrant restart before its flush)
    fake.held["prod"].clear()

    state, _ = _resume(tmp_path, manifest)

    assert sorted(fake.prod_put_ids()) == sorted(ids), "the recorded batch is re-sent: prod lacks it"
    assert state["prod_receipt"]["resumed_partial"] is False
    assert not stale.exists(), "the first attempt's batch record is spent"
    assert not list(tmp_path.glob("*.chunks.jsonl")), "no record outlives a closed resume"
    assert state["prod_receipt"]["prod_calls"]["rows"] == 3


# ---------------------------------- task 808: the OLDER "twin and prod already hold every id" exit
@needs_pack
def test_the_older_landed_exit_never_closes_over_rows_the_twin_dropped(tmp_path, monkeypatch):
    """Review 797 S4, the 2026-10-09 tranche-14 shape: twin = pack minus the dropped ids, prod = the
    WHOLE pack. Prod lacks no id, so the older exit saw "twin and prod hold every id" and would write
    DONE over the rows the twin never held. With the twin artifact the drain refuses AT that exit
    (review 811 S1: no fall-through to the resume path, no writer lock, no second full prod lookup)
    with the resume path's `ResumeCollisionError`: REFUSED, no DONE, zero PUTs, and no more prod
    POSTs than main's single membership lookup. (Mutation: reverting the condition fails this.)"""
    fake = _install(monkeypatch, FakeQdrant())
    ids = _ids()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _twin_leg(artifacts, drop_ids=frozenset({ids[2]}))
    fake.held["prod"] = set(ids)
    fake.counts.clear()

    pack_dir, rec = _drain_once(tmp_path, monkeypatch, artifacts, fake, apply=True)

    assert rec["status"] == "REFUSED" and rec["stage"] == "live-apply-preflight", rec
    assert rec["reason"].startswith("ResumeCollisionError:") and "did not write" in rec["reason"]
    assert ids[2] in rec["reason"] and "task-664" in rec["reason"]
    assert not (pack_dir / "DONE").exists()
    assert ("prod", "PUT") not in fake.counts and fake.puts["prod"] == []
    assert fake.counts.get(("prod", "POST"), 0) <= 1, "one membership lookup, as on main"
    assert not PQ.twin_set_matches_prod_pack_ids(
        ids, [], PQ.twin_written_ids_path(artifacts, PACK))


@needs_pack
def test_a_replay_cannot_see_the_twin_dropped_rows_on_prod(tmp_path, monkeypatch):
    """Every drain without --apply is a replay and its "prod" lookup reads the TWIN, so the incident
    state is invisible to it: it closes through the 788 exit ("nothing to do"), asks prod nothing
    and writes nothing. Only the --apply run refuses (previous test). Pinned so nobody reads the
    dry result as proof of the older exit."""
    fake = _install(monkeypatch, FakeQdrant())
    ids = _ids()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _twin_leg(artifacts, drop_ids=frozenset({ids[2]}))
    fake.held["prod"] = set(ids)
    fake.counts.clear()

    pack_dir, rec = _drain_once(tmp_path, monkeypatch, artifacts, fake, apply=False)

    assert rec["status"] == "PROVED-DRY-RUN" and "nothing to do" in rec["would"], rec
    assert not any(role == "prod" for role, _ in fake.counts)
    assert not (pack_dir / "DONE").exists()


@needs_pack
def test_prod_equal_to_the_twin_set_is_still_a_done_no_op(tmp_path, monkeypatch):
    """prod == twin == pack minus dropped: unchanged happy path (closes through the 788 exit)."""
    fake = _install(monkeypatch, FakeQdrant())
    ids = _ids()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _twin_leg(artifacts, drop_ids=frozenset({ids[2]}))
    fake.held["prod"] = set(ids[:2])
    fake.counts.clear()

    pack_dir, rec = _drain_once(tmp_path, monkeypatch, artifacts, fake, apply=True)

    assert rec["status"] == "LANDED" and (pack_dir / "DONE").exists()
    assert ("prod", "PUT") not in fake.counts
    # and a twin set that IS the whole pack passes the older exit's test
    full = tmp_path / "full.json"
    full.write_text(json.dumps(sorted(ids)))
    assert PQ.twin_set_matches_prod_pack_ids(ids, [], full)


@needs_pack
def test_without_a_twin_artifact_the_older_exit_keeps_todays_behaviour(tmp_path, monkeypatch):
    """No twin-written-ids artifact (task-664 incident-2 shape; ruling T14 open): no twin truth to
    compare against, so twin and prod both holding every id still marks DONE. Pinned so a later
    ruling changes it deliberately."""
    fake = _install(monkeypatch, FakeQdrant())
    ids = _ids()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    fake.held["twin"] = set(ids)
    fake.held["prod"] = set(ids)
    fake.counts.clear()

    pack_dir, rec = _drain_once(tmp_path, monkeypatch, artifacts, fake, apply=True)

    assert rec["status"] == "LANDED" and (pack_dir / "DONE").exists(), rec
    assert ("prod", "PUT") not in fake.counts
    assert PQ.twin_set_matches_prod_pack_ids(ids, [], PQ.twin_written_ids_path(artifacts, PACK))
