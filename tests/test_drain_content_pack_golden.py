"""Golden (characterization) test of the drain's `content_pack` path, end to end on real Qdrant.

`drain()` handles three manifest kinds through three near-copies of one protocol. The sealed twin-first
path is pinned by `test_drain_transaction.py` and the payload-patch path by its own golden; this file
pins the content-pack path so that a later move of write-path code is provable behaviour-identical.

Four scenarios run `drain()` against two throwaway collections (the twin `aw_e8` and prod `aw_e9`
behind the `aw_prod` alias), on the four real dila_juri rows of `fixtures/drain_content_pack/points.jsonl`
(the rows of the sealed golden, with their vectors):

  clean          a pack of four rows lands: twin leg, verify, prod leg, verify, DONE
  supersede      a pack plus one `superseded` tombstone of a stored old row at a url the pack writes:
                 the old row is deleted on twin and prod, the change-log segment says so
  refused        a pack whose tombstone names a url the pack does not write: REFUSED by the dry check
                 before any write; both collections unchanged, no DONE marker
  recapture      in-place rows: two of the four pack ids are already on BOTH collections and one on the
                 twin ONLY, so the pre-write lookup, the inplace-ids file and the `expected_delta`
                 arithmetic run; prod's layout count lags by one (as an optimizer can make it), which
                 reaches the `count_note` branch of the verify
  twin_write_started  the twin upsert dies before it wrote; the state file says `twin-write-started`
                 (with a tombstone, so the replay's change-log reason pre-map runs); the next drain
                 replays the twin leg from the recorded ids and lands prod
  crash_resume   the prod upsert dies after the twin leg; the state file says `twin-write-complete`;
                 the next drain resumes prod-only to DONE without a second twin write

Each scenario's record is compared with `fixtures/drain_content_pack/content_pack.golden.json`: the
ordered write effects (which collection got which write, how many ids, which ids), the drain results,
every receipt/state/marker/ledger file, the lock protocol, the change-log segment, and every point
(id, vectors, payload) both collections hold afterwards. Timestamps, durations, paths and the manifest
sha are normalised by the harness the sealed golden uses; prose fields by `test_drain_transaction`.

Needs a throwaway Qdrant (`AW_TEST_QDRANT_URL`); skips without one. Regenerate the golden ONLY for a
deliberate, reviewed change of what the drain writes: `AW_REGEN_GOLDEN=1`.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import sys

import pytest

pytestmark = pytest.mark.usefixtures("no_sparse_rule")

import drain_transaction_harness as H
from cytir.index import promote_queue as Q
from cytir.index import promote_sealed_pack as PSP
from test_drain_transaction import normalise_prose

FIX = H.HERE / "fixtures" / "drain_content_pack"
POINTS = FIX / "points.jsonl"
GOLDEN = FIX / "content_pack.golden.json"
OLD_ROW = "00000000-0000-5000-8000-0000000000c1"
PACK_ID = "cp-golden-batch-00001"


@pytest.fixture
def url():
    u = H.qdrant_url()
    yield u
    try:
        H.clear_live(u)
    except Exception as exc:  # noqa: BLE001 -- teardown must not turn a pass into an error
        print(f"teardown: could not clear {u}: {exc}", file=sys.stderr)


def _points() -> list[dict]:
    return [json.loads(l) for l in POINTS.read_text().splitlines() if l.strip()]


def _old_row(for_url: str) -> dict:
    return {"id": OLD_ROW,
            "payload": {"url": for_url, "source_type": "official", "snippet": "old"},
            "vector": {"dense": [0.1] * H.MRL, "bm25": {"indices": [1], "values": [1.0]}}}


def _world(url: str, *, old_row_url: str | None = None) -> None:
    for name in (H.TWIN, H.PROD):
        H.reset_collection(url, name)
    H.set_alias(url, H.ALIAS, H.PROD)
    if old_row_url:
        for coll in (H.TWIN, H.PROD):
            H.api(url, f"/collections/{coll}/points?wait=true", {"points": [_old_row(old_row_url)]})


def _stale(p: dict) -> dict:
    """The row as an earlier capture stored it: same id, an older snippet."""
    return {"id": p["id"], "vector": p["vector"], "payload": {**p["payload"], "snippet": "stale"}}


def _seed_inplace(url: str, pts: list[dict]) -> None:
    """Pack ids 0 and 1 already on BOTH collections, id 2 on the twin ONLY (id 3 on neither)."""
    for coll, rows in ((H.TWIN, pts[:3]), (H.PROD, pts[:2])):
        H.api(url, f"/collections/{coll}/points?wait=true", {"points": [_stale(p) for p in rows]})


def _lag_prod_count(monkeypatch):
    """Prod's layout fingerprint reads one point too many AFTER its write (an optimizer lag): the
    membership proof still passes, the count delta differs from `expected_delta`: `count_note`."""
    real = Q.LAYOUT.fingerprint
    seen = {"prod": 0}

    def fp(u, coll, *a, **k):
        fpr = real(u, coll, *a, **k)
        if coll == H.PROD:
            seen["prod"] += 1
            if seen["prod"] % 2 == 0:      # the second read of a leg is its `layout_after`
                fpr = {**fpr, "points_count": fpr["points_count"] + 1}
        return fpr
    monkeypatch.setattr(Q.LAYOUT, "fingerprint", fp)
    return real


def _queue(tmp: pathlib.Path, *, tombstone_url: str | None = None) -> pathlib.Path:
    """A queue dir holding one sealed content-pack manifest over the fixture rows."""
    queue = tmp / "queue"
    pack_dir = queue / PACK_ID
    pack_dir.mkdir(parents=True)
    pts = pack_dir / "points.jsonl"
    shutil.copy(POINTS, pts)
    manifest = {"schema": Q.MANIFEST_SCHEMA, "kind": "content_pack", "pack_id": PACK_ID,
                "family": H.FAMILY, "collection": H.TWIN, "pack_points": str(pts),
                "pack_points_sha256": hashlib.sha256(pts.read_bytes()).hexdigest(),
                "rows": len(_points())}
    if tombstone_url:
        tomb = pack_dir / "tombstones.jsonl"
        tomb.write_text(json.dumps({"id": OLD_ROW, "url": tombstone_url, "reason": "superseded"}) + "\n")
        manifest.update(tombstones=str(tomb), tombstones_count=1,
                        tombstones_sha256=hashlib.sha256(tomb.read_bytes()).hexdigest())
    body = json.dumps(manifest, indent=1, sort_keys=True).encode()
    (pack_dir / "manifest.json").write_bytes(body)
    (pack_dir / "READY").write_text(hashlib.sha256(body).hexdigest() + "\n")
    return queue


_REAL_UPSERT, _REAL_DELETE = PSP.upsert, Q.PD.delete_points_verified


class Recorder:
    """The ordered write effects of the run. Wraps the two write primitives of the content-pack legs
    (`upsert`, `delete_points_verified`) and passes through to the real Qdrant. `fail_on` names the
    collection whose upsert dies (once), standing for a crash before that leg wrote.

    HONESTY: the crash is a `KeyboardInterrupt` raised from a pass-through wrapper, so the drain's
    `except BaseException` handler RUNS and the golden pins the receipt and artifacts that handler
    writes. A real SIGKILL runs no handler: the state file the legs wrote before the kill is what
    survives, and this golden does not pin that exact on-disk shape for a kill."""

    def __init__(self, monkeypatch, fail_on: str | None = None):
        self.effects: list = []
        self.fail = fail_on
        real_up, real_del = _REAL_UPSERT, _REAL_DELETE     # never a wrapper of an earlier scenario

        def role(coll):
            return {H.TWIN: "twin", H.PROD: "prod"}.get(coll, coll)

        def up(u, coll, pts, batch, pause_s=0.0):
            if self.fail and coll == self.fail:
                self.fail = None
                self.effects.append(["upsert-died", role(coll), len(pts)])
                raise KeyboardInterrupt(f"simulated crash before the {role(coll)} leg wrote")
            self.effects.append(["upsert", role(coll), len(pts), sorted(str(p["id"]) for p in pts)])
            return real_up(u, coll, pts, batch, pause_s)

        def dele(u, coll, ids, **kw):
            self.effects.append(["delete", role(coll), len(ids), sorted(str(i) for i in ids)])
            return real_del(u, coll, ids, **kw)

        monkeypatch.setattr(Q.PSP, "upsert", up)
        monkeypatch.setattr(Q.PD, "delete_points_verified", dele)


def _drain(queue, url, tmp, artifacts, **extra):
    return Q.drain(queue, twin_url=url, prod_url=url, task="2864-golden", apply=True, batch=3,
                   pause_s=0, artifact_root=artifacts, refresh_index=False, prod_alias=H.ALIAS, **extra)


def _tree(root, tmp):
    """`H.snapshot_tree`, with the pack's points file (copied from the fixture) kept as its sha only."""
    t = H.snapshot_tree(root, tmp) if root.exists() else {}
    return {n: ("sha256:" + hashlib.sha256(x.encode()).hexdigest() if n.endswith("points.jsonl") else x)
            for n, x in t.items()}


def _record(tmp, url, queue, artifacts, results, calls, rec, log_dir=None) -> dict:
    out = {
        "results": json.loads(H.normalise(json.dumps(results, sort_keys=True), tmp)),
        "effects": rec.effects,
        "queue": _tree(queue, tmp),
        "artifacts": _tree(artifacts, tmp),
        "lock_protocol": calls,
        "twin_points": H.dump_points(url, H.TWIN),
        "prod_points": H.dump_points(url, H.PROD),
    }
    if log_dir is not None:
        from cytir.index import change_log as CL
        segs = sorted(log_dir.glob("*.jsonl"))
        out["change_log"] = [
            {"header": {k: v for k, v in h.items() if k in ("n", "op", "kind", "reason")},
             "lines": sorted(({k: l[k] for k in ("op", "id", "reason") if k in l} for l in body),
                            key=lambda d: json.dumps(d, sort_keys=True))}
            for h, body in (CL.read_segment(s) for s in segs)]
    return out


def run_scenarios(tmp: pathlib.Path, url: str, monkeypatch) -> dict:
    from cytir.core import paths
    out: dict = {}
    pts = _points()
    first_url = pts[0]["payload"]["url"]

    def scenario(name, *, tombstone_url=None, old_row_url=None, crash_on=None, recapture=False):
        d = tmp / name
        d.mkdir()
        _world(url, old_row_url=old_row_url)
        queue = _queue(d, tombstone_url=tombstone_url)
        artifacts, log_dir = d / "artifacts", d / "changelog"
        monkeypatch.setenv("AW_CHANGE_LOG_DIR", str(log_dir))
        paths.reset_settings_cache()
        calls = H.install_fakes(monkeypatch, Q, PSP)
        if recapture:
            _seed_inplace(url, pts)
            real_fp = _lag_prod_count(monkeypatch)
        rec = Recorder(monkeypatch, fail_on=crash_on)
        status = []
        results = []
        if crash_on:
            with pytest.raises(BaseException) as ei:        # the drain stops: a write was issued
                _drain(queue, url, d, artifacts)
            status.append(f"raised {type(ei.value).__name__}")
            mid = {"queue": _tree(queue, d), "artifacts": _tree(artifacts, d),
                   "twin_ids": sorted(str(p["id"]) for p in H.dump_points(url, H.TWIN)),
                   "prod_ids": sorted(str(p["id"]) for p in H.dump_points(url, H.PROD))}
            results = _drain(queue, url, d, artifacts)       # the next run resumes prod-only
        else:
            results = _drain(queue, url, d, artifacts)
        if recapture:
            monkeypatch.setattr(Q.LAYOUT, "fingerprint", real_fp)     # the lag is this scenario's only
        got = _record(d, url, queue, artifacts, results, calls, rec, log_dir)
        got["run_status"] = status
        if crash_on:
            got["after_crash"] = mid
        out[name] = got

    scenario("clean")
    scenario("supersede", tombstone_url=first_url, old_row_url=first_url)
    scenario("refused", tombstone_url="https://example.invalid/not-in-this-pack", old_row_url="https://example.invalid/not-in-this-pack")
    scenario("recapture", recapture=True)
    scenario("twin_write_started", tombstone_url=first_url, old_row_url=first_url, crash_on=H.TWIN)
    scenario("crash_resume", crash_on=H.PROD)
    return out


def _normalised(got: dict) -> dict:
    got = json.loads(json.dumps(got, sort_keys=True))
    return {name: normalise_prose(sc) for name, sc in got.items()}


def test_content_pack_path_is_identical_to_the_recorded_golden(tmp_path, url, monkeypatch):
    got = run_scenarios(tmp_path, url, monkeypatch)
    # the scenarios' own claims, independent of the golden
    assert got["clean"]["results"][0]["status"] == "LANDED"
    assert len(got["clean"]["twin_points"]) == len(got["clean"]["prod_points"]) == 4
    assert got["supersede"]["results"][0]["status"] == "LANDED"
    assert OLD_ROW not in {str(p["id"]) for p in got["supersede"]["prod_points"] + got["supersede"]["twin_points"]}
    assert got["refused"]["results"][0]["status"] == "REFUSED"
    assert got["refused"]["results"][0]["stage"] == "content_pack_check"
    assert not any(n.endswith("/DONE") for n in got["refused"]["queue"])
    assert [str(p["id"]) for p in got["refused"]["prod_points"]] == [OLD_ROW]     # nothing written
    assert got["crash_resume"]["results"][0]["status"] == "LANDED"
    rc = got["recapture"]["results"][0]
    assert rc["status"] == "LANDED" and rc["twin"]["tombstoned"] == 0
    assert "points_count moved by +3, expected +2" in got["recapture"]["artifacts"][
        f"{PACK_ID}-ledger.jsonl"], "the count_note branch was not reached"
    assert len(got["recapture"]["prod_points"]) == 4
    tw = got["twin_write_started"]
    assert tw["run_status"] and tw["results"][0]["status"] == "LANDED"
    assert '"stage": "twin-write-started"' in tw["after_crash"]["artifacts"][
        f"{PACK_ID}-content-pack-twin-state.json"]
    assert OLD_ROW not in {str(p["id"]) for p in tw["prod_points"] + tw["twin_points"]}
    if os.environ.get("AW_REGEN_GOLDEN") == "1":
        GOLDEN.write_text(json.dumps(got, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
        pytest.skip("golden regenerated")
    want = _normalised(json.loads(GOLDEN.read_text()))
    got = _normalised(got)
    assert sorted(got) == sorted(want)
    for name in want:
        for key in want[name]:
            if key in ("queue", "artifacts"):
                assert sorted(got[name][key]) == sorted(want[name][key]), f"{name}/{key} files differ"
                for f in want[name][key]:
                    assert got[name][key][f] == want[name][key][f], f"{name}/{key}/{f} differs"
            else:
                assert got[name][key] == want[name][key], f"{name}/{key} differs"
        assert sorted(got[name]) == sorted(want[name])


def test_content_pack_run_is_deterministic_run_to_run(tmp_path, url, monkeypatch):
    """The comparison above is only a proof if two runs of the SAME code agree."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    ga = _normalised(run_scenarios(a, url, monkeypatch))
    gb = _normalised(run_scenarios(b, url, monkeypatch))
    assert json.dumps(ga, sort_keys=True) == json.dumps(gb, sort_keys=True)


# ======================================================================================================
# Task 2911 pins: the protocol steps of `apply_content_pack_side` / `live_apply_content_pack` that the
# golden above does not see (it never runs a big pack, a superseded_generation pack, a moved alias, a
# write that silently did not land, or a non-default write pause, and its fake lock ignores its arguments).
# Each runs the real drain on real Qdrant; each was shown red under one temporary mutation of the code.
# ======================================================================================================
MOVED = "aw_moved2911"
SG_PACK = "cp-sg-batch-00001"
SG_OLD, SG_NEW = "00000000-0000-5000-8000-0000000000d1", "00000000-0000-5000-8000-0000000000d2"
SG_URL = "https://www.legifrance.gouv.fr/loda/article_lc/LEGIARTI000000000777"
PINS_TASK = "2911-pin"


@pytest.fixture
def url2(url):
    yield url
    try:
        if MOVED in {c["name"] for c in H.api(url, "/collections")["collections"]}:
            H.api(url, f"/collections/{MOVED}", method="DELETE")
    except Exception as exc:  # noqa: BLE001 -- teardown must not turn a pass into an error
        print(f"teardown: could not clear {url}: {exc}", file=sys.stderr)


def _pin_boot(tmp, url, monkeypatch, *, tombstone_url=None, old_row_url=None):
    """A queue of the four fixture rows (plus an optional superseded tombstone), the twin-first world,
    the fake locks. Returns (queue, artifacts, lock-calls)."""
    from cytir.core import paths
    _world(url, old_row_url=old_row_url)
    H.reset_collection(url, MOVED)
    queue = _queue(tmp, tombstone_url=tombstone_url)
    monkeypatch.setenv("AW_CHANGE_LOG_DIR", str(tmp / "changelog"))
    paths.reset_settings_cache()
    return queue, tmp / "artifacts", H.install_fakes(monkeypatch, Q, PSP)


def _pin_drain(queue, url, artifacts, *, pause_s=0, batch=3, task=PINS_TASK, **extra):
    return Q.drain(queue, twin_url=url, prod_url=url, task=task, apply=True, batch=batch, pause_s=pause_s,
                   artifact_root=artifacts, refresh_index=False, prod_alias=H.ALIAS, **extra)


def _ids_on(url, coll):
    return sorted(str(p["id"]) for p in H.dump_points(url, coll))


def _spy_upserts(monkeypatch):
    """`(collection, pause_s, got a checkpoint)` of every upsert, passed on with pause 0 (nothing sleeps)."""
    seen: list = []
    real = PSP.upsert

    def up(u, coll, pts, batch, pause_s=0.0, **k):
        seen.append((coll, pause_s, "checkpoint" in k))
        return real(u, coll, pts, batch, 0.0, **k)
    monkeypatch.setattr(PSP, "upsert", up)
    return seen


def test_the_prod_leg_upsert_gets_the_pause_the_write_pause_rule_computes(tmp_path, url2, monkeypatch):
    """The PROD write site of the content-pack path asks `PG.write_pause(role, rows, pause_s)`; the twin
    leg is not slowed. The rule is patched to a visible value, so a call that bypassed it is seen."""
    from cytir.index import prod_guard as PG
    monkeypatch.setattr(PG, "write_pause", lambda role, rows, p: p + (7.0 if role == "prod" else 0.0))
    queue, artifacts, _ = _pin_boot(tmp_path, url2, monkeypatch)
    seen = _spy_upserts(monkeypatch)
    assert _pin_drain(queue, url2, artifacts, pause_s=0.25)[0]["status"] == "LANDED"
    assert {p for c, p, _ in seen if c == H.TWIN} == {0.25}
    assert {p for c, p, _ in seen if c == H.PROD} == {7.25}


def test_the_prod_lock_is_built_for_the_configured_host_the_prod_lock_path_and_this_pack(
        tmp_path, url2, monkeypatch):
    from cytir.index.drain.locks import PROD_LOCK_PATH
    queue, artifacts, _ = _pin_boot(tmp_path, url2, monkeypatch)
    built: list = []

    class Spy(H.FakeLock):
        def __init__(self, *a, **k):
            built.append((a, k))
    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", Spy)
    assert _pin_drain(queue, url2, artifacts)[0]["status"] == "LANDED"
    assert built == [((), {"host": "lockhost", "lock_path": PROD_LOCK_PATH,
                           "purpose": f"task{PINS_TASK}-{PACK_ID}"})]


def test_an_ordinary_pack_takes_no_chunk_checkpoint_on_either_leg(tmp_path, url2, monkeypatch):
    queue, artifacts, _ = _pin_boot(tmp_path, url2, monkeypatch)
    seen = _spy_upserts(monkeypatch)
    lookups: list = []
    real = PSP.already_present

    def present(u, coll, ids, *a, **k):
        lookups.append((coll, "checkpoint" in k))
        return real(u, coll, ids, *a, **k)
    monkeypatch.setattr(PSP, "already_present", present)
    res = _pin_drain(queue, url2, artifacts)[0]
    assert res["status"] == "LANDED" and "prod_calls" not in res["prod"]     # an ordinary receipt is unchanged
    assert [c for c, _, ck in seen if ck] == [] and {c for c, _, _ in seen} == {H.TWIN, H.PROD}
    assert not [c for c in lookups if c[1]]
    assert not list(artifacts.glob("*.chunks.jsonl"))


def test_a_big_pack_checkpoints_the_prod_leg_only_and_a_landing_spends_the_records(tmp_path, url2, monkeypatch):
    """4 rows against a big-pack floor of 3: the PROD leg's pre-write lookup and its upsert carry a chunk
    checkpoint, the twin leg's never; once every batch is sent no record is left."""
    from cytir.index import prod_guard as PG
    monkeypatch.setattr(PG, "PROD_BIG_PACK_ROWS", 3)
    monkeypatch.setattr(PG, "PROD_BIG_PACK_PAUSE_S", 0.0)
    queue, artifacts, _ = _pin_boot(tmp_path, url2, monkeypatch)
    seen = _spy_upserts(monkeypatch)
    lookups: list = []
    real = PSP.already_present

    def present(u, coll, ids, *a, **k):
        lookups.append((coll, len(ids), "checkpoint" in k))
        return real(u, coll, ids, *a, **k)
    monkeypatch.setattr(PSP, "already_present", present)
    res = _pin_drain(queue, url2, artifacts)[0]
    assert res["status"] == "LANDED" and "prod_calls" in res["prod"] and "prod_calls" not in res["twin"]
    assert sorted({(c, ck) for c, _, ck in seen}) == [(H.TWIN, False), (H.PROD, True)]
    assert (H.PROD, 4, True) in lookups
    assert not [x for x in lookups if x[0] == H.TWIN and x[2]]
    assert not list(artifacts.glob("*.chunks.jsonl"))              # spent by the landing
    assert _ids_on(url2, H.PROD) == _ids_on(url2, H.TWIN) and len(_ids_on(url2, H.PROD)) == 4


def test_a_big_pack_prod_leg_that_dies_after_one_batch_leaves_its_write_record(tmp_path, url2, monkeypatch):
    import re
    from cytir.index import prod_guard as PG
    monkeypatch.setattr(PG, "PROD_BIG_PACK_ROWS", 3)
    monkeypatch.setattr(PG, "PROD_BIG_PACK_PAUSE_S", 0.0)
    queue, artifacts, _ = _pin_boot(tmp_path, url2, monkeypatch)
    real, puts = PSP._api, {"prod": 0}

    def api(u, path, body=None, method=None, *a, **k):
        m = re.match(r"/collections/([^/]+)/points\?", path)
        if m and method == "PUT" and body and "points" in body and m.group(1) == H.PROD:
            puts["prod"] += 1
            if puts["prod"] == 2:
                raise KeyboardInterrupt("simulated crash before the second prod batch")
        return real(u, path, body, method, *a, **k)
    monkeypatch.setattr(PSP, "_api", api)
    with pytest.raises(KeyboardInterrupt):
        _pin_drain(queue, url2, artifacts)
    assert len(list(artifacts.glob("*-prod-write.chunks.jsonl"))) == 1     # the one batch that landed
    assert len(_ids_on(url2, H.PROD)) == 3 and len(_ids_on(url2, H.TWIN)) == 4


def test_the_alias_moving_during_the_twin_leg_refuses_before_any_prod_write(tmp_path, url2, monkeypatch):
    """The re-check under the lock, right before the prod write. The check after the write also exists,
    so the pin is on the prod collection staying untouched, not only on the refusal."""
    queue, artifacts, _ = _pin_boot(tmp_path, url2, monkeypatch)
    rec = Recorder(monkeypatch)
    inner = PSP.upsert

    def up(u, coll, pts, batch, pause_s=0.0, **k):
        out = inner(u, coll, pts, batch, pause_s, **k)
        if coll == H.TWIN:
            H.set_alias(url2, H.ALIAS, MOVED)
        return out
    monkeypatch.setattr(PSP, "upsert", up)
    with pytest.raises(Q.TFA.ApplyAbort, match="moved mid-drain"):
        _pin_drain(queue, url2, artifacts)
    assert [e[1] for e in rec.effects] == ["twin"]                 # nothing was sent to prod
    assert _ids_on(url2, H.PROD) == [] and _ids_on(url2, MOVED) == [] and len(_ids_on(url2, H.TWIN)) == 4
    assert not (queue / PACK_ID / "DONE").exists()


def test_the_alias_moving_after_the_prod_write_is_the_last_check_before_done(tmp_path, url2, monkeypatch):
    queue, artifacts, _ = _pin_boot(tmp_path, url2, monkeypatch)
    inner = PSP.upsert

    def up(u, coll, pts, batch, pause_s=0.0, **k):
        out = inner(u, coll, pts, batch, pause_s, **k)
        if coll == H.PROD:
            H.set_alias(url2, H.ALIAS, MOVED)
        return out
    monkeypatch.setattr(PSP, "upsert", up)
    with pytest.raises(Q.TFA.ApplyAbort, match="moved mid-drain"):
        _pin_drain(queue, url2, artifacts)
    assert not (queue / PACK_ID / "DONE").exists()


def test_a_prod_upsert_that_did_not_land_fails_the_membership_proof(tmp_path, url2, monkeypatch):
    queue, artifacts, _ = _pin_boot(tmp_path, url2, monkeypatch)
    inner = PSP.upsert

    def up(u, coll, pts, batch, pause_s=0.0, **k):
        return len(pts) if coll == H.PROD else inner(u, coll, pts, batch, pause_s, **k)   # claims, writes nothing
    monkeypatch.setattr(PSP, "upsert", up)
    with pytest.raises(BaseException) as ei:
        _pin_drain(queue, url2, artifacts)
    assert "pack id(s) missing after upsert (membership check failed)" in str(ei.value)
    assert not (queue / PACK_ID / "DONE").exists()


def test_a_prod_delete_that_did_not_land_fails_the_membership_proof(tmp_path, url2, monkeypatch):
    first = json.loads(POINTS.read_text().splitlines()[0])["payload"]["url"]
    queue, artifacts, _ = _pin_boot(tmp_path, url2, monkeypatch, tombstone_url=first, old_row_url=first)
    inner = Q.PD.delete_points_verified

    def dele(u, coll, ids, **kw):
        return len(ids) if coll == H.PROD else inner(u, coll, ids, **kw)                 # claims, deletes nothing
    monkeypatch.setattr(Q.PD, "delete_points_verified", dele)
    with pytest.raises(BaseException) as ei:
        _pin_drain(queue, url2, artifacts)
    assert "tombstoned id(s) still present after delete (membership check failed)" in str(ei.value)
    assert not (queue / PACK_ID / "DONE").exists()


def test_the_pack_ids_are_checked_against_the_i1_id_rule_before_anything_is_written(tmp_path, url2, monkeypatch):
    """`IR.refuse_unless_i1` runs in the side function before `write_attempt.started`: a refusal there
    leaves the collection untouched."""
    queue, artifacts, _ = _pin_boot(tmp_path, url2, monkeypatch)
    seen: list = []

    def refuse(collection, points):
        seen.append((collection, len(points)))
        raise Q.QueueError("pinned refusal")
    monkeypatch.setattr(Q.IR, "refuse_unless_i1", refuse)
    res = _pin_drain(queue, url2, artifacts)[0]               # a clean REFUSED: the drain goes on
    assert res["status"] == "REFUSED" and "pinned refusal" in json.dumps(res)
    assert seen and seen[0] == (H.TWIN, 4)
    assert _ids_on(url2, H.TWIN) == [] and _ids_on(url2, H.PROD) == []


# ---------------------------------------------------------- superseded_generation (a pure tombstone pack)
def _sg_payload(snippet: str) -> dict:
    return {"url": SG_URL, "source_type": "official", "provenance": "dila-loda-2911", "snippet": snippet}


def _sg_row(rid: str, snippet: str) -> dict:
    return {"id": rid, "payload": _sg_payload(snippet),
            "vector": {"dense": [0.1] * H.MRL, "bm25": {"indices": [1], "values": [1.0]}}}


def _sg_world(url: str) -> None:
    """Twin and prod both hold the OLD generation row and its newer sibling at one document url."""
    for name in (H.TWIN, H.PROD):
        H.reset_collection(url, name)
        H.api(url, f"/collections/{name}/points?wait=true",
              {"points": [_sg_row(SG_OLD, "old generation"), _sg_row(SG_NEW, "new generation")]})
    H.set_alias(url, H.ALIAS, H.PROD)


def _sg_queue(tmp: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """A pure `superseded_generation` pack: no points, one tombstone, the declared export of the old row."""
    import gzip
    queue = tmp / "queue"
    pack_dir = queue / SG_PACK
    pack_dir.mkdir(parents=True)
    pts = pack_dir / "points.jsonl"
    pts.write_text("")
    export = pack_dir / "export.jsonl.gz"
    with gzip.open(export, "wt") as fh:
        fh.write(json.dumps(_sg_row(SG_OLD, "old generation")) + "\n")
    tomb = pack_dir / "tombstones.jsonl"
    tomb.write_text(json.dumps({"id": SG_OLD, "url": SG_URL, "reason": Q.SUPERSEDED_GENERATION}) + "\n")
    manifest = {"schema": Q.MANIFEST_SCHEMA, "kind": "content_pack", "pack_id": SG_PACK, "family": H.FAMILY,
                "collection": H.TWIN, "pack_points": str(pts),
                "pack_points_sha256": hashlib.sha256(pts.read_bytes()).hexdigest(), "rows": 0,
                "tombstones": str(tomb), "tombstones_count": 1,
                "tombstones_sha256": hashlib.sha256(tomb.read_bytes()).hexdigest(),
                "export": {"path": str(export), "rows": 1,
                           "sha256": hashlib.sha256(export.read_bytes()).hexdigest()}}
    body = json.dumps(manifest, indent=1, sort_keys=True).encode()
    (pack_dir / "manifest.json").write_bytes(body)
    (pack_dir / "READY").write_text(hashlib.sha256(body).hexdigest() + "\n")
    return queue, export


def _sg_boot(tmp, url, monkeypatch):
    from cytir.core import paths
    _sg_world(url)
    queue, export = _sg_queue(tmp)
    monkeypatch.setenv("AW_CHANGE_LOG_DIR", str(tmp / "changelog"))
    paths.reset_settings_cache()
    H.install_fakes(monkeypatch, Q, PSP)
    return queue, export, tmp / "artifacts"


def test_a_superseded_generation_pack_deletes_the_old_row_on_twin_then_prod(tmp_path, url2, monkeypatch):
    queue, _, artifacts = _sg_boot(tmp_path, url2, monkeypatch)
    rec = Recorder(monkeypatch)
    res = _pin_drain(queue, url2, artifacts)[0]
    assert res["status"] == "LANDED", res
    assert [e[:2] for e in rec.effects if e[0] == "delete"] == [["delete", "twin"], ["delete", "prod"]]
    assert rec.effects[0][0] == "upsert" and [e[0] for e in rec.effects] == ["upsert", "delete", "upsert", "delete"]
    assert res["twin"]["tombstoned"] == res["prod"]["tombstoned"] == 1
    for coll in (H.TWIN, H.PROD):
        assert _ids_on(url2, coll) == [SG_NEW]                      # the newer generation stays
    assert (queue / SG_PACK / "DONE").exists()


def test_a_superseded_generation_export_is_reverified_before_the_prod_leg(tmp_path, url2, monkeypatch):
    """The export is checked by the dry check at the start; the PROD leg checks it AGAIN under the lock
    (a replay from a twin-write record never ran the dry check). Here the export file changes while the
    twin leg runs: the prod leg must refuse before it deletes anything."""
    queue, export, artifacts = _sg_boot(tmp_path, url2, monkeypatch)
    rec = Recorder(monkeypatch)
    inner = Q.PD.delete_points_verified

    def dele(u, coll, ids, **kw):
        out = inner(u, coll, ids, **kw)
        if coll == H.TWIN:
            export.write_bytes(export.read_bytes() + b"tampered")        # the sha256 no longer matches
        return out
    monkeypatch.setattr(Q.PD, "delete_points_verified", dele)
    with pytest.raises(BaseException) as ei:
        _pin_drain(queue, url2, artifacts)
    assert "sha256" in str(ei.value)
    assert [e[:2] for e in rec.effects if e[0] == "delete"] == [["delete", "twin"]]
    assert _ids_on(url2, H.PROD) == sorted([SG_NEW, SG_OLD])        # prod untouched
    assert not (queue / SG_PACK / "DONE").exists()


def test_a_pack_with_no_superseded_generation_tombstone_does_not_reverify_an_export(tmp_path, url2, monkeypatch):
    first = json.loads(POINTS.read_text().splitlines()[0])["payload"]["url"]
    queue, artifacts, _ = _pin_boot(tmp_path, url2, monkeypatch, tombstone_url=first, old_row_url=first)
    called: list = []
    monkeypatch.setattr(Q, "verify_removed_rows_export", lambda *a, **k: called.append(1))
    assert _pin_drain(queue, url2, artifacts)[0]["status"] == "LANDED"
    assert called == []


def test_the_side_receipt_separates_the_tombstones_that_were_there_from_those_already_gone(url2):
    """`apply_content_pack_side` on real Qdrant: of two tombstoned ids one is stored, one is not. The receipt
    says 2 requested, 1 verified gone, 1 not found (absent before), and the stored row really is deleted."""
    H.reset_collection(url2, H.TWIN)
    H.api(url2, f"/collections/{H.TWIN}/points?wait=true", {"points": [_old_row("https://example.invalid/old")]})
    gone = "00000000-0000-5000-8000-0000000000c2"
    rec = Q.apply_content_pack_side("twin", url2, H.TWIN, [], [{"id": OLD_ROW}, {"id": gone}], batch=2, pause_s=0)
    assert (rec["role"], rec["written"], rec["tombstoned"]) == ("twin", 0, 1)
    assert (rec["tombstones_requested"], rec["tombstones_not_found"]) == (2, 1)
    assert _ids_on(url2, H.TWIN) == []


def test_each_leg_writes_to_its_own_endpoint(tmp_path, url2, monkeypatch):
    """Twin and prod are told apart by the url string here (the same throwaway server answers both): the
    twin leg writes to `twin_url`, the prod leg to `prod_url`."""
    first = json.loads(POINTS.read_text().splitlines()[0])["payload"]["url"]
    queue, artifacts, _ = _pin_boot(tmp_path, url2, monkeypatch, tombstone_url=first, old_row_url=first)
    sent: list = []
    inner = PSP.upsert

    def up(u, coll, pts, batch, pause_s=0.0, **k):
        sent.append((coll, u))
        return inner(u, coll, pts, batch, pause_s, **k)
    monkeypatch.setattr(PSP, "upsert", up)
    prod_url = url2 + "/"
    res = Q.drain(queue, twin_url=url2, prod_url=prod_url, task=PINS_TASK, apply=True, batch=3, pause_s=0,
                  artifact_root=artifacts, refresh_index=False, prod_alias=H.ALIAS)[0]
    assert res["status"] == "LANDED", res
    assert sorted(set(sent)) == [(H.TWIN, url2), (H.PROD, prod_url)]


# ---------------------------------------------------------------- task 2926: pins before the resume code moves
BEHIND = "cp-zz-batch-00001"


def _sg_two_packs(tmp, url, monkeypatch):
    """The `superseded_generation` pack of `_sg_boot` plus an ordinary pack (the four fixture rows) behind it."""
    queue, export, artifacts = _sg_boot(tmp, url, monkeypatch)
    _queue(tmp)                       # adds PACK_ID to the same `queue` dir
    # ... renamed so that it sorts AFTER the sg pack (strict name order within a family)
    old, new = queue / PACK_ID, queue / BEHIND
    old.rename(new)
    man = new / "manifest.json"
    body = man.read_text().replace(str(old), str(new)).replace(PACK_ID, BEHIND).encode()
    man.write_bytes(body)
    (new / "READY").write_text(hashlib.sha256(body).hexdigest() + "\n")
    return queue, export, artifacts


def _after_a_prod_leg_that_died(tmp, url, monkeypatch):
    """Pass 1 over the sg pack ALONE: the twin leg lands (state `twin-write-complete`), the prod upsert dies."""
    queue, export, artifacts = _sg_two_packs(tmp, url, monkeypatch)
    Recorder(monkeypatch, fail_on=H.PROD)
    with pytest.raises(KeyboardInterrupt):
        _pin_drain(queue, url, artifacts, only=[SG_PACK])
    state = json.loads((artifacts / f"{SG_PACK}-twin-state.json").read_text()) \
        if (artifacts / f"{SG_PACK}-twin-state.json").exists() else None
    return queue, export, artifacts, state


def test_p3b_a_content_pack_prod_only_resume_that_refuses_for_good_stops_the_drain(tmp_path, url2, monkeypatch):
    """P3b (trace 2912, policy 2): the OTHER policy of the same state. The sg pack sits at `twin-write-complete`;
    its export is damaged, so the prod-only resume refuses at the re-verify every time. The drain STOPS: the
    ordinary pack behind it does not land in that pass, and prod is untouched. Once the export is restored a
    later pass lands both (the healthy pack was only waiting)."""
    from cytir.index.drain import common as DC
    monkeypatch.setattr(DC, "_RESUME_RETRY_BACKOFF_S", (0, 0, 0))
    monkeypatch.setattr(Q, "_RESUME_RETRY_BACKOFF_S", (0, 0, 0))
    queue, export, artifacts, _ = _after_a_prod_leg_that_died(tmp_path, url2, monkeypatch)
    assert _ids_on(url2, H.TWIN) == [SG_NEW] and _ids_on(url2, H.PROD) == sorted([SG_NEW, SG_OLD])
    good = export.read_bytes()
    export.write_bytes(good + b"tampered")                               # persistent: the sha256 never matches
    resumed = []
    real_resume = Q.resume_content_pack_prod_only
    monkeypatch.setattr(Q, "resume_content_pack_prod_only", lambda **k: resumed.append(k["pack_id"]) or real_resume(**k))
    for _ in range(2):                                                    # every pass: the same outcome
        with pytest.raises(BaseException) as ei:
            _pin_drain(queue, url2, artifacts)
        assert "sha256" in str(ei.value)
        assert not (queue / BEHIND / "DONE").exists() and not (queue / SG_PACK / "DONE").exists()
        assert not (queue / BEHIND / "receipt.json").exists()          # the healthy pack was never reached
        assert _ids_on(url2, H.PROD) == sorted([SG_NEW, SG_OLD])        # prod untouched
    assert resumed == [SG_PACK, SG_PACK]                                 # a prod-only resume, never a re-plan
    export.write_bytes(good)
    res = {r["pack_id"]: r for r in _pin_drain(queue, url2, artifacts)}
    assert res[SG_PACK]["status"] == res[BEHIND]["status"] == "LANDED"
    assert (queue / BEHIND / "DONE").exists() and (queue / SG_PACK / "DONE").exists()


def _p6_boot(tmp_path, url2, monkeypatch):
    """A kill inside the twin leg of the sg pack: state `twin-write-started`, the twin delete never ran."""
    queue, export, artifacts = _sg_boot(tmp_path, url2, monkeypatch)
    real_del = Q.PD.delete_points_verified
    state = {"die": True}

    def dele(u, coll, ids, **kw):
        if state["die"] and coll == H.TWIN:
            state["die"] = False
            raise KeyboardInterrupt("killed inside the twin leg")
        return real_del(u, coll, ids, **kw)
    monkeypatch.setattr(Q.PD, "delete_points_verified", dele)
    with pytest.raises(KeyboardInterrupt):
        _pin_drain(queue, url2, artifacts)
    st = json.loads(next(artifacts.glob(f"{SG_PACK}*twin*state*.json")).read_text())
    assert st["stage"] == "twin-write-started"
    assert _ids_on(url2, H.TWIN) == sorted([SG_NEW, SG_OLD])             # the twin delete never ran
    seen: list = []
    resumed: list = []
    real_re = Q.UGS.reverify_generation_export
    monkeypatch.setattr(Q.UGS, "reverify_generation_export",
                        lambda *a, **k: seen.append("reverify") or real_re(*a, **k))
    real_resume = Q.resume_content_pack_prod_only
    monkeypatch.setattr(Q, "resume_content_pack_prod_only", lambda **k: resumed.append(1) or real_resume(**k))
    monkeypatch.setattr(Q.PD, "delete_points_verified",
                        lambda u, coll, ids, **kw: seen.append(("delete", coll)) or real_del(u, coll, ids, **kw))
    return queue, export, artifacts, seen, resumed


def test_p6_replay_after_a_twin_write_started_record_replays_both_legs_and_reverifies_before_the_prod_delete(
        tmp_path, url2, monkeypatch):
    """P6a: a `twin-write-started` record of a `superseded_generation` pack is REPLAYED, not resumed prod-only
    (the dry check is skipped): the twin delete runs again, then the prod leg re-verifies the export BEFORE its
    delete, and lands."""
    queue, export, artifacts, seen, resumed = _p6_boot(tmp_path, url2, monkeypatch)
    res = _pin_drain(queue, url2, artifacts)[0]
    assert res["status"] == "LANDED" and resumed == []
    assert seen == [("delete", H.TWIN), "reverify", ("delete", H.PROD)]
    assert _ids_on(url2, H.TWIN) == _ids_on(url2, H.PROD) == [SG_NEW]
    assert (queue / SG_PACK / "DONE").exists()


def test_p6_replay_with_a_damaged_export_runs_the_twin_leg_and_stops_the_prod_leg_then_resumes_prod_only(
        tmp_path, url2, monkeypatch):
    """P6b: the export is damaged between the passes. The replay's TWIN leg still runs (nothing re-verifies the
    export on a replay's twin leg), the PROD leg refuses at the re-verify, prod is untouched, the drain stops. The
    twin leg verified, so the state moved to `twin-write-complete`: after the export is restored the next pass is a
    PROD-ONLY RESUME (one call), which re-verifies the export again before the prod delete, and lands."""
    queue, export, artifacts, seen, resumed = _p6_boot(tmp_path, url2, monkeypatch)
    good = export.read_bytes()
    export.write_bytes(good + b"tampered")
    with pytest.raises(BaseException) as ei:
        _pin_drain(queue, url2, artifacts)
    assert "sha256" in str(ei.value) and resumed == []
    assert seen == [("delete", H.TWIN), "reverify"]
    assert _ids_on(url2, H.TWIN) == [SG_NEW] and _ids_on(url2, H.PROD) == sorted([SG_NEW, SG_OLD])
    assert not (queue / SG_PACK / "DONE").exists()
    st = json.loads(next(artifacts.glob(f"{SG_PACK}*twin*state*.json")).read_text())
    assert st["stage"] == "twin-write-complete"
    export.write_bytes(good)
    seen.clear()
    res = _pin_drain(queue, url2, artifacts)[0]
    assert res["status"] == "LANDED" and resumed == [1]
    assert seen == ["reverify", ("delete", H.PROD)]
    assert _ids_on(url2, H.TWIN) == _ids_on(url2, H.PROD) == [SG_NEW]


# ---------------------------------------------------------------- task 2929: pins before the resume code moves
# `resume_content_pack_prod_only` moves to `cytir/index/drain/resume.py` at step 13 (see the note in the resume golden).
def _ledger(artifacts, pack_id):
    rows = [json.loads(x) for x in (artifacts / f"{pack_id}-resume-ledger.jsonl").read_text().splitlines()]
    for r in rows:
        r.pop("ts")
    return rows


class _Up:
    """`PSP.upsert` wrapper for the resume pins: records `(collection, batch, pause_s, checkpoint path, retry-leg depth)`
    and passes on with pause 0. `die`: the first PROD upsert dies (a crash before the prod leg wrote)."""

    def __init__(self, monkeypatch, die=False):
        self.calls, self.die = [], die
        real = _REAL_UPSERT

        def up(u, coll, pts, batch, pause_s=0.0, **k):
            ck = k.get("checkpoint")
            self.calls.append((coll, batch, pause_s, None if ck is None else ck.path, PSP._LEG["depth"]))
            if self.die and coll == H.PROD:
                self.die = False
                raise KeyboardInterrupt("simulated crash before the prod leg wrote")
            return real(u, coll, pts, batch, 0.0, **k)
        monkeypatch.setattr(PSP, "upsert", up)


def _spies(monkeypatch):
    """The runner's pid, the writer lock's arguments and the layout fingerprints the resume asks for."""
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
    real_fp = Q.LAYOUT.fingerprint

    def fp(u, c, **k):
        out = real_fp(u, c, **k)
        seen["fp"].append((u, c, out))
        return out
    monkeypatch.setattr(Q.LAYOUT, "fingerprint", fp)
    return seen


def _cp_first_run(tmp_path, url2, monkeypatch, *, big=False):
    """Pass 1: the twin leg lands (rows, one tombstone, two rows already on prod), the prod upsert dies. State
    `twin-write-complete`."""
    from cytir.index import prod_guard as PG
    if big:
        monkeypatch.setattr(PG, "PROD_BIG_PACK_ROWS", 3)
        monkeypatch.setattr(PG, "PROD_BIG_PACK_PAUSE_S", 0.0)
    first = json.loads(POINTS.read_text().splitlines()[0])["payload"]["url"]
    queue, artifacts, locks = _pin_boot(tmp_path, url2, monkeypatch, tombstone_url=first, old_row_url=first)
    _seed_inplace(url2, _points())
    up = _Up(monkeypatch, die=True)
    with pytest.raises(KeyboardInterrupt):
        _pin_drain(queue, url2, artifacts, pause_s=0.25)
    del locks[:]
    up.calls.clear()
    return queue, artifacts, locks, up


def _cp_resume_args(queue, artifacts, url, **over):
    mpath = queue / PACK_ID / "manifest.json"
    sha = hashlib.sha256(mpath.read_bytes()).hexdigest()
    kw = dict(pack_id=PACK_ID, manifest=json.loads(mpath.read_text()), twin_url=url, prod_url=url, task=PINS_TASK,
              batch=3, pause_s=0, artifact_root=artifacts, prod_collection=H.PROD, prod_alias=H.ALIAS,
              state_record=Q.read_content_pack_twin_state(artifacts, PACK_ID, manifest_sha=sha))
    kw.update(over)
    return kw


def _cp_locks(*, prod_lock=True):
    p = f"{PACK_ID}-promotion-queue-resume"
    seq = [("assert_locks_free", PINS_TASK), ("acquire", "TWIN-WRITE-LOCK", PINS_TASK, p),
           ("acquire", "PROD-WRITE-LOCK", PINS_TASK, p)]
    if prod_lock:
        seq += ["prod-lock-acquire", "prod-lock-release"]
    return seq + [("release", "PROD-WRITE-LOCK", PINS_TASK), ("release", "TWIN-WRITE-LOCK", PINS_TASK)]


def test_2929_content_resume_collaborators_ledger_and_receipt(tmp_path, url2, monkeypatch):
    queue, artifacts, locks, up = _cp_first_run(tmp_path, url2, monkeypatch)
    seen = _spies(monkeypatch)
    forms = []
    real_form = Q.verify_written_form
    monkeypatch.setattr(Q, "verify_written_form", lambda *a, **k: forms.append((a, k)) or real_form(*a, **k))
    res = _pin_drain(queue, url2, artifacts, pause_s=0.25)[0]
    assert res["status"] == "LANDED" and res["note"] == "resumed prod-only from a twin-write-complete state record"
    assert (res["prod"]["written"], res["prod"]["tombstoned"]) == (4, 1) and "prod_calls" not in res["prod"]
    assert locks == _cp_locks()
    assert seen["pids"] == [("TWIN-WRITE-LOCK", os.getpid()), ("PROD-WRITE-LOCK", os.getpid())]
    assert seen["lock"] == [{"host": "lockhost", "lock_path": Q.PROD_LOCK_PATH,
                             "purpose": f"task{PINS_TASK}-{PACK_ID}-resume"}]
    assert [c for _, c, _ in seen["fp"]] == [H.PROD] * 3 and {u for u, _, _ in seen["fp"]} == {url2}
    assert up.calls == [(H.PROD, 3, 0.25, None, 2)]                 # batch, pause, no checkpoint, two legs deep
    (fa, fk), = forms
    assert fa[:2] == (url2, H.PROD) and len(fa[2]) == 4 and fk["twin"] == (url2, H.TWIN)
    assert sorted(fk["inplace_ids"]) == sorted(p["id"] for p in _points()[:2])
    led = _ledger(artifacts, PACK_ID)
    assert [x["status"] for x in led] == ["RESUME-PREFLIGHT-OK", "PRESERVATION-VERIFIED", "RESUME-PREDECESSOR-BOUND",
                                          "TWIN-GATE-PASS", "PROD-VERIFIED", "TWIN-PROD-VERIFIED-OK"]
    counts = {"pack_ids_count": 4, "tomb_ids_count": 1}
    assert led[0] == {"status": "RESUME-PREFLIGHT-OK", "armed": True, "proof": counts}
    assert led[1]["proof"] == {"status": "PASS", "pack_id": PACK_ID, "scope": (
        "narrow: RESUME of a content pack's prod leg only (upsert + tombstone-delete); twin was already written "
        "and verified by a prior run and is never rewritten here")}
    assert led[2] == {"status": "RESUME-PREDECESSOR-BOUND", "proof": counts}
    assert led[3]["proof"] == {"status": "PASS", "mechanical": True, "pack_id": PACK_ID, "note": (
        "twin already verified by the ORIGINAL apply; re-confirmed present/absent by this resume's own preflight")}
    assert led[4]["proof"] == {"status": "PASS", "role": "prod", "written": 4, "tombstoned": 1, "payload_form": {
        "checked": 4, "collection": H.PROD, "compared": "all payload keys + dense", "inplace_checked": 2,
        "inplace_source": "prod", "inplace_total": 2, "mismatches": 0, "profile": "normalized", "twin_checked": 4}}
    assert led[5] == {"status": "TWIN-PROD-VERIFIED-OK",
                      "recovery": "resumed from immutable failed predecessor after verified twin",
                      "rollback": "artifact and target readback proofs retained; no destructive automatic rollback"}
    assert _ids_on(url2, H.PROD) == _ids_on(url2, H.TWIN) and OLD_ROW not in _ids_on(url2, H.PROD)


def test_2929_a_big_content_pack_resume_writes_with_a_chunk_checkpoint_named_for_the_manifest(
        tmp_path, url2, monkeypatch):
    queue, artifacts, locks, up = _cp_first_run(tmp_path, url2, monkeypatch, big=True)
    res = _pin_drain(queue, url2, artifacts, pause_s=0.25)[0]
    sha = hashlib.sha256((queue / PACK_ID / "manifest.json").read_bytes()).hexdigest()
    (coll, batch, pause, ck, depth), = up.calls
    assert (coll, batch, depth) == (H.PROD, 3, 2) and ck == artifacts / f"{PACK_ID}-{sha[:12]}-prod-write.chunks.jsonl"
    assert res["status"] == "LANDED" and "prod_calls" in res["prod"]
    assert not list(artifacts.glob("*.chunks.jsonl"))                # spent by the landing
    assert _ids_on(url2, H.PROD) == _ids_on(url2, H.TWIN)


def test_2929_the_content_resume_only_resumes_a_twin_write_complete_record(tmp_path, url2, monkeypatch):
    queue, artifacts, locks, up = _cp_first_run(tmp_path, url2, monkeypatch)
    args = _cp_resume_args(queue, artifacts, url2)
    assert args["state_record"]["stage"] == "twin-write-complete"
    started = {**args["state_record"], "stage": "twin-write-started"}
    with pytest.raises(Q.QueueError, match="requires a 'twin-write-complete' state record"):
        Q.resume_content_pack_prod_only(**{**args, "state_record": started})
    assert locks == [] and up.calls == [] and _ids_on(url2, H.PROD) != _ids_on(url2, H.TWIN)


def test_2929_the_content_resume_takes_the_attempt_and_lands_prod_only_when_called_directly(tmp_path, url2, monkeypatch):
    queue, artifacts, locks, up = _cp_first_run(tmp_path, url2, monkeypatch)
    wa = Q.WriteAttempt()
    state = Q.resume_content_pack_prod_only(**_cp_resume_args(queue, artifacts, url2, write_attempt=wa))
    assert wa.started is True and state["prod_receipt"]["written"] == 4
    assert state["prod_before"]["collection"] == H.PROD and state["prod_before"]["points_count"] == 3
    assert locks == _cp_locks() and [c[0] for c in up.calls] == [H.PROD]
    assert _ids_on(url2, H.PROD) == _ids_on(url2, H.TWIN) and OLD_ROW not in _ids_on(url2, H.PROD)


@pytest.mark.parametrize("what", ["a pack row", "the tombstoned row"])
def test_2929_the_content_resume_premise_is_the_twin_state_before_any_prod_write(tmp_path, url2, monkeypatch, what):
    queue, artifacts, locks, up = _cp_first_run(tmp_path, url2, monkeypatch)
    if what == "a pack row":
        H.api(url2, f"/collections/{H.TWIN}/points/delete?wait=true", {"points": [_points()[3]["id"]]}, method="POST")
        match = r"resume premise failed -- 1 pack id\(s\) the twin leg reported written are no longer present"
    else:
        H.api(url2, f"/collections/{H.TWIN}/points?wait=true", {"points": [_old_row("https://example.invalid/old")]})
        match = r"resume premise failed -- 1 tombstone id\(s\) the twin leg reported deleted are still present"
    before = _ids_on(url2, H.PROD)
    with pytest.raises(Q.QueueError, match=match):
        Q.resume_content_pack_prod_only(**_cp_resume_args(queue, artifacts, url2))
    assert up.calls == [] and _ids_on(url2, H.PROD) == before
    assert locks == _cp_locks(prod_lock=False)                  # no prod lock taken, both runner locks given back
    assert not (artifacts / f"{PACK_ID}-resume-ledger.jsonl").exists()


def test_2929_the_alias_moved_before_the_content_resume_refuses_before_the_prod_write(tmp_path, url2, monkeypatch):
    queue, artifacts, locks, up = _cp_first_run(tmp_path, url2, monkeypatch)
    H.set_alias(url2, H.ALIAS, MOVED)
    before = _ids_on(url2, H.PROD)
    with pytest.raises(BaseException, match="moved mid-drain"):
        Q.resume_content_pack_prod_only(**_cp_resume_args(queue, artifacts, url2))
    assert up.calls == [] and _ids_on(url2, H.PROD) == before and _ids_on(url2, MOVED) == []
    assert _ledger(artifacts, PACK_ID)[-1]["status"] == "FAILED"


def _cp_fault_after_the_prod_write(tmp_path, url2, monkeypatch, *, match, on_present):
    """The content resume's prod upsert and tombstone delete land; the NEXT prod lookups are its own verify's:
    the first asks for the pack ids, the second for the tombstoned id. One of them is made to lie. The drain
    stops (ABORTED, no DONE), the ledger records the failure, both runner locks come back in reverse order."""
    queue, artifacts, locks, up = _cp_first_run(tmp_path, url2, monkeypatch)
    st = {"written": False, "n": 0}
    real_del, real_present = Q.PD.delete_points_verified, PSP.already_present

    def dele(u, coll, ids, **kw):
        out = real_del(u, coll, ids, **kw)
        st["written"] = st["written"] or coll == H.PROD
        return out

    def present(u, coll, ids, *a, **k):
        got = real_present(u, coll, ids, *a, **k)
        if st["written"] and coll == H.PROD:
            st["n"] += 1
            return on_present(st["n"], got)
        return got
    monkeypatch.setattr(Q.PD, "delete_points_verified", dele)
    monkeypatch.setattr(PSP, "already_present", present)
    with pytest.raises(BaseException, match=match):
        _pin_drain(queue, url2, artifacts, pause_s=0.25)
    rec = json.loads((queue / PACK_ID / "receipt.json").read_text())
    assert rec["status"] == "ABORTED" and re.search(match, rec["reason"])
    assert not (queue / PACK_ID / "DONE").exists()
    assert locks[-2:] == [("release", "PROD-WRITE-LOCK", PINS_TASK), ("release", "TWIN-WRITE-LOCK", PINS_TASK)]
    assert _ledger(artifacts, PACK_ID)[-1]["status"] == "FAILED"


def test_2929_a_prod_upsert_missing_from_the_content_resume_membership_proof_aborts(tmp_path, url2, monkeypatch):
    _cp_fault_after_the_prod_write(tmp_path, url2, monkeypatch, match=r"1 pack id\(s\) missing after resume upsert",
                                   on_present=lambda n, got: sorted(got)[1:] if n == 1 else got)


def test_2929_a_tombstoned_row_still_on_prod_after_the_content_resume_aborts(tmp_path, url2, monkeypatch):
    _cp_fault_after_the_prod_write(tmp_path, url2, monkeypatch, match=r"1 tombstoned id\(s\) still present after resume",
                                   on_present=lambda n, got: [OLD_ROW] if n == 2 else got)


def test_2929_the_alias_moving_after_the_content_resume_write_is_the_last_check_before_done(tmp_path, url2, monkeypatch):
    _cp_fault_after_the_prod_write(
        tmp_path, url2, monkeypatch, match=r"moved mid-drain",
        on_present=lambda n, got: (H.set_alias(url2, H.ALIAS, MOVED), got)[1] if n == 1 else got)


def test_2929_each_leg_of_the_content_resume_talks_to_its_own_endpoint(tmp_path, url2, monkeypatch):
    queue, artifacts, locks, up = _cp_first_run(tmp_path, url2, monkeypatch)
    sent: list = []
    real, real_fp, real_del = PSP._api, Q.LAYOUT.fingerprint, Q.PD.delete_points_verified

    def api(u, path, body=None, method=None, *a, **k):
        m = re.match(r"/collections/([^/]+)", path)
        if m:
            sent.append((u, m.group(1)))
        return real(u, path, body, method, *a, **k)

    def fp(u, c, **k):
        sent.append((u, c))
        return real_fp(u, c, **k)

    def dele(u, c, ids, **k):
        sent.append((u, c))
        return real_del(u, c, ids, **k)
    monkeypatch.setattr(PSP, "_api", api)
    monkeypatch.setattr(Q.LAYOUT, "fingerprint", fp)
    monkeypatch.setattr(Q.PD, "delete_points_verified", dele)
    prod_url = url2 + "/"
    Q.resume_content_pack_prod_only(**_cp_resume_args(queue, artifacts, url2, prod_url=prod_url))
    assert set(sent) == {(url2, H.TWIN), (prod_url, H.PROD)} and (prod_url, H.PROD) in sent


def test_2929_a_transient_error_in_the_content_resume_is_retried_by_the_drain(tmp_path, url2, monkeypatch):
    """The drain runs the prod-only content resume through `_retry_resume`: a transport error (here a refused
    connection before the leg starts) is retried and the pack lands; the backoff is zeroed so nothing sleeps."""
    from cytir.index.drain import common as DC
    monkeypatch.setattr(DC, "_RESUME_RETRY_BACKOFF_S", (0, 0, 0))
    queue, artifacts, locks, up = _cp_first_run(tmp_path, url2, monkeypatch)
    real, calls = Q.resume_content_pack_prod_only, []

    def flaky(**k):
        calls.append(k["pack_id"])
        if len(calls) == 1:
            raise ConnectionError("blip")
        return real(**k)
    monkeypatch.setattr(Q, "resume_content_pack_prod_only", flaky)
    res = _pin_drain(queue, url2, artifacts, pause_s=0.25)[0]
    assert res["status"] == "LANDED" and calls == [PACK_ID, PACK_ID]
    assert (queue / PACK_ID / "DONE").exists() and _ids_on(url2, H.PROD) == _ids_on(url2, H.TWIN)


# ---- task 2938: the budget decisions of the content-pack path, as END STATE -------------------------------

def _budget_args(*, spent_s, budget_s=100.0, margin_s=1.0):
    """drain() kwargs: a `budget_s` window of which `spent_s` is already gone."""
    import time
    return dict(budget_seconds=budget_s, started_at_epoch=time.time() - spent_s, stop_margin_s=margin_s)


def _ops(queue, pack, legs):
    man = json.loads((queue / pack / "manifest.json").read_text())
    return legs * ((man.get("rows") or 0) + (man.get("tombstones_count") or 0))


def test_2938_a_pending_resume_that_does_not_fit_the_rest_of_the_budget_stops_the_drain(tmp_path, url2, monkeypatch):
    """B6: the sg pack has a pending prod-only resume (`twin-write-complete`). 80 s of a 100 s budget are spent;
    the resume is estimated at 50 s (fits a FRESH window, not the rest of this one). The drain STOPS: no result,
    no receipt for either pack, the ordinary pack behind it is not landed (it would otherwise land over a known
    twin != prod gap), prod untouched. A pass with the whole budget resumes and lands both."""
    queue, export, artifacts, state = _after_a_prod_leg_that_died(tmp_path, url2, monkeypatch)
    sha = hashlib.sha256((queue / SG_PACK / "manifest.json").read_bytes()).hexdigest()
    assert Q.read_content_pack_twin_state(artifacts, SG_PACK, manifest_sha=sha)["stage"] == "twin-write-complete"
    prod_before = _ids_on(url2, H.PROD)
    aborted = (queue / SG_PACK / "receipt.json").read_bytes()          # the first pass's ABORTED receipt
    monkeypatch.setattr(Q, "ESTIMATED_CONTENT_PACK_OPS_PER_SECOND", _ops(queue, SG_PACK, 1) / 50.0)
    Recorder(monkeypatch)                                  # the healthy fakes again (the prod upsert no longer dies)
    res = _pin_drain(queue, url2, artifacts, **_budget_args(spent_s=80))
    assert res == []                                        # stopped: nothing skipped, nothing recorded
    assert not (queue / BEHIND / "receipt.json").exists() and not (queue / BEHIND / "DONE").exists()
    assert not (queue / SG_PACK / "DONE").exists() and (queue / SG_PACK / "receipt.json").read_bytes() == aborted
    assert _ids_on(url2, H.PROD) == prod_before
    monkeypatch.setattr(Q, "ESTIMATED_CONTENT_PACK_OPS_PER_SECOND", 150.0)
    res = {r["pack_id"]: r for r in _pin_drain(queue, url2, artifacts, **_budget_args(spent_s=0))}
    assert res[SG_PACK]["status"] == res[BEHIND]["status"] == "LANDED"
    assert (queue / SG_PACK / "DONE").exists() and (queue / BEHIND / "DONE").exists()


def test_2938_a_pending_resume_that_can_never_fit_leaves_a_loud_receipt_and_stops(tmp_path, url2, monkeypatch):
    """B6/B5: the same pending resume estimated at 500 s against a 100 s budget can never fit any run. The drain
    stops AND leaves a `BUDGET-EXCEEDS-TIMEOUT` receipt that names the resume stage, on disk and in the results."""
    queue, export, artifacts, _ = _after_a_prod_leg_that_died(tmp_path, url2, monkeypatch)
    monkeypatch.setattr(Q, "ESTIMATED_CONTENT_PACK_OPS_PER_SECOND", _ops(queue, SG_PACK, 1) / 500.0)
    Recorder(monkeypatch)
    res = _pin_drain(queue, url2, artifacts, **_budget_args(spent_s=0))
    assert [r["status"] for r in res] == ["BUDGET-EXCEEDS-TIMEOUT"]
    got = json.loads((queue / SG_PACK / "receipt.json").read_text())
    assert got["status"] == "BUDGET-EXCEEDS-TIMEOUT" and got["resume_stage"] == "twin-write-complete"
    assert got["budget_seconds"] == 100.0 and got["estimated_seconds"] == pytest.approx(500.0)
    assert not (queue / BEHIND / "receipt.json").exists() and not (queue / SG_PACK / "DONE").exists()


def test_2938_a_fresh_pack_that_does_not_fit_this_run_is_skipped_and_the_drain_goes_on(tmp_path, url2, monkeypatch):
    """B5: the big ordinary pack (estimated 50 s) with 30 s left of a 100 s budget: `SKIPPED-BUDGET` (the name the
    digest reads) carrying `estimated_seconds`, no receipt file, no DONE, still READY; the smaller sg pack behind
    it (named second via `only`) still lands in the same pass. A fresh pack estimated above the whole window gets
    the loud `BUDGET-EXCEEDS-TIMEOUT` receipt instead."""
    queue, export, artifacts = _sg_two_packs(tmp_path, url2, monkeypatch)
    big_ops, small_ops = _ops(queue, BEHIND, 2), _ops(queue, SG_PACK, 2)
    assert small_ops * 50.0 / big_ops < 29 < 50, (big_ops, small_ops)     # the small pack fits what is left
    monkeypatch.setattr(Q, "ESTIMATED_CONTENT_PACK_OPS_PER_SECOND", big_ops / 50.0)
    Recorder(monkeypatch)
    res = _pin_drain(queue, url2, artifacts, only=[BEHIND, SG_PACK], **_budget_args(spent_s=70))
    by = {r["pack_id"]: r for r in res}
    assert by[BEHIND]["status"] == "SKIPPED-BUDGET" and by[BEHIND]["estimated_seconds"] == pytest.approx(50.0)
    assert not (queue / BEHIND / "receipt.json").exists() and not (queue / BEHIND / "DONE").exists()
    assert (queue / BEHIND / "READY").exists()
    assert by[SG_PACK]["status"] == "LANDED" and (queue / SG_PACK / "DONE").exists()
    monkeypatch.setattr(Q, "ESTIMATED_CONTENT_PACK_OPS_PER_SECOND", big_ops / 500.0)
    res = _pin_drain(queue, url2, artifacts, **_budget_args(spent_s=0))
    assert [r["status"] for r in res] == ["BUDGET-EXCEEDS-TIMEOUT"]
    assert json.loads((queue / BEHIND / "receipt.json").read_text())["status"] == "BUDGET-EXCEEDS-TIMEOUT"


def test_2938_a_content_pack_resume_is_not_applied_in_a_dry_run_or_under_a_blocked_gate(tmp_path, url2, monkeypatch):
    """C1: the sg pack has a pending prod-only resume. Without `--apply` the drain only PROVES: status
    PROVED-DRY-RUN with the resume note, prod untouched, no DONE, the resume never called. With `--apply` and a gate
    contract that is not `mechanical`: PROVED-BLOCKED, naming the gate. Then the real apply resumes and lands."""
    queue, export, artifacts, _ = _after_a_prod_leg_that_died(tmp_path, url2, monkeypatch)
    Recorder(monkeypatch)
    prod_before = _ids_on(url2, H.PROD)
    called = []
    real_resume = Q.resume_content_pack_prod_only
    monkeypatch.setattr(Q, "resume_content_pack_prod_only", lambda **k: called.append(1) or real_resume(**k))
    res = Q.drain(queue, twin_url=url2, prod_url=url2, task=PINS_TASK, apply=False, batch=3, pause_s=0,
                  artifact_root=artifacts, refresh_index=False, prod_alias=H.ALIAS, only=[SG_PACK])
    assert [r["status"] for r in res] == ["PROVED-DRY-RUN"] and res[0]["blocked_on"] == []
    assert "resume pending" in res[0]["note"] and "twin-write-complete" in res[0]["note"]
    battery = tmp_path / "gate-battery.json"
    battery.write_text(json.dumps({"schema": Q.GATE_CONTRACT_SCHEMA, "adr": "ADR-110", "pr": 302,
                                   "profile": "battery-required"}))
    res = Q.drain(queue, twin_url=url2, prod_url=url2, task=PINS_TASK, apply=True, batch=3, pause_s=0,
                  artifact_root=artifacts, refresh_index=False, prod_alias=H.ALIAS, only=[SG_PACK],
                  contract_path=battery)
    assert [r["status"] for r in res] == ["PROVED-BLOCKED"] and "battery-required" in res[0]["blocked_on"][0]
    assert not called and _ids_on(url2, H.PROD) == prod_before and not (queue / SG_PACK / "DONE").exists()
    res = _pin_drain(queue, url2, artifacts, only=[SG_PACK])
    assert called == [1] and res[0]["status"] == "LANDED" and (queue / SG_PACK / "DONE").exists()
    assert _ids_on(url2, H.PROD) == _ids_on(url2, H.TWIN)


def test_2938_a_content_pack_the_german_parts_guard_refuses_writes_nothing_and_is_retried_every_run(
        tmp_path, url2, monkeypatch):
    """C5: the guard's verdict (a fact about the writer interpreter of THIS run) refuses the content pack at stage
    `german_parts`: a REFUSED receipt carrying the reason, nothing on the twin or on prod, no state file, no DONE.
    The stage is not cached: the next run (guard passing) lands the pack."""
    queue, artifacts, _ = _pin_boot(tmp_path, url2, monkeypatch)
    twin_before, prod_before = _ids_on(url2, H.TWIN), _ids_on(url2, H.PROD)
    real = Q.GPG.guarded
    monkeypatch.setattr(Q.GPG, "guarded", lambda fn, pack_id, rows: ({"rows_checked": 1}, "German rows without parts"))
    res = _pin_drain(queue, url2, artifacts)
    assert [(r["status"], r["stage"], r["reason"]) for r in res] == [("REFUSED", "german_parts", "German rows without parts")]
    saved = json.loads((queue / PACK_ID / "receipt.json").read_text())
    assert saved["status"] == "REFUSED" and saved["stage"] == "german_parts"
    assert _ids_on(url2, H.TWIN) == twin_before and _ids_on(url2, H.PROD) == prod_before
    assert not (queue / PACK_ID / "DONE").exists() and not list(artifacts.glob("*twin-state*"))
    monkeypatch.setattr(Q.GPG, "guarded", real)
    res = _pin_drain(queue, url2, artifacts)
    assert [r["status"] for r in res] == ["LANDED"] and (queue / PACK_ID / "DONE").exists()


def test_2938_held_out_cards_of_an_earlier_run_are_dropped_when_the_content_check_refuses(tmp_path, url2, monkeypatch):
    """C7: the pack's `held-out-cards` file (left by a run that held cards out) is removed when the dry check
    REFUSES the pack (stage `content_pack_check`) on a live run: a refused pack holds no cards. A dry run (no
    `--apply`) keeps the file."""
    from cytir.index.drain.receipts import HELD_OUT_CARDS_FILE
    bad = "https://example.invalid/not-in-this-pack"
    queue, artifacts, _ = _pin_boot(tmp_path, url2, monkeypatch, tombstone_url=bad, old_row_url=bad)
    held = queue / PACK_ID / HELD_OUT_CARDS_FILE
    held.write_text('{"id": "x"}\n')
    res = Q.drain(queue, twin_url=url2, prod_url=url2, task=PINS_TASK, apply=False, batch=3, pause_s=0,
                  artifact_root=artifacts, refresh_index=False, prod_alias=H.ALIAS)
    assert res[0]["status"] == "REFUSED" and res[0]["stage"] == "content_pack_check" and held.exists()
    res = _pin_drain(queue, url2, artifacts)
    assert res[0]["status"] == "REFUSED" and res[0]["stage"] == "content_pack_check"
    assert not held.exists()
