"""`cyt dump` takes its delta base from the fingerprint ledger and advances it after signing
(task 2590, option B hooks; spec `delta-fingerprint-base.md` 5-6). The ledger is master-only."""
from __future__ import annotations

import json
import pathlib

import pytest
from qdrant_client.http import models as qmodels

import cytir.core.qdrant as Q
from cytir.core import paths
from cytir.core import serving_profile as SP
from cytir.distribute import apply as apply_mod
from cytir.distribute import dump, pack
from cytir.distribute import fingerprint_ledger as FL

DIM = SP.get(dump.DEFAULT_PROFILE).dense_dim
COLL = "aw_test"


def _vec(seed):
    v = [0.0] * DIM
    v[0], v[1] = 1.0, seed
    return v


def _point(i, seed=None):
    return qmodels.PointStruct(
        id=i, vector={"dense": _vec(seed if seed is not None else i / 10)},
        payload={"url": f"https://example.org/{i}", "title": f"T{i}", "lang": "en",
                 "source_type": "web_page", "license": "CC0-1.0", "trust_tier": 0.5,
                 "last_seen": "2026-09-20T00:00:00Z"})


@pytest.fixture
def env(tmp_path, monkeypatch):
    c = Q.qdrant_client(":memory:")
    c.create_collection(COLL, vectors_config={"dense": qmodels.VectorParams(
        size=DIM, distance=qmodels.Distance.COSINE)})
    c.upsert(COLL, points=[_point(i) for i in (1, 2, 3, 4)])
    monkeypatch.setattr(Q, "qdrant_client", lambda *a, **k: c)
    monkeypatch.delenv("AW_CHANGE_LOG_DIR", raising=False)
    monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("AW_FINGERPRINT_LEDGER_ROOT", str(tmp_path / "ledger"))
    paths.reset_settings_cache()
    (tmp_path / "packs").mkdir()
    yield c, tmp_path
    paths.reset_settings_cache()


def _ledger_off(monkeypatch):
    monkeypatch.setenv("AW_FINGERPRINT_LEDGER_ROOT", "")
    paths.reset_settings_cache()


def _dump(tmp, name, *extra, base=None, seq=None, reasons=None, fn=dump.cli):
    argv = ["--collection", COLL, "--out", str(tmp / "packs" / name), "--dump-id", name,
            "--no-change-log"]
    if base:
        argv += ["--delta-base-manifest", str(tmp / "packs" / base / "manifest.json"),
                 "--sequence", str(seq)]
    if reasons:
        (tmp / f"{name}-r.json").write_text(json.dumps(reasons))
        argv += ["--tombstone-reasons", str(tmp / f"{name}-r.json")]
    return fn([*argv, *extra])


def _manifest(tmp, name):
    return json.loads((tmp / "packs" / name / "manifest.json").read_text())


def _chain(env):
    c, tmp = env
    assert _dump(tmp, "full") == 0
    c.upsert(COLL, points=[_point(2, 0.77), _point(7)])
    c.delete(COLL, points_selector=qmodels.PointIdsList(points=[3]))
    assert _dump(tmp, "d1", base="full", seq=1, reasons={"3": "superseded"}) == 0
    c.upsert(COLL, points=[_point(7, 0.55), _point(3, 0.33)])
    c.delete(COLL, points_selector=qmodels.PointIdsList(points=[4]))
    assert _dump(tmp, "d2", base="d1", seq=2, reasons={"4": "aged_out"}) == 0


def test_three_hop_through_the_real_cli(env):
    c, tmp = env
    led = FL.Ledger(tmp / "ledger")
    assert _dump(tmp, "full") == 0
    full = _manifest(tmp, "full")
    st0 = led.verify()   # the manifest's fingerprints match the ledger
    assert (st0["sequence"], st0["count"], st0["sidecar_sha256"]) == (
        0, full["fingerprints"]["count"], full["fingerprints"]["sha256"])
    c.upsert(COLL, points=[_point(2, 0.77), _point(7)])
    c.delete(COLL, points_selector=qmodels.PointIdsList(points=[3]))
    assert _dump(tmp, "d1", base="full", seq=1, reasons={"3": "superseded"}) == 0
    assert led.verify()["sequence"] == 1
    c.upsert(COLL, points=[_point(7, 0.55), _point(3, 0.33)])
    c.delete(COLL, points_selector=qmodels.PointIdsList(points=[4]))
    # without the ledger this is refused: delta 1 carries no fingerprints block
    assert _dump(tmp, "d2", base="d1", seq=2, reasons={"4": "aged_out"}) == 0
    d2 = _manifest(tmp, "d2")
    ids = {r["id"] for r in FL.pack_rows(tmp / "packs" / "d2", d2)}
    assert ids == {"7", "3"} and FL.pack_tombstones(tmp / "packs" / "d2", d2) == ["4"]
    # delta 2's base was the fold over (full, d1); the ledger now equals the live collection
    live = {r["id"]: pack.row_fingerprint(r) for r in dump.scroll_rows(c, COLL)}
    assert dict(led.items()) == dict(sorted(live.items(), key=lambda kv: pack.point_id_key(kv[0])))
    assert led.verify()["pack_manifest_sha256"] == pack.sha256_file(str(tmp / "packs/d2/manifest.json"))
    FL.check_rebuild(tmp / "packs", led)
    assert "fingerprints" not in d2 and not (tmp / "packs/d2/fingerprints.tsv.gz").exists()


def test_ledger_unset_is_todays_behaviour(env, monkeypatch):
    c, tmp = env
    assert _dump(tmp, "full") == 0
    c.upsert(COLL, points=[_point(2, 0.77), _point(7)])
    c.delete(COLL, points_selector=qmodels.PointIdsList(points=[3]))
    assert _dump(tmp, "with", base="full", seq=1, reasons={"3": "superseded"}) == 0
    _ledger_off(monkeypatch)
    assert _dump(tmp, "with2", base="full", seq=1, reasons={"3": "superseded"}) == 0
    a, b = _manifest(tmp, "with"), _manifest(tmp, "with2")
    for m in (a, b):
        m.pop("created_at", None)
        m["dump_id"] = m["pack_id"] = "x"
    assert a == b
    # unset: no ledger file appears, and delta-on-delta is refused as before (no fingerprints block)
    before = sorted(p.name for p in (tmp / "ledger").iterdir())
    c.upsert(COLL, points=[_point(9)])
    assert _dump(tmp, "d2", base="with2", seq=2) == 1
    assert sorted(p.name for p in (tmp / "ledger").iterdir()) == before


def test_stale_ledger_refuses_and_writes_nothing(env, monkeypatch, capsys):
    c, tmp = env
    assert _dump(tmp, "full") == 0
    c.upsert(COLL, points=[_point(2, 0.77)])
    _ledger_off(monkeypatch)
    assert _dump(tmp, "d1", base="full", seq=1) == 0     # published, the ledger did not move
    monkeypatch.setenv("AW_FINGERPRINT_LEDGER_ROOT", str(tmp / "ledger"))
    paths.reset_settings_cache()
    c.upsert(COLL, points=[_point(3, 0.66)])
    assert _dump(tmp, "d2", base="d1", seq=2) == 1
    assert "REFUSED" in capsys.readouterr().err
    assert not (tmp / "packs/d2/manifest.json").exists()
    assert list((tmp / "packs/d2").iterdir()) == []


def test_missing_ledger_refuses_a_delta(env):
    c, tmp = env
    assert _dump(tmp, "full") == 0
    (tmp / "ledger/state.json").unlink()
    assert _dump(tmp, "d1", base="full", seq=1) == 1
    assert not (tmp / "packs/d1/manifest.json").exists()


def test_wrong_sequence_refuses(env):
    c, tmp = env
    assert _dump(tmp, "full") == 0
    assert _dump(tmp, "d1", base="full", seq=2) == 1   # the ledger is at 0, so seq 1 is next


def test_advance_failure_keeps_the_signed_pack_and_exits_non_zero(env, monkeypatch, capsys):
    c, tmp = env
    assert _dump(tmp, "full") == 0
    c.upsert(COLL, points=[_point(2, 0.77)])

    def boom(ledger, d):
        raise FL.LedgerCorrupt("boom")
    monkeypatch.setattr(FL, "advance_from_pack_dir", boom)
    assert _dump(tmp, "d1", base="full", seq=1) == 3
    err = capsys.readouterr().err
    assert "LEDGER LAGGING" in err and "rebuild_by_fold" in err
    assert (tmp / "packs/d1/manifest.json").exists()      # the pack stays
    monkeypatch.undo()
    # the ledger lags: the next delta refuses; rebuild_by_fold repairs
    st = FL.Ledger(tmp / "ledger").verify()
    assert st["sequence"] == 0
    FL.rebuild_by_fold(tmp / "packs", FL.Ledger(tmp / "ledger"))
    assert FL.Ledger(tmp / "ledger").verify()["sequence"] == 1


def test_a_second_full_rebases_the_ledger(env):
    c, tmp = env
    _chain(env)
    c.upsert(COLL, points=[_point(20)])
    assert _dump(tmp, "full2") == 0
    st = FL.Ledger(tmp / "ledger").verify()
    assert st["sequence"] == 0 and st["pack_manifest_sha256"] == pack.sha256_file(
        str(tmp / "packs/full2/manifest.json"))


def test_cumulative_pack_is_refused_by_apply(env, monkeypatch):
    """The cumulative workaround (diff against the full again) names the FULL as its base: a
    consumer who applied delta 1 refuses it. Hence the ledger, not a bigger pack."""
    c, tmp = env
    assert _dump(tmp, "full") == 0
    c.upsert(COLL, points=[_point(2, 0.77)])
    assert _dump(tmp, "d1", base="full", seq=1) == 0
    c.upsert(COLL, points=[_point(3, 0.66)])
    _ledger_off(monkeypatch)
    assert _dump(tmp, "cum", base="full", seq=2) == 0    # cumulative: base is the full
    assert _manifest(tmp, "cum")["base_manifest_sha256"] == pack.sha256_file(
        str(tmp / "packs/full/manifest.json"))
    state = tmp / "apply-state.json"
    state.write_text(json.dumps({"applied_manifest_sha256": pack.sha256_file(
        str(tmp / "packs/d1/manifest.json")), "sequence": 1}))
    monkeypatch.setattr(Q, "qdrant_client", lambda *a, **k: c)
    with pytest.raises(apply_mod.ApplyRefusal, match="does not match"):
        apply_mod.main(["--manifest", str(tmp / "packs/cum"), "--qdrant-url", ":memory:",
                        "--collection", "x", "--allow-unsigned", "--state-file", str(state)])


def test_the_ledger_root_setting_is_off_by_default(monkeypatch):
    monkeypatch.delenv("AW_FINGERPRINT_LEDGER_ROOT", raising=False)
    monkeypatch.setenv("AW_WORK_ROOT", "/tmp/aw-x")
    paths.reset_settings_cache()
    try:
        assert dump._configured_ledger() is None
    finally:
        paths.reset_settings_cache()


def _hold(tmp):
    import fcntl
    (tmp / "ledger").mkdir(exist_ok=True)
    fh = open(tmp / "ledger/.lock", "a")
    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fh


def test_a_held_lock_refuses_delta_and_full_and_writes_nothing(env, capsys):
    c, tmp = env
    assert _dump(tmp, "full") == 0
    before = FL.Ledger(tmp / "ledger").verify()
    fh = _hold(tmp)
    try:
        c.upsert(COLL, points=[_point(2, 0.77)])
        assert _dump(tmp, "d1", base="full", seq=1) == 1
        assert _dump(tmp, "full2") == 1
    finally:
        fh.close()
    assert "held by another dump" in capsys.readouterr().err
    assert list((tmp / "packs/d1").iterdir()) == [] and list((tmp / "packs/full2").iterdir()) == []
    assert FL.Ledger(tmp / "ledger").verify() == before
    assert _dump(tmp, "d1", base="full", seq=1) == 0   # released: the next run proceeds


def test_the_lock_is_released_after_the_dump(env):
    c, tmp = env
    assert _dump(tmp, "full") == 0
    fh = _hold(tmp)   # would raise BlockingIOError if the dump kept it
    fh.close()


def test_ledger_off_takes_no_lock(env, monkeypatch):
    c, tmp = env
    _ledger_off(monkeypatch)
    assert _dump(tmp, "full") == 0
    assert not (tmp / "ledger").exists()


def test_a_fork_at_advance_is_a_conflict_exit_4(env, monkeypatch, capsys):
    c, tmp = env
    assert _dump(tmp, "full") == 0
    c.upsert(COLL, points=[_point(2, 0.77)])
    real = FL.advance_from_pack_dir

    def other_dump_won(ledger, d):   # the ledger moved while this pack was being built
        _ledger_commit_other(tmp)
        return real(ledger, d)
    monkeypatch.setattr(FL, "advance_from_pack_dir", other_dump_won)
    assert _dump(tmp, "d1", base="full", seq=1) == 4
    err = capsys.readouterr().err
    assert "LEDGER CONFLICT: do not publish" in err and "LEDGER LAGGING" not in err


def _ledger_commit_other(tmp):
    led = FL.Ledger(tmp / "ledger")
    d = tmp / "packs/full"
    m = json.loads((d / "manifest.json").read_text())
    other = tmp / "packs/other"
    other.mkdir()
    # a different seq-1 pack on the same base, folded first
    FL.update_from_pack(led, {**m, "schema": pack.DELTA_PACK_SCHEMA, "sequence": 1,
                              "base_manifest_sha256": pack.sha256_file(str(d / "manifest.json")),
                              "pack_id": "other", "base_dump_id": "full", "shards": []},
                        "ab" * 32, [], [])


def test_the_unseeded_refusal_names_the_repair(env, capsys):
    c, tmp = env
    assert _dump(tmp, "full") == 0
    (tmp / "ledger/state.json").unlink()
    assert _dump(tmp, "d1", base="full", seq=1) == 1
    err = capsys.readouterr().err
    assert "rebuild_by_fold" in err and "next FULL" in err


def _log_writer(tmp):
    log = tmp / "log"
    log.mkdir(exist_ok=True)

    def seg(seq, lines):
        h = {"schema": "aw.change-log.v1", "pack_id": f"p{seq}", "pack_sha": "ab" * 32,
             "landed_at": "2026-11-01T00:00:00Z", "n": len(lines), "seq": seq}
        (log / f"2026110{seq}T000000Z-p{seq}.jsonl").write_text(
            "\n".join(json.dumps(x) for x in [h, *lines]) + "\n")
        (log / ".seq").write_text(f"{seq}\n")
    return log, seg


def _log_delta(tmp, name, base, seq, log):
    return dump.cli(["--collection", COLL, "--out", str(tmp / "packs" / name), "--dump-id", name,
                     "--delta-base-manifest", str(tmp / "packs" / base / "manifest.json"),
                     "--sequence", str(seq), "--from-change-log", str(log)])


def test_change_log_path_two_hops_with_the_ledger_on(env):
    """full -> d1 (change log) -> d2 (change log); d2's base is the fold, not a stored sidecar."""
    c, tmp = env
    log, seg = _log_writer(tmp)
    led = FL.Ledger(tmp / "ledger")
    seg(1, [{"id": "1", "op": "replace"}])
    assert _dump(tmp, "full", "--from-change-log", str(log)) == 0
    c.upsert(COLL, points=[_point(2, 0.77)])
    seg(2, [{"id": "2", "op": "replace"}])
    assert _log_delta(tmp, "d1", "full", 1, log) == 0
    m1 = _manifest(tmp, "d1")
    assert m1["delta_source"]["path"] == "change-log"
    assert {r["id"] for r in FL.pack_rows(tmp / "packs/d1", m1)} == {"2"}
    assert led.verify()["sequence"] == 1
    c.upsert(COLL, points=[_point(3, 0.66), _point(9)])
    seg(3, [{"id": "3", "op": "replace"}, {"id": "9", "op": "replace"}])
    assert _log_delta(tmp, "d2", "d1", 2, log) == 0
    m2 = _manifest(tmp, "d2")
    assert m2["delta_source"]["path"] == "change-log"
    assert {r["id"] for r in FL.pack_rows(tmp / "packs/d2", m2)} == {"3", "9"}
    assert m2["base_manifest_sha256"] == pack.sha256_file(str(tmp / "packs/d1/manifest.json"))
    # d2's base was the fold over (full, d1): the ledger now equals the live collection
    live = {r["id"]: pack.row_fingerprint(r) for r in dump.scroll_rows(c, COLL)}
    assert dict(led.items()) == dict(sorted(live.items(), key=lambda kv: pack.point_id_key(kv[0])))
    assert led.verify()["sequence"] == 2
    FL.check_rebuild(tmp / "packs", led)


# NB: payload edits, not vector re-upserts: the in-memory Qdrant renormalises a re-upserted vector
# to a different ulp (see rehearsal.py), so only a payload round trip is bit-identical here.
def test_net_no_op_in_the_log_yields_no_row_against_the_ledger_base(env):
    """A row changed and changed back between two deltas is named by the log (twice) but the
    fetched row equals the ledger base: no row ships, and it is counted as a no-op."""
    c, tmp = env
    log, seg = _log_writer(tmp)
    seg(1, [{"id": "1", "op": "replace"}])
    assert _dump(tmp, "full", "--from-change-log", str(log)) == 0
    c.upsert(COLL, points=[_point(1, 0.42)])
    c.set_payload(COLL, payload={"title": "edited"}, points=[2])   # 2 changes ...
    seg(2, [{"id": "2", "op": "replace"}, {"id": "1", "op": "replace"}])
    c.set_payload(COLL, payload={"title": "T2"}, points=[2])      # ... and 2 changes back
    seg(3, [{"id": "2", "op": "replace"}])
    assert _log_delta(tmp, "d1", "full", 1, log) == 0
    m = _manifest(tmp, "d1")
    assert {r["id"] for r in FL.pack_rows(tmp / "packs/d1", m)} == {"1"}
    assert FL.pack_tombstones(tmp / "packs/d1", m) == []
    assert m["delta_source"]["no_ops_dropped"] == 1   # id 2 dropped BY the base comparison


def test_net_no_op_on_the_scroll_path_yields_no_row_against_the_ledger_base(env):
    c, tmp = env
    assert _dump(tmp, "full") == 0
    c.set_payload(COLL, payload={"title": "edited"}, points=[2])
    c.set_payload(COLL, payload={"title": "T2"}, points=[2])      # changed and changed back
    c.upsert(COLL, points=[_point(7)])
    assert _dump(tmp, "d1", base="full", seq=1) == 0
    m = _manifest(tmp, "d1")
    assert {r["id"] for r in FL.pack_rows(tmp / "packs/d1", m)} == {"7"}
    # the scroll builder keeps no no-op counter (only the log builder writes `no_ops_dropped`):
    # pin that id 2 is dropped by the comparison, i.e. it is live and equals the ledger base
    live = {r["id"]: pack.row_fingerprint(r) for r in dump.scroll_rows(c, COLL)}
    assert dict(FL.Ledger(tmp / "ledger").items())["2"] == live["2"]


def test_three_hop_apply_side_equality(env):
    """The stranger applies full, d1, d2 in order, each checked the way `cyt apply` checks it
    (shard hashes, tombstone count), upserting rows and deleting tombstones; the resulting row
    set equals the master's. Pack-row level: `cyt apply`/`cyt rehearse` need a Qdrant with UUID
    point ids, and this fixture's master uses integer ids."""
    c, tmp = env
    _chain(env)
    mine: dict[str, str] = {}
    prev = None
    for name in ("full", "d1", "d2"):
        d = tmp / "packs" / name
        m = _manifest(tmp, name)
        apply_mod.verify_shard_hashes(m, d)
        tombs = []
        if prev is not None:
            assert m["base_manifest_sha256"] == prev    # the chain the stranger's state file enforces
            tombs = FL.pack_tombstones(d, m)
            assert apply_mod.verify_tombstones(m, d) == len(tombs)
        for r in FL.pack_rows(d, m):
            mine[r["id"]] = pack.row_fingerprint(r)
        for t in tombs:
            del mine[t]
        prev = pack.sha256_file(str(d / "manifest.json"))
    master = {r["id"]: pack.row_fingerprint(r) for r in dump.scroll_rows(c, COLL)}
    assert mine == master and set(mine) == {"1", "2", "3", "7"}


def test_help_documents_the_exit_codes(capsys):
    with pytest.raises(SystemExit) as e:
        dump.cli(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for line in ("  1  REFUSED", "  3  LEDGER LAGGING", "  4  LEDGER CONFLICT"):
        assert line in out
    assert "rebuild_by_fold" in out and "do NOT publish" in out


def test_tombstones_gz_is_bit_exact_across_seconds(env, monkeypatch):
    """task 2605: no wall-clock in a tombstone line, so two builds >= 1.1 s apart are bit-identical."""
    import time
    c, tmp = env
    assert _dump(tmp, "full") == 0
    c.delete(COLL, points_selector=qmodels.PointIdsList(points=[3]))
    _ledger_off(monkeypatch)
    assert _dump(tmp, "t1", base="full", seq=1, reasons={"3": "superseded"}) == 0
    time.sleep(1.1)
    assert _dump(tmp, "t2", base="full", seq=1, reasons={"3": "superseded"}) == 0
    f1, f2 = (tmp / "packs" / n / "tombstones.jsonl.gz" for n in ("t1", "t2"))
    assert f1.read_bytes()[4:8] == b"\0\0\0\0"          # gz header mtime
    assert f1.read_bytes() == f2.read_bytes()
    assert _manifest(tmp, "t1")["tombstones"]["sha256"] == _manifest(tmp, "t2")["tombstones"]["sha256"]
