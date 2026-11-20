"""Task 2586: the master-side fingerprint ledger (option B, specs/delta-fingerprint-base.md section 5).

Synthetic rows, a real full -> delta1 -> delta2 pack chain on disk, no Qdrant."""
from __future__ import annotations

import json
import pathlib
import shutil

import pytest

from cytir.distribute import dump, pack
from cytir.distribute import fingerprint_ledger as FL
from cytir.ops import public_cut


def _row(i, tag="a", dense=(0.1, 0.2)):
    return pack.row_to_record(str(i), list(dense), {"url": f"https://x/{i}", "v": tag}, None)


def _write_pack(root, name, rows, *, delta=None, tombs=()):
    """delta = (seq, base_manifest_sha256) or None for a FULL dump. Returns (dir, sha)."""
    root.mkdir(parents=True, exist_ok=True)
    d = root / name
    d.mkdir()
    shards, total, fps = dump.write_full_shards(iter(rows), d, shard_rows=3)
    m = dump.build_manifest(dump_id=name, collection="c", dims=2, distance="cosine",
                            shards=shards, total_points=len(rows), any_row_has_sparse=False,
                            fingerprints=None if delta else fps, profile=None)
    m["created_at"] = f"2026-11-{1 + len(list(root.iterdir())):02d}T00:00:00Z"
    if delta:
        (d / "fingerprints.tsv.gz").unlink()
        m["schema"] = pack.DELTA_PACK_SCHEMA
        m["pack_id"] = name
        m["sequence"], m["base_manifest_sha256"] = delta
        m["base_dump_id"] = "x"
        cnt, sha = pack.write_tombstones(
            [pack.Tombstone(id=str(t), reason="aged_out", at="") for t in tombs],
            str(d / "tombstones.jsonl.gz"))
        m["tombstones"] = {"path": "tombstones.jsonl.gz", "count": cnt, "sha256": sha}
    (d / "manifest.json").write_text(json.dumps(m, sort_keys=True))
    return d, pack.sha256_file(str(d / "manifest.json"))


def _apply_all(packs, ledger):
    for d, sha in packs:
        m = json.loads((d / "manifest.json").read_text())
        FL.update_from_pack(ledger, m, sha, FL.pack_rows(d, m), FL.pack_tombstones(d, m))


@pytest.fixture()
def chain(tmp_path):
    packs = tmp_path / "packs"
    full = _write_pack(packs, "full", [_row(i) for i in (1, 2, 3, 4, 5)])
    # delta1: replace 2, tombstone 3, add 6
    d1 = _write_pack(packs, "d1", [_row(2, "b"), _row(6)], delta=(1, full[1]), tombs=[3])
    # delta2: replace 2 again, re-add 3, tombstone 5
    d2 = _write_pack(packs, "d2", [_row(2, "c"), _row(3, "z")], delta=(2, d1[1]), tombs=[5])
    return tmp_path, packs, [full, d1, d2]


def _expected(packs_state):
    """Independent model: a dict fold over rows."""
    st = {}
    for rows, tombs in packs_state:
        for r in rows:
            st[r["id"]] = pack.row_fingerprint(r)
        for t in tombs:
            st.pop(str(t), None)
    return dict(sorted(st.items(), key=lambda kv: pack.point_id_key(kv[0])))


def test_full_ledger_is_byte_identical_to_the_dump_sidecar(chain):
    tmp, packs, ps = chain
    led = FL.Ledger(tmp / "led")
    _apply_all(ps[:1], led)
    full_manifest = json.loads((ps[0][0] / "manifest.json").read_text())
    assert led.read_state()["sidecar_sha256"] == full_manifest["fingerprints"]["sha256"]


def test_three_hop_delta2_base_equals_fold_of_full_and_delta1(chain):
    tmp, packs, ps = chain
    led = FL.Ledger(tmp / "led")
    _apply_all(ps[:2], led)
    base = FL.base_for_delta(led, expected_manifest_sha256=ps[1][1],
                             expected_full_manifest_sha256=ps[0][1], since_seq=1)
    got = {i: fp for i, fp, _ in pack.iter_fingerprints(base.path, expected_count=base.count)}
    want = _expected([([_row(i) for i in (1, 2, 3, 4, 5)], []),
                      ([_row(2, "b"), _row(6)], [3])])
    assert got == want and "3" not in got
    scratch = FL.Ledger(tmp / "scratch")
    st = FL.rebuild_by_fold(packs, scratch, upto_sequence=1)
    assert st["sidecar_sha256"] == base.sha256


def test_fold_equals_incremental_byte_for_byte_replaced_twice_and_readded(chain):
    tmp, packs, ps = chain
    led = FL.Ledger(tmp / "led")
    _apply_all(ps, led)
    assert dict(led.items()) == _expected([([_row(i) for i in (1, 2, 3, 4, 5)], []),
                                           ([_row(2, "b"), _row(6)], [3]),
                                           ([_row(2, "c"), _row(3, "z")], [5])])
    FL.check_rebuild(packs, led, tmp / "scratch")
    scratch = FL.Ledger(tmp / "scratch")
    assert led.sidecar_path().read_bytes() == scratch.sidecar_path().read_bytes()
    assert led.state_path.read_bytes() == scratch.state_path.read_bytes()


def test_check_rebuild_catches_a_drifted_ledger(chain):
    tmp, packs, ps = chain
    led = FL.Ledger(tmp / "led")
    _apply_all(ps[:2], led)  # ledger behind the packs on disk: repairable lag, not corruption
    with pytest.raises(FL.LedgerStale):
        FL.check_rebuild(packs, led, tmp / "scratch")


def test_missing_ledger_refuses_by_name(tmp_path):
    with pytest.raises(FL.LedgerMissing):
        FL.base_for_delta(FL.Ledger(tmp_path / "none"), expected_manifest_sha256="x")


def test_corrupt_sidecar_and_state_refuse(chain):
    tmp, packs, ps = chain
    led = FL.Ledger(tmp / "led")
    _apply_all(ps[:2], led)
    side = led.sidecar_path()
    good = side.read_bytes()
    side.write_bytes(good[:-8])
    with pytest.raises(FL.LedgerCorrupt):
        FL.base_for_delta(led, expected_manifest_sha256=ps[1][1])
    side.write_bytes(good)
    led.state_path.write_text("{not json")
    with pytest.raises(FL.LedgerCorrupt):
        FL.base_for_delta(led, expected_manifest_sha256=ps[1][1])
    led.state_path.write_text(json.dumps({"schema": "other"}))
    with pytest.raises(FL.LedgerCorrupt):
        led.read_state()
    side.unlink()
    led.state_path.write_text(json.dumps({
        "schema": FL.LEDGER_SCHEMA, "algo": pack.FINGERPRINT_ALGO,
        "base_full_manifest_sha256": "a", "pack_manifest_sha256": "b", "sequence": 1,
        "count": 1, "sidecar": side.name, "sidecar_sha256": "c"}))
    with pytest.raises(FL.LedgerMissing):
        led.verify()


def test_stale_ledger_refuses(chain):
    tmp, packs, ps = chain
    led = FL.Ledger(tmp / "led")
    _apply_all(ps[:2], led)
    with pytest.raises(FL.LedgerStale):  # delta2 published, ledger at 1
        FL.base_for_delta(led, expected_manifest_sha256=ps[2][1])


def test_ledger_from_before_a_rebase_refuses(chain):
    tmp, packs, ps = chain
    led = FL.Ledger(tmp / "led")
    _apply_all(ps[:2], led)
    full2 = _write_pack(tmp / "packs2", "full2", [_row(i, "n") for i in (1, 2)])
    with pytest.raises(FL.LedgerRebased):
        FL.base_for_delta(led, expected_manifest_sha256=ps[1][1],
                          expected_full_manifest_sha256=full2[1])
    _apply_all([full2], led)  # a new FULL replaces the ledger in one step
    assert FL.base_for_delta(led, expected_manifest_sha256=full2[1],
                             expected_full_manifest_sha256=full2[1]).count == 2


def test_apply_out_of_order_or_without_ledger_is_refused(chain):
    tmp, packs, ps = chain
    led = FL.Ledger(tmp / "led")
    _apply_all(ps[:1], led)
    m2 = json.loads((ps[2][0] / "manifest.json").read_text())
    with pytest.raises(FL.LedgerChainError):  # delta 2 first
        FL.update_from_pack(led, m2, ps[2][1], FL.pack_rows(ps[2][0], m2))
    with pytest.raises(FL.LedgerMissing):  # no ledger at all, delta first
        FL.update_from_pack(FL.Ledger(tmp / "empty"), m2, ps[2][1], [])


def test_crash_between_pack_signed_and_ledger_advanced(chain):
    tmp, packs, ps = chain
    led = FL.Ledger(tmp / "led")
    _apply_all(ps[:1], led)
    d, sha = ps[1]
    m = json.loads((d / "manifest.json").read_text())
    (led.root / "fingerprints.tmp.tsv.gz").write_bytes(b"half")  # crash mid-write
    with pytest.raises(FL.LedgerStale):  # delta1 published, ledger at 0: refuse
        FL.base_for_delta(led, expected_manifest_sha256=sha)
    r = FL.update_from_pack(led, m, sha, FL.pack_rows(d, m), FL.pack_tombstones(d, m))
    assert not r["noop"]  # repaired by replaying the pack
    assert FL.base_for_delta(led, expected_manifest_sha256=sha).sequence == 1
    r = FL.update_from_pack(led, m, sha, FL.pack_rows(d, m), FL.pack_tombstones(d, m))
    assert r["noop"]  # ledger advanced, caller's marker did not: replay is a no-op
    assert list(led.root.glob("fingerprints.*.tsv.gz")) == [led.sidecar_path()]


def test_fold_refuses_a_fork_and_ignores_an_orphan(chain):
    tmp, packs, ps = chain
    shutil.rmtree(ps[1][0])  # delta1 lost: delta2's base is not in the chain
    assert [c[1]["dump_id"] for c in FL.chain_of(packs)] == ["full"]
    _write_pack(packs, "d1b", [_row(9)], delta=(1, ps[0][1]))
    _write_pack(packs, "d1c", [_row(8)], delta=(1, ps[0][1]))
    with pytest.raises(FL.LedgerChainError):
        FL.chain_of(packs)


def test_tombstone_and_row_for_one_id_in_one_pack_refused(chain):
    tmp, packs, ps = chain
    led = FL.Ledger(tmp / "led")
    _apply_all(ps[:1], led)
    d, sha = _write_pack(tmp / "bad", "bad", [_row(2, "q")], delta=(1, ps[0][1]), tombs=[2])
    m = json.loads((d / "manifest.json").read_text())
    with pytest.raises(FL.LedgerChainError):
        FL.update_from_pack(led, m, sha, FL.pack_rows(d, m), ["2"])


def test_unknown_tombstone_refuses(chain):
    tmp, packs, ps = chain
    led = FL.Ledger(tmp / "led")
    _apply_all(ps[:1], led)
    d, sha = _write_pack(tmp / "bad", "bad", [_row(7)], delta=(1, ps[0][1]), tombs=[99])
    m = json.loads((d / "manifest.json").read_text())
    with pytest.raises(FL.LedgerChainError, match="99"):
        FL.update_from_pack(led, m, sha, FL.pack_rows(d, m), ["99"])
    assert led.verify()["sequence"] == 0  # nothing committed


def test_full_is_checked_against_its_own_sidecar(tmp_path):
    def rich(i):
        return pack.row_to_record(
            f"00000000-0000-4000-8000-{i:012d}", [0.1234567890123 * i, -1e-9 / 3],
            {"url": f"https://x/{i}", "t": "äö"},
            ([1, 4294967000], [0.5, 0.25]) if i % 2 else None)
    d, sha = _write_pack(tmp_path / "p", "full", [rich(i) for i in range(1, 8)])
    led = FL.Ledger(tmp_path / "led")
    FL.advance_from_pack_dir(led, d)  # sparse + non-float32 dense: equals the dump's block
    m = json.loads((d / "manifest.json").read_text())
    for field, val in (("sha256", "0" * 64), ("algo", "other"), ("count", 1)):
        bad = {**m, "fingerprints": {**m["fingerprints"], field: val}}
        with pytest.raises(FL.LedgerCorrupt):
            FL.update_from_pack(FL.Ledger(tmp_path / "l2"), bad, sha, FL.pack_rows(d, m))
    nofp = {k: v for k, v in m.items() if k != "fingerprints"}
    with pytest.raises(FL.LedgerCorrupt):
        FL.update_from_pack(FL.Ledger(tmp_path / "l3"), nofp, sha, FL.pack_rows(d, m))


def test_full_over_full_crash_keeps_the_old_committed_state(tmp_path, monkeypatch):
    f1 = _write_pack(tmp_path / "p1", "f1", [_row(1), _row(2)])
    f2 = _write_pack(tmp_path / "p2", "f2", [_row(1, "n"), _row(3)])
    led = FL.Ledger(tmp_path / "led")
    FL.advance_from_pack_dir(led, f1[0])
    real = FL.os.replace

    def boom(src, dst):
        if str(dst).endswith(FL.STATE_NAME):
            raise OSError("crash before the commit point")
        return real(src, dst)

    monkeypatch.setattr(FL.os, "replace", boom)
    with pytest.raises(OSError):
        FL.advance_from_pack_dir(led, f2[0])
    monkeypatch.setattr(FL.os, "replace", real)
    assert led.verify()["pack_manifest_sha256"] == f1[1]  # old state still valid
    with pytest.raises(FL.LedgerStale):
        FL.base_for_delta(led, expected_manifest_sha256=f2[1])
    FL.advance_from_pack_dir(led, f2[0])  # replay advances
    assert FL.base_for_delta(led, expected_manifest_sha256=f2[1]).count == 2
    assert len(list(led.root.glob("fingerprints.*.tsv.gz"))) == 1


def test_advance_from_pack_dir_walks_the_chain(chain):
    tmp, packs, ps = chain
    led = FL.Ledger(tmp / "led")
    for d, _ in ps:
        FL.advance_from_pack_dir(led, d)
    FL.check_rebuild(packs, led, tmp / "scratch")


def test_replaying_the_base_full_after_deltas_refuses(chain):
    tmp, packs, ps = chain
    led = FL.Ledger(tmp / "led")
    _apply_all(ps[:2], led)
    with pytest.raises(FL.LedgerChainError):
        FL.advance_from_pack_dir(led, ps[0][0])


def test_ancestor_manifest_json_refuses(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(FL.LedgerLocationError):
        FL.Ledger(tmp_path / "deep" / "led")


def test_unsorted_rows_refuse(tmp_path):
    m = dump.build_manifest(dump_id="f", collection="c", dims=2, distance="cosine", shards=[],
                            total_points=2, any_row_has_sparse=False, profile=None)
    m["fingerprints"] = {"algo": pack.FINGERPRINT_ALGO, "sha256": "x", "count": 2}
    with pytest.raises(pack.FingerprintOrderError):
        FL.update_from_pack(FL.Ledger(tmp_path / "led"), m, "s", [_row(2), _row(1)])


def test_ledger_never_ships(tmp_path):
    app = pathlib.Path(FL.__file__).resolve().parents[2]
    for bad in (app / "data" / "ledger", app / "cytir" / "ledger"):
        with pytest.raises(FL.LedgerLocationError):
            FL.Ledger(bad)
    dumpdir = tmp_path / "dump"
    dumpdir.mkdir()
    (dumpdir / "manifest.json").write_text("{}")
    with pytest.raises(FL.LedgerLocationError):
        FL.Ledger(dumpdir)
    doc = public_cut.load_allowlist(app)
    shipped, _ = public_cut.select(doc, ["cytir/distribute/fingerprint_ledger.py"])
    assert "cytir/distribute/fingerprint_ledger.py" in shipped
    for p in ("fingerprint-ledger/line/state.json", "fingerprint-ledger/line/fingerprints.1.tsv.gz"):
        assert not any(public_cut._matches(pat, p)
                       for r in doc["rules"] for pat in r["paths"])


def test_the_drain_does_not_import_the_ledger():
    app = pathlib.Path(FL.__file__).resolve().parents[2]
    for p in app.glob("cytir/**/promote_queue*.py"):
        assert "fingerprint_ledger" not in p.read_text()


def test_check_rebuild_refuses_a_non_empty_scratch(chain):
    tmp, packs, ps = chain
    led = FL.Ledger(tmp / "led")
    _apply_all(ps, led)
    scratch = tmp / "scratch"
    scratch.mkdir()
    (scratch / "junk").write_text("x")
    with pytest.raises(FL.LedgerLocationError):
        FL.check_rebuild(packs, led, scratch)
    FL.check_rebuild(packs, led)  # temp scratch: passes, leaves nothing behind


def test_check_rebuild_without_scratch_uses_a_fresh_dir(chain):
    tmp, packs, ps = chain
    led = FL.Ledger(tmp / "led")
    _apply_all(ps[:2], led)
    with pytest.raises(FL.LedgerStale):
        FL.check_rebuild(packs, led)
    with pytest.raises(FL.LedgerStale):  # a second call is not fooled by the first
        FL.check_rebuild(packs, led)


def test_a_new_full_repairs_a_garbage_state(chain):
    tmp, packs, ps = chain
    led = FL.Ledger(tmp / "led")
    _apply_all(ps[:2], led)
    led.state_path.write_text("{garbage")
    with pytest.raises(FL.LedgerCorrupt):  # a delta never repairs it
        FL.update_from_pack(led, json.loads((ps[2][0] / "manifest.json").read_text()), ps[2][1], [])
    d, sha = ps[0]
    m = json.loads((d / "manifest.json").read_text())
    r = FL.update_from_pack(led, m, sha, FL.pack_rows(d, m))
    assert r["state"]["sequence"] == 0
    led.verify()
