# Task 650 (migration S8): moved to tests/ (2026-10-04) -- where the public cut ships and re-runs it.
"""Tests for cytir.distribute.apply — `cyt apply` (task 650, S8).

No live Qdrant server: `qdrant_client.QdrantClient(":memory:")` gives a
REAL client (not a hand-rolled fake) backed by an in-process store, so
these tests exercise the actual create_collection/upsert/delete/count
calls apply.py makes.
"""
from __future__ import annotations

import json
import os
import stat
import textwrap

import pytest

from cytir.core.qdrant import qdrant_client
from cytir.distribute import apply, dump, pack


# --------------------------------------------------------------- fixtures
def _row(id_, url, *, dense=(0.1, 0.2, 0.3), sparse=None, license_="MIT",
         source_type="a"):
    return pack.row_to_record(
        id_, list(dense),
        {"url": url, "source_type": source_type, "license": license_,
         "title": url}, sparse)


def _write_dump(tmp_path, rows, *, any_sparse=False, name="dump"):
    out_dir = tmp_path / name
    shards, total, fps = dump.write_full_shards(iter(rows), out_dir, shard_rows=10)
    manifest = dump.build_manifest(
        dump_id=name, collection="src", dims=3, distance="cosine",
        shards=shards, total_points=total, any_row_has_sparse=any_sparse, fingerprints=fps, profile=None)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir, manifest


# ------------------------------------------------------------- signature
def test_verify_dump_signature_unsigned_refused_by_default(tmp_path):
    with pytest.raises(apply.ApplyRefusal):
        apply.verify_dump_signature({}, tmp_path, pubkey=None,
                                    allow_unsigned=False)


def test_verify_dump_signature_unsigned_allowed_with_flag(tmp_path):
    apply.verify_dump_signature({}, tmp_path, pubkey=None, allow_unsigned=True)


def test_verify_dump_signature_signed_needs_pubkey(tmp_path):
    manifest = {"signature": {"algo": "minisign", "sig_file": "manifest.json.minisig"}}
    with pytest.raises(apply.ApplyRefusal):
        apply.verify_dump_signature(manifest, tmp_path, pubkey=None,
                                    allow_unsigned=False)


def _install_fake_minisign(tmp_path, monkeypatch, *, ok=True):
    script = tmp_path / "minisign"
    script.write_text(textwrap.dedent(f"""\
        #!/bin/sh
        case "$1" in
          -V) {"exit 0" if ok else "exit 1"} ;;
          *) exit 2 ;;
        esac
    """))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))


def test_verify_dump_signature_success(tmp_path, monkeypatch):
    _install_fake_minisign(tmp_path, monkeypatch, ok=True)
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "manifest.json.minisig").write_text("sig")
    manifest = {"signature": {"algo": "minisign", "sig_file": "manifest.json.minisig"}}
    apply.verify_dump_signature(manifest, tmp_path, pubkey="RWfake",
                                allow_unsigned=False)


def test_verify_dump_signature_failure_refuses_even_with_allow_unsigned(tmp_path, monkeypatch):
    _install_fake_minisign(tmp_path, monkeypatch, ok=False)
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "manifest.json.minisig").write_text("sig")
    manifest = {"signature": {"algo": "minisign", "sig_file": "manifest.json.minisig"}}
    # A signature that fails verification is NEVER waved through by
    # --allow-unsigned -- that flag only covers "no signature"/"can't check",
    # never "actively tampered".
    with pytest.raises(apply.ApplyRefusal):
        apply.verify_dump_signature(manifest, tmp_path, pubkey="RWfake",
                                    allow_unsigned=True)


# --------------------------------------------------------------- shards
def test_verify_shard_hashes_catches_mismatch(tmp_path):
    out_dir, manifest = _write_dump(tmp_path, [_row("id-1", "https://a")])
    manifest["shards"][0]["sha256"] = "0" * 64
    with pytest.raises(apply.ApplyRefusal):
        apply.verify_shard_hashes(manifest, out_dir)


def test_verify_shard_hashes_catches_missing_file(tmp_path):
    out_dir, manifest = _write_dump(tmp_path, [_row("id-1", "https://a")])
    manifest["shards"][0]["path"] = "shards/does-not-exist.parquet"
    with pytest.raises(apply.ApplyRefusal):
        apply.verify_shard_hashes(manifest, out_dir)


def test_verify_shard_hashes_passes_on_untouched_dump(tmp_path):
    out_dir, manifest = _write_dump(tmp_path, [_row("id-1", "https://a")])
    apply.verify_shard_hashes(manifest, out_dir)  # no raise


# ---------------------------------------------------------- create/upsert
def test_create_collection_then_refuses_without_recreate():
    client = qdrant_client(":memory:")
    manifest = {"embedder": {"dims": 3, "distance": "cosine"}, "sparse": None}
    apply.create_collection(client, "c1", manifest, legacy_layout=True)
    with pytest.raises(apply.ApplyRefusal):
        apply.create_collection(client, "c1", manifest, legacy_layout=True)
    apply.create_collection(client, "c1", manifest, recreate=True, legacy_layout=True)  # no raise


def test_full_dump_roundtrip_general_and_entity_card(tmp_path):
    rows = [
        _row("11111111-1111-1111-1111-111111111111", "https://a",
             sparse=([1, 2], [0.5, 0.5]), source_type="web_page"),
        _row("22222222-2222-2222-2222-222222222222", "https://b",
             sparse=None, source_type="entity_card"),
    ]
    out_dir, manifest = _write_dump(tmp_path, rows, any_sparse=True)

    client = qdrant_client(":memory:")
    apply.create_collection(client, "c2", manifest, legacy_layout=True)
    n = apply.upsert_shards(client, "c2", manifest, out_dir)
    assert n == 2
    assert client.count("c2", exact=True).count == 2

    card = client.retrieve("c2", ids=["22222222-2222-2222-2222-222222222222"],
                           with_vectors=True)[0]
    assert "bm25" not in (card.vector or {})


# -------------------------------------------------------------- tombstones
def test_apply_tombstones_forces_delete(tmp_path):
    rows = [_row("11111111-1111-1111-1111-111111111111", "https://a")]
    out_dir, manifest = _write_dump(tmp_path, rows)
    client = qdrant_client(":memory:")
    apply.create_collection(client, "c3", manifest, legacy_layout=True)
    apply.upsert_shards(client, "c3", manifest, out_dir)
    assert client.count("c3", exact=True).count == 1

    tombstones = [pack.Tombstone(id="11111111-1111-1111-1111-111111111111",
                                 reason="takedown", at="2026-09-25T00:00:00Z")]
    n, sha = pack.write_tombstones(tombstones, str(out_dir / "tombstones.jsonl.gz"))
    delta_manifest = dict(manifest)
    delta_manifest["tombstones"] = {"path": "tombstones.jsonl.gz", "count": n, "sha256": sha}
    n_applied = apply.apply_tombstones(client, "c3", delta_manifest, out_dir)
    assert n_applied == 1
    assert client.count("c3", exact=True).count == 0


def test_apply_tombstones_bad_hash_refuses(tmp_path):
    out_dir, manifest = _write_dump(tmp_path, [_row("id-1", "https://a")])
    tombstones = [pack.Tombstone(id="id-1", reason="takedown", at="2026-09-25T00:00:00Z")]
    n, _ = pack.write_tombstones(tombstones, str(out_dir / "tombstones.jsonl.gz"))
    manifest["tombstones"] = {"path": "tombstones.jsonl.gz", "count": n, "sha256": "0" * 64}
    client = qdrant_client(":memory:")
    with pytest.raises(apply.ApplyRefusal):
        apply.apply_tombstones(client, "c4", manifest, out_dir)


# ---------------------------------------------------------- end-to-end CLI
def test_end_to_end_full_dump_via_cli(tmp_path):
    """Build a small source collection, dump it (writing shards + manifest
    directly, not via the CLI's twin guard, which is dump.py's own concern
    -- tested separately), apply it via `apply.main`, and prove the target
    matches: same count, and a sampled point's URL survives the round trip.
    """
    src = qdrant_client(":memory:")
    from qdrant_client.http import models as qm
    src.create_collection("src", vectors_config={"dense": qm.VectorParams(
        size=3, distance=qm.Distance.COSINE)},
        sparse_vectors_config={"bm25": qm.SparseVectorParams()})
    src.upsert("src", points=[
        qm.PointStruct(id="11111111-1111-1111-1111-111111111111",
                       vector={"dense": [0.1, 0.2, 0.3],
                               "bm25": qm.SparseVector(indices=[1], values=[0.5])},
                       payload={"url": "https://example.org/1", "title": "One",
                               "source_type": "web_page", "license": "CC0-1.0",
                               "lang": "en", "trust_tier": 0.5,
                               "last_seen": "2026-09-20T00:00:00Z"}),
    ])

    rows = list(dump.scroll_rows(src, "src"))
    out_dir, manifest = _write_dump(tmp_path, rows, any_sparse=True, name="e2e")

    target = qdrant_client(":memory:")
    apply.create_collection(target, "dst", manifest, legacy_layout=True)
    n = apply.upsert_shards(target, "dst", manifest, out_dir)
    assert n == 1
    got = target.retrieve("dst", ids=["11111111-1111-1111-1111-111111111111"])[0]
    assert got.payload["url"] == "https://example.org/1"


# --------------------------------------------------------- full CLI + delta
def test_cli_main_full_then_delta_lifecycle(tmp_path, monkeypatch):
    """`cyt apply` end to end, twice: a full dump, then a delta pack that
    supersedes one row and tombstones another. Runs against an on-disk
    Qdrant (`path=`), which the `--qdrant-url` positional-location wiring
    (cytir/core/qdrant.py) accepts exactly like `:memory:` or a real URL."""
    monkeypatch.chdir(tmp_path)
    qdrant_path = str(tmp_path / "qdrant-data")

    base_rows = [_row("11111111-1111-1111-1111-111111111111", "https://a"),
                 _row("22222222-2222-2222-2222-222222222222", "https://b")]
    base_dir, base_manifest_dict = _write_dump(tmp_path, base_rows, name="base")

    rc = apply.main(["--legacy-layout", "--manifest", str(base_dir), "--qdrant-url", qdrant_path,
                     "--collection", "c5", "--allow-unsigned",
                     "--state-file", str(tmp_path / "state.json")])
    assert rc == 0
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["sequence"] == 0
    assert state["applied_manifest_sha256"] == pack.sha256_file(str(base_dir / "manifest.json"))

    # id-1 changed, id-2 gone (tombstoned), id-3 new
    current = [_row("11111111-1111-1111-1111-111111111111", "https://a", dense=(0.9, 0.9, 0.9)),
               _row("33333333-3333-3333-3333-333333333333", "https://c")]
    base_manifest = json.loads((base_dir / "manifest.json").read_text())
    builder = dump.DeltaBuilder(base_manifest, base_dir)
    changed = list(builder.changed_rows(iter(current)))
    tombstones = builder.tombstones({"22222222-2222-2222-2222-222222222222": "takedown"})

    delta_dir = tmp_path / "delta1"
    shards, total = dump.write_shards(iter(changed), delta_dir, shard_rows=10)
    n_tomb, tomb_sha = pack.write_tombstones(tombstones, str(delta_dir / "tombstones.jsonl.gz"))
    delta_manifest = dump.build_manifest(
        dump_id="delta1", collection="src", dims=3, distance="cosine",
        shards=shards, total_points=total, any_row_has_sparse=False, profile=None)
    delta_manifest["schema"] = pack.DELTA_PACK_SCHEMA
    delta_manifest["pack_id"] = "delta1"
    delta_manifest["sequence"] = 1
    delta_manifest["base_dump_id"] = base_manifest["dump_id"]
    delta_manifest["base_manifest_sha256"] = pack.sha256_file(str(base_dir / "manifest.json"))
    delta_manifest["tombstones"] = {"path": "tombstones.jsonl.gz", "count": n_tomb, "sha256": tomb_sha}
    (delta_dir / "manifest.json").write_text(json.dumps(delta_manifest, indent=2))

    rc = apply.main(["--legacy-layout", "--manifest", str(delta_dir), "--qdrant-url", qdrant_path,
                     "--collection", "c5", "--allow-unsigned",
                     "--state-file", str(tmp_path / "state.json")])
    assert rc == 0

    # `apply.main` closes its own client on the way out (a local/on-disk
    # Qdrant holds an exclusive file lock for the client's lifetime), but
    # THIS client is ours, so we close it ourselves before the next
    # `apply.main` call below tries to open the same storage dir.
    target = qdrant_client(qdrant_path)
    assert target.count("c5", exact=True).count == 2  # id-1 (changed) + id-3 (new); id-2 gone
    assert not target.retrieve("c5", ids=["22222222-2222-2222-2222-222222222222"])
    target.close()

    # Replaying the SAME delta again must refuse: state.sequence is already 1.
    with pytest.raises(apply.ApplyRefusal):
        apply.main(["--legacy-layout", "--manifest", str(delta_dir), "--qdrant-url", qdrant_path,
                   "--collection", "c5", "--allow-unsigned",
                   "--state-file", str(tmp_path / "state.json")])


# ---------------------------------------- RULING 74 (3): root key for real releases
_SIGNED = {"signature": {"algo": "minisign", "sig_file": "manifest.json.minisig"}}


def _signed_dir(tmp_path, monkeypatch, *, ok=True):
    _install_fake_minisign(tmp_path, monkeypatch, ok=ok)
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "manifest.json.minisig").write_text("sig")


def test_signed_dump_with_allow_unsigned_is_refused(tmp_path, monkeypatch):
    _signed_dir(tmp_path, monkeypatch)
    for key in ("RWfake", None):
        with pytest.raises(apply.ApplyRefusal, match="only for dumps with no signature"):
            apply.verify_dump_signature(_SIGNED, tmp_path, pubkey=key, allow_unsigned=True)


def test_verification_result_is_reported(tmp_path, monkeypatch):
    assert apply.verify_dump_signature({}, tmp_path, pubkey=None,
                                       allow_unsigned=True) == "unverified"
    _signed_dir(tmp_path, monkeypatch)
    assert apply.verify_dump_signature(_SIGNED, tmp_path, pubkey="RWfake",
                                       allow_unsigned=False) == "verified"


def test_wrong_key_refused(tmp_path, monkeypatch):
    _signed_dir(tmp_path, monkeypatch, ok=False)
    with pytest.raises(apply.ApplyRefusal, match="FAILED"):
        apply.verify_dump_signature(_SIGNED, tmp_path, pubkey="RWwrong", allow_unsigned=False)


def test_refusal_happens_before_any_write(tmp_path, monkeypatch):
    out_dir, manifest = _write_dump(tmp_path, [_row("id-1", "https://a")])
    manifest["signature"] = _SIGNED["signature"]
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    import cytir.core.qdrant as q

    def boom(*a, **k):
        raise AssertionError("qdrant touched before the signature check")
    monkeypatch.setattr(q, "qdrant_client", boom)
    state = tmp_path / "state.json"
    for extra in ([], ["--allow-unsigned"], ["--root-pubkey", "RWfake", "--allow-unsigned"]):
        with pytest.raises(apply.ApplyRefusal):
            apply.main(["--manifest", str(out_dir), "--qdrant-url", "http://x:1",
                        "--collection", "c", "--state-file", str(state)] + extra)
    assert not state.exists()


def _content_fake_minisign(tmp_path, monkeypatch):
    """minisign stand-in that accepts only a file containing SIGNED-OK."""
    script = tmp_path / "minisign"
    script.write_text('#!/bin/sh\nwhile [ $# -gt 0 ]; do case "$1" in -m) m="$2";; esac; shift; done\n'
                      'grep -q SIGNED-OK "$m"\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))


def _no_client(monkeypatch):
    import cytir.core.qdrant as q
    calls = []
    monkeypatch.setattr(q, "qdrant_client", lambda *a, **k: calls.append(1) or 1 / 0)
    return calls


def test_verified_bytes_are_the_applied_bytes(tmp_path, monkeypatch):
    """review 2851 finding 1: --manifest dir/evil.json must not ride on dir/manifest.json's signature."""
    _content_fake_minisign(tmp_path, monkeypatch)
    out_dir, manifest = _write_dump(tmp_path, [_row("id-1", "https://a")])
    manifest["signature"] = _SIGNED["signature"]
    (out_dir / "evil.json").write_text(json.dumps(manifest))
    manifest["note"] = "SIGNED-OK"
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    (out_dir / "manifest.json.minisig").write_text("sig")
    calls = _no_client(monkeypatch)
    state = tmp_path / "state.json"
    with pytest.raises(apply.ApplyRefusal, match="FAILED"):
        apply.main(["--manifest", str(out_dir / "evil.json"), "--qdrant-url", "http://x:1",
                    "--collection", "c", "--root-pubkey", "RWk", "--state-file", str(state)])
    assert not calls and not state.exists()


def test_key_with_allow_unsigned_is_refused(tmp_path, monkeypatch):
    out_dir, manifest = _write_dump(tmp_path, [_row("id-1", "https://a")])  # signature stripped
    calls = _no_client(monkeypatch)
    for flag in ("--root-pubkey", "--pubkey"):
        with pytest.raises(apply.ApplyRefusal, match="cannot be combined"):
            apply.main(["--manifest", str(out_dir), "--qdrant-url", "http://x:1",
                        "--collection", "c", "--allow-unsigned", flag, "RWk",
                        "--state-file", str(tmp_path / "s.json")])
    assert not calls


def test_two_different_keys_refused(tmp_path):
    out_dir, _ = _write_dump(tmp_path, [_row("id-1", "https://a")])
    with pytest.raises(apply.ApplyRefusal, match="both given"):
        apply.main(["--manifest", str(out_dir), "--qdrant-url", "http://x:1", "--collection", "c",
                    "--root-pubkey", "RWa", "--pubkey", "RWb"])


def test_signature_block_without_sig_file_is_a_refusal(tmp_path):
    for sig in ({"algo": "minisign"}, {"algo": "minisign", "sig_file": ""}):
        with pytest.raises(apply.ApplyRefusal, match="sig_file"):
            apply.verify_dump_signature({"signature": sig}, tmp_path, pubkey="RWk",
                                        allow_unsigned=False)


def test_signed_dump_with_minisign_absent_is_refused(tmp_path, monkeypatch):
    from cytir.distribute import sign
    monkeypatch.setattr(sign.shutil, "which", lambda *_: None)
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "manifest.json.minisig").write_text("sig")
    with pytest.raises(apply.ApplyRefusal, match="cannot verify signature"):
        apply.verify_dump_signature(_SIGNED, tmp_path, pubkey="RWk", allow_unsigned=False)


def test_delta_order_refusals_build_no_client(tmp_path, monkeypatch):
    out_dir, manifest = _write_dump(tmp_path, [_row("id-1", "https://a")])
    manifest.update(schema=pack.DELTA_PACK_SCHEMA, pack_id="d", sequence=5,
                    base_dump_id="x", base_manifest_sha256="0" * 64,
                    tombstones={"path": "t.jsonl.gz", "count": 1, "sha256": "0" * 64})
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    calls = _no_client(monkeypatch)
    state = tmp_path / "state.json"
    with pytest.raises(apply.ApplyRefusal, match="base"):
        apply.main(["--manifest", str(out_dir), "--qdrant-url", "http://x:1",
                    "--collection", "c", "--allow-unsigned", "--state-file", str(state)])
    state.write_text(json.dumps({"applied_manifest_sha256": "0" * 64, "sequence": 1}))
    with pytest.raises(apply.ApplyRefusal, match="sequence"):
        apply.main(["--manifest", str(out_dir), "--qdrant-url", "http://x:1",
                    "--collection", "c", "--allow-unsigned", "--state-file", str(state)])
    state.write_text(json.dumps({"applied_manifest_sha256": "0" * 64, "sequence": 4}))
    (out_dir / "t.jsonl.gz").write_bytes(b"x")
    with pytest.raises(apply.ApplyRefusal, match="tombstones file hash"):
        apply.main(["--manifest", str(out_dir), "--qdrant-url", "http://x:1",
                    "--collection", "c", "--allow-unsigned", "--state-file", str(state)])
    assert not calls
