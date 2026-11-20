"""`cyt torrent` (task 687, S8): a deterministic v1 metainfo from the manifest's file list."""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from cytir import cli
from cytir.distribute import dump, idf, pack, torrent

PIECE = 16384


def _row(i):
    return pack.row_to_record(
        f"00000000-0000-0000-0000-{i:012d}", [0.1, 0.2, 0.3],
        {"url": f"https://example.org/{i}", "source_type": "a", "license": "MIT",
         "title": "".join(hashlib.sha256(f"{i}{k}".encode()).hexdigest() for k in range(8))}, ([i, i + 1], [1.0, 2.0]))


def _release(tmp_path, name="rel", n=30, signed=True):
    out = tmp_path / name
    acc = idf.IdfAccumulator()
    shards, total, fps = dump.write_full_shards(
        iter([_row(i) for i in range(1, n + 1)]), out, shard_rows=10, idf=acc)
    manifest = dump.build_manifest(
        dump_id=name, collection="src", dims=3, distance="cosine", shards=shards,
        total_points=total, any_row_has_sparse=True, idf_snapshot=acc.write(out),
        fingerprints=fps, profile=None)
    if signed:
        manifest["signature"] = {"algo": "minisign", "sig_file": "manifest.json.minisig"}
        (out / "manifest.json.minisig").write_text("untrusted comment: sig\nAAAA\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out, manifest


def _paths(blob):
    return [b"/".join(f[b"path"]).decode() for f in torrent.bdecode(blob)[b"info"][b"files"]]


def test_bencode_matches_the_specification_examples():
    assert torrent.bencode({"spam": "eggs", "cow": "moo"}) == b"d3:cow3:moo4:spam4:eggse"
    assert torrent.bencode([b"spam", 42, -3, 0]) == b"l4:spami42ei-3ei0ee"
    assert torrent.bdecode(b"d3:cow3:moo4:spaml1:a1:bee") == {b"cow": b"moo", b"spam": [b"a", b"b"]}
    with pytest.raises(torrent.TorrentRefusal):
        torrent.bdecode(b"i1eXX")


def test_two_runs_give_the_same_bytes_and_infohash_whatever_the_file_times(tmp_path):
    out, _ = _release(tmp_path)
    first, r1 = torrent.make_torrent(out)
    for p in out.rglob("*"):
        if p.is_file():
            os.utime(p, (1_000_000_000, 1_000_000_000))
    second, r2 = torrent.make_torrent(out)
    assert first == second and r1["infohash"] == r2["infohash"]
    assert hashlib.sha1(torrent.bencode(torrent.bdecode(first)[b"info"])).hexdigest() == r1["infohash"]


def test_the_same_release_built_in_another_directory_has_the_same_infohash(tmp_path):
    a, _ = _release(tmp_path / "x")
    b, _ = _release(tmp_path / "y")
    # the shards differ only by nothing: same rows, same parquet writer, same manifest but for
    # `created_at`; pin the manifest so the comparison is about the packaging
    (b / "manifest.json").write_bytes((a / "manifest.json").read_bytes())
    (b / "fingerprints.tsv.gz").write_bytes((a / "fingerprints.tsv.gz").read_bytes())
    for s in a.glob("shards/*"):
        (b / "shards" / s.name).write_bytes(s.read_bytes())
    assert torrent.make_torrent(a)[1]["infohash"] == torrent.make_torrent(b)[1]["infohash"]


def test_trackers_and_webseeds_do_not_change_the_infohash(tmp_path):
    out, _ = _release(tmp_path)
    plain, r0 = torrent.make_torrent(out)
    rich, r1 = torrent.make_torrent(out, trackers=["udp://t.example:1/announce"],
                                    webseeds=["https://mirror.example/rel/"])
    assert r0["infohash"] == r1["infohash"] and plain != rich
    meta = torrent.bdecode(rich)
    assert meta[b"announce"] == b"udp://t.example:1/announce"
    assert meta[b"url-list"] == [b"https://mirror.example/rel/"]


def test_the_file_list_is_the_manifests_and_a_stray_file_stays_out(tmp_path):
    out, manifest = _release(tmp_path)
    (out / "NOTES.private.txt").write_text("not named by the manifest")
    blob, report = torrent.make_torrent(out)
    paths = _paths(blob)
    assert "NOTES.private.txt" not in paths
    assert paths == sorted(p for p, _, _ in torrent.declared_files(manifest))
    assert {"manifest.json", "manifest.json.minisig", "bm25-idf.tsv.gz",
            "fingerprints.tsv.gz", "shards/shard-00000.parquet"} <= set(paths)
    assert report["files"] == len(paths)


def test_the_pieces_are_the_sha1_of_the_concatenated_files_across_file_boundaries(tmp_path):
    out, _ = _release(tmp_path, n=400)
    blob, report = torrent.make_torrent(out, piece_length=PIECE)
    info = torrent.bdecode(blob)[b"info"]
    stream = b"".join((out / p).read_bytes() for p in _paths(blob))
    assert len(stream) == report["bytes"] and len(stream) > 2 * PIECE
    want = b"".join(hashlib.sha1(stream[i:i + PIECE]).digest()
                    for i in range(0, len(stream), PIECE))
    assert info[b"pieces"] == want and info[b"piece length"] == PIECE
    assert [f[b"length"] for f in info[b"files"]] == [(out / p).stat().st_size for p in _paths(blob)]


def test_the_piece_length_is_a_function_of_the_size():
    assert torrent.choose_piece_length(10) == torrent.MIN_PIECE
    assert torrent.choose_piece_length(1 << 30) == 1 << 20
    assert torrent.choose_piece_length(10 ** 15) == torrent.MAX_PIECE
    assert torrent.choose_piece_length(100 << 30) == torrent.MAX_PIECE


@pytest.mark.parametrize("damage, match", [
    ("missing_shard", "not in"),
    ("shard_bytes", "bytes on disk"),
    ("shard_sha", "sha256"),
    ("idf_sha", "sha256"),
    ("no_manifest", "manifest.json"),
    ("traversal", "plain relative path"),
])
def test_a_release_that_would_not_apply_is_not_packaged(tmp_path, damage, match):
    out, manifest = _release(tmp_path)
    shard = out / manifest["shards"][0]["path"]
    if damage == "missing_shard":
        shard.unlink()
    elif damage == "shard_bytes":
        shard.write_bytes(shard.read_bytes() + b"x")
    elif damage == "shard_sha":
        data = bytearray(shard.read_bytes())
        data[-1] ^= 1
        shard.write_bytes(bytes(data))
        manifest["shards"][0]["bytes"] = len(data)
        (out / "manifest.json").write_text(json.dumps(manifest))
    elif damage == "idf_sha":
        manifest["sparse"]["idf_snapshot"]["sha256"] = "0" * 64
        (out / "manifest.json").write_text(json.dumps(manifest))
    elif damage == "no_manifest":
        (out / "manifest.json").unlink()
    else:
        manifest["shards"][0]["path"] = "../outside.parquet"
        (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(torrent.TorrentRefusal, match=match):
        torrent.make_torrent(out)


def test_verify_accepts_the_directory_it_describes_and_refuses_any_other_bytes(tmp_path):
    out, manifest = _release(tmp_path)
    blob, _ = torrent.make_torrent(out)
    t = tmp_path / "rel.torrent"
    t.write_bytes(blob)
    assert torrent.verify_torrent(t, out)["files"] == len(_paths(blob))
    # a byte changed AND the manifest re-sealed to match: only the torrent can tell
    shard = out / manifest["shards"][0]["path"]
    data = bytearray(shard.read_bytes())
    data[-1] ^= 1
    shard.write_bytes(bytes(data))
    manifest["shards"][0]["sha256"] = pack.sha256_file(str(shard))
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(torrent.TorrentRefusal, match="does not describe"):
        torrent.verify_torrent(t, out)


def test_aw_torrent_is_a_verb_and_writes_beside_the_directory(tmp_path, capsys):
    out, _ = _release(tmp_path)
    assert cli.main(["torrent", "make", str(out)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["torrent"] == str(out) + ".torrent"
    assert report["magnet"].startswith(f"magnet:?xt=urn:btih:{report['infohash']}")
    assert cli.main(["torrent", "verify", report["torrent"], str(out)]) == 0
    (out / "manifest.json").unlink()
    assert cli.main(["torrent", "verify", report["torrent"], str(out)]) == 1


def test_private_is_opt_in_and_the_default_torrent_is_unchanged(tmp_path):
    rel, _ = _release(tmp_path)
    blob, rep = torrent.make_torrent(rel)
    info = torrent.bdecode(blob)[b"info"]
    assert set(info) == {b"name", b"piece length", b"pieces", b"files"}
    assert rep["private"] is False
    assert rep["infohash"] == hashlib.sha1(torrent.bencode(info)).hexdigest()
    assert blob == torrent.make_torrent(rel, private=False)[0]


def test_private_sets_the_key_changes_the_infohash_and_stays_deterministic(tmp_path):
    rel, _ = _release(tmp_path)
    pub, prep = torrent.make_torrent(rel, trackers=["http://t.example/announce"])
    a, rep = torrent.make_torrent(rel, trackers=["http://t.example/announce"], private=True)
    b, _ = torrent.make_torrent(rel, trackers=["http://t.example/announce"], private=True)
    assert torrent.bdecode(a)[b"info"][b"private"] == 1
    assert rep["private"] is True
    assert a == b
    assert rep["infohash"] != prep["infohash"]


def test_private_refuses_without_a_tracker(tmp_path):
    rel, _ = _release(tmp_path)
    with pytest.raises(torrent.TorrentRefusal, match="--tracker"):
        torrent.make_torrent(rel, private=True)


def test_verify_accepts_both_kinds_and_says_which(tmp_path):
    rel, _ = _release(tmp_path)
    for private in (False, True):
        blob, _ = torrent.make_torrent(rel, trackers=["http://t.example/a"], private=private)
        f = tmp_path / f"x{private}.torrent"
        f.write_bytes(blob)
        assert torrent.verify_torrent(f, rel)["private"] is private


def _golden_dir(tmp_path):
    d = tmp_path / "golden"
    d.mkdir()
    (d / "shard-0.jsonl").write_bytes(b"alpha\n" * 3000)
    (d / "shard-1.jsonl").write_bytes(b"bravo-line\n" * 2500)
    manifest = {
        "schema": "aw.dump-manifest.v1", "dump_id": "golden-1", "created_at": "2026-01-01T00:00:00Z",
        "source": "fixture", "embedder": "none", "counts": {}, "license": "MIT",
        "shards": [{"path": "shard-0.jsonl", "sha256": hashlib.sha256(b"alpha\n" * 3000).hexdigest(),
                    "bytes": 18000},
                   {"path": "shard-1.jsonl", "sha256": hashlib.sha256(b"bravo-line\n" * 2500).hexdigest(),
                    "bytes": 27500}],
    }
    (d / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return d


def test_golden_infohashes_pin_the_default_and_private_info_dicts(tmp_path):
    # A red result here means published infohashes would change.
    d = _golden_dir(tmp_path)
    kw = dict(piece_length=PIECE, trackers=["http://t.example/announce"],
              webseeds=["http://w.example/golden/"])
    blob, rep = torrent.make_torrent(d, **kw)
    assert rep["infohash"] == "2464a3637161975dc12c3eadb1c1ca65e6deb209"
    assert hashlib.sha256(blob).hexdigest() == "b3ad85d13b01a47526362087b7e7df96a3ecf5dd97b4b485dc885950a31fc0bf"
    assert torrent.make_torrent(d, private=True, **kw)[1]["infohash"] == "ae320779b72a15a4237332e5f4dcbcb106b770f3"


def test_cli_private_without_tracker_exits_nonzero_and_says_refused(tmp_path, capsys):
    out, _ = _release(tmp_path)
    assert cli.main(["torrent", "make", str(out), "--private"]) != 0
    err = capsys.readouterr().err
    assert err.startswith("REFUSED:") and "--tracker" in err
    assert not (tmp_path / "rel.torrent").exists()
