#!/usr/bin/env python3
"""cytir.distribute.torrent — `cyt torrent` (task 687, S8; specs/dump-format.md §6b).

A deterministic BitTorrent v1 metainfo for a dump or delta-pack directory. Two runs on the
same directory give the same bytes and so the same infohash; that infohash is what the
release notes publish next to the signed manifest.

WHAT SHIPS. The file list is the manifest's, never the directory's (release-doctrine #1:
a file the manifest does not name is not part of the release, and a stray file beside the
shards stays out of the torrent):

    manifest.json, its detached signature, every shard, the fingerprint sidecar,
    the tombstone file, the BM25 IDF snapshot

A file the manifest declares and the directory lacks, or whose bytes do not match the
declared `sha256`, refuses: a torrent of a release that would not apply is a mistake that
cannot be recalled once seeded. The check is made in the same single read that hashes the
pieces.

WHY IT IS DETERMINISTIC. Files sorted by path; `piece length` a function of the total size
(or given); no creation date, no `created by`, no `comment`, nothing from the clock or the
host. Trackers and webseeds sit outside the `info` dict, so they never change the infohash.

NOT DONE. v1 only (SHA-1 pieces, no BEP 52 v2 hashes). SHA-1 is what every client speaks
today; integrity of the release comes from the minisign signature over the manifest and the
sha256 of every file in it, not from the torrent's hash.

Dependencies: none (a 40-line bencoder/decoder below); a library would add a licence and a
second implementation to audit for 40 lines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Iterable, Optional

_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cytir.distribute import pack

MIN_PIECE = 1 << 18      # 256 KiB
MAX_PIECE = 1 << 24      # 16 MiB
TARGET_PIECES = 2000


class TorrentRefusal(ValueError):
    pass


# ------------------------------------------------------------------- bencode
def bencode(obj) -> bytes:
    if isinstance(obj, bool):
        raise TypeError("bool is not bencodable")
    if isinstance(obj, int):
        return b"i%de" % obj
    if isinstance(obj, str):
        obj = obj.encode("utf-8")
    if isinstance(obj, (bytes, bytearray)):
        return b"%d:%s" % (len(obj), bytes(obj))
    if isinstance(obj, (list, tuple)):
        return b"l" + b"".join(bencode(x) for x in obj) + b"e"
    if isinstance(obj, dict):
        items = sorted((k.encode("utf-8") if isinstance(k, str) else k, v)
                       for k, v in obj.items())
        return b"d" + b"".join(bencode(k) + bencode(v) for k, v in items) + b"e"
    raise TypeError(f"cannot bencode {type(obj).__name__}")


def bdecode(data: bytes):
    value, end = _decode(data, 0)
    if end != len(data):
        raise TorrentRefusal("trailing bytes after the bencoded value")
    return value


def _decode(data: bytes, i: int):
    c = data[i:i + 1]
    if c == b"i":
        j = data.index(b"e", i)
        return int(data[i + 1:j]), j + 1
    if c == b"l":
        out, i = [], i + 1
        while data[i:i + 1] != b"e":
            v, i = _decode(data, i)
            out.append(v)
        return out, i + 1
    if c == b"d":
        out, i = {}, i + 1
        while data[i:i + 1] != b"e":
            k, i = _decode(data, i)
            out[k], i = _decode(data, i)
        return out, i + 1
    if c.isdigit():
        j = data.index(b":", i)
        n = int(data[i:j])
        return data[j + 1:j + 1 + n], j + 1 + n
    raise TorrentRefusal(f"bad bencode at byte {i}")


# ----------------------------------------------------------------- file list
def declared_files(manifest: dict) -> list[tuple[str, Optional[str], Optional[int]]]:
    """`(relative path, declared sha256 or None, declared bytes or None)` for every file
    the manifest names, plus `manifest.json` itself. Sorted by path."""
    files: dict[str, tuple[Optional[str], Optional[int]]] = {"manifest.json": (None, None)}

    def add(path: str, sha: Optional[str], size: Optional[int]) -> None:
        if path in files:
            raise TorrentRefusal(f"manifest names {path!r} twice")
        files[path] = (sha, size)

    for s in manifest.get("shards", []):
        add(s["path"], s.get("sha256"), s.get("bytes"))
    for key in ("fingerprints", "tombstones"):
        block = manifest.get(key)
        if block:
            add(block["path"], block.get("sha256"), None)
    idf = (manifest.get("sparse") or {}).get("idf_snapshot")
    if idf:
        add(idf["path"], idf.get("sha256"), idf.get("bytes"))
    sig = manifest.get("signature")
    if sig:
        add(sig["sig_file"], None, None)
    for path in files:
        parts = path.split("/")
        if path.startswith("/") or ".." in parts or "" in parts:
            raise TorrentRefusal(f"manifest path {path!r} is not a plain relative path")
    return [(p, *files[p]) for p in sorted(files, key=lambda p: p.encode("utf-8"))]


def choose_piece_length(total_bytes: int) -> int:
    want = max(total_bytes // TARGET_PIECES, 1)
    piece = MIN_PIECE
    while piece < want and piece < MAX_PIECE:
        piece <<= 1
    return piece


# ---------------------------------------------------------------------- build
def build_info(directory: pathlib.Path, *, piece_length: Optional[int] = None,
               private: bool = False) -> tuple[dict, dict]:
    """The `info` dict for a release directory, and a report. One read of every file."""
    directory = pathlib.Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise TorrentRefusal(f"{directory} has no manifest.json: not a dump or pack directory")
    manifest = pack.validate_manifest(json.loads(manifest_path.read_text()))
    listing = declared_files(manifest)

    sizes = []
    for rel, _sha, declared_bytes in listing:
        path = directory / rel
        if not path.is_file():
            raise TorrentRefusal(f"the manifest names {rel!r} and it is not in {directory}")
        size = path.stat().st_size
        if declared_bytes is not None and size != declared_bytes:
            raise TorrentRefusal(f"{rel}: {size} bytes on disk, manifest says {declared_bytes}")
        sizes.append(size)
    total = sum(sizes)
    if not total:
        raise TorrentRefusal("nothing to package")
    piece = piece_length or choose_piece_length(total)
    if piece < 16384 or piece & (piece - 1):
        raise TorrentRefusal(f"piece length {piece} is not a power of two >= 16 KiB")

    pieces: list[bytes] = []
    hasher, filled = hashlib.sha1(), 0
    for (rel, declared_sha, _), size in zip(listing, sizes):
        sha = hashlib.sha256()
        with open(directory / rel, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                sha.update(chunk)
                view = memoryview(chunk)
                while len(view):
                    take = min(len(view), piece - filled)
                    hasher.update(view[:take])
                    filled += take
                    view = view[take:]
                    if filled == piece:
                        pieces.append(hasher.digest())
                        hasher, filled = hashlib.sha1(), 0
        if declared_sha is not None and sha.hexdigest() != declared_sha:
            raise TorrentRefusal(
                f"{rel}: sha256 {sha.hexdigest()} does not match the manifest's {declared_sha}")
    if filled:
        pieces.append(hasher.digest())

    info = {
        "name": str(manifest.get("pack_id") or manifest["dump_id"]),
        "piece length": piece,
        "pieces": b"".join(pieces),
        "files": [{"length": size, "path": rel.split("/")}
                  for (rel, _, _), size in zip(listing, sizes)],
    }
    if private:
        info["private"] = 1  # BEP 27: tracker-only swarm, no DHT/PEX
    report = {"files": len(listing), "bytes": total, "piece_length": piece,
              "pieces": len(pieces), "private": bool(private)}
    return info, report


def infohash(info: dict) -> str:
    return hashlib.sha1(bencode(info)).hexdigest()


def make_torrent(directory: pathlib.Path, *, piece_length: Optional[int] = None,
                 trackers: Iterable[str] = (), webseeds: Iterable[str] = (),
                 private: bool = False) -> tuple[bytes, dict]:
    """`(.torrent bytes, report)`. Trackers and webseeds are outside `info`: they do not
    change the infohash, so a mirror list can change without a new release identity."""
    trackers, webseeds = list(trackers), list(webseeds)
    if private and not trackers:
        raise TorrentRefusal("--private needs at least one --tracker: a private swarm has no DHT")
    info, report = build_info(directory, piece_length=piece_length, private=private)
    meta: dict = {"info": info}
    if trackers:
        meta["announce"] = trackers[0]
        meta["announce-list"] = [[t] for t in trackers]
    if webseeds:
        meta["url-list"] = webseeds
    report["infohash"] = infohash(info)
    report["name"] = info["name"]
    report["magnet"] = f"magnet:?xt=urn:btih:{report['infohash']}&dn={info['name']}"
    return bencode(meta), report


def verify_torrent(torrent_path: pathlib.Path, directory: pathlib.Path) -> dict:
    """Rebuilds the metainfo from `directory` with the torrent's own piece length and
    refuses unless the `info` dicts are identical (so: same files, same bytes)."""
    meta = bdecode(pathlib.Path(torrent_path).read_bytes())
    want = meta[b"info"]
    info, report = build_info(directory, piece_length=want[b"piece length"],
                              private=want.get(b"private") == 1)
    got = bdecode(bencode(info))
    if got != want:
        raise TorrentRefusal(
            f"{torrent_path} does not describe {directory}: infohash "
            f"{hashlib.sha1(bencode(want)).hexdigest()} != {infohash(info)}")
    report["infohash"] = infohash(info)
    return report


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None, prog: str = "cyt torrent") -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="write the .torrent for a dump/pack directory")
    m.add_argument("directory")
    m.add_argument("--out", default=None,
                   help="default: <directory>.torrent, beside the directory")
    m.add_argument("--piece-length", type=int, default=None)
    m.add_argument("--tracker", action="append", default=[])
    m.add_argument("--webseed", action="append", default=[])
    m.add_argument("--private", action="store_true",
                   help="BEP 27 private=1 in info, for rehearsals of unreleased dumps only; "
                        "needs --tracker; changes the infohash")
    v = sub.add_parser("verify", help="refuse unless a .torrent describes a directory")
    v.add_argument("torrent")
    v.add_argument("directory")
    args = ap.parse_args(argv)
    if args.cmd == "make":
        data, report = make_torrent(pathlib.Path(args.directory),
                                    piece_length=args.piece_length,
                                    trackers=args.tracker, webseeds=args.webseed,
                                    private=args.private)
        out = pathlib.Path(args.out or str(pathlib.Path(args.directory).resolve()) + ".torrent")
        out.write_bytes(data)
        report["torrent"] = str(out)
    else:
        report = verify_torrent(pathlib.Path(args.torrent), pathlib.Path(args.directory))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TorrentRefusal as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        raise SystemExit(1)
