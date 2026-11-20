#!/usr/bin/env python3
"""cytir.distribute.apply — `cyt apply` (task 650, S8; specs/dump-format.md §7).

The self-hoster's side of the release: verify signature and shard hashes,
create the collection from the manifest's own mirror contract (dims,
distance, sparse leg), upsert a full dump or a delta pack's shards, and — for
a delta pack — apply every tombstone as a forced, unconditional delete.

Every refusal in here is a real refusal (engineering-doctrine #3, fail loud
or don't fail): a bad signature, a shard hash mismatch, a delta whose base
does not match local state, or an out-of-sequence pack all stop before
touching Qdrant, not after.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Optional

import pathlib as _pathlib

# Started by path (`python aw/...py`), sys.path[0] is this file's own directory, not the repo
# root: put the root first so `import cytir` resolves (pinned by the script start-guard test).
_REPO_ROOT = str(_pathlib.Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cytir.distribute import idf as idf_snapshot
from cytir.distribute import pack
from cytir.distribute.sign import (
    SignatureVerificationFailed,
    SigningToolMissing,
    verify_manifest,
)

DEFAULT_STATE_FILE = "aw-apply-state.json"
DEFAULT_BATCH = 500
DEFAULT_PAUSE_S = 0.0

_DISTANCE_BY_NAME = {
    "cosine": "Cosine", "euclid": "Euclid", "dot": "Dot", "manhattan": "Manhattan",
}


class ApplyRefusal(RuntimeError):
    pass


# --------------------------------------------------------------- signature
def verify_dump_signature(manifest: dict, manifest_dir: pathlib.Path, *,
                          pubkey: Optional[str], allow_unsigned: bool,
                          manifest_bytes: Optional[bytes] = None) -> str:
    """Return "verified" or "unverified". RULING 74 (3), specs/release-key-and-seeding.md §3.

    A signed dump applies only with a root public key and a valid signature.
    `allow_unsigned` covers a dump with NO signature block and nothing else.
    `manifest_bytes`: the exact bytes the caller parsed; the signature is checked over
    those bytes, not over whatever `manifest_dir/manifest.json` holds.
    """
    sig = manifest.get("signature")
    if not sig:
        if not allow_unsigned:
            raise ApplyRefusal(
                "REFUSED: this dump has no `signature` block (unsigned). "
                "Pass --allow-unsigned to accept it anyway — never do this "
                "for a real release, only for local/CI rehearsal.")
        return "unverified"
    if allow_unsigned:
        raise ApplyRefusal(
            "REFUSED: --allow-unsigned is only for dumps with no signature; "
            "this dump IS signed, so pass --root-pubkey and drop --allow-unsigned.")
    if not pubkey:
        raise ApplyRefusal(
            "REFUSED: this dump IS signed, but no --root-pubkey was given to "
            "verify it against. Pass the publisher's root minisign public "
            "key (published in the README and NOTICE — specs/release-key-and-seeding.md §3).")
    if not isinstance(sig, dict) or not sig.get("sig_file"):
        raise ApplyRefusal("REFUSED: the `signature` block names no `sig_file`; "
                           "cannot verify this dump.")
    sig_path = manifest_dir / sig["sig_file"]
    manifest_path = manifest_dir / "manifest.json"
    try:
        if manifest_bytes is None:
            verify_manifest(str(manifest_path), str(sig_path), pubkey)
        else:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                tmp = pathlib.Path(td) / "manifest.json"
                tmp.write_bytes(manifest_bytes)
                verify_manifest(str(tmp), str(sig_path), pubkey)
    except SigningToolMissing as e:
        raise ApplyRefusal(f"REFUSED: cannot verify signature: {e}") from e
    except SignatureVerificationFailed as e:
        raise ApplyRefusal(f"REFUSED: signature verification FAILED — this "
                           f"dump may be tampered or corrupt: {e}") from e
    return "verified"


# ------------------------------------------------------------------- shards
def verify_shard_hashes(manifest: dict, manifest_dir: pathlib.Path) -> None:
    bad = []
    for shard in manifest.get("shards", []):
        path = manifest_dir / shard["path"]
        if not path.is_file():
            bad.append((shard["path"], "missing"))
            continue
        got = pack.sha256_file(str(path))
        if got != shard["sha256"]:
            bad.append((shard["path"], f"sha256 mismatch: got {got}"))
    if bad:
        raise ApplyRefusal(
            f"REFUSED: {len(bad)} shard(s) failed hash verification: {bad}")


# ------------------------------------------------------------- idf snapshot
def verify_idf_snapshot(manifest: dict, manifest_dir: pathlib.Path) -> Optional[dict]:
    """The pinned BM25 IDF snapshot, when the manifest declares one: missing, altered or
    miscounted is a refusal before Qdrant is touched. Returns its block, or None."""
    try:
        return idf_snapshot.verify_snapshot(manifest, manifest_dir)
    except idf_snapshot.IdfSnapshotRefusal as e:
        raise ApplyRefusal(f"REFUSED: {e}") from e


# --------------------------------------------------------------- collection
def create_collection(client, collection: str, manifest: dict, *,
                      recreate: bool = False, legacy_layout: bool = False,
                      log=lambda m: print(m, file=sys.stderr)) -> dict:
    """Create the apply target through the ONE creator (`cytir.core.serving_profile`).

    A manifest with a `profile` block creates exactly that profile (digest-checked,
    named-profile-checked, width-checked). A manifest from before the block existed is
    REFUSED unless the caller names the old layout on purpose (`legacy_layout=True`,
    `--legacy-layout`): fp32, no quantizer, no on-disk flags, no payload indexes.
    Either way the layout created is logged, and on a Qdrant server the collection is read
    back and compared with the profile. Returns the receipt."""
    from cytir.core import serving_profile as SP

    embedder = manifest["embedder"]
    if str(embedder.get("distance", "cosine")).lower() not in _DISTANCE_BY_NAME:
        raise ApplyRefusal(
            f"REFUSED: unknown distance {embedder.get('distance')!r} in manifest")
    block = manifest.get("profile")
    if block is not None:
        try:
            profile = SP.ServingProfile.from_block(block)
        except (ValueError, KeyError, TypeError) as e:
            raise ApplyRefusal(f"REFUSED: manifest profile block: {e}") from e
        if profile.dense_dim != embedder["dims"]:
            raise ApplyRefusal(
                f"REFUSED: manifest profile {profile.name!r} is {profile.dense_dim}-d "
                f"but the shards are {embedder['dims']}-d")
        profile = profile.with_(distance=str(embedder.get("distance", "cosine")).lower())
        layout = f"profile {profile.name} (digest {block['digest'][:12]})"
        if legacy_layout:
            log("[apply] --legacy-layout ignored: the manifest declares a profile")
    else:
        if not legacy_layout:
            raise ApplyRefusal(
                "REFUSED: this manifest declares no serving profile (a dump from before "
                "slice 8 step 1), so there is no declared layout to create. The old "
                "layout is fp32, no quantizer, no on-disk flags and no payload indexes: "
                "a different, much heavier collection than the one the publisher "
                "serves. Pass --legacy-layout to create it knowingly.")
        sparse = manifest.get("sparse")
        profile = SP.legacy_apply(
            embedder["dims"], str(embedder.get("distance", "cosine")).lower(),
            sparse_name=sparse["vector_name"] if sparse else None,
            # The manifest names the modifier (mirror contract). A manifest from before it
            # was declared names none, and keeps the old, un-weighted collection.
            sparse_idf=bool(sparse) and sparse.get("modifier") == "idf")
        layout = "LEGACY layout (fp32, no quantizer, no payload indexes)"
    log(f"[apply] creating {collection} with {layout}")
    try:
        SP.create_collection(client, collection, profile,
                             on_exists="recreate" if recreate else "refuse",
                             log=log)
    except SP.CollectionExistsError:
        raise ApplyRefusal(
            f"REFUSED: collection {collection!r} already exists. Pass "
            f"--recreate to drop and rebuild it from this dump (only for a "
            f"fresh full-dump apply — never for a delta pack).") from None
    try:
        receipt = SP.verify_created(client, collection, profile, log=log)
    except SP.ShapeMismatch as e:
        raise ApplyRefusal(f"REFUSED: {e}") from e
    return {"layout": layout, **receipt}


# ------------------------------------------------------------------- upsert
def upsert_shards(client, collection: str, manifest: dict, manifest_dir: pathlib.Path,
                  *, batch: int = DEFAULT_BATCH, pause_s: float = DEFAULT_PAUSE_S) -> int:
    import time

    from qdrant_client.http import models as qmodels

    sparse_name = (manifest.get("sparse") or {}).get("vector_name")
    total = 0
    for shard in manifest.get("shards", []):
        buf = []
        for row in pack.read_parquet_shard(str(manifest_dir / shard["path"])):
            vector: dict = {"dense": row["dense"]}
            if sparse_name and row.get("sparse_indices") is not None:
                vector[sparse_name] = qmodels.SparseVector(
                    indices=row["sparse_indices"], values=row["sparse_values"])
            buf.append(qmodels.PointStruct(id=row["id"], vector=vector,
                                           payload=row["payload"]))
            if len(buf) >= batch:
                client.upsert(collection, points=buf)
                total += len(buf)
                buf = []
                if pause_s:
                    time.sleep(pause_s)
        if buf:
            client.upsert(collection, points=buf)
            total += len(buf)
    return total


def verify_tombstones(manifest: dict, manifest_dir: pathlib.Path) -> int:
    """Read and validate EVERY tombstone (file hash, then each line's id and `reason`) before
    anything is written. An unknown reason refuses the whole pack with one line and no partial
    apply (`specs/dump-format.md` §4). Returns the tombstone count."""
    tomb = manifest.get("tombstones")
    if not tomb or not tomb.get("count"):
        return 0
    path = manifest_dir / tomb["path"]
    got_sha = pack.sha256_file(str(path))
    if got_sha != tomb["sha256"]:
        raise ApplyRefusal(
            f"REFUSED: tombstones file hash mismatch (got {got_sha}, "
            f"manifest says {tomb['sha256']})")
    n = 0
    try:
        for _ in pack.read_tombstones(str(path)):
            n += 1
    except (ValueError, KeyError) as e:
        raise ApplyRefusal(
            f"REFUSED: tombstone {n + 1} of {path.name} is not valid ({e}); nothing was written"
        ) from e
    return n


def apply_tombstones(client, collection: str, manifest: dict,
                     manifest_dir: pathlib.Path, *,
                     batch: int = DEFAULT_BATCH) -> int:
    """A tombstone is a MANDATORY, forced delete — never optional, never
    conditional on the id being present (Qdrant's delete is a no-op on a
    missing id, which is exactly the semantics a tombstone needs: "this id
    must not exist afterward", not "this id must have existed before")."""
    from qdrant_client.http import models as qmodels

    tomb = manifest.get("tombstones")
    if not tomb or not tomb.get("count"):
        return 0
    path = manifest_dir / tomb["path"]
    got_sha = pack.sha256_file(str(path))
    if got_sha != tomb["sha256"]:
        raise ApplyRefusal(
            f"REFUSED: tombstones file hash mismatch (got {got_sha}, "
            f"manifest says {tomb['sha256']})")
    ids: list[str] = []
    n = 0
    for t in pack.read_tombstones(str(path)):
        ids.append(t.id)
        if len(ids) >= batch:
            client.delete(collection, points_selector=qmodels.PointIdsList(points=ids))
            n += len(ids)
            ids = []
    if ids:
        client.delete(collection, points_selector=qmodels.PointIdsList(points=ids))
        n += len(ids)
    return n


# --------------------------------------------------------------------- state
def load_state(path: pathlib.Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def save_state(path: pathlib.Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False))


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="cyt apply",
        description="apply a public dump or delta pack to your own Qdrant "
                     "— specs/dump-format.md §7")
    ap.add_argument("--manifest", required=True,
                     help="path to a dump/pack directory, or its manifest.json")
    ap.add_argument("--qdrant-url", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--root-pubkey", default=None,
                     help="the publisher's root minisign public key string "
                          "(published in the README and NOTICE, never inside "
                          "the dump); required for a signed dump. Today the given key "
                          "verifies the manifest signature directly (no keys.json "
                          "chain yet)")
    ap.add_argument("--pubkey", default=None, help="old name of --root-pubkey")
    ap.add_argument("--allow-unsigned", action="store_true",
                     help="accept a dump that carries NO signature (refused on a "
                          "signed dump) — never for a real release, only "
                          "local/CI rehearsal. \"Signed\" is decided by the "
                          "manifest's own `signature` field, which a mirror can "
                          "strip; the defence is to give the key and never pass "
                          "this flag for a release")
    ap.add_argument("--recreate", action="store_true",
                     help="drop and recreate the collection if it already "
                          "exists (full-dump mode only)")
    ap.add_argument("--legacy-layout", action="store_true",
                     help="a manifest without a serving-profile block (a dump from "
                          "before slice 8 step 1) applies only with this flag: it "
                          "creates the OLD bare layout (fp32, no quantizer, no payload "
                          "indexes)")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--pause-s", type=float, default=DEFAULT_PAUSE_S)
    ap.add_argument("--state-file", default=DEFAULT_STATE_FILE,
                     help="tracks which dump/pack this Qdrant instance has "
                          "applied, so a later delta can check its base "
                          "and its sequence number")
    args = ap.parse_args(argv)
    if args.root_pubkey and args.pubkey and args.root_pubkey != args.pubkey:
        raise ApplyRefusal("REFUSED: --root-pubkey and --pubkey were both given "
                           "with different values; give one.")
    args.pubkey = args.root_pubkey or args.pubkey
    if args.pubkey and args.allow_unsigned:
        raise ApplyRefusal("REFUSED: --allow-unsigned cannot be combined with "
                           "--root-pubkey: giving a key means you expect a signed release.")
    # task 864 (review 869 S1): from an aw_e10 T0 to the flip only the drain writes the twin
    from cytir.index import twin_write_hold as _TWH
    _TWH.refuse_cli_write(args, tool='cytir/distribute/apply.py')

    from cytir.core.qdrant import qdrant_client  # lazy: CLI-only dependency

    manifest_path = pathlib.Path(args.manifest)
    if manifest_path.is_dir():
        manifest_dir = manifest_path
        manifest_path = manifest_path / "manifest.json"
    else:
        manifest_dir = manifest_path.parent
    manifest_bytes = manifest_path.read_bytes()   # read once: verify and parse the same bytes
    manifest = pack.validate_manifest(json.loads(manifest_bytes))

    verification = verify_dump_signature(manifest, manifest_dir, pubkey=args.pubkey,
                                         allow_unsigned=args.allow_unsigned,
                                         manifest_bytes=manifest_bytes)
    if verification == "unverified":
        print("WARNING: applying an UNSIGNED dump (--allow-unsigned); "
              "recorded as `unverified` in the state file", file=sys.stderr)
    verify_shard_hashes(manifest, manifest_dir)
    idf_block = verify_idf_snapshot(manifest, manifest_dir)

    state_path = pathlib.Path(args.state_file)
    state = load_state(state_path)
    is_delta = manifest["schema"] == pack.DELTA_PACK_SCHEMA
    # Every refusal that needs no server runs before the client is built.
    # Only "collection exists" must ask the server and stays below.
    if is_delta:
        if state.get("applied_manifest_sha256") != manifest["base_manifest_sha256"]:
            raise ApplyRefusal(
                f"REFUSED: this pack's base "
                f"({manifest['base_manifest_sha256']}) does not match "
                f"what {state_path} recorded as last applied "
                f"({state.get('applied_manifest_sha256')}). Apply packs "
                f"in order, starting from a full dump.")
        expected_seq = state.get("sequence", 0) + 1
        if manifest["sequence"] != expected_seq:
            raise ApplyRefusal(
                f"REFUSED: pack sequence {manifest['sequence']} != "
                f"expected {expected_seq} (last applied: "
                f"{state.get('sequence', 0)}). Packs must be applied in "
                f"order.")
        verify_tombstones(manifest, manifest_dir)   # every reason, before any write
    client = qdrant_client(args.qdrant_url, timeout=300)
    try:
        receipt = None  # the layout receipt; a delta creates no collection
        if is_delta:
            if not client.collection_exists(args.collection):
                raise ApplyRefusal(
                    f"REFUSED: collection {args.collection!r} does not "
                    f"exist — apply the full dump this pack is based on "
                    f"first.")
        else:
            receipt = create_collection(client, args.collection, manifest,
                                        recreate=args.recreate,
                                        legacy_layout=args.legacy_layout)

        before = client.count(args.collection, exact=True).count
        n_upserted = upsert_shards(client, args.collection, manifest, manifest_dir,
                                   batch=args.batch, pause_s=args.pause_s)
        n_tombstoned = apply_tombstones(
            client, args.collection, manifest, manifest_dir,
            batch=args.batch) if is_delta else 0
        after = client.count(args.collection, exact=True).count

        save_state(state_path, {
            "collection": args.collection,
            "applied_dump_id": manifest.get("pack_id", manifest.get("dump_id")),
            "applied_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "sequence": manifest.get("sequence", 0),
            "verification": verification,
            **({"idf_snapshot_sha256": idf_block["sha256"]} if idf_block
               else {k: v for k, v in state.items() if k == "idf_snapshot_sha256"}),
        })

        print(json.dumps({
            "collection": args.collection, "mode": "delta" if is_delta else "full",
            "collection_layout": receipt, "points_before": before, "points_after": after,
            "rows_upserted": n_upserted, "tombstones_applied": n_tombstoned,
            "idf_snapshot": ({"verified": True, "n": idf_block["n"],
                              "vocabulary": idf_block["vocabulary"]}
                             if idf_block else None),
        }, indent=2))
        return 0
    finally:
        # A local/on-disk Qdrant (self-hoster test rigs, the CI rehearsal)
        # holds an exclusive file lock on its storage dir for the client's
        # lifetime -- an un-closed client here would leave the NEXT `cyt
        # apply` invocation against the same path unable to open it.
        client.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ApplyRefusal as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
