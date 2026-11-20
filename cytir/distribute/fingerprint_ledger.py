"""Master-side fingerprint ledger (task 2586, option B of specs/delta-fingerprint-base.md).

The ledger holds `id -> fingerprint` for the exact state a stranger has after the
last PUBLISHED pack, so delta #2 has a base. It is a cache of signed artifacts:
`rebuild_by_fold` rebuilds it from the packs, and `check_rebuild` proves the
stored copy equals the fold. It NEVER ships: it lives under the master's work
root (`<work_root>/fingerprint-ledger/<line>/`), not in a dump directory, not
under `data/`, not in a torrent file list.

On disk (per line):
  state.json                  small index; written LAST, atomically, = the commit point
  fingerprints.<seq>.<sha16>.tsv.gz   the dump's own sidecar format (sorted by point id,
                              zero gzip mtime), written by `pack.FingerprintWriter`

The fingerprint is `pack.row_fingerprint`, imported, not copied: byte-identical
to a FULL dump's sidecar. Not imported by the drain (`promote_queue`).
"""
from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from cytir.distribute import pack

LEDGER_SCHEMA = "aw.fingerprint-ledger.v1"
STATE_NAME = "state.json"
LEDGER_DIRNAME = "fingerprint-ledger"


class LedgerError(RuntimeError):
    """Base: the ledger cannot give a trustworthy base. Never fall back."""


class LedgerMissing(LedgerError):
    """No state, or the sidecar the state names is gone."""


class LedgerCorrupt(LedgerError):
    """The sidecar does not match the state (sha, count) or the state is malformed."""


class LedgerStale(LedgerError):
    """The ledger describes another manifest than the last PUBLISHED one."""


class LedgerRebased(LedgerError):
    """The ledger descends from another FULL dump than the current line."""


class LedgerChainError(LedgerError):
    """A pack does not extend the ledger (base sha or sequence mismatch)."""


class LedgerLocationError(LedgerError):
    """The ledger root would sit somewhere that ships."""


@dataclass(frozen=True)
class Base:
    """What a delta builder needs: feed `path`/`count` to `pack.iter_fingerprints`."""
    path: str
    count: int
    sha256: str
    pack_manifest_sha256: str
    sequence: int


def default_root(line: str) -> pathlib.Path:
    from cytir.core import paths  # noqa: PLC0415

    return pathlib.Path(paths.work_path(LEDGER_DIRNAME, line))


def _check_location(root: pathlib.Path) -> None:
    root = root.resolve()
    for d in (root, *root.parents):
        if (d / "manifest.json").exists():
            raise LedgerLocationError(f"{d} holds a manifest.json: that is a dump directory")
    app = pathlib.Path(__file__).resolve().parents[2]
    for shipped in (app / "data", app / "cytir"):
        if root == shipped or shipped in root.parents:
            raise LedgerLocationError(f"{root} is under {shipped}: the ledger must never ship")


class Ledger:
    def __init__(self, root: os.PathLike | str):
        self.root = pathlib.Path(root)
        _check_location(self.root)

    @property
    def state_path(self) -> pathlib.Path:
        return self.root / STATE_NAME

    def exists(self) -> bool:
        return self.state_path.exists()

    def read_state(self) -> dict:
        try:
            raw = self.state_path.read_text()
        except FileNotFoundError:
            raise LedgerMissing(f"no ledger state at {self.state_path}") from None
        try:
            st = json.loads(raw)
        except ValueError as e:
            raise LedgerCorrupt(f"{self.state_path} is not JSON: {e}") from e
        need = ("schema", "algo", "base_full_manifest_sha256", "pack_manifest_sha256",
                "sequence", "count", "sidecar", "sidecar_sha256")
        if (not isinstance(st, dict) or st.get("schema") != LEDGER_SCHEMA
                or any(k not in st for k in need)):
            raise LedgerCorrupt(f"{self.state_path} is malformed")
        if st["algo"] != pack.FINGERPRINT_ALGO:
            raise LedgerCorrupt(f"ledger algo {st['algo']!r} != {pack.FINGERPRINT_ALGO!r}")
        return st

    def sidecar_path(self, st: Optional[dict] = None) -> pathlib.Path:
        return self.root / (st or self.read_state())["sidecar"]

    def verify(self) -> dict:
        """State + sidecar agree (sha256 and row count). Returns the state."""
        st = self.read_state()
        path = self.sidecar_path(st)
        if not path.exists():
            raise LedgerMissing(f"ledger sidecar {path} is missing")
        got = pack.sha256_file(str(path))
        if got != st["sidecar_sha256"]:
            raise LedgerCorrupt(f"ledger sidecar sha256 {got} != state {st['sidecar_sha256']}")
        try:
            for _ in pack.iter_fingerprints(str(path), expected_count=st["count"]):
                pass
        except (ValueError, OSError, EOFError) as e:
            raise LedgerCorrupt(f"ledger sidecar unreadable: {e}") from e
        return st

    def items(self) -> Iterator[tuple[str, str]]:
        st = self.read_state()
        for id_, fp, _ in pack.iter_fingerprints(str(self.sidecar_path(st)),
                                                 expected_count=st["count"]):
            yield id_, fp

    def _commit(self, tmp_sidecar: pathlib.Path, block: dict, *, base_full: str,
                pack_sha: str, sequence: int) -> dict:
        # the file name carries the manifest sha, so a new FULL (sequence 0 again)
        # never overwrites the sidecar the committed state still names
        name = f"fingerprints.{sequence}.{pack_sha[:16]}.tsv.gz"
        with open(tmp_sidecar, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp_sidecar, self.root / name)
        st = {"schema": LEDGER_SCHEMA, "algo": pack.FINGERPRINT_ALGO,
              "base_full_manifest_sha256": base_full, "pack_manifest_sha256": pack_sha,
              "sequence": sequence, "count": block["count"], "sidecar": name,
              "sidecar_sha256": block["sha256"]}
        tmp_state = self.root / (STATE_NAME + ".tmp")
        tmp_state.write_text(json.dumps(st, sort_keys=True, indent=1) + "\n")
        with open(tmp_state, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp_state, self.state_path)  # the commit point
        dfd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        for stray in self.root.glob("fingerprints.*.tsv.gz"):
            if stray.name != name:
                stray.unlink(missing_ok=True)
        return st


def _fp_rows(rows: Iterable[dict]) -> Iterator[tuple[str, str]]:
    """(id, fp) in strictly increasing point-id order; refuses anything else."""
    last = None
    for row in rows:
        key = pack.point_id_key(row["id"])
        if last is not None and not key > last:
            raise pack.FingerprintOrderError(
                f"pack rows are not strictly increasing at {row['id']!r}")
        last = key
        yield row["id"], pack.row_fingerprint(row)


def update_from_pack(ledger: Ledger, manifest: dict, manifest_sha256: str,
                     rows: Iterable[dict], tombstone_ids: Iterable[str] = ()) -> dict:
    """Advance the ledger by one PUBLISHED pack. `rows` are the pack's shard rows
    (payload as the JSON string the shard holds), in point-id order.
    FULL pack: replaces the ledger (a rebase). DELTA pack: must name the ledger's
    current manifest and be sequence + 1. Re-applying the pack the ledger already
    describes is a no-op (repairs a crash before the caller's own marker moved).
    A delta tombstone for an id the ledger never held, or a FULL whose declared
    `fingerprints` block differs from the ledger's own, is refused.
    Not locked: run it under the drain single-flight lock (delta-change-log 3.4).
    Returns {"state", "noop"}."""
    pack.validate_manifest(manifest)
    ledger.root.mkdir(parents=True, exist_ok=True)
    tmp = ledger.root / "fingerprints.tmp.tsv.gz"
    if manifest["schema"] == pack.DUMP_MANIFEST_SCHEMA:
        base_full, seq = manifest_sha256, int(manifest.get("sequence", 0))
        cur = None
        if ledger.exists():
            try:
                cur = ledger.read_state()
            except LedgerCorrupt:
                cur = None  # a new FULL replaces the ledger: it repairs a garbage state.json
        if cur is not None:
            if cur["base_full_manifest_sha256"] == manifest_sha256 and cur["sequence"] > 0:
                raise LedgerChainError(
                    f"replaying the base FULL {manifest_sha256} would rewind the ledger "
                    f"from sequence {cur['sequence']} to 0")
        declared = manifest.get("fingerprints")
        if not declared:
            raise LedgerCorrupt("FULL manifest has no `fingerprints` block (pre-task-687 dump)")
        w = pack.FingerprintWriter(str(tmp))
        for id_, fp in _fp_rows(rows):
            w.add(id_, fp)
        block = w.close()
        if (declared.get("algo") != block["algo"] or declared.get("sha256") != block["sha256"]
                or declared.get("count") != block["count"]):
            raise LedgerCorrupt(
                f"FULL {manifest_sha256}: the ledger's own fingerprints (algo {block['algo']}, "
                f"sha256 {block['sha256']}, count {block['count']}) differ from the dump's "
                f"declared block {declared}")
    else:
        st = ledger.verify()  # a corrupt or missing ledger refuses here
        if st["pack_manifest_sha256"] == manifest_sha256:
            return {"state": st, "noop": True}
        if manifest["base_manifest_sha256"] != st["pack_manifest_sha256"]:
            raise LedgerChainError(
                f"pack base {manifest['base_manifest_sha256']} != ledger "
                f"{st['pack_manifest_sha256']}")
        if int(manifest["sequence"]) != st["sequence"] + 1:
            raise LedgerChainError(
                f"pack sequence {manifest['sequence']} != ledger {st['sequence']} + 1")
        base_full, seq = st["base_full_manifest_sha256"], int(manifest["sequence"])
        tomb = set(tombstone_ids)
        new = list(_fp_rows(rows))  # one pack's changed rows; the base streams
        clash = tomb & {i for i, _ in new}
        if clash:
            raise LedgerChainError(
                f"ids both changed and tombstoned in one pack: {sorted(clash)[:3]}")
        w = pack.FingerprintWriter(str(tmp))
        new_it = iter(new)
        nxt = next(new_it, None)
        seen_tomb = set()
        for id_, fp, key in pack.iter_fingerprints(str(ledger.sidecar_path(st)),
                                                   expected_count=st["count"]):
            while nxt is not None and pack.point_id_key(nxt[0]) < key:
                w.add(*nxt)
                nxt = next(new_it, None)
            if nxt is not None and nxt[0] == id_:
                w.add(*nxt)
                nxt = next(new_it, None)
            elif id_ in tomb:
                seen_tomb.add(id_)
            else:
                w.add(id_, fp)
        while nxt is not None:
            w.add(*nxt)
            nxt = next(new_it, None)
        block = w.close()
        unknown = sorted(tomb - seen_tomb)
        if unknown:
            raise LedgerChainError(
                f"pack {manifest_sha256}: {len(unknown)} tombstone(s) name ids the ledger "
                f"never held {unknown[:5]}: the pack was built against a different base")
    return {"state": ledger._commit(tmp, block, base_full=base_full,
                                    pack_sha=manifest_sha256, sequence=seq),
            "noop": False}


def base_for_delta(ledger: Ledger, *, expected_manifest_sha256: str,
                   expected_full_manifest_sha256: Optional[str] = None,
                   since_seq: Optional[int] = None) -> Base:
    """The base a delta builder diffs against. Refuses (named error) if the
    ledger is missing or corrupt, is not at the last PUBLISHED manifest, or
    descends from another full dump than `expected_full_manifest_sha256` (a
    rebase). `since_seq`, if given, must equal the ledger's sequence."""
    st = ledger.verify()
    if st["pack_manifest_sha256"] != expected_manifest_sha256:
        raise LedgerStale(
            f"ledger describes manifest {st['pack_manifest_sha256']} (seq {st['sequence']}), "
            f"last published is {expected_manifest_sha256}")
    if (expected_full_manifest_sha256 is not None
            and st["base_full_manifest_sha256"] != expected_full_manifest_sha256):
        raise LedgerRebased(
            f"ledger descends from full {st['base_full_manifest_sha256']}, "
            f"current line is {expected_full_manifest_sha256}")
    if since_seq is not None and st["sequence"] != since_seq:
        raise LedgerStale(f"ledger is at sequence {st['sequence']}, builder asked {since_seq}")
    return Base(str(ledger.sidecar_path(st)), st["count"], st["sidecar_sha256"],
                st["pack_manifest_sha256"], st["sequence"])


# ------------------------------------------------------------------- fold
def iter_shard_rows(path: str) -> Iterator[dict]:
    """Raw rows: the payload stays the shard's JSON string, which is what is hashed."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    for batch in pq.ParquetFile(path).iter_batches():
        for row in batch.to_pylist():
            yield dict(row)


def pack_rows(pack_dir: pathlib.Path, manifest: dict) -> Iterator[dict]:
    for shard in manifest.get("shards", []):
        yield from iter_shard_rows(str(pack_dir / shard["path"]))


def pack_tombstones(pack_dir: pathlib.Path, manifest: dict) -> list[str]:
    t = manifest.get("tombstones")
    if not t:
        return []
    return [x.id for x in pack.read_tombstones(str(pack_dir / t["path"]))]


def chain_of(packs_dir: os.PathLike | str) -> list[tuple[pathlib.Path, dict, str]]:
    """The newest FULL dump and its delta chain (each names the previous manifest
    sha, sequence + 1), in apply order, as (dir, manifest, manifest sha256).
    Refuses a gap or a fork."""
    root = pathlib.Path(packs_dir)
    found = []
    for d in sorted(p for p in root.iterdir() if (p / "manifest.json").is_file()):
        mp = d / "manifest.json"
        found.append((d, pack.validate_manifest(json.loads(mp.read_text())),
                      pack.sha256_file(str(mp))))
    fulls = [f for f in found if f[1]["schema"] == pack.DUMP_MANIFEST_SCHEMA]
    if not fulls:
        raise LedgerMissing(f"no full dump under {packs_dir}")
    chain = [max(fulls, key=lambda f: (f[1]["created_at"], f[2]))]
    by_base: dict[str, list] = {}
    for f in found:
        if f[1]["schema"] == pack.DELTA_PACK_SCHEMA:
            by_base.setdefault(f[1]["base_manifest_sha256"], []).append(f)
    while True:
        nxt = by_base.get(chain[-1][2], [])
        if len(nxt) > 1:
            raise LedgerChainError(f"fork: {len(nxt)} deltas name manifest {chain[-1][2]}")
        if not nxt:
            return chain
        want = int(chain[-1][1].get("sequence", 0)) + 1
        if int(nxt[0][1]["sequence"]) != want:
            raise LedgerChainError(f"sequence gap: got {nxt[0][1]['sequence']}, want {want}")
        chain.append(nxt[0])


def advance_from_pack_dir(ledger: Ledger, pack_dir: os.PathLike | str) -> dict:
    """Advance the ledger from a signed pack ON DISK: the manifest's sha256 is the
    file's, rows and tombstones are re-read from the shards. For the dump hook,
    where the row generator is already consumed. Same return as `update_from_pack`."""
    d = pathlib.Path(pack_dir)
    mp = d / "manifest.json"
    manifest = pack.validate_manifest(json.loads(mp.read_text()))
    return update_from_pack(ledger, manifest, pack.sha256_file(str(mp)),
                            pack_rows(d, manifest), pack_tombstones(d, manifest))


def rebuild_by_fold(packs_dir: os.PathLike | str, ledger: Ledger,
                    upto_sequence: Optional[int] = None) -> dict:
    """Rebuild `ledger` from signed artifacts only: the full's rows re-hashed, then
    each delta's rows re-hashed and its tombstones removed. Returns the state."""
    state = None
    for d, manifest, sha in chain_of(packs_dir):
        if (upto_sequence is not None and manifest["schema"] == pack.DELTA_PACK_SCHEMA
                and int(manifest["sequence"]) > upto_sequence):
            break
        state = advance_from_pack_dir(ledger, d)["state"]
    return state


def check_rebuild(packs_dir: os.PathLike | str, ledger: Ledger,
                  scratch: Optional[os.PathLike | str] = None) -> None:
    """The invariant: fold(packs) == stored, byte for byte (sidecar sha + state).
    The fold runs in a fresh temporary directory; a `scratch` you name must be absent
    or empty (a non-empty one would be folded INTO, and a stale state would pass)."""
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    stored = ledger.verify()
    own = scratch is None
    if own:
        scratch = tempfile.mkdtemp(prefix="aw-ledger-check-")
    else:
        sp = pathlib.Path(scratch)
        if sp.exists() and any(sp.iterdir()):
            raise LedgerLocationError(f"scratch {sp} is not empty: refusing to fold into it")
    try:
        rebuilt = rebuild_by_fold(packs_dir, Ledger(scratch))
    finally:
        if own:
            shutil.rmtree(scratch, ignore_errors=True)
    if rebuilt == stored:
        return
    if rebuilt["sequence"] > stored["sequence"]:
        raise LedgerStale(f"the ledger lags the signed packs: sequence {stored['sequence']} "
                          f"< {rebuilt['sequence']}; repair with rebuild_by_fold")
    raise LedgerCorrupt(f"fold rebuild differs from the stored ledger: {rebuilt} != {stored}")
