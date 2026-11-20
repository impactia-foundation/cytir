"""Cut a big sealed pack into drain-sized sub-packs WITHOUT copying or re-sealing
a byte (task 680, 2026-09-30).

WHY THIS EXISTS
---------------
The drain loads a pack's whole join (texts + payloads + the float matrix) in
memory, and one pack is one promotion transaction. The ten OpenAlex packs are
2.5M, 1.0M, 0.9M, ... rows -- six of them exceed even the ADR-083 mode-1 cap on
their own, and ERIC-sized packs (~160k) are what the 30 GB box has been taking
(task 629 §6.1). The sealed pack is the preservation unit, so it is not rewritten:
a sub-pack is a directory of symlinks to a run of its shards.

A shard is the natural cut. Chunk shard `shard-NNNN.texts.jsonl.gz` and vector
shard `shard-NNNN.{vectors.f16,vectors.manifest.json,row_ids.jsonl}` carry the
same 50,000 point ids in the same order (measured on all ten OpenAlex packs,
task 680: 136 of 136 shard tags paired), which is what makes a run of shards a
complete, self-contained pack. Anything else -- a tag with no partner, a tag
whose id sets differ -- refuses (engineering-doctrine #3): a sub-pack that
silently dropped a vector would store rows without their embeddings.

Symlinks are for the drain's reads only. They are NOT an ADR-105 persisted-input
store (`persisted_inputs` refuses to index a symlink) and nothing here files
text; the pages' text lives in the L1 tree the registry already globs.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

CHUNK_SUFFIX = ".texts.jsonl.gz"
VECTOR_SUFFIXES = (".vectors.f16", ".vectors.manifest.json", ".row_ids.jsonl")


class SubpackError(RuntimeError):
    """The shards cannot be cut into a complete sub-pack."""


@dataclass(frozen=True)
class Subpack:
    index: int
    tags: tuple[str, ...]
    rows: int


def _chunk_tags(chunks_dir: Path) -> list[str]:
    return sorted(p.name.removesuffix(CHUNK_SUFFIX)
                  for p in chunks_dir.glob("shard-*" + CHUNK_SUFFIX))


def _vector_rows(vectors_dir: Path, tag: str) -> int:
    man = vectors_dir / f"{tag}.vectors.manifest.json"
    if not man.is_file():
        raise SubpackError(f"{vectors_dir}: no vector manifest for {tag}")
    return int(json.loads(man.read_text())["n"])


def plan_subpacks(chunks_dir: str, vectors_dir: str,
                  shards_per_pack: int) -> list[Subpack]:
    """Consecutive runs of `shards_per_pack` shards. Refuses when the chunk and
    vector shard tags are not the same set."""
    if shards_per_pack < 1:
        raise SubpackError("shards_per_pack must be >= 1")
    c, v = Path(chunks_dir), Path(vectors_dir)
    tags = _chunk_tags(c)
    if not tags:
        raise SubpackError(f"{chunks_dir}: no shard-*{CHUNK_SUFFIX}")
    vtags = sorted(p.name.removesuffix(".vectors.manifest.json")
                   for p in v.glob("shard-*.vectors.manifest.json"))
    if tags != vtags:
        only_c = sorted(set(tags) - set(vtags))[:3]
        only_v = sorted(set(vtags) - set(tags))[:3]
        raise SubpackError(
            f"chunk shards and vector shards are not the same set "
            f"(chunks only: {only_c}, vectors only: {only_v}) -- a run of "
            "shards would not be a complete pack")
    out = []
    for i in range(0, len(tags), shards_per_pack):
        run = tuple(tags[i:i + shards_per_pack])
        out.append(Subpack(index=i // shards_per_pack, tags=run,
                           rows=sum(_vector_rows(v, t) for t in run)))
    return out


def _link(src: Path, dst: Path) -> None:
    if not src.exists():
        raise SubpackError(f"{src} does not exist")
    if dst.is_symlink():
        if os.readlink(dst) == str(src):
            return                      # idempotent re-run
        dst.unlink()
    elif dst.exists():
        raise SubpackError(f"{dst} exists and is not a symlink -- refusing to "
                           "overwrite real data with a link")
    dst.symlink_to(src)


def materialize_subpack(chunks_dir: str, vectors_dir: str, sub: Subpack,
                        out_dir: str) -> tuple[str, str]:
    """`<out_dir>/chunks` and `<out_dir>/vectors`: symlinks to `sub`'s shards.
    Returns the two paths. Safe to re-run."""
    c, v, o = Path(chunks_dir), Path(vectors_dir), Path(out_dir)
    (o / "chunks").mkdir(parents=True, exist_ok=True)
    (o / "vectors").mkdir(parents=True, exist_ok=True)
    for tag in sub.tags:
        _link(c / f"{tag}{CHUNK_SUFFIX}", o / "chunks" / f"{tag}{CHUNK_SUFFIX}")
        for suffix in VECTOR_SUFFIXES:
            _link(v / f"{tag}{suffix}", o / "vectors" / f"{tag}{suffix}")
    return str(o / "chunks"), str(o / "vectors")
