#!/usr/bin/env python3
# Moved from the pipeline-era `normalize_sealed_pack_shape` module (task 677, task 643 S5); the old path was removed (task 683 S9).
"""Reshape a sealed pack's text/metadata rows to match one vector shard set.

Task 641 (growth lane). `promote_sealed_pack.py` reads exactly one artefact
shape: a `chunks_dir` of `shard-*.texts.jsonl.gz` whose `point_id` set is
EXACTLY the `vectors_dir`'s `row_ids.jsonl` id set -- no more, no fewer (its
own `plan()` refuses a partial join rather than reshape or truncate). Two
already-sealed families do not arrive in that shape, for two different
reasons, and this is the one reader that fixes both -- not a new family of
one-off scripts per family:

1. **`cc_authority_frontier_w7` (CC tranche 1, ~205k rows).** The re-embed
   that fixed its vector dtype (log.md 2026-09-22 19:42Z) staged only
   `{point_id, text}` per row into `reembed_1024fp16/stage/` -- exactly what
   the embedder needs and NOTHING else, so `url`/`title`/`domain`/
   `source_type`/`trust_tier` are gone from that shape. The ORIGINAL packs
   (`packs/tranche1/<cell>-pack-<NNNN>.jsonl`) carry all of it, keyed by the
   same `point_id`. This is a MERGE: text from the re-embed shards (it is
   the text that was actually embedded -- the vector and the stored text
   must agree), every other field from the original packs.

2. **`de_openlegaldata_exact` (OLD, ~34k rows).** One staged chunk shard
   (`stages/aw-task633-openlegaldata-v1/chunks-shard0/chunks/shard-0000.
   texts.jsonl.gz`, 34,323 rows, full metadata) was split AFTER embedding
   into two vector parts (`vector_rows` 19,999 + 14,324, a pack-size-cap
   split at a document boundary, `data/expansion-pack-admission-status.
   v1.json#de-openlegaldata-shard0-part{1,2}`). The one chunk shard now
   over-covers either vectors part alone. This is a SELECT: every field
   from the one source, restricted to one vectors part's id set.

Both are `--text-from`/`--metadata-from` of the SAME operation: read every
`shard-*.texts.jsonl.gz` under `--metadata-from` (and, if given, a SEPARATE
`--text-from`) into one `point_id -> row` map each; for every id the target
`--vectors` shard set declares, look up its metadata (and its text, from
`--text-from` if given, else from the metadata map itself); the merged row
IS the metadata row with `text` set to the text-source row's `text`. Refuses
--- never truncates or pads --- on any id the vectors declare that a source
does not, and (a WARNING, not a refusal: a source may legitimately cover a
wider population than one vectors part) reports how many metadata ids were
never selected.

DEFAULT IS A DRY RUN: prints the counts and writes nothing until --apply.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
from typing import Iterator

ROOT = pathlib.Path(__file__).resolve().parents[2]

from cytir.gates import mechanical_content_check as MCC  # noqa: E402 -- backfill_lang, ONE home
from cytir.index import promote_sealed_pack as PSP  # noqa: E402 -- read_vectors/SEALED_DIM, ONE home


class NormalizeError(RuntimeError):
    """A refusal. Never caught here."""


def _shard_rows(chunks_dir: str) -> Iterator[dict]:
    """Every row of every `.jsonl`/`.jsonl.gz` file directly under `chunks_dir`.

    This is a SOURCE-side reader, deliberately more permissive than
    `promote_sealed_pack.read_records` (which insists on `shard-*.texts.
    jsonl.gz` because THAT shape is the one contract every downstream
    consumer relies on). This module's whole job is normalizing shapes that
    are not yet that contract -- CC tranche 1's original packs are plain,
    uncompressed `<cell>-pack-<NNNN>.jsonl`, never gzipped and never named
    `shard-*` -- so the source reader has to read what upstream actually
    produced. The OUTPUT this module writes (`write_shard`) still goes
    through the one real contract; only the input side is widened.
    """
    import gzip as _gzip
    paths = sorted(pathlib.Path(chunks_dir).glob("*.jsonl.gz")) + \
        sorted(pathlib.Path(chunks_dir).glob("*.jsonl"))
    if not paths:
        raise NormalizeError(
            f"no *.jsonl or *.jsonl.gz file directly under {chunks_dir}")
    for path in paths:
        opener = _gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_id_map(chunks_dir: str) -> dict[str, dict]:
    """`point_id -> row` over every row under `chunks_dir`.

    Refuses a duplicate id within the SAME source (a pack defect, not
    something to silently pick one of -- `promote_sealed_pack.project_rows`
    makes the identical refusal for the same reason at a later stage).
    """
    out: dict[str, dict] = {}
    for row in _shard_rows(chunks_dir):
        pid = row.get("point_id")
        if not pid:
            raise NormalizeError(
                f"{chunks_dir}: a row carries no point_id -- nothing to key "
                "this join on")
        if pid in out:
            raise NormalizeError(
                f"{chunks_dir}: duplicate point_id {pid} -- two rows minting "
                "one id is a pack defect, not something to de-duplicate here")
        out[pid] = row
    return out


def target_ids(vectors_dir: str, mrl: int) -> list[str]:
    """The exact id set `promote_sealed_pack.read_vectors` will address.

    `mrl` does not change WHICH ids exist, only their truncated width, so
    any legal value reads the same id set -- 256 is used here only because
    it is always <= the sealed 1024-d width and therefore always a legal
    truncation to ask `read_vectors` for.
    """
    _matrix, ids = PSP.read_vectors(vectors_dir, min(mrl, PSP.SEALED_DIM))
    return ids


def merge(*, metadata_from: list[str], text_from: str | None,
         vectors_dir: str, mrl: int,
         backfill_lang: bool = False) -> tuple[list[dict], dict]:
    """Return (merged rows in vector-shard order, a report dict).

    `backfill_lang`: after the merge, run `mechanical_content_check.
    backfill_lang` over the merged rows -- fills an EMPTY `lang` from the
    same tracked detector the language gate uses, never touches a row that
    already declares one, records `lang_source` on every row it fills
    (orchestrator ruling 2026-09-23, CC tranche 1, task 652). OFF by
    default: normalizing a pack's SHAPE and backfilling a payload field are
    two different operations, and a caller that does not ask for the
    backfill must get exactly the merge it asked for.
    """
    meta: dict[str, dict] = {}
    for src in metadata_from:
        chunk = load_id_map(src)
        overlap = set(chunk) & set(meta)
        if overlap:
            raise NormalizeError(
                f"{src}: {len(overlap)} id(s) already seen in an earlier "
                "--metadata-from source (e.g. "
                f"{sorted(overlap)[:3]}) -- metadata sources must be "
                "disjoint, or one of them is not the source it claims to be")
        meta.update(chunk)

    text_map: dict[str, dict] | None = load_id_map(text_from) if text_from else None

    ids = target_ids(vectors_dir, mrl)
    id_set = set(ids)
    missing_meta = sorted(id_set - set(meta))
    if missing_meta:
        raise NormalizeError(
            f"{len(missing_meta)} of {len(ids)} vector id(s) have no "
            f"metadata row (e.g. {missing_meta[:3]}) -- refusing a partial "
            "join rather than promoting a row with no url/title/licence")
    if text_map is not None:
        missing_text = sorted(id_set - set(text_map))
        if missing_text:
            raise NormalizeError(
                f"{len(missing_text)} of {len(ids)} vector id(s) have no row "
                f"in --text-from (e.g. {missing_text[:3]}) -- refusing "
                "rather than storing a vector beside no text at all")

    merged = []
    for pid in ids:
        row = dict(meta[pid])
        if text_map is not None:
            row["text"] = text_map[pid].get("text")
            if not row["text"]:
                raise NormalizeError(
                    f"{pid}: --text-from row carries no text -- the vector "
                    "for this id would be stored beside empty text")
        merged.append(row)

    unused_meta = sorted(set(meta) - id_set)
    report = {
        "vector_ids": len(ids),
        "metadata_sources": metadata_from,
        "metadata_rows_total": len(meta),
        "metadata_rows_unused": len(unused_meta),
        "metadata_rows_unused_sample": unused_meta[:5],
        "text_source": text_from,
        "merged_rows": len(merged),
    }
    if backfill_lang:
        merged, lang_report = MCC.backfill_lang(merged)
        report["lang_backfill"] = lang_report
    return merged, report


def write_shard(rows: list[dict], out_dir: pathlib.Path,
                tag: str = "shard-0000") -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.texts.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metadata-from", action="append", required=True,
                    help="dir(s) of shard-*.texts.jsonl.gz carrying "
                         "url/title/source_type/trust_tier/etc; repeatable, "
                         "must be id-disjoint across repeats")
    ap.add_argument("--text-from", default=None,
                    help="dir of shard-*.texts.jsonl.gz carrying the "
                         "AUTHORITATIVE embedded text, if different from "
                         "--metadata-from (omit when one source has both)")
    ap.add_argument("--vectors", required=True,
                    help="the vectors_dir whose row_ids.jsonl id set is the "
                         "target -- every id it declares must resolve")
    ap.add_argument("--mrl", type=int, default=256)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--backfill-lang", action="store_true",
                    help="fill an empty `lang` field from the tracked "
                         "language detector when confident and in scope "
                         "(orchestrator ruling 2026-09-23); never touches a "
                         "row that already declares a lang")
    ap.add_argument("--apply", action="store_true",
                    help="without this the run is a DRY RUN and writes nothing")
    args = ap.parse_args(argv)

    rows, report = merge(metadata_from=args.metadata_from,
                         text_from=args.text_from, vectors_dir=args.vectors,
                         mrl=args.mrl, backfill_lang=args.backfill_lang)
    print(f"[normalize] {json.dumps(report, indent=1)}")
    if args.apply:
        path = write_shard(rows, pathlib.Path(args.out_dir))
        print(f"[normalize] wrote {len(rows):,} rows -> {path}")
    else:
        print(f"[normalize] DRY RUN -- {len(rows):,} row(s) would be written "
             f"to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
