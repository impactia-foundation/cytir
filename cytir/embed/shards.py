#!/usr/bin/env python3
"""Embed staged chunk shards at FULL 1024 dims, store fp16 binary.

MOVED HERE from task 413a's `embed_shards.py` in task 657 (task 643
§6.1/§6.2): task 413a's `embed_shards.py` is now a shim
(`from cytir.embed.shards import *`) so the existing callers --
task 633's `run_embed_lease.sh`'s default `AW_EMBEDDER`,
task 417's `run_admitted_stage_embed.py`'s `--embedder` argument --
keep working untouched. New code calls `cytir.embed.stage.embed_stage()` or
invokes this module directly (`python -m cytir.embed.shards`).

The only behavioural change from the moved copy: `--model`/`--revision`
default from `cytir.core.model_pin` (`EMBED_MODEL_ID` /
`embed_model_revision()`) instead of restating the pin as a second literal
-- the exact fork `cytir/core/model_pin.py`'s own docstring exists to close
(task 432: "document vectors came from whatever HuggingFace served that
day" was the write-side defect that module fixed; this file was the other
un-pinned write path it had not yet reached). `cytir.core.model_pin.assert_pin`
is called once the model loads, so a caller who overrides `--model`/
`--revision` away from the committed pin gets a loud, named refusal instead
of a silent divergent vector geometry.

Runs on the rented/owned GPU box, one process per GPU, detached.  Reads
``shard-NNNN.texts.jsonl.gz`` and writes the task-413 §3 artifact contract:

    shard-NNNN.vectors.f16            raw row-major LE float16 (n,1024), NO header
    shard-NNNN.vectors.manifest.json  n, dim, dtype, byte_order, sha256, ordering,
                                      embedder model + PINNED + RESOLVED revision,
                                      tokenizer class (recorded, not inferred)
    shard-NNNN.row_ids.jsonl          {"i":k,"point_id":..} -- the ordering contract

The whole point of the run is the FULL width: any serving width is a later,
free truncation of this file (`cytir.core.qdrant`'s / `serving_config.py`'s
`mrl_trunc`, applied at promotion time). Nothing here chooses a serving width.

Resumable: a shard whose manifest exists AND whose .f16 matches the recorded
byte length and sha256 is skipped.  An interrupted shard restarts (only that
shard), because a partial .f16 is indistinguishable from a short one and we
refuse to guess.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np

from cytir.core.model_pin import EMBED_MODEL_ID, assert_pin, embed_model_revision

DIM = 1024


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def log(msg: str) -> None:
    sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), msg))
    sys.stderr.flush()


def load_model(args):
    from sentence_transformers import SentenceTransformer
    assert_pin(args.model, args.revision, where="cytir.embed.shards.load_model")
    kw = {"trust_remote_code": True, "device": args.device}
    if args.revision:
        kw["revision"] = args.revision
    model = SentenceTransformer(args.model, **kw)
    model.max_seq_length = args.max_seq
    # fp16 inference: the serving embedder (TEI) already runs fp16, and the
    # artifact is stored fp16 regardless, so this is a throughput choice and
    # not a fidelity one. It is recorded, not assumed -- crosscheck_mrl proves
    # the vectors still land on the corpus originals.
    if args.compute_dtype == "float16":
        model = model.half()
    native = model.get_sentence_embedding_dimension()
    if native != DIM:
        raise SystemExit("model native dim %d != %d; this run stores full width"
                         % (native, DIM))
    tok = model.tokenizer
    info = {
        "model": args.model,
        "revision_pinned": args.revision or None,
        "revision_resolved": resolve_revision(args.model, args.revision),
        "output_width": DIM,
        "max_seq_length": model.max_seq_length,
        "normalized": True,
        "compute_dtype": args.compute_dtype,
        "document_prompt": None,
        "tokenizer_class": type(tok).__name__,
        "tokenizer_is_fast": bool(getattr(tok, "is_fast", False)),
        "tokenizer_vocab_size": int(getattr(tok, "vocab_size", -1)),
        "torch": __import__("torch").__version__,
    }
    log("model loaded: %s" % json.dumps(info))
    return model, info


def resolve_revision(model_id: str, pinned: str) -> str | None:
    """MEASURED revision: read the commit the local snapshot actually is."""
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(model_id, revision=pinned or None,
                                 local_files_only=True)
        return os.path.basename(os.path.realpath(path))
    except Exception as exc:  # pragma: no cover - environment dependent
        log("revision resolve failed: %r" % (exc,))
        return None


def read_texts(path: str):
    """Return (ids, dense inputs). The input is the dense channel from
    `record_text_channels` (ADR-060), the same call the builders use; a row with
    no `dense_text` falls back to `text` and the shard logs the count once."""
    from cytir.core.common import record_text_channels
    ids, texts = [], []
    fallback = 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            ids.append(row["point_id"])
            if not row.get("dense_text"):
                fallback += 1
            texts.append(record_text_channels(row)[0])
    if fallback:
        log("%s: %d of %d rows have no dense_text; embedded `text` instead"
            % (os.path.basename(path), fallback, len(ids)))
    return ids, texts


def row_ids_match_texts(rid_path: str, texts_path: str, expected: int) -> bool:
    """Re-derive the exact staged ordering before trusting a resume marker."""
    count = 0
    with gzip.open(texts_path, "rt", encoding="utf-8") as texts, \
            open(rid_path, encoding="utf-8") as row_ids:
        for count, text_line in enumerate(texts, 1):
            rid_line = row_ids.readline()
            if not rid_line:
                return False
            text_row = json.loads(text_line)
            rid_row = json.loads(rid_line)
            if (rid_row.get("i") != count - 1
                    or rid_row.get("point_id") != text_row.get("point_id")):
                return False
        if row_ids.readline():
            return False
    return count == expected


def shard_done(vec_path: str, man_path: str, rid_path: str,
               texts_path: str, expected_embedder: dict | None = None) -> bool:
    if not all(os.path.exists(path) for path in
               (vec_path, man_path, rid_path, texts_path)):
        return False
    try:
        with open(man_path) as fh:
            man = json.load(fh)
        expect = man["n"] * DIM * 2
        recorded_embedder = man.get("embedder") or {}
        embedder_fields = (
            "model", "revision_pinned", "revision_resolved", "output_width",
            "max_seq_length", "normalized", "compute_dtype")
        if (man.get("schema") != "aw.413a.vectors.v1"
                or man.get("row_ids_file") != os.path.basename(rid_path)
                or os.path.getsize(vec_path) != expect
                or sha256_file(vec_path) != man.get("sha256")
                or sha256_file(rid_path) != man.get("row_ids_sha256")
                or sha256_file(texts_path) != man.get("texts_sha256")
                or (expected_embedder is not None and any(
                    recorded_embedder.get(field) != expected_embedder.get(field)
                    for field in embedder_fields))):
            return False
        return row_ids_match_texts(rid_path, texts_path, man["n"])
    except Exception:
        return False


def embed_shard(model, info, texts_path: str, outdir: str, args) -> dict:
    tag = os.path.basename(texts_path)[: -len(".texts.jsonl.gz")]
    vec_path = os.path.join(outdir, tag + ".vectors.f16")
    man_path = os.path.join(outdir, tag + ".vectors.manifest.json")
    rid_path = os.path.join(outdir, tag + ".row_ids.jsonl")

    if shard_done(vec_path, man_path, rid_path, texts_path, info):
        log("%s already complete, skipping" % tag)
        with open(man_path) as fh:
            man = json.load(fh)
        man["resumed"] = True
        return man

    ids, texts = read_texts(texts_path)
    n = len(ids)
    log("%s encoding n=%d" % (tag, n))
    t0 = time.time()

    tmp = vec_path + ".part"
    written = 0
    with open(tmp, "wb") as fout:
        for start in range(0, n, args.chunk):
            block = texts[start: start + args.chunk]
            vec = model.encode(
                block, batch_size=args.batch, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False)
            vec = np.asarray(vec, dtype=np.float32)
            if vec.shape != (len(block), DIM):
                raise SystemExit("%s: bad encode shape %r" % (tag, vec.shape))
            if not np.isfinite(vec).all():
                raise SystemExit("%s: non-finite values in encode output" % tag)
            fout.write(np.ascontiguousarray(
                vec.astype("<f2")).tobytes(order="C"))
            written += len(block)
            if start and (start // args.chunk) % 20 == 0:
                rate = written / max(time.time() - t0, 1e-9)
                log("%s %d/%d %.1f rows/s" % (tag, written, n, rate))
        fout.flush()
        os.fsync(fout.fileno())

    if written != n or os.path.getsize(tmp) != n * DIM * 2:
        raise SystemExit("%s: byte-size contract violated (%d rows, %d bytes)"
                         % (tag, written, os.path.getsize(tmp)))
    os.replace(tmp, vec_path)

    rid_tmp = rid_path + ".part"
    with open(rid_tmp, "w") as fh:
        for i, pid in enumerate(ids):
            fh.write(json.dumps({"i": i, "point_id": pid}) + "\n")
    os.replace(rid_tmp, rid_path)

    seconds = time.time() - t0
    manifest = {
        "schema": "aw.413a.vectors.v1",
        "shard": tag,
        "n": n,
        "dim": DIM,
        "dtype": "float16",
        "byte_order": "little",
        "layout": "row-major, no header",
        "bytes": n * DIM * 2,
        "sha256": sha256_file(vec_path),
        "row_ids_file": os.path.basename(rid_path),
        "row_ids_sha256": sha256_file(rid_path),
        "texts_sha256": sha256_file(texts_path),
        "ordering": "row i of vectors.f16 == line i of row_ids.jsonl == "
                    "line i of the staged texts.jsonl.gz",
        "embedder": info,
        "encode_seconds": round(seconds, 2),
        "rows_per_second": round(n / max(seconds, 1e-9), 2),
        "embedded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "producer_sha256": sha256_file(os.path.abspath(__file__)),
        "host": os.uname().nodename,
        "device": args.device,
    }
    tmp = man_path + ".part"
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    os.replace(tmp, man_path)
    log("%s DONE n=%d %.1f rows/s sha=%s"
        % (tag, n, manifest["rows_per_second"], manifest["sha256"][:12]))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    ap.add_argument("--model", default=EMBED_MODEL_ID)
    ap.add_argument("--revision", default=embed_model_revision())
    ap.add_argument("--max-seq", type=int, default=2560)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=4096,
                    help="rows held in RAM per flush")
    ap.add_argument("--device", default="auto",
                    help="auto (default): cuda:0 if CUDA is available, else cpu")
    ap.add_argument("--compute-dtype", default="auto",
                    choices=["auto", "float16", "float32"],
                    help="auto (default): float16 on cuda, float32 on cpu")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--only-shard", default="",
                    help="embed exactly this shard tag (for bounded GPU leases)")
    return ap


def resolve_device_dtype(device: str, compute_dtype: str,
                         cuda_available: bool) -> tuple:
    """Resolve the "auto" defaults to the hardware actually present.

    "auto" device -> cuda:0 when CUDA is available, else cpu.
    "auto" compute dtype -> float16 on cuda, float32 on cpu.
    Explicit values pass through unchanged.
    """
    if device in (None, "auto"):
        device = "cuda:0" if cuda_available else "cpu"
    if compute_dtype in (None, "auto"):
        compute_dtype = "float16" if device.startswith("cuda") else "float32"
    return device, compute_dtype


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.device, args.compute_dtype = resolve_device_dtype(
        args.device, args.compute_dtype, _cuda_available())
    log("resolved device=%s compute_dtype=%s" % (args.device, args.compute_dtype))

    os.makedirs(args.outdir, exist_ok=True)
    shards = sorted(f for f in os.listdir(args.indir)
                    if f.endswith(".texts.jsonl.gz"))
    if not shards:
        raise SystemExit("embedding input inventory is empty")
    if args.only_shard:
        wanted = args.only_shard + ".texts.jsonl.gz"
        if wanted not in shards:
            raise SystemExit("requested --only-shard is absent: %s" % args.only_shard)
        shards = [wanted]
    mine = [s for i, s in enumerate(shards) if i % args.world == args.rank]
    log("rank %d/%d owns %d of %d shards" % (args.rank, args.world, len(mine),
                                             len(shards)))
    # Consent refusal at the embed stage (task 677, task 643 S5): a recorded
    # decline on any host in the shards this rank will embed refuses BEFORE
    # the model loads or a GPU second is spent. No bypass.
    from cytir.consent.embed_gate import assert_stage_consent
    gate = assert_stage_consent(args.indir, shards_only=mine, log=log)
    log("consent gate clean: %s" % json.dumps(gate, sort_keys=True))
    model, info = load_model(args)
    total = 0
    t0 = time.time()
    for name in mine:
        man = embed_shard(model, info, os.path.join(args.indir, name),
                          args.outdir, args)
        total += man["n"]
        log("rank %d cumulative rows=%d elapsed=%.0fs avg=%.1f rows/s"
            % (args.rank, total, time.time() - t0,
               total / max(time.time() - t0, 1e-9)))
    log("rank %d COMPLETE rows=%d" % (args.rank, total))


if __name__ == "__main__":
    main()
