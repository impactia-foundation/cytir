#!/usr/bin/env python3
"""AW bake-off — generalized INDEX stage. passages.jsonl -> embed (dense [+
learned-sparse for bge-m3]) + BM25 sparse -> a DISTINCT Qdrant collection per
embedder, so the live `aw_passages` serving index is never touched.

  python3 -m cytir.embed.bakeoff --in build/passages_canon.jsonl \
      --embedder qwen3-0.6b --collection bakeoff_qwen3-0.6b \
      --qdrant-url http://<qdrant-host>:6333

Embedders (see EMBEDDERS below): bge-m3, qwen3-0.6b, qwen3-4b, arctic-l-v2.
Quantization defaults to `binary` (ADR-011 serving design). Sparse legs: dense
is always stored; bge-m3 also stores its learned-sparse `lexical` leg; a classic
BM25 sparse leg (aw_common.bm25_tokens) is always computed+stored from text.
"""
from __future__ import annotations

import argparse
import json

import sys

from cytir.core.common import (DENSE_VEC, SPARSE_LEXICAL, SPARSE_BM25, bm25_tokens,
                       compose_index_text)
from cytir.core.model_pin import (EMBED_MODEL_ID, embed_model_revision, pin_note)


def qdrant_bindings():
    """Load Qdrant only for collection operations.

    The embedder registry and MRL projection are also principal producer
    contracts.  Keeping their import path free of the optional client lets an
    isolated GPU producer use the same implementation without carrying a
    database client; actual index operations still fail loudly if it is absent.
    """
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise RuntimeError(
            "qdrant-client is required for Qdrant collection operations") from exc
    return QdrantClient, models

# The embedder registry and `load_embedder` live in `cytir.embed.model` (task 682, S6);
# re-exported here (and by the old-path shim). One implementation.
from cytir.embed.model import EMBEDDERS, load_embedder  # noqa: E402,F401

# mrl_trunc moved to `serving_config.py` (task 478) so the truncation width and
# the truncation transform live in the same module, and so callers that need only
# the transform stop importing this module's embedder stack. Re-exported here
# because ~10 scripts do `from index_bakeoff import mrl_trunc`; there is still
# exactly one implementation.
from cytir.core.serving_config import mrl_trunc  # noqa: E402,F401


def bakeoff_profile(collection, dim, legs, quant, on_disk=False):
    """The `ServingProfile` the bake-off has always created (request kwargs unchanged):
    fp32 dense (server default datatype), the chosen quantizer held in RAM, a plain-IDF
    bm25 sparse leg when asked for, server-default hnsw / payload storage, plain
    payload indexes. `lexical` is not part of the profile: see `ensure_collection`."""
    from cytir.core import serving_profile as SP

    quantizer = {"none": "none", "int8": "int8", "binary": "binary_1bit"}.get(quant)
    if quantizer is None:
        raise ValueError(quant)
    return SP.ServingProfile(
        name=f"bakeoff-{collection}", dense_dim=int(dim), dense_datatype="float32",
        datatype_explicit=False, dense_on_disk=bool(on_disk), quantizer=quantizer,
        quant_always_ram=True,
        sparse_name=SPARSE_BM25 if "bm25" in legs else None, sparse_idf=True,
        sparse_index_on_disk=None, sparse_index_datatype=None,
        hnsw_m=None, hnsw_on_disk=None, on_disk_payload=None,
        payload_indexes=(("lang", "keyword", False), ("tier", "keyword", False),
                         ("domain", "keyword", False), ("trust_tier", "float", False)),
        note="index bake-off collection")


def ensure_collection(client, collection, dim, legs, quant, recreate=True, on_disk=False):
    """Create `collection` through the one creator (`serving_profile.create_collection`).
    An existing collection is kept when not `recreate`, else deleted and rebuilt. The
    learned-sparse `lexical` leg (bge-m3 only) is added by an update after creation,
    because a profile carries a single sparse vector."""
    from cytir.core import serving_profile as SP

    _, models = qdrant_bindings()
    profile = bakeoff_profile(collection, dim, legs, quant, on_disk)
    created = SP.create_collection(
        client, collection, profile, on_exists="recreate" if recreate else "keep",
        index_failure="warn", log=lambda *a: None)
    if created and "lexical" in legs:
        client.update_collection(collection, sparse_vectors_config={
            SPARSE_LEXICAL: models.SparseVectorParams()})


# `point_id` moved to `cytir.core.ids.record_point_id` (task 682, S6): the uuid5 mint
# belongs in aw/core. Same function, re-exported under the old name.
from cytir.core.ids import record_point_id as point_id  # noqa: E402,F401


# `existing_ids` moved to `cytir.embed.collection_stage` (task 682, S6); re-exported.
from cytir.embed.collection_stage import existing_ids  # noqa: E402,F401


def main():
    from tqdm import tqdm

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--embedder", required=True, choices=list(EMBEDDERS))
    ap.add_argument("--collection", required=True)
    ap.add_argument("--qdrant-url", required=True)  # no default host (public cut)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--quant", choices=["none", "binary", "int8"], default="binary")
    ap.add_argument("--legs", default=None,
                    help="comma list of extra sparse legs to store: "
                         "lexical,bm25 (dense always stored). Default: "
                         "bm25 (+lexical for bge-m3).")
    ap.add_argument("--limit", type=int, default=0, help="cap #passages (smoke test)")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="keep the existing collection and skip passages already "
                         "indexed (crash-safe on the flaky GPU box).")
    # Thermal duty-cycling for heat-limited GPU boxes: pause every N batches to shed
    # heat so a long embed does not build to a hard thermal fault.
    ap.add_argument("--cooldown-every", type=int, default=0,
                    help="pause every N batches (0=off)")
    ap.add_argument("--cooldown-sec", type=float, default=20.0,
                    help="seconds to pause for cooldown")
    ap.add_argument("--max-seq", type=int, default=512,
                    help="truncate inputs to this many tokens (uniform across "
                         "embedders; Qwen3's 32k default OOMs + skews the compare)")
    ap.add_argument("--mrl-dim", type=int, default=0,
                    help="Matryoshka: store dense truncated to this many dims "
                         "(+renormalized). 0 = full. Tests storage/recall tradeoff.")
    ap.add_argument("--on-disk", action="store_true",
                    help="store dense vectors on disk (1B-scale serving design, "
                         "ADR-011). Default off: keep in RAM (right for small "
                         "bake-off corpora — avoids the qdrant nofile blow-up).")
    args = ap.parse_args()
    # task 864 (review 869 S1): from an aw_e10 T0 to the flip only the drain writes the twin
    from cytir.index import twin_write_hold as _TWH
    _TWH.refuse_cli_write(args, tool='cytir/embed/bakeoff.py')

    spec = EMBEDDERS[args.embedder]
    # resolve sparse legs to store
    if args.legs is not None:
        legs = {l.strip() for l in args.legs.split(",") if l.strip()}
    else:
        legs = {"bm25"} | ({"lexical"} if spec["has_lexical"] else set())
    if "lexical" in legs and not spec["has_lexical"]:
        print(f"[warn] {args.embedder} has no learned-sparse leg; dropping 'lexical'")
        legs.discard("lexical")

    import time
    recs = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    if args.limit:
        recs = recs[: args.limit]

    dense_dim = args.mrl_dim if (args.mrl_dim and args.mrl_dim < spec["dim"]) else spec["dim"]
    from cytir.core.qdrant import qdrant_connect
    _, models = qdrant_bindings()
    client = qdrant_connect(args.qdrant_url, timeout=300)
    ensure_collection(client, args.collection, dense_dim, legs, args.quant,
                      recreate=not (args.append or args.resume), on_disk=args.on_disk)

    if args.resume:
        done = existing_ids(client, args.collection)
        before = len(recs)
        recs = [r for r in recs if point_id(r) not in done]
        print(f"[resume] {len(done)} already indexed; {len(recs)}/{before} remaining")

    print(f"{len(recs)} passages | embedder={args.embedder} dim={spec['dim']} "
          f"legs={sorted(legs) + ['dense']} quant={args.quant}"
          f"{' cooldown %ds/%db' % (args.cooldown_sec, args.cooldown_every) if args.cooldown_every else ''}")

    import torch
    cuda = torch.cuda.is_available()
    print(f"Loading {spec['model']} ({'GPU/fp16' if cuda else 'CPU/fp32'})…")
    encode_docs, _encode_q, spec = load_embedder(args.embedder, use_fp16=cuda, max_seq=args.max_seq)
    if "_st_prompts" in spec:
        print(f"  sentence-transformers prompts available: {spec['_st_prompts']}")
        print(f"  doc prompt={spec['doc_prompt_name']!r} query prompt={spec['query_prompt_name']!r}")

    total = 0
    for b, i in enumerate(tqdm(range(0, len(recs), args.batch), desc="embed+upsert")):
        chunk = recs[i:i + args.batch]
        # task 274 (bug #2): embed + BM25 from title+site_name+body (EVERY chunk),
        # not the body alone — so the page's highest-signal summary + entity token
        # are IN both vectors and a thin-body/vocab-mismatch page can enter the
        # candidate pool at all (the title was previously payload-only, used only
        # by the post-pool title-centrality rerank). entity_card records are
        # exempt (ADR-048); compose_index_text is a no-op where the title is
        # already redundant with the body (byte-parity).
        itext = [compose_index_text(r) for r in chunk]
        dense, lex = encode_docs(itext, batch=args.batch) \
            if spec["kind"] == "st" else encode_docs(itext)
        dense = mrl_trunc(dense, args.mrl_dim)
        points = []
        for j, r in enumerate(chunk):
            vec = {DENSE_VEC: dense[j]}
            if "lexical" in legs and lex is not None:
                lw = lex[j]
                vec[SPARSE_LEXICAL] = models.SparseVector(
                    indices=[int(k) for k in lw.keys()],
                    values=[float(v) for v in lw.values()])
            if "bm25" in legs:
                bm = bm25_tokens(itext[j], r.get("lang"))
                vec[SPARSE_BM25] = models.SparseVector(
                    indices=list(bm.keys()), values=list(bm.values()))
            payload = {k: v for k, v in r.items() if k != "text"}
            points.append(models.PointStruct(id=point_id(r), vector=vec, payload=payload))
        client.upsert(args.collection, points=points)
        total += len(points)
        # thermal duty-cycle: let the GPUs shed heat on the flaky box
        if args.cooldown_every and (b + 1) % args.cooldown_every == 0:
            time.sleep(args.cooldown_sec)

    n = client.get_collection(args.collection).points_count
    print(f"\nDone. {n} points in `{args.collection}`.")


if __name__ == "__main__":
    main()
