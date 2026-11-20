#!/usr/bin/env python3
"""The ONE stage that embeds records into a Qdrant collection (task 682, migration
S6; task 643 section 6, "one embed stage parameterized by the collection layout").

WHY ONE STAGE
-------------
Before this module three scripts each created a collection, loaded an embedder,
truncated to an MRL width, built a point and upserted it:

* the retired `embed_p2_config_b.py` script  -- deleted (task 1315); the retired config-B 128-dim layout (removed 2026-10-28, task 1378).
* `cytir/index/stage_pack_b1.py`      -- every live re-fetch, news, HN and coverage lane
  (the 653 L1 sweep, `news_refresh_stage.sh`, `hn_discussion_stage.sh`, the retired `cov351_run.sh` (git-blob:2606d459c27b3037293ed6544fce26cd35093759; deleted task 1966),
  `direct_compose.py`): int8 at `--mrl-dim`, bulk hnsw m=0.
* a TEI encoder and a torch encoder, each written out inline.

The documented path built a layout production no longer serves (task 643 section 2.1:
"A stranger therefore builds a different index from ours, with no warning"). A second
way to do one verb is a fork, so the layout is now a PARAMETER and the work is done
once. The two scripts above are thin callers of this module and keep their CLIs.

THE LAYOUT IS DATA
------------------
`resolve_layout(name)` returns a `Layout`: a `serving_config.ServingConfig` (dense
width x datatype x quantizer x rescore, the one home for the served shape) plus the
three things `ServingConfig` does not carry: the sparse-index flavour, the HNSW `m`
at creation, and the payload indexes.

    served        the layout of the TWIN (aw_e8: the gate instrument and the
                  rollback copy) and of the live staging lanes: 256-d float16, product
                  quantization x8, rescore x3 at query time, BM25 sparse (IDF, on-disk
                  float16 index), HNSW m=16. It IS `serving_config.FLIP_SERVING_CONFIG`
                  ("256-pq8"); this module names no width of its own. THE DEFAULT.
                  It is NOT what production serves. Prod (aw_e9) is the profile
                  `prod-e9` (`cytir.core.serving_profile`, proven equal to a sealed read of
                  aw_e9) and differs on: HNSW m (16 vs 8), HNSW max_indexing_threads
                  (0 vs 8), sparse index datatype (float16 vs uint8), payload indexes
                  (no `tier`/`lei` on prod; every one on-disk on prod, in RAM here),
                  and the optimizer/segment settings (default_segment_number 30 and
                  max_segment_size 3.9M on prod, server defaults here;
                  max_optimization_threads 4 on prod). Dense vectors, the quantizer and
                  `on_disk` flags match. Tim decided not to make `served` equal prod.
    pack-staging  `stage_pack_b1`'s layout: int8 at the caller's `--mrl-dim`, hnsw
                  m=0 (bulk; packs are consumed by scroll, not queried).

What is and is not in the collection:

* Rescore and oversampling are QUERY-side parameters, not stored. After building a
  `served` collection the aw-api must run with `AW_MRL_DIM=256 AW_DENSE_RESCORE=1
  AW_DENSE_OVERSAMPLING=1.0` (`serving_config.flip_deploy_env`); `main` prints them.
* Product quantization and HNSW only apply once Qdrant has optimized a segment
  (indexing threshold, ~20k points by default). A small collection is searched by an
  exact scan: correct, just not quantized.

STATUS OF THE DOOR (task 696 slice 8 step 3)
--------------------------------------------
`cyt embed index` and `cyt source run --embed` are RETIRED: they projected rows with a
generic materializer that is not production's projection. They refuse and name
`cyt index build` (the retired config-B 128-dim layout and its projection were removed,
task 1378). The layout/creator/encoder machinery above is shared with `stage_pack_b1`
(the live lanes), which uses the production projection.

PARAMETERS THAT STAY WITH THE CALLER
------------------------------------
`--max-seq` differs on purpose (512 here and in `stage_pack_b1`; `cytir.embed.shards` uses
2560 for the sealed-shard artifact). The stage takes it as an argument and records
nothing about it; it is not a layout property.

THE SEALED-SHARD FLAVOUR
------------------------
`cytir.embed.shards` (the GPU worker the ERIC chain, growth compose and the admitted-stage
chain call, under `cytir.embed.stage`'s GPU lock) writes full-width 1024-d float16 FILES
and lets `promote_sealed_pack` truncate to the serving width at promotion, through the
same `serving_config.mrl_trunc`. Its output is a sealed artifact, not a collection, so
it stays a separate module; `cyt embed shards` is its door in the one CLI verb.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from cytir.core.common import DENSE_VEC, SPARSE_BM25

ARCTIC = "arctic-l-v2"

#: There is NO default layout (task 1283, R1 remainder): `served` is the pre-e9 arm shape
#: (hnsw m=16, float16 sparse) and the only door that defaulted to it is retired. A caller
#: names its layout, or sets `AW_EMBED_LAYOUT`.
LAYOUT_NAMES = ("served", "pack-staging")


@dataclass(frozen=True)
class Layout:
    """A collection layout: the `ServingConfig` plus what it does not carry."""

    name: str
    serving: object            # cytir.core.serving_config.ServingConfig
    sparse: str                # "serving" (on-disk f16 index) | "plain-idf"
    hnsw_m: int                # m at creation; 0 = bulk, no graph
    payload_indexes: dict      # field -> qdrant PayloadSchemaType
    verify_shape: bool         # read the collection back and assert it (served)

    @property
    def dense_dim(self) -> int:
        return self.serving.dense_dim


@dataclass
class Row:
    """One point to write: id, the text the dense leg embeds, the sparse leg
    (`(indices, values)`, or None to store no sparse vector) and the payload."""

    pid: str
    dense_text: str
    sparse: tuple | None
    payload: dict


def resolve_layout(name: str | None = None, *, dense_dim: int | None = None) -> Layout:
    """The layout called `name` (or `$AW_EMBED_LAYOUT`); naming none is refused.

    `pack-staging` needs `dense_dim` (the caller states the width; there is no
    default, task 508). `served` fixes its own width.
    """
    from qdrant_client import models as qm

    from cytir.core import serving_config as SC

    name = name or os.environ.get("AW_EMBED_LAYOUT")
    if not name:
        raise ValueError("no embed layout named: pass one or set CYTIR_EMBED_LAYOUT "
                         f"(one of {', '.join(LAYOUT_NAMES)}); there is no default")
    if name == "served":
        cfg = SC.get(SC.FLIP_SERVING_CONFIG)
        return Layout("served", cfg, "serving", 16, dict(cfg.payload_indexes), True)
    if name == "pack-staging":
        if not dense_dim:
            raise ValueError("layout 'pack-staging' needs an explicit dense_dim "
                             "(task 508: no default width)")
        cfg = SC.ServingConfig(
            name=f"pack-staging-{dense_dim}", dense_dim=int(dense_dim),
            dense_datatype="float32", quantizer="int8", rescore=False,
            oversampling=1.0, note="stage_pack_b1's scroll-consumed staging layout")
        indexes = {"lang": qm.PayloadSchemaType.KEYWORD,
                   "tier": qm.PayloadSchemaType.KEYWORD,
                   "domain": qm.PayloadSchemaType.KEYWORD,
                   "provenance": qm.PayloadSchemaType.KEYWORD,
                   "trust_tier": qm.PayloadSchemaType.FLOAT}
        return Layout("pack-staging", cfg, "plain-idf", 0, indexes, False)
    raise ValueError(f"unknown embed layout {name!r}; known: {list(LAYOUT_NAMES)}")


def create_collection(client, name: str, layout: Layout, *, recreate: bool,
                      log=print) -> bool:
    """Create `name` in `layout`. Returns True if it created one.

    A collection that exists is left alone (resume semantics) unless `recreate`,
    which deletes it first. `served` is read back and asserted
    (`serving_config.assert_collection_matches`): the quantizer key is an untagged
    serde field and a typo returns 200 and stores nothing (task 412 SS10.1).
    """
    from cytir.core import serving_profile as SP

    profile = layout_profile(layout)
    created = SP.create_collection(
        client, name, profile, on_exists="recreate" if recreate else "keep",
        index_failure="warn", verify=layout.verify_shape, log=lambda *a: None)
    if created:
        cfg = layout.serving
        log(f"created {name}: layout={layout.name} dense({cfg.dense_dim},"
            f"{cfg.dense_datatype},{cfg.quantizer},on_disk) + sparse(bm25,idf,{layout.sparse})"
            f" + hnsw(m={layout.hnsw_m},on_disk) + on_disk_payload")
    return created


def layout_profile(layout: Layout):
    """The `ServingProfile` a `Layout` has always created (request kwargs byte-equal)."""
    from qdrant_client import models

    from cytir.core import serving_profile as SP

    cfg = layout.serving
    if layout.sparse not in ("serving", "plain-idf"):
        raise ValueError(f"unknown sparse flavour {layout.sparse!r}")
    plain = layout.sparse == "plain-idf"
    return SP.ServingProfile(
        name=layout.name if layout.name != "pack-staging" else f"pack-staging-{cfg.dense_dim}",
        dense_dim=cfg.dense_dim, dense_datatype=cfg.dense_datatype,
        # float32 is the server default: leave `datatype` out, so the stored config
        # reads back exactly as the retired scripts' collections did (task 682 proof).
        datatype_explicit=cfg.dense_datatype != "float32",
        dense_on_disk=cfg.dense_on_disk, quantizer=cfg.quantizer,
        quant_always_ram=cfg.quant_always_ram,
        sparse_index_on_disk=None if plain else True,
        sparse_index_datatype=None if plain else "float16",
        hnsw_m=layout.hnsw_m,
        payload_indexes=tuple(
            (f, "float" if t == models.PayloadSchemaType.FLOAT else "keyword", False)
            for f, t in layout.payload_indexes.items()),
        note=cfg.note)


def existing_ids(client, collection) -> set:
    """Every point id already in the collection (for --resume)."""
    ids, offset = set(), None
    while True:
        pts, offset = client.scroll(collection, limit=10000, offset=offset,
                                    with_payload=False, with_vectors=False)
        ids.update(str(p.id) for p in pts)
        if offset is None:
            break
    return ids


def tei_doc_encoder(url: str, tei_batch: int):
    """`encode_docs`-shaped callable backed by a TEI sidecar.

    DOCUMENTS get no prompt prefix -- arctic-embed-l-v2 prefixes queries only --
    which is why this cannot reuse `cytir.serve.models._tei_embed` (the query path,
    which sends `prompt_name`). Fails loud on any TEI error: a silent fallback to a
    different embedder would write vectors that are wrong in a way no count check
    can see. (Moved verbatim from `cytir/index/stage_pack_b1.py`, task 682.)
    """
    import numpy as np
    import requests

    def encode(texts, batch=None):  # signature mirrors load_embedder's
        # the caller's `batch` is the upsert batch, not the TEI request size:
        # the TEI request size is bounded by max_batch_tokens and is fixed here
        del batch
        out = []
        for i in range(0, len(texts), tei_batch):
            r = requests.post(f"{url.rstrip('/')}/embed",
                              json={"inputs": texts[i:i + tei_batch],
                                    "normalize": True, "truncate": True},
                              timeout=180)
            if r.status_code != 200:
                raise RuntimeError(f"TEI {r.status_code} at row {i}: "
                                   f"{r.text[:300]}")
            out.extend(r.json())
        return np.asarray(out, dtype="float32"), None
    return encode


def build_encoder(*, tei_url: str = "", tei_batch: int = 1, max_seq: int = 512,
                  log=print):
    """`encode_docs(texts, batch=...) -> (vectors, None)` for arctic-embed-l-v2.

    TEI sidecar when `tei_url` is given, else the in-process model (GPU fp16 when
    CUDA is present, CPU fp32 otherwise; the pinned revision either way).
    """
    if tei_url:
        log(f"Embedding docs via TEI at {tei_url} (batch {tei_batch})")
        return tei_doc_encoder(tei_url, tei_batch)
    import torch

    from cytir.embed.model import load_embedder
    cuda = torch.cuda.is_available()
    log(f"Loading {ARCTIC} ({'GPU/fp16' if cuda else 'CPU/fp32'})…")
    encode_docs, _q, _spec = load_embedder(ARCTIC, use_fp16=cuda, max_seq=max_seq)
    return encode_docs


def embed_rows(client, collection: str, items: Sequence,
               materialize: Callable[[Sequence], "list[Row]"], layout: Layout,
               encode_docs, *, batch: int = 256, log=print) -> dict:
    """Embed and upsert `items`, `batch` at a time.

    `materialize(chunk)` turns a chunk of the caller's items into `Row`s, so a
    caller that builds sparse vectors or payloads lazily keeps its memory profile.
    Dense vectors are truncated to `layout.dense_dim` (`mrl_trunc`: prefix, then
    L2-renormalize) -- the single home of that transform.
    """
    from qdrant_client import models

    from cytir.core.serving_config import mrl_trunc

    t0 = time.time()
    total = 0
    for i in range(0, len(items), batch):
        rows = materialize(items[i:i + batch])
        dense, _ = encode_docs([r.dense_text for r in rows], batch=batch)
        dense = mrl_trunc(dense, layout.dense_dim)
        points = []
        for row, dv in zip(rows, dense):
            vec = {DENSE_VEC: dv}
            if row.sparse is not None:
                vec[SPARSE_BM25] = models.SparseVector(
                    indices=row.sparse[0], values=row.sparse[1])
            points.append(models.PointStruct(id=row.pid, vector=vec,
                                             payload=row.payload))
        client.upsert(collection, points=points, wait=False)
        total += len(points)
        if (i // batch) % 25 == 0:
            rate = total / max(time.time() - t0, 1)
            log(f"  {total}/{len(items)}  {rate:.0f} pts/s")
    return {"points": total, "seconds": time.time() - t0, "t0": t0}


def wait_green(client, collection: str, *, iters: int, sleep: float = 2.0):
    """Poll until the collection is green (at most `iters` polls); returns its info."""
    from qdrant_client import models

    for _ in range(iters):
        if client.get_collection(collection).status == models.CollectionStatus.GREEN:
            break
        time.sleep(sleep)
    return client.get_collection(collection)


#: the one verb that replaced the retired door
REPLACEMENT_VERB = "cyt index build"
RETIRED_DOOR_MESSAGE = (
    "REFUSED: `cyt embed index` is retired: it built an index that differs from the one "
    "production serves (point ids, dense text, BM25 weights and payload all differ). The one "
    "build verb is `" + REPLACEMENT_VERB + "`; get a ready-made index with `cyt apply` (a "
    "published dump), or stop at the sealed pack and its vectors (`cyt source run`, "
    "`cyt embed shards`).")


def main(argv=None, *, prog: str | None = None) -> int:
    """`cyt embed index`: RETIRED. It always refuses and names `cyt index build`."""
    raise SystemExit(RETIRED_DOOR_MESSAGE)


if __name__ == "__main__":
    raise SystemExit(main())
