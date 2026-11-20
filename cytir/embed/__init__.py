"""cytir.embed — the embed stage (task 657, migration S6 early move, task 643
§6.1). Moved out of task 413a's scripts and task 633's scripts (a
production path that had two task numbers in it, embedding for every
sealed-pack promotion since task 413), one file at a time; each move leaves
a shim at the old path (task 643 §6.2).

Modules:
    shards.py  the GPU embed worker (moved from
               task 413a's `embed_shards.py`): reads staged chunk
               shards, writes the sealed 1024-d fp16 vector artifact
               `promote_sealed_pack.py` reads. MODEL/REVISION now import
               `cytir.core.model_pin` instead of restating the pin (the fork
               `cytir.core.model_pin`'s own docstring exists to prevent).
    stage.py   the GPU-lock + invocation wrapper (moved from the lock
               discipline in task 633's `run_embed_lease.sh`): the
               one door other package code and the ERIC materialization
               chain call to embed a sealed chunk-stage directory.
    collection_stage.py
               the ONE stage that embeds records into a Qdrant collection,
               parameterized by the collection layout (task 682, S6):
               `served` (256-d float16, PQ x8, rescore, BM25; the default),
               and `pack-staging` (the packs'
               scroll-consumed layout). `cyt embed index` (its old door) is retired;
               `cytir/index/stage_pack_b1.py` calls it and keeps its CLI
               (the retired config-B 128-dim layout (removed 2026-10-28, task 1378)).
    model.py   the embedder registry and `load_embedder` (moved from
               the old `index_bakeoff` script, which re-exports them).

`cyt embed {index,shards,stage}` (cytir/cli.py) is the one verb over these: `index`
writes a collection; `shards` and `stage` write the sealed full-width fp16
vector files that promotion truncates to the serving width.
"""
from __future__ import annotations
