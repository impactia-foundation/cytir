"""cytir.index — build/apply/promote a serving collection (task 643 §6.1 target
layout). `compose.py` (task 657) embeds a sealed chunk stage, verifies its own
point ids reproduce under a NAMED, checked mint scheme, and composes+enqueues
a `cytir/index/promote_queue.py` promotion-queue pack. `apply.py` (task 646 S4,
moved from task 351's `apply_content_pack_shared.py`) is the twin-first
canonical-dump coverage-pack transaction: the ONE applier in the repo that
writes production. The index-side modules that used to sit in `pipeline/` now live here (task 677,
task 643 S5): `layout.py` (was `collection_layout`), `readiness.py`, `dense_precision.py`,
`twin_first.py`, `supersede.py`, `upsert.py`, `point_delete.py`, `payload_carry.py`,
`preservation_contract.py`, `content_pack_preservation.py`, `promote_sealed_pack.py`,
`normalize_sealed_pack_shape.py`. The old `pipeline/` paths are alias shims.
`promote_queue.py` (the live drain; the former retired `promote_queue` alias shim is deleted, and
the nightly unit still runs that path) and `news_window_filter.py` moved here too.
`compose.py` imports `fetch_collection_vector_config` from `promote_queue`.
the aw-does-not-import-the-scripts test holds the arrow (architecture-doctrine
#4: `cytir/` never imports the legacy scripts tree).
"""
from __future__ import annotations
