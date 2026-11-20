"""ONE home for the embedder identity — model id AND revision (task 432).

This file is a MIRROR CONTRACT. Vectors written into the index and vectors
computed for a query at serve time must come from the same model at the same
revision, or the geometry silently stops matching. Write side and read side
therefore import their identity from here rather than each restating it.

MOVED HERE from the retired model_pin script in task 646 S2 (task 643 §6.2):
previously the aw-api container image contained only `cytir/serve/`
(never `pipeline/`), so the read side could not import this module across
the directory boundary and instead carried a hand-maintained
**byte-identical copy** at `cytir/serve/model_pin.py`, with
a model-pin test failing if the two files ever differed. Now
both the retired script (once a shim: `from cytir.core.model_pin import *`)
and `cytir/serve/models.py` import this module directly — the package
is installed into the aw-api image (`server/Dockerfile`), so there is
nothing left to keep byte-identical. `test_server_copy_is_byte_identical`
became `test_server_imports_the_same_pin_object` (import identity, not a
second copy).

WHAT IS PINNED, AND WHY THIS VALUE
----------------------------------
`ac6544c8a46e00af67e330e85a9028c66b8cfd9a` is not a new choice. It is what
production already runs, verified three ways on 2026-08-19:

  1. task 422 — the live embed driver
     passes `--revision ac6544c8a46e00af67e330e85a9028c66b8cfd9a`, and
     task 413a carries it as its argparse default.
  2. HuggingFace's API reports it as the model's current `main` sha
     (lastModified 2025-07-28).
  3. the build box's HF cache and the serving box's TEI cache
     (`/opt/aw/tei-data/.../snapshots/`) each hold exactly this one snapshot.

So pinning changes nothing about what loads today; it stops the day HuggingFace
moves `main` from being an invisible re-embed of the whole corpus.

Task 429 (blocker E) found the write side unpinned: `load_embedder()` took no
revision, so document vectors came from whatever HuggingFace served that day.
Task 432 found the read side worse — the TEI sidecar is launched without
`--revision` and reports `"model_sha": null`, i.e. serving cannot even tell you
which revision it is answering queries with.
"""
from __future__ import annotations

import os
import re

# --- the pin ---------------------------------------------------------------
EMBED_MODEL_ID = "Snowflake/snowflake-arctic-embed-l-v2.0"
EMBED_MODEL_REVISION = "ac6544c8a46e00af67e330e85a9028c66b8cfd9a"

#: Set AW_MODEL_REVISION to move the pin for a deliberate re-embed campaign.
#: It is honoured, never silently: `embed_model_revision()` returns the override
#: and `pin_note()` says loudly that the value is not the committed pin. The
#: vast.ai embed workers already pass this variable (task 375).
REVISION_ENV = "AW_MODEL_REVISION"

_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def embed_model_revision() -> str:
    """The revision to load. Honours AW_MODEL_REVISION; validates its shape.

    A revision is a 40-hex commit sha, never a branch name: `main` is a moving
    target and pinning to it pins nothing.
    """
    override = os.environ.get(REVISION_ENV, "").strip()
    if not override:
        return EMBED_MODEL_REVISION
    if not _HEX40.match(override):
        raise ValueError(
            f"{REVISION_ENV}={override!r} is not a 40-hex commit sha. A branch or "
            f"tag is not a pin — it moves. Use the full commit sha.")
    return override


def is_pinned_default() -> bool:
    return embed_model_revision() == EMBED_MODEL_REVISION


def pin_note() -> str:
    """One line for logs and receipts; screams when the pin is overridden."""
    rev = embed_model_revision()
    if rev == EMBED_MODEL_REVISION:
        return f"embedder {EMBED_MODEL_ID}@{rev} (committed pin)"
    return (f"embedder {EMBED_MODEL_ID}@{rev} — OVERRIDE via {REVISION_ENV}; "
            f"the committed pin is {EMBED_MODEL_REVISION}. Vectors written now "
            f"will NOT match the index built under the pin.")


def assert_pin(model_id: str, revision: str | None, where: str) -> None:
    """Fail loud when a call site's model identity disagrees with the pin.

    `revision=None` means the caller did not pin at all, which is the exact
    defect this module exists to remove — so it is an error, not a warning.
    """
    expected_rev = embed_model_revision()
    problems = []
    if model_id != EMBED_MODEL_ID:
        problems.append(f"model id {model_id!r} != pinned {EMBED_MODEL_ID!r}")
    if revision is None:
        problems.append("no revision pinned (reproducibility is nominal, not real)")
    elif revision != expected_rev:
        problems.append(f"revision {revision!r} != pinned {expected_rev!r}")
    if problems:
        raise ValueError(
            f"embedder pin mismatch at {where}: " + "; ".join(problems) +
            ". Write-time and read-time embeddings must come from the same model "
            "at the same revision or the vector geometry silently diverges. "
            "The pin lives in cytir/core/model_pin.py.")
