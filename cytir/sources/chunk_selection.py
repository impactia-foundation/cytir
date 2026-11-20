"""The declared chunk policy of a family's contract row as an explicit structured selection.

One definition for the adapters that chunk under `CHUNK_DECLARED` through the structured chunker:
the file the policy registry names for the contract's sha256, and the contract's tokenizer
revision. The run needs no `AW_CHUNK*` variable, and a policy that is not the declared one never
runs. (Task 1392. The older adapters carry private copies of this function; folding them onto
this one is a follow-up, not part of the ERIC slice.)
"""
from __future__ import annotations

import pathlib

from cytir.sources.family import RunContext


def declared_policy_selection(ctx: RunContext):
    from cytir.chunk import registry as chunk_registry
    from cytir.core import common as C
    from cytir.chunk import dispatch as _dispatch

    cp = (ctx.contract or {}).get("chunk_policy")
    if not cp:
        raise ValueError("the run has no contract row with a declared chunk_policy")
    entry = chunk_registry.entry(cp["sha256"])
    repo = pathlib.Path(__file__).resolve().parents[2]
    return _dispatch.explicit_structured_chunker_selection(
        str(repo / entry["file"]), cp["sha256"], cp["tokenizer_revision"])
