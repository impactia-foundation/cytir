"""The shape of a rank-head file (ADR-036), in one place: the serving loader checks it, the trainer emits it.

A head file is JSON. `variant` v1/v2 needs `w` (the 1024-d logistic weight on the frozen query vector) and `b`;
v3 and the optional regressor read `w_beta`/`b_beta` as well. A file that lacks a key the variant reads used to
load half-formed and fail per query; it is refused here instead.
"""
from __future__ import annotations

REQUIRED_KEYS = ("variant", "beta_gen", "beta_max", "trust_scale", "w", "b")
VARIANTS = ("v1", "v2", "v3")


def validate_head(head: object) -> dict:
    """Return `head` if it is a well-formed rank-head dict, else raise ValueError naming what is wrong."""
    if not isinstance(head, dict):
        raise ValueError("rank-head file must be a JSON object")
    missing = [k for k in REQUIRED_KEYS if k not in head]
    if missing:
        raise ValueError(f"rank-head file is missing keys: {', '.join(missing)}")
    if head["variant"] not in VARIANTS:
        raise ValueError(f"rank-head variant {head['variant']!r} not in {VARIANTS}")
    w = head["w"]
    if not isinstance(w, list) or not w or not all(isinstance(x, (int, float)) for x in w):
        raise ValueError("rank-head `w` must be a non-empty list of numbers")
    for k in ("beta_gen", "beta_max", "trust_scale", "b"):
        if not isinstance(head[k], (int, float)) or isinstance(head[k], bool):
            raise ValueError(f"rank-head `{k}` must be a number")
    if head["variant"] == "v3" and ("w_beta" not in head or "b_beta" not in head):
        raise ValueError("rank-head variant v3 needs `w_beta` and `b_beta`")
    return head
