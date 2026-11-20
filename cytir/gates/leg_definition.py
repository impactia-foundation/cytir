"""What a flip-gate LEG is, in one place: the env overrides and the search call (task 865, review 866 M1).

`data/flip-gate-contract.v1.json` `legs` defines three arms per side. A leg arm is the serving code run
with the SIDE's own flags plus the leg's overrides, called exactly as the gate calls it. Both sides
must take their leg from here -- the incumbent read on prod (`cytir.gates.prod_legs`) and the candidate
read on the build box -- or "candidate minus incumbent" measures two instruments, not two indexes.

* `full-stack`: no override; the side's own ratified weights and flags.
* `dense-only`: "dense-only weights, floor/head flags off, dense policy otherwise identical". The weights
  file is the contract's; the 14 floor/head flags are the ones task 524's reference launch turned off
  (they act outside the fusion weights, so the weights file alone does not turn them off).
* `sparse-only` (task 863, the POLICY PR "sparse-only arm definition"): the MIRROR of dense-only. The
  weights file `data/gates/fusion_weights.sparse_only.v1.json` is the dense-only file with `dense` 0.0
  and `bm25` 1.0 (every other key byte-equal: the same zero weights, the same non-weight knobs); the
  same 14 floor/head flags off EXCEPT `AW_BM25_PER_TOKEN`, which stays as the side serves it: it is
  part of the BM25 LEG itself (for 2..4-token queries it adds the per-token BM25 queries and so
  changes which rows the leg returns), not a floor or a head acting after fusion.

The gate reads k = 10 (the 629 arm runner and `cytir.gates.authority`), and passes each row's `lang` and
`country`.
"""
from __future__ import annotations

LEGS = ("full-stack", "dense-only", "sparse-only")
GATE_K = 10
#: task 524's dense-only reference launch (`launch_dense_only.sh`, DENSE_ONLY_FLAGS), the contract's
#: "floor/head flags off"
FLOOR_HEAD_OFF = {
    "AW_AUTHORITY": "0", "AW_AUTHLEG": "0", "AW_HUBFLOOR": "0", "AW_NAMEGATE": "0",
    "AW_NAMECOV_W": "0.0", "AW_QHEAD": "0", "AW_REFCANON": "0", "AW_RELFLOOR": "0",
    "AW_TRUSTFLOOR": "0", "AW_COMMERCIAL": "0", "AW_BM25_PER_TOKEN": "0",
    "AW_OFF_COUNTRY_W": "0.0", "AW_OFF_COUNTRY_NONBOUND_W": "0.0", "AW_RANKHEAD": "0",
}
#: sha256 of the contract's dense-only weights file (`legs.dense-only.weights` of the flip-gate contract, task 524)
DENSE_ONLY_WEIGHTS_SHA256 = "484534a7730e24a9374a8d5638f88d3224b958242dd43553316e0eae9cfe081a"
#: the flags the BM25 leg itself reads: a sparse-only arm leaves them as the side serves them
SPARSE_LEG_FLAGS = ("AW_BM25_PER_TOKEN",)
#: the sparse-only overrides: the floor/head flags off, minus the BM25 leg's own
SPARSE_ONLY_OFF = {k: v for k, v in FLOOR_HEAD_OFF.items() if k not in SPARSE_LEG_FLAGS}
#: sha256 of the sparse-only weights file (`legs.sparse-only.weights_file` of the flip-gate contract,
#: task 863): the dense-only file with dense 0.0 and bm25 1.0
SPARSE_ONLY_WEIGHTS_SHA256 = "25987044c245cc471a57c9ef29ad03151a2685cfb94a93db96a60c0006dcad72"


class LegUndefined(ValueError):
    """The contract does not define this leg well enough to read it."""


def leg_env(leg: str, weights_path: str | None, weights_sha256: str | None) -> dict[str, str]:
    """The env overrides of `leg`, applied ON TOP of the side's own flags."""
    if leg == "full-stack":
        if weights_path:
            raise LegUndefined("full-stack serves the side's own ratified weights: pass no --weights")
        return {}
    if leg == "dense-only":
        if not weights_path:
            raise LegUndefined("dense-only needs the contract's dense-only weights file")
        if weights_sha256 != DENSE_ONLY_WEIGHTS_SHA256:
            raise LegUndefined(f"dense-only weights sha256 {weights_sha256} is not the contract's "
                               f"{DENSE_ONLY_WEIGHTS_SHA256}")
        return {**FLOOR_HEAD_OFF, "AW_FUSION_WEIGHTS": weights_path}
    if leg == "sparse-only":
        if not weights_path:
            raise LegUndefined("sparse-only needs the contract's sparse-only weights file")
        if weights_sha256 != SPARSE_ONLY_WEIGHTS_SHA256:
            raise LegUndefined(f"sparse-only weights sha256 {weights_sha256} is not the contract's "
                               f"{SPARSE_ONLY_WEIGHTS_SHA256}")
        return {**SPARSE_ONLY_OFF, "AW_FUSION_WEIGHTS": weights_path}
    raise LegUndefined(f"unknown leg {leg!r}; one of {LEGS}")


def search_kwargs(row: dict, k: int = GATE_K) -> dict:
    """The arguments the gate passes to `retrieval.search` for one row (besides the query)."""
    return {"k": k, "lang": row.get("lang") or None, "country": row.get("country") or None}


def _client_module(retrieval):
    """The module that owns the Qdrant client: `qdrant_io` since S10 step 6 (task 1104); a tree from
    before that step kept the client in `retrieval` itself."""
    return getattr(retrieval, "qdrant_io", None) or retrieval


def apply_qdrant_timeout(retrieval, seconds: float | None) -> float:
    """Set the serving code's Qdrant client timeout for the gate's reads (task 937); return the one in force.

    None leaves the side's own `AW_QDRANT_TIMEOUT` (the module's `_QDRANT_TIMEOUT`) as it is. A number
    replaces it in the loaded module only -- never in an env file or a live env -- and drops any client
    built with the old value. Ranking does not depend on the timeout; only whether a cold query completes.
    """
    if seconds is not None:
        if not seconds > 0:
            raise ValueError(f"--qdrant-timeout-s {seconds}: must be > 0")
        io = _client_module(retrieval)
        io._QDRANT_TIMEOUT = float(seconds)
        io._client = None
    return _client_module(retrieval)._QDRANT_TIMEOUT
