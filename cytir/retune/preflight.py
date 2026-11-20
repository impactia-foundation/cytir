"""The serving-contract preflight (task 711, kept by task 724): refuse before ONE evaluation query.

Moved from the retired offline tuner. It proves that the process which will score the weights
serves the way prod serves: the collection's dense width and vector names, TEI's model and
revision against the pin, and (when prod's `/health` is given) the server's own echo of its
effective config and of the fusion-weights file it loads.
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import os

from cytir.core.serve_layout import PACKAGE_NAMES

#: where the image holds the serving package: the live image was built before the rename (the old package
#: name under /app), the first one after it holds the new name; prod's echo names whichever runs, so both are read
CONTAINER_PREFIXES = tuple(f"/app/{name}/serve/" for name in PACKAGE_NAMES)
HOST_MARKERS = tuple(f"/{name}/serve/" for name in PACKAGE_NAMES)


def _norm_path(v):
    """A file the server names is compared by its place in the serving package, not by the host path."""
    if isinstance(v, dict):
        return {k: _norm_path(x) for k, x in v.items()}
    if isinstance(v, str):
        for prefix in CONTAINER_PREFIXES:
            if v.startswith(prefix):
                return "serve/" + v[len(prefix):]
        flat = v.replace(os.sep, "/")
        for marker in HOST_MARKERS:
            if marker in flat:
                return "serve/" + flat.split(marker, 1)[1]
    return v


def _unmapped_image_paths(v):
    """Every `/app/...` string in `v` that no known image prefix covers (see CONTAINER_PREFIXES)."""
    if isinstance(v, dict):
        return [p for x in v.values() for p in _unmapped_image_paths(x)]
    if isinstance(v, str) and v.startswith("/app/") and not v.startswith(CONTAINER_PREFIXES):
        return [v]
    return []


def config_parity_problems(served: dict, local: dict) -> list[str]:
    """The server's echo (`/health` -> `retrieval.effective_config()`) of prod vs this process.
    The collection NAME differs by design (alias vs flag). The weights file is compared by its
    sha256 -- and `incumbent_only` says the arm runs the incumbent, whose sha must equal prod's."""
    out = []
    for k in sorted(set(served) | set(local)):
        if k == "collection":
            continue
        for side, value in (("prod echoes", served.get(k)), ("this process computes", local.get(k))):
            for path in _unmapped_image_paths(value):
                out.append(f"effective_config[{k}]: {side} {path!r}, an image path under none of "
                           f"{CONTAINER_PREFIXES}; refused, not compared as if it were mapped")
        a, b = _norm_path(served.get(k)), _norm_path(local.get(k))
        if k == "fusion_weights":
            a, b = (a or {}).get("sha256"), (b or {}).get("sha256")
        if a != b:
            out.append(f"effective_config[{k}]: prod echoes {a!r}, this process computes {b!r}")
    return out


def serving_contract_problems(collection: dict, mrl_dim: int, tei_info: dict | None, *,
                              model_id: str, revision: str, dense: str, sparse: str,
                              name: str = "?", cached_revisions: list[str] | None = None) -> list[str]:
    """What must hold before one evaluation query goes to the collection.

    `collection` is qdrant's `GET /collections/{name}` `result`; `mrl_dim` is the width this process
    embeds at (`models.MRL_DIM`, the server's AW_MRL_DIM); `tei_info` is TEI's `GET /info`."""
    out = []
    params = (collection.get("config") or {}).get("params") or {}
    vectors = params.get("vectors") or {}
    if "size" in vectors:
        out.append(f"collection {name} has one unnamed vector; the server queries the named "
                   f"vector {dense!r}")
    elif dense not in vectors:
        out.append(f"collection {name} has no dense vector {dense!r} (has {sorted(vectors)})")
    elif int(vectors[dense].get("size", -1)) != int(mrl_dim):
        out.append(f"dense width: collection {name} serves {vectors[dense].get('size')}-dim "
                   f"{dense!r} vectors, this process embeds at {mrl_dim} (AW_MRL_DIM). "
                   f"Qdrant refuses every such query.")
    if sparse not in (params.get("sparse_vectors") or {}):
        out.append(f"collection {name} has no sparse vector {sparse!r} "
                   f"(has {sorted(params.get('sparse_vectors') or {})})")
    if tei_info is None:
        out.append("TEI /info unreachable: the embedder cannot be identified")
    else:
        if tei_info.get("model_id") != model_id:
            out.append(f"TEI serves model {tei_info.get('model_id')!r}, the pin is {model_id!r}")
        sha = tei_info.get("model_sha")
        if sha is None and cached_revisions == [revision]:
            pass      # launched without --revision, but its HF cache holds the pinned snapshot ONLY
        elif sha is None:
            out.append(f"TEI reports model_sha=null (launched without --revision): the query "
                       f"embedder's revision is unknown; the pin is {revision}")
        elif sha != revision:
            out.append(f"TEI serves revision {sha}, the pin is {revision}")
    return out


def cached_revisions(snapshots_dir: str | None) -> list[str] | None:
    """Revisions in a HF cache `.../models--X/snapshots/` dir (the evidence for a TEI that was
    launched without --revision: if only the pinned snapshot exists, only it can be served)."""
    if not snapshots_dir:
        return None
    try:
        return sorted(os.listdir(snapshots_dir))
    except OSError:
        return None


def check(qdrant_url: str, collection: str, tei_url: str, *, prod_health: dict | None,
          incumbent_sha256: str, tei_snapshots: str | None = None) -> dict:
    """Run inside the arm process (the server is imported). Raises SystemExit listing EVERY
    problem; returns the facts it proved. Network reads only (collection config, TEI /info, one
    probe embedding)."""
    from cytir.serve import models, retrieval
    from cytir.serve.schema import DENSE_VEC, SPARSE_BM25
    try:
        coll = IH.fetch_json(f"{qdrant_url}/collections/{collection}", timeout=30)["result"]
    except Exception as e:  # noqa: BLE001 -- an unreadable contract is a refusal, not a pass
        raise SystemExit(f"PREFLIGHT FAILED: cannot read collection {collection} at {qdrant_url}: {e!r}")
    try:
        tei_info = IH.fetch_json(f"{tei_url}/info", timeout=30)
    except Exception:  # noqa: BLE001
        tei_info = None
    problems = serving_contract_problems(
        coll, models.MRL_DIM, tei_info, model_id=models.EMBED_MODEL_ID,
        revision=models.EMBED_REVISION, dense=DENSE_VEC, sparse=SPARSE_BM25, name=collection,
        cached_revisions=cached_revisions(tei_snapshots))
    size = (((coll.get("config") or {}).get("params") or {}).get("vectors") or {}).get(DENSE_VEC, {}).get("size")
    if not problems:
        vec, _ = models._embed_query_tei("cyt retune preflight probe")
        if len(vec) != size:
            problems.append(f"probe embedding has {len(vec)} dims, collection {collection} serves {size}")
    if prod_health is not None:
        served = prod_health.get("config")
        if not served:
            problems.append("prod /health echoes no effective config: cannot prove this process "
                            "runs prod's serving knobs")
        else:
            problems += config_parity_problems(served, retrieval.effective_config())
            want = (served.get("fusion_weights") or {}).get("sha256")
            if want and want != incumbent_sha256:
                problems.append(f"the incumbent (sha256 {incumbent_sha256[:12]}) is not the file prod "
                                f"loads ({served['fusion_weights'].get('path')}, sha256 {want[:12]})")
    if problems:
        raise SystemExit("PREFLIGHT FAILED (no evaluation query was sent):\n  - " + "\n  - ".join(problems))
    return {"collection": collection, "points": coll.get("points_count"), "dense_dim": size,
            "mrl_dim": models.MRL_DIM, "tei_model_sha": (tei_info or {}).get("model_sha"),
            "prod_parity_checked": prod_health is not None}
