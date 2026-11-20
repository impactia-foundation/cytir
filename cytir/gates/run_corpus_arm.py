#!/usr/bin/env python3
"""Read-only, SHA-bound Task370 policy-off corpus arm using the serving scorer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
from cytir.core.digest import canonical_sha256  # noqa: E402
from cytir.gates import authority as gate  # noqa: E402

PROBES = ROOT / "data" / "gates" / "acceptance_probes.v1.json"
BOUND_RE = re.compile(r"^([0-9a-f]{64})=(.+)$")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def bound(value: str, label: str) -> tuple[Path, str]:
    match = BOUND_RE.match(value)
    if not match: raise SystemExit(f"{label} must be SHA256=PATH")
    path = Path(match.group(2))
    if not path.is_file(): raise SystemExit(f"{label} path missing: {path}")
    actual = digest(path)
    if actual != match.group(1): raise SystemExit(f"{label} SHA256 mismatch: expected={match.group(1)} actual={actual}")
    return path, actual


def task369():
    from cytir.gates import run_arm  # noqa: PLC0415  (moved from the task-369 scripts, task 677)
    return run_arm


def load_qvec_source(path: Path, cache_path: Path, cache_hash: str) -> tuple[dict, int]:
    """Verify immutable cache membership against its committed source manifest.

    The direct schema avoids a builder-specific alias: cache_sha256, query_count,
    query_keys_sha256, and query_sources[{path,sha256,rows}] are all mandatory.
    """
    doc = json.loads(path.read_text())
    required = {"cache_sha256", "query_count", "query_keys_sha256", "query_sources"}
    if required - set(doc): raise SystemExit(f"qvec query-source manifest missing {sorted(required-set(doc))}")
    if doc["cache_sha256"] != cache_hash: raise SystemExit("qvec source does not bind this cache SHA256")
    cache = json.loads(cache_path.read_text())
    if not isinstance(cache, dict) or not all(isinstance(key, str) for key in cache): raise SystemExit("cache must be a query-vector object")
    keys = sorted(cache)
    if doc["query_count"] != len(keys) or doc["query_keys_sha256"] != canonical_sha256(keys):
        raise SystemExit("qvec source does not bind exact cache keys/count")
    sources = doc["query_sources"]
    if not isinstance(sources, list) or not sources: raise SystemExit("qvec source has no query_sources")
    for source in sources:
        if not isinstance(source, dict) or {"path", "sha256", "rows"} - set(source): raise SystemExit("invalid qvec query source row")
        if not isinstance(source["path"], str) or not re.fullmatch(r"[0-9a-f]{64}", str(source["sha256"])) or not isinstance(source["rows"], int) or source["rows"] < 0:
            raise SystemExit("invalid qvec query source binding")
    return doc, len(keys)


def load_corpus(path: Path, target: dict) -> dict:
    doc = json.loads(path.read_text())
    required = {"identity", "qdrant_url", "collection", "status", "points_count", "representation", "supersede_sha256", "source_sha256"}
    if required - set(doc): raise SystemExit(f"corpus wrapper missing {sorted(required-set(doc))}")
    for key in ("identity", "qdrant_url", "collection", "status", "points_count"):
        if doc[key] != target[key]: raise SystemExit(f"corpus wrapper {key} does not match verified target")
    for key in ("supersede_sha256", "source_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(doc[key])): raise SystemExit(f"corpus wrapper invalid {key}")
    return doc


def assert_env() -> dict[str, str]:
    expected = dict(task369().PINNED_ENV)
    expected.update({"AW_COUNTRY_SCOPE_MODE": "policy_only", "AW_QHEAD_COUNTRY_LEGS": "0", "AW_QHEAD_COUNTRY_TRACE": "1"})
    bad = {key: (os.environ.get(key), value) for key, value in expected.items() if os.environ.get(key) != value}
    if bad: raise SystemExit(f"Task370 policy-off environment mismatch: {bad}")
    if not os.environ.get("AW_QHEAD_PATH") or not Path(os.environ["AW_QHEAD_PATH"]).is_file(): raise SystemExit("AW_QHEAD_PATH must name a versioned readable artifact")
    return expected


def serving_binding(retrieval) -> dict:
    """What dense policy and fusion weights this arm ACTUALLY ran.

    Read back from the loaded module's own constants, never from the
    environment it was asked to run with -- the running system echoes its
    effective config, so a stale or ignored knob shows up as a difference
    instead of being believed (deployment-doctrine 2).

    Task 604: every sealed arm before this one recorded 39 pinned keys and
    NONE of the dense/fusion knobs, so no arm in this repo could prove which
    ranker it had measured. It had, in fact, measured one production does not
    serve -- the `fusion_weights.json` fallback and rescore off, against
    production's candidate533 and rescore x3, worth 82 golds and +6 h@1.

    `retrieval.effective_config()` is the same surface `/health` echoes (task
    522), so a gate arm and the deployed API now answer the question the same
    way, from the same function.
    """
    config = retrieval.effective_config()
    weights = config.get("fusion_weights") or {}
    return {
        "schema": "aw.gate-serving-binding.v1",
        "mrl_dim": config.get("mrl_dim"),
        "dense_rescore": bool(config.get("dense_rescore")),
        "dense_oversampling": float(config.get("dense_oversampling")),
        "dense_ignore_quant": bool(config.get("dense_ignore_quant")),
        # may be None -- `retrieval` reads it as `int(...) or None`, and an arm
        # that recorded 0 for "the server's own default" would be lying about
        # a knob in exactly the way this block exists to stop.
        "dense_hnsw_ef": (int(retrieval._DENSE_HNSW_EF)
                          if retrieval._DENSE_HNSW_EF else None),
        "fusion_weights_path": weights.get("path"),
        "fusion_weights_sha256": weights.get("sha256"),
        "effective_config": config,
    }


#: Keys `trace()` adds BESIDE the four leg id lists. Additive, deliberately:
#: `cytir.gates.news_daily.cut_arm` and `cytir.gates.content_guard` both read this
#: object's existing keys, so changing any of their shapes would reach two live
#: gates. A new key beside them reaches nobody who does not ask for it.
#:
#: `urls` is what makes a per-LEG rank computable at all. ADR-043 A8 rule 4 wants
#: three arms -- dense-only, BM25-only and fused -- and a leg returns point ids
#: while a gold is a URL, so without the mapping the leg orders can be compared to
#: each other but never scored against the gold set. `retrieval` already builds it
#: (`_LAST_QUERY_PLAN_TRACE["candidate_urls"]`, retrieval.py:4377); it was simply
#: never carried out of the trace.
#:
#: `scores` is each leg's own score per id, which the sealed-IDF BM25 arm needs as
#: the baseline its re-scored order is compared against.
TRACE_EXTRA_KEYS = ("urls", "scores")

#: Written only when `capture_sparse=True`. Separate from TRACE_EXTRA_KEYS because
#: these two are NOT free: they cost one extra qdrant retrieve per query and they
#: grow the arm artifact, so a caller asks for them rather than getting them.
TRACE_SPARSE_KEYS = ("bm25_query", "sparse")

LEG_NAMES = ("dense", "bm25", "country_dense", "country_bm25")


def trace(retrieval, *, row: dict | None = None,
          capture_sparse: bool = False) -> dict:
    """The per-query trace an arm records. One home for the shape.

    `capture_sparse` adds the two inputs ADR-109's SEALED BM25 arm needs and
    nothing else can supply: the QUERY's bm25 sparse vector, and each traced
    candidate's STORED bm25 values. With them, `gate_leg_arms.sealed_bm25_order`
    re-ranks the pool under a sealed IDF table, which makes the BM25 arm
    layout-invariant by construction (A8 rule 4). Without them the BM25 reading is
    Qdrant's live one, whose idf comes from segment headers and therefore moves
    with an optimizer merge that wrote nothing.

    **The stored values are restricted to the QUERY's token indices, and that is
    not an optimisation — it is the arithmetic.** The sealed score is
    `sum_t idf(t)*q(t)*d(t)`, and `q(t)` is zero for every token outside the
    query, so a token the query does not contain cannot affect any candidate's
    score. Capturing whole stored vectors would multiply the arm's size by the
    average document vocabulary (~100 tokens against a 2-8 token query) to record
    numbers that are all multiplied by zero. Measured on the B01 shape: ~4 values
    per candidate rather than ~100.

    It is opt-in because it is not free: one extra `retrieve(..., with_vectors)`
    per query, and a larger artifact. An arm cut without it is not broken --
    `gate_leg_arms.sealed_arm_available()` reports the sealed arm as unavailable
    rather than silently reading the live leg under its name.
    """
    raw = retrieval._LAST_QUERY_PLAN_TRACE
    if not raw:
        empty = {"fast_path": True, **{name: [] for name in LEG_NAMES},
                 "urls": {}, "scores": {}}
        if capture_sparse:
            empty.update({"bm25_query": {}, "sparse": {}})
        return empty
    legs = raw.get("legs") or {}
    out = {"fast_path": False,
           **{name: [str(x.get("id")) for x in legs.get(name, [])]
              for name in LEG_NAMES}}
    # pid -> url, for exactly the ids the legs returned. Filtered rather than
    # copied whole, so an arm does not grow by the size of the entire candidate
    # pool; and `.get` rather than `[...]`, because a pool id with no recorded
    # url is a real (if rare) state and must not raise inside an arm cut.
    candidate_urls = raw.get("candidate_urls") or {}
    wanted = {pid for name in LEG_NAMES for pid in out[name]}
    out["urls"] = {pid: candidate_urls[pid] for pid in wanted
                   if candidate_urls.get(pid)}
    out["scores"] = {name: {str(x.get("id")): x.get("score")
                            for x in legs.get(name, ())}
                     for name in ("dense", "bm25")}
    if capture_sparse:
        out.update(capture_sparse_inputs(retrieval, row, wanted))
    return out


def capture_sparse_inputs(retrieval, row: dict | None,
                          wanted: set[str]) -> dict:
    """The query's bm25 vector and each traced candidate's stored bm25 values.

    Recomputes the query vector with `schema.bm25_tokens` -- the SAME function
    retrieval uses to build the live query (`retrieval.py` line ~3787) -- rather
    than reading it out of the trace, because putting it in the trace means
    editing `retrieval.py`, and `retrieval_sha256` is pinned evidence in every arm
    manifest. A gate capture may not move the serving binding.

    Refuses quietly in only one direction: with no `row` there is no query text to
    tokenise, so the capture is recorded as empty WITH a reason, never as a query
    that had no tokens -- an empty query vector scores every candidate 0.0 and
    would read as a leg that ranked nothing.
    """
    if not row or not (row.get("query") or row.get("q")):
        return {"bm25_query": {}, "sparse": {},
                "sparse_capture_skipped":
                    "no query text was passed to trace(row=...); the query's "
                    "bm25 vector cannot be recomputed, and an empty one would "
                    "score every candidate 0.0"}
    from cytir.serve.schema import COLLECTION, SPARSE_BM25, bm25_tokens
    query = bm25_tokens(row.get("query") or row.get("q"), row.get("lang"))
    out: dict = {"bm25_query": {str(k): float(v) for k, v in query.items()}}
    if not query or not wanted:
        out["sparse"] = {}
        return out
    indices = set(query)
    stored: dict[str, dict[str, float]] = {}
    points = (getattr(retrieval, "qdrant_io", None) or retrieval).client().retrieve(
        collection_name=COLLECTION, ids=sorted(wanted),
        with_payload=False, with_vectors=[SPARSE_BM25])
    for point in points:
        vectors = point.vector or {}
        sparse = vectors.get(SPARSE_BM25) if isinstance(vectors, dict) else None
        if sparse is None:
            continue
        pairs = zip(getattr(sparse, "indices", ()) or (),
                    getattr(sparse, "values", ()) or ())
        kept = {str(i): float(v) for i, v in pairs if int(i) in indices}
        if kept:
            stored[str(point.id)] = kept
    out["sparse"] = stored
    return out


def score(retrieval, row: dict, *, capture_sparse: bool = False) -> dict:
    retrieval._LAST_QUERY_PLAN_TRACE.clear()
    hits = retrieval.search(row["query"], k=10, lang=row.get("lang"), country=row.get("country"))
    urls = [hit.get("url") or "" for hit in hits]
    exact = next((i for i, url in enumerate(urls, 1) if gate.norm_url(url) == gate.norm_url(row["target_url"])), None)
    return {"urls": urls, "rank": exact,
            "trace": trace(retrieval, row=row, capture_sparse=capture_sparse)}


def write_artifact(out_dir: Path, arm_id: str, result: dict) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", arm_id): raise SystemExit("arm-id must be lowercase [a-z0-9_-]")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    path = out_dir / f"task370-arm.{arm_id}.{hashlib.sha256(payload.encode()).hexdigest()}.json"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError: raise SystemExit(f"refusing to overwrite existing arm artifact: {path}")
    with os.fdopen(fd, "w", encoding="utf-8") as f: f.write(payload)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--qvec-query-source-manifest", required=True)
    ap.add_argument("--target-manifest", required=True); ap.add_argument("--corpus-manifest", required=True)
    ap.add_argument("--out-dir", required=True, type=Path); ap.add_argument("--arm-id", required=True)
    ap.add_argument(
        "--capture-sparse", action="store_true",
        help=("also record each query's bm25 vector and its traced candidates' "
              "STORED bm25 values, restricted to the query's token indices -- "
              "the inputs ADR-109's sealed BM25 gate arm needs (A8 rule 4). "
              "Costs one extra qdrant retrieve per query and a larger artifact."))
    args = ap.parse_args()
    if not PROBES.is_file(): raise SystemExit(f"REFUSED: probe fixture missing: {PROBES} (the public cut ships none)")
    cache_path, cache_hash = bound(args.cache, "--cache")
    qvec_path, qvec_hash = bound(args.qvec_query_source_manifest, "--qvec-query-source-manifest")
    target_path, target_hash = bound(args.target_manifest, "--target-manifest")
    corpus_path, corpus_hash = bound(args.corpus_manifest, "--corpus-manifest")
    qvec_source, cache_vectors = load_qvec_source(qvec_path, cache_path, cache_hash)
    expected_env = assert_env(); target = task369().verify_target(str(target_path)); corpus = load_corpus(corpus_path, target)
    miss = gate.install_cached_embedder(str(cache_path)); from cytir.serve import retrieval  # noqa: E702
    result = {"cache_miss": 0, "sets": {}, "probes": {}, "task370": {"target_manifest": target, "target_manifest_sha256": target_hash, "corpus_manifest": corpus, "corpus_manifest_sha256": corpus_hash, "cache_sha256": cache_hash, "cache_vectors": cache_vectors, "qvec_query_source_manifest_sha256": qvec_hash, "qvec_query_source": qvec_source, "retrieval_sha256": digest(ROOT / "cytir/serve/retrieval.py"), "qhead_sha256": digest(Path(os.environ["AW_QHEAD_PATH"])), "pinned_env": expected_env, "effective_env": {key: os.environ.get(key) for key in expected_env}, "serving_binding": serving_binding(retrieval), "probe_fixture_sha256": digest(PROBES)}}
    rows_by_set = [(name, gate.load_rows(path)) for name, path in gate.SETS if Path(path).exists()]
    if Path(gate.SLICEB).exists(): rows_by_set.append(("slice_b", gate.load_sliceb(gate.SLICEB)))
    for name, rows in rows_by_set:
        per_q, h1, h10, total, reciprocal = {}, 0, 0, 0, 0.0
        for index, row in enumerate(rows):
            qid = str(row.get("id") or row.get("qid") or f"{name}-{index}")
            try:
                gold_urls, gold_domains = gate.gold_of(row); value = score(retrieval, {"query": row.get("query") or row.get("q"), "lang": row.get("lang"), "country": row.get("country"), "target_url": (gold_urls or [""])[0]}, capture_sparse=args.capture_sparse)
                value["rank"] = gate.rank_of(value["urls"], gold_urls, gold_domains) if gold_urls or gold_domains else None
            except Exception as exc: value = {"urls": [f"__ERROR__:{type(exc).__name__}"], "rank": None, "trace": {}}
            per_q[qid] = value
            if gold_urls or gold_domains:
                total += 1; rank = value["rank"]; h1 += rank == 1; h10 += bool(rank and rank <= 10); reciprocal += 1 / rank if rank else 0.0
        result["sets"][name] = {"n": len(rows), "n_gold": total, "h1": h1, "h10": h10, "mrr": round(reciprocal / total, 4) if total else None, "per_q": per_q}
    for probe in json.loads(PROBES.read_text()):
        try: result["probes"][probe["id"]] = {**score(retrieval, probe, capture_sparse=args.capture_sparse), "spec": probe}
        except Exception as exc: result["probes"][probe["id"]] = {"urls": [f"__ERROR__:{type(exc).__name__}"], "rank": None, "trace": {}, "spec": probe}
    result["cache_miss"] = miss[0]
    path = write_artifact(args.out_dir, args.arm_id, result); print(f"wrote {path}; cache_miss={miss[0]}")


if __name__ == "__main__": main()
