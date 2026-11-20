#!/usr/bin/env python3
"""Run one Task-369 arm using gate_authority's immutable cached query vectors.

No collection writes occur.  The caller chooses the one arm variable with
environment: legacy/current head, learned_only/current head, or
learned_only/retrained head (AW_QHEAD_PATH).  This runner records enough
configuration and edge diagnostics for compare_arms.py to reject a fake A/B.
"""
from __future__ import annotations
from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import argparse, hashlib, json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [ROOT]
from cytir.gates import authority as gate  # noqa: E402
from cytir.core.serve_layout import SERVER_REL  # noqa: E402

PINNED_ENV = {
    "AW_AUTHLEG":"1", "AW_AUTHORITY":"1", "AW_AUTHORITY_MAXBOOST":"0.8", "AW_AUTHORITY_MIN":"0.3",
    "AW_AUTHORITY_MIN_CENTRALITY":"0.3", "AW_AUTHORITY_RELGATE":"0.7", "AW_AUTHORITY_W":"1.0",
    "AW_BATCH_LEGS":"1", "AW_BM25_PER_TOKEN":"1", "AW_COMMERCIAL":"1", "AW_DEDUP_V2":"1",
    "AW_DENSE_HNSW_EF":"384", "AW_EMBEDDER":"arctic", "AW_FOLD_ACCENTS":"1", "AW_HUBFLOOR":"1",
    "AW_HUBFLOOR_DENSE_MAX":"0.88", "AW_HUBFLOOR_DISTINCT_MAX":"0.10", "AW_HUBFLOOR_MINBETA":"0.5",
    "AW_HUBFLOOR_W":"-1.5", "AW_OFF_COUNTRY_LOCBOUND_ONLY":"0", "AW_OFF_COUNTRY_NONBOUND_CLASSES":"ref",
    "AW_OFF_COUNTRY_NONBOUND_W":"0.0", "AW_OFF_COUNTRY_W":"-2.0", "AW_QDRANT_TIMEOUT":"8",
    "AW_QHEAD":"1", "AW_QHEAD_KNOBS":"country_scope,off_country_mag", "AW_RANKHEAD":"1", "AW_RELFLOOR":"1",
    "AW_RERANK":"0", "AW_TEI_URL":"", "AW_TRUSTFLOOR":"1", "AW_TRUSTFLOOR_MINBETA":"0.24",
    "AW_TRUSTFLOOR_POOL_MAX":"0.66", "AW_TRUSTFLOOR_REL_BETA":"0.7", "AW_TRUSTFLOOR_TAU":"0.6", "AW_TWO_STAGE":"1",
}

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def edge_rows(path):
    return json.load(open(path, encoding="utf-8"))

def verify_target(path):
    """Assert the caller supplied an independently captured R5b twin identity.

    The manifest is deliberately an input, not a fact copied into this script:
    it is recorded after a quiet-state verification and stops a stale scratch
    collection (notably :6442) being mislabeled as R5b.
    """
    want = json.load(open(path, encoding="utf-8"))
    required = {"qdrant_url", "collection", "status", "points_count", "identity"}
    missing = required - set(want)
    if missing:
        raise SystemExit(f"target manifest missing {sorted(missing)}")
    if want["qdrant_url"].rstrip("/") != os.environ["AW_QDRANT_URL"].rstrip("/") or want["collection"] != os.environ["AW_COLLECTION"]:
        raise SystemExit("target manifest URL/collection does not match this arm")
    endpoint = f"{want['qdrant_url'].rstrip('/')}/collections/{want['collection']}"
    try:
        with IH.urlopen(endpoint, timeout=10) as response:
            got = json.load(response)["result"]
    except Exception as exc:
        raise SystemExit(f"cannot verify target collection at {endpoint}: {type(exc).__name__}: {exc}")
    actual = {"status": got.get("status"), "points_count": got.get("points_count")}
    expected = {"status": want["status"], "points_count": want["points_count"]}
    if actual != expected:
        raise SystemExit(f"target manifest mismatch: expected={expected}, actual={actual}")
    return want

def build_cache(path, edges):
    """One immutable cache for gate golds *and* scope edge rows."""
    import numpy as np
    from cytir.serve import models
    queries = sorted(set(gate.all_queries()) | {e["query"] for e in edges})
    vecs = {}
    for q in queries:
        models.embed_query(q)
        vecs[q] = np.asarray(models.last_full_query_vec(), dtype="float32").tolist()
    with open(path, "x", encoding="utf-8") as f:  # never overwrite the yardstick
        json.dump(vecs, f, sort_keys=True)
    print(f"wrote immutable cache {path}: {len(vecs)} vectors")

def assert_env():
    bad = {k: (os.environ.get(k), v) for k, v in PINNED_ENV.items() if os.environ.get(k) != v}
    if bad:
        raise SystemExit(f"task369 production-env mismatch: {bad}")
    if os.environ.get("AW_COUNTRY_SCOPE_MODE") not in {"legacy", "learned_only"}:
        raise SystemExit("set CYTIR_COUNTRY_SCOPE_MODE=legacy|learned_only")
    if not os.environ.get("AW_COLLECTION") or not os.environ.get("AW_QDRANT_URL"):
        raise SystemExit("CYTIR_COLLECTION and CYTIR_QDRANT_URL are required")
    if not os.environ.get("AW_HUBFLOOR_TABLE"):
        raise SystemExit("CYTIR_HUBFLOOR_TABLE is required")
    for key in ("AW_QHEAD_PATH", "AW_HUBFLOOR_TABLE"):
        path = os.environ.get(key)
        if not path or not os.path.isfile(path):
            raise SystemExit(f"{key} is missing or not a file: {path!r}")

def add_edges(out, cache, edges, target):
    # gate.run installs the cached embedder in this process. Reuse it: edges never
    # invoke a live embedder and cache misses are measurement failures.
    gate.install_cached_embedder(cache)
    from cytir.serve import models, retrieval
    results = []
    errors = 0
    for e in edges:
        try:
            country = e.get("country")
            hits = retrieval.search(e["query"], k=10, lang=e.get("lang"), country=country)
            urls = [h.get("url") or "" for h in hits]
            rank = gate.rank_of(urls, e.get("gold_urls") or [], e.get("gold_domains") or [])
            if country:
                # An explicit country is an input contract, not an inferred scope.
                # Record its actual ranked output; never manufacture an observed
                # classifier result from the request parameter.
                results.append({"id": e["id"], "requested_country": country, "urls": urls,
                                "gold_rank": rank, "max_gold_rank": e.get("max_gold_rank")})
                continue
            else:
                models.embed_query(e["query"])
                _hard, resolved, conf = retrieval._effective_free_text_country_scope(
                    e["query"], retrieval._query_jurisdiction(e["query"])[1], models.last_full_query_vec())
                resolved = resolved or _hard
            results.append({"id": e["id"], "scope": resolved, "confidence": round(float(conf), 6),
                            "expect_scope": e.get("expect_scope"), "urls": urls, "gold_rank": rank,
                            "max_gold_rank": e.get("max_gold_rank")})
        except Exception as exc:  # recorded; comparator makes errors fatal
            errors += 1
            results.append({"id": e["id"], "error": f"{type(exc).__name__}: {exc}", "expect_scope": e.get("expect_scope")})
    doc = json.load(open(out))
    qhead_path = os.environ["AW_QHEAD_PATH"]
    hubfloor_path = os.environ["AW_HUBFLOOR_TABLE"]
    retrieval_path = os.path.join(ROOT, SERVER_REL, "retrieval.py")
    doc["task369"] = {"cache_sha256": digest(cache), "cache_vectors": len(json.load(open(cache))),
                       "mode": os.environ["AW_COUNTRY_SCOPE_MODE"], "qhead_path": qhead_path,
                       "qhead_sha256": digest(qhead_path),
                       "retrieval_sha256": digest(retrieval_path),
                       "collection": os.environ["AW_COLLECTION"], "edge_results": results, "edge_errors": errors,
                       "pinned_env": {k: os.environ.get(k) for k in PINNED_ENV},
                       "hubfloor_table": hubfloor_path,
                       "hubfloor_sha256": digest(hubfloor_path)}
    doc["task369"]["target_manifest"] = {"identity": target["identity"], "status": target["status"],
                                        "points_count": target["points_count"], "collection": target["collection"],
                                        "qdrant_url": target["qdrant_url"]}
    with open(out, "w", encoding="utf-8") as f: json.dump(doc, f, indent=2, sort_keys=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--out")
    ap.add_argument("--edges", default=os.path.join(ROOT, "data", "gates", "scope_edges.json"))
    ap.add_argument("--target-manifest")
    ap.add_argument("--build-cache", action="store_true")
    args = ap.parse_args(); edges = edge_rows(args.edges)
    if args.build_cache:
        build_cache(args.cache, edges); return
    assert_env()
    if not args.target_manifest: ap.error("--target-manifest is required for an arm run")
    target = verify_target(args.target_manifest)
    if not args.out: ap.error("--out is required unless --build-cache")
    gate.run(args.out, args.cache)
    add_edges(args.out, args.cache, edges, target)

if __name__ == "__main__": main()
