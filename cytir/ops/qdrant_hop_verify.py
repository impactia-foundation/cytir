"""qdrant_hop_verify.py — fingerprint a Qdrant collection and prove a version hop
did not change it.

Used by the `chain` mode (`cytir.ops.qdrant_hop_chain`) at every hop of the 1.12.x -> 1.18.x staged
upgrade chain (ADR-070 / task 305: a 1.12-era snapshot cannot be restored
directly into >=1.18 — the storage backend moved RocksDB -> Gridstore and the
per-segment storage-type enum was renamed at 1.13.0).

A hop is only "OK" if the collection is byte-for-byte the same *contract*
afterwards, not merely loadable:

  count      exact count (not the approximate `points_count` field, which lies
             while an optimizer is running — memory `aw-heavy-rewrite-degrades-prod`)
  config     the load-bearing config-B fields (dense size/distance/on_disk,
             sparse modifier/on_disk/datatype, hnsw m/ef_construct/on_disk,
             scalar-int8 quantization, on_disk_payload)
  indexes    the payload index set + their types
  content    payload + dense + sparse of a fixed sample of point ids, digested
             (drift doctrine: compare content, not just counts — offsetting
             errors hide behind clean aggregates)
  queries    replays the exact query shapes AW serves (dense leg, sparse BM25
             leg, RRF fusion, facet, payload-filtered scroll) with vectors
             captured from the collection itself, and compares ids + scores

Deliberately speaks raw REST via urllib, NOT qdrant-client: the point of this
tool is to observe what several *server* versions do, so it must not route
through a client library whose own behaviour varies with the server version.

  chain:    qdrant_hop_verify.py chain --help   (the whole staged walk with docker; its runbook)
  capture:  qdrant_hop_verify.py capture --url http://localhost:6421 \
                --collection aw_b1 --out baseline.json
  compare:  qdrant_hop_verify.py compare --url http://localhost:6421 \
                --collection aw_b1 --baseline baseline.json --label v1.13.6

Exit 0 = identical. Exit 1 = a real difference (fail the hop). Exit 2 = usage/
transport error. Never exits 0 on a check it could not run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.error

from cytir.core import internal_http as IH
from cytir.index import prod_guard as PG

# Score tolerance across server versions. MEASURED, not assumed: 1.16.x changed
# how int8-quantized dense scores are computed/rescored, so a self-match that
# scored 0.995651 on 1.12.4 scores 0.997568 on 1.16.3 (task 339 dry run). Scores
# are NOT bit-stable across the RocksDB->Gridstore era; rank ORDER is the real
# contract, and it is checked separately and strictly (tie-aware). This default
# is deliberately tight so a hop that shifts scores has to say so out loud.
SCORE_EPS = 1e-3

# Separate epsilon for deciding what counts as a TIE. Tie detection and score
# tolerance are different questions: ties are exact-equal scores (RRF's 1/(60+k),
# identical BM25 sums), while tolerance is how far a score may drift and still
# mean the same thing. Using one number for both makes the tie groups swallow the
# whole result list as soon as the tolerance is loosened — which silently turns a
# strict order check into a set check. Keep them apart.
TIE_EPS = 1e-9

# How many floats of a stored vector are meaningful for an identity digest.
# Vectors are stored f32; 6 decimals is well inside f32 precision and immune to
# repr differences.
VEC_ROUND = 6


# ------------------------------------------------------------------ transport -
class QdrantError(RuntimeError):
    pass


def _req(url: str, method: str = "GET", body=None, timeout: float = 600.0):
    try:
        return IH.fetch_json(url, body=body, method=method, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise QdrantError(f"{method} {url} -> HTTP {e.code}: "
                          f"{e.read().decode()[:600]}") from None
    except Exception as e:  # transport — never swallow, never treat as "empty"
        raise QdrantError(f"{method} {url} -> {type(e).__name__}: {e}") from None


#: Names the reason for scrolling a Qdrant that carries a prod name: set ONLY by cytir.ops.qdrant_hop_chain
#: on the calls that target the hop container (127.0.0.1:$PORT, stopped copy, not serving).
HOP_COPY_ENV = "AW_HOP_VERIFY_HOP_COPY"
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
#: The SERVING Qdrant's REST port on the prod box. It is loopback there, so loopback alone proves nothing.
SERVING_PORT = 6333


def _scroll_allowance(url: str) -> dict:
    """`PG.check` kwargs for a scroll: the override counts for a LOOPBACK url only."""
    reason = os.environ.get(HOP_COPY_ENV, "").strip()
    host, port = PG._hostport(url)
    if port in (SERVING_PORT, PG.PROD_PORT):
        return {}
    if any(u.strip() and PG._hostport(u.strip()) == (host, port)
           for u in os.environ.get("AW_PROD_URL", "").split(",")):
        return {}
    if reason and host in _LOOPBACK:
        return {"allow_prod_scroll": True, "reason": reason}
    return {}


class Q:
    def __init__(self, url: str, collection: str, timeout: float = 600.0):
        self.url = url.rstrip("/")
        self.c = collection
        self.timeout = timeout

    def get(self, path):
        return _req(f"{self.url}{path}", timeout=self.timeout)["result"]

    def post(self, path, body):
        return _req(f"{self.url}{path}", "POST", body, timeout=self.timeout)["result"]

    def info(self):
        return self.get(f"/collections/{self.c}")

    def exact_count(self):
        return self.post(f"/collections/{self.c}/points/count", {"exact": True})["count"]

    def scroll(self, limit, with_vector=False, with_payload=True, offset=None, flt=None):
        path = f"/collections/{self.c}/points/scroll"
        try:
            PG.check(self.url, "POST", path, collection=self.c, **_scroll_allowance(self.url))
        except PG.ProdReadRefused as e:
            raise QdrantError(str(e)) from None
        b = {"limit": limit, "with_payload": with_payload, "with_vector": with_vector}
        if offset is not None:
            b["offset"] = offset
        if flt is not None:
            b["filter"] = flt
        return self.post(path, b)

    def by_ids(self, ids, with_vector=True):
        return self.post(f"/collections/{self.c}/points",
                         {"ids": ids, "with_payload": True, "with_vector": with_vector})

    def query(self, body):
        return self.post(f"/collections/{self.c}/points/query", body)["points"]

    def facet(self, key, limit=20):
        return self.post(f"/collections/{self.c}/facet", {"key": key, "limit": limit})


# -------------------------------------------------------------- config subset -
def config_subset(info: dict) -> dict:
    """The load-bearing config-B fields. Curated on purpose: newer servers add
    incidental keys, and a fingerprint that fails on those is a fingerprint
    nobody trusts. The FULL config is stored alongside for human diff."""
    p = info.get("config", {}).get("params", {})
    out = {"on_disk_payload": p.get("on_disk_payload")}

    vec = p.get("vectors") or {}
    if isinstance(vec, dict) and "size" in vec:      # unnamed single vector
        vec = {"": vec}
    for name, v in sorted(vec.items()):
        out[f"dense.{name}"] = {
            "size": v.get("size"), "distance": v.get("distance"),
            "on_disk": v.get("on_disk"), "datatype": v.get("datatype"),
        }

    for name, v in sorted((p.get("sparse_vectors") or {}).items()):
        idx = v.get("index") or {}
        out[f"sparse.{name}"] = {
            "modifier": v.get("modifier"),
            "on_disk": idx.get("on_disk"), "datatype": idx.get("datatype"),
        }

    h = info.get("config", {}).get("hnsw_config") or {}
    out["hnsw"] = {"m": h.get("m"), "ef_construct": h.get("ef_construct"),
                   "on_disk": h.get("on_disk")}

    q = (info.get("config", {}).get("quantization_config") or {})
    sc = q.get("scalar") or {}
    out["quantization"] = {"kind": "scalar" if sc else (list(q) or [None])[0],
                           "type": sc.get("type"), "always_ram": sc.get("always_ram"),
                           "quantile": sc.get("quantile")}
    return out


def payload_indexes(info: dict) -> dict:
    out = {}
    for field, spec in (info.get("payload_schema") or {}).items():
        out[field] = spec.get("data_type") if isinstance(spec, dict) else spec
    return out


# ------------------------------------------------------------------- digests --
def _canon_point(p: dict) -> str:
    v = p.get("vector") or {}
    norm_v = {}
    if isinstance(v, dict):
        for name, val in sorted(v.items()):
            if isinstance(val, dict) and "indices" in val:       # sparse
                pairs = sorted(zip(val.get("indices") or [], val.get("values") or []))
                norm_v[name] = [[int(i), round(float(x), 5)] for i, x in pairs]
            elif isinstance(val, list):                          # dense
                norm_v[name] = [round(float(x), VEC_ROUND) for x in val]
            else:
                norm_v[name] = val
    return json.dumps({"id": str(p.get("id")), "payload": p.get("payload"),
                       "vector": norm_v},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def content_digest(points: list) -> dict:
    lines = sorted(_canon_point(p) for p in points)
    h = hashlib.sha256()
    for ln in lines:
        h.update(ln.encode())
        h.update(b"\n")
    return {"n": len(lines), "sha256": h.hexdigest()}


def hits(res: list) -> list:
    return [[str(p["id"]), round(float(p.get("score", 0.0)), 6)] for p in res]


# ------------------------------------------------------------------- capture --
def build_queries(q: Q, info: dict, seed: int, k: int, per_leg: int = 2) -> dict:
    """Derive the replayable query set FROM THE DATA, so compare/capture use
    byte-identical inputs on every server version. Stored in the fingerprint."""
    dense_names = list((info["config"]["params"].get("vectors") or {}).keys())
    sparse_names = list((info["config"]["params"].get("sparse_vectors") or {}).keys())
    rnd = random.Random(seed)

    probe, _ = _pair(q.scroll(max(64, per_leg * 32), with_vector=True))
    if not probe:
        raise QdrantError("collection is empty — nothing to fingerprint")
    rnd.shuffle(probe)

    queries = {}
    for dn in dense_names:
        for i, pt in enumerate(p for p in probe if (p.get("vector") or {}).get(dn)):
            if i >= per_leg:
                break
            queries[f"dense[{dn}]#{i}"] = {
                "kind": "dense", "using": dn, "limit": k,
                "vector": [float(x) for x in pt["vector"][dn]],
            }
    for sn in sparse_names:
        for i, pt in enumerate(p for p in probe if (p.get("vector") or {}).get(sn)):
            if i >= per_leg:
                break
            sv = pt["vector"][sn]
            queries[f"sparse[{sn}]#{i}"] = {
                "kind": "sparse", "using": sn, "limit": k,
                "vector": {"indices": [int(x) for x in sv["indices"]],
                           "values": [float(x) for x in sv["values"]]},
            }
    # the exact hybrid shape AW serves
    d0 = next((v for kk, v in queries.items() if kk.startswith("dense[")), None)
    s0 = next((v for kk, v in queries.items() if kk.startswith("sparse[")), None)
    if d0 and s0:
        queries["rrf"] = {"kind": "rrf", "limit": k, "dense": d0, "sparse": s0}

    # facet + filtered scroll on a real indexed keyword field
    kw = [f for f, t in payload_indexes(info).items() if t == "keyword"]
    if kw:
        key = sorted(kw)[0]
        queries[f"facet[{key}]"] = {"kind": "facet", "key": key, "limit": 20}
        fc = q.facet(key, 20).get("hits") or []
        if fc:
            val = fc[0]["value"]
            queries[f"filter[{key}={val}]"] = {
                "kind": "filter_scroll", "key": key, "value": val, "limit": k}
    return queries


def _pair(scroll_result):
    return scroll_result.get("points", []), scroll_result.get("next_page_offset")


def run_queries(q: Q, queries: dict) -> dict:
    out = {}
    for name, spec in queries.items():
        kind = spec["kind"]
        if kind == "dense":
            out[name] = hits(q.query({"query": spec["vector"], "using": spec["using"],
                                      "limit": spec["limit"]}))
        elif kind == "sparse":
            out[name] = hits(q.query({"query": spec["vector"], "using": spec["using"],
                                      "limit": spec["limit"]}))
        elif kind == "rrf":
            d, s = spec["dense"], spec["sparse"]
            out[name] = hits(q.query({
                "prefetch": [
                    {"query": d["vector"], "using": d["using"], "limit": 50},
                    {"query": s["vector"], "using": s["using"], "limit": 50}],
                "query": {"fusion": "rrf"}, "limit": spec["limit"]}))
        elif kind == "facet":
            res = q.facet(spec["key"], spec["limit"]).get("hits") or []
            out[name] = sorted([[str(h["value"]), int(h["count"])] for h in res])
        elif kind == "filter_scroll":
            flt = {"must": [{"key": spec["key"], "match": {"value": spec["value"]}}]}
            pts, _ = _pair(q.scroll(spec["limit"], with_payload=False, flt=flt))
            out[name] = sorted(str(p["id"]) for p in pts)
        else:
            raise QdrantError(f"unknown query kind {kind!r}")
    return out


def fingerprint(url, collection, seed, k, digest_points, timeout, queries=None,
                sample_ids=None, per_leg=2):
    q = Q(url, collection, timeout)
    info = q.info()
    fp = {
        "collection": collection,
        "server_version": _req(f"{url.rstrip('/')}/").get("version"),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": info.get("status"),
        "exact_count": q.exact_count(),
        "segments_count": info.get("segments_count"),
        "config": config_subset(info),
        "config_full": info.get("config"),
        "payload_indexes": payload_indexes(info),
    }
    if sample_ids is None:
        pts, _ = _pair(q.scroll(digest_points, with_vector=True))
        sample_ids = [p["id"] for p in pts]
        fp["scroll_first_ids"] = [str(i) for i in sample_ids[:20]]
    fp["sample_ids"] = sample_ids
    # fetch BY ID (not by scroll) so the sample is identical regardless of any
    # scroll-ordering change between server versions
    fp["content"] = content_digest(q.by_ids(sample_ids) if sample_ids else [])
    fp["queries"] = (queries if queries is not None
                     else build_queries(q, info, seed, k, per_leg))
    fp["query_results"] = run_queries(q, fp["queries"])
    return fp


# ------------------------------------------------------------------- compare --
def _cmp_ranked(b: list, c: list, eps: float):
    """Compare two scored result lists as a RANKING, not as a byte string.

    Ties are real: with RRF (and with any leg where several docs share a score)
    the order inside a tie group is arbitrary and varies run to run — comparing
    the raw id sequence turns that into a false alarm. So: build the tie groups
    from the BASELINE scores, require the current result to fill the same groups
    in the same order, and report score drift separately (it is the thing that
    actually changed across the 1.12->1.18 quantization rewrite).

    Returns (ok, detail-string)."""
    groups, cursor = [], 0
    while cursor < len(b):
        s = b[cursor][1]
        j = cursor
        while j < len(b) and abs(b[j][1] - s) <= TIE_EPS:
            j += 1
        groups.append(set(str(x[0]) for x in b[cursor:j]))
        cursor = j

    problems, pos = [], 0
    for gi, g in enumerate(groups):
        got = set(str(x[0]) for x in c[pos:pos + len(g)])
        if got != g:
            problems.append(f"tie-group {gi} (rank {pos}): expected {sorted(g)[:3]} "
                            f"got {sorted(got)[:3]}")
        pos += len(g)

    bs = {str(x[0]): x[1] for x in b}
    drift = [abs(bs[str(x[0])] - x[1]) for x in c if str(x[0]) in bs]
    mx = max(drift) if drift else 0.0
    overlap = len(set(bs) & set(str(x[0]) for x in c))
    rank1 = (b and c and str(b[0][0]) == str(c[0][0]))
    stats = (f"overlap@{len(b)}={overlap}/{len(b)}, rank1={'same' if rank1 else 'CHANGED'}, "
             f"max score drift {mx:.6f}")
    if problems:
        return False, stats + " | " + "; ".join(problems[:2])
    if mx > eps:
        return False, stats + f" | drift > tolerance {eps}"
    return True, f"order identical ({len(groups)} tie-groups), " + stats


def compare(base: dict, cur: dict, label: str, eps: float = SCORE_EPS,
            ann_drift: str = "fail") -> int:
    bad = []
    ok = []

    def chk(name, b, c, eq=None):
        good = (b == c) if eq is None else eq(b, c)
        (ok if good else bad).append((name, b, c))

    chk("exact_count", base["exact_count"], cur["exact_count"])
    chk("status_green", "green", cur["status"])
    for key in sorted(set(base["config"]) | set(cur["config"])):
        chk(f"config.{key}", base["config"].get(key), cur["config"].get(key))
    chk("payload_indexes", base["payload_indexes"], cur["payload_indexes"])
    chk("content_digest", base["content"], cur["content"])

    ann = []
    for name in sorted(set(base["query_results"]) | set(cur["query_results"])):
        b = base["query_results"].get(name)
        c = cur["query_results"].get(name)
        if b is None or c is None:
            bad.append((f"query.{name}", b, c))
            continue
        if len(b) != len(c):
            bad.append((f"query.{name}:len", len(b), len(c)))
            continue
        if b and isinstance(b[0], list) and len(b[0]) == 2 and isinstance(b[0][1], float):
            verdict, detail = _cmp_ranked(b, c, eps)
            (ok if verdict else ann).append((f"query.{name}", f"{len(b)} hits", detail))
        elif b != c:
            bad.append((f"query.{name}", b, c))
        else:
            ok.append((f"query.{name}", f"{len(b)} entries", "identical"))

    # SEVERITY MODEL (decided empirically in the task-339 dry run).
    #
    # Everything checked above is a DATA contract: exact count, config-B fields,
    # payload indexes, and a content digest over a fixed point-id sample
    # (payload + dense + sparse, byte-normalised). If those hold, the collection
    # that came out of the hop IS the collection that went in.
    #
    # The ranked queries are a SEARCH contract, and it is not the same thing.
    # Measured across 1.13.6 -> 1.16.3 on a provably untouched collection: the
    # BM25 leg, facets and filtered scrolls stay byte-identical, while the dense
    # leg keeps only 5-8 of its top 10 and scores move by up to 0.012 — qdrant
    # rewrote HNSW traversal and int8-quantized rescoring in that window. That is
    # a real ranking change, and judging it belongs to the eval harness
    # (cytir.gates.promote_verify Tier 2 / run_eval.py), not to a structural fingerprint:
    # a fingerprint can say "the engine moved", never "the answers got worse".
    # So when the content digest proves the data is intact, --ann-drift=warn
    # reports the movement QUANTIFIED (overlap@k, rank-1 stability, score drift)
    # and lets the chain reach the gate that can judge it. The default, fail, is
    # for hops that are supposed to be completely invisible.
    content_clean = base["content"] == cur["content"]
    warn = []
    for name, n, detail in ann:
        if ann_drift == "warn" and content_clean:
            warn.append((name, n, "ANN DRIFT (content identical): " + detail))
        else:
            bad.append((name, n, detail))

    print(f"--- hop verify [{label}] server={cur.get('server_version')} "
          f"collection={cur['collection']} ---")
    for name, b, c in ok:
        print(f"  OK    {name}: {_short(c)}")
    for name, b, c in warn:
        print(f"  WARN  {name}: {_short(c)}")
    for name, b, c in bad:
        print(f"  FAIL  {name}: baseline={_short(b)} != now={_short(c)}")
    print(f"--- {len(ok)} ok / {len(warn)} warn / {len(bad)} FAILED ---")
    return 1 if bad else 0


def _short(v, n=220):
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + f"...(+{len(s)-n})"


# ---------------------------------------------------------------------- main --
def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    if argv and argv[0] == "chain":     # task 2370: the staged version walk (was a shell driver)
        from cytir.ops.qdrant_hop_chain import main as chain_main
        return int(chain_main(argv[1:], prog=f"{prog or 'qdrant_hop_verify.py'} chain") or 0)
    ap = argparse.ArgumentParser(prog=prog or "qdrant_hop_verify.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["capture", "compare"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--out")
    ap.add_argument("--baseline")
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=339)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--digest-points", type=int, default=500,
                    help="how many points to content-digest (fixed id sample)")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--ann-drift", choices=["fail", "warn"], default="fail",
                    help="severity when a ranked query's ORDER changes while the "
                         "collection content is provably identical. Use 'warn' only "
                         "when a real eval gate runs downstream.")
    ap.add_argument("--queries-per-leg", type=int, default=2,
                    help="how many probe queries to capture per named vector")
    ap.add_argument("--score-tolerance", type=float, default=SCORE_EPS,
                    help="max |score delta| tolerated per hit. Rank ORDER is "
                         "always compared strictly (tie-aware) regardless.")
    a = ap.parse_args(argv)

    try:
        if a.mode == "capture":
            if not a.out:
                ap.error("capture needs --out")
            fp = fingerprint(a.url, a.collection, a.seed, a.topk,
                             a.digest_points, a.timeout, per_leg=a.queries_per_leg)
            with open(a.out, "w") as fh:
                json.dump(fp, fh, indent=1)
            print(f"captured {a.collection} @ {fp['server_version']}: "
                  f"{fp['exact_count']:,} pts, {len(fp['queries'])} queries, "
                  f"content sha={fp['content']['sha256'][:16]} -> {a.out}")
            return 0

        if not a.baseline:
            ap.error("compare needs --baseline")
        with open(a.baseline) as fh:
            base = json.load(fh)
        cur = fingerprint(a.url, a.collection, a.seed, a.topk, a.digest_points,
                          a.timeout, queries=base["queries"],
                          sample_ids=base["sample_ids"])
        return compare(base, cur, a.label or a.url, a.score_tolerance, a.ann_drift)
    except QdrantError as e:
        print(f"TRANSPORT/SERVER ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
