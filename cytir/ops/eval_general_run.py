"""Task 206 Part A (restored into cyt ops, task 1088) — the STANDING general-query eval scorer for data/eval_general.yaml.

General queries have NO single gold URL, so we do NOT score hit@k-on-gold. Instead
an offline Sonnet judge grades the live/dev AW top-k for:
  FLOOD    = fraction of top-k that are HIGH-TRUST but OFF-TOPIC (the antirez/nbb class)
  rel@1    = is the #1 result actually on-topic
plus a mechanical routing metric from the rank-head intent probe:
  routing  = P(general) for the query, and whether it clears the gate (>=0.5).

Two stages (co-location: retrieval must run beside the twin on the GPU box; the LLM
judge needs the API key on the orchestrator host):

  # (GPU box) run twin retrieval + head routing for each config -> results.jsonl
  cyt ops eval-general-run retrieve --config off   --out results_off.jsonl
  cyt ops eval-general-run retrieve --config v2_206 --out results_v2_206.jsonl

  # (orchestrator host) Sonnet-judge the dumped top-k, emit per-query scores + summary
  cyt ops eval-general-run judge --results results_off.jsonl --out scored_off.json

The gate/probe harness wires this in as: FLOOD(mean) must NOT rise vs the committed
baseline (data/eval_general_baseline.json), and routing-gated rate on flood_prone
must NOT fall. Head-independent — any future ranking change is measured here."""
from __future__ import annotations
import argparse, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlsplit

from cytir.consent.block import ConsentBlocked
from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HITRUST = 0.85


def host(u):
    h = urlsplit(u or "").netloc.lower()
    return h[4:] if h.startswith("www.") else h


# ------------------------------------------------------------------ retrieve
def cmd_retrieve(args):
    import numpy as np
    D206 = args.workdir
    ART = os.path.join(D206, "artifacts")
    os.environ.setdefault("AW_COLLECTION", "aw_v4_sat")
    os.environ.setdefault("AW_QDRANT_URL", "http://localhost:6333")
    os.environ.update(AW_TWO_STAGE="1", AW_BM25_PER_TOKEN="1", AW_KSTAB="1",
                      AW_AUTHLEG="1", AW_TRUST_RELGATE="0")
    if args.config != "off":
        src = "head_clf.json" if args.config == "v2_205" else "head_clf_206.json"
        base = args.head205_dir if args.config == "v2_205" else ART
        clf = json.load(open(os.path.join(base, src)))
        head = {"variant": "v2", "beta_gen": 0.7, "beta_max": 0.9,
                "w": clf["w"], "b": clf["b"]}
        hp = os.path.join(ART, f"rankhead_gen_{args.config}.json")
        json.dump(head, open(hp, "w"))
        os.environ["AW_RANKHEAD"] = "1"
        os.environ["AW_RANKHEAD_PATH"] = hp
    else:
        os.environ["AW_RANKHEAD"] = "0"

    # eval query -> 1024-d (from pool_vecs) ; routing P(general) for BOTH heads
    ids = set(json.load(open(os.path.join(ART, "eval_general.ids.json"))))
    pool_ids = json.load(open(os.path.join(ART, "pool_ids.json")))
    pv = np.load(os.path.join(ART, "pool_vecs.npy"))
    idx = {pid: i for i, pid in enumerate(pool_ids)}
    import yaml
    eg = os.path.join(D206, "data", "eval_general.yaml")
    if not os.path.exists(eg):
        eg = os.path.join(HERE, "data", "eval_general.yaml")
    ev = yaml.safe_load(open(eg))["queries"]
    ev = [e for e in ev if e["id"] in ids]
    vec = {e["query"]: pv[idx[e["id"]]] for e in ev}

    def sig(x, clf):
        w = np.asarray(clf["w"], "float32")
        return float(1 / (1 + np.exp(-(x @ w + clf["b"]))))
    clf205 = json.load(open(os.path.join(args.head205_dir, "head_clf.json")))
    clf206 = json.load(open(os.path.join(ART, "head_clf_206.json")))

    sys.path.insert(0, os.path.join(D206, "server"))
    import cytir.serve.models as models
    MRL = 128

    def _emb(q):
        full = np.asarray(vec[q], "float32")
        models._tls.full_qvec = full
        v = full[:MRL]
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).tolist(), None
    models.embed_query = _emb
    import cytir.serve.retrieval as R

    rows = []
    for e in ev:
        q = e["query"]
        try:
            res = R.search(q, k=args.k)
        except Exception as ex:  # noqa: BLE001
            res = []
            sys.stderr.write(f"ERR {e['id']}: {ex}\n")
        x = np.asarray(vec[q], "float32")
        rows.append({
            "id": e["id"], "query": q, "lang": e["lang"], "src": e["src"],
            "vertical": e.get("vertical"), "flood_prone": e.get("flood_prone"),
            "p_general_205": sig(x, clf205), "p_general_206": sig(x, clf206),
            "results": [{"rank": i + 1, "url": r.get("url"), "title": r.get("title"),
                         "trust_tier": r.get("trust_tier"), "source_type": r.get("source_type"),
                         "score": r.get("score")} for i, r in enumerate(res)],
        })
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[retrieve {args.config}] {len(rows)} queries -> {args.out}")


# ------------------------------------------------------------------ judge
JUDGE_SYS = (
    "You grade an authoritative-URL search engine on a GENERAL/informational query "
    "(no single correct gold URL). You get the query and its top-k results "
    "(rank, url, title, trust_tier 0..1). For EACH result decide if it is FLOOD: a "
    "HIGH-TRUST page (trust_tier>=0.85, i.e. gov/official/encyclopedic/registry) whose "
    "TOPIC is OFF — not plausibly about this query (e.g. a Canada.ca or opensource.org "
    "or a random statute page returned for 'march madness teams'). A high-trust page "
    "that IS on-topic is NOT flood. A low-trust on/off-topic page is NOT flood.\n"
    "Also decide rel_at_1: is the rank-1 result plausibly ON-TOPIC / useful for the "
    "query (yes/no).\n"
    "Reply ONLY JSON: {\"flood_ranks\":[<ranks that are flood>],\"rel_at_1\":true|false}."
)
API_URL = "https://api.anthropic.com/v1/messages"
_cost = {"in": 0, "out": 0}
_lock = threading.Lock()


def _call(model, user, key, retries=4):
    body = json.dumps({"model": model, "max_tokens": 300, "system": JUDGE_SYS,
                       "messages": [{"role": "user", "content": user}]}).encode()
    last = None
    for a in range(retries):
        try:
            spec = HttpSpec(timeout=90.0, headers={
                "x-api-key": key, "anthropic-version": "2023-06-01",
                "content-type": "application/json"})
            with make_client(spec) as client:
                r = client.post(API_URL, content=body)
                r.raise_for_status()                 # urllib raised on 4xx/5xx; httpx does not
                d = r.json()
            with _lock:
                _cost["in"] += d["usage"]["input_tokens"]
                _cost["out"] += d["usage"]["output_tokens"]
            return "".join(b["text"] for b in d["content"] if b["type"] == "text")
        except ConsentBlocked:
            raise                                    # a declined host is never retried
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (a + 1))
    raise RuntimeError(f"api failed: {last}")


def judge_one(row, model, key):
    res = row["results"]
    if not res:
        return {"id": row["id"], "flood": 0.0, "rel_at_1": False, "empty": True,
                "flood_prone": row.get("flood_prone")}
    lines = "\n".join(
        f'{r["rank"]}. trust={r.get("trust_tier")} {host(r.get("url"))} | {(r.get("title") or "")[:90]}'
        for r in res)
    user = f"QUERY: {row['query']}\nRESULTS:\n{lines}"
    txt = _call(model, user, key)
    s = txt.find("{")
    try:
        o, _ = json.JSONDecoder().raw_decode(txt[s:]) if s >= 0 else ({}, 0)
    except Exception:  # noqa: BLE001
        o = {}
    fr = [int(x) for x in o.get("flood_ranks", []) if isinstance(x, (int, float))]
    k = len(res)
    return {"id": row["id"], "query": row["query"], "flood_prone": row.get("flood_prone"),
            "flood": len(set(fr)) / k, "flood_ranks": sorted(set(fr)), "k": k,
            "rel_at_1": bool(o.get("rel_at_1", False)),
            "p_general_205": row.get("p_general_205"), "p_general_206": row.get("p_general_206")}


def cmd_judge(args):
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        sys.exit("ANTHROPIC_API_KEY not set")
    rows = [json.loads(l) for l in open(args.results)]
    scored = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(judge_one, r, args.model, key) for r in rows]
        for fu in as_completed(futs):
            scored.append(fu.result())
    n = len(scored)
    flood = sum(s["flood"] for s in scored) / n
    rel1 = sum(1 for s in scored if s["rel_at_1"]) / n
    fp = [s for s in scored if s.get("flood_prone")]
    flood_fp = sum(s["flood"] for s in fp) / len(fp) if fp else 0.0
    gated206 = sum(1 for s in scored if (s.get("p_general_206") or 0) >= 0.5) / n
    gated205 = sum(1 for s in scored if (s.get("p_general_205") or 0) >= 0.5) / n
    gated206_fp = sum(1 for s in fp if (s.get("p_general_206") or 0) >= 0.5) / len(fp) if fp else 0.0
    gated205_fp = sum(1 for s in fp if (s.get("p_general_205") or 0) >= 0.5) / len(fp) if fp else 0.0
    summ = {"n": n, "n_flood_prone": len(fp),
            "FLOOD_mean": round(flood, 4), "FLOOD_mean_flood_prone": round(flood_fp, 4),
            "rel_at_1_mean": round(rel1, 4),
            "routing_gated_all_205": round(gated205, 4), "routing_gated_all_206": round(gated206, 4),
            "routing_gated_floodprone_205": round(gated205_fp, 4),
            "routing_gated_floodprone_206": round(gated206_fp, 4),
            "model": args.model}
    json.dump({"summary": summ, "scored": scored}, open(args.out, "w"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summ, indent=1))


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieve")
    r.add_argument("--config", required=True, choices=["off", "v2_205", "v2_206"])
    r.add_argument("--out", required=True)
    r.add_argument("--k", type=int, default=10)
    r.add_argument("--workdir", required=True, help="the task-206 work dir (artifacts/, optional data/, server/)")
    r.add_argument("--head205-dir", required=True, help="directory holding the task-205 head_clf.json")
    j = sub.add_parser("judge")
    j.add_argument("--results", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--model", default="claude-sonnet-4-5")
    j.add_argument("--workers", type=int, default=10)
    args = ap.parse_args(argv)
    (cmd_retrieve if args.cmd == "retrieve" else cmd_judge)(args)


if __name__ == "__main__":
    main()
