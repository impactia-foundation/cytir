"""Score ONE fusion-weights file with the SERVED ranker, in a fresh process.

    python -m cytir.retune.arm --weights W.json --out arm.json [--heldout-cache C.json] [--no-chooser]

The weights file is the one variable (task 533's method). The process env is prod's own serving env
(`cytir.ops.prod_env.resolve_env`, compose + overlays) with four values pointed at the twin: collection,
qdrant, TEI, and the weights file. Queries go through `GET /search` of the server's own FastAPI app
(`cytir.serve.app.api`, in process): the handler, `retrieval.search()`, the trust head and every other
serving term are the production code. This module holds no scoring.

Why a process per weights file: `retrieval` reads parts of the weights at import time.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from cytir.core import envnames as _envnames
import pathlib
import subprocess
import sys
import tempfile

import yaml

from cytir.retune import guard, judge
from cytir.retune.score import aggregate, hit_rank

REPO = pathlib.Path(__file__).resolve().parents[2]
ARM_SCHEMA = "aw.retune-arm.v1"
K = 10
#: Timing only; the twin is shared with the drain. Not a ranking knob.
TWIN_EXTRA = {"AW_QDRANT_TIMEOUT": "60"}
#: The hash salt of every arm process (see `build_env`). Not a ranking knob; the same in every arm.
HASH_SEED = "0"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_env(weights: str, collection: str, qdrant_url: str, tei_url: str,
              base: dict | None = None) -> dict:
    """The env of one arm: no inherited AW_* (a caller's stray knob must not reach the ranker),
    prod's serving env with its overlays, then the four twin values."""
    from cytir.ops.prod_env import resolve_env
    env = {k: v for k, v in (os.environ if base is None else base).items() if not _envnames.is_knob(k)}
    env.update(resolve_env(overlays=True).env)
    env.update({"AW_COLLECTION": collection, "AW_QDRANT_URL": qdrant_url, "AW_TEI_URL": tei_url,
                "AW_FUSION_WEIGHTS": str(pathlib.Path(weights).resolve()),
                "AW_WEBAPP": str(REPO / "webapp" / "index.html"), **TWIN_EXTRA})
    # Pinned: `search()` breaks exact score ties by set order, and Python salts string hashes per
    # process, so two arms with the same weights would otherwise differ on ~9 % of requests
    # (task 752 finding D6). One seed for every arm makes the weights file the only variable.
    env["PYTHONHASHSEED"] = HASH_SEED
    return env


def run_chooser(rows: list[dict], search_get, k: int = K, threads: int = 1) -> dict:
    """Score `rows` (the v2 chooser) through `search_get(q, k) -> (status, json)`.
    A non-200 raises: an error is not a miss (task 711)."""
    def one(row):
        status, body = search_get(row["query"], k)
        if status != 200:
            raise RuntimeError(f"/search answered {status} for {row['id']}: {str(body)[:200]}")
        urls = [r.get("url") or "" for r in body.get("results", [])]
        return row["id"], {"urls": urls, "rank": hit_rank(row["target_url"], urls), "type": row.get("type")}
    if threads > 1:
        with concurrent.futures.ThreadPoolExecutor(threads) as ex:
            done = list(ex.map(one, rows))
    else:
        done = [one(r) for r in rows]
    per_q = dict(done)
    return {**aggregate(per_q), "per_q": per_q}


def _testclient_get():
    from fastapi.testclient import TestClient
    from cytir.serve import app as srv
    client = TestClient(srv.api)

    def get(q, k):
        r = client.get("/search", params={"q": q, "k": k})
        return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)
    return get


def worker(args) -> dict:
    guard.refuse_non_twin(os.environ.get("AW_QDRANT_URL", ""), os.environ.get("AW_COLLECTION", ""))
    sys.path[:0] = [str(REPO), str(REPO / "server")]
    from cytir.serve import retrieval
    want = sha256_file(args.weights)
    got = retrieval.weights_fingerprint().get("sha256")
    if got != want:
        raise SystemExit(f"REFUSED: the server loaded weights sha256 {got}, the arm was asked for {want}")
    doc = {"schema": ARM_SCHEMA, "weights_path": args.weights, "weights_sha256": want,
           "weights": json.load(open(args.weights)), "code": args.code_sha, "collection": os.environ.get("AW_COLLECTION"),
           "sets": {}}
    if args.prod_health:
        from cytir.retune import preflight
        doc["preflight"] = preflight.check(os.environ["AW_QDRANT_URL"], os.environ["AW_COLLECTION"],
                                           os.environ["AW_TEI_URL"],
                                           prod_health=json.load(open(args.prod_health)),
                                           incumbent_sha256=want, tei_snapshots=args.tei_snapshots)
    doc["served_config"] = retrieval.effective_config()
    if not args.no_chooser:
        rows = yaml.safe_load(open(args.chooser))["eval"]
        doc["chooser_sha256"] = sha256_file(args.chooser)
        doc["sets"]["chooser"] = run_chooser(rows, _testclient_get(), K, args.threads)
    if args.heldout_cache:
        from cytir.gates import authority as gate
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "leg.json")
            gate.run(out, args.heldout_cache)       # installs the pinned-vector embedder: LAST
            leg = json.load(open(out))
        doc["heldout_cache_sha256"] = sha256_file(args.heldout_cache)
        doc["gate_set_version"] = leg.get("gate_set_version")
        doc["cache_miss"] = leg.get("cache_miss")
        doc["sets"].update(leg["sets"])
    return doc


def spawn(weights: str, out: str, *, collection: str, qdrant_url: str, tei_url: str, chooser: str,
          heldout_cache: str | None = None, no_chooser: bool = False, prod_health: str | None = None,
          threads: int = 1, niceness: bool = True, tei_snapshots: str | None = None,
          code_sha: str | None = None) -> subprocess.Popen:
    """Start one arm as a child process under prod's env, at idle priority."""
    cmd = [sys.executable, "-m", "cytir.retune.arm", "--weights", weights, "--out", out,
           "--chooser", chooser, "--threads", str(threads)]
    if heldout_cache:
        cmd += ["--heldout-cache", heldout_cache]
    if no_chooser:
        cmd += ["--no-chooser"]
    if prod_health:
        cmd += ["--prod-health", prod_health]
    if code_sha:
        cmd += ["--code-sha", code_sha]
    if tei_snapshots:
        cmd += ["--tei-snapshots", tei_snapshots]
    if niceness:
        cmd = ["nice", "-n", "19", "ionice", "-c3"] + cmd
    env = build_env(weights, collection, qdrant_url, tei_url)
    env["PYTHONPATH"] = os.pathsep.join(x for x in (str(REPO), env.get("PYTHONPATH", "")) if x)
    return subprocess.Popen(cmd, env=env, cwd=str(REPO))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chooser", default=str(REPO / "data" / "eval_tasks.v2.yaml"))
    ap.add_argument("--heldout-cache")
    ap.add_argument("--no-chooser", action="store_true")
    ap.add_argument("--prod-health", help="JSON of prod's /health: run the serving-contract preflight")
    ap.add_argument("--tei-snapshots", help="HF cache snapshots dir of the TEI model (evidence for a TEI "
                                            "launched without --revision)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--code-sha", help="the code the arm runs (recorded; a reused arm must match)")
    args = ap.parse_args(argv)
    guard.refuse_serving_path(args.out, "--out")
    try:
        doc = worker(args)
    except SystemExit as e:
        if str(e.code).startswith("PREFLIGHT FAILED"):
            print(e.code, file=sys.stderr, flush=True)
            return 5
        raise
    judge.assert_no_errors(doc, args.out)
    tmp = args.out + ".part"
    with open(tmp, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, args.out)
    for name, s in doc["sets"].items():
        print(f"  {name:18s} n={s['n']:>4} h@1={s['h1']:>4} h@10={s['h10']:>4} mrr={s['mrr']}", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
