"""`cyt gate`: score a pinned gold slice against a RUNNING `cyt serve`, over HTTP (refactor slice 8,
step 5; specs/aw-index-serve-gate.md sections 2 and 3).

    cyt gate --api URL --slice data/gate-slices/ci-fixture.v1.json
            [--k 10] [--verdict-out FILE] [--baseline FILE] [--serve-receipt FILE]
            [--allow-identity-mismatch]

ONE SCORER. The golds are scored by the per-gold scorer of `cytir.gates.authority` (`gold_of`,
`rank_of`, `norm_url`, and the ADR-043 per-gold rule `rank_regressed`): not a copy of it. What this
module adds is the transport (`GET /search` of the server that is running) and the verdict file.
`rank` is the scorer's rank (it matches a gold at page level, ignoring the `#fragment`); `rank_exact`
is the first result whose normalised URL, fragment included, equals a gold URL (a fedlex article is
a fragment of one page: the page-level rank alone cannot tell two articles apart).

THE VERDICT (`aw.gate-verdict.v1`): the slice's identity (sha256 of its bytes), the server's
identity (the sha256 of its `serve-receipt.json` when `--serve-receipt` names it, plus its profile
digest and IDF table; always the `/health` config digest), and per gold the rank, `rank_exact`,
`hit_at_k` and the top-k URLs. Per class: h@1, h@k, MRR.

THE PASS RULE (`passes_against_baseline`, the ONE function that says what "pass" means): with
`--baseline FILE` (a verdict of an earlier run on the SAME slice), the run passes when no gold ranks
worse than in the baseline, by `rank` or by `rank_exact`; a gold found there and missing now is
worse. Without a baseline nothing is compared: the verdict is written and the exit is 0 with
`pass: null` (a first run records the baseline). Design question Q3 of the spec is open with the
founder; to change the rule, change that one function.

IDENTITY. A baseline made on another serving profile digest or another IDF table is not a like-for-
like comparison: the gate REFUSES (exit 3, both values named) unless `--allow-identity-mismatch` is
given, which records each mismatch in the verdict (`baseline.identity_mismatches`).

EMPTY IS NOT A MISS. A server with 0 points, or a run where EVERY query returns no result, is a
refusal: nothing was measured, and no verdict (no baseline either) is written. One empty answer
among non-empty ones stays a miss.

`proves`: what a green run of THIS verdict can claim, set from the embedder identity the gate can
see (the query embedder's `/info` that `cyt serve` records in its receipt: a stand-in announces
itself). `plumbing-only` for a stand-in, `ranking` only when the identity is the pinned model,
`unknown` (and it says why) when the gate cannot know.

EXIT: 0 pass (or no baseline), 1 fail (a gold regressed), 2 usage, 3 refused: an unreadable slice,
a baseline of another slice or identity, a server that is unreachable, not healthy or empty, ANY
query that errors, or ANY unexpected exception. An error is never a miss, and exit 1 means ONLY
that a gold ranks worse.
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import argparse
import datetime
import hashlib
import http.client
import json
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request

from cytir.gates import authority as AUTH

SCHEMA = "aw.gate-verdict.v1"
SLICE_SCHEMA = "aw.gate-slice.v1"
RULE = "no-gold-worse-than-baseline"
EXIT_PASS, EXIT_FAIL, EXIT_USAGE, EXIT_REFUSED = 0, 1, 2, 3
PROVES_PLUMBING = "plumbing-only (stand-in vectors; not a ranking claim)"
PROVES_RANKING = "ranking"


class GateRefused(RuntimeError):
    """The gate cannot score (exit 3). Never an empty result."""


def sha256_file(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- the slice
def load_slice(path) -> tuple[dict, str]:
    try:
        raw = pathlib.Path(path).read_bytes()
        doc = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise GateRefused(f"slice {path}: {type(exc).__name__}: {exc}") from exc
    if doc.get("schema") != SLICE_SCHEMA or not doc.get("golds"):
        raise GateRefused(f"slice {path}: not an {SLICE_SCHEMA} file with golds")
    ids = [g.get("id") for g in doc["golds"]]
    if None in ids or len(set(ids)) != len(ids):
        raise GateRefused(f"slice {path}: every gold needs a unique id")
    for g in doc["golds"]:
        if not g.get("query") or not AUTH.gold_of(g)[0]:
            raise GateRefused(f"slice {path}: gold {g.get('id')!r} needs a query and a target_url")
    return doc, hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------- the transport
def _get_json(url: str, timeout: float):
    try:
        with IH.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        raise GateRefused(f"{url}: HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as exc:
        raise GateRefused(f"{url}: {type(exc).__name__}: {exc}") from exc


def server_identity(api: str, timeout: float) -> dict:
    """What the running server says it is (`/health`): refuses a server that is not `ok`."""
    health = _get_json(api.rstrip("/") + "/health", timeout)
    if not isinstance(health, dict) or health.get("status") != "ok":
        raise GateRefused(f"{api}/health: status "
                          f"{health.get('status') if isinstance(health, dict) else health!r}, not ok")
    points = health.get("points")
    if not isinstance(points, int) or isinstance(points, bool) or points <= 0:
        raise GateRefused(f"{api}/health: the collection holds {points!r} points: an empty index "
                          "measures nothing (is `cyt serve` on the collection `cyt index build` made?)")
    cfg = health.get("config") or {}
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return {"api": api, "collection": cfg.get("collection"), "points": health.get("points"),
            "config_sha256": hashlib.sha256(blob.encode()).hexdigest()}


def receipt_identity(path) -> dict:
    """The server's `serve-receipt.json`: its sha256 and the two identities the verdict pins."""
    try:
        raw = pathlib.Path(path).read_bytes()
        doc = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise GateRefused(f"--serve-receipt {path}: {type(exc).__name__}: {exc}") from exc
    idf = doc.get("idf_table") or {}
    return {"sha256": hashlib.sha256(raw).hexdigest(), "schema": doc.get("schema"),
            "collection": doc.get("collection"), "points_count": doc.get("points_count"),
            "profile": doc.get("profile"), "query_embedding": doc.get("query_embedding"),
            "idf_table": {k: idf.get(k) for k in ("sha256", "collection", "points_count",
                                                  "matches_target")}}


def search(api: str, gold: dict, k: int, timeout: float) -> list[str]:
    """The URLs the server returns for one gold's query, in order. Any failure raises."""
    params = {"q": gold["query"], "k": k}
    if gold.get("lang"):
        params["lang"] = gold["lang"]
    if gold.get("country"):
        params["country"] = gold["country"]
    doc = _get_json(f"{api.rstrip('/')}/search?{urllib.parse.urlencode(params)}", timeout)
    if (not isinstance(doc, dict) or not isinstance(doc.get("results"), list) or doc.get("error")
            or doc.get("status") == "degraded"):
        raise GateRefused(f"query {gold['id']!r}: the server answered without results: "
                          f"{str(doc)[:200]}")
    urls = [r.get("url") if isinstance(r, dict) else None for r in doc["results"]]
    if not all(isinstance(u, str) and u for u in urls):
        raise GateRefused(f"query {gold['id']!r}: a result row without a url: {str(doc)[:200]}")
    return urls


# --------------------------------------------------------------------------- scoring
def score_gold(gold: dict, urls: list[str], k: int) -> dict:
    """One gold's row, by the authority scorer (`gold_of` + `rank_of`) plus the exact rank."""
    gold_urls, gold_domains = AUTH.gold_of(gold)
    rank = AUTH.rank_of(urls, gold_urls, gold_domains)
    exact = {AUTH.norm_url(g) for g in gold_urls}
    rank_exact = next((i for i, u in enumerate(urls, 1) if AUTH.norm_url(u) in exact), None)
    return {"id": gold["id"], "class": gold.get("class"), "family": gold.get("family"),
            "query": gold["query"], "target_url": gold_urls[0], "rank": rank,
            "rank_exact": rank_exact, "hit_at_k": rank is not None and rank <= k,
            "top": urls[:k]}


def golds_outcome_digest(rows: list[dict]) -> str:
    """sha256 over the ranking OUTCOME of a verdict: the sorted `(id, hit_at_k, rank, rank_exact)`
    of every gold, nothing else (not `top`, scores, timestamps). A committed baseline carries it,
    and a sealed contract pins it, so a re-record cannot move a gold quietly."""
    outcome = sorted([r["id"], r["hit_at_k"], r["rank"], r["rank_exact"]] for r in rows)
    return hashlib.sha256(json.dumps(outcome, separators=(",", ":")).encode()).hexdigest()


def class_metrics(rows: list[dict], k: int) -> dict:
    out: dict = {}
    for name in sorted({r["class"] for r in rows}, key=str):
        cls = [r for r in rows if r["class"] == name]
        ranks = [r["rank"] for r in cls]
        out[str(name)] = {
            "n": len(cls), "h@1": sum(1 for x in ranks if x == 1),
            f"h@{k}": sum(1 for x in ranks if x is not None and x <= k),
            "mrr": round(sum(1.0 / x for x in ranks if x) / len(cls), 4)}
    return out


def embedder_proves(receipt: dict | None) -> str:
    """What a run can claim, from the query embedder's identity as `cyt serve` recorded it (the
    TEI `/info` in the receipt). A stand-in announces itself (`stand_in`); `ranking` needs the
    pinned model id AND revision; anything else is `unknown`, with the reason."""
    if not receipt:
        return "unknown (no --serve-receipt: the gate cannot see the query embedder)"
    qe = receipt.get("query_embedding") or {}
    info = qe.get("info")
    if not isinstance(info, dict):
        return (f"unknown (the server's query embedder mode is {qe.get('mode')!r} and it did not "
                "report an identity)")
    if info.get("stand_in"):
        return PROVES_PLUMBING
    from cytir.core import model_pin
    try:
        pinned = (model_pin.EMBED_MODEL_ID, model_pin.embed_model_revision())
    except ValueError:
        pinned = None
    if pinned and (info.get("model_id"), info.get("model_sha")) == pinned:
        return PROVES_RANKING
    return (f"unknown (the query embedder reports {info.get('model_id')!r} / "
            f"{info.get('model_sha')!r}, not the pinned model)")


# --------------------------------------------------------------------------- THE RULE
def passes_against_baseline(rows: list[dict], baseline_rows: list[dict]) -> tuple[bool, dict]:
    """THE pass rule, in one place (design question Q3, open with the founder): the run passes when
    no gold ranks worse than in the baseline verdict of the same slice, by `rank` or by
    `rank_exact` (ADR-043's per-gold rule, `authority.rank_regressed`). Returns `(passed, detail)`;
    `detail` lists the regressions, the improvements and the golds the baseline does not know."""
    base = {r["id"]: r for r in baseline_rows}
    regressions, improved, unknown = [], [], []
    for r in rows:
        b = base.get(r["id"])
        if b is None:
            unknown.append(r["id"])
            continue
        for measure in ("rank", "rank_exact"):
            if AUTH.rank_regressed(b.get(measure), r[measure]):
                regressions.append({"id": r["id"], "measure": measure,
                                    "baseline": b.get(measure), "now": r[measure]})
            elif r[measure] is not None and (b.get(measure) is None or r[measure] < b[measure]):
                improved.append({"id": r["id"], "measure": measure,
                                 "baseline": b.get(measure), "now": r[measure]})
    return (not regressions), {"regressions": regressions, "improved": improved,
                               "not_in_baseline": unknown}


# --------------------------------------------------------------------------- the run
def run(*, api: str, slice_path, k: int = 10, baseline_path=None, serve_receipt=None,
        labels: dict | None = None, timeout: float = 60.0, allow_identity_mismatch: bool = False,
        log=print) -> tuple[int, dict]:
    doc, slice_sha = load_slice(slice_path)
    baseline = None
    if baseline_path:
        try:
            baseline = json.loads(pathlib.Path(baseline_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GateRefused(f"--baseline {baseline_path}: {type(exc).__name__}: {exc}") from exc
        if not isinstance(baseline, dict) or baseline.get("schema") != SCHEMA:
            raise GateRefused(f"--baseline {baseline_path}: not an {SCHEMA} verdict")
        if (baseline.get("slice") or {}).get("sha256") != slice_sha:
            raise GateRefused(
                f"--baseline {baseline_path} was scored on slice sha256 "
                f"{(baseline.get('slice') or {}).get('sha256')}, this slice is {slice_sha}: a "
                "changed slice needs a new baseline, made on purpose")
    server = server_identity(api, timeout)
    rows, answered = [], []
    for gold in doc["golds"]:
        urls = search(api, gold, k, timeout)
        answered.append(urls)
        rows.append(score_gold(gold, urls, k))
    if not any(answered):
        raise GateRefused(f"all {len(rows)} queries returned no result: an empty answer to every "
                          "query measures nothing, it is not a set of misses")
    receipt = receipt_identity(serve_receipt) if serve_receipt else None
    verdict: dict = {
        "schema": SCHEMA,
        "created_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "slice": {"name": doc.get("name"), "sha256": slice_sha, "n_golds": len(rows)},
        "k": k, "labels": dict(labels or {}), "server": server,
        "serve_receipt": receipt, "proves": embedder_proves(receipt),
        "classes": class_metrics(rows, k), "golds": rows,
        "golds_outcome_digest": golds_outcome_digest(rows), "rule": RULE,
        "baseline": None, "pass": None}
    code = EXIT_PASS
    if baseline is not None:
        same, mismatches = {}, []
        for field, mine, theirs in (
                ("serve_receipt.profile.digest",
                 ((receipt or {}).get("profile") or {}).get("digest"),
                 (((baseline.get("serve_receipt") or {}).get("profile")) or {}).get("digest")),
                ("serve_receipt.idf_table.sha256",
                 ((receipt or {}).get("idf_table") or {}).get("sha256"),
                 (((baseline.get("serve_receipt") or {}).get("idf_table")) or {}).get("sha256"))):
            same[field] = (mine == theirs)
            if mine != theirs:
                mismatches.append({"field": field, "baseline": theirs, "now": mine})
        if mismatches and not allow_identity_mismatch:
            raise GateRefused(
                "the baseline was made on another identity, so this is not a like-for-like "
                "comparison: " + "; ".join(f"{m['field']}: baseline {m['baseline']!r}, now "
                                           f"{m['now']!r}" for m in mismatches)
                + " (record a new baseline on purpose, or pass --allow-identity-mismatch to "
                  "compare anyway and record the mismatch)")
        ok, detail = passes_against_baseline(rows, baseline["golds"])
        verdict["baseline"] = {"path": pathlib.Path(baseline_path).name,
                               "sha256": sha256_file(baseline_path), **detail,
                               "identity_equal": same, "identity_mismatches": mismatches,
                               "identity_mismatch_allowed": bool(mismatches)}
        verdict["pass"] = ok
        code = EXIT_PASS if ok else EXIT_FAIL
    for r in rows:
        log(f"  {r['id']:16s} rank={r['rank']!s:>4} exact={r['rank_exact']!s:>4}  {r['target_url']}")
    return code, verdict


def build_parser(prog: str = "cyt gate") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    ap.add_argument("--api", required=True, metavar="URL", help="a running `cyt serve`")
    ap.add_argument("--slice", required=True, metavar="FILE", help="a pinned gold slice")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--verdict-out", default=None, metavar="FILE")
    ap.add_argument("--baseline", default=None, metavar="FILE",
                    help="a verdict of an earlier run on the same slice: no gold may rank worse")
    ap.add_argument("--serve-receipt", default=None, metavar="FILE",
                    help="the server's serve-receipt.json (its sha256 goes into the verdict)")
    ap.add_argument("--allow-identity-mismatch", action="store_true",
                    help="compare against a baseline made on another profile digest or IDF table "
                         "(refused by default); the verdict records each mismatch")
    ap.add_argument("--label", action="append", default=[], metavar="K=V",
                    help="a fact about the run the server cannot tell (e.g. vectors=stand-in), "
                         "recorded in the verdict")
    ap.add_argument("--timeout", type=float, default=60.0)
    return ap


def main(argv=None, *, prog: str = "cyt gate") -> int:
    args = build_parser(prog).parse_args(argv)
    try:
        labels = {}
        for item in args.label:
            key, eq, val = item.partition("=")
            if not eq or not key:
                raise GateRefused(f"--label {item!r}: expected K=V")
            labels[key] = val
        code, verdict = run(api=args.api, slice_path=args.slice, k=args.k,
                            baseline_path=args.baseline, serve_receipt=args.serve_receipt,
                            labels=labels, timeout=args.timeout,
                            allow_identity_mismatch=args.allow_identity_mismatch)
        if args.verdict_out:
            pathlib.Path(args.verdict_out).write_text(
                json.dumps(verdict, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    except GateRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as exc:  # noqa: BLE001  exit 1 means ONLY "a gold regressed": a crash is a refusal
        print(f"REFUSED: unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    cmp = verdict["baseline"]
    print(json.dumps({"slice": verdict["slice"], "proves": verdict["proves"],
                      "classes": verdict["classes"], "pass": verdict["pass"],
                      "regressions": (cmp or {}).get("regressions"),
                      "verdict": args.verdict_out}, indent=1))
    info = ((verdict["serve_receipt"] or {}).get("query_embedding") or {}).get("info") or {}
    if info.get("device"):
        print(f"embedder: {info.get('model_id')}@{info.get('model_sha')} {info['device']}/"
              f"{info.get('dtype')} sentence-transformers {info.get('sentence_transformers')} "
              f"torch {info.get('torch')}")
    if verdict["proves"] != PROVES_RANKING:
        print(f"proves: {verdict['proves']}")
    if cmp is None:
        print("no --baseline: nothing compared, exit 0 (record this verdict as the baseline)")
    elif code == EXIT_FAIL:
        print("FAIL: a gold ranks worse than in the baseline", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
