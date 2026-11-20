"""The retune command: measure on the twin, EMIT a candidate and a report, deploy nothing.

    python -m cytir.retune.emit --collection <twin-collection> --qdrant-url <twin-url> \\
        --heldout-cache <the pinned 1,418-vector cache> [--prod-health-url https://<prod host>]

Reads: the twin (read-only: qdrant search + TEI embeds), prod's `/health` once (optional). Writes:
`data/retune/candidates/<date>-<sha>.json|.weights.json|.report.md` and arm documents in a work dir.
It has no deploy step. Promotion of weights is a PR that Tim merges (ADR-099 amendment 3):
`docs` in the report name the file prod loads and what the PR changes.

Refusals (exit 2, nothing measured): a deploy-shaped flag, an output directory that is a serving
path, a non-loopback qdrant (prod is never the target), an instrument whose digest is not the
flip-gate contract's pin. Exit 5: the serving-contract preflight failed.
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import time

from cytir.core.paths import data_path
from cytir.retune import arm as armmod
from cytir.retune import grid, guard, judge

REPO = armmod.REPO
CONTRACT = data_path("flip-gate-contract.v1.json")   # shipped package data: resolved, not a checkout path
DEFAULT_OUT = REPO / "data" / "retune" / "candidates"
EXIT_REFUSED = guard.EXIT_REFUSED
EXIT_PREFLIGHT = 5
EXIT_ARM = 4

#: Any argv token that starts with one of these names an action this command does not have.
DEPLOY_SHAPED = ("--deploy", "--promote", "--apply", "--install", "--restart", "--stage", "--flip",
                 "--rollback", "--write-live", "--overwrite", "--push", "--live-weights",
                 "--promote-target", "--weights-out", "--out-weights", "--incumbent-weights")


Refused = guard.Refused


def refuse_deploy_shaped(argv: list[str], environ: dict | None = None) -> None:
    environ = os.environ if environ is None else environ
    for tok in argv:
        flag = tok.split("=", 1)[0]
        if any(flag == d or flag.startswith(d) for d in DEPLOY_SHAPED):
            raise Refused(f"{flag!r} is deploy-shaped. The retune emits a candidate and a report; "
                          f"promotion is a PR that Tim merges (ADR-099 amendment 3).")
    if environ.get("AW_RETUNE_MODE", "emit") not in ("emit", ""):
        raise Refused(f"AW_RETUNE_MODE={environ['AW_RETUNE_MODE']!r}: the prod/candidate modes of the "
                      f"old retune_fusion.sh are gone; there is only emit")


def contract_pins() -> dict:
    inst = json.loads(CONTRACT.read_text())["instruments"]
    return {"chooser": inst["629"]["sha256"], "heldout_cache": inst["1418"]["query_cache_sha256"],
            "gate_set_version": inst["1418"]["gate_set_version"]}


def check_instruments(chooser: str, cache: str) -> dict:
    pins = contract_pins()
    for what, path in (("chooser", chooser), ("heldout_cache", cache)):
        if not os.path.isfile(path):
            raise Refused(f"{what} file {path!r} does not exist")
    got = {"chooser": armmod.sha256_file(chooser), "heldout_cache": armmod.sha256_file(cache)}
    for k in ("chooser", "heldout_cache"):
        if got[k] != pins[k]:
            raise Refused(f"{k} digest {got[k][:12]} is not the flip-gate contract's pin {pins[k][:12]} "
                          f"(a moved yardstick is not an A/B)")
    return {**got, "gate_set_version": pins["gate_set_version"]}


def incumbent_path() -> str:
    """The file prod loads, from the ONE place that says (cytir.ops.prod_env: compose + overlays)."""
    from cytir.ops.prod_env import resolve_env
    return resolve_env(overlays=True).env["AW_FUSION_WEIGHTS"]


def write_weights(path: pathlib.Path, weights: dict) -> str:
    data = (json.dumps(weights, indent=1) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def verify_reused(doc: dict, job: dict, code: str, collection: str) -> None:
    """A reused arm must be the same measurement: same weights bytes, same collection, same code."""
    want = armmod.sha256_file(job["weights"])
    bad = []
    if doc.get("weights_sha256") != want:
        bad.append(f"weights sha256 {str(doc.get('weights_sha256'))[:12]} != {want[:12]}")
    if doc.get("collection") != collection:
        bad.append(f"collection {doc.get('collection')!r} != {collection!r}")
    if not doc.get("code") or doc.get("code") != code:
        bad.append(f"code {doc.get('code')!r} != {code!r} (an arm from another checkout, or none recorded)")
    if bad:
        raise SystemExit(f"REFUSED to reuse arm {job['name']}: " + "; ".join(bad))


def _run_arms(jobs: list[dict], workers: int, *, stamps: dict, points_fn, code: str, resume: bool,
              **common) -> None:
    """jobs: [{name, weights, out, ...spawn kwargs}]. At most `workers` children at once; any
    non-zero child aborts the run (an arm that failed is not a zero). The twin's point count is
    stamped before and after every arm. An existing arm file is refused unless `resume`, and then
    verified. Every finished arm is refused if it holds an `__ERROR__` row (M2)."""
    pending, running = list(jobs), []
    while pending or running:
        while pending and len(running) < workers:
            j = pending.pop(0)
            if os.path.exists(j["out"]):
                if not resume:
                    raise SystemExit(f"REFUSED: {j['out']} exists; use --resume to verify and reuse it, or a fresh --work-dir")
                doc = json.load(open(j["out"]))
                verify_reused(doc, j, code, common["collection"])
                judge.assert_no_errors(doc, j["name"])
                print(f"[arm] reuse {j['name']} (verified: weights, collection, code)", flush=True)
                stamps[j["name"]] = [None, None]
                continue
            print(f"[arm] start {j['name']}", flush=True)
            kw = {k: v for k, v in j.items() if k not in ("name", "weights", "out")}
            stamps[j["name"]] = [points_fn(), None]
            running.append((j, armmod.spawn(j["weights"], j["out"], code_sha=code, **common, **kw)))
        time.sleep(2)
        for j, p in list(running):
            rc = p.poll()
            if rc is None:
                continue
            running.remove((j, p))
            if rc != 0:
                for _, q in running:
                    q.terminate()
                raise SystemExit(rc if rc == EXIT_PREFLIGHT else EXIT_ARM)
            stamps[j["name"]][1] = points_fn()
            judge.assert_no_errors(json.load(open(j["out"])), j["name"])
            print(f"[arm] done  {j['name']}", flush=True)


def _fetch_health(url: str, dest: pathlib.Path) -> dict:
    with IH.urlopen(url.rstrip("/") + "/health", timeout=30) as r:
        data = json.load(r)
    dest.write_text(json.dumps(data))
    return data


def render_report(c: dict) -> str:
    """One screen, decision first (pr-writing)."""
    v = c["bar"]["verdict"]
    cand = c.get("candidate")
    ch = c["chooser_summary"]
    lines = [f"# Fusion retune, {c['emitted_at'][:10]}: {v}", ""]
    if cand is None:
        lines += ["**Decision: nothing to promote.** No single-key change beat the live weights on the chooser."]
    elif v == "PASS":
        lines += [f"**Decision: you may promote `{cand['change']}`.** It passes the ADR-043 ranking bar on the "
                  f"chooser and the held-out battery. Promotion is a PR you merge (recipe below)."]
    else:
        lines += [f"**Decision: do not promote `{cand['change']}`. The live weights stay.** Verdict {v}: "
                  + "; ".join(c["bar"]["reasons"][:3])]
    lines += ["",
              f"- Index: `{c['index']['collection']}` on the twin, {c['index']['points']:,} points"
              + (f" (prod {c['index']['prod_points']:,})" if c['index'].get('prod_points') else "") + f". Prod's weights: "
              f"`{c['incumbent']['name']}` (sha256 `{c['incumbent']['sha256'][:12]}`).",
              f"- Null pairs (same weights again, run last): moved {c['null'].get('pairs', [c['null']['moved']])} gold(s). Band = {c['bar']['band']}.",
              f"- Twin point count seen across arms: {', '.join(f'{p:,}' for p in c['index'].get('points_seen', [])) or 'not stamped'}. "
              f"Server code `{(c.get('code') or 'unknown')[:12]}`; prod's deployed code is not verified.",
              f"- Chooser (629, v2), h@1 / h@10 / MRR vs live: {ch['d_h1']:+d} / {ch['d_h10']:+d} / {ch['d_mrr']:+.4f}.",
              f"- Held-out battery ({c['instruments']['gate_set_version']}): "
              + (f"{c['heldout_summary']['regressed']} gold(s) lost rank, {c['heldout_summary']['improved']} gained."
                 if c.get("heldout_summary") else "not run (no candidate)."),
              f"- Arms measured: {len(c['arms'])}. Wall time {c['wall_seconds'] // 60} min.",
              ""]
    lines += ["## Promotion recipe (only if you decide to promote)", "",
              "The file prod loads is `cytir/serve/fusion_weights.candidate533.json` (`/app/cytir/serve/...` in the "
              "container, a read-only bind mount of the box's `cytir/serve/`). Its pointer has ONE home: "
              "`FLIP_FUSION_WEIGHTS_FILE` and `FLIP_FUSION_WEIGHTS_SHA256` in `cytir/core/serving_config.py`. "
              "`deploy/docker-compose.flip.yaml` and `cytir/ops/serving_env.default.json` are generated from it. "
              "A promotion PR adds the candidate as a NEW versioned file in `cytir/serve/`, moves those two "
              "constants, and regenerates the two files. It never edits the served file in place (task 522). "
              "The box then needs the file and a container restart: a separate, announced deploy.", ""]
    lines += ["<details><summary>Per class</summary>", "",
              "| class | n gold | h@1 live→cand | h@10 live→cand | MRR Δ | reordered |", "|---|--:|--:|--:|--:|--:|"]
    for r in c["classes"]:
        lines.append(f"| {r['class']} | {r['n_gold']} | {r['a']['h1']}→{r['b']['h1']} | {r['a']['h10']}→{r['b']['h10']} "
                     f"| {r['d_mrr']:+.4f} | {r['reordered']} |")
    lines += ["", "**Arms (chooser, vs live)**", "", "| arm | net h@1+h@10 | regressed golds |", "|---|--:|--:|"]
    for name, a in c["arms"].items():
        lines.append(f"| {name} | {a['chooser_net']:+d} | {a['regressed']} |")
    if c["regressed"]:
        lines += ["", "**Golds that lost rank (candidate)**", ""]
        lines += [f"- `{r['class']}/{r['id']}`: {r['from']} → {r['to']}" for r in c["regressed"][:40]]
    lines += ["", "</details>", ""]
    return "\n".join(lines)


NULL_PAIRS = 2     # same weights again, run LAST so they span the time of the comparison


def run(args) -> int:
    t0 = time.time()
    chooser = args.chooser
    pins = check_instruments(chooser, args.heldout_cache)
    code = _git_sha()
    if not code:
        raise Refused("no code sha: set AW_RETUNE_CODE_SHA (the runner does) or run inside a git checkout")
    inc_path = incumbent_path()
    incumbent = json.load(open(inc_path))
    work = pathlib.Path(args.work_dir or tempfile.mkdtemp(prefix="aw-retune-"))
    work.mkdir(parents=True, exist_ok=True)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    health_file, prod_points = None, None
    if args.prod_health_url:
        health_file = work / "prod-health.json"
        prod_points = _fetch_health(args.prod_health_url, health_file).get("points")

    stamps: dict = {}
    common = dict(collection=args.collection, qdrant_url=args.qdrant_url, tei_url=args.tei_url,
                  chooser=chooser, threads=args.threads, tei_snapshots=args.tei_snapshots)
    ctl = dict(stamps=stamps, points_fn=lambda: _points(args), code=code, resume=args.resume)
    props = grid.proposals(incumbent, args.keys)
    if args.max_arms:
        props = props[: args.max_arms]
    wfiles = {}
    for name, w in props:
        p = work / f"w-{name}.json"
        write_weights(p, w)
        wfiles[name] = (p, w)
    # 1. incumbent (with the preflight) + one-key proposals, chooser only
    jobs = [{"name": "incumbent", "weights": inc_path, "out": str(work / "arm-incumbent.json"),
             "prod_health": str(health_file) if health_file else None}]
    jobs += [{"name": n, "weights": str(p), "out": str(work / f"arm-{n}.json")} for n, (p, _) in wfiles.items()]
    _run_arms(jobs, args.workers, **ctl, **common)
    docs = {j["name"]: json.load(open(j["out"])) for j in jobs}
    inc = docs.pop("incumbent")
    winner, scored = judge.pick_winner(docs, inc)

    # 2. the winner on the held-out battery, with its incumbent
    hi = hc = None
    if winner:
        wpath, wdict = wfiles[winner]
        hjobs = [{"name": "incumbent-heldout", "weights": inc_path, "out": str(work / "held-incumbent.json"),
                  "no_chooser": True, "heldout_cache": args.heldout_cache},
                 {"name": f"{winner}-heldout", "weights": str(wpath), "out": str(work / "held-candidate.json"),
                  "no_chooser": True, "heldout_cache": args.heldout_cache}]
        _run_arms(hjobs, args.workers, **ctl, **common)
        hi, hc = (json.load(open(j["out"])) for j in hjobs)
        for doc in (hi, hc):
            if doc.get("gate_set_version") != pins["gate_set_version"] or doc.get("cache_miss"):
                raise SystemExit(f"held-out instrument broken: gate_set_version={doc.get('gate_set_version')} "
                                 f"cache_miss={doc.get('cache_miss')}")

    # 3. the null pairs LAST: the incumbent weights again, same sets as the comparison
    njobs = [{"name": f"null-{i + 1}", "weights": inc_path, "out": str(work / f"arm-null-{i + 1}.json"),
              **({"heldout_cache": args.heldout_cache} if winner else {})} for i in range(NULL_PAIRS)]
    _run_arms(njobs, args.workers, **ctl, **common)
    nulls = [json.load(open(j["out"])) for j in njobs]

    full_inc = {"sets": {**inc["sets"], **(hi["sets"] if hi else {})}}
    null_cmps = [judge.compare(full_inc, n) for n in nulls]
    null_cmp = max(null_cmps, key=lambda c: c["moved"])
    started = [v[0] for v in stamps.values() if v[0] is not None]
    ended = [v[1] for v in stamps.values() if v[1] is not None]
    points_seen = sorted({p for p in started + ended if p is not None})

    result = {"schema": "aw.retune-candidate.v1",
              "emitted_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "instruments": pins, "code": code,
              "index": {"collection": args.collection, "qdrant_url": args.qdrant_url,
                        "points": (inc.get("preflight") or {}).get("points") or (points_seen[0] if points_seen else None),
                        "prod_points": prod_points, "points_seen": points_seen, "points_stamps": stamps,
                        "serving": "twin (read-only)"},
              "incumbent": {"name": pathlib.Path(inc_path).name, "path": inc_path,
                            "sha256": inc["weights_sha256"], "weights": incumbent,
                            "prod_parity_checked": bool(health_file)},
              "null": {"moved": null_cmp["moved"], "pairs": [c["moved"] for c in null_cmps]},
              "arms": {n: {"chooser_net": c["chooser_net"], "regressed": len(c["regressed"])} for n, c in scored.items()},
              "candidate": None, "regressed": []}
    if winner:
        full_cand = {"sets": {**docs[winner]["sets"], **hc["sets"]}}
        worst_null = nulls[null_cmps.index(null_cmp)]
        full_null = {"sets": {**worst_null["sets"]}}
        cmp = judge.compare(full_inc, full_cand)
        bar = judge.bar(cmp, null_cmp)
        if bar["verdict"] == "PASS" and len(points_seen) > 1:
            bar = {"verdict": "INCONCLUSIVE", "band": bar["band"],
                   "reasons": [f"the twin's point count moved during the run {points_seen}: arms did not see one index"]}
        stem = f"{result['emitted_at'][:10].replace('-', '')}-{hc['weights_sha256'][:12]}"
        wsha = write_weights(out_dir / f"{stem}.weights.json", wdict)
        changed = {k: [incumbent.get(k), wdict[k]] for k in wdict if wdict[k] != incumbent.get(k)}
        (out_dir / f"{stem}.perquery.csv").write_text(judge.perquery_csv(full_inc, full_null, full_cand))
        result.update({"bar": bar, "classes": cmp["rows"],
                       "regressed": cmp["regressed"], "improved": cmp["improved"],
                       "candidate": {"change": winner, "changed_keys": changed, "weights": wdict,
                                     "weights_file": f"{stem}.weights.json", "weights_sha256": wsha,
                                     "promotable": bar["verdict"] == "PASS"}})
        held = [r for r in cmp["rows"] if not r["class"].startswith("chooser")]
        result["heldout_summary"] = {"regressed": len([r for r in cmp["regressed"] if not r["class"].startswith("chooser")]),
                                     "improved": len([r for r in cmp["improved"] if not r["class"].startswith("chooser")]),
                                     "classes": len(held)}
        result["chooser_summary"] = _chooser_sum(cmp)
    else:
        stem = f"{result['emitted_at'][:10].replace('-', '')}-none-{inc['weights_sha256'][:12]}"
        result.update({"bar": {"verdict": "NO_CANDIDATE", "band": null_cmp["moved"],
                               "reasons": ["no single-key move gained on the chooser"]},
                       "classes": null_cmp["rows"], "chooser_summary": {"d_h1": 0, "d_h10": 0, "d_mrr": 0.0}})
    result["wall_seconds"] = int(time.time() - t0)
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=1) + "\n")
    (out_dir / f"{stem}.report.md").write_text(render_report(result))
    print(f"[emit] {result['bar']['verdict']}: wrote {out_dir / (stem + '.report.md')}", flush=True)
    return 0


def _chooser_sum(cmp: dict) -> dict:
    ch = [r for r in cmp["rows"] if r["class"].startswith("chooser:")]
    return {"d_h1": sum(r["d_h1"] for r in ch), "d_h10": sum(r["d_h10"] for r in ch),
            "d_mrr": cmp["chooser_d_mrr"]}


def _points(args) -> int | None:
    try:
        with IH.urlopen(f"{args.qdrant_url}/collections/{args.collection}", timeout=30) as r:
            return json.load(r)["result"]["points_count"]
    except Exception:  # noqa: BLE001
        return None


def _git_sha() -> str | None:
    """The code the arms ran. The runner ships `git archive` (no .git) and passes its HEAD."""
    import subprocess
    if os.environ.get("AW_RETUNE_CODE_SHA"):
        return os.environ["AW_RETUNE_CODE_SHA"]
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    refuse_deploy_shaped(argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--qdrant-url", required=True)
    ap.add_argument("--tei-url", default="http://localhost:8899")
    ap.add_argument("--heldout-cache", required=True, help="the pinned 1,418-vector query cache")
    ap.add_argument("--chooser", default=str(REPO / "data" / "eval_tasks.v2.yaml"))
    ap.add_argument("--prod-health-url", help="read prod's /health once to prove the incumbent is what prod loads")
    ap.add_argument("--tei-snapshots", help="HF cache snapshots dir of the TEI model, when that TEI reports "
                                            "model_sha=null")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--work-dir")
    ap.add_argument("--keys", nargs="*", help="only move these weight keys")
    ap.add_argument("--resume", action="store_true", help="reuse arm files already in --work-dir, after verifying "
                                                          "weights sha, collection and code sha")
    ap.add_argument("--max-arms", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    guard.refuse_serving_path(args.out_dir, "--out-dir")
    if args.work_dir:
        guard.refuse_serving_path(args.work_dir, "--work-dir")
    guard.refuse_non_twin(args.qdrant_url, args.collection)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
