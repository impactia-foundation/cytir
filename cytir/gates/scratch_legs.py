"""The flip gate's LEG arms read on a CANDIDATE or scratch collection (task 863, spec `aw-e10-rebuild`
section 5 "component arms"; runbook P6).

The gate reads three legs per side (`data/flip-gate-contract.v1.json` `legs`). What a leg IS -- the env
overrides, k, the per-row search arguments -- lives in ONE place, `cytir.gates.leg_definition`; the
incumbent's legs on prod are read by `cytir.gates.prod_legs`. This module is the other side: the same
serving code (`cytir/serve`, in-process) pointed at a NON-prod Qdrant (the candidate on the build
box, or a component arm on a scratch endpoint), with the side's flags from an env file and the leg
from `leg_definition`, so "arm minus control" measures two indexes and not two instruments.

Per query: `retrieval.search(query, **leg_definition.search_kwargs(row))`, the gold's rank by the gate's
scorer (`cytir.gates.authority.rank_of`), and whether the gold is PRESENT in the collection (by the indexed
`url` field: the gold url without its fragment, http/https, with and without `www.`, with and without
the trailing slash). An absent gold and a missed gold both rank None; the presence column keeps them
apart (a comparison restricts to golds present in every compared arm).

Two instruments, one query file each, built here from the committed sets (never a hand list):
  * `629`: the 629 chooser, `data/eval_tasks.v2.yaml` (`target_url`; class = `type`);
  * `heldout`: the held-out battery, every set of `cytir.gates.authority.SETS` (`gold_of`; class = set).

Refuses a prod collection (`cytir.index.prod_guard`) and a live name (`cytir.index.live_target`). Reads only.

    python -m cytir.gates.scratch_legs --qdrant http://127.0.0.1:6484 --collection e863_e1 \\
        --leg dense-only --weights <dense-only weights> --env-file <side flags> \\
        --tei http://127.0.0.1:8902 --instrument 629 --out e863_e1.629.dense-only.json
"""
from __future__ import annotations

import argparse
import json
import os
from cytir.core import envnames as _envnames
import pathlib
import sys
import time
import urllib.parse

from cytir.gates import leg_definition as LD
# the helpers the incumbent's reader uses (one parse of the side's env, one TEI pin check, one file sha)
from cytir.gates.prod_legs import ProdLegRefused, check_tei_pin, map_image_paths, parse_env_text, sha256_file

SCHEMA = "aw.scratch-leg-read.v1"
ROOT = pathlib.Path(__file__).resolve().parents[2]
from cytir.core.serve_layout import SERVER_REL  # noqa: E402
INSTRUMENTS = ("629", "heldout")
#: keys the side's env file may not set: the reader names the collection, the Qdrant and the TEI
OWN_KEYS = ("AW_QDRANT_URL", "AW_COLLECTION", "AW_TEI_URL")


class ScratchLegRefused(RuntimeError):
    pass


def queries_of(instrument: str) -> list[dict]:
    """The rows of one instrument: {id, query, lang, country, gold_urls, gold_domains, cls}."""
    from cytir.gates import authority as gate
    if instrument == "629":
        import yaml
        from cytir.core.paths import data_path
        chooser = data_path("eval_tasks.v2.yaml")
        if not os.path.isfile(chooser):
            raise ScratchLegRefused(f"the 629 chooser set is missing: {chooser} (the public cut ships none)")
        rows = yaml.safe_load(open(chooser, encoding="utf-8"))["eval"]
        return [{"id": r["id"], "query": r.get("question") or r["query"], "lang": r.get("lang"),
                 "country": r.get("country"), "gold_urls": [r["target_url"]], "gold_domains": [],
                 "cls": r.get("type")} for r in rows]
    if instrument == "heldout":
        out = []
        for name, path in gate.SETS:
            if not os.path.exists(path):
                continue
            for r in gate.load_rows(path):
                urls, domains = gate.gold_of(r)
                if not urls and not domains:
                    continue
                q = r.get("query") or r.get("q")
                out.append({"id": f"{name}|{r.get('id') or q}", "query": q, "lang": r.get("lang"),
                            "country": r.get("country"), "gold_urls": urls, "gold_domains": domains,
                            "cls": name})
        return out
    raise ScratchLegRefused(f"instrument {instrument!r}: one of {INSTRUMENTS}")


def url_variants(url: str) -> list[str]:
    """The spellings a stored `url` may carry for one gold (fragment stripped)."""
    u = (url or "").split("#")[0].strip()
    p = urllib.parse.urlsplit(u)
    if not p.netloc:
        return [u] if u else []
    hosts = {p.netloc, p.netloc[4:] if p.netloc.startswith("www.") else "www." + p.netloc}
    paths = {p.path, p.path.rstrip("/") if p.path.endswith("/") else p.path + "/"}
    out = set()
    for scheme in ("https", "http"):
        for h in hosts:
            for path in paths:
                out.add(urllib.parse.urlunsplit((scheme, h, path, p.query, "")))
    return sorted(out)


def present(qdrant: str, collection: str, gold_urls: list[str]) -> bool | None:
    """True when a stored row carries one of the gold's url spellings; None for a domain-only gold."""
    from cytir.index import promote_sealed_pack as PSP
    variants = sorted({v for g in gold_urls for v in url_variants(g)})
    if not variants:
        return None
    res = PSP._api(qdrant, f"/collections/{collection}/points/scroll",
                   {"limit": 1, "with_payload": False, "with_vector": False,
                    "filter": {"must": [{"key": "url", "match": {"any": variants}}]}})
    return bool(res.get("points"))


def read(qdrant: str, collection: str, *, leg: str, weights: str | None, env_file: str, tei: str,
         instrument: str, out: str, qvec_cache: str | None = None, qdrant_timeout_s: float | None = None,
         log=print) -> dict:
    from cytir.index import live_target as LT
    from cytir.index import prod_guard as PG
    if PG.is_prod(qdrant, collection):
        raise ScratchLegRefused(f"{qdrant} / {collection} is prod: the incumbent's legs are read by "
                                "cytir.gates.prod_legs, never here")
    try:
        LT.refuse_live_target(qdrant, collection)
    except LT.LiveTargetRefused as exc:
        raise ScratchLegRefused(f"{exc}") from exc
    weights_sha = sha256_file(weights) if weights else None
    overrides = LD.leg_env(leg, os.path.abspath(weights) if weights else None, weights_sha)
    try:
        env = map_image_paths(parse_env_text(pathlib.Path(env_file).read_text()))
        pin = check_tei_pin(tei)
    except ProdLegRefused as exc:
        raise ScratchLegRefused(str(exc)) from exc
    for k in OWN_KEYS:
        _envnames.pop_knob(env, k)
    os.environ.update(env)
    os.environ.update(overrides)
    os.environ.update({"AW_QDRANT_URL": qdrant, "AW_COLLECTION": collection, "AW_TEI_URL": tei})
    queries = queries_of(instrument)
    sys.path[:0] = [str(ROOT)]
    from cytir.gates import authority as gate
    if qvec_cache:      # the contract's pinned 1418 instrument: cached query vectors, not live TEI (task 926)
        gate.install_cached_embedder(qvec_cache)
    from cytir.serve import retrieval
    qdrant_timeout = LD.apply_qdrant_timeout(retrieval, qdrant_timeout_s)
    effective = retrieval.effective_config()
    rows, t0 = [], time.time()
    for i, q in enumerate(queries):
        kw = LD.search_kwargs(q, LD.GATE_K)
        row = {"id": q["id"], "cls": q["cls"], "lang": kw["lang"], "country": kw["country"]}
        try:
            hits = retrieval.search(q["query"], **kw)
            urls = [h.get("url") or "" for h in hits]
            row.update(status="OK", rank=gate.rank_of(urls, q["gold_urls"], q["gold_domains"]),
                       urls=urls)
        except Exception as exc:   # noqa: BLE001 -- recorded; a failed query is never a miss
            row.update(status="ERROR", reason=f"{type(exc).__name__}: {exc}"[:300])
        row["present"] = present(qdrant, collection, q["gold_urls"])
        rows.append(row)
        if (i + 1) % 200 == 0:
            log(f"[{collection} {leg} {instrument}] {i + 1}/{len(queries)} {time.time() - t0:.0f}s", flush=True)
    doc = {"schema": SCHEMA, "collection": collection, "qdrant": qdrant, "leg": leg,
           "leg_overrides": overrides, "k": LD.GATE_K, "instrument": instrument,
           "weights": {"path": os.path.abspath(weights), "sha256": weights_sha} if weights else None,
           "env_file": {"path": os.path.abspath(env_file), "sha256": sha256_file(env_file), "keys": len(env)},
           "embedder": ({**pin, "qvec_cache": qvec_cache, "sha256": sha256_file(qvec_cache)} if qvec_cache else pin),
           "gate_set_version": gate.GATE_SET_VERSION, "effective_config": effective,
           "qdrant_timeout_s": qdrant_timeout, "qdrant_timeout_s_override": qdrant_timeout_s,
           "retrieval_sha256": sha256_file(ROOT / SERVER_REL / "retrieval.py"),
           "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "seconds": round(time.time() - t0, 1),
           "counts": {s: sum(1 for r in rows if r["status"] == s) for s in ("OK", "ERROR")},
           "rows": rows}
    pathlib.Path(out).write_text(json.dumps(doc, ensure_ascii=False) + "\n")
    return doc


def main(argv=None, prog=None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    ap.add_argument("--qdrant", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--leg", required=True, choices=LD.LEGS)
    ap.add_argument("--weights", default=None, help="the leg's weights file (dense-only, sparse-only)")
    ap.add_argument("--env-file", required=True, help="the side's CYTIR_* serving flags")
    ap.add_argument("--tei", required=True, help="a revision-pinned TEI")
    ap.add_argument("--instrument", required=True, choices=INSTRUMENTS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qvec-cache", default=None, help="the gate's immutable query-vector cache (instrument 1418)")
    ap.add_argument("--qdrant-timeout-s", type=float, default=None,
                    help="the serving code's Qdrant client timeout for this read (default: the side's env)")
    a = ap.parse_args(argv)
    try:
        doc = read(a.qdrant, a.collection, leg=a.leg, weights=a.weights, env_file=a.env_file, tei=a.tei,
                   instrument=a.instrument, out=a.out, qvec_cache=a.qvec_cache,
                   qdrant_timeout_s=a.qdrant_timeout_s)
    except (ScratchLegRefused, LD.LegUndefined) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out": a.out, **doc["counts"], "seconds": doc["seconds"]}))
    return 0 if not doc["counts"]["ERROR"] else 3


if __name__ == "__main__":
    sys.exit(main())
