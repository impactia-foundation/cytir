"""Landing check for one queued pack (task 701; moved from the task-699 scripts `verify_land.py`,
`domcount.py`, `spamcheck.py`, which ran on the prod path from outside the repo).

READ-ONLY: the pack's receipt, indexed id retrievals and exact `domain` counts. It never writes twin or prod.

    python -m cytir.ops.landing_check pack --queue-dir Q --pack-id P --chunks-dir D [--exclude-file F] \\
        --twin-url U --twin-collection C --prod-url U --prod-collection C [--sample 3000]
    python -m cytir.ops.landing_check domains --chunks-dir D [--exclude-file F] ...same urls... [--top 12]

`pack` PASSES (exit 0) only when ALL hold, each printed:
  1. the receipt status is LANDED;
  2. on each side staged rows == written + every counted drop (`skipped_by_projection`, `excluded_undated`,
     `excluded_out_of_scope_language` = the declared exclusion file, `already_present`) + the content-check
     word-floor drops, i.e. nothing vanished unnamed;
  3. twin and prod hold the same point count;
  4. a sample of the NON-excluded stage ids is served 100 % on both sides. The ids the content check DROPPED
     (the receipt's `mechanical_content.problem_sample`, the word-floor drops among them) are not in the
     sample: they are in the stage and never written, by design (task 760; shard59 of task 749 printed
     `sample_served false` on a correct landing). The receipt names only the first 20 of them; a drop it does
     not name (`failing_rows` above 20) may fall in the sample, and that many misses are tolerated, printed as
     `sample_unnamed_content_drops_tolerated`;
  5. the declared-excluded ids, read per row with the row's `reason` (the format `cytir.index.exclude_file`
     supports): ids excluded as `already_on_twin_and_prod` (ruling 23.5, ADR-116: the compose-time served
     exclusion) must be PRESENT on twin and prod; every other excluded id (spam, floor trim, an overlap file
     whose rows carry no reason) must be ABSENT. The two groups are counted and reported separately
     (`served_excluded_*`, `excluded_other_in_file` / `excluded_found_*`); a served id that is missing is a
     landing defect, not a pass.
`domains` prints, per top domain of the stage after exclusion, sealed rows vs serving rows on twin and
prod (flags `CHECK` when serving < sealed or twin != prod).
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import argparse
import glob
import gzip
import json
import random
import sys
import urllib.request
from pathlib import Path

from cytir.index.served_exclusion import SERVED_REASON

DROP_KEYS = ("skipped_by_projection", "excluded_undated", "excluded_out_of_scope_language",
             "already_present")


def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"}, method="POST")
    return json.load(IH.urlopen(req, timeout=120))


def _get(url: str) -> dict:
    return json.load(IH.urlopen(url, timeout=60))


def points_count(base: str, collection: str) -> int:
    return _get(f"{base.rstrip('/')}/collections/{collection}")["result"]["points_count"]


def served(base: str, collection: str, ids: list[str]) -> int:
    n = 0
    for i in range(0, len(ids), 500):
        n += len(_post(f"{base.rstrip('/')}/collections/{collection}/points",
                       {"ids": ids[i:i + 500], "with_payload": False, "with_vector": False})["result"])
    return n


def domain_count(base: str, collection: str, domain: str) -> int:
    return _post(f"{base.rstrip('/')}/collections/{collection}/points/count",
                 {"filter": {"must": [{"key": "domain", "match": {"value": domain}}]},
                  "exact": True})["result"]["count"]


def stage_ids(chunks_dir: str | Path) -> list[tuple[str, str]]:
    out = []
    files = sorted(glob.glob(str(chunks_dir) + "/*.texts.jsonl.gz"))
    if not files:
        raise SystemExit(f"REFUSED: no *.texts.jsonl.gz under {chunks_dir}")
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    out.append((r["point_id"], r.get("domain", "")))
    return out


def excluded_rows(path: str | Path | None) -> dict[str, str | None]:
    """`{point_id: reason or None}` of an exclusion file (`.jsonl`, optionally `.gz`)."""
    if not path:
        return {}
    opener = gzip.open if str(path).endswith(".gz") else open
    out: dict[str, str | None] = {}
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[str(r["point_id"])] = r.get("reason")
    return out


def excluded_ids(path: str | Path | None) -> set[str]:
    return set(excluded_rows(path))


def accounting(side: dict, mech: dict) -> dict:
    rows = side.get("rows") or {}
    drops = {k: int(rows.get(k) or 0) for k in DROP_KEYS}
    floor = int(mech.get("word_floor_dropped") or 0)
    written = int(side.get("written") or 0)
    vectors = int(rows.get("vectors") or 0)
    return {"vectors": vectors, "written": written, **drops, "word_floor_dropped": floor,
            "balanced": vectors == written + sum(drops.values()) + floor}


def content_drops(mech: dict) -> tuple[set[str], int]:
    """`(named ids, unnamed count)` of the rows the mechanical content check dropped from a pack. The receipt
    keeps `problem_sample` (the first 20 problems, each dropped) and the count `failing_rows`."""
    named = set((mech.get("problem_sample") or {}))
    return named, max(0, int(mech.get("failing_rows") or 0) - len(named))


def check_pack(queue_dir: str, pack_id: str, chunks_dir: str, *, twin: tuple[str, str],
               prod: tuple[str, str], exclude_file: str | None = None, sample: int = 3000,
               seed: int = 3) -> dict:
    receipt_path = Path(queue_dir) / pack_id / "receipt.json"
    res: dict = {"pack_id": pack_id, "checks": {}}
    if not receipt_path.is_file():
        res["checks"]["receipt_landed"] = False
        res["ok"] = False
        return res
    rec = json.loads(receipt_path.read_text())
    mech = rec.get("mechanical_content") or {}
    res["checks"]["receipt_landed"] = rec.get("status") == "LANDED"
    res["receipt_status"] = rec.get("status")
    res["accounting"] = {s: accounting(rec.get(s) or {}, mech) for s in ("twin", "prod")}
    res["checks"]["accounted"] = all(a["balanced"] for a in res["accounting"].values())
    t, p = points_count(*twin), points_count(*prod)
    res["counts"] = {"twin": t, "prod": p}
    res["checks"]["twin_equals_prod"] = t == p
    ids = stage_ids(chunks_dir)
    rows = excluded_rows(exclude_file)
    ex = set(rows)
    served_ids = sorted(i for i, why in rows.items() if why == SERVED_REASON)
    other_ids = sorted(i for i, why in rows.items() if why != SERVED_REASON)
    dropped, unnamed = content_drops(mech)
    keep = [i for i, _ in ids if i not in ex and i not in dropped]
    samp = random.Random(seed).sample(keep, min(sample, len(keep)))
    res["content_dropped_named"] = len(dropped)
    res["sample_unnamed_content_drops_tolerated"] = unnamed
    res["stage_rows"] = len(ids)
    res["excluded_in_file"] = len(ex)
    res["served_excluded_in_file"] = len(served_ids)
    res["excluded_other_in_file"] = len(other_ids)
    for name, (base, coll) in (("twin", twin), ("prod", prod)):
        res["checks"][f"sample_served_{name}"] = len(samp) - served(base, coll, samp) <= unnamed if samp else False
        # the served group must be ON the target (it is why the pack did not write it) ...
        have = served(base, coll, served_ids) if served_ids else 0
        res[f"served_excluded_found_{name}"] = have
        res["checks"][f"served_excluded_present_{name}"] = have == len(served_ids)
        # ... every other excluded id must NOT be
        found = served(base, coll, other_ids) if other_ids else 0
        res[f"excluded_found_{name}"] = found
        res["checks"][f"excluded_absent_{name}"] = found == 0
    res["ok"] = all(res["checks"].values())
    return res


def check_domains(chunks_dir: str, *, twin: tuple[str, str], prod: tuple[str, str],
                  exclude_file: str | None = None, top: int = 12) -> dict:
    ex = excluded_ids(exclude_file)
    dom: dict[str, int] = {}
    for pid, d in stage_ids(chunks_dir):
        if pid not in ex:
            dom[d] = dom.get(d, 0) + 1
    rows = []
    for d, n in sorted(dom.items(), key=lambda x: -x[1])[:top]:
        tw, pr = domain_count(*twin, d), domain_count(*prod, d)
        rows.append({"domain": d, "sealed": n, "twin": tw, "prod": pr, "check": not (tw >= n and pr >= n and tw == pr)})
    return {"domains": len(dom), "rows_sealed": sum(dom.values()), "top": rows,
            "ok": not any(r["check"] for r in rows)}


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pack", "domains"):
        p = sub.add_parser(name)
        p.add_argument("--chunks-dir", required=True)
        p.add_argument("--exclude-file", default=None)
        for s in ("twin", "prod"):
            p.add_argument(f"--{s}-url", required=True)
            p.add_argument(f"--{s}-collection", required=True)
    pk = sub.choices["pack"]
    pk.add_argument("--queue-dir", required=True)
    pk.add_argument("--pack-id", required=True)
    pk.add_argument("--sample", type=int, default=3000)
    sub.choices["domains"].add_argument("--top", type=int, default=12)
    a = ap.parse_args(argv)
    twin, prod = (a.twin_url, a.twin_collection), (a.prod_url, a.prod_collection)
    if a.cmd == "pack":
        res = check_pack(a.queue_dir, a.pack_id, a.chunks_dir, twin=twin, prod=prod,
                         exclude_file=a.exclude_file, sample=a.sample)
    else:
        res = check_domains(a.chunks_dir, twin=twin, prod=prod, exclude_file=a.exclude_file, top=a.top)
    print(json.dumps(res, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
