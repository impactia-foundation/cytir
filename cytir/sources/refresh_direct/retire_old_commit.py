"""Retire the old-commit rows of refreshed `github_repo_docs` repos (task 2634; spec
`gh-repo-docs-retire-old-commit`, section 4). PREPARE ONLY: this builds a pure
`superseded_generation` pack and its removed-rows export; it writes no twin, no prod and nothing
in the promotion queue. The drain edit that accepts a `survivor_url` is a separate reviewed PR.

Why: a refresh re-stages each doc under a new commit url, so the composer's exact-url supersede
finds nothing and the old commit's rows keep serving next to the new ones.

Steps (one door, `main`):
1. ONE filtered scroll over the twin's github slice (`domain` keyword; payload `url` and
   `provenance`, ids only, no vectors), grouped client-side by owner/repo.
2. A repo is eligible only when the heads file records a head AND that head's pack has a DONE
   marker and a LANDED receipt in the queue (read only): the new rows are served.
3. Old rows = rows of an eligible repo whose sha is not the head. The survivor of an old row is a
   row of the same provenance at the same path under the head sha (`is_sibling_commit_url`). An old
   row with no survivor (the path vanished) is COUNTED in `no_survivor` and not tombstoned: the
   drain refuses a pack whole when one id has no survivor, and a page is never emptied.
4. At most `--cap` tombstones per pack (default 5,000), whole repos only, never a split repo;
   a repo larger than the cap is reported as `over_cap` and skipped.
5. The pack goes through the one pack writer (`cytir.news.compose.write_pack`) with every tombstone
   `{id, url, reason: superseded_generation, survivor_url}`, no points; the export through
   `url_group_supersede.write_export` into the operator's `removed_rows_export_dir`.

`--scan-only` prints per-repo old/new/no-survivor counts as JSON. `--dry-run` prints the plan and
writes nothing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from cytir.core import twin as TWIN
from cytir.index import github_sibling_commit as GSC
from cytir.index import row_decode as RD
from cytir.sources.refresh_direct import github_repo_docs as GRD

REASON = "superseded_generation"
DEFAULT_CAP = 5000
SCROLL_LIMIT = 1024
DEFAULT_DOMAIN = "github.com"
FAMILY = "news"   # the refresh lane's composer stamps this family on every pack (direct_compose)


def _api(url: str, path: str, body: dict | None = None):
    from cytir.index import promote_sealed_pack as PSP  # noqa: PLC0415 -- the one guarded client
    return PSP._api(url, path, body)


def scan_slice(twin_url: str, collection: str, *, domain: str = DEFAULT_DOMAIN, api=_api
               ) -> "dict[str, list[dict]]":
    """ONE filtered scroll: `{owner/repo (case-folded): [{id, url, provenance, sha, path}]}` of the
    github blob rows the twin serves. Payload fields `url`/`provenance` only; no vectors."""
    decode = RD.reader(collection)
    by_repo: dict[str, list[dict]] = {}
    offset = None
    while True:
        body = {"limit": SCROLL_LIMIT, "with_payload": ["url", "provenance"], "with_vector": False,
                "filter": {"must": [{"key": "domain", "match": {"value": domain}}]}}
        if offset is not None:
            body["offset"] = offset
        res = api(twin_url, f"/collections/{collection}/points/scroll", body) or {}
        for p in res.get("points", []):
            pl = decode(p.get("payload") or {})
            parsed = GSC.parse_blob_url(pl.get("url"))
            if parsed is None:
                continue
            owner, repo, sha, path = parsed
            by_repo.setdefault(f"{owner}/{repo}".lower(), []).append(
                {"id": p["id"], "url": pl["url"], "provenance": pl.get("provenance"),  # canonical row
                 "sha": sha, "path": path})
        offset = res.get("next_page_offset")
        if offset is None:
            return by_repo


def pack_landed(queue_dir: pathlib.Path, pack_id: str) -> bool:
    """The head's pack is served on both sides: a DONE marker AND a receipt with status LANDED."""
    d = pathlib.Path(queue_dir) / str(pack_id)
    try:
        return (d / "DONE").is_file() and json.loads((d / "receipt.json").read_text()).get("status") == "LANDED"
    except (OSError, ValueError, AttributeError):
        return False


def plan_repo(rows: list[dict], head_sha: str) -> dict:
    """Pure. `{old, new, tombstones, no_survivor}` for one repo's rows against its head."""
    new = [r for r in rows if r["sha"] == head_sha]
    old = [r for r in rows if r["sha"] != head_sha]
    tombs, orphans = [], []
    for r in old:
        surv = next((n for n in new if n["provenance"] == r["provenance"]  # canonical row
                     and GSC.is_sibling_commit_url(r["url"], n["url"])), None)
        if surv is None or not r["provenance"]:  # canonical row
            orphans.append(r["id"])
        else:
            tombs.append({"id": r["id"], "url": r["url"], "reason": REASON,
                          "survivor_url": surv["url"]})
    return {"old": len(old), "new": len(new), "tombstones": tombs, "no_survivor": orphans,
            "no_provenance_old": sum(1 for r in old if not r["provenance"]),  # canonical row
            "no_provenance_new": sum(1 for r in new if not r["provenance"])}  # canonical row


def build_plan(by_repo: dict, heads: dict, queue_dir: pathlib.Path, *, cap: int = DEFAULT_CAP) -> dict:
    """Pure over its inputs (reads the queue's markers only). Repos in sorted order; a repo that
    would take the pack past `cap` is `deferred` to the next pack, a repo alone over `cap` is
    `over_cap`. Heads keys are matched case-folded."""
    heads_ci = {k.lower(): v for k, v in heads.items()}
    report: dict[str, dict] = {}
    tombs: list[dict] = []
    deferred: list[str] = []
    for key in sorted(by_repo):
        head = heads_ci.get(key)
        if not head:
            report[key] = {"status": "no_head"}
            continue
        if not pack_landed(queue_dir, head.get("pack_id")):
            report[key] = {"status": "pack_not_landed", "pack_id": head.get("pack_id")}
            continue
        p = plan_repo(by_repo[key], head["head_sha"])
        entry = {"status": "eligible", "old": p["old"], "new": p["new"],
                 "tombstones": len(p["tombstones"]), "no_survivor": len(p["no_survivor"]),
                 "no_provenance_old": p["no_provenance_old"],
                 "no_provenance_new": p["no_provenance_new"]}
        if len(p["tombstones"]) > cap:
            entry["status"] = "over_cap"
        elif len(tombs) + len(p["tombstones"]) > cap:
            entry["status"] = "deferred"
            deferred.append(key)
        else:
            tombs.extend(p["tombstones"])
        report[key] = entry
    return {"repos": report, "tombstones": tombs, "deferred": deferred,
            "no_survivor": sum(v.get("no_survivor", 0) for v in report.values())}


def _summary(plan: dict) -> dict:
    return {"tombstones": len(plan["tombstones"]), "no_survivor": plan["no_survivor"],
            "deferred_repos": plan["deferred"],
            "repos": {k: v for k, v in plan["repos"].items() if v["status"] == "eligible"
                      and (v["old"] or v["no_survivor"])},
            "skipped": {k: v["status"] for k, v in plan["repos"].items() if v["status"] != "eligible"}}


def write_retire_pack(plan: dict, *, pack_id: str, out_dir: pathlib.Path, twin_url: str,
                      collection: str, export_dir: pathlib.Path) -> dict:
    """The one pack writer + the one exporter. Export first (it verifies every row on the twin),
    then the pure pack naming it."""
    from cytir.index import url_group_supersede as US  # noqa: PLC0415
    from cytir.news import compose as C  # noqa: PLC0415
    tombs = plan["tombstones"]
    path = pathlib.Path(export_dir) / f"{pack_id}-removed-rows.jsonl.gz"
    export = US.write_export(twin_url, collection, [t["id"] for t in tombs],
                             {t["id"]: t["url"] for t in tombs}, path)
    export = {k: export[k] for k in ("path", "sha256", "rows")}
    return C.write_pack(str(out_dir), kept=[], tombstones=tombs, pack_id=pack_id, family=FAMILY,
                        collection=collection, days=[time.strftime("%Y-%m-%d", time.gmtime())],
                        extra_manifest={"arm": "gh_repo_docs_retire_old_commit",
                                        "export": export,
                                        "tombstones_by_reason": {REASON: len(tombs)},
                                        "no_survivor_not_tombstoned": plan["no_survivor"]})


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    ap.add_argument("--heads", default=None, help="heads file (default: the refresh lane's)")
    ap.add_argument("--queue-dir", required=True, help="promotion queue, READ ONLY (DONE/receipts)")
    ap.add_argument("--twin-url", default=None)
    ap.add_argument("--collection", default=None)
    ap.add_argument("--domain", default=DEFAULT_DOMAIN)
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--pack-id", default=None)
    ap.add_argument("--out-dir", default=None, help="where the pack is written (never the queue)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--scan-only", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    twin_url = args.twin_url or TWIN.url()
    collection = args.collection or TWIN.collection()
    heads = GRD.load_heads(pathlib.Path(args.heads) if args.heads else GRD.heads_path())
    queue = pathlib.Path(args.queue_dir)
    plan = build_plan(scan_slice(twin_url, collection, domain=args.domain), heads, queue, cap=args.cap)
    if args.scan_only:
        print(json.dumps({"repos": plan["repos"]}, indent=1, sort_keys=True))
        return 0
    summary = _summary(plan)
    if args.dry_run or not plan["tombstones"]:
        print(json.dumps({"dry_run": bool(args.dry_run), **summary}, indent=1, sort_keys=True))
        return 0
    if not (args.pack_id and args.out_dir):
        print("--pack-id and --out-dir are required to write a pack", file=sys.stderr)
        return 2
    out = pathlib.Path(args.out_dir).resolve()
    if queue.resolve() in (out, *out.parents):
        print(f"--out-dir {out} lies inside the promotion queue {queue}: this lane never enqueues",
              file=sys.stderr)
        return 2
    from cytir.index import url_group_supersede as US  # noqa: PLC0415
    manifest = write_retire_pack(plan, pack_id=args.pack_id, out_dir=out / args.pack_id,
                                 twin_url=twin_url, collection=collection,
                                 export_dir=US.export_dir(queue))
    print(json.dumps({"pack_id": args.pack_id, "tombstones": manifest["tombstones_count"],
                      "export": manifest["export"], **summary}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], "python -m cytir.sources.refresh_direct.retire_old_commit"))
