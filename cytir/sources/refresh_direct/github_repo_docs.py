#!/usr/bin/env python3
"""`github_repo_docs` discovery adapter for the refresh-cadence lane (task 972).

For each repository a pack lists, call the acquirer's `run` in-process
(`cytir.sources.github_repo_docs.acquire`: pins the branch to a commit, keeps the exact API responses and blobs, writes
Markdown) into a PER-RUN preserve root, then return the sealed Markdown files as
candidates. A per-run root matters: the acquirer accepts an old terminal receipt on a
re-run, so a shared root would hand back last week's commit.

Candidate URL = `https://github.com/<owner>/<repo>/blob/<commit>/<path>`, the form
the acquirer's own `source_url` carries and the form the existing github rows use.
The licence is the pack route's declared SPDX id (checked live against
`GET /repos/{owner}/{repo}` when the route was added); the acquirer's own
receipt licence is only used as a cross-check, and a mismatch is refused.

The pack (`sources/us/github-repo-docs-tracked.yaml`) lists one repository per route:
`host: github.com/<owner>/<repo>`, `acquire.params.query: {branch, max_files}`,
`license:` (SPDX); an optional top-level `repo_list:` (JSONL in data/, task 996) appends one
route per row after the routes. Out of scope: tombstones for files removed from the repo.

Head-sha skip (task 2630). `github-repo-docs-heads.json` (the refresh lane's work dir, beside
`state.json`; master-side state, never shipped) maps `owner/repo` -> `{head_sha, tree_sha, landed_at,
pack_id}`. For a repo with a recorded head, `discover` makes ONE commit lookup; an equal head means
the repo is skipped (no tree listing, no raw GETs, no candidates, so no rows and no tombstones: its
landed rows stay) and counted in `orgs_unchanged`. A missing/unreadable file, or a repo with no
entry, behaves as before. The heads are written by `record_heads`, called by the composer only AFTER
the pack is queued (`queued: true`): a staged-only or failed run must not advance them, or a repo
whose rows never landed would be skipped forever. `repo_heads` in the result carries the heads this
run fetched (repos with an error are left out, so they are retried).
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import io
import json
import os
from pathlib import Path

from cytir.extract.github_agent_resources import candidates_from_receipt  # noqa: F401 -- task 1115: moved unchanged

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MAX_FILES = 6


UNKNOWN_LICENSES = {"", "UNKNOWN", "UNKNOWN-PER-REPO", "NOASSERTION", "NONE"}


def load_repo_list(path, *, defaults: dict) -> list[dict]:
    """Routes from a pack's `repo_list:` JSONL (task 996), in FILE order. Each row becomes a route
    with the pack defaults (trust); the licence is the row's own SPDX id. `tree: docs` + no file cap
    (the acquirer's docs-tree default) because the list is the phase-1 full-docs set."""
    repos = []
    for line in Path(path).read_text("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        repos.append({"owner_repo": row["owner_repo"], "branch": row.get("branch") or "main",
                      "sha": (row.get("sha") or "").lower() or None,
                      "max_files": None, "exclude_regex": None, "tree": "docs",
                      "license": (row.get("license") or "").strip(),
                      "in_index": row.get("in_index"),
                      "trust_tier": defaults["trust_tier"]})
    return repos


def load_repos(pack_path: str) -> list[dict]:
    from cytir.sources import pack_loader  # noqa: PLC0415
    doc = pack_loader.load_pack(REPO_ROOT / pack_path)
    repos = []
    for plan in pack_loader.plans(doc):
        host = (plan.host or "").split("/")
        if len(host) != 3 or host[0] != "github.com":
            continue
        q = plan.params.get("query") or {}
        repos.append({"owner_repo": f"{host[1]}/{host[2]}", "branch": q.get("branch") or "main",
                      "max_files": int(q.get("max_files") or DEFAULT_MAX_FILES),
                      "exclude_regex": q.get("exclude_regex") or None,
                      "tree": q.get("tree") or "capped",   # task 986: "docs" = full docs tree
                      "license": (plan.route_license or "").strip(),
                      "trust_tier": plan.effective_trust()})
    if doc.get("repo_list"):
        seen = {r["owner_repo"].casefold() for r in repos}   # a route wins over a list row
        listed = load_repo_list(REPO_ROOT / doc["repo_list"],
                                defaults={"trust_tier": repos[0]["trust_tier"] if repos
                                          else float(doc.get("trust_tier") or 0.75)})
        repos += [r for r in listed if r["owner_repo"].casefold() not in seen]
    return repos


def window(repos: list[dict], *, offset: int = 0, limit: int | None = None,
           in_index: bool | None = None) -> list[dict]:
    """Deterministic batch: optional `in_index` filter first (list rows only; a route has none),
    then `[offset : offset+limit]` over the merged order (routes, then list order)."""
    if in_index is not None:
        repos = [r for r in repos if r.get("in_index") is in_index]
    return repos[offset:(offset + limit) if limit is not None else None]


HEADS_NAME = "github-repo-docs-heads.json"
HEADS_SCHEMA = "aw.github-repo-docs-heads.v1"


def heads_path(opts: dict | None = None) -> Path:
    """The heads file: `adapter_opts["heads_path"]` or `<refresh_direct_work>/github-repo-docs-heads.json`."""
    if (opts or {}).get("heads_path"):
        return Path(opts["heads_path"])
    from cytir.core import paths  # noqa: PLC0415
    return Path(paths.get_value("refresh_direct_work")) / HEADS_NAME


def load_heads(path: Path) -> dict:
    """`{owner_repo: {head_sha, ...}}`; `{}` when the file is missing or unreadable (no skips)."""
    try:
        doc = json.loads(Path(path).read_text("utf-8"))
        heads = doc["repos"]
        return {k: v for k, v in heads.items() if isinstance(v, dict) and v.get("head_sha")}
    except (OSError, ValueError, KeyError, AttributeError):
        return {}


def record_heads(path: Path, repo_heads: dict, *, pack_id: str, now: _dt.datetime | None = None) -> int:
    """Merge `repo_heads` into the file (atomic write); returns how many repos it recorded. Call only
    after the pack is queued."""
    if not repo_heads:
        return 0
    stamp = (now or _dt.datetime.now(_dt.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    path = Path(path)
    try:
        doc = json.loads(path.read_text("utf-8"))
        repos = dict(doc["repos"])
    except (OSError, ValueError, KeyError, TypeError):
        repos = {}
    for owner_repo, h in repo_heads.items():
        repos[owner_repo] = {"head_sha": h["head_sha"], "tree_sha": h.get("tree_sha"),
                             "landed_at": stamp, "pack_id": pack_id}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"schema": HEADS_SCHEMA, "repos": repos}, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)
    return len(repo_heads)


def lookup_head(repo: dict) -> dict | None:
    """ONE `GET /repos/{r}/commits/{branch}` (the acquirer's own first call). `{head_sha, tree_sha}`,
    or None on any failure (then the repo is fetched in full and the acquirer reports the error)."""
    from urllib.parse import quote  # noqa: PLC0415
    from cytir.sources.github_repo_docs import acquire  # noqa: PLC0415
    settings = acquire.Settings(root=Path("."), max_files=0, retries=2, timeout=30.0)
    client = acquire.GitHubClient(os.environ.get("GH_TOKEN", "").strip(), settings)
    try:
        enc = "/".join(quote(p, safe="") for p in repo["owner_repo"].split("/"))
        resp = client.get(f"https://api.github.com/repos/{enc}/commits/{quote(repo['branch'], safe='')}")
        if resp.status_code != 200:
            return None
        body = resp.json()
        sha = str(body.get("sha", "")).lower()
        if not acquire.GIT_SHA_RE.fullmatch(sha):
            return None
        return {"head_sha": sha, "tree_sha": ((body.get("commit") or {}).get("tree") or {}).get("sha")}
    except Exception:  # noqa: BLE001 -- a failed lookup only forfeits the skip
        return None
    finally:
        client.close()


def acquire_argv(queue: Path, root: Path, max_files: int | None, tree: str = "capped",
                 workers: int = 1) -> list[str]:
    """The acquirer's `run` command line (the arguments the earlier out-of-process call passed)."""
    cap = ["--max-files", str(max_files)] if max_files is not None else []
    return ["run", "--queue", str(queue), "--root", str(root),
            *cap, "--tree", tree, "--workers", str(workers)]


def run_acquire(queue: Path, root: Path, max_files: int | None, tree: str = "capped",
                workers: int = 1) -> None:
    """Run the acquirer in-process (a token comes from its own env; see the operator file)."""
    from cytir.sources.github_repo_docs import acquire  # noqa: PLC0415
    args = acquire.parser().parse_args(acquire_argv(queue, root, max_files, tree, workers))
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = acquire.run(args)
    except acquire.AcquisitionError as exc:
        raise RuntimeError(f"acquire_github_repo_docs exit 2: {exc}") from exc
    if rc:
        raise RuntimeError(f"acquire_github_repo_docs exit {rc}: {err.getvalue()[-400:]}")


def read_receipt(root: Path, owner_repo: str) -> dict:
    from cytir.extract.github_agent_resources import receipt_name  # noqa: PLC0415
    return json.loads((root / "receipts" / receipt_name(owner_repo)).read_text("utf-8"))


def present_at_pin(root: Path, repo: dict) -> bool:
    """True when `root` already holds a valid terminal receipt for the repo at the pinned SHA
    (a row without a SHA accepts any valid terminal receipt)."""
    from cytir.extract.github_agent_resources import load_terminal  # noqa: PLC0415
    rec = load_terminal(root, repo["owner_repo"])
    if not rec or rec.get("status") != "complete":
        return False
    return not repo.get("sha") or str(rec.get("revision", "")).lower() == repo["sha"]


def discover(source_row: dict, *, repos=None, acquire=None, root: Path | None = None,
             head_lookup=None, heads: dict | None = None) -> dict:
    """`source_row["adapter_opts"]` (task 996, set by the composer): `repo_offset`, `repo_limit`,
    `repo_in_index` (bool|None), `preserve_root` (reuse this root, skip repos present at the pin),
    `workers` (acquirer --workers), `heads_path` (the heads file; tests). `head_lookup(repo)` and
    `heads` are test doubles for `lookup_head` and the heads file."""
    opts = source_row.get("adapter_opts") or {}
    repos = repos if repos is not None else load_repos(source_row["pack_path"])
    if not repos:
        raise ValueError("github_repo_docs: no github.com/<owner>/<repo> routes in "
                         f"{source_row.get('pack_path')!r}")
    repos = window(repos, offset=int(opts.get("repo_offset") or 0), limit=opts.get("repo_limit"),
                   in_index=opts.get("repo_in_index"))
    if not repos:
        raise ValueError("github_repo_docs: the --repo-offset/--repo-limit/--repo-in-index "
                         "window selects no repository")
    for r in repos:
        if r["license"].upper() in UNKNOWN_LICENSES:
            raise ValueError(f"{r['owner_repo']}: the route declares no known licence -- not acquired")
    acquire = acquire or run_acquire
    if opts.get("preserve_root"):
        run_root = Path(opts["preserve_root"])
    else:
        from cytir.core import paths  # noqa: PLC0415
        base = Path(root or paths.get_value("refresh_github_docs_root"))
        run_root = base / _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root.mkdir(parents=True, exist_ok=True)
    workers = int(opts.get("workers") or 1)
    reuse = bool(opts.get("preserve_root"))
    todo = [r for r in repos if not (reuse and present_at_pin(run_root, r))]
    if heads is None:
        from cytir.core import paths  # noqa: PLC0415
        try:
            heads = load_heads(heads_path(opts))
        except paths.SettingRefused:        # no work root configured: nothing recorded, no skips
            heads = {}
    unchanged: dict[str, dict] = {}
    looked_up: dict[str, dict] = {}
    heads_lookup_failed = 0                # task 2718: lookup returned None -> fetched unconditionally
    if heads:
        lookup = head_lookup or lookup_head
        for r in todo:
            prior = heads.get(r["owner_repo"])
            if not prior:
                continue                       # never landed: no lookup, fetch as before
            now = lookup(r)
            if now and now["head_sha"] == str(prior["head_sha"]).lower():
                unchanged[r["owner_repo"]] = now
            elif now:
                looked_up[r["owner_repo"]] = now
            else:
                heads_lookup_failed += 1
        todo = [r for r in todo if r["owner_repo"] not in unchanged]
    errors: dict[str, str] = {}
    # One acquirer run per (file cap, tree) group, so `--workers` parallelises across repos.
    groups: dict[tuple, list[dict]] = {}
    for r in todo:
        groups.setdefault((r["max_files"], r.get("tree", "capped")), []).append(r)
    for n, ((max_files, tree), grp) in enumerate(groups.items()):
        queue = run_root / f"queue-{n}.jsonl"
        queue.write_text("".join(json.dumps(
            {"owner_repo": r["owner_repo"], "branch": r["branch"], "license": r["license"],
             "mode": "deep", **({"exclude_regex": r["exclude_regex"]} if r.get("exclude_regex") else {})}
        ) + "\n" for r in grp))
        extra = {"workers": workers} if workers != 1 else {}
        try:
            if tree == "capped" and not extra:
                acquire(queue, run_root, max_files)
            else:                        # injected test doubles keep the 3-argument form
                acquire(queue, run_root, max_files, tree, **extra)
        except Exception as exc:  # noqa: BLE001 -- named; repos with a valid receipt are still read
            for r in grp:
                errors[r["owner_repo"]] = f"{type(exc).__name__}: {exc}"
    cands = []
    repo_heads: dict[str, dict] = {}
    for repo in repos:
        if repo["owner_repo"] in unchanged:
            continue                           # skipped: no receipt, no rows, no tombstones
        try:
            receipt = read_receipt(run_root, repo["owner_repo"])
            cands.extend(candidates_from_receipt(run_root, receipt, repo))
            head = str(receipt["revision"]).lower()
            seen = looked_up.get(repo["owner_repo"]) or {}
            repo_heads[repo["owner_repo"]] = {
                "head_sha": head, "tree_sha": seen.get("tree_sha") if seen.get("head_sha") == head else None}
            errors.pop(repo["owner_repo"], None)
        except Exception as exc:  # noqa: BLE001
            errors.setdefault(repo["owner_repo"], f"{type(exc).__name__}: {exc}")
    return {"adapter": "github_repo_docs", "orgs_checked": len(repos), "orgs_errored": len(errors),
            "orgs_reused": len(repos) - len(todo) - len(unchanged), "orgs_unchanged": len(unchanged),
            "heads_lookup_failed": heads_lookup_failed,
            "repo_heads": {k: v for k, v in repo_heads.items() if k not in errors},
            "errors": errors, "candidates_found": len(cands), "candidates": cands,
            "preserve_root": str(run_root)}
