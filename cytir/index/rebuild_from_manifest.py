"""cytir.index.rebuild_from_manifest -- `cyt index rebuild`: reconstruct an AW collection from one
sealed production manifest.

This is the repo-native entry point intended for a fresh clone (the one rebuild door; moved
from the pipeline tree in task 1234, behaviour unchanged).  The manifest
supplies ordered argv arrays, never shell fragments.  The runner verifies the
source-preservation contract, checks out the exact pipeline git revision into
the work directory when necessary, substitutes only documented placeholders,
and records every phase in an fsync'd append-only run ledger.

Example:

    cyt index rebuild run \
      --manifest /path/to/production-preservation.json \
      --artifact-root /path/to/aw-artifacts --work /path/to/rebuild
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from cytir.index.preservation_contract import ContractError, verify_manifest

#: The checkout this module runs from: the default `--repo` the manifest's git revision is pinned in.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _now() -> str:
    return time.strftime("%FT%TZ", time.gmtime())


def _append(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps({"ts": _now(), **event}, sort_keys=True) + "\n")
        out.flush()
        os.fsync(out.fileno())


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read rebuild manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError("rebuild manifest must be a JSON object")
    return value


def _git(repo: Path, *args: str, check: bool = True) -> str:
    run = subprocess.run(["git", "-C", str(repo), *args], text=True,
                         capture_output=True)
    if check and run.returncode:
        raise ContractError(
            f"git {' '.join(args)} failed in {repo}: {run.stderr.strip()}")
    return run.stdout.strip()


def _reusable_worktree(target: Path, git_sha: str) -> bool:
    """True when `target` is a complete worktree already at `git_sha` (reuse it untouched)."""
    if not (target / ".git").exists():
        return False
    run = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], text=True,
                         capture_output=True)
    return run.returncode == 0 and run.stdout.strip() == git_sha


def _refuse_dirty_worktree(target: Path) -> None:
    """A COMPLETE worktree at another sha that holds uncommitted changes is someone's work, not
    an interrupted add: refuse instead of deleting it (orchestrator review of task 1713)."""
    if not (target / ".git").exists():
        return
    run = subprocess.run(["git", "-C", str(target), "status", "--porcelain"], text=True,
                         capture_output=True)
    if run.returncode == 0 and run.stdout.strip():
        raise SystemExit(f"rebuild: refusing to remove {target}: it is a complete worktree with "
                         f"uncommitted changes (not an interrupted add). Clean or remove it by hand.")


def _clear_interrupted_worktree(repo: Path, target: Path) -> None:
    """Task 1713: a `kill -9` during `git worktree add` leaves a half-made `target` dir and/or a
    (possibly locked) `.git/worktrees/<name>` entry, and the next `add` then refuses.  Remove both
    so `add` can run.  Only called when `target` is not a complete worktree at the wanted sha."""
    _refuse_dirty_worktree(target)
    listing = _git(repo, "worktree", "list", "--porcelain")
    registered = any(
        line == f"worktree {target}" or line == f"worktree {target.resolve()}"
        for line in listing.splitlines())
    if registered:
        print(f"rebuild: recovering registered pinned worktree {target}", file=sys.stderr)
        subprocess.run(["git", "-C", str(repo), "worktree", "unlock", str(target)],
                       text=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", "--force",
                        str(target)], text=True, capture_output=True)
    if target.exists():
        print(f"rebuild: removing half-made pinned worktree dir {target}", file=sys.stderr)
        shutil.rmtree(target)
    if registered or "prunable" in listing:
        print("rebuild: pruning stale worktree entries", file=sys.stderr)
        _git(repo, "worktree", "prune")


def _pinned_repo(repo: Path, work: Path, git_sha: str) -> Path:
    _git(repo, "cat-file", "-e", f"{git_sha}^{{commit}}")
    if _git(repo, "rev-parse", "HEAD") == git_sha:
        return repo
    target = work / f"repo-{git_sha[:12]}"
    if _reusable_worktree(target, git_sha):
        return target
    _clear_interrupted_worktree(repo, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    run = subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "--detach", str(target), git_sha],
        text=True, capture_output=True)
    if run.returncode:
        raise ContractError(f"cannot create pinned worktree: {run.stderr.strip()}")
    return target


def _commands(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return manifest["rebuild"]["commands"]


def _require_workdir(work: Path, *, resume: bool) -> None:
    if work.exists() and any(work.iterdir()) and not resume:
        raise ContractError(
            f"work directory is not empty: {work}; choose an empty directory "
            "or pass --resume explicitly")


def _expand(argv: list[str], values: dict[str, str]) -> list[str]:
    out = []
    for arg in argv:
        try:
            rendered = arg.format_map(values)
        except KeyError as exc:
            raise ContractError(f"unknown rebuild placeholder {exc.args[0]!r}") from exc
        if "{" in rendered or "}" in rendered:
            raise ContractError(f"unresolved rebuild placeholder in {arg!r}")
        out.append(rendered)
    return out


def _status(manifest: dict[str, Any]) -> int:
    status = manifest.get("status", "unknown")
    print(f"status={status} collection={manifest.get('collection', {}).get('name', 'unknown')}")
    gaps = manifest.get("blocking_gaps") or []
    for gap in gaps:
        if isinstance(gap, dict):
            print(f"BLOCKED {gap.get('id', '?')}: {gap.get('detail', gap)}")
        else:
            print(f"BLOCKED: {gap}")
    return 0 if status == "sealed" else 1


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__)
    ap.add_argument("command", choices=("status", "plan", "run"))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--registry")
    ap.add_argument("--artifact-root")
    ap.add_argument("--work")
    ap.add_argument("--repo", default=str(REPO_ROOT))
    ap.add_argument("--phase", action="append", default=[],
                    help="run/print only this phase (repeatable); dependencies are caller-owned")
    ap.add_argument("--resume", action="store_true",
                    help="allow an existing non-empty work directory (run only)")
    args = ap.parse_args(argv)

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = _read_manifest(manifest_path)
    if args.command == "status":
        return _status(manifest)
    if manifest.get("status") != "sealed":
        return _status(manifest)
    if not args.artifact_root or not args.work:
        ap.error("plan/run require --artifact-root and --work")
    try:
        proof = verify_manifest(
            manifest_path, registry_path=args.registry,
            artifact_root=args.artifact_root, deep=True,
            expect_collection=manifest.get("collection", {}).get("name"))
        repo = Path(args.repo).expanduser().resolve()
        work = Path(args.work).expanduser().resolve()
        if args.command == "run":
            _require_workdir(work, resume=args.resume)
            pinned = _pinned_repo(repo, work, proof["git_sha"])
        else:
            # Planning is read-only: prove the commit exists but do not create a
            # detached worktree merely to render argv.
            _git(repo, "cat-file", "-e", f"{proof['git_sha']}^{{commit}}")
            pinned = repo if _git(repo, "rev-parse", "HEAD") == proof["git_sha"] \
                else work / f"repo-{proof['git_sha'][:12]}"
        values = {
            "repo": str(pinned),
            "artifact_root": str(Path(args.artifact_root).expanduser().resolve()),
            "work": str(work),
            "manifest": str(manifest_path),
            "collection": proof["collection"],
            "release_id": proof["release_id"],
        }
        selected = [row for row in _commands(manifest)
                    if not args.phase or row["phase"] in args.phase]
        unknown = sorted(set(args.phase) - {row["phase"] for row in selected})
        if unknown:
            raise ContractError(f"unknown rebuild phases: {', '.join(unknown)}")
        rendered = [(row["phase"], _expand(row["argv"], values)) for row in selected]
        print(json.dumps({"contract": proof, "pinned_repo": str(pinned),
                          "commands": rendered}, indent=2, sort_keys=True))
        if args.command == "plan":
            return 0

        work.mkdir(parents=True, exist_ok=True)
        ledger = work / "rebuild-run.jsonl"
        _append(ledger, {"status": "START", "contract": proof,
                         "pinned_repo": str(pinned)})
        env = dict(os.environ, AW_ARTIFACT_ROOT=values["artifact_root"],
                   AW_PRESERVATION_MANIFEST=str(manifest_path),
                   AW_PRESERVATION_COLLECTION=proof["collection"],
                   AW_PRESERVATION_DEEP="1")
        for phase, command in rendered:
            _append(ledger, {"status": "PHASE-START", "phase": phase,
                             "argv": command})
            run = subprocess.run(command, cwd=pinned, env=env)
            if run.returncode:
                _append(ledger, {"status": "FAILED", "phase": phase,
                                 "exit_code": run.returncode})
                return run.returncode
            _append(ledger, {"status": "PHASE-PASS", "phase": phase})
        _append(ledger, {"status": "COMPLETE", "phases": len(rendered)})
        return 0
    except ContractError as exc:
        print(f"REBUILD REFUSED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
