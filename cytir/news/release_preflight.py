#!/usr/bin/env python3
"""Immutable-release and language-gate preflight for the news scheduler.

The scheduler runs code on two machines.  A matching ``GIT_SHA`` alone cannot
prove that the release is complete (2026-08-03 omitted ``enrich_tier1.py``),
and the normal application venv cannot safely run fastText (numpy 2 disables
the content language gate).  This is deliberately a small, dependency-free
tool so it can run *before* any ingest work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from cytir.news.source_queue import host_meta_relpath, queue_relpath


SCHEMA = "aw.news-immutable-release.v1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
REQUIRED_RELEASE_FILES = frozenset({
    "deploy/scripts/ensure-staging-qdrant.sh",
    "GIT_SHA",
    "cytir/consent/gate.py",
    "cytir/consent/policy.py",
    "cytir/news/dump_points.py",
    "cytir/news/enrich_tier1.py",
    "cytir/acquire/fetch_process.py",
    "cytir/acquire/discovery_ingest.py",
    "cytir/acquire/preserved_page_fetch.py",
    "cytir/acquire/preserved_stage_io.py",
    "cytir/sources/hn/freshness_ingest.py",
    host_meta_relpath(),
    queue_relpath(),
    "cytir/news/recapture_skip.py",
    "deploy/scripts/news_refresh_cron.sh",
    "deploy/scripts/news_refresh_delta.sh",
    "deploy/scripts/news_refresh_stage.sh",
    "cytir/news/release_preflight.py",
    "cytir/news/routine_apply.py",
    "cytir/news/stage_contract.py",
    "cytir/index/news_window_filter.py",
    "cytir/index/dump_netnew.py",
    "deploy/langgate/requirements.txt",
    "cytir/acquire/sitemap_ingest.py",
    "cytir/index/stage_pack_b1.py",
})


class PreflightError(RuntimeError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_files(root: Path) -> dict[str, str]:
    """The complete regular-file closure, never a hand-picked dependency list."""
    return {
        path.relative_to(root).as_posix(): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
        # Bytecode caches are derived by running the release, not part of it; counting
        # them made the 2026-09-24 daily news run refuse on its own __pycache__.
        and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def write_manifest(root: Path, git_sha: str, out: Path, *,
                   skipped_links: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """`skipped_links` (path + target of each link member the cut did not extract) is recorded
    only when non-empty, so a release without such links keeps its digest."""
    if not GIT_SHA.fullmatch(git_sha):
        raise PreflightError("git SHA must be 40 lowercase hex characters")
    root = root.resolve()
    out = out.resolve()
    if out.is_relative_to(root):
        raise PreflightError("release manifest must live outside its release root")
    value = {
        "schema": SCHEMA,
        "git_sha": git_sha,
        "files": release_files(root),
    }
    if skipped_links:
        value["skipped_links"] = skipped_links
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


def load_and_validate_release(root: Path, manifest_path: Path,
                              expected_git_sha: str,
                              expected_manifest_sha256: str) -> dict[str, Any]:
    if not GIT_SHA.fullmatch(expected_git_sha):
        raise PreflightError("expected release SHA is not a 40-character lowercase git SHA")
    if not SHA256.fullmatch(expected_manifest_sha256):
        raise PreflightError("expected release-manifest SHA-256 is malformed")
    if not root.is_dir() or root.is_symlink():
        raise PreflightError(f"release root is absent or not a real directory: {root}")
    if not manifest_path.is_file() or digest(manifest_path) != expected_manifest_sha256:
        raise PreflightError("release manifest absent or SHA-256 mismatch")
    try:
        value = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"release manifest is unreadable: {exc}") from exc
    if value.get("schema") != SCHEMA or value.get("git_sha") != expected_git_sha:
        raise PreflightError("release manifest schema or git SHA mismatch")
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise PreflightError("release manifest has no complete file closure")
    if any(not isinstance(k, str) or not SHA256.fullmatch(str(v)) for k, v in files.items()):
        raise PreflightError("release manifest has malformed file hashes")
    missing_runtime = sorted(REQUIRED_RELEASE_FILES - set(files))
    if missing_runtime:
        raise PreflightError(
            f"release omits required news runtime files: {missing_runtime[:3]}"
        )
    actual = release_files(root.resolve())
    if files != actual:
        missing = sorted(set(files) - set(actual))
        extra = sorted(set(actual) - set(files))
        changed = sorted(k for k in set(files) & set(actual) if files[k] != actual[k])
        raise PreflightError(
            "release closure mismatch "
            f"missing={missing[:3]} extra={extra[:3]} changed={changed[:3]}")
    git_file = root / "GIT_SHA"
    if not git_file.is_file() or git_file.read_text().strip() != expected_git_sha:
        raise PreflightError("release GIT_SHA file absent or mismatched")
    return value


def validate_langgate(release_root: Path, python: Path,
                      expected_requirements_sha256: str) -> dict[str, Any]:
    if not SHA256.fullmatch(expected_requirements_sha256):
        raise PreflightError("expected langgate requirements SHA-256 is malformed")
    requirements = release_root / "deploy/langgate/requirements.txt"
    if not requirements.is_file() or digest(requirements) != expected_requirements_sha256:
        raise PreflightError("langgate requirements lock absent or SHA-256 mismatch")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise PreflightError(f"pinned langgate interpreter is absent/non-executable: {python}")
    probe = r'''
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root))   # the release's own `aw` (the retired pipeline shim is gone)
import numpy
if int(numpy.__version__.split(".", 1)[0]) >= 2:
    raise SystemExit("numpy>=2 in langgate interpreter: " + numpy.__version__)
import fasttext  # noqa: F401 -- make an absent dependency explicit
from cytir.acquire import fetch_process as fp
status = fp.assert_langgate_active("news scheduler preflight")
if not status.get("active"):
    raise SystemExit("langgate did not report active")
print(json.dumps({"python": sys.executable, "numpy": numpy.__version__,
                  "lang_gate": status["active"]}, sort_keys=True))
'''
    result = subprocess.run(
        [str(python), "-c", probe, str(release_root)], text=True,
        capture_output=True, check=False,
        env={**os.environ, "AW_LANGGATE_OPTIONAL": "", "PYTHONDONTWRITEBYTECODE": "1"})
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().replace("\n", " ")
        raise PreflightError(f"pinned langgate interpreter failed active probe: {detail}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise PreflightError("langgate probe emitted invalid evidence") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("write-manifest")
    make.add_argument("--release-dir", type=Path, required=True)
    make.add_argument("--git-sha", required=True)
    make.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("check")
    check.add_argument("--release-dir", type=Path, required=True)
    check.add_argument("--release-manifest", type=Path, required=True)
    check.add_argument("--expected-git-sha", required=True)
    check.add_argument("--expected-release-manifest-sha256", required=True)
    check.add_argument("--langgate-python", type=Path)
    check.add_argument("--expected-langgate-requirements-sha256")
    check.add_argument("--skip-langgate", action="store_true",
                       help="closure-only check for the non-extraction orchestrator host")
    args = parser.parse_args()
    try:
        if args.command == "write-manifest":
            value = write_manifest(args.release_dir, args.git_sha, args.out)
            print(json.dumps({"schema": value["schema"], "files": len(value["files"]),
                              "manifest_sha256": digest(args.out)}, sort_keys=True))
        else:
            release = load_and_validate_release(
                args.release_dir, args.release_manifest, args.expected_git_sha,
                args.expected_release_manifest_sha256)
            if args.skip_langgate:
                if args.langgate_python or args.expected_langgate_requirements_sha256:
                    raise PreflightError("--skip-langgate cannot be combined with langgate inputs")
                gate: dict[str, Any] = {"checked": False,
                                         "reason": "orchestrator host has no extraction venv"}
            else:
                if not args.langgate_python or not args.expected_langgate_requirements_sha256:
                    raise PreflightError("langgate interpreter and requirements SHA are required")
                gate = validate_langgate(args.release_dir, args.langgate_python,
                                         args.expected_langgate_requirements_sha256)
            print(json.dumps({"schema": "aw.news-preflight.v1", "git_sha": release["git_sha"],
                              "release_file_count": len(release["files"]), "langgate": gate},
                             sort_keys=True))
    except PreflightError as exc:
        raise SystemExit(f"FATAL news scheduler preflight: {exc}") from exc


if __name__ == "__main__":
    main()
