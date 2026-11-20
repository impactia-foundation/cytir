#!/usr/bin/env python3
"""Acquire repository-native documentation with a preserved L0 -> L1 chain.

This adapter repairs the Task 288 Route-B failure without reproducing its
passage-only design.  For every input repository it:

* resolves and pins the requested branch to an immutable commit;
* preserves the exact GitHub API responses and selected source blobs;
* emits deterministic, parent-digest-bound Markdown;
* emits a receipted outlink inventory, including an empty inventory; and
* writes an atomic terminal receipt before marking the repository complete.

The output root is append/resume safe.  A terminal receipt is accepted on a
rerun only after all referenced artifacts have been revalidated.  Failed
repositories remain retryable and cannot make the run manifest complete.
No chunks or vectors are produced here: those stages consume the sealed
Markdown after their policies are pinned by the release manifest.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any, Iterable
from urllib.parse import quote

import httpx

from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client


from cytir.acquire.preserved_stage_io import (artifact_path,  # noqa: F401  # re-exported: read as GH.<name> by test_task517_context7_titles.py:60 and test_aw_github_repo_docs_acquire.py:26
                                           atomic_write, sha256_bytes, sha256_file,
                                           store_artifact)

from cytir.extract.github_agent_resources import (  # noqa: F401 -- task 1115: moved unchanged; re-exported for the importers  # re-exported: read as GH.<name> by test_task517_context7_titles.py:60 and test_aw_github_repo_docs_acquire.py:26
    SCHEMA, DOC_EXTENSIONS, DOC_DIR, SKIP,
    SHA256_RE, GIT_SHA_RE, MD_LINK_RE, AUTOLINK_RE,
    AcquisitionError, validate_artifact, utc_now, normalize_owner_repo,
    receipt_name, select_documents, source_provenance, source_to_markdown,
    compose_document_title, extract_outlinks, quarantine_receipt, validate_receipt,
    load_terminal)
canonical_repo = normalize_owner_repo  # task 1121: the package def was renamed; the `canon*` name stays on this exempt driver only
from cytir.acquire.github_docs_tree import (CHANGELOG_FILE, PERMISSIVE, ROOT_DOC,  # noqa: F401 -- task 986: shared with cyt ops  # re-exported: read as GH.<name> by test_task517_context7_titles.py:60 and test_aw_github_repo_docs_acquire.py:26
                                         TREE_DEFAULT_MAX_FILE_BYTES, TREE_DEFAULT_MAX_FILES,
                                         TREE_DIR, TREE_EXTENSIONS, TREE_SKIP_DIRS, TREE_SKIP_FILES,
                                         select_docs_tree)

RUN_SCHEMA = "aw.github-doc-acquisition-run.v1"


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )


def iter_queue(path: Path) -> Iterable[dict[str, Any]]:
    seen: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AcquisitionError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict) or not row.get("owner_repo"):
                raise AcquisitionError(f"{path}:{line_number}: missing owner_repo")
            row = dict(row)
            row["owner_repo"] = canonical_repo(str(row["owner_repo"]))
            identity = row["owner_repo"].casefold()
            if identity in seen:
                raise AcquisitionError(f"{path}:{line_number}: duplicate {identity}")
            seen.add(identity)
            yield row


@dataclass(frozen=True)
class Settings:
    root: Path
    max_files: int
    retries: int
    timeout: float
    tree: str = "capped"
    max_file_bytes: int = TREE_DEFAULT_MAX_FILE_BYTES
    include_changelog: bool = False


class GitHubClient:
    def __init__(self, token: str, settings: Settings):
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.client = make_client(HttpSpec(
            timeout=settings.timeout, follow_redirects=True, headers=headers))
        self.settings = settings
        self.rate_lock = threading.Lock()
        self.not_before = 0.0

    def close(self) -> None:
        self.client.close()

    def get(self, url: str) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(self.settings.retries + 1):
            with self.rate_lock:
                delay = self.not_before - time.time()
            if delay > 0:
                time.sleep(delay)
            try:
                response = self.client.get(url)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError(
                        f"retryable status {response.status_code}",
                        request=response.request, response=response)
                if response.status_code == 403 and response.headers.get("x-ratelimit-remaining") == "0":
                    reset = float(response.headers.get("x-ratelimit-reset", time.time() + 60))
                    with self.rate_lock:
                        self.not_before = max(self.not_before, reset + 2)
                    raise httpx.HTTPStatusError(
                        "GitHub API rate limit exhausted", request=response.request,
                        response=response)
                return response
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last = exc
                if attempt < self.settings.retries:
                    time.sleep(min(30.0, 2.0 ** attempt))
        raise AcquisitionError(f"request failed after retries: {url}: {last}")


def api_evidence(response: httpx.Response) -> bytes:
    record = {
        "url": str(response.request.url),
        "status_code": response.status_code,
        "headers": {
            key.lower(): value for key, value in response.headers.items()
            if key.lower() in {"content-type", "etag", "last-modified", "x-github-request-id"}
        },
        "body_b64": __import__("base64").b64encode(response.content).decode("ascii"),
    }
    return (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()


def decode_git_blob(response: httpx.Response, expected_sha1: str) -> bytes:
    """Decode a documented GitHub Git-blob response and bind it to tree SHA-1."""
    if response.status_code != 200:
        raise AcquisitionError(f"Git blob API returned {response.status_code}")
    try:
        value = response.json()
        if value.get("encoding") != "base64":
            raise ValueError("unsupported Git blob encoding")
        raw = base64.b64decode(str(value["content"]), validate=False)
    except (KeyError, TypeError, ValueError) as exc:
        raise AcquisitionError("Git blob API response is malformed") from exc
    git_sha1 = hashlib.sha1(
        f"blob {len(raw)}\0".encode() + raw, usedforsecurity=False
    ).hexdigest()
    if git_sha1 != expected_sha1.casefold():
        raise AcquisitionError(
            f"Git blob SHA-1 mismatch: {git_sha1} != {expected_sha1}")
    return raw


def acquire_repository(client: GitHubClient, record: dict[str, Any],
                       settings: Settings) -> dict[str, Any]:
    owner_repo = record["owner_repo"]
    branch = str(record.get("branch") or "main")
    encoded_repo = "/".join(quote(part, safe="") for part in owner_repo.split("/"))
    commit_response = client.get(
        f"https://api.github.com/repos/{encoded_repo}/commits/{quote(branch, safe='')}")
    commit_artifact = store_artifact(settings.root, "original/api", "json", api_evidence(commit_response))
    if commit_response.status_code != 200:
        metadata_response = client.get(f"https://api.github.com/repos/{encoded_repo}")
        metadata_artifact = store_artifact(
            settings.root, "original/api", "json", api_evidence(metadata_response))
        if metadata_response.status_code == 200:
            default_branch = str(metadata_response.json().get("default_branch") or "")
            if default_branch and default_branch != branch:
                branch = default_branch
                commit_response = client.get(
                    f"https://api.github.com/repos/{encoded_repo}/commits/"
                    f"{quote(branch, safe='')}")
                commit_artifact = store_artifact(
                    settings.root, "original/api", "json", api_evidence(commit_response))
        if commit_response.status_code != 200:
            reason = (
                "repository_unavailable" if metadata_response.status_code == 404
                else "branch_or_commit_unavailable"
            )
            return quarantine_receipt(
                record, branch, reason,
                [commit_artifact, metadata_artifact],
                f"commit={commit_response.status_code}; metadata={metadata_response.status_code}",
            )
    commit = commit_response.json()
    revision = str(commit.get("sha", "")).lower()
    if not GIT_SHA_RE.fullmatch(revision):
        raise AcquisitionError(f"{owner_repo}: commit response lacks a full revision")

    tree_response = client.get(
        f"https://api.github.com/repos/{encoded_repo}/git/trees/{revision}?recursive=1")
    tree_artifact = store_artifact(settings.root, "original/api", "json", api_evidence(tree_response))
    if tree_response.status_code != 200:
        raise AcquisitionError(f"{owner_repo}: tree returned {tree_response.status_code}")
    tree = tree_response.json()
    tree_truncated = tree.get("truncated") is True
    nodes = tree.get("tree")
    if not isinstance(nodes, list):
        raise AcquisitionError(f"{owner_repo}: tree response lacks nodes")

    license_id = str(record.get("license") or "NOASSERTION").upper()
    permissive = license_id in PERMISSIVE
    if tree_truncated and permissive:
        raise AcquisitionError(
            f"{owner_repo}: recursive tree was truncated for deep posture")
    if settings.tree == "docs":
        selected = select_docs_tree(nodes, permissive, settings.max_files, settings.max_file_bytes,
                                    settings.include_changelog, record.get("exclude_regex"))
    else:
        selected = select_documents(nodes, permissive, settings.max_files,
                                     record.get("exclude_regex"))
    if not selected:
        return quarantine_receipt(
            record, branch, "no_eligible_documentation",
            [commit_artifact, tree_artifact], "tree contains no admissible documentation blob",
            revision=revision,
        )

    documents: list[dict[str, Any]] = []
    skipped_empty: list[str] = []
    outlink_count = 0
    link_stats: dict[str, int] = {}
    for node in selected:
        source_path = str(node["path"])
        if node.get("size") == 0:
            # Task 1003: the pinned Git tree itself says this file is 0 bytes (a
            # placeholder). That is a file property, not a fetch error: skip, record.
            skipped_empty.append(source_path)
            continue
        source_url = (
            f"https://github.com/{owner_repo}/blob/{revision}/"
            + quote(source_path, safe="/")
        )
        raw_url = (
            f"https://raw.githubusercontent.com/{owner_repo}/{revision}/"
            + quote(source_path, safe="/")
        )
        response = client.get(raw_url)
        blob_api_artifact = None
        if response.status_code == 200:
            raw = response.content
            media_type = response.headers.get(
                "content-type", "application/octet-stream")
        else:
            # raw.githubusercontent can return a persistent 404 for a blob that
            # is present in the immutable Git tree. Fall back to GitHub's
            # documented Git-blob endpoint, preserve that exact API response,
            # and verify the decoded bytes against the tree's Git SHA-1.
            blob_sha1 = str(node.get("sha", "")).casefold()
            blob_response = client.get(
                f"https://api.github.com/repos/{encoded_repo}/git/blobs/{blob_sha1}")
            blob_api_artifact = store_artifact(
                settings.root, "original/api", "json", api_evidence(blob_response))
            raw = decode_git_blob(blob_response, blob_sha1)
            media_type = "application/vnd.github+json; decoded-git-blob"
        expected_size = node.get("size")
        if not raw or (isinstance(expected_size, int) and len(raw) != expected_size):
            # A tree entry that is not 0 bytes but fetched empty or at another size is a
            # fetch error: keep the repo-level refusal, with path and expected size.
            held = store_artifact(settings.root, "original/blobs", "raw", raw)
            return quarantine_receipt(
                record, branch,
                "selected_document_empty" if not raw else "selected_document_size_mismatch",
                [commit_artifact, tree_artifact, *(
                    [blob_api_artifact] if blob_api_artifact else []), held],
                f"{source_path}: fetched {len(raw)} bytes, tree size {expected_size}",
                revision=revision,
            )
        blob_sha1 = str(node.get("sha", "")).lower()
        original = store_artifact(settings.root, "original/blobs", "raw", raw)
        try:
            markdown_bytes = source_to_markdown(raw, source_path=source_path)
            markdown = store_artifact(settings.root, "markdown", "md", markdown_bytes)
            outlinks = extract_outlinks(
                markdown_bytes.decode("utf-8"), source_url, link_stats)
            outlink_count += len(outlinks)
            outlink_payload = (
                json.dumps({
                    "source_url": source_url,
                    "repository": owner_repo,
                    "revision": revision,
                    "source_path": source_path,
                    "retrieved_at": utc_now(),
                    "links": outlinks,
                }, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode()
            outlink_artifact = store_artifact(settings.root, "outlinks", "json", outlink_payload)
            document = {
                "source_path": source_path,
                "source_url": source_url,
                "git_blob_sha1": blob_sha1,
                "media_type": media_type,
                "original": original,
                "markdown": markdown,
                "outlinks": {**outlink_artifact, "records": len(outlinks)},
                # The L0 binding that used to be the document's first six lines
                # (task 517), and the title a reader and the ranker see.
                "provenance": source_provenance(
                    owner_repo=owner_repo, revision=revision,
                    source_path=source_path, parent_sha256=original["body_sha256"]),
                "title": compose_document_title(
                    markdown_bytes.decode("utf-8"), source_path),
            }
            if blob_api_artifact:
                document["blob_api_response"] = blob_api_artifact
            documents.append(document)
        except (AcquisitionError, httpx.HTTPError):
            raise
        except Exception as exc:
            # Task 1012: an unexpected exception while converting one selected file is
            # a file property, not a transport/auth failure (those raise above, in
            # client.get). Quarantine the repo with the file path; the run goes on.
            return quarantine_receipt(
                record, branch, f"exception: {type(exc).__name__}: {exc}",
                [commit_artifact, tree_artifact, original],
                f"{source_path}: {type(exc).__name__}: {exc}", revision=revision)

    if not documents:
        return quarantine_receipt(
            record, branch, "no_docs_content",
            [commit_artifact, tree_artifact],
            f"every selected file is 0 bytes: {skipped_empty}", revision=revision,
        )
    acquired_at = utc_now()
    return {
        "schema": SCHEMA,
        "status": "complete",
        "skipped_empty": skipped_empty,
        "unparsable_links": link_stats.get("unparsable_links", 0),
        "acquired_at": acquired_at,
        "source_profile": "software_docs",
        "repository": owner_repo,
        "requested_branch": branch,
        "revision": revision,
        "license": license_id,
        "posture": "deep" if permissive else "pointer",
        "route": "documented_source_api",
        "route_receipt": {
            "higher_priority_routes_checked": [
                "source_owned_incremental_feed:not_applicable",
                "source_owned_bulk_dump:not_selected_repository_tree_is_canonical",
            ],
            "reason_selected_route_is_needed": "pin repository tree and exact source blobs",
            "consent_and_robots_result": "GitHub documented API/raw content route",
            "fetch_vantage": os.environ.get("AW_FETCH_VANTAGE") or socket.gethostname(),
        },
        "tree": {
            "commit_response": commit_artifact,
            "tree_response": tree_artifact,
            "node_count": len(nodes),
            "truncated": tree_truncated,
        },
        "counts": {
            "original_documents": len(documents),
            "markdown_documents": len(documents),
            "outlink_documents": len(documents),
            "outlinks": outlink_count,
        },
        "documents": documents,
        "downstream": {"chunks": "pending", "vectors": "pending"},
    }


def run(args: argparse.Namespace) -> int:
    queue = list(iter_queue(args.queue))
    if args.max_repos is not None:
        queue = queue[:args.max_repos]
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    max_files = args.max_files if args.max_files is not None else (
        TREE_DEFAULT_MAX_FILES if args.tree == "docs" else 6)
    settings = Settings(root=root, max_files=max_files, retries=args.retries, timeout=args.timeout,
                        tree=args.tree, max_file_bytes=args.max_file_bytes,
                        include_changelog=args.include_changelog)
    script_path = Path(__file__).resolve()
    run_state = {
        "schema": RUN_SCHEMA,
        "status": "running",
        "started_at": utc_now(),
        "pid": os.getpid(),
        "queue": {"path": str(args.queue), "sha256": sha256_file(args.queue),
                  "repositories": len(queue)},
        "producer": {"path": "aw/sources/github_repo_docs/acquire.py",
                     "sha256": sha256_file(script_path),
                     "git_revision": args.git_revision},
        "parameters": {"max_files": max_files, "tree": args.tree, "workers": args.workers,
                       "retries": args.retries, "timeout": args.timeout},
    }
    atomic_json(root / "RUN.json", run_state)

    already: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for record in queue:
        terminal = load_terminal(root, record["owner_repo"])
        if terminal is None:
            pending.append(record)
        else:
            already[record["owner_repo"].casefold()] = terminal

    token = os.environ.get("GH_TOKEN", "").strip()
    if not token and len(pending) > 50:
        raise AcquisitionError(
            "GH_TOKEN is required for a queue larger than 50 repositories")
    client = GitHubClient(token, settings)
    completed = sum(1 for x in already.values() if x["status"] == "complete")
    quarantined = sum(1 for x in already.values() if x["status"] == "quarantined")
    failed = 0
    originals = sum(x["counts"]["original_documents"] for x in already.values())
    outlinks = sum(x["counts"]["outlinks"] for x in already.values())
    ledger = root / "repo-ledger.jsonl"
    errors = root / "errors"
    errors.mkdir(exist_ok=True)
    try:
        with ledger.open("a", encoding="utf-8") as log, ThreadPoolExecutor(
                max_workers=args.workers) as executor:
            futures = {
                executor.submit(acquire_repository, client, record, settings): record
                for record in pending
            }
            for index, future in enumerate(as_completed(futures), 1):
                record = futures[future]
                owner_repo = record["owner_repo"]
                try:
                    receipt = future.result()
                    validate_receipt(root, receipt, owner_repo)
                    receipt_path = root / "receipts" / receipt_name(owner_repo)
                    atomic_json(receipt_path, receipt)
                    if receipt["status"] == "complete":
                        completed += 1
                    else:
                        quarantined += 1
                    originals += receipt["counts"]["original_documents"]
                    outlinks += receipt["counts"]["outlinks"]
                    event = {"at": utc_now(), "repository": owner_repo,
                             "status": receipt["status"], "receipt": str(receipt_path.relative_to(root)),
                             "documents": receipt["counts"]["original_documents"],
                             "outlinks": receipt["counts"]["outlinks"]}
                except Exception as exc:  # each failure stays explicit and retryable
                    failed += 1
                    event = {"at": utc_now(), "repository": owner_repo,
                             "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                    atomic_json(errors / receipt_name(owner_repo), event)
                log.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                log.flush()
                os.fsync(log.fileno())
                if index % args.progress_every == 0 or index == len(pending):
                    print(
                        f"PROGRESS terminal={completed + quarantined}/{len(queue)} "
                        f"complete={completed} quarantined={quarantined} "
                        f"attempt_failed={failed} originals={originals} outlinks={outlinks}",
                        flush=True,
                    )
    finally:
        client.close()

    remaining = sum(1 for record in queue if load_terminal(root, record["owner_repo"]) is None)
    run_state.update({
        "status": "complete" if remaining == 0 else "partial",
        "finished_at": utc_now(),
        "counts": {"queue_repositories": len(queue), "terminal_repositories": len(queue) - remaining,
                   "remaining_repositories": remaining, "attempt_failures": failed,
                   "complete_repositories": completed,
                   "quarantined_repositories": quarantined,
                   "original_documents": originals, "markdown_documents": originals,
                   "outlink_documents": originals, "outlinks": outlinks},
    })
    atomic_json(root / "RUN.json", run_state)
    print(json.dumps(run_state, indent=2, sort_keys=True), flush=True)
    return 0 if remaining == 0 else 2


def status(args: argparse.Namespace) -> int:
    queue = list(iter_queue(args.queue))
    terminal = complete = quarantined = originals = outlinks = 0
    for record in queue:
        receipt = load_terminal(args.root.resolve(), record["owner_repo"])
        if receipt:
            terminal += 1
            complete += receipt["status"] == "complete"
            quarantined += receipt["status"] == "quarantined"
            originals += receipt["counts"]["original_documents"]
            outlinks += receipt["counts"]["outlinks"]
    value = {"queue_repositories": len(queue), "terminal_repositories": terminal,
             "remaining_repositories": len(queue) - terminal,
             "complete_repositories": complete,
             "quarantined_repositories": quarantined,
             "original_documents": originals, "markdown_documents": originals,
             "outlink_documents": originals, "outlinks": outlinks}
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--queue", type=Path, required=True)
    run_parser.add_argument("--root", type=Path, required=True)
    run_parser.add_argument("--workers", type=int, default=4)
    run_parser.add_argument("--max-files", type=int, default=None,
                            help="per-repo file cap; default 6 (capped mode) or 400 (--tree docs)")
    run_parser.add_argument("--tree", choices=("capped", "docs"), default="capped",
                            help="capped = the pointer/deep selection (<= 6 files); docs = the full docs tree")
    run_parser.add_argument("--max-file-bytes", type=int, default=TREE_DEFAULT_MAX_FILE_BYTES)
    run_parser.add_argument("--include-changelog", action="store_true")
    run_parser.add_argument("--retries", type=int, default=4)
    run_parser.add_argument("--timeout", type=float, default=60.0)
    run_parser.add_argument("--progress-every", type=int, default=25)
    run_parser.add_argument("--max-repos", type=int)
    run_parser.add_argument("--git-revision", default=os.environ.get("AW_GIT_SHA"))
    run_parser.set_defaults(func=run)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--queue", type=Path, required=True)
    status_parser.add_argument("--root", type=Path, required=True)
    status_parser.set_defaults(func=status)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        raise SystemExit(args.func(args))
    except AcquisitionError as exc:
        print(f"FATAL: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
