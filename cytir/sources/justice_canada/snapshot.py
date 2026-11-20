"""Seal one pinned official Justice Canada bilingual-law source snapshot.

Moved from the flat pipeline script by task 1402 (S7); the old path is a
forwarding shim. The family adapter `cytir.sources.families.justice_canada` calls `validate` (no
write) in `acquire` and `seal` in `seal_snapshot`. The code is the old driver's, not rewritten:
`seal` is `validate` plus the stage receipt and `RUN.json`, byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire import stage_receipt as SS


VERSION = "justice-canada-laws-snapshot.v3-exhaustive-attempt-evidence"
COMMIT = re.compile(r"^[0-9a-f]{40}$")
REPO = "justicecanada/laws-lois-xml"


def validate(*, root: Path, artifact_root: Path, archive: Path, headers: Path,
             curl_metrics: Path, transport: Path, ls_remote: Path,
             ls_remote_stderr: Path, ls_remote_started: Path,
             ls_remote_completed: Path, ls_remote_raw: Path,
             ls_remote_rc: Path, fetch_started: Path, fetch_completed: Path,
             fetch_rc: Path, curl_stderr: Path, commit: str,
             requested_url: str) -> dict:
    """Check every piece of evidence; write nothing. Returns what `seal` records."""
    root = root.resolve()
    artifact_root = artifact_root.resolve()
    paths = [archive, headers, curl_metrics, transport, ls_remote,
             ls_remote_stderr, ls_remote_started, ls_remote_completed,
             ls_remote_raw, ls_remote_rc, fetch_started, fetch_completed,
             fetch_rc, curl_stderr]
    paths = [path.resolve() for path in paths]
    for path in paths + [root]:
        path.relative_to(artifact_root)
    if not COMMIT.fullmatch(commit):
        raise ValueError("invalid pinned Git commit")
    expected_url = (
        f"https://codeload.github.com/{REPO}/tar.gz/{commit}"
    )
    if requested_url != expected_url:
        raise ValueError("snapshot URL is not pinned to the exact commit")
    if ls_remote.read_text(encoding="utf-8") != f"{commit}\tHEAD\n":
        raise ValueError("git ls-remote HEAD evidence disagrees with pinned commit")
    if ls_remote_raw.read_text(encoding="utf-8") != f"{commit}\tHEAD\n":
        raise ValueError("raw git ls-remote evidence disagrees with pinned commit")
    if ls_remote_rc.read_text(encoding="utf-8") != "0\n":
        raise ValueError("git ls-remote did not complete successfully")
    if fetch_rc.read_text(encoding="utf-8") != "0\n":
        raise ValueError("codeload fetch did not complete successfully")
    for timestamp in (ls_remote_started, ls_remote_completed,
                      fetch_started, fetch_completed):
        if not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\n?",
            timestamp.read_text(encoding="utf-8"),
        ):
            raise ValueError("invalid git remote observation timestamp")

    fetch = json.loads(transport.read_text(encoding="utf-8"))
    if (
        fetch.get("schema") != "aw.http-fetch-metadata.v1"
        or fetch.get("requested_url") != requested_url
        or fetch.get("http_status") != 200
        or not str(fetch.get("final_url") or "").startswith(
            "https://codeload.github.com/"
        )
    ):
        raise ValueError("invalid codeload HTTP transport evidence")
    for key, path in (("body", archive), ("response_headers", headers)):
        artifact = fetch.get(key) or {}
        if (
            artifact.get("bytes") != path.stat().st_size
            or artifact.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"{key} disagrees with transport metadata")
    metrics = json.loads(curl_metrics.read_text(encoding="utf-8"))
    if (
        int(metrics.get("http_code") or metrics.get("response_code") or 0) != 200
        or metrics.get("url_effective") != fetch["final_url"]
    ):
        raise ValueError("curl metrics disagree with canonical transport metadata")

    members = xml_members = 0
    languages: set[str] = set()
    expected_root = f"laws-lois-xml-{commit}"
    expected_prefix = f"{expected_root}/"
    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar:
            if (
                member.name.startswith("/")
                or ".." in Path(member.name).parts
                or (
                    member.name != expected_root
                    and not member.name.startswith(expected_prefix)
                )
            ):
                raise ValueError(f"unsafe or wrong-commit archive member: {member.name!r}")
            members += 1
            if member.isfile() and member.name.lower().endswith(".xml"):
                xml_members += 1
                lowered = member.name.lower()
                if "/eng/" in lowered or "/en/" in lowered:
                    languages.add("en")
                if "/fra/" in lowered or "/fr/" in lowered:
                    languages.add("fr")
    if members == 0 or xml_members == 0:
        raise ValueError("Justice Canada snapshot contains no XML source files")

    return {"paths": paths, "members": members, "xml_members": xml_members,
            "languages": languages}


def seal(*, root: Path, artifact_root: Path, archive: Path, headers: Path,
         curl_metrics: Path, transport: Path, ls_remote: Path,
         ls_remote_stderr: Path, ls_remote_started: Path,
         ls_remote_completed: Path, ls_remote_raw: Path,
         ls_remote_rc: Path, fetch_started: Path, fetch_completed: Path,
         fetch_rc: Path, curl_stderr: Path, commit: str,
         requested_url: str) -> dict:
    found = validate(
        root=root, artifact_root=artifact_root, archive=archive, headers=headers,
        curl_metrics=curl_metrics, transport=transport, ls_remote=ls_remote,
        ls_remote_stderr=ls_remote_stderr, ls_remote_started=ls_remote_started,
        ls_remote_completed=ls_remote_completed, ls_remote_raw=ls_remote_raw,
        ls_remote_rc=ls_remote_rc, fetch_started=fetch_started,
        fetch_completed=fetch_completed, fetch_rc=fetch_rc, curl_stderr=curl_stderr,
        commit=commit, requested_url=requested_url)
    root, artifact_root = root.resolve(), artifact_root.resolve()
    paths, members = found["paths"], found["members"]
    xml_members, languages = found["xml_members"], found["languages"]
    row = {
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "commit": commit,
        "members": members,
        "repository": REPO,
        "requested_url": requested_url,
        "xml_members": xml_members,
    }
    stage = SS.seal(
        artifact_root=artifact_root,
        artifact_paths=paths,
        receipt_path=root / "stages/original/receipt.json",
        records=1,
        fmt="exact-pinned-codeload-tar-gzip+git-head-observation+http-transport",
        logical_sha256=hashlib.sha256(
            (canonical_json(row) + "\n").encode("utf-8")
        ).hexdigest(),
        producer_id="pipeline/acquire_justice_canada_laws_snapshot.py",
        producer_version=VERSION,
    )
    run = {
        "schema": "aw.justice-canada-laws-source-run.v1",
        "version": VERSION,
        "status": "source_acquired_pending_source_native_projection",
        "source": "Justice Canada bilingual consolidated Acts and Regulations XML repository",
        "repository": REPO,
        "commit": commit,
        "counts": {
            "source_archives": 1,
            "archive_members": members,
            "xml_members": xml_members,
        },
        "observed_language_paths": sorted(languages),
        "stages": {"original": stage},
        "content_scope": "full pinned repository XML snapshot; document/section/chunk counts pending source-native bilingual projection",
        "next_stage": "preserve XML document/version/section hierarchy into full Markdown, exact production selector join, parent-bound relations/outlinks, current chunks",
        "index_admission": "blocked_until_projection_profile_fidelity_chunks_and_exact_production_join_seal",
    }
    encoded = (json.dumps(run, indent=2, sort_keys=True) + "\n").encode("utf-8")
    run_path = root / "RUN.json"
    if run_path.exists():
        if sha256_file(run_path) != hashlib.sha256(encoded).hexdigest():
            raise ValueError("immutable Justice Canada RUN collision")
    else:
        atomic_publish(run_path, encoded)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "root", "artifact-root", "archive", "headers", "curl-metrics",
        "transport", "ls-remote", "ls-remote-stderr", "ls-remote-started",
        "ls-remote-completed",
        "ls-remote-raw", "ls-remote-rc", "fetch-started",
        "fetch-completed", "fetch-rc", "curl-stderr",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--requested-url", required=True)
    args = parser.parse_args()
    result = seal(
        root=args.root,
        artifact_root=args.artifact_root,
        archive=args.archive,
        headers=args.headers,
        curl_metrics=args.curl_metrics,
        transport=args.transport,
        ls_remote=args.ls_remote,
        ls_remote_stderr=args.ls_remote_stderr,
        ls_remote_started=args.ls_remote_started,
        ls_remote_completed=args.ls_remote_completed,
        ls_remote_raw=args.ls_remote_raw,
        ls_remote_rc=args.ls_remote_rc,
        fetch_started=args.fetch_started,
        fetch_completed=args.fetch_completed,
        fetch_rc=args.fetch_rc,
        curl_stderr=args.curl_stderr,
        commit=args.commit,
        requested_url=args.requested_url,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
