"""Seal the RFC Editor's source-owned rsync text corpus and full projections.

Moved from the flat pipeline script by task 1395; the family adapters
`cytir.sources.families.rfc_editor` call `rfc_row` (one file -> one markdown row, one place) and
`materialize` (the whole-tree seal). The rsync FETCH stays in the operator shell.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.parse

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire import preserved_stage_io as PS
from cytir.acquire import stage_receipt as SS


VERSION = "rfc-editor-rsync-preserved.v1"
#: frozen label: stage receipts sealed before the move carry this producer string; a rewritten
#: label would make the same tree seal to different bytes (task 1395 characterization)
PRODUCER_ID = "pipeline/acquire_rfc_editor_rsync_preserved.py"
RFC_NAME = re.compile(r"rfc([0-9]+)\.txt")
URL_RE = re.compile(r"https?://[^\s<>\"']+")


class RfcAcquisitionError(RuntimeError):
    pass


def _logical(rows: list[dict]) -> tuple[bytes, str]:
    data = b"".join((canonical_json(row) + "\n").encode() for row in rows)
    return data, hashlib.sha256(data).hexdigest()


def inventory(raw_dir: Path) -> list[Path]:
    files = sorted(path for path in raw_dir.iterdir()
                   if path.is_file() and RFC_NAME.fullmatch(path.name))
    if not files:
        raise RfcAcquisitionError("RFC rsync tree contains no RFC text files")
    numbers = [int(RFC_NAME.fullmatch(path.name).group(1)) for path in files]
    if len(numbers) != len(set(numbers)):
        raise RfcAcquisitionError("duplicate RFC number in rsync tree")
    return files


def rfc_row(number: int, raw: bytes, original_row: dict) -> tuple[dict | None, dict | None]:
    """One RFC text file -> `(markdown row, None)` or `(None, quarantine row)` (empty text)."""
    encoding = "utf-8"
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        encoding = "iso-8859-1"
        text = raw.decode(encoding)
    source_url = f"https://www.rfc-editor.org/rfc/rfc{number}.txt"
    document_id = f"rfc-editor:rfc:{number}"
    if not text.strip():
        return None, {"document_id": document_id, "reason": "empty_source_text",
                      "source_sha256": original_row["sha256"]}
    return {
        "document_id": document_id, "url": source_url, "lang": "en",
        "title": f"RFC {number}", "markdown": text,
        "source_relative_path": original_row["relative_path"],
        "source_bytes": original_row["bytes"],
        "source_sha256": original_row["sha256"],
        "source_text_encoding": encoding,
    }, None


def outlinks(row: dict) -> list[dict]:
    """The outlink records of one markdown row (unique targets, malformed hosts skipped)."""
    seen: set[str] = set()
    out = []
    for target in URL_RE.findall(row["markdown"]):
        target = target.rstrip(".,;:)]}")
        try:
            parsed = urllib.parse.urlsplit(target)
        except ValueError:
            # RFC prose contains URI grammar examples that deliberately use
            # malformed bracketed hosts. They are source text, not crawlable
            # outlinks; retain the text and exclude only this derived token.
            continue
        if not parsed.hostname or target in seen:
            continue
        seen.add(target)
        out.append({"document_id": row["document_id"], "source_url": row["url"],
                    "target_url": target})
    return out


def materialize(*, raw_dir: Path, before_list: Path, after_list: Path,
                root: Path, module: str, started_at: str, completed_at: str,
                rsync_version: str, fetch_vantage: str) -> dict:
    files = inventory(raw_dir)
    if not before_list.is_file() or not after_list.is_file():
        raise RfcAcquisitionError("missing before/after rsync inventories")
    root.mkdir(parents=True, exist_ok=True)
    transport = {
        "schema": "aw.rsync-fetch-metadata.v1", "protocol": "rsync",
        "module": module, "requested_route": f"rsync://{module}/",
        "final_route": f"rsync://{module}/", "exit_status": 0,
        "started_at": started_at, "completed_at": completed_at,
        "fetch_vantage": fetch_vantage, "rsync_version": rsync_version,
        "inventory_before": {"bytes": before_list.stat().st_size,
                             "sha256": sha256_file(before_list)},
        "inventory_after": {"bytes": after_list.stat().st_size,
                            "sha256": sha256_file(after_list)},
    }
    metadata = root / "input/rsync.transport.json"
    body = (json.dumps(transport, indent=2, sort_keys=True) + "\n").encode()
    if metadata.exists():
        if sha256_file(metadata) != hashlib.sha256(body).hexdigest():
            raise RfcAcquisitionError("immutable rsync metadata collision")
    else:
        atomic_publish(metadata, body)

    original_rows = []
    for path in files:
        match = RFC_NAME.fullmatch(path.name)
        original_rows.append({"rfc_number": int(match.group(1)),
                              "relative_path": f"input/rfcs/{path.name}",
                              "bytes": path.stat().st_size,
                              "sha256": sha256_file(path)})
    original_data, original_sha = _logical(original_rows)
    original_artifacts = files + [before_list, after_list, metadata]
    original = SS.seal(
        artifact_root=root, artifact_paths=original_artifacts,
        receipt_path=root / "stages/original/receipt.json",
        records=len(files), fmt="exact-rsync-file-tree+canonical-transport-metadata",
        logical_sha256=original_sha,
        producer_id=PRODUCER_ID,
        producer_version=VERSION)

    markdown_rows: list[dict] = []
    outlink_rows: list[dict] = []
    quarantine_rows: list[dict] = []
    for path, original_row in zip(files, original_rows):
        row, quarantine = rfc_row(original_row["rfc_number"], path.read_bytes(), original_row)
        if quarantine:
            quarantine_rows.append(quarantine)
            continue
        markdown_rows.append(row)
        outlink_rows.extend(outlinks(row))
    markdown_rows.sort(key=lambda row: int(row["document_id"].rsplit(":", 1)[1]))
    outlink_rows.sort(key=lambda row: (row["document_id"], row["target_url"]))

    stages = {"original": original}
    for name, rows in (("markdown", markdown_rows), ("outlinks", outlink_rows),
                       ("quarantine", quarantine_rows)):
        data, logical_sha = _logical(rows)
        artifact = PS.store_artifact(root, f"stages/{name}/data", "jsonl", data)
        stages[name] = SS.seal(
            artifact_root=root, artifact_paths=[PS.artifact_path(root, artifact)],
            receipt_path=root / f"stages/{name}/receipt.json",
            records=len(rows), fmt="canonical-jsonl+deterministic-gzip",
            logical_sha256=logical_sha,
            producer_id=PRODUCER_ID,
            producer_version=VERSION)
    run = {
        "schema": "aw.rfc-editor-rsync-acquisition-run.v1", "version": VERSION,
        "status": "complete", "source": "RFC Editor text corpus",
        "route": "source_owned_rsync_module", "fetch_vantage": fetch_vantage,
        "license": "IETF Trust Legal Provisions; redistribution permitted; modification restrictions apply",
        "license_reference": "https://www.rfc-editor.org/series/rfc-use/",
        "counts": {"original_documents": len(files),
                   "markdown_documents": len(markdown_rows),
                   "outlink_records": len(outlink_rows),
                   "quarantined_documents": len(quarantine_rows)},
        "stages": stages,
        "production_subtraction": "pending_source-native_rfc_number_membership_gate",
        "outlink_cadence": ["after_completed_wave", "monthly"],
        "downstream": {"chunks": "pending_generic_structured_profile",
                       "vectors": "pending_1024d_fp16"},
    }
    run_body = (json.dumps(run, indent=2, sort_keys=True) + "\n").encode()
    run_path = root / "RUN.json"
    if run_path.exists():
        if sha256_file(run_path) != hashlib.sha256(run_body).hexdigest():
            raise RfcAcquisitionError("immutable RUN collision")
    else:
        atomic_publish(run_path, run_body)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw-dir", "before-list", "after-list", "root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("module", "started-at", "completed-at", "rsync-version",
                 "fetch-vantage"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    result = materialize(
        raw_dir=args.raw_dir.resolve(), before_list=args.before_list.resolve(),
        after_list=args.after_list.resolve(), root=args.root.resolve(),
        module=args.module, started_at=args.started_at,
        completed_at=args.completed_at, rsync_version=args.rsync_version,
        fetch_vantage=args.fetch_vantage)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
