#!/usr/bin/env python3
"""Project exact current BFS DAM records from the sealed official catalog."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import urljoin

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire.deterministic_gzip import deterministic_gzip_text
from cytir.index.preservation_contract import _validate_stage
from cytir.acquire import stage_receipt as SS


SOURCE_ID = "bfs-dam-current-exact-key-reconstruction-v1"
VERSION = "bfs-dam-current-materialization.v1"
EXPECTED_RAW = 185_931
EXPECTED_PROJECTION = 158_500
EXPECTED_CURRENT = 158_499


def _artifact_path(root: Path, uri: str) -> Path:
    if not uri.startswith("artifact://"):
        raise ValueError("BFS stage contains nonportable artifact URI")
    path = (root / uri.removeprefix("artifact://")).resolve()
    if root != path and root not in path.parents:
        raise ValueError("BFS artifact escapes global root")
    return path


def _principal_path(root: Path, stage: dict, label: str) -> Path:
    validated = _validate_stage(stage, label, artifact_root=root, deep=True)
    rows = [row for row in validated["artifacts"]
            if row["uri"].endswith((".jsonl", ".jsonl.gz"))]
    if len(rows) != 1:
        raise ValueError(f"{label} requires one principal JSONL artifact")
    return _artifact_path(root, rows[0]["uri"])


def raw_key(asset: dict) -> tuple[str, str, str]:
    ids = asset.get("ids") or {}
    description = asset.get("description") or {}
    language = re.split(
        r"[/,]", str(description.get("language") or "").strip())[0].strip().lower()
    title = str(((description.get("titles") or {}).get("main")) or "").strip()
    dam_id = str(ids.get("damId") or "")
    if not dam_id or language not in {"de", "fr", "it", "en"} or not title:
        return "", "", ""
    return f"https://www.bfs.admin.ch/asset/{language}/{dam_id}", title, language


def _named(values: object) -> list[str]:
    result = []
    for value in values if isinstance(values, list) else []:
        if not isinstance(value, dict):
            continue
        name = str(value.get("name") or "").strip()
        code = str(value.get("code") or "").strip()
        if name:
            result.append(f"{name} ({code})" if code else name)
    return result


def source_links(asset: dict, canonical_url: str) -> list[dict]:
    links = []
    seen = set()
    for raw in asset.get("links") or []:
        if not isinstance(raw, dict):
            continue
        href = str(raw.get("href") or "").strip()
        if not href:
            continue
        target = urljoin("https://www.bfs.admin.ch", href)
        relation = str(raw.get("rel") or "related").strip() or "related"
        context = str(raw.get("text") or relation).strip() or relation
        key = (target, relation, context)
        if key not in seen:
            links.append({"url": target, "relation": relation, "context": context})
            seen.add(key)
    canonical = (canonical_url, "canonical-record", "BFS DAM asset page")
    if canonical not in seen:
        links.insert(0, {"url": canonical[0], "relation": canonical[1],
                         "context": canonical[2]})
    return links


def render(asset: dict, legacy: dict, canonical_url: str, lang: str) -> str:
    ids = asset.get("ids") or {}
    bfs = asset.get("bfs") or {}
    description = asset.get("description") or {}
    titles = description.get("titles") or {}
    bibliography = description.get("bibliography") or {}
    categorization = description.get("categorization") or {}
    shop = asset.get("shop") or {}
    title = str(titles.get("main") or legacy.get("title") or "").strip()
    lines = [f"# {title}", "", "## Record identity", "",
             f"- Canonical asset: {canonical_url}",
             f"- BFS DAM asset ID: {ids.get('damId')}",
             f"- API UUID: {ids.get('uuid')}",
             f"- Language: {lang}"]
    for label, value in (("Content ID", ids.get("contentId")),
                         ("Language-copy ID", ids.get("languageCopyId")),
                         ("GNP", ids.get("gnp"))):
        if value not in (None, ""):
            lines.append(f"- {label}: {value}")
    lines.extend(["", "## Description", ""])
    for label, value in (("Super-title", titles.get("super")),
                         ("Subtitle", titles.get("sub")),
                         ("Period", bibliography.get("period")),
                         ("Remark", bibliography.get("remark"))):
        text = str(value or "").strip()
        if text:
            lines.append(f"- {label}: {text}")
    summary = str(legacy.get("text") or "").strip()
    if summary:
        lines.extend(["", summary])
    lines.extend(["", "## Publication metadata", ""])
    lifecycle = bfs.get("lifecycle") or {}
    model = bfs.get("articleModel") or {}
    for label, value in (("Lifecycle", lifecycle.get("name")),
                         ("Lifecycle code", lifecycle.get("code")),
                         ("Article model", model.get("name")),
                         ("Article model code", model.get("code")),
                         ("Embargo/release time", bfs.get("embargo")),
                         ("Order number", shop.get("orderNr")),
                         ("Stock flag", shop.get("stock"))):
        if value not in (None, ""):
            lines.append(f"- {label}: {value}")
    lines.extend(["", "## Classifications", ""])
    classification_rows = 0
    for family in sorted(categorization):
        values = _named(categorization[family])
        if values:
            classification_rows += len(values)
            lines.append(f"### {family}")
            lines.extend(f"- {value}" for value in values)
            lines.append("")
    if not classification_rows:
        lines.append("No source-declared classifications.\n")
    links = source_links(asset, canonical_url)
    lines.extend(["## Source links", ""])
    lines.extend(f"- [{row['context']}]({row['url']}) — {row['relation']}"
                 for row in links)
    return "\n".join(lines).strip() + "\n"


class StageWriter:
    def __init__(self, root: Path, output: Path, publish_to: Path, name: str,
                 shard_bytes: int = 32 * 1024 * 1024):
        self.root = root; self.output = output; self.publish_to = publish_to
        self.name = name
        self.shard_bytes = shard_bytes; self.buffer = bytearray()
        self.logical = hashlib.sha256(); self.records = 0; self.paths: list[Path] = []

    def append(self, row: dict) -> None:
        encoded = (canonical_json(row) + "\n").encode()
        self.buffer.extend(encoded); self.logical.update(encoded); self.records += 1
        if len(self.buffer) >= self.shard_bytes:
            self.flush()

    def flush(self, allow_empty: bool = False) -> None:
        if not self.buffer and not allow_empty:
            return
        path = self.output / "stages" / self.name / "data" / (
            f"shard-{len(self.paths):06d}.jsonl.gz")
        path.parent.mkdir(parents=True, exist_ok=True)
        with deterministic_gzip_text(path, compresslevel=6) as target:
            target.write(self.buffer.decode())
        self.paths.append(path); self.buffer.clear()

    def seal(self) -> dict:
        self.flush(allow_empty=not self.paths)
        return SS.seal(
            artifact_root=self.root, artifact_paths=self.paths,
            receipt_path=self.output / "stages" / self.name / "receipt.json",
            records=self.records,
            fmt="canonical-jsonl+deterministic-gzip",
            logical_sha256=self.logical.hexdigest(),
            producer_id="pipeline/materialize_bfs_dam_current.py",
            producer_version=VERSION, publish_from=self.output,
            publish_to=self.publish_to)


def materialize(*, artifact_root: Path, source_run_path: Path,
                identity_run_path: Path, output: Path,
                expected_raw: int = EXPECTED_RAW,
                expected_projection: int = EXPECTED_PROJECTION,
                expected_current: int = EXPECTED_CURRENT) -> dict:
    artifact_root = artifact_root.resolve(); output = output.resolve()
    if output.exists():
        raise ValueError(f"refusing existing BFS materialization: {output}")
    if not output.is_relative_to(artifact_root):
        raise ValueError("BFS output must be under artifact root")
    source_run = json.loads(source_run_path.read_text())
    identity_run = json.loads(identity_run_path.read_text())
    if (source_run.get("schema") != "aw.bfs-dam-preserved-source-run.v1"
            or identity_run.get("schema") != "aw.bfs-dam-current-identity-run.v1"
            or identity_run.get("source_id") != SOURCE_ID):
        raise ValueError("BFS source/identity RUN contract failed")
    original = _validate_stage(source_run["stages"]["original"], "source.original",
                               artifact_root=artifact_root, deep=True)
    projection = _validate_stage(source_run["stages"]["projection"], "source.projection",
                                 artifact_root=artifact_root, deep=True)
    identity = _validate_stage(identity_run["stages"]["identity"], "identity",
                               artifact_root=artifact_root, deep=True)
    if (original["records"] != expected_raw
            or projection["records"] != expected_projection
            or identity["records"] != expected_current):
        raise ValueError("BFS source/current population drift")
    raw_path = _principal_path(artifact_root, source_run["stages"]["original"],
                               "source.original")
    projection_path = _principal_path(
        artifact_root, source_run["stages"]["projection"], "source.projection")
    identity_path = _principal_path(
        artifact_root, identity_run["stages"]["identity"], "identity")

    legacy = {}
    with projection_path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            key = (str(row.get("url") or ""), str(row.get("title") or ""),
                   str(row.get("lang") or ""))
            if key in legacy:
                raise ValueError("duplicate BFS projection key")
            legacy[key] = row
    current = {}
    with gzip.open(identity_path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            key = (row["current_url"], row["current_title"], row["current_lang"])
            if key in current:
                raise ValueError("duplicate BFS current key")
            current[key] = row
    if len(legacy) != expected_projection or len(current) != expected_current:
        raise ValueError("BFS projection/current exact-key drift")

    partial = output.with_name(output.name + f".partial.{os.getpid()}")
    partial.mkdir(parents=True)
    markdown = StageWriter(artifact_root, partial, output, "markdown")
    outlinks = StageWriter(artifact_root, partial, output, "outlinks")
    raw_rows = matched = links = 0; matched_keys = set()
    try:
        with raw_path.open(encoding="utf-8") as source:
            for raw_ordinal, line in enumerate(source):
                asset = json.loads(line); raw_rows += 1
                key = raw_key(asset)
                identity_row = current.get(key)
                if identity_row is None:
                    continue
                legacy_row = legacy.get(key)
                legacy_sha = hashlib.sha256(canonical_json(legacy_row).encode()).hexdigest()
                if (legacy_row is None
                        or legacy_sha != identity_row["legacy_source_record_sha256"]
                        or key in matched_keys):
                    raise ValueError("BFS raw/projection/current binding drift")
                matched_keys.add(key); matched += 1
                raw_sha = hashlib.sha256(canonical_json(asset).encode()).hexdigest()
                ids = asset.get("ids") or {}
                record_id = f"bfs-dam:{ids.get('damId')}:{key[2]}"
                parent_identity = {
                    "source_run_sha256": sha256_file(source_run_path),
                    "identity_run_sha256": sha256_file(identity_run_path),
                    "raw_artifact_uri": original["artifacts"][0]["uri"],
                    "raw_row_ordinal": raw_ordinal,
                    "raw_record_sha256": raw_sha,
                    "legacy_projection_sha256": legacy_sha,
                    "dam_id": ids.get("damId"),
                    "api_uuid": ids.get("uuid"),
                }
                rendered = render(asset, legacy_row, key[0], key[2])
                markdown_sha = hashlib.sha256(rendered.encode()).hexdigest()
                markdown.append({
                    "schema": "aw.bfs-dam-current-parent.v1",
                    "source_id": SOURCE_ID, "source_profile": "structured_data_feed",
                    "document_id": identity_row["point_id"],
                    "point_id": identity_row["point_id"],
                    "source_record_id": record_id, "url": key[0],
                    "title": key[1], "lang": key[2], "markdown": rendered,
                    "markdown_sha256": markdown_sha,
                    "parent_source_sha256": raw_sha,
                    "parent_identity": parent_identity,
                    "extraction": {"projection_recipe": VERSION,
                                   "source_text_truncation_applied": False,
                                   "full_source_record_preserved": True},
                })
                rows = source_links(asset, key[0])
                if not rows:
                    outlinks.append({
                        "schema": "aw.parent-source-link.v1",
                        "document_id": identity_row["point_id"],
                        "source_record_id": record_id, "source_url": key[0],
                        "parent_markdown_sha256": markdown_sha,
                        "parent_identity": parent_identity, "ordinal": 0,
                        "is_empty": True, "target_url": None,
                        "relation": None, "anchor_or_context": None})
                else:
                    for ordinal, row in enumerate(rows):
                        outlinks.append({
                            "schema": "aw.parent-source-link.v1",
                            "document_id": identity_row["point_id"],
                            "source_record_id": record_id, "source_url": key[0],
                            "parent_markdown_sha256": markdown_sha,
                            "parent_identity": parent_identity, "ordinal": ordinal,
                            "is_empty": False, "target_url": row["url"],
                            "relation": row["relation"],
                            "anchor_or_context": row["context"]})
                        links += 1
        if (raw_rows != expected_raw or matched != expected_current
                or matched_keys != set(current)):
            raise ValueError("BFS raw/current materialization conservation failed")
        markdown_stage = markdown.seal(); outlink_stage = outlinks.seal()
        result = {
            "schema": "aw.bfs-dam-current-materialization-run.v1",
            "status": "complete", "source_id": SOURCE_ID,
            "source_profile": "structured_data_feed",
            "counts": {"raw_asset_records": raw_rows,
                       "legacy_projection_records": len(legacy),
                       "current_source_documents": matched,
                       "markdown_parents": markdown.records,
                       "outlink_rows_including_empty": outlinks.records,
                       "resolved_source_links": links,
                       "quarantine_records": 0,
                       "legacy_keys_absent_current": 1},
            "inputs": {
                "source_run": {"uri": "artifact://" + source_run_path.relative_to(
                    artifact_root).as_posix(), "bytes": source_run_path.stat().st_size,
                    "sha256": sha256_file(source_run_path)},
                "identity_run": {"uri": "artifact://" + identity_run_path.relative_to(
                    artifact_root).as_posix(), "bytes": identity_run_path.stat().st_size,
                    "sha256": sha256_file(identity_run_path)}},
            "identity_policy": "current production UUIDs are parent identities; legacy UUIDs are not emitted",
            "stages": {"original": original, "markdown": markdown_stage,
                       "outlinks": outlink_stage},
            "admission": "requires structured profile, fidelity, token and portable-stage gates before embedding",
        }
        atomic_publish(partial / "RUN.json", (
            json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        os.rename(partial, output)
        return result
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--identity-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(materialize(
        artifact_root=args.artifact_root.resolve(),
        source_run_path=args.source_run.resolve(),
        identity_run_path=args.identity_run.resolve(),
        output=args.output.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
