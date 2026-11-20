"""RFC Editor parents: the prod-net-new selection and the reversible heading Markdown.

`net_new` builds a sealed RFC parent holding only the source-native RFC numbers that
production does not hold (or, with `selection="production"`, only those it does).
`restage` derives heading-aware Markdown from the preserved RFC text bytes; the
decoration is reversible to the decoded source, and `render` proves it.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire import preserved_stage_io as PS
from cytir.index.preservation_contract import _validate_stage
from cytir.acquire import stage_receipt as SS

NET_NEW_VERSION = "rfc-prod-source-native-selection.v2"
NET_NEW_PRODUCER = "aw/extract/rfc_editor.py#net_new"
RESTAGE_VERSION = "rfc-editor-heading-markdown.v2"
RESTAGE_PRODUCER = "aw/extract/rfc_editor.py#restage"
RFC_NAME = re.compile(r"rfc(?P<number>\d+)\.txt")
NUMBERED = re.compile(r"^\s*(?P<number>\d+(?:\.\d+)*\.?)\s+(?P<title>\S.*)$")
APPENDIX = re.compile(r"^\s*(Appendix\s+[A-Z](?:\.[0-9]+)?)\.?\s+(\S.*)$", re.I)



def _jsonl(rows: list[dict]) -> tuple[bytes, str]:
    data = b"".join((canonical_json(row) + "\n").encode() for row in rows)
    return data, hashlib.sha256(data).hexdigest()


def _read_stage_rows(root: Path, stage: dict) -> list[dict]:
    checked = _validate_stage(stage, "input", artifact_root=root, deep=True)
    receipt = json.loads((root / checked["receipt"]["uri"].removeprefix(
        "artifact://")).read_text(encoding="utf-8"))
    shards = [root / row["uri"].removeprefix("artifact://")
              for row in receipt["artifacts"] if row["uri"].endswith(".jsonl.gz")]
    if len(shards) != 1:
        raise ValueError("RFC input stage must have exactly one JSONL shard")
    with gzip.open(shards[0], "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def net_new(*, root: Path, heading_run: Path, membership_path: Path,
                  output_run: Path, selection: str = "net-new",
                  artifact_root: Path | None = None) -> dict:
    if selection not in {"net-new", "production"}:
        raise ValueError("selection must be net-new or production")
    artifact_root = (artifact_root or root).resolve()
    root = root.resolve()
    output_run = output_run.resolve()
    if not output_run.is_relative_to(artifact_root):
        raise ValueError("output RUN must be below the global artifact root")
    run = json.loads(heading_run.read_text(encoding="utf-8"))
    markdown = _read_stage_rows(root, run["stages"]["markdown"])
    outlinks = _read_stage_rows(root, run["stages"]["outlinks"])
    membership = json.loads(membership_path.read_text(encoding="utf-8"))
    prod = membership.get("rfc_numbers")
    if (not isinstance(prod, list) or len(prod) != len(set(prod))
            or len(prod) != membership.get("counts", {}).get("distinct_rfc_numbers")):
        raise ValueError("invalid production RFC membership artifact")
    source_numbers = {int(row["document_id"].rsplit(":", 1)[1]) for row in markdown}
    prod_numbers = set(map(int, prod))
    overlap = source_numbers & prod_numbers
    if overlap != prod_numbers:
        raise ValueError("production RFC membership contains source-unresolved RFC numbers")
    kept = [row for row in markdown
            if ((int(row["document_id"].rsplit(":", 1)[1]) in prod_numbers)
                == (selection == "production"))]
    kept_ids = {row["document_id"] for row in kept}
    kept_links = [row for row in outlinks if row["document_id"] in kept_ids]
    linked_ids = {row["document_id"] for row in kept_links}
    empty_rows = [{"document_id": value, "status": "no_external_outlinks"}
                  for value in sorted(kept_ids - linked_ids)]
    admission = output_run.parent
    membership_copy = admission / "membership/prod-aw_e3-rfcs.v1.json"
    membership_bytes = membership_path.read_bytes()
    if membership_copy.exists() and membership_copy.read_bytes() != membership_bytes:
        raise ValueError("immutable RFC membership copy collision")
    if not membership_copy.exists():
        atomic_publish(membership_copy, membership_bytes)
    original_rows = []
    original_paths = [membership_copy]
    for row in kept:
        path = root / row["source_relative_path"]
        number = int(row["document_id"].rsplit(":", 1)[1])
        if sha256_file(path) != row["source_sha256"]:
            raise ValueError("net-new RFC source artifact digest mismatch")
        original_rows.append({"rfc_number": number,
                              "relative_path": row["source_relative_path"],
                              "bytes": path.stat().st_size,
                              "sha256": row["source_sha256"]})
        original_paths.append(path)
    original_rows.sort(key=lambda row: row["rfc_number"])
    _, original_logical = _jsonl(original_rows)
    original_stage = SS.seal(
        artifact_root=artifact_root, artifact_paths=original_paths,
        receipt_path=admission / "stages/original/receipt.json",
        records=len(kept), fmt="exact-rsync-files+prod-membership-subtraction",
        logical_sha256=original_logical, producer_id=NET_NEW_PRODUCER,
        producer_version=NET_NEW_VERSION)
    stages = {"original": original_stage}
    for name, rows in (("markdown", kept), ("outlinks", kept_links),
                       ("outlink_empty_parents", empty_rows), ("quarantine", [])):
        data, logical = _jsonl(rows)
        relative = (admission / "stages" / name / "data").relative_to(artifact_root)
        artifact = PS.store_artifact(artifact_root, str(relative), "jsonl", data)
        stages[name] = SS.seal(
            artifact_root=artifact_root,
            artifact_paths=[PS.artifact_path(artifact_root, artifact)],
            receipt_path=admission / f"stages/{name}/receipt.json",
            records=len(rows), fmt="canonical-jsonl+deterministic-gzip",
            logical_sha256=logical, producer_id=NET_NEW_PRODUCER, producer_version=NET_NEW_VERSION)
    result = {
        "schema": ("aw.rfc-net-new-parent-run.v1" if selection == "net-new"
                   else "aw.rfc-production-parent-run.v1"),
        "version": NET_NEW_VERSION, "status": "complete",
        "selection": selection,
        "source_id": ("rfc-editor-net-new-prod-subtracted"
                      if selection == "net-new" else "rfc-editor-production-current"),
        "counts": {"source_documents": len(source_numbers),
                   "production_rfc_numbers": len(prod_numbers),
                   "production_overlap": len(overlap),
                   ("net_new_documents" if selection == "net-new"
                    else "production_documents"): len(kept),
                   ("net_new_outlinks" if selection == "net-new"
                    else "production_outlinks"): len(kept_links),
                   "empty_outlink_parents": len(empty_rows)},
        "membership": {"bytes": membership_copy.stat().st_size,
                       "sha256": sha256_file(membership_copy),
                       "uri": ("artifact://"
                               + membership_copy.relative_to(artifact_root).as_posix())},
        "source_heading_run_sha256": sha256_file(heading_run),
        "stages": stages,
    }
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if output_run.exists() and output_run.read_bytes() != encoded:
        raise ValueError("immutable RFC net-new RUN collision")
    if not output_run.exists():
        atomic_publish(output_run, encoded)
    for name, stage in stages.items():
        _validate_stage(stage, name, artifact_root=artifact_root, deep=True)
    return result


def _heading_level(line: str, previous_blank: bool, next_blank: bool) -> int | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 140 or not (previous_blank or next_blank):
        return None
    match = NUMBERED.fullmatch(line)
    if match:
        token = match.group("number").rstrip(".")
        return min(2 + token.count("."), 6)
    if APPENDIX.fullmatch(line):
        return 2
    if (len(stripped) <= 90 and any(char.isalpha() for char in stripped)
            and stripped.upper() == stripped and not stripped.endswith((".", ":", ";"))):
        return 2
    return None


def render(number: int, text: str, *, relative_path: str, source_sha256: str,
           encoding: str) -> tuple[str, list[dict]]:
    lines = text.splitlines(keepends=True)
    headings = []
    converted = []
    for index, line in enumerate(lines):
        bare = line.rstrip("\r\n")
        ending = line[len(bare):]
        previous_blank = index == 0 or not lines[index - 1].strip()
        next_blank = index + 1 == len(lines) or not lines[index + 1].strip()
        level = _heading_level(bare, previous_blank, next_blank)
        if level is not None:
            headings.append({"line": index + 1, "level": level, "text": bare})
            converted.append("#" * level + " " + bare + ending)
        else:
            converted.append(line)
    prefix = ("---\n" + f"aw_rfc_number: {number}\n"
              f"aw_source_path: \"{relative_path}\"\n"
              f"aw_source_sha256: {source_sha256}\n"
              f"aw_source_encoding: {encoding}\n---\n\n# RFC {number}\n\n")
    markdown = prefix + "".join(converted)
    # Prove that Markdown decoration is reversible to the entire decoded source.
    body = markdown[len(prefix):]
    restored = body.splitlines(keepends=True)
    for heading in headings:
        at = heading["line"] - 1
        marker = "#" * heading["level"] + " "
        if not restored[at].startswith(marker):
            raise ValueError("RFC heading reconstruction marker mismatch")
        restored[at] = restored[at][len(marker):]
    if "".join(restored) != text:
        raise ValueError("RFC Markdown is not byte-reversible to decoded source")
    return markdown, headings


def restage(*, root: Path, parent_run: Path, output_run: Path) -> dict:
    parent = json.loads(parent_run.read_text(encoding="utf-8"))
    original = _validate_stage(parent["stages"]["original"], "original",
                               artifact_root=root, deep=True)
    _validate_stage(parent["stages"]["outlinks"], "outlinks",
                               artifact_root=root, deep=True)
    _validate_stage(parent["stages"]["quarantine"], "quarantine",
                    artifact_root=root, deep=True)
    rows = []
    source_paths = [path for path in (root / "input/rfcs").glob("rfc*.txt")
                    if RFC_NAME.fullmatch(path.name)]
    for path in sorted(source_paths,
                       key=lambda p: int(RFC_NAME.fullmatch(p.name).group("number"))):
        match = RFC_NAME.fullmatch(path.name)
        if match is None:
            continue
        number = int(match.group("number"))
        raw = path.read_bytes()
        encoding = "utf-8"
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            encoding = "iso-8859-1"
            text = raw.decode(encoding)
        relative = path.relative_to(root).as_posix()
        source_sha = hashlib.sha256(raw).hexdigest()
        markdown, headings = render(number, text, relative_path=relative,
                                    source_sha256=source_sha, encoding=encoding)
        rows.append({
            "document_id": f"rfc-editor:rfc:{number}",
            "url": f"https://www.rfc-editor.org/rfc/rfc{number}.txt",
            "lang": "en", "title": f"RFC {number}", "markdown": markdown,
            "source_relative_path": relative, "source_sha256": source_sha,
            "source_text_encoding": encoding,
            "source_heading_count": len(headings),
            "source_heading_map_sha256": hashlib.sha256(
                (canonical_json(headings) + "\n").encode()).hexdigest(),
            "source_reconstruction_sha256": hashlib.sha256(text.encode()).hexdigest(),
        })
    if len(rows) != original["records"]:
        raise ValueError("RFC original/Markdown document conservation failure")
    data, logical = _jsonl(rows)
    artifact = PS.store_artifact(root, "stages/markdown-v2/data", "jsonl", data)
    markdown_stage = SS.seal(
        artifact_root=root, artifact_paths=[PS.artifact_path(root, artifact)],
        receipt_path=root / "stages/markdown-v2/receipt.json", records=len(rows),
        fmt="canonical-jsonl+deterministic-gzip+reversible-heading-map",
        logical_sha256=logical, producer_id=RESTAGE_PRODUCER, producer_version=RESTAGE_VERSION)
    run = {**parent, "schema": "aw.rfc-editor-heading-markdown-run.v2",
           "version": RESTAGE_VERSION, "status": "complete",
           "stages": {**parent["stages"], "markdown": markdown_stage},
           "heading_projection": {
               "method": "reversible source-line heading decoration",
               "documents": len(rows),
               "documents_with_headings": sum(row["source_heading_count"] > 0
                                               for row in rows),
               "source_text_truncation_applied": False}}
    encoded = (json.dumps(run, indent=2, sort_keys=True) + "\n").encode()
    if output_run.exists() and output_run.read_bytes() != encoded:
        raise ValueError("immutable RFC v2 RUN collision")
    if not output_run.exists():
        atomic_publish(output_run, encoded)
    _validate_stage(markdown_stage, "markdown", artifact_root=root, deep=True)
    return run
