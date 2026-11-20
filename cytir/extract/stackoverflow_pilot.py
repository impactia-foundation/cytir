#!/usr/bin/env python3
"""Build a bounded, source-native Stack Overflow Q+A preservation pilot.

This is deliberately separate from ``ingest_se.py``.  The historical ingester
flattens HTML, caps text at 8,000 characters, and retains a large answer map in
memory.  This stage instead streams the official Posts/Users archives, selects
an exact deterministic bottom-k sample in four accepted-question score cells,
and preserves the exact selected XML rows before emitting full Markdown and
parent-bound relations/outlinks.

No point-count projection or GPU admission is made here.  The output is a
measured source/Markdown pilot for the ``community_qa`` preservation profile.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
import heapq
import html
import json
from pathlib import Path
import subprocess
import tempfile
from typing import BinaryIO, Iterable, Iterator
from urllib.parse import urljoin

from lxml import etree
from lxml import html as lxml_html

from cytir.acquire.gzip_pinned import deterministic_gzip
from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.index.preservation_contract import _validate_stage
import cytir.acquire.stage_receipt as SS


VERSION = "stackoverflow-qa-source-native-pilot.v1"
PRODUCER = "pipeline/build_stackoverflow_qa_pilot.py"
SITE = "stackoverflow.com"
PROFILE = "community_qa"
ALLOWED_LICENSES = {"CC BY-SA 2.5", "CC BY-SA 3.0", "CC BY-SA 4.0"}
CELLS = {
    "accepted_qscore_0_1": (0, 1),
    "accepted_qscore_2_4": (2, 4),
    "accepted_qscore_5_9": (5, 9),
    "accepted_qscore_10_plus": (10, None),
}


@dataclass
class SampledQuestion:
    cell: str
    rank: int
    raw: bytes
    attrs: dict[str, str]
    answers: list[tuple[int, int, bytes, dict[str, str]]]


def _score(attrs: dict[str, str]) -> int:
    try:
        return int(attrs.get("Score") or 0)
    except ValueError:
        return 0


def _cell(score: int) -> str:
    for name, (lower, upper) in CELLS.items():
        if score >= lower and (upper is None or score <= upper):
            return name
    raise AssertionError(score)


def _sample_rank(cell: str, question_id: str) -> int:
    # Sampling only: this digest is not a document or point identity.
    value = f"{VERSION}\0{cell}\0{question_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest(), "big")


def _xml_rows(stream: BinaryIO) -> Iterator[tuple[bytes, dict[str, str]]]:
    """Yield exact source row bytes and parsed attributes from dump XML.

    Stack Exchange dump tables are one ``row`` element per physical line.
    Keeping the original line makes the selected witness byte-exact rather
    than relying on XML reserialization, which may reorder or normalize it.
    """
    parser = etree.XMLParser(huge_tree=True, recover=False)
    for raw in stream:
        stripped = raw.strip()
        if not stripped.startswith(b"<row "):
            continue
        element = etree.fromstring(stripped, parser=parser)
        if element.tag != "row":
            raise ValueError("unexpected Stack Exchange XML element")
        yield raw, {str(key): str(value) for key, value in element.attrib.items()}


def select_posts(stream: BinaryIO, *, sample_per_cell: int) -> dict:
    if sample_per_cell <= 0:
        raise ValueError("sample_per_cell must be positive")
    heaps: dict[str, list[tuple[int, str]]] = {name: [] for name in CELLS}
    selected: dict[str, SampledQuestion] = {}
    answer_to_question: dict[str, str] = {}
    counts = Counter()

    for raw, attrs in _xml_rows(stream):
        post_type = attrs.get("PostTypeId")
        if post_type == "1":
            counts["questions_seen"] += 1
            question_id = attrs.get("Id")
            accepted_id = attrs.get("AcceptedAnswerId")
            if not question_id or not accepted_id:
                continue
            counts["questions_with_accepted_answer"] += 1
            question_score = _score(attrs)
            if question_score < 0:
                counts["accepted_questions_negative_score_excluded"] += 1
                continue
            cell = _cell(question_score)
            counts[f"eligible_{cell}"] += 1
            rank = _sample_rank(cell, question_id)
            heap = heaps[cell]
            candidate = (-rank, question_id)
            if len(heap) >= sample_per_cell and candidate <= heap[0]:
                continue
            if len(heap) >= sample_per_cell:
                _, evicted_id = heapq.heapreplace(heap, candidate)
                evicted = selected.pop(evicted_id)
                answer_to_question.pop(evicted.attrs["AcceptedAnswerId"], None)
            else:
                heapq.heappush(heap, candidate)
            selected[question_id] = SampledQuestion(
                cell=cell, rank=rank, raw=raw, attrs=attrs, answers=[])
            answer_to_question[accepted_id] = question_id
        elif post_type == "2":
            counts["answers_seen"] += 1
            parent_id = attrs.get("ParentId")
            if not parent_id or parent_id not in selected:
                continue
            answer_id = attrs.get("Id")
            if not answer_id:
                continue
            question = selected[parent_id]
            answer_score = _score(attrs)
            accepted = answer_id == question.attrs.get("AcceptedAnswerId")
            # Preserve the accepted answer plus the two best answers scoring
            # >=3, which measures the prior AMIN=3/QMIN=8/KANS=2 arm.
            question.answers.append((1 if accepted else 0, answer_score, raw, attrs))
            question.answers.sort(key=lambda item: (-item[0], -item[1], int(item[3]["Id"])))
            accepted_rows = [item for item in question.answers if item[0]]
            qualifying = [item for item in question.answers if not item[0] and item[1] >= 3][:2]
            question.answers = accepted_rows[:1] + qualifying

    rows = []
    for question_id, question in sorted(selected.items(), key=lambda item: int(item[0])):
        accepted_id = question.attrs["AcceptedAnswerId"]
        accepted = next((item for item in question.answers
                         if item[3].get("Id") == accepted_id), None)
        top = sorted((item for item in question.answers if item[1] >= 3),
                     key=lambda item: (-item[1], int(item[3]["Id"])))[:2]
        rows.append({
            "cell": question.cell,
            "sample_rank_sha256_integer": str(question.rank),
            "question": _raw_record(question.raw, question.attrs),
            "accepted_answer": _raw_record(accepted[2], accepted[3]) if accepted else None,
            "top_answers_score_ge_3": [_raw_record(item[2], item[3]) for item in top],
            "comparison_prior_amin3_qmin8_kans2": {
                "included": _score(question.attrs) >= 8 or bool(top),
                "question_min_score": 8,
                "answer_min_score": 3,
                "answer_cap": 2,
                "selected_answer_ids": [item[3]["Id"] for item in top],
            },
        })
    counts["selected_questions"] = len(rows)
    counts["selected_with_accepted_answer_present"] = sum(
        row["accepted_answer"] is not None for row in rows)
    counts["prior_arm_included"] = sum(
        row["comparison_prior_amin3_qmin8_kans2"]["included"] for row in rows)
    return {"schema": "aw.stackoverflow-qa-post-selection.v1",
            "version": VERSION, "sample_per_cell": sample_per_cell,
            "counts": dict(sorted(counts.items())), "rows": rows}


def _raw_record(raw: bytes, attrs: dict[str, str]) -> dict:
    return {"raw_xml_base64": base64.b64encode(raw).decode("ascii"),
            "raw_xml_bytes": len(raw),
            "raw_xml_sha256": hashlib.sha256(raw).hexdigest(),
            "attributes": attrs}


def select_users(stream: BinaryIO, *, owner_ids: set[str]) -> dict:
    users = {}
    rows_seen = 0
    for raw, attrs in _xml_rows(stream):
        rows_seen += 1
        user_id = attrs.get("Id")
        if user_id in owner_ids:
            users[user_id] = _raw_record(raw, attrs)
    return {"schema": "aw.stackoverflow-qa-user-selection.v1",
            "version": VERSION,
            "counts": {"users_seen": rows_seen,
                       "owner_ids_requested": len(owner_ids),
                       "owner_ids_found": len(users),
                       "owner_ids_missing": len(owner_ids - set(users))},
            "users": dict(sorted(users.items(), key=lambda item: int(item[0])))}


def _owner_ids(selection: dict) -> set[str]:
    values = set()
    for row in selection["rows"]:
        records = [row["question"], row.get("accepted_answer"),
                   *row["top_answers_score_ge_3"]]
        for record in records:
            if record and record["attributes"].get("OwnerUserId"):
                values.add(record["attributes"]["OwnerUserId"])
    return values


def _tag(node: etree._Element) -> str:
    value = node.tag
    return value.rsplit("}", 1)[-1].casefold() if isinstance(value, str) else ""


def _source_link_target(base_url: str, raw_target: str) -> tuple[str, bool]:
    """Resolve a source-declared link, preserving malformed targets explicitly.

    Stack Exchange source rows can legally contain historical malformed hrefs
    (for example an unmatched IPv6 bracket).  They are part of the exact XML
    evidence and must neither abort the whole document nor be silently dropped.
    """
    raw_target = str(raw_target or "").strip()
    if not raw_target:
        return "", True
    try:
        return urljoin(base_url, raw_target), True
    except ValueError:
        return raw_target, False


def _inline(node: etree._Element, base_url: str) -> str:
    name = _tag(node)
    if name == "br":
        return "\n"
    if name == "code":
        value = "".join(node.itertext())
        fence = "`" * (max(1, max((len(part) for part in value.split("`")), default=0)) + 1)
        return f"{fence}{value}{fence}"
    if name == "a":
        label = "".join(node.itertext()).strip()
        target, valid = _source_link_target(base_url, node.get("href") or "")
        if target and not valid:
            return f"{label or 'Source link'} (malformed source href: `{target.replace('`', '')}`)"
        return f"[{label or target}]({target})" if target else label
    if name == "img":
        label = str(node.get("alt") or "image").strip()
        target, valid = _source_link_target(base_url, node.get("src") or "")
        if target and not valid:
            return f"{label} (malformed source image href: `{target.replace('`', '')}`)"
        return f"![{label}]({target})" if target else label
    values = [node.text or ""]
    for child in node:
        values.append(_inline(child, base_url))
        values.append(child.tail or "")
    return "".join(values)


def html_fragment_to_markdown(fragment: str, base_url: str) -> str:
    """Render a complete SE HTML fragment without any byte/token cap."""
    nodes = lxml_html.fragments_fromstring(fragment or "", no_leading_text=False)
    blocks: list[str] = []
    for node in nodes:
        if isinstance(node, str):
            if node.strip():
                blocks.append(node.strip())
            continue
        name = _tag(node)
        if name == "pre":
            code = "".join(node.itertext())
            fence = "`" * max(3, max((len(part) for part in code.split("`")), default=0) + 1)
            blocks.append(f"{fence}\n{code.rstrip()}\n{fence}")
        elif name in {"ul", "ol"}:
            ordered = name == "ol"
            items = []
            for ordinal, item in enumerate(
                    [child for child in node if _tag(child) == "li"], 1):
                prefix = f"{ordinal}." if ordered else "-"
                items.append(f"{prefix} {_inline(item, base_url).strip()}")
            blocks.append("\n".join(items))
        elif name == "blockquote":
            rendered = _inline(node, base_url).strip()
            blocks.append("\n".join(f"> {line}" for line in rendered.splitlines()))
        elif name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            blocks.append(f"{'#' * int(name[1])} {_inline(node, base_url).strip()}")
        else:
            rendered = _inline(node, base_url).strip()
            if rendered:
                blocks.append(rendered)
    return "\n\n".join(blocks).strip()


def _links(fragment: str, base_url: str) -> list[dict]:
    root = lxml_html.fragment_fromstring(fragment or "", create_parent="div")
    links = []
    seen = set()
    for anchor in root.xpath(".//a[@href]"):
        target, valid = _source_link_target(
            base_url, str(anchor.get("href") or "").strip())
        key = (target, " ".join("".join(anchor.itertext()).split()))
        if target and key not in seen:
            seen.add(key)
            links.append({"target_url": target, "anchor_or_context": key[1] or None,
                          "relation": ("cited-link" if valid
                                       else "malformed-source-link")})
    return links


def _attribution(attrs: dict[str, str], users: dict) -> dict:
    user_id = attrs.get("OwnerUserId")
    user = users.get(user_id) if user_id else None
    display = ((user or {}).get("attributes") or {}).get("DisplayName")
    display = display or attrs.get("OwnerDisplayName")
    return {"owner_user_id": user_id, "owner_display_name": display,
            "owner_profile_url": f"https://{SITE}/users/{user_id}" if user_id else None,
            "user_row_present": user is not None if user_id else None}


def materialize(selection: dict, user_selection: dict) -> dict:
    markdown_rows, outlink_rows, quarantine = [], [], []
    users = user_selection["users"]
    metrics = Counter()
    for row in selection["rows"]:
        question = row["question"]["attributes"]
        question_id = question["Id"]
        accepted_record = row.get("accepted_answer")
        if accepted_record is None:
            quarantine.append({"question_id": question_id,
                               "reason": "accepted_answer_missing_from_source_stream"})
            continue
        accepted = accepted_record["attributes"]
        records = [("question", question), ("accepted_answer", accepted)]
        if any(record.get("ContentLicense") not in ALLOWED_LICENSES
               for _, record in records):
            quarantine.append({"question_id": question_id,
                               "reason": "missing_or_unrecognized_source_license"})
            continue
        attributions = {kind: _attribution(record, users) for kind, record in records}
        if any(not (value["owner_user_id"] or value["owner_display_name"])
               for value in attributions.values()):
            quarantine.append({"question_id": question_id,
                               "reason": "irreversible_attribution_identity_missing"})
            continue
        question_url = f"https://{SITE}/questions/{question_id}"
        answer_url = f"https://{SITE}/a/{accepted['Id']}"
        question_md = html_fragment_to_markdown(question.get("Body") or "", question_url)
        answer_md = html_fragment_to_markdown(accepted.get("Body") or "", answer_url)
        if not question.get("Title") or not question_md or not answer_md:
            quarantine.append({"question_id": question_id,
                               "reason": "principal_question_or_answer_content_missing"})
            continue
        document_id = hashlib.sha256(
            f"stackoverflow-qa-document\0{question_id}\0{accepted['Id']}".encode()).hexdigest()
        tags = html.unescape(question.get("Tags") or "")
        markdown = (
            f"# {question['Title'].strip()}\n\n"
            f"- Question ID: {question_id}\n"
            f"- Accepted answer ID: {accepted['Id']}\n"
            f"- Question score: {_score(question)}\n"
            f"- Accepted answer score: {_score(accepted)}\n"
            f"- Question created: {question.get('CreationDate')}\n"
            f"- Accepted answer created: {accepted.get('CreationDate')}\n"
            f"- Question license: {question['ContentLicense']}\n"
            f"- Accepted answer license: {accepted['ContentLicense']}\n"
            f"- Tags: {tags}\n"
            f"- Question author: {attributions['question']['owner_display_name'] or attributions['question']['owner_user_id']}\n"
            f"- Answer author: {attributions['accepted_answer']['owner_display_name'] or attributions['accepted_answer']['owner_user_id']}\n\n"
            f"## Question\n\n{question_md}\n\n"
            f"## Accepted answer\n\n{answer_md}\n")
        markdown_rows.append({
            "document_id": document_id, "source_profile": PROFILE,
            "site": SITE, "lang": "en", "url": question_url,
            "answer_url": answer_url,
            "source_links": [
                {"relation": "canonical-question", "url": question_url},
                {"relation": "accepted-answer", "url": answer_url},
            ],
            "title": question["Title"], "tags": tags, "cell": row["cell"],
            "question": {"post_id": question_id, "score": _score(question),
                         "creation_time": question.get("CreationDate"),
                         "license": question["ContentLicense"],
                         "attribution": attributions["question"],
                         "raw_xml_sha256": row["question"]["raw_xml_sha256"]},
            "accepted_answer": {"post_id": accepted["Id"],
                                "score": _score(accepted),
                                "creation_time": accepted.get("CreationDate"),
                                "license": accepted["ContentLicense"],
                                "attribution": attributions["accepted_answer"],
                                "raw_xml_sha256": accepted_record["raw_xml_sha256"]},
            "comparison_prior_amin3_qmin8_kans2": row[
                "comparison_prior_amin3_qmin8_kans2"],
            "markdown": markdown,
            "extraction": {"producer": PRODUCER, "version": VERSION,
                           "method": "source-native-html-fragment-to-markdown",
                           "full_prechunk_content": True, "content_cap": None},
        })
        links = [
            {"target_url": question_url, "anchor_or_context": "canonical question",
             "relation": "canonical-question"},
            {"target_url": answer_url, "anchor_or_context": "accepted answer",
             "relation": "accepted-answer"},
        ]
        for kind, attrs in records:
            profile = attributions[kind]["owner_profile_url"]
            if profile:
                links.append({"target_url": profile,
                              "anchor_or_context": f"{kind} author",
                              "relation": "attribution-profile"})
            links.extend(_links(attrs.get("Body") or "",
                                question_url if kind == "question" else answer_url))
        seen = set()
        for link in links:
            key = (link["target_url"], link["relation"], link["anchor_or_context"])
            if key in seen:
                continue
            seen.add(key)
            outlink_rows.append({"document_id": document_id,
                                 "source_url": question_url,
                                 "ordinal": len(seen) - 1, "is_empty": False, **link})
        if not seen:
            outlink_rows.append({"document_id": document_id,
                                 "source_url": question_url, "ordinal": 0,
                                 "is_empty": True, "target_url": None,
                                 "anchor_or_context": None, "relation": None})
        metrics[f"retained_{row['cell']}"] += 1
        metrics["retained_questions"] += 1
        metrics["prior_arm_included"] += bool(
            row["comparison_prior_amin3_qmin8_kans2"]["included"])
        metrics["markdown_utf8_bytes"] += len(markdown.encode("utf-8"))
    markdown_rows.sort(key=lambda value: int(value["question"]["post_id"]))
    outlink_rows.sort(key=lambda value: (int(value["source_url"].rsplit("/", 1)[-1]),
                                         value["ordinal"]))
    quarantine.sort(key=lambda value: int(value["question_id"]))
    return {"markdown": markdown_rows, "outlinks": outlink_rows,
            "quarantine": quarantine, "metrics": dict(sorted(metrics.items()))}


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _jsonl_bytes(rows: Iterable[dict]) -> bytes:
    return b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)


def _publish_gzip(path: Path, raw: bytes) -> None:
    atomic_publish(path, deterministic_gzip(raw))


def _run_7z(archive: Path, member: str, consumer) -> object:
    with tempfile.NamedTemporaryFile(prefix="aw-so-7z-", suffix=".stderr") as stderr:
        process = subprocess.Popen(["7z", "x", "-so", str(archive), member],
                                   stdout=subprocess.PIPE, stderr=stderr)
        assert process.stdout is not None
        try:
            value = consumer(process.stdout)
        finally:
            process.stdout.close()
        rc = process.wait()
        if rc != 0:
            stderr.seek(0)
            raise ValueError(f"7z {member} stream failed rc={rc}: {stderr.read()[-2000:]!r}")
        return value


def _stage(*, artifact_root: Path, root: Path, name: str,
           paths: list[Path], records: int, fmt: str, logical: bytes) -> dict:
    return SS.seal(artifact_root=artifact_root, artifact_paths=paths,
                   receipt_path=root / "stages" / name / "receipt.json",
                   records=records, fmt=fmt,
                   logical_sha256=hashlib.sha256(logical).hexdigest(),
                   producer_id=PRODUCER, producer_version=VERSION)


def build(*, posts_archive: Path, users_archive: Path, inventory_run: Path,
          census_run: Path, root: Path, artifact_root: Path,
          sample_per_cell: int) -> dict:
    if (root / "RUN.json").exists():
        raise ValueError("Stack Overflow pilot RUN already exists")
    root.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(inventory_run.read_text())
    if inventory.get("status") != "source_archives_crc_and_members_complete_pending_projection":
        raise ValueError("Stack Overflow archive inventory is not terminal")
    _validate_stage(inventory["stages"]["archive_inventory"],
                    "stackoverflow.archive_inventory", artifact_root=artifact_root,
                    deep=True)
    census = json.loads(census_run.read_text())
    if census.get("status") != "complete":
        raise ValueError("Stack Overflow source census is not terminal")
    _validate_stage(census["stages"]["census"], "stackoverflow.census",
                    artifact_root=artifact_root, deep=True)

    posts_checkpoint = root / "checkpoints/posts-selection.json.gz"
    if posts_checkpoint.exists():
        selection = json.loads(gzip.decompress(posts_checkpoint.read_bytes()))
        if selection.get("sample_per_cell") != sample_per_cell:
            raise ValueError("posts checkpoint sample configuration drift")
    else:
        selection = _run_7z(
            posts_archive, "Posts.xml",
            lambda stream: select_posts(stream, sample_per_cell=sample_per_cell))
        _publish_gzip(posts_checkpoint, _json_bytes(selection))

    users_checkpoint = root / "checkpoints/users-selection.json.gz"
    if users_checkpoint.exists():
        user_selection = json.loads(gzip.decompress(users_checkpoint.read_bytes()))
    else:
        owners = _owner_ids(selection)
        user_selection = _run_7z(
            users_archive, "Users.xml",
            lambda stream: select_users(stream, owner_ids=owners))
        _publish_gzip(users_checkpoint, _json_bytes(user_selection))

    result = materialize(selection, user_selection)
    markdown_body = _jsonl_bytes(result["markdown"])
    outlinks_body = _jsonl_bytes(result["outlinks"])
    quarantine_body = _jsonl_bytes(result["quarantine"])
    metrics_value = {
        "schema": "aw.stackoverflow-qa-pilot-metrics.v1", "version": VERSION,
        "selection_counts": selection["counts"],
        "user_counts": user_selection["counts"],
        "materialization_counts": result["metrics"],
        "classification": "bounded source-native pilot; no document/point scale projection",
        "comparison_arms": {
            "accepted_question_score_cells": list(CELLS),
            "prior_task226": {"answer_min_score": 3,
                              "question_min_score": 8, "answer_cap": 2}},
    }
    metrics_body = _json_bytes(metrics_value)
    outputs = {
        "markdown": root / "data/markdown.jsonl.gz",
        "outlinks": root / "data/outlinks.jsonl.gz",
        "quarantine": root / "data/quarantine.jsonl.gz",
        "pilot_metrics": root / "data/pilot-metrics.json",
    }
    _publish_gzip(outputs["markdown"], markdown_body)
    _publish_gzip(outputs["outlinks"], outlinks_body)
    _publish_gzip(outputs["quarantine"], quarantine_body)
    atomic_publish(outputs["pilot_metrics"], metrics_body)
    original_paths = [posts_archive, users_archive, inventory_run, census_run,
                      posts_checkpoint, users_checkpoint]
    original_logical = (
        gzip.decompress(posts_checkpoint.read_bytes())
        + gzip.decompress(users_checkpoint.read_bytes())
        + inventory_run.read_bytes() + census_run.read_bytes())
    stages = {
        "original": _stage(
            artifact_root=artifact_root, root=root, name="original",
            paths=original_paths, records=len(selection["rows"]),
            fmt="official-StackOverflow-Posts+Users-7z+exact-selected-XML-rows",
            logical=original_logical),
        "markdown": _stage(
            artifact_root=artifact_root, root=root, name="markdown",
            paths=[outputs["markdown"]], records=len(result["markdown"]),
            fmt="canonical-jsonl+deterministic-gzip+full-source-native-QA-Markdown",
            logical=markdown_body),
        "outlinks": _stage(
            artifact_root=artifact_root, root=root, name="outlinks",
            paths=[outputs["outlinks"]], records=len(result["outlinks"]),
            fmt="canonical-jsonl+deterministic-gzip+parent-bound-QA-relations",
            logical=outlinks_body),
        "quarantine": _stage(
            artifact_root=artifact_root, root=root, name="quarantine",
            paths=[outputs["quarantine"]], records=len(result["quarantine"]),
            fmt="canonical-jsonl+deterministic-gzip+explicit-source-gate-failures",
            logical=quarantine_body),
        "pilot_metrics": _stage(
            artifact_root=artifact_root, root=root, name="pilot_metrics",
            paths=[outputs["pilot_metrics"]], records=1,
            fmt="canonical-json+measured-yield-no-point-projection",
            logical=metrics_body),
    }
    run = {
        "schema": "aw.stackoverflow-qa-source-native-pilot-run.v1",
        "version": VERSION,
        "status": "source_native_pilot_complete_pending_profile_fidelity_chunk_measurement",
        "source_profile": PROFILE, "site": SITE,
        "counts": {"selected_questions": len(selection["rows"]),
                   "markdown_documents": len(result["markdown"]),
                   "outlink_rows": len(result["outlinks"]),
                   "quarantine": len(result["quarantine"])},
        "source_census_sha256": sha256_file(census_run),
        "archive_inventory_sha256": sha256_file(inventory_run),
        "content_fidelity": {"full_original_archives": True,
                             "exact_selected_xml_rows": True,
                             "full_prechunk_markdown": True,
                             "source_content_cap": None},
        "stages": stages,
        "downstream": "profile/fidelity audit; source-native Q+A chunk pilot; exact token and UUID gates; no scale ratification before measured yield",
    }
    atomic_publish(root / "RUN.json", _json_bytes(run))
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("posts-archive", "users-archive", "inventory-run",
                 "census-run", "root", "artifact-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--sample-per-cell", type=int, default=256)
    args = parser.parse_args()
    value = build(
        posts_archive=args.posts_archive.resolve(),
        users_archive=args.users_archive.resolve(),
        inventory_run=args.inventory_run.resolve(),
        census_run=args.census_run.resolve(), root=args.root.resolve(),
        artifact_root=args.artifact_root.resolve(),
        sample_per_cell=args.sample_per_cell)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
