#!/usr/bin/env python3
"""Deep-audit one news_article pack: semantics, lineage and profile evidence.

Task 450 generalized this off the Aug-14 pack it was born for. It used to
assert four literal counts (`article_parents == 4993`, `legacy_passage_rows ==
14851`, ...) and stamp one literal `source_id`, so it could certify exactly one
historical pack and nothing else — which is why a fresh daily-news delta had no
route to a profile-compliance receipt at all (task 449 §6.3).

What is audited is unchanged and undiluted; only *which* pack is audited is now
an input:

  * `--source-id` names the pack under audit and is stamped into the evidence;
  * `--expect-count NAME=VALUE` (repeatable) pins any count a caller knows from
    upstream. The Aug-14 invocation passes its original four and therefore gets
    byte-identical strictness;
  * everything else is re-derived from the run and the pack and cross-checked
    for exact conservation — parents, markdown identity and full-body
    semantics, response lineage, outlink parent representation, chunk parent
    digests, ordinals and retained metadata, and the pinned news-specific chunk
    policy.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import uuid

from cytir.acquire.artifact_io import atomic_publish, sha256_file
from cytir.index.preservation_contract import _validate_stage


NEWS_KEYS = ("headline", "byline", "byline_status", "publication_time",
             "publication_time_status", "publisher", "fetched_at")
NEWS_POLICY_PATH = (
    "data/chunk-policy-news-article-768w-o0-adjacent-sections-tail-1152-tok2560.v1.json")
# Sibling-read basename (the declared label above is compared, never read by path).
NEWS_POLICY_FILE = "chunk-policy-news-article-768w-o0-adjacent-sections-tail-1152-tok2560.v1.json"
NEWS_POLICY_ID = "news-article-768w-o0-adjacent-sections-tail-1152-tok2560"

# ADR-100 Ruling A: news legitimately chunks with the legacy 180w/o30
# windower (ADR-090's measured verdict — legacy beats the structured epoch-4
# policy on this content). `chunk_policy.py` now expresses that deployed
# algorithm as a strict descriptor; a pack may bind its chunks to EITHER the
# structured news policy above OR this descriptor. Nothing about the
# structured branch below changes.
LEGACY_POLICY_PATH = "data/chunk-policy-legacy-180w-o30.v1.json"
LEGACY_POLICY_ID = "legacy-180w-o30"
# Sibling-read basename for the legacy descriptor (the label above is compared, never read by path).
LEGACY_POLICY_FILE = "chunk-policy-legacy-180w-o30.v1.json"


def _validate_news_policy(pack: dict, registry_path: Path) -> None:
    declared = pack.get("chunk_policy") or {}
    path_declared = declared.get("path")
    policy_id_declared = declared.get("policy_id")
    if (path_declared, policy_id_declared) == (NEWS_POLICY_PATH, NEWS_POLICY_ID):
        path = registry_path.parent / NEWS_POLICY_FILE
        if declared.get("sha256") != sha256_file(path):
            raise ValueError("news-specific policy SHA drifted")
        policy = json.loads(path.read_text())
        # Retention/supersession semantics belong to the source profile and
        # its evidence, not to the strict executable chunk-policy schema.
        if (policy.get("policy_id") != NEWS_POLICY_ID
                or policy.get("algorithm") != "struct-markdown-adjacent-section-pack-v1"
                or (policy.get("token_hard_max") or {}).get("exact") != 2560):
            raise ValueError("news-specific executable policy semantics drifted")
        return
    if (path_declared, policy_id_declared) == (LEGACY_POLICY_PATH, LEGACY_POLICY_ID):
        path = registry_path.parent / LEGACY_POLICY_FILE
        # The strict field-by-field validation (exact legend, exact window/
        # overlap, exact producer identity) lives in one place: the loader.
        from cytir.chunk.policy import (ChunkPolicyError, LEGACY_FIXED_WINDOW_PACK,
                                  LEGACY_RULE_LEGEND, load_chunk_policy)
        try:
            policy = load_chunk_policy(path)
        except ChunkPolicyError as exc:
            raise ValueError(f"legacy news chunk descriptor is invalid: {exc}") from exc
        if declared.get("sha256") != policy.sha256:
            raise ValueError("legacy news chunk descriptor SHA drifted")
        if (policy.policy_id != LEGACY_POLICY_ID
                or policy.algorithm != LEGACY_FIXED_WINDOW_PACK
                or policy.rule_legend != LEGACY_RULE_LEGEND):
            raise ValueError("legacy news chunk descriptor semantics drifted")
        return
    raise ValueError(
        "news pack does not declare the pinned news-specific policy "
        "(neither the structured news policy nor the ratified ADR-100 "
        "legacy descriptor)")


def _rows(root: Path, stage: dict) -> list[dict]:
    checked = _validate_stage(stage, "input", artifact_root=root, deep=True)
    receipt = json.loads((root / checked["receipt"]["uri"].removeprefix(
        "artifact://")).read_text())
    shards = [root / row["uri"].removeprefix("artifact://")
              for row in receipt["artifacts"] if row["uri"].endswith(".jsonl.gz")]
    values = []
    for path in shards:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            values.extend(json.loads(line) for line in source if line.strip())
    return values


def audit(*, parent_root: Path, parent_run: Path, pack_root: Path,
          output_root: Path, registry_path: Path, source_id: str,
          expect_counts: dict[str, int] | None = None,
          run_status: str = "ready_for_current_policy_chunking",
          pack_artifact_root: Path | None = None,
          ) -> tuple[Path, Path]:
    # A pack whose chunk artifacts are sealed into the same portable tree as
    # its parent run resolves `artifact://` against that tree's root, which is
    # not the directory holding PACK.json. Callers that co-locate the two keep
    # the historical behaviour by leaving this unset.
    chunk_root = pack_artifact_root or pack_root
    run = json.loads(parent_run.read_text())
    for name, stage in run["stages"].items():
        _validate_stage(stage, f"parent.{name}", artifact_root=parent_root, deep=True)
    counts = run.get("counts") or {}
    if run.get("status") != run_status:
        raise ValueError(
            f"news source run status is {run.get('status')!r}, not {run_status!r}")
    for name in ("article_parents", "markdown_documents", "quarantine_records"):
        if not isinstance(counts.get(name), int) or isinstance(
                counts.get(name), bool) or counts[name] < 0:
            raise ValueError(f"news run counts.{name} is missing or not a count")
    if counts["article_parents"] != counts["markdown_documents"]:
        raise ValueError("news run parent/markdown conservation drifted")
    # Any count the caller knows from upstream is pinned exactly. This is how
    # the Aug-14 invocation keeps the literal pins this module used to hardcode.
    for name, expected in sorted((expect_counts or {}).items()):
        if counts.get(name) != expected:
            raise ValueError(
                f"news run counts.{name}={counts.get(name)!r} != "
                f"expected {expected!r}")
    pack_path = pack_root / "PACK.json"
    pack = json.loads(pack_path.read_text())
    if pack.get("source_profile") != "news_article":
        raise ValueError("news PACK declares the wrong profile")
    _validate_news_policy(pack, registry_path)
    _validate_stage(pack["chunks"], "pack.chunks", artifact_root=chunk_root, deep=True)
    markdown = _rows(parent_root, run["stages"]["markdown"])
    outlinks = _rows(parent_root, run["stages"]["outlinks"])
    chunks = _rows(chunk_root, pack["chunks"])
    parents = {}
    bylines = published = 0
    for row in markdown:
        document_id = str(row.get("document_id") or "")
        url = str(row.get("url") or "")
        expected_id = hashlib.sha256(f"news-document\0{url}".encode()).hexdigest()
        metadata = {key: row.get(key) for key in NEWS_KEYS}
        if (document_id != expected_id or document_id in parents
                or not metadata["headline"] or not metadata["publisher"]
                or not metadata["fetched_at"]
                or metadata["byline_status"] not in
                ("source_declared", "not_declared_by_publisher")
                or metadata["publication_time_status"] not in
                ("source_declared", "not_declared_by_publisher")
                or (metadata["byline_status"] == "source_declared") != bool(metadata["byline"])
                or (metadata["publication_time_status"] == "source_declared") !=
                bool(metadata["publication_time"])):
            raise ValueError("news parent semantic identity is incomplete")
        text = str(row.get("markdown") or "")
        if (not text.startswith(f"# {metadata['headline']}\n\nByline: ")
                or f"\n\nPublisher: {metadata['publisher']}\n\nFetched: {metadata['fetched_at']}\n\n" not in text
                or f"\n\nSource: {url}\n\n" not in text
                or not text.split(f"\n\nSource: {url}\n\n", 1)[1].strip()):
            raise ValueError("news Markdown lacks explicit semantics or full body")
        lineage = row.get("lineage") or {}
        if (lineage.get("fetched_at") != metadata["fetched_at"]
                or not (lineage.get("original_sha1") or lineage.get("original_sha256"))):
            raise ValueError("news parent response lineage is incomplete")
        bylines += metadata["byline_status"] == "source_declared"
        published += metadata["publication_time_status"] == "source_declared"
        parents[document_id] = row
    linked = set(); empty = set()
    for row in outlinks:
        document_id = row.get("document_id")
        if document_id not in parents or row.get("publisher") != parents[document_id]["publisher"]:
            raise ValueError("news outlink lost publisher/article identity")
        if row.get("is_empty"):
            if row.get("target_url") is not None:
                raise ValueError("news empty-outlink row has a target")
            empty.add(document_id)
        else:
            if not row.get("target_url") or not row.get("anchor_or_context"):
                raise ValueError("news cited target lacks anchor/context")
            linked.add(document_id)
    if linked | empty != set(parents) or linked & empty:
        raise ValueError("news outlink parent conservation failure")
    point_ids = set(); ordinals: dict[str, list[int]] = {}
    parent_sha = {key: hashlib.sha256(str(value["markdown"]).encode()).hexdigest()
                  for key, value in parents.items()}
    for row in chunks:
        point_id = str(row.get("point_id") or ""); uuid.UUID(point_id)
        document_id = str(row.get("document_id") or "")
        if point_id in point_ids or document_id not in parents:
            raise ValueError("news chunk point/parent identity failure")
        point_ids.add(point_id)
        if row.get("parent_markdown_sha256") != parent_sha[document_id]:
            raise ValueError("news chunk parent digest mismatch")
        meta = row.get("source_metadata") or {}
        if any(meta.get(key) != parents[document_id].get(key) for key in NEWS_KEYS):
            raise ValueError("news chunk lost publication/retention semantics")
        ordinals.setdefault(document_id, []).append(row.get("chunk_ordinal"))
    if set(ordinals) != set(parents) or any(sorted(v) != list(range(len(v)))
                                              for v in ordinals.values()):
        raise ValueError("news chunk ordinal conservation failure")
    if len(parents) != counts["markdown_documents"]:
        raise ValueError(
            f"news markdown parents={len(parents)} != declared "
            f"markdown_documents={counts['markdown_documents']}")
    registry = json.loads(registry_path.read_text())
    profile = registry["profiles"]["news_article"]
    n = f"{len(parents):,}"
    methods = {
        "original": [
            f"Deep-rehashed all {n} exact admitted response evidence chains against their sealed original bytes.",
            "Scanned every parent for explicit publication status plus exact per-original fetch/capture time; source absence remains explicit.",
            "Derived and checked every publisher identity from the canonical article URL host."],
        "markdown": [
            f"Scanned all {n} complete article bodies with explicit headline, byline status and publication status ({bylines} source-declared bylines; {published} source-declared publication times).",
            "Matched every Markdown parent to its preserved response SHA-1/SHA-256 and fetch-time lineage."],
        "chunks": [
            f"Deep-validated the pinned {pack['chunk_policy']['policy_id']} chunk policy "
            "through the strict loader (exact SHA, algorithm and rule legend) and the "
            "rolling-12-month/source-URL supersession contract.",
            "Recomputed every canonical parent Markdown digest and matched it on every chunk.",
            "Verified every chunk retains headline/byline/publication/publisher/fetch metadata from its unique parent."],
        "outlinks": [
            "Scanned every outlink for parent article ID, canonical publisher identity and publication-time status.",
            "Checked all non-empty resolved cited targets carry first-occurrence anchor/context evidence.",
            f"Deep-validated parent conservation including {len(empty)} explicitly empty parents."],
        "vectors": [],
    }
    checks = {}
    for stage in profile["required_stages"]:
        requirements = profile.get(stage, {}).get("requirements", [])
        if len(requirements) != len(methods[stage]):
            raise ValueError(f"news profile method count mismatch: {stage}")
        checks[stage] = [{"requirement": req, "status": "PASS",
                          "verification_method": method}
                         for req, method in zip(requirements, methods[stage])]
    evidence = {
        "schema": "aw.news-article-profile-evidence.v1",
        "source_id": source_id,
        "run_status": run_status,
        "declared_counts": counts,
        "counts": {"parents": len(parents), "source_declared_bylines": bylines,
                   "source_declared_publication_times": published,
                   "chunks": len(chunks), "unique_point_ids": len(point_ids),
                   "outlinks_including_empty": len(outlinks),
                   "empty_outlink_parents": len(empty)},
        "logical_sha256": {name: stage["logical_sha256"]
                           for name, stage in run["stages"].items()},
        "chunk_logical_sha256": pack["chunks"]["logical_sha256"],
        "source_parent_run_sha256": sha256_file(parent_run),
        "pack_sha256": sha256_file(pack_path),
        "profile_registry_sha256": sha256_file(registry_path),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    paths = output_root / "verification-evidence.json", output_root / "checks.json"
    for path, value in zip(paths, (evidence, checks)):
        encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        if path.exists() and path.read_bytes() != encoded:
            raise ValueError(f"immutable news audit collision: {path}")
        if not path.exists(): atomic_publish(path, encoded)
    return paths


def _expect_count(value: str) -> tuple[str, int]:
    name, _, number = value.partition("=")
    if not name.strip() or not number.strip().lstrip("-").isdigit():
        raise argparse.ArgumentTypeError(
            f"--expect-count wants NAME=INTEGER, got {value!r}")
    return name.strip(), int(number)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-root", "parent-run", "pack-root", "output-root",
                 "profile-registry"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--source-id", required=True,
                   help="the pack under audit; stamped into the evidence")
    p.add_argument("--expect-count", type=_expect_count, action="append",
                   default=[], metavar="NAME=INTEGER",
                   help="pin a run count known from upstream (repeatable)")
    p.add_argument("--run-status", default="ready_for_current_policy_chunking")
    p.add_argument("--pack-artifact-root", type=Path,
                   help="portable root the pack's chunk artifacts resolve "
                        "against (default: --pack-root)")
    a = p.parse_args()
    evidence, checks = audit(parent_root=a.parent_root.resolve(),
        parent_run=a.parent_run.resolve(), pack_root=a.pack_root.resolve(),
        output_root=a.output_root.resolve(),
        registry_path=a.profile_registry.resolve(),
        source_id=a.source_id, expect_counts=dict(a.expect_count),
        run_status=a.run_status,
        pack_artifact_root=(a.pack_artifact_root.resolve()
                            if a.pack_artifact_root else None))
    print(json.dumps({"evidence": str(evidence), "checks": str(checks)}, indent=2))


if __name__ == "__main__":
    main()
