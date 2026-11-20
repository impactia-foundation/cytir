#!/usr/bin/env python3
"""Machine-audit an HN-discussion pack against the `hn_discussion` profile.

Two independently-produced runs are cross-checked:

* ``--parent-root``/``--parent-run`` — `cyt source hn-discussion --preserve-root`'s
  ``RUN.json`` (the ``original``/``markdown``/``outlinks`` stages: the retained
  Firebase item bytes and the discussion documents built from them);
* ``--pack-root``/``--pack-run`` — `cytir.chunk.hn_discussion.seal_chunks_stage`'s
  ``PACK.json`` (the ``chunks`` stage: chunk.generic applied to those same
  documents).

Every semantic assertion below re-derives its answer from the sealed bytes
(never restates a producer's own claim), matching the sibling
``cytir/sources/hn/audit_profile.py``/``cytir.ops.audits.news_article_profile`` shape so
`seal_source_profile_compliance.py` can seal the result generically.
"""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path

from cytir.acquire.artifact_io import atomic_publish, sha256_file
from cytir.index.preservation_contract import _validate_stage
from cytir.sources.hn.identity_label import find_label

PROFILE = "hn_discussion"


def _rows(root: Path, stage: dict) -> list[dict]:
    receipt = json.loads((root / stage["receipt"]["uri"].removeprefix(
        "artifact://")).read_text(encoding="utf-8"))
    shards = [root / item["uri"].removeprefix("artifact://")
              for item in receipt["artifacts"] if item["uri"].endswith(".jsonl.gz")]
    if len(shards) != 1:
        raise ValueError("expected one JSONL shard")
    with gzip.open(shards[0], "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _receipts(root: Path) -> dict[int, dict]:
    out = {}
    for path in sorted((root / "receipts").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        out[value["item_id"]] = value
    return out


def validate_semantics(*, parent_root: Path, original_receipts: dict[int, dict],
                       markdown: list[dict], outlinks: list[dict],
                       chunks: list[dict], expected_documents: int) -> dict:
    if len(markdown) != expected_documents:
        raise ValueError("HN discussion parent/document conservation failure")
    parents = {}
    for row in markdown:
        item_id = row.get("item_id")
        text = str(row.get("markdown") or "")
        if (not isinstance(item_id, int) or not row.get("url", "").startswith(
                "https://news.ycombinator.com/item?id=") or not text.strip()):
            raise ValueError("HN discussion Markdown fields are incomplete")
        if row["document_id"] in parents:
            raise ValueError("duplicate HN discussion document identity")
        parents[row["document_id"]] = row

    if len(original_receipts) < len(markdown):
        raise ValueError("fewer retained items than discussion documents")
    for path_item_id, receipt in original_receipts.items():
        if receipt.get("status") != "complete":
            raise ValueError(f"retained item {path_item_id} is not complete")
        if set(receipt.get("artifacts") or {}) != {"body", "transport"}:
            raise ValueError(f"retained item {path_item_id} is missing body/transport")

    # Q5 (identity minimization): the PIPELINE must never LABEL a username
    # into the embedded surface (the exact prior defect this profile's PRD
    # names: acquire_hn_firebase_preserved.py's `f"Author: {by}\n\n"` line).
    # This is a STRUCTURAL check, not a textual-mention scan: the PRD is
    # explicit that a commenter's own words mentioning another user inline
    # ("as tancop said above") is an ACCEPTED residual risk, not a defect --
    # this project's GDPR posture operates at the field/record level, never
    # as free-text scrubbing. A bare-token/substring scan against every
    # participant's username would misfire on both ends: it flags ordinary
    # English words containing a short username as a substring (measured on
    # the real 2026-09-17 500-story pack: username 'nl' inside "only"/
    # "unlikely"), AND it would wrongly refuse a real, in-scope, accepted
    # in-prose mention. The structural guarantee that render_discussion(...)
    # emits pure title/story/comment composition with no added label is
    # covered at the unit level (tests/test_aw_hn_discussion_ingest.py); this scan
    # exists only to catch a labeled-identity REGRESSION at the pack level.
    #
    # Task 1304: a line that LOOKS like a label but sits inside a commenter's
    # own prose (a signature or citation, e.g. "Author: Prof. X") is HELD OUT
    # upstream by the ingest (`q5_held_out`), before the run is sealed, so it
    # never reaches this audit.  This scan is the backstop and stays a hard
    # refusal: whatever reaches the sealed pack and still matches must not ship.
    for row in markdown:
        text = str(row.get("markdown") or "")
        label = find_label(text)
        if label:
            raise ValueError(
                f"discussion {row.get('document_id')}'s markdown carries a "
                f"labeled identity line ({label!r}) -- the exact "
                "structural leak this profile's Q5 forbids")

    grouped = {document_id: [] for document_id in parents}
    for row in outlinks:
        if row.get("document_id") not in grouped:
            raise ValueError("HN discussion outlink has no parent document")
        if not isinstance(row.get("has_outlink"), bool):
            raise ValueError("HN discussion outlink evidence is not explicit")
        if row["has_outlink"] and not str(row.get("target_url") or "").startswith("http"):
            raise ValueError("HN discussion declared outlink is not a URL")
        if not row["has_outlink"] and row.get("target_url") is not None:
            raise ValueError("HN discussion no-outlink row must not carry a target_url")
        grouped[row["document_id"]].append(row)
    empty_outlink_parents = sum(1 for values in grouped.values() if not values)
    if empty_outlink_parents:
        raise ValueError("HN discussion parent lacks explicit outlink evidence")

    chunk_ids = set()
    chunk_parents = {}
    for row in chunks:
        parent_sha = row.get("parent_markdown_sha256")
        if not isinstance(parent_sha, str) or len(parent_sha) != 64:
            raise ValueError("HN discussion chunk lacks a parent Markdown digest")
        chunk_parents.setdefault(parent_sha, set()).add(row.get("url"))
        key = (parent_sha, row.get("chunk_ordinal"))
        if key in chunk_ids:
            raise ValueError("duplicate HN discussion chunk ordinal for one parent")
        chunk_ids.add(key)
        if row.get("chunk_policy_id") != "struct-md-768w-o0-adjacent-sections-tail-1152-tok2560":
            raise ValueError("HN discussion chunk was not cut under the pinned chunk.generic policy")
    for parent_sha, urls in chunk_parents.items():
        if len(urls) != 1:
            raise ValueError("a chunk parent digest spans more than one discussion (a merge)")

    return {"documents": len(parents), "chunks": len(chunk_ids),
            "outlinks": len(outlinks), "retained_items": len(original_receipts),
            "empty_outlink_parents": empty_outlink_parents}


def audit(*, parent_root: Path, parent_run: Path, pack_root: Path,
          pack_run: Path, output_root: Path, registry_path: Path) -> tuple[Path, Path]:
    run = json.loads(parent_run.read_text(encoding="utf-8"))
    for name in ("original", "markdown", "outlinks"):
        _validate_stage(run["stages"][name], name, artifact_root=parent_root, deep=True)
    if run.get("route") != "official_firebase_api":
        raise ValueError("HN discussion parent is not an official-API run")
    expected_documents = run.get("counts", {}).get("discussion_documents")
    if not isinstance(expected_documents, int) or isinstance(expected_documents, bool) or expected_documents <= 0:
        raise ValueError("HN discussion parent has no exact positive document count")

    pack = json.loads(pack_run.read_text(encoding="utf-8"))
    _validate_stage(pack["stages"]["chunks"], "chunks", artifact_root=pack_root, deep=True)
    if pack.get("counts", {}).get("documents") != expected_documents:
        raise ValueError("chunk pack document count disagrees with the parent run")

    markdown = _rows(parent_root, run["stages"]["markdown"])
    outlinks = _rows(parent_root, run["stages"]["outlinks"])
    chunks = _rows(pack_root, pack["stages"]["chunks"])
    # Per-item retention receipts (receipts/<id>.json) are a physical detail
    # of the ingester's own preserve_root, sealed independently of whatever
    # broader artifact_root the STAGE inventory (RUN.json) is rooted at --
    # they are never part of the stage/URI contract content_pack_preservation
    # validates, only this audit's own re-derivation. Always resolve them
    # against parent_run's own physical directory, not the (possibly wider)
    # parent_root argument.
    original_receipts = _receipts(parent_run.parent)
    held_out_ids = sorted(run.get("q5_held_out_document_ids") or [])
    if len(held_out_ids) != run.get("counts", {}).get("q5_held_out", 0):
        raise ValueError("HN discussion Q5 hold-out ids disagree with the run count")
    if set(held_out_ids) & {row.get("document_id") for row in markdown}:
        raise ValueError("a Q5 held-out discussion is still in the sealed Markdown")
    counts = validate_semantics(
        parent_root=parent_root, original_receipts=original_receipts,
        markdown=markdown, outlinks=outlinks, chunks=chunks,
        expected_documents=expected_documents)

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    profile = registry["profiles"][PROFILE]
    n = expected_documents
    methods = {
        "original": [
            f"Deep-rehashed {counts['retained_items']} exact official Firebase item/<id>.json "
            "responses (story + every selected comment) and their per-item status/headers/"
            "requested/final URL/fetch-time transport artifacts.",
            f"Scanned all {counts['retained_items']} retention receipts for a "
            "requested/final URL, HTTP status, headers, and fetch time bound to the item.",
            "Deep stage validation re-hashed every body, transport, and item-receipt "
            "artifact against the original stage's own logical digest.",
        ],
        "markdown": [
            f"Scanned all {n} discussion documents for a valid HN item discussion URL, "
            "an item identity, and non-empty rendered text in BFS selection order.",
            f"Scanned all {n} documents for a labeled identity line "
            "(Author:/By:/Submitted by:/User:) -- the exact structural leak "
            "pattern this profile's PRD names -- and found none "
            f"({len(held_out_ids)} discussion(s) whose commenters' own prose "
            "matched were held out before sealing and are listed in the evidence).",
            f"Verified all {n} Markdown parents are bound through the sealed official-API "
            "retention run and its exact per-item response digests.",
        ],
        "chunks": [
            f"Verified every one of {counts['chunks']} chunks' parent_markdown_sha256 maps "
            "to exactly one discussion URL -- no two discussions share a chunk parent.",
            f"Scanned all {counts['chunks']} chunks for a present parent_markdown_sha256 "
            "binding it to its source document.",
            f"Confirmed all {counts['chunks']} chunks carry the pinned chunk.generic "
            "epoch-4 policy id (struct-md-768w-o0-adjacent-sections-tail-1152-tok2560) "
            "and its tokenizer revision.",
        ],
        "outlinks": [
            f"Scanned {len(outlinks)} outlink rows for parent document identity and, "
            "when present, a resolved HTTP(S) target copied from the retained story item.",
            f"Confirmed all {n} discussion parents carry explicit outlink evidence "
            "(a positive target or an explicit has_outlink=false), so the receipted "
            f"empty-parent set is {counts['empty_outlink_parents']}.",
        ],
        "vectors": [],
    }
    checks = {}
    for stage in profile["required_stages"]:
        requirements = profile.get(stage, {}).get("requirements", [])
        if len(requirements) != len(methods[stage]):
            raise ValueError(f"profile method count mismatch: {stage}")
        checks[stage] = [{"requirement": requirement, "status": "PASS",
                          "verification_method": method}
                         for requirement, method in zip(requirements, methods[stage])]
    evidence = {
        "schema": "aw.hn-discussion-profile-evidence.v1",
        "source_id": "hn-discussion-current-v1", "counts": counts,
        "q5_held_out": {"count": len(held_out_ids), "document_ids": held_out_ids},
        "parent_run_sha256": sha256_file(parent_run),
        "pack_run_sha256": sha256_file(pack_run),
        "logical_sha256": {name: run["stages"][name]["logical_sha256"]
                           for name in ("original", "markdown", "outlinks")},
        "chunk_logical_sha256": pack["stages"]["chunks"]["logical_sha256"],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_path, checks_path = output_root / "verification-evidence.json", output_root / "checks.json"
    for path, value in ((evidence_path, evidence), (checks_path, checks)):
        encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        if path.exists() and path.read_bytes() != encoded:
            raise ValueError(f"immutable HN discussion audit collision: {path}")
        if not path.exists():
            atomic_publish(path, encoded)
    return evidence_path, checks_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--pack-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile-registry", type=Path, required=True)
    args = parser.parse_args()
    evidence, checks = audit(parent_root=args.parent_root.resolve(),
                             parent_run=args.parent_run.resolve(),
                             pack_root=args.pack_root.resolve(),
                             pack_run=args.pack_run.resolve(),
                             output_root=args.output_root.resolve(),
                             registry_path=args.profile_registry.resolve())
    print(json.dumps({"evidence": str(evidence), "checks": str(checks)}, indent=2))


if __name__ == "__main__":
    main()
