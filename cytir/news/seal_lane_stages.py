#!/usr/bin/env python3
"""Seal one daily-news arm's preserved stages as hash-bound evidence.

The daily news lane produced no stage evidence at all: `seal_stage_receipt` /
`preserved_stage_io` / `stage-receipt` appeared nowhere in it, so a pack of its
rows could not name a single `news_article` stage (task 449 §6.1).  Task 450's
fetch path now seals the *originals* and emits an
`aw.news-lane-document.v1` parent per page; this producer turns a day's spool
into the portable, count-conserving stage inventory the preservation contract
reads:

    original  -> markdown -> (chunks) -> vectors
                          \\-> outlinks

What is sealed here
-------------------
* ``original``  — one row per **admitted** preserved article response (HTTP 200,
  HTML, on-host), each binding the sealed body and transport digests and the
  content-addressed receipt that carries our outbound request identity.
* ``markdown``  — the `aw.news-lane-document.v1` parents, in the exact shape
  `stage_structured_markdown_for_embedding.py` consumes (`document_id`, `url`,
  `markdown`, `markdown_sha256`, plus the headline/byline/publication-time/
  publisher/fetched-at semantics `news_article` requires).
* ``outlinks``  — every article-cited target with its first-occurrence
  anchor/context, plus an explicit receipted empty row for a parent that cites
  nothing, so "no outlinks" and "never looked" stay distinguishable.
* ``chunk_join`` — the document -> emitted point-id join for the passages this
  run actually produced.
* ``chunks``    — one row per distinct emitted passage, carrying its parent
  Markdown digest, its ordinal within that parent, and the publication
  semantics `news_article` requires, bound to the ADR-100 legacy chunk
  descriptor the lane actually runs.
* ``vectors``   — the passages that reached the staged collection, read back
  from that collection rather than assumed from the chunk set, so the
  consent/projection drop between the two is a *measured* quarantine.

The ``chunks`` stage, and ADR-100
---------------------------------
Task 450 refused to seal ``chunks``: ``news_article.chunks`` requires a
"pinned news-specific policy", this lane chunks with the legacy 180w/o30
windower, and ``chunk_policy.py`` could not express that rule — so no honest
policy artifact existed to declare.  ADR-100 Ruling A (PR #75, merged) makes
the deployed legacy rule a first-class strict descriptor
(``data/chunk-policy-legacy-180w-o30.v1.json``), and task 451 implemented it.
This producer now declares that descriptor: it is loaded through the strict
loader (never re-hashed by hand here), and its identity is written into every
chunk row and into the arm's ``PACK.json``.  Nothing about what gets chunked
changes — this is provenance made explicit, exactly as the ADR says.

Parents that emit no passage
----------------------------
The legacy chunker has a 15-word floor, so a preserved page whose principal
content is shorter than that yields a Markdown parent and **no chunk at all**.
Those parents are quarantined at the ``original -> markdown`` edge with an
explicit reason and their observed body word count, rather than being carried
into a ``markdown`` stage whose parents would then not all resolve to chunks.
Their response bytes stay sealed in ``original`` — nothing is dropped, the
count is simply accounted where it actually stops.

Every count is derived from the data and cross-checked; nothing is pinned as a
literal and nothing is copied from one artifact into another so that a later
check can rediscover it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire.deterministic_gzip import deterministic_gzip_text
from cytir.core import projection as PROJ
from cytir.chunk import policy as CP
from cytir.acquire import preserved_page_fetch as PF
from cytir.acquire import preserved_stage_io as PS
from cytir.acquire import stage_receipt as SS


RUN_SCHEMA = "aw.news-lane-run.v1"
ORIGINAL_SCHEMA = "aw.news-lane-original.v1"
OUTLINK_SCHEMA = "aw.news-lane-outlink.v1"
CHUNK_JOIN_SCHEMA = "aw.news-lane-document-chunk-join.v1"
CHUNK_SCHEMA = "aw.news-lane-chunk.v1"
VECTOR_SCHEMA = "aw.news-lane-vector.v1"
DOCUMENT_SCHEMA = "aw.news-lane-document.v1"
PACK_SCHEMA = "aw.news-lane-pack.v1"
RUN_STATUS = "news_article_stages_sealed"
# A frozen LABEL written into every sealed receipt's producer.id (its bytes feed manifest shas), not a
# path anything opens. Declared in `cytir.ops.public_cut.FROZEN_RECEIPT_LABELS`.
PRODUCER_ID = "pipeline/seal_news_lane_stages.py"
PROFILE = "news_article"

# The publication semantics `news_article` requires every chunk to retain, in
# the one spelling `cytir.ops.audits.news_article_profile.NEWS_KEYS` reads them.
NEWS_KEYS = ("headline", "byline", "byline_status", "publication_time",
             "publication_time_status", "publisher", "fetched_at")

# ADR-100 Ruling A: the ratified descriptor for the rule this lane runs. The
# path is the declaration; every field of it — including its SHA — comes back
# out of the strict loader, so this producer cannot declare a policy the
# loader would reject.
CHUNK_POLICY_ARTIFACT = "data/chunk-policy-legacy-180w-o30.v1.json"

# The lane's own arm routing (news_refresh_stage.sh). One home: a second copy
# that drifted would silently move a publisher between packs.
ARM_ROUTES = (("swissinfo.ch", "swissinfo"), ("srf.ch", "chmedia"),
              ("rts.ch", "chmedia"), ("rsi.ch", "chmedia"))

class NewsLaneSealError(RuntimeError):
    """The day's spool cannot be sealed as conserving news_article evidence."""


def load_declared_chunk_policy(path: Path) -> "CP.LegacyChunkPolicy":
    """The ADR-100 descriptor, through the strict loader and nothing else."""
    try:
        policy = CP.load_chunk_policy(path)
    except CP.ChunkPolicyError as exc:
        raise NewsLaneSealError(f"chunk policy {path} is not loadable: {exc}") from exc
    if policy.algorithm != CP.LEGACY_FIXED_WINDOW_PACK:
        raise NewsLaneSealError(
            f"the news lane runs the legacy chunker; {path} declares "
            f"{policy.algorithm!r}")
    return policy


def _policy_binding(policy: "CP.LegacyChunkPolicy") -> dict:
    return {"path": CHUNK_POLICY_ARTIFACT, "policy_id": policy.policy_id,
            "sha256": policy.sha256, "algorithm": policy.algorithm,
            "rule_legend": policy.rule_legend}


def arm_of(url: str) -> str | None:
    """Route one URL to its pack arm, or None when it belongs to no arm."""
    host = (urlsplit(url).netloc or "").split("@")[-1].split(":")[0].casefold()
    for suffix, arm in ARM_ROUTES:
        if host == suffix or host.endswith("." + suffix):
            return arm
    return None


def news_document_id(url: str) -> str:
    return hashlib.sha256(f"news-document\0{url}".encode()).hexdigest()


def _canonical_url(url: str) -> str:
    return (url or "").split("#")[0]


def load_receipts(spool: Path) -> list[dict]:
    """Every sealed publisher response in this spool, validated, sorted."""
    root = spool / "pages"
    receipts = []
    for path in sorted((root / "receipts").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") != PF.RECEIPT_SCHEMA:
            raise NewsLaneSealError(f"not a publisher-page receipt: {path}")
        for artifact in value["artifacts"].values():
            PS.validate_artifact(root, artifact)
        receipts.append(value)
    if not receipts:
        raise NewsLaneSealError(f"no sealed publisher responses under {root}")
    return receipts


def load_documents(spool: Path) -> list[dict]:
    path = spool / "news_documents.jsonl"
    documents = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != DOCUMENT_SCHEMA:
                raise NewsLaneSealError(f"{path}:{number} is not a news document")
            if row.get("document_id") != news_document_id(row.get("url") or ""):
                raise NewsLaneSealError(f"{path}:{number} document identity drift")
            documents.append(row)
    if not documents:
        raise NewsLaneSealError(f"no news documents in {path}")
    return documents


def resolve_parents(documents: list[dict]) -> tuple[list[dict], list[dict]]:
    """One parent per document, or an explicit refusal to name one.

    Several queued URLs can redirect onto one page, so a run legitimately holds
    more than one fetch of the same document. Agreement is judged on
    ``body_markdown_sha256`` — the extracted CONTENT — never on
    ``markdown_sha256``, which covers the rendered envelope and so includes the
    ``Fetched:`` line: two byte-identical fetches of one page have different
    envelope digests purely because they happened at different times, and
    grouping on that would report every repeat as a conflict. When the content
    agrees there is one parent and the extra fetches are simply repeats. When it
    DISAGREES the page changed under us mid-run — a rotating hub or listing page
    (rts.ch's configured sitemap yields these; task 449 §4.2) — and there is no
    honest answer to "which fetch produced this row". Guessing "the last one"
    would be a claim about upsert ordering that no receipt supports, so the
    document is quarantined instead: its points are claimed by no parent and
    cannot enter a pack.

    -> (resolved parents, divergent parents) — both sorted by document_id.
    """
    groups: dict[str, list[dict]] = {}
    for row in documents:
        groups.setdefault(row["document_id"], []).append(row)
    resolved, divergent = [], []
    for document_id, rows in sorted(groups.items()):
        rows.sort(key=lambda row: (row["fetched_at"],
                                   row["lineage"]["receipt_object_id"]))
        if len({row["body_markdown_sha256"] for row in rows}) == 1:
            resolved.append({**rows[0], "fetch_repeats": len(rows)})
        else:
            divergent.extend(rows)
    return resolved, divergent


def _seal_rows(*, artifact_root: Path, out_root: Path, name: str,
               rows: list[dict], fmt: str, producer_version: str) -> dict:
    """Write one deterministic JSONL gzip shard and seal its stage receipt."""
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / f"{name}.jsonl.gz"
    digest = hashlib.sha256()
    if path.exists():
        raise NewsLaneSealError(f"refusing to replace a sealed artifact: {path}")
    with deterministic_gzip_text(path) as handle:
        for row in rows:
            encoded = canonical_json(row) + "\n"
            handle.write(encoded)
            digest.update(encoded.encode("utf-8"))
    return SS.seal(
        artifact_root=artifact_root, artifact_paths=[path],
        receipt_path=out_root / f"{name}.receipt.json", records=len(rows),
        fmt=fmt, logical_sha256=digest.hexdigest(),
        producer_id=PRODUCER_ID, producer_version=producer_version)


def _edge(*, inputs: int, parents: int, quarantined: int, outputs: int,
          quarantine_receipt: dict | None) -> dict:
    if inputs != parents + quarantined:
        raise NewsLaneSealError(
            f"lineage conservation failed: {inputs} != {parents} + {quarantined}")
    edge = {"input_records": inputs, "emitted_parent_records": parents,
            "quarantined_records": quarantined, "output_records": outputs}
    if quarantined:
        if quarantine_receipt is None:
            raise NewsLaneSealError("non-zero quarantine requires a receipt")
        edge["quarantine_receipt"] = quarantine_receipt
    elif quarantine_receipt is not None:
        raise NewsLaneSealError("zero quarantine must not claim a receipt")
    return edge


def served_identities(gated_rows: Path, arm: str) -> tuple[dict, dict, int]:
    """Served point id per gated row, through the deployed projector.

    -> (served_of_chunk_point, document_of_served, unprojectable)
    """
    served_of_chunk: dict[str, str] = {}
    document_of_served: dict[str, str] = {}
    unprojectable = 0
    with gated_rows.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if arm_of(row.get("url") or "") != arm:
                continue
            document_id = row.get("news_document_id")
            chunk_point = row.get("point_id")
            if not document_id or not chunk_point:
                raise NewsLaneSealError(
                    f"{gated_rows}:{number} carries no news_document_id/point_id")
            projected = PROJ.project_record(row)
            if projected is None:
                unprojectable += 1
                continue
            served = projected[0]
            previous = document_of_served.setdefault(served, document_id)
            if previous != document_id:
                raise NewsLaneSealError(
                    f"served point {served} is claimed by two documents "
                    f"({previous} and {document_id})")
            served_of_chunk[chunk_point] = served
    if not served_of_chunk:
        raise NewsLaneSealError(f"no {arm} rows in {gated_rows}")
    return served_of_chunk, document_of_served, unprojectable


def document_body(document: dict) -> str:
    """The principal content the chunker was fed, out of its own envelope."""
    marker = f"\n\nSource: {document['url']}\n\n"
    text = document["markdown"]
    if marker not in text:
        raise NewsLaneSealError(
            f"document {document['document_id']} has no Source: envelope line")
    return text.split(marker, 1)[1]


def seal_arm(*, spool: Path, artifact_root: Path, arm: str, out_root: Path,
             day: str, producer_version: str, chunk_policy_path: Path,
             staged_point_ids: set[str], gated_rows: Path) -> dict:
    policy = load_declared_chunk_policy(chunk_policy_path)
    receipts = [r for r in load_receipts(spool)
                if arm_of(r["requested_url"]) == arm]
    if not receipts:
        raise NewsLaneSealError(f"no preserved responses for arm {arm!r}")
    documents, divergent = resolve_parents(
        [d for d in load_documents(spool) if arm_of(d["url"]) == arm])

    # An admitted original is a response that could carry article content AND
    # stayed on the publisher we asked. Everything else is preserved evidence
    # of what the publisher actually returned, not an article.
    admissible: dict[str, dict] = {}
    non_admitted = []
    for receipt in receipts:
        if (PF.is_admissible_article(receipt) and
                arm_of(_canonical_url(receipt["final_url"])) == arm):
            admissible[receipt["object_id"]] = receipt
        else:
            non_admitted.append({
                **receipt,
                "_reason": ("http_status" if receipt["http_status"] != 200 else
                            "non_html" if not PF.is_admissible_article(receipt)
                            else "off_arm_redirect")})

    # Several queued URLs can redirect onto one article, so one document may
    # have several admitted responses. Exactly one of them is that document's
    # `original`; the rest are repeat fetches, inventoried but never counted as
    # separate sources.
    by_document: dict[str, list[dict]] = {}
    for receipt in admissible.values():
        by_document.setdefault(
            news_document_id(_canonical_url(receipt["final_url"])),
            []).append(receipt)
    for rows in by_document.values():
        rows.sort(key=lambda row: (row["fetched_at"], row["object_id"]))

    parent_receipt: dict[str, dict] = {}
    for document in documents:
        document_id = document["document_id"]
        lineage = document["lineage"]
        receipt = admissible.get(lineage["receipt_object_id"])
        if receipt is None or news_document_id(
                _canonical_url(receipt["final_url"])) != document_id:
            raise NewsLaneSealError(
                f"markdown parent {document_id} names no admitted original")
        if (lineage["original_sha256"] != receipt["body_sha256"] or
                lineage["fetched_at"] != receipt["fetched_at"] or
                document["markdown_sha256"] != hashlib.sha256(
                    document["markdown"].encode("utf-8")).hexdigest()):
            raise NewsLaneSealError(
                f"markdown parent {document_id} does not bind its sealed original")
        parent_receipt[document_id] = receipt

    # The legacy chunker's own 15-word floor means a preserved page can yield a
    # Markdown parent and no passage at all. Such a parent stops here: it is
    # quarantined at this edge with its observed body length, rather than
    # entering a `markdown` stage in which some parents would resolve to no
    # chunk. Its response bytes stay sealed in `original`.
    resolved_ids = {row["document_id"] for row in documents}
    chunkless = [row for row in documents
                 if not set(row["chunk_point_ids"])]
    documents = [row for row in documents if set(row["chunk_point_ids"])]

    original_receipt = {
        document_id: parent_receipt.get(document_id) or rows[0]
        for document_id, rows in by_document.items()}
    repeat_fetches = [
        {"schema": "aw.news-lane-repeat-fetch.v1",
         "document_id": document_id,
         "requested_url": receipt["requested_url"],
         "final_url": receipt["final_url"],
         "fetched_at": receipt["fetched_at"],
         "original_sha256": receipt["body_sha256"],
         "receipt_object_id": receipt["object_id"]}
        for document_id, rows in sorted(by_document.items()) for receipt in rows
        if receipt["object_id"] != original_receipt[document_id]["object_id"]]

    original_rows = [{
        "schema": ORIGINAL_SCHEMA,
        "document_id": document_id,
        "url": _canonical_url(receipt["final_url"]),
        "requested_url": receipt["requested_url"],
        "final_url": receipt["final_url"],
        "publisher": (urlsplit(receipt["final_url"]).netloc or "").casefold(),
        "http_status": receipt["http_status"],
        "content_type": receipt["content_type"],
        "fetched_at": receipt["fetched_at"],
        "original_sha256": receipt["body_sha256"],
        "original_bytes": receipt["body_bytes"],
        "transport_sha256": receipt["artifacts"]["transport"]["body_sha256"],
        "receipt_object_id": receipt["object_id"],
        "artifacts": receipt["artifacts"],
    } for document_id, receipt in sorted(original_receipt.items())]

    documents.sort(key=lambda row: row["document_id"])
    markdown_rows = [{
        **{key: value for key, value in document.items()
           if key not in ("outlinks", "chunk_point_ids")},
        "parent_response": document["lineage"],
    } for document in documents]

    outlink_rows: list[dict] = []
    for document in documents:
        base = {"schema": OUTLINK_SCHEMA,
                "document_id": document["document_id"],
                "url": document["url"], "publisher": document["publisher"],
                "publication_time": document["publication_time"],
                "publication_time_status": document["publication_time_status"],
                "parent_markdown_sha256": document["markdown_sha256"]}
        if not document["outlinks"]:
            outlink_rows.append({**base, "ordinal": 0, "is_empty": True,
                                 "target_url": None, "anchor_or_context": None,
                                 "nofollow": None})
            continue
        for link in document["outlinks"]:
            if not link.get("target_url") or not link.get("anchor_or_context"):
                raise NewsLaneSealError(
                    f"cited target without anchor/context in "
                    f"{document['document_id']}")
            outlink_rows.append({**base, "ordinal": link["ordinal"],
                                 "is_empty": False,
                                 "target_url": link["target_url"],
                                 "anchor_or_context": link["anchor_or_context"],
                                 "nofollow": link["nofollow"]})

    # A page can emit two passages with identical text, and passage identity is
    # content-derived, so one document legitimately names one point id twice —
    # the collection holds one point for it. Collapse those inside the document
    # and count them; a point claimed by two DIFFERENT documents is a real
    # identity failure and still refuses.
    chunk_join_rows = []
    repeated_points = 0
    for document in documents:
        distinct = list(dict.fromkeys(document["chunk_point_ids"]))
        repeated_points += len(document["chunk_point_ids"]) - len(distinct)
        chunk_join_rows.append({
            "schema": CHUNK_JOIN_SCHEMA,
            "document_id": document["document_id"], "url": document["url"],
            "parent_markdown_sha256": document["markdown_sha256"],
            "chunk_point_ids": distinct,
            "chunk_count": len(distinct),
            "emitted_chunk_rows": len(document["chunk_point_ids"]),
        })
    point_ids = [point for row in chunk_join_rows
                 for point in row["chunk_point_ids"]]
    if len(point_ids) != len(set(point_ids)):
        owners: dict[str, list[str]] = {}
        for row in chunk_join_rows:
            for point in row["chunk_point_ids"]:
                owners.setdefault(point, []).append(row["url"])
        shared = sorted(point for point, urls in owners.items() if len(urls) > 1)
        raise NewsLaneSealError(
            f"{len(shared)} point ids are claimed by two documents "
            f"(first: {shared[0]} <- {owners[shared[0]]})")

    quarantine_rows = [{
        "schema": "aw.news-lane-original-quarantine.v1",
        "requested_url": receipt["requested_url"],
        "final_url": receipt["final_url"],
        "http_status": receipt["http_status"],
        "content_type": receipt["content_type"],
        "fetched_at": receipt["fetched_at"],
        "original_sha256": receipt["body_sha256"],
        "receipt_object_id": receipt["object_id"],
        "reason": receipt["_reason"],
    } for receipt in sorted(non_admitted, key=lambda r: r["object_id"])]
    divergent_ids = {row["document_id"] for row in divergent}
    extraction_quarantine = [{
        "schema": "aw.news-lane-extraction-quarantine.v1",
        "document_id": document_id,
        "url": _canonical_url(receipt["final_url"]),
        "original_sha256": receipt["body_sha256"],
        "receipt_object_id": receipt["object_id"],
        "reason": ("divergent_parents_within_run" if document_id in divergent_ids
                   else "no_principal_content_extracted"),
        "fetches": sum(1 for row in divergent
                       if row["document_id"] == document_id) or 1,
        "body_markdown_sha256": sorted(
            {row["body_markdown_sha256"] for row in divergent
             if row["document_id"] == document_id}) or None,
    } for document_id, receipt in sorted(original_receipt.items())
        if document_id not in resolved_ids]
    extraction_quarantine += [{
        "schema": "aw.news-lane-extraction-quarantine.v1",
        "document_id": row["document_id"],
        "url": row["url"],
        "original_sha256": row["lineage"]["original_sha256"],
        "receipt_object_id": row["lineage"]["receipt_object_id"],
        "reason": "no_chunk_emitted_under_chunk_policy",
        "fetches": row["fetch_repeats"],
        "body_markdown_sha256": [row["body_markdown_sha256"]],
        "markdown_sha256": row["markdown_sha256"],
        "body_words": len(document_body(row).split()),
        "chunk_policy": _policy_binding(policy),
    } for row in chunkless]
    extraction_quarantine.sort(key=lambda row: (row["reason"],
                                                row["document_id"]))

    # One row per distinct emitted passage. `chunk_ordinal` is that passage's
    # position in the order its parent actually emitted them (the chunker's own
    # output order, carried through `chunk_point_ids`), renumbered across the
    # distinct set so a parent that emitted the same text twice — one point in
    # the collection — still describes a gapless 0..n-1 sequence.
    parent_of = {row["document_id"]: row for row in documents}
    chunk_rows = [{
        "schema": CHUNK_SCHEMA,
        "point_id": point_id,
        "document_id": row["document_id"],
        "url": row["url"],
        "chunk_ordinal": ordinal,
        "parent_markdown_sha256": row["parent_markdown_sha256"],
        "chunk_policy": _policy_binding(policy),
        "source_metadata": {key: parent_of[row["document_id"]][key]
                            for key in NEWS_KEYS},
    } for row in chunk_join_rows
        for ordinal, point_id in enumerate(row["chunk_point_ids"])]

    # The staged collection is read back, not assumed. Its ids live in the
    # SERVED identity space, so each chunk's served id is re-derived through
    # the deployed projector and the read-back has to agree: a staged point
    # this arm's rows do not account for refuses the seal outright.
    served_of_chunk, document_of_served, unprojectable = served_identities(
        gated_rows, arm)
    unknown = sorted(set(staged_point_ids) - set(document_of_served))
    if unknown:
        raise NewsLaneSealError(
            f"{len(unknown)} staged point ids are not produced by this arm's "
            f"gated rows under the deployed projector (first: {unknown[0]})")
    served_chunks: dict[str, list[dict]] = {}
    for row in chunk_rows:
        served = served_of_chunk.get(row["point_id"])
        if served is not None and served in staged_point_ids:
            served_chunks.setdefault(served, []).append(row)
    vector_rows = [{
        "schema": VECTOR_SCHEMA,
        "point_id": served,
        "document_id": document_of_served[served],
        "url": served_chunks[served][0]["url"] if served in served_chunks else None,
        "chunk_point_ids": sorted(row["point_id"] for row in
                                  served_chunks.get(served, [])),
        "parent_markdown_sha256": (served_chunks[served][0]
                                   ["parent_markdown_sha256"]
                                   if served in served_chunks else None),
    } for served in sorted(staged_point_ids)]
    unstaged_rows = [{
        "schema": "aw.news-lane-unstaged-chunk.v1",
        "point_id": row["point_id"], "document_id": row["document_id"],
        "url": row["url"], "chunk_ordinal": row["chunk_ordinal"],
        "served_point_id": served_of_chunk.get(row["point_id"]),
        "reason": ("chunk_row_not_projectable"
                   if row["point_id"] not in served_of_chunk else
                   "served_point_absent_from_staged_collection"),
    } for row in chunk_rows
        if served_of_chunk.get(row["point_id"]) not in staged_point_ids]

    seal = lambda name, rows, fmt: _seal_rows(  # noqa: E731 — one shape, many uses
        artifact_root=artifact_root, out_root=out_root, name=name, rows=rows,
        fmt=fmt, producer_version=producer_version)
    stages = {
        "original": seal("original", original_rows, "news-lane-original-jsonl"),
        "markdown": seal("markdown", markdown_rows, "news-lane-markdown-jsonl"),
        "outlinks": seal("outlinks", outlink_rows, "news-lane-outlinks-jsonl"),
        "chunk_join": seal("chunk-join", chunk_join_rows,
                           "news-lane-chunk-join-jsonl"),
        "chunks": seal("chunks", chunk_rows, "news-lane-chunks-jsonl"),
        "vectors": seal("vectors", vector_rows, "news-lane-vectors-jsonl"),
    }
    quarantine_receipt = None
    if extraction_quarantine:
        quarantine_receipt = seal(
            "extraction-quarantine", extraction_quarantine,
            "news-lane-extraction-quarantine-jsonl")["receipt"]
    if quarantine_rows:
        stages["non_admitted_responses"] = seal(
            "non-admitted-responses", quarantine_rows,
            "news-lane-non-admitted-jsonl")
    if repeat_fetches:
        stages["repeat_fetches"] = seal(
            "repeat-fetches", repeat_fetches, "news-lane-repeat-fetch-jsonl")
    unstaged_receipt = None
    if unstaged_rows:
        unstaged_receipt = seal(
            "unstaged-chunks", unstaged_rows,
            "news-lane-unstaged-chunk-jsonl")["receipt"]

    lineage = {
        "original_to_markdown": _edge(
            inputs=len(original_rows), parents=len(documents),
            quarantined=len(extraction_quarantine), outputs=len(documents),
            quarantine_receipt=quarantine_receipt),
        # Every parent that reaches `markdown` emitted at least one passage —
        # the ones that did not were quarantined at the edge above — so this
        # edge conserves with nothing held back.
        "markdown_to_chunks": _edge(
            inputs=len(documents), parents=len(documents), quarantined=0,
            outputs=len(chunk_rows), quarantine_receipt=None),
        # Several chunks can share one served identity (same canonical URL and
        # anchor), so the parents of this edge are the chunk rows that reached
        # the collection, not the number of points they collapsed into.
        "chunks_to_vectors": _edge(
            inputs=len(chunk_rows),
            parents=len(chunk_rows) - len(unstaged_rows),
            quarantined=len(unstaged_rows), outputs=len(vector_rows),
            quarantine_receipt=unstaged_receipt),
    }
    run = {
        "schema": RUN_SCHEMA,
        "status": RUN_STATUS,
        "arm": arm,
        "day": day,
        "profile": "news_article",
        "producer": {"id": PRODUCER_ID, "version": producer_version},
        "counts": {
            "preserved_responses": len(receipts),
            "article_parents": len(documents),
            "markdown_documents": len(documents),
            "admitted_originals": len(original_rows),
            "non_admitted_responses": len(quarantine_rows),
            "repeat_fetches": len(repeat_fetches),
            "quarantine_records": len(extraction_quarantine),
            "divergent_parent_documents": len(divergent_ids),
            "repeat_fetches_collapsed": sum(
                row["fetch_repeats"] - 1 for row in documents + chunkless),
            "outlink_rows": len(outlink_rows),
            "empty_outlink_parents": sum(
                1 for row in outlink_rows if row["is_empty"]),
            "emitted_points": len(point_ids),
            "repeated_points_within_documents": repeated_points,
            "chunkless_parents": len(chunkless),
            "chunk_records": len(chunk_rows),
            "vector_records": len(vector_rows),
            "unstaged_chunks": len(unstaged_rows),
            "unprojectable_gated_rows": unprojectable,
            "chunks_sharing_a_served_point": len(chunk_rows)
            - len(unstaged_rows) - len(vector_rows),
        },
        "chunk_policy": _policy_binding(policy),
        "stages": stages,
        "lineage": lineage,
    }
    pack = {
        "schema": PACK_SCHEMA,
        "source_profile": PROFILE,
        "arm": arm,
        "day": day,
        # `cytir.ops.audits.news_article_profile` reads exactly these two keys: the
        # declared policy (ADR-100) and the chunks stage it re-validates.
        "chunk_policy": {"path": CHUNK_POLICY_ARTIFACT,
                         "policy_id": policy.policy_id,
                         "sha256": policy.sha256},
        "chunks": stages["chunks"],
    }
    for name, value in (("RUN.json", run), ("PACK.json", pack)):
        path = out_root / name
        encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        if path.exists():
            if sha256_file(path) != hashlib.sha256(encoded).hexdigest():
                raise NewsLaneSealError(f"immutable run collision: {path}")
        else:
            atomic_publish(path, encoded)
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--arm", required=True,
                        choices=sorted({arm for _, arm in ARM_ROUTES}))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--day", required=True)
    parser.add_argument("--producer-version", required=True)
    parser.add_argument("--chunk-policy", type=Path, required=True,
                        help="the ADR-100 descriptor this lane's chunks are "
                             "bound to (data/chunk-policy-legacy-180w-o30.v1.json)")
    parser.add_argument("--staged-point-ids", type=Path, required=True,
                        help="one point id per line, read back from the arm's "
                             "staged collection")
    parser.add_argument("--gated-rows", type=Path, required=True,
                        help="the arm's gated JSONL, whose rows are re-projected "
                             "to recover each chunk's served identity")
    args = parser.parse_args()
    try:
        staged = {line.strip() for line in
                  args.staged_point_ids.read_text().splitlines() if line.strip()}
        if not staged:
            raise NewsLaneSealError(
                f"no staged point ids in {args.staged_point_ids}")
        run = seal_arm(spool=args.spool.resolve(),
                       artifact_root=args.artifact_root.resolve(),
                       arm=args.arm, out_root=args.out.resolve(),
                       day=args.day, producer_version=args.producer_version,
                       chunk_policy_path=args.chunk_policy.resolve(),
                       staged_point_ids=staged,
                       gated_rows=args.gated_rows.resolve())
    except (NewsLaneSealError, OSError, KeyError, ValueError) as exc:
        print(f"NEWS-LANE SEAL FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"arm": run["arm"], "status": run["status"],
                      "counts": run["counts"],
                      "chunk_policy": run["chunk_policy"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
