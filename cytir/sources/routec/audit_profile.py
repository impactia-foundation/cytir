#!/usr/bin/env python3
"""Deep profile audits of the sealed Route-C packs.

Four families: Datatracker review records, Datatracker Internet-Draft records,
Datatracker meeting/governance records (one pack per record kind) and W3C
technical reports. The three Datatracker audits share one skeleton; W3C keeps
its own because its parents and outlinks carry different identity fields.
The record counts a pack was sealed against are read from its per-pack
declaration (`cytir.extract.routec.declaration`); a missing declaration is a refusal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from cytir.acquire.artifact_io import sha256_file
from cytir.extract.routec import ietf_datatracker_reviews as reviews
from cytir.extract.routec import ietf_internet_drafts as drafts
from cytir.extract.routec import ietf_meeting_records as meetings
from cytir.extract.routec import w3c_technical_reports as w3c
from cytir.extract.routec.declaration import load_declaration
from cytir.sources import audit as shared

STRUCTURED = "structured_data_feed"
ARTIFACT_KEYS = {"response_body", "response_headers", "transport", "consent"}
REVIEW_SCOPE = "complete official Datatracker review record and full displayed review text"
DRAFT_SCOPE = ("complete Datatracker document record; "
               "linked draft text is not claimed as embedded")
COUNT_KEYS = ("route_urls", "route_reported_production_points")
DRAFT_COUNT_KEYS = COUNT_KEYS + (
    "source_parents", "source_parent_reported_production_points",
    "redirected_current_rfc_records", "redirected_current_rfc_reported_production_points")


def _replay_parent(row: dict, *, artifact_root: Path, label: str, schema: str,
                   source_id: str, id_prefix: str, scope: str,
                   render: Callable[[str, str, str, str], tuple[str, str, dict]],
                   logical: Callable[[Path, dict], bytes],
                   datatracker_hosts: bool, record_kind: str | None = None,
                   failure: str = "parent source replay failed") -> None:
    """Re-render one Datatracker parent from its preserved response and compare."""
    url = str(row.get("url") or "")
    final_url = str(row.get("canonical_url") or "")
    identity = row.get("parent_identity") or {}
    artifacts = identity.get("artifacts") or {}
    if set(artifacts) != ARTIFACT_KEYS:
        raise ValueError(f"{label} artifact inventory drift: {url}")
    body = logical(artifact_root, artifacts["response_body"])
    body_sha = hashlib.sha256(body).hexdigest()
    title, markdown, version = render(body.decode("utf-8", "replace"), url,
                                      final_url, body_sha)
    extraction = row.get("extraction") or {}
    hosts_ok = (not datatracker_hosts or (
        urlsplit(url).hostname == "datatracker.ietf.org"
        and urlsplit(final_url).hostname == "datatracker.ietf.org"))
    if (row.get("schema") != schema or row.get("source_id") != source_id
            or row.get("source_profile") != STRUCTURED
            or (record_kind is not None and row.get("record_kind") != record_kind)
            or row.get("source_record_id") != url
            or row.get("document_id") != hashlib.sha256(
                f"{id_prefix}\0{url}".encode()).hexdigest()
            or not hosts_ok
            or row.get("title") != title or row.get("markdown") != markdown
            or row.get("markdown_sha256") != hashlib.sha256(markdown.encode()).hexdigest()
            or row.get("parent_source_sha256") != body_sha
            or identity.get("body_sha256") != body_sha
            or identity.get("transport_kind") != "consent-gated-direct-after-cc-miss"
            or not (identity.get("terminal_receipt") or {}).get("sha256")
            or row.get("source_version") != version
            or not isinstance(row.get("reported_production_points"), int)
            or row.get("reported_production_points", 0) < 1
            or row.get("content_scope") != scope
            or extraction.get("full_prechunk_principal_content") is not True
            or extraction.get("source_text_truncation_applied") is not False):
        raise ValueError(f"{label} {failure}: {url}")


def validate_review_parent(row: dict, *, artifact_root: Path) -> None:
    _replay_parent(
        row, artifact_root=artifact_root, label="Datatracker review",
        schema="aw.ietf-datatracker-review-record-parent.v1",
        source_id=reviews.SOURCE_ID, id_prefix="ietf-review-record",
        scope=REVIEW_SCOPE, logical=reviews._logical, datatracker_hosts=True,
        render=lambda html, url, final, sha: reviews.render_review_record(
            html, url=url, final_url=final, body_sha256=sha))


def validate_draft_parent(row: dict, *, artifact_root: Path) -> None:
    _replay_parent(
        row, artifact_root=artifact_root, label="Datatracker draft",
        schema="aw.ietf-datatracker-draft-record-parent.v1",
        source_id=drafts.SOURCE_ID, id_prefix="ietf-draft-record",
        scope=DRAFT_SCOPE, logical=drafts._logical, datatracker_hosts=True,
        render=lambda html, url, final, sha: drafts.render_draft_record(
            html, url=url, final_url=final, body_sha256=sha))


def validate_meeting_parent(row: dict, *, kind: str, artifact_root: Path) -> None:
    config = meetings.CONFIG[kind]
    _replay_parent(
        row, artifact_root=artifact_root, label=f"Datatracker {kind}",
        schema=f"aw.ietf-datatracker-{config['slug']}-record-parent.v1",
        source_id=meetings.source_id(kind), id_prefix=f"ietf-{kind}-record",
        scope=config["scope"], logical=meetings._logical, datatracker_hosts=False,
        record_kind=kind, failure="parent replay failed",
        render=lambda html, url, final, sha: meetings.render_meeting_record(
            html, kind=kind, url=url, final_url=final, body_sha256=sha))


def _load_run(path: Path, *, label: str, schema: str, source_id: str,
              expected_counts: dict[str, int], record_kind: str | None = None) -> dict:
    run = json.loads(path.read_text())
    counts = run.get("counts") or {}
    if (run.get("schema") != schema or run.get("status") != "complete"
            or (record_kind is not None and run.get("record_kind") != record_kind)
            or run.get("source_id") != source_id
            or run.get("source_profile") != STRUCTURED
            or run.get("point_identity", {}).get("ledger_subtraction_authorized") is not False
            or any(counts.get(name) != value for name, value in expected_counts.items())):
        raise ValueError(f"{label} RUN conservation failed")
    return run


def _datatracker_audit(*, label: str, run: dict, parent_run_path: Path,
                       artifact_root: Path, pack_root: Path, registry_path: Path,
                       output_root: Path, n_parents: int, n_points: int,
                       validate: Callable[[dict], None], evidence_head: dict,
                       evidence_counts: Callable[..., dict],
                       methods: Callable[[int], dict[str, list[str]]],
                       strict_urls: bool) -> tuple[Path, Path]:
    """The shared Datatracker skeleton: replay parents, outlinks, pack, chunks."""
    counts = run["counts"]
    shared.validate_run_stages(run, artifact_root)
    parents = shared.stage_rows(artifact_root, run["stages"]["markdown"])
    outlinks = shared.stage_rows(artifact_root, run["stages"]["outlinks"])
    if len(parents) != n_parents or len(outlinks) != counts["outlink_rows_including_empty"]:
        raise ValueError(f"{label} parent/outlink count drift")
    by_id: dict[str, dict] = {}
    points = 0
    for row in parents:
        validate(row)
        points += row["reported_production_points"]
        if row["document_id"] in by_id:
            raise ValueError(f"duplicate {label} parent")
        by_id[row["document_id"]] = row
    if points != n_points:
        raise ValueError(f"{label} point conservation failed")
    linked, empty = shared.check_outlinks(
        outlinks, by_id, label=label, bind_source_url=strict_urls,
        exclusive=strict_urls)
    pack_path = pack_root / "PACK.json"
    pack = json.loads(pack_path.read_text())
    distribution = shared.pack_gate(pack, profile=STRUCTURED, documents=n_parents, label=label)
    chunks = shared.stage_rows(artifact_root, pack["chunks"])
    point_ids = shared.check_chunks(chunks, by_id, profile=STRUCTURED, label=label,
                                    bind_url=strict_urls)
    profile = json.loads(registry_path.read_text())["profiles"][STRUCTURED]
    checks = shared.profile_checks(profile, methods(len(empty)), label=label)
    evidence = {**evidence_head,
        "counts": evidence_counts(parents=parents, points=points, chunks=chunks,
                                  point_ids=point_ids, outlinks=outlinks,
                                  linked=linked, empty=empty),
        "point_identity": run["point_identity"],
        "parent_run_sha256": sha256_file(parent_run_path),
        "pack_sha256": sha256_file(pack_path), "token_distribution": distribution}
    return shared.publish_evidence(output_root, evidence, checks, label=label)


def _evidence_counts_reviews(*, parents, points, chunks, point_ids, outlinks, linked, empty):
    return {"parents": len(parents), "reported_parent_points": points,
            "chunks": len(chunks), "unique_point_ids": len(point_ids),
            "outlink_rows": len(outlinks), "linked_parents": len(linked),
            "empty_parents": len(empty)}


def audit_reviews(*, artifact_root: Path, parent_run_path: Path, pack_root: Path,
                  registry_path: Path, output_root: Path,
                  declaration: Path) -> tuple[Path, Path]:
    expected = load_declaration(declaration, source_id=reviews.SOURCE_ID, keys=COUNT_KEYS)
    urls, points = expected["route_urls"], expected["route_reported_production_points"]
    run = _load_run(parent_run_path, label="Datatracker review",
        schema="aw.routec-ietf-datatracker-review-record-run.v1",
        source_id=reviews.SOURCE_ID,
        expected_counts={"route_urls": urls, "route_reported_production_points": points,
                         "source_parents": urls, "direct_responses": urls})
    return _datatracker_audit(label="Datatracker review", run=run,
        parent_run_path=parent_run_path, artifact_root=artifact_root,
        pack_root=pack_root, registry_path=registry_path, output_root=output_root,
        n_parents=urls, n_points=points, strict_urls=True,
        validate=lambda row: validate_review_parent(row, artifact_root=artifact_root),
        evidence_head={"schema": "aw.routec-ietf-datatracker-review-profile-evidence.v1",
                       "source_id": reviews.SOURCE_ID},
        evidence_counts=_evidence_counts_reviews,
        methods=lambda empty: {
            "original": [
                f"Deep-rehashed all {urls} terminal official Datatracker review responses and transport receipts.",
                "Bound every review to its exact stable requested and current final URL identity.",
                "Verified response-body, header, transport, consent and terminal-receipt checksums."],
            "markdown": [
                f"Re-rendered all {urls} complete Datatracker records and full displayed review text.",
                "Preserved review name/version metadata and exact requested/final URL identity.",
                "Matched every admitted parent to its exact preserved response digest."],
            "chunks": [
                "Verified chunks never merge distinct Datatracker review records.",
                "Matched every chunk to its parent Markdown digest and response identity.",
                "Verified the pinned Datatracker review projection and contiguous parent-local ordinals.",
                "Verified the pinned tokenizer and every embedding text has at most 2,560 tokens."],
            "outlinks": [
                "Checked each relation retains exact review-record identity and source context.",
                "Checked every non-empty target resolves against the current final Datatracker URL.",
                f"Verified all {urls} parents are represented, including {empty} explicit empty outputs."],
            "vectors": []})


def audit_drafts(*, artifact_root: Path, parent_run_path: Path, pack_root: Path,
                 registry_path: Path, output_root: Path,
                 declaration: Path) -> tuple[Path, Path]:
    d = load_declaration(declaration, source_id=drafts.SOURCE_ID, keys=DRAFT_COUNT_KEYS)
    urls, route_points = d["route_urls"], d["route_reported_production_points"]
    records, record_points = d["source_parents"], d["source_parent_reported_production_points"]
    rfc, rfc_points = (d["redirected_current_rfc_records"],
                       d["redirected_current_rfc_reported_production_points"])
    run = _load_run(parent_run_path, label="Datatracker draft",
        schema="aw.routec-ietf-datatracker-draft-record-run.v1",
        source_id=drafts.SOURCE_ID,
        expected_counts={"route_urls": urls, "route_reported_production_points": route_points,
            "source_parents": records, "source_parent_reported_production_points": record_points,
            "redirected_current_rfc_records": rfc,
            "redirected_current_rfc_reported_production_points": rfc_points})
    redirected = run.get("redirected_current_rfc_records") or []
    if len(redirected) != rfc:
        raise ValueError("Datatracker draft RUN conservation failed")
    if (sum(row["reported_production_points"] for row in redirected) != rfc_points
            or any(row.get("disposition") != "current-response-is-RFC-record-not-draft-record"
                   or not urlsplit(row.get("final_url") or "").path.startswith("/doc/rfc")
                   for row in redirected)):
        raise ValueError("Datatracker RFC redirect classification failed")

    def counts(*, parents, points, chunks, point_ids, outlinks, linked, empty):
        return {"parents": len(parents), "reported_parent_points": points,
                "redirected_rfc_records": len(redirected),
                "redirected_rfc_reported_points": rfc_points,
                "chunks": len(chunks), "unique_point_ids": len(point_ids),
                "outlink_rows": len(outlinks), "linked_parents": len(linked),
                "empty_parents": len(empty)}
    return _datatracker_audit(label="Datatracker draft", run=run,
        parent_run_path=parent_run_path, artifact_root=artifact_root,
        pack_root=pack_root, registry_path=registry_path, output_root=output_root,
        n_parents=records, n_points=record_points, strict_urls=True,
        validate=lambda row: validate_draft_parent(row, artifact_root=artifact_root),
        evidence_head={"schema": "aw.routec-ietf-datatracker-draft-profile-evidence.v1",
                       "source_id": drafts.SOURCE_ID},
        evidence_counts=counts,
        methods=lambda empty: {
            "original": [
                f"Deep-rehashed all {urls:,} terminal official Datatracker responses and transport receipts.",
                "Bound each record to its requested draft identity, current final URL and displayed version.",
                f"Verified response and terminal-receipt checksums, including {rfc} explicit current-RFC redirects."],
            "markdown": [
                f"Re-rendered all {records:,} complete Datatracker document records with semantic field boundaries.",
                "Preserved draft name/version metadata and exact stable requested/final URL identity.",
                "Matched every admitted parent to its exact preserved response digest."],
            "chunks": [
                "Verified chunks never merge distinct Datatracker records.",
                "Matched every chunk to its parent Markdown digest and response identity.",
                "Verified the pinned Datatracker projection and contiguous parent-local ordinals.",
                "Verified the pinned tokenizer and every embedding text has at most 2,560 tokens."],
            "outlinks": [
                "Checked each relation retains exact draft-record identity and source context.",
                "Checked every non-empty target resolves against the current final Datatracker URL.",
                f"Verified all {records} parents are represented, including {empty} explicit empty outputs."],
            "vectors": []})


def audit_meeting_records(*, artifact_root: Path, parent_run_path: Path,
                          pack_root: Path, registry_path: Path, output_root: Path,
                          kind: str, declaration: Path) -> tuple[Path, Path]:
    sid = meetings.source_id(kind)
    expected = load_declaration(declaration, source_id=sid, keys=COUNT_KEYS)
    urls, points = expected["route_urls"], expected["route_reported_production_points"]
    label = f"Datatracker {kind}"
    run = _load_run(parent_run_path, label=label, record_kind=kind, source_id=sid,
        schema=f"aw.routec-ietf-datatracker-{kind}-record-run.v1",
        expected_counts={"route_urls": urls, "route_reported_production_points": points,
                         "source_parents": urls, "direct_responses": urls})

    def counts(*, parents, points, chunks, point_ids, outlinks, linked, empty):
        return {"parents": len(parents), "reported_parent_points": points,
                "chunks": len(chunks), "unique_point_ids": len(point_ids),
                "outlink_rows": len(outlinks), "empty_parents": len(empty)}
    return _datatracker_audit(label=label, run=run, parent_run_path=parent_run_path,
        artifact_root=artifact_root, pack_root=pack_root, registry_path=registry_path,
        output_root=output_root, n_parents=urls, n_points=points, strict_urls=False,
        validate=lambda row: validate_meeting_parent(
            row, kind=kind, artifact_root=artifact_root),
        evidence_head={"schema": f"aw.routec-ietf-datatracker-{kind}-profile-evidence.v1",
                       "source_id": sid, "record_kind": kind},
        evidence_counts=counts,
        methods=lambda empty: {
            "original": [
                f"Deep-rehashed all {urls} exact official Datatracker {kind} responses and transport receipts.",
                f"Bound every {kind} record to its stable requested and final URL identity.",
                "Verified body, header, transport, consent and terminal-receipt checksums."],
            "markdown": [
                f"Re-rendered complete Datatracker records and full displayed {kind} text.",
                f"Preserved exact {kind} record/version and requested/final URL identity.",
                "Matched every admitted parent to its exact preserved response digest."],
            "chunks": [
                f"Verified chunks never merge distinct Datatracker {kind} records.",
                "Matched every chunk to its parent Markdown digest and response identity.",
                "Verified the pinned projection and contiguous parent-local ordinals.",
                "Verified the pinned tokenizer and every embedding text has at most 2,560 tokens."],
            "outlinks": [
                f"Checked every relation retains exact {kind} identity and context.",
                "Checked every non-empty target resolves against the current final URL.",
                f"Verified all {urls} parents, including {empty} explicit empty outputs."],
            "vectors": []})


def validate_w3c_parent(row: dict) -> None:
    url = str(row.get("url") or "")
    markdown = str(row.get("markdown") or "")
    identity = row.get("parent_identity") or {}
    version = row.get("source_version") or {}
    extraction = row.get("extraction") or {}
    if (row.get("schema") != "aw.w3c-technical-report-parent.v1"
            or row.get("source_id") != w3c.SOURCE_ID
            or row.get("source_profile") != "software_docs"
            or row.get("source_record_id") != url
            or not row.get("document_id") or not w3c.is_technical_report_url(url)
            or not markdown.strip()
            or row.get("markdown_sha256") != hashlib.sha256(markdown.encode()).hexdigest()
            or row.get("parent_source_sha256") !=
                (identity.get("response_body") or {}).get("body_sha256")
            or not (identity.get("terminal_receipt") or {}).get("sha256")
            or not (identity.get("response_headers") or {}).get("body_sha256")
            or not (identity.get("transport") or {}).get("body_sha256")
            or not (identity.get("consent") or {}).get("body_sha256")
            or version.get("requested_path") != urlsplit(url).path
            or version.get("identity_basis") != "exact-requested-TR-path+preserved-response"
            or extraction.get("full_prechunk_principal_content") is not True
            or extraction.get("source_text_truncation_applied") is not False):
        raise ValueError(f"W3C technical-report parent identity/fidelity failed: {url}")


def audit_w3c(*, artifact_root: Path, parent_run_path: Path, pack_root: Path,
              registry_path: Path, output_root: Path) -> tuple[Path, Path]:
    """W3C reports derive their counts from the RUN (no pinned count, no declaration)."""
    run = json.loads(parent_run_path.read_text())
    counts = run.get("counts") or {}
    if (run.get("schema") != "aw.routec-w3c-technical-report-run.v1"
            or run.get("status") != "complete"
            or run.get("source_id") != w3c.SOURCE_ID
            or run.get("source_profile") != "software_docs"
            or run.get("point_identity", {}).get("ledger_subtraction_authorized") is not False
            or counts.get("route_urls") !=
                counts.get("technical_report_parents") + counts.get("residual_urls")
            or counts.get("route_reported_production_points") !=
                counts.get("technical_report_reported_production_points") +
                counts.get("residual_reported_production_points")):
        raise ValueError("W3C parent RUN identity/conservation failed")
    shared.validate_run_stages(run, artifact_root)
    parents = shared.stage_rows(artifact_root, run["stages"]["markdown"])
    outlinks = shared.stage_rows(artifact_root, run["stages"]["outlinks"])
    if len(parents) != counts["technical_report_parents"]:
        raise ValueError("W3C parent count drift")
    by_id: dict[str, dict] = {}
    for row in parents:
        validate_w3c_parent(row)
        if row["document_id"] in by_id:
            raise ValueError("duplicate W3C parent document ID")
        by_id[row["document_id"]] = row
    linked: set = set()
    empty: set = set()
    for row in outlinks:
        parent = by_id.get(str(row.get("document_id") or ""))
        if (parent is None or row.get("source_record_id") != parent["source_record_id"]
                or row.get("source_url") != parent["url"]
                or row.get("parent_markdown_sha256") != parent["markdown_sha256"]
                or row.get("parent_identity") != parent["parent_identity"]):
            raise ValueError("W3C outlink lost parent identity")
        if row.get("is_empty"):
            if row.get("target_url") is not None:
                raise ValueError("W3C empty outlink has a target")
            empty.add(parent["document_id"])
        else:
            if not row.get("target_url") or not row.get("anchor_or_context"):
                raise ValueError("W3C outlink lacks target/context")
            linked.add(parent["document_id"])
    if linked | empty != set(by_id) or linked & empty:
        raise ValueError("W3C outlink parent conservation failed")
    pack_path = pack_root / "PACK.json"
    pack = json.loads(pack_path.read_text())
    shared.pack_gate(pack, profile="software_docs",
                     documents=len(parents), label="W3C")
    chunks = shared.stage_rows(artifact_root, pack["chunks"])
    point_ids = shared.check_chunks(chunks, by_id, profile="software_docs",
                                    label="W3C", bind_url=True)
    profile = json.loads(registry_path.read_text())["profiles"]["software_docs"]
    n = len(parents)
    checks = shared.profile_checks(profile, {
        "original": [
            ("Bound each admitted report to its exact requested /TR/ version path, "
             "final URL, declared canonical when present, and preserved response identity."),
            (f"Deep-rehashed terminal receipt plus exact body, headers, transport and "
             f"consent artifacts for all {n} admitted reports.")],
        "markdown": [
            (f"Scanned all {n} complete prechunk principal-content Markdown "
             "parents with document-local heading structure."),
            "Checked exact /TR/ path/version identity and preserved canonical metadata.",
            "Recomputed every Markdown digest and bound every parent to the preserved body digest."],
        "chunks": [
            "Checked every chunk retains its W3C report path and a non-empty section heading.",
            "Matched every chunk to its parent Markdown digest and exact parent identity.",
            "Verified pinned tokenizer/policy, unique UUIDs, contiguous parent ordinals and max <=2560."],
        "outlinks": [
            "Checked each relation retains exact W3C report source path and response identity.",
            "Checked every non-empty target retains first-occurrence anchor/context.",
            f"Verified all {n} parents are represented, including {len(empty)} explicit empty parents."],
        "vectors": []}, label="W3C")
    evidence = {
        "schema": "aw.routec-w3c-technical-report-profile-evidence.v1",
        "source_id": w3c.SOURCE_ID,
        "counts": {"parents": n, "chunks": len(chunks),
                   "unique_point_ids": len(point_ids), "outlink_rows": len(outlinks),
                   "linked_parents": len(linked), "empty_parents": len(empty),
                   "residual_route_urls": counts["residual_urls"]},
        "point_identity": run["point_identity"],
        "source_parent_run_sha256": sha256_file(parent_run_path),
        "pack_sha256": sha256_file(pack_path),
        "profile_registry_sha256": sha256_file(registry_path),
        "parent_logical_sha256": {
            name: stage["logical_sha256"] for name, stage in run["stages"].items()},
        "chunk_logical_sha256": pack["chunks"]["logical_sha256"],
    }
    return shared.publish_evidence(output_root, evidence, checks, label="W3C")


FAMILIES = ("ietf-datatracker-reviews", "ietf-internet-drafts",
            "ietf-meeting-records", "w3c-technical-reports")


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog or "audit_profile.py", description=__doc__)
    parser.add_argument("family", choices=FAMILIES)
    for name in ("artifact-root", "parent-run", "pack-root", "profile-registry",
                 "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--declaration", type=Path,
                        help="per-pack count declaration; required except for W3C")
    parser.add_argument("--record-kind", choices=sorted(meetings.CONFIG),
                        help="ietf-meeting-records only")
    args = parser.parse_args(argv)
    common = dict(artifact_root=args.artifact_root.resolve(),
                  parent_run_path=args.parent_run.resolve(),
                  pack_root=args.pack_root.resolve(),
                  registry_path=args.profile_registry.resolve(),
                  output_root=args.output_root.resolve())
    if args.family == "w3c-technical-reports":
        evidence, checks = audit_w3c(**common)
    else:
        declaration = args.declaration.resolve() if args.declaration else None
        if args.family == "ietf-datatracker-reviews":
            evidence, checks = audit_reviews(**common, declaration=declaration)
        elif args.family == "ietf-internet-drafts":
            evidence, checks = audit_drafts(**common, declaration=declaration)
        else:
            if not args.record_kind:
                parser.error("ietf-meeting-records needs --record-kind")
            evidence, checks = audit_meeting_records(
                **common, kind=args.record_kind, declaration=declaration)
    print(json.dumps({"evidence": str(evidence), "checks": str(checks)}, indent=2))
    return 0
