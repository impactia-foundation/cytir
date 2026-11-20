#!/usr/bin/env python3
"""The public cut, as a mechanical export (refactor slice 1, task 696, 2026-10-03).

    cyt ops cut --out DIR                  # export exactly the allowlisted tree, or refuse
    cyt ops cut --check                    # same scan, write nothing
    cyt ops cut --verify TREE              # an exported tree equals its manifest
    cyt ops cut --out DIR --tighten        # lower the baseline to today's (never raises it); a PR
                                          # branch may only lower files it changed
    cyt ops cut --out DIR --tighten --scheduled   # the daily job: lower every file
    cyt ops cut --record-baseline          # first recording only (refuses if one exists)
    cyt ops cut --admit-moved FROM TO [--reason TEXT]   # a git-proven move carries its mentions
    cyt ops cut --admit-stored FILE KIND COUNT --reason TEXT   # a stored value, by a ruling

WHAT IT DECLARES. `data/public-cut.v1.json` is an ALLOWLIST. Every shipped path is named by a
rule in that file. A rule that matches nothing (and is not marked `optional`) is an error, so
the file cannot rot into a wish list. A path that a `never` pattern names (`pipeline/`,
`tasks/`, `plan.md`, files naming hosts) can never ship: a rule that selects one is refused as
a contradiction, with no baseline to hide behind. Working assumption Q8 (specs/
oss-refactor-scoreboard.md), not yet ruled by Tim.

WHAT IT REFUSES. A shipped text file that names a non-shipped path (`pipeline/`, `tasks/`),
carries a host or a path of one of our boxes, or carries a secret-shaped literal is a LEAK.
Leaks exist today, so `data/public-cut-leak-baseline.v1.json` records them per (kind, file)
with a count. It is a ratchet: a leak that is not in the baseline, or a count above it, is
refused with the list. A count below it passes (and `--tighten` records it). The baseline never
grows. The slices that remove the leaks (3, 4, 7, 12) lower it; the cut is real at zero.

THE ONE DECLARED WAY TO RAISE IT. A file that MOVES (say from the private tree into `cytir/`)
carries the mentions it already had; the ratchet sees a "new" file. `--admit-moved FROM TO`
raises the baseline for TO, and only when git shows FROM -> TO as a rename on this branch
against main, and only up to the counts FROM had on main. A new file, a copy, or a mention
the move gained is still refused. Each admission is recorded in the baseline's `admitted`
list (what, counts, date, reason, branch), which `--tighten` keeps. Hand edits are not the way.

THE SECOND DECLARED WAY: A STORED VALUE. Some mentions are stored enum values, not paths a stranger
must change (a hostname as a word inside a reason string that prod rows carry; renaming needs a
table version bump and a payload rewrite). `--admit-stored FILE KIND COUNT --reason TEXT` records
a `stored-value` entry (file, category, exact count, date, reason, branch) in `admitted`. The scan
subtracts the admitted count from that (kind, file) before the baseline comparison. The count is a
ceiling AND a floor: above it the extra mention is a new leak and refuses; below it the entry is
stale and refuses until it is re-admitted at the lower count. Only the mentions the ruling named
are admitted; every other mention in the file stays in the baseline. `--tighten` keeps the entry.

WHAT IT PRODUCES. A fresh tree (no git history, no timestamps, sorted) plus
`PUBLIC-CUT-MANIFEST.json`: the allowlist digest and every file with its sha256, so two exports
of the same commit are byte-identical and a stranger can verify a copy. Signing the manifest
belongs to slice 13.

The scan patterns are built from fragments so that this file does not itself trip them.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import pathlib
import re
import stat
import subprocess
import sys

APP_ROOT = pathlib.Path(__file__).resolve().parents[2]
ALLOWLIST_REL = "data/public-cut.v1.json"
BASELINE_REL = "data/public-cut-leak-baseline.v1.json"
MANIFEST_NAME = "PUBLIC-CUT-MANIFEST.json"
SCHEMA = "aw.public-cut.v1"
BASELINE_SCHEMA = "aw.public-cut-leak-baseline.v1"
MAX_SCAN_BYTES = 2_000_000
KINDS = ("pipeline_ref", "tasks_ref", "host_literal", "secret_literal")

# Patterns, spelled in pieces: a leak scanner that is itself a leak is a bad joke.
_P, _T = "pipe" + "line", "ta" + "sks"
_PATTERNS: dict[str, re.Pattern[str]] = {
    "pipeline_ref": re.compile(rf"(?<![\w.-]){_P}/[A-Za-z0-9_.-]+"
                               rf"|^\s*(?:from|import)\s+{_P}\b", re.M),
    "tasks_ref": re.compile(rf"(?<![\w.-]){_T}/[A-Za-z0-9_.-]+"),
    "host_literal": re.compile("|".join((
        "heim" + "dall", "play" + "ground", "/media/" + "nvme4t", "/mnt/" + "wwn-",
        # the host name; the public contact mailbox `<name>@<domain>` is not a host literal
        r"(?<![\w@])ben" + r"der(?![\w@])",
        "/ro" + "ot/aw", "/home/" + "claude", "your-" + "storagebox"))),
    "secret_literal": re.compile("|".join((
        r"AKIA[0-9A-Z]{16}", r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
        r"gh[pousr]_[A-Za-z0-9]{36,}", r"xox[abprs]-[A-Za-z0-9-]{10,}",
        r"(?i:(?:api[_-]?key|secret|passw(?:or)?d|token)[\"']?\s*[:=]\s*[\"'][A-Za-z0-9/+_-]{24,}[\"'])"))),
}

#: The publisher's identity (task 993, decision 1): a brand name is not infrastructure, so it is
#: removed before the host scan. Only a whole token counts: `<box>.<brand>` is still a host.
BRAND_TERMS: tuple[str, ...] = ("impact" + "IA", "impactia" + ".org", "websear" + ".ch")
_BRAND = re.compile(r"(?<![\w@.-])(?:" + "|".join(re.escape(t) for t in BRAND_TERMS) + r")(?![\w-])",
                    re.I)

#: Frozen labels that live INSIDE sealed receipts: (file, exact literal, reason). The scanner skips
#: ONLY that exact literal in that exact file, and reports how many it skipped as `exempt: N`.
#: A stale entry (file shipped, literal gone) is refused, so the list can only shrink. These
#: entries and the patterns above are the only split-string spellings in the package.
_FROZEN_REASON = ("frozen label inside sealed receipts; changing it changes receipt bytes "
                  "(tasks 739, 764)")
#: A retired producer path that the contracts file records under `producer_ids` with status
#: `retired-path` (no artefact carries it), as opposed to a `frozen-label` kept in sealed receipts.
_RETIRED_PATH_REASON = "retired producer path recorded in the contract (not a receipt label)"
#: A key name in a data file that cannot be renamed (not a receipt label, not a host).
_REGISTRY_KEY_REASON = ("registry key name in data/assembly/sealed-populations.v1.json, frozen label; "
                        "named, never resolved as a host (task 2099)")
#: A test that asserts a frozen label the package still writes (task 2250).
_DESCRIPTOR_LABEL_REASON = ("asserts the frozen producer label the acquire stage writes into legacy "
                            "descriptors")
#: A test INPUT: a pointer the code under test must reject (task 2264).
_TEST_INPUT_REASON = ("test input: a " + _T + "/ or " + _P + "/ path the code under test must handle or reject "
                      "(receipt bytes unaffected)")
#: A moved test asserts the old producer label or release layout as a VALUE (task 2249): the label is
#: frozen in sealed receipts, or the test builds a throwaway pre-move release tree in tmp_path.
_TEST_VALUE_REASON = ("value a moved test asserts: a producer label frozen in sealed receipts (changing it changes receipt bytes) or the pre-move release "
                      "layout it writes into a throwaway tmp_path tree (task 2249)")
FROZEN_RECEIPT_LABELS: tuple[tuple[str, str, str], ...] = (
    # task 1484: producer paths declared under producer_ids by the batched contract PRs
    ("data/source-family-contracts.v1.json", _P + "/acquire_official_bulk_source.py",
     _RETIRED_PATH_REASON),
    ("data/source-family-contracts.v1.json", _P + "/acquire_official_bulk_source_direct.py",
     _RETIRED_PATH_REASON),
    ("data/source-family-contracts.v1.json", _P + "/ingest_ctgov_psy.py",
     _RETIRED_PATH_REASON),
    ("data/source-family-contracts.v1.json", _T + "/426-scripts/fold_w3_scratch_into_ledger.py",
     _RETIRED_PATH_REASON),
    ("data/source-family-contracts.v1.json", _T + "/420-scripts/fold_swissmedic_pdf_into_ledger.py",
     _RETIRED_PATH_REASON),
    ("data/source-family-contracts.v1.json", _P + "/derive_routec_generic_point_witness.py",
     _RETIRED_PATH_REASON),
    ("data/source-family-contracts.v1.json", _P + "/derive_ietf_datatracker_draft_point_witness.py",
     _RETIRED_PATH_REASON),
    ("data/source-family-contracts.v1.json", _P + "/derive_w3c_point_witness.py",
     _RETIRED_PATH_REASON),
    ("data/source-family-contracts.v1.json", _T + "/426-scripts/materialize_scratch_locator_family.py",
     _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/426-scripts/seal_scratch_atomic_terminal_point_witness.py",
     _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/417-scripts/materialize_r0_astra_pdf_worker.py",
     _RETIRED_PATH_REASON),
    ("data/source-family-contracts.v1.json", _T + "/417-scripts/seal_r0_astra_pdf_terminal_point_witness.py",
     _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/417-scripts/materialize_iris_meta_current.py",
     _RETIRED_PATH_REASON),
    ("data/source-family-contracts.v1.json", _T + "/417-scripts/seal_r0_iris_terminal_point_witness.py",
     _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/417-scripts/materialize_dila_jade_candidates.py",
     _RETIRED_PATH_REASON),
    ("data/source-family-contracts.v1.json", _T + "/417-scripts/seal_ce_dila_terminal_point_witness.py",
     _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/417-scripts/resolve_r0_datatracker_canonical_identities.py",
     _RETIRED_PATH_REASON),
    # task 2232: frozen task-417 HTML lane producer labels declared under producer_ids
    ("data/source-family-contracts.v1.json", _T + "/417-scripts/compose_r0_astra_html_parent.py",
     _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/417-scripts/seal_r0_astra_html_terminal_point_witness.py",
     _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/417-scripts/compose_r0_swissmedic_html_parent.py",
     _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/417-scripts/seal_r0_swissmedic_html_terminal_point_witness.py",
     _FROZEN_REASON),
    # task 2238: frozen one-shot lane producer labels declared under producer_ids
    ("data/source-family-contracts.v1.json", _T + "/427-scripts/materialize_l1b.py",
     _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/477-scripts/materialize_adr092_direct_captures.py",
     _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/426-scripts/materialize_adr092_cc_parents.py",
     _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/482-scripts/build_cc_supplement_consent.py",
     _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/482-scripts/build_f2_offline_admission_consent.py",
     _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/494-scripts/seal_4crawl_subset_consent_evidence.py",
     _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/494-scripts/seal_4crawl_servingonly_frontier_subset.py",
     _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/482-scripts/seal_adr092_cc_chain_subset.py",
     _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/495-scripts/wrap_reused_task427_bridge_as_sweep.py",
     _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/495-scripts/rebind_bridge_reused_leg_and_chain.py",
     _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/464-scripts/redispose_crawl30_sweep.py",
     _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _P + "/mint_claim_set_point_witness.py",
     _FROZEN_REASON),
    ("cytir/corpus/stage_structured_markdown.py",
     _P + "/stage_structured_markdown_for_embedding.py", _FROZEN_REASON),
    ("cytir/news/seal_lane_stages.py", _P + "/seal_news_lane_stages.py", _FROZEN_REASON),
    ("cytir/extract/bfs_dam_current.py", _P + "/materialize_bfs_dam_current.py", _FROZEN_REASON),
    ("cytir/news/seal_source_evidence.py", _P + "/news_refresh_stage.sh", _FROZEN_REASON),
    ("cytir/news/seal_source_evidence.py", _P + "/consent_gate.py", _FROZEN_REASON),
    ("cytir/index/seal_content_pack_intent.py", _P + "/upsert_supersede.group_key", _FROZEN_REASON),
    # decision D12 (Tim 2026-10-25, task 1313): producer ids written into sealed receipts;
    # `portableize_pack` refuses a re-run whose receipt bytes differ, so the id cannot be renamed.
    ("cytir/embed/adapt_pack_to_stage.py", _T + "/417-scripts/adapt_pack_to_413a_stage.py", _FROZEN_REASON),
    ("cytir/embed/portableize_pack.py", _T + "/417-scripts/portableize_embedding_pack.py", _FROZEN_REASON),
    # task 1899: HN discussion producer ids sealed into stage receipts (`producer_id=`) by the live daily stage.
    ("cytir/sources/hn/discussion_ingest.py", _P + "/hn_discussion_ingest.py", _FROZEN_REASON),
    ("cytir/chunk/hn_discussion.py", _P + "/chunk_hn_discussion_records.py", _FROZEN_REASON),
    # task 1403: the Service-Public dump producer label is declared under producer_ids
    ("data/source-family-contracts.v1.json", _P + "/acquire_service_public_dump_preserved.py",
     _FROZEN_REASON),
    # task 1390: the ERIC inventory producer label is frozen in sealed receipts (PR #1147 keeps it
    # byte-for-byte as PRODUCER_ID); the contracts file declares it under producer_ids.
    ("data/source-family-contracts.v1.json", _P + "/acquire_eric_api_inventory.py", _FROZEN_REASON),
    # task 1400: the gesetze L1 rows carry producer id `<path>:parse_law_zip`; the contracts file
    # declares it under producer_ids. The plain path is a substring of both labels, so one entry
    # covers every occurrence.
    ("data/source-family-contracts.v1.json", _P + "/ingest_gesetze.py", _FROZEN_REASON),
    # task 1395: the RFC Editor stage receipts carry this producer id
    ("cytir/sources/rfc_editor/preserved.py", _P + "/acquire_rfc_editor_rsync_preserved.py", _FROZEN_REASON),
    # task 2138: the sealed technical-product-markdown stage receipts carry this producer id; the
    # ported module keeps it byte-for-byte as PRODUCER
    ("cytir/sources/technical_manufacturer/product_markdown.py",
     _P + "/materialize_technical_product_catalog.py", _FROZEN_REASON),
    # task 1404: the PubMed L1 stage receipts carry this producer id (task 417); the moved parser
    # keeps it byte-for-byte as PRODUCER
    ("cytir/extract/pubmed.py", _P + "/ingest_pubmed.py:iter_pubmed_articles", _FROZEN_REASON),
    # task 1409: the sitemap-ingest producer label is frozen in sealed receipts; the module moved in task 683
    ("data/source-family-contracts.v1.json", _P + "/sitemap_ingest.py", _FROZEN_REASON),
    # task 1386: the ERIC inventory stage receipts carry this producer id; a renamed id would seal
    # the same inventory to different bytes
    ("cytir/sources/eric/inventory.py", _P + "/acquire_eric_api_inventory.py", _FROZEN_REASON),
    # task 1386 (CI fix, 2026-10-28): the frozen golden receipt carries the same producer id
    ("data/samples/eric/golden/RUN.json", _P + "/acquire_eric_api_inventory.py", _FROZEN_REASON),
    # task 1497 (batch B: tasks 1395, 1392, 1402, 1405, 1411): the contracts file declares these producer
    # labels under producer_ids
    ("data/source-family-contracts.v1.json",
     _P + "/acquire_rfc_editor_rsync_preserved.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json",
     _P + "/materialize_eric_pdf_markdown.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json",
     _P + "/finalize_eric_pdf_materialization_workers.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json",
     _P + "/audit_eric_pdf_partition_preflight.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json",
     _P + "/acquire_justice_canada_laws_snapshot.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json",
     _P + "/materialize_justice_canada_residual.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json",
     _P + "/materialize_github_acquisition_parent.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json",
     _P + "/acquire_statcan_wds_preserved.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json",
     _P + "/materialize_context7_exact_paths.py", _FROZEN_REASON),
    # task 1724: the Justice Canada materializers stamp these ids into sealed stage receipts
    ("cytir/sources/justice_canada/render.py", _P + "/materialize_routec_justice_canada_laws.py", _FROZEN_REASON),
    ("cytir/sources/justice_canada/residual.py", _P + "/materialize_justice_canada_residual.py", _FROZEN_REASON),
    # task 1436: the KALI full-text producers stamp these ids into sealed stage receipts
    ("cytir/sources/kali/derive_production_markdown.py", _P + "/derive_kali_production_markdown.py", _FROZEN_REASON),
    ("cytir/sources/kali/combine_production_markdown.py", _P + "/combine_kali_production_markdown.py", _FROZEN_REASON),
    # task 1401: the technical-manufacturer catalog stage receipts carry this producer id; a renamed
    # id would seal the same run to different bytes (tests pin the sealed bytes)
    ("cytir/sources/technical_manufacturer/catalog.py",
     _P + "/acquire_technical_manufacturer_catalog.py", _FROZEN_REASON),
    # task 2250: the acquire stage writes this frozen producer label into legacy chunk descriptors
    ("tests/test_legacy_chunk_descriptor.py", _P + "/fetch_process.py",
     _DESCRIPTOR_LABEL_REASON),
    # task 2264: a pinned frozen producer label, and two fake pointers the ledger must reject
    ("tests/test_aw_corpus_materialize_github_parent.py",
     _P + "/materialize_github_acquisition_parent.py", _FROZEN_REASON),
    # task 2264: tests OF the leak machinery need private-path inputs; one row per literal
    ("tests/test_aw_contract_module_ids.py", _P + "/aw_b1_license.py", _TEST_INPUT_REASON),
    ("tests/test_aw_contract_module_ids.py", _P + "/x.py", _TEST_INPUT_REASON),
    ("tests/test_aw_contract_module_ids.py", _P + "/some_tool.py", _TEST_INPUT_REASON),
    ("tests/test_aw_contract_module_ids.py", _P + "/y.py", _TEST_INPUT_REASON),
    ("tests/test_aw_public_cut_strip.py", _T + "/12-note.md", _TEST_INPUT_REASON),
    ("tests/test_aw_public_cut_strip.py", _T + "/9-a.md", _TEST_INPUT_REASON),
    ("tests/test_aw_public_cut_strip.py", _T + "/5-probe.jsonl#a", _TEST_INPUT_REASON),
    ("tests/test_aw_public_cut_strip.py", _T + "/1-x.md", _TEST_INPUT_REASON),
    ("tests/test_aw_public_cut_strip.py", _P + "/x.py", _TEST_INPUT_REASON),
    ("tests/test_aw_public_cut_strip.py", _P + "/p.py", _TEST_INPUT_REASON),
    ("tests/test_aw_contract_producer_ids.py", _T + "/417-scripts/x.py", _TEST_INPUT_REASON),
    # task 1394: the official bulk-dump stage receipts carry these producer ids
    ("cytir/sources/official_bulk/bulk.py", _P + "/acquire_official_bulk_source.py", _FROZEN_REASON),
    ("tests/test_aw_family_official_bulk.py", _P + "/acquire_official_bulk_source.py", _FROZEN_REASON),
    ("tests/test_aw_family_official_bulk.py", _P + "/acquire_official_bulk_source_direct.py",
     _FROZEN_REASON),
    ("cytir/sources/official_bulk/direct.py", _P + "/acquire_official_bulk_source_direct.py",
     _FROZEN_REASON),
    # task 2249: package tests moved out of the private tree that assert the old label as a value
    ("tests/test_aw_news_t4_sitemap_characterization.py",
     _P + "/sitemap_ingest.py", _TEST_VALUE_REASON),
    ("tests/test_aw_news_t4_discovery_characterization.py",
     _P + "/discovery_ingest.py", _TEST_VALUE_REASON),
    ("tests/test_aw_slice3b_openlegaldata.py",
     _P + "/acquire_openlegaldata_hf_dump.py", _TEST_VALUE_REASON),
    ("tests/test_aw_slice3b_openlegaldata.py",
     _P + "/materialize_openlegaldata.py", _TEST_VALUE_REASON),
    ("tests/test_aw_slice3b_stackoverflow.py",
     _P + "/materialize_stackoverflow_qa_production_tranche.py", _TEST_VALUE_REASON),
    ("tests/test_aw_slice3b_stage_receipt.py",
     _P + "/materialize_x.py", _TEST_VALUE_REASON),
    # task 1392: the ERIC materialization receipts carry these producer ids; a renamed id would seal
    # the same inputs to different bytes
    ("cytir/sources/eric/materialize.py", _P + "/materialize_eric_pdf_markdown.py", _FROZEN_REASON),
    ("cytir/sources/eric/finalize_workers.py",
     _P + "/finalize_eric_pdf_materialization_workers.py", _FROZEN_REASON),
    ("cytir/ops/eric_partition_preflight.py", _P + "/audit_eric_pdf_partition_preflight.py",
     _FROZEN_REASON),
    # task 1402: the Justice Canada snapshot stage receipt carries this producer id; a renamed id would
    # seal the same snapshot to different bytes. (The contracts-file entries ride with the contract PR: a
    # label absent from the shipped file is a stale entry the cut refuses.)
    ("cytir/sources/justice_canada/snapshot.py", _P + "/acquire_justice_canada_laws_snapshot.py", _FROZEN_REASON),
    # task 1950: the truncated-CC recovery stage receipts carry this producer id; a renamed id would
    # seal the same recovery to different bytes
    ("cytir/news/truncated_cc_recovery.py", _P + "/acquire_news_truncated_cc_recovery.py", _FROZEN_REASON),
    # task 1523: the FINMA PDF pilot stage receipts carry this producer id (3 stages); a renamed id would
    # seal the same pilot to different bytes
    ("cytir/sources/finma/materialize_pilot.py", _P + "/materialize_finma_pdf_pilot.py", _FROZEN_REASON),
    # task 1509: the Stack Overflow tranche rows carry this producer id (extraction.producer); the label
    # stays the pre-move path so sealed rows keep their bytes
    ("cytir/sources/stackoverflow/materialize_production_tranche.py",
     _P + "/materialize_stackoverflow_qa_production_tranche.py", _FROZEN_REASON),
    # task 1405: the StatCan preserved-original stage receipts carry this producer id; the pinned
    # parity test (tests/test_aw_family_statcan_wds_preserved.py) fixes the sealed bytes
    ("cytir/sources/statcan/preserved.py", _P + "/acquire_statcan_wds_preserved.py", _FROZEN_REASON),
    # task 1411: the context7_exact materializer's PRODUCER label is written into the sealed stage receipts
    ("cytir/sources/context7/materialize.py", _P + "/materialize_context7_exact_paths.py", _FROZEN_REASON),
    # task 1594: the four derived-parent stage receipts carry this producer id
    ("cytir/corpus/materialize_github_parent.py", _P + "/materialize_github_acquisition_parent.py", _FROZEN_REASON),
    # task 1522: the OpenAlex pilot/partition stage receipts carry these producer ids; a renamed id would
    # seal the same rows to different bytes
    ("cytir/sources/openalex/materialize_pilot.py", _P + "/materialize_openalex_academic_pilot.py", _FROZEN_REASON),
    ("cytir/sources/openalex/finalize_partitions.py", _P + "/finalize_openalex_academic_partitions.py",
     _FROZEN_REASON),
    # task 1403: the Service-Public dump producer id is written into sealed stage receipts
    ("cytir/sources/service_public/dump.py", _P + "/acquire_service_public_dump_preserved.py",
     _FROZEN_REASON),
    # task 1435 (task 1316 / D6): the openlegaldata producer label is frozen in served rows; the
    # contracts file declares it under producer_ids (status frozen-label, code_home cytir/extract)
    ("data/source-family-contracts.v1.json", _P + "/materialize_openlegaldata.py", _FROZEN_REASON),
    # task 1883: the LODA current-IDs deriver writes this producer id into sealed stage receipts (loda_exact)
    ("cytir/extract/loda_current.py", _P + "/derive_loda_production_markdown.py", _FROZEN_REASON),
    # task 1437: the JORF full-text stage receipts carry these producer ids (sealed jorf_recent runs)
    ("cytir/sources/jorf/recover_archives.py", _P + "/recover_jorf_production_from_archives.py", _FROZEN_REASON),
    ("cytir/sources/jorf/full_markdown.py", _P + "/materialize_jorf_full_markdown.py", _FROZEN_REASON),
    ("cytir/sources/jorf/filter_full_parents.py", _P + "/filter_jorf_full_content_parents.py", _FROZEN_REASON),
    # task 1432: the Route-C materializer stage receipts carry these producer ids; a renamed id would
    # seal the same rows to different bytes (parity digests in tests/test_aw_family_routec_generic.py)
    ("cytir/sources/routec/preserved_html.py", _P + "/materialize_routec_preserved_html.py", _FROZEN_REASON),
    ("cytir/sources/routec/direct_preserved_html.py", _P + "/materialize_routec_direct_preserved_html.py", _FROZEN_REASON),
    # task 1410: the exact-URL preserved stage receipts carry these producer ids; the pinned parity
    # tests (tests/test_aw_family_exact_url_preserved.py) fix the sealed bytes
    ("cytir/sources/exact_url/cc_preserved.py", _P + "/acquire_exact_url_cc_preserved.py", _FROZEN_REASON),
    ("cytir/sources/exact_url/direct_preserved.py", _P + "/acquire_exact_url_direct_preserved.py", _FROZEN_REASON),
    # task 1401: the technical-manufacturer catalog producer label, declared under producer_ids
    ("data/source-family-contracts.v1.json", _P + "/acquire_technical_manufacturer_catalog.py", _FROZEN_REASON),
    # task 1404: the contracts file declares the PubMed L1 producer label under producer_ids (twice:
    # the block and its amendment row); the label is frozen in sealed L1 receipts (task 417)
    ("data/source-family-contracts.v1.json", _P + "/ingest_pubmed.py:iter_pubmed_articles", _FROZEN_REASON),
    # task 1408: the rights-canon crawl-driver path is declared (retired-path) under producer_ids.
    ("data/source-family-contracts.v1.json", _P + "/ingest_rights_pack.py", _RETIRED_PATH_REASON),
    # task 1413: the contracts file declares the SwissNAMES3D L1 producer id (frozen in the sealed
    # swisstopo-structured-v3-exact receipt, task 417) and the retired driver path, under producer_ids
    ("data/source-family-contracts.v1.json", _T + "/417-scripts/derive_swisstopo_structured_l1.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _P + "/ingest_swisstopo.py",
     _RETIRED_PATH_REASON),
    # task 1412: the bfs-pxweb producer labels (the retired flat driver; the sealed L1 receipt's
    # one-shot script id) are declared under producer_ids in the contracts file.
    ("data/source-family-contracts.v1.json", _P + "/ingest_bfs.py", _RETIRED_PATH_REASON),
    ("data/source-family-contracts.v1.json", _T + "/417-scripts/derive_bfs_structured_l1.py",
     _FROZEN_REASON),
    # task 1410: the contracts file declares the exact-URL preserved producer labels under producer_ids
    ("data/source-family-contracts.v1.json", _P + "/acquire_exact_url_cc_preserved.py", _RETIRED_PATH_REASON),
    ("data/source-family-contracts.v1.json", _P + "/acquire_exact_url_direct_preserved.py", _RETIRED_PATH_REASON),
    ("data/source-family-contracts.v1.json", _P + "/derive_ca_canadaca_point_witness.py", _RETIRED_PATH_REASON),
    ("data/source-family-contracts.v1.json", _T + "/417-scripts/seal_r0_cochrane_terminal_point_witness.py", _RETIRED_PATH_REASON),
    ("data/source-family-contracts.v1.json", _T + "/417-scripts/seal_r0_cochrane_direct_terminal_point_witness.py", _RETIRED_PATH_REASON),
    # task 1414: producer_ids key of scratch_de_laender_juris_exact (status retired-path)
    ("data/source-family-contracts.v1.json", _P + "/ingest_de_laender_juris.py", _RETIRED_PATH_REASON),
    # task 1432: the contracts file declares the Route-C materializer producer paths under producer_ids
    ("data/source-family-contracts.v1.json", _P + "/materialize_routec_preserved_html.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _P + "/materialize_routec_direct_preserved_html.py", _FROZEN_REASON),
    # task 1436: producer_ids keys of kali_exact (status frozen-label, ids in sealed stage receipts)
    ("data/source-family-contracts.v1.json", _P + "/derive_kali_production_markdown.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _P + "/combine_kali_production_markdown.py", _FROZEN_REASON),
    # task 1437: producer_ids keys of jorf_recent (status frozen-label, orchestrator ruling)
    ("data/source-family-contracts.v1.json", _P + "/materialize_jorf_full_markdown.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _P + "/recover_jorf_production_from_archives.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _P + "/filter_jorf_full_content_parents.py", _FROZEN_REASON),
    # task 1524: technical-product producer ids and route labels written into sealed receipts
    ("cytir/ops/audits/technical_product_admission.py",
     _P + "/audit_technical_product_admission.py", _FROZEN_REASON),
    ("cytir/sources/technical_manufacturer/seeds.py",
     _P + "/prepare_technical_manufacturer_seeds.py", _FROZEN_REASON),
    ("cytir/sources/technical_manufacturer/catalog_frontier.py",
     _P + "/build_technical_manufacturer_catalog_frontier.py", _FROZEN_REASON),
    ("cytir/sources/technical_manufacturer/catalog_frontier.py",
     "heim" + "dall-direct-quarantine", _FROZEN_REASON),
    ("cytir/sources/technical_manufacturer/catalog_frontier.py",
     "heim" + "dall-residential-direct-after-cc-miss", _FROZEN_REASON),
    # task 1396: producer_ids keys of finma_circulars and finma_exact (status retired-path)
    ("data/source-family-contracts.v1.json", _P + "/ingest_finma_circulars.py", _RETIRED_PATH_REASON),
    ("data/source-family-contracts.v1.json", _P + "/chain_finma_full_admission.sh", _RETIRED_PATH_REASON),
    # task 1587: the task-1551 drivers' producer labels are declared frozen-label in the contracts
    # file (producer_ids); the labels are written into sealed receipts, so the literals stay
    ("data/source-family-contracts.v1.json", _P + "/materialize_routec_justice_canada_laws.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _P + "/materialize_bfs_dam_current.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _P + "/materialize_aug14_news_current_articles.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _P + "/acquire_news_truncated_cc_recovery.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _P + "/ingest_cantonal_cc.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _P + "/ingest_zhlex.py", _FROZEN_REASON),
    # task 1704: the CA-store news lane producers (task 431) are declared under producer_ids
    ("data/source-family-contracts.v1.json", _T + "/431-scripts/materialize_ca_store_news_document.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/431-scripts/close_ca_store_news_admission.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/431-scripts/seal_ca_store_news_terminal_point_witness.py", _FROZEN_REASON),
    # task 2099: the registry key that names the artifact root; the data file cannot be renamed
    ("cytir/index/assemble_plan.py", "heim" + "dall_path", _REGISTRY_KEY_REASON),
    # task 2531: the task-2530 producer_ids labels (frozen one-shots, declared under producer_ids in
    # source-family-contracts); named, never opened
    ("data/source-family-contracts.v1.json", _T + "/513-scripts/mint_family_claim_fragments.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/417-scripts/census_r0_remainder.py", _FROZEN_REASON),
    # task 2533: the two older producer_ids labels (task 2490) the census still counted undeclared
    ("data/source-family-contracts.v1.json", _T + "/299-scripts/build299_retailer_cards.py", _FROZEN_REASON),
    ("data/source-family-contracts.v1.json", _T + "/417-scripts/seal_task413a_wikipedia_point_witness.py", _FROZEN_REASON),
)

#: The `data/` files that `cytir/` names in code and the cut does NOT ship (task 2277). A stranger's copy
#: opens none of them unless an operator supplies it. ONE table; a row is (path, state, reason).
#:   degrades -- the reader meets the absent file with ONE named reason (never a silent pass); a test
#:               covers both branches.
#:   baseline -- named in code by a master-side build or ops verb, not yet audited for the absent-file
#:               branch (task 2277 follow-up). The ratchet: this state may only shrink. A row whose
#:               file ships, or that no code names any more, is refused as stale.
_ASM = ("index-assembly input: a per-build plan, census or ledger the master's assemble verb reads for one "
        "collection; a stranger's rebuild declares its own (unaudited absent-file branch)")
_EVAL = ("master-side eval or gate slice (held-out queries and golds), read by an operator verb and never at "
         "serve time (unaudited absent-file branch)")
_GATE = ("gate contract, slice or verdict for the master's flip and daily-news gates; the stranger's gates run "
         "on the shipped contracts (unaudited absent-file branch)")
_CADENCE = ("generated refresh-cadence table, written by the refresh verb and read back by the compose steps "
            "(unaudited absent-file branch)")
_REPO = ("master-side repo ledger or census read by an ops verb on the private tree; no serving path reads it "
         "(unaudited absent-file branch)")
_ASM_BUILD = ("per-build assembly input read by `cyt ops build-assembly-plan` under --root; absent, it exits 2 with "
              "the named reason `per-build assembly input <path> not found under <root>` (require_input); a "
              "stranger's rebuild supplies its own tree")
_ASM_LOADER = ("per-build assembly record read by the assemble loader; absent, it prints ONE named stderr note "
               "(`assemble: <path> not found -> no extras|no exclusions`) and continues with the empty default")
OPTIONAL_DATA_READS: tuple[tuple[str, str, str], ...] = (
    ("data/refresh_manifest.json", "degrades",
     "operator-supplied: freshness_ingest and weekly_delta take it from a REQUIRED --manifest (no default is opened), "
     "the literal in code is a provenance label or a basename key; it holds model-authored text, which authored_text_guard "
     "refuses as page content, and one host literal, so it stays private; scan-manifest names the missing file and exits 2"),
    ("data/assembly/assembly-plan.current.v1.json", "degrades", _ASM_BUILD),
    ("data/assembly/cancellation-track1a-retained-bind.v1.json", "degrades", _ASM_BUILD),
    ("data/assembly/claim-manifests.v1.json", "degrades", _ASM_BUILD),
    ("data/assembly/fold-report.v1.json", "degrades", _ASM_BUILD),
    ("data/assembly/g4-task472-cc-url-suppression-supplement.v1.json", "degrades", _ASM_BUILD),
    ("data/assembly/g4-tombstone-suppression.v1.json", "degrades", _ASM_BUILD),
    ("data/assembly/g4-tombstone-url-suppression.v1.json", "degrades", _ASM_BUILD),
    ("data/assembly/g6-mixed-url-resolution.v1.json", "degrades", _ASM_BUILD),
    ("data/assembly/g7-convergent-identity-rows.v2.json", "degrades", _ASM_BUILD),
    ("data/assembly/pack-route-exclusions.v1.json", "degrades", _ASM_LOADER),
    ("data/assembly/pack-route-extra-sources.v1.json", "degrades", _ASM_LOADER),
    ("data/assembly/reseal-task477-direct-captures.v1.json", "degrades", _ASM_BUILD),
    ("data/eval.yaml", "baseline", _EVAL),
    ("data/card-waivers.v1.jsonl", "degrades",
     "sealed per-id waivers of the master's card guard; `load_waivers` treats an absent default file as no waivers "
     "(the strict case) and refuses with CardWaiverError when a manifest pins its hash and the file is absent"),
    ("data/eval_tasks.v2.yaml", "baseline",
     "master-side 629 chooser set, a pinned yardstick of the flip gate; the retune, serve-parity, leg-read and "
     "flip-gate readers each refuse with a named missing-file message (never a pass), run in an exported cut"),
    ("data/gates/acceptance_probes.v1.json", "degrades",
     "gate probe fixture of the corpus-arm A/B; `compare_corpus_arms` prints `MEASUREMENT ERROR: probe fixture "
     "missing` and exits 3, `run_corpus_arm` exits with `REFUSED: probe fixture missing`: the gate refuses, never passes"),
    ("data/production-rebuild-lock.v1.json", "degrades",
     "the master's blocked rebuild manifest, a gate input: the flip gate reports `no global preservation manifest` "
     "and refuses, `index apply` raises ContractError naming the file; the drain verbs read it under a REQUIRED "
     "operator-supplied --release-dir, so no default path of a stranger's tree opens it"),
    ("data/entry-doors.v1.json", "degrades",
     "operator registry of the full tree (task 2894); `cyt ops entry-doors --check` prints 'registry file not shipped "
     "in this tree' and returns 0 in a cut"),
    ("data/change-log-writers.v1.json", "degrades",
     "operator registry of the full tree (task 2849); no verb opens it: `load_registry` is a library read by the "
     "writer-registry guard test of the full tree, and `cytir/ops/change_log_writers.py` imports without it (task 2903)"),
    ("data/eval_github_docs_gold.v1.jsonl", "degrades",
     "master-side gold set; `cyt ops github-docs-gold` prints `REFUSED: no gold set at <path> (the public cut ships "
     "none)` and returns 1 (task 2903, run in an exported cut)"),
    ("data/spam-host-lists.v1.json", "degrades",
     "sealed hold-out host lists of the master's tranche pre-screen; `cc_tranche_delta` prints `REFUSED: spam host "
     "lists <path> not found` and returns before any work when the default is absent (task 2903)"),
    ("data/test-suite-roster.v2.jsonl", "degrades",
     "operator roster of the full tree; its only reader `cytir/ops/test_suite_roster.py` is itself not shipped in the "
     "cut, so no stranger's command opens it (task 2903)"),
    ("data/experiments-ledger.v1.jsonl", "degrades",
     "operator ledger of the full tree; `cyt ops experiments-ledger` prints `ledger not shipped in this tree` and "
     "returns 0 when the default file is absent (task 2903)"),
    ("data/lessons-ledger.v2.jsonl", "degrades",
     "operator ledger of the full tree; `cyt ops lessons` prints `ledger not shipped in this tree` and returns 0 "
     "when the default file is absent (task 2903)"),
    ("data/eval/eval-civic.yaml", "baseline", _EVAL),
    ("data/eval/eval-coding.yaml", "baseline", _EVAL),
    ("data/eval/eval-entity.yaml", "baseline", _EVAL),
    ("data/eval/eval-legal.yaml", "baseline", _EVAL),
    ("data/eval/eval-reference.yaml", "baseline", _EVAL),
    ("data/eval/eval-research.yaml", "baseline", _EVAL),
    ("data/eval_agentic.yaml", "baseline", _EVAL),
    ("data/eval_authoritative.yaml", "baseline", _EVAL),
    ("data/eval_commercial.yaml", "baseline", _EVAL),
    ("data/eval_edgecases.yaml", "baseline", _EVAL),
    ("data/eval_entity_card_golds.yaml", "baseline", _EVAL),
    ("data/eval_entity_cards.yaml", "baseline", _EVAL),
    ("data/eval_foundation.yaml", "baseline", _EVAL),
    ("data/eval_freshness_heldout.yaml", "baseline", _EVAL),
    ("data/eval_general.yaml", "baseline", _EVAL),
    ("data/eval_jurisdiction.yaml", "baseline", _EVAL),
    ("data/eval_legal_authority.yaml", "baseline", _EVAL),
    ("data/eval_names.yaml", "baseline", _EVAL),
    ("data/eval_realistic.yaml", "baseline", _EVAL),
    ("data/eval_relevance_vs_trust.yaml", "baseline", _EVAL),
    ("data/eval_tasks.yaml", "baseline", _EVAL),
    ("data/eval_vocab_robustness.yaml", "baseline", _EVAL),
    ("data/eval_zero_signal.yaml", "baseline", _EVAL),
    ("data/flip-gate-verdict.v1.json", "baseline", _GATE),
    ("data/gates/scope_edges.json", "baseline", _EVAL),
    ("data/growth-pending-sets.v1.json", "baseline", _ASM),
    ("data/mechanism-callers.v1.json", "baseline", _REPO),
    ("data/mirror-audit-declared.v1.json", "baseline", _REPO),
    ("data/news-daily-gate-slice.v1.json", "baseline", _GATE),
    ("data/oss-refactor-live-entrypoints.v1.json", "baseline", _REPO),
    ("data/pipeline-reach-census.v1.json", "baseline", _REPO),
    ("data/production-flip-ruling-aw_e8.v1.json", "baseline", _ASM),
    ("data/r0-1024-coverage-census.v1.json", "baseline", _ASM),
    ("data/r0-427-collision-ledger-adr101c-supplement.v1.json", "baseline", _ASM),
    ("data/r0-427-collision-ledger.v1.json", "baseline", _ASM),
    ("data/refresh-cadence.v1.json", "baseline", _CADENCE),
)


def _const_strs(node) -> list[str]:
    """String constants under `node` in SOURCE order (ast.walk is breadth-first: it yields the right operand of
    `A / "data" / "x.json"` before the left one, which hid the writer registry from this guard, task 2903)."""
    cs = [c for c in ast.walk(node) if isinstance(c, ast.Constant) and isinstance(c.value, str)]
    return [c.value for c in sorted(cs, key=lambda c: (c.lineno, c.col_offset))]


_JOIN_NAMES = ("join", "joinpath", "Path", "PurePath", "PosixPath")


def _is_path_join(n) -> bool:
    """`a / b`, `os.path.join(..)`, `p.joinpath(..)`, `Path("data", ..)`."""
    if isinstance(n, ast.BinOp):
        return isinstance(n.op, ast.Div)
    if isinstance(n, ast.Call):
        f = n.func
        return (getattr(f, "attr", None) or getattr(f, "id", "")) in _JOIN_NAMES
    return False


_DATA_EXT = (".json", ".jsonl", ".yaml", ".yml", ".csv", ".txt", ".tsv")
_DATA_LITERAL = re.compile(r"(?<![\w./-])data/([\w.\-/*]+\.(?:json|jsonl|yaml|yml|csv|txt|tsv))")


def data_literals(root: pathlib.Path = APP_ROOT) -> dict[str, list[str]]:
    """{`data/...` path: [cytir/ files that name it in CODE]}. Docstrings and comments are not code. A path
    is named as one literal (`"data/x.json"`), as joined parts (`ROOT / "data" / "x.json"`,
    `os.path.join(r, "data", "x.json")`) or as `f"{DATA}/x.json"`. Over-inclusive on purpose: a provenance
    label that names the file is declared too, so a reader cannot hide behind being 'just a label'."""
    found: dict[str, set[str]] = {}
    for p in sorted((root / "cytir").rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel == "cytir/ops/public_cut.py":      # the contract's own module names files to declare them
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        docs = {id(n.body[0].value) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        for n in ast.walk(tree):
            hits: list[str] = []
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
                hits += ["data/" + m.group(1) for m in _DATA_LITERAL.finditer(n.value)]
            elif isinstance(n, ast.JoinedStr):
                for i, part in enumerate(n.values):
                    if isinstance(part, ast.Constant) and isinstance(part.value, str) and i and isinstance(
                            n.values[i - 1], ast.FormattedValue):
                        fv = n.values[i - 1].value
                        if isinstance(fv, ast.Name) and fv.id.upper() in ("DATA", "DATA_DIR"):
                            m = re.match(r"/([\w.\-/]+\.(?:json|jsonl|yaml|yml|csv|txt|tsv))", part.value)
                            if m:
                                hits.append("data/" + m.group(1))
            elif _is_path_join(n):
                ss = _const_strs(n)
                if "data" in ss and ss.index("data") + 1 < len(ss) and ss[-1].endswith(_DATA_EXT):
                    hits.append("data/" + "/".join(ss[ss.index("data") + 1:]))
            for h in hits:
                found.setdefault(h, set()).add(rel)
    return {k: sorted(v) for k, v in sorted(found.items())}


def unshipped_data_reads(root: pathlib.Path = APP_ROOT) -> tuple[list[str], list[str]]:
    """(problems naming the cytir/ file, stale OPTIONAL_DATA_READS rows). A problem is a tracked `data/` file
    that `cytir/` names, that the allowlist does not ship, and that OPTIONAL_DATA_READS does not declare.
    A name that matches no tracked file (an output, a glob nothing ships) is not a shipped-or-not question."""
    files = tracked_files(root)
    shipped, _ = select(load_allowlist(root), files)
    tracked = set(files)
    declared = {r[0] for r in OPTIONAL_DATA_READS}
    literals = data_literals(root)
    problems, used = [], set()
    for path, readers in literals.items():
        if _has_wild(path):
            ok = any(_matches(path, f) for f in shipped)
        else:
            ok = path in shipped or path not in tracked
        if ok:
            continue
        if path in declared:
            used.add(path)
        else:
            problems.append(f"{readers[0]}{' (+%d more)' % (len(readers) - 1) if len(readers) > 1 else ''} "
                            f"names {path}, which the cut does not ship and OPTIONAL_DATA_READS does not declare")
    # in the exported tree the declared files are absent by design: a row is judged only where it is tracked
    stale = [f"{r[0]}: {'shipped' if r[0] in shipped else 'no cytir/ code names it'}"
             for r in OPTIONAL_DATA_READS if r[0] not in used and (r[0] in shipped or r[0] in tracked)]
    return problems, stale


class CutRefusal(Exception):
    """The allowlist contradicts itself or a rule matches nothing: a config error, not a leak."""


# --------------------------------------------------------------------------- globs


def _glob_re(pat: str) -> re.Pattern[str]:
    """`**` crosses `/`, `*` and `?` do not. A pattern with no wildcard names one file."""
    out, i = [], 0
    while i < len(pat):
        if pat.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pat.startswith("**", i):
            out.append(".*")
            i += 2
        elif pat[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pat[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pat[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def _matches(pat: str, rel: str) -> bool:
    return bool(_glob_re(pat).match(rel))


def _has_wild(pat: str) -> bool:
    return any(c in pat for c in "*?")


# --------------------------------------------------------------------------- tree


def tracked_files(root: pathlib.Path) -> list[str]:
    """App-relative paths a clean clone holds (`git ls-files`); a plain walk when `root` is
    not a checkout (the exported tree itself)."""
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", "."],
                             check=True, capture_output=True).stdout
        rels = [p for p in out.decode().split("\0") if p]
        if rels:
            return sorted(rels)
    except (OSError, subprocess.CalledProcessError):
        pass
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                  if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts)


def load_allowlist(root: pathlib.Path = APP_ROOT) -> dict:
    path = root / ALLOWLIST_REL
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CutRefusal(f"cannot read the allowlist {ALLOWLIST_REL}: {e}") from e
    if doc.get("schema") != SCHEMA:
        raise CutRefusal(f"{ALLOWLIST_REL}: schema must be {SCHEMA!r}, got {doc.get('schema')!r}")
    for key in ("rules", "never", "scan_exempt"):
        if not isinstance(doc.get(key), list):
            raise CutRefusal(f"{ALLOWLIST_REL}: `{key}` must be a list")
    ids = [r.get("id") for r in doc["rules"]]
    if len(ids) != len(set(ids)) or not all(ids):
        raise CutRefusal(f"{ALLOWLIST_REL}: every rule needs a unique `id`")
    for r in doc["rules"]:
        if not r.get("why") or not r.get("paths"):
            raise CutRefusal(f"{ALLOWLIST_REL}: rule {r.get('id')!r} needs `why` and `paths`")
    return doc


def select(doc: dict, files: list[str]) -> tuple[dict[str, str], list[str]]:
    """({shipped path: rule id}, problems). The first rule that names a path owns it."""
    problems: list[str] = []
    never = [n["path"] if isinstance(n, dict) else n for n in doc["never"]]
    shipped: dict[str, str] = {}
    for rule in doc["rules"]:
        optional = bool(rule.get("optional"))
        for pat in rule["paths"]:
            hits = [f for f in files if _matches(pat, f)]
            if not hits:
                if not optional:
                    problems.append(f"rule {rule['id']}: `{pat}` matches no tracked file "
                                    "(declare it `optional` or remove it)")
                continue
            for f in hits:
                forbidden = next((n for n in never if _matches(n, f)), None)
                if forbidden is None:
                    shipped.setdefault(f, rule["id"])
                elif not _has_wild(pat):
                    problems.append(f"rule {rule['id']} names `{f}`, which `never` forbids "
                                    f"(`{forbidden}`)")
                # a wildcard that sweeps up a forbidden path simply does not ship it
    return shipped, problems



# --------------------------------------------------------------------------- derived copies (task 2223)

#: A shipped contract keeps its evidence trail in the master. The CUT removes the internal
#: pointers from the PROSE fields `data/public-cut.v1.json` `strip` declares (ruling 60(c),
#: log 2026-11-01 task 2213). The master is never edited; the shipped copy is a derived artifact.
#: Only a documentary field may be declared: the last key must be a PROSE key and no step of the
#: path may be a key a loader reads at runtime. Anything else is refused as a config error.
PROSE_KEYS = frozenset({"evidence", "basis", "note", "notes", "why", "recorded_in", "source",
                        "open_keys_why", "producers", "verdict", "consulted_by", "evaluated_by", "recorded", "status",
                        "stale_claim_note", "measured_sensitivity"})
RUNTIME_KEYS = frozenset({"consent", "producer_ids", "code_home", "cite", "instrument",
                          "role", "route", "route_allowed", "ruling_cite", "minted_by",
                          "declared_by", "dense_surface", "sparse_surface", "dense", "sparse",
                          "chunk_fields", "identity_fields", "family_id", "id", "sha256", "key"})
#: A pointer inside a sentence: a `<dir>/<name>` path of the two private trees, and the
#: parenthesised "(task 123)" form. The sentence keeps its words.
_POINTER = re.compile(rf"(?<![\w.-])(?:{_T}|{_P})/[A-Za-z0-9_.#:/-]+")
_TASK_PAREN = re.compile(r"\s*\((?:see |in )?tasks? \d+[a-z]?(?:[,/ ]+(?:and )?\d+[a-z]?)*\)")


def _check_strip_path(path: str) -> list[str]:
    segs = path.split(".")
    if not segs or not all(segs):
        raise CutRefusal(f"strip path {path!r} is empty or has an empty step")
    if segs[-1] not in PROSE_KEYS:
        raise CutRefusal(f"strip path {path!r}: `{segs[-1]}` is not a prose key "
                         f"({sorted(PROSE_KEYS)})")
    bad = [x for x in segs if x in RUNTIME_KEYS]
    if bad:
        raise CutRefusal(f"strip path {path!r} passes through {bad}, which a loader reads at "
                         "runtime: the cut refuses to declare it")
    return segs


#: Runtime keys whose pointers the cut may RELABEL (ruling 68(c), Tim 2026-11-04): `producer_ids`
#: (the KEYS of that map), `code_home` and `cite` (their values). A relabel never deletes a key;
#: it swaps a private-tree pointer for a neutral frozen label. Any other key is refused.
RELABEL_KEYS = frozenset({"producer_ids", "code_home", "cite"})
_TASK_POINTER = re.compile(rf"(?<![\w.-]){_T}/(\d+[a-z]?)-([A-Za-z0-9_.#:/-]+)")


def _check_relabel_path(path: str) -> list[str]:
    segs = path.split(".")
    if not segs or not all(segs):
        raise CutRefusal(f"relabel path {path!r} is empty or has an empty step")
    if segs[-1] not in RELABEL_KEYS:
        raise CutRefusal(f"relabel path {path!r}: `{segs[-1]}` is not a relabel key "
                         f"({sorted(RELABEL_KEYS)})")
    return segs


def frozen_label(text: str) -> str:
    """Swap every private-tree task pointer in `text` for `frozen:task-<N>:<stem>`; the key keeps its
    meaning (which task, which one-shot producer) and names nothing a stranger cannot find."""
    def one(m: re.Match) -> str:
        tail = m.group(2).rstrip(".;,")
        stem = tail.rsplit("/", 1)[-1].split("#", 1)[0]
        stem = stem.rsplit(".", 1)[0] if "." in stem else stem
        return f"frozen:task-{m.group(1)}:{stem}" if stem else f"frozen:task-{m.group(1)}"
    return _TASK_POINTER.sub(one, text)


def _relabel_node(node, segs: list[str]) -> int:
    """Apply one declared relabel path. Returns the number of keys or values changed."""
    n = 0
    if isinstance(node, list):
        for item in node:
            n += _relabel_node(item, segs)
        return n
    if not isinstance(node, dict):
        return 0
    head, rest = segs[0], segs[1:]
    if head == "*":
        for v in list(node.values()):
            n += _relabel_node(v, rest)
        return n
    if head not in node:
        return 0
    if rest:
        return _relabel_node(node[head], rest)
    val = node[head]
    if isinstance(val, str):
        new = frozen_label(val)
        if new != val:
            node[head] = new
            return 1
    elif isinstance(val, dict) and head == "producer_ids":
        pairs = getattr(val, "pairs", None) or list(val.items())
        renamed = [(frozen_label(k), v) for k, v in pairs]
        keys = [k for k, _v in renamed]
        if len(set(keys)) != len(keys):
            raise CutRefusal("relabel of producer_ids would make two keys equal")
        n = sum(1 for (a, _), (b, _) in zip(pairs, renamed) if a != b)
        if n:
            val.clear()
            val.update(renamed)
            if hasattr(val, "pairs"):
                val.pairs = renamed
    return n


def strip_rules(doc: dict) -> dict[str, dict]:
    """{file: {"fields": [path, ...], "relabel": [path, ...], "reseal": bool}} from `strip`, validated."""
    out: dict[str, dict] = {}
    for e in doc.get("strip", []):
        if not e.get("path") or not e.get("why") or not (e.get("fields") or e.get("relabel")):
            raise CutRefusal(f"{ALLOWLIST_REL}: every `strip` entry needs path, why, fields or relabel")
        if e["path"] in out:
            raise CutRefusal(f"{ALLOWLIST_REL}: `strip` names {e['path']!r} twice")
        for f in e.get("fields", []):
            _check_strip_path(f)
        for f in e.get("relabel", []):
            _check_relabel_path(f)
        out[e["path"]] = {"fields": list(e.get("fields", [])), "relabel": list(e.get("relabel", [])),
                          "reseal": bool(e.get("reseal"))}
    return out


def _is_bare_leak(x: str) -> bool:
    t = x.strip()
    return bool(_BARE_VALUE.match(t)) and bool(scan_text(t))


def _clean_sentence(x: str) -> str:
    return _POINTER.sub("", _TASK_PAREN.sub("", x)).replace("  ", " ").replace(" ,", ",").strip()


def _strip_node(node, segs: list[str]) -> int:
    """Apply one declared path; a list met on the way is walked item by item. Returns the number
    of values changed or dropped. A bare leaking item is dropped, a sentence keeps its words."""
    n = 0
    if isinstance(node, list):
        for item in node:
            n += _strip_node(item, segs)
        return n
    if not isinstance(node, dict):
        return 0
    head, rest = segs[0], segs[1:]
    if head == "*":
        for v in list(node.values()):
            n += _strip_node(v, rest)
        return n
    if head not in node:
        return 0
    if rest:
        return _strip_node(node[head], rest)
    val = node[head]
    if isinstance(val, str):
        if _is_bare_leak(val):
            del node[head]
            return 1
        new = _clean_sentence(val)
        if new != val:
            node[head] = new
            return 1
    elif isinstance(val, list):
        keep = []
        for item in val:
            if isinstance(item, str):
                if _is_bare_leak(item):
                    n += 1
                    continue
                new = _clean_sentence(item)
                n += new != item
                keep.append(new)
            else:
                keep.append(item)
        node[head] = keep
    return n


class _Obj(dict):
    """A JSON object that remembers every (key, value) pair in order, duplicates included: the
    master carries shadowed duplicate keys that a plain `dict` re-dump would silently drop. The
    dict part holds the LAST value per key (what every loader sees) and is the one the strip edits."""

    def __init__(self, pairs):
        super().__init__(pairs)
        self.pairs = list(pairs)


def _dump(x, level: int = 0) -> str:
    """json.dumps(indent=1, ensure_ascii=False) layout, but an `_Obj` writes its shadowed duplicates."""
    pad, inner = " " * level, " " * (level + 1)
    if isinstance(x, _Obj):
        if not x.pairs:
            return "{}"
        last = {k: i for i, (k, _v) in enumerate(x.pairs) if k in x}
        rows = [f"{inner}{json.dumps(k, ensure_ascii=False)}: "
                f"{_dump(x[k] if last[k] == i else v, level + 1)}" for i, (k, v) in enumerate(x.pairs) if k in x]
        return "{\n" + ",\n".join(rows) + f"\n{pad}}}"
    if isinstance(x, list):
        if not x:
            return "[]"
        return "[\n" + ",\n".join(f"{inner}{_dump(v, level + 1)}" for v in x) + f"\n{pad}]"
    return json.dumps(x, ensure_ascii=False)


def derive_text(text: str, rule: dict) -> tuple[str, int]:
    """(shipped text, values changed) for one declared file. Re-dumped JSON: the master's hand
    formatting is not kept, so a self-sealed file is re-sealed (`reseal`) after the strip."""
    doc = json.loads(text, object_pairs_hook=_Obj)
    changed = sum(_strip_node(doc, _check_strip_path(f)) for f in rule["fields"])
    changed += sum(_relabel_node(doc, _check_relabel_path(f)) for f in rule.get("relabel", []))

    def dump(d) -> str:
        return _dump(d) + "\n"
    if rule.get("reseal"):
        from cytir.sources.contract import pre_amendment_sha256
        _recorded, computed = pre_amendment_sha256(dump(doc))
        key = next(k for k, v in doc.items() if isinstance(v, list) and v
                   and isinstance(v[0], dict) and "old_file_sha256" in v[0])
        doc[key][-1]["pre_amendment_file_sha256"] = computed
        out = dump(doc)
        recorded, again = pre_amendment_sha256(out)
        if recorded != again:
            raise CutRefusal("the re-sealed copy does not verify")
        return out, changed
    return dump(doc), changed


# --------------------------------------------------------------------------- scan


def _is_text(data: bytes) -> bool:
    return b"\0" not in data[:4096]


def scan_text(text: str) -> dict[str, int]:
    text = _BRAND.sub(" ", text)
    return {k: n for k, p in _PATTERNS.items() if (n := len(p.findall(text)))}


#: Hosts are left out: the module's own patterns are spelled in pieces on purpose.
_SPLIT_KINDS = ("pipeline_ref", "tasks_ref")
def _split_hint(*names: str) -> re.Pattern[str]:
    """Cheap pre-filter for `scan_split_literals`: the file holds a name whole, or a name cut by a
    string boundary (closing quote, joiner, opening quote). Any other file is neither parsed nor
    walked: a name cut by anything else is not an all-constant string assembly."""
    cut = [re.escape(n[:i]) + "[\"'][\\s()+,]*[frbFRB]*[\"']" + re.escape(n[i])
           for n in names for i in range(1, len(n))]
    whole = [re.escape(n) + "/?[\"']" for n in names]    # a piece that ends with the name (and slash)
    return re.compile("|".join(whole + cut))


_SPLIT_HINT = _split_hint(_P, _T)
_JOINERS = {"os.path.join", "posixpath.join", "path.join"}
_PATHS = {"Path", "PurePath", "PurePosixPath", "pathlib.Path", "pathlib.PurePath",
          "pathlib.PurePosixPath"}


def _dotted(n: ast.AST) -> str:
    parts = []
    while isinstance(n, ast.Attribute):
        parts.append(n.attr)
        n = n.value
    if isinstance(n, ast.Name):
        parts.append(n.id)
        return ".".join(reversed(parts))
    return ""


def _const_value(n: ast.AST, env: dict[str, str] | None = None) -> str | None:
    """The string an ALL-CONSTANT assembly evaluates to, else None. Handles adjacent constants
    (the parser has already folded them), `+`, `"/".join(<tuple|list>)`, `os.path.join(...)`
    and `Path("a") / "b"` chains. Any non-constant part (`REPO / "x" / name`) gives None: that
    is the honest form of building a path."""
    if isinstance(n, ast.Constant):
        return n.value if isinstance(n.value, str) else None
    if isinstance(n, ast.Name):                     # a module-level all-constant binding
        return (env or {}).get(n.id)
    if isinstance(n, ast.JoinedStr):                # an f-string whose parts are all constant
        parts = []
        for e in n.values:
            if isinstance(e, ast.FormattedValue):
                if e.conversion != -1 or e.format_spec is not None:
                    return None
                e = e.value
            v = _const_value(e, env)
            if v is None:
                return None
            parts.append(v)
        return "".join(parts)
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
        a, b = _const_value(n.left, env), _const_value(n.right, env)
        return a + b if a is not None and b is not None else None
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
        a, b = _const_value(n.left, env), _const_value(n.right, env)
        return a.rstrip("/") + "/" + b if a is not None and b is not None else None
    if isinstance(n, ast.Call) and not n.keywords:
        f = _dotted(n.func)
        if (isinstance(n.func, ast.Attribute) and n.func.attr == "join" and len(n.args) == 1
                and isinstance(n.func.value, ast.Constant) and isinstance(n.func.value.value, str)
                and isinstance(n.args[0], (ast.Tuple, ast.List))):
            parts = [_const_value(e, env) for e in n.args[0].elts]
            return None if None in parts else n.func.value.value.join(parts)
        if f in _JOINERS or f in _PATHS:
            parts = [_const_value(e, env) for e in n.args]
            return "/".join(parts) if parts and None not in parts else None
    return None



def scan_split_literals(text: str, skip: tuple[str, ...] = ()) -> dict[str, int]:
    """Leaks hidden by splitting a literal in a Python file (`"pipe" "line/x"`, `"pipe" + "line/x"`,
    `"/".join(("pipe" "line", "x"))`, `Path("pipe" "line") / "x"`): the maximal all-constant
    assembly is evaluated and scanned like text. What the plain scan already sees in the assembly's
    own source is not counted twice. `skip` are verbatim registered labels."""
    if not _SPLIT_HINT.search(text):                # most files: no parse, no walk
        return {}
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {}
    out: dict[str, int] = {}
    env: dict[str, str] = {}                        # one pass over module-level bindings, no more
    for st in tree.body:
        if (isinstance(st, ast.Assign) and len(st.targets) == 1
                and isinstance(st.targets[0], ast.Name)
                and (bv := _const_value(st.value, env)) is not None):
            env[st.targets[0].id] = bv

    def visit(n: ast.AST) -> None:
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
            return                                  # a docstring is prose, the text scan has it
        v = (_const_value(n, env)
             if isinstance(n, (ast.Constant, ast.BinOp, ast.Call, ast.JoinedStr)) else None)
        if v is not None:
            if (_P in v or _T in v) and not any(x in v for x in skip):
                got = {k: c for k, c in scan_text(v).items() if k in _SPLIT_KINDS}
                if got:                             # get_source_segment is O(file): only on a hit
                    have = scan_text(ast.get_source_segment(text, n) or "")
                    for k, c in got.items():
                        if (d := c - have.get(k, 0)) > 0:
                            out[k] = out.get(k, 0) + d
            return
        for c in ast.iter_child_nodes(n):
            visit(c)
    visit(tree)
    return out


_BARE_VALUE = re.compile(r"^\S+$")


def prose_exempt_paths(doc: dict) -> list[str]:
    """The explicit file list of `prose_exempt` (RULING 46, option c): digest-pinned contract data
    whose provenance PROSE is not scanned. Never a glob, so a new file is not exempt by accident."""
    return [e["path"] for e in doc.get("prose_exempt", [])]


def label_exempt_paths(doc: dict) -> list[str]:
    """The `prose_exempt` entries marked `"labels": true` (decision D13, task 1313): the RULING 46
    exemption also covers their LABEL fields, the bare `tasks/...` or `pipeline/...` values that name
    where a fact was recorded. The stranger sees a label they cannot resolve; nothing breaks. Host and
    secret literals in those values still count. Each file repoints its labels at its next re-seal
    (a re-seal changes the file's digest pin, so it is never a drive-by edit); the entry is then
    removed. Never a glob."""
    return [e["path"] for e in doc.get("prose_exempt", []) if e.get("labels")]


def scan_bare_values(text: str) -> dict[str, int]:
    """Leaks in a prose-exempt JSON file: only string VALUES that are one bare token (a path or
    `path#symbol`) count. Sentences, and keys, do not."""
    try:
        doc = json.loads(text)
    except ValueError:
        return scan_text(text)
    vals: list[str] = []

    def add(t: str) -> None:
        t = t.strip()
        if _BARE_VALUE.match(t):
            vals.append(t)

    def walk(x) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                add(k)                      # a path used as a key is a bare token too
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            add(x)
    walk(doc)
    return scan_text("\n".join(vals))


def scan(root: pathlib.Path, shipped: dict[str, str], exempt: list[str],
         exempted: dict[str, int] | None = None,
         prose_exempt: list[str] | None = None,
         label_exempt: list[str] | None = None,
         strip: dict[str, dict] | None = None) -> dict[str, dict[str, int]]:
    """{kind: {file: count}} over the shipped text files that are not scan-exempt. Occurrences of a
    FROZEN_RECEIPT_LABELS literal in its own file are skipped and counted into `exempted`
    ({file: n}) when given."""
    leaks: dict[str, dict[str, int]] = {k: {} for k in KINDS}
    for rel in sorted(shipped):
        if any(_matches(e, rel) for e in exempt):
            continue
        p = root / rel
        try:
            if p.stat().st_size > MAX_SCAN_BYTES:
                continue
            data = p.read_bytes()
        except OSError:
            continue
        if not _is_text(data):
            continue
        text = data.decode("utf-8", errors="replace")
        if strip and rel in strip:
            text = derive_text(text, strip[rel])[0]
        orig = text
        for frel, literal, _reason in FROZEN_RECEIPT_LABELS:
            if frel != rel:
                continue
            n = text.count(literal)
            if not n and strip and rel in strip and strip[rel].get("relabel"):
                continue                 # the relabel swapped the literal for a frozen label
            if not n:
                raise CutRefusal(f"FROZEN_RECEIPT_LABELS entry {literal!r} is not in {rel}: "
                                 "remove the stale entry")
            text = text.replace(literal, " ")
            if exempted is not None:
                exempted[rel] = exempted.get(rel, 0) + n
        found = scan_bare_values(text) if rel in (prose_exempt or ()) else scan_text(text)
        if rel.endswith(".py"):
            skip = [lit for frel, lit, _r in FROZEN_RECEIPT_LABELS if frel == rel]
            for k, n in scan_split_literals(orig, skip).items():
                found[k] = found.get(k, 0) + n
        for kind, n in found.items():
            if rel in (label_exempt or ()) and kind in ("pipeline_ref", "tasks_ref"):
                continue
            leaks[kind][rel] = n
    return leaks


def _baseline_doc(root: pathlib.Path) -> dict | None:
    p = root / BASELINE_REL
    if not p.is_file():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("schema") != BASELINE_SCHEMA:
        raise CutRefusal(f"{BASELINE_REL}: schema must be {BASELINE_SCHEMA!r}")
    return doc


def load_baseline(root: pathlib.Path) -> dict[str, dict[str, int]] | None:
    doc = _baseline_doc(root)
    if doc is None:
        return None
    return {k: dict(doc.get("leaks", {}).get(k, {})) for k in KINDS}


def load_admitted(root: pathlib.Path) -> list[dict]:
    """The admission records (`--admit-moved`); empty when none. Kept verbatim by every write."""
    doc = _baseline_doc(root)
    return list(doc.get("admitted", [])) if doc else []


def apply_stored_admissions(leaks: dict[str, dict[str, int]], admitted: list[dict]
                            ) -> tuple[dict[str, dict[str, int]], list[str]]:
    """Subtract each `stored-value` admission (file, category, exact count) from the raw scan.
    Returns (effective leaks, refusals). Above the count the excess stays a leak (the baseline
    refuses it); below it the admission is stale and is a refusal line."""
    eff = {k: dict(v) for k, v in leaks.items()}
    refusals: list[str] = []
    for a in admitted:
        if a.get("kind") != "stored-value":
            continue
        rel, cat, n = a.get("file"), a.get("category"), a.get("count")
        if cat not in KINDS or not isinstance(n, int) or n < 1:
            refusals.append(f"admitted stored-value entry is malformed: {a}")
            continue
        have = eff[cat].get(rel, 0)
        if have < n:
            refusals.append(f"{cat}: {rel} admits {n} stored value(s) but has {have}; "
                            "re-admit at the lower count (an admission never outlives its "
                            "mentions)")
        elif have == n:
            eff[cat].pop(rel)
        else:
            eff[cat][rel] = have - n
    return eff, refusals


def admit_stored(root: pathlib.Path, rel: str, kind: str, count: int, reason: str,
                 raw: dict) -> dict:
    """Record a `stored-value` admission for (rel, kind) at exactly `count` (<= today's raw
    count). Refuses without a reason, on an unknown kind, or above today's count."""
    if kind not in KINDS:
        raise CutRefusal(f"unknown leak kind {kind!r} (one of {', '.join(KINDS)})")
    if not reason.strip():
        raise CutRefusal("--admit-stored needs --reason (quote the ruling)")
    have = raw[kind].get(rel, 0)
    if count < 1 or count > have:
        raise CutRefusal(f"cannot admit {count} {kind} in {rel}: the file has {have}")
    try:
        branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip()
    except CutRefusal:
        branch = ""
    record = {"kind": "stored-value", "file": rel, "category": kind, "count": count,
              "date": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
              "reason": reason, "branch": branch}
    kept = [a for a in load_admitted(root)
            if not (a.get("kind") == "stored-value"
                    and (a.get("file"), a.get("category")) == (rel, kind))]
    base_leaks = load_baseline(root)
    if base_leaks is None:
        raise CutRefusal(f"{BASELINE_REL} is missing (first run: --record-baseline)")
    admitted = kept + [record]
    eff, _ = apply_stored_admissions(raw, admitted)
    now = eff[kind].get(rel, 0)
    if now:                     # the baseline may only fall here
        base_leaks[kind][rel] = min(base_leaks[kind].get(rel, now), now)
    else:
        base_leaks[kind].pop(rel, None)
    write_baseline(root, base_leaks, admitted)
    return record


def totals(leaks: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {k: {"files": len(v), "mentions": sum(v.values())} for k, v in leaks.items()}


def diff_against_baseline(leaks, baseline) -> tuple[list[str], list[str]]:
    """(new leaks, shrunk entries). A new leak is a (kind, file) absent from the baseline or
    with a higher count. Both lists are human lines."""
    new, shrunk = [], []
    for kind in KINDS:
        base = baseline.get(kind, {})
        for rel, n in sorted(leaks[kind].items()):
            if rel not in base:
                new.append(f"{kind}: {rel} ({n}x) is not in the baseline")
            elif n > base[rel]:
                new.append(f"{kind}: {rel} grew {base[rel]} -> {n}")
        for rel, n in sorted(base.items()):
            now = leaks[kind].get(rel, 0)
            if now < n:
                shrunk.append(f"{kind}: {rel} {n} -> {now}")
    return new, shrunk


def branch_changed_files(root: pathlib.Path, base: str | None = None) -> set[str]:
    """Files (relative to `root`) this branch changed against origin/main: committed, staged,
    unstaged and untracked."""
    ref = base or "origin/main"
    names: set[str] = set()
    for args in (("diff", "--name-only", "--relative", f"{ref}...HEAD"),
                 ("diff", "--name-only", "--relative", "HEAD"),
                 ("ls-files", "-o", "--exclude-standard")):
        out = _git(root, *args).decode("utf-8", errors="replace")
        names.update(x for x in out.splitlines() if x)
    return names


def untouched_shrinks(root: pathlib.Path, leaks, baseline, base: str | None = None) -> list[str]:
    """Baseline entries `--tighten` would lower for a file this branch did not change."""
    changed = branch_changed_files(root, base)
    return sorted({f"{kind}: {rel}" for kind in KINDS
                   for rel, n in baseline.get(kind, {}).items()
                   if leaks[kind].get(rel, 0) < n and rel not in changed})


def write_baseline(root: pathlib.Path, leaks: dict[str, dict[str, int]],
                   admitted: list[dict] | None = None) -> None:
    """Write the baseline. `admitted` defaults to the list already on disk, so `--tighten` and
    `--record-baseline` never drop it."""
    if admitted is None:
        admitted = load_admitted(root)
    doc = {"schema": BASELINE_SCHEMA,
           "note": ("Ratchet: `aw ops cut` refuses a leak absent from this list or above its "  # rename-codemod: keep (persisted: note written into data/public-cut-leak-baseline.v1.json)
                    "count. It only shrinks (`aw ops cut --tighten`); the one way up is "  # rename-codemod: keep (persisted: baseline note)
                    "`aw ops cut --admit-moved FROM TO`, recorded in `admitted`. "  # rename-codemod: keep (persisted: baseline note)
                    "Zero is the real cut."),
           "totals": totals(leaks),
           "leaks": {k: dict(sorted(leaks[k].items())) for k in KINDS}}
    if admitted:
        doc["admitted"] = admitted
    (root / BASELINE_REL).write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n",
                                     encoding="utf-8")


# --------------------------------------------------------------------------- export


def _sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(root: pathlib.Path, shipped: dict[str, str], out: pathlib.Path,
           allowlist_sha: str, strip: dict[str, dict] | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise CutRefusal(f"--out {out} is not empty; the cut is always a fresh tree")
    entries = []
    for rel in sorted(shipped):
        src, dst = root / rel, out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        master = src.read_bytes()
        extra: dict = {}
        if strip and rel in strip:
            text, n = derive_text(master.decode("utf-8"), strip[rel])
            dst.write_bytes(text.encode("utf-8"))
            extra = {"derived_from_sha256": hashlib.sha256(master).hexdigest(),
                     "values_stripped": n, "resealed": bool(strip[rel]["reseal"])}
        else:
            dst.write_bytes(master)
        dst.chmod(0o755 if src.stat().st_mode & stat.S_IXUSR else 0o644)
        entries.append({"path": rel, "rule": shipped[rel], "sha256": _sha256(dst), **extra})
    manifest = {"schema": "aw.public-cut-manifest.v1", "allowlist_sha256": allowlist_sha,
                "files": entries}
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def verify_tree(tree: pathlib.Path) -> list[str]:
    """Problems found comparing an exported tree with its own manifest: a missing file, a
    changed file, or a file the manifest does not name. Empty list = byte-identical."""
    try:
        manifest = json.loads((tree / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"cannot read {MANIFEST_NAME}: {e}"]
    want = {e["path"]: e["sha256"] for e in manifest.get("files", [])}
    problems = []
    for rel, sha in sorted(want.items()):
        p = tree / rel
        if not p.is_file():
            problems.append(f"missing: {rel}")
        elif _sha256(p) != sha:
            problems.append(f"changed: {rel}")
    have = {p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file()}
    for rel in sorted(have - set(want) - {MANIFEST_NAME}):
        problems.append(f"not in the manifest: {rel}")
    return problems


# --------------------------------------------------------------------------- admit a move


def _git(root: pathlib.Path, *args: str) -> bytes:
    try:
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        detail = getattr(e, "stderr", b"") or b""
        raise CutRefusal(f"git {' '.join(args[:2])} failed: {detail.decode().strip() or e}") from e


def _resolve_base(root: pathlib.Path, base: str | None) -> str:
    """The commit this branch is measured against: the merge-base of HEAD and main."""
    for ref in ([base] if base else ["origin/main", "main"]):
        try:
            return _git(root, "merge-base", ref, "HEAD").decode().strip()
        except CutRefusal:
            continue
    raise CutRefusal(f"cannot find a base to compare with ({base or 'origin/main, main'})")


def git_renames(root: pathlib.Path, base_commit: str) -> dict[str, str]:
    """{new path: old path} for every rename git finds between `base_commit` and the working
    tree (committed, staged or edited), app-relative."""
    out = _git(root, "diff", "--find-renames", "--name-status", "--relative", "-z",
               base_commit, "--", ".").decode().split("\0")
    renames, i = {}, 0
    while i < len(out) and out[i]:
        status = out[i]
        if status[0] in "RC":
            renames[out[i + 2]] = out[i + 1] if status[0] == "R" else ""
            i += 3
        else:
            i += 2
    return renames


def admit_moved(root: pathlib.Path, frm: str, to: str, reason: str, leaks: dict,
                base: str | None = None) -> dict:
    """Raise the baseline for `to`, to the counts `frm` had on main, if git shows frm -> to as
    a rename on this branch. Returns the admission record. Refuses (CutRefusal) otherwise."""
    base_commit = _resolve_base(root, base)
    found = git_renames(root, base_commit)
    if found.get(to) != frm:
        raise CutRefusal(f"git does not show {frm} -> {to} as a rename against main "
                         f"(renames found: {found or 'none'}); a new file or a copy is not "
                         "admitted: fix its mentions instead")
    old = _git(root, "show", f"{base_commit}:./{frm}").decode("utf-8", errors="replace")
    had = scan_text(old)
    counts = {k: min(had.get(k, 0), leaks[k][to]) for k in KINDS if leaks[k].get(to)}
    counts = {k: n for k, n in counts.items() if n}
    base_leaks = load_baseline(root)
    if base_leaks is None:
        raise CutRefusal(f"{BASELINE_REL} is missing (first run: --record-baseline)")
    for k, n in counts.items():
        base_leaks[k][to] = max(base_leaks[k].get(to, 0), n)
    for k in KINDS:                      # the old path is gone: its entry moves with the file
        base_leaks[k].pop(frm, None)
    try:
        branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip()
    except CutRefusal:
        branch = ""
    record = {"kind": "moved", "from": frm, "to": to, "counts": counts,
              "date": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
              "reason": reason, "branch": branch}
    kept = [a for a in load_admitted(root) if (a.get("from"), a.get("to")) != (frm, to)]
    write_baseline(root, base_leaks, kept + [record])
    return record


def scan_shipped(root: pathlib.Path, doc: dict, shipped: dict[str, str],
                 exempted: dict[str, int] | None = None) -> dict[str, dict[str, int]]:
    """The one way to scan the shipped files with the allowlist's own exemptions (prose, labels,
    strip rules). `plan` and the refactor scoreboard both call it, so the two counts agree."""
    return scan(root, shipped, doc["scan_exempt"], exempted, prose_exempt_paths(doc),
                label_exempt_paths(doc), strip_rules(doc))


def effective_leaks(root: pathlib.Path, doc: dict, shipped: dict[str, str]) -> dict[str, dict[str, int]]:
    """Today's leaks as the cut check counts them: `scan_shipped` minus the stored-value admissions."""
    return apply_stored_admissions(scan_shipped(root, doc, shipped), load_admitted(root))[0]


def plan(root: pathlib.Path = APP_ROOT) -> dict:
    """Everything the command decides, without writing: the shipped map, the config problems,
    the leaks and their comparison with the baseline."""
    doc = load_allowlist(root)
    files = tracked_files(root)
    shipped, problems = select(doc, files)
    exempted: dict[str, int] = {}
    leaks = scan_shipped(root, doc, shipped, exempted)
    for e in doc.get("prose_exempt", []):
        if e["path"] not in shipped:
            problems.append(f"prose_exempt names {e['path']!r}, which is not a shipped file")
        if not e.get("why"):
            problems.append(f"prose_exempt {e['path']!r} needs a `why`")
    raw = leaks
    leaks, stale = apply_stored_admissions(raw, load_admitted(root))
    baseline = load_baseline(root)
    new, shrunk = diff_against_baseline(leaks, baseline) if baseline is not None else ([], [])
    new = stale + new
    return {"raw": raw, "doc": doc, "shipped": shipped, "problems": problems, "leaks": leaks,
            "baseline": baseline, "new": new, "shrunk": shrunk, "exempted": exempted,
            "allowlist_sha256": _sha256(root / ALLOWLIST_REL)}


def main(argv: list[str] | None = None, prog: str = "cyt ops cut") -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=pathlib.Path, help="empty or new directory for the exported tree")
    ap.add_argument("--verify", type=pathlib.Path, metavar="TREE",
                    help="check an exported tree against its manifest (run before installing)")
    ap.add_argument("--check", action="store_true", help="scan and compare, write nothing")
    ap.add_argument("--root", type=pathlib.Path, default=APP_ROOT, help=argparse.SUPPRESS)
    ap.add_argument("--tighten", action="store_true",
                    help="lower the baseline to today's leaks (refuses to raise any entry)")
    ap.add_argument("--scheduled", action="store_true",
                    help="with --tighten: the daily job; tighten every file, not just this branch's")
    ap.add_argument("--record-baseline", action="store_true",
                    help="write the baseline from today's leaks (only when none exists)")
    ap.add_argument("--admit-moved", nargs=2, metavar=("FROM", "TO"),
                    help="a file git shows as moved FROM -> TO against main carries the "
                         "mentions FROM had on main (nothing more); recorded in `admitted`")
    ap.add_argument("--admit-stored", nargs=3, metavar=("FILE", "KIND", "COUNT"),
                    help="admit COUNT mentions of KIND in FILE as stored values, by a ruling "
                         "(quote it in --reason); the count is exact, growth still refuses")
    ap.add_argument("--reason", default="", help="why, for the admission record")
    ap.add_argument("--base", help=argparse.SUPPRESS)
    ap.add_argument("--json", action="store_true", help="print the numbers as JSON")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    if args.verify:
        problems = verify_tree(args.verify.resolve())
        for line in problems:
            print(f"  - {line}", file=sys.stderr)
        print(f"verify {args.verify}: " + (f"REFUSED, {len(problems)} problem(s)" if problems
                                         else "byte-identical to its manifest"))
        return 1 if problems else 0
    if not (args.out or args.check or args.tighten or args.record_baseline
            or args.admit_moved or args.admit_stored):
        ap.error("give --out DIR, --check, --tighten, --record-baseline, --admit-moved "
                 "or --admit-stored")

    try:
        p = plan(root)
    except CutRefusal as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    if p["problems"]:
        print("REFUSED: the allowlist is inconsistent:", file=sys.stderr)
        for line in p["problems"]:
            print(f"  - {line}", file=sys.stderr)
        return 2

    if args.admit_stored:
        rel, kind, cnt = args.admit_stored
        try:
            rec = admit_stored(root, rel, kind, int(cnt), args.reason, p["raw"])
            p = plan(root)
        except (CutRefusal, ValueError) as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        print(f"admitted stored {kind} x{rec['count']} in {rel} (recorded in {BASELINE_REL})")

    if args.admit_moved:
        frm, to = args.admit_moved
        try:
            rec = admit_moved(root, frm, to, args.reason, p["leaks"], args.base)
            p = plan(root)
        except CutRefusal as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        print(f"admitted {frm} -> {to}: {json.dumps(rec['counts'])} (recorded in {BASELINE_REL})")

    summary = {"files": len(p["shipped"]), "leaks": totals(p["leaks"])}
    if args.record_baseline:
        if p["baseline"] is not None and any(p["baseline"].values()):
            print(f"REFUSED: {BASELINE_REL} already records leaks; it only shrinks (--tighten)",
                  file=sys.stderr)
            return 2
        write_baseline(root, p["leaks"])
        print(f"recorded {BASELINE_REL}: {json.dumps(summary['leaks'])}")
        return 0
    if p["baseline"] is None:
        print(f"REFUSED: {BASELINE_REL} is missing (first run: --record-baseline)",
              file=sys.stderr)
        return 2
    if p["new"]:
        print(f"REFUSED: {len(p['new'])} new leak(s) outside the baseline "
              f"({BASELINE_REL} may only shrink):", file=sys.stderr)
        for line in p["new"]:
            print(f"  - {line}", file=sys.stderr)
        print("Fix the file (do not raise the baseline): the cut ships no path into the "
              "private tree and no host or secret literal.", file=sys.stderr)
        return 1
    if args.tighten:
        if not args.scheduled:
            try:
                foreign = untouched_shrinks(root, p["leaks"], p["baseline"], args.base)
            except CutRefusal as e:
                print(f"REFUSED: cannot tell which files this branch changed ({e}); "
                      "the daily job uses --scheduled", file=sys.stderr)
                return 2
            if foreign:
                print(f"REFUSED: --tighten would lower {len(foreign)} file(s) this branch did not "
                      f"change (e.g. {foreign[0]}); the daily job tightens, a PR only adds "
                      "admissions", file=sys.stderr)
                return 2
        write_baseline(root, p["leaks"])
        print(f"tightened {BASELINE_REL}: {len(p['shrunk'])} entr(ies) lowered; "
              f"{json.dumps(summary['leaks'])}")
        return 0
    if p["shrunk"]:
        print(f"note: {len(p['shrunk'])} baseline entr(ies) are above today's count; "
              "the daily `--tighten --scheduled` job lowers it (do not tighten in a PR)", file=sys.stderr)
    if args.out:
        try:
            export(root, p["shipped"], args.out.resolve(), p["allowlist_sha256"],
                   strip_rules(p["doc"]))
        except CutRefusal as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
    if args.json:
        print(json.dumps(summary, indent=1))
    else:
        t = summary["leaks"]
        print(f"cut: {summary['files']} files" + (f" -> {args.out}" if args.out else "")
              + "; baseline leaks (files/mentions) "
              + ", ".join(f"{k} {v['files']}/{v['mentions']}" for k, v in t.items())
              + f"; exempt: {sum(p['exempted'].values())} (frozen receipt labels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
