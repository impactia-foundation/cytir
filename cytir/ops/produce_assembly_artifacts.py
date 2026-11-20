"""Produce the ADR-101 assembly-guard input artifacts from the sealed corpus.

READ-ONLY over the corpus: opens files for reading only, issues no network
request, no Qdrant call, and writes only under ``--out``.

This is the *artifact producer* half of the ADR-101 flip gate.  The guard suite
(the assembly guards module) never touches the corpus; it consumes what
this script emits.  Keeping the two apart is what lets the gate run from a cold
clone against a committed plan, and what lets the gate be unit-tested with
synthetic fixtures.

What it emits (all under ``--out``):

  claim-manifests.v1.json
      The G1 claim-manifest index: one manifest per contributing set, at
      point-ID granularity, with the claims artifact URI + re-verified sha256 +
      record count.  Sets that contribute no vectors carry an explicit
      ``mode`` of ``tombstone`` or ``cancelled`` -- never an absence.

  g4-tombstone-suppression.v1.json
      The G4 suppression record: the exact point IDs where an ADR-091 tombstone
      is re-claimed by task 427, listed inline (there are 4).

  g6-mixed-url-resolution.v1.json
      The G6 unit ruling for every URL whose old points are not all won by one
      set, resolved by the ADR-101 / task-456 tie-break (whole URL follows the
      majority winner of its old points; on a tie the sealed set wins).

  g7-identity-collapse.v1.json + g7-identity-collapse-ids.jsonl.gz
      The G7 evidence: every task-427 new point ID that equals an old point ID
      the same stage claims to replace.  These are the IDs the applier must
      UPSERT in place and must never place in the build's delete set.

  g8-task427-winning-point-ids.jsonl.gz + g8-task427-winning-points.v1.json
      The winning-subset point IDs (ADR-101 Ruling 1) that a census
      ``_replacement_point_witness`` for task 427 must carry.  This is the
      *input* to minting that witness, not the witness itself.

Usage (on the box that holds the corpus)::

    python3 produce_assembly_artifacts.py \
        --corpus-root /mnt/.../aw-corpus \
        --collision-ledger data/r0-427-collision-ledger.v1.json \
        --out /root/task462/out
"""
from __future__ import annotations

import argparse
import gzip
from cytir.acquire.gzip_pinned import deterministic_gzip_open
import hashlib
import io
import json
import sys
import uuid
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

SCHEMA_MANIFESTS = "aw.assembly-claim-manifests.v1"
SCHEMA_MANIFEST = "aw.assembly-claim-manifest.v1"

# ---------------------------------------------------------------------------
# Set classification.  Verbatim from the retired task-456 pass-2 and
# task456_pass3.py -- the classes are read off each family's own PACK.json
# chunk policy, not guessed from the name.  Reproduced here (rather than
# imported) because the ledger those scripts produced is the sha-bound input
# the guard checks against, so any divergence must show up as a hard mismatch.
# ---------------------------------------------------------------------------
ATOMIC_EPOCH1 = {
    "scratch_ai_hf_model_cards_exact", "scratch_bav_docs_exact",
    "scratch_bazl_docs_exact", "scratch_bk_digital_exact",
    "scratch_boc_valet_exact", "scratch_ca_canadaca_cc_exact",
    "scratch_cass_juri_exact", "scratch_ce_juri_exact",
    "scratch_ch_business_exact", "scratch_ch_cantonal_law_exact",
    "scratch_ctgov_psy_exact", "scratch_de_laender_juris_exact",
    "scratch_de_laender_law_exact", "scratch_edoeb_pages_exact",
    "scratch_elcom_docs_exact", "scratch_endorsement_discovery_exact",
    "scratch_entscheidsuche_exact", "scratch_genesis_stats_exact",
    "scratch_insee_definition_exact", "scratch_mh_authority_pack_exact",
    "scratch_modelcontextprotocol_io_docs_exact", "scratch_r4_coverage_exact",
    "scratch_rights_canon_exact", "scratch_sbfi_vet_exact",
    "scratch_scc_a2aj_exact", "scratch_sitemap_ingest_exact",
    "scratch_skills_transition_exact", "scratch_statcan_daily_exact",
    "scratch_statcan_wds_exact", "scratch_tech_gapfill_exact",
    "scratch_us_fed_cc_exact", "scratch_wave6_cc_exact",
    "scratch_wave7_cc_exact", "scratch_weko_docs_exact",
}
EXACT_EPOCH4 = {
    "astra_html_exact", "astra_pdf_exact", "bfs_dam_exact",
    "ca_canadaca_captured_exact", "ca_canadaca_direct_exact",
    "ce_dila_unique_exact", "cochrane_direct_exact",
    "cochrane_preserved_exact", "context7_exact",
    "datatracker_canonical_exact", "fedlex_exact", "finma_exact",
    "ietf_drafts_exact", "iris_exact", "jorf_recent", "justice_exact",
    "justice_residual_exact", "kali_exact", "loda_exact", "rfc_exact",
    "routec_generic_complete_exact", "service_public_dila_exact",
    "swissmedic_html_exact", "swissmedic_pdf_exact", "w3c_exact",
    "scratch_ca_store_news_exact",
}
TOMBSTONE_SETS = {"natural_person_policy_drop"}

# ADR-101 Ruling 1, as a table over the competing set's class.
WINNER_BY_CLASS = {
    "atomic_epoch1": "task427_postconsent",
    "exact_epoch4": "this_set",
    "reembed_epoch3_in_place": "this_set",
    "named_slice_payload_rule": "this_set",
    "tombstone": "nobody",
    "adr092_remainder": "task427_postconsent",
}

TASK427 = "task427_postconsent"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def raw(point_id: str) -> bytes:
    return uuid.UUID(str(point_id)).bytes


def log(message: str) -> None:
    print(message, flush=True)


def write_point_ids(path: Path, point_ids: list[str]) -> None:
    """Deterministic gzip: fixed mtime so the sha256 is reproducible."""
    with path.open("wb") as handle:
        with deterministic_gzip_open(handle, compresslevel=6) as raw_out:
            with io.TextIOWrapper(raw_out, encoding="utf-8") as target:
                for point_id in point_ids:
                    target.write(
                        json.dumps({"point_id": point_id}, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------


def classify(label: str) -> str:
    if label in TOMBSTONE_SETS:
        return "tombstone"
    if label in EXACT_EPOCH4:
        return "exact_epoch4"
    if label in ATOMIC_EPOCH1:
        return "atomic_epoch1"
    raise SystemExit(f"unclassified replacement family: {label}")


def load_admitted_old_points(stage: Path, expect_sha: str
                             ) -> tuple[set[bytes], dict[bytes, str]]:
    path = stage / "admitted-old-points.jsonl.gz"
    digest = sha256_file(path)
    if digest != expect_sha:
        raise SystemExit(f"427 admitted-old-points drift: {digest} != {expect_sha}")
    keys: set[bytes] = set()
    urls: dict[bytes, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            key = raw(row["old_point_id"])
            if key in keys:
                raise SystemExit("duplicate old_point_id in 427 stage ledger")
            keys.add(key)
            urls[key] = row["url"]
    return keys, urls


def build_owner_map(root: Path, census: dict, wiki_witness: Path,
                    census_dir: Path, X: set[bytes], X_url: dict[bytes, str],
                    ) -> tuple[dict[bytes, str], dict[str, dict]]:
    """Owner class per claimed old point, plus the per-family claim bindings.

    Byte-for-byte the pass-2 rule: a point is owned by the replacement witness
    that carries it, else by the 413-A Wikipedia witness, else by the ADR-092
    remainder, else by a payload-rule named slice attributed by claimed host.
    """
    owner: dict[bytes, str] = {}
    bindings: dict[str, dict] = {}
    witnesses = census["inputs"]["replacement_point_witnesses"]
    for index, (label, binding) in enumerate(sorted(witnesses.items()), 1):
        uri = binding["point_artifact"]["uri"]
        path = root / uri.removeprefix("artifact://")
        actual = sha256_file(path)
        if actual != binding["point_artifact"]["sha256"]:
            raise SystemExit(f"claims artifact drift for {label}: {actual}")
        cls = classify(label)
        rows = 0
        hits = 0
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                rows += 1
                key = raw(json.loads(line)["point_id"])
                if key in X:
                    if key in owner:
                        raise SystemExit("double owner among replacement witnesses")
                    owner[key] = f"{cls}|{label}"
                    hits += 1
        if rows != binding["records"]:
            raise SystemExit(f"claims row drift for {label}: {rows}")
        bindings[label] = {
            "class": cls, "uri": uri, "sha256": actual,
            "bytes": path.stat().st_size, "records": rows,
            "contested_with_task427": hits,
        }
        log(f"[inv] {index:2d}/61 {label:<45} rows={rows:>9,} x427={hits:>8,}")

    # 413-A Wikipedia: an in-place re-embed; its witness is a binary ID bucket
    # set, not a points.jsonl.gz.
    receipt = json.loads((wiki_witness / "RECEIPT.json").read_text())
    wiki_hits: set[bytes] = set()
    wiki_ids = 0
    wiki_artifacts = []
    for artifact in receipt["artifacts"]:
        uri = artifact["uri"]
        if "/buckets/ids-" not in uri:
            continue
        path = root / uri.removeprefix("artifact://")
        body = path.read_bytes()
        actual = hashlib.sha256(body).hexdigest()
        if artifact.get("sha256") and actual != artifact["sha256"]:
            raise SystemExit(f"wikipedia bucket drift: {uri}")
        if len(body) % 16:
            raise SystemExit(f"malformed wikipedia bucket: {uri}")
        wiki_ids += len(body) // 16
        wiki_artifacts.append({"uri": uri, "bytes": len(body), "sha256": actual})
        for offset in range(0, len(body), 16):
            key = body[offset:offset + 16]
            if key in X:
                wiki_hits.add(key)
    for key in wiki_hits:
        if key in owner:
            raise SystemExit("wikipedia overlaps a replacement witness")
        owner[key] = "reembed_epoch3_in_place|wikipedia_413a"
    bindings["wikipedia_413a"] = {
        "class": "reembed_epoch3_in_place",
        "uri": f"artifact://{wiki_witness.relative_to(root).as_posix()}/RECEIPT.json",
        "sha256": sha256_file(wiki_witness / "RECEIPT.json"),
        "bytes": (wiki_witness / "RECEIPT.json").stat().st_size,
        "records": wiki_ids,
        "buckets": len(wiki_artifacts),
        "contested_with_task427": len(wiki_hits),
    }
    log(f"[inv] wikipedia_413a ids={wiki_ids:,} x427={len(wiki_hits):,}")

    remainder_hits = 0
    with gzip.open(census_dir / "remaining-points.jsonl.gz", "rt",
                   encoding="utf-8") as source:
        for line in source:
            key = raw(json.loads(line)["point_id"])
            if key in X:
                if key in owner:
                    raise SystemExit("remainder overlaps a sealed owner")
                owner[key] = "adr092_remainder|remainder"
                remainder_hits += 1
    log(f"[inv] remainder x427={remainder_hits:,}")

    leftover = X - set(owner)
    hosts: Counter[str] = Counter()
    for key in leftover:
        host = (urlsplit(X_url[key]).hostname or "").lower()
        hosts[host] += 1
        owner[key] = f"named_slice_payload_rule|{host}"
    log(f"[inv] leftover named-slice {len(leftover):,} hosts={dict(hosts)}")
    if len(owner) != len(X):
        raise SystemExit("owner map does not cover every claimed old point")
    return owner, bindings


def winner_of(owner_label: str) -> str:
    cls, _, name = owner_label.partition("|")
    winner = WINNER_BY_CLASS.get(cls)
    if winner is None:
        raise SystemExit(f"no ADR-101 rule for owner class: {cls}")
    return TASK427 if winner == TASK427 else ("nobody" if winner == "nobody" else name)


# ---------------------------------------------------------------------------


def main(argv=None, prog=None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--collision-ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifact-prefix", default="_work/task462",
                        help="corpus-relative directory the emitted point-ID "
                             "files are published under; the URIs in the "
                             "manifests resolve against it")
    args = parser.parse_args(argv)
    prefix = args.artifact_prefix.strip("/")

    root: Path = args.corpus_root
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    ledger = json.loads(args.collision_ledger.read_text())
    ledger_sha = sha256_file(args.collision_ledger)
    basis = ledger["basis"]

    stage = root / basis["task427_admitted_old_points"]["uri"].removeprefix("artifact://")
    stage = stage.parent
    census_dir = root / basis["census_v31_run"]["uri"].removeprefix("artifact://")
    census_dir = census_dir.parent
    wiki_witness = root / "_work/task417/r0-witness/wikipedia-task413a-v1"
    retained = root / "_work/task426/r0-adr092-retained-parent-l1b-0d90bb55146d"
    cc_parts = [root / f"_work/task426/r0-adr092-cc-parents-1d22ba19-p{i}"
                for i in range(9)]

    for label, key in (("stage RUN", "task427_stage_run"),
                       ("census RUN", "census_v31_run")):
        path = root / basis[key]["uri"].removeprefix("artifact://")
        actual = sha256_file(path)
        if actual != basis[key]["sha256"]:
            raise SystemExit(f"{label} drift: {actual}")
    log("[inv] basis hashes re-verified")

    X, X_url = load_admitted_old_points(
        stage, basis["task427_admitted_old_points"]["sha256"])
    log(f"[inv] task427 claimed old points {len(X):,}")

    census = json.loads((census_dir / "RUN.json").read_text())
    owner, bindings = build_owner_map(root, census, wiki_witness, census_dir,
                                      X, X_url)

    # ---- winner per point, and the reconciliation against the ledger -------
    winners: dict[bytes, str] = {key: winner_of(label)
                                 for key, label in owner.items()}
    won = sum(1 for value in winners.values() if value == TASK427)
    log(f"[inv] task427 wins {won:,} loses {len(X) - won:,}")

    # ---- G4: tombstone exclusivity ----------------------------------------
    tombstone_hits = sorted(
        str(uuid.UUID(bytes=key)) for key, label in owner.items()
        if label.startswith("tombstone|"))
    log(f"[g4] tombstone points re-claimed by task427: {len(tombstone_hits)}")

    # ---- G6: mixed-URL unit ruling ----------------------------------------
    doc_old: dict[str, set[bytes]] = {}
    doc_rows: Counter[str] = Counter()
    with gzip.open(stage / "admitted-documents.jsonl.gz", "rt",
                   encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            doc_old.setdefault(row["url"], set()).update(
                raw(point) for point in row["old_point_ids"])

    stage_run = json.loads((stage / "RUN.json").read_text())
    new_ids: set[bytes] = set()
    new_by_url: Counter[str] = Counter()
    new_rows = 0
    for shard in stage_run["artifacts"]["stages"]:
        with gzip.open(stage / shard["records"]["file"], "rt",
                       encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                new_rows += 1
                new_ids.add(raw(row["point_id"]))
                new_by_url[row["url"]] += 1
    if new_rows != len(new_ids):
        raise SystemExit(f"task427 emits duplicate new point IDs: "
                         f"rows={new_rows} distinct={len(new_ids)}")
    orphan_rows = sum(count for url, count in new_by_url.items()
                      if url not in doc_old)
    if orphan_rows:
        raise SystemExit(f"{orphan_rows} task427 rows on URLs absent from "
                         "admitted-documents; the stage is not self-consistent")
    log(f"[inv] task427 new point ids {len(new_ids):,}")

    url_class: dict[str, str] = {}
    for url, keys in doc_old.items():
        seen = {winners[key] for key in keys}
        url_class[url] = ("427_wins_whole_url" if seen == {TASK427}
                          else "427_loses_whole_url" if TASK427 not in seen
                          else "mixed_url_needs_unit_ruling")
        doc_rows[url] = new_by_url.get(url, 0)

    mixed = sorted(url for url, cls in url_class.items()
                   if cls == "mixed_url_needs_unit_ruling")
    resolutions = []
    for url in mixed:
        # The ratified tie-break, applied literally: every old point votes,
        # including a point whose winner is "nobody" (an ADR-091 tombstone),
        # because the ledger's own owner map makes "nobody" a winner class.
        # Whether a policy tombstone should instead GOVERN its whole URL is an
        # unruled policy question -- it is flagged, not decided, here.
        tally = Counter(winners[key] for key in doc_old[url])
        top = max(tally.values())
        leaders = sorted(name for name, count in tally.items() if count == top)
        # majority winner of the URL's old points; on a tie the SEALED set
        # wins (task 427 is the challenger).
        if len(leaders) == 1:
            unit_winner, basis_note = leaders[0], "majority"
        else:
            sealed = [name for name in leaders if name != TASK427]
            if not sealed:
                raise SystemExit(f"tie with no sealed leader on {url}")
            unit_winner, basis_note = sealed[0], "tie-break:sealed-set-wins"
        resolutions.append({
            "url": url,
            "old_points": len(doc_old[url]),
            "task427_new_rows": doc_rows[url],
            "point_winner_tally": dict(sorted(tally.items())),
            "unit_winner": unit_winner,
            "basis": basis_note,
            "task427_rows_loaded": doc_rows[url] if unit_winner == TASK427 else 0,
            "task427_rows_suppressed": 0 if unit_winner == TASK427 else doc_rows[url],
        })
    mixed_rows = sum(item["task427_new_rows"] for item in resolutions)
    mixed_loaded = sum(item["task427_rows_loaded"] for item in resolutions)
    log(f"[g6] mixed urls={len(mixed)} rows={mixed_rows} "
        f"loaded={mixed_loaded} suppressed={mixed_rows - mixed_loaded}")

    rows_by_class = Counter()
    for url, cls in url_class.items():
        rows_by_class[cls] += doc_rows[url]

    # ---- G7: identity collapse -------------------------------------------
    collapsed = sorted(str(uuid.UUID(bytes=key)) for key in (new_ids & X))
    log(f"[g7] identity-collapsed new point ids: {len(collapsed):,}")
    collapse_path = out / "g7-identity-collapse-ids.jsonl.gz"
    write_point_ids(collapse_path, collapsed)

    # ---- G8 input: the winning-subset point IDs ---------------------------
    winning_ids = sorted(str(uuid.UUID(bytes=key))
                         for key, value in winners.items() if value == TASK427)
    winning_path = out / "g8-task427-winning-point-ids.jsonl.gz"
    write_point_ids(winning_path, winning_ids)
    log(f"[g8] winning-subset point ids written: {len(winning_ids):,}")

    # ---- the CC materializer claims --------------------------------------
    # The stage binds every terminal status; only `admitted` will ever carry a
    # vector.  Both counts are published: the claim is what the set binds, the
    # win is what it will supply.
    cc_bindings = []
    cc_points = 0
    cc_by_status: Counter[str] = Counter()
    for part in cc_parts:
        terminals = part / "terminals.jsonl.gz"
        if not terminals.exists():
            raise SystemExit(f"cc materializer partition missing terminals: {part}")
        count = 0
        with gzip.open(terminals, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                bound = len(row.get("old_point_ids") or [])
                count += bound
                cc_by_status[str(row.get("status") or "unknown")] += bound
        cc_points += count
        cc_bindings.append({
            "uri": f"artifact://{terminals.relative_to(root).as_posix()}",
            "sha256": sha256_file(terminals),
            "bytes": terminals.stat().st_size,
            "old_point_ids": count,
        })
        log(f"[inv] cc {part.name} old_point_ids={count:,}")
    log(f"[inv] cc by status {dict(sorted(cc_by_status.items()))}")

    # ---- the payload-rule ("named slice") sets ----------------------------
    # These are subtracted from the remainder by a predicate over payload, not
    # by an enumerated point witness.  wikipedia and gesetze also publish a
    # real witness; the rest publish nothing a claim can be intersected with,
    # which is guard G1's failure mode -- see produce_named_slice_claims.py.
    named_counts = dict(census["named_counts"])
    gesetze_binding = census["inputs"].get("gesetze_point_witness") or {}
    payload_rule_sets = {name: count for name, count in named_counts.items()
                         if name not in {"wikipedia", "gesetze"}}
    log(f"[inv] payload-rule sets without a point witness: "
        f"{sum(payload_rule_sets.values()):,} points across "
        f"{len(payload_rule_sets)} sets")

    # ---- the Track-1A retained bind: URL + count only ----------------------
    retained_alias = retained / "replacement-alias-ledger.jsonl.gz"
    retained_urls = 0
    if retained_alias.exists():
        with gzip.open(retained_alias, "rt", encoding="utf-8") as source:
            retained_urls = sum(1 for _ in source)

    # ---- emit -------------------------------------------------------------
    sets = []
    for label, binding in sorted(bindings.items()):
        cls = binding["class"]
        mode = ("tombstone" if cls == "tombstone" else
                "in_place_reembed" if cls == "reembed_epoch3_in_place" else
                "replacement")
        contested = binding["contested_with_task427"]
        # The 413-A witness carries 11 IDs that are no longer in current
        # membership; the census's own intersection is the honest win count.
        current_points = (int(census["named_counts"]["wikipedia"])
                          if label == "wikipedia_413a" else binding["records"])
        sets.append({
            "schema": SCHEMA_MANIFEST,
            "set_id": label,
            "kind": "sealed_replacement_family",
            "class": cls,
            "mode": mode,
            "stage": "sealed-1024-vector",
            "contributes_vectors": mode != "tombstone",
            "claims": {
                "granularity": "point_id",
                "uri": binding["uri"],
                "sha256": binding["sha256"],
                "bytes": binding["bytes"],
                "count": binding["records"],
                "verified": True,
            },
            "adjudication": {
                "contested_with_task427": contested,
                "winner_on_contested": WINNER_BY_CLASS[cls],
                "points_won": 0 if mode == "tombstone" else (
                    current_points
                    - (contested if WINNER_BY_CLASS[cls] == TASK427 else 0)),
                "points_lost_to_task427":
                    contested if WINNER_BY_CLASS[cls] == TASK427 else 0,
            },
            **({"tombstones": {
                "granularity": "point_id",
                "uri": binding["uri"],
                "sha256": binding["sha256"],
                "count": binding["records"],
                "reclaimed_by_a_winning_set": contested,
            }} if mode == "tombstone" else {}),
            **({"census_witness": {
                "uri": binding["uri"],
                "sha256": binding["sha256"],
                "records": binding["records"],
                "shape": "membership/points.jsonl.gz",
                "consumed_by_census": "r0-remainder-census-v31",
                "scope": "all_claims",
            }} if label != "wikipedia_413a" and mode != "tombstone" else {}),
        })

    for name, count in sorted(payload_rule_sets.items()):
        sets.append({
            "schema": SCHEMA_MANIFEST,
            "set_id": f"named_slice:{name}",
            "kind": "payload_rule_named_slice",
            "class": "named_slice_payload_rule",
            "mode": "replacement",
            "stage": "sealed-1024-vector",
            "contributes_vectors": True,
            "claims": {
                "granularity": "payload_predicate",
                "uri": "artifact://retired/task417-census_r0_remainder.py"
                       "::named_slice",
                "count": count,
                "verified": False,
                "why_insufficient":
                    "the set is selected by a predicate over payload "
                    "(domain / source_type / URL prefix), not by an enumerated "
                    "point witness. A predicate is not a claim: two predicates "
                    "can select the same point and nothing raises. Task 456 "
                    "found 176 arxiv points also claimed by task 427 only by "
                    "elimination, because there was no arxiv claim set to "
                    "intersect. Remedy: produce the named-slice claims first.",
            },
            "adjudication": {
                "contested_with_task427": None,
                "winner_on_contested": "this_set",
                "points_won": count,
                "points_lost_to_task427": 0,
            },
        })

    if gesetze_binding:
        gesetze_path = root / gesetze_binding["uri"].removeprefix("artifact://")
        gesetze_sha = sha256_file(gesetze_path)
        if gesetze_sha != gesetze_binding["sha256"]:
            raise SystemExit(f"gesetze witness drift: {gesetze_sha}")
        sets.append({
            "schema": SCHEMA_MANIFEST,
            "set_id": "gesetze",
            "kind": "payload_rule_named_slice",
            "class": "named_slice_payload_rule",
            "mode": "replacement",
            "stage": "sealed-1024-vector",
            "contributes_vectors": True,
            "claims": {
                "granularity": "point_id",
                "uri": gesetze_binding["uri"],
                "sha256": gesetze_sha,
                "count": int(named_counts["gesetze"]),
                "verified": True,
            },
            "adjudication": {
                "contested_with_task427": 0,
                "winner_on_contested": "this_set",
                "points_won": int(named_counts["gesetze"]),
                "points_lost_to_task427": 0,
            },
            "census_witness": {
                "uri": gesetze_binding["uri"],
                "sha256": gesetze_sha,
                "records": int(named_counts["gesetze"]),
                "shape": "membership/points.jsonl.gz",
                "consumed_by_census": "r0-remainder-census-v31",
                "scope": "all_claims",
            },
        })

    sets.append({
        "schema": SCHEMA_MANIFEST,
        "set_id": TASK427,
        "kind": "rechunk_replacement_stage",
        "class": "rechunk_epoch4_from_l1b",
        "mode": "replacement",
        "stage": "embedded-verified",
        "contributes_vectors": True,
        "claims": {
            "granularity": "point_id",
            "uri": basis["task427_admitted_old_points"]["uri"],
            "sha256": basis["task427_admitted_old_points"]["sha256"],
            "count": basis["task427_admitted_old_points"]["rows"],
            "verified": True,
        },
        "vector": {
            "root": "artifact://_work/task427/post-consent-embedding-vectors-f78b4b75-r2",
            "rows": len(new_ids), "dim": 1024, "dtype": "float16",
        },
        "adjudication": {
            "points_won": won,
            "points_lost": len(X) - won,
            "rows_declared": len(new_ids),
            "rows_loaded": rows_by_class["427_wins_whole_url"] + mixed_loaded,
            "rows_suppressed": (rows_by_class["427_loses_whole_url"]
                                + mixed_rows - mixed_loaded),
            "rows_on_mixed_urls": mixed_rows,
        },
        "identity_collapse": {
            "count": len(collapsed),
            "uri": f"artifact://{prefix}/{collapse_path.name}",
            "sha256": sha256_file(collapse_path),
            "note": "new point IDs equal to an old point ID the same stage "
                    "replaces; the applier MUST upsert these in place",
        },
    })

    sets.append({
        "schema": SCHEMA_MANIFEST,
        "set_id": "cc_materializer",
        "kind": "sealed_chunk_stage",
        "class": "exact_epoch4",
        "mode": "replacement",
        "stage": "chunked-not-embedded",
        "contributes_vectors": True,
        "claims": {
            "granularity": "point_id",
            "uri": "artifact://_work/task426/r0-adr092-cc-parents-1d22ba19-p0..p8/terminals.jsonl.gz",
            "partitions": cc_bindings,
            "count": cc_points,
            "verified": True,
        },
        "adjudication": {
            "contested_with_task427": 0,
            "winner_on_contested": "this_set",
            "points_won": cc_by_status.get("admitted", 0),
            "points_bound_all_statuses": cc_points,
            "points_by_status": dict(sorted(cc_by_status.items())),
            "points_lost_to_task427": 0,
        },
        "pending": {
            "reason": "vectors not yet produced (embed-only lane, census bucket B1)",
        },
    })

    sets.append({
        "schema": SCHEMA_MANIFEST,
        "set_id": "track1a_retained_bind",
        "kind": "sealed_chunk_stage",
        "class": "rechunk_epoch4_from_l1b",
        "mode": "cancelled",
        "stage": "chunked-verified-not-embedded",
        "contributes_vectors": False,
        "claims": {
            "granularity": "url",
            "uri": "artifact://_work/task426/r0-adr092-retained-parent-l1b-0d90bb55146d/replacement-alias-ledger.jsonl.gz",
            "count": retained_urls,
            "verified": False,
            "why_insufficient": "URL + point_count only; publishes no point ID, "
                                "so no claim can be adjudicated at point-ID "
                                "granularity (guard G1 fails closed)",
        },
        "cancellation": {
            "uri": "data/assembly/cancellation-track1a-retained-bind.v1.json",
        },
    })

    manifests = {
        "schema": SCHEMA_MANIFESTS,
        "producer": "aw ops produce-assembly-artifacts",
        "collision_ledger": {
            "uri": f"artifact://{args.collision_ledger.name}",
            "sha256": ledger_sha,
        },
        "membership": {
            "uri": basis["membership_receipt"]["uri"],
            "sha256": basis["membership_receipt"]["sha256"],
        },
        "sets": sets,
        "totals": {
            "sets": len(sets),
            "contributing": sum(1 for item in sets if item["contributes_vectors"]),
            "point_id_granularity": sum(
                1 for item in sets if item["claims"]["granularity"] == "point_id"),
            "cancelled": sum(1 for item in sets if item["mode"] == "cancelled"),
        },
    }

    def write(name: str, payload: dict) -> None:
        (out / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        log(f"[out] {name}")

    write("claim-manifests.v1.json", manifests)
    write("g4-tombstone-suppression.v1.json", {
        "schema": "aw.assembly-suppression-record.v1",
        "guard": "G4",
        "rule": "ADR-101 Ruling 1.3 -- a tombstoned point accepts no replacement",
        "tombstone_set": "natural_person_policy_drop",
        "reclaimed_by": TASK427,
        "count": len(tombstone_hits),
        "point_ids": tombstone_hits,
        "action": "strike from task427's claim; the point is deleted, not replaced",
    })
    write("g6-mixed-url-resolution.v1.json", {
        "schema": "aw.assembly-unit-resolution.v1",
        "guard": "G6",
        "tie_break": "whole URL follows the majority winner of its old points; "
                     "on a tie the sealed set wins",
        "urls": len(resolutions),
        "old_points": sum(item["old_points"] for item in resolutions),
        "task427_new_rows": mixed_rows,
        "task427_rows_loaded": mixed_loaded,
        "task427_rows_suppressed": mixed_rows - mixed_loaded,
        "resolutions": resolutions,
    })
    write("g7-identity-collapse.v1.json", {
        "schema": "aw.assembly-identity-collapse.v1",
        "guard": "G7",
        "set_id": TASK427,
        "count": len(collapsed),
        "ids": {
            "uri": f"artifact://{prefix}/{collapse_path.name}",
            "sha256": sha256_file(collapse_path),
            "bytes": collapse_path.stat().st_size,
        },
        "required_apply_property": {
            "mode": "upsert_in_place",
            "id_collision_policy": "upsert_in_place",
            "delete_set_excludes_loaded_ids": True,
        },
        "sample": collapsed[:8],
    })
    write("g8-task427-winning-points.v1.json", {
        "schema": "aw.assembly-witness-input.v1",
        "guard": "G8",
        "set_id": TASK427,
        "count": len(winning_ids),
        "ids": {
            "uri": f"artifact://{prefix}/{winning_path.name}",
            "sha256": sha256_file(winning_path),
            "bytes": winning_path.stat().st_size,
        },
        "next": "mint a census replacement point witness over exactly these IDs "
                "in the membership/points.jsonl.gz shape "
                "census_r0_remainder.py::_replacement_point_witness requires",
    })
    write("url-class-rows.v1.json", {
        "schema": "aw.assembly-url-class-rows.v1",
        "urls": {cls: sum(1 for value in url_class.values() if value == cls)
                 for cls in sorted(set(url_class.values()))},
        "task427_new_rows": dict(sorted(rows_by_class.items())),
        "task427_new_rows_total": sum(rows_by_class.values()),
    })
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main() or 0)
