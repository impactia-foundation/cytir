#!/usr/bin/env python3
"""cytir.index.stage_pack_b1 — embed + stage a coverage pack in aw_b1-NATIVE shape
(task 322; first user: the `aw_dfrlaw_cov` DILA French-law pack).

Supersedes the retired config-B 128-dim layout (removed 2026-10-28, task 1378) for packs destined for aw_b1: same locked
config-B collection schema (arctic-l-v2 dense MRL-128 int8 on_disk + BM25 IDF
sparse), but the records are projected through the SAME machinery
`build_aw_b1.project_record` uses, so a post-build delta apply is a plain
scroll+upsert with no re-projection:

  * identity  — aw_b1_identity: content_digest = sha256(normalize(text)),
                classify_tier (no source-native per-chunk anchor -> Tier B,
                digest-as-anchor), point_id = uuid5(NS, canonical_url#anchor).
  * sparse    — `sparse_text` when present, otherwise `text`, composed with
                title/site name -> aw_common.bm25_tokens with AW_FOLD_ACCENTS=1
                (tok epoch 2) -> bm25_saturate.saturate_sparse.
  * dense     — `dense_text` when present, otherwise `text`, independently
                composed with title/site name -> arctic-embed-l-v2 MRL-128 +
                L2 renorm (embed epoch 1, config B).
  * payload   — aw_b1 conventions (tier = identity tier, snippet <=25 words,
                text dropped, country via aw_country, license via
                aw_b1_license, rule_epochs stamped, master-only fields under
                `_master` per ADR-039).
  * consent   — DUMP-BASIS stamps (task 322 orchestrator ruling, flagged for
                Tim): records ingested from an official open-data dump carry
                consent_state="dump" + consent_basis/consent_source/
                consent_checked_at reflecting the dump license grant, NOT a
                robots probe of the canonical-URL host (legifrance's WAF 403s
                robots from every vantage; the Licence Ouverte grant is the
                admission basis, exactly like fedlex SPARQL / gesetze dumps).

  AW_FOLD_ACCENTS=1 python3 -m cytir.index.stage_pack_b1 \
      --in legi_code.jsonl --in loda.jsonl ... \
      --collection aw_dfrlaw_cov --qdrant-url http://localhost:6398 \
      --mrl-dim 128 \
      --consent-source "dila-opendata echanges.dila.gouv.fr (Licence Ouverte v2.0)" \
      --consent-checked-at 2026-07-25 \
      --attribution "République française — DILA (Légifrance)" \
      --license-version "Licence Ouverte / Open Licence v2.0"

  `--mrl-dim` is REQUIRED, no default (task 508 / task 500 §7.2): the dense
  vector width to truncate to and store. Every caller states it explicitly --
  a code default is how a lane ships the wrong width for a week undetected.

Resumable (--resume): skips point_ids already in the collection. Collection is
created with hnsw m=0 (bulk; packs are consumed by scroll — finalize to m=16
only if the pack should also be queryable directly).
"""
from __future__ import annotations

import argparse
import json
import os
import time

# fold ON before importing the tokenizer path (mirror of build_aw_b1)
os.environ.setdefault("AW_FOLD_ACCENTS", "1")

import cytir.core.common as AC
import cytir.core.ids as ID
import cytir.core.projection as PROJ
from cytir.index import payload_carry as PC

from cytir.core import bm25_tokenizer as _TOK  # noqa: E402

_TOK._FOLD_ACCENTS = True

from cytir.core.qdrant import qdrant_connect

# Task 682 (S6): the collection layout, the TEI and in-process encoders, the dense
# truncation, the upsert loop and the green wait are ONE stage,
# `cytir.embed.collection_stage`. This script keeps what is its own: the aw_b1 projection,
# the consent stamps, the payload-carry refusal and the CLI. `tei_doc_encoder` is the
# stage's (re-exported: the carry-guard test names it in main()).
from cytir.embed import collection_stage as STAGE
from cytir.embed.collection_stage import tei_doc_encoder  # noqa: F401

ARCTIC = STAGE.ARCTIC
DENSE = AC.DENSE_VEC
SPARSE = AC.SPARSE_BM25

#: The layout this script has always created: int8 at the caller's `--mrl-dim`,
#: bulk hnsw m=0 (packs are consumed by scroll, not queried).
LAYOUT = "pack-staging"


def ensure_collection(client, name: str, recreate: bool, mrl_dim: int):
    STAGE.create_collection(
        client, name, STAGE.resolve_layout(LAYOUT, dense_dim=mrl_dim),
        recreate=recreate, log=lambda m: print(m, flush=True))


def project(rec: dict, args, consent: dict) -> tuple[str, str, str, dict] | None:
    """(point_id, dense_text, sparse_text, payload); None => skip.

    `record_text_channels` is the shared contract with the generic config-B
    embedder.  Keeping selection here rather than in a Fedlex-only wrapper
    makes the corrected channel split part of the normal recurring pack path.
    """
    return PROJ.project_record(
        rec, consent=consent, attribution=args.attribution,
        license_version=args.license_version,
        preserve_rule_epochs=getattr(args, "preserve_rule_epochs", False),
        # task 984: the exact chunk word count rides the STAGING payload to the content-pack
        # composer, which declares it in the manifest and strips it from the pack rows. A record
        # without it adds nothing to the payload.
        extra_passthrough=("chunk_n_words",),
    )


def load_records(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", action="append", required=True)
    ap.add_argument("--collection", required=True)
    # task 508 / 500 §7.2: NO default. A silent default is how a lane ends up
    # shipping the wrong width for a week (the news-lane repoint to 1024 is
    # exactly this risk) -- every caller states the width it wants explicitly.
    # The width the CURRENT r7 release ships at is 128; callers that need that
    # behaviour pass --mrl-dim 128 themselves (news_refresh_stage.sh does, via
    # its own AW_NEWS_MRL_DIM config default -- see that script's header).
    ap.add_argument("--mrl-dim", type=int, required=True,
                    help="dense vector width to truncate/store at (config, not "
                         "a code default -- e.g. 128 for the current r7 news "
                         "route, 1024 for the post-flip repoint)")
    ap.add_argument("--qdrant-url", default="http://localhost:6398")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--resume", action="store_true")
    # task 360: upserting a small delta into a collection that ALREADY holds
    # tens of millions of unrelated points (a cloned twin, not something
    # built incrementally by this script) makes --resume's existing_ids()
    # call do a FULL unfiltered scroll of the whole collection just to check
    # for prior staged ids that provably cannot be there (content-addressed
    # ids from DIFFERENT text). --no-recreate skips recreate like --resume
    # does, WITHOUT that scan -- caller is asserting the id set is
    # collision-free (verify independently, e.g. chunk.py's own
    # assign_chunk_ids id-distinctness check, before relying on this).
    ap.add_argument("--no-recreate", action="store_true",
                    help="skip collection recreate, skip the existing_ids scan too "
                         "(unlike --resume). For a delta upsert into a large "
                         "pre-existing collection this script didn't build.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tei-url", default=os.environ.get("AW_DOC_TEI_URL", ""),
                    help="embed documents through a TEI sidecar serving the "
                         "same checkpoint instead of loading the in-process "
                         "torch model (env AW_DOC_TEI_URL). Prove parity "
                         "against already-indexed vectors before using it.")
    ap.add_argument("--tei-batch", type=int,
                    default=int(os.environ.get("AW_DOC_TEI_BATCH", "1")),
                    help="rows per TEI request; the live container runs "
                         "max_batch_tokens=2048 and a card is up to ~1.1k "
                         "composed tokens, so 1 is the safe default.")
    ap.add_argument("--consent-state", default="dump")
    ap.add_argument("--consent-basis", default="official-dump-license")
    ap.add_argument("--consent-source", required=True)
    ap.add_argument("--consent-checked-at", required=True)
    ap.add_argument("--attribution", default=None)
    ap.add_argument("--license-version", default=None)
    # task 565 / ADR-090 amendment 1. Without this, EVERY row this script
    # stages lands with `aw_b1_identity.EPOCHS` -- `chunk: 1` -- whatever
    # generation it was actually cut under, because `project_record`'s
    # `preserve_rule_epochs` was reachable only from the chunk@2 worker.
    # `rule_epochs.chunk = 1` is what every existing generation discriminator
    # reads as PRODUCTION LEGACY (task 536's index_partition_hosts
    # --min-chunk-epoch is the one that matters), so a chunk.reference epoch-5
    # delta staged through this path would have been indistinguishable from
    # the incumbent's own rows the moment it landed -- the fossil-record
    # failure in corpus-doctrine 4, manufactured at write time.
    #
    # It is OPT-IN and it FAILS CLOSED: a record whose `rule_epochs` is
    # missing or is not a string-to-int object raises out of
    # `project_record` rather than silently falling back to the default set.
    ap.add_argument("--preserve-rule-epochs", action="store_true",
                    help="carry each record's own rule_epochs (and its "
                         "`stamps.prov_id`) into the payload instead of "
                         "stamping the build-wide default set. Required for "
                         "any pack whose rows were cut under a non-baseline "
                         "chunk policy; refuses a record that cannot state "
                         "its own epochs.")
    args = ap.parse_args()
    # task 1397: the one caller of collection_stage's write primitives holds the aw_e10 twin hold
    from cytir.index import twin_write_hold as _TWH
    _TWH.refuse_cli_write(args, tool='cytir.index.stage_pack_b1')

    assert _TOK._FOLD_ACCENTS, "AW_FOLD_ACCENTS must be ON (tok epoch 2)"
    ID.self_test()
    consent = {
        "consent_state": args.consent_state,
        "consent_basis": args.consent_basis,
        "consent_source": args.consent_source,
        "consent_checked_at": args.consent_checked_at,
    }

    client = qdrant_connect(args.qdrant_url, timeout=600)
    skip_recreate = args.resume or args.no_recreate
    ensure_collection(client, args.collection, recreate=not skip_recreate,
                      mrl_dim=args.mrl_dim)
    done: set = set()
    if args.resume:
        done = STAGE.existing_ids(client, args.collection)
        print(f"[resume] {len(done)} already staged", flush=True)
    elif args.no_recreate:
        print("[no-recreate] skipping existing_ids scan -- assumed "
              "collision-free delta upsert", flush=True)

    projected = []
    stats = {"in": 0, "skipped": 0, "dup": 0, "resumed": 0}
    seen = set()
    for rec in load_records(args.inp):
        stats["in"] += 1
        row = project(rec, args, consent)
        if row is None:
            stats["skipped"] += 1
            continue
        pid = row[0]
        if pid in seen:
            stats["dup"] += 1
            continue
        seen.add(pid)
        if pid in done:
            stats["resumed"] += 1
            continue
        projected.append(row)
        if args.limit and len(projected) >= args.limit:
            break
    print(f"[project] {stats} -> {len(projected)} to embed", flush=True)

    # -- PAYLOAD CARRY: a refusal BEFORE the embedder is loaded (task 584).
    # Task 580 staged and applied 187,222 rows carrying no `title`,
    # `trust_tier`, `lang` or `domain`, measured a -17 h@1, and concluded
    # against the lever those rows were built to test. The apply-time gate
    # (`gate_payload_contract.py`) catches that shape now, but it catches it
    # after the GPU has been spent; the same floors applied here cost one pass
    # over the projected payloads and fail in seconds. Same CONTRACT object,
    # so the two surfaces cannot come to disagree. No flag turns it off: a
    # pack that cannot meet the contract is a build defect, and waiving it is
    # a decision about the standard, not a step inside the run.
    try:
        pcv = PC.assert_scoreable((r[3] for r in projected),
                                  label=f"{args.collection} (staged)")
    except PC.PayloadCarryError as exc:
        raise SystemExit(f"[stage] REFUSED: {exc}")
    print(f"[stage] payload carry OK: {pcv['rows_checked']:,} row(s), "
          + ", ".join(f"{f['field']} {f['present']:.1%}"
                      for f in pcv["fields"]), flush=True)

    if args.tei_url:
        # Same checkpoint, out of process. Used when the box has no working
        # torch install; validated against the in-process vectors already in
        # the twin before it may be trusted — see task 398's
        # tei_parity_check (n=40, min cosine 0.9997, median 0.999999).
        print(f"Embedding docs via TEI at {args.tei_url} "
              f"(batch {args.tei_batch})", flush=True)
        encode_docs = tei_doc_encoder(args.tei_url, args.tei_batch)
    else:
        encode_docs = STAGE.build_encoder(max_seq=args.max_seq,
                                          log=lambda m: print(m, flush=True))

    def materialize(chunk):
        rows = []
        for pid, dense_composed, sparse_composed, payload in chunk:
            fidx, fval = PROJ.saturated_bm25(sparse_composed, payload.get("lang"))
            rows.append(STAGE.Row(pid, dense_composed,
                                  (fidx, fval) if fidx else None, payload))
        return rows

    layout = STAGE.resolve_layout(LAYOUT, dense_dim=args.mrl_dim)
    res = STAGE.embed_rows(client, args.collection, projected, materialize, layout,
                           encode_docs, batch=args.batch,
                           log=lambda m: print(m, flush=True))
    t0, total = res["t0"], res["points"]

    print("waiting for green…", flush=True)
    info = STAGE.wait_green(client, args.collection, iters=600)
    print(f"DONE {info.points_count} points in `{args.collection}` "
          f"status={info.status} embed_rate={total/max(time.time()-t0,1):.0f} pts/s "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
