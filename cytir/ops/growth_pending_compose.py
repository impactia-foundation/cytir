#!/usr/bin/env python3
"""Unattended embed -> compose -> enqueue for any sealed chunk stage the
growth lane's own pipelines produce (task 664, 2026-09-26).

WHY THIS EXISTS
----------------
2026-09-25 overnight: CC tranche 2/3's GPU embed finished cleanly, SE
tranche 8's chunk stage finished cleanly, and BOTH sat idle for ~12 hours
with the GPU at 0% utilization -- because the last step (embed if not
already embedded, then compose a promotion-queue manifest, then enqueue)
was something only an AGENT knew how to do, and no agent was awake. The
lane's actual throughput ceiling was never the GPU or the network; it was
"is someone watching".

`cytir.index.compose.embed_compose_enqueue()` (task 657) is already the ONE
door for "embed (GPU-locked) -> verify identity -> compose -> enqueue",
generic across every family that stages through the `aw.413a.stage.v1`
chunk shape + `aw.413a.vectors.v1` vector shape -- which CC, ERIC and SE
all do, because they all chunk through
`stage_structured_markdown_for_embedding.py` / `adapt_pack_to_413a_stage.py`
and embed through `cytir.embed.shards` (directly or via the
task 413a's `embed_shards.py` shim). What was missing was not a
compose primitive; it was something to ask, unattended and on a schedule,
"is there a sealed stage nobody has composed yet?" -- this module is that
question, asked per family, and the answer acted on.

ERIC already has its own equivalent unattended loop
(the build box's chain_all_tranches.sh, which chains
raw materialize output all the way through compose+enqueue) -- this module
does not duplicate that; it exists for the families that do NOT have one:
CC (embedded by a throwaway per-tranche script, task 634) and SE (chunked
by a hand-run RESUME.md recipe, task 631). Adding a family here means
writing one `discover_*_pending()` function; the embed/compose/enqueue
step is never re-derived.

DEPLOYMENT
----------
Runs as `aw-growth-pending-compose.timer` on the build box (every 15 min) --
see the build box's aw-growth-pending-compose.service unit. Idempotent
and safe to run concurrently with itself or with a promotion drain: it only
ever WRITES a queue pack's manifest.json + READY once (skips any pack_id
that already has a manifest.json in --queue-dir), and embedding is GPU-lock
guarded (`cytir.embed.stage.embed_stage`) so a concurrent embed elsewhere is a
clean, retried refusal, never a corrupt write.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from cytir.core import paths as PATHS
from cytir.core import twin as TWIN
from cytir.index import compose as AIC
from cytir.index import manifest_rows as MR
from cytir.index import subpack as SUBPACK

_REPO_ROOT = PATHS.REPO_ROOT

#: The queue is the operator setting `queue_dir` (cytir.core.paths), resolved on use; it used to be
#: a literal of ours here, which a scratch run (AW_WORK_ROOT) would have silently ignored.


class PendingJob:
    """One discovered, not-yet-enqueued sealed chunk stage."""

    def __init__(self, *, pack_id: str, family: str, chunks_dir: str,
                vectors_dir: str, collection: str = TWIN.collection(), mrl: int = 256,
                ceiling: int | None = None, license_from: str | None = None,
                family_license: str | None = None,
                family_payload: dict | None = None,
                family_payload_basis: str | None = None,
                title_from: str | None = None,
                exclude_ids_file: str | None = None,
                exclude_ids_reason: str | None = None):
        self.pack_id = pack_id
        self.family = family
        self.chunks_dir = chunks_dir
        self.vectors_dir = vectors_dir
        self.collection = collection
        self.mrl = mrl
        self.ceiling = ceiling
        self.license_from = license_from
        self.family_license = family_license
        self.family_payload = family_payload
        self.family_payload_basis = family_payload_basis
        self.title_from = title_from
        self.exclude_ids_file = exclude_ids_file
        self.exclude_ids_reason = exclude_ids_reason

    def __repr__(self) -> str:
        return f"PendingJob({self.pack_id!r}, family={self.family!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, PendingJob) and vars(self) == vars(other)


# ------------------------------------------------------------- CC discovery
# Shape: <task634_root>/tranches/t<N>_gpu/<part>/{chunks,vectors}, where
# <part> is either a domain-hash letter (a, b, c, d, ...) or "all" for a
# tranche with no part split -- exactly what task 634's
# gpu_embed_driver.sh and stage_cc_tranche.py already produce. A part is
# STAGED (ready to discover) once its chunks dir holds at least one
# shard-*.texts.jsonl.gz; it is READY TO COMPOSE once every chunk shard has
# a matching vector shard (embed already ran) -- this module composes those
# immediately and, for a staged-but-not-yet-embedded part, still returns it
# (embed_compose_enqueue embeds it first).

_CC_TRANCHE_DIR_RE = re.compile(r"^t(\d+)_gpu$")

#: Task 664, 2026-09-26 (real incident: tranche2/4/5's first drain attempt
#: after the anchor fix all refused with `ProjectionError: unresolved
#: license`): `cc_authority_frontier_w7` rows carry no top-level `license`
#: (confirmed on real rows, kept in the test data as task664_cc_real_rows.jsonl)
#: and CC spans many origin hosts no single host-table rule can cover, so
#: its licence basis is FAMILY-level, not per-row or per-host -- ADR-111's
#: opt-out+minimum-record reading (cytir.core.projection's own
#: comment on this exact family, task 641). The value is the identical
#: string the family's original, pre-unattended landing (tranche2/3, via a
#: bespoke script) already declared in its manifest -- not a new grant
#: invented here.
_FAMILY_LICENSE = {
    "cc_authority_frontier_w7": "ADR-111-opt-out-minimum-record",
}


#: Task 664, 2026-09-26, incident 2 (cc-authority-frontier-w7-tranche2):
#: the seal-watchdog can re-stage the SAME gated fetch output into a
#: differently-partitioned pack (an unlettered "all" pack duplicating an
#: earlier lettered a-d split) -- composing it anyway just recreates the
#: twin/prod resume gap `cytir/index/promote_queue.py`'s AlreadyLandedError
#: handling exists to patch around, and wastes an embed+compose cycle on
#: content that is already served. CC packs are minted per-URL
#: (cytir.core.common.mint_passage_point_id via the row's own url+anchor),
#: so two genuinely different rows essentially never collide by id; a
#: fraction this high is "duplicate", not "mostly overlapping legitimate
#: content" by coincidence.
_DUPLICATE_FRACTION = 0.99


def _stage_already_landed_fraction(chunks_dir: str, *, twin_url: str | None,
                                   collection: str) -> float:
    """Fraction of `chunks_dir`'s point_ids already present in twin --
    `0.0` (never blocks discovery) if `twin_url` is not given, the stage is
    empty, or the check itself fails for any reason (unreachable twin, a
    malformed shard). Best-effort only, same discipline as every other
    resilience check in this module: it self-heals next tick rather than
    ever turning a discovery pass into a hard failure."""
    if not twin_url:
        return 0.0
    try:
        ids = [str(row["point_id"]) for row in AIC.PSP.read_records(chunks_dir)]
    except Exception:  # noqa: BLE001 -- best-effort only, never blocks discovery
        return 0.0
    if not ids:
        return 0.0
    try:
        present = AIC.PSP.already_present(twin_url, collection, ids)
    except Exception:  # noqa: BLE001
        return 0.0
    return len(present) / len(ids)


def discover_cc_pending(task634_root: str, *,
                        family: str = "cc_authority_frontier_w7",
                        twin_url: str | None = None,
                        collection: str = TWIN.collection(),
                        log=lambda msg: None) -> list[PendingJob]:
    root = Path(task634_root) / "tranches"
    if not root.is_dir():
        return []
    jobs: list[PendingJob] = []
    for tranche_dir in sorted(root.iterdir()):
        m = _CC_TRANCHE_DIR_RE.match(tranche_dir.name)
        if not m or not tranche_dir.is_dir():
            continue
        tranche_num = m.group(1)
        for part_dir in sorted(tranche_dir.iterdir()):
            if not part_dir.is_dir():
                continue
            chunks_dir = part_dir / "chunks"
            vectors_dir = part_dir / "vectors"
            if not chunks_dir.is_dir():
                continue
            if not any(chunks_dir.glob("*.texts.jsonl.gz")):
                continue
            part = part_dir.name
            pack_id = (f"cc-authority-frontier-w7-tranche{tranche_num}"
                      if part == "all"
                      else f"cc-authority-frontier-w7-tranche{tranche_num}-{part}")
            fraction = _stage_already_landed_fraction(
                str(chunks_dir), twin_url=twin_url, collection=collection)
            if fraction >= _DUPLICATE_FRACTION:
                log(f"[growth-pending-compose] {pack_id}: SKIPPED -- "
                   f"{fraction:.1%} of its ids are already on twin under "
                   "(presumably) a different pack_id -- a duplicate stage, "
                   "not composed")
                continue
            jobs.append(PendingJob(
                pack_id=pack_id, family=family, chunks_dir=str(chunks_dir),
                vectors_dir=str(vectors_dir),
                family_license=_FAMILY_LICENSE.get(family)))
    return jobs


# ------------------------------------------------------------- SE discovery
# Shape: <se_root>/production/stages/<LABEL>-413a-portable-v3/STAGE-RECEIPT.json
# (chunking done) and <se_root>/vectors/release=*/source=<LABEL>/vectors
# (embed done, if it exists yet -- globbed rather than reconstructed from
# the tranche ordinal, because the release name embeds an ordinal
# (aw-task631-so-tranche8-v1) this module has no independent way to
# recompute from the offset alone; the SOURCE half of that path is just the
# stage's own label, which this function already has).

#: Task 680, 2026-09-30: the admission ledger is NOT a serving record. Until
#: this task SE discovery skipped every stage `data/expansion-pack-admission-
#: status.v1.json` names ("admitted through an OLDER mechanism -- a full-corpus
#: rebuild"). The twin said otherwise: Stack Overflow tranches 1-7 (91,959
#: points, embedded at 1024-d fp16 since 08-19 .. 09-23) had 0 of 1,400 sampled
#: point ids and 0 of 700 sampled urls on aw_e8, while stackoverflow.com held
#: 25,502 points (tranches 8-9 only). A ledger row records that vectors EXIST;
#: only the collection records that a row is SERVED. SE now asks the same
#: question CC always asked -- `_stage_already_landed_fraction` on the twin --
#: and the ledger is no longer consulted (enumerated != staged != serving).
def _find_vectors_dir(vectors_root: str | Path, labels: list[str]) -> str | None:
    """The sealed vector dir for a stage, or None. Two shapes exist on disk:
    `release=*/source=<label>/vectors/` (every pack from task 631 on) and
    `release=*/source=<label>/` with the shards directly inside it (the three
    aw-e3-rebuild SO pilot tranches). SO's pilot tranche 000000 also carries a
    different LABEL (`...score10-production-tranche-...`) than its stage
    (`...qa-score10-tranche-...`), so callers pass every label to try."""
    root = Path(vectors_root)
    for label in labels:
        for pattern in (f"release=*/source={label}/vectors",
                        f"release=*/source={label}"):
            for d in sorted(root.glob(pattern)):
                if any(d.glob("*.vectors.manifest.json")):
                    return str(d)
    return None


#: Task 664, 2026-09-26: found live -- `stackoverflow_qa`'s raw rows carry
#: `license: null` at the top level (Stack Exchange's licence is per-record,
#: ADR-103 amendment 2, not host-uniform, and `stackoverflow.com` is not in
#: `aw_b1_license.py`'s host table), so every tranche this driver composed
#: refused at promotion with "unresolved license" -- `cytir.index.compose`'s
#: `compose_pack()` had no `license_from` passthrough at all until this same
#: task added one. `source_metadata.accepted_answer.license` is the SAME
#: path `promote_sealed_pack.py --license-from` already uses for the sibling
#: family `stackexchange_nonso_qa_p0` (task 641, commit a131dbc1) -- one
#: convention for this row shape, not a second one invented here.
_FAMILY_LICENSE_FROM = {
    "stackoverflow_qa": "source_metadata.accepted_answer.license",
    # Task 680: the 41 already-served non-SO packs carry exactly these manifest
    # fields (read from the landed stackexchange-nonso-qa-math...-part18-v1
    # manifest, 2026-09-30) -- one convention per row shape, copied from the
    # served precedent, not a second invented here.
    "stackexchange_nonso_qa_p0": "source_metadata.accepted_answer.license",
}

#: Task 664, 2026-09-26 (SE tranche 8): `stackoverflow_qa` rows carry no
#: top-level source_type/trust_tier/title, so every pack this module
#: composed refused at promotion with PayloadCarryError (0% carry on all
#: three). Coordinator's ruling: derive from the SAME family's own already-
#: SERVED precedent (checked on the twin: tranches 1-8's landed rows),
#: never invent a number -- source_type="community", trust_tier=0.55,
#: tier=B. Cited via family_payload_basis so the manifest states its
#: derivation, not a bare number.
_FAMILY_PAYLOAD = {
    "stackoverflow_qa": {"source_type": "community", "trust_tier": 0.55},
    "stackexchange_nonso_qa_p0": {"source_type": "community",
                                  "trust_tier": 0.55},
}
_FAMILY_PAYLOAD_BASIS = {
    "stackoverflow_qa": ("family precedent: served SE rows, tranche 1-8 on "
                        "prod/twin (source_type=community, trust_tier=0.55, "
                        "tier=B) -- coordinator ruling, task 664, "
                        "2026-09-26"),
}
_FAMILY_TITLE_FROM = {
    "stackoverflow_qa": "source_metadata.title",
    "stackexchange_nonso_qa_p0": "source_metadata.title",
}


def vectors_complete(chunks_dir: str, vectors_dir: str) -> bool:
    """True when every chunk shard has a sealed vector shard beside it (same
    `shard-NNNN` tag, manifest present). Such a job must NOT go through
    `embed_stage`: that door always runs `check_gpu_idle()`, so while the ERIC
    chain holds the A6000 every already-embedded SO/SE pack would be refused
    `GPUBusyError` each cycle, and on an idle GPU it would queue for the lock to
    re-open a finished embed. Task 680: the 16 packs this sweep found had their
    vectors for 1-5 weeks."""
    want = {Path(p).name.removesuffix(".texts.jsonl.gz")
            for p in Path(chunks_dir).glob("shard-*.texts.jsonl.gz")}
    have = {Path(p).name.removesuffix(".vectors.manifest.json")
            for p in Path(vectors_dir).glob("shard-*.vectors.manifest.json")}
    return bool(want) and want <= have


def _se_label_aliases(label: str) -> list[str]:
    """Labels a SO stage's vectors may carry. The ledger and the e3-rebuild
    vector dir of pilot tranche 000000 say `...score10-production-tranche-...`
    where the stage says `...qa-score10-tranche-...` (see
    `_find_vectors_dir`)."""
    return [label, label.replace("-qa-score10-tranche-",
                                 "-score10-production-tranche-")]


def _is_duplicate_stage(pack_id: str, chunks_dir: str, *, twin_url: str | None,
                        collection: str, log) -> bool:
    fraction = _stage_already_landed_fraction(
        chunks_dir, twin_url=twin_url, collection=collection)
    if fraction >= _DUPLICATE_FRACTION:
        log(f"[growth-pending-compose] {pack_id}: SKIPPED -- {fraction:.1%} "
            "of its ids are already served on the twin")
        return True
    return False


def discover_se_pending(se_root: str, *,
                        family: str = "stackoverflow_qa",
                        vectors_roots: list[str] | None = None,
                        twin_url: str | None = None,
                        collection: str = TWIN.collection(),
                        skip=None,
                        log=lambda msg: None) -> list[PendingJob]:
    """`skip(pack_id) -> bool` is asked BEFORE the twin check: a stage that is
    already enqueued needs no id probe against the twin (55+ stages x 20k ids
    every 15 minutes is real CPU on the box the ERIC chain is running on)."""
    stages_root = Path(se_root) / "production" / "stages"
    if not stages_root.is_dir():
        return []
    roots = [str(Path(se_root) / "vectors"), *(vectors_roots or [])]
    jobs: list[PendingJob] = []
    for stage_dir in sorted(stages_root.iterdir()):
        if not stage_dir.is_dir() or not stage_dir.name.endswith("-413a-portable-v3"):
            continue
        if not (stage_dir / "STAGE-RECEIPT.json").is_file():
            continue
        label = stage_dir.name[:-len("-413a-portable-v3")]
        if skip is not None and skip(label):
            continue
        if _is_duplicate_stage(label, str(stage_dir), twin_url=twin_url,
                               collection=collection, log=log):
            continue
        vectors_dir = next(
            (v for r in roots
             if (v := _find_vectors_dir(r, _se_label_aliases(label)))), None
        ) or str(Path(se_root) / "vectors"
                 / f"release=pending/source={label}/vectors")
        jobs.append(PendingJob(
            pack_id=label, family=family, chunks_dir=str(stage_dir),
            vectors_dir=vectors_dir,
            license_from=_FAMILY_LICENSE_FROM.get(family),
            family_payload=_FAMILY_PAYLOAD.get(family),
            family_payload_basis=_FAMILY_PAYLOAD_BASIS.get(family),
            title_from=_FAMILY_TITLE_FROM.get(family)))
    return jobs


# ------------------------------------------------------- non-SO SE discovery
# Shape: <nonso_root>/<site[-partN]>/stage-413a-portable-v3/STAGE-RECEIPT.json
# (task 631). Until task 680 nothing discovered these: the timer only knew the
# SO tranche tree, so math.stackexchange.com parts 19-27 (176,525 points,
# embedded 2026-09-23) sat unqueued for a week while parts 1-18 had been
# queued by hand. Pack ids follow the landed convention
# `stackexchange-nonso-qa-<unit>-v1`; the family carries the SAME manifest
# fields as the 41 served packs (see _FAMILY_* above).

_NONSO_FAMILY = "stackexchange_nonso_qa_p0"
_NONSO_FAMILY_BASIS = (
    "family precedent: 41 served non-SO packs on prod/twin (source_type="
    "community, trust_tier=0.55) -- read from the landed math.stackexchange.com"
    "-part18-v1 manifest, task 680, 2026-09-30")


def discover_nonso_pending(nonso_root: str, *,
                           vectors_roots: list[str] | None = None,
                           twin_url: str | None = None,
                           collection: str = TWIN.collection(),
                           skip=None,
                           log=lambda msg: None) -> list[PendingJob]:
    root = Path(nonso_root)
    if not root.is_dir():
        return []
    jobs: list[PendingJob] = []
    for unit_dir in sorted(root.iterdir()):
        stage_dir = unit_dir / "stage-413a-portable-v3"
        if not (stage_dir / "STAGE-RECEIPT.json").is_file():
            continue
        pack_id = f"stackexchange-nonso-qa-{unit_dir.name}-v1"
        if skip is not None and skip(pack_id):
            continue
        if _is_duplicate_stage(pack_id, str(stage_dir), twin_url=twin_url,
                               collection=collection, log=log):
            continue
        vectors_dir = next(
            (v for r in (vectors_roots or [])
             if (v := _find_vectors_dir(r, [pack_id]))), None
        ) or str(root / "vectors-pending" / pack_id / "vectors")
        jobs.append(PendingJob(
            pack_id=pack_id, family=_NONSO_FAMILY, chunks_dir=str(stage_dir),
            vectors_dir=vectors_dir,
            license_from=_FAMILY_LICENSE_FROM[_NONSO_FAMILY],
            family_payload=_FAMILY_PAYLOAD[_NONSO_FAMILY],
            family_payload_basis=_NONSO_FAMILY_BASIS,
            title_from=_FAMILY_TITLE_FROM[_NONSO_FAMILY]))
    return jobs


# ------------------------------------------------ OpenLegalData (task 680)
# the OpenLegalData shard loop writes one chunk stage per acquisition shard:
# `stages/de-openlegaldata-v1/chunks-shard<N>/{PACK.json,chunks/shard-0000.texts.
# jsonl.gz}`. This discovery globs that tree (the roster is the tree, never a typed
# list: census-roster-from-manifest) and answers, per shard, which of the things a
# pack needs is still missing. It never embeds: a stage with no vectors is logged
# AWAITING VECTORS and left alone, because the embed is a deliberate GPU decision
# (the A6000 is shared; the 53 shards were embedded on a rented box, task 680) and
# `embed_stage` here would queue behind or refuse against the ERIC chain every 15
# minutes. A stage whose L1 is not filed is logged AWAITING L1: the drain refuses
# every page without an L1 record (ADR-105, require=l1), and a refused pack sits in
# the queue holding the row cap (the OpenAlex gap, task 686).

_OLD_FAMILY = "de_openlegaldata_v1"
_OLD_STAGES = "stages/de-openlegaldata-v1"
_OLD_VECTORS = "vectors/release=aw-task680-openlegaldata-v1"
_OLD_SHARD_RE = re.compile(r"^chunks-shard(\d+)$")
#: Copied from the two landed shard-0 packs (de-openlegaldata-shard0-part1/2,
#: DONE on prod/twin): the same family, the same manifest fields. Not a new grant.
_OLD_PAYLOAD = {"source_type": "official", "trust_tier": 0.9}
_OLD_PAYLOAD_BASIS = (
    "family precedent: the two landed shard-0 packs de-openlegaldata-shard0-part1/2 "
    "(DONE; source_type=official, trust_tier=0.9) -- read from their queue manifests, "
    "task 680, 2026-10-01")


def discover_openlegaldata_pending(corpus_root: str, *,
                                   twin_url: str | None = None,
                                   collection: str = TWIN.collection(), skip=None,
                                   log=lambda msg: None) -> list[PendingJob]:
    from cytir.sources.families import openlegaldata as OLD

    root = Path(corpus_root)
    stages = root / _OLD_STAGES
    if not stages.is_dir():
        return []
    found = sorted(
        (int(m.group(1)), d) for d in stages.iterdir()
        if (m := _OLD_SHARD_RE.match(d.name)) and d.is_dir())
    jobs: list[PendingJob] = []
    for shard, stage in found:
        chunks_dir = stage / "chunks"
        if not (stage / "PACK.json").is_file() or not any(
                chunks_dir.glob("*.texts.jsonl.gz")):
            continue
        pack_id = f"de-openlegaldata-shard{shard}"
        if skip is not None and skip(pack_id):
            continue
        vectors_dir = root / _OLD_VECTORS / f"source=de-openlegaldata-shard{shard}"
        if not vectors_complete(str(chunks_dir), str(vectors_dir)):
            log(f"[growth-pending-compose] {pack_id}: AWAITING VECTORS (chunked, "
                "not embedded; embed is a deliberate GPU decision)")
            continue
        if not OLD.is_l1_filed(root, shard):
            log(f"[growth-pending-compose] {pack_id}: AWAITING L1 (ADR-105 "
                f"require=l1: file l1/release={OLD.l1_release_name(shard)} first)")
            continue
        if _is_duplicate_stage(pack_id, str(chunks_dir), twin_url=twin_url,
                               collection=collection, log=log):
            continue
        jobs.append(PendingJob(
            pack_id=pack_id, family=_OLD_FAMILY, chunks_dir=str(chunks_dir),
            vectors_dir=str(vectors_dir), family_payload=dict(_OLD_PAYLOAD),
            family_payload_basis=_OLD_PAYLOAD_BASIS,
            title_from="source_metadata.title"))
    return jobs


# ------------------------------------------ family-pack stages (task 696)
# `cytir.sources.family_stage` cuts a `run_family` pack of a `chunk.atomic` family into stages of
# <= 150k rows under `stages/<family>-v1/chunks-shardN` and files each stage's L1 (ADR-105). This
# discovery globs that tree (the roster is the tree) and answers, per stage, what is still
# missing. Unlike OpenLegalData it DOES embed (the stage is small, ~40 min, and the GPU lock is
# the existing convention); like OpenLegalData it refuses to compose before the L1 is filed, and
# it refuses a stage nobody has measured for overlap (`OVERLAP.json`): the overlap file is passed
# to compose as the declared exclusion (the #495 path), the count is reported, never hidden.

def discover_family_stage_pending(corpus_root: str, family_id: str, *,
                                  twin_url: str | None = None,
                                  collection: str = TWIN.collection(), skip=None,
                                  log=lambda msg: None) -> list[PendingJob]:
    from cytir.sources import family_stage as FS

    jobs: list[PendingJob] = []
    for shard, stage in FS.list_stages(corpus_root, family_id):
        pack_id = FS.pack_id(family_id, shard)
        if not (stage / "PACK.json").is_file():
            continue
        if skip is not None and skip(pack_id):
            continue
        if not FS.is_l1_filed(corpus_root, family_id, shard):
            log(f"[growth-pending-compose] {pack_id}: AWAITING L1 (ADR-105 require=l1: "
                f"file l1/release={FS.l1_release(family_id, shard)} first)")
            continue
        overlap = FS.read_overlap(stage)
        if overlap is None:
            log(f"[growth-pending-compose] {pack_id}: AWAITING OVERLAP CHECK "
                "(python -m cytir.sources.family_stage overlap)")
            continue
        if overlap.get("needs_ruling"):
            log(f"[growth-pending-compose] {pack_id}: AWAITING RULING -- {overlap['urls_served']:,} "
                f"of {overlap['urls']:,} urls ({overlap['url_share']:.1%}) are already served "
                "under other ids, over the drain's ceiling, and no exclusion is declared "
                "(python -m cytir.sources.family_stage overlap --exclude-served-urls)")
            continue
        exclude = (str(stage / overlap["exclude_file"]) if overlap.get("rows_excluded") else None)
        jobs.append(PendingJob(
            pack_id=pack_id, family=family_id, chunks_dir=str(stage / "chunks"),
            vectors_dir=str(FS.vectors_dir(corpus_root, family_id, shard)),
            exclude_ids_file=exclude,
            exclude_ids_reason=overlap.get("exclude_reason") if exclude else None))
    return jobs


# ------------------------------------------------- declared sets (task 680)
# A SET is a family whose sealed packs are too big for one promotion, or carry
# rows the admission floor does not admit: the ten OpenAlex packs (6,404,873
# points, 5,438,224 after the ADR-102 L2 ladder trim). It is declared as data in
# `data/growth-pending-sets.v1.json` -- paths relative to the corpus root, a
# sha-pinned exclusion file, the shard run length -- so a stranger can add one
# without touching this module (engineering-doctrine #2). Each source pack is cut
# into sub-packs by `cytir.index.subpack` (symlinks, nothing copied or re-sealed).
#
# A set that is not `active` is HELD, and a hold names the condition that
# un-parks it (engineering-doctrine #8): discovery logs both, every sweep, so the
# reason is never only in someone's head.

SETS_SCHEMA = "aw.growth-pending-sets.v1"


def _load_sets(sets_file: str) -> list[dict]:
    doc = json.loads(Path(sets_file).read_text())
    if doc.get("schema") != SETS_SCHEMA:
        raise ValueError(f"{sets_file}: schema {doc.get('schema')!r}, "
                         f"expected {SETS_SCHEMA!r}")
    return doc["sets"]


def discover_sets_pending(sets_file: str, corpus_root: str, *,
                          twin_url: str | None = None,
                          collection: str = TWIN.collection(), skip=None,
                          log=lambda msg: None) -> list[PendingJob]:
    root = Path(corpus_root)
    jobs: list[PendingJob] = []
    for st in _load_sets(sets_file):
        if st.get("status") != "active":
            hold = st.get("hold") or {}
            log(f"[growth-pending-compose] set {st['set_id']}: HELD -- "
                f"{hold.get('reason', 'no reason recorded')} "
                f"(un-parks when: {hold.get('unhold_when', 'NOT RECORDED')})")
            continue
        exclude = st.get("exclude_ids") or {}
        for src in st["sources"]:
            chunks_dir = str(root / src["chunks"])
            vectors_dir = str(root / src["vectors"])
            subs = SUBPACK.plan_subpacks(chunks_dir, vectors_dir,
                                         int(st["shards_per_pack"]))
            for sub in subs:
                pack_id = f"{st['pack_id_prefix']}{src['label']}-s{sub.index:04d}"
                if skip is not None and skip(pack_id):
                    continue
                out_c, out_v = SUBPACK.materialize_subpack(
                    chunks_dir, vectors_dir, sub,
                    str(root / st["subpack_root"] / pack_id))
                if _is_duplicate_stage(pack_id, out_c, twin_url=twin_url,
                                       collection=collection, log=log):
                    continue
                jobs.append(PendingJob(
                    pack_id=pack_id, family=st["family"], chunks_dir=out_c,
                    vectors_dir=out_v,
                    license_from=st.get("license_from"),
                    family_license=st.get("family_license"),
                    family_payload=st.get("family_payload"),
                    family_payload_basis=st.get("family_payload_basis"),
                    title_from=st.get("title_from"),
                    exclude_ids_file=(str(root / exclude["path"])
                                      if exclude else None),
                    exclude_ids_reason=exclude.get("reason")))
    return jobs


# ------------------------------------------------------------------ queue depth
#: Files that end a pack's life in the queue (promote_queue writes them).
_TERMINAL_MARKERS = ("DONE", "SUPERSEDED", "PARKED")


def _is_cached_refusal(pack: Path) -> bool:
    """True when the drain will SKIP this pack: its last receipt is a
    REFUSED at a cacheable stage and was made at the manifest's current sha
    (`promote_queue.py`'s SKIPPED-REFUSED-UNCHANGED rule). Such a pack is
    not inventory the drain owes -- only a reseal (new manifest sha) makes it
    owed again, and the cap must not count it (task 686: ten refused
    OpenAlex packs held the whole 1,000,000-row cap)."""
    try:
        prior = json.loads((pack / "receipt.json").read_text())
        sha = hashlib.sha256((pack / "manifest.json").read_bytes()).hexdigest()
    except (OSError, ValueError):
        return False
    return (prior.get("status") == "REFUSED"
            and prior.get("stage") in AIC.PQ._CACHEABLE_REFUSAL_STAGES
            and prior.get("manifest_sha") == sha)


def queue_depth_rows(queue_dir: str) -> int:
    """Rows in packs that are READY, not yet terminal and not a cached
    refusal: the inventory the drain still owes (flow-doctrine #2: inventory
    rots; do not fill the queue faster than the one promotion door drains
    it)."""
    total = 0
    qd = Path(queue_dir)
    if not qd.is_dir():
        return 0
    for pack in qd.iterdir():
        if not (pack / "READY").is_file() or not (pack / "manifest.json").is_file():
            continue
        if any((pack / m).exists() for m in _TERMINAL_MARKERS):
            continue
        if _is_cached_refusal(pack):
            continue
        try:
            # the inventory the drain still OWES is the rows it will write (`rows_to_write_max`; a
            # pack composed before that name existed answers with its legacy `rows`, as before)
            total += MR.rows_to_write_max(json.loads((pack / "manifest.json").read_text())) or 0
        except (OSError, ValueError):
            continue
    return total


# ------------------------------------------------------------------ filter

def already_enqueued(queue_dir: str, pack_id: str) -> bool:
    """A pack_id counts as enqueued once its manifest.json exists -- the
    same file `compose_pack` writes, checked BEFORE embedding anything (an
    already-enqueued pack is a no-op here even if its vectors somehow
    changed; re-composing a landed/queued pack is a different, deliberate
    operation, not this driver's job)."""
    # a pack whose every id was already served composed to nothing: settled, never re-composed
    return ((Path(queue_dir) / pack_id / "manifest.json").exists()
            or AIC.SE.is_settled(queue_dir, pack_id))


#: Real incident, task 664, 2026-09-26: CC tranche2/4/5 composed at
#: 12:18-12:20Z under a manifest missing `family_license`, `plan()` then
#: REFUSED all three with "unresolved license" (promote_queue.py's own
#: `_CACHEABLE_REFUSAL_STAGES` cache -- correctly refuses to keep re-running
#: `plan()` on an unchanged manifest). #381 fixed the composer 12 minutes
#: later, but `already_enqueued()`'s plain existence check meant this
#: module never revisited them -- they had a manifest.json, so they read as
#: "done" forever, needing a human to notice, delete the stale pack dir by
#: hand, and force a rediscovery. `_refusal_is_stale` below closes that: a
#: pack whose last recorded attempt was a CACHEABLE content refusal
#: (promote_queue.py's own whitelist, reused rather than re-guessed) from
#: BEFORE the composer code last changed is eligible for automatic
#: recompose -- the fix that might cure it did not exist yet when the
#: refusal was written. Never applies to a transient-stage refusal (lock
#: contention, an API timeout -- promote_queue.py already retries those
#: itself every drain) or to a pack with no receipt yet (nothing refused).
def _compose_code_last_changed(repo_root: Path | None = None) -> str | None:
    """ISO-8601 commit time of the last commit touching either compose
    module (`git log -1 --format=%cI`) in `repo_root` (default: this
    module's own checkout -- the same self-refreshed tree `growth_pending_
    compose.sh` just reset to origin/main). `None` if git is unavailable or
    the call fails for any reason -- this NEVER blocks ordinary discovery,
    it only disables the staleness re-check for that one run (self-heals
    next tick)."""
    import subprocess
    root = repo_root or _REPO_ROOT
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "log", "-1", "--format=%cI", "--",
             "cytir/index/compose.py", "cytir/ops/growth_pending_compose.py"],
            capture_output=True, text=True, timeout=10, check=True)
    except Exception:  # noqa: BLE001 -- best-effort staleness check only
        return None
    ts = out.stdout.strip()
    return ts or None


def _refusal_is_stale(queue_dir: str, pack_id: str, *,
                      code_changed_at: str | None) -> bool:
    if code_changed_at is None:
        return False
    receipt_path = Path(queue_dir) / pack_id / "receipt.json"
    if not receipt_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if receipt.get("status") != "REFUSED":
        return False
    if receipt.get("stage") not in AIC.PQ._CACHEABLE_REFUSAL_STAGES:
        return False
    checked_at = receipt.get("checked_at")
    if not checked_at:
        return False
    # ISO-8601 `Z`/`+00:00`-suffixed timestamps compare correctly as plain
    # strings (same fixed-width format, same UTC offset convention every
    # caller in this codebase uses) -- no datetime parsing needed.
    return checked_at < code_changed_at


def pending_jobs(*, cc_root: str | None, se_root: str | None,
                 queue_dir: str | None = None,
                 nonso_root: str | None = None,
                 vectors_roots: list[str] | None = None,
                 sets_file: str | None = None,
                 corpus_root: str | None = None,
                 openlegaldata: bool = False,
                 family_stages: list[str] | None = None,
                 twin_url: str | None = None,
                 log=lambda msg: None) -> list[PendingJob]:
    queue_dir = queue_dir or AIC.default_queue_dir()
    code_changed_at = _compose_code_last_changed()

    def settled(pack_id: str) -> bool:
        # Enqueued and not a stale refusal: nothing for this sweep to do, and
        # no reason to probe the twin for it (see discover_se_pending).
        return (already_enqueued(queue_dir, pack_id)
                and not _refusal_is_stale(queue_dir, pack_id,
                                          code_changed_at=code_changed_at))

    jobs: list[PendingJob] = []
    if cc_root:
        jobs.extend(discover_cc_pending(cc_root, twin_url=twin_url, log=log))
    if se_root:
        jobs.extend(discover_se_pending(se_root, vectors_roots=vectors_roots,
                                        twin_url=twin_url, skip=settled,
                                        log=log))
    if nonso_root:
        jobs.extend(discover_nonso_pending(nonso_root,
                                           vectors_roots=vectors_roots,
                                           twin_url=twin_url, skip=settled,
                                           log=log))
    if sets_file:
        jobs.extend(discover_sets_pending(
            sets_file, corpus_root or "", twin_url=twin_url, skip=settled,
            log=log))
    if openlegaldata:
        jobs.extend(discover_openlegaldata_pending(
            corpus_root or "", twin_url=twin_url, skip=settled, log=log))
    for family_id in family_stages or []:
        jobs.extend(discover_family_stage_pending(
            corpus_root or "", family_id, twin_url=twin_url, skip=settled, log=log))
    return [j for j in jobs if not settled(j.pack_id)]


# --------------------------------------------------------------------- run

def run(*, cc_root: str | None, se_root: str | None,
       queue_dir: str | None = None, qdrant_url: str | None = None,
       ceilings: dict[str, int] | None = None,
       nonso_root: str | None = None,
       vectors_roots: list[str] | None = None,
       sets_file: str | None = None,
       corpus_root: str | None = None,
       openlegaldata: bool = False,
       family_stages: list[str] | None = None,
       only_packs: list[str] | None = None,
       max_queue_rows: int | None = None,
       prod_url: str | None = None, prod_collection: str = "aw_prod",
       log=lambda msg: print(msg, flush=True)) -> list[dict]:
    """Compose+enqueue every discovered pending job, embedding first if
    needed. One job's failure is logged and does NOT stop the others --
    each family/tranche/part is an independent unit of work, same
    discipline `promote_queue.py`'s own drain uses per pack.

    `prod_url` (ruling 23.5, task 702): with the twin (`qdrant_url`) it turns on the compose-time
    already-served exclusion for every job (`cytir.index.served_exclusion`). A pack whose ids are all
    served is recorded (`_composed-to-nothing/`) and reported as `all-served`, not as a failure."""
    ceilings = ceilings or {}
    queue_dir = queue_dir or AIC.default_queue_dir()
    jobs = pending_jobs(cc_root=cc_root, se_root=se_root, queue_dir=queue_dir,
                        nonso_root=nonso_root, vectors_roots=vectors_roots,
                        sets_file=sets_file, corpus_root=corpus_root,
                        openlegaldata=openlegaldata, family_stages=family_stages,
                        twin_url=qdrant_url, log=log)
    if only_packs:
        # an operator's narrowing (a dry rehearsal into a scratch queue, one stage at a time);
        # the unattended timer never passes it
        jobs = [j for j in jobs if j.pack_id in only_packs]
    results = []
    if not jobs:
        log("[growth-pending-compose] nothing pending")
        return results
    for job in jobs:
        if max_queue_rows is not None:
            depth = queue_depth_rows(queue_dir)
            if depth >= max_queue_rows:
                log(f"[growth-pending-compose] queue holds {depth:,} ready rows "
                    f"(cap {max_queue_rows:,}): {job.pack_id} and the rest wait "
                    "for the drain")
                results.append({"pack_id": job.pack_id, "status": "deferred"})
                continue
        log(f"[growth-pending-compose] {job.pack_id} ({job.family}): "
           f"chunks={job.chunks_dir} vectors={job.vectors_dir}")
        try:
            result = AIC.embed_compose_enqueue(
                pack_id=job.pack_id, family=job.family,
                chunks_dir=job.chunks_dir, vectors_dir=job.vectors_dir,
                collection=job.collection, mrl=job.mrl,
                qdrant_url=qdrant_url, queue_dir=queue_dir,
                license_from=job.license_from,
                family_license=job.family_license,
                family_payload=job.family_payload,
                family_payload_basis=job.family_payload_basis,
                title_from=([job.title_from] if job.title_from else ()),
                skip_embed=vectors_complete(job.chunks_dir, job.vectors_dir),
                served_targets=AIC.served_targets_from_args(
                    qdrant_url, job.collection, prod_url, prod_collection),
                exclude_ids_file=job.exclude_ids_file,
                exclude_ids_reason=job.exclude_ids_reason,
                ceiling=ceilings.get(job.family), log=log)
            if result.get("all_served"):
                results.append({"pack_id": job.pack_id, "status": "all-served"})
                continue
            results.append({"pack_id": job.pack_id, "status": "enqueued",
                            "ready_sha": result["ready_sha"]})
        except Exception as exc:  # noqa: BLE001 -- one job's failure, not the run's
            log(f"[growth-pending-compose] {job.pack_id}: FAILED "
               f"{type(exc).__name__}: {exc}")
            results.append({"pack_id": job.pack_id, "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}"})
    return results


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cc-root", default=None,
                    help="task634 work root, e.g. <work-root>/task634 "
                         "(omit to skip CC discovery)")
    ap.add_argument("--se-root", default=None,
                    help="n1-stackexchange-v1 work root (omit to skip SE discovery)")
    ap.add_argument("--se-nonso-root", default=None,
                    help="n1-stackexchange-nonso-v1 work root -- one "
                         "<site[-partN]>/stage-413a-portable-v3 per pack "
                         "(omit to skip non-SO SE discovery)")
    ap.add_argument("--vectors-root", action="append", default=[],
                    metavar="DIR",
                    help="repeatable: a `vectors/` tree holding "
                         "release=*/source=<label>[/vectors] (the corpus "
                         "`vectors/` dir). The SE root's own vectors/ is "
                         "always searched")
    ap.add_argument("--sets-file", default=None,
                    help="data/growth-pending-sets.v1.json -- declared sets "
                         "(big or trimmed packs, cut into sub-packs); held "
                         "sets are logged, never composed")
    ap.add_argument("--corpus-root", default=None,
                    help="root the sets file's relative paths resolve against")
    ap.add_argument("--openlegaldata", action="store_true",
                    help="discover OpenLegalData chunk stages under --corpus-root "
                         "(needs vectors AND L1 filed; never embeds)")
    ap.add_argument("--family-stage", action="append", default=[], metavar="FAMILY",
                    help="repeatable: discover `cytir.sources.family_stage` stages of this "
                         "family under --corpus-root (needs L1 filed AND an overlap check; "
                         "embeds under the GPU lock)")
    ap.add_argument("--only-pack", action="append", default=[], metavar="PACK_ID",
                    help="repeatable: compose only these discovered packs (operator rehearsal)")
    ap.add_argument("--max-queue-rows", type=int, default=None,
                    help="stop composing once the queue holds this many READY, "
                         "not-yet-terminal rows; the rest wait for the drain "
                         "(default: no cap)")
    ap.add_argument("--queue-dir", default=None,
                    help="promotion queue (default: the operator setting queue_dir)")
    ap.add_argument("--qdrant-url", default=TWIN.url(),
                    help="twin URL to check mrl against the served dense width; "
                         "empty string to skip the check")
    ap.add_argument("--prod-url", default=None,
                    help="prod URL (the tunnel); with --qdrant-url it declares, at compose, the "
                         "ids the twin AND prod already serve as the pack's exclusion "
                         "(ruling 23.5). Omit: not checked.")
    ap.add_argument("--prod-collection", default="aw_prod")
    ap.add_argument("--ceiling", action="append", default=[],
                    metavar="FAMILY=N",
                    help="repeatable: a per-family enqueue ceiling, e.g. "
                         "eric_official_pdf=4500000")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # Fail at START, before the first job and before the queue is read: the GPU lock directory
    # and (unless --queue-dir names one) the queue are operator settings, and a job that refused
    # one in the middle of a run left the unit red only after discovery had already run.
    needed = ["gpu_lock_dir"] + ([] if args.queue_dir else ["queue_dir"])
    code = PATHS.startup_refusal("growth_pending_compose", *needed)
    if code:
        return code
    args.queue_dir = args.queue_dir or AIC.default_queue_dir()
    # task 696 slice 4: say where the queue and the write locks are, and why (never raises).
    print(PATHS.location_line("queue_dir", args.queue_dir, "queue_dir"), flush=True)
    print(PATHS.lock_location_line(), flush=True)
    ceilings = {}
    for item in args.ceiling:
        family, _, value = item.partition("=")
        ceilings[family] = int(value)
    results = run(cc_root=args.cc_root, se_root=args.se_root,
                  queue_dir=args.queue_dir,
                  qdrant_url=args.qdrant_url or None, ceilings=ceilings,
                  nonso_root=args.se_nonso_root,
                  vectors_roots=args.vectors_root,
                  sets_file=args.sets_file, corpus_root=args.corpus_root,
                  openlegaldata=args.openlegaldata,
                  family_stages=args.family_stage,
                  only_packs=args.only_pack or None,
                  max_queue_rows=args.max_queue_rows,
                  prod_url=args.prod_url, prod_collection=args.prod_collection)
    failed = [r for r in results if r["status"] == "failed"]
    print(json.dumps({"jobs": len(results), "failed": len(failed)}, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
