#!/usr/bin/env python3
# Moved from the pipeline-era `promote_sealed_pack` module (task 677, task 643 S5); the old path was retired.
"""Promote a SEALED 1024-d fp16 expansion pack into a 256-dim serving collection.

Task 629 (W1 / Pool A). This is the one capability the 100M spec named as
missing: *"nothing in this repo has yet promoted a sealed 1024-d pack into a
256-dim serving collection"* (`specs/portfolio-100m.md` §1.4). Every component it
needs already existed and is reused here rather than reimplemented; what did not
exist was the glue that **mints new points from a pre-computed vector shard**.

WHAT ALREADY EXISTED, AND IS NOT COPIED
---------------------------------------
* `cyt index surface-delta` reads exactly this artefact shape
  (`shard-*.vectors.f16` + `shard-*.row_ids.jsonl` + `chunks/shard-*.texts.jsonl.gz`)
  — but it is SAME-ID-ONLY: its join guard refuses unless every id is already in
  the target, and it writes `PUT /points/vectors`, a vector-only update. It
  cannot admit a row.
* `serving_config.mrl_trunc` truncates AND RENORMALISES (the collection distance
  is Cosine; the renormalise is not optional).
* `aw_b1_projection.project_record` mints the point id, composes the dense and
  sparse surfaces, and builds the payload — with `require_license` and
  `validate_precomputed_identity` already on its signature.
* `aw_b1_projection.saturated_bm25` builds the tf-saturated `bm25` posting. The
  sealed packs are DENSE ONLY (`vector_bytes == rows * 1024 * 2`), so the sparse
  leg is regenerated here, through the project's own tokeniser, never a second
  implementation of it.
* `twin_first_apply.twin_first` sequences preflight → preservation → lock → twin →
  verify → gate → prod → verify, ledgering every transition.
* Product quantization needs nothing from an applier: the config is set once at
  collection creation (`serving_config.quantization_config`) and Qdrant re-derives
  codes when it indexes the segment. Note the consequence, which this module
  reports rather than hides: quantization is SKIPPED in appendable segments, so
  freshly upserted points serve un-quantized until the optimizer indexes them and
  a gate arm cut before then measures a different ranker.

THE FOUR THINGS THIS MODULE REFUSES ON, AND WHY EACH ONE IS HERE
----------------------------------------------------------------
1. **A row whose recomputed point id disagrees with the pack's own** — under
   `--identity-check reproduce`, the default. The vector shard and the text shard
   are joined by `point_id`, and if the id this projection mints is not the id the
   pack recorded, the vector we would store was computed for a different row than
   the payload beside it — the defect task 546 found shipping silently (context7:
   53,822 of 84,961 staged rows addressing nothing, while the collection's real
   rows for the same URLs sat under other ids, invisible to `points_count`).

   **The 2026-09-22 "measured limit" was WRONG, and the record is corrected
   rather than quietly dropped.** It read: *"Pool A's own point ids could NOT be
   reproduced from their rows … 252 combinations were tried … and none minted
   the pack's `ec99b688-…`"*, and the `remint-b1` ruling of the same day rests
   entirely on it. Task 629 re-ran the search on 2026-09-23 over
   url-field × anchor-form × text-field × uuid5 namespace × key shape, found the
   recipe on the FIRST row of each pack, and then re-derived **every** row of
   all four W1 packs:

       B01 technical_manufacturer      445 /    445   anchor `document:<doc>:chunk-<n>`
       B02 service_public_fiche      8,324 /  8,324   anchor `chunk-<n>`
       B03 statcan_wds              10,136 / 10,136   anchor `<source_record_id>`
       B05 rfc_editor_net_new       61,212 / 61,212   anchor `chunk-<n>`

   All four are `aw_common.mint_passage_point_id(url, anchor, text)` over
   `canonical_text` (or `text`, where the pack has no `canonical_text`) — which
   is exactly what `pack_surface.py`'s docstring said all along. The earlier
   search missed the anchor forms the staging module actually mints
   (`stage_structured_markdown_for_embedding.py`, the packs' own producer).

   **RULED 2026-09-23 (coordinator): `pack-declared` IS the canonical mint for
   W1, and `remint-b1` is RETIRED for it.** `mint_passage_point_id` is the one
   home for a source-native passage id, so writing the pack's id is writing the
   id the scheme prescribes — not an acceptance on trust. What makes it safe
   here is that the id is arithmetic this module can redo, per family, and the
   family contract records the recipe and the count beside the mode.

   `remint-b1` stays in `IDENTITY_MODES` because it is the right answer for a
   pack whose ids genuinely cannot be re-derived: it stores the id
   `aw_b1_identity` mints, which is the id a b1 rebuild will mint, with the
   receipt carrying a `pack_id -> minted_id` map so the sealed vectors stay
   traceable. It was simply not the right answer for W1, and it could not have
   been for B02 or B05 at all: `aw_b1_identity.mint_point_id` has no ordinal in
   its anchor, and one service-public fiche is 1,462 chunks over 21 distinct
   headings, so the re-mint collides by construction.

   `reproduce` stays the default: it compares against what *this* projection
   mints (the b1 scheme), which is a different and stricter question.
2. **An id the target already holds.** This module ADMITS; it does not overwrite.
   An existing id means the row is superseded, not new, and silently overwriting
   it would make a supersession look like growth (memory
   `present-is-not-complete`). Promoting such a row is a different operation with
   a different gate, so it is named (`--allow-supersede REASON`) and the reason
   lands in the receipt.
3. **A page the target already serves, under any id.** Refusal 2 asks the right
   question of the wrong key: an id encodes the mint scheme that made it, so a
   pack minted by `aw_common.mint_passage_point_id` never collides with the
   incumbent rows the b1 rebuild minted for the SAME url. Task 629 measured two
   real W1 packs where that was 99% of the pack -- `service_public_fiche`
   2,958 of 2,997 urls, `statcan_wds` 10,040 of 10,136 -- both reporting
   `already_present: 0`. `already_served_urls` closes it on the url, which
   survives a change of mint scheme, and it is cheap on a net-new pack because
   `url` is indexed and the scroll only runs for a batch that already counted
   non-zero.
4. **An unlicensed row.** ADR-039's amendment scopes `require_license=False` to
   *replacement of existing points* and says in terms that it "may admit no new
   unlicensed content". This is net-new admission, so `require_license=True` is
   not configurable here.

DEFAULT IS A DRY RUN. Without `--apply` this reads, projects, joins, runs every
guard and writes the receipt — and makes no write of any kind.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import gzip
import hashlib
import json
import os
import pathlib
import threading
import time
from typing import Iterable, Iterator, Mapping

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]

from cytir.core import projection as PROJ  # noqa: E402
from cytir.core import common as AC  # noqa: E402
from cytir.core.search_result_url import rejected_reason  # noqa: E402 -- task 2242
from cytir.index import layout as LAYOUT  # noqa: E402
from cytir.index import payload_carry as PC  # noqa: E402
from cytir.corpus import persisted as PI  # noqa: E402
from cytir.core import serving_config as serving_config  # noqa: E402
from cytir.index import payload_profile as PP  # noqa: E402 -- the ONE payload projection (task 675)
from cytir.index import sparse_rule as SR  # noqa: E402 -- the ONE BM25 prune rule (task 717)
from cytir.index import dry_run as DRY  # noqa: E402 -- a dry drain never writes (2026-10-03)
from cytir.index import prod_guard as PG  # noqa: E402 -- no scroll, and no replay, reaches prod (2026-10-07)
from cytir.core import internal_http as IH  # noqa: E402 -- the ONE door to Qdrant (consent step 5, task 1648)
from cytir.index import chunk_checkpoint as CK  # noqa: E402 -- chunk-level resume of long prod calls (task 786)
from cytir.index import id_rule as IR  # noqa: E402 -- the ONE I1 id rule; the aw_e10 writer guard (task 847)

SCHEMA = "aw.sealed-pack-promotion.v1"

#: The sealed packs' stored width. Not a parameter: it is a property of the
#: artefact (`vector_bytes == rows * 1024 * 2` for all 32 admitted packs), and a
#: shard whose manifest disagrees is a different artefact that must not be read
#: through this reader.
SEALED_DIM = 1024

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"

#: Coordinator's ruling, task 664, 2026-09-26: a pack whose urls this
#: collection already serves (under other point ids) refuses WHOLE only
#: past this share -- below it, the served rows are dropped and the rest
#: lands (see `plan()`'s already_served_urls block). Mirrors the mechanical
#: content gate's word-floor ceiling (promote_queue.py's
#: WORD_FLOOR_FAIL_SHARE_MAX) -- same "an ordinary overlap at scale is not
#: a duplicate-pack signal" reasoning, a different check.
ALREADY_SERVED_FAIL_SHARE_MAX = 0.05


class PromotionError(RuntimeError):
    """A refusal. Never caught inside this module and never downgraded."""


class AlreadyLandedError(PromotionError):
    """Some or all of `plan()`'s own POINT IDS are already in `collection` --
    a distinct subclass of the generic already-present refusal (task 664,
    2026-09-26 incident), not because the write itself is treated any
    differently here, but so a caller with TWO targets (twin, then prod)
    can tell "this population already landed HERE" apart from every other
    refusal reason (identity mismatch, join gap, missing persisted input,
    payload-carry, already-served-under-a-different-id) without parsing a
    message string.

    THE INCIDENT THIS EXISTS FOR: prod's qdrant restarted mid-write during
    `cc-authority-frontier-w7-tranche2-d`'s PROD leg, after its TWIN leg had
    already landed and verified (185,184 ids). The next drain re-planned
    against the TWIN (as `dry_plan()` always does) and got refused with the
    plain `PromotionError` this used to raise -- a stage-"plan" refusal,
    cached forever by `promote_queue.py`'s skip-cache, so the twin/prod gap
    could never close by itself. `written_ids` (never `present`, the
    ALREADY-LANDED subset) rides on the exception so a caller can ask the
    SAME question of a DIFFERENT target (did prod get them too, or just
    twin?) without re-deriving the join.

    Note the name: it is raised when ANY of the ids is present, not only when
    all of them are. `present` (task 690) is the already-landed subset itself. The caller
    needs it to tell a RESUME (twin holds this pack's ids because this pack,
    or a duplicate of it, landed there) from a PARTIAL OVERLAP (an earlier,
    different pack landed a handful of the same ids on both targets, and the
    rest are new). Task 690: `eric-official-pdf-tranche-130000` shared ONE id
    of 169,070 with its own smoke pack, and the drain reported "169069 of
    169069 on NEITHER twin nor prod" without naming that one id.
    """

    def __init__(self, message: str, *, ids: frozenset[str] = frozenset(),
                 present: frozenset[str] = frozenset()):
        super().__init__(message)
        self.ids = ids
        self.present = present


class ResumeCollisionError(AlreadyLandedError):
    """A prod-leg RESUME (task 788) found ids on the target that the twin leg did NOT write.

    A resume expects prod to hold a subset of the twin leg's own written ids (its partial write).
    An id outside that set is not our partial write: it is a genuine collision (the remint-b1
    case, task 664), and writing next to it would be a supersession. `present` is the colliding
    subset only."""


def resume_partition(written_ids: list[str], present: Iterable[str],
                     twin_ids: Iterable[str], *, collection: str) -> dict:
    """Split a sealed pack's projected ids for a prod-leg RESUME (task 788). Pure: no network.

    `written_ids`: what `plan()` projects for this pack. `present`: the subset the target holds
    (asked by id, `already_present`). `twin_ids`: the ids the twin leg wrote and verified (the
    resume's source of truth).

    The rule a first-time apply uses ("any id present refuses") is replaced by the one a resume
    needs: present ids MUST be twin-written ids (our own partial write; skipped, never re-sent),
    and the ids to write are exactly the twin-written ids the target lacks. Refuses on
      * a present id the twin leg did not write -> `ResumeCollisionError` (a real collision);
      * a twin-written id this pack does not project -> `PromotionError` (the pack or its scope
        changed since the twin leg; a resume would not be the same population).
    """
    projected = set(written_ids)
    twin = set(twin_ids)
    have = set(present)
    foreign = sorted(have - twin)
    if foreign:
        raise ResumeCollisionError(
            f"resume: {len(foreign)} of {len(have)} id(s) already in {collection} are NOT in the twin "
            f"leg's written ids (e.g. {foreign[:3]}). That is not this pack's partial write; it is a "
            "collision with rows another write put there. Overwriting them is a supersession, a "
            "different operation with a different gate. Reconcile by hand (task 664 protocol).",
            ids=frozenset(written_ids), present=frozenset(foreign))
    unknown = sorted(twin - projected)
    if unknown:
        raise PromotionError(
            f"resume: {len(unknown)} of {len(twin)} twin-written id(s) are not projected by this "
            f"pack now (e.g. {unknown[:3]}). The pack or its scope changed after the twin leg; "
            "refusing to resume a different population.")
    # The write set is the TWIN's decision, never the pack's: a projected id the twin leg did not
    # write (the drain dropped it: a mechanical-content problem, a prose/binary hold-out, an
    # already-served url) is skipped and counted. Tranche 14 (2026-10-09): a resume that wrote the
    # pack instead put 213 gate-rejected rows on prod only.
    to_write = [pid for pid in written_ids if pid in twin and pid not in have]
    return {"twin_written_ids": len(twin), "projected": len(projected),
            "present_before": len(have),
            "dropped_by_twin_decision": len(projected - twin),
            "to_write": len(to_write), "_to_write": to_write,
            "_present": sorted(have), "_dropped": sorted(projected - twin)}


# --------------------------------------------------------------- reading a pack
def _text_shards(chunks_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(chunks_dir, "shard-*.texts.jsonl.gz")))


def _vector_manifests(vec_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(vec_dir, "*.vectors.manifest.json")))


def read_records(chunks_dir: str) -> Iterator[dict]:
    """Every chunk row of a pack, in shard order."""
    shards = _text_shards(chunks_dir)
    if not shards:
        raise PromotionError(
            f"no shard-*.texts.jsonl.gz under {chunks_dir} — for the two "
            "partitioned OpenAlex tranches the chunk shards live under "
            "`partitioned/.../workers/partition-NN/pack/chunks` (and under the "
            "`resume-v2` checkpoint for partitions 00 and 01), not the pack root")
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def read_vectors(vec_dir: str, mrl: int) -> tuple[np.ndarray, list[str]]:
    """The sealed 1024-d fp16 matrix, MRL-truncated to `mrl` and RENORMALISED.

    The row order is the manifest's own, and the ids come from the sibling
    `row_ids.jsonl` in that same order — which is what the manifest's `ordering`
    field asserts. A count mismatch between the two is a refusal rather than a
    zip() that silently truncates to the shorter one.
    """
    manifests = _vector_manifests(vec_dir)
    if not manifests:
        raise PromotionError(f"no *.vectors.manifest.json under {vec_dir}")
    mats, ids = [], []
    for man_path in manifests:
        man = json.loads(pathlib.Path(man_path).read_text())
        tag = os.path.basename(man_path).removesuffix(".vectors.manifest.json")
        f16 = os.path.join(vec_dir, tag + ".vectors.f16")
        rowids = os.path.join(vec_dir, tag + ".row_ids.jsonl")
        raw = np.fromfile(f16, dtype="<f2")
        n = int(man["n"])
        if raw.size != n * SEALED_DIM:
            raise PromotionError(
                f"{f16}: {raw.size} float16 values is not n*{SEALED_DIM} "
                f"({n}*{SEALED_DIM}={n * SEALED_DIM}) — this reader is for the "
                "sealed 1024-d artefact and must not reshape a different one")
        shard_ids = [json.loads(x)["point_id"]
                     for x in open(rowids, encoding="utf-8") if x.strip()]
        if len(shard_ids) != n:
            raise PromotionError(
                f"{rowids} has {len(shard_ids)} ids for {n} vectors — the "
                "manifest's ordering contract is what aligns them, so a "
                "mismatch means the pair cannot be joined at all")
        mats.append(raw.reshape(n, SEALED_DIM).astype("float32"))
        ids += shard_ids
    matrix = np.concatenate(mats, 0) if mats else np.zeros((0, SEALED_DIM), "float32")
    # ONE home for truncate+renormalise, shared with stage_pack_b1 and the
    # serving config, so a promotion and a rebuild cannot disagree by an ulp.
    return np.asarray(serving_config.mrl_trunc(matrix, mrl), dtype="float32"), ids


# ------------------------------------------------- the family's payload fields
#: `payload_carry_contract.RECORD_CONTRACT` requires a pre-projection record to
#: carry `url`, `source_type`, `title`, `trust_tier`, `lang` and `domain` at >=99%.
#: **Pool A's rows carry only `url` and `lang`.** So a promotion cannot just read
#: the rows: two of the missing four are mechanical functions of the row, and two
#: are DECLARED properties of the family.
#:
#: This is guard #1 of the 100M spec (`the family declaration, declared BEFORE
#: acquisition`) doing its job at the last possible moment. `source_type` and
#: `trust_tier` are POLICY facts -- which §4 layer the family sits in, and which
#: trust class that layer grants -- so this module REFUSES rather than guessing
#: them. They come from the family contract, passed in as `--family-payload`.
#:
#: The immediate consequence, worth stating because it looks like a nuisance and
#: is actually the point: without a declared contract, NOTHING promotes. The 99%
#: floor cannot be met by a pack whose rows never carried the fields.
DECLARED_PAYLOAD_FIELDS = ("source_type", "trust_tier")
DERIVABLE_PAYLOAD_FIELDS = ("domain", "title")


def derive_domain(url: str) -> str | None:
    """The registrable host of a url, as production's `domain` field spells it.

    Mechanical: `aw_b1_projection` reads `record["domain"]` and never derives it,
    so a row without one resolves NO licence (every host rule is keyed on it) and
    fails the payload floor. That is why the whole news family read `unresolved
    license` on a host that is IN the table.

    The fold is `aw_common.fold_www`, not a copy of it. This function used to
    strip a literal `www.` of its own, which is how B03 (statcan) came within
    one dry run of entering as `domain = www150.statcan.gc.ca` while all 13,861
    rows of its own family already serving carry `domain = statcan.gc.ca` --
    the licence would still have resolved (longest-suffix) and the family would
    have been split in two for every `domain` filter and every site-search.
    Task 629, measured on the twin before the change; the counts are in
    `fold_www`'s docstring.
    """
    if not url:
        return None
    from urllib.parse import urlsplit
    return AC.fold_www(urlsplit(url).hostname or "") or None


def with_domain(record: dict) -> dict:
    """`record` plus its derived `domain`, if it did not carry one.

    Separate from `apply_family_payload` because it is not about a family's
    declaration at all: it is a mechanical read of the row's own url, and it is
    what every licence host rule is keyed on. Never overwrites a domain the row
    carries -- the pack is the authority on its own rows.
    """
    if record.get("domain"):
        return record
    domain = derive_domain(record.get("url") or "")
    if not domain:
        return record
    return {**record, "domain": domain}


def apply_family_payload(record: dict, declared: Mapping,
                         title_from: Iterable[str]) -> tuple[dict, list[str]]:
    """Fill the payload fields the row does not carry. Returns (record, missing).

    Never overwrites a value the row already has -- the pack is the authority on
    its own rows; this only fills absences.
    """
    out = dict(with_domain(record))     # one home for the derivation
    if not out.get("title"):
        for path in title_from:
            value = dotted(out, path)
            if isinstance(value, str) and value.strip():
                out["title"] = value.strip()
                break
    for field in DECLARED_PAYLOAD_FIELDS:
        if not out.get(field) and declared.get(field) is not None:
            out[field] = declared[field]
    missing = [f for f in (*DECLARED_PAYLOAD_FIELDS, *DERIVABLE_PAYLOAD_FIELDS)
               if not out.get(f)]
    return out, missing


# ------------------------------------------------- POLICY 4.7 publication date
#: POLICY 4.7: *"`published_date` capture is mandatory at ingest (a news record
#: without a date is inadmissible -- it cannot age out)"*. Pool A's news packs
#: carry the date only NESTED, at `source_metadata.publication_time`, beside a
#: `source_metadata.publication_time_status` that says whether the PUBLISHER
#: declared it. Both matter: a value whose status is `not_declared_by_publisher`
#: is our inference, not the publisher's statement, and admitting it would make
#: the rolling window turn on a date nobody published. Measured on
#: news-current-articles-v5b: 3,333 of 4,993 documents carry a source-declared
#: time.
#:
#: The rule is EXCLUDE, never backfill -- the same disposition
#: the retired `apply_news_dates` script already applies post-hoc for staged news
#: collections (task 303). Applying it at ADMISSION means an undated row never
#: enters, rather than entering and being deleted later.
DEFAULT_PUBLISHED_STATUS_REQUIRES = "source_declared"


def dotted(record: Mapping, path: str):
    """`a.b.c` over nested mappings; None if any hop is absent."""
    cursor: object = record
    for part in path.split("."):
        if not isinstance(cursor, Mapping):
            return None
        cursor = cursor.get(part)
    return cursor


def lift_published(record: dict, *, path: str, status_path: str | None,
                   status_requires: str | None) -> tuple[dict, str | None]:
    """Return (record with `published` set, reason it was NOT set).

    Never mutates the input, and never invents a date: the only value it writes
    is the one the record already carries at `path`.
    """
    value = dotted(record, path)
    if not value:
        return record, f"no value at {path}"
    if status_path:
        status = dotted(record, status_path)
        if status_requires and status != status_requires:
            return record, (f"{status_path} is {status!r}, not "
                            f"{status_requires!r} -- that date is not the "
                            "publisher's statement")
    return {**record, "published": value}, None


#: task 641: `aw_b1_projection.project_record` resolves a licence from
#: `record["license"]` (a truthy top-level value wins) or, failing that, a
#: per-host rule in `cytir.core.license`. Some families' STAGED rows
#: carry the licence only nested (e.g. stackexchange_nonso_qa_p0's per-answer
#: `source_metadata.accepted_answer.license` -- Stack Exchange licences are
#: per-record, not host-uniform, so no host rule can stand in for it) --
#: `stage_structured_markdown_for_embedding.py` dropped the top-level field the
#: same way it dropped `source_type`/`trust_tier`/`title`, which is exactly
#: why `apply_family_payload`/`--title-from` already exist for those three.
#: `--license-from` is the same lift for the fourth field, mirroring
#: `lift_published` byte-for-byte rather than growing a second mini-DSL.
def lift_license(record: dict, *, path: str) -> tuple[dict, str | None]:
    """Return (record with `license` set, reason it was NOT set).

    Never overwrites a `license` the row already carries at the top level --
    a family whose rows are already self-sufficient (e.g. `loda_exact`) must
    not have a lift silently substitute a different value for one the row
    already asserts.
    """
    if record.get("license"):
        return record, None
    value = dotted(record, path)
    if not value:
        return record, f"no value at {path}"
    return {**record, "license": value}, None


# ------------------------------------------------------------------- projection
#: The full precomputed-identity set `aw_b1_projection._validate_precomputed_identity`
#: checks. Pool A's packs predate it: they carry `point_id` and nothing else from
#: this list, so asking for the full check refuses every row on
#: `canonical_url: None`. The strong check is used where the record can answer it
#: and the id check is used everywhere; which one ran is COUNTED, never assumed.
FULL_IDENTITY_FIELDS = ("canonical_url", "content_digest", "identity_tier",
                        "identity_anchor_component", "point_id", "id_salt")


IDENTITY_MODES = ("reproduce", "pack-declared", "remint-b1")


def project_rows(records: Iterator[dict], *, consent: dict | None,
                 attribution: str | None, license_version: str | None,
                 identity_check: str = "reproduce",
                 identity_reason: str | None = None,
                 published_from: str | None = None,
                 published_status_from: str | None = None,
                 published_status_requires: str | None = None,
                 require_published: bool = False,
                 family_payload: Mapping | None = None,
                 title_from: Iterable[str] = (),
                 license_from: str | None = None,
                 family_license: str | None = None,
                 exclude_ids: frozenset | Mapping[str, str] | None = None,
                 id_rule: str | None = None):
    """`project_record` over a pack's rows, with identity and licence enforced.

    `id_rule` (task 864, the aw_e10 writer contract): the `id_rule` the TARGET collection declares
    (`cytir.index.id_rule.declared_id_rule`; `plan()` passes it). `"I1"`: whatever the identity mode,
    the id WRITTEN is `id_rule.mint_i1` of the projected row (ADR-067 I1 + am.1: a card with an
    identity gets `canon#card:<identity>`), and the pack's own id stays the vector join key
    (`vector_key`), as under `remint-b1`. The identity check of the mode still runs first and
    still refuses a row it refused before: it proves the vector belongs to the row. Two rows of one
    pack that mint ONE I1 id (identical text on one page) are one record under I1: the first is
    written, the others are counted in `i1_merged` and listed in `i1_merged_rows` (never silently
    dropped). A row with no I1 id refuses the pack. None = every collection before aw_e10,
    unchanged.

    Identity is the guard that makes the vector/text join trustworthy: the vector
    shard is addressed by `point_id`, so if the id this projection mints from the
    row's own `canonical_url` + anchor is not the id the pack recorded, the vector
    we would store belongs to a different row than the payload beside it. That is
    the defect task 546 found shipping silently.

    Two strengths of that check exist, and this returns which one each row got:

    * `full` — `project_record(validate_precomputed_identity=True)`, which checks
      all six of `FULL_IDENTITY_FIELDS`. Available only for rows that carry them.
    * `point_id` — the recomputed id compared against the row's own `point_id`.
      Weaker (it does not re-verify the digest or the tier) and still decisive for
      the join, because the join key IS the id.

    A row carrying NO `point_id` is a refusal in every mode: there would be
    nothing to compare and nothing to address the vector by.

    Three modes, and the returned mapping says which one every row took:

    * `reproduce` (default) — the minted id must equal the pack's, or refuse.
    * `pack-declared` — accept the pack's id, with a reason the receipt carries.
      It does NOT weaken the join (the record is still matched to its vector by
      its own id) but the pairing rests on the pack's receipts rather than on
      arithmetic redone here, so those rows are counted separately.
    * `remint-b1` (RULED for this wave) — store the id `aw_b1_identity` mints.
      The join to the vector still runs on the PACK's id, because that is what
      addresses the shard; only what gets WRITTEN changes. Every row's
      `pack_id -> minted_id` pair is returned so the receipt can carry the map.

    `require_license=True` is ADR-039's amendment, not a parameter here.

    `exclude_ids` (orchestrator ruling, 2026-09-23, CC tranche 1, task 641):
    ids a caller has already determined are out of the product's language
    scope (`mechanical_content_check.language_scope_filter`, run on the
    pack's RAW rows before this). Excluded here, BEFORE the payload-carry
    census below ever sees them -- a population-level presence floor is
    meant to measure whether IN-SCOPE rows carry their fields, not to be
    dragged down by rows that were never going into the index at all.

    A plain `frozenset`/`set` excludes with the reason
    `"out_of_scope_language"` (unchanged default, every existing caller).
    A `Mapping[str, str]` (id -> reason) carries its OWN per-id reason
    instead -- Tim's ruling 2026-09-24 (CC tranche 1, task 652) excludes a
    second, larger and DIFFERENTLY-CAUSED population (rows the detector
    could not confidently classify at all, not rows it confidently placed
    outside EN/DE/FR/IT), and a row's excluded-because reason is exactly
    the fact `mechanical_content_check.lang_drop_detail`'s receipt exists
    to preserve (corpus-doctrine: every row reaches one terminal, with a
    reason that is actually true of it).
    """
    if identity_check not in IDENTITY_MODES:
        raise PromotionError(f"--identity-check must be one of {IDENTITY_MODES}")
    if id_rule not in (None,) + IR.ID_RULES:
        raise PromotionError(f"unknown target id_rule {id_rule!r} (known: {IR.ID_RULES})")
    if identity_check == "pack-declared" and not identity_reason:
        raise PromotionError(
            "--identity-check pack-declared requires a reason: it accepts ids "
            "this module cannot re-derive, and an unexplained acceptance is how "
            "a trust decision becomes a default")
    out, skipped = {}, 0
    modes = {"full": 0, "point_id": 0, "pack_declared": 0, "reminted": 0}
    i1_merged: list[dict] = []
    #: I1 id -> [(pack id, joined entry)] of every row of the pack that mints it (more than one)
    i1_groups: dict[str, list] = {}
    if id_rule is not None:
        modes.update(id_rule=id_rule, i1_reminted=0, i1_merged=0)
    #: rows excluded by the POLICY 4.7 date rule, with the reason -- reported,
    #: never silently dropped (ADR-086: every row reaches one terminal).
    undated: list[dict] = []
    #: rows excluded because their detected language is outside the
    #: product's scope (`exclude_ids`) -- reported the same way `undated`
    #: is, never silently dropped.
    out_of_scope_dropped: list[dict] = []
    #: task 2224: rows `project_record` rejected (site-search result URL, empty url/text),
    #: with the reason -- a reported terminal; their vectors are never written.
    projection_rejected: list[dict] = []
    #: minted id -> the pack id whose vector it carries. The join to the vector
    #: shard runs on the PACK id in every mode, so this is what lets a re-minted
    #: row still find its embedding -- and what makes the sealed pack's lineage
    #: traceable from a served point.
    vector_key: dict[str, str] = {}
    for rec in records:
        declared = rec.get("point_id")
        if not declared:
            raise PromotionError(
                "a pack row carries no `point_id`; the vector shard is addressed "
                "by id, so such a row can be neither joined nor checked")
        if exclude_ids and str(declared) in exclude_ids:
            reason = (exclude_ids[str(declared)]
                     if isinstance(exclude_ids, Mapping)
                     else "out_of_scope_language")
            out_of_scope_dropped.append(
                {"point_id": declared, "url": rec.get("url"), "reason": reason})
            continue
        # DOMAIN FIRST, and unconditionally. `aw_b1_projection` reads
        # `record["domain"]` and never derives it, every licence host rule is
        # keyed on it, and a row without one resolves NO licence -- so this is
        # licence-critical. It used to happen only inside `apply_family_payload`,
        # i.e. only when a caller passed `--family-payload`, which couples a
        # mechanical derivation to an unrelated option: a pack whose rows already
        # carry every declared payload field but no `domain` fails with
        # "unresolved license" on a host that is IN the table, and the operator's
        # fix is to pass a family payload they do not otherwise need. Found while
        # dry-running B01's own fixture (task 629, 2026-09-22), before the batch
        # rather than during it.
        rec = with_domain(rec)
        if family_payload is not None:
            rec, missing = apply_family_payload(rec, family_payload, title_from)
            if missing:
                raise PromotionError(
                    f"{rec.get('url')!r}: the record still lacks "
                    f"{missing} after the family's declared payload was applied. "
                    "`payload_carry_contract.RECORD_CONTRACT` floors all of "
                    "url/source_type/title/trust_tier/lang/domain at 99%, and "
                    "Pool A's rows carry only url and lang, so `source_type` and "
                    "`trust_tier` MUST come from the family contract "
                    "(--family-payload) and `title` from a declared path "
                    "(--title-from). This module will not guess a POLICY trust "
                    "class.")
        if published_from:
            rec, why = lift_published(
                rec, path=published_from, status_path=published_status_from,
                status_requires=published_status_requires)
            if why and require_published:
                undated.append({"point_id": declared, "url": rec.get("url"),
                                "reason": why})
                continue
        if license_from:
            rec, _why = lift_license(rec, path=license_from)
            # No refusal on a missing lift here: `require_license=True` inside
            # `project_record` below is the one place that enforces a licence
            # exists, so a row this lift could not fill still gets the SAME
            # "unresolved license" refusal every other family gets, not a
            # second, earlier error message for the same fact.
        full = all(rec.get(f) is not None for f in FULL_IDENTITY_FIELDS)
        row = PROJ.project_record(
            rec, consent=consent, attribution=attribution,
            license_version=license_version,
            validate_precomputed_identity=full, require_license=True,
            family_license=family_license)
        if row is None:
            skipped += 1
            # task 2224: `project_record` returns None for exactly two causes; name the
            # one that fired so the row reaches a REPORTED terminal (ADR-086).
            # task 2242: compose's reason vocabulary, one shared constant.
            projection_rejected.append({"point_id": str(declared), "url": rec.get("url"),
                                        "reason": rejected_reason(rec.get("url"))})
            continue
        point_id, dense_text, sparse_text, payload = row
        if str(point_id) != str(declared):
            if identity_check == "pack-declared":
                # The pack's id is both the join key and the target address.
                point_id = str(declared)
                modes["pack_declared"] += 1
            elif identity_check == "remint-b1":
                # Keep the MINTED id (what a rebuild will mint); the vector is
                # still addressed by the pack's id, recorded in `vector_key`.
                modes["reminted"] += 1
            else:
                raise PromotionError(
                    f"identity drift: the pack records point_id {declared!r} for "
                    f"{payload.get('url')!r} but this projection mints "
                    f"{point_id!r}. The vector shard is addressed by the pack's "
                    "id, so promoting this row would store a vector computed for "
                    "one row beside another row's payload. See this module's "
                    "docstring: Pool A's ids are not reproducible from their "
                    "rows, so this mode refuses them by design.")
        else:
            modes["full" if full else "point_id"] += 1
        if id_rule == IR.ID_RULE_I1:
            i1, why = IR.mint_i1(payload)
            if i1 is None:
                raise PromotionError(
                    f"{payload.get('url')!r}: the target declares id_rule I1 and this row has no I1 "
                    f"id ({why}); nothing can be written under a guessed id")
            if i1 in out:
                # one record under I1: the survivor is chosen after the loop, by the rebuild's rule
                i1_groups.setdefault(i1, [(vector_key[i1], out[i1])]).append(
                    (str(declared), (dense_text, sparse_text, payload, rec)))
                continue
            point_id = i1
        if point_id in out:
            raise PromotionError(
                f"duplicate point id {point_id} within the pack — two rows "
                "minting one id means the identity scheme collided, which is a "
                "pack defect and not something to de-duplicate here")
        out[point_id] = (dense_text, sparse_text, payload, rec)
        vector_key[point_id] = str(declared)
    if published_from and require_published:
        modes["excluded_undated"] = len(undated)
    if exclude_ids:
        # Kept inside `modes` (a dict, already returned) rather than as a
        # new tuple element -- `project_rows` is unpacked positionally by
        # every existing caller (`plan()` and a dozen tests), and a 6th
        # element there would be a silent breaking change to all of them.
        modes["excluded_out_of_scope_language"] = len(out_of_scope_dropped)
        modes["out_of_scope_dropped_rows"] = out_of_scope_dropped
    for i1, members in i1_groups.items():
        # Review 869 S2: the SAME survivor rule as the aw_e10 rebuild's E3' (`id_rule.choose_survivor`:
        # the canonical URL spelling, else the newest `last_seen`, else the smallest id), so the stored
        # url is the canonical spelling. The rebuild ANDs the members' consent flags; a pack row's
        # flags come from one composer run, and members that disagree refuse the pack (nothing guessed).
        flags = {(m[1][2].get("search_ok"), m[1][2].get("train_ok")) for m in members}
        if len(flags) > 1:
            raise PromotionError(
                f"{len(members)} rows of the pack mint one I1 id {i1} ({members[0][1][2].get('url')!r}) "
                f"with different consent flags {sorted(flags, key=str)}: one record cannot carry both")
        kind = "C" if any(is_card_payload(m[1][2]) for m in members) else "P"
        keep = IR.choose_survivor([(pid, entry[2]) for pid, entry in members], kind)
        for pid, entry in members:
            if pid == keep:
                out[i1], vector_key[i1] = entry, pid
            else:
                i1_merged.append({"point_id": pid, "url": entry[2].get("url"), "i1": i1, "kept": keep})
    if exclude_ids:
        # Task 1011: an exclusion file names the ids a TARGET holds (the minted id, which under an
        # `id_rule` or `remint-b1` is NOT the pack's declared id), so the declared-id test above
        # lets those rows through. Test the id the row would be WRITTEN under as well; the dropped
        # row carries the pack id, because the vector shard is addressed by it.
        for minted in [k for k in out if k in exclude_ids]:
            entry = out.pop(minted)
            out_of_scope_dropped.append({
                "point_id": vector_key.pop(minted), "minted_id": minted,
                "url": entry[2].get("url"),
                "reason": (exclude_ids[minted] if isinstance(exclude_ids, Mapping)
                           else "out_of_scope_language")})
        modes["excluded_out_of_scope_language"] = len(out_of_scope_dropped)
    if id_rule is not None:
        modes["i1_merged"] = len(i1_merged)
        modes["i1_reminted"] = sum(1 for k, v in vector_key.items() if k != v)
    if i1_merged:
        modes["i1_merged_rows"] = i1_merged
    shared = sorted({r["point_id"] for r in projection_rejected} & {vector_key[k] for k in out})
    if shared:
        # task 2242 (compose's task-2222 message): the exclusion is by declared id, so it would
        # drop the accepted row too; refuse here, BEFORE the join check mislabels it.
        raise PromotionError(
            f"refusing to plan: {len(shared)} declared point_id(s) are shared by a record the "
            f"projection accepts and a record it rejects (the declared-id exclusion would silently "
            f"drop the accepted row too); first: {shared[0]}. Nothing written.")
    if projection_rejected:
        # only a pack with a rejected row records this: every other plan keeps its shape
        modes["projection_rejected_rows"] = projection_rejected
    return out, skipped, modes, vector_key, undated


def is_card_payload(payload: Mapping) -> bool:
    from cytir.index.supersede import is_card
    return is_card(payload.get("source_type"), payload.get("entity_kind"), payload.get("qid"))


# ------------------------------------------------------------------ the target
#: Cap on how much of a Qdrant error body we echo into a log line / receipt
#: -- a malformed-filter or bad-vector 400 body is a small JSON object, this
#: is only a guard against something unbounded coming back.
_API_ERROR_BODY_CAP = 4000


#: Retries for a TIMEOUT only (never for an HTTPError -- that is a real
#: answer from the server, not silence). Found live, task 641, 2026-09-23:
#: a PUT to prod timed out at 300s under heavy unrelated load on the build box
#: (measured load average 18.90) and aborted the whole drain, leaving that
#: pack's remaining rows unwritten -- reconciled by hand afterwards. A PUT
#: `points?wait=true` upsert is idempotent (each point carries its own id;
#: retrying it re-applies the same content, never duplicates), so a retry
#: on timeout is safe for every caller of `_api`, not just upserts.
_API_TIMEOUT_RETRIES = 3
_API_TIMEOUT_BACKOFF_S = (5, 15, 30)
#: Retries for an HTTP STATUS that says "try again", never for any other (task 786, 2026-10-08).
#: Tranche 14's prod leg aborted on one `POST .../points -> HTTP 408` (a retrieve of 500 ids: the
#: server or the tunnel gave up waiting, the call itself was small). Before this a 408 or a 5xx was a
#: final answer, so one blip on call 1 of 2,820 stopped a 1.4M-row pack. The set is the transient
#: statuses (408, 429, 500-599); every other 4xx is a real refusal and is never retried. Bounded:
#: `_API_STATUS_RETRIES` retries per call, exponential waits, then the same `PromotionError` as before
#: (the drain's clean, resumable ABORT). A PUT upsert/delete is idempotent, as for the timeout retry.
_API_STATUS_RETRIES = 4
_API_STATUS_BACKOFF_S = (5, 15, 45, 90)
_RETRYABLE_STATUS = frozenset({408, 429}) | frozenset(range(500, 600))

#: The ids one `already_present` request carries. 500 was the value in force on 2026-10-08, so a
#: 1,409,761-id pack is 2,820 requests, not one (the incident's premise "one huge request" is false).
#: A caller may ask for more, never more than the max.
ALREADY_PRESENT_BATCH = 500
ALREADY_PRESENT_BATCH_MAX = 5000


class CallStats:
    """What the prod calls of THIS process cost (task 786): the receipt of a big pack records it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.retries = 0
        self.retried_statuses: dict[str, int] = {}
        self.longest_call_s = 0.0
        self.lookup_chunks = 0
        self.lookup_chunks_resumed = 0

    def snapshot(self) -> dict:
        with self._lock:
            return {"calls": self.calls, "retries": self.retries,
                    "retried_statuses": dict(self.retried_statuses),
                    "longest_call_s": round(self.longest_call_s, 3),
                    "lookup_chunks": self.lookup_chunks,
                    "lookup_chunks_resumed": self.lookup_chunks_resumed}

    def mark(self) -> dict:
        """Start of a leg: reset the longest call and return the snapshot the leg's receipt subtracts."""
        with self._lock:
            self.longest_call_s = 0.0
        return self.snapshot()

    def note_call(self, seconds: float) -> None:
        with self._lock:
            self.calls += 1
            self.longest_call_s = max(self.longest_call_s, seconds)

    def note_retry(self, why: str) -> None:
        with self._lock:
            self.retries += 1
            self.retried_statuses[why] = self.retried_statuses.get(why, 0) + 1

    def note_chunks(self, total: int, resumed: int) -> None:
        with self._lock:
            self.lookup_chunks += total
            self.lookup_chunks_resumed += resumed


#: Calls made to a PROD target only (`PG.check` says which); a twin call is not counted.
PROD_STATS = CallStats()

#: A LEG's whole status-retry allowance (task 787 review). Per call the ladder is 155 s at most, but a leg is
#: thousands of calls: 3 retries on each of 2,820 retrieves and 1,410 PUTs is about 76 h, against a 6 h unit
#: limit, and a kill mid-write leaves a half-landed sealed pack. After this many status retries in one leg the
#: next transient status aborts at once, cleanly. Inside `retry_leg()` only; outside it there is no leg budget.
LEG_STATUS_RETRY_BUDGET = 20
_LEG = {"depth": 0, "left": None}
_LEG_LOCK = threading.Lock()


@contextlib.contextmanager
def retry_leg():
    """One write/lookup leg of the drain: the status-retry budget is shared by every `_api` call inside.
    Re-entrant: a nested leg (a content pack's lookup, then its side) shares the outer budget. Also usable as
    a decorator."""
    with _LEG_LOCK:
        if _LEG["depth"] == 0:
            _LEG["left"] = LEG_STATUS_RETRY_BUDGET
        _LEG["depth"] += 1
    try:
        yield
    finally:
        with _LEG_LOCK:
            _LEG["depth"] -= 1
            if _LEG["depth"] == 0:
                _LEG["left"] = None


def _spend_leg_retry() -> bool:
    """True when the leg may retry once more (always True outside a leg)."""
    with _LEG_LOCK:
        if _LEG["left"] is None:
            return True
        if _LEG["left"] <= 0:
            return False
        _LEG["left"] -= 1
        return True


#: Said in the final abort of a prod call: a 408 is Qdrant's own 5 s header-receive timeout, i.e. the SSH
#: forward accepted the connection and the request bytes arrived late (task 787 probe on Qdrant 1.18.3).
_TUNNEL_HINT = (" -- first suspect: the prod tunnel (aw-prod-tunnel, :6334). Check `ss -tn` Send-Q on the "
                "forward port, then restart aw-prod-tunnel; a retry already opens a new connection.")


def _api(url: str, path: str, body: dict | None = None, method: str | None = None, *,
         allow_prod_scroll: bool = False, reason: str | None = None,
         retry_status: bool = True):
    import socket
    import urllib.error
    req_method = method or ("POST" if body is not None else "GET")
    DRY.refuse_http(req_method, path)
    # incident 2026-10-07: the ONE gate for prod. Refuses a scroll (and, in a replay, any query).
    is_prod = PG.check(url, req_method, path, allow_prod_scroll=allow_prod_scroll, reason=reason)
    attempt = 0
    status_attempt = 0
    while True:
        started = time.monotonic()
        try:
            payload = IH.fetch_json(url.rstrip("/") + path, body=body, method=req_method, timeout=300)
            if is_prod:
                PROD_STATS.note_call(time.monotonic() - started)
            break
        except urllib.error.HTTPError as exc:
            # Bug found live, task 641, 2026-09-23: a bare `HTTPError: HTTP
            # Error 400: Bad Request` with no body told us NOTHING about why
            # -- Qdrant's error responses carry the actual cause (a vector
            # name/dimension mismatch, a bad payload type, a malformed
            # filter, a point id format) in the body, which urllib does not
            # include in the exception's default string form. Read it
            # BEFORE it's gone -- the body stream is only readable once, and
            # only while the exception is live here. Never retried: this IS
            # an answer from the server, not silence.
            raw = exc.read()
            try:
                detail = raw.decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001 -- best-effort only, never masks the real error
                detail = repr(raw)
            detail = detail[:_API_ERROR_BODY_CAP]
            err = PromotionError(
                f"{req_method} {path} -> HTTP {exc.code} {exc.reason}: "
                f"{detail}")
            err.http_status = exc.code  # lets a caller tell a 5xx (transient) from a 4xx
            if is_prod:
                PROD_STATS.note_call(time.monotonic() - started)
            # `retry_status=False`: a fail-closed probe (live_target.read_holdings) must refuse at once,
            # not after 155 s of backoff -- CI timed out on exactly that (PR #718, listing-fails).
            if not retry_status or exc.code not in _RETRYABLE_STATUS:
                raise err from exc
            if status_attempt >= _API_STATUS_RETRIES or not _spend_leg_retry():
                if is_prod:
                    err = PromotionError(str(err) + _TUNNEL_HINT
                                         + ("" if status_attempt >= _API_STATUS_RETRIES else
                                            f" (the leg's {LEG_STATUS_RETRY_BUDGET}-retry budget is spent)"))
                    err.http_status = exc.code
                raise err from exc
            wait_s = _API_STATUS_BACKOFF_S[min(status_attempt, len(_API_STATUS_BACKOFF_S) - 1)]
            print(f"[_api] {req_method} {path}: HTTP {exc.code} (retry "
                  f"{status_attempt + 1}/{_API_STATUS_RETRIES}), retrying in {wait_s}s")
            if is_prod:
                PROD_STATS.note_retry(str(exc.code))
            time.sleep(wait_s)
            status_attempt += 1
            continue
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            # Found live, task 641, 2026-09-23: a PUT to prod timed out at
            # 300s under heavy unrelated load on the build box (measured load
            # average 18.90) and aborted the whole drain, leaving that
            # pack's remaining rows unwritten -- reconciled by hand
            # afterwards. A raw TimeoutError (not always wrapped in
            # URLError by urllib, depending on which read timed out) and a
            # URLError whose CAUSE is a timeout are the only two shapes
            # retried here; a URLError for anything else (connection
            # refused, DNS) is a real failure and must not be retried as if
            # it were transient contention. `points?wait=true` upsert is
            # idempotent (each point carries its own id; retrying re-applies
            # the same content, never duplicates), so this is safe for
            # every caller of `_api`, not just upserts.
            is_timeout = isinstance(exc, (TimeoutError, socket.timeout)) or \
                isinstance(getattr(exc, "reason", None), (TimeoutError, socket.timeout))
            if not is_timeout or attempt >= _API_TIMEOUT_RETRIES:
                raise
            wait_s = _API_TIMEOUT_BACKOFF_S[min(attempt, len(_API_TIMEOUT_BACKOFF_S) - 1)]
            print(f"[_api] {req_method} {path}: timed out (attempt "
                 f"{attempt + 1}/{_API_TIMEOUT_RETRIES}), retrying in "
                 f"{wait_s}s")
            if is_prod:
                PROD_STATS.note_retry("timeout")
            time.sleep(wait_s)
            attempt += 1
    if payload.get("status") not in ("ok", None):
        raise PromotionError(f"{path} returned {payload.get('status')!r}")
    return payload.get("result")


def _ordered_map(fn, items: list, workers: int) -> list:
    """`[fn(x) for x in items]`, with up to `workers` calls in flight.

    Order-preserving, so a caller sees exactly what the sequential loop saw.
    Threads, not processes: every `fn` here is one blocking HTTP round trip
    (`_api`), and the cost of a plan() against the prod box is round-trip
    latency over the build box->serving box tunnel, not CPU (task 689).
    """
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    import concurrent.futures as _cf
    with _cf.ThreadPoolExecutor(max_workers=min(workers, len(items))) as ex:
        return list(ex.map(fn, items))


def already_present(url: str, collection: str, ids: list[str],
                    batch: int | None = None, workers: int = 1, *,
                    checkpoint: "CK.ChunkCheckpoint | None" = None) -> list[str]:
    """Which of `ids` the collection already holds. A batched id retrieve, which
    is indexed and cheap — never a payload scroll (memory
    `prod-serving-box-fragile-scrolls`).

    `batch` is `ALREADY_PRESENT_BATCH` (500) unless asked, and never above
    `ALREADY_PRESENT_BATCH_MAX` (5,000): a pack of N ids is ceil(N/batch) bounded requests, never one.
    Each request retries a transient status (408, 429, 5xx) with backoff, bounded, inside `_api`.

    `checkpoint` (task 786): a chunk the checkpoint already holds is not asked again, and a chunk that
    answers is recorded, so a retry after an abort re-enters at the first unfinished chunk. Only the
    caller that asks a question whose answer cannot change between attempts (the PRE-write presence
    check) passes one; the post-write membership proof never does.

    `workers` > 1 (task 689) issues that many retrieves concurrently. Same
    read-only, indexed, id-keyed calls and the same result, in the same order;
    only the round trips overlap."""
    batch = ALREADY_PRESENT_BATCH if batch is None else batch
    if not 1 <= batch <= ALREADY_PRESENT_BATCH_MAX:
        raise PromotionError(
            f"already_present batch {batch} outside 1..{ALREADY_PRESENT_BATCH_MAX}")
    chunks = [ids[i:i + batch] for i in range(0, len(ids), batch)]
    workers = PG.lookup_workers(url, collection, workers)
    resumed = [0]

    def _one(chunk: list[str]) -> list[str]:
        key = CK.chunk_key(collection, chunk) if checkpoint is not None else None
        if key is not None:
            hit = checkpoint.get(key)
            if hit is not None:
                resumed[0] += 1
                return hit
        got = _api(url, f"/collections/{collection}/points",
                   {"ids": chunk, "with_payload": False, "with_vector": False})
        PG.pace(url, collection)
        found_chunk = [str(r["id"]) for r in (got or [])]
        if key is not None:
            checkpoint.put(key, found_chunk)
        return found_chunk

    found: list[str] = []
    try:
        for part in _ordered_map(_one, chunks, workers):
            found += part
    except BaseException as exc:
        exc.lookup_failed = True    # the lookup itself failed: a checkpoint of it is worth keeping (task 787)
        raise
    if PG.is_prod(url, collection):
        PROD_STATS.note_chunks(len(chunks), resumed[0])
    return found


def retrieve_payloads(url: str, collection: str, ids: list[str], fields: Iterable[str],
                      batch: int = 500, workers: int = 1) -> dict[str, dict]:
    """`{id: {field: value}}` for the ids the collection holds, `fields` only. The same indexed,
    batched, id-keyed retrieve as `already_present` (never a scroll), asking for a short payload
    projection instead of none. An id the collection does not hold is absent from the result,
    so its keys ARE the `already_present` answer."""
    keep = list(fields)
    chunks = [ids[i:i + batch] for i in range(0, len(ids), batch)]
    workers = PG.lookup_workers(url, collection, workers)

    def _one(chunk: list[str]) -> dict[str, dict]:
        got = _api(url, f"/collections/{collection}/points",
                   {"ids": chunk, "with_payload": keep, "with_vector": False})
        PG.pace(url, collection)
        return {str(r["id"]): dict(r.get("payload") or {}) for r in (got or [])}

    found: dict[str, dict] = {}
    for part in _ordered_map(_one, chunks, workers):
        found.update(part)
    return found


def already_served_urls(url: str, collection: str, urls: list[str],
                        batch: int = 256, workers: int = 1, *,
                        allow_prod_scroll: bool = False, reason: str | None = None) -> list[str]:
    """Which of `urls` the collection already serves a row for.

    THE HOLE THIS CLOSES, AND HOW IT WAS FOUND
    ------------------------------------------
    `already_present` asks the right question of the wrong key. It checks POINT
    IDS, and an id encodes the mint scheme that produced it. Task 629 measured
    the consequence on two real W1 packs: `service_public_fiche` and
    `statcan_wds` were both labelled `net_new_addition`, both reported
    `already_present: 0`, and both would have added a second, parallel
    representation of pages the index already serves --

        B02  2,958 of 2,997 urls already served  (8,285 of 8,324 rows, 99.5%)
        B03 10,040 of 10,136 urls already served (10,040 of 10,136 rows, 99.1%)

    -- because their ids come from `aw_common.mint_passage_point_id` while the
    incumbent rows for the same urls were minted by the b1 rebuild. Different
    ids for the same page: the id check passes, `points_count` goes up, and a
    supersession has been recorded as growth (memory `present-is-not-complete`).

    COST. `url` is an indexed keyword on every serving collection, so this is a
    filtered COUNT per batch and a scroll only for the batches that have a hit.
    A genuinely net-new pack therefore pays N/batch counts and no scroll at all
    -- measured 0 scrolls for B05's 5,259 urls -- which is what makes it safe to
    run against prod (memory `prod-serving-box-fragile-scrolls`).

    PROD IS NOT ASKED THIS (incident 2026-10-07). "Is this URL served" has the same answer on the
    twin (twin-mirror rule), and the scroll for a batch with hits is a scan of the serving box:
    a landed pack hits on every batch, and 187k scrolls made search answer 503. Ask the TWIN. A
    scroll against a prod target is refused in `_api` (`cytir.index.prod_guard`) unless the caller
    passes `allow_prod_scroll=True` and a reason; the counts stay allowed, paced and 2-wide.
    """
    workers = PG.lookup_workers(url, collection, workers)
    served: set[str] = set()
    chunks = []
    for i in range(0, len(urls), batch):
        chunk = [u for u in urls[i:i + batch] if u]
        if chunk:
            chunks.append(chunk)

    def _flt(chunk: list[str]) -> dict:
        return {"must": [{"key": "url", "match": {"any": chunk}}]}

    def _count(chunk: list[str]) -> int:
        got = _api(url, f"/collections/{collection}/points/count",
                   {"exact": True, "filter": _flt(chunk)})
        PG.pace(url, collection)
        return int((got or {}).get("count") or 0)

    # The N/batch filtered COUNTs are independent read-only round trips: run
    # them `workers` wide (task 689). The scrolls stay sequential, and only
    # for a batch that counted a hit, exactly as before.
    counts = _ordered_map(_count, chunks, workers)
    for chunk, count in zip(chunks, counts):
        if not count:
            continue
        flt = _flt(chunk)
        offset = None
        while True:
            body = {"limit": 1024, "with_payload": ["url"],
                    "with_vector": False, "filter": flt}
            if offset is not None:
                body["offset"] = offset
            res = _api(url, f"/collections/{collection}/points/scroll", body,
                       **({"allow_prod_scroll": True, "reason": reason}
                          if allow_prod_scroll else {}))
            PG.pace(url, collection)
            for p in (res or {}).get("points", []):
                u = (p.get("payload") or {}).get("url")
                if u:
                    served.add(u)
            offset = (res or {}).get("next_page_offset")
            if offset is None:
                break
    return sorted(served)


def _served_drop_decision(joined: dict, served: list[str]
                          ) -> tuple[frozenset[str], float]:
    """Given `plan()`'s `joined` (`{point_id: (dense, sparse, payload,
    row)}`) and the urls `already_served_urls` found, return
    `(already_served_pids, served_share)`: which of `joined`'s point ids
    carry one of the served urls, and what fraction of the pack's DISTINCT
    urls that is. A pure function (no network, no vectors fixture) so the
    share arithmetic and the drop set are unit-testable at any scale --
    task 664, 2026-09-26: `plan()`'s own real fixture has only 3 urls
    total, too small to exercise a sub-ceiling drop-and-continue case."""
    served_set = set(served)
    already_served_pids = frozenset(
        pid for pid, (_d, _s, payload, _r) in joined.items()
        if (payload.get("url") or "") in served_set)
    pack_url_count = len({(payload.get("url") or "")
                          for _pid, (_d, _s, payload, _r) in joined.items()})
    served_share = len(served) / pack_url_count if pack_url_count else 0.0
    return already_served_pids, served_share


def target_width(url: str, collection: str) -> int:
    info = _api(url, f"/collections/{collection}")
    vectors = info["config"]["params"].get("vectors") or {}
    dense = vectors.get(DENSE_VECTOR) or {}
    size = dense.get("size")
    if not size:
        raise PromotionError(
            f"{collection} declares no `{DENSE_VECTOR}` vector size; refusing "
            "to guess the serving width")
    return int(size)


# ----------------------------------------------------------------------- upsert
def upsert(url: str, collection: str, points: list[dict], batch: int,
          pause_s: float = 0.0, parallel: int = 1, *,
          profile_collection: str | None = None,
          checkpoint: "CK.ChunkCheckpoint | None" = None) -> int:
    """Batched PUT, with an optional pause between batches.

    `pause_s` exists for one reason: a burst of heavy writes degrades the
    30 GB prod box (memory `aw-heavy-rewrite-degrades-prod` -- an
    optimization storm and 60s API timeouts from an unthrottled bulk
    rewrite). Default 0 keeps every existing caller's behaviour unchanged
    (run_batch.py's W1 batches never throttled and are not being changed
    here); the promotion queue driver is the caller that passes a nonzero
    value.

    `profile_collection` (slice 8 step 2): the DECLARED collection whose payload form `collection`
    stores, when `collection` itself is not declared (a stranger's build target stores prod's form,
    `aw_e9`'s). None = `collection`'s own declaration, as before.

    `parallel` (task 641, 2026-09-23 -- SE throughput: sequential single-
    connection PUTs were the dominant cost, measured 0.66ms/point i.e. a
    single-threaded ceiling well under 1.33M rows/8h): >1 fires that many
    batches concurrently via a thread pool -- `_api`'s urllib call is
    blocking I/O, so threads (not processes) already overlap network wait.
    `pause_s` is ignored when `parallel > 1` (concurrency IS the throttle;
    the two don't compose meaningfully). Use this for the TWIN leg only
    (the build box, not RAM-limited); the PROD leg (the 30 GB box) stays
    `parallel=1` with a measured `pause_s`, per the coordinator's ruling.
    """
    # The writer door for the collection's declared payload form (task 675,
    # cytir.index.payload_profile). The WHOLE list is projected before the first PUT,
    # so a value the profile refuses leaves the target untouched.
    # Task 717: a collection that DECLARES the aw_e9 BM25 prune rule (keyed by the PHYSICAL
    # `collection`, not `profile_collection`: the twin and a stranger's target declare none and
    # stay raw) gets every sparse vector cut by the ONE rule and the rule's epoch in rule_epochs.
    # Before the payload projection, so the stamped rule_epochs is the one that gets its code.
    # Task 847 (aw_e10 writer contract): a collection that DECLARES `"id_rule": "I1"` (keyed by the
    # PHYSICAL `collection`, like the BM25 rule) takes a batch only when every point is keyed by the
    # I1 id of its own row. Checked on the pack rows, before the projection drops `tier` and
    # `point_id`, and before the first PUT: the whole batch is refused.
    IR.refuse_unless_i1(collection, points)
    rule = SR.SparseRule.for_collection(collection)
    if rule is not None:
        points = rule.project_points(points)
    points = PP.Projector.for_collection(
        profile_collection or collection).project_points(points)
    if parallel <= 1:
        written = 0
        for i in range(0, len(points), batch):
            chunk = points[i:i + batch]
            # task 786: a batch the checkpoint holds was PUT (and answered) by an earlier attempt of this
            # same transaction; the terminal membership proof still asks prod for every id.
            key = CK.chunk_key(collection, [p["id"] for p in chunk]) if checkpoint is not None else None
            if key is not None and checkpoint.get(key) is not None:
                written += len(chunk)
                continue
            _api(url, f"/collections/{collection}/points?wait=true",
                 {"points": chunk}, method="PUT")
            if key is not None:
                checkpoint.put(key, [])
            written += len(chunk)
            if pause_s and i + batch < len(points):
                time.sleep(pause_s)
    else:
        import concurrent.futures as _cf
        chunks = [points[i:i + batch] for i in range(0, len(points), batch)]

        def _put(chunk):
            _api(url, f"/collections/{collection}/points?wait=true",
                 {"points": chunk}, method="PUT")
            return len(chunk)

        written = 0
        with _cf.ThreadPoolExecutor(max_workers=parallel) as ex:
            for n in ex.map(_put, chunks):
                written += n
    if rule is not None:
        # The guard on what the collection STORES, not on what this process meant to send.
        readback = rule.verify_written(
            lambda ids: _api(url, f"/collections/{collection}/points",
                             {"ids": ids, "with_payload": ["rule_epochs"],
                              "with_vector": [SPARSE_VECTOR]}),
            points)
        # The result is kept (the content-pack receipt carries it) and printed (the journal).
        SR.LAST_READBACK[collection] = readback
        print(f"[sparse-rule] readback {collection}: {readback}", flush=True)
    return written


def payload_profile_preflight(collection: str, payloads: Iterable[Mapping], *,
                              profile_collection: str | None = None) -> dict:
    """Project every payload for `collection`'s declared profile WITHOUT writing.

    Raises `PP.PayloadProfileError` (undeclared collection, wrong tables file,
    a value a closed key has no code for). A drain calls this for BOTH targets
    before the twin leg, so a refusal never leaves twin ahead of prod.
    """
    projector = PP.Projector.for_collection(profile_collection or collection)
    # Task 717: under a declared BM25 prune rule the written rule_epochs carries the rule's
    # epoch, a different enum value. Check THAT one now, before the twin leg: an epoch
    # combination without a code would otherwise refuse at the prod leg, after the twin wrote.
    # (Presence of the rule is read from the declaration only; the term set is not loaded.)
    stamp = SR.declared_epoch(collection)
    rows = 0
    for payload in payloads:
        if stamp is not None:
            payload = SR.stamp_epoch(payload, stamp)
        projector.project(payload)
        rows += 1
    return {"collection": collection, "profile": projector.profile,
            "open_keys": sorted(projector.open_keys), "rows_checked": rows,
            **({"sparse_rule_epoch": stamp} if stamp is not None else {})}


def build_points(joined, matrix, id_index, mrl: int) -> list[dict]:
    points = []
    for point_id, (_dense_text, sparse_text, payload, _rec) in joined.items():
        vec = matrix[id_index[point_id]]
        if vec.shape[0] != mrl:
            raise PromotionError(
                f"{point_id}: truncated vector is {vec.shape[0]}-d, expected {mrl}")
        vectors: dict = {DENSE_VECTOR: [float(x) for x in vec]}
        indices, values = PROJ.saturated_bm25(sparse_text, payload.get("lang"))
        if indices:
            vectors[SPARSE_VECTOR] = {"indices": list(indices),
                                      "values": [float(v) for v in values]}
        points.append({"id": point_id, "vector": vectors, "payload": payload})
    return points


# ---------------------------------------------------------------------- receipt
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _projection_rejected_report(rows: list[dict]) -> dict:
    """Count per reason plus the first 50 rows (task 2224)."""
    by_reason: dict[str, int] = {}
    for row in rows:
        by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1
    return {"total": len(rows), "by_reason": by_reason, "rows": rows[:50]}


def plan(*, chunks_dir: str, vectors_dir: str, qdrant: str, collection: str,
         mrl: int, consent: dict | None = None, attribution: str | None = None,
         license_version: str | None = None,
         allow_supersede: str | None = None,
         require_persisted: str = "l1",
         identity_check: str = "reproduce",
         identity_reason: str | None = None,
         published_from: str | None = None,
         published_status_from: str | None = None,
         published_status_requires: str | None = None,
         require_published: bool = False,
         family_payload: Mapping | None = None,
         title_from: Iterable[str] = (),
         license_from: str | None = None,
         family_license: str | None = None,
         exclude_ids: frozenset | Mapping[str, str] | None = None,
         lookup_workers: int = 1,
         persisted_check: bool = True,
         lookup_checkpoint: "CK.ChunkCheckpoint | None" = None,
         resume_twin_ids: Iterable[str] | None = None) -> dict:
    """Everything up to (but not including) the write. Pure measurement.

    `resume_twin_ids` (task 788): the ids the TWIN leg of this pack wrote and verified, passed ONLY
    by the prod-leg resume (`promote_queue.resume_prod_only`). It replaces the first-time rule "any
    id already present refuses" with `resume_partition`'s: present ids must be twin-written ids (our
    partial write, skipped), anything else refuses as a collision, and the returned `_joined` holds
    exactly the twin-written ids the target lacks. Every other check runs on the WHOLE projection,
    as before. None (default) = every existing caller, unchanged. Never combined with
    `allow_supersede` (a resume never overwrites).

    `lookup_workers` (task 689): concurrency of the read-only already-present /
    already-served round trips against `qdrant`. Default 1 = the sequential
    behaviour every other caller has always had.

    `persisted_check` (slice 8 step 2): the ADR-105 persisted-input check needs the corpus-wide
    sqlite index only our own boxes have. A `Single`-transaction build (a stranger's) has none, and
    passes False: the check is then RECORDED as skipped in the receipt (`persisted_inputs`), never
    as a pass. Default True = every existing caller, unchanged."""
    started = time.time()
    declared = target_width(qdrant, collection)
    if declared != mrl:
        raise PromotionError(
            f"--mrl {mrl} does not match {collection}'s declared dense width "
            f"{declared}; a promotion may not choose a width the target does "
            "not serve")

    matrix, vector_ids = read_vectors(vectors_dir, mrl)
    # task 864: the id the target's declaration names (aw_e10: I1) is the id written, for every
    # identity mode; None for every collection that declares no `id_rule` (unchanged).
    target_id_rule = IR.declared_id_rule(collection)
    joined, skipped, identity_modes, vector_key, undated = project_rows(
        read_records(chunks_dir), consent=consent, attribution=attribution,
        license_version=license_version, identity_check=identity_check,
        identity_reason=identity_reason, published_from=published_from,
        exclude_ids=exclude_ids,
        published_status_from=published_status_from,
        published_status_requires=published_status_requires,
        require_published=require_published, family_payload=family_payload,
        title_from=title_from, license_from=license_from,
        family_license=family_license, id_rule=target_id_rule)

    # The join to the vector shard ALWAYS runs on the pack's id, in every mode.
    # `vector_key` maps what will be written back to what addresses the shard.
    vec_set = set(vector_ids)
    rec_set = {vector_key[pid] for pid in joined}
    # A row EXCLUDED by the date rule leaves its vector with no row, which is a
    # legitimate asymmetry rather than a broken join -- so it is subtracted from
    # the vector side and reported, not counted as a defect. A row excluded by
    # `exclude_ids` (out-of-scope language) is the identical shape: its
    # embedding still exists in the shard, there is just no row to pair it
    # with any more, on purpose.
    vec_set -= {str(row["point_id"]) for row in undated}
    vec_set -= {str(row["point_id"])
               for row in identity_modes.get("out_of_scope_dropped_rows", ())}
    # task 864: a row merged into an earlier row of the pack under ONE I1 id keeps its vector unused
    vec_set -= {str(row["point_id"]) for row in identity_modes.get("i1_merged_rows", ())}
    # task 2224: a row `project_record` rejected has no row to write; its vector is a named terminal
    # (`projection_rejected`), never written. Every OTHER vector-without-row stays a refusal.
    vec_set -= {str(row["point_id"]) for row in identity_modes.get("projection_rejected_rows", ())}
    only_vectors = sorted(vec_set - rec_set)
    only_records = sorted(rec_set - vec_set)
    if only_vectors or only_records:
        raise PromotionError(
            f"vector/text join is not total: {len(only_vectors)} id(s) with a "
            f"vector and no row (e.g. {only_vectors[:3]}), {len(only_records)} "
            f"with a row and no vector (e.g. {only_records[:3]}). A partial "
            "join would store some rows without their embedding and embed "
            "others without their payload; both are silent.")

    shard_index = {pid: i for i, pid in enumerate(vector_ids)}
    # What gets WRITTEN is `joined`'s keys; the presence check must run on those,
    # not on the pack's ids -- under remint-b1 they are different sets, and
    # checking the wrong one is how a re-minted collision would overwrite a live
    # row (the ruling's cost (a), enforced rather than noted).
    id_index = {pid: shard_index[vector_key[pid]] for pid in joined}
    written_ids = sorted(joined)
    if resume_twin_ids is not None and allow_supersede:
        raise PromotionError(
            "resume: a prod-leg resume never overwrites, and this pack declares allow_supersede; "
            "a supersede pack's prod leg is not resumed by this path (reconcile by hand)")
    present = already_present(qdrant, collection, written_ids, workers=lookup_workers,
                              **({} if lookup_checkpoint is None
                                 else {"checkpoint": lookup_checkpoint}))
    resume = None
    if resume_twin_ids is not None:
        # Task 788: a RESUME of the prod leg. The presence question was just asked by id (above);
        # what changes is the rule applied to its answer, never the question.
        resume = resume_partition(written_ids, present, resume_twin_ids, collection=collection)
    elif present and not allow_supersede:
        # `AlreadyLandedError`, not the bare `PromotionError` every other
        # refusal in this function raises -- task 664, 2026-09-26: a caller
        # applying the SAME written_ids to TWO targets (twin, then prod)
        # needs to distinguish "already landed here" from a genuine content
        # refusal, and `ids=frozenset(written_ids)` (not `present`, the
        # already-there subset -- the CALLER needs the full population this
        # plan would have written, to ask the same membership question of a
        # different target) lets it do that without re-deriving the join.
        raise AlreadyLandedError(
            f"{len(present)} of {len(written_ids)} ids are already in "
            f"{collection} "
            f"(e.g. {present[:3]}). This module ADMITS new rows; overwriting an "
            "existing id is a supersession, which is a different operation with "
            "a different gate. Re-run with --allow-supersede '<reason>' if that "
            "is genuinely what is intended, and the reason lands in the receipt.",
            ids=frozenset(written_ids), present=frozenset(present))

    # The SAME question, asked of the key that survives a change of mint scheme.
    # `already_present` above is blind to a page the collection serves under a
    # different id, and task 629 measured two W1 packs where that was 99% of the
    # pack. A `net_new_addition` that lands on a served url is a replacement,
    # which is a different operation with a different gate (ADR-083 supersede +
    # ADR-043's boundary bar), so it refuses here rather than at the apply.
    pack_urls = sorted({(payload.get("url") or "")
                        for _pid, (_d, _s, payload, _r) in joined.items()})
    if PG.is_prod(qdrant, collection):
        # incident 2026-10-07: prod is never asked the URL question. The twin leg's dry plan
        # asked the twin (equal to prod by the twin-mirror rule), applied the share ceiling and
        # produced the drop set; the prod leg writes exactly the twin's written ids.
        served = []
        served_checked_against = "not asked of prod: the twin leg decided (prod_guard)"
    else:
        served = already_served_urls(qdrant, collection, pack_urls,
                                     workers=lookup_workers)
        served_checked_against = f"{collection} at {qdrant} (not prod)"
    already_served_pids: frozenset[str] = frozenset()
    if served and not allow_supersede:
        # Coordinator's ruling, task 664, 2026-09-26 -- the SAME per-URL,
        # share-based principle as the mechanical content gate's word
        # floor (promote_queue.py's WORD_FLOOR_FAIL_SHARE_MAX): a handful
        # of pages this collection already serves under another point id
        # is an ordinary, expected overlap at this scale (real incident:
        # tranche5, 2 of 17,417 urls), not evidence the pack is a
        # duplicate. Those rows are DROPPED from this apply (never
        # silently written as a second parallel representation) and
        # recorded (`already_served_skipped` in the receipt) -- the pack
        # refuses WHOLE only when more than
        # `ALREADY_SERVED_FAIL_SHARE_MAX` of its urls are already served,
        # which IS the duplicate-pack signal. Computed by
        # `_served_drop_decision`, a pure helper kept separate from this
        # function's real-vectors/real-chunks plumbing so the share
        # arithmetic and the drop set are unit-testable at any scale
        # without a matching vectors fixture.
        already_served_pids, served_share = _served_drop_decision(joined, served)
        if served_share > ALREADY_SERVED_FAIL_SHARE_MAX:
            raise PromotionError(
                f"already_served_urls: {len(served)} of {len(pack_urls)} "
                f"({served_share:.2%}) — over the "
                f"{ALREADY_SERVED_FAIL_SHARE_MAX:.0%} ceiling; {collection} "
                "already serves a row for MOST of these pages under other "
                f"point ids (e.g. {served[:3]}). This is a sign the whole "
                "pack is a duplicate, not an isolated overlap -- re-run "
                "with --allow-supersede '<reason>' once the removal of the "
                "incumbent rows and the boundary delta have been ruled.")

    # ADR-105: the page behind every admitted row must have persisted text in a
    # registered store. Checked BEFORE the write, on the row's own url.
    pages = [{"url": (payload.get("url") or ""),
              "sha": payload.get("digest") or payload.get("content_digest")}
             for _pid, (_d, _s, payload, _r) in joined.items()]
    if persisted_check:
        persisted = PI.assert_pages_persisted(pages, require=require_persisted)
    else:
        persisted = {"verdict": "SKIPPED", "pages": len(pages), "require": require_persisted,
                     "reason": "persisted_check=False: no persisted-input index on this target "
                               "(ADR-105 is checked where the index exists; skipped is not a pass)"}

    # Payload carry floors, before anything is written (task 584's lesson: the
    # apply-time gate catches this shape only after the cost is paid).
    carry = PC.assert_scoreable((payload for _pid, (_d, _s, payload, _r)
                                 in joined.items()),
                                label=f"{collection} (sealed-pack promotion)")

    fingerprint = LAYOUT.fingerprint(qdrant, collection)
    projected = len(joined)
    resume_present = resume_dropped = None
    if resume is not None:
        resume_present = resume.pop("_present")
        resume_dropped = resume.pop("_dropped")
        # Every check above ran on the whole projection; what is handed to the write is only the
        # twin-written ids the target lacks.
        keep = resume.pop("_to_write")
        joined = {pid: joined[pid] for pid in keep}
        id_index = {pid: id_index[pid] for pid in keep}
    return {
        "schema": SCHEMA,
        "planned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chunks_dir": chunks_dir,
        "vectors_dir": vectors_dir,
        "target": {"url": qdrant, "collection": collection,
                   "declared_dense_width": declared},
        "rows": {"vectors": len(vector_ids), "projected": projected,
                 "skipped_by_projection": skipped,
                 "excluded_undated": len(undated),
                 "excluded_out_of_scope_language":
                     identity_modes.get("excluded_out_of_scope_language", 0),
                 "already_present": len(present),
                 # Reported even when it is 0 and even when a supersede reason
                 # made it non-fatal: "0 of 5,259" is the statement that makes
                 # a net-new claim checkable, and a count that only appears on
                 # failure teaches nobody anything.
                 "distinct_urls": len(pack_urls),
                 "already_served_urls": (None if PG.is_prod(qdrant, collection) else len(served)),
                 # only a prod target records this, so every other receipt keeps its recorded shape
                 **({"already_served_checked_against": served_checked_against}
                    if PG.is_prod(qdrant, collection) else {})},
        "out_of_scope_language_dropped":
            identity_modes.pop("out_of_scope_dropped_rows", None),
        # task 864: rows of this pack that minted the I1 id of an earlier row (one record under I1);
        # only an I1 target records it, so every other receipt keeps its recorded shape
        **({"i1_merged": identity_modes.pop("i1_merged_rows")}
           if "i1_merged_rows" in identity_modes else {}),
        # task 2224: rows the projection rejected, by reason; absent when there are none
        **({"projection_rejected": _projection_rejected_report(
                identity_modes.pop("projection_rejected_rows"))}
           if "projection_rejected_rows" in identity_modes else {}),
        "family_payload": ({"declared": dict(family_payload),
                            "title_from": list(title_from),
                            "why": ("Pool A rows carry only url and lang; "
                                    "source_type and trust_tier are POLICY "
                                    "facts from the family contract and domain "
                                    "is derived from the url")}
                           if family_payload is not None else None),
        "published_rule": ({
            "path": published_from,
            "status_path": published_status_from,
            "status_requires": published_status_requires,
            "required": require_published,
            "excluded": undated[:50],
            "excluded_total": len(undated),
            "disposition": ("POLICY 4.7: an undated news record is "
                            "inadmissible because it cannot age out. Excluded "
                            "at admission, never backfilled -- the same "
                            "disposition apply_news_dates.py applies post-hoc."),
        } if published_from else None),
        "identity_check": {
            "mode": identity_check,
            "reason": identity_reason,
            **identity_modes,
            "note": ("`full` rows were checked against all six precomputed "
                     "identity fields; `point_id` rows carry only the id, which "
                     "is still the join key, and were checked against that; "
                     "`pack_declared` rows carry an id this module could NOT "
                     "re-derive and were accepted on the pack's receipts, which "
                     "is weaker and is why the count is here; `reminted` rows "
                     "are written under the id aw_b1_identity mints, which is "
                     "the id a rebuild will mint, with the vector still "
                     "addressed by the pack's id via `vector_key`."),
        },
        # Cost (b) of the re-mint ruling: the sealed pack's lineage stays
        # traceable from any served point, so its sha-verified vectors are
        # provably the ones serving.
        "vector_key": {"minted_id_to_pack_id": vector_key,
                       "why": ("the vector shard is addressed by the pack's id "
                               "in every mode; this maps what was written back "
                               "to what carries its embedding")},
        "sealed_dim": SEALED_DIM,
        "mrl": mrl,
        "allow_supersede": allow_supersede,
        # only a resume records this, so every other receipt keeps its recorded shape
        **({"resume": resume} if resume is not None else {}),
        "persisted_inputs": persisted,
        "payload_carry": carry,
        "layout_before": fingerprint,
        "layout_note": ("Product quantization is skipped in appendable segments, "
                        "so these points serve un-quantized until the optimizer "
                        "indexes them; a gate arm cut before `n == points_count` "
                        "measures a different ranker (serving_config.py:61-65, "
                        "ADR-043 A8 rule 2)."),
        "plan_seconds": round(time.time() - started, 3),
        "_joined": joined,
        "_matrix": matrix,
        "_id_index": id_index,
        "_already_served_pids": already_served_pids,
        # only a resume returns these (task 788): the twin-written ids the target already held, and
        # the projected ids the twin leg dropped (never to be on the target)
        **({"_resume_present": resume_present, "_resume_dropped": resume_dropped}
           if resume is not None else {}),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack-chunks", required=True,
                    help="the pack's chunks/ dir (shard-*.texts.jsonl.gz)")
    ap.add_argument("--vectors", required=True,
                    help="the sealed vector dir (*.vectors.f16 + row_ids.jsonl)")
    ap.add_argument("--qdrant", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--mrl", type=int, required=True,
                    help="must equal the target's declared dense width")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--require-persisted", default="l1", choices=("l1", "l1b"))
    ap.add_argument("--allow-supersede", default=None,
                    help="REASON; permits writing ids the target already holds")
    ap.add_argument("--identity-check", default="reproduce",
                    choices=IDENTITY_MODES,
                    help="reproduce (default): the minted id must equal the "
                         "pack's. remint-b1 (ruled for W1): write the id "
                         "aw_b1_identity mints, i.e. the id a rebuild will "
                         "mint. pack-declared: accept the pack's id, with "
                         "--identity-reason. See the module docstring")
    ap.add_argument("--identity-reason", default=None)
    ap.add_argument("--published-from", default=None,
                    help="dotted path to the publication date on the row, e.g. "
                         "source_metadata.publication_time")
    ap.add_argument("--published-status-from", default=None,
                    help="dotted path to the date's status field, e.g. "
                         "source_metadata.publication_time_status")
    ap.add_argument("--published-status-requires",
                    default=DEFAULT_PUBLISHED_STATUS_REQUIRES,
                    help="the status value that counts as the PUBLISHER having "
                         "declared the date (default: %(default)s)")
    ap.add_argument("--family-payload", default=None,
                    help="JSON file (or inline JSON) with the family contract's "
                         "declared source_type and trust_tier; required, because "
                         "Pool A rows carry neither and this module will not "
                         "guess a POLICY trust class")
    ap.add_argument("--title-from", action="append", default=[],
                    help="dotted path(s) to the row's title, tried in order")
    ap.add_argument("--license-from", default=None,
                    help="dotted path to a per-row licence nested off the "
                         "top level (e.g. source_metadata.accepted_answer."
                         "license); never overwrites a top-level `license` "
                         "the row already carries")
    ap.add_argument("--family-license", default=None,
                    help="a licence basis stamped on every row that has "
                         "none (per-row and host-table resolution both take "
                         "priority) -- for a family whose licence.basis is "
                         "a family-level rule, not per-host (e.g. "
                         "ADR-111's opt-out+minimum-record basis for "
                         "cc_authority_frontier_w7); never overwrites a "
                         "`license` a row already carries")
    ap.add_argument("--require-published", action="store_true",
                    help="POLICY 4.7: exclude a row with no publisher-declared "
                         "date instead of admitting it")
    ap.add_argument("--attribution", default=None)
    ap.add_argument("--license-version", default=None)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="without this the run is a DRY RUN and writes nothing")
    ap.add_argument(
        "--record-written-ids", default=None,
        help=("write the point ids this run WOULD write (dry run) or DID write "
              "(--apply) to this path, one JSON list. Needed by ADR-043 am.3's "
              "`rollback-restore` null protocol, which has to remove exactly "
              "what was added and nothing else. Kept OUT of the receipt on "
              "purpose: 222,499 ids is an 8 MB receipt, and a receipt is a thing "
              "people read."))
    args = ap.parse_args(argv)

    family_payload = None
    if args.family_payload:
        raw = args.family_payload
        family_payload = json.loads(
            pathlib.Path(raw).read_text() if os.path.exists(raw) else raw)
    receipt = plan(chunks_dir=args.pack_chunks, vectors_dir=args.vectors,
                   qdrant=args.qdrant, collection=args.collection,
                   mrl=args.mrl, attribution=args.attribution,
                   license_version=args.license_version,
                   allow_supersede=args.allow_supersede,
                   require_persisted=args.require_persisted,
                   identity_check=args.identity_check,
                   identity_reason=args.identity_reason,
                   published_from=args.published_from,
                   published_status_from=args.published_status_from,
                   published_status_requires=args.published_status_requires,
                   require_published=args.require_published,
                   family_payload=family_payload,
                   title_from=tuple(args.title_from),
                   license_from=args.license_from,
                   family_license=args.family_license)
    joined = receipt.pop("_joined")
    matrix = receipt.pop("_matrix")
    id_index = receipt.pop("_id_index")
    already_served_pids = receipt.pop("_already_served_pids", frozenset())
    if already_served_pids:
        joined = {pid: v for pid, v in joined.items()
                 if pid not in already_served_pids}
        receipt["already_served_skipped"] = len(already_served_pids)

    print(f"[plan] {receipt['rows']} — dry run"
          if not args.apply else f"[plan] {receipt['rows']}")
    print(LAYOUT.render(receipt["layout_before"], "target before"))

    points = None
    if args.record_written_ids:
        points = build_points(joined, matrix, id_index, args.mrl)
        ids = [point["id"] for point in points]
        pathlib.Path(args.record_written_ids).write_text(
            json.dumps(ids, indent=1) + "\n")
        receipt["written_ids_path"] = args.record_written_ids
        receipt["written_ids_count"] = len(ids)
        print(f"[ids] {len(ids):,} -> {args.record_written_ids}")

    if args.apply:
        if points is None:
            points = build_points(joined, matrix, id_index, args.mrl)
        t0 = time.time()
        written = upsert(args.qdrant, args.collection, points, args.batch)
        receipt["written"] = written
        receipt["write_seconds"] = round(time.time() - t0, 3)
        receipt["layout_after"] = LAYOUT.fingerprint(args.qdrant, args.collection)
        print(LAYOUT.render(receipt["layout_after"], "target after"))
        print(f"[apply] wrote {written:,} point(s) in "
              f"{receipt['write_seconds']:.1f}s")
    else:
        receipt["written"] = 0
        receipt["dry_run"] = True

    pathlib.Path(args.receipt).write_text(json.dumps(receipt, indent=1) + "\n")
    print(f"[receipt] {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
