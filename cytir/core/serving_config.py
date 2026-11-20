# Moved from `serving_config.py` (task 677, task 643 S5); the old path is an alias shim.
"""The ONE home for the serving-collection shape (dim x quantizer x residency).

Why this module exists
----------------------
Task 412 found the serving configuration scattered across six build scripts that
disagreed with each other and with the live collection, and concluded that
"production sits at the worst point on the entire frontier" partly because nobody
could see the whole config in one place. Task 478 then needed to build the SAME
shape three ways (bake arms) and a fourth time for real (the assembly binder).
Copying `create_collection` a fourth time would have forked the contract, so the
shape moved here and every caller imports it.

Three separate consumers must agree on `dense_dim`, and that agreement is a
MIRROR CONTRACT (engineering-doctrine 6):

  * build side  -- `mrl_trunc(vec, dim)` when a 1024-dim arctic vector is stored
  * query side  -- `cytir/serve/models.py` slices the TEI 1024-dim vector to
                   the same width and L2-renormalizes
  * the collection itself -- `VectorParams(size=dim)`

If the three ever disagree the geometry silently stops matching (or, at best,
Qdrant rejects the query on a dim mismatch). `test_serving_config.py` asserts the
build-side and query-side widths are read from this module, not re-declared.

The MRL contract (arctic-embed-l-v2, ADR-018)
---------------------------------------------
Arctic's Matryoshka property holds for a PREFIX of the full-width vector, and the
prefix must be re-normalized because truncation does not preserve the L2 norm and
the collection distance is Cosine. The canonical implementation is
`pipeline.index_bakeoff.mrl_trunc` -- truncate to the first `d` dims, then
L2-renormalize -- and it is re-exported here so callers get the width and the
transform from the same import.

The vendor card characterizes exactly two widths, 1024 and 256; 128 (what
production ships) has no vendor characterization at all. Task 410/415 measured
full-1024 beating MRL-256 at 10/10 cells, and task 410 measured 128 -> 256 as
worth +12.2 pp (Wikipedia) / +4.5 pp (anchored law) on deep-fact dense h@1.

What "4-bit" means in Qdrant (measured, task 412 SS10.1)
--------------------------------------------------------
The arctic model card's route to 128 bytes/vector is FAISS `pq256x4fs` -- 256 dims
at 4 bits. Qdrant has no fast-scan PQ. Its `product.compression` ratio is defined
against float32, and `get_bucket_size` maps x4->1, x8->2, x16->4, x32->8, x64->16
dims per code byte, so bytes/vector = ceil(dim / bucket). The nearest true
equivalent of the card's target is therefore:

    product quantization at compression "x8", dim 256  ->  exactly 128.0 bytes/vector

That is what `256-pq8` below is, and it is what "the 256/4-bit config" names
everywhere in task 478. It is NOT scalar quantization: Qdrant has no 4-bit scalar
quantizer (`{scalar, product, binary, turbo}`, `types.rs:895`). Recall parity with
the card's FAISS claim is not implied by the byte count and has to be measured --
which is the whole point of the bake.

Two traps this module exists to make unrepeatable (task 412 SS10.1, SS10.7)
---------------------------------------------------------------------------
1. `QuantizationConfig` is `#[serde(untagged)]`. Sending two quantizer keys, or a
   typo'd one, returns HTTP 200 and silently stores only the first match. Every
   config built here is meant to be read back and asserted -- see
   `assert_collection_matches`.
2. Scalar and product quantization are SKIPPED ENTIRELY in appendable (unindexed)
   segments -- `supports_appendable()` is true only for Binary and Turbo. A
   collection that has not been optimized will answer searches with HTTP 200 from
   an UNQUANTIZED full scan. `assert_collection_matches` therefore also checks
   `indexed_vectors_count`, and callers must not measure before green.
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field, replace

import numpy as np
from qdrant_client import models as qm


def mrl_trunc(vecs, d):
    """Matryoshka: truncate each vector to the first d dims and L2-renormalize.

    THE canonical implementation. It used to live in `index_bakeoff.py`, which
    ~10 scripts import it from; that module also pulls in `aw_common` and the
    embedder stack, so anything wanting the truncation had to drag in the whole
    bakeoff rig. It moved here -- beside the width it is applied at -- and
    `index_bakeoff` now re-exports it, so every existing
    `from index_bakeoff import mrl_trunc` keeps working against ONE implementation
    rather than acquiring a second copy (engineering-doctrine 1).

    The renormalize is not optional. Arctic emits unit vectors at the full 1024,
    a PREFIX of a unit vector is not a unit vector, and the collection distance is
    Cosine -- Qdrant normalizes internally for Cosine, but the query and document
    sides must agree about what was stored, and every other consumer of these
    vectors (the rank head, the numpy reference scorer) assumes unit norm.

    `d` <= 0 or >= dim is a no-op and returns the input UNCHANGED, including its
    type. Callers that need an array must not rely on the return being a list.
    """
    if not d:
        return vecs
    a = np.asarray(vecs, dtype="float32")
    if d >= a.shape[1]:
        return vecs
    a = a[:, :d]
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (a / n).tolist()

# ---------------------------------------------------------------------------
# payload indexes
# ---------------------------------------------------------------------------
# config-B payload indexes matching aw_v4_sat (memory aw-qdrant-collection-shape).
# `url` is present from birth so no delta applier ever has to create it after the
# fact (task 348 supersede-on-upsert reconciles page groups via an indexed url
# match).
#
# `published` is the DRIFT this module makes visible rather than inheriting: it is
# indexed on the live collection (10 indexes, 5,205,217 points -- see
# `task 403 2026-08-12/qdrant-collection-twin.json`) but is created by
# no code in the repo. It was written out-of-band during task 354. A fresh
# `create_collection` has never reproduced it, so every rebuild silently drops a
# live index. It is declared here so the rebuild keeps it; ADR-100 / the news lane
# filter on `published` and a full-scan on an unindexed field is exactly the
# `prod-serving-box-fragile-scrolls` failure.
PAYLOAD_INDEXES: dict[str, qm.PayloadSchemaType] = {
    "domain": qm.PayloadSchemaType.KEYWORD,
    "url": qm.PayloadSchemaType.KEYWORD,
    "source_type": qm.PayloadSchemaType.KEYWORD,
    "country": qm.PayloadSchemaType.KEYWORD,
    "tier": qm.PayloadSchemaType.KEYWORD,
    "lang": qm.PayloadSchemaType.KEYWORD,
    "trust_tier": qm.PayloadSchemaType.FLOAT,
    "register_id": qm.PayloadSchemaType.KEYWORD,
    "lei": qm.PayloadSchemaType.KEYWORD,
    "published": qm.PayloadSchemaType.KEYWORD,
}

#: ADR-124 / ruling 64 (task 1391): the REBUILD roster = the served roster + the master-only
#: `family_id` keyword index. Prod and the twin have no such index and no `family_id` values
#: (backfill happens at the rebuild, never in place), so the served roster above stays sealed.
FAMILY_ID_INDEXES: dict[str, qm.PayloadSchemaType] = {
    **PAYLOAD_INDEXES, "family_id": qm.PayloadSchemaType.KEYWORD}

DENSE = "dense"
SPARSE_BM25 = "bm25"

# Qdrant version the whole 478 measurement chain is pinned to. Production runs
# this exact build (`GET /` on the serving box -> 1.18.3 / db8fa43fcb...), and
# task 412's capability probe was read against the same commit's source, so its
# results transfer. the staging-qdrant ensure script used to
# default to v1.12.4, which has neither `turbo` nor binary `encoding` and whose
# snapshots do not restore into >=1.18 at all (ADR-070) -- so a bake or a build
# run on the staging default would have measured a different engine AND would
# have produced an unshippable artifact. Fixed by task 483: that script's
# default now matches this constant and it refuses to run against a
# mismatched image. (Note: the actual staging container was still
# running the old v1.12.4 image as of task 483 -- fixing the default made the
# restore script correctly refuse it rather than silently perpetuating it;
# the container itself still needs a real upgrade before its own restore
# path will pass again.)
QDRANT_IMAGE = os.environ.get("AW_QDRANT_IMAGE", "qdrant/qdrant:v1.18.3")


@dataclass(frozen=True)
class ServingConfig:
    """One point on the (dim x quantizer x residency x rescore) surface."""

    name: str
    dense_dim: int
    # Storage dtype of the ORIGINAL vectors. Only read when rescore is on; task
    # 412 measured 9.4 MB resident of 21 GB with rescore off, i.e. dead weight.
    dense_datatype: str  # "float32" | "float16"
    quantizer: str       # "none" | "int8" | "pq_x8" | "binary_1bit"
    rescore: bool
    oversampling: float
    note: str = ""
    # `always_ram=False` only has any effect when the raw vectors are on_disk
    # (task 412 SS10.5, `quantized_vectors.rs:2214`). Production has on_disk=True,
    # so its always_ram=False is honoured. Keep both together.
    dense_on_disk: bool = True
    quant_always_ram: bool = False
    payload_indexes: dict = field(default_factory=lambda: dict(PAYLOAD_INDEXES))

    # -- Qdrant argument construction -------------------------------------
    def vectors_config(self) -> dict:
        dt = {"float32": qm.Datatype.FLOAT32,
              "float16": qm.Datatype.FLOAT16}[self.dense_datatype]
        return {DENSE: qm.VectorParams(
            size=self.dense_dim, distance=qm.Distance.COSINE,
            on_disk=self.dense_on_disk, datatype=dt)}

    def sparse_vectors_config(self) -> dict:
        # IDF is applied query-side by Qdrant's modifier and is NEVER stored;
        # what IS stored is the ADR-028 saturated weight, which is bounded to
        # (0, 2.2] and therefore safe in float16 (relative error ~1e-3). Plain
        # uint8 stays unusable -- fractional weights collapse to {0,1,2}.
        return {SPARSE_BM25: qm.SparseVectorParams(
            modifier=qm.Modifier.IDF,
            index=qm.SparseIndexParams(on_disk=True,
                                       datatype=qm.Datatype.FLOAT16))}

    def quantization_config(self):
        if self.quantizer == "none":
            return None
        if self.quantizer == "int8":
            return qm.ScalarQuantization(scalar=qm.ScalarQuantizationConfig(
                type=qm.ScalarType.INT8, always_ram=self.quant_always_ram))
        if self.quantizer == "pq_x8":
            return qm.ProductQuantization(product=qm.ProductQuantizationConfig(
                compression=qm.CompressionRatio.X8,
                always_ram=self.quant_always_ram))
        if self.quantizer == "binary_1bit":
            return qm.BinaryQuantization(binary=qm.BinaryQuantizationConfig(
                encoding=qm.BinaryQuantizationEncoding.ONE_BIT,
                always_ram=self.quant_always_ram))
        raise ValueError(f"unknown quantizer {self.quantizer!r}")

    def search_quantization_params(self):
        """The query-side half. Binary and PQ are UNUSABLE without rescore.

        Task 412 measured binary-without-rescore collapsing -29.0 pp at d128 and
        -20.9 pp at d256, and measured that rescore alone is not enough -- it only
        re-ranks what the beam already found, so oversampling >= 3 is what
        actually recovers recall (binary d128: 48.97 at x1 -> 51.10 at x3).

        Rescore is NOT a per-collection stored setting; it is a query parameter.
        That is why it lives on this object beside the quantizer instead of being
        hardcoded in `retrieval.py` (where task 412 found it as an unreachable
        constant with no env knob).
        """
        return qm.QuantizationSearchParams(
            rescore=self.rescore, oversampling=self.oversampling)

    @property
    def appendable_is_quantized(self) -> bool:
        """Does an UN-indexed (appendable) segment still search correctly?

        This is the property that decides how much un-indexed tail is tolerable,
        and it is a property of the QUANTIZER, not a number someone picks.

        `QuantizationConfig::supports_appendable()` (`types.rs:912`) is true only
        for Binary and Turbo. So:

          * `int8` / `pq_x8` -- quantization is SKIPPED in appendable segments.
            Points sitting there are searched by an UNQUANTIZED full scan, which
            is task 412 SS10.7's trap: the measurement silently stops being a
            measurement of the quantizer. Tolerance must be tight.
          * `binary_1bit` (and turbo) -- an appendable segment carries correct
            binary codes. Nothing is wrong with those points.
          * `none` -- there is no quantizer to skip. The residual cannot affect
            quantization fidelity at all, because there is none to affect.

        For the lenient cases the residual still means those points are found by
        exact scan rather than through the HNSW graph. That biases recall
        slightly UPWARD (exact beats an approximate beam), never downward, so it
        cannot manufacture a win for a losing arm -- and at the fractions
        observed here (0.07-0.56%) it is far below the resolution of either
        instrument.
        """
        return self.quantizer in ("none", "binary_1bit", "turbo")

    def indexed_tolerance(self) -> float:
        return (INDEXED_TOLERANCE_LENIENT if self.appendable_is_quantized
                else INDEXED_TOLERANCE_STRICT)

    def code_bytes_per_vector(self) -> int:
        """MEASURED bytes/vector of the quantized codes (task 412 SS10.2).

        Not nominal arithmetic: binary codes are padded to a 16-byte boundary
        with a 16 B/vec floor, and scalar int8 carries +4 B/vec of metadata.
        """
        d = self.dense_dim
        if self.quantizer == "none":
            return 0
        if self.quantizer == "int8":
            return d + 4
        if self.quantizer == "pq_x8":
            return -(-d // 2)          # 2 dims per code byte
        if self.quantizer == "binary_1bit":
            return max(16, -(-(-(-d // 8)) // 16) * 16)  # ceil to 16 B, floor 16
        raise ValueError(self.quantizer)


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------
# Every arm task 478's bake measures, plus the shipped baseline. Adding a row here
# is how a new candidate enters the bake -- there is no other list.
CONFIGS: dict[str, ServingConfig] = {
    # --- the shipped baseline. Must reproduce production EXACTLY, so it keeps
    # f32 originals and rescore off. ADR-023 is vindicated for int8 (task 412
    # SS11.4 at n=3,000): int8/d128 rescore-off scores 52.30 against an
    # unquantized ceiling of 51.93, i.e. indistinguishable from no quantization.
    "prod-128-int8": ServingConfig(
        name="prod-128-int8", dense_dim=128, dense_datatype="float32",
        quantizer="int8", rescore=False, oversampling=1.0,
        note="the shipped config (aw_e3 / aw_v4_sat). Baseline, not a candidate."),

    # --- the widen-only arm: same quantizer, twice the dims. Isolates the
    # dimension effect from the quantizer effect.
    "256-int8": ServingConfig(
        name="256-int8", dense_dim=256, dense_datatype="float16",
        quantizer="int8", rescore=False, oversampling=1.0,
        note="dimension doubled, quantizer unchanged. 260 B/vec."),

    # --- THE TARGET. "256/4-bit" = product x8 at dim 256 = exactly 128.0 B/vec,
    # fewer bytes than production pays today.
    "256-pq8": ServingConfig(
        name="256-pq8", dense_dim=256, dense_datatype="float16",
        quantizer="pq_x8", rescore=True, oversampling=1.0,
        note="THE FLIP TARGET. The model card's 4-bit route, in Qdrant terms."),

    # --- task 412's co-leader, at a QUARTER of the target's code bytes. 412
    # explicitly deferred the tie-break to "build both on the build-box twin at
    # full scale and run the gate", which is this bake. Carried because it is
    # free to add and it is the arm the ADR-011/071 design actually names.
    "256-bin": ServingConfig(
        name="256-bin", dense_dim=256, dense_datatype="float16",
        quantizer="binary_1bit", rescore=True, oversampling=3.0,
        note="ADR-011/071's named design. 32 B/vec. Most scale-sensitive arm."),

    # --- the control. Unquantized at the candidate width: bounds every 256 arm
    # from above and turns "is the quantizer costing us anything" into a
    # measurement instead of an argument.
    "256-none": ServingConfig(
        name="256-none", dense_dim=256, dense_datatype="float16",
        quantizer="none", rescore=False, oversampling=1.0,
        note="CONTROL: the unquantized 256 ceiling. Not shippable (1024 B/vec)."),

    # --- the ceiling. Task 412's single best cell: 67.50 deep-fact h@1, exactly
    # the unquantized full-width ceiling, at 128 B/vec. Rejected there on DISK
    # (rescore must read 2,048 B/pt of originals -> ~552 GB at 100M), not on
    # quality. Carried as the upper bound the target is being judged against.
    "1024-bin": ServingConfig(
        name="1024-bin", dense_dim=1024, dense_datatype="float16",
        quantizer="binary_1bit", rescore=True, oversampling=3.0,
        note="CEILING: best measured cell in task 412; rejected on disk, not quality."),
}

#: The REBUILD shape (ADR-124, ruling 64, task 1391): `256-pq8` + the `family_id` index. Kept out of
#: CONFIGS on purpose: CONFIGS is the roster the sealed creator goldens enumerate, and the four old
#: creators keep the served shape. The aw_e11 rebuild names this constant.
FAMILY_ID_CONFIG: ServingConfig = replace(
    CONFIGS["256-pq8"], name="256-pq8-family", payload_indexes=dict(FAMILY_ID_INDEXES),
    note="the rebuild (aw_e11) shape: 256-pq8 + family_id index; ADR-124.")

# There is deliberately NO default config name. It was "prod-128-int8", the retired
# shape, and a caller that named nothing silently built it (slice 8 step 1). A caller
# names a config, or sets AW_SERVING_CONFIG, or this refuses.
def get(name: str | None = None) -> ServingConfig:
    name = name or os.environ.get("AW_SERVING_CONFIG")
    if not name:
        raise ValueError("no serving config named: pass one of "
                         f"{sorted(CONFIGS)} (or set CYTIR_SERVING_CONFIG); there is no default")
    if name not in CONFIGS:
        raise KeyError(f"unknown serving config {name!r}; "
                       f"known: {sorted(CONFIGS)}")
    return CONFIGS[name]


# ---------------------------------------------------------------------------
# the flip's compose-facing env (task 522)
# ---------------------------------------------------------------------------
# `ServingConfig` above is the QDRANT-side shape. `cytir/serve/retrieval.py`
# reads a SEPARATE set of env vars to query that shape (`AW_DENSE_RESCORE`,
# `AW_DENSE_OVERSAMPLING`, `AW_DENSE_IGNORE_QUANT`) plus `AW_MRL_DIM`
# (`cytir/serve/models.py`) to slice the query embedding to the same
# width. Task 519 found these missing from `deploy/docker-compose.yaml`
# entirely -- not wrong, ABSENT -- so `prod_env_run.py` faithfully ran every
# gate arm at rescore OFF against a config the ADR-011 amendment had already
# named "256-pq8, rescore=True, oversampling=1.0 (3.0 until task 2831)". The compose file is not
# wrong today because it serves the DIFFERENT, still-live `prod-128-int8`
# config (int8 keeps rescore off by design -- `test_int8_keeps_rescore_off`
# above); the gap was that the FLIP's config had never been written down
# anywhere the compose file could read it.
#
# `flip_deploy_env()` is that one place. `cytir/ops/gen_flip_compose_env.py`
# generates/validates `deploy/docker-compose.flip.yaml` (the flip's overlay,
# applied on top of `docker-compose.yaml` on flip day only) from it, and
# `test_flip_compose_env.py` fails if the tracked overlay ever
# disagrees with what this function returns.
FLIP_SERVING_CONFIG = "256-pq8"

# THE ONE HOME for "the weights the flip ships with" (task 552). Every other
# site that needs this fact imports or is verified against these two
# constants: `deploy/docker-compose.flip.yaml` is GENERATED from them
# (`cytir/ops/gen_flip_compose_env.py`), `data/flip-gate-contract.v1.json`
# carries only a pointer to them (`legs.full-stack.candidate_weights_source`,
# same pattern as its `latency.source` pointer into `deploy/aw-monitor.env`
# rather than a second copy of the number), and `cytir/gates/flip_gate.py`'s
# `assert_sealed_pass` imports this module directly to refuse a verdict whose
# candidate full-stack arm served a different sha256. Never hand-edit a copy
# of this sha256 anywhere else -- that is exactly the fork task 552 closed.
#
# History: task 519 (ADR-099 amendment, PR #91) promoted `candidate519`
# (`trust` 3.0, `academic` -1.0, `machine` 0.0) as the flip's weights. Task
# 533's later re-tune on the flip's actual post-delta corpus produced
# `candidate533` (one key different: `academic` -1.0 -> -2.0), which strictly
# dominates candidate519 on both the 629-query chooser and the held-out
# 1,418-query gate (task 533 doc §5) and was the file every task-538 gate arm
# actually measured (`task 538 run_gate_538.sh` exports
# `AW_BUILD_WEIGHTS` at it). This constant did not follow that promotion --
# task 551 found the resulting harnesses defaulting to the now-stale
# candidate519 and refused rather than guess; task 552 is the fix: point the
# ONE home at what task 538's gate actually measured. Promoting either
# candidate into a flip that actually ships remains Tim's call, by PR
# (memory: legislative-prs) -- this is a pointer correction, not a new
# promotion, and task 533 §8.6 still recommends HOLD on the flip itself for
# reasons unrelated to which weights file it would carry.
FLIP_FUSION_WEIGHTS_FILE = "fusion_weights.candidate533.json"
FLIP_FUSION_WEIGHTS_CONTAINER_PATH = f"/app/cytir/serve/{FLIP_FUSION_WEIGHTS_FILE}"
FLIP_FUSION_WEIGHTS_SHA256 = (
    "2f1d7a654424b863d433000a35ae9a87b8979c5cddf250638a07fbdee8d3c16e")


def flip_deploy_env(name: str = FLIP_SERVING_CONFIG) -> dict[str, str]:
    """The env the flip's `aw-api` service must carry for
    serving config `name`, as compose-file string values.

    Includes the fusion-weights promotion alongside the dense knobs because
    they ship in the SAME release (task 522): a 256-dim collection scored
    with the wrong fusion weights is a different but equally silent
    mis-measurement of the same kind task 519 found for the dense policy.
    """
    cfg = get(name)
    return {
        "AW_MRL_DIM": str(cfg.dense_dim),
        "AW_DENSE_RESCORE": "1" if cfg.rescore else "0",
        "AW_DENSE_OVERSAMPLING": str(cfg.oversampling),
        # The flip queries the quantizer it was built with; `ignore=True` is
        # a diagnostic arm (task 517), never a deploy default.
        "AW_DENSE_IGNORE_QUANT": "0",
        "AW_FUSION_WEIGHTS": FLIP_FUSION_WEIGHTS_CONTAINER_PATH,
    }


# ---------------------------------------------------------------------------
# the same env, for a GATE arm rather than a container (task 604)
# ---------------------------------------------------------------------------
# `flip_deploy_env()` is written for compose: its `AW_FUSION_WEIGHTS` is a path
# INSIDE the container. A gate arm runs from a release directory on the build box,
# so it needs the same policy with the weights resolved against that release's
# own `cytir/serve/`. Everything else is identical, and deliberately so --
# the whole point is that there is one home for "what production serves".
#
# Why this exists at all: task 603 found the ADR-043 content gate scoring the
# dense leg with rescore OFF while production serves it ON x3, and task 604
# found the same gap one level worse -- `retrieval.py:_weights_path()` falls
# back to `fusion_weights.json` when `AW_FUSION_WEIGHTS` is unset, and no gate
# driver sets it, so every post-flip content-pack arm scored `trust 2.5 /
# academic -0.6 / machine -1.0` while production served candidate533's `3.0 /
# -2.0 / 0.0`. Measured read-only on the live twin (task 604 sec.2): 82 golds
# in a different position and h@1 213 -> 219. Pre-flip the fallbacks happened
# to equal what production ran, which is why nothing ever caught it -- the
# same shape as task 600's `AW_MRL_DIM` finding, and the same fix: take the
# number from the system being judged.
#
# `AW_MRL_DIM` is deliberately NOT in here. The query width is a property of
# the TARGET COLLECTION (task 600: `gate_query_width` reads it from the
# collection, because Qdrant rejects every other width outright), not of the
# serving profile, and a gate may legitimately measure a collection that is
# not the one production serves. It is recorded in the arm, not bound here.
def gate_dense_env(server_dir, name: str = FLIP_SERVING_CONFIG) -> dict[str, str]:
    """The dense/fusion env a gate arm must carry to measure what prod serves.

    `server_dir` is the release's `cytir/serve` directory -- the one
    `retrieval.py` resolves its own default against.
    """
    env = flip_deploy_env(name)
    return {
        "AW_DENSE_RESCORE": env["AW_DENSE_RESCORE"],
        "AW_DENSE_OVERSAMPLING": env["AW_DENSE_OVERSAMPLING"],
        "AW_DENSE_IGNORE_QUANT": env["AW_DENSE_IGNORE_QUANT"],
        "AW_FUSION_WEIGHTS": str(pathlib.Path(server_dir) /
                                 FLIP_FUSION_WEIGHTS_FILE),
    }


# ---------------------------------------------------------------------------
# CANDIDATE ranking knobs -- declared here, armed nowhere (task 606)
# ---------------------------------------------------------------------------
# `flip_deploy_env()` above is what production RUNS. This table is what a
# ranking candidate would add to it, and it is deliberately not merged into
# either that function or `gate_dense_env()`: a knob that changes ranking
# reaches production through a deploy PR and the flip-gate battery, never
# through a default. It lives here rather than in a task script so that "what
# is proposed, at what value, and what it still has to clear" is data with one
# home instead of a sentence in a changelog (engineering-doctrine 2).
#
# `bar` is not decoration. ADR-043 scores a ranking change on the per-gold
# zero-loss bar over the gold core; an entry may be armed only after an arm
# pair on the twin reports that bar met, and the experiment id is what connects
# the arming to the measurement that justified it.
CANDIDATE_RANKING_KNOBS: dict[str, dict] = {
    "AW_BM25_ZERO_GUARD": {
        "value": "1",
        "default_in_code": "0",
        "what": "drop non-positive sparse-leg scores before min-max. A filtered "
                "sparse query against a PLAIN (unindexed) segment returns "
                "filter-matching points that do not contain the term, scored "
                "0.0; admitted to the normalisation they collapse `lo` and "
                "re-scale the whole BM25 leg.",
        "lineage": "task 336 guarded the per-token sub-legs and the stage-2 "
                   "coverage oracle after a 90-point upsert on aw_b1 broke 4 "
                   "of 663 gate golds. The MAIN legs were never guarded.",
        "exposure": "38 of 1,394 BM25 legs on the 1,418-query battery come "
                    "back short of their candidate window, so they are the "
                    "legs a plain segment can pad (task 606, measured on "
                    "aw_e8). Qdrant keeps a sub-threshold segment plain while "
                    "still reporting green/optimizer-ok, so `wait_green` in "
                    "the content-pack applier does not wait it away.",
        "measured": "task 606: inert on today's aw_e8 -- 0 golds moved, core "
                    "h@1 219 unchanged -- because the collection currently "
                    "carries no plain segment. The value is insurance against "
                    "the state a live write creates, not a lift.",
        "bar": "ADR-043 ranking bar (per-gold zero-loss on the gold core) on a "
               "collection that DOES carry a fresh plain segment, plus the "
               "flip-gate battery, before arming.",
        "experiment": "E606-6",
    },
}


def candidate_ranking_env(names: "list[str] | None" = None) -> dict[str, str]:
    """The env for an arm that measures one or more candidate ranking knobs.

    Never merged into `flip_deploy_env()`: an arm asks for these by name, so a
    knob cannot reach a deploy by being in the same dictionary as the dense
    policy.
    """
    wanted = list(CANDIDATE_RANKING_KNOBS) if names is None else list(names)
    unknown = [n for n in wanted if n not in CANDIDATE_RANKING_KNOBS]
    if unknown:
        raise KeyError(f"unknown candidate ranking knob(s): {unknown}")
    return {n: CANDIDATE_RANKING_KNOBS[n]["value"] for n in wanted}


#: Schema of the block every arm records about the configuration it RAN under.
#: Read back from the loaded `retrieval` module, not from the environment it
#: was asked to run with: an arm must be able to prove its own dense policy,
#: which is exactly what no sealed arm in this repo could do before task 604.
GATE_BINDING_SCHEMA = "aw.gate-serving-binding.v1"


def gate_serving_binding_problems(binding: object,
                                  name: str = FLIP_SERVING_CONFIG) -> list[str]:
    """Human-readable reasons this arm did not measure what production serves.

    Empty list = the arm ran production's dense policy and production's fusion
    weights. Fail-closed: a missing or malformed block is a problem, never a
    pass, because "no evidence" and "evidence of agreement" are the two things
    task 600 confused when an arm of 1,418 errors was written and handed on.
    """
    if not isinstance(binding, dict):
        return ["the arm records no serving binding -- it cannot prove which "
                "dense policy or fusion weights it ran (task 604)"]
    problems: list[str] = []
    if binding.get("schema") != GATE_BINDING_SCHEMA:
        problems.append(f"serving binding schema {binding.get('schema')!r} is "
                        f"not {GATE_BINDING_SCHEMA}")
    cfg = get(name)
    want = {"dense_rescore": cfg.rescore,
            "dense_oversampling": float(cfg.oversampling),
            "dense_ignore_quant": False}
    for key, value in want.items():
        got = binding.get(key)
        if key == "dense_oversampling":
            got = None if got is None else float(got)
        if got != value:
            problems.append(
                f"{key}={got!r}, production serves {value!r} "
                f"(cytir.core.serving_config.flip_deploy_env, {name})")
    if binding.get("fusion_weights_sha256") != FLIP_FUSION_WEIGHTS_SHA256:
        problems.append(
            f"fusion weights sha256 {binding.get('fusion_weights_sha256')} != "
            f"the promoted {FLIP_FUSION_WEIGHTS_SHA256} "
            f"({FLIP_FUSION_WEIGHTS_FILE}) -- this arm scored a ranker "
            f"production does not run")
    return problems


# ---------------------------------------------------------------------------
# the flip's qdrant-side attach mount (task 596, re-pointed by task 597,
# retired by task 618)
# ---------------------------------------------------------------------------
# Task 596 used this pointer to attach the flip CANDIDATE (`aw_e8`) into the
# serving qdrant ahead of the flip, because the serving box's local disk had
# no room for a second 135 GB collection. Task 597 (2026-09-17) re-pointed it
# at the UNSERVED rollback collection (`aw_e3`) once `aw_e8` -- by then the
# SERVED collection -- was relocated onto local disk, where a served
# collection belongs (random 4 KB reads: local p50 0.49 ms / p95 1.08 ms vs
# the network volume's p50 1.32 ms / p95 2.58 ms; novel-query first-hit was
# 4-9 s cold on the volume).
#
# Task 618 (2026-09-19, ADR-098 am.1 addendum, Tim ruled 2026-09-17 "4. ok")
# retired `aw_e3` from the serving box entirely: after a 2-day bake with no
# rollback exercised, the collection was deleted from `deploy-qdrant-1` and
# its volume copy removed (preserved copies re-verified first -- the build box
# twin, the build box snapshot, off-box dump; see
# `data/production-flip-ruling-aw_e8.v1.json`'s `rollback.amendment` and
# `task 618`). Nothing the serving box runs today
# needs an off-local bind mount: `aw_e8` (served) lives in
# `docker-compose.yaml`'s own local `aw-qdrant` volume, and there is no
# second collection to attach. `flip_qdrant_volumes()` therefore returns no
# mounts.
#
# The MECHANISM is not removed, only its current use: a future flip that
# again needs to attach an off-local collection (a rebuild too large for
# local disk, a new rollback copy on the scratch volume) re-points this the
# same way task 596 -> 597 -> 618 did, by PR (legislative-prs) -- name the
# collection and its host dir, return the one `"host:container"` entry.
# Compose merges a service's `volumes` list by CONCATENATION across `-f`
# files (verified, task 596), so a re-added entry is additive to
# `docker-compose.yaml`'s own qdrant volumes, never a replacement of them.
#
# SAFETY: never bind a collection's path into a qdrant container while any
# OTHER qdrant process still has it mounted -- two processes reading/writing
# the same collection directory concurrently corrupts it.
#
# Task 668 (2026-09-28) tried re-pointing this a fourth time, for `aw_e9`,
# then REVERTED it the same day once the real numbers came in: `/` has 89 GB
# free (measured), `aw_e8`'s local copy is 149 GB and `aw_e9`'s optimized
# footprint is 113 GB -- serving `aw_e9` off this attach mount would mean
# serving it from `/mnt/promote-scratch`, exactly the cold-latency
# antipattern this mechanism's own history (task 597) exists to avoid
# (network volume p50 1.32ms/p95 2.58ms vs local p50 0.49ms/p95 1.08ms, 4-9s
# cold novel-query latency). `aw_e9` reaches local disk by RELOCATION
# instead (`deploy/scripts/flip-aw_e9-methodA.sh` rev 2: `aw_e8` backed up to
# the volume, then deleted from local, `aw_e9` copied in from its offline
# restore) -- the same shape as task 597's own aw_e8/aw_e3 swap, not an
# attach. `flip_qdrant_volumes()` is back to carrying only the P18 mount.
def flip_qdrant_volumes() -> list[str]:
    """Extra bind mount(s) the flip's `qdrant` service needs, as
    compose-file `service.volumes` entries (`"host:container"` strings) --
    appended to `docker-compose.yaml`'s own qdrant volumes by
    `cytir/ops/gen_flip_compose_env.py`, never replacing them.

    As of task 668 (2026-09-28) this is back to empty (see the module
    comment above for why the aw_e9 attempt was reverted the same day):
    `aw_e8` (served) lives in `docker-compose.yaml`'s own local `aw-qdrant`
    volume, and the flip reaches `aw_e9` by relocating it onto that same
    local volume rather than attaching it off-local.

    P18 (task 637 section 16) adds one entry that is NOT about attaching a
    collection: the snapshot staging directory. It is a bind mount rather
    than a named volume because it has to land on `/mnt/promote-scratch`,
    the only volume on the serving box with room for a 113 GB tar."""
    return [flip_qdrant_temp_mount()]


# ---------------------------------------------------------------------------
# P18 -- where qdrant STAGES a snapshot (task 637 section 16)
# ---------------------------------------------------------------------------
#: Qdrant builds a snapshot tar under `<storage>/tmp/` and only then copies it
#: to the snapshots directory. So snapshotting a 113 GB collection needs ~113 GB
#: free ON THE STORAGE VOLUME, which prod's 301 GB local disk (106 GB free) does
#: not have -- and neither did the twin's volume, where a 140.7 GB staging file
#: took `/media/disk2` to 65 GB free and aborted task 637's own build.
#:
#: `docker-compose.yaml`'s qdrant comment has named this pattern since task 337
#: ("re-point snapshots + QDRANT__STORAGE__TEMP_PATH at it"). It was never
#: configured. This is that comment turned into a tracked setting.
#:
#: VERIFIED BY BEHAVIOUR, not by a config echo: with the env var set, qdrant
#: creates `<temp_path>/tmp/` and NEVER creates `<storage>/tmp/`; without it,
#: `<storage>/tmp/` appears. `GET /telemetry` does NOT report temp_path at any
#: details level, so a monitor has to check the directory qdrant actually made
#: (`cyt ops check-qdrant-temp-path`).
QDRANT_TEMP_PATH_CONTAINER = "/qdrant/snapshot-tmp"

#: The host directory bound at QDRANT_TEMP_PATH_CONTAINER on the SERVING box.
#: `/mnt/promote-scratch` is the 738 GB Hetzner Cloud Volume (450 GB free);
#: task 597 forbids serving a collection off it, which this is not -- a snapshot
#: tar is one big sequential write, the workload a network volume is good at.
QDRANT_TEMP_PATH_HOST_PROD = "/mnt/promote-scratch/qdrant-tmp"


def flip_qdrant_environment() -> dict:
    """Env the flip's `qdrant` service needs, rendered into the tracked overlay.

    Kept here rather than in the compose file for the same reason
    `flip_qdrant_volumes()` is: a knob that only exists in a hand-edited compose
    file is a knob the next `up -d` silently drops (task 596's finding, and the
    reason `test_flip_compose_env.py` exists)."""
    return {"QDRANT__STORAGE__TEMP_PATH": QDRANT_TEMP_PATH_CONTAINER}


def flip_qdrant_temp_mount() -> str:
    """The prod bind mount that backs `flip_qdrant_environment()`'s temp path."""
    return "%s:%s" % (QDRANT_TEMP_PATH_HOST_PROD, QDRANT_TEMP_PATH_CONTAINER)


# ---------------------------------------------------------------------------
# collection creation + the read-back assertion
# ---------------------------------------------------------------------------
def create_collection(client, name: str, cfg: ServingConfig,
                      bulk: bool = True, with_payload_indexes: bool = True,
                      log=print) -> None:
    """Create a serving-shaped collection.

    `bulk=True` creates at `hnsw m=0` -- no graph is built during the load, which
    is the ADR-022 / `qdrant-bulk-build-dim-change` bulk convention. Call
    `finalize_collection` afterwards to PATCH m=16 and build the graph once.

    NOTE the ordering constraint this signature encodes: quantization can only be
    chosen at creation. Task 412 SS10.6 read the ConfigMismatchOptimizer and found
    that PATCHing `quantization_config` on a live collection rebuilds EVERY
    segment whose stored quantization differs -- where "differs" is plain `!=`, so
    even a change to `always_ram` alone forces a full rebuild with new segment ids
    and a rebuilt HNSW. This vindicates ADR-023's standing rule: bake the
    quantizer at creation, never PATCH it.
    """
    from cytir.core import serving_profile as SP
    profile = SP.profile_from_serving_config(cfg, with_payload_indexes=with_payload_indexes)
    SP.create_collection(client, name, profile, on_exists="refuse", index_failure="raise",
                         verify=True, bulk=bulk, log=log)


def finalize_collection(client, name: str, log=print) -> None:
    log(f"[finalize] {name}: PATCH hnsw m=16")
    client.update_collection(name, hnsw_config=qm.HnswConfigDiff(m=16))


# How much of a collection may legitimately sit in an un-indexed appendable tail.
#
# MEASURED, task 478, at 1,824,204 points -- the residual is real, permanent, and
# quantizer-dependent:
#
#     binary one_bit   green at 1,823,000 indexed   residual 1,204   (0.066%)
#     none             green at 1,814,000 indexed   residual 10,204  (0.56%)
#     int8, product    green at 1,824,204 indexed   residual 0       (exact)
#
# The pattern is not noise, it is the mechanism. `supports_appendable()` is TRUE
# for Binary and Turbo, so for those the optimizer has no reason to convert a
# small appendable segment -- it is already correctly quantized. For scalar and
# product it is FORCED to convert, because quantization is skipped in appendable
# segments, which is why those two reach exact equality. And with no quantizer
# there is nothing to force at all, so `none` drifts furthest.
#
# This is the MIRROR IMAGE of task 412 SS10.7's trap. That trap says "a green
# collection serving 200s is not evidence the index exists", and the check for it
# is `indexed == points`. Written for the trap alone, that check demands something
# that for binary and for `none` can never happen: it cost this task four hours on
# a wait that was never going to finish, and then failed a perfectly good
# unquantized control arm.
#
# So the bar is chosen by `ServingConfig.appendable_is_quantized`, i.e. by whether
# an un-indexed point is actually searched WRONG -- not by one magic number.
INDEXED_TOLERANCE_STRICT = float(os.environ.get("AW_INDEXED_TOL_STRICT", "0.005"))
INDEXED_TOLERANCE_LENIENT = float(os.environ.get("AW_INDEXED_TOL_LENIENT", "0.05"))


# The collection-shape check, split into a normaliser + one pure rule set, so
# that a caller reading the collection over raw HTTP gets the SAME rules as one
# holding a qdrant client. Task 529 measured the cost of not having that split:
# `cytir/gates/promote_verify.py` -- the tool `task 508`
# names for the promote verification -- carried its own copy of this check,
# frozen at the RETIRED 128-int8 "config B" (`dense["size"] == 128 and ... quant
# == "int8"`). Against the ratified 256-pq8 candidate that copy FAILS by
# construction, and it read the quantizer only out of
# `quantization_config.scalar`, so a product-quantized collection read back as
# `None` whatever it actually held. Engineering doctrine 1: a copy is a forked
# contract, not parity.


def shape_from_info(info) -> dict:
    """Normalise a qdrant-client `get_collection` model to the shape dict."""
    params = info.config.params
    dense = (params.vectors[DENSE] if isinstance(params.vectors, dict)
             else params.vectors)
    qc = info.config.quantization_config
    kind = None
    if qc is not None:
        for k in ("scalar", "product", "binary", "turbo"):
            if getattr(qc, k, None) is not None:
                kind = k
                break
    return {"dense_size": dense.size,
            "dense_datatype": getattr(dense, "datatype", None),
            "quantizer_kind": kind,
            "points": info.points_count or 0,
            "indexed": info.indexed_vectors_count or 0,
            "status": str(info.status)}


def shape_from_json(info: dict) -> dict:
    """Normalise the raw `GET /collections/<name>` body — no client needed."""
    params = (info.get("config") or {}).get("params") or {}
    vectors = params.get("vectors") or {}
    dense = (vectors.get(DENSE)
             if isinstance(vectors, dict) and DENSE in vectors else vectors)
    dense = dense if isinstance(dense, dict) else {}
    qc = (info.get("config") or {}).get("quantization_config") or {}
    kind = None
    for k in ("scalar", "product", "binary", "turbo"):
        if qc.get(k) is not None:
            kind = k
            break
    return {"dense_size": dense.get("size"),
            "dense_datatype": dense.get("datatype"),
            "quantizer_kind": kind,
            "points": info.get("points_count") or 0,
            "indexed": info.get("indexed_vectors_count") or 0,
            "status": str(info.get("status"))}


def collection_shape_problems(shape: dict, cfg: ServingConfig,
                              check_indexed: bool = True) -> list[str]:
    """Every way this collection is not what `cfg` asked for.

    This is the countermeasure to task 412's untagged-serde trap: a doubled or
    typo'd quantizer key returns HTTP 200 and stores something else. It is also
    the countermeasure to the appendable-segment trap -- scalar and product
    quantization do not apply until the optimizer has run, and an unoptimized
    collection answers searches with 200 from an unquantized scan. A green
    collection serving 200s is not evidence that the index exists.
    """
    problems: list[str] = []
    if shape["dense_size"] != cfg.dense_dim:
        problems.append(f"dense size {shape['dense_size']} != {cfg.dense_dim}")

    got_dt = shape["dense_datatype"]
    want_dt = {"float32": "float32", "float16": "float16"}[cfg.dense_datatype]
    # float32 is the default and may read back as None rather than the literal.
    if not (got_dt is None and want_dt == "float32") and str(
            got_dt or "").lower().replace("datatype.", "") != want_dt:
        problems.append(f"dense datatype {got_dt!r} != {want_dt!r}")

    want_kind = {"none": None, "int8": "scalar",
                 "pq_x8": "product", "binary_1bit": "binary"}[cfg.quantizer]
    if shape["quantizer_kind"] != want_kind:
        problems.append(
            f"quantizer read back as {shape['quantizer_kind']!r}, asked for "
            f"{want_kind!r} (untagged-serde trap: check for a doubled key)")

    if check_indexed:
        pts, idx = shape["points"], shape["indexed"]
        residual = max(pts - idx, 0)
        tol = cfg.indexed_tolerance()
        if pts and residual > pts * tol:
            problems.append(
                f"indexed_vectors_count {idx:,} < points_count {pts:,} by "
                f"{residual:,} ({residual/pts:.2%} > {tol:.2%} for quantizer "
                f"{cfg.quantizer!r}): quantization is SKIPPED in appendable "
                f"segments for scalar and product, so any measurement taken now "
                f"may be partly of an UNQUANTIZED scan")
        if str(shape["status"]).lower().endswith("green") is False:
            problems.append(f"status {shape['status']!r} is not green")

    return problems


def assert_collection_matches(client, name: str, cfg: ServingConfig,
                              check_indexed: bool = True) -> dict:
    """Read the collection back and refuse if it is not what we asked for."""
    shape = shape_from_info(client.get_collection(name))
    problems = collection_shape_problems(shape, cfg, check_indexed)
    if problems:
        raise RuntimeError(f"collection {name} does not match config "
                           f"{cfg.name}: " + "; ".join(problems))
    return {"points": shape["points"], "indexed": shape["indexed"],
            "unindexed_residual": max(shape["points"] - shape["indexed"], 0),
            "status": shape["status"]}


_QUANTIZER_KIND_TO_CFG = {None: "none", "scalar": "int8",
                          "product": "pq_x8", "binary": "binary_1bit"}


def assert_dim_matches_shape(shape: dict, name: str, expected_dim: int,
                             source: str) -> None:
    """Task 563 -- the guard every pack applier runs before it writes a
    single vector: the embedding width the pack itself declares
    (`expected_dim`, read from the pack's own collection or its embedding
    manifest/`--mrl`, per `source`) must equal the width `name` already
    holds. Takes an already-normalised `shape` dict (`shape_from_info` or
    `shape_from_json` -- either caller's own way of reaching the target,
    client or raw HTTP) so this one function serves both.

    `cyt index pack-delta` (task 322) and `cyt index surface-delta` (task
    546) each grew their own scroll/upsert loop with no dimension check of
    any kind -- task 560 found a 128-dim pack silently aimed at the 256-dim
    `aw_e8` and only caught it by hand. This is the ONE guard both import,
    rather than each growing its own ad hoc comparison (doctrine 1: one
    truth, one home).

    It is deliberately NOT a second comparator. It mirrors `shape`'s own
    CURRENT datatype and quantizer, unchanged, into a synthetic
    `ServingConfig` whose only chosen field is `dense_dim` -- so
    `collection_shape_problems`, the exact comparator `assert_collection_
    matches` runs, can never refuse for a datatype or quantizer reason, and
    dimension is the only axis left that can disagree.
    """
    dt = str(shape["dense_datatype"] or "").lower().replace("datatype.", "")
    dense_datatype = "float16" if dt == "float16" else "float32"
    kind = shape["quantizer_kind"]
    if kind not in _QUANTIZER_KIND_TO_CFG:
        raise RuntimeError(
            f"collection {name} holds quantizer kind {kind!r}, which "
            f"assert_dim_matches has no ServingConfig equivalent for and "
            f"cannot mirror to isolate the dimension check")
    mirror = ServingConfig(
        name=f"<{source} declares dim={expected_dim}>",
        dense_dim=expected_dim, dense_datatype=dense_datatype,
        quantizer=_QUANTIZER_KIND_TO_CFG[kind], rescore=False,
        oversampling=1.0)
    problems = collection_shape_problems(shape, mirror, check_indexed=False)
    if problems:
        raise RuntimeError(
            f"embedding-dimension mismatch: {source} declares dim="
            f"{expected_dim}, but collection {name} holds dense_size="
            f"{shape['dense_size']}; refusing to write (" +
            "; ".join(problems) + ")")


def assert_dim_matches(client, name: str, expected_dim: int, source: str) -> dict:
    """`assert_dim_matches_shape`, for a caller that already holds a
    `qdrant_client.QdrantClient` (e.g. `cyt index pack-delta`). A caller that
    only has the raw HTTP surface (e.g. `cyt index surface-delta`, which
    is sealed-tested against a fake `cytir.core.qdrant_http.api` and must never
    construct a second client stack) calls `assert_dim_matches_shape`
    directly with a `shape_from_json(...)` result instead.
    """
    shape = shape_from_info(client.get_collection(name))
    assert_dim_matches_shape(shape, name, expected_dim, source)
    return {"points": shape["points"], "indexed": shape["indexed"],
            "unindexed_residual": max(shape["points"] - shape["indexed"], 0),
            "status": shape["status"]}
