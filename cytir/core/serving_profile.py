"""The ONE declared serving profile and the ONE collection creator (refactor slice 8, step 1).

Before this module there were four ways to create a Qdrant collection in `cytir/`
(`serving_config.create_collection`, `collection_stage.create_collection`,
`distribute.apply.create_collection`, `bm25_saturate.ensure_dst`), each with its own
defaults: `cyt apply` made fp32 with no quantizer, no on-disk flags and no payload
indexes. Now there is one object, `ServingProfile`, that declares everything Qdrant
stores at creation time (vector size/dtype, quantizer, on-disk flags, HNSW, sparse
index, optimizer, payload indexes), and one function, `create_collection`, that turns
it into a collection. The other four are thin shims that choose a profile and a
policy (what to do when the collection exists, what to do when an index fails).

Three named profiles describe live things, each proven equal to a SEALED read of the
real collection (`data/serving-profiles/*.config.v1.json`, a read-only
`GET /collections/<name>`; the serving-profile drift test compares the profile,
and a collection the creator really made on a Qdrant service, to the fixture field by
field):

  prod-e9   what prod serves (aw_e9, 637's shape: m=8, uint8 sparse, 8 payload indexes,
            on-disk indexes, tuned optimizer).
  twin-e8   what the twin holds (aw_e8: m=16, float16 sparse, 10 indexes).
  served    the layout `cyt embed` and the live staging lanes create today. It equals
            twin-e8 except for the twin's runtime-tuned optimizer thread cap. It is NOT
            prod-e9: the two differ on five axes (see the test). Nobody changed that here.

`ServingProfile.manifest()` is one JSON block; its digest goes into the dump manifest.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, fields, replace
from typing import Optional

from qdrant_client import models as qm

from cytir.core.digest import canonical_utf8_sha256

DENSE = "dense"
SPARSE_BM25 = "bm25"

# Qdrant 1.18.3 server defaults for every field a profile may leave unset. They exist
# so `ServingProfile.expected_shape()` can say what a collection created from a profile
# WILL read back as; the CI service proves they still hold on the pinned image.
SERVER_DEFAULTS = {
    "shard_number": 1, "replication_factor": 1, "write_consistency_factor": 1,
    "hnsw_m": 16, "ef_construct": 100, "full_scan_threshold": 10000,
    "max_indexing_threads": 0, "default_segment_number": 0, "max_segment_size": None,
    "indexing_threshold": 10000, "max_optimization_threads": None,
    "memmap_threshold": None, "on_disk_payload": True,
}

_DATATYPES = {"float32": qm.Datatype.FLOAT32, "float16": qm.Datatype.FLOAT16,
              "uint8": qm.Datatype.UINT8}
_DISTANCES = {"cosine": qm.Distance.COSINE, "dot": qm.Distance.DOT,
              "euclid": qm.Distance.EUCLID,
              "manhattan": qm.Distance.MANHATTAN}


#: The version of the profile block a dump manifest carries. A reader accepts a block
#: of this version or older (a field the older block lacks takes its default) and REFUSES
#: a newer block, or a block with a field this code does not know, naming both versions.
#: Bump it when a field is added that an older reader must not silently drop.
PROFILE_SCHEMA_VERSION = 1
_BLOCK_META = ("schema", "profile_schema_version")


def canonical_digest(stored: dict) -> str:
    """sha256 of the canonical JSON of a profile declaration AS STORED. The manifest is
    the authority, not the dataclass of the code that reads it: a field a later version
    adds must not invalidate a block an earlier version wrote."""
    return canonical_utf8_sha256(stored)


class ProfileBlockRefusal(ValueError):
    """A manifest profile block that this code will not create a collection from."""


class ShapeMismatch(RuntimeError):
    """A collection read back from the server is not the shape its profile declares."""


class CollectionExistsError(RuntimeError):
    """`create_collection(on_exists="refuse")` found the name taken."""


@dataclass(frozen=True)
class ServingProfile:
    """Everything a collection is created with. `None` = "not passed: server default"."""

    name: str
    dense_dim: int
    dense_datatype: str = "float16"       # what the vectors ARE (read-back is checked)
    datatype_explicit: bool = True        # False: leave `datatype` out of the request
    dense_on_disk: Optional[bool] = True
    distance: str = "cosine"
    quantizer: str = "pq_x8"              # none | int8 | pq_x8 | binary_1bit
    quant_always_ram: bool = False
    sparse_name: Optional[str] = SPARSE_BM25   # None = no sparse vector at all
    sparse_idf: bool = True
    sparse_index_on_disk: Optional[bool] = True
    sparse_index_datatype: Optional[str] = "float16"
    hnsw_m: Optional[int] = 16            # m at creation; None = no hnsw_config sent
    hnsw_on_disk: Optional[bool] = True
    hnsw_max_indexing_threads: Optional[int] = None
    on_disk_payload: Optional[bool] = True
    # optimizer knobs sent at creation: a sorted tuple of (key, value)
    optimizers: tuple = ()
    # payload indexes: (field, "keyword"|"float", on_disk). on_disk False = the plain
    # enum schema the older creators send; True = the typed params object.
    payload_indexes: tuple = ()
    note: str = ""
    # The token cap the collection's dense vectors are DECLARED to be embedded at (the
    # `embedder.max_seq_length` of the packs that feed it). None = not declared: a pack's
    # cap is recorded, never checked (task 2538). Left out of `manifest()` while None, so
    # every digest written before this field existed stays valid.
    embedder_max_seq: Optional[int] = None

    # Duck-types `serving_config.ServingConfig` for the read-back check
    # (`dense_dim`, `dense_datatype`, `quantizer`, `name`). `indexed_tolerance` is left
    # out on purpose: the creator reads back with check_indexed=False.

    # --- the manifest block ------------------------------------------------------
    def manifest(self) -> dict:
        d = asdict(self)
        if d.get("embedder_max_seq") is None:
            d.pop("embedder_max_seq", None)
        d["optimizers"] = [list(kv) for kv in self.optimizers]
        d["payload_indexes"] = [list(i) for i in self.payload_indexes]
        d["schema"] = "aw.serving-profile.v1"
        d["profile_schema_version"] = PROFILE_SCHEMA_VERSION
        return d

    def digest(self) -> str:
        return canonical_digest(self.manifest())

    def block(self) -> dict:
        """What a dump manifest carries: name, digest, schema version, the declaration."""
        return {"name": self.name, "digest": self.digest(),
                "profile_schema_version": PROFILE_SCHEMA_VERSION,
                "profile": self.manifest()}

    @classmethod
    def from_block(cls, block: dict) -> "ServingProfile":
        """Rebuild from a manifest block. The digest is checked over the block AS STORED
        (not over this class), so a block an older version wrote still verifies after a
        field is added here. Refuses (`ProfileBlockRefusal`, a ValueError):

          * a digest that does not match the stored declaration (edited block)
          * a block of a NEWER schema version, or one carrying a field this code does not
            know (named, with both versions); a field the block lacks takes its default
          * a block that names a built-in profile (`prod-e9`, ...) but differs from it:
            the digest alone is an internal checksum, only a signature makes a manifest
            tamper-evident, and this makes the NAME mean something too."""
        if not isinstance(block, dict) or not isinstance(block.get("profile"), dict):
            raise ProfileBlockRefusal("no `profile` declaration in the block")
        stored = block["profile"]
        got = canonical_digest(stored)
        if got != block.get("digest"):
            raise ProfileBlockRefusal(
                f"serving profile {block.get('name')!r}: digest {block.get('digest')!r} "
                f"does not match its declaration ({got}): the manifest was edited")
        ver = stored.get("profile_schema_version")
        if ver != block.get("profile_schema_version") or not isinstance(ver, int):
            raise ProfileBlockRefusal(
                f"serving profile {block.get('name')!r}: schema version "
                f"{block.get('profile_schema_version')!r} (block) vs {ver!r} (declaration)")
        if ver > PROFILE_SCHEMA_VERSION:
            raise ProfileBlockRefusal(
                f"serving profile {block.get('name')!r} is schema v{ver}; this code reads "
                f"up to v{PROFILE_SCHEMA_VERSION}. Upgrade `aw` to apply this dump.")
        raw = {k: v for k, v in stored.items() if k not in _BLOCK_META}
        unknown = sorted(set(raw) - {f.name for f in fields(cls)})
        if unknown:
            raise ProfileBlockRefusal(
                f"serving profile {block.get('name')!r} (schema v{ver}) carries field(s) "
                f"{unknown} that this code (reads up to v{PROFILE_SCHEMA_VERSION}) does "
                f"not know: it would drop them silently. Upgrade `aw` to apply this dump.")
        raw["optimizers"] = tuple(tuple(kv) for kv in raw.get("optimizers", ()))
        raw["payload_indexes"] = tuple(tuple(i) for i in raw.get("payload_indexes", ()))
        prof = cls(**raw)
        if block.get("name") != prof.name:
            raise ProfileBlockRefusal(
                f"serving profile block is named {block.get('name')!r} but its "
                f"declaration is named {prof.name!r}")
        builtin = builtin_profile(prof.name)
        if builtin is not None:
            want = builtin.manifest()
            differs = sorted(k for k in raw if k in want and
                             json.loads(json.dumps(stored[k])) != want[k])
            if differs:
                raise ProfileBlockRefusal(
                    f"serving profile {prof.name!r} is a built-in profile, but the block "
                    f"differs from it on {differs}: a different layout must carry a "
                    f"different name")
        return prof

    def with_(self, **kw) -> "ServingProfile":
        return replace(self, **kw)

    # --- Qdrant request construction --------------------------------------------
    def vectors_config(self) -> dict:
        kw = dict(size=self.dense_dim, distance=_DISTANCES[self.distance.lower()])
        if self.dense_on_disk is not None:
            kw["on_disk"] = self.dense_on_disk
        if self.datatype_explicit:
            kw["datatype"] = _DATATYPES[self.dense_datatype]
        return {DENSE: qm.VectorParams(**kw)}

    def sparse_vectors_config(self) -> Optional[dict]:
        if self.sparse_name is None:
            return None
        index = None
        if self.sparse_index_on_disk is not None or self.sparse_index_datatype is not None:
            ikw = {}
            if self.sparse_index_on_disk is not None:
                ikw["on_disk"] = self.sparse_index_on_disk
            if self.sparse_index_datatype is not None:
                ikw["datatype"] = _DATATYPES[self.sparse_index_datatype]
            index = qm.SparseIndexParams(**ikw)
        return {self.sparse_name: qm.SparseVectorParams(
            modifier=qm.Modifier.IDF if self.sparse_idf else None, index=index)}

    def quantization_config(self):
        if self.quantizer == "none":
            return None
        if self.quantizer == "int8":
            return qm.ScalarQuantization(scalar=qm.ScalarQuantizationConfig(
                type=qm.ScalarType.INT8, always_ram=self.quant_always_ram))
        if self.quantizer == "pq_x8":
            return qm.ProductQuantization(product=qm.ProductQuantizationConfig(
                compression=qm.CompressionRatio.X8, always_ram=self.quant_always_ram))
        if self.quantizer == "binary_1bit":
            return qm.BinaryQuantization(binary=qm.BinaryQuantizationConfig(
                encoding=qm.BinaryQuantizationEncoding.ONE_BIT,
                always_ram=self.quant_always_ram))
        raise ValueError(f"unknown quantizer {self.quantizer!r}")

    def hnsw_config(self, m: Optional[int]):
        if m is None:
            return None
        kw = {"m": m}
        if self.hnsw_on_disk is not None:
            kw["on_disk"] = self.hnsw_on_disk
        if self.hnsw_max_indexing_threads is not None:
            kw["max_indexing_threads"] = self.hnsw_max_indexing_threads
        return qm.HnswConfigDiff(**kw)

    def index_schema(self, ftype: str, on_disk: bool):
        if not on_disk:
            return {"keyword": qm.PayloadSchemaType.KEYWORD,
                    "float": qm.PayloadSchemaType.FLOAT}[ftype]
        if ftype == "keyword":
            return qm.KeywordIndexParams(type="keyword", on_disk=True)
        if ftype == "float":
            return qm.FloatIndexParams(type="float", on_disk=True)
        raise ValueError(f"unknown payload index type {ftype!r}")

    # --- what a collection made from this profile reads back as ------------------
    def expected_shape(self) -> dict:
        d = SERVER_DEFAULTS
        opt = dict(self.optimizers)
        quant = None
        if self.quantizer == "int8":
            quant = {"kind": "scalar", "type": "int8", "always_ram": self.quant_always_ram}
        elif self.quantizer == "pq_x8":
            quant = {"kind": "product", "compression": "x8",
                     "always_ram": self.quant_always_ram}
        elif self.quantizer == "binary_1bit":
            quant = {"kind": "binary", "encoding": "one_bit",
                     "always_ram": self.quant_always_ram}
        sparse = {}
        if self.sparse_name is not None:
            sparse[self.sparse_name] = {
                "index_on_disk": bool(self.sparse_index_on_disk),
                "index_datatype": self.sparse_index_datatype or "float32",
                "modifier": "idf" if self.sparse_idf else None}
        return {
            "vectors": {DENSE: {"size": self.dense_dim, "distance": self.distance.capitalize(),
                                "on_disk": bool(self.dense_on_disk),
                                "datatype": self.dense_datatype}},
            "sparse": sparse,
            "shard_number": d["shard_number"],
            "replication_factor": d["replication_factor"],
            "write_consistency_factor": d["write_consistency_factor"],
            "on_disk_payload": (d["on_disk_payload"] if self.on_disk_payload is None
                                else bool(self.on_disk_payload)),
            "hnsw": {"m": d["hnsw_m"] if self.hnsw_m is None else self.hnsw_m,
                     "ef_construct": d["ef_construct"],
                     "full_scan_threshold": d["full_scan_threshold"],
                     "max_indexing_threads": (d["max_indexing_threads"]
                                              if self.hnsw_max_indexing_threads is None
                                              else self.hnsw_max_indexing_threads),
                     "on_disk": bool(self.hnsw_on_disk)},
            "optimizer": {k: opt.get(k, d[k]) for k in (
                "default_segment_number", "max_segment_size", "indexing_threshold",
                "max_optimization_threads", "memmap_threshold")},
            "quantization": quant,
            "payload_indexes": {f: {"type": t, "on_disk": bool(od)}
                                for f, t, od in self.payload_indexes},
        }


def shape_of(result: dict) -> dict:
    """Normalise the `result` of a raw `GET /collections/<name>` into the shape dict
    `ServingProfile.expected_shape()` predicts. Counts, status, WAL and flush settings
    are not creation parameters and are left out. Absent == the server default."""
    cfg = result["config"]
    params = cfg["params"]
    vec = params["vectors"]
    vec = vec if DENSE in vec else {DENSE: vec}
    q = cfg.get("quantization_config")
    quant = None
    if q:
        kind, body = next(iter(q.items()))
        quant = {"kind": kind, **{k: v for k, v in body.items()
                                   if k in ("type", "compression", "encoding", "always_ram")}}
        quant.setdefault("always_ram", False)
    sparse = {}
    for n, s in (params.get("sparse_vectors") or {}).items():
        idx = s.get("index") or {}
        sparse[n] = {"index_on_disk": bool(idx.get("on_disk")),
                     "index_datatype": idx.get("datatype") or "float32",
                     "modifier": s.get("modifier")}
    h, o = cfg["hnsw_config"], cfg["optimizer_config"]
    return {
        "vectors": {n: {"size": v["size"], "distance": v["distance"],
                        "on_disk": bool(v.get("on_disk")),
                        "datatype": v.get("datatype") or "float32"} for n, v in vec.items()},
        "sparse": sparse,
        "shard_number": params["shard_number"],
        "replication_factor": params["replication_factor"],
        "write_consistency_factor": params["write_consistency_factor"],
        "on_disk_payload": bool(params.get("on_disk_payload")),
        "hnsw": {"m": h["m"], "ef_construct": h["ef_construct"],
                 "full_scan_threshold": h["full_scan_threshold"],
                 "max_indexing_threads": h["max_indexing_threads"],
                 "on_disk": bool(h.get("on_disk"))},
        "optimizer": {k: o.get(k) for k in (
            "default_segment_number", "max_segment_size", "indexing_threshold",
            "max_optimization_threads", "memmap_threshold")},
        "quantization": quant,
        "payload_indexes": {f: {"type": s["data_type"],
                                "on_disk": bool((s.get("params") or {}).get("on_disk"))}
                            for f, s in (result.get("payload_schema") or {}).items()},
    }


# ---------------------------------------------------------------------------------
# the named profiles
# ---------------------------------------------------------------------------------
# The roster the code creates today (`ServingConfig.payload_indexes`, task 354 + 403).
_SERVED_INDEXES = (
    ("domain", "keyword", False), ("url", "keyword", False),
    ("source_type", "keyword", False), ("country", "keyword", False),
    ("tier", "keyword", False), ("lang", "keyword", False),
    ("trust_tier", "float", False), ("register_id", "keyword", False),
    ("lei", "keyword", False), ("published", "keyword", False))

SERVED = ServingProfile(
    name="served", dense_dim=256, dense_datatype="float16", quantizer="pq_x8",
    payload_indexes=_SERVED_INDEXES,
    note="pre-e9 arm shape; selectable by name; no default anywhere")

#: ADR-124 / ruling 64 (task 1391): the rebuild's profile = SERVED + the master-only `family_id`
#: keyword index, nothing else. SERVED itself stays sealed (prod has no such index; the backfill
#: is at the rebuild, not in place). Created by `serving_config.FAMILY_ID_CONFIG`.
SERVED_FAMILY_ID = SERVED.with_(
    name="served-family-id",
    payload_indexes=_SERVED_INDEXES + (("family_id", "keyword", False),),
    note="served + family_id index (ADR-124); the aw_e11 rebuild shape")

TWIN_E8 = SERVED.with_(
    name="twin-e8", optimizers=(("max_optimization_threads", 3),),
    note="the twin aw_e8: `served` plus a runtime-tuned optimizer thread cap")

PROD_E9 = ServingProfile(
    name="prod-e9", dense_dim=256, dense_datatype="float16", quantizer="pq_x8",
    sparse_index_datatype="uint8", hnsw_m=8, hnsw_max_indexing_threads=8,
    optimizers=(("default_segment_number", 30), ("indexing_threshold", 10000),
                ("max_optimization_threads", 4), ("max_segment_size", 3900000)),
    payload_indexes=(
        ("url", "keyword", True), ("domain", "keyword", True), ("lang", "keyword", True),
        ("country", "keyword", True), ("source_type", "keyword", True),
        ("trust_tier", "float", True), ("published", "keyword", True),
        ("register_id", "keyword", True)),
    note="what prod serves (aw_e9, task 637 shape); sealed in data/serving-profiles/")

#: The `source_cards` collection layout (ADR-021 / ADR-017): 128-wide int8 dense + bm25 with the card
#: payload indexes. Same request as the old script's own `create_collection` except that hnsw
#: `ef_construct` is not sent (the server default is the 100 the script spelled out).
SOURCE_CARDS = ServingProfile(
    name="source-cards", dense_dim=128, dense_datatype="float32", datatype_explicit=False,
    quantizer="int8", sparse_index_on_disk=None, sparse_index_datatype=None,
    payload_indexes=(("disc_class", "keyword", False), ("jurisdiction", "keyword", False),
                     ("disposition", "keyword", False), ("domain", "keyword", False),
                     ("trust_tier", "float", False)),
    note="the out-of-pool source-authority card collection (ADR-021 / ADR-017)")

#: Named, public, resolvable by `cyt dump --profile` and `cyt apply`.
PROFILES = {p.name: p for p in (SERVED, SERVED_FAMILY_ID, TWIN_E8, PROD_E9)}


def get(name: str) -> ServingProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown serving profile {name!r}; known: {sorted(PROFILES)}") from None


def builtin_profile(name: str) -> Optional["ServingProfile"]:
    """The built-in profile a NAME stands for, or None for a name that is not one."""
    if name in PROFILES:
        return PROFILES[name]
    m = re.fullmatch(r"pack-staging-(\d+)", name)
    return pack_staging(int(m.group(1))) if m else None


def pack_staging(dense_dim: int) -> ServingProfile:
    """`stage_pack_b1`'s scroll-consumed staging layout (the width is the caller's)."""
    return ServingProfile(
        name=f"pack-staging-{int(dense_dim)}", dense_dim=int(dense_dim),
        dense_datatype="float32", datatype_explicit=False, quantizer="int8",
        sparse_index_on_disk=None, sparse_index_datatype=None, hnsw_m=0,
        payload_indexes=(("lang", "keyword", False), ("tier", "keyword", False),
                         ("domain", "keyword", False), ("provenance", "keyword", False),
                         ("trust_tier", "float", False)),
        note="stage_pack_b1's scroll-consumed staging layout")


def legacy_apply(dims: int, distance: str, *, sparse_name: Optional[str],
                 sparse_idf: bool) -> ServingProfile:
    """What `cyt apply` created before dump manifests carried a profile: fp32, no
    quantizer, no on-disk flags, no payload indexes. Kept so an OLD manifest still
    applies exactly as it did."""
    return ServingProfile(
        name="legacy-apply", dense_dim=int(dims), dense_datatype="float32",
        datatype_explicit=False, dense_on_disk=None, distance=distance, quantizer="none",
        sparse_name=sparse_name, sparse_idf=sparse_idf, sparse_index_on_disk=None,
        sparse_index_datatype=None, hnsw_m=None, hnsw_on_disk=None, on_disk_payload=None,
        note="manifest without a profile block (pre step-1 dumps)")


def mirror_int8(dense_dim: int, distance: str, hnsw_m: int) -> ServingProfile:
    """`bm25_saturate`'s pass-2 target: the source's dense width, int8, plain-idf sparse."""
    return ServingProfile(
        name="bm25-saturate-mirror", dense_dim=int(dense_dim), dense_datatype="float32",
        datatype_explicit=False, distance=distance, quantizer="int8",
        sparse_index_on_disk=None, sparse_index_datatype=None, hnsw_m=int(hnsw_m),
        note="one-shot BM25-saturation rebuild target")


# ---------------------------------------------------------------------------------
# THE creator
# ---------------------------------------------------------------------------------
def create_collection(client, name: str, profile: ServingProfile, *,
                      on_exists: str = "refuse", index_failure: str = "raise",
                      verify: bool = False, bulk: bool = False, log=print) -> bool:
    """Create `name` from `profile`. The ONLY place in `cytir/` that calls Qdrant's
    create_collection (the serving-profile drift test ratchets this).

    on_exists      "refuse" raise `CollectionExistsError` | "keep" return False, untouched |
                   "recreate" delete it first (the caller has already decided that is safe)
    index_failure  "raise" a payload index that fails aborts | "warn" print and go on
    verify         read the collection back and refuse a shape that is not the profile
    bulk           create at hnsw m=0 (no graph during a load); `finalize_collection`
                   PATCHes the profile's m afterwards
    Returns True when it created the collection.
    """
    if on_exists not in ("refuse", "keep", "recreate"):
        raise ValueError(f"on_exists {on_exists!r}")
    if index_failure not in ("raise", "warn"):
        raise ValueError(f"index_failure {index_failure!r}")
    if client.collection_exists(name):
        if on_exists == "keep":
            return False
        if on_exists == "refuse":
            raise CollectionExistsError(
                f"{name} already exists. Refusing to delete a collection that may be "
                f"someone's index; drop it deliberately first.")
        client.delete_collection(name)
    m = 0 if (bulk and profile.hnsw_m is not None) else profile.hnsw_m
    kwargs = dict(collection_name=name, vectors_config=profile.vectors_config())
    sparse = profile.sparse_vectors_config()
    if sparse is not None:
        kwargs["sparse_vectors_config"] = sparse
    hnsw = profile.hnsw_config(m)
    if hnsw is not None:
        kwargs["hnsw_config"] = hnsw
    if profile.on_disk_payload is not None:
        kwargs["on_disk_payload"] = profile.on_disk_payload
    quant = profile.quantization_config()
    if quant is not None:
        kwargs["quantization_config"] = quant
    if profile.optimizers:
        kwargs["optimizers_config"] = qm.OptimizersConfigDiff(**dict(profile.optimizers))
    client.create_collection(**kwargs)
    for fld, ftype, on_disk in profile.payload_indexes:
        try:
            client.create_payload_index(
                name, field_name=fld, field_schema=profile.index_schema(ftype, on_disk))
        except Exception as exc:  # noqa: BLE001
            if index_failure == "raise":
                raise
            import sys
            print(f"[embed] WARNING payload index {fld!r} on {name} failed: {exc!r}",
                  file=sys.stderr, flush=True)
    log(f"[create] {name} profile={profile.name} dim={profile.dense_dim} "
        f"dtype={profile.dense_datatype} quant={profile.quantizer} "
        f"m={m} idx={len(profile.payload_indexes)}")
    if verify:
        from cytir.core import serving_config as SC
        SC.assert_collection_matches(client, name, profile, check_indexed=False)
    return True


def shape_diff(a: dict, b: dict, path="") -> dict:
    """Every leaf where two shape dicts differ: {path: (a, b)}."""
    out = {}
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k, "<absent>"), b.get(k, "<absent>")
        if isinstance(x, dict) and isinstance(y, dict):
            out.update(shape_diff(x, y, f"{path}{k}."))
        elif x != y:
            out[f"{path}{k}"] = (x, y)
    return out


def is_server(client) -> bool:
    """True only for a client talking to a Qdrant SERVER. The in-process local client
    (`:memory:` or a path) accepts a quantizer, HNSW and payload-index request and stores
    none of it, so a read-back from it proves nothing about the layout."""
    inner = getattr(client, "_client", None)
    return inner is not None and type(inner).__name__ == "QdrantRemote"


def read_shape(client, name: str) -> dict:
    """`shape_of` of a collection read through a client (`get_collection`)."""
    d = client.get_collection(name).model_dump(mode="json")
    return shape_of({"config": d["config"], "payload_schema": d.get("payload_schema")})


def verify_created(client, name: str, profile: ServingProfile, *, log=print) -> dict:
    """Read the collection back and compare it, field by field, to what `profile` declares.
    A server that disagrees raises `ShapeMismatch`. A local client cannot store the layout:
    that is said in the receipt (`verified: False`), never claimed as a verified shape."""
    if not is_server(client):
        reason = ("a local/in-process Qdrant stores no quantizer, HNSW graph or payload "
                  "index: the layout was REQUESTED, not verified")
        log(f"[verify] {name}: NOT VERIFIED ({reason})")
        return {"verified": False, "reason": reason, "profile": profile.name}
    diffs = shape_diff(read_shape(client, name), profile.expected_shape())
    if diffs:
        raise ShapeMismatch(
            f"collection {name} read back from the server is not profile "
            f"{profile.name!r}: " + "; ".join(f"{k}: read {a!r}, profile {b!r}"
                                               for k, (a, b) in diffs.items()))
    log(f"[verify] {name}: read back from the server equals profile {profile.name!r}")
    return {"verified": True, "reason": "read back from a Qdrant server",
            "profile": profile.name}


def finalize_collection(client, name: str, profile: ServingProfile, log=print) -> None:
    """After a bulk load: PATCH hnsw to the profile's m so the graph is built once."""
    log(f"[finalize] {name}: PATCH hnsw m={profile.hnsw_m}")
    client.update_collection(name, hnsw_config=qm.HnswConfigDiff(m=profile.hnsw_m))


def profile_from_serving_config(cfg, *, with_payload_indexes: bool = True) -> ServingProfile:
    """The profile a `serving_config.ServingConfig` has always created (kwargs byte-equal)."""
    return ServingProfile(
        name=cfg.name, dense_dim=cfg.dense_dim, dense_datatype=cfg.dense_datatype,
        dense_on_disk=cfg.dense_on_disk, quantizer=cfg.quantizer,
        quant_always_ram=cfg.quant_always_ram,
        payload_indexes=tuple(
            (f, "float" if t == qm.PayloadSchemaType.FLOAT else "keyword", False)
            for f, t in cfg.payload_indexes.items()) if with_payload_indexes else (),
        note=cfg.note)
