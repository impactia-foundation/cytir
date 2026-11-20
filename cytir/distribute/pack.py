"""cytir.distribute.pack — the declared row/manifest schema (task 650, S8).

Implements `specs/dump-format.md`. This is the ONE place the public dump's
payload projection is defined — `cytir/distribute/dump.py` (write side) and
`cytir/distribute/apply.py` (read side) both import it rather than each
re-deriving their own idea of "the compact record" (engineering-doctrine #1,
One Truth, One Home).

Two whitelists, not a deny-list (release-doctrine #1: "a field ships because
it is named below, never because it merely exists on the source payload").
`cytir/core/common.py:compact_payload()` is a deny-list (`MASTER_ONLY_KEYS`)
used for the internal staging->prod ship, where the two sides trust each
other's payload shape. A stranger's download does not get that trust: a new
payload key added to the live index tomorrow is simply absent from the dump
until this file names it, not shipped by default and fixed later.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

DUMP_MANIFEST_SCHEMA = "aw.dump-manifest.v1"
DELTA_PACK_SCHEMA = "aw.delta-pack.v1"

# --------------------------------------------------------------- projection
#: The per-record consent disposition (ADR-115; specs/dump-format.md §3.4). These four ship on
#: EVERY row, as an explicit `null` when the source row carries no value: an absent flag is
#: "unknown", and a consumer must never read it as `true` (release-doctrine, consent re-resolved
#: per source, never inherited). `consent_checked_at` and `consent_source` stay master-only.
CONSENT_FIELDS = ("search_ok", "train_ok", "consent_basis", "consent_state")

#: The content digest ships under its stored name, at 128 bits (specs/dump-format.md §3.5).
DIGEST_FIELD = "content_digest"
DIGEST_HEX = 32

#: Every source_type except entity_card (specs/dump-format.md §3.1).
DEFAULT_PROJECTION_FIELDS = (
    "url", "title", "snippet", "lang", "source_type", "trust_tier",
    "last_seen", "country", "license", "domain", "tier", DIGEST_FIELD,
) + CONSENT_FIELDS

#: source_type == "entity_card" (specs/dump-format.md §3.2, ADR-112 point 3).
#: Deliberately excludes `dense_text`/`sparse_text`/`facet_terms` — the
#: officer-bearing fields — by never naming them, not by stripping them.
ENTITY_CARD_PROJECTION_FIELDS = (
    "url", "title", "snippet", "lang", "trust_tier", "last_seen", "country",
    "license", "register_id", "register_scheme", "resolve_confidence",
    "source_type",
) + CONSENT_FIELDS

#: The trailing officer/signatory clause a card's dense text may carry. Two
#: phrasings are known to occur in the pipeline today, both appended as the
#: LAST sentence of the composed text (never mid-string):
#:   - `cytir/corpus/register_cards.py:compose_register_card` and
#:     `cytir/corpus/card_channels.py:OFFICER_CLAUSE` —
#:     "Registered signatories/officers (commercial register): <names>."
#:   - `cytir/corpus/card_recipe.py`'s wikidata-phase-3 builder (the
#:     `by_facet.get("person")` branch) — "Registered signatories/officers:
#:     <names>." with NO "(commercial register)" parenthetical.
#: A hardcoded regex naming only one exact phrasing is a landmine: any
#: future card generation that promotes this builder's `dense_text` into a
#: served `snippet` (the way `compose_register_card` already does) would
#: leak names through undetected. Match the INVARIANT marker phrase
#: ("Registered signatories/officers") and strip everything after it,
#: whatever the suffix — never re-litigate this decision by hand-listing
#: every phrasing that shows up. Anchoring on "strip to end" is also why an
#: officer's own name may itself contain a period (e.g. "Dr. A. Muster")
#: without truncating wrong — a mid-string `[^.]*\.` stop-at-first-period
#: regex would, "strip to end" does not.
_OFFICER_CLAUSE_RE = re.compile(
    r"\s*Registered signatories/officers\b.*$",
    re.S,
)

#: Mirrors `compose_register_card`'s own 220-char snippet window so the
#: dump's snippet is the same LENGTH as the served one, just name-free.
_ENTITY_SNIPPET_MAX_CHARS = 220


def _name_free_entity_snippet(payload: dict) -> str:
    """A entity-card snippet with the officer clause removed before, not
    after, truncation (a blind `dense_text[:220]` slice — what the live
    payload's own `snippet` field is — can leave a truncated fragment of a
    name behind; recomposing from the stripped text cannot)."""
    dense = payload.get("dense_text") or payload.get("snippet") or ""
    stripped = _OFFICER_CLAUSE_RE.sub("", dense).strip()
    if not stripped:
        stripped = str(payload.get("title") or "").strip()
    if len(stripped) <= _ENTITY_SNIPPET_MAX_CHARS:
        return stripped
    head = stripped[:_ENTITY_SNIPPET_MAX_CHARS].rsplit(" ", 1)[0]
    return (head or stripped[:_ENTITY_SNIPPET_MAX_CHARS]) + "…"


class ProjectionRefusal(ValueError):
    """A row cannot be projected onto the declared dump schema as-is."""


_HEX = re.compile(r"[0-9a-f]+")


def _digest32(value) -> Optional[str]:
    """The shipped digest: the first 128 bits (32 hex) of sha256 over the whitespace-normalised
    chunk text (`cytir.core.ids.content_digest`; the normalized payload profile keeps the same
    prefix, `cytir.index.payload_profile.DIGEST_PREFIX_HEX`). Any other length or alphabet is NOT
    this recipe's output (a legacy sha1, a placeholder) and is not shipped under its name."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if len(v) in (64, DIGEST_HEX) and _HEX.fullmatch(v):
        return v[:DIGEST_HEX]
    return None


def _decode_consent(payload: dict) -> dict:
    """A `normalized`-profile collection (the served one) stores `consent_basis` and
    `consent_state` as integer codes of the committed table. Decode them through
    `cytir.index.payload_profile.decode_payload` (the table the server uses) before projecting, so
    the dump ships the string. A code the table does not hold is a `ProjectionRefusal`."""
    coded = {k: payload[k] for k in ("consent_basis", "consent_state")
             if isinstance(payload.get(k), int) and not isinstance(payload.get(k), bool)}
    if not coded:
        return payload
    from cytir.index import payload_profile as PP
    try:
        return {**payload, **PP.decode_payload(coded, "normalized", PP.load_tables())}
    except PP.PayloadProfileError as e:
        raise ProjectionRefusal(f"consent code not in the payload enum table: {e}") from e


def _consent_value(key: str, value):
    """`None` stays `None`. A flag is a bool; basis/state are strings. Anything else (a code
    that survived decoding, a non-bool flag) is refused, never shipped."""
    if value is None or value == "":
        return None
    if key in ("search_ok", "train_ok"):
        if isinstance(value, bool):
            return value
    elif isinstance(value, str):
        return value
    raise ProjectionRefusal(
        f"consent field {key!r} carries {value!r}: not a bool flag / string value")


def project_payload(payload: dict) -> dict:
    """Project one scrolled Qdrant payload onto the declared dump schema.

    Returns a NEW dict; never mutates `payload`. Raises `ProjectionRefusal`
    on a payload with no usable `url` — the dump's join key — rather than
    silently emitting an unlocatable row.
    """
    if not payload.get("url"):
        raise ProjectionRefusal(f"payload has no url: {payload!r}")
    if (payload.get("source_type") or "") == "entity_card":
        fields, digest = ENTITY_CARD_PROJECTION_FIELDS, False
    else:
        fields, digest = DEFAULT_PROJECTION_FIELDS, True
    row = {k: payload[k] for k in fields
           if k not in CONSENT_FIELDS and k != DIGEST_FIELD
           and payload.get(k) not in (None, "", [], {})}
    payload = _decode_consent(payload)
    if digest and _digest32(payload.get(DIGEST_FIELD)):
        row[DIGEST_FIELD] = _digest32(payload.get(DIGEST_FIELD))
    for k in CONSENT_FIELDS:
        row[k] = _consent_value(k, payload.get(k))
    if (payload.get("source_type") or "") == "entity_card":
        row["snippet"] = _name_free_entity_snippet(payload)
        row["source_type"] = "entity_card"
    return row


def ships_sparse(payload: dict) -> bool:
    """Whether a projected row is allowed to carry a sparse (bm25) leg.

    Entity cards never do (ADR-112 point 3) — the sparse channel is where
    `cytir.sources.zefix.facets:person_sparse_tokens` and
    `compose_register_card`'s `facet_terms` put officer names as plain BM25
    tokens; dropping the whole leg is the only guarantee that survives a
    future change to how that text is built.
    """
    return (payload.get("source_type") or "") != "entity_card"


# ------------------------------------------------------------------ tombstones
# The vocabulary lives in the stdlib-only leaf `tombstone_vocab` (task 2532): the change log needs it and the
# drain's import closure is a guarded surface, so it must not import this writer module. Re-exported here.
from cytir.distribute.tombstone_vocab import TAKEDOWN_REASONS, TOMBSTONE_REASONS  # noqa: E402,F401


@dataclass(frozen=True)
class Tombstone:
    id: str
    reason: str
    at: str = ""          # accepted for old callers and old packs; never written (dump-format.md section 4)

    def __post_init__(self):
        if self.reason not in TOMBSTONE_REASONS:
            raise ValueError(
                f"tombstone reason {self.reason!r} not in {sorted(TOMBSTONE_REASONS)}")
        if not self.id:
            raise ValueError("tombstone requires a non-empty id")

    def to_json(self) -> dict:
        return {"id": self.id, "reason": self.reason}


def open_gzip_text_writer(path: str, compresslevel: int = 9):
    """A gzip text writer whose bytes depend on the content only: header mtime 0 and no
    file name (`gzip.open` stamps the wall clock, so two dumps of the same rows differed in
    their sidecar bytes and sha256; found by the server-side rehearsal, task 2339)."""
    import io

    from cytir.acquire.gzip_pinned import deterministic_gzip_open

    raw = open(path, "wb")
    gz = deterministic_gzip_open(raw, compresslevel=compresslevel)

    class _Writer(io.TextIOWrapper):
        def close(self):
            try:
                super().close()
            finally:
                raw.close()

    return _Writer(gz, encoding="utf-8")


def write_tombstones(tombstones: Iterable[Tombstone], path: str) -> tuple[int, str]:
    """Write `tombstones.jsonl.gz`; returns (count, sha256 of the gz file)."""
    n = 0
    with open_gzip_text_writer(path) as fh:
        for t in tombstones:
            fh.write(json.dumps(t.to_json(), ensure_ascii=False) + "\n")
            n += 1
    return n, sha256_file(path)


def read_tombstones(path: str) -> Iterator[Tombstone]:
    import gzip

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            yield Tombstone(id=d["id"], reason=d["reason"], at=d.get("at", ""))


# ------------------------------------------------------------------------ IO
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: Parquet shard row shape (specs/dump-format.md §2). Kept as one place so
#: dump.py (write) and apply.py (read) share the exact column list/order.
SHARD_COLUMNS = ("id", "payload", "dense", "dense_dims",
                  "sparse_indices", "sparse_values")


def row_to_record(point_id: str, dense: list[float], payload: dict,
                   sparse: Optional[tuple[list[int], list[float]]]) -> dict:
    """One Parquet row, ready for `write_parquet_shard`."""
    indices, values = (sparse if sparse else (None, None))
    return {
        "id": point_id,
        "payload": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        "dense": [float(x) for x in dense],
        "dense_dims": len(dense),
        "sparse_indices": list(indices) if indices is not None else None,
        "sparse_values": [float(x) for x in values] if values is not None else None,
    }


def write_parquet_shard(rows: list[dict], path: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("id", pa.string()),
        ("payload", pa.string()),
        ("dense", pa.list_(pa.float32())),
        ("dense_dims", pa.int32()),
        # int64, NOT int32: a BM25 sparse index is `int(md5(term)[:8], 16)`
        # (cytir/core/common.py:_hash) -- the full unsigned 32-bit range,
        # 0..4294967295. A signed int32 column overflows on roughly half of
        # real term hashes ("Value 2167715904 too large to fit in C integer
        # type" on the first real-twin shard task 654 wrote; a synthetic
        # fixture with indices like [1, 5, 100] never exercises this).
        # int64 rather than a pyarrow uint32 column for the widest read-side
        # compatibility (many Parquet readers/languages have no native
        # unsigned type).
        ("sparse_indices", pa.list_(pa.int64())),
        ("sparse_values", pa.list_(pa.float32())),
    ])
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd")


def read_parquet_shard(path: str) -> Iterator[dict]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches():
        for row in batch.to_pylist():
            row = dict(row)
            row["payload"] = json.loads(row["payload"])
            yield row


# ------------------------------------------------------------ fingerprints
#: A full dump carries one `id<TAB>fingerprint` line per row in a gzip sidecar,
#: declared in the manifest's `fingerprints` block, so the NEXT delta builds
#: against the previous dump's digests instead of re-reading its shards
#: (task 687, S8). The fingerprint is a change-detection stability key, not a
#: security digest: 16 hex chars (64 bit) is plenty to tell "this row changed".
#:
#: The sidecar is SORTED by `point_id_key` (strictly increasing, so no id
#: repeats). That is Qdrant's own scroll order, so a delta streams the
#: current snapshot against the sidecar as a merge-join with O(1) memory
#: (task 687 refactor): no `{id: fp}` map exists at any corpus size.
FINGERPRINT_HEX = 16
FINGERPRINTS_FILENAME = "fingerprints.tsv.gz"
#: Declared in the manifest block; a base written under another algorithm is
#: refused by the delta builder (its fingerprints are not comparable).
FINGERPRINT_ALGO = "aw-row-fp-v2-f32"


def row_fingerprint(row: dict) -> str:
    """The per-row change-detection key (NOT a security digest): sha256 over
    the projected payload text, the dense vector as EXACT float32 bits, and
    the sparse leg (int64 indices, float32 values), first 16 hex chars.

    Why float32 bits and not rounded decimals (task 687 refactor): Qdrant's
    REST scroll returns a float32 as its shortest decimal text (`-0.06305`),
    which Python parses to a double that is NOT the float32 (99.97% of values
    on the real twin). The same float32 read back from Parquet, or over gRPC,
    is the exact double (`-0.06305000185966492`). A 4-decimal rounding of the
    two disagrees whenever the value sits on a rounding tie (0.56% of rows:
    335 of 60,000 on the twin). Both doubles narrow to the SAME float32, so
    hashing the float32 bits is independent of the transport and of the
    Parquet round trip. The sparse leg is in the key so a re-tokenised row
    with unchanged text and dense vector is still shipped as changed."""
    from array import array

    h = hashlib.sha256()
    h.update(row["payload"].encode("utf-8"))
    h.update(b"\0")
    h.update(array("f", row["dense"]).tobytes())
    h.update(b"\0")
    if row.get("sparse_indices") is not None:
        h.update(array("q", row["sparse_indices"]).tobytes())
        h.update(b"\0")
        h.update(array("f", row["sparse_values"]).tobytes())
    return h.hexdigest()[:FINGERPRINT_HEX]


class FingerprintOrderError(ValueError):
    """Ids were not strictly increasing in `point_id_key` order."""


def point_id_key(point_id: str) -> tuple:
    """Qdrant's scroll order over point ids: unsigned integers (numeric)
    first, then UUIDs (canonical lower-case hyphenated text compares like
    the 128-bit number). Any other string (synthetic test ids only; Qdrant
    accepts nothing else) sorts last, lexicographically."""
    if point_id.isascii() and point_id.isdigit():
        return (0, int(point_id), "")
    if (len(point_id) == 36
            and point_id[8] == point_id[13] == point_id[18] == point_id[23] == "-"):
        return (1, 0, point_id)
    return (2, 0, point_id)


class FingerprintWriter:
    """Append-only writer for the per-row fingerprint sidecar. Refuses an id
    that is not strictly greater than the previous one: an unsorted sidecar
    would make the next delta's merge-join mislabel rows, so the writer fails
    closed instead of sorting in memory."""

    def __init__(self, path: str):
        import gzip

        self.path = path
        self.count = 0
        self._last = None
        self._fh = open_gzip_text_writer(path, compresslevel=3)

    def add(self, id_: str, fingerprint: str) -> None:
        key = point_id_key(id_)
        if self._last is not None and not key > self._last:
            self._fh.close()
            raise FingerprintOrderError(
                f"id {id_!r} is not after the previous id in point-id order "
                f"(duplicate or out-of-order scroll); the fingerprint sidecar "
                f"must be strictly sorted")
        self._last = key
        self._fh.write(f"{id_}\t{fingerprint}\n")
        self.count += 1

    def close(self) -> dict:
        """Closes the file; returns the manifest's `fingerprints` block."""
        self._fh.close()
        return {"path": pathlib.Path(self.path).name, "count": self.count,
                "algo": FINGERPRINT_ALGO, "sha256": sha256_file(self.path)}


def check_fingerprints_sidecar(path: str, *, expected_sha256: str) -> None:
    """Refuses a sidecar whose sha256 does not match the manifest's block
    (the block is covered by the manifest signature, so a swapped sidecar
    must fail here, not silently diff). One streamed pass, O(1) memory."""
    got = sha256_file(path)
    if got != expected_sha256:
        raise ValueError(f"fingerprint sidecar {path} sha256 {got} != "
                         f"manifest {expected_sha256}")


def iter_fingerprints(path: str, *, expected_count: int
                      ) -> Iterator[tuple[str, str, tuple]]:
    """Stream `(id, fingerprint, key)` in sidecar order, O(1) memory. Order is
    checked as it streams and the row count against the manifest when the
    stream ends: a reordered or short sidecar raises instead of diffing."""
    import gzip

    n, last = 0, None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            id_, _, fp = line.rstrip("\n").partition("\t")
            key = point_id_key(id_)
            if last is not None and not key > last:
                raise FingerprintOrderError(
                    f"fingerprint sidecar {path} is not strictly sorted at {id_!r}")
            last = key
            n += 1
            yield id_, fp, key
    if n != expected_count:
        raise ValueError(f"fingerprint sidecar has {n} ids, manifest "
                         f"says {expected_count}")


# ------------------------------------------------------------------- manifest
REQUIRED_MANIFEST_FIELDS = ("schema", "dump_id", "created_at", "source",
                            "embedder", "counts", "shards", "license")
REQUIRED_DELTA_FIELDS = REQUIRED_MANIFEST_FIELDS + (
    "pack_id", "sequence", "base_dump_id", "base_manifest_sha256")


def validate_manifest(manifest: dict) -> dict:
    """Fail-loud shape check; returns `manifest` unchanged for chaining."""
    schema = manifest.get("schema")
    if schema == DUMP_MANIFEST_SCHEMA:
        required = REQUIRED_MANIFEST_FIELDS
    elif schema == DELTA_PACK_SCHEMA:
        required = REQUIRED_DELTA_FIELDS
    else:
        raise ValueError(
            f"manifest schema {schema!r} not one of "
            f"{DUMP_MANIFEST_SCHEMA!r}/{DELTA_PACK_SCHEMA!r}")
    missing = [f for f in required if f not in manifest]
    if missing:
        raise ValueError(f"manifest missing required field(s): {missing}")
    return manifest
