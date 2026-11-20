# Dump format v0 — the public artifact contract

Task 650 (migration S8, task 643 §6.2). Governs everything `cyt dump` writes and
everything `cyt apply` reads: the full dump and the daily delta packs. This is a
**declared projection** (release-doctrine #1): what ships is what this file
names, never whatever a scroll happens to return. A payload key this file does
not name does not ship, full stop — a new field is absent from the dump until
this file is edited to include it.

Governing rulings: ADR-039 (per-record `license`, compact dump), ADR-111
(general-web opt-out basis, snippet included), ADR-112 (ODC-By 1.0 database
licence, entity-card name-free projection, mandatory tombstones).

## 0. Two kinds of pack, one delta shape

- **A full dump** (`cyt dump`): every currently-live point, sharded.
- **A delta pack** (`cyt dump --delta`, or the daily cron): adds/replaces since
  a named base, plus tombstones. Same row shape as a full dump; the pack
  manifest additionally names its base and its tombstones.

Internally, `cytir.index.promote_queue` already drains a queue of sealed packs
under its own schema, `aw.promotion-queue-pack.v1` (`schema`, `pack_id`,
`family`, `pack_chunks`, `vectors`, `mrl`, `collection`). That schema is our
**internal** delta — twin/prod promotion, not a stranger's download — and nothing
here changes it. This spec reuses its naming (`schema`, `pack_id`, a `family`
per shard) so the two are legible side by side; unifying them into one format
is a follow-up, not done by this task.

## 1. Manifest — `manifest.json`

One manifest per dump or delta pack, `schema: "aw.dump-manifest.v1"` for a
full dump, `schema: "aw.delta-pack.v1"` for a delta. Required fields:

```jsonc
{
  "schema": "aw.dump-manifest.v1",
  "dump_id": "2026-09-24T00-00-00Z-full",     // stable id, also the dir name
  "created_at": "2026-09-24T00:03:11Z",
  "source": {
    "collection": "aw_e8",
    "qdrant_role": "twin"                      // "twin" always; never "prod" (§5)
  },
  // --- the mirror contract (release-doctrine #5): everything a consumer
  // needs to QUERY the shipped vectors correctly, not just store them.
  "embedder": {
    "model_id": "Snowflake/snowflake-arctic-embed-l-v2.0",
    "revision": "ac6544c8a46e00af67e330e85a9028c66b8cfd9a",
    "dims": 128,                                // MRL width actually shipped
    "quantization": "float32",                  // dump ships float32; the
                                                 // served int8 quantization is
                                                 // a serving-side optimization,
                                                 // not part of the artifact
    "distance": "cosine"
  },
  "sparse": {
    "vector_name": "bm25",
    "tokenizer": "aw-bm25-v1",                  // cytir/core/common.py
                                                 // bm25_tokens + German
                                                 // decompounding (compound_split);
                                                 // cytir/serve/schema.py
                                                 // mirrors it query-side
    "modifier": "idf",                          // the collection scores a sparse hit as
                                                 // stored_weight * ln((n+1)/(df+0.5));
                                                 // `cyt apply` creates it so. Absent =
                                                 // an older dump: no modifier.
    "idf_snapshot": {                           // §6a; null when no row ships a sparse leg
      "schema": "aw.bm25-idf-snapshot.v1",
      "path": "bm25-idf.tsv.gz",
      "formula": "ln((n + 1) / (df + 0.5))",
      "n": 44000123,                            // rows that ship a sparse vector
      "vocabulary": 9876543,                    // lines in the file
      "sha256": "…", "bytes": 81234567
    }
  },
  "counts": {"points": 44123998, "shards": 883},
  "shards": [
    {"path": "shards/shard-00000.parquet", "rows": 50000,
     "sha256": "…", "bytes": 41123456}
    // …
  ],
  "license": {
    "database": "ODC-By-1.0",
    "database_url": "https://opendatacommons.org/licenses/by/1-0/",
    "notice": "authoritative-web index compilation (c) impactIA. Attribution "
              "required for the compilation; each record also carries its "
              "own `license` field, which governs the source content at "
              "that record's `url` and is NOT a distribution gate (ADR-112)."
  },
  "signature": {
    "algo": "minisign",
    "pubkey_id": "RW…",                         // minisign key id, published
                                                  // out of band (NOTICE / repo README)
    "sig_file": "manifest.json.minisig"
  }
}
```

A delta pack's manifest (`aw.delta-pack.v1`) carries everything above under
the same keys, plus:

```jsonc
{
  "schema": "aw.delta-pack.v1",
  "pack_id": "2026-09-25-delta-0001",
  "sequence": 1,                       // monotonic; apply refuses out of order
  "base_dump_id": "2026-09-24T00-00-00Z-full",
  "base_manifest_sha256": "…",         // the FULL dump (or prior delta) this
                                        // pack applies onto; apply refuses a
                                        // pack whose base does not match its
                                        // local state (release-doctrine #3)
  "tombstones": {
    "path": "tombstones.jsonl.gz",
    "count": 412,
    "sha256": "…"
  }
}
```

No `shards`/`counts.points` means the pack carries only tombstones (a pure
takedown/aged-out pack), which is legal.

A **full dump** also declares `fingerprints` (task 687): a gzip sidecar
`fingerprints.tsv.gz` with one `id<TAB>fp` line per row, **sorted by point id**
in Qdrant's scroll order (unsigned integers numerically, then UUIDs; strictly
increasing, so an id never repeats). `fp` is the first 16 hex chars of a sha256
over the row's projected payload text, its dense vector as **exact float32
bits**, and its sparse leg (int64 indices, float32 values): a change-detection
key, not a security digest. It is not a rounding of decimals: a float32 reaches
Python as REST text, as an exact double (gRPC) or from Parquet, and 4-decimal
roundings of those forms disagree on 0.56% of real rows (task 687 refactor);
the float32 bits are the same in all of them.

Every `.gz` file we write has a zero header mtime and no file name, so two dumps of the same rows have the same bytes and sha256.

```jsonc
"fingerprints": {"path": "fingerprints.tsv.gz", "count": 100000,
                 "algo": "aw-row-fp-v2-f32", "sha256": "…"}
```

A builder refuses a base whose `algo` differs from its own. The delta is a
**merge-join** of the sidecar and the scrolled snapshot, both in id order: peak
memory does not depend on the corpus size, and a snapshot that is out of order
or repeats an id is refused.

`cyt dump --delta-base-manifest` diffs against this sidecar and never opens the
base shards. A base manifest without the block is refused (re-dump the base).
The sidecar is covered by the manifest signature through its `sha256`.

Optional keys (task 2504, `delta-change-log.md` 3.4): `change_log_segment` (`{stamp, seq}`: the landing
stamp `YYYYMMDDTHHMMSSZ` and the monotonic segment counter; where the next delta starts reading the master's change log; a self-hoster
ignores it) and, in a delta, `delta_source: {path: "change-log"|"scroll", note, no_ops_dropped?}`.

## 2. Shards — Parquet, zstd

Each shard is one zstd-compressed Parquet file, row-per-point, schema:

| column | type | notes |
|---|---|---|
| `id` | string | the point id **as served** — an opaque key, the one a delta and a tombstone address. It is a uuid5, but NOT always the `cytir.core.ids` I1 id: fedlex and loda serve the older text-based `uuid5(NAMESPACE_URL, url#anchor:sha1(text))`, and the payload's own `point_id` is a different, older id (task 705). Never re-derive it from `url`; a stranger's own rebuild mints other ids. |
| `payload` | string (JSON) | the **declared** compact payload — §3 |
| `dense` | list\<float32\> | present on every row |
| `dense_dims` | int32 | redundant with the manifest, kept per-row so a shard is self-describing without the manifest open |
| `sparse_indices` | list\<int32\>, nullable | absent (null) when the row carries no sparse leg (every `entity_card` row, §3.2) |
| `sparse_values` | list\<float32\>, nullable | same nullability as `sparse_indices` |

`payload` is JSON rather than one Parquet column per field on purpose: the
corpus has dozens of family-specific metadata keys (`register_id`, `canton`,
`legal_form`, …) and a fixed columnar schema would either explode into a wide
sparse table or silently drop whatever this spec didn't anticipate. A JSON
payload column is honest about that and keeps the declared-field discipline
in one place (`cytir/distribute/pack.py`), not scattered across a Parquet schema
migration every time a family gains a field.

## 3. The payload projection (release-doctrine #1, ADR-112)

`cyt dump` never writes a scrolled Qdrant payload verbatim. Every row goes
through one of two **whitelists** — a field ships because it is named below,
never because it merely exists on the source payload.

### 3.1 Default projection (every `source_type` except `entity_card`)

```
url, title, snippet, lang, source_type, trust_tier, last_seen, country,
license, domain, tier, content_digest,
search_ok, train_ok, consent_basis, consent_state
```

This is ADR-039's compact shape (`url + license + snippet + title + dense +
sparse + lang + trust_tier + last_seen`) plus the handful of locating/ranking
fields (`domain`, `tier`, `content_digest`) that already ship uncontroversially
in `aw_common.compact_payload()`, plus the consent disposition (§3.4). Master-only fields never
enter this set: `attribution`, `owner`, `author`, `license_version`, `outlinks`,
`consent_checked_at`, `consent_source`, any `_master` block. The dense vector ships; the sparse vector (named `bm25`) ships when the
source point carries one.

**Several rows may share one `url`** (task 1040, RULING 55). A Fedlex article longer than 300
words is cut into 240-word windows, and every window keeps the article's BARE url (`…#art_N`).
The rows stay distinct by `id` and `content_digest` (tier-B identity: url + digest). The window's
place is the served payload field `window` = `{"index": K, "count": N}` (1-based; absent on a
one-row article). `window` is a field of the STORED payload only. The search API response and the
MCP item do not carry it, and it is NOT in the v0 dump projection above, so a dump consumer cannot
read the window order from any shipped field. Adding it to the dump is a release decision, not
made here.

### 3.2 Entity-card projection (`source_type == "entity_card"`, ADR-112 point 3)

```
url, title, snippet, lang, trust_tier, last_seen, country, license,
register_id, register_scheme, resolve_confidence, source_type
```

Differences from the default projection, all deliberate:

- **No sparse leg at all.** `sparse_indices`/`sparse_values` are always null
  for an entity-card row, even if the source point carries a `bm25` sparse
  vector. the register-card builder's `person_sparse_tokens` and
  `compose_register_card`'s `facet_terms` put officer names into the sparse
  channel as plain BM25 tokens — dropping the leg entirely, rather than
  trying to filter individual tokens out of it, is the only guarantee that
  holds under a future change to how that sparse text is built.
- **No `dense_text` field.** The card-text payload field
  (`compose_register_card`'s `dense`, stored verbatim as `rec["dense_text"]`)
  carries the officer clause (`"Registered signatories/officers (commercial
  register): <names>."`) and is not on the whitelist at all — not stripped,
  simply never selected.
- **`snippet` is recomposed name-free**, not sliced from `dense_text` like the
  live payload's own `snippet = dense_text[:220]`. `cytir/distribute/pack.py`
  strips the officer-clause sentence (it is always the trailing sentence
  `compose_register_card` appends) from `dense_text` before truncating, so
  the shipped snippet never contains a name even when the live snippet
  (built for serving, not for the dump) does. See
  `_name_free_entity_snippet()` / `test_pack.py`.
- **The dense embedding still ships.** Per Tim's ruling (memo 645 Q2,
  2026-09-23 08:00Z), the vector may encode officer names as a residual,
  indirect leak — accepted because the names are already public register
  facts. Only *readable text* is restricted.

Live serving (websear.ch) is untouched by any of this — it keeps `dense_text`,
`sparse_text` and the officer-bearing snippet to build and rank. This
projection only governs what `cyt dump`/`cyt apply` ever see.

### 3.4 Consent flags (ADR-115, task 768)

`search_ok`, `train_ok` (booleans), `consent_basis`, `consent_state` (strings) ship on **every**
row, under the names the served payload carries. The consent was resolved live for the source,
so a consumer needs the answer to honour it (`train_ok=false` rows are in the dump: they may be
searched, and a trainer must skip them). A row whose source payload carries no value ships
**`null`** for that field. `null` means *not recorded*; it is never a default `true`, and a
consumer must treat it as unknown. A field's value is never rewritten by dump or apply. The dump
reads the **raw** twin; a coded value from a `normalized`-profile collection (an integer
`consent_basis`/`consent_state`) is decoded through the server's own enum table
(`cytir.index.payload_profile.decode_payload`) and the string ships; a code the table does not
hold is refused (`ProjectionRefusal`), never shipped as a code. `consent_checked_at`
and `consent_source` (a per-host sentence) stay master-only.

### 3.5 The content digest (task 768)

`content_digest` is **128 bits, 32 lowercase hex**: the first 32 hex of
`sha256(utf8(normalize(chunk_text)))`, with `normalize` = collapse every whitespace run to one
ASCII space and strip (`cytir.core.ids.content_digest`; no case or accent folding). The raw twin stores
the full 64 hex (100 of 100 rows read by id); the served `normalized` profile stores the 32-hex
prefix; the dump ships the prefix either way. A value of any other length or alphabet is not this recipe's output and is **not
shipped**. **What it is for.** The digest hashes the *chunk text*, and the dump ships only a
short `snippet`. Measured on 100 twin rows by id (task 768): it
equals the recomputation over the shipped `snippet` on **49 of 100** rows (the chunk is shorter than
the snippet window) and is **not reproducible from any shipped field on the other 51**. So it is an
**opaque content identity**: "did the content change" (compare two dumps of one id) and "have I
seen this text" (dedup), checkable only by someone who holds the chunk text, e.g. the fetched
page. It is not a proof of the shipped snippet. The change fingerprints (§1) now cover the
consent fields and the 32-hex form, so the first delta against a pre-768 base is O(corpus) once.

### 3.3 The licence field is not a distribution gate

Per ADR-112 point 2, a family whose `license` value is `"unknown"` (or any
other string) is **not** excluded from the dump — absence of a known licence
instrument plus a robots.txt that does not opt out already resolves to
"may index and redistribute" (ADR-111's logic). `cyt dump` still refuses to
ship a row with **no `license` key at all** (empty/missing, not merely
`"unknown"`): `cytir/distribute/license_check.py` (previously a script that
nothing called) asserts *presence*
of the field across the export, never its content. Below `--license-threshold`
(default 0.95, matching the tool's pre-existing default) the export refuses
whole and prints the offending families; it never filters by what the license
string says.

## 4. Tombstones (ADR-112 point 4, release-doctrine #3)

`tombstones.jsonl.gz`, one JSON object per line. Every `.gz` file we write has a zero header mtime and no file name, so two dumps of the same rows have the same bytes and sha256.

```jsonc
{"id": "…", "reason": "superseded"}
```

A line carries no timestamp: the pack's `created_at` and the change log hold the time, so the same rows give the same bytes. (An older pack may carry an `at` key; readers ignore it. No schema bump: pre-launch, nothing consumed it.)

`reason` is one of `superseded | aged_out | takedown | superseded_generation` (no other value —
`cytir/distribute/pack.py:TOMBSTONE_REASONS`). A tombstone is a **mandatory,
forced delete** for `cyt apply` (§5): it is never optional, never merged with
a same-id row that also appears in the pack's shards (a pack that contains
both an upsert and a tombstone for the same id is malformed and `cyt apply`
refuses it), and it always wins — same precedence rule ADR-091/ADR-101a
already give a tombstone inside the served index, extended here into the pack
format.

`superseded_generation` (ADR-114 am.1) marks an old generation of rows retired because the new
rows are already served at the same urls. **PROPOSED widening (task 2497, Tim merges; not yet
ratified):** `superseded_generation` marks a non-takedown forced delete of a row the index retires:
an old generation, a tracking-param copy, or a defective row (a binary file or an error page whose
url leaves the index). The change log maps the queue's `canonical_duplicate`, `binary_content`,
`soft_404` and `restore_rollback` to it (`change_log.QUEUE_TO_PUBLIC_REASON`). It is an ordinary
forced delete. It is not a takedown:
a takedown list is built from `takedown` alone (`cytir/distribute/pack.py:TAKEDOWN_REASONS`), and
no takedown wording (receipt, `TAKEDOWN.md`) covers this reason.

The update queue (not this format) knows three more reasons (ADR-123, `cytir/index/promote_queue.py`):
`canonical_duplicate`, `binary_content` and `restore_rollback`. They are **not** values of this
field today. Only `takedown` feeds a takedown list; none of the three does. `restore_rollback` is
operator-only and will never be a public value. `canonical_duplicate` and `binary_content` become
public values only after `cyt apply` ships a tolerant reader (an unknown reason other than `takedown`
deletes the id and logs the reason); until then `cyt dump` still needs one of the four values above
for every id a delta finds gone.

An unknown `reason` refuses the pack **before any write**: `cyt apply` reads and validates every
tombstone (`cytir/distribute/apply.py:verify_tombstones`) before it upserts a single shard row, so a
bad line never leaves a partial apply.

Irrevocability, stated where a self-hoster will read it (also in
`cytir/distribute/apply.py`'s own `--help` and in `TAKEDOWN.md`): once a torrent
copy of a dump is in the wild, a later tombstone reaches everyone who *applies*
a later pack. It cannot reach a copy that nobody ever re-syncs.

## 5. Where a dump may come from

`cyt dump` always reads from the **twin** (the build box, port 6457 today), never
prod. This is enforced in code (`cytir/distribute/dump.py`), not only by
convention: the default `--qdrant-url` is the twin's own URL, and a small,
hard denylist of known prod-tunnel endpoints (`127.0.0.1:16333`,
`localhost:16333` — `cytir.index.promote_queue`'s own `--prod-url` default)
is refused unconditionally, with no override flag that can lift it. Reading
from anywhere else (a CI fixture instance, a self-hoster's own scratch
Qdrant) is allowed only outside that denylist and requires
`--i-know-this-is-not-the-twin` to say so explicitly.

## 6. Signing

The manifest (not the shards individually — their hashes are already inside
it) is signed with **minisign** (ISC licence, Apache-compatible), a detached
signature at `manifest.json.minisig`. `cytir/distribute/sign.py` shells out to
the `minisign` binary; it does not reimplement Ed25519. `cyt dump --sign-key
<seckey>` signs; omitting `--sign-key` ships an explicitly **unsigned** dump
with a loud warning — `cyt apply` refuses an unsigned dump unless
`--allow-unsigned` is passed (for the CI rehearsal / local dev only; never for
a real release). The public key is published out of band (repo README/NOTICE,
task 643 §5.5's doc set), not inside the dump itself.

### 6a. The pinned BM25 IDF snapshot (full dumps)

A Qdrant sparse index scores `stored_weight * ln((n + 1) / (df + 0.5))` (`cytir/index/layout.py`).
The stored weights are in the shards; `n` and `df` belong to the corpus. A **full dump** ships
them as integers, so a self-hoster's scores can be checked against ours and a query-side
re-weighting can use the same table:

- `bm25-idf.tsv.gz`: one `term_hash<TAB>df` line per term, ascending, gzip with a zero mtime
  (the same corpus gives the same bytes). `cytir/distribute/idf.py`.
- `n` and the file's `sha256` sit in `sparse.idf_snapshot`, which the manifest signature covers.
- It is counted from the dump's own rows in the pass that writes the shards. It is not read
  from the live twin's index header, which carries stale postings (task 627).
- `cyt apply` verifies it before touching Qdrant: file present, `sha256`, schema and formula,
  strictly ascending, `1 <= df <= n`, vocabulary count, `n <= points`. Any miss refuses. The
  state file records the verified `idf_snapshot_sha256`.
- A **delta pack** ships none (its rows are a change set). The snapshot of the full dump it is
  based on stays the reference; the consumer's own Qdrant moves `n` and `df` as packs apply.
- Qdrant servers count `n` over points that HAVE a sparse vector; `qdrant-client` local mode
  counts all points. The CI rehearsal accounts for that and says so.

### 6b. Torrent packaging

`cyt torrent make DIR` writes `DIR.torrent`: a BitTorrent v1 metainfo whose files are exactly
the manifest's: `manifest.json`, its `.minisig`, every shard, the fingerprint sidecar, the
tombstone file and the IDF snapshot. A file the manifest does not name is not in it; a named
file that is missing or whose `sha256`/`bytes` differ refuses (`cytir/distribute/torrent.py`).

- Deterministic: files sorted by path, piece length a function of the total size (or
  `--piece-length`), no date, no creator, no comment. Two runs give the same bytes and
  infohash. `--tracker` and `--webseed` sit outside `info` and never change the infohash.
- `cyt torrent make --private` sets BEP 27 `private=1` in `info` and needs a `--tracker`. It is
  for rehearsals of unreleased data only: a published release is never private (it must stay
  fetchable by DHT and webseed when our tracker is gone). The flag is part of `info`, so a
  private and a public torrent of the same directory have different infohashes.
- `cyt torrent verify FILE DIR` refuses unless the torrent describes the directory byte for
  byte. `cyt dump --torrent` makes it after signing, so the `.minisig` is in it.
- Publish the infohash beside the signed manifest. v1 (SHA-1 pieces) only; integrity comes from
  the signature over the manifest and the `sha256` of every file in it.

## 7. `cyt apply` contract (self-hoster side)

1. Verify `manifest.json.minisig` against the published public key (unless
   `--allow-unsigned`).
2. Verify every shard's `sha256` before touching Qdrant.
3. For a **full dump**: create the collection from the manifest's `embedder`/
   `sparse` block (dense + optional sparse named vectors, matching dims and
   distance), then upsert every shard.
4. For a **delta pack**: refuse if `base_manifest_sha256` does not match the
   locally-recorded state (§1); otherwise upsert the pack's shards, then apply
   every tombstone as a forced point delete, in the order the pack lists them.
5. Report point counts before/after and the shard/tombstone counts applied —
   `cyt apply` states the resulting state, not its own exit code, as the
   measure of success (release-doctrine #4).

## 8. Not done by this spec

- Merging the internal promotion-queue format (`aw.promotion-queue-pack.v1`)
  and this one into a single schema.
- Publishing the minisign public key, and the seeding/hosting of the torrent (the packaging is
  done: §6b).
- BEP 52 (v2) hashes.
- An IDF snapshot for a delta pack (§6a: by design none).
