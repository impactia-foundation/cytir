"""cytir.index.row_decode -- ONE canonical in-memory row for a stored payload of ANY profile (task 855).

WHY. A reader of the twin was written for the RAW twin (`aw_e8`, profile `prod`): string enums
and a 64-hex `content_digest`. From the aw_e10 flip the twin is a CODED copy of prod (ruling 27
Q1, profile `rebuild_e10`): ten keys hold the integer code of a committed table, the digest keeps
its first 128 bits (32 hex), and `stamps`, `point_id`, `tier`, `build_id` are not stored. A raw
reader on a coded row compares 32 hex with 64 hex (never equal), matches no family on an integer
`provenance`, and crashes on `.strip()` / `.get()` of an int. Task 847 S2 counted about eight such
readers on the live path.

WHAT. Every reader of a stored row calls :func:`decode_row` (or a :class:`RowDecoder` resolved
once per collection) BEFORE it reads a key. The result is the CANONICAL row, the same for a raw
row and for the same row coded:

* an enum key holding an integer is decoded with the committed table
  (`data/payload-enum-tables.v1.json`). A string is already a value (a raw row, or an OPEN key
  written raw because its value has no code). No table value is an int (tested in
  `payload_profile`), so the spelling says which it is; a code with no value raises.
* ``content_digest`` is ALWAYS its first 32 chars (:func:`digest_key`), for every profile.
  The projection keeps 32 chars and drops the rest; 32 cannot be turned back into 64, so the
  canonical digest is the 32-char form and every raw reader compares on it -- a pack's 64-hex
  digest goes through :func:`digest_key` too. For a sha256 hex digest two texts share the prefix
  with probability about n^2 / 2^129 (1.5e-21 at n = 1e9 rows). NOT every digest is a hash: the
  StackExchange family writes `se-<host>-<answer id>`, and on a long host name the first 32 chars
  hold no answer id, so every answer on that host has ONE key (aw_e9 already stores it cut so). A
  reader therefore never decides on the digest ALONE (every routed compare also matches url and
  family). No data migration: the stored bytes stay as they are on both profiles.
* ``published`` is in its one contract form (`cytir.core.published`), as every writer stores it.
* a key the PROFILE never stores is removed and named in ``row.absent`` (a frozenset): a reader
  tells "this profile does not carry `tier`" (``"tier" in row.absent``) from "this row lacks it".

So ``decode_row(project(raw, P), profile=P) == decode_row(raw, profile=P)`` for every profile P
(tested on real twin rows), and the raw twin row decoded under `prod` differs from the coded one
only by ``row.absent``.

WHICH PROFILE. :func:`profile_of` reads the declaration (`data/collection-payload-profiles.v1.json`);
it never infers a profile from a row. An UNDECLARED name (an alias such as `aw_prod`, a scratch or
test collection) reads as ``None``. Its raw rows decode as usual, but an integer CODE in it is
refused (:class:`RowDecodeError`): an index arm built with `grow=True` numbers values in the order
it meets them, so the committed table may give such a code the wrong value. Resolve an alias to its
physical name, or declare the collection. :func:`as_canonical_row` (rows a caller hands in, no collection
named) decodes codes with the committed table. A staged declaration (task 847) reads like any
other: a reader is not a writer.

THE FLIP-DAY SWITCH is data, not code: the twin setting (`twin_collection`, cytir.core.twin) names
the new twin and its declaration names its profile. :class:`RowDecoder` caches per collection on
the two data files' (mtime, size), so a process whose checkout changes picks the new declaration
up without a restart (a unit that runs from a fixed release directory reads that release's file).
Stdlib only.
"""
from __future__ import annotations

import json
import os
import pathlib
import threading
from typing import Any, Iterable, Mapping

from cytir.core.published import normalize_payload_published
from cytir.index import payload_profile as PP

#: The digest key and its canonical length (the projection's `prefix16`).
DIGEST_KEY = "content_digest"
DIGEST_HEX = PP.DIGEST_PREFIX_HEX

#: Every key any profile codes (`enum`) -- the keys whose integer spelling is a code.
ENUM_KEYS: frozenset[str] = frozenset(
    k for rules in PP.PROFILE_RULES.values() for k, r in rules.items() if r == "enum")


class RowDecodeError(PP.PayloadProfileError):
    """A stored code has no value in the committed table: the table is behind the collection."""


class CanonicalRow(dict):
    """A decoded payload. A plain dict to every reader; `absent` names the keys the row's profile
    never stores, `profile` the declared profile it was read under (None = undeclared)."""

    __slots__ = ("absent", "profile")

    def __init__(self, data: Mapping[str, Any], *, absent: frozenset[str] = frozenset(),
                 profile: str | None = None):
        super().__init__(data)
        self.absent = absent
        self.profile = profile

    def __reduce__(self):            # copy/pickle keep the markers
        return (_rebuild_row, (dict(self), self.absent, self.profile))


def _rebuild_row(data, absent, profile):
    return CanonicalRow(data, absent=absent, profile=profile)


def digest_key(value: Any) -> Any:
    """The canonical spelling of a `content_digest`: its first 32 chars (128 bits), exactly what
    the projection stores. A 64-hex raw digest and its 32-hex stored form map to the same key;
    a non-string (None, absent) is returned as is."""
    return value[:DIGEST_HEX] if isinstance(value, str) else value


def dropped_keys(profile: str | None) -> frozenset[str]:
    """The keys `profile` never stores (rule `drop`); empty for None and for `prod`."""
    if profile is None:
        return frozenset()
    if profile not in PP.PROFILE_RULES:
        raise PP.PayloadProfileError("unknown payload profile %r" % (profile,))
    return frozenset(k for k, r in PP.PROFILE_RULES[profile].items() if r == "drop")


def _inverse(tables: Mapping[str, Mapping[str, int]]) -> dict[str, dict[int, str]]:
    return {k: {code: value for value, code in t.items()} for k, t in tables.items()}


def _is_code(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def decode_row(payload: Mapping[str, Any] | None, *, profile: str | None,
               tables: Mapping[str, Mapping[str, int]] | None = None,
               _inverse_tables: Mapping[str, Mapping[int, str]] | None = None,
               refuse_codes_of: str | None = None) -> CanonicalRow:
    """The canonical row of a stored payload read under `profile` (see the module docstring).
    `tables` = the FORWARD tables (`payload_profile.load_tables()`); default: the committed file.
    `refuse_codes_of` = the name of an UNDECLARED collection: an integer code in it is refused.
    Idempotent: decoding a canonical row returns an equal row."""
    inv = _inverse_tables
    if inv is None:
        inv = _inverse(tables) if tables is not None else _default_tables()[1]
    absent = dropped_keys(profile)
    out: dict[str, Any] = {}
    for k, v in normalize_payload_published(payload or {}).items():
        if k in absent:
            continue
        if k in ENUM_KEYS and _is_code(v):
            if refuse_codes_of is not None:
                raise RowDecodeError(
                    "collection %r is not declared in %s and stores a CODE in %s (%r): its table "
                    "is not known (a build may number values its own way). Read the physical, "
                    "declared collection, or declare it." % (refuse_codes_of, PP.PROFILES_PATH.name,
                                                             k, v))
            try:
                out[k] = json.loads(inv[k][v])
            except KeyError:
                raise RowDecodeError(
                    "stored %s code %r has no value in %s: the committed table is behind the "
                    "collection (a code is added by a reviewed data PR)" % (k, v, PP.TABLES_PATH.name)
                ) from None
        elif k == DIGEST_KEY:
            out[k] = digest_key(v)
        else:
            out[k] = v
    return CanonicalRow(out, absent=absent, profile=profile)


# ----------------------------------------------------------------------- declarations
def _stat_key(path: pathlib.Path) -> tuple:
    try:
        st = os.stat(path)
        return (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return (str(path), None, None)


_LOCK = threading.Lock()
_TABLES_CACHE: dict[tuple, dict] = {}
_DECODER_CACHE: dict[tuple, "RowDecoder"] = {}


def _default_tables() -> tuple[dict[str, dict[str, int]], dict[str, dict[int, str]]]:
    """(forward, inverse) of the committed table file, cached on its (mtime, size)."""
    key = _stat_key(PP.TABLES_PATH)
    with _LOCK:
        hit = _TABLES_CACHE.get(key)
    if hit is None:
        fwd = PP.load_tables()
        hit = (fwd, _inverse(fwd))
        with _LOCK:
            _TABLES_CACHE.clear()
            _TABLES_CACHE[key] = hit
    return hit


def declaration_of(collection: str, *, profiles_path: pathlib.Path | str | None = None
                   ) -> dict | None:
    """The declared entry of a PHYSICAL collection name, or None when it is not declared."""
    return PP.load_profiles(profiles_path).get(collection)


def profile_of(collection: str, *, profiles_path: pathlib.Path | str | None = None) -> str | None:
    """The declared payload profile of `collection` (`prod` = raw; `normalized`, `rebuild_e10`
    ... = coded), or None for an undeclared name. Read from the declaration, never from a row."""
    entry = declaration_of(collection, profiles_path=profiles_path)
    return None if entry is None else entry["profile"]


class RowDecoder:
    """`decode_row` for ONE collection, resolved once: the profile from its declaration and the
    tables (sha-checked against the declaration's `tables_sha256` when it pins one)."""

    def __init__(self, collection: str | None, profile: str | None, *,
                 tables: Mapping[str, Mapping[str, int]] | None = None, declared: bool = True):
        self.collection = collection
        self.profile = profile
        self.declared = declared
        self.absent = dropped_keys(profile)
        if tables is None:
            self.tables, self._inv = _default_tables()
        else:
            self.tables, self._inv = dict(tables), _inverse(tables)

    @classmethod
    def for_collection(cls, collection: str, *, profiles_path: pathlib.Path | str | None = None,
                       tables_path: pathlib.Path | str | None = None) -> "RowDecoder":
        """The decoder of `collection` (cached on the declaration and table files' mtime and
        size, so a data commit is picked up by a running process)."""
        key = (collection, _stat_key(pathlib.Path(profiles_path or PP.PROFILES_PATH)),
               _stat_key(pathlib.Path(tables_path or PP.TABLES_PATH)))
        with _LOCK:
            hit = _DECODER_CACHE.get(key)
        if hit is not None:
            return hit
        entry = declaration_of(collection, profiles_path=profiles_path)
        profile = None if entry is None else entry["profile"]
        if entry is not None and PP.PROFILE_RULES[profile]:
            tables = PP.load_tables(tables_path, expect_sha256=entry.get("tables_sha256"))
        elif tables_path is not None:
            tables = PP.load_tables(tables_path)
        else:
            tables = None
        dec = cls(collection, profile, tables=tables, declared=entry is not None)
        with _LOCK:
            if len(_DECODER_CACHE) > 64:
                _DECODER_CACHE.clear()
            _DECODER_CACHE[key] = dec
        return dec

    def __call__(self, payload: Mapping[str, Any] | None) -> CanonicalRow:
        return self.decode(payload)

    def decode(self, payload: Mapping[str, Any] | None) -> CanonicalRow:
        return decode_row(payload, profile=self.profile, _inverse_tables=self._inv,
                          refuse_codes_of=None if self.declared else self.collection)

    def decode_map(self, rows: Mapping[str, Mapping[str, Any] | None]) -> dict[str, CanonicalRow]:
        """`{id: payload}` -> `{id: canonical row}` (the shape of `point_delete.fetch_payload_fields`)."""
        return {i: self.decode(p) for i, p in rows.items()}

    def decode_points(self, points: Iterable[Mapping[str, Any]]) -> list[dict]:
        """Qdrant point dicts (`{"id", "payload", ...}`) with each payload decoded; other keys kept."""
        return [dict(p, payload=self.decode(p.get("payload"))) for p in points]


def reader(collection: str) -> RowDecoder:
    """Shorthand for :meth:`RowDecoder.for_collection` -- what a reader calls."""
    return RowDecoder.for_collection(collection)


def same_digest(a: Any, b: Any) -> bool:
    """True when two digests name the same text in canonical form (either may be 64 or 32 hex);
    False when either is empty."""
    ka, kb = digest_key(a), digest_key(b)
    return bool(ka) and bool(kb) and ka == kb


def as_canonical_row(row: Mapping[str, Any] | None) -> CanonicalRow:
    """`row` as a canonical row: returned as is when it already is one (its `absent` kept), else
    decoded key by key under no declared profile. For a check that receives rows from several
    callers (some decode with the collection's profile, some hand in what Qdrant returned)."""
    if isinstance(row, CanonicalRow):
        return row
    return decode_row(row, profile=None)


def as_canonical_rows(rows: Mapping[str, Mapping[str, Any] | None]) -> dict[str, CanonicalRow]:
    """`{id: payload}` -> `{id: canonical row}` through :func:`as_canonical_row`."""
    return {i: as_canonical_row(p) for i, p in rows.items()}
