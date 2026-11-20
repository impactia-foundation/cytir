"""cytir.index.payload_profile -- the ONE definition of a collection's payload form.

A collection stores its payload in a *profile*, and the profile is a property of
the COLLECTION, declared in data -- never inferred from a sample row, never a
flag a caller remembers to pass.

* ``prod``       -- the raw form (aw_e8, the twin).
* ``normalized`` -- task 637 section 3.3 (aw_e9, served since 2026-09-29): ten keys
  become the integer code of a committed table, ``stamps`` is dropped and
  ``content_digest`` keeps its first 128 bits (32 hex).

Every writer that puts a payload into a declared collection calls
:func:`project_payload` -- the streaming index builder (``cytir.index.from_collection``) and the promotion drain (``cytir/index/promote_sealed_pack.upsert``).
A reader that needs the original values calls :func:`decode_payload`. The two
declarations that make this a contract are data files:

* ``data/collection-payload-profiles.v1.json`` -- collection -> profile (+ the
  sha256 of the enum tables it was built with);
* ``data/payload-enum-tables.v1.json`` -- code -> value, per key, with the
  provenance of the build manifest they were read from.

Unknown values (task 675, the conservative rule): a CLOSED key (declared in the
collection's ``closed_keys``) refuses a value its table does not hold. An OPEN key
(declared in ``open_keys``) is written RAW when its value has no code, and
decoded as itself. Nothing is ever assigned a code by a writer: a code is part of
the contract, and a contract changes by a reviewed data PR.

Stdlib only; no import from ``pipeline/`` (the dependency arrow points inward).
"""
from __future__ import annotations

from cytir.core.paths import data_path

import hashlib
import json
import pathlib
from typing import Any, Mapping, MutableMapping

from cytir.core.published import normalize_payload_published

REPO = pathlib.Path(__file__).resolve().parents[2]
PROFILES_PATH = data_path("collection-payload-profiles.v1.json")
TABLES_PATH = data_path("payload-enum-tables.v1.json")

SCHEMA_PROFILES = "aw.collection-payload-profiles/v1"
#: A declaration's `state` (task 847). `live` (the default when the key is absent): every writer
#: projects with it. `staged`: declared, reviewed and tested BEFORE the collection exists or serves
#: (aw_e10 during its build); a live writer is refused, a build may write it, and `live_target`
#: does not count it as a live name. The flip commit sets it `live`.
STATE_LIVE, STATE_STAGED = "live", "staged"
STATES = (STATE_LIVE, STATE_STAGED)
SCHEMA_TABLES = "aw.payload-enum-tables/v1"

# --- the rules of each profile (moved out of the retired experiment builder) --- #
# `drop`     : the key is not written at all.
# `enum`     : the value is replaced by its code in a committed table (information-
#              preserving: the table decodes it).
# `prefix16` : a hex digest is truncated to its first 32 hex chars (128 bits).
# `point_id` is NOT in `normalized`: it equals the Qdrant id on only 14.56% of
# aw_e8 points and is the only carrier of the canonical id for the rest (task 637
# section 3.3a). `rebuild_ids_aligned` drops it and is valid ONLY for a rebuild that
# makes the Qdrant id the canonical id.
PROFILE_RULES: dict[str, dict[str, str]] = {
    "prod": {},
    "normalized": {
        "stamps": "drop",
        "rule_epochs": "enum",
        "build_id": "enum",
        "chunk_rule": "enum",
        "consent_source": "enum",
        "consent_basis": "enum",
        "consent_state": "enum",
        "provenance": "enum",
        "prov_id": "enum",
        "content_digest": "prefix16",
    },
}
PROFILE_RULES["rebuild_ids_aligned"] = dict(PROFILE_RULES["normalized"], point_id="drop")
# `rebuild_e10` (ADR-122 point 4, ADR-067 am.1 point 3, spec aw-e10-rebuild E5 + E8): the
# normalized profile of aw_e9 for a rebuild that re-keys every row to I1, so the Qdrant id IS
# the canonical id and `point_id` names nothing; `tier` and `build_id` have no reader after the
# reader audit of task 713 section 9. Only the deleted `aw.index.rebuild_e10` wrote it (task 1312; the `rebuild_e10` profile key stays).
PROFILE_RULES["rebuild_e10"] = dict(PROFILE_RULES["normalized"], point_id="drop", tier="drop",
                                    build_id="drop")

DIGEST_PREFIX_HEX = 32


class PayloadProfileError(RuntimeError):
    """The profile contract refused: an undeclared collection, a table that does
    not match its declared sha256, or a value the profile has no code for."""


class StagedCollectionError(PayloadProfileError):
    """A live writer asked for a collection whose declaration is still `staged`."""


class UnknownValueError(PayloadProfileError):
    """A closed key carried a value its enum table does not hold."""

    def __init__(self, key: str, value: Any, collection: str | None = None):
        self.key, self.value, self.collection = key, value, collection
        super().__init__(
            "REFUSED: payload key %r carries a value with no code in the %s enum "
            "table (%s). A code is part of the contract and is added by a reviewed "
            "data PR (data/payload-enum-tables.v1.json), never by a writer."
            % (key, collection or "declared", json.dumps(value, ensure_ascii=False)[:160]))


# ------------------------------------------------------------------ the data files
def _canon(value: Any) -> str:
    """The table's value spelling: the exact `json.dumps` the builder used."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def file_sha256(path: pathlib.Path | str) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load_profiles(path: pathlib.Path | str | None = None) -> dict[str, dict]:
    path = PROFILES_PATH if path is None else path
    doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA_PROFILES:
        raise PayloadProfileError("%s: schema %r != %r" % (path, doc.get("schema"),
                                                          SCHEMA_PROFILES))
    for name, entry in doc["collections"].items():
        if entry["profile"] not in PROFILE_RULES:
            raise PayloadProfileError("%s: collection %s declares unknown profile %r"
                                      % (path, name, entry["profile"]))
        if "serving" in entry and not isinstance(entry["serving"], bool):
            raise PayloadProfileError("%s: collection %s: `serving` must be true or false, got %r"
                                      % (path, name, entry["serving"]))
        state = entry.get("state", STATE_LIVE)
        if state not in STATES:
            raise PayloadProfileError("%s: collection %s: `state` must be one of %s, got %r"
                                      % (path, name, list(STATES), state))
        if state == STATE_STAGED and entry.get("serving") is True:
            raise PayloadProfileError("%s: collection %s is `staged` and `serving`: a staged "
                                      "collection serves nothing until the flip commit sets it live"
                                      % (path, name))
    return doc["collections"]


def is_staged(entry: Mapping[str, Any]) -> bool:
    """True for a declaration that a build may write and no live writer may (task 847)."""
    return entry.get("state", STATE_LIVE) == STATE_STAGED


def staged_collections(declared: Mapping[str, dict] | None = None) -> frozenset[str]:
    declared = load_profiles() if declared is None else declared
    return frozenset(n for n, e in declared.items() if is_staged(e))


def entry_for_collection(collection: str, declared: Mapping[str, dict] | None = None,
                         *, allow_staged: bool = False) -> dict:
    """The declared entry for a PHYSICAL collection name. An undeclared collection
    is a refusal, never a default: a writer must not guess a profile. A `staged`
    collection (task 847: `aw_e10` before its flip) is a refusal too, unless the
    caller says it is not a live writer (`allow_staged`): the writer contract of a
    staged collection switches on by ONE data commit, `state` -> `live`."""
    declared = load_profiles() if declared is None else declared
    if collection not in declared:
        raise PayloadProfileError(
            "REFUSED: collection %r has no declared payload profile in %s. Declare "
            "it (a data PR); a writer never infers the profile from a row."
            % (collection, PROFILES_PATH.name))
    if is_staged(declared[collection]) and not allow_staged:
        raise StagedCollectionError(
            "REFUSED: collection %r is declared `staged` in %s: its writer contract is OFF until "
            "the flip commit sets `state` to `live` (task 847 runbook, step F7). No live writer "
            "writes a staged collection." % (collection, PROFILES_PATH.name))
    return declared[collection]


def load_tables(path: pathlib.Path | str | None = None, *,
                expect_sha256: str | None = None) -> dict[str, dict[str, int]]:
    """The FORWARD tables `{key: {value_json: code}}` from the committed data file.

    `expect_sha256` is the collection's declared `tables_sha256`; a table that does
    not hash to it is the wrong table, and encoding against it would give a value
    the wrong integer.
    """
    path = pathlib.Path(TABLES_PATH if path is None else path)
    if expect_sha256 is not None and file_sha256(path) != expect_sha256:
        raise PayloadProfileError(
            "REFUSED: %s hashes to %s, the profile declares %s"
            % (path.name, file_sha256(path)[:16], expect_sha256[:16]))
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA_TABLES:
        raise PayloadProfileError("%s: schema %r != %r" % (path, doc.get("schema"),
                                                          SCHEMA_TABLES))
    return {key: {value: int(code) for code, value in table.items()}
            for key, table in doc["tables"].items()}


def tables_document(inverted_tables: Mapping[str, Mapping[str, str]],
                    provenance: Mapping[str, Any]) -> dict:
    """The committed file's content from a build manifest's `payload_enum_tables`
    (`{key: {code: value_json}}`): codes in numeric order, keys sorted."""
    tables = {k: {str(c): v for c, v in sorted(t.items(), key=lambda kv: int(kv[0]))}
              for k, t in sorted(inverted_tables.items())}
    return {"schema": SCHEMA_TABLES, "provenance": dict(provenance), "tables": tables}


# ---------------------------------------------------------------------- projection
def project_payload(payload: Mapping[str, Any], profile: str,
                    tables: MutableMapping[str, dict[str, int]], *,
                    grow: bool = False, open_keys: frozenset[str] = frozenset(),
                    collection: str | None = None) -> dict:
    """The ONE projection of a raw payload into `profile`.

    * `grow=True` (index BUILD only): an unseen enum value is assigned the next
      code, exactly as the builder always did.
    * `grow=False` (every writer into a live collection): an unseen value of a
      closed key raises :class:`UnknownValueError`; of an `open_keys` key it is
      kept RAW.
    * Idempotent: an enum key already holding an int is left alone, so projecting
      a projected row is the identity (no table value is an int -- tested).
    """
    # Every profile writes `published` in the ONE contract form (task 676,
    # cytir.core.published): an ISO date in UTC (or a reduced-precision ISO year /
    # year-month the source knows no better). Done first, so it holds for the raw
    # `prod` profile too and is idempotent under re-projection.
    payload = normalize_payload_published(payload)
    rules = PROFILE_RULES[profile]
    if not rules:
        return payload
    out: dict[str, Any] = {}
    for k, v in payload.items():
        rule = rules.get(k)
        if rule is None:
            out[k] = v
        elif rule == "drop":
            continue
        elif rule == "enum":
            if isinstance(v, int) and not isinstance(v, bool):
                out[k] = v
                continue
            key = _canon(v)
            tbl = tables.setdefault(k, {}) if grow else tables.get(k, {})
            if key in tbl:
                out[k] = tbl[key]
            elif grow:
                tbl[key] = len(tbl)
                out[k] = tbl[key]
            elif k in open_keys:
                out[k] = v
            else:
                raise UnknownValueError(k, v, collection)
        elif rule == "prefix16":
            out[k] = v[:DIGEST_PREFIX_HEX] if isinstance(v, str) else v
        else:
            raise PayloadProfileError("unknown payload rule %r" % rule)
    return out


class Projector:
    """A collection's projection, resolved once (the tables are read and hashed a
    single time, not per batch)."""

    def __init__(self, collection: str, entry: Mapping[str, Any], *,
                 tables_path: pathlib.Path | str | None = None):
        self.collection = collection
        self.profile = entry["profile"]
        self.open_keys = frozenset(entry.get("open_keys") or ())
        self.identity = not PROFILE_RULES[self.profile]
        self.tables: dict[str, dict[str, int]] = {}
        if not self.identity:
            self.tables = load_tables(tables_path,
                                      expect_sha256=entry.get("tables_sha256"))
            enum_keys = {k for k, r in PROFILE_RULES[self.profile].items() if r == "enum"}
            stray = set(self.open_keys) - enum_keys
            if stray:
                raise PayloadProfileError(
                    "%s: open_keys %s are not enum keys of profile %s"
                    % (collection, sorted(stray), self.profile))

    @classmethod
    def for_collection(cls, collection: str, *,
                       profiles_path: pathlib.Path | str | None = None,
                       tables_path: pathlib.Path | str | None = None,
                       allow_staged: bool = False) -> "Projector":
        """The projector a live writer uses for a PHYSICAL collection name, read
        from the declaration. Undeclared -> refusal; staged -> refusal unless
        `allow_staged` (a reader or a test of the staged contract, never a writer)."""
        entry = entry_for_collection(collection, load_profiles(profiles_path),
                                     allow_staged=allow_staged)
        return cls(collection, entry, tables_path=tables_path)

    def project(self, payload: Mapping[str, Any]) -> dict:
        return project_payload(payload, self.profile, self.tables, grow=False,
                               open_keys=self.open_keys, collection=self.collection)

    def project_points(self, points: list[dict]) -> list[dict]:
        """A copy of `points` whose payloads are projected. Refuses the WHOLE batch
        on an unknown closed value, before any point is sent."""
        return [dict(p, payload=self.project(p.get("payload") or {})) for p in points]


def decode_payload(payload: Mapping[str, Any], profile: str,
                   tables: Mapping[str, Mapping[str, int]]) -> dict:
    """Back to the original values. `stamps` and the digest's last 128 bits are not
    recoverable from the row (they are stored elsewhere / not carried); everything
    else is."""
    rules = PROFILE_RULES[profile]
    inverse = {k: {code: value for value, code in t.items()} for k, t in tables.items()}
    out: dict[str, Any] = {}
    for k, v in payload.items():
        if rules.get(k) == "enum" and isinstance(v, int) and not isinstance(v, bool):
            try:
                out[k] = json.loads(inverse[k][v])
            except KeyError:
                raise PayloadProfileError("code %r has no value in table %r" % (v, k))
        else:
            out[k] = v
    return out


def form_of(payload: Mapping[str, Any], profile: str,
            open_keys: frozenset[str] = frozenset()) -> str:
    """`projected` / `raw` / `mixed` / `n/a` -- what a stored row actually is,
    judged by its enum keys (an int is a code, anything else is a raw value).
    `open_keys` are ignored: either spelling is legal for them."""
    enum_keys = [k for k, r in PROFILE_RULES[profile].items()
                 if r == "enum" and k in payload and k not in open_keys]
    if not enum_keys:
        return "n/a"
    coded = [isinstance(payload[k], int) and not isinstance(payload[k], bool)
             for k in enum_keys]
    if all(coded):
        return "projected"
    if not any(coded):
        return "raw"
    return "mixed"
