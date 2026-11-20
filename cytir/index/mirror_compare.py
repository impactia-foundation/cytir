"""Twin row against prod row, by CONTENT (task 833: guards G1 and G2 of task 831).

WHY. A Tier-B point id is `uuid5(canonical_url + "#" + content_digest)` (`cytir.core.ids`,
`classify_tier` + `mint_point_id`). So "the same id is present on both sides" already means "the
same digest on both sides", and a check that compares the digest of a present id cannot fail. The
drain's finalize of a `CARRIED-BY-FLIP` pack did exactly that, and 6,247 rows that the twin had
rewritten in place stayed stale on prod `aw_e9` with every check green (task 830). The rewrite
changed what the digest does not cover: consent, licence, title, site name, the `www.` url
spelling, provenance, rule epochs and the dense vector.

WHAT "EQUAL" MEANS HERE (one definition, for the drain's prod verify and for the finalize):

* REFERENCE FORM (task 855). The reference row is first read as its CANONICAL row
  (`cytir.index.row_decode.as_canonical_row`): a coded twin row (enum codes, 32-hex digest; the twin after
  the aw_e10 flip) and a raw one (aw_e8, a pack row) give the same values, so the steps below never
  re-project a projected row (it was `<unstorable>` on every coded row: `stamp_epoch` on an int).
* PAYLOAD. The reference row is put into the TARGET's stored form by the writer's own two steps
  (`promote_sealed_pack.upsert`): the BM25 rule's epoch stamp in `rule_epochs`
  (`sparse_rule.stamp_epoch`, when the target declares a rule), then the payload projection
  (`payload_profile.Projector.project`). Then every key of the union of both rows must match.
  No key is skipped: not the digest, not a timestamp, not a key one side lacks.
  - Numbers match within a relative 1e-9 (a JSON store may re-spell a float); nothing looser.
  - An OPEN key (`open_keys` of the declaration: written raw while its value had no code) matches
    when both spellings project to the same value under the current tables.
  - `allow_unstamped_rule_epochs` (the finalize only): the stored `rule_epochs` may also be the
    UNSTAMPED projection. A row that a build or a flip carried predates the epoch marker: the
    aw_e9 build pruned under the rule but never stamped it (task 717), and task 830's EQUAL class
    holds exactly that form. A row the drain has just written must carry the stamp (G2 never
    allows this).
* DENSE. Both `dense` vectors, unit-normalised (the collections are Cosine, and Qdrant stores
  them normalised), differ by at most 1e-3 per component and have a cosine of at least 0.9999.
  Task 830: equal rows differ by at most 4e-4 (fp16 storage); a re-embedded row by more than 1e-2.

A row missing on either side, a reference row the target's profile cannot hold, and a missing
dense vector are all MISMATCHES, never passes. This module never calls Qdrant: the caller reads
both sides (by id) and hands the rows in.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from cytir.index import payload_profile as PP
from cytir.index import row_decode as RD
from cytir.index import sparse_rule as SR

DENSE = "dense"
DENSE_ATOL = 1e-3
DENSE_MIN_COS = 0.9999
FLOAT_REL_TOL = 1e-9
#: how many differing rows a receipt or an error names
EXAMPLES = 20


def values_equal(a: Any, b: Any) -> bool:
    """Deep equality where numbers match within `FLOAT_REL_TOL` (bool is never a number here)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b or math.isclose(a, b, rel_tol=FLOAT_REL_TOL, abs_tol=0.0)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(values_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b))
    return a == b


def dense_of(row: Mapping[str, Any] | None) -> list[float] | None:
    """The `dense` vector of a stored or a pack row (`{"vector": {"dense": [...]}}`), or None."""
    vec = (row or {}).get("vector")
    if isinstance(vec, dict):
        vec = vec.get(DENSE)
    return vec if isinstance(vec, list) and vec else None


def dense_equal(a: list[float] | None, b: list[float] | None) -> bool:
    if a is None or b is None or len(a) != len(b):
        return False
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:                       # a zero vector has no direction: compare as stored
        return max(abs(x - y) for x, y in zip(a, b)) <= DENSE_ATOL
    ua, ub = [x / na for x in a], [y / nb for y in b]
    cos = sum(x * y for x, y in zip(ua, ub))
    return cos >= DENSE_MIN_COS and max(abs(x - y) for x, y in zip(ua, ub)) <= DENSE_ATOL


class Comparator:
    """Reference rows (twin or pack, raw form) against stored rows of one TARGET collection."""

    def __init__(self, collection: str, *, projector: PP.Projector | None = None,
                 allow_unstamped_rule_epochs: bool = False):
        self.collection = collection
        self.projector = projector or PP.Projector.for_collection(collection)
        self.epoch = SR.declared_epoch(collection)
        self.allow_unstamped = allow_unstamped_rule_epochs

    def target_form(self, payload: Mapping[str, Any]) -> dict:
        """The payload as the writer would store it in this collection (stamp, then project).
        Raises `PP.UnknownValueError` / `SR.SparseRuleError` when it cannot be stored at all."""
        p = dict(payload)
        if self.epoch is not None:
            p = SR.stamp_epoch(p, self.epoch)
        return self.projector.project(p)

    def _open_key_equal(self, key: str, want: Any, got: Any) -> bool:
        if key not in self.projector.open_keys:
            return False
        try:
            return values_equal(self.projector.project({key: want})[key],
                                self.projector.project({key: got})[key])
        except PP.PayloadProfileError:
            return False

    def _unstamped_rule_epochs(self, payload: Mapping[str, Any]) -> Any:
        try:
            return self.projector.project(dict(payload)).get("rule_epochs")
        except PP.PayloadProfileError:
            return None

    def diff(self, ref: Mapping[str, Any] | None, stored: Mapping[str, Any] | None) -> list[str]:
        """The keys on which `stored` is not `ref` in this collection's form; [] when equal.
        `dense` names a vector difference; `<absent-on-...>` and `<unstorable:...>` a row that
        cannot be compared at all (a mismatch, never a pass)."""
        if ref is None:
            return ["<absent-on-reference>"]
        if stored is None:
            return ["<absent-on-target>"]
        try:
            src = ref.get("payload") or {}
            raw = RD.as_canonical_row(src)
            if self.projector.identity and RD.DIGEST_KEY in src:
                # a RAW target stores the full digest: compare the reference's own spelling
                raw = RD.CanonicalRow(dict(raw, **{RD.DIGEST_KEY: src[RD.DIGEST_KEY]}),
                                      absent=raw.absent, profile=raw.profile)
            want = self.target_form(raw)
        except (PP.PayloadProfileError, SR.SparseRuleError) as exc:
            return [f"<unstorable:{getattr(exc, 'key', type(exc).__name__)}>"]
        got = stored.get("payload") or {}
        keys = []
        for k in sorted(set(want) | set(got)):
            if k in want and k in got and values_equal(want[k], got[k]):
                continue
            if k in want and k in got and self._open_key_equal(k, want[k], got[k]):
                continue
            if (k == "rule_epochs" and self.allow_unstamped and k in got
                    and values_equal(self._unstamped_rule_epochs(raw), got[k])):
                continue
            keys.append(k)
        if not dense_equal(dense_of(ref), dense_of(stored)):
            keys.append(DENSE)
        return keys

    def compare(self, ids: Iterable[str], refs: Mapping[str, Any], stored: Mapping[str, Any]) -> dict:
        """`{checked, equal, mismatched, by_key, examples}` over `ids`; `examples` names the first
        `EXAMPLES` differing ids with their keys."""
        checked = equal = 0
        by_key: dict[str, int] = {}
        examples: list[dict] = []
        mismatched = 0
        for pid in ids:
            checked += 1
            keys = self.diff(refs.get(pid), stored.get(pid))
            if not keys:
                equal += 1
                continue
            mismatched += 1
            for k in keys:
                by_key[k] = by_key.get(k, 0) + 1
            if len(examples) < EXAMPLES:
                examples.append({"id": pid, "keys": keys})
        return {"collection": self.collection, "checked": checked, "equal": equal,
                "mismatched": mismatched, "by_key": dict(sorted(by_key.items())),
                "examples": examples}
