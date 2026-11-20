"""cytir.index.sparse_rule -- the ONE BM25 prune rule of the aw_e9 layout (task 717).

The rule (task 637 S6, `--prune-df-gt 0.02 --prune-df-lt 2`): drop a sparse term when its
document frequency in the frozen df map is above 2 % of the corpus (a near-stopword) or is
below 2 (a hapax). A term the map does not hold is KEPT. The aw_e9 rebuild applied it to
33.7M rows; the drain wrote every later row unpruned (task 708 section 3.1: 38.2 % of prod's
rows, 74 extra postings each). This module is the single definition both writers import:

* the index builder (the collection builder, via
  the arm builder) derives the term set with :func:`build_prune_set` and cuts
  every row with :func:`prune_postings`;
* the promotion drain (`cytir.index.promote_sealed_pack.upsert`) cuts every row it writes to a
  collection that DECLARES the rule, with the same :func:`prune_postings`, against the sealed
  term set of the build (:class:`DropTerms`).

A collection declares the rule in `data/collection-payload-profiles.v1.json`
(`sparse_rule`), keyed by the PHYSICAL collection written. A declared collection MUST carry the
key: an object (the rule) or an explicit `null` ("no rule: this collection stays raw", the twin
`aw_e8`). A declared collection WITHOUT the key is a refusal, so a collection that comes after a
flip (`aw_e10`) cannot take raw writes by omission. A collection the profiles file does not declare
at all (a stranger's build target) writes raw, as before.

What a stored row says: the rule's identity is written into `rule_epochs` as
`{"bm25_prune": <epoch>}`. A row written by the rebuild carries no marker (it predates it) and
an unpruned drain row carries none either, so the marker proves "pruned" and its absence proves
nothing; the posting count is the only evidence for a row without it.

The term set is a sealed input, not a live call: a sorted little-endian uint32 file whose
sha256 and term count the declaration pins. It is hashed ONCE per process (the first
:meth:`SparseRule.for_collection`, which the drain makes at its start, before any twin write) and
the resolved rule is reused for every pack. Where the file lives is the operator setting
`sparse_rule_drop_terms` (env `AW_SPARSE_RULE_DROP_TERMS`); no default, so a box without it
refuses by name rather than writing an unpruned row. `python -m cytir.index.sparse_rule
build-drop-terms` writes it from the build's own df map.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from typing import Any, Callable, Iterable, Mapping, Sequence

SPARSE_VECTOR = "bm25"
RULE_EPOCH_KEY = "bm25_prune"
SETTING = "sparse_rule_drop_terms"
#: The readback after a prod write looks at this many evenly spaced rows of the write.
VERIFY_SAMPLE = 200


class SparseRuleError(RuntimeError):
    """The rule refused: a missing or wrong term set, a row that cannot carry the marker, or
    a written vector that holds a term the rule drops."""


# ----------------------------------------------------------- the rule (moved from the builder)
def qdrant_idf(n, df):
    """Qdrant's own BM25 IDF (task 627 §3.2 validated this inversion)."""
    return math.log(1.0 + (n - df + 0.5) / (df + 0.5))


def df_from_idf(n, idf):
    """Invert qdrant_idf for a threshold: the df at which IDF == idf."""
    # idf = ln(1 + (n-df+0.5)/(df+0.5))  =>  e^idf - 1 = (n-df+0.5)/(df+0.5)
    k = math.exp(idf) - 1.0
    return (n + 0.5 - 0.5 * k) / (k + 1.0)


def build_prune_set(df, n, *, df_gt=None, idf_lt=None, df_lt=None):
    """The term set to REMOVE, and the rule that produced it."""
    rule = {}
    drop = set()
    if df_gt is not None:
        cut = df_gt * n
        drop |= {t for t, v in df.items() if v > cut}
        rule["df_gt_frac"] = df_gt
        rule["df_gt_abs"] = cut
        rule["idf_at_cut"] = qdrant_idf(n, max(1.0, cut))
    if idf_lt is not None:
        cut = df_from_idf(n, idf_lt)
        drop |= {t for t, v in df.items() if v > cut}
        rule["idf_lt"] = idf_lt
        rule["df_at_idf_cut_abs"] = cut
        rule["df_at_idf_cut_frac"] = cut / n
    if df_lt is not None:
        drop |= {t for t, v in df.items() if v < df_lt}
        rule["df_lt_abs"] = df_lt
        rule["idf_at_df_lt"] = qdrant_idf(n, max(1.0, df_lt - 1))
    return drop, rule


def prune_postings(indices: Sequence[int], values: Sequence[float], drop) -> tuple[list, list]:
    """THE cut: the postings whose term is not in `drop`, in their original order, values
    untouched. `drop` is a set of term ids (the builder) or a :class:`DropTerms` (the drain);
    both answer membership, and a test holds them equal on real rows."""
    gone = (drop.contains_each(indices) if hasattr(drop, "contains_each")
            else [i in drop for i in indices])
    kept = [k for k, g in enumerate(gone) if not g]
    return [indices[k] for k in kept], [values[k] for k in kept]


# ------------------------------------------------------------------------- the sealed term set
def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def write_drop_terms(path: pathlib.Path | str, terms: Iterable[int]) -> dict:
    """Write `terms` as a sorted, de-duplicated little-endian uint32 file. A term outside
    uint32 is a refusal (Qdrant sparse indices are u32)."""
    import numpy as np
    arr = np.fromiter(terms, dtype=np.int64)
    if arr.size and (arr.min() < 0 or arr.max() > 0xFFFFFFFF):
        raise SparseRuleError("a term id is outside uint32: min %d max %d" % (arr.min(), arr.max()))
    arr = np.unique(arr).astype("<u4")
    path = pathlib.Path(path)
    arr.tofile(path)
    return {"terms": int(arr.size), "sha256": _sha256_file(path)}


class DropTerms:
    """The term set of the rule, memory-mapped (231 MB on disk for aw_e9; never a Python set).
    Hashes to the declared sha256 and holds the declared number of terms, or it is refused."""

    def __init__(self, path: pathlib.Path | str, *, expect_sha256: str, expect_terms: int):
        import numpy as np
        path = pathlib.Path(path)
        if not path.is_file():
            raise SparseRuleError("REFUSED: the BM25 prune term set %s does not exist" % path)
        got = _sha256_file(path)
        if got != expect_sha256:
            raise SparseRuleError("REFUSED: %s hashes to %s, the collection's sparse_rule declares %s"
                                  % (path, got[:16], expect_sha256[:16]))
        self.path = path
        self.sha256 = got
        self._a = np.memmap(path, dtype="<u4", mode="r")
        if int(self._a.size) != int(expect_terms):
            raise SparseRuleError("REFUSED: %s holds %d terms, the declaration says %d"
                                  % (path, self._a.size, expect_terms))

    def __len__(self) -> int:
        return int(self._a.size)

    def contains_each(self, indices: Sequence[int]) -> list[bool]:
        import numpy as np
        if not len(indices):
            return []
        q = np.asarray(indices, dtype=np.int64)
        out = np.zeros(q.shape, dtype=bool)
        ok = (q >= 0) & (q <= 0xFFFFFFFF)
        qq = q[ok].astype(np.uint32)
        pos = np.searchsorted(self._a, qq)
        pos = np.minimum(pos, self._a.size - 1)
        out[ok] = self._a[pos] == qq
        return out.tolist()

    def __contains__(self, term: int) -> bool:
        return bool(self.contains_each([term])[0])


# ----------------------------------------------------------------------------- the marker
def _declared_entry(collection: str, profiles_path=None) -> Mapping[str, Any] | None:
    """The `sparse_rule` object of the PHYSICAL collection, or None for "no rule". None means
    one of two things only: the profiles file does not declare the collection (a stranger's
    target), or it declares `"sparse_rule": null`. A declared collection that lacks the key is
    refused: silence must never mean "write raw"."""
    from cytir.index import payload_profile as PP
    declared = PP.load_profiles(profiles_path)
    if collection not in declared:
        return None
    if "sparse_rule" not in declared[collection]:
        raise SparseRuleError(
            "REFUSED: %s is declared in %s without a `sparse_rule` key: declare the BM25 prune rule, "
            "or `\"sparse_rule\": null` to say it takes no rule. A collection that comes after a flip "
            "must not take raw writes by omission" % (collection, PP.PROFILES_PATH.name))
    return declared[collection]["sparse_rule"] or None


def declared_epoch(collection: str, *, profiles_path=None) -> int | None:
    """The epoch the PHYSICAL collection's declaration stamps, or None (no rule). Reads the
    declaration only: the term set is not touched, so the pre-twin preflight stays cheap."""
    entry = _declared_entry(collection, profiles_path)
    return int(entry["epoch"]) if entry else None


def stamp_epoch(payload: Mapping[str, Any], epoch: int, rule_id: str = "the BM25 prune rule") -> dict:
    """A copy whose `rule_epochs` carries the rule's epoch. The marker needs a dict to join; a
    coded int (an already projected row) or a missing key is refused, because a row pruned
    under the rule that cannot say so is the defect this field exists for."""
    ep = payload.get("rule_epochs")
    if not isinstance(ep, dict):
        raise SparseRuleError(
            "REFUSED: rule_epochs is %r, not an object: a row written under %s must carry its "
            "epoch in rule_epochs" % (ep, rule_id))
    return dict(payload, rule_epochs=dict(ep, **{RULE_EPOCH_KEY: int(epoch)}))


# ---------------------------------------------------------------------------- the declaration
#: Resolved rules of this process, so a drain hashes the 231 MB term set once, not per upsert.
_RULES: dict = {}
#: The last readback per collection (what the receipt and the journal show).
LAST_READBACK: dict[str, dict] = {}


class SparseRule:
    """A collection's declared rule, resolved once (the term set is hashed a single time)."""

    def __init__(self, collection: str, entry: Mapping[str, Any], drop: DropTerms):
        self.collection = collection
        self.id = entry["id"]
        self.epoch = int(entry["epoch"])
        self.drop = drop

    @classmethod
    def for_collection(cls, collection: str, *, profiles_path=None,
                       drop_path: pathlib.Path | str | None = None) -> "SparseRule | None":
        """The rule of the PHYSICAL collection `collection`, or None when it takes none (the
        twin's explicit null, a stranger's undeclared target). A declared rule whose term set is
        not configured is a refusal, never a quiet raw write; so is a declared collection with no
        `sparse_rule` key. Resolved once per (collection, file): the term set is hashed on the
        first call and the same object comes back after that."""
        entry = _declared_entry(collection, profiles_path)
        if not entry:
            return None
        if drop_path is None:
            from cytir.core import paths
            drop_path = getattr(paths.operator_settings(), SETTING)
        if not drop_path:
            raise SparseRuleError(
                "REFUSED: %s declares the BM25 prune rule %s and the operator setting `%s` "
                "(env AW_SPARSE_RULE_DROP_TERMS) is empty: a prod write without the rule would store "
                "an unpruned row" % (collection, entry["id"], SETTING))
        path = pathlib.Path(drop_path)
        try:
            st = path.stat()
            key = (collection, str(path), st.st_size, st.st_mtime_ns, entry["drop_terms_sha256"],
                   int(entry["terms"]), int(entry["epoch"]), entry["id"])
        except OSError:
            key = None                      # DropTerms refuses a missing file by name
        if key is not None and key in _RULES:
            return _RULES[key]
        rule = cls(collection, entry, DropTerms(path, expect_sha256=entry["drop_terms_sha256"],
                                                expect_terms=entry["terms"]))
        if key is not None:
            _RULES[key] = rule
        return rule

    def stamp_payload(self, payload: Mapping[str, Any]) -> dict:
        return stamp_epoch(payload, self.epoch, self.id)

    # -- the vector
    def project_point(self, point: Mapping[str, Any]) -> dict:
        """The point as it is written to a collection under this rule: sparse vector pruned,
        marker stamped. A row whose every term is dropped loses the bm25 vector, as in the
        rebuild. Refuses when the result holds a dropped term or has more postings than
        the rule allows (raw count minus the dropped ones)."""
        vector = dict(point.get("vector") or {})
        sp = vector.get(SPARSE_VECTOR)
        if sp:
            idx, val = list(sp.get("indices", [])), list(sp.get("values", []))
            kept_i, kept_v = prune_postings(idx, val, self.drop)
            bound = len(idx) - sum(self.drop.contains_each(idx))
            leftover = [t for t, g in zip(kept_i, self.drop.contains_each(kept_i)) if g]
            if leftover or len(kept_i) > bound:
                raise SparseRuleError(
                    "REFUSED: point %s would be written to %s with %d postings against a bound "
                    "of %d under %s, dropped terms left: %s" % (
                        point.get("id"), self.collection, len(kept_i), bound, self.id, leftover[:5]))
            if kept_i:
                vector[SPARSE_VECTOR] = {"indices": kept_i, "values": kept_v}
            else:
                del vector[SPARSE_VECTOR]
        return dict(point, vector=vector, payload=self.stamp_payload(point.get("payload") or {}))

    def project_points(self, points: list[dict]) -> list[dict]:
        """The WHOLE list, before the first PUT: one refused row leaves the target untouched."""
        return [self.project_point(p) for p in points]

    # -- the readback
    def verify_written(self, fetch: Callable[[list], list], written: Sequence[Mapping[str, Any]],
                       *, sample: int = VERIFY_SAMPLE) -> dict:
        """Read back an evenly spaced sample of the rows just written (one retrieve of at most
        `sample` ids, which is <= 500) and prove, on what the collection STORES: no dropped
        term, the same terms as sent, and the stored `rule_epochs` code equal to the sent one.
        `fetch(ids)` is the retrieve with `with_payload: ["rule_epochs"], with_vector: ["bm25"]`."""
        if not written:
            return {"rule": self.id, "checked": 0, "drop_terms_sha256": self.drop.sha256}
        step = max(1, len(written) // sample)
        chosen = list(written[::step][:sample])
        got = {str(r["id"]): r for r in (fetch([p["id"] for p in chosen]) or [])}
        bad = []
        for p in chosen:
            row = got.get(str(p["id"]))
            if row is None:
                bad.append((p["id"], "absent"))
                continue
            sent = (p.get("vector") or {}).get(SPARSE_VECTOR)
            stored = (row.get("vector") or {}).get(SPARSE_VECTOR) or {}
            sidx = list(stored.get("indices", []))
            if any(self.drop.contains_each(sidx)):
                bad.append((p["id"], "stores a dropped term"))
            elif sorted(sidx) != sorted((sent or {}).get("indices", [])):
                bad.append((p["id"], "stored terms differ from the terms sent"))
            elif (row.get("payload") or {}).get("rule_epochs") != (p.get("payload") or {}).get("rule_epochs"):
                bad.append((p["id"], "rule_epochs differs from the one sent"))
        if bad:
            raise SparseRuleError(
                "%s: %d of %d sampled prod row(s) are not in the %s form after the write, e.g. %s"
                % (self.collection, len(bad), len(chosen), self.id, bad[:5]))
        return {"rule": self.id, "epoch": self.epoch, "checked": len(chosen), "mismatches": 0,
                "drop_terms_sha256": self.drop.sha256}


# --------------------------------------------------------------------------------------- build
def _cmd_build(a: argparse.Namespace) -> int:
    """Write the sealed term set from the build's own df map (`df-full.json`, the file
    the arm builder's `scan_df` wrote) with :func:`build_prune_set`, and print the
    declaration's numbers."""
    blob = json.load(open(a.df))
    n, df = blob["n"], {int(k): v for k, v in blob["df"].items()}
    drop, rule = build_prune_set(df, n, df_gt=a.df_gt, df_lt=a.df_lt)
    written = write_drop_terms(a.out, drop)
    print(json.dumps({"df_n": n, "df_terms": len(df), "rule": rule, **written}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cytir.index.sparse_rule")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-drop-terms", help="write the sealed term set from a df map")
    b.add_argument("--df", required=True, help="the build's df map JSON ({n, df: {term: count}})")
    b.add_argument("--out", required=True)
    b.add_argument("--df-gt", type=float, required=True)
    b.add_argument("--df-lt", type=int, required=True)
    b.set_defaults(fn=_cmd_build)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
