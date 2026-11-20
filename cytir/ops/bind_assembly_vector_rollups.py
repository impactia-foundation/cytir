"""Bind each contributing set's sealed vector rollup into the claim manifests.

READ-ONLY over the corpus; rewrites only the manifest index passed as ``--in``
to ``--out``.

The ADR-101 claim-manifest contract is
``{set_id, vector_root, rows, dim, dtype, receipt_sha256, mode, claims,
tombstones}``.  ``produce_assembly_artifacts.py`` mints the claims half by
walking point IDs.  This stage mints the vector half by following each set's
own witness ``RUN.json`` to ``inputs.portable_vector_receipt`` and reading the
row count out of the receipt it names.

Without it, guard G1 cannot check the contract, and guards G5 and G7 cannot
tell a set that has no vectors yet from a set whose vectors exist but were
never declared -- which is the difference between WAITING and FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

RECEIPT_KEYS = ("portable_vector_receipt", "vector_receipt",
                "recursive_vector_verification")
ROW_KEYS = ("rows", "records", "vector_rows", "points")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows_from(payload: dict) -> int | None:
    for key in ROW_KEYS:
        value = payload.get(key)
        if isinstance(value, int):
            return value
    counts = payload.get("counts")
    if isinstance(counts, dict):
        for key in ROW_KEYS + ("embedded", "vectors"):
            value = counts.get(key)
            if isinstance(value, int):
                return value
    verification = payload.get("verification")
    if isinstance(verification, dict):
        for key in ROW_KEYS:
            value = verification.get(key)
            if isinstance(value, int):
                return value
    return None


def bind_one(root: Path, run_uri: str) -> dict | None:
    """Follow a witness RUN.json to its sealed vector rollup receipt."""
    run_path = root / run_uri.removeprefix("artifact://")
    if not run_path.is_file():
        return {"error": f"witness RUN not readable: {run_uri}"}
    run = json.loads(run_path.read_text())
    inputs = run.get("inputs") or {}
    binding = None
    for key in RECEIPT_KEYS:
        candidate = inputs.get(key)
        if isinstance(candidate, dict) and candidate.get("uri"):
            binding = (key, candidate)
            break
    if binding is None:
        return {"error": "witness RUN names no vector receipt"}
    key, candidate = binding
    receipt_path = root / candidate["uri"].removeprefix("artifact://")
    if not receipt_path.is_file():
        return {"error": f"vector receipt not readable: {candidate['uri']}"}
    actual = sha256_file(receipt_path)
    if candidate.get("sha256") and actual != candidate["sha256"]:
        return {"error": f"vector receipt digest drift: {actual}"}
    receipt = json.loads(receipt_path.read_text())
    rows = _rows_from(receipt)
    out = {
        "receipt_key": key,
        "receipt_uri": candidate["uri"],
        "receipt_sha256": actual,
        "root": receipt.get("source") or receipt.get("root")
        or str(Path(candidate["uri"]).parent),
        "dim": receipt.get("dim") or (receipt.get("vector") or {}).get("dim"),
        "dtype": receipt.get("dtype")
        or (receipt.get("vector") or {}).get("dtype"),
        "status": receipt.get("status"),
    }
    if rows is None:
        out["error"] = "vector receipt declares no row count"
    else:
        out["rows"] = rows
    return out


def _derive_row_split(row: dict) -> None:
    """Turn a set's point-level loss into a row-level suppression (guard G5).

    Task 456 §5: the ``source-native-atomic-record@1`` families are strictly
    one vector row per old point, so the rows they forfeit equal the points
    they forfeit.  The exact-production families re-chunk, so their rows do not
    map 1:1 -- but they forfeit nothing under ADR-101 Ruling 1, so the
    distinction does not bite.  Both facts are asserted here, not assumed: an
    atomic family whose row count differs from its claim count breaks the
    premise the whole arithmetic rests on, and the run refuses.
    """
    adjudication = row.setdefault("adjudication", {})
    rows = int(row["vector"]["rows"])
    lost = int(adjudication.get("points_lost_to_task427") or 0)
    claims = int((row.get("claims") or {}).get("count") or 0)
    if row.get("class") == "atomic_epoch1":
        if rows != claims:
            raise SystemExit(
                f"{row['set_id']}: atomic family declares {rows:,} vector rows "
                f"for {claims:,} claimed points; the one-row-per-point premise "
                "task 456 §5 relies on does not hold")
        suppressed = lost
    else:
        if lost:
            raise SystemExit(
                f"{row['set_id']}: a re-chunking family forfeits {lost:,} "
                "points, so its row-level suppression cannot be derived from "
                "the point count; it must be measured")
        suppressed = 0
    adjudication["rows_declared"] = rows
    adjudication["rows_suppressed"] = suppressed
    adjudication["rows_loaded"] = rows - suppressed
    adjudication["row_split_basis"] = (
        "one-row-per-old-point (atomic_epoch1)" if row.get("class")
        == "atomic_epoch1" else "forfeits nothing under ADR-101 Ruling 1")


def main(argv=None, prog=None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--census-run", type=Path, required=True)
    parser.add_argument("--in", dest="source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    manifests = json.loads(args.source.read_text())
    census = json.loads(args.census_run.read_text())
    witnesses = census["inputs"]["replacement_point_witnesses"]

    bound = 0
    unbound: list[str] = []
    for row in manifests["sets"]:
        if not row.get("contributes_vectors"):
            continue
        vector = row.get("vector") or {}
        # A previous pass records {"declared": false, "why": ...} for a set it
        # could not bind.  That block is truthy, so testing for presence would
        # count an admitted failure as a success on every later run.
        if vector and vector.get("declared") is not False and vector.get("rows"):
            bound += 1
            _derive_row_split(row) if "rows_declared" not in (
                row.get("adjudication") or {}) else None
            continue
        witness = witnesses.get(row["set_id"])
        if witness is None:
            unbound.append(f"{row['set_id']}: no witness RUN to follow")
            row["vector"] = {
                "declared": False,
                "why": "the set publishes no witness RUN naming a vector "
                       "rollup receipt, so its row count, dim and dtype cannot "
                       "be bound at gate time",
            }
            continue
        result = bind_one(args.corpus_root, witness["run"]["uri"])
        if result and "rows" in result:
            row["vector"] = {"declared": True, **result}
            bound += 1
            _derive_row_split(row)
            print(f"[vec] {row['set_id']:<45} rows={result['rows']:>9,}",
                  flush=True)
        else:
            detail = (result or {}).get("error", "unknown")
            unbound.append(f"{row['set_id']}: {detail}")
            row["vector"] = {"declared": False, "why": detail,
                             **{k: v for k, v in (result or {}).items()
                                if k != "error"}}
            print(f"[vec] {row['set_id']:<45} UNBOUND ({detail})", flush=True)

    manifests["vector_binding"] = {
        "producer": "aw ops bind-assembly-vector-rollups",
        "bound": bound,
        "unbound": len(unbound),
        "unbound_detail": sorted(unbound),
    }
    args.out.write_text(json.dumps(manifests, indent=2, sort_keys=True) + "\n")
    from cytir.ops._provenance import write_provenance
    write_provenance(args.out, "bind-assembly-vector-rollups", {"manifests_in": args.source, "census_run": args.census_run})
    print(f"[vec] bound={bound} unbound={len(unbound)} -> {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main() or 0)
