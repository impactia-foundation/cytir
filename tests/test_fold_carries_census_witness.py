#!/usr/bin/env python3
"""A fragment must not silently drop the census witness its index entry carried.

The fold replaces an index entry wholesale with the normalized fragment. That
is right for everything a fragment is authoritative about -- rows, mode,
delete set -- but the census witness is bound elsewhere: by `--fresh-witness`
on the run, or by `produce_assembly_artifacts.py` for the sealed families the
census consumed at v31.

Task 513 hit this while binding row IDs onto the 13 payload-route sets. Those
sets already carry a v31 census witness; giving one a fragment so G5/G7 can
state a row split would, without this carry-forward, have dropped the witness
and turned guard G8 (census witness parity) from PASS to FAIL -- the "an
incomplete fragment is worse than no fragment" trap task 511 §5 named, arriving
by a route nobody had looked at.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
import uuid
from pathlib import Path


def _load(name: str):
    import importlib
    return importlib.import_module("cytir.ops." + name) if name in _MOVED else _load_here(name)


_MOVED = {"fold_assembly_claim_fragments"}


def _load_here(name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FOLD = _load("fold_assembly_claim_fragments")

WITNESS = {"uri": "artifact://w/points.jsonl.gz", "sha256": "0" * 64,
           "records": 128990, "shape": "membership/points.jsonl.gz",
           "scope": "all_claims", "consumed_by_census": "r0-remainder-census-v31"}


def _ids(n, offset=0):
    return [str(uuid.UUID(int=i + offset)) for i in range(n)]


def _write(path: Path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    import hashlib
    with path.open("wb") as fh:
        with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
            for p in points:
                gz.write((json.dumps({"point_id": p}) + "\n").encode())
    b = path.read_bytes()
    return {"uri": path.name, "sha256": hashlib.sha256(b).hexdigest(),
            "bytes": len(b), "count": len(points)}


def _run(tmp_path, monkeypatch, index_sets):
    root = tmp_path / "corpus"
    frags = tmp_path / "frags"
    frags.mkdir(parents=True, exist_ok=True)
    new = _write(root / "new.jsonl.gz", _ids(3, 900))
    old = _write(root / "old.jsonl.gz", _ids(3, 500))
    dele = _write(root / "del.jsonl.gz", _ids(3, 500))
    t427 = _write(root / "t427.jsonl.gz", _ids(2, 700))
    tomb = _write(root / "tomb.jsonl.gz", _ids(2, 800))
    (frags / "claim-manifest-a_family.v1.json").write_text(json.dumps({
        "schema": "aw.assembly-claim-manifest.v1", "set_id": "a_family",
        "mode": "replacement", "contributes_vectors": True,
        "claims": {**new, "granularity": "point_id", "verified": True},
        "vector": {"rows": 3, "dim": 1024, "dtype": "float16"},
        "old_point_ids_superseded": old,
        "delete_set": dele,
    }))
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"schema": "aw.assembly-claim-manifests.v1",
                                 "sets": index_sets}))
    out, report = tmp_path / "out.json", tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "fold", "--corpus-root", str(root), "--index", str(index),
        "--fragments", str(frags), "--out", str(out), "--report", str(report),
        "--task427-claims", t427["uri"],
        "--task427-claims-sha256", t427["sha256"],
        "--tombstone-claims", tomb["uri"]])
    assert FOLD.main() == 0
    folded = json.loads(out.read_text())
    return {row["set_id"]: row for row in folded["sets"]}


def test_a_fragment_inherits_the_index_entrys_census_witness(tmp_path, monkeypatch):
    sets = _run(tmp_path, monkeypatch, [
        {"set_id": "a_family", "schema": "aw.assembly-claim-manifest.v1",
         "census_witness": dict(WITNESS)}])
    assert sets["a_family"]["census_witness"] == WITNESS, \
        "the fold dropped a census witness the index already carried"


def test_a_set_with_no_prior_witness_still_has_none(tmp_path, monkeypatch):
    sets = _run(tmp_path, monkeypatch, [
        {"set_id": "a_family", "schema": "aw.assembly-claim-manifest.v1"}])
    assert sets["a_family"].get("census_witness") is None


def test_the_carry_forward_never_invents_a_witness_for_a_new_set(tmp_path, monkeypatch):
    sets = _run(tmp_path, monkeypatch, [])
    assert sets["a_family"].get("census_witness") is None


def test_an_in_place_reembed_that_states_its_superseded_set_keeps_the_full_claim(
        tmp_path, monkeypatch):
    """wikipedia_413a claims 19,227,117 and stages 19,227,106 -- the projection
    excluded 11. Taking claims := new ids would shrink the claim by exactly the
    points that are NOT carried, so they would be neither loaded nor deleted:
    the quiet-drop shape task 513 found for service_public_dila_exact at a scale
    of 35,351, here at a scale of 11."""
    root = tmp_path / "corpus"
    frags = tmp_path / "frags"
    frags.mkdir(parents=True, exist_ok=True)
    old_ids = _ids(5, 100)             # the set claims 5 production points
    new_ids = old_ids[:4]              # 4 are carried, in place
    new = _write(root / "new.jsonl.gz", new_ids)
    old = _write(root / "old.jsonl.gz", old_ids)
    coll = _write(root / "coll.jsonl.gz", new_ids)
    dele = _write(root / "del.jsonl.gz", old_ids[4:])   # the 1 not carried
    t427 = _write(root / "t427.jsonl.gz", _ids(2, 700))
    tomb = _write(root / "tomb.jsonl.gz", _ids(2, 800))
    (frags / "claim-manifest-inplace.v1.json").write_text(json.dumps({
        "schema": "aw.assembly-claim-manifest.v1", "set_id": "an_inplace_set",
        "mode": "in_place_reembed", "contributes_vectors": True,
        "claims": {**new, "granularity": "point_id", "verified": True},
        "vector": {"rows": 4, "dim": 1024, "dtype": "float16"},
        "old_point_ids_superseded": old,
        "identity_collapse": coll,
        "delete_set": dele,
    }))
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"schema": "aw.assembly-claim-manifests.v1",
                                 "sets": []}))
    out, report = tmp_path / "out.json", tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "fold", "--corpus-root", str(root), "--index", str(index),
        "--fragments", str(frags), "--out", str(out), "--report", str(report),
        "--task427-claims", t427["uri"],
        "--task427-claims-sha256", t427["sha256"],
        "--tombstone-claims", tomb["uri"]])
    assert FOLD.main() == 0
    rep = json.loads(report.read_text())
    cons = rep["conservation"]["an_inplace_set"]
    assert cons["superseded"] == 5, "the claim shrank to the carried rows"
    assert cons["delete_set"] == 1 and cons["identity_collapse"] == 4
    assert cons["conserves"] is True


def test_an_in_place_reembed_with_no_superseded_statement_is_unchanged(
        tmp_path, monkeypatch):
    """The fallback must stay exactly what it was for the sets that rely on it."""
    root = tmp_path / "corpus"
    frags = tmp_path / "frags"
    frags.mkdir(parents=True, exist_ok=True)
    ids = _ids(4, 100)
    new = _write(root / "new.jsonl.gz", ids)
    coll = _write(root / "coll.jsonl.gz", ids)
    t427 = _write(root / "t427.jsonl.gz", _ids(2, 700))
    tomb = _write(root / "tomb.jsonl.gz", _ids(2, 800))
    (frags / "claim-manifest-inplace2.v1.json").write_text(json.dumps({
        "schema": "aw.assembly-claim-manifest.v1", "set_id": "plain_inplace",
        "mode": "in_place_reembed", "contributes_vectors": True,
        "claims": {**new, "granularity": "point_id", "verified": True},
        "vector": {"rows": 4, "dim": 1024, "dtype": "float16"},
        "identity_collapse": coll,
    }))
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"schema": "aw.assembly-claim-manifests.v1",
                                 "sets": []}))
    out, report = tmp_path / "out.json", tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "fold", "--corpus-root", str(root), "--index", str(index),
        "--fragments", str(frags), "--out", str(out), "--report", str(report),
        "--task427-claims", t427["uri"],
        "--task427-claims-sha256", t427["sha256"],
        "--tombstone-claims", tomb["uri"]])
    assert FOLD.main() == 0
    rep = json.loads(report.read_text())
    cons = rep["conservation"]["plain_inplace"]
    assert cons["superseded"] == 4 and cons["conserves"] is True
