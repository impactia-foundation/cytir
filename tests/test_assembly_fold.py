"""Unit tests for the fold and the ADR-101a corrections.

The fold is where a producer's own vocabulary meets the ADR-101 contract, and
where a mis-declared mode turns into a silent delete.  Every normalization it
performs, and every defect it refuses, has a fixture here.
"""
from __future__ import annotations

import gzip
import importlib.util
import io
import json
import sys
import uuid
from pathlib import Path

import pytest


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


def _ids(n: int, offset: int = 0) -> list[str]:
    return [str(uuid.UUID(int=index + offset)) for index in range(n)]


def _write_ids(path: Path, points: list[str], bare: bool = False) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8") as target:
                for point in points:
                    target.write(
                        (json.dumps(point) if bare
                         else json.dumps({"point_id": point})) + "\n")
    return {"uri": path.name, "sha256": FOLD.sha256_file(path),
            "count": len(points)}


# ---------------------------------------------------------------------------
# reading point-ID artifacts
# ---------------------------------------------------------------------------


def test_read_ids_accepts_both_artifact_shapes(tmp_path):
    """Producers write bare strings; my own stages write {"point_id": ...}."""
    bare = _write_ids(tmp_path / "bare.jsonl.gz", _ids(5), bare=True)
    wrapped = _write_ids(tmp_path / "wrapped.jsonl.gz", _ids(5))
    assert FOLD.read_ids(tmp_path, bare, "bare") == \
           FOLD.read_ids(tmp_path, wrapped, "wrapped")


def test_read_ids_refuses_digest_drift(tmp_path):
    binding = _write_ids(tmp_path / "x.jsonl.gz", _ids(3))
    binding["sha256"] = "0" * 64
    with pytest.raises(FOLD.ArtifactDefect, match="digest drift"):
        FOLD.read_ids(tmp_path, binding, "x")


def test_read_ids_refuses_a_count_that_disagrees(tmp_path):
    binding = _write_ids(tmp_path / "x.jsonl.gz", _ids(3))
    binding["count"] = 4
    # the wording moved from "fragment declares" to "the binding declares" when
    # the reader became shared (task 513b): it now serves fragments AND claim
    # manifests, and "binding" is what both have. The refusal is the assertion.
    with pytest.raises(FOLD.ArtifactDefect, match=r"3 IDs read.*declares 4"):
        FOLD.read_ids(tmp_path, binding, "x")


def test_a_defect_is_raised_not_exited_so_the_fold_can_quarantine(tmp_path):
    """Exiting on the first bad artifact reports one defect per run, which is
    the slowest possible way to find four."""
    assert issubclass(FOLD.ArtifactDefect, Exception)
    assert not issubclass(FOLD.ArtifactDefect, SystemExit)


def test_read_ids_refuses_a_duplicate_point_id(tmp_path):
    binding = _write_ids(tmp_path / "x.jsonl.gz", _ids(3) + _ids(1))
    binding.pop("count")
    binding.pop("sha256")
    with pytest.raises(FOLD.ArtifactDefect, match="duplicate point ID"):
        FOLD.read_ids(tmp_path, binding, "x")


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_load_fragment_refuses_unparseable_json(tmp_path):
    """The task-467 defect: adjacent string literals, which JSON has no rule for."""
    path = tmp_path / "claim-manifest-broken.v1.json"
    path.write_text('{"schema": "aw.assembly-claim-manifest.v1",\n'
                    ' "set_id": "x"\n "y"}\n')
    with pytest.raises(SystemExit, match="not valid JSON"):
        FOLD.load_fragment(path)


def test_load_fragment_refuses_a_missing_contract_field(tmp_path):
    path = tmp_path / "claim-manifest-thin.v1.json"
    path.write_text(json.dumps({
        "schema": "aw.assembly-claim-manifest.v1", "set_id": "x",
        "mode": "replacement", "contributes_vectors": True}))
    with pytest.raises(SystemExit, match="omits"):
        FOLD.load_fragment(path)


def test_load_fragment_refuses_the_wrong_schema(tmp_path):
    path = tmp_path / "claim-manifest-alien.v1.json"
    path.write_text(json.dumps({"schema": "something.else", "set_id": "x",
                                "mode": "replacement",
                                "contributes_vectors": True,
                                "claims": {}, "vector": {}}))
    with pytest.raises(SystemExit, match="wrong schema"):
        FOLD.load_fragment(path)


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------


def _fragment(**overrides) -> dict:
    fragment = {
        "schema": "aw.assembly-claim-manifest.v1",
        "set_id": "task472_example",
        "mode": "replacement",
        "contributes_vectors": True,
        "claims": {"granularity": "point_id", "count": 100,
                   "uri": "artifact://new.jsonl.gz", "sha256": "a" * 64},
        "vector": {"rows": 100, "dim": 1024, "dtype": "float16"},
        "identity_collapse": {"count": 10, "uri": "artifact://collapse.jsonl.gz"},
        "delete_set": {"count": 90, "uri": "artifact://delete.jsonl.gz"},
        "old_point_ids_superseded": {"count": 100,
                                     "uri": "artifact://old.jsonl.gz"},
    }
    fragment.update(overrides)
    return fragment


def test_normalize_moves_new_ids_out_of_the_claims_field():
    """`claims` means OLD points in ADR-101 and NEW points in the fragments."""
    manifest, note = FOLD.normalize(_fragment(), "frag.json")
    assert manifest["claims"]["uri"] == "artifact://old.jsonl.gz"
    assert manifest["new_point_ids"]["uri"] == "artifact://new.jsonl.gz"
    assert any("NEW point IDs" in line for line in note["notes"])


def test_normalize_corrects_a_mis_declared_in_place_reembed():
    """The task-467 defect: every ID collapses, nothing is deleted, mode says
    'replacement'. An applier that believes it may delete 339,307 card rows."""
    fragment = _fragment(set_id="task467_cards", mode="replacement",
                         identity_collapse={"count": 100,
                                            "uri": "artifact://c.jsonl.gz"})
    fragment.pop("delete_set")
    fragment.pop("old_point_ids_superseded")
    manifest, note = FOLD.normalize(fragment, "frag.json")
    assert manifest["mode"] == "in_place_reembed"
    assert manifest["normalization"]["declared_mode"] == "replacement"
    assert any("mode corrected" in line for line in note["notes"])
    # an in-place re-embed's claims ARE its row IDs -- one and the same value
    assert manifest["claims"]["uri"] == manifest["new_point_ids"]["uri"]


def test_normalize_leaves_an_honest_in_place_reembed_alone():
    fragment = _fragment(set_id="rii", mode="in_place_reembed",
                         identity_collapse={"count": 100,
                                            "uri": "artifact://new.jsonl.gz"},
                         delete_set={"count": 0})
    manifest, _ = FOLD.normalize(fragment, "frag.json")
    assert manifest["mode"] == "in_place_reembed"
    assert manifest["normalization"]["declared_mode"] == "in_place_reembed"


def test_normalize_does_not_reclassify_a_real_replacement():
    """Collapse < rows and a non-empty delete set is a genuine replacement."""
    manifest, _ = FOLD.normalize(_fragment(), "frag.json")
    assert manifest["mode"] == "replacement"


def test_normalize_flags_a_fragment_with_no_superseded_artifact():
    fragment = _fragment()
    fragment.pop("old_point_ids_superseded")
    manifest, note = FOLD.normalize(fragment, "frag.json")
    assert "claims" not in manifest
    assert any("reconstructed" in line for line in note["notes"])


# ---------------------------------------------------------------------------
# normalization -- the net-new-addition branch
# ---------------------------------------------------------------------------


def _addition(**overrides) -> dict:
    """A well-formed net-new addition, on task 500's shape."""
    fragment = {
        "schema": "aw.assembly-claim-manifest.v1",
        "set_id": "task500_news_1024_add",
        "lane": "flip",
        "mode": "net_new_addition",
        "contributes_vectors": True,
        "contributes_production_points": False,
        "claims": {"granularity": "point_id", "count": 17959,
                   "claims_are_production_points": False,
                   "uri": "claimed-point-ids.jsonl.gz", "sha256": "b" * 64,
                   "verified": True},
        "membership_basis": {"receipt_sha256": "c" * 64, "rows": 34_464_401},
        "vector": {"rows": 17959, "dim": 1024, "dtype": "float16"},
    }
    fragment.update(overrides)
    return fragment


def test_normalize_keeps_an_additions_claim_instead_of_dropping_it():
    """The task-500 defect: an addition fell through to the reconstruct case,
    `claims` was rebuilt as the empty union, and G1 then failed the set for
    publishing no claim -- while its 17,959 enumerated IDs sat on disk."""
    manifest, note = FOLD.normalize(_addition(), "frag.json")
    assert manifest["mode"] == "net_new_addition"
    assert manifest["new_point_ids"]["uri"] == "claimed-point-ids.jsonl.gz"
    assert manifest["additions"]["count"] == 17959
    assert manifest["additions"]["rows"] == 17959
    assert manifest["claims"]["count"] == 0
    assert manifest["claims"]["claims_are_production_points"] is False
    assert manifest["contributes_production_points"] is False
    assert manifest["delete_set"] is None
    assert manifest["identity_collapse"] is None
    assert any("net-new addition" in line for line in note["notes"])
    assert not any("claims must be reconstructed" in line
                   for line in note["notes"])


def test_normalize_carries_an_additions_membership_basis_through():
    """G7 ties the disjointness proof to the membership the plan builds
    against, so the basis has to survive the fold."""
    manifest, _ = FOLD.normalize(_addition(), "frag.json")
    assert manifest["membership_basis"]["receipt_sha256"] == "c" * 64


@pytest.mark.parametrize("field,value", [
    ("identity_collapse", {"count": 5, "uri": "artifact://c.jsonl.gz"}),
    ("delete_set", {"count": 5, "uri": "artifact://d.jsonl.gz"}),
    ("old_point_ids_superseded", {"count": 5, "uri": "artifact://o.jsonl.gz"}),
])
def test_normalize_refuses_an_addition_that_binds_old_points(field, value):
    """A replacement wearing the wrong label. The difference between the two
    is whether the applier deletes live production points."""
    with pytest.raises(SystemExit, match="supersede, collapse or delete"):
        FOLD.normalize(_addition(**{field: value}), "frag.json")


def test_normalize_accepts_an_addition_that_states_its_delete_set_is_empty():
    """`{"count": 0}` says the same thing as omitting the field; refusing it
    would make the honest producer the one that fails."""
    manifest, _ = FOLD.normalize(
        _addition(delete_set={"count": 0}), "frag.json")
    assert manifest["mode"] == "net_new_addition"


def test_normalize_refuses_an_addition_that_does_not_disclaim_production_points():
    with pytest.raises(SystemExit,
                       match="contributes_production_points: false"):
        fragment = _addition()
        fragment.pop("contributes_production_points")
        FOLD.normalize(fragment, "frag.json")


def test_normalize_refuses_an_addition_whose_claim_does_not_say_which_it_means():
    """`claims` means NEW IDs in the fragments and OLD ones in ADR-101. For an
    addition the two readings differ by the entire membership."""
    fragment = _addition()
    fragment["claims"].pop("claims_are_production_points")
    with pytest.raises(SystemExit, match="claims_are_production_points"):
        FOLD.normalize(fragment, "frag.json")


@pytest.mark.parametrize("field", ["uri", "sha256"])
def test_normalize_refuses_an_addition_that_does_not_pin_its_row_ids(field):
    fragment = _addition()
    fragment["claims"].pop(field)
    with pytest.raises(SystemExit, match=f"pins no {field}"):
        FOLD.normalize(fragment, "frag.json")


def test_normalize_refuses_an_empty_addition():
    fragment = _addition()
    fragment["claims"]["count"] = 0
    with pytest.raises(SystemExit, match="zero row IDs"):
        FOLD.normalize(fragment, "frag.json")


def test_the_in_place_correction_never_captures_an_addition():
    """The mode-corrector fires when every new ID collapses onto an old one.
    An addition collapses none, and must not be reclassified into a mode whose
    row IDs the applier believes are already in the collection."""
    manifest, _ = FOLD.normalize(_addition(), "frag.json")
    assert manifest["mode"] == "net_new_addition"
    assert manifest["normalization"]["declared_mode"] == "net_new_addition"


def test_asserts_old_points_reads_an_explicit_zero_as_no_assertion():
    assert FOLD.asserts_old_points(None) is False
    assert FOLD.asserts_old_points({}) is False
    assert FOLD.asserts_old_points({"count": 0}) is False
    assert FOLD.asserts_old_points({"count": 1}) is True
    assert FOLD.asserts_old_points({"uri": "x.jsonl.gz"}) is True


# ---------------------------------------------------------------------------
# the fold end to end -- what it MEASURES about an addition
# ---------------------------------------------------------------------------


def _fold_run(tmp_path: Path, minted: list[str], *, declared_rows: int = 3,
              other_claims: list[str] | None = None,
              tombstones: list[str] | None = None,
              remainder: list[str] | None = None,
              monkeypatch=None) -> dict:
    """Drive FOLD.main() over one addition and one replacement, and return the
    fold report.  Everything the addition checks intersect against is real."""
    root = tmp_path / "corpus"
    frags = tmp_path / "fragments"
    frags.mkdir(parents=True, exist_ok=True)

    add_ids = _write_ids(root / "add.jsonl.gz", minted)
    other_claims = other_claims if other_claims is not None else _ids(2, 500)
    other_new = _write_ids(root / "other-new.jsonl.gz", _ids(2, 900))
    other_old = _write_ids(root / "other-old.jsonl.gz", other_claims)
    other_del = _write_ids(root / "other-del.jsonl.gz", other_claims)
    t427 = _write_ids(root / "t427.jsonl.gz", _ids(2, 700))
    tomb = _write_ids(root / "tomb.jsonl.gz",
                      tombstones if tombstones is not None else _ids(2, 800))

    (frags / "claim-manifest-addition.v1.json").write_text(json.dumps(
        _addition(set_id="an_addition",
                  claims={**add_ids, "granularity": "point_id",
                          "claims_are_production_points": False,
                          "verified": True},
                  vector={"rows": declared_rows, "dim": 1024,
                          "dtype": "float16"})))
    (frags / "claim-manifest-other.v1.json").write_text(json.dumps({
        "schema": "aw.assembly-claim-manifest.v1", "set_id": "a_replacement",
        "mode": "replacement", "contributes_vectors": True,
        "claims": {**other_new, "granularity": "point_id", "verified": True},
        "vector": {"rows": 2, "dim": 1024, "dtype": "float16"},
        "old_point_ids_superseded": other_old,
        "delete_set": other_del,
    }))

    index = tmp_path / "index.json"
    index.write_text(json.dumps({
        "schema": "aw.assembly-claim-manifests.v1", "sets": []}))
    out, report = tmp_path / "out.json", tmp_path / "report.json"
    argv = ["fold", "--corpus-root", str(root), "--index", str(index),
            "--fragments", str(frags), "--out", str(out), "--report",
            str(report),
            "--task427-claims", t427["uri"],
            "--task427-claims-sha256", t427["sha256"],
            "--tombstone-claims", tomb["uri"]]
    if remainder is not None:
        rem = _write_ids(root / "remainder.jsonl.gz", remainder)
        argv += ["--remainder", rem["uri"]]
    monkeypatch.setattr(sys, "argv", argv)
    assert FOLD.main() == 0
    return json.loads(report.read_text())


def test_the_fold_counts_an_additions_minted_ids_on_disk(tmp_path, monkeypatch):
    report = _fold_run(tmp_path, _ids(3, 100), monkeypatch=monkeypatch)
    conservation = report["conservation"]["an_addition"]
    assert conservation["conserves"] is True
    assert conservation["addition_vector_rows"] == 3
    assert conservation["addition_minted_ids"] == 3
    assert conservation["superseded"] == 0
    assert report["additions"]["an_addition"]["rows"] == 3
    assert report["clean"] is True


def test_the_fold_refuses_an_addition_whose_vectors_and_ids_disagree(
        tmp_path, monkeypatch):
    """Two of the three numbers agreeing is what a missing shard looks like."""
    report = _fold_run(tmp_path, _ids(3, 100), declared_rows=4,
                       monkeypatch=monkeypatch)
    assert report["conservation"]["an_addition"]["conserves"] is False
    assert any("rows == minted IDs == vectors" in line
               for line in report["findings"])


def test_the_fold_catches_a_minted_id_that_is_another_sets_old_point(
        tmp_path, monkeypatch):
    """The ID is uuid5(url # digest): a collision means same URL, same content
    -- an in-place re-embed that called itself an addition, and the applier
    would upsert over a live vector nothing subtracted."""
    shared = _ids(1, 100)
    report = _fold_run(tmp_path, _ids(3, 100), other_claims=shared + _ids(1, 501),
                       monkeypatch=monkeypatch)
    hits = report["additions"]["an_addition"][
        "minted_ids_in_another_sets_claims"]
    assert hits == {"a_replacement": 1}
    assert any("must belong to no production point" in line
               for line in report["findings"])
    assert report["clean"] is False


def test_the_fold_catches_a_minted_id_that_is_tombstoned(tmp_path, monkeypatch):
    report = _fold_run(tmp_path, _ids(3, 100), tombstones=_ids(1, 102),
                       monkeypatch=monkeypatch)
    assert report["additions"]["an_addition"]["minted_ids_tombstoned"] == 1
    assert any("may not resurrect" in line for line in report["findings"])


def test_the_fold_catches_a_minted_id_inside_the_census_remainder(
        tmp_path, monkeypatch):
    """The remainder is a subset of the membership, so a hit here is proof the
    set is not net-new -- without needing the 34.4M-point join."""
    report = _fold_run(tmp_path, _ids(3, 100), remainder=_ids(1, 101),
                       monkeypatch=monkeypatch)
    assert report["additions"]["an_addition"][
        "minted_ids_in_census_remainder"] == 1
    assert any("not a net-new addition" in line for line in report["findings"])


def test_the_fold_reports_a_clean_addition_against_the_remainder(
        tmp_path, monkeypatch):
    report = _fold_run(tmp_path, _ids(3, 100), remainder=_ids(2, 600),
                       monkeypatch=monkeypatch)
    assert report["additions"]["an_addition"][
        "minted_ids_in_census_remainder"] == 0
    assert report["clean"] is True


# ---------------------------------------------------------------------------
# The lane gate.  Track 2's expansion fragments live in the same directory and
# match the same glob, and one of them (task 488's) pins no new_point_ids
# artifact at all -- so before this gate the fold died on a KeyError partway
# through, having already adjudicated half the flip lane.
# ---------------------------------------------------------------------------

def _lane_fragment(tmp_path: Path, name: str, set_id: str, **over) -> Path:
    body = {"schema": FOLD.SCHEMA_MANIFEST, "set_id": set_id,
            "mode": "replacement", "contributes_vectors": True,
            "claims": {"count": 1}, "vector": {"rows": 1}}
    body.update(over)
    path = tmp_path / name
    path.write_text(json.dumps(body, indent=1))
    return path


def test_lane_gate_keeps_flip_fragments_and_excludes_expansion(tmp_path):
    flip = _lane_fragment(tmp_path, "claim-manifest-a.v1.json", "set_a")
    expansion = _lane_fragment(tmp_path, "claim-manifest-b.v1.json", "set_b",
                               lane="expansion", lane_note="Track 2, not flip")

    in_lane, out_of_lane = FOLD.partition_by_lane([flip, expansion])

    assert in_lane == [flip]
    assert out_of_lane == [{"fragment": "claim-manifest-b.v1.json",
                            "set_id": "set_b", "lane": "expansion",
                            "lane_note": "Track 2, not flip"}]


def test_a_fragment_with_no_lane_field_is_the_flip_lane(tmp_path):
    """Every fragment sealed before the field existed must still fold."""
    legacy = _lane_fragment(tmp_path, "claim-manifest-a.v1.json", "set_a")
    assert "lane" not in json.loads(legacy.read_text())
    in_lane, out_of_lane = FOLD.partition_by_lane([legacy])
    assert in_lane == [legacy] and out_of_lane == []


def test_an_explicit_flip_lane_is_kept(tmp_path):
    explicit = _lane_fragment(tmp_path, "claim-manifest-a.v1.json", "set_a",
                              lane=FOLD.FLIP_LANE)
    in_lane, _ = FOLD.partition_by_lane([explicit])
    assert in_lane == [explicit]


def test_an_unknown_lane_is_refused_not_skipped(tmp_path):
    """A skip is how a real claim set goes missing with nothing to show it."""
    odd = _lane_fragment(tmp_path, "claim-manifest-a.v1.json", "set_a",
                         lane="something-new")
    with pytest.raises(SystemExit, match="does not recognise"):
        FOLD.partition_by_lane([odd])


def test_the_gate_runs_before_the_required_field_check(tmp_path):
    """Task 488's fragment omits new_point_ids; it must never be loaded."""
    expansion = _lane_fragment(tmp_path, "claim-manifest-b.v1.json", "set_b",
                               lane="expansion")
    in_lane, out_of_lane = FOLD.partition_by_lane([expansion])
    assert in_lane == []
    assert out_of_lane[0]["set_id"] == "set_b"


# ---------------------------------------------------------------------------
# contested = a UNION of point sets, never a sum of pair counts
#
# `task494_4crawl_supplement_cc_chunks` reported **points_won = -883** on the
# real corpus: it claims 2,270 old points and shares points with two other
# sets, and summing the two pair counts (3,153) counts every point both other
# sets also claim twice.  A negative winning population is not a rounding
# error -- it is the number a census witness scope is sealed against, which is
# the failure task 498 sec.7 caught in a different consumer of the same bug.
# ---------------------------------------------------------------------------

def _replacement_fragment(root: Path, set_id: str, old: list[str],
                          new: list[str]) -> dict:
    return {
        "schema": "aw.assembly-claim-manifest.v1", "set_id": set_id,
        "mode": "replacement", "contributes_vectors": True,
        "claims": {**_write_ids(root / f"{set_id}-new.jsonl.gz", new),
                   "granularity": "point_id", "verified": True},
        "vector": {"rows": len(new), "dim": 1024, "dtype": "float16"},
        "old_point_ids_superseded": _write_ids(
            root / f"{set_id}-old.jsonl.gz", old),
        "delete_set": _write_ids(root / f"{set_id}-del.jsonl.gz", old),
    }


def _three_way_fold(tmp_path: Path, monkeypatch) -> dict:
    """Three replacement sets, ONE old point claimed by all three."""
    root = tmp_path / "corpus"
    frags = tmp_path / "fragments"
    frags.mkdir(parents=True, exist_ok=True)
    shared = _ids(1, 10)
    for name, own, offset in (("set_a", 20, 200), ("set_b", 40, 300),
                              ("set_c", 60, 400)):
        (frags / f"claim-manifest-{name}.v1.json").write_text(json.dumps(
            _replacement_fragment(root, name, shared + _ids(2, own),
                                  _ids(2, offset))))
    t427 = _write_ids(root / "t427.jsonl.gz", _ids(2, 700))
    tomb = _write_ids(root / "tomb.jsonl.gz", _ids(2, 800))
    index = tmp_path / "index.json"
    index.write_text(json.dumps(
        {"schema": "aw.assembly-claim-manifests.v1", "sets": []}))
    out, report = tmp_path / "out.json", tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "fold", "--corpus-root", str(root), "--index", str(index),
        "--fragments", str(frags), "--out", str(out), "--report", str(report),
        "--task427-claims", t427["uri"],
        "--task427-claims-sha256", t427["sha256"],
        "--tombstone-claims", tomb["uri"]])
    assert FOLD.main() == 0
    return json.loads(out.read_text())


def test_contested_is_a_union_not_a_sum_of_pair_counts(tmp_path, monkeypatch):
    index = _three_way_fold(tmp_path, monkeypatch)
    by_id = {row["set_id"]: row for row in index["sets"]}
    for set_id in ("set_a", "set_b", "set_c"):
        adj = by_id[set_id]["adjudication"]
        # each set shares exactly ONE point, with two other sets; summing the
        # pair counts would say 2 and drive points_won from 2 down to 1
        assert adj["contested_with_another_set"] == 1, set_id
        assert adj["points_won"] == 2, set_id


def test_points_won_can_never_go_negative(tmp_path, monkeypatch):
    index = _three_way_fold(tmp_path, monkeypatch)
    for row in index["sets"]:
        adj = row.get("adjudication") or {}
        if "points_won" in adj:
            assert adj["points_won"] >= 0, row["set_id"]
            assert adj["points_won"] <= adj["claims"], row["set_id"]


# ---------------------------------------------------------------------------
# resolve_row_split -- ADR-101b/c row-level loser suppression (task 505
# sec.6 / task 511 sec.2)
# ---------------------------------------------------------------------------

def test_row_split_uncontested_loads_everything():
    out = FOLD.resolve_row_split("set_a", contested=0, rows_declared=100,
                                 split=None)
    assert out == {"rows_loaded": 100, "rows_suppressed": 0}


def test_row_split_contested_no_artifact_stays_pending():
    out = FOLD.resolve_row_split("set_a", contested=50, rows_declared=100,
                                 split=None)
    assert out["rows_pending_adjudication"] == 100
    assert "no adjudicated winner" in out["why_no_row_split"]


def test_row_split_exact_coverage_resolves():
    split = {"covered_old_points": 50, "lost_old_points": 50,
             "lost_urls": 12, "rows_declared": 100,
             "rows_suppressed": 60, "rows_loaded": 40,
             "pairs": ["ADR-101b.1a"]}
    out = FOLD.resolve_row_split("set_a", contested=50, rows_declared=100,
                                 split=split)
    assert out["rows_loaded"] == 40
    assert out["rows_suppressed"] == 60
    assert "row_split_basis" in out
    assert "rows_pending_adjudication" not in out


def test_row_split_win_and_lose_coverage_resolves_on_covered_not_lost():
    """task492's real shape: it WINS 25,022 points from task427 (no
    suppression -- those rows load normally) and LOSES 113,522 to task467
    (suppressed). `contested` is the union of both sides (138,216 in the
    real case); `lost_old_points` alone only ever counts the losing side and
    would never reach `contested` for a set that also wins somewhere --
    `covered_old_points` (win-or-lose) is the field that must match."""
    split = {"covered_old_points": 90, "lost_old_points": 60,
             "lost_urls": 15, "rows_declared": 100,
             "rows_suppressed": 40, "rows_loaded": 60,
             "pairs": ["ADR-101b.1a", "ADR-101c.2b"]}
    out = FOLD.resolve_row_split("set_a", contested=90, rows_declared=100,
                                 split=split)
    assert out["rows_loaded"] == 60
    assert out["rows_suppressed"] == 40
    assert "rows_pending_adjudication" not in out


def test_row_split_partial_coverage_stays_pending_not_a_wrong_split():
    """The set has 50 contested points; the artifact only accounts for 30 of
    them (e.g. one of two pairs is unresolved). A partial split is a wrong
    split with extra steps -- it must NOT resolve."""
    split = {"covered_old_points": 30, "lost_old_points": 30,
             "lost_urls": 8, "rows_declared": 100,
             "rows_suppressed": 35, "rows_loaded": 65,
             "pairs": ["ADR-101b.1a"]}
    out = FOLD.resolve_row_split("set_a", contested=50, rows_declared=100,
                                 split=split)
    assert out["rows_pending_adjudication"] == 100
    assert "resolves 30" in out["why_no_row_split"]


def test_row_split_over_coverage_also_stays_pending():
    """The artifact claims MORE contested points than this fold measures --
    exact match only, never >=, because a mismatch in EITHER direction means
    the artifact is stale against what this fold just re-derived."""
    split = {"covered_old_points": 80, "lost_old_points": 80,
             "lost_urls": 8, "rows_declared": 100,
             "rows_suppressed": 35, "rows_loaded": 65,
             "pairs": ["ADR-101b.1a"]}
    out = FOLD.resolve_row_split("set_a", contested=50, rows_declared=100,
                                 split=split)
    assert out["rows_pending_adjudication"] == 100


def test_row_split_refuses_a_stale_rows_declared():
    """The artifact was derived against an OLDER binder plan whose shard
    count for this set differed -- refuse rather than load a wrong split."""
    split = {"covered_old_points": 50, "lost_old_points": 50,
             "lost_urls": 12, "rows_declared": 999,
             "rows_suppressed": 60, "rows_loaded": 40,
             "pairs": ["ADR-101b.1a"]}
    with pytest.raises(FOLD.ArtifactDefect, match="stale row-suppression"):
        FOLD.resolve_row_split("set_a", contested=50, rows_declared=100,
                               split=split)


def _three_way_fold_row_suppression(tmp_path: Path, monkeypatch,
                                    row_suppression: dict) -> dict:
    """Same three-way, one-shared-point fixture as `_three_way_fold`, plus
    a `--row-suppression` artifact -- so the integration path (main() reading
    the flag, not just the pure function) is exercised too."""
    root = tmp_path / "corpus"
    frags = tmp_path / "fragments"
    frags.mkdir(parents=True, exist_ok=True)
    shared = _ids(1, 10)
    for name, own, offset in (("set_a", 20, 200), ("set_b", 40, 300),
                              ("set_c", 60, 400)):
        (frags / f"claim-manifest-{name}.v1.json").write_text(json.dumps(
            _replacement_fragment(root, name, shared + _ids(2, own),
                                  _ids(2, offset))))
    t427 = _write_ids(root / "t427.jsonl.gz", _ids(2, 700))
    tomb = _write_ids(root / "tomb.jsonl.gz", _ids(2, 800))
    index = tmp_path / "index.json"
    index.write_text(json.dumps(
        {"schema": "aw.assembly-claim-manifests.v1", "sets": []}))
    row_sup_path = tmp_path / "row-suppression.json"
    row_sup_path.write_text(json.dumps(row_suppression))
    out, report = tmp_path / "out.json", tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "fold", "--corpus-root", str(root), "--index", str(index),
        "--fragments", str(frags), "--out", str(out), "--report", str(report),
        "--task427-claims", t427["uri"],
        "--task427-claims-sha256", t427["sha256"],
        "--tombstone-claims", tomb["uri"],
        "--row-suppression", str(row_sup_path)])
    assert FOLD.main() == 0
    return json.loads(out.read_text())


def test_row_suppression_flag_resolves_only_the_set_it_covers(
        tmp_path, monkeypatch):
    """set_a's one contested point is covered end-to-end; set_b and set_c
    (same contest, no artifact entry) stay pending -- proving the flag is
    consumed per-set, not as a blanket override."""
    index = _three_way_fold_row_suppression(tmp_path, monkeypatch, {
        "ratified": True, "reason": "test fixture",
        "sets": {"set_a": {"covered_old_points": 1, "lost_old_points": 1,
                           "lost_urls": 1, "rows_declared": 2,
                           "rows_suppressed": 1, "rows_loaded": 1,
                           "pairs": ["ADR-101b.1a"]}}})
    by_id = {row["set_id"]: row for row in index["sets"]}
    a = by_id["set_a"]["adjudication"]
    assert a["rows_loaded"] == 1 and a["rows_suppressed"] == 1
    assert "rows_pending_adjudication" not in a
    for sid in ("set_b", "set_c"):
        adj = by_id[sid]["adjudication"]
        assert adj["rows_pending_adjudication"] == 2, sid


def test_unratified_row_suppression_artifact_is_ignored(tmp_path, monkeypatch):
    """`ratified: false` (an ADR-101b/c PR that has not merged) must not
    resolve anything -- exactly like G2's own gate."""
    index = _three_way_fold_row_suppression(tmp_path, monkeypatch, {
        "ratified": False, "reason": "draft marker still present", "sets": {
            "set_a": {"covered_old_points": 1, "lost_old_points": 1,
                      "lost_urls": 1, "rows_declared": 2,
                      "rows_suppressed": 1, "rows_loaded": 1,
                      "pairs": ["ADR-101b.1a"]}}})
    by_id = {row["set_id"]: row for row in index["sets"]}
    assert by_id["set_a"]["adjudication"]["rows_pending_adjudication"] == 2
