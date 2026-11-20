"""Route-C per-pack count declarations and the shared audit helpers."""
import gzip
import json
from pathlib import Path

import pytest

from cytir.extract.routec import ietf_datatracker_reviews as reviews
from cytir.extract.routec import ietf_meeting_records as meetings
from cytir.extract.routec.declaration import SCHEMA, load_declaration
from cytir.sources import audit as shared
from cytir.sources.routec import audit_profile

DATA = Path(__file__).resolve().parents[1] / "data" / "routec-pack-declarations"

# The counts the materializers and audits pinned in code before they became data.
PINNED_BEFORE = {
    "routec-ietf-datatracker-review-records-current": {
        "route_urls": 613, "route_reported_production_points": 1629},
    "routec-ietf-datatracker-internet-draft-records-current": {
        "route_urls": 1325, "route_reported_production_points": 18140,
        "source_parents": 1318, "source_parent_reported_production_points": 17680,
        "redirected_current_rfc_records": 7,
        "redirected_current_rfc_reported_production_points": 460},
    "routec-iana-official-registry-records-current": {
        "route_urls": 190, "direct_responses": 177, "cc_responses": 13},
    **{meetings.source_id(kind): {"route_urls": urls, "route_reported_production_points": pts}
       for kind, (urls, pts) in {
           "agenda": (163, 194), "minutes": (152, 1257), "charter": (58, 228),
           "bof_request": (43, 227), "narrative_minutes": (6, 62),
           "conflict_review": (11, 11)}.items()},
}


def _write(path: Path, source_id: str, counts: dict) -> Path:
    path.write_text(json.dumps({"schema": SCHEMA, "source_id": source_id, "counts": counts}))
    return path


@pytest.mark.parametrize("source_id", sorted(PINNED_BEFORE))
def test_shipped_declaration_equals_the_counts_once_pinned_in_code(source_id: str) -> None:
    keys = tuple(PINNED_BEFORE[source_id])
    assert load_declaration(DATA / f"{source_id}.v1.json", source_id=source_id,
                            keys=keys) == PINNED_BEFORE[source_id]


def test_every_shipped_declaration_is_accounted_for() -> None:
    assert {p.name.removesuffix(".v1.json") for p in DATA.glob("*.json")} == set(PINNED_BEFORE)


def test_declaration_refuses_instead_of_guessing(tmp_path: Path) -> None:
    keys = ("route_urls",)
    with pytest.raises(ValueError, match="required"):
        load_declaration(None, source_id="s", keys=keys)
    with pytest.raises(ValueError, match="unreadable"):
        load_declaration(tmp_path / "absent.json", source_id="s", keys=keys)
    other = _write(tmp_path / "o.json", "other", {"route_urls": 1})
    with pytest.raises(ValueError, match="not 's'"):
        load_declaration(other, source_id="s", keys=keys)
    short = _write(tmp_path / "m.json", "s", {})
    with pytest.raises(ValueError, match="missing"):
        load_declaration(short, source_id="s", keys=keys)
    for bad in (True, -1, "3"):
        path = _write(tmp_path / "b.json", "s", {"route_urls": bad})
        with pytest.raises(ValueError, match="invalid"):
            load_declaration(path, source_id="s", keys=keys)
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"schema": "x", "source_id": "s", "counts": {}}))
    with pytest.raises(ValueError, match="wrong schema"):
        load_declaration(path, source_id="s", keys=keys)


def _resolution(tmp_path: Path) -> Path:
    path = tmp_path / "resolution.jsonl.gz"
    row = {"url": "https://datatracker.ietf.org/doc/review-x",
           "disposition": reviews.DISPOSITION, "reported_production_points": 2}
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    return path


def test_materializer_selector_follows_the_declaration(tmp_path: Path) -> None:
    kwargs = dict(artifact_root=tmp_path, routes_path=tmp_path / "routes",
                  resolution_path=_resolution(tmp_path), output_root=tmp_path / "out")
    with pytest.raises(ValueError, match="declaration required"):
        reviews.materialize(**kwargs, declaration=None)
    drift = _write(tmp_path / "d.json", reviews.SOURCE_ID,
                   {"route_urls": 2, "route_reported_production_points": 2})
    with pytest.raises(ValueError, match="selector drift"):
        reviews.materialize(**kwargs, declaration=drift)
    right = _write(tmp_path / "r.json", reviews.SOURCE_ID,
                   {"route_urls": 1, "route_reported_production_points": 2})
    with pytest.raises(Exception) as caught:  # past the selector gate, no routes file
        reviews.materialize(**kwargs, declaration=right)
    assert "selector drift" not in str(caught.value)


def test_audit_refuses_without_declaration_and_on_run_count_drift(tmp_path: Path) -> None:
    paths = dict(artifact_root=tmp_path, parent_run_path=tmp_path / "run.json",
                 pack_root=tmp_path, registry_path=tmp_path / "reg.json",
                 output_root=tmp_path / "out")
    with pytest.raises(ValueError, match="declaration required"):
        audit_profile.audit_reviews(**paths, declaration=None)
    run = {"schema": "aw.routec-ietf-datatracker-review-record-run.v1",
           "status": "complete", "source_id": reviews.SOURCE_ID,
           "source_profile": "structured_data_feed",
           "point_identity": {"ledger_subtraction_authorized": False},
           "counts": {"route_urls": 613, "route_reported_production_points": 1629,
                      "source_parents": 613, "direct_responses": 613}}
    paths["parent_run_path"].write_text(json.dumps(run))
    shipped = DATA / f"{reviews.SOURCE_ID}.v1.json"
    # run agrees with the shipped declaration: it passes the RUN gate (and fails later)
    with pytest.raises(Exception) as caught:
        audit_profile.audit_reviews(**paths, declaration=shipped)
    assert "RUN conservation" not in str(caught.value)
    smaller = _write(tmp_path / "s.json", reviews.SOURCE_ID,
                     {"route_urls": 612, "route_reported_production_points": 1629})
    with pytest.raises(ValueError, match="RUN conservation failed"):
        audit_profile.audit_reviews(**paths, declaration=smaller)


def test_shared_audit_helpers(tmp_path: Path) -> None:
    assert shared.contiguous([[1, 0, 2], []]) and not shared.contiguous([[0, 2]])
    ready = {"status": "ready_for_gpu_queue", "source_profile": "p",
             "counts": {"documents": 3}, "token_distribution": {"over_hard_max": 0, "max": 2560}}
    assert shared.pack_gate(ready, profile="p", documents=3, label="X")["max"] == 2560
    ready["token_distribution"]["max"] = 2561
    with pytest.raises(ValueError, match="X PACK gate failed"):
        shared.pack_gate(ready, profile="p", documents=3, label="X")
    profile = {"required_stages": ["original"], "original": {"requirements": ["a", "b"]}}
    with pytest.raises(ValueError, match="method count mismatch: original"):
        shared.profile_checks(profile, {"original": ["only one"]}, label="X")
    checks = shared.profile_checks(profile, {"original": ["m1", "m2"]}, label="X")
    assert [c["status"] for c in checks["original"]] == ["PASS", "PASS"]
    shared.publish_evidence(tmp_path, {"a": 1}, checks, label="X")
    shared.publish_evidence(tmp_path, {"a": 1}, checks, label="X")  # same bytes: idempotent
    with pytest.raises(ValueError, match="immutable X audit collision"):
        shared.publish_evidence(tmp_path, {"a": 2}, checks, label="X")


def _parent() -> dict:
    return {"document_id": "d", "source_record_id": "r", "url": "u",
            "markdown_sha256": "m", "parent_identity": {"i": 1}, "source_version": {"v": 1}}


def _link(**over) -> dict:
    row = {"document_id": "d", "schema": "aw.parent-source-link.v1",
           "source_record_id": "r", "source_url": "u", "parent_markdown_sha256": "m",
           "parent_identity": {"i": 1}, "source_version": {"v": 1}, "ordinal": 0,
           "is_empty": False, "target_url": "t", "relation": "x", "anchor_or_context": "c"}
    return {**row, **over}


def test_shared_outlink_and_chunk_checks() -> None:
    parents = {"d": _parent()}
    linked, empty = shared.check_outlinks([_link()], parents, label="X",
                                          bind_source_url=True, exclusive=True)
    assert linked == {"d"} and empty == set()
    with pytest.raises(ValueError, match="lost parent identity"):
        shared.check_outlinks([_link(source_url="other")], parents, label="X",
                              bind_source_url=True, exclusive=True)
    shared.check_outlinks([_link(source_url="other")], parents, label="X",
                          bind_source_url=False, exclusive=False)
    mixed = [_link(), _link(ordinal=1, is_empty=True, target_url=None, relation=None,
                            anchor_or_context=None)]
    with pytest.raises(ValueError, match="conservation failed"):
        shared.check_outlinks(mixed, parents, label="X", bind_source_url=True, exclusive=True)
    shared.check_outlinks(mixed, parents, label="X", bind_source_url=True, exclusive=False)
    chunk = {"point_id": "6f0a7c3e-0000-4000-8000-000000000001", "document_id": "d",
             "source_profile": "p", "url": "u", "source_record_id": "r",
             "parent_markdown_sha256": "m", "parent_identity": {"i": 1}, "chunk_ordinal": 0}
    assert shared.check_chunks([chunk], parents, profile="p", label="X", bind_url=True) \
        == {chunk["point_id"]}
    with pytest.raises(ValueError, match="chunk ordinal conservation failed"):
        shared.check_chunks([{**chunk, "chunk_ordinal": 1}], parents, profile="p",
                            label="X", bind_url=True)
