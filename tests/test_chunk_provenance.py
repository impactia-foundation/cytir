#!/usr/bin/env python3
"""Contract tests for `aw.chunk-provenance.v1` (task 407).

These are the pure-function cases where a regression would be SILENT and
expensive: an id that stops discriminating two policies, a master-only block
leaking into the compact dump, an "unknown" that quietly acquires a plausible
value. Everything else about provenance is verified against artifacts on the build
host, not here (verification-doctrine 1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cytir.chunk import provenance as CP  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "tasks" / "407-artifacts"

E3 = "cf0f5244dc9b27494bb8683e31a55392a3f9ea50db8c4ddc1de274d7ebbbd4a7"
E4 = "7dcb54c497e1855ad91ddafe80ff34b1da14f3d23b3af26dc7c7b53ab70ce6bf"


def _struct(sha: str, epoch: int, legend: str) -> dict:
    return CP.build(
        family="generic-structured",
        chunker={"mode": "structured", "chunk_epoch": epoch,
                 "rule_legend": legend, "policy_sha256": sha,
                 "chunk_tokenizer_revision": "ac6544c8" * 5},
        channels={"canonical": "hierarchy-plus-body",
                  "dense": "leaf-heading-plus-body",
                  "sparse": "hierarchy-plus-body", "channels_split": True},
        confidence="stamped")


def test_epoch3_and_epoch4_policies_get_different_ids():
    """The whole reason the epoch int is not enough.

    Tim, 2026-08-13: epoch 3 and epoch 4 differ ONLY in
    `token_hard_max.exact` (2000 -> 2560) and produce identical output on every
    unit that does not hit the backstop. If provenance keyed on the epoch alone
    these rows would be indistinguishable forever.
    """
    a = _struct(E3, 3, "struct-md-768w-o0-adjacent-sections-tail-1152@3")
    b = _struct(E4, 4, "struct-md-768w-o0-adjacent-sections-tail-1152-tok2560@4")
    assert a["prov_id"] != b["prov_id"]


def test_id_discriminates_policies_that_share_an_epoch():
    """FIVE tracked policy artifacts declare epoch 3. Same epoch, same legend
    shape, different bytes must still separate."""
    a = _struct(E3, 3, "struct-md-768w-o0-adjacent-sections-tail-1152@3")
    b = _struct("0" * 64, 3, "struct-md-768w-o0-adjacent-sections-tail-1152@3")
    assert a["chunker"]["chunk_epoch"] == b["chunker"]["chunk_epoch"] == 3
    assert a["prov_id"] != b["prov_id"]


def test_id_is_stable_across_confidence_and_evidence():
    """A stamped row and the same rule recovered from a manifest are the same
    rule, so they must intern to the same id."""
    stamped = _struct(E4, 4, "x@4")
    recovered = CP.build(
        family="generic-structured", chunker=stamped["chunker"],
        channels=stamped["channels"], confidence="recovered",
        evidence=["recovered from a shard manifest"])
    assert stamped["prov_id"] == recovered["prov_id"]


def test_master_only_producer_does_not_change_the_id_and_never_ships():
    """ADR-039: which RUN made a row is master-only, and is not part of the
    rule — otherwise a redistributor gets two ids with identical content."""
    bare = _struct(E4, 4, "x@4")
    withrun = CP.build(
        family="generic-structured", chunker=bare["chunker"],
        channels=bare["channels"],
        producer={"module": "example-producer/chunk.py",
                  "producer_revision": "feff1f09", "input": {"path": "example/input"}},
        confidence="stamped")
    assert bare["prov_id"] == withrun["prov_id"]
    assert "producer" not in CP.compact(withrun)
    assert "producer" in withrun


def test_unknown_requires_a_reason():
    with pytest.raises(CP.ProvenanceError, match="unknown_reason"):
        CP.build(family="unknown", chunker={},
                 channels={"canonical": "unknown", "dense": "unknown",
                           "sparse": "unknown"},
                 confidence="unknown", unknown_fields=["family"])


def test_recovered_requires_cited_evidence():
    with pytest.raises(CP.ProvenanceError, match="recovered"):
        _ = CP.build(
            family="generic-legacy-window",
            chunker={"chunk_epoch": 1},
            channels={"canonical": "body", "dense": "body", "sparse": "body"},
            confidence="recovered")


def test_unnamed_surface_is_rejected():
    with pytest.raises(CP.ProvenanceError, match="named surface"):
        CP.build(family="generic-legacy-window", chunker={"chunk_epoch": 1},
                 channels={"canonical": "body", "dense": "whatever",
                           "sparse": "body"},
                 confidence="stamped")


def test_legend_key_must_match_its_content():
    rec = _struct(E4, 4, "x@4")
    tampered = json.loads(json.dumps(rec))
    tampered["chunker"]["chunk_epoch"] = 99
    with pytest.raises(CP.ProvenanceError, match="does not match"):
        CP.validate_legend({rec["prov_id"]: tampered})


# --- the shipped artifacts of this task -------------------------------------

@pytest.mark.skipif(not (ART / "provenance-legend.json").exists(),
                    reason="task-407 artifacts not present")
def test_shipped_legend_is_internally_consistent():
    CP.validate_legend(json.loads((ART / "provenance-legend.json").read_text()))


@pytest.mark.skipif(not (ART / "provenance-legend.json").exists(),
                    reason="task-407 artifacts not present")
def test_interned_legend_never_speaks_for_one_wave():
    """A legend record is SHARED by several waves. Wave-specific evidence in it
    would make the first wave to mint the id speak for all of them — which is
    exactly what happened before this test existed (a p0-era-passages row
    resolved to a record whose evidence read "wave=p0-era-fw")."""
    legend = json.loads((ART / "provenance-legend.json").read_text())
    names = set(json.loads((ART / "wave-provenance-map.json").read_text()))
    for pid, rec in legend.items():
        for line in rec.get("evidence", []):
            for wave in names:
                assert f"wave={wave}" not in line, (pid, wave, line)


@pytest.mark.skipif(not (ART / "wave-provenance-map.json").exists(),
                    reason="task-407 artifacts not present")
def test_every_mapped_wave_resolves_and_unknowns_stay_unknown():
    legend = json.loads((ART / "provenance-legend.json").read_text())
    wave_map = json.loads((ART / "wave-provenance-map.json").read_text())
    total = 0
    for wave, entry in wave_map.items():
        total += entry["rows"]
        a = entry["assignment"]
        ids = ([a["prov_id"]] if a["kind"] == "uniform"
               else [r["prov_id"] for r in a.get("rules", [])])
        for pid in ids:
            assert pid in legend, (wave, pid)
        if entry["confidence"] == "unknown":
            assert all(legend[i]["family"] == "unknown" for i in ids), wave
            assert all(legend[i].get("unknown_reason") for i in ids), wave
        else:
            assert all(legend[i]["family"] != "unknown" for i in ids), wave
    # re-derivation: the map must account for the store's full row count
    assert total == 53_990_262, total
