#!/usr/bin/env python3
"""The pack route must not build what the flip does not own.

`_plan_pack_sources` plans every `source=` directory under the release. Task 478
§2.5 called that deliberate over-collection and said the claim manifests'
`claimed(S)` term would narrow it -- which happens for the `stage` and `twin`
routes and never happened for the pack route, because 122 sources stage under
ONE binder set id. Task 513 measured 6,521,047 expansion-lane rows staged into
the flip build as a result, including 6.4M OpenAlex rows whose own fragment says
"NOT flip-lane ... are NOT in the Sep-8 flip's membership set".

The exclusion is derived from evidence tiers, and the properties that matter
are that it reaches exactly as far as the measurement, that every source
carries a citation, and that it does NOT reach the four expansion-ledger packs
which are flip-lane.

Task 514 (its exclusion-derivation script, doc
build-phase B3 §4) added two more tiers on top of task 513's
original three (`fragment_lane`, `expansion_portfolio_class`, `family_only`):
`sealed_excluded_marker` (a source carries its own `EXCLUDED.json`) and
`unclaimed_by_any_set` (no contributing set's vector receipt and no
slice-ledger source list names it -- ADR-101 Ruling 2 guards G1/G3). Six
sources moved into these two tiers, extending the measured exclusion from
task 513's 6,521,047 to task 514's 6,532,370 staged rows (§5's stated
arithmetic).
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

from tests.helpers.repo_files import repo_file_or_skip

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from cytir.index import assemble_inputs as AI  # noqa: E402

ARTIFACT = repo_file_or_skip("data/assembly/pack-route-exclusions.v1.json")
CLASSES = repo_file_or_skip("data/assembly/expansion-portfolio-classes.v1.json")

# The independently measured Track-2 staged population: task 513 §6.1 measured
# 6,521,047 (its staged-pack source census over the real
# staged tree); task 514 §4/§5 (build-phase B3 doc and its
# exclusion-derivation script) extended it by
# 11,323 rows across six sources with no contributing-set claim at all
# (`sealed_excluded_marker` + `unclaimed_by_any_set` tiers), for a current
# sealed measurement of 6,532,370.
MEASURED_TRACK2_ROWS = 6_532_370


def _doc():
    return json.loads(ARTIFACT.read_text())


def test_the_tracked_exclusion_artifact_loads_and_every_source_is_cited():
    ex = AI.load_pack_route_exclusions()
    assert ex, "no pack-route exclusions load from the tracked artifact"
    for source, row in ex.items():
        assert row["lane"] != "flip"
        assert row["named_by"], f"{source} is excluded with no evidence"
        assert row["evidence_tier"] in (
            "fragment_lane", "expansion_portfolio_class", "family_only",
            # task 514 (build-phase B3 doc §4): two more tiers for
            # sources no contributing set claims at all.
            "sealed_excluded_marker", "unclaimed_by_any_set")
        for cite in row["named_by"]:
            assert cite.get("kind"), f"{source}: a citation must name its kind"


def test_nothing_is_excluded_on_family_evidence_while_staging_rows():
    """Family evidence is only ever acceptable for a source contributing nothing."""
    for row in _doc()["evidence"]:
        if row["evidence_tier"] == "family_only":
            assert row["staged_rows"] == 0, (
                f"{row['source']} is excluded on family evidence alone but stages "
                f"{row['staged_rows']:,} rows -- it needs a claim fragment")


def test_the_exclusion_reaches_exactly_as_far_as_the_measurement():
    doc = _doc()
    residue = sum(r["rows"] for r in doc["staged_but_no_evidence_names_them"])
    assert doc["excluded_staged_rows"] + residue == MEASURED_TRACK2_ROWS, (
        "the rule reaches further or less far than the measured Track-2 population")
    assert residue == 0, (
        "an expansion-lane row no evidence names must block the load: "
        f"{doc['staged_but_no_evidence_names_them']}")


def test_the_four_flip_lane_ledger_packs_are_not_excluded():
    """Expansion-ledger membership is not by itself a Track-2 signal."""
    doc = _doc()
    kept = {row["source"] for row in doc["kept_as_flip_by_portfolio_class"]}
    assert kept, "no ledger pack was kept as flip -- the guard against over-reach is not firing"
    ex = set(AI.load_pack_route_exclusions())
    assert not (kept & ex), f"a flip-lane ledger pack was excluded: {kept & ex}"
    # the three context7 packs belong to the flip family `context7_exact`
    assert sum(1 for s in kept if s.startswith("context7-")) == 3


def test_every_portfolio_class_the_ledger_uses_on_a_staged_source_is_ruled():
    """An unknown class must refuse, not default -- both defaults have a victim."""
    classes = json.loads(CLASSES.read_text())["classes"]
    for name, row in classes.items():
        assert row["disposition"] in ("flip", "expansion")
        assert row.get("why"), f"{name} has a disposition with no evidence"


def test_load_refuses_an_artifact_with_the_wrong_schema(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"schema": "something.else", "evidence": []}))
    with pytest.raises(SystemExit):
        AI.load_pack_route_exclusions(p)


def test_an_absent_artifact_means_no_exclusions(tmp_path):
    assert AI.load_pack_route_exclusions(tmp_path / "nope.json") == {}
