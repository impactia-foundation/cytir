"""The surface-corrected populations must point at the CORRECTED stages.

Task 528a fixed four families in place on `aw_e8`. Nothing on disk knew that:
`sealed-populations.v1.json` still named the stages whose surfaces the
collection no longer held, so a rebuild would have re-embedded the old text and
reverted every fix in silence -- handoff risk #1 of the task-528a surface-delta doc
§7. Task 528a-2 sealed the corrected stages and moved the pointers.

This pins the pointer, not the prose: a future edit that puts a superseded path
back under `stage` fails here rather than at the next rebuild.

WHAT TASK 533 CHANGED, AND WHY THE TEST HAD TO. The first cut of this file
asserted the correction by NAME -- `"task528a2" in p["stage"]`. That reads as a
pin but is a ratchet stuck at one generation: the next correction in the same
lane fails the test for being newer, which is the opposite of what the file
exists to prevent. Task 533 re-materialised the F2 family (168,128 rows on the
composed surface) and carved task482 its own release, and both edits tripped it.

The property is stated directly instead: a corrected family's `stage` must not
be any path in its own supersession chain, must be one of the generations this
lane has ratified, and must keep the original under `superseded_stage`. Adding
a generation is a one-line edit to `GENERATIONS`, which is a visible, reviewable
statement that a new correction landed -- not a rename that quietly widens a
substring match.
"""
from __future__ import annotations

import json
import pathlib

from tests.helpers.repo_files import repo_file_or_skip

REPO = pathlib.Path(__file__).resolve().parents[1]
SEALED = repo_file_or_skip("data/assembly/sealed-populations.v1.json")

# The corrections this lane has made, oldest first. A corrected family's stage
# and vectors must name one of these.
GENERATIONS = ("aw-task528a2-", "aw-task533-")

# set_id -> the generations whose stage this family may legitimately be on.
CORRECTED = {
    "task467_entity_card_1024_reembed": GENERATIONS,
    "task472_f2_html_1024_embed": GENERATIONS,
    "task500_news_1024_add": GENERATIONS,
    # task 533 carved this one out of task472's merged release, so it is now a
    # corrected family in its own right rather than a folded one.
    "task482_f2_consent_supplement_chunks": GENERATIONS,
}
# Task 533 emptied this. `assembly_binder plan` REFUSED the folded arrangement
# on BOTH families that used it -- one F2 release of 138,837 rows carrying two
# sets, one news release of 28,525 carrying three -- because a population's
# declared `rows` must equal the sum of its own vector shards. Worse, the sets
# folded INTO those releases still named their PRE-correction stages, so the
# one arrangement a rebuild could execute would have reverted task 528a's
# surface fix on 10,566 news rows and 11,763 F2 rows in silence. Every set now
# names a release holding exactly its own rows.
FOLDED: dict[str, str] = {}


def populations():
    doc = json.loads(SEALED.read_text())
    return {p["set_id"]: p for p in doc["populations"]}


def _on_a_ratified_generation(uri, generations):
    return any(g in uri for g in generations)


def test_corrected_families_point_at_a_ratified_corrected_generation():
    pops = populations()
    for sid, generations in CORRECTED.items():
        p = pops[sid]
        assert _on_a_ratified_generation(p["stage"], generations), (sid, p["stage"])
        assert _on_a_ratified_generation(p["vectors"], generations), (sid, p["vectors"])
        # the original is kept, so a reader can see what moved and why
        assert p["superseded_stage"] and p["superseded_stage"] != p["stage"], sid
        assert p["superseded_vectors"] != p["vectors"], sid
        assert p["surface_corrected_by"], sid


def test_folded_families_say_which_stage_now_covers_them():
    pops = populations()
    for sid, covers in FOLDED.items():
        p = pops[sid]
        assert p.get("surfaces_now_staged_with") == covers, sid
        assert p.get("surface_note"), sid


def test_no_population_borrows_another_set_s_release():
    """The invariant that makes `assembly_binder plan` able to run at all.

    A set that points at a release covering several sets declares a row count
    the release cannot match, and plan refuses the family outright -- which is
    how BOTH corrected families became unbuildable from a clean clone without
    anything in the repo saying so.
    """
    pops = populations()
    for sid, p in pops.items():
        assert not p.get("surfaces_now_staged_with"), (
            f"{sid} borrows {p['surfaces_now_staged_with']}'s release; give it "
            f"a release holding exactly its own rows")
        assert "stage_rows" not in p or p["stage_rows"] == p.get("rows"), (
            sid, p.get("stage_rows"), p.get("rows"))


def test_no_corrected_family_still_points_at_a_superseded_stage():
    """The failure this exists to prevent, stated as an assertion."""
    pops = populations()
    for sid, p in pops.items():
        if sid in CORRECTED:
            assert p["stage"] != p["superseded_stage"], sid
            assert p["vectors"] != p["superseded_vectors"], sid


def test_every_corrected_family_declares_rows_equal_to_its_own_release():
    """The refusal task 533 hit, stated as an assertion.

    `assembly_binder plan` compares a population's declared `rows` against the
    sum of its vector shards and refuses a mismatch. Task 528a-2 sealed two
    sets into one 138,837-row release and left task472 declaring 127,077, so
    plan refused the whole family and a clean clone could not build it. Nothing
    in the repo said so until the tool was run.
    """
    pops = populations()
    for sid in CORRECTED:
        p = pops[sid]
        assert isinstance(p.get("rows"), int) and p["rows"] > 0, sid
        # a family whose release covers more than its own rows must say so
        # explicitly rather than leaving plan to discover it
        assert "stage_rows" not in p or p["stage_rows"] == p["rows"], (
            sid, p.get("stage_rows"), p["rows"])
