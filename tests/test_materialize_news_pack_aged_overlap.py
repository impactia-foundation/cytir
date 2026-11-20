#!/usr/bin/env python3
"""Task 625 -- regression test for the aged-out/stale/fresh tombstone overlap.

Found running the real catch-up pipeline against 2026-09-17's delta, in two
separate incidents:

1. A point can be BOTH a stale same-URL-group predecessor (today's pack
   carries a newer version at that URL) and independently >36mo old by the
   twin's own stored `published` date. `materialize_news_pack.main()` used
   to raise "point has two tombstone reasons" on that overlap and refuse the
   whole pack.

2. A point's id can be re-minted IDENTICAL by today's own artifact (content
   unchanged since it first aged past the retention cutoff) while ALSO
   appearing on the independently-computed, pre-pack aged-ids candidate
   list. Nothing checked this second overlap, so the apply wrote the point
   fresh and then immediately deleted it as "aged out, no replacement" in
   the same transaction -- caught only by verify_target's exact-membership
   check, AFTER the twin write+delete had already executed (609 points on
   2026-09-17: task 625's catch-up doc §7.1).

`resolve_aged_out_overlap` is the fix for both, isolated into a pure
function so this does not need a live Qdrant to test (verification-doctrine
1).

Moved from task 385's exploration scripts
(task 646 S4, alongside the module's own move to
`cytir/sources/families/news.py`).
"""
import sys

from cytir.sources.families.news import resolve_aged_out_overlap


def test_overlap_is_dropped_from_aged_out_not_raised():
    aged_ids, dropped, dropped_fresh = resolve_aged_out_overlap(
        aged_ids_raw=["a", "b", "c"], stale_ids=["b", "z"])
    assert aged_ids == ["a", "c"], aged_ids
    assert dropped == ["b"], dropped
    assert dropped_fresh == [], dropped_fresh


def test_no_overlap_is_a_no_op():
    aged_ids, dropped, dropped_fresh = resolve_aged_out_overlap(
        aged_ids_raw=["a", "b"], stale_ids=["x", "y"])
    assert aged_ids == ["a", "b"], aged_ids
    assert dropped == [], dropped
    assert dropped_fresh == [], dropped_fresh


def test_empty_inputs():
    assert resolve_aged_out_overlap([], []) == ([], [], [])
    assert resolve_aged_out_overlap([], ["x"]) == ([], [], [])
    assert resolve_aged_out_overlap(["a"], []) == (["a"], [], [])


def test_full_overlap_empties_aged_out_entirely():
    aged_ids, dropped, dropped_fresh = resolve_aged_out_overlap(
        aged_ids_raw=["a", "b"], stale_ids=["a", "b", "c"])
    assert aged_ids == [], aged_ids
    assert dropped == ["a", "b"], dropped
    assert dropped_fresh == [], dropped_fresh


def test_fresh_overlap_is_dropped_as_still_current():
    """The 2026-09-17 incident: an aged candidate re-minted by today's own
    artifact must be dropped, not deleted out from under its own fresh
    write."""
    aged_ids, dropped_stale, dropped_fresh = resolve_aged_out_overlap(
        aged_ids_raw=["a", "b", "c"], stale_ids=[], fresh_ids=["b"])
    assert aged_ids == ["a", "c"], aged_ids
    assert dropped_stale == [], dropped_stale
    assert dropped_fresh == ["b"], dropped_fresh


def test_stale_overlap_takes_precedence_over_fresh_overlap():
    """A point cannot be both -- stale_ids and fresh_ids (this pack's own
    minted ids) are disjoint by construction (existing_group_points excludes
    `keep`), but the function must not double-count if ever called with
    inputs that violate that invariant: stale wins, since it is checked
    first and fresh is computed against what stale already removed."""
    aged_ids, dropped_stale, dropped_fresh = resolve_aged_out_overlap(
        aged_ids_raw=["a"], stale_ids=["a"], fresh_ids=["a"])
    assert aged_ids == [], aged_ids
    assert dropped_stale == ["a"], dropped_stale
    assert dropped_fresh == [], dropped_fresh


def test_real_20260917_incident_ids_reproduce_the_609_point_loss():
    """Real record, not synthetic: five of the 609 point ids from the actual
    2026-09-17 catch-up pack that were written fresh and then immediately
    deleted as aged-out in the same transaction, plus one real id that was a
    genuine (non-overlapping) aged-out tombstone from the same pack --
    verbatim from the corrected
    tombstone-pointers.22aff40e8a...json's `aged_out_dropped_as_still_current`
    and `aged_out[0].predecessor_id` fields (task 625's catch-up doc
    sec 7.1a). Without the fix, all six would have been treated as aged-out;
    with it, only the genuine one is."""
    overlap_sample = [
        "004d3f0d-0036-5f34-a957-aaae799c7904",
        "005630c1-65f8-557a-810b-e9adc1ebdee4",
        "00a9ba2d-200d-5a36-bd17-873fcbe7c58e",
        "019911fe-e825-5630-8eaa-ac8c4ec8e979",
        "02ba7a83-c709-5d54-84bd-8b6659c38f77",
    ]
    genuine_aged = "00a0f20e-68ea-5c4d-bf49-ce1ac0269f84"
    aged_ids_raw = sorted(overlap_sample + [genuine_aged])
    # fresh_ids stands in for the pack's own artifact ids: in the real
    # incident these 5 ids ARE artifact rows (content unchanged since aging
    # past the cutoff); genuine_aged is not one.
    aged_ids, dropped_stale, dropped_fresh = resolve_aged_out_overlap(
        aged_ids_raw=aged_ids_raw, stale_ids=[], fresh_ids=overlap_sample)
    assert aged_ids == [genuine_aged], aged_ids
    assert dropped_stale == [], dropped_stale
    assert sorted(dropped_fresh) == sorted(overlap_sample), dropped_fresh


def test_pointers_counts_shape_is_exactly_source_group_and_aged_out():
    """apply_content_pack_shared.py's verify_tombstone_pointers asserts EXACT
    dict equality: `counts != {"source_group": N, "aged_out": M}` aborts the
    whole preflight with "tombstone pointer counts mismatch". A first version
    of the overlap fix added a third key here (the drop count) and broke
    every pack that ever hit the overlap -- found running it for real against
    2026-09-17. This is not exercised by main() (no live qdrant in this
    suite, see the module docstring) so it is asserted directly against the
    dict-literal shape in the source, which is the next-cheapest thing to a
    live check and would have caught the regression before it shipped."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent
           / "cytir" / "sources" / "families" / "news.py").read_text()
    match = re.search(r'"counts":\s*\{([^}]*)\}', src)
    assert match, "no counts={...} dict literal found in cytir/sources/families/news.py"
    body = match.group(1)
    assert '"source_group"' in body and '"aged_out"' in body, body
    assert "aged_out_dropped_as_superseded" not in body, (
        "counts must be exactly {source_group, aged_out} -- extra keys break "
        "apply_content_pack_shared.py's exact dict-equality check: " + body)


if __name__ == "__main__":
    fails = 0
    for fn in (test_overlap_is_dropped_from_aged_out_not_raised,
               test_no_overlap_is_a_no_op, test_empty_inputs,
               test_full_overlap_empties_aged_out_entirely,
               test_fresh_overlap_is_dropped_as_still_current,
               test_stale_overlap_takes_precedence_over_fresh_overlap,
               test_real_20260917_incident_ids_reproduce_the_609_point_loss,
               test_pointers_counts_shape_is_exactly_source_group_and_aged_out):
        try:
            fn()
            print(f"  {fn.__name__}  OK")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL {fn.__name__}: {e}")
    print("PASS" if not fails else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)
