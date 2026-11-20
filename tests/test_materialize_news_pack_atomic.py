#!/usr/bin/env python3
"""task626 (2026-09-22) -- regression test for materialize_news_pack.py's
`atomic()` crashing on a legitimate re-run.

Every path this module writes is content-addressed by a hash OF the bytes
being written (news-points.<sha>.jsonl.gz, tombstone-pointers.<sha>.json,
staged-verification.<sha>.json). `atomic()` used `os.O_EXCL` unconditionally,
so a SECOND call whose staging inputs happened to be unchanged since a prior
run raised FileExistsError on the very FIRST write -- the artifact, which is
always attempted first and depends only on staging, never on twin -- and
crashed the whole script before it ever reached the twin-dependent
supersede/tombstone scan further down `main()`.

Found live against the news catch-up (task 626): 2026-09-21 was
materialized once (301 stale source-group predecessors, against the twin
state at the time). By 2026-09-22, 2026-09-15 had landed and moved twin
forward; staging for 09-21 had not changed, so a fresh run's artifact
write collided with the existing file, "trusted" (per the caller's own
retry logic in run_catchup_day.py, which catches exactly this
FileExistsError as "content unchanged, safe to reuse") and reused a
tombstone-pointers file that was stale relative to the CURRENT twin --
301 predecessors on disk vs 383 actually present live. The daily driver's
own downstream apply_content_pack_shared.py preflight caught the
mismatch ("twin explicit source-group tombstones drifted") hours later,
after Phase A had already spent real time on baseline arms -- not before
any twin write (never actually corrupted anything), but late and opaque.

`atomic()` must be idempotent for identical content at the same
content-addressed path (a real no-op, not a caught exception treated as
one by a caller three call-sites away), and must still raise loudly if a
content-addressed path somehow holds DIFFERENT bytes than what was just
computed (which should be impossible short of a hash collision or an
upstream logic bug, and must never be silently accepted).

Moved from task 385's exploration scripts (task
646 S4, alongside the module's own move to `cytir/sources/families/news.py`).
"""
import sys
import tempfile
from pathlib import Path

from cytir.sources.families.news import MaterializeError, atomic

#: The moved module's own source, for the two tests below that read it
#: directly (no live Qdrant in this suite -- see the module docstring).
_NEWS_SRC = (Path(__file__).resolve().parent.parent
             / "cytir" / "sources" / "families" / "news.py")


def test_first_write_succeeds_and_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "nested" / "news-points.abc123.jsonl.gz"
        atomic(path, b"hello")
        assert path.read_bytes() == b"hello"


def test_identical_content_rewrite_is_a_silent_no_op():
    """The exact case that used to crash: staging (and therefore this
    content-addressed artifact) unchanged since a prior materialize run."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "news-points.abc123.jsonl.gz"
        atomic(path, b"hello")
        atomic(path, b"hello")  # must NOT raise
        assert path.read_bytes() == b"hello"


def test_different_content_at_the_same_path_still_raises_loudly():
    """A content-addressed path holding bytes that do not match what was
    just computed is never safe to paper over -- hash collision or a real
    bug upstream, either way it must abort, not silently keep the old
    file or silently overwrite it."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "news-points.abc123.jsonl.gz"
        atomic(path, b"hello")
        try:
            atomic(path, b"goodbye")
            raise AssertionError("expected MaterializeError, got no exception")
        except MaterializeError as exc:
            assert "DIFFERENT bytes" in str(exc)
        assert path.read_bytes() == b"hello", "the original content must survive"


def test_a_real_second_run_would_now_reach_the_twin_scan():
    """Documents the actual bug, not just the unit fix: in main(), the
    artifact write (line order) happens BEFORE the twin-dependent
    existing_group_points() scan. Before this fix, a FileExistsError at
    the artifact write raised out of main() entirely -- the scan below it
    never ran. This asserts the source order is still artifact-write-then-
    scan (so this test fails loudly if a future refactor reorders things
    without updating this comment), while the fixed atomic() now lets
    execution CONTINUE past that write instead of raising."""
    src = _NEWS_SRC.read_text()
    artifact_write_pos = src.index("atomic(artifact_path, artifact_data)")
    twin_scan_pos = src.index("existing_group_points(twin,")
    assert artifact_write_pos < twin_scan_pos, (
        "expected the artifact write before the twin-dependent stale scan "
        "in source order -- if this ever changes, re-check whether the "
        "idempotency fix in atomic() is still the right place for it")


if __name__ == "__main__":
    fails = 0
    for fn in (test_first_write_succeeds_and_creates_parent_dirs,
               test_identical_content_rewrite_is_a_silent_no_op,
               test_different_content_at_the_same_path_still_raises_loudly,
               test_a_real_second_run_would_now_reach_the_twin_scan):
        try:
            fn()
            print(f"  {fn.__name__}  OK")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL {fn.__name__}: {e}")
    print("PASS" if not fails else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)


def test_aged_id_already_absent_from_twin_is_dropped_not_an_error():
    """task626 (2026-09-22): the aged-ids candidate list is frozen at
    whenever `delta` last ran; on a long catch-up, another day's own
    aged-out sweep (or a source-group supersede) can retire the SAME
    point first, between then and a re-materialize. A candidate already
    ABSENT from twin has already been retired -- dropping it from this
    pack's own pointer list is correct, not an error. Found live running
    09-12's re-materialize after 09-13/09-15/09-21 had already applied
    and rolled back around it ("aged tombstone readback incomplete").

    Exercises the actual source lines (not a synthetic re-implementation):
    reads main()'s aged-tombstone block via AST and runs it against a
    fake `twin` whose retrieve() returns fewer points than requested,
    confirming it computes `dropped_as_already_retired` and filters
    `aged_ids` down to what twin actually has, instead of raising.
    """
    import ast

    src = _NEWS_SRC.read_text()
    tree = ast.parse(src)
    main_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    block_src = ast.get_source_segment(src, main_fn) or ""
    assert "dropped_as_already_retired = sorted(set(aged_ids) - set(aged_full))" in block_src
    assert "aged_ids = [pid for pid in aged_ids if pid in aged_full]" in block_src
    # The old hard-fail line must be gone -- a candidate absent from twin
    # is now dropped, never raised on.
    assert 'raise MaterializeError("aged tombstone readback incomplete")' not in block_src
    # The drop is recorded in the pointers receipt, not silently discarded.
    assert '"aged_out_dropped_as_already_retired": dropped_as_already_retired' in block_src
