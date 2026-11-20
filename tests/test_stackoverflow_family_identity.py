"""Characterization: the family makes what the old Stack Overflow path makes.

`cytir/extract/stackoverflow_qa.py` and `cytir/sources/families/stackoverflow_qa.py` put the production
frontier's SELECTION and the live tranches' RENDERING on the family door. This test runs the OLD code
on the committed sample (`data/samples/stackoverflow-qa/`, 24 real rows of the dump's `Posts.xml`) and
compares with the family's pins in `tests/test_aw_family_stackoverflow_qa.py`:

  * the old frontier builder (`cytir.sources.stackoverflow.frontier`: `consume_posts`,
    `_eligible_rows`) picks the same ten questions the family takes, and none of the four negatives;
  * the old tranche materializer (`select_exact_posts`, `select_users`, `materialize`) makes Markdown
    rows whose sha256 is the family's pin;
  * the old stage script's own chunking call (`generic_chunking.chunk_generic_records` on an explicit
    structured selection, `dense_text or text`, `sparse_text or text`) makes the dense and sparse
    input whose sha256 the family's pins hold.

It moved to tests/ (task 2256); it goes when the old
scripts go (task 643 S9). Nothing here touches the two pinned files whose bytes feed
`producer.revision`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import pathlib
import sqlite3

import pytest

pytest.importorskip("lxml")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
TESTS = REPO / "tests" / "test_aw_family_stackoverflow_qa.py"
POSTS = REPO / "data" / "samples" / "stackoverflow-qa" / "Posts.xml"
POLICY = REPO / "data" / "chunk-policy-struct-md-768w-o0-adjacent-sections-tail-1152-tok2560.v1.json"


def _pins():
    spec = importlib.util.spec_from_file_location("stackoverflow_family_pins", TESTS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_old_frontier_picks_the_questions_the_family_takes():
    import cytir.sources.stackoverflow.frontier as old
    pins = _pins()
    conn = sqlite3.connect(":memory:")
    with POSTS.open("rb") as fh:
        old.consume_posts(fh, conn)
    wanted = old.requested_user_ids(conn)
    assert wanted == set()                      # no selected post names an owner by id
    old.consume_users(io.BytesIO(b"<users></users>"), conn, wanted)
    eligible = [old._row(c) for c in old._eligible_rows(conn)]
    assert [str(r["question_id"]) for r in eligible] == sorted(pins.PAIRS, key=int)
    assert {str(r["question_id"]): str(r["accepted_answer_id"]) for r in eligible} == pins.PAIRS


def test_the_old_materializer_and_the_old_stage_chunking_make_the_family_pins(word_only_chunking_allowed):
    import cytir.extract.stackoverflow_pilot as pilot
    import cytir.sources.stackoverflow.frontier as frontier
    import cytir.sources.stackoverflow.materialize_production_tranche as old
    from cytir.acquire.artifact_io import canonical_json
    from cytir.core.common import explicit_structured_chunker_selection
    from cytir.chunk.generic import chunk_generic_records
    pins = _pins()
    conn = sqlite3.connect(":memory:")
    with POSTS.open("rb") as fh:
        frontier.consume_posts(fh, conn)
    frontier.consume_users(io.BytesIO(b"<users></users>"), conn, set())
    rows = [frontier._row(c) for c in frontier._eligible_rows(conn)]
    with POSTS.open("rb") as fh:
        selection = old.select_exact_posts(fh, frontier=rows)
    users = pilot.select_users(io.BytesIO(b"<users></users>"), owner_ids=set())
    result = old.materialize(selection, users)
    assert result["quarantine"] == [] and len(result["markdown"]) == 10
    md = "".join(canonical_json({k: v for k, v in r.items() if k != "extraction"}) + "\n"
                 for r in result["markdown"])
    assert hashlib.sha256(md.encode()).hexdigest() == pins.MARKDOWN_SHA256
    sel = explicit_structured_chunker_selection(
        str(POLICY), pins.POLICY_SHA256, "ac6544c8a46e00af67e330e85a9028c66b8cfd9a")
    dense, sparse = [], []
    for row in result["markdown"]:
        units = list(chunk_generic_records(row["markdown"], selection=sel))
        assert len(units) == 1
        for n, u in enumerate(units):
            for acc, v in ((dense, u.dense_text or u.text), (sparse, u.sparse_text or u.text)):
                acc.append(canonical_json({"url": row["url"], "ordinal": n, "text": v}) + "\n")
    assert [hashlib.sha256("".join(a).encode()).hexdigest() for a in (dense, sparse)] == [
        pins.DENSE_SURFACE_SHA256, pins.SPARSE_SURFACE_SHA256]
