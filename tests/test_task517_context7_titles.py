#!/usr/bin/env python3
"""Task 517: the provenance front matter, and the title it was hiding.

Every test here runs against REAL sealed records copied verbatim out of the
task-417 context7 chain (`tests/testdata/task517_sealed_context7_records.json`):

  * `materialized_parent_v1` -- one row of the sealed `markdown` stage of
    `missing-git-sources-repos-001875-002124-v1`, i.e. exactly what the v1
    materializer wrote and what was chunked and embedded into `aw_e4`/`aw_e5`/
    `aw_e6`;
  * `chunk_zero_with_the_front_matter` / `chunk_one_without_it` -- the first two
    rows of that pack's `shard-0000.texts.jsonl.gz`, the file the assembly
    binder reads.

The defect, as shipped: all 126,427 staged context7 rows carried NO title, and
72,330 of them opened with the four-line YAML block -- as the snippet, and as
the head of the dense embedding, with `section_heading` literally "---".

Run: python3 -m pytest tests/test_task517_context7_titles.py
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cytir.sources.github_repo_docs.acquire as GH  # noqa: E402
from cytir.ops.audits import context7_software_docs_pack as AUDIT  # noqa: E402
from cytir.index import serving_metadata as SM  # noqa: E402
import cytir.sources.context7.materialize as MAT  # noqa: E402
from cytir.index import payload_shape_contract as PSC  # noqa: E402

SEALED = json.loads(
    (pathlib.Path(__file__).resolve().parent.parent / "tests" / "testdata"
     / "task517_sealed_context7_records.json").read_text())
PARENT = SEALED["materialized_parent_v1"]
CHUNK0 = SEALED["chunk_zero_with_the_front_matter"]
CHUNK1 = SEALED["chunk_one_without_it"]
CONTRACT = PSC.load()

# The blob the v1 materializer read, recovered by removing the header it added.
RAW = PARENT["markdown"].split("---\n\n", 1)[1].encode()


# ------------------------------------------------------- the shipped defect
def test_the_sealed_record_really_does_open_with_the_provenance():
    """Not a hypothetical. This is the record that was embedded."""
    assert PARENT["markdown"].startswith('---\naw_repository: "daltron/')
    assert CHUNK0["text"].startswith("---\naw_repository:")
    assert CHUNK0["section_heading"] == "---"
    assert "title" not in PARENT
    assert "title" not in (CHUNK0.get("source_metadata") or {})


def test_v1_would_fail_the_new_audit_and_v2_passes():
    """The v1 body is refused by name; the v2 body of the same document is not."""
    assert AUDIT._front_matter_absent(PARENT["markdown"]) is False
    assert AUDIT._front_matter_absent(
        GH.source_to_markdown(RAW, source_path="README.md").decode()) is True


# ------------------------------------------------------------------ the fix
def test_source_to_markdown_no_longer_prepends_the_block():
    out = GH.source_to_markdown(RAW, source_path="README.md").decode()
    assert not out.startswith("---\naw_")
    assert "aw_parent_sha256" not in out
    # and it is still the same document
    assert "NotificationBanner" in out


def test_the_provenance_survives_as_record_metadata():
    block = GH.source_provenance(
        owner_repo=PARENT["repository"], revision=PARENT["revision"],
        source_path=PARENT["source_path"],
        parent_sha256=PARENT["parent_original_sha256"])
    assert block == {"repository": "daltron/notificationbanner",
                     "revision": PARENT["revision"],
                     "source_path": "README.md",
                     "parent_sha256": PARENT["parent_original_sha256"]}
    # nothing the front matter carried is lost
    assert set(block) == {"repository", "revision", "source_path",
                          "parent_sha256"}


def test_the_title_is_the_one_production_gives_this_document():
    """`https://github.com/Daltron/NotificationBanner/blob/master/README.md` is
    titled `Written in Swift 5` in the collection this build replaces -- on all
    five of its chunks. The rule re-derives it from the blob."""
    body = GH.source_to_markdown(RAW, source_path="README.md").decode()
    assert GH.compose_document_title(body, "README.md") == "Written in Swift 5"


def test_the_title_falls_back_to_the_basename_only_with_no_heading():
    assert GH.compose_document_title("no headings here\n", "docs/guide.rst") \
        == "guide.rst"
    assert GH.compose_document_title("## Second level first\n", "x/y.md") \
        == "Second level first"
    assert GH.compose_document_title("# Closing hashes ###\n", "x/y.md") \
        == "Closing hashes"


def test_the_materializer_declares_a_new_generation():
    """A change in what the collection CONTAINS has to be a visible diff."""
    assert MAT.VERSION == "context7-exact-path-materialization.v2"


# ------------------------------------------------- the projection reads it
def test_the_projection_titles_the_row_once_the_record_carries_one():
    """`aw_serving_metadata._pack_title` already reads `source_metadata.title`;
    it had nothing to read. This is the whole distance between 0 titles and
    126,427 of them."""
    assert SM._pack_title(CHUNK0) is None
    titled = dict(CHUNK0)
    titled["source_metadata"] = dict(CHUNK0["source_metadata"],
                                     title="Written in Swift 5")
    assert SM._pack_title(titled) == "Written in Swift 5"


# ------------------------------------------------------------- the guard
def test_the_guard_matches_the_real_shipped_snippet_and_not_the_next_chunk():
    pattern = PSC.provenance_head_pattern(CONTRACT)
    assert pattern is not None
    # the staged snippet is the chunk text with whitespace collapsed
    snippet0 = " ".join(CHUNK0["text"].split())
    snippet1 = " ".join(CHUNK1["text"].split())
    assert pattern.match(snippet0)
    assert not pattern.match(snippet1)


def test_the_guard_catches_the_rfc_shape_too():
    """The first pattern named the four github keys and would have missed the
    RFC family, whose block opens `aw_rfc_number`."""
    pattern = PSC.provenance_head_pattern(CONTRACT)
    assert pattern.match('--- aw_rfc_number: 8446 aw_source_path: "input/rfcs/'
                         'rfc8446.txt" ---')


def test_the_guard_catches_the_fedlex_shape_it_was_generalized_from():
    pattern = PSC.provenance_head_pattern(CONTRACT)
    assert pattern.match("Official source identity - ELI: `cc/11/529_488_529`")


def test_the_guard_does_not_fire_on_a_document_that_starts_with_a_rule():
    """A Markdown horizontal rule is not a provenance block."""
    pattern = PSC.provenance_head_pattern(CONTRACT)
    assert not pattern.match("--- and then some prose about aw_ things")
    assert not pattern.match("Written in Swift 5 A notification banner")
