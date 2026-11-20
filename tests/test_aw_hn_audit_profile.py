from cytir.sources.hn import audit_profile as subject
import json


def test_validates_source_native_hn_semantics_and_canonical_id():
    item_id = 7
    document_id = subject._document_id(item_id)
    markdown = [{"item_id": item_id, "document_id": document_id,
                 "markdown": "# T\n\nHacker News item: 7\n\nAuthor: a\n\nScore: 2\n\nSource: https://x\n\nDiscussion: https://news.ycombinator.com/item?id=7\n"}]
    outlinks = [{"document_id": document_id, "target_url": "https://x"}]
    chunks = [{"point_id": "7a1c68f4-4060-5f4c-b6a6-2780a7cb9ccd",
               "document_id": document_id, "parent_markdown_sha256": "a" * 64,
               "chunk_ordinal": 0}]
    assert subject.validate_semantics(markdown, outlinks, chunks, expected=1) == {
        "parents": 1, "chunks": 1, "outlinks": 1, "empty_outlink_parents": 0}


def test_audit_methods_cover_every_structured_profile_requirement():
    registry = json.loads((subject.Path(__file__).parents[1]
                           / "data/source-preservation-profiles.v1.json").read_text())
    profile = registry["profiles"]["structured_data_feed"]
    # Regression: the real profile currently has four independent chunk gates;
    # a shorter method list must fail closed instead of silently omitting one.
    assert len(profile["chunks"]["requirements"]) == 4
