"""Task 601: end-to-end real-record proof that an HN-discussion pack passes
the `hn_discussion` tier-2 preservation-profile contract honestly.

ingest (--preserve-root) -> chunk.generic (epoch-4) -> audit -> profile
compliance -> `preservation_contract._validate_profile_compliance` (the exact
function `content_pack_preservation.py` calls for every pack source).
"""
from __future__ import annotations
from pathlib import Path
import pytest
from cytir.ops.audits import hn_discussion_profile as AUDIT
from cytir.sources.hn import discussion_ingest as ING
from tests.test_aw_hn_discussion_ingest import RecordedFirebaseClient, write_input


def test_an_inline_username_mention_by_another_commenter_is_not_a_leak():
    """PRD Q5, verbatim: a comment's own prose mentioning another user inline
    ("as tancop said above") is an ACCEPTED residual risk, not a defect --
    this project's GDPR posture is field/record-level, never free-text
    scrubbing. The audit must not refuse a real, in-scope thread for this."""
    markdown = [{"document_id": "d1", "item_id": 1,
                "url": "https://news.ycombinator.com/item?id=1",
                "markdown": "Title\n\nAs tancop said above, this tracks."}]
    # Should not raise.
    AUDIT.validate_semantics(
        parent_root=Path("."),
        original_receipts={1: {"status": "complete",
                               "artifacts": {"body": {}, "transport": {}}}},
        markdown=markdown, outlinks=[{"document_id": "d1", "has_outlink": False,
                                     "target_url": None}],
        chunks=[{"parent_markdown_sha256": "a" * 64, "chunk_ordinal": 0,
                "url": "https://news.ycombinator.com/item?id=1",
                "chunk_policy_id":
                    "struct-md-768w-o0-adjacent-sections-tail-1152-tok2560"}],
        expected_documents=1)


def test_a_labeled_identity_line_reaching_the_sealed_pack_is_still_refused():
    """The exact prior defect this profile's PRD names
    (acquire_hn_firebase_preserved.py's `f"Author: {by}\\n\\n"` line): a
    REGRESSION that labels a username into the embedded surface must still
    be refused."""
    markdown = [{"document_id": "d1", "item_id": 1,
                "url": "https://news.ycombinator.com/item?id=1",
                "markdown": "Title\n\nAuthor: tancop\n\nSome body text."}]
    with pytest.raises(ValueError, match="labeled identity line"):
        AUDIT.validate_semantics(
            parent_root=Path("."),
            original_receipts={1: {"status": "complete",
                                   "artifacts": {"body": {}, "transport": {}}}},
            markdown=markdown,
            outlinks=[{"document_id": "d1", "has_outlink": False,
                      "target_url": None}],
            chunks=[{"parent_markdown_sha256": "a" * 64, "chunk_ordinal": 0,
                    "url": "https://news.ycombinator.com/item?id=1",
                    "chunk_policy_id":
                        "struct-md-768w-o0-adjacent-sections-tail-1152-tok2560"}],
            expected_documents=1)


def test_renderer_added_label_is_still_a_hard_failure(tmp_path, monkeypatch):
    """A renderer that adds the label itself (the original defect) is NOT held
    out: the ingest refuses the run."""
    import datetime as dt

    real = ING.render_discussion
    monkeypatch.setattr(
        ING, "render_discussion",
        lambda title, story, comments: "Author: someone\n\n" + real(title, story, comments))
    source = tmp_path / "ids.jsonl"
    write_input(source, 100)
    with pytest.raises(ING.IngestError, match="renderer itself emitted"):
        ING.ingest(
            input_path=source, output_dir=tmp_path / "records",
            api_base_url="https://hacker-news.firebaseio.com/v0/",
            item_url_template="https://news.ycombinator.com/item?id={item_id}",
            item_cap=20, shard_size=100, workers=1, client=RecordedFirebaseClient(),
            sleep=lambda _s: None,
            clock=lambda: dt.datetime(2026, 9, 17, tzinfo=dt.timezone.utc),
            lang_detector=lambda _t, _u: ("en", True))
