"""Q5 label classification: prose line vs renderer-added label."""
from cytir.sources.hn import identity_label as IDL


def test_clean_text_is_clean():
    assert IDL.classify_discussion("Title\n\nfirst comment", ["Title", "first comment"]) == IDL.CLEAN


def test_prose_line_in_a_source_block_is_prose():
    blocks = ["Title", "quantum text", "Author: Prof. Ross of Southampton"]
    assert IDL.classify_discussion("\n\n".join(blocks), blocks) == IDL.PROSE


def test_label_only_in_the_composition_is_renderer():
    blocks = ["Title", "comment"]
    assert IDL.classify_discussion("Author: bob\n\n" + "\n\n".join(blocks), blocks) == IDL.RENDERER


def test_mid_sentence_mention_is_not_a_label():
    assert IDL.find_label("the author: said so") is None
    assert IDL.find_label("x\n\nBy: me") is not None


def test_match_stitched_across_a_block_join_holds_out_with_a_named_reason():
    # Task 1445 case (i): `\s*` spans the "\n\n" join of two source blocks.
    for blocks in (["Title", "By", ": quoted"], ["Title", "see Author:", "Author:", "Bob"]):
        rendered = "\n\n".join(blocks)
        assert IDL.classify_discussion(rendered, blocks) == IDL.RENDERER  # old behaviour
        assert IDL.classify_with_reason(rendered, blocks) == (
            IDL.PROSE, IDL.REASON_SPANS_BLOCKS)


def test_renderer_label_still_fails_even_next_to_a_stitched_match():
    blocks = ["Title", "By", ": quoted"]
    rendered = "Author: bob\n\n" + "\n\n".join(blocks)
    assert IDL.classify_with_reason(rendered, blocks)[0] == IDL.RENDERER
    plain = ["Title", "comment"]
    assert IDL.classify_with_reason("Author: bob\n\n" + "\n\n".join(plain), plain)[0] == IDL.RENDERER


def test_prose_reason_and_clean_unchanged():
    blocks = ["Title", "Author: Prof. X"]
    assert IDL.classify_with_reason("\n\n".join(blocks), blocks) == (IDL.PROSE, IDL.REASON_PROSE)
    assert IDL.classify_with_reason("Title\n\nhi", ["Title", "hi"]) == (IDL.CLEAN, None)


def test_renderer_label_right_after_a_stitched_match_still_fails():
    rendered = "T\n\nBy\n\n:\n\nAuthor: bob"
    assert IDL.classify_with_reason(rendered, ["T", "By", ":"])[0] == IDL.RENDERER


def test_pure_stitched_case_still_holds_out():
    rendered = "T\n\nBy\n\n:\n\nx"
    assert IDL.classify_with_reason(rendered, ["T", "By", ":", "x"]) == (
        IDL.PROSE, IDL.REASON_SPANS_BLOCKS)


def test_label_heading_its_own_block_matches_from_the_label_not_the_blank_line():
    # Task 1539 (1): `^\s*` used to start the match at the blank line, so the
    # matched text began with a newline and looked like part of the first block.
    assert IDL.find_label("T\n\nBy: alice") == "By: a"
    assert IDL.find_label("T\n\n\n\nAuthor: x") == "Author: x"


def test_label_heading_second_block_is_renderer_when_only_the_composition_has_it():
    assert IDL.classify_with_reason("T\n\nBy: alice", ["T"]) == (IDL.RENDERER, None)


def test_label_heading_second_block_is_prose_when_a_source_block_carries_it():
    for blocks in (["T", "By: alice"], ["T\n\nBy: alice"]):
        assert IDL.classify_with_reason("T\n\nBy: alice", blocks) == (IDL.PROSE, IDL.REASON_PROSE)
