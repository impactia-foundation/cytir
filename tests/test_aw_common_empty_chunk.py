import hashlib

import cytir.core.common as AC


EMPTY = {
    "text": "",
    "url": "https://en.wikipedia.org/wiki/List_of_places_in_California_(G)",
    "point_id": "example",
    "chunk_n_words": 0,
    "chunk_ordinal": 2,
    "chunk_unit": "adjacent_sections_composed_token_part",
    "chunk_rule": "struct-md-768w-o0-adjacent-sections-tail-1152@3",
    "content_digest": hashlib.sha256(b"").hexdigest(),
}


def test_known_zero_word_tail_is_ledgerable():
    result = AC.empty_chunk_quarantine(dict(EMPTY))
    assert result is not None
    assert result["point_id"] == "example"


def test_empty_chunk_predicate_stays_narrow():
    mutations = (
        {"text": "content"},
        {"url": ""},
        {"chunk_n_words": 1},
        {"content_digest": "0" * 64},
    )
    for mutation in mutations:
        assert AC.empty_chunk_quarantine({**EMPTY, **mutation}) is None
