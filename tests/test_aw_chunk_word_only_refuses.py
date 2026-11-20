"""task 2395: generic `chunk_markdown` refuses word-only chunking unless asked."""
import pytest

from cytir.chunk import chunker as CH

DOC = "# Title\n\n" + "\n\n".join(f"## S{i}\n\n" + ("word " * 200) for i in range(4))


@pytest.fixture
def no_tokenizer(monkeypatch):
    """Force the 'unavailable' state whatever is installed on this box."""
    monkeypatch.setattr(CH, "_ALLOW_WORD_ONLY_FOR_TESTS", False)
    monkeypatch.setattr(CH, "_tokenizer_cache", {})
    monkeypatch.setitem(CH._TOKENIZER_STATE, "status", "unavailable")


def test_unavailable_without_flag_raises_named_exception(no_tokenizer):
    with pytest.raises(CH.TokenizerUnavailable) as e:
        CH.chunk_markdown(DOC)
    assert "tokenizer_status='unavailable'" in str(e.value)
    with pytest.raises(CH.TokenizerUnavailable):
        CH.chunk_texts(DOC)


def test_unavailable_with_flag_gives_word_only_result(no_tokenizer):
    out = CH.chunk_markdown(DOC, allow_word_only=True)
    assert out and CH.tokenizer_status() == "unavailable"


def test_tokenizer_ok_output_is_unchanged(monkeypatch):
    class Tok:  # stand-in tokenizer: 1 token per word, far under any ceiling
        pass
    monkeypatch.setattr(CH, "_ALLOW_WORD_ONLY_FOR_TESTS", False)
    monkeypatch.setattr(CH, "_tokenizer_cache", {(CH.TOKENIZER_MODEL, CH.TOKENIZER_REVISION): Tok()})
    monkeypatch.setitem(CH._TOKENIZER_STATE, "status", "ok")
    monkeypatch.setattr("cytir.chunk.token_counter.count", lambda tok, t: len(t.split()))
    ref = CH.chunk_markdown(DOC, allow_word_only=True)
    assert [c.text for c in CH.chunk_markdown(DOC)] == [c.text for c in ref]
