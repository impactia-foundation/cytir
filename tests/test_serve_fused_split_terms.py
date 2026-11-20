import pytest

from cytir.serve.article_tokens import glued_article_variants
from cytir.serve.fused_split_terms import MAX_VARIANTS, fused_split_variants as fsv
from cytir.serve.schema import bm25_tokens
from cytir.serve.settings import Query


def test_flag_is_off_by_default():
    assert Query.__dataclass_fields__["query_fused_split"].metadata["default"] == "0"


@pytest.mark.parametrize("q,added", [
    ("Art. 336c illness", ["336"]),
    ("RFC8446 tls", []),  # letter-first: ONE token on both sides, untouched
    ("iso27001 controls", []),
    ("art336 OR", []),
    ("Cas9 knockout", []),  # the split `cas` moved gx11 6 -> 8
    ("ipv4 address", []),
    ("10gb", ["gb", "10"]),
    ("4bis", ["bis"]),
    ("§ 5a ZGB", []),  # the lone letter and the one-digit part are both dropped by the tokenizer
    ("rfc 8446", []),  # letter-first fuse dropped: it lost golds (ts11, ts13, ts39, ts44)
    ("Art. 336 c", ["336c"]),
    ("art 4 bis", ["4bis"]),
    ("article 41 du code", []),
    ("iso 27001", []),
    ("SR 220", []),
    ("art.336c", ["336"]),
])
def test_cases(q, added):
    assert fsv(q) == (f"{q} {' '.join(added)}" if added else q)


@pytest.mark.parametrize("q", [
    "pi is 3.14", "date 2024-11-16", "version v2.1 notes", "2.5g protein", "1.2.3b", "a3f9c2e7b1d4",
    "h264a codec", "sqlite3 bs4 pop3", "c 336", "kündigungsfrist arbeitsvertrag", "", "12 34", "ab cd",
    "10:30 am",
])
def test_no_explosion(q):
    assert fsv(q) == q


def test_cap_and_no_tf_bump():
    q = " ".join(f"{i}ab" for i in range(30))
    assert len(fsv(q).split()) - 30 <= MAX_VARIANTS
    assert fsv("336 336c") == "336 336c"  # `336` already present, not duplicated


def test_original_tokens_always_stay():
    for q in ("Art. 336c", "rfc 8446", "iso27001", "Art. 336 c"):
        assert set(bm25_tokens(q, "de")) <= set(bm25_tokens(fsv(q), "de"))


def test_german_decompounding_adds_only_the_variants():
    off = bm25_tokens("kündigungsfrist art 336c", "de")
    on = bm25_tokens(fsv("kündigungsfrist art 336c"), "de")
    assert set(off) <= set(on) and len(set(on) - set(off)) == 1  # `336`


@pytest.mark.parametrize("q", [
    "Art. 336c illness", "kündigungsfrist arbeitsvertrag art 335c schweiz", "art 264c", "BGB § 312g",
    "Artikel 41a OR", "article 17b CO", "art. 8a", "7B model", "3a pillar", "4o", "405B"])
def test_strict_superset_of_article_lever(q):
    for lang in ("de", "fr", "it", "en"):
        assert set(bm25_tokens(glued_article_variants(q), lang)) <= set(bm25_tokens(fsv(q), lang))
