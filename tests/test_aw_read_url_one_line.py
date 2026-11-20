"""Task 2749: fedlex filestore HTML is ONE line in every language (measured 2026-11-12, 2.6 MB, 1-3 newlines).

`read_url` therefore keeps the HTML twin; the line-blind part is solved by the fragment trim in
`cyt read`. These fixtures are shrunk copies with the live shape: one DOCTYPE line, then one line."""
import pytest

from cytir.reader.extract import extract
from cytir.serve import read_url as RU
from tests.helpers.repo_files import repo_file_or_skip

TABLE = {"_langs": ["de", "fr", "it", "en"], "cc/27/317_321_377": "20261001"}
LANGS = ["de", "fr", "it", "en"]


def _fixture(lang):
    return repo_file_or_skip(f"tests/fixtures/fedlex_one_line/or-{lang}.html").read_text(encoding="utf8")


@pytest.mark.parametrize("lang", LANGS)
def test_twin_is_same_html_variant_and_one_line_in_every_language(lang):
    url = f"https://www.fedlex.admin.ch/eli/cc/27/317_321_377/{lang}#art_336_c"
    twin = RU.read_url_for({"url": url}, TABLE)
    assert twin.endswith(f"-20261001-{lang}-html.html#art_336_c")
    assert _fixture(lang).count("\n") == 1          # the shape that truncates in a line-based fetch tool


@pytest.mark.parametrize("lang", LANGS)
def test_fragment_trim_gives_the_article_not_the_one_line_law(lang):
    text, anchored = extract(_fixture(lang), f"https://x/{lang}#art_336_c")
    assert anchored and "Art. 336c" in text and "filler" not in text and "after" not in text
