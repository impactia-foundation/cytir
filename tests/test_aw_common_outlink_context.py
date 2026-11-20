from bs4 import BeautifulSoup

from cytir.core.common import extract_outlink_context_records


def test_context_outlinks_share_canonical_first_occurrence_contract():
    soup = BeautifulSoup("""
      <a href="/inside" rel="nofollow">Internal guide</a>
      <a href="https://Other.Example/path?utm_source=x">External source</a>
      <a href="https://other.example/path">Later duplicate</a>
      <a href="mailto:no@example.test">Email</a>
    """, "html.parser")
    rows = extract_outlink_context_records(
        soup, "https://authority.example/page")
    assert rows == [
        {"url": "https://authority.example/inside", "nofollow": True,
         "anchor_or_context": "Internal guide"},
        {"url": "https://other.example/path", "nofollow": False,
         "anchor_or_context": "External source"},
    ]

