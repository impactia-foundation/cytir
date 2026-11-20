from cytir.extract import bfs_dam_current as M


def asset():
    return {
        "ids": {"damId": 42, "uuid": "source-uuid", "contentId": 9,
                "languageCopyId": 10, "gnp": "GNP-1"},
        "bfs": {"lifecycle": {"name": "Current", "code": "CURRENT"},
                "articleModel": {"name": "Table", "code": "TABLE"},
                "embargo": "2026-01-01T00:00:00Z", "provisional": False},
        "description": {
            "language": "EN",
            "titles": {"main": "Population", "sub": "Annual"},
            "bibliography": {"period": "2025"},
            "categorization": {
                "prodima": [{"name": "Population", "code": "01"}]},
        },
        "shop": {"orderNr": "asset-42", "stock": False},
        "links": [{"href": "/statistics/population", "rel": "theme",
                   "text": "Population topic"}],
    }


def test_raw_key_uses_parser_compatible_first_language():
    value = asset()
    value["description"]["language"] = "EN/DE"
    assert M.raw_key(value) == (
        "https://www.bfs.admin.ch/asset/en/42", "Population", "en")


def test_render_preserves_source_identity_semantics_and_links():
    url = "https://www.bfs.admin.ch/asset/en/42"
    markdown = M.render(asset(), {"title": "Population", "text": "Full summary."},
                        url, "en")
    for expected in ("# Population", "## Record identity", "BFS DAM asset ID: 42",
                     "Full summary.", "## Publication metadata",
                     "## Classifications", "Population (01)",
                     "## Source links", "Population topic"):
        assert expected in markdown
    assert markdown.endswith("\n")


def test_source_links_resolve_relative_targets_and_add_canonical_parent():
    url = "https://www.bfs.admin.ch/asset/en/42"
    links = M.source_links(asset(), url)
    assert links[0] == {"url": url, "relation": "canonical-record",
                        "context": "BFS DAM asset page"}
    assert links[1] == {
        "url": "https://www.bfs.admin.ch/statistics/population",
        "relation": "theme", "context": "Population topic"}
