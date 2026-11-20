from cytir.extract.routec import iana_registries as subject


def test_renderer_preserves_registry_metadata_tables_and_links():
    html = '''<html><head><title>Example Registry</title></head><body><article>
      <h1>Example Registry</h1><dl><dt>Last Updated</dt><dd>2026-08-16</dd></dl>
      <table><tr><th>Name</th><th>Value</th></tr><tr><td>alpha</td><td>42</td></tr></table>
      <p>See <a href="/go/rfc1234">RFC 1234</a>.</p></article></body></html>'''
    title, markdown, version = subject.render_registry_markdown(
        html, url="https://www.iana.org/assignments/example/example.xhtml",
        body_sha256="a" * 64, transport_kind="fixture")
    assert title == "Example Registry"
    assert "Name | Value" in markdown and "alpha | 42" in markdown
    assert "Last Updated" in markdown and "2026-08-16" in markdown
    assert "RFC 1234" in markdown
    assert version["record_last_updated"] == "2026-08-16"


def test_renderer_rejects_non_record_navigation_page():
    try:
        subject.render_registry_markdown(
            "<html><body><nav>links only</nav></body></html>",
            url="https://www.iana.org/", body_sha256="b" * 64,
            transport_kind="fixture")
    except ValueError as error:
        assert "main/article" in str(error)
    else:
        raise AssertionError("navigation-only page was admitted")
