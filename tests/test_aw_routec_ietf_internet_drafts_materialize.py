import hashlib

import pytest

from cytir.extract.routec.ietf_internet_drafts import render_draft_record


HTML = """<html><head><title>draft-example-02 - Example</title></head><body><main>
<h1>Example Protocol draft-example-02</h1><p>Abstract</p><p>Complete abstract.</p>
<a href="https://www.ietf.org/archive/id/draft-example-02.txt">txt</a>
</main></body></html>"""


def test_renderer_preserves_record_and_distinguishes_linked_text() -> None:
    url = "https://datatracker.ietf.org/doc/draft-example"
    title, markdown, version = render_draft_record(
        HTML, url=url, final_url=url + "/", body_sha256=hashlib.sha256(HTML.encode()).hexdigest())
    assert title == "draft-example-02 - Example"
    assert "Complete abstract." in markdown
    assert version["draft_name"] == "draft-example"
    assert version["displayed_version"] == "draft-example-02"


def test_renderer_rejects_non_draft_record() -> None:
    with pytest.raises(ValueError, match="draft identity absent"):
        render_draft_record("<html><body><main><h1>About</h1></main></body></html>",
            url="https://datatracker.ietf.org/about", final_url="https://datatracker.ietf.org/about/",
            body_sha256="a" * 64)


def test_current_rfc_redirect_is_not_rendered_as_a_draft_record() -> None:
    with pytest.raises(ValueError, match="draft identity absent"):
        render_draft_record("<html><body><main><h1>RFC 9852</h1></main></body></html>",
            url="https://datatracker.ietf.org/doc/draft-old",
            final_url="https://datatracker.ietf.org/doc/rfc9852/",
            body_sha256="b" * 64)
