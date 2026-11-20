import hashlib
import pytest

from cytir.extract.routec.ietf_datatracker_reviews import render_review_record


HTML="""<html><head><title>Last Call Review of draft-example-02</title></head><body><main>
<h1>Last Call Review review-draft-example-02-genart-lc-user-2026-08-16-00</h1>
<pre>Complete review text with findings.</pre></main></body></html>"""


def test_review_renderer_preserves_complete_displayed_review_text():
    url="https://datatracker.ietf.org/doc/review-draft-example-02-genart-lc-user-2026-08-16"
    title,markdown,version=render_review_record(HTML,url=url,final_url=url+"/",body_sha256=hashlib.sha256(HTML.encode()).hexdigest())
    assert title.startswith("Last Call Review")
    assert "Complete review text with findings." in markdown
    assert version["review_name"].startswith("review-draft-example")


def test_review_renderer_rejects_non_review_record():
    with pytest.raises(ValueError,match="review identity absent"):
        render_review_record("<html><body><main><h1>Agenda</h1></main></body></html>",
            url="https://datatracker.ietf.org/doc/agenda-100",final_url="https://datatracker.ietf.org/doc/agenda-100/",body_sha256="a"*64)
