import hashlib

import pytest

from cytir.extract.routec import ietf_meeting_records as subject


@pytest.mark.parametrize(("kind", "url", "needle"), [
    ("agenda", "https://datatracker.ietf.org/doc/agenda-114-example/02", "Full agenda item"),
    ("minutes", "https://datatracker.ietf.org/doc/minutes-114-example", "Full minutes text"),
])
def test_render_preserves_full_displayed_record(kind: str, url: str, needle: str) -> None:
    html = (f"<html><head><title>{kind.title()} IETF114: example</title></head>"
            f"<body><main><h1>{kind.title()} IETF114: example</h1>"
            f"<pre>{needle}</pre></main></body></html>")
    digest = hashlib.sha256(html.encode()).hexdigest()
    title, markdown, version = subject.render_meeting_record(
        html, kind=kind, url=url, final_url=url+"/", body_sha256=digest)
    assert kind.title() in title
    assert needle in markdown
    assert version["record_kind"] == kind
    assert version["requested_url"] == url


def test_minutes_selector_excludes_narrative_minutes() -> None:
    pattern = subject.CONFIG["minutes"]["pattern"]
    assert pattern.fullmatch("https://datatracker.ietf.org/doc/minutes-114-example")
    assert not pattern.fullmatch(
        "https://datatracker.ietf.org/doc/narrative-minutes-interim-2024-iesg")


def test_untitled_agenda_is_identified_by_displayed_heading() -> None:
    url = "https://datatracker.ietf.org/doc/agenda-interim-2021-roll-02-roll-01"
    html = ("<html><head><title>Untitled</title></head><body><main>"
            "<h1>(Untitled) agenda-interim-2021-roll-02-roll-01-01</h1>"
            "<pre>Full source agenda.</pre></main></body></html>")
    digest = hashlib.sha256(html.encode()).hexdigest()
    _, markdown, _ = subject.render_meeting_record(
        html, kind="agenda", url=url, final_url=url+"/", body_sha256=digest)
    assert "Full source agenda." in markdown


@pytest.mark.parametrize(("kind", "url", "heading"), [
    ("charter", "https://datatracker.ietf.org/doc/charter-ietf-example",
     "Example charter-ietf-example-01"),
    ("bof_request", "https://datatracker.ietf.org/doc/bofreq-example",
     "Example bofreq-example-02"),
    ("narrative_minutes",
     "https://datatracker.ietf.org/doc/narrative-minutes-interim-2024-iesg",
     "Example narrative-minutes-interim-2024-iesg-01"),
    ("conflict_review",
     "https://datatracker.ietf.org/doc/conflict-review-example",
     "IETF conflict review for draft-example conflict-review-example-00"),
])
def test_governance_record_identity_is_source_native(kind: str, url: str,
                                                      heading: str) -> None:
    html = (f"<html><head><title>Example</title></head><body><main><h1>{heading}</h1>"
            f"<pre>Full {kind} source text.</pre></main></body></html>")
    digest = hashlib.sha256(html.encode()).hexdigest()
    _, markdown, version = subject.render_meeting_record(
        html, kind=kind, url=url, final_url=url+"/", body_sha256=digest)
    assert f"Full {kind} source text." in markdown
    assert version["record_kind"] == kind
