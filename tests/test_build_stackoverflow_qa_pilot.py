from __future__ import annotations

import io
import xml.sax.saxutils as saxutils

import cytir.extract.stackoverflow_pilot as subject


def _row(**attrs: str) -> bytes:
    encoded = " ".join(f'{key}="{saxutils.escape(str(value), {chr(34): "&quot;"}).replace(chr(10), "&#10;")}"'
                       for key, value in attrs.items())
    return f"  <row {encoded} />\n".encode()


def test_score_cells_exact_rows_full_content_and_prior_arm():
    long_code = "x = 1\n" * 3000
    rows = [b"<posts>\n"]
    questions = [("1", "2", "0"), ("3", "4", "2"),
                 ("5", "6", "5"), ("7", "8", "10")]
    for question, answer, score in questions:
        rows.append(_row(Id=question, PostTypeId="1", AcceptedAnswerId=answer,
                         Score=score, OwnerUserId="10", CreationDate="2020-01-01",
                         ContentLicense="CC BY-SA 4.0", Title=f"Question {question}",
                         Tags="<python>", Body="<p>Full question</p>"))
        body = f'<pre><code>{long_code}</code></pre><p><a href="https://example.org/ref">ref</a></p>'
        rows.append(_row(Id=answer, PostTypeId="2", ParentId=question,
                         Score="4", OwnerUserId="11", CreationDate="2020-01-02",
                         ContentLicense="CC BY-SA 4.0", Body=body))
    rows.append(b"</posts>\n")
    selection = subject.select_posts(io.BytesIO(b"".join(rows)), sample_per_cell=1)
    assert selection["counts"]["selected_questions"] == 4
    assert selection["counts"]["selected_with_accepted_answer_present"] == 4
    assert selection["counts"]["prior_arm_included"] == 4
    assert {row["cell"] for row in selection["rows"]} == set(subject.CELLS)
    assert all(row["question"]["raw_xml_sha256"] for row in selection["rows"])

    users = subject.select_users(io.BytesIO(
        b"<users>\n" +
        _row(Id="10", DisplayName="Questioner") +
        _row(Id="11", DisplayName="Answerer") + b"</users>\n"),
        owner_ids={"10", "11"})
    result = subject.materialize(selection, users)
    assert len(result["markdown"]) == 4
    assert not result["quarantine"]
    assert long_code in result["markdown"][0]["markdown"]
    assert len(result["markdown"][0]["markdown"]) > 8000
    assert all(row["question"]["license"] == "CC BY-SA 4.0"
               for row in result["markdown"])
    assert all(row["lang"] == "en" and len(row["source_links"]) == 2
               for row in result["markdown"])
    relations = {row["relation"] for row in result["outlinks"]}
    assert {"canonical-question", "accepted-answer", "attribution-profile",
            "cited-link"} <= relations


def test_bottom_k_is_deterministic_and_byte_exact():
    rows = [b"<posts>\n"]
    raw_by_question = {}
    for ordinal in range(20):
        question = str(100 + ordinal * 2)
        answer = str(101 + ordinal * 2)
        raw = _row(Id=question, PostTypeId="1", AcceptedAnswerId=answer,
                   Score="10", OwnerDisplayName="q", CreationDate="2020-01-01",
                   ContentLicense="CC BY-SA 4.0", Title=question, Body="<p>Q</p>")
        raw_by_question[question] = raw
        rows.extend([raw, _row(Id=answer, PostTypeId="2", ParentId=question,
                               Score="1", OwnerDisplayName="a", CreationDate="2020-01-02",
                               ContentLicense="CC BY-SA 4.0", Body="<p>A</p>")])
    rows.append(b"</posts>\n")
    payload = b"".join(rows)
    first = subject.select_posts(io.BytesIO(payload), sample_per_cell=3)
    second = subject.select_posts(io.BytesIO(payload), sample_per_cell=3)
    assert first == second
    assert len(first["rows"]) == 3
    for row in first["rows"]:
        qid = row["question"]["attributes"]["Id"]
        import base64
        assert base64.b64decode(row["question"]["raw_xml_base64"]) == raw_by_question[qid]


def test_missing_attribution_is_explicit_quarantine():
    posts = b"<posts>\n" + _row(
        Id="1", PostTypeId="1", AcceptedAnswerId="2", Score="10",
        CreationDate="2020", ContentLicense="CC BY-SA 4.0", Title="Q",
        Body="<p>Question</p>") + _row(
        Id="2", PostTypeId="2", ParentId="1", Score="4",
        CreationDate="2020", ContentLicense="CC BY-SA 4.0",
        Body="<p>Answer</p>") + b"</posts>\n"
    selection = subject.select_posts(io.BytesIO(posts), sample_per_cell=1)
    result = subject.materialize(selection, {"users": {}, "counts": {}})
    assert result["markdown"] == []
    assert result["quarantine"] == [{
        "question_id": "1", "reason": "irreversible_attribution_identity_missing"}]


def test_malformed_source_href_is_preserved_explicitly_without_aborting_document():
    fragment = '<p>See <a href="http://[bad">historical reference</a>.</p>'
    markdown = subject.html_fragment_to_markdown(
        fragment, "https://stackoverflow.com/questions/1")
    assert "historical reference" in markdown
    assert "malformed source href" in markdown
    assert "http://[bad" in markdown
    assert subject._links(fragment, "https://stackoverflow.com/questions/1") == [{
        "target_url": "http://[bad",
        "anchor_or_context": "historical reference",
        "relation": "malformed-source-link",
    }]


def test_negative_score_accepted_question_is_measured_not_sampled():
    posts = b"<posts>\n" + _row(
        Id="1", PostTypeId="1", AcceptedAnswerId="2", Score="-2",
        OwnerDisplayName="q", CreationDate="2020",
        ContentLicense="CC BY-SA 4.0", Title="Q", Body="<p>Q</p>") + _row(
        Id="2", PostTypeId="2", ParentId="1", Score="4",
        OwnerDisplayName="a", CreationDate="2020",
        ContentLicense="CC BY-SA 4.0", Body="<p>A</p>") + b"</posts>\n"
    selection = subject.select_posts(io.BytesIO(posts), sample_per_cell=1)
    assert selection["rows"] == []
    assert selection["counts"]["accepted_questions_negative_score_excluded"] == 1
