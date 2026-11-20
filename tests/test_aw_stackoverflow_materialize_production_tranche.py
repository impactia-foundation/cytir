import io

import pytest

import cytir.sources.stackoverflow.materialize_production_tranche as subject


def test_only_explicit_first_through_seventh_bounded_frontier_is_authorized():
    subject._require_authorized_frontier(
        {"status": "exact_frontier_complete_first_tranche_authorized"})
    subject._require_authorized_frontier(
        {"status": "exact_second_tranche_authorized_only"})
    subject._require_authorized_frontier(
        {"status": "exact_third_tranche_authorized_only"})
    subject._require_authorized_frontier(
        {"status": "exact_fourth_tranche_authorized_only"})
    subject._require_authorized_frontier(
        {"status": "exact_fifth_tranche_authorized_only"})
    subject._require_authorized_frontier(
        {"status": "exact_sixth_tranche_authorized_only"})
    subject._require_authorized_frontier(
        {"status": "exact_seventh_tranche_authorized_only"})
    with pytest.raises(ValueError, match="not terminal"):
        subject._require_authorized_frontier({"status": "exact_frontier_complete"})


def test_the_generic_nth_tranche_authorizer_status_is_also_accepted():
    """Task 663, 2026-09-25 regression: tranche 8's real authorize run (the
    generic `authorize_stackoverflow_qa_nth_tranche.py`, PR #295) produced
    `exact_tranche_08_authorized_only`, and this allowlist -- never updated
    when the generic script shipped -- refused it with a bare "not
    terminal", stalling tranche 8's materialize step. Ordinal 8+ follows a
    PATTERN (`exact_tranche_{ordinal:02d}_authorized_only`), not a new
    literal per tranche -- this test covers several ordinals, not just 8,
    so a regression at tranche 9+ fails here too."""
    for ordinal in (8, 9, 12, 99):
        subject._require_authorized_frontier(
            {"status": f"exact_tranche_{ordinal:02d}_authorized_only"})
    # still refuses garbage that merely LOOKS close to the pattern
    with pytest.raises(ValueError, match="not terminal"):
        subject._require_authorized_frontier(
            {"status": "exact_tranche_08_authorized_only_extra"})
    with pytest.raises(ValueError, match="not terminal"):
        subject._require_authorized_frontier(
            {"status": "exact_tranche_authorized_only"})
    with pytest.raises(ValueError, match="not terminal"):
        subject._require_authorized_frontier({"status": None})


def _row(**attrs):
    value = " ".join(f'{key}="{value}"' for key, value in attrs.items())
    return f"  <row {value} />\n".encode()


def test_exact_post_selection_preserves_full_rows_and_frontier_identity():
    q = _row(Id="10", PostTypeId="1", AcceptedAnswerId="20", Score="10",
             Body="&lt;p&gt;question&lt;/p&gt;", Title="title",
             ContentLicense="CC BY-SA 4.0", OwnerUserId="1")
    a = _row(Id="20", PostTypeId="2", ParentId="10", Score="3",
             Body="&lt;p&gt;answer&lt;/p&gt;", ContentLicense="CC BY-SA 4.0",
             OwnerUserId="2")
    from cytir.extract.stackoverflow_pilot import _xml_rows
    parsed = {attrs["Id"]: attrs for _, attrs in _xml_rows(io.BytesIO(q + a))}
    from cytir.acquire.artifact_io import canonical_json
    import hashlib
    frontier = [{"question_id": 10, "accepted_answer_id": 20,
                 "selection_key": "a" * 64,
                 "question_attrs_sha256": hashlib.sha256(canonical_json(parsed["10"]).encode()).hexdigest(),
                 "answer_attrs_sha256": hashlib.sha256(canonical_json(parsed["20"]).encode()).hexdigest()}]
    value = subject.select_exact_posts(io.BytesIO(q + a), frontier=frontier)
    assert value["counts"] == {"selected_questions": 1, "selected_answers": 1}
    assert value["rows"][0]["question"]["raw_xml_bytes"] == len(q)
    assert value["rows"][0]["accepted_answer"]["raw_xml_bytes"] == len(a)
    assert value["rows"][0]["cell"] == "accepted_qscore_10_plus"


def test_exact_post_selection_fails_closed_on_missing_answer():
    q = _row(Id="10", PostTypeId="1", AcceptedAnswerId="20", Score="10")
    from cytir.extract.stackoverflow_pilot import _xml_rows
    from cytir.acquire.artifact_io import canonical_json
    import hashlib
    attrs = next(_xml_rows(io.BytesIO(q)))[1]
    frontier = [{"question_id": 10, "accepted_answer_id": 20,
                 "selection_key": "b" * 64,
                 "question_attrs_sha256": hashlib.sha256(canonical_json(attrs).encode()).hexdigest(),
                 "answer_attrs_sha256": "c" * 64}]
    with pytest.raises(ValueError, match="source rows are incomplete"):
        subject.select_exact_posts(io.BytesIO(q), frontier=frontier)


def test_adapter_has_no_source_content_cap():
    source = subject.Path(subject.__file__).read_text()
    assert "[:8000]" not in source
    assert "source_content_cap\": None" in source
