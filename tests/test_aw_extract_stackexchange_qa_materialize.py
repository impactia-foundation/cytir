import gzip
import json
from pathlib import Path

import pytest

from cytir.extract.stackexchange_qa import build, materialize, _owner_user_id


FIXTURE = Path(__file__).resolve().parent / "fixtures/security_se_sample200.jsonl"


def _real_rows():
    with open(FIXTURE, "r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def test_owner_user_id_parses_the_real_permalink_shape():
    assert _owner_user_id("https://security.stackexchange.com/users/12144") == "12144"
    assert _owner_user_id(None) is None
    assert _owner_user_id("") is None


def test_materialize_on_real_task226_rows_emits_one_document_per_row_no_quarantine():
    rows = _real_rows()
    result = materialize(rows, site="security.stackexchange.com")
    # Every real row in the sample has a license, title, markdown text, url,
    # and either a display name or a resolvable numeric user id -- nothing
    # should quarantine on this real slice.
    assert result["metrics"]["quarantine"] == 0
    assert len(result["markdown"]) == len(rows)
    doc_ids = {row["document_id"] for row in result["markdown"]}
    assert len(doc_ids) == len(rows)  # one row per (question, single answer) -> unique
    # Real deleted-user edge case (master_author_url null) must still carry a
    # display name and a null owner_user_id, not a fabricated one.
    deleted_user_rows = [r for r in result["markdown"]
                         if r["accepted_answer"]["attribution"]["owner_profile_url"] is None]
    assert deleted_user_rows
    for row in deleted_user_rows:
        attribution = row["accepted_answer"]["attribution"]
        assert attribution["owner_user_id"] is None
        assert attribution["owner_display_name"]
    # A real accepted-answer row and a real non-accepted (top) row both exist
    # in the sample; is_accepted must reflect the source row, not default true.
    accepted_flags = {row["accepted_answer"]["is_accepted"] for row in result["markdown"]}
    assert accepted_flags == {True, False}
    # license travels through, exactly as read (one of the three ADR-039 bands).
    licenses = {row["accepted_answer"]["license"] for row in result["markdown"]}
    assert licenses <= {"CC-BY-SA-2.5", "CC-BY-SA-3.0", "CC-BY-SA-4.0"}


def test_materialize_quarantines_a_row_with_no_license():
    rows = _real_rows()[:2]
    rows[0] = dict(rows[0])
    rows[0]["license"] = "unknown"
    result = materialize(rows, site="security.stackexchange.com")
    assert result["metrics"]["quarantine"] == 1
    assert result["quarantine"][0]["reason"] == "missing_or_unrecognized_license"
    assert len(result["markdown"]) == 1


def test_build_seals_a_real_run_against_the_actual_sample_and_a_stub_dump(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    passages = artifacts / "security.stackexchange.com.passages.jsonl"
    passages.write_bytes(FIXTURE.read_bytes())
    dump = artifacts / "security.stackexchange.com.7z"
    dump.write_bytes(b"stub-dump-bytes-for-sha256-evidence-only")
    root = artifacts / "site-run"

    run = build(site="security.stackexchange.com", passages_jsonl=passages,
               dump=dump, root=root, artifact_root=artifacts,
               expected_input_rows=200)

    assert run["status"] == "source_native_site_complete_pending_profile_fidelity_chunk_measurement"
    assert run["counts"]["selected_source_rows"] == 200
    assert run["counts"]["markdown_documents"] == 200
    assert run["counts"]["quarantine"] == 0
    # every stage's receipt must independently validate against the artifact root
    from cytir.index.preservation_contract import _validate_stage
    for name, stage in run["stages"].items():
        _validate_stage(stage, f"test.{name}", artifact_root=artifacts, deep=True)

    with gzip.open(root / "data/markdown.jsonl.gz", "rt", encoding="utf-8") as source:
        markdown_rows = [json.loads(line) for line in source]
    assert len(markdown_rows) == 200
    for row in markdown_rows:
        assert row["markdown"]
        assert row["url"].startswith("https://security.stackexchange.com/a/")
        assert row["document_id"]


def test_build_windows_are_disjoint_and_exhaustive_and_cheaper_than_the_whole_file(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    passages = artifacts / "security.stackexchange.com.passages.jsonl"
    passages.write_bytes(FIXTURE.read_bytes())
    dump = artifacts / "security.stackexchange.com.7z"
    dump.write_bytes(b"stub-dump-bytes-for-sha256-evidence-only")

    run_a = build(site="security.stackexchange.com", passages_jsonl=passages,
                 dump=dump, root=artifacts / "window-a", artifact_root=artifacts,
                 expected_input_rows=120, offset=0, limit=120,
                 expected_total_rows=200)
    run_b = build(site="security.stackexchange.com", passages_jsonl=passages,
                 dump=dump, root=artifacts / "window-b", artifact_root=artifacts,
                 expected_input_rows=80, offset=120, limit=120,
                 expected_total_rows=200)

    assert run_a["counts"]["selected_source_rows"] == 120
    assert run_b["counts"]["selected_source_rows"] == 80

    def _doc_ids(root: Path) -> set[str]:
        with gzip.open(root / "data/markdown.jsonl.gz", "rt", encoding="utf-8") as source:
            return {json.loads(line)["document_id"] for line in source}

    ids_a, ids_b = _doc_ids(artifacts / "window-a"), _doc_ids(artifacts / "window-b")
    assert not (ids_a & ids_b)  # disjoint
    assert len(ids_a) + len(ids_b) == 200  # exhaustive over the real 200-row sample

    # the preserved original-evidence copy holds only this window's rows, not
    # the whole shared source file -- the point of windowing at all.
    with gzip.open(artifacts / "window-a/data/source-passages.jsonl.gz", "rt",
                   encoding="utf-8") as source:
        assert sum(1 for _ in source) == 120

    from cytir.index.preservation_contract import _validate_stage
    for name, stage in run_a["stages"].items():
        _validate_stage(stage, f"test.window_a.{name}", artifact_root=artifacts, deep=True)


def test_build_refuses_a_total_row_count_that_drifted_from_expectation(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    passages = artifacts / "security.stackexchange.com.passages.jsonl"
    passages.write_bytes(FIXTURE.read_bytes())
    dump = artifacts / "security.stackexchange.com.7z"
    dump.write_bytes(b"stub")
    root = artifacts / "site-run"
    with pytest.raises(ValueError, match="source file row conservation failed"):
        build(site="security.stackexchange.com", passages_jsonl=passages,
             dump=dump, root=root, artifact_root=artifacts,
             expected_input_rows=120, offset=0, limit=120,
             expected_total_rows=199)


def test_build_refuses_a_row_count_that_drifted_from_expectation(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    passages = artifacts / "security.stackexchange.com.passages.jsonl"
    passages.write_bytes(FIXTURE.read_bytes())
    dump = artifacts / "security.stackexchange.com.7z"
    dump.write_bytes(b"stub")
    root = artifacts / "site-run"
    with pytest.raises(ValueError, match="row conservation failed"):
        build(site="security.stackexchange.com", passages_jsonl=passages,
             dump=dump, root=root, artifact_root=artifacts,
             expected_input_rows=199)
