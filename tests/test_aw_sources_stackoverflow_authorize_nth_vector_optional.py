"""Task 691: a prior tranche with a terminal stage chain but no vector rollup receipt.

The growth timer embeds and seals raw vector shards, not a rollup, so the generic authorizer
must accept `vector=None` for a prior -- and must say so in the decision, never omit it."""
import json
from pathlib import Path

import pytest

from cytir.acquire.artifact_io import sha256_file
from cytir.sources.stackoverflow.authorize_nth import build
from cytir.sources.stackoverflow.authorize_third import _validate_terminal_tranche
from test_aw_sources_stackoverflow_authorize_nth import (
    _dump, _rows, _terminal_tranche_evidence, _write_evidence)
from cytir.acquire import stage_receipt as SS


def test_prior_without_vector_rollup_is_accepted_and_recorded(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    source_root = artifacts / "source"
    source_root.mkdir(parents=True)
    frontier = source_root / "eligible.jsonl.gz"
    _rows(frontier, 20001)
    from cytir.sources.stackoverflow.authorize_second import _smallest
    ranked, _ = _smallest(frontier, 20000)
    tranche = _dump(source_root / "first.jsonl.gz", ranked[0:10000])
    stage = SS.seal(
        artifact_root=artifacts, artifact_paths=[frontier, tranche],
        receipt_path=source_root / "receipt.json", records=20001,
        fmt="fixture", logical_sha256=sha256_file(frontier),
        producer_id="fixture", producer_version="v1")
    frontier_run = source_root / "RUN.json"
    frontier_run.write_text(json.dumps({
        "schema": "aw.stackoverflow-qa-production-frontier-run.v1",
        "status": "exact_frontier_complete_first_tranche_authorized",
        "stages": {"frontier": stage}}, sort_keys=True))
    evidence = _write_evidence(source_root, "first", 13086)  # tranche 9's real chunk count
    evidence["vector"] = None
    result = build(frontier_run=frontier_run, frontier=frontier,
                   priors=[{"tranche": tranche, **evidence}], ordinal=2,
                   root=artifacts / "decision", artifact_root=artifacts)
    assert result["decision"]["selector"]["offset"] == 10000
    bound = result["decision"]["prior_tranches_terminal_evidence"]
    assert "absent" in bound["first_vector"]
    assert "sha256" in bound["first_stage"]


def test_a_broken_stage_chain_is_still_refused_when_the_vector_is_absent():
    ev = _terminal_tranche_evidence(13086)
    ev["stage"]["rows"] = 1
    with pytest.raises(ValueError, match="portable stage"):
        _validate_terminal_tranche(**{**ev, "vector": None}, label="x")


def test_a_present_but_broken_vector_is_still_refused():
    ev = _terminal_tranche_evidence(13086)
    ev["vector"]["rows"] = 1
    with pytest.raises(ValueError, match="vectors are not terminal"):
        _validate_terminal_tranche(**ev, label="x")


def test_a_prior_with_quarantined_questions_is_terminal_when_fully_accounted():
    """Task 691 ordinal 21: 9,999 documents + 1 quarantined == 10,000 selected."""
    ev = _terminal_tranche_evidence(13043)
    ev["parent"]["counts"].update(markdown_documents=9999, quarantine=1)
    ev["pack"]["counts"]["documents"] = 9999
    _validate_terminal_tranche(**ev, label="prior-#21")


def test_a_prior_that_loses_a_question_without_quarantine_is_refused():
    ev = _terminal_tranche_evidence(13043)
    ev["parent"]["counts"].update(markdown_documents=9999, quarantine=0)
    ev["pack"]["counts"]["documents"] = 9999
    with pytest.raises(ValueError, match="source tranche is not terminal"):
        _validate_terminal_tranche(**ev, label="x")


def test_a_pack_that_disagrees_with_the_admitted_documents_is_refused():
    ev = _terminal_tranche_evidence(13043)
    ev["parent"]["counts"].update(markdown_documents=9999, quarantine=1)
    ev["pack"]["counts"]["documents"] = 10000
    with pytest.raises(ValueError, match="chunk/token evidence"):
        _validate_terminal_tranche(**ev, label="x")
