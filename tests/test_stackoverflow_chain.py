"""Tests for cytir/ops/stackoverflow_tranches.py (task 691): the resumable
authorize -> materialize -> chunk -> stage -> L1 loop for the Stack Overflow frontier.

Counts in the fixtures are the real ones of tranche 9 (offset 80000): 10,000 questions,
35,906 source rows, 13,086 chunks (ratio 1.3086)."""
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest

from cytir.ops import stackoverflow_tranches as SO

REAL = dict(questions=10_000, source_rows=35_906, chunks=13_086)


def _w(path: Path, obj=None, text: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text is not None else json.dumps(obj))
    return path


def seal_tranche(root: Path, offset: int, limit: int = 10_000, vector: str | None = None) -> None:
    lay = SO.Layout(root)
    lab = SO.label(offset, limit)
    _w(lay.parent(lab) / "RUN.json", {})
    _w(lay.pack(lab) / "PACK.json", {})
    _w(lay.audit(lab) / "source-profile-compliance-receipt.json", {})
    _w(lay.audit(lab) / "content-fidelity-receipt.json", {})
    _w(lay.stage(lab) / "STAGE-RECEIPT.json", {})
    if offset == 0:
        _w(lay.frontier_dir / "frontier/first-010000.jsonl.gz", text="x")
    else:
        _w(lay.decision(lab) / f"frontier/tranche-{offset:06d}.jsonl.gz", text="x")
    if vector:
        _w(root / "vectors" / vector / "RECEIPT.portable.v4.json", {})


def make_frontier(root: Path, records: int) -> None:
    lay = SO.Layout(root)
    _w(lay.frontier_dir / "RUN.json", {"stages": {"frontier": {"records": records}}})


def test_roster_is_globbed_from_the_tree_and_sorted(tmp_path):
    seal_tranche(tmp_path, 10_000, vector="release=a/source=" + SO.label(10_000))
    seal_tranche(tmp_path, 0, vector="release=old/source=stackoverflow-score10-production-"
                                      "tranche-000000-009999-v1")
    seal_tranche(tmp_path, 20_000)
    roster = SO.discover_roster(tmp_path)
    assert [p.offset for p in roster] == [0, 10_000, 20_000]
    assert roster[0].vector is not None            # found under the older label alias
    assert roster[1].vector is not None
    assert roster[2].vector is None                # embed pending: absent, not an error


def test_roster_gap_is_refused_not_skipped(tmp_path):
    seal_tranche(tmp_path, 0)
    seal_tranche(tmp_path, 20_000)
    with pytest.raises(SO.ChainError, match="gap/overlap"):
        SO.discover_roster(tmp_path)


def test_in_flight_tranche_without_a_stage_is_not_a_prior(tmp_path):
    seal_tranche(tmp_path, 0)
    lay = SO.Layout(tmp_path)
    _w(lay.parent(SO.label(10_000)) / "RUN.json", {})   # materialized, not yet staged
    assert [p.offset for p in SO.discover_roster(tmp_path)] == [0]


def test_vector_receipt_picks_the_highest_version(tmp_path):
    lab = SO.label(0)
    for v in (4, 5):
        _w(tmp_path / f"vectors/release=r/source={lab}/RECEIPT.portable.v{v}.json", {})
    assert SO.find_vector_receipt(tmp_path, lab).name == "RECEIPT.portable.v5.json"


def _counts(**over):
    run = {"counts": {"selected_questions": REAL["questions"],
                      "markdown_documents": REAL["questions"], "quarantine": 0,
                      "selected_source_rows": REAL["source_rows"]}}
    pack = {"counts": {"documents": REAL["questions"], "chunks": REAL["chunks"]}}
    stage = {"records": REAL["chunks"], "rows": REAL["chunks"], "status": "complete"}
    return dict(in_questions=REAL["questions"], expected_source_rows=REAL["source_rows"],
                parent_run=run, pack=pack, stage=stage, l1_rows=REAL["questions"],
                log=lambda m: None) | over


def test_audit_counts_real_tranche_shape():
    out = SO.audit_counts(**_counts())
    assert out["chunks_per_question"] == 1.3086
    assert out["documents"] + out["quarantined"] == out["questions"]


@pytest.mark.parametrize("mutate,match", [
    (lambda k: k["parent_run"]["counts"].update(markdown_documents=9_999), "materialize"),
    (lambda k: k["parent_run"]["counts"].update(selected_source_rows=1), "source rows"),
    (lambda k: k["pack"]["counts"].update(documents=9_999), "chunk"),
    (lambda k: k["stage"].update(records=1), "stage"),
    (lambda k: k.update(l1_rows=9_999), "L1"),
])
def test_audit_counts_refuses_every_mismatch(mutate, match):
    kw = _counts()
    mutate(kw)
    with pytest.raises(SO.ChainError, match=match):
        SO.audit_counts(**kw)


def test_disk_guard_refuses_under_floor(tmp_path):
    with pytest.raises(SO.ChainError, match="under 300"):
        SO.check_disk([tmp_path], 300, usage=lambda p: 299.0)
    SO.check_disk([tmp_path], 300, usage=lambda p: 300.0)


class FakeRunner:
    """Stands in for subprocess.run and writes what each real step publishes."""

    def __init__(self, root: Path, fail_on: str | None = None):
        self.root, self.fail_on, self.steps = root, fail_on, []

    def __call__(self, argv, env=None):
        assert env["CUDA_VISIBLE_DEVICES"] == ""          # CPU only, never the GPU
        text = " ".join(argv)
        lay = SO.Layout(self.root)

        def arg(name):
            return Path(argv[argv.index(name) + 1])
        if "cytir.sources.stackoverflow.authorize_nth" in text:
            step = "authorize"
            off = sum(1 for a in argv if a == "--prior-tranche") * 10_000
            lim = int(arg("--limit").name)
            lab = SO.label(off, lim)
            assert str(arg("--root")) == str(lay.decision(lab))
            _w(lay.decision(lab) / "RUN.json", {"decision": {
                "selector": {"offset": off, "limit": lim},
                "counts": {"questions": lim,
                           "expected_source_rows_question_answer_user": 2 * lim + 5}}})
            _w(lay.decision(lab) / f"frontier/tranche-{off:06d}-{off + lim - 1:06d}.jsonl.gz", text="x")
            self.authorize_argv = argv
        elif "cytir.ops.run_observed_one_shot" in text:
            step = "materialize"
            assert argv[argv.index("--") + 1:argv.index("--") + 5] == ["ionice", "-c3", "nice", "-n19"]
            parent = arg("--root")
            lim = int(argv[argv.index("--expected-questions") + 1])
            _w(parent / "RUN.json", {"schema": "aw.stackoverflow-qa-production-tranche-run.v1",
                "status": "source_native_tranche_complete_pending_profile_fidelity_chunk_measurement",
                "counts": {
                "selected_questions": lim, "markdown_documents": lim, "quarantine": 0,
                "selected_source_rows": int(argv[argv.index("--expected-source-rows") + 1])}})
            md = parent / "data/markdown.jsonl.gz"
            md.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(md, "wt") as fh:
                for i in range(lim):
                    fh.write(json.dumps({"i": i}) + "\n")
            _w(lay.attempt(parent.name) / "OUTCOME.json", {"status": "succeeded", "rc": 0})
        elif "cytir.corpus.stage_structured_markdown" in text:
            step = "chunk"
        elif "cytir.sources.stackoverflow.audit_profile" in text:
            step = "chunk_audit"
            lim = int(arg("--expected-selected").name)
            _w(arg("--pack-root") / "PACK.json",
               {"counts": {"documents": lim, "chunks": int(lim * 1.3086)}})
        elif "cytir.index.seal_content_fidelity_audit" in text:
            step = "chunk_fidelity"
            _w(arg("--receipt"), {})
        elif "cytir.index.seal_source_profile_compliance" in text:
            step = "chunk_compliance"
            _w(arg("--receipt"), {})
        elif "cytir.embed.portableize_pack" in text:
            step = "portableize"
            _w(arg("--pack") / "PACK.portable.v2.json", {})
        elif "cytir.embed.adapt_pack_to_stage" in text:
            step = "adapt"
            pack = json.loads((arg("--pack") / "PACK.json").read_text())
            n = pack["counts"]["chunks"]
            _w(arg("--output") / "STAGE-RECEIPT.json", {"records": n, "rows": n, "status": "complete"})
        elif "register_l1_release" in text:
            step = "l1"
            release = argv[argv.index("--release") + 1]
            d = arg("--corpus-root") / "l1" / f"release={release}"
            d.mkdir(parents=True)
            (d / "_register_manifest.json").write_text("{}")
            (d / "markdown.jsonl.gz").write_bytes((arg("--source") / "markdown.jsonl.gz").read_bytes())
        elif "verify_l1_coverage" in text:
            step = "verify"
        else:
            raise AssertionError(text)
        self.steps.append(step)
        return type("R", (), {"returncode": 3 if step == self.fail_on else 0})()


def _tree_with_nine_tranches(root: Path, frontier_records: int) -> None:
    for i in range(9):
        seal_tranche(root, i * 10_000)
    make_frontier(root, frontier_records)


def test_run_ordinal_seals_the_next_tranche_through_every_step(tmp_path):
    _tree_with_nine_tranches(tmp_path, 551_326)
    # run_ordinal asserts the authorized window; the fake authorizer derives it from the roster
    runner = FakeRunner(tmp_path)
    result = SO.run_ordinal(artifact_root=tmp_path, python_bin="py", runner=runner,
                            log=lambda m: None)
    assert runner.steps == ["authorize", "materialize", "chunk", "chunk_audit", "chunk_fidelity", "chunk_compliance", "portableize", "adapt", "l1", "verify"]
    assert result["ordinal"] == 10 and result["offset"] == 90_000
    assert result["label"] == "stackoverflow-qa-score10-tranche-090000-099999-v1"
    assert result["l1_release"] == "so-tranche-090000-w1"
    assert result["questions"] == result["l1_rows"] == 10_000
    # every earlier tranche was passed, taken from the tree; an absent vector is passed as "-"
    argv = runner.authorize_argv
    assert argv.count("--prior-tranche") == 9 and argv.count("-") == 9
    # the new tranche is now part of the roster, so the next call asks for ordinal 11
    assert [p.offset for p in SO.discover_roster(tmp_path)][-1] == 90_000


def test_run_ordinal_resumes_and_skips_finished_steps(tmp_path):
    _tree_with_nine_tranches(tmp_path, 551_326)
    SO.run_ordinal(artifact_root=tmp_path, python_bin="py", runner=FakeRunner(tmp_path),
                   log=lambda m: None)
    # The L1 manifest of a finished tranche exists, so a re-run of its tail does nothing.
    lay = SO.Layout(tmp_path)
    lab = SO.label(90_000)
    shutil.rmtree(lay.stage(lab))
    runner = FakeRunner(tmp_path)
    # stage gone => it is not a prior, so the same ordinal resumes: only the missing steps run
    SO.run_ordinal(artifact_root=tmp_path, python_bin="py", runner=runner, log=lambda m: None)
    assert runner.steps == ["adapt", "verify"]


def test_half_written_materialize_root_refuses_instead_of_overwriting(tmp_path):
    _tree_with_nine_tranches(tmp_path, 551_326)
    lay = SO.Layout(tmp_path)
    lab = SO.label(90_000)
    runner = FakeRunner(tmp_path)
    runner.fail_on = "materialize"
    with pytest.raises(SO.ChainError, match="step failed"):
        SO.run_ordinal(artifact_root=tmp_path, python_bin="py", runner=runner, log=lambda m: None)
    shutil.rmtree(lay.parent(lab))            # the failed run left no terminal RUN
    lay.attempt(lab).mkdir(parents=True, exist_ok=True)
    with pytest.raises(SO.ChainError, match="inspect"):
        SO.run_ordinal(artifact_root=tmp_path, python_bin="py", runner=FakeRunner(tmp_path),
                       log=lambda m: None)


def test_last_partial_tranche_and_exhaustion(tmp_path):
    for i in range(55):
        seal_tranche(tmp_path, i * 10_000)
    make_frontier(tmp_path, 551_326)
    out = SO.run_ordinal(artifact_root=tmp_path, python_bin="py", runner=FakeRunner(tmp_path),
                         log=lambda m: None)
    assert out["offset"] == 550_000 and out["limit"] == 1_326
    assert out["label"] == "stackoverflow-qa-score10-tranche-550000-551325-v1"
    assert SO.run_ordinal(artifact_root=tmp_path, python_bin="py", runner=FakeRunner(tmp_path),
                          log=lambda m: None) is None


def test_loop_writes_state_and_next_and_stops_on_failure(tmp_path):
    corpus, state = tmp_path / "corpus", tmp_path / "state"
    corpus.mkdir()
    _tree_with_nine_tranches(corpus, 551_326)
    runner = FakeRunner(corpus)
    SO.run_loop(artifact_root=corpus, state_dir=state, python_bin="py", argv=["cmd"],
                max_ordinals=2, runner=runner, log=lambda m: None, usage=lambda p: 1e6)
    st = json.loads((state / "STATE.json").read_text())
    assert sorted(st["tranches"]) == ["10", "11"] and st["status"].startswith("stopped: reached")
    assert "sealed by this driver: 2 tranche(s), 20,000 questions" in (state / "NEXT.md").read_text()
    runner = FakeRunner(corpus, fail_on="authorize")
    with pytest.raises(SO.ChainError):
        SO.run_loop(artifact_root=corpus, state_dir=state, python_bin="py", argv=["cmd"],
                    max_ordinals=1, runner=runner, log=lambda m: None, usage=lambda p: 1e6)
    st = json.loads((state / "STATE.json").read_text())
    assert st["failed"]["error"].startswith("step failed")
    with pytest.raises(SO.ChainError, match="free space"):
        SO.run_loop(artifact_root=corpus, state_dir=state, python_bin="py", argv=["cmd"],
                    max_ordinals=1, runner=FakeRunner(corpus), log=lambda m: None,
                    usage=lambda p: 10.0)


def test_resume_command_carries_the_persisted_index_env_and_refuses_without_it(tmp_path, monkeypatch):
    monkeypatch.delenv("AW_PERSISTED_INDEX_SQLITE_DIR", raising=False)
    args = SO.build_parser().parse_args(["--artifact-root", str(tmp_path), "--state-dir", str(tmp_path / "s")])
    cmd = SO.resume_command(args, {"AW_PERSISTED_INDEX_SQLITE_DIR": "/nvme/pi"})
    assert cmd[0] == "AW_PERSISTED_INDEX_SQLITE_DIR=/nvme/pi" and "--state-dir" in cmd
    assert SO.resume_command(args, {}) is None
    assert SO.main(["--artifact-root", str(tmp_path), "--state-dir", str(tmp_path / "s")]) == 2
