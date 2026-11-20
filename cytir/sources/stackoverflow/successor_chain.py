"""Chunk + audit + seal for one Stack Overflow tranche (the successor chain).

Called by `cytir.ops.stackoverflow_tranches` once the source-native materialize step has finished
(task 1981; ported from a shell chain, design in task 1978). The driver is the only caller and there
is no CLI of its own (one door).

Every step stays a subprocess (`python -m cytir...`): the step `main()`s read `sys.argv`, and steps 1-2
run under `nice -n 10 ionice -c3` (Linux tools, as before). `build_steps` returns the exact argv
lists; `run_successor` runs them in order through the injected `runner` and stops at the first
nonzero exit. `--verification-method` and `--classification` are sealed into receipts as values:
keep them byte-identical.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

RELEASE_DIR = Path(__file__).resolve().parents[3]
POLICY = "data/chunk-policy-struct-md-768w-o0-adjacent-sections-tail-1152-tok2560.v1.json"
PROFILE_REGISTRY = "data/source-preservation-profiles.v1.json"
TOKENIZER_REVISION = "ac6544c8a46e00af67e330e85a9028c66b8cfd9a"
VERIFICATION_METHOD = ("Deep source/archive/XML/Markdown semantic audit with AST no-cap check "
                       "and exact selected-row digests")
NICE = ["nice", "-n", "10", "ionice", "-c3"]


class SuccessorChainError(RuntimeError):
    """A precondition or a step of the chain failed."""


@dataclass(frozen=True)
class SuccessorCfg:
    """Everything the chain needs; no defaults (the caller knows its tranche)."""
    artifact_root: Path
    parent_root: Path
    parent_outcome: Path
    pack_root: Path
    audit_root: Path
    source_adapter: Path
    producer_revision: str
    expected_schema: str
    expected_status: str
    expected_selected: int
    expected_cell: str
    source_id: str
    classification: str
    python_bin: str


def check_parent_terminal(outcome: Path, run: Path, schema: str, status: str) -> None:
    """The observed materialize attempt succeeded and its RUN is terminal (was an inline gate)."""
    if not Path(outcome).is_file():
        raise SuccessorChainError(f"Stack Overflow source pilot OUTCOME is absent: {outcome}")
    got = json.loads(Path(outcome).read_text())
    if got.get("status") != "succeeded" or got.get("rc") != 0:
        raise SuccessorChainError("Stack Overflow source pilot predecessor did not succeed")
    if not Path(run).is_file():
        raise SuccessorChainError(f"Stack Overflow source pilot RUN is absent: {run}")
    doc = json.loads(Path(run).read_text())
    if doc.get("schema") != schema or doc.get("status") != status:
        raise SuccessorChainError("Stack Overflow source pilot RUN is absent or nonterminal")


def build_steps(cfg: SuccessorCfg, *, release_dir: Path = RELEASE_DIR) -> list[list[str]]:
    """The four argv lists, in run order."""
    rel = Path(release_dir)
    py = cfg.python_bin
    parent_run = f"{cfg.parent_root}/RUN.json"
    audit = f"{cfg.audit_root}"
    registry = f"{rel}/{PROFILE_REGISTRY}"
    stage = [*NICE, py, "-m", "cytir.corpus.stage_structured_markdown",
             "--artifact-root", f"{cfg.artifact_root}",
             "--parent-root", f"{cfg.parent_root}", "--parent-run", parent_run,
             "--out", f"{cfg.pack_root}", "--policy", f"{rel}/{POLICY}",
             "--tokenizer-revision", TOKENIZER_REVISION,
             "--producer-revision", cfg.producer_revision,
             "--source-profile", "community_qa"]
    audit_args = ["--artifact-root", f"{cfg.artifact_root}", "--parent-run", parent_run,
                  "--pack-root", f"{cfg.pack_root}", "--output-root", audit,
                  "--profile-registry", registry, "--source-id", cfg.source_id,
                  "--expected-schema", cfg.expected_schema, "--expected-status", cfg.expected_status,
                  "--expected-selected", str(cfg.expected_selected),
                  "--classification", cfg.classification]
    if cfg.expected_cell:
        audit_args += ["--expected-cell", cfg.expected_cell]
    audit_step = [*NICE, py, "-m", "cytir.sources.stackoverflow.audit_profile", *audit_args]
    fidelity = [py, "-m", "cytir.index.seal_content_fidelity_audit",
                "--artifact-root", f"{cfg.artifact_root}", "--parent-run", parent_run,
                "--adapter-source", f"{cfg.source_adapter}", "--source-id", cfg.source_id,
                "--verification-method", VERIFICATION_METHOD,
                "--receipt", f"{audit}/content-fidelity-receipt.json",
                "--envelope", f"{audit}/content-fidelity-envelope.json"]
    compliance = [py, "-m", "cytir.index.seal_source_profile_compliance",
                  "--artifact-root", f"{cfg.artifact_root}", "--profile-registry", registry,
                  "--profile", "community_qa", "--source-id", cfg.source_id,
                  "--checks", f"{audit}/checks.json",
                  "--verification-evidence", f"{audit}/verification-evidence.json",
                  "--receipt", f"{audit}/source-profile-compliance-receipt.json"]
    return [stage, audit_step, fidelity, compliance]


def run_successor(cfg: SuccessorCfg, *, env: dict, runner, log, release_dir: Path = RELEASE_DIR) -> None:
    """Gate, then the four steps. Raises SuccessorChainError at the first failure."""
    check_parent_terminal(cfg.parent_outcome, Path(f"{cfg.parent_root}/RUN.json"),
                          cfg.expected_schema, cfg.expected_status)
    # `python -m cytir...` steps load THIS checkout's package (the caller may not set PYTHONPATH).
    prior = env.get("PYTHONPATH")
    step_env = dict(env, PYTHONPATH=f"{release_dir}{os.pathsep + prior if prior else ''}")
    for argv in build_steps(cfg, release_dir=release_dir):
        log(f"[successor] $ {' '.join(argv[:8])} ...")
        result = runner(argv, env=step_env)
        if result.returncode != 0:
            raise SuccessorChainError(f"successor step failed rc={result.returncode}: "
                                      f"{' '.join(argv[:8])} ...")
