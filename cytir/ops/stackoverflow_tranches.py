#!/usr/bin/env python3
"""Stack Overflow Q+A production frontier: authorize -> materialize -> chunk -> stage -> L1,
one 10k-question tranche at a time, resumable (task 691, 2026-10-02).

WHAT WAS MISSING
----------------
The 551,326-pair licensed frontier is ranked once and immutable. Tranches 1-9 were landed by
hand-run recipes (task 631 `run_trancheN_*.sh`: a hand-typed list of every earlier tranche's
receipts per ordinal, then materialize, chunk chain, portableize, adapt). ~47 ordinals remained.
This module is that loop as product code. It adds no extraction or chunk logic; every step is
the existing product script, run as a subprocess (their import-time effects and memory never
accumulate here), exactly as the retired OpenLegalData chain did:

  1. `python -m cytir.sources.stackoverflow.authorize_nth` -- the generic authorizer. The prior
     tranches' receipts are GLOBBED FROM THE TREE (never a hand list) and the roster is printed
     (corpus-doctrine 8). The authorizer derives the offset from the priors' own selection keys.
  2. the stackoverflow QA production-tranche materializer (kept in the legacy tree, see below) under
     `run_observed_one_shot` (CPU only: `ionice -c3 nice -n19`, `CUDA_VISIBLE_DEVICES=`).
  3. `cytir/sources/stackoverflow/successor_chain.py` -- chunk + stage + profile + fidelity.
  4. `python -m cytir.embed.portableize_pack`, then `python -m cytir.embed.adapt_pack_to_stage`
     -> the sealed `...-413a-portable-v3` stage.
  5. `cytir.corpus.register_l1_release` + `cytir.corpus.verify_l1_coverage` -- the ADR-105 L1
     filing the promotion drain requires (`require=l1`); without it the pack is refused.

WHAT THIS DOES NOT DO: embed, compose, or enqueue. `aw-growth-pending-compose.timer`
(`cytir.ops.growth_pending_compose`, `discover_se_pending`) reads the sealed stage, takes the
GPU lock, embeds and enqueues. This driver never takes the GPU lock and never imports torch.
Because embedding is the timer's job, the authorizer accepts a prior tranche that has a terminal
stage but no vector rollup receipt (the timer seals raw vector shards, not a rollup); the
decision records that the vector evidence was absent.

COUNTS. Every stage prints `in= admitted= excluded(reasons)=` and refuses on mismatch
(corpus-doctrine 2). RESUME: each step has a marker artifact; a finished step is skipped. A
failure stops the loop with a non-zero exit; STATE.json and NEXT.md say which ordinal and why
(engineering-doctrine 3). A half-written step directory is never overwritten.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from cytir.sources.stackoverflow.successor_chain import SuccessorCfg, SuccessorChainError, run_successor

REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILY_ID = "stackoverflow_qa"
STACK = "_work/next-waves/n1-stackexchange-v1"
NAME_PREFIX = "stackoverflow-qa-score10-tranche-"
TRANCHE_LIMIT = 10_000
MIN_FREE_GB = 300
SOURCE_PROFILE = "community_qa"
_NAME_RE = re.compile(r"^stackoverflow-qa-score10-tranche-(\d{6})-(\d{6})-v1$")
_VECTOR_RE = re.compile(r"^RECEIPT\.portable\.v(\d+)\.json$")
# The first tranche's vectors were sealed under an older label.
_VECTOR_ALIASES = ("stackoverflow-score10-production-tranche-",)


class ChainError(RuntimeError):
    pass


def say(msg: str) -> None:
    print(msg, flush=True)     # a detached run's log must not sit in a stdio buffer


def label(offset: int, limit: int = TRANCHE_LIMIT) -> str:
    return f"{NAME_PREFIX}{offset:06d}-{offset + limit - 1:06d}-v1"


def l1_release_name(offset: int) -> str:
    return f"so-tranche-{offset:06d}-w1"


@dataclass
class Layout:
    """Where every artifact of one tranche lives (the tree the earlier recipes built)."""
    artifact_root: Path

    @property
    def stack(self) -> Path:
        return self.artifact_root / STACK

    def decision(self, lab: str) -> Path:
        return self.stack / "decisions" / lab

    def parent(self, lab: str) -> Path:
        return self.stack / "production" / lab

    def pack(self, lab: str) -> Path:
        return self.stack / "production" / "packs" / lab

    def audit(self, lab: str) -> Path:
        return self.stack / "production" / "audits" / lab

    def stage(self, lab: str) -> Path:
        return self.stack / "production" / "stages" / f"{lab}-413a-portable-v3"

    def attempt(self, lab: str) -> Path:
        return self.stack / "launch-attempts" / f"{lab}-attempt-0001"

    @property
    def frontier_dir(self) -> Path:
        return self.stack / "frontiers" / "stackoverflow-qa-score10-production-v1"

    def l1_markdown(self, offset: int) -> Path:
        return (self.artifact_root / "l1" / f"release={l1_release_name(offset)}"
                / "markdown.jsonl.gz")

    def l1_manifest(self, offset: int) -> Path:
        return self.l1_markdown(offset).with_name("_register_manifest.json")


@dataclass
class Prior:
    offset: int
    limit: int
    label: str
    tranche: Path
    parent: Path
    pack: Path
    profile: Path
    fidelity: Path
    stage: Path
    vector: Path | None

    def evidence(self) -> list[Path]:
        return [self.tranche, self.parent, self.pack, self.profile, self.fidelity, self.stage]


def find_vector_receipt(artifact_root: Path, lab: str) -> Path | None:
    """The highest-version `RECEIPT.portable.v*.json` of a tranche, wherever it was sealed.

    Two shapes exist on disk (`release=*/source=<label>/` and `.../vectors/`), and the
    first tranche has an older label. None when the embed left only raw shards."""
    labels = [lab] + [lab.replace(NAME_PREFIX, alias, 1) for alias in _VECTOR_ALIASES]
    best: tuple[int, Path] | None = None
    for name in labels:
        for pattern in (f"release=*/source={name}/RECEIPT.portable.v*.json",
                        f"release=*/source={name}/vectors/RECEIPT.portable.v*.json"):
            for path in (artifact_root / "vectors").glob(pattern):
                m = _VECTOR_RE.match(path.name)
                if m and (best is None or int(m.group(1)) > best[0]):
                    best = (int(m.group(1)), path)
    return best[1] if best else None


def discover_roster(artifact_root: Path) -> list[Prior]:
    """Every sealed tranche, from the tree: a production dir with a terminal stage receipt.

    Refuses (never silently skips) a roster that is not contiguous from offset 0 with
    contiguous windows -- that is a gap or an overlap (corpus-doctrine 8)."""
    lay = Layout(artifact_root)
    found: list[Prior] = []
    for parent_dir in sorted((lay.stack / "production").glob(f"{NAME_PREFIX}*-v1")):
        m = _NAME_RE.match(parent_dir.name)
        if not m:
            raise ChainError(f"unparseable tranche dir in the tree: {parent_dir}")
        offset, end = int(m.group(1)), int(m.group(2))
        lab = parent_dir.name
        stage = lay.stage(lab) / "STAGE-RECEIPT.json"
        if not stage.is_file():
            continue            # an in-flight tranche is not a prior of anything
        if offset == 0:
            files = sorted((lay.frontier_dir / "frontier").glob("first-*.jsonl.gz"))
        else:
            files = sorted((lay.decision(lab) / "frontier").glob("*.jsonl.gz"))
        if len(files) != 1:
            raise ChainError(f"{lab}: expected exactly one selection file, found {files}")
        found.append(Prior(
            offset=offset, limit=end - offset + 1, label=lab, tranche=files[0],
            parent=parent_dir / "RUN.json", pack=lay.pack(lab) / "PACK.json",
            profile=lay.audit(lab) / "source-profile-compliance-receipt.json",
            fidelity=lay.audit(lab) / "content-fidelity-receipt.json", stage=stage,
            vector=find_vector_receipt(artifact_root, lab)))
    found.sort(key=lambda p: p.offset)
    expected = 0
    for prior in found:
        if prior.offset != expected:
            raise ChainError(f"roster gap/overlap: {prior.label} starts at {prior.offset}, "
                             f"expected {expected} (roster: {[p.label for p in found]})")
        missing = [str(p) for p in prior.evidence() if not p.is_file()]
        if missing:
            raise ChainError(f"{prior.label}: missing terminal evidence {missing}")
        expected = prior.offset + prior.limit
    return found


def print_roster(roster: list[Prior], log=say) -> None:
    log(f"[stackoverflow] roster from the tree: {len(roster)} sealed tranche(s), "
        f"{sum(p.limit for p in roster):,} questions")
    for p in roster:
        log(f"[stackoverflow]   {p.label}  vector_rollup="
            f"{'yes' if p.vector else 'ABSENT (embed pending or timer-sealed)'}")


def frontier_total(artifact_root: Path) -> int:
    run = json.loads((Layout(artifact_root).frontier_dir / "RUN.json").read_text())
    return int(run["stages"]["frontier"]["records"])


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def check_disk(paths: list[Path], min_gb: float = MIN_FREE_GB, usage=free_gb) -> None:
    for path in paths:
        free = usage(path)
        if free < min_gb:
            raise ChainError(f"free space {free:.0f} GB on {path} is under {min_gb} GB; stopping")


def _run(argv: list[str], *, env: dict, runner, log) -> None:
    log(f"[stackoverflow] $ {' '.join(str(a) for a in argv[:2])} ... {' '.join(str(a) for a in argv[-4:])}")
    result = runner([str(a) for a in argv], env=env)
    if result.returncode != 0:
        raise ChainError(f"step failed rc={result.returncode}: {' '.join(str(a) for a in argv[:4])} ...")


def _jload(path: Path) -> dict:
    return json.loads(path.read_text())


def _count_lines(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def audit_counts(*, in_questions: int, expected_source_rows: int, parent_run: dict,
                 pack: dict, stage: dict, l1_rows: int | None, log=say) -> dict:
    """The accounting identity per stage. Raises on any mismatch (corpus-doctrine 2)."""
    c = parent_run["counts"]
    docs, quarantined = c["markdown_documents"], c["quarantine"]
    if c["selected_questions"] != in_questions or docs + quarantined != in_questions:
        raise ChainError(f"materialize: in={in_questions} admitted={docs} "
                         f"excluded(quarantine)={quarantined} selected={c['selected_questions']}")
    if c["selected_source_rows"] != expected_source_rows:
        raise ChainError(f"materialize source rows {c['selected_source_rows']} "
                         f"!= authorized {expected_source_rows}")
    log(f"[stackoverflow] materialize: in={in_questions} admitted={docs} "
        f"excluded(quarantine)={quarantined} source_rows={c['selected_source_rows']}")
    chunks = pack["counts"]["chunks"]
    if pack["counts"]["documents"] != docs:
        raise ChainError(f"chunk: in={docs} documents, pack has {pack['counts']['documents']}")
    log(f"[stackoverflow] chunk: in={docs} documents admitted={chunks} chunks "
        f"excluded(documents with no chunk)=0 ratio={chunks / docs:.4f}")
    if stage["records"] != chunks or stage["rows"] != chunks or stage["status"] != "complete":
        raise ChainError(f"stage: in={chunks} chunks, stage records={stage['records']} "
                         f"rows={stage['rows']} status={stage['status']}")
    log(f"[stackoverflow] stage: in={chunks} admitted={stage['records']} excluded()=0")
    if l1_rows is not None:
        if l1_rows != docs:
            raise ChainError(f"L1: in={docs} documents, L1 holds {l1_rows}")
        log(f"[stackoverflow] L1: in={docs} admitted={l1_rows} excluded()=0")
    return {"questions": in_questions, "documents": docs, "quarantined": quarantined,
            "source_rows": c["selected_source_rows"], "chunks": chunks,
            "chunks_per_question": round(chunks / in_questions, 4),
            "l1_rows": l1_rows, "stage_records": stage["records"]}


def authorize_argv(roster: list[Prior], *, ordinal: int, limit: int, lay: Layout,
                   python_bin: str) -> list[str]:
    argv = [python_bin, "-m", "cytir.sources.stackoverflow.authorize_nth",
            "--frontier-run", str(lay.frontier_dir / "RUN.json"),
            "--frontier", str(lay.frontier_dir / "frontier/eligible-score10.jsonl.gz"),
            "--root", str(lay.decision(label(sum(p.limit for p in roster), limit))),
            "--artifact-root", str(lay.artifact_root),
            "--ordinal", str(ordinal), "--limit", str(limit)]
    for p in roster:
        argv += ["--prior-tranche", str(p.tranche), "--prior-parent", str(p.parent),
                 "--prior-pack", str(p.pack), "--prior-profile", str(p.profile),
                 "--prior-fidelity", str(p.fidelity), "--prior-stage", str(p.stage),
                 "--prior-vector", str(p.vector) if p.vector else "-"]
    return argv


def run_ordinal(*, artifact_root: Path, python_bin: str, runner=subprocess.run,
                log=say, run_verify=True) -> dict | None:
    """Land the next tranche. None when the frontier is exhausted."""
    lay = Layout(artifact_root)
    roster = discover_roster(artifact_root)
    print_roster(roster, log)
    offset = sum(p.limit for p in roster)
    total = frontier_total(artifact_root)
    if offset >= total:
        log(f"[stackoverflow] frontier exhausted: {offset:,} of {total:,}")
        return None
    limit = min(TRANCHE_LIMIT, total - offset)
    ordinal = len(roster) + 1
    lab = label(offset, limit)
    t0 = time.time()
    # REPO_ROOT first: `python -m cytir...` steps must load THIS checkout's package.
    env = dict(os.environ,
               PYTHONPATH=os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "pipeline")]),
               CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    log(f"[stackoverflow] ordinal {ordinal}: offset={offset} limit={limit} label={lab}")
    decision_dir = lay.decision(lab)
    if not (decision_dir / "RUN.json").is_file():
        if decision_dir.exists():
            raise ChainError(f"half-written authorization root, inspect: {decision_dir}")
        _run(authorize_argv(roster, ordinal=ordinal, limit=limit, lay=lay,
                            python_bin=python_bin), env=env, runner=runner, log=log)
    decision = _jload(decision_dir / "RUN.json")["decision"]
    sel = decision["selector"]
    if sel["offset"] != offset or sel["limit"] != limit:
        raise ChainError(f"authorized window {sel['offset']}+{sel['limit']} != {offset}+{limit}")
    questions = decision["counts"]["questions"]
    source_rows = decision["counts"]["expected_source_rows_question_answer_user"]
    if questions != limit:
        raise ChainError(f"authorize: in={limit} admitted={questions}")
    log(f"[stackoverflow] authorize: in={limit} admitted={questions} excluded()=0 "
        f"expected_source_rows={source_rows}")

    parent, attempt = lay.parent(lab), lay.attempt(lab)
    # The producer: its BYTES are hashed below into AW_PRODUCER_REVISION (chunk provenance
    # producer.revision) and bound by the fidelity receipt (adapter_source_sha256). This path IS the
    # producer revision; moving the file changes the digest of FUTURE tranches only (task 1509).
    materializer = REPO_ROOT / "cytir/sources/stackoverflow/materialize_production_tranche.py"
    archives = lay.stack / "acquisition"
    if not (parent / "RUN.json").is_file():
        if parent.exists() or attempt.exists():
            raise ChainError(f"materialize root exists without a terminal RUN, inspect: "
                             f"{parent} {attempt}")
        _run([python_bin, "-m", "cytir.ops.run_observed_one_shot",
              "--attempt-root", attempt, "--artifact-root", artifact_root,
              "--release", str(REPO_ROOT), "--",
              "ionice", "-c3", "nice", "-n19", "env", "CUDA_VISIBLE_DEVICES=",
              python_bin, "-m", "cytir.sources.stackoverflow.materialize_production_tranche",
              "--posts-archive", archives / "stackoverflow-posts-v1/input/stackoverflow.com-Posts.7z",
              "--users-archive", archives / "stackoverflow-users-v1/input/stackoverflow.com-Users.7z",
              "--inventory-run", archives / "stackoverflow-source-archive-inventory-v1/RUN.json",
              "--frontier-run", decision_dir / "RUN.json",
              "--tranche", decision_dir / f"frontier/tranche-{offset:06d}-{offset + limit - 1:06d}.jsonl.gz",
              "--root", parent, "--artifact-root", artifact_root,
              "--expected-questions", questions, "--expected-source-rows", source_rows],
             env=env, runner=runner, log=log)

    pack, audit, stage = lay.pack(lab), lay.audit(lab), lay.stage(lab)
    if not (audit / "source-profile-compliance-receipt.json").is_file():
        if pack.exists() or audit.exists():
            raise ChainError(f"chunk/audit dirs exist without the terminal receipt, inspect: "
                             f"{pack} {audit}")
        producer = hashlib.sha256(materializer.read_bytes()).hexdigest()
        cfg = SuccessorCfg(
            artifact_root=artifact_root, parent_root=parent, parent_outcome=attempt / "OUTCOME.json",
            pack_root=pack, audit_root=audit, source_adapter=materializer, producer_revision=producer,
            expected_schema="aw.stackoverflow-qa-production-tranche-run.v1",
            expected_status="source_native_tranche_complete_pending_profile_fidelity_chunk_measurement",
            expected_selected=questions, expected_cell="accepted_qscore_10_plus",
            source_id=f"stackoverflow-qa-score10-production-tranche-{offset:06d}-{offset + limit - 1:06d}-v1",
            classification=(f"measured exact ordinal-{ordinal} production tranche (offset={offset}, "
                            f"limit={limit}); full {total}-pair frontier remains selection-only"),
            python_bin=python_bin)
        try:
            run_successor(cfg, env=env, runner=runner, log=log, release_dir=REPO_ROOT)
        except SuccessorChainError as exc:
            raise ChainError(str(exc)) from exc
    if not (pack / "PACK.portable.v2.json").is_file():
        _run([python_bin, "-m", "cytir.embed.portableize_pack",
              "--artifact-root", artifact_root, "--pack", pack], env=env, runner=runner, log=log)
    if not (stage / "STAGE-RECEIPT.json").is_file():
        if stage.exists():
            raise ChainError(f"stage dir exists without its receipt, inspect: {stage}")
        _run([python_bin, "-m", "cytir.embed.adapt_pack_to_stage",
              "--artifact-root", artifact_root, "--pack", pack,
              "--pack-manifest", "PACK.portable.v2.json", "--output", stage],
             env=env, runner=runner, log=log)

    if not lay.l1_manifest(offset).is_file():
        _run([python_bin, "-m", "cytir.corpus.register_l1_release",
              "--source", parent / "data", "--release", l1_release_name(offset),
              "--corpus-root", artifact_root, "--file-suffix", "markdown.jsonl.gz"],
             env=env, runner=runner, log=log)
    if run_verify:
        _run([python_bin, "-m", "cytir.corpus.verify_l1_coverage",
              "--passages", lay.l1_markdown(offset), "--sample", "300",
              "--label", f"so-tranche-{offset:06d}"], env=env, runner=runner, log=log)

    counts = audit_counts(
        in_questions=questions, expected_source_rows=source_rows,
        parent_run=_jload(parent / "RUN.json"), pack=_jload(pack / "PACK.json"),
        stage=_jload(stage / "STAGE-RECEIPT.json"),
        l1_rows=_count_lines(lay.l1_markdown(offset)), log=log)
    counts.update(ordinal=ordinal, offset=offset, limit=limit, label=lab,
                  stage=str(stage), l1_release=l1_release_name(offset),
                  seconds=round(time.time() - t0, 1), status="sealed")
    return counts


def _write_state(state_dir: Path, state: dict, argv: list[str]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    tmp = state_dir / "STATE.json.tmp"
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    tmp.replace(state_dir / "STATE.json")
    done = state["tranches"]
    chunks = sum(t["chunks"] for t in done.values())
    questions = sum(t["questions"] for t in done.values())
    (state_dir / "NEXT.md").write_text(
        "# Stack Overflow frontier driver (task 691)\n\n"
        f"updated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, pid {os.getpid()}\n\n"
        f"- sealed by this driver: {len(done)} tranche(s), {questions:,} questions, {chunks:,} chunks\n"
        f"- last: {state.get('last') or 'none'}\n"
        f"- status: {state.get('status')}\n"
        f"- failed: {state.get('failed') or 'none'}\n\n"
        "Sealed stages are CPU-only products. Embed, compose and enqueue belong to "
        "aw-growth-pending-compose.timer; this driver never takes the GPU lock.\n\n"
        "Relaunch (resumes; sealed steps are skipped, a half-written step refuses):\n\n"
        f"    {' '.join(argv)}\n")


def run_loop(*, artifact_root: Path, state_dir: Path, python_bin: str, argv: list[str],
             max_ordinals: int | None = None, min_free_gb: float = MIN_FREE_GB,
             runner=subprocess.run, log=say, usage=free_gb, run_verify=True) -> dict:
    sf = state_dir / "STATE.json"
    state = json.loads(sf.read_text()) if sf.is_file() else {"tranches": {}}
    state.update(failed=None, status="running")
    n = 0
    while max_ordinals is None or n < max_ordinals:
        try:
            check_disk([state_dir, artifact_root], min_free_gb, usage)
            result = run_ordinal(artifact_root=artifact_root, python_bin=python_bin,
                                 runner=runner, log=log, run_verify=run_verify)
        except ChainError as exc:
            state.update(failed={"error": str(exc)}, status="stopped: failed")
            _write_state(state_dir, state, argv)
            raise
        if result is None:
            state["status"] = "done: frontier exhausted"
            _write_state(state_dir, state, argv)
            break
        state["tranches"][str(result["ordinal"])] = result
        state["last"] = f"ordinal {result['ordinal']} ({result['label']})"
        _write_state(state_dir, state, argv)
        log(f"[stackoverflow] sealed: {result}")
        n += 1
    else:
        state["status"] = f"stopped: reached --max-ordinals {max_ordinals}"
        _write_state(state_dir, state, argv)
    return state


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifact-root", type=Path, required=True, help="the corpus root (aw-corpus)")
    ap.add_argument("--state-dir", type=Path, required=True,
                    help="STATE.json and NEXT.md go here (NOT under the corpus, not a boot drive)")
    ap.add_argument("--max-ordinals", type=int, default=None,
                    help="stop after N tranches (default: until the frontier is exhausted)")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    return ap


PERSISTED_INDEX_ENV = "AW_PERSISTED_INDEX_SQLITE_DIR"


def resume_command(args, environ=None) -> list[str] | None:
    """The relaunch command NEXT.md prints, env included. None when the env is unset.

    `verify_l1_coverage` opens the persisted index. Without the env it falls back to the
    HDD copy and sat in D state for hours (task 691, 2026-10-03). The drain uses the NVMe
    one, so the driver must too: a resume that loses the env is refused, not run slowly.
    """
    value = (os.environ if environ is None else environ).get(PERSISTED_INDEX_ENV)
    if not value:
        return None
    cmd = [f"{PERSISTED_INDEX_ENV}={value}", sys.executable, "-m", "cytir.ops.stackoverflow_tranches",
           "--artifact-root", str(args.artifact_root), "--state-dir", str(args.state_dir),
           "--min-free-gb", str(args.min_free_gb)]
    if args.max_ordinals is not None:
        cmd += ["--max-ordinals", str(args.max_ordinals)]
    return cmd


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cmd = resume_command(args)
    if cmd is None:
        print(f"FATAL: {PERSISTED_INDEX_ENV} is not set; point it at the NVMe persisted index "
              "(the one the drain uses)", file=sys.stderr)
        return 2
    try:
        run_loop(artifact_root=args.artifact_root, state_dir=args.state_dir,
                 python_bin=sys.executable, argv=cmd, max_ordinals=args.max_ordinals,
                 min_free_gb=args.min_free_gb)
    except ChainError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
