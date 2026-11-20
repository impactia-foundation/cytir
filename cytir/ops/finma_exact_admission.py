"""FINMA exact retained-PDF admission: pack -> profile audit -> seals -> portable stage (task 1396).

This is the OPS chain of the `finma_exact` family (backfilled; 2,956 retained PDFs of
`finma-production-current-exact-retained-pdfs-v1`), moved out of a shell script. It adds no
extraction or chunk logic: every step is an existing module, run as an observed one-shot
(`cytir.ops.run_observed_one_shot`) under a fresh attempt namespace, CPU only (`CUDA_VISIBLE_DEVICES=`
empty, `ionice`/`nice` on the pack step). A step that does not report `succeeded` with rc 0 stops
the chain.

  gate     the immutable release verifies against its own `SHA256SUMS`; the combined FINMA parent
           run (`RUN.json`) must say `ready_for_web_pdf_profile_fidelity_and_current_chunking`,
           profile `web_pdf`, 2,956 source records, markdown documents and outlink parents, and
           `markdown_changed` false. The chain waits for the combine supervisor PID to exit first.
  fresh    the pack, audit, stage and attempt roots must not exist yet.
  pack     `cytir.corpus.stage_structured_markdown` under the release's chunk policy.
  profile  the web-PDF profile audit, run as the package module `PROFILE_AUDIT_MODULE`.
  seals    source-profile compliance seal, content-fidelity seal, `portableize_pack`,
           `adapt_pack_to_stage`, in ONE observed step (so the attempt record names the group).

Environment (all required, as the shell chain had them): AW_ARTIFACT_ROOT, AW_FINMA_COMBINE_WAITER_PID,
AW_FINMA_PARENT_ROOT, AW_FINMA_PACK_ROOT, AW_FINMA_AUDIT_ROOT, AW_FINMA_STAGE_ROOT,
AW_FINMA_ATTEMPT_ROOT, AW_RELEASE_DIR, plus
AW_FINMA_COMBINE_ADAPTER (the combine adapter named in the fidelity seal). `AW_PYTHON_BIN` picks the
interpreter (default: this one).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

PROFILE_AUDIT_MODULE = "cytir.ops.audits.finma_web_pdf_profile"
SOURCE_ID = "finma-production-current-exact-retained-pdfs-v1"
EXPECTED_DOCUMENTS = 2956
TOKENIZER_REVISION = "ac6544c8a46e00af67e330e85a9028c66b8cfd9a"
POLICY_FILE = "data/chunk-policy-struct-md-768w-o0-adjacent-sections-tail-1152-tok2560.v1.json"
PARENT_SCHEMA = "aw.finma-production-materialization-run.v1"
PARENT_STATUS = "ready_for_web_pdf_profile_fidelity_and_current_chunking"
VERIFICATION_METHOD = "finma-exact-retained-full-pdf-docling-contextual-outlinks-no-content-cap"
ENV_KEYS = ("AW_ARTIFACT_ROOT", "AW_FINMA_COMBINE_WAITER_PID", "AW_FINMA_PARENT_ROOT",
            "AW_FINMA_PACK_ROOT", "AW_FINMA_AUDIT_ROOT", "AW_FINMA_STAGE_ROOT",
            "AW_FINMA_ATTEMPT_ROOT", "AW_RELEASE_DIR", "AW_FINMA_COMBINE_ADAPTER")


class AdmissionRefused(Exception):
    """A precondition or a step failed; the message names which."""


def parent_gate(run: dict) -> None:
    """The combined parent run must be the complete 2,956-document web-PDF materialization."""
    counts = run.get("counts") or {}
    if (run.get("schema") != PARENT_SCHEMA or run.get("status") != PARENT_STATUS
            or run.get("source_profile") != "web_pdf"
            or counts.get("source_records") != EXPECTED_DOCUMENTS
            or counts.get("markdown_documents") != EXPECTED_DOCUMENTS
            or counts.get("outlink_parents") != EXPECTED_DOCUMENTS
            or (run.get("metadata_normalization") or {}).get("markdown_changed") is not False):
        raise AdmissionRefused("FINMA combined parent gate failed")


def read_env(environ=None) -> dict:
    environ = os.environ if environ is None else environ
    missing = [k for k in ENV_KEYS if not environ.get(k)]
    if missing:
        raise AdmissionRefused("set " + ", ".join(missing))
    return {k: environ[k] for k in ENV_KEYS}


def release_revision(release: pathlib.Path) -> str:
    sums = release / "SHA256SUMS"
    if not sums.is_file():
        raise AdmissionRefused(f"no SHA256SUMS in release {release}")
    return hashlib.sha256(sums.read_bytes()).hexdigest()


def verify_release(release: pathlib.Path, py: str) -> None:
    if subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=release,
                      stdout=subprocess.DEVNULL).returncode != 0:
        raise AdmissionRefused(f"release {release} fails its own SHA256SUMS")


def wait_for_pid(pid: int, poll: float = 15.0) -> None:
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            pass
        time.sleep(poll)


def step_pythonpath(env: dict) -> str:
    """The release directory only: every step is a package module."""
    return env["AW_RELEASE_DIR"]


def run_observed(name: str, cmd: list[str], *, env: dict, py: str, extra_env: dict | None = None):
    attempt = pathlib.Path(env["AW_FINMA_ATTEMPT_ROOT"]) / name
    child_env = dict(os.environ, PYTHONPATH=env["AW_RELEASE_DIR"])
    subprocess.run([py, "-m", "cytir.ops.run_observed_one_shot", "--attempt-root", str(attempt),
                    "--artifact-root", env["AW_ARTIFACT_ROOT"], "--release", env["AW_RELEASE_DIR"],
                    "--", *cmd], env=child_env, check=False)
    outcome = json.loads((attempt / "OUTCOME.json").read_text())
    if outcome.get("status") != "succeeded" or outcome.get("rc") != 0:
        raise AdmissionRefused(f"observed FINMA admission step failed: {outcome}")


def seal_steps(py: str, env: dict) -> list[list[str]]:
    rel, art, audit = env["AW_RELEASE_DIR"], env["AW_ARTIFACT_ROOT"], env["AW_FINMA_AUDIT_ROOT"]
    pack = env["AW_FINMA_PACK_ROOT"]
    registry = f"{rel}/data/source-preservation-profiles.v1.json"
    return [
        [py, "-m", "cytir.index.seal_source_profile_compliance", "--artifact-root", art,
         "--profile-registry", registry, "--profile", "web_pdf", "--source-id", SOURCE_ID,
         "--checks", f"{audit}/checks.json",
         "--verification-evidence", f"{audit}/verification-evidence.json",
         "--receipt", f"{audit}/source-profile-compliance.json"],
        [py, "-m", "cytir.index.seal_content_fidelity_audit", "--artifact-root", art,
         "--parent-run", f"{env['AW_FINMA_PARENT_ROOT']}/RUN.json",
         "--adapter-source", env["AW_FINMA_COMBINE_ADAPTER"], "--source-id", SOURCE_ID,
         "--verification-method", VERIFICATION_METHOD,
         "--receipt", f"{audit}/content-fidelity.json",
         "--envelope", f"{audit}/content-fidelity-envelope.json"],
        [py, "-m", "cytir.embed.portableize_pack", "--artifact-root", art, "--pack", pack],
        [py, "-m", "cytir.embed.adapt_pack_to_stage", "--artifact-root", art, "--pack", pack,
         "--pack-manifest", "PACK.portable.v2.json", "--output", env["AW_FINMA_STAGE_ROOT"]],
    ]


def run_seals(env: dict, py: str) -> None:
    """The inner body of the `seals` observed step: the four seals in order, first failure stops."""
    child_env = dict(os.environ, PYTHONPATH=step_pythonpath(env))
    for cmd in seal_steps(py, env):
        subprocess.run(cmd, env=child_env, check=True)


def admit(env: dict, py: str) -> str:
    release = pathlib.Path(env["AW_RELEASE_DIR"])
    revision = release_revision(release)
    verify_release(release, py)
    wait_for_pid(int(env["AW_FINMA_COMBINE_WAITER_PID"]))
    run_json = pathlib.Path(env["AW_FINMA_PARENT_ROOT"]) / "RUN.json"
    if not run_json.is_file():
        raise AdmissionRefused(f"no {run_json}")
    parent_gate(json.loads(run_json.read_text()))
    for key in ("AW_FINMA_PACK_ROOT", "AW_FINMA_AUDIT_ROOT", "AW_FINMA_STAGE_ROOT",
                "AW_FINMA_ATTEMPT_ROOT"):
        if os.path.exists(env[key]):
            raise AdmissionRefused(f"{key} already exists: {env[key]}")
    os.makedirs(env["AW_FINMA_ATTEMPT_ROOT"])
    art, rel = env["AW_ARTIFACT_ROOT"], env["AW_RELEASE_DIR"]
    run_observed("pack", ["ionice", "-c3", "nice", "-n15", "env", "CUDA_VISIBLE_DEVICES=",
                          f"PYTHONPATH={step_pythonpath(env)}", py, "-m", "cytir.corpus.stage_structured_markdown",
                          "--artifact-root", art, "--parent-root", env["AW_FINMA_PARENT_ROOT"],
                          "--parent-run", str(run_json), "--out", env["AW_FINMA_PACK_ROOT"],
                          "--policy", f"{rel}/{POLICY_FILE}", "--tokenizer-revision", TOKENIZER_REVISION,
                          "--producer-revision", revision, "--source-profile", "web_pdf"],
                 env=env, py=py)
    run_observed("profile", ["env", f"PYTHONPATH={env['AW_RELEASE_DIR']}", py, "-m", PROFILE_AUDIT_MODULE,
                             "--artifact-root", art, "--parent-run", str(run_json),
                             "--pack-root", env["AW_FINMA_PACK_ROOT"],
                             "--output-root", env["AW_FINMA_AUDIT_ROOT"],
                             "--profile-registry", f"{rel}/data/source-preservation-profiles.v1.json",
                             "--source-id", SOURCE_ID, "--expected-documents",
                             str(EXPECTED_DOCUMENTS)], env=env, py=py)
    run_observed("seals", [py, "-m", "cytir.ops.finma_exact_admission", "--seals"], env=env, py=py)
    return env["AW_FINMA_STAGE_ROOT"]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    py = os.environ.get("AW_PYTHON_BIN") or sys.executable
    try:
        env = read_env()
        if argv == ["--seals"]:
            run_seals(env, py)
            return 0
        stage = admit(env, py)
    except AdmissionRefused as exc:
        print(f"finma_exact_admission: {exc}", file=sys.stderr)
        return 1
    print(f"FINMA exact retained PDF admission terminal at {stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
