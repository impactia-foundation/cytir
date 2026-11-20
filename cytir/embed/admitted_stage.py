#!/usr/bin/env python3
"""Wait for one portable admitted stage, then embed/verify/seal it fail-closed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"
REVISION = "ac6544c8a46e00af67e330e85a9028c66b8cfd9a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_ref(root: Path, ref: dict, label: str) -> Path:
    uri = str(ref.get("uri") or "")
    if not uri.startswith("artifact://"):
        raise ValueError(f"{label} lacks a portable artifact URI")
    relative = Path(uri.removeprefix("artifact://"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} has an unsafe artifact URI")
    path = root / relative
    if (not path.is_file() or path.stat().st_size != ref.get("bytes")
            or sha256_file(path) != ref.get("sha256")):
        raise ValueError(f"{label} bytes/SHA do not resolve below global root")
    return path


def validate_admission(*, root: Path, stage: Path, profile: Path,
                       fidelity: Path, expected_rows: int,
                       expected_pack_sha256: str) -> dict:
    stage_receipt = json.loads((stage / "STAGE-RECEIPT.json").read_text())
    if (stage_receipt.get("schema") != "aw.stage-receipt.v1"
            or stage_receipt.get("status") != "complete"
            or stage_receipt.get("records") != expected_rows
            or stage_receipt.get("rows") != expected_rows
            or stage_receipt.get("deterministic_encoding") is not True
            or (stage_receipt.get("producer") or {}).get("version")
            != "v3-recursive-global-artifact-root"):
        raise ValueError("stage is not the exact recursive-portable terminal stage")
    pack_path = checked_ref(root, stage_receipt.get("pack") or {}, "pack")
    if sha256_file(pack_path) != expected_pack_sha256:
        raise ValueError("stage pack SHA differs from admitted pack")
    chunk_receipt_path = checked_ref(
        root, stage_receipt.get("chunk_receipt") or {}, "chunk receipt")
    pack_value = json.loads(pack_path.read_text())
    pack_chunk_ref = (pack_value.get("chunks") or {}).get("receipt") or {}
    nested_receipt_path = checked_ref(root, pack_chunk_ref, "PACK chunk receipt")
    if nested_receipt_path != chunk_receipt_path:
        raise ValueError("stage and PACK bind different chunk receipts")
    nested_receipt = json.loads(nested_receipt_path.read_text())
    nested_artifacts = nested_receipt.get("artifacts")
    if not isinstance(nested_artifacts, list) or not nested_artifacts:
        raise ValueError("PACK chunk receipt has no artifact inventory")
    if sum(int(row.get("bytes") or -1) for row in nested_artifacts) != nested_receipt.get("bytes"):
        raise ValueError("PACK chunk artifact byte sum differs from receipt")
    for index, ref in enumerate(nested_artifacts):
        checked_ref(root, ref, f"PACK chunk artifact {index}")
    artifacts = stage_receipt.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("stage has no artifact inventory")
    if sum(int(row.get("bytes") or -1) for row in artifacts) != stage_receipt.get("bytes"):
        raise ValueError("stage artifact byte sum differs from receipt")
    for index, ref in enumerate(artifacts):
        checked_ref(root, ref, f"stage artifact {index}")

    profile_value = json.loads(profile.read_text())
    checks = profile_value.get("checks") or {}
    if (profile_value.get("schema") != "aw.source-profile-compliance-receipt.v1"
            or not checks or any(check.get("status") != "PASS"
                                 for values in checks.values() for check in values)):
        raise ValueError("source-profile receipt is not all-PASS")
    fidelity_value = json.loads(fidelity.read_text())
    if (fidelity_value.get("schema") != "aw.content-fidelity-receipt.v1"
            or fidelity_value.get("scope") != "admitted-original-and-principal-content"
            or fidelity_value.get("original_bytes_complete") is not True
            or fidelity_value.get("markdown_content_complete") is not True
            or fidelity_value.get("source_text_truncation_applied") is not False):
        raise ValueError("content-fidelity receipt is not admissible")
    return stage_receipt


def gpu_writer_pids() -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True)
    return [int(line) for line in result.stdout.splitlines() if line.strip()]


def verification_argv(*, python: str, verifier: Path, stage: Path,
                      vectors: Path, output: Path,
                      artifact_root: Path | None = None) -> list[str]:
    """Build the immutable Task413A verifier CLI.

    Legacy hash-bound verifiers resolve stage/vector artifacts from explicit
    paths only.  The portable-aware verifier also accepts the global root so
    ``artifact://`` source bindings can be rehashed; callers must declare that
    capability rather than guessing a CLI flag.
    """
    argv = [python, str(verifier), "--stage", str(stage), "--vectors",
            str(vectors)]
    if artifact_root is not None:
        argv.extend(("--artifact-root", str(artifact_root)))
    argv.extend(("--out", str(output)))
    return argv


def verifier_supports_artifact_root(*, python: str, verifier: Path) -> bool:
    """Preflight the portable verifier contract before spending GPU time."""
    result = subprocess.run(
        [python, str(verifier), "--help"], text=True, capture_output=True,
        check=False)
    return result.returncode == 0 and "--artifact-root" in result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--stage-producer-pid", type=int, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-pack-sha256", required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--embedder", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--existing-embed-pid", type=int)
    parser.add_argument("--output-name", default="RECEIPT.portable.v4.json")
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    stage_receipt_path = args.stage / "STAGE-RECEIPT.json"
    while not stage_receipt_path.is_file():
        if not Path(f"/proc/{args.stage_producer_pid}").exists():
            raise ValueError("stage producer exited without a terminal receipt")
        time.sleep(args.poll_seconds)
    validate_admission(
        root=root, stage=args.stage.resolve(), profile=args.profile.resolve(),
        fidelity=args.fidelity.resolve(), expected_rows=args.expected_rows,
        expected_pack_sha256=args.expected_pack_sha256)
    if not verifier_supports_artifact_root(
            python=args.python, verifier=args.verifier.resolve()):
        raise ValueError(
            "portable stage verifier lacks required --artifact-root support")
    if args.existing_embed_pid is not None:
        while Path(f"/proc/{args.existing_embed_pid}").exists():
            time.sleep(args.poll_seconds)
        if not args.vectors.is_dir():
            raise ValueError("existing embed exited without vector output")
    else:
        if args.vectors.exists():
            raise ValueError("refusing pre-existing vector output")
        while gpu_writer_pids():
            print(f"[admitted-embed] waiting sole GPU writer {gpu_writer_pids()}",
                  flush=True)
            time.sleep(args.poll_seconds)
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        subprocess.run([
            args.python, str(args.embedder), "--in", str(args.stage.resolve()),
            "--out", str(args.vectors.resolve()), "--model", MODEL,
            "--revision", REVISION, "--max-seq", "2560", "--batch", "16",
            "--chunk", "4096", "--device", "cuda:0", "--compute-dtype", "float16",
            "--rank", "0", "--world", "1"], check=True, env=env)
    if args.verification.exists():
        raise ValueError("refusing pre-existing verification output")
    # The verifier writes one immutable file but does not create an arbitrary
    # caller-selected namespace.  Establish that namespace after the collision
    # check.  This does not touch completed vector bytes, so verifier failures
    # remain safely retryable without re-embedding.
    args.verification.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(verification_argv(
        python=args.python, verifier=args.verifier.resolve(),
        stage=args.stage.resolve(), vectors=args.vectors.resolve(),
        output=args.verification.resolve(),
        artifact_root=root),
        check=True)
    release = Path(__file__).resolve().parents[2]
    subprocess.run([
        args.python, str(release / "cytir/embed/vector_rollup.py"),
        "--stage", str(args.stage.resolve()), "--vectors",
        str(args.vectors.resolve()), "--artifact-root", str(root),
        "--verification", str(args.verification.resolve()), "--output-name",
        args.output_name], check=True)


if __name__ == "__main__":
    main()
