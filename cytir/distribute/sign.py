"""cytir.distribute.sign — minisign detached signatures for the dump manifest
(task 650, S8; specs/dump-format.md §6).

Shells out to the `minisign` binary (ISC licence, Apache-compatible) rather
than reimplementing Ed25519 — a hand-rolled signer is exactly the kind of
"closer to an embedding than to a document" false economy this project's own
doctrine warns against for cryptographic code specifically: get it from the
audited tool, don't re-derive it. `minisign` is not declared as a Python
dependency anywhere (it is a standalone binary); the CI release-rehearsal
workflow installs it the same way `.github/workflows/ci.yml`'s `secrets` job
already installs `gitleaks` — a pinned release tarball, not a package
manager.

`cyt dump` treats signing as OPTIONAL but loud: omitting `--sign-key` ships an
explicitly unsigned dump (manifest.json with no `signature` block) and prints
a warning; `cyt apply` refuses an unsigned dump unless `--allow-unsigned` is
passed. There is no silent "signed with nothing" fallback (engineering-
doctrine #3, fail loud or don't fail) — if `--sign-key` is given and the
`minisign` binary is missing, `sign_manifest` raises rather than skipping.
"""
from __future__ import annotations

import shutil
import subprocess


class SigningToolMissing(RuntimeError):
    pass


class SignatureVerificationFailed(RuntimeError):
    pass


def _minisign_path() -> str:
    path = shutil.which("minisign")
    if not path:
        raise SigningToolMissing(
            "the `minisign` binary is not on PATH. Install it "
            "(https://github.com/jedisct1/minisign) or omit --sign-key to "
            "ship an explicitly unsigned dump.")
    return path


def sign_manifest(manifest_path: str, secret_key_path: str, *,
                   password: str | None = None) -> str:
    """Detached-sign `manifest_path`; returns the signature file's path
    (`<manifest_path>.minisig`, minisign's own naming convention)."""
    minisign = _minisign_path()
    cmd = [minisign, "-S", "-s", secret_key_path, "-m", manifest_path]
    proc = subprocess.run(
        cmd, input=(password + "\n") if password is not None else None,
        text=True, capture_output=True)
    if proc.returncode != 0:
        raise SignatureVerificationFailed(
            f"minisign signing failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()}")
    return manifest_path + ".minisig"


def verify_manifest(manifest_path: str, sig_path: str, pubkey: str) -> None:
    """Raises `SignatureVerificationFailed` on any mismatch; returns None on
    success. `pubkey` is the minisign public key STRING (the `RW...` value
    from a `.pub` file), not a path — the caller reads it from wherever the
    public key is published (repo NOTICE / README, never inside the dump
    itself, specs/dump-format.md §6)."""
    minisign = _minisign_path()
    cmd = [minisign, "-V", "-P", pubkey, "-m", manifest_path, "-x", sig_path]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise SignatureVerificationFailed(
            f"minisign verification FAILED for {manifest_path}: "
            f"{(proc.stderr or proc.stdout).strip()}")


def minisign_available() -> bool:
    return shutil.which("minisign") is not None
