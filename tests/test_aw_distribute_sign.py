# Task 650 (migration S8): lives in tests/ with the rest of the package's tests (task 696 slice 2).
"""Tests for cytir.distribute.sign (task 650, S8).

No real `minisign` binary is assumed on the test runner (it is not a Python
dependency — see sign.py's docstring). These tests install a small FAKE
`minisign` shell script on PATH so the wiring (argument shape, exit-code
handling, error surfacing) is proven without needing the real tool; a
separate CI release-rehearsal job (task 650 step 4) exercises the real
binary end to end.
"""
from __future__ import annotations

import os
import stat
import textwrap

import pytest

from cytir.distribute import sign


def _install_fake_minisign(tmp_path, monkeypatch, *, sign_ok=True, verify_ok=True):
    script = tmp_path / "minisign"
    script.write_text(textwrap.dedent(f"""\
        #!/bin/sh
        case "$1" in
          -S)
            # -S -s <seckey> -m <manifest>
            {"exit 0" if sign_ok else "echo 'fake sign failure' >&2; exit 1"}
            ;;
          -V)
            # -V -P <pubkey> -m <manifest> -x <sigfile>
            {"exit 0" if verify_ok else "echo 'fake verify failure' >&2; exit 1"}
            ;;
          *)
            echo "unexpected args: $@" >&2
            exit 2
            ;;
        esac
    """))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))


def test_minisign_available_false_when_not_on_path(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent-bin-dir")
    assert sign.minisign_available() is False


def test_sign_manifest_missing_tool_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", "/nonexistent-bin-dir")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(sign.SigningToolMissing):
        sign.sign_manifest(str(manifest), str(tmp_path / "key.sec"))


def test_sign_manifest_success(tmp_path, monkeypatch):
    _install_fake_minisign(tmp_path, monkeypatch, sign_ok=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    sig_path = sign.sign_manifest(str(manifest), str(tmp_path / "key.sec"))
    assert sig_path == str(manifest) + ".minisig"


def test_sign_manifest_failure_raises(tmp_path, monkeypatch):
    _install_fake_minisign(tmp_path, monkeypatch, sign_ok=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(sign.SignatureVerificationFailed):
        sign.sign_manifest(str(manifest), str(tmp_path / "key.sec"))


def test_verify_manifest_success(tmp_path, monkeypatch):
    _install_fake_minisign(tmp_path, monkeypatch, verify_ok=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    sig = tmp_path / "manifest.json.minisig"
    sig.write_text("fake-sig")
    sign.verify_manifest(str(manifest), str(sig), pubkey="RWfakepubkey")


def test_verify_manifest_failure_raises(tmp_path, monkeypatch):
    _install_fake_minisign(tmp_path, monkeypatch, verify_ok=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    sig = tmp_path / "manifest.json.minisig"
    sig.write_text("fake-sig")
    with pytest.raises(sign.SignatureVerificationFailed):
        sign.verify_manifest(str(manifest), str(sig), pubkey="RWfakepubkey")
