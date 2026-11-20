"""The NEXT.md resume command of `run_windows` is pasteable (task 696 slice 4, round 2).

`AW_OPERATOR_URL=<your-url>` is parsed by bash as a redirection (it reads a file named `your-url>`),
so a resume command copied from NEXT.md failed or did something else. The command now renders the
running process's effective identity and overlay, and quotes every placeholder.
"""
from __future__ import annotations

import shlex
from pathlib import Path

import pytest

pytest.importorskip("qdrant_client")

from cytir.sources import run_windows as RW  # noqa: E402


def _resume_tokens(text: str) -> list[str]:
    block = text.split("Resume (idempotent; never delete a DONE window):")[1].split("The compose timer")[0]
    lex = shlex.shlex(block.replace("\\\n", " "), posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    return list(lex)


def _render(monkeypatch, **env):
    for name in ("AW_OPERATOR_URL", "AW_OPERATOR_CONTACT", "AW_OPERATOR_OVERLAY", "AW_WORK_ROOT",
                 "AW_PERSISTED_INDEX_SQLITE_DIR"):
        monkeypatch.delenv(name, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return RW.next_md("fam", "/out/r", "/out/c", {"leg": "fr"}, "http://t", "http://p")


def test_unconfigured_placeholders_are_quoted_so_bash_reads_no_redirection(monkeypatch, tmp_path):
    from cytir.core import user_agent as UA
    monkeypatch.setattr(UA, "CONFIG_FILE", tmp_path / "operator.json")
    text = _render(monkeypatch)
    tokens = _resume_tokens(text)
    assert not [t for t in tokens if t in ("<", ">", "<<", ">>", "|")], tokens   # no redirection token
    assert "AW_OPERATOR_URL=<your-url>" in tokens and "AW_PERSISTED_INDEX_SQLITE_DIR=<persisted>" in tokens
    assert "<checkout>" not in text


def test_the_effective_identity_overlay_and_work_root_of_the_process_are_rendered(monkeypatch, tmp_path):
    overlay = tmp_path / "ov dir" / "operator.json"        # a space: must be quoted
    text = _render(monkeypatch, AW_OPERATOR_URL="https://example.org/about", AW_OPERATOR_CONTACT="a@example.org",
                   AW_OPERATOR_OVERLAY=str(overlay), AW_WORK_ROOT="/w", AW_PERSISTED_INDEX_SQLITE_DIR="/p")
    tokens = _resume_tokens(text)
    assert "AW_OPERATOR_URL=https://example.org/about" in tokens
    assert "AW_OPERATOR_CONTACT=a@example.org" in tokens
    assert f"AW_OPERATOR_OVERLAY={overlay}" in tokens and "AW_WORK_ROOT=/w" in tokens
    assert "AW_PERSISTED_INDEX_SQLITE_DIR=/p" in tokens
    assert "python" in tokens and "cytir.sources.run_windows" in tokens
    assert tokens[0] == "cd" and tokens[1] == str(Path(RW.__file__).resolve().parents[2])
