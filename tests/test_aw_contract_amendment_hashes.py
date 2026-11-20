"""Task 2486: the hash convention of the contract amendments rows is checkable.

Convention (the notes in the rows say the same):
- `old_file_sha256` is the sha256 of the contracts file at a committed revision (the
  `origin/main` head before the task). Recomputed here from git history: the value of the
  LATEST row must equal the sha of some revision of the file in the file's history.
- `pre_amendment_file_sha256` is the sha256 of the file text with the row's own object (and
  the comma before it) cut out. It is never a committed blob (the row is appended in the same
  commit), so git cannot reproduce it; the textual cut is the check, run by
  `cytir.sources.contract.pre_amendment_sha256` for the LATEST row only.
- Older rows are historic. Many were batched or rebased and name no committed revision; they
  are not re-checked. One row is a recorded exception: task 1317 / D10 (2026-10-25), whose
  pre_amendment value c3a1a984... reproduces neither from git nor from a textual cut.
  The sealed value is never rewritten; the test pins it as it is.
The git half skips when no git history is available (the public cut ships no `.git`).
"""
import hashlib
import json
import subprocess

import pytest

from cytir.core.paths import data_path
from cytir.sources.contract import pre_amendment_sha256
from tests.helpers.repo_files import REPO_ROOT

_REL = "data/source-family-contracts.v1.json"
_EXC_TASK = "task 1317 / D10"
_EXC_PRE = "c3a1a984f3c35d0cf7c289ffefe19661c6513834537875b5468f39b76a9c1752"


def _git(*args: str) -> bytes | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                             timeout=120, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _history_shas() -> set[str]:
    log = _git("log", "--format=%H", "--", _REL)
    if not log or _git("rev-parse", "--is-shallow-repository") != b"false\n":
        pytest.skip("no complete git history for the contracts file (public cut or shallow clone)")
    shas = set()
    for rev in log.decode().split():
        blob = _git("show", f"{rev}:./{_REL}")
        if blob is not None:
            shas.add(hashlib.sha256(blob).hexdigest())
    return shas


def _rows() -> list[dict]:
    return json.loads(data_path("source-family-contracts.v1.json").read_text("utf-8"))["amendments"]


def test_latest_row_old_file_sha256_is_a_committed_revision():
    shas = _history_shas()
    last = _rows()[-1]
    assert last["old_file_sha256"] in shas, (
        f"{last['task']}: old_file_sha256 matches no committed revision of {_REL}")


def test_a_wrong_old_file_sha256_is_caught():
    assert "0" * 64 not in _history_shas()


def test_latest_row_pre_amendment_reproduces_from_the_text():
    rec, comp = pre_amendment_sha256(data_path("source-family-contracts.v1.json").read_text("utf-8"))
    assert rec == comp


def test_1317_row_is_the_recorded_exception():
    row = next(r for r in _rows() if r["task"] == _EXC_TASK)
    assert row["pre_amendment_file_sha256"] == _EXC_PRE  # sealed, never rewritten
    assert _EXC_PRE not in _history_shas()  # not a committed blob either
