"""Task 2496: `cytir.sources.contract amend` edits the sealed contracts file textually and appends a
correct amendments row, or refuses and writes nothing."""
import hashlib
import json
import shutil
import subprocess

import pytest

from cytir.core.paths import data_path
from cytir.sources import contract as C

_SYNTH = '''{
 "schema": "aw.source-family-contract.v2",
 "families": [
  {
   "family_id": "alpha",
   "status": "backfilled",
   "licence": {
    "spdx": "CC0-1.0",
    "basis": [
     "a",
     "b"
    ]
   }
  },
  {
   "family_id": "beta",
   "status": "declared"
  }
 ],
 "amendments": [
    {
      "date": "2026-10-25",
      "task": "task 1",
      "old_file_sha256": "00",
      "pre_amendment_file_sha256": "00"
    },
    {"date": "2026-11-02", "task": "task 2", "old_file_sha256": "11", "pre_amendment_file_sha256": "11"}
  ]
}
'''


def _ok(_text):
    return []


def _args(synth, *extra, sets=("a=1",)):
    out = ["--contracts", str(synth), "--family", "beta", "--reason", "r", "--task", "7", *extra]
    for s in sets:
        out += ["--set", s]
    return out


@pytest.fixture
def synth(tmp_path):
    p = tmp_path / "contracts.json"
    p.write_text(_SYNTH, encoding="utf-8")
    return p


def test_insert_new_key_is_textual_and_row_is_correct(synth):
    old, new = C.amend(synth, "beta", [("producer_ids", {"x/y.py": {"status": "frozen-label"}})],
                       "declare", "7", today="2026-11-05", check=_ok)
    assert old == _SYNTH
    assert ('  {\n   "family_id": "beta",\n   "producer_ids": {\n    "x/y.py": {\n     "status": '
            '"frozen-label"\n    }\n   },\n   "status": "declared"') in new
    row = json.loads(new)["amendments"][-1]
    assert row["task"] == "task 7" and row["date"] == "2026-11-05"
    assert row["families"] == ["beta"] and row["keys"] == ["producer_ids"]
    assert row["old_file_sha256"] == hashlib.sha256(_SYNTH.encode()).hexdigest()
    rec, comp = C.pre_amendment_sha256(new)
    assert rec == comp


def test_byte_identical_outside_the_two_edits(synth):
    _, new = C.amend(synth, "beta", [("extra", 1)], "r", "7", today="2026-11-05", check=_ok)
    marker = '\n   "status": "declared"'
    cut = new.index('\n   "extra"')
    assert new[:cut] == _SYNTH[:cut]
    tail = new[new.index(marker):]
    row_start = tail.index(',\n    {"date": "2026-11-05"')
    row_end = tail.index("\n  ]")
    assert tail[:row_start] + tail[row_end:] == _SYNTH[_SYNTH.index(marker):]


def test_dotted_key_replaces_only_that_value(synth):
    _, new = C.amend(synth, "alpha", [("licence.spdx", "MIT")], "r", "7", check=_ok)
    assert json.loads(new)["families"][0]["licence"] == {"spdx": "MIT", "basis": ["a", "b"]}
    assert new[:new.index('"spdx"')] == _SYNTH[:_SYNTH.index('"spdx"')]


def test_dot_inside_a_key_is_escaped(synth):
    _, new = C.amend(synth, "beta", [("producer_ids.vendor/x\\.py", {"s": 1})], "r", "7", check=_ok)
    assert json.loads(new)["families"][1]["producer_ids"] == {"vendor/x.py": {"s": 1}}


def test_unknown_family_refused(synth):
    with pytest.raises(C.FamilyContractError, match="unknown family"):
        C.amend(synth, "nope", [("a", 1)], "r", "7", check=_ok)


def test_check_failure_refused_and_nothing_written(synth, capsys):
    # the synthetic file has no roster and no required blocks: the real check must fail it
    assert C.amend_main(_args(synth)) == 1
    assert "would fail --check" in capsys.readouterr().err
    assert synth.read_text() == _SYNTH


def test_invalid_json_value_refused(synth, capsys):
    assert C.amend_main(_args(synth, sets=("a={bad",))) == 1
    assert "not valid JSON" in capsys.readouterr().err
    assert synth.read_text() == _SYNTH


def test_dry_run_prints_diff_and_writes_nothing(synth, capsys, monkeypatch):
    monkeypatch.setattr(C, "check_text", _ok)
    assert C.amend_main(_args(synth, "--dry-run")) == 0
    assert '+   "a": 1' in capsys.readouterr().out
    assert synth.read_text() == _SYNTH


def test_write_replaces_the_file(synth, monkeypatch):
    monkeypatch.setattr(C, "check_text", _ok)
    assert C.amend_main(_args(synth)) == 0
    assert json.loads(synth.read_text())["families"][1]["a"] == 1


def _vcs(cwd, *a):
    subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                   check=True, capture_output=True)


def test_dirty_file_refused(synth, capsys, monkeypatch):
    monkeypatch.setattr(C, "check_text", _ok)
    _vcs(synth.parent, "init", "-q")
    _vcs(synth.parent, "add", ".")
    _vcs(synth.parent, "commit", "-qm", "x")
    synth.write_text(_SYNTH + " ", encoding="utf-8")
    assert C.amend_main(_args(synth)) == 1
    assert "uncommitted" in capsys.readouterr().err
    assert synth.read_text() == _SYNTH + " "


def test_real_file_copy_amend_passes_check_and_hash_convention(tmp_path):
    p = tmp_path / "source-family-contracts.v1.json"
    shutil.copy(data_path("source-family-contracts.v1.json"), p)
    raw = p.read_bytes()
    fid = json.loads(raw)["families"][0]["family_id"]
    old, new = C.amend(p, fid, [("producer_ids.zz/t\\.py", {"status": "frozen-label", "meaning": "m", "code_home": "none (frozen label; test)",
                                           "declared": "task 2496", "date": "2026-11-05"})],
                       "synthetic test edit", "2496")
    assert C.check_text(new) == []
    rec, comp = C.pre_amendment_sha256(new)
    assert rec == comp
    assert json.loads(new)["amendments"][-1]["old_file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert old == raw.decode()
