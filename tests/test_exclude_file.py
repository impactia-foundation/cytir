"""Tests for cytir/index/exclude_file.py and the drain's use of it (task 680)."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from cytir.index import exclude_file as XF


def _write(path: Path, rows: list[dict], *, gz: bool = True) -> str:
    opener = gzip.open if gz else open
    with opener(path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return XF.sha256_file(path)


def test_load_exclude_ids_reads_a_pinned_gz_file(tmp_path):
    f = tmp_path / "trimmed.jsonl.gz"
    sha = _write(f, [{"point_id": "a", "work_id": "W1"},
                     {"point_id": "b", "work_id": "W2"}])
    got = XF.load_exclude_ids(str(f), sha256=sha, reason="below_admission_floor")
    assert got == {"a": "below_admission_floor", "b": "below_admission_floor"}


def test_load_exclude_ids_reads_plain_jsonl(tmp_path):
    f = tmp_path / "trimmed.jsonl"
    sha = _write(f, [{"point_id": "a"}], gz=False)
    assert XF.load_exclude_ids(str(f), sha256=sha, reason="r") == {"a": "r"}


def test_load_exclude_ids_refuses_a_hash_mismatch(tmp_path):
    f = tmp_path / "trimmed.jsonl.gz"
    _write(f, [{"point_id": "a"}])
    with pytest.raises(XF.ExcludeFileError, match="not the manifest's pinned"):
        XF.load_exclude_ids(str(f), sha256="0" * 64, reason="r")


def test_load_exclude_ids_refuses_without_a_pinned_hash_or_a_reason(tmp_path):
    f = tmp_path / "trimmed.jsonl.gz"
    sha = _write(f, [{"point_id": "a"}])
    with pytest.raises(XF.ExcludeFileError, match="exclude_ids_sha256"):
        XF.load_exclude_ids(str(f), sha256=None, reason="r")
    with pytest.raises(XF.ExcludeFileError, match="exclude_ids_reason"):
        XF.load_exclude_ids(str(f), sha256=sha, reason=None)


def test_load_exclude_ids_refuses_a_missing_file(tmp_path):
    with pytest.raises(XF.ExcludeFileError, match="does not exist"):
        XF.load_exclude_ids(str(tmp_path / "nope.jsonl.gz"),
                            sha256="0" * 64, reason="r")


def test_load_exclude_ids_refuses_a_file_that_excludes_nothing(tmp_path):
    f = tmp_path / "empty.jsonl.gz"
    sha = _write(f, [])
    with pytest.raises(XF.ExcludeFileError, match="0 point ids"):
        XF.load_exclude_ids(str(f), sha256=sha, reason="r")


def test_drain_merges_the_manifest_exclusion_file_into_its_exclude_map(tmp_path):
    """The drain derives `exclude_ids` in ONE place for the dry plan and the
    write (promote_queue.language_scope_and_waiver_exclusions); a manifest's
    exclusion file must arrive there, with its reason, and never overwrite a
    more specific reason a language rule already gave the same id."""
    from cytir.index import promote_queue as PQ

    f = tmp_path / "trimmed.jsonl.gz"
    sha = _write(f, [{"point_id": "t-1"}, {"point_id": "t-2"}])
    manifest = {"pack_id": "p", "pack_chunks": str(tmp_path),
                "exclude_ids_file": str(f), "exclude_ids_sha256": sha,
                "exclude_ids_reason": "below_admission_floor"}
    exclude_map, lang_report, waiver_report = \
        PQ.language_scope_and_waiver_exclusions(manifest)
    assert exclude_map == {"t-1": "below_admission_floor",
                           "t-2": "below_admission_floor"}
    assert lang_report is None and waiver_report is None


def test_drain_refuses_a_manifest_whose_exclusion_file_hash_is_wrong(tmp_path):
    from cytir.index import promote_queue as PQ

    f = tmp_path / "trimmed.jsonl.gz"
    _write(f, [{"point_id": "t-1"}])
    manifest = {"pack_id": "p", "pack_chunks": str(tmp_path),
                "exclude_ids_file": str(f), "exclude_ids_sha256": "0" * 64,
                "exclude_ids_reason": "below_admission_floor"}
    with pytest.raises(XF.ExcludeFileError):
        PQ.language_scope_and_waiver_exclusions(manifest)
