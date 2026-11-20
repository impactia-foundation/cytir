"""Tests for the ERIC L1 seal (cytir.sources.eric.l1_seal) (task 669/C).

Real incident: `eric-official-pdf-tranche-020000` refused 10,000/10,000
pages with `UnpersistedInputs` (ADR-105) -- its consolidated markdown
(`finalize_eric_pdf_materialization_workers.py`'s own `markdown/
documents.jsonl.gz` output shape) was never registered into the `l1/`
tree at all; no ERIC chain script ever calls `register_l1_release.py`.
Row shape below (`url`/`markdown_sha256`/`markdown`) is the exact real
shape read off the operator host's `education-eric-v1/materialized/official-pdf-
tranche-020000-.../markdown/documents.jsonl.gz` during this investigation.
"""
from __future__ import annotations

import gzip
import json
import pathlib

import pytest

from cytir.sources.eric import l1_seal as S
from cytir.corpus import register_l1_release as RL1
from cytir.corpus import verify_l1_coverage as VLC

REAL_ERIC_ROW = {
    "document_id": "d" * 64,
    "lang": None,
    "markdown": "DOCUMENT RESUME\n\nED 036150\n\nBy- Someone\nA Study\n",
    "markdown_sha256": "5d8987361375381397aed3ae6510f5e6b1cca5b787f2b2f2523fb53eea3ae7a0",
    "parent_identity": {"eric_id": "ED036150"},
    "parent_response": {},
    "source_links": [],
    "source_record_id": "ED036150",
    "source_version": "official ERIC full-text PDF response",
    "title": "ERIC ED036150",
    "url": "https://files.eric.ed.gov/fulltext/ED036150.pdf",
}


def _write_markdown_shard(parent_root: pathlib.Path, rows: list[dict]) -> None:
    markdown_dir = parent_root / S.MARKDOWN_SUBDIR
    markdown_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(markdown_dir / S.MARKDOWN_FILENAME, "wt",
                   encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_seal_registers_the_consolidated_markdown_shard(tmp_path, monkeypatch):
    parent_root = tmp_path / "materialized" / "tranche-020000.init"
    _write_markdown_shard(parent_root, [REAL_ERIC_ROW])
    corpus_root = tmp_path / "corpus"
    monkeypatch.setattr(VLC.PI, "assert_pages_persisted",
                        lambda pages, **kw: {"verdict": "PASS"})

    result = S.seal(str(parent_root), release_name="eric-tranche-020000",
                    corpus_root=str(corpus_root), refresh_index=False,
                    log=lambda m: None)

    assert result["register"]["files"] == 1
    dest = corpus_root / "l1" / "release=eric-tranche-020000"
    assert dest.is_dir()
    assert (dest / S.MARKDOWN_FILENAME).is_file()
    assert result["verify"]["status"] == "PASS"


def test_seal_raises_when_registration_finds_no_markdown_shard(tmp_path):
    parent_root = tmp_path / "materialized" / "empty.init"
    (parent_root / S.MARKDOWN_SUBDIR).mkdir(parents=True)
    corpus_root = tmp_path / "corpus"

    with pytest.raises(S.EricL1SealError, match="L1 registration FAILED"):
        S.seal(str(parent_root), release_name="eric-empty",
              corpus_root=str(corpus_root), refresh_index=False,
              log=lambda m: None)


def test_seal_refuses_to_refresh_an_index_the_drain_does_not_read(
        tmp_path, monkeypatch):
    """Real incident, task 679, 2026-09-30: tranche 080000's seal ran 21 h
    without AW_PERSISTED_INDEX_SQLITE_DIR and began building a second index
    on the HDD. With the refresh requested and the variable unset the seal
    must refuse BEFORE registering or touching any index."""
    monkeypatch.delenv(S.SQLITE_DIR_ENV, raising=False)
    parent_root = tmp_path / "materialized" / "tranche-080000.init"
    _write_markdown_shard(parent_root, [REAL_ERIC_ROW])
    corpus_root = tmp_path / "corpus"

    with pytest.raises(S.EricL1SealError, match=S.SQLITE_DIR_ENV):
        S.seal(str(parent_root), release_name="eric-tranche-080000",
              corpus_root=str(corpus_root), log=lambda m: None)
    assert not (corpus_root / "l1").exists()  # refused before registering


def test_seal_cli_exits_nonzero_without_the_index_dir(tmp_path, monkeypatch,
                                                      capsys):
    monkeypatch.delenv(S.SQLITE_DIR_ENV, raising=False)
    parent_root = tmp_path / "materialized" / "tranche-080000.init"
    _write_markdown_shard(parent_root, [REAL_ERIC_ROW])
    rc = S.main(["--parent-root", str(parent_root), "--release-name", "x",
                "--corpus-root", str(tmp_path / "corpus")])
    assert rc == 1
    assert S.SQLITE_DIR_ENV in capsys.readouterr().err


def test_seal_raises_when_the_index_still_does_not_see_it(tmp_path, monkeypatch):
    """The exact CC tranche4 incident, reproduced for ERIC: registration
    succeeds, but the persisted-input index still refuses -- must raise,
    never silently report success."""
    parent_root = tmp_path / "materialized" / "tranche-020000.init"
    _write_markdown_shard(parent_root, [REAL_ERIC_ROW])
    corpus_root = tmp_path / "corpus"

    def fake_assert(pages, **kw):
        raise VLC.PI.UnpersistedInputs(
            "[persisted-inputs] REFUSED eric-tranche-020000: 1 of 1 page(s) "
            "have no persisted L1 record", {"verdict": "REFUSED"})
    monkeypatch.setattr(VLC.PI, "assert_pages_persisted", fake_assert)

    with pytest.raises(S.EricL1SealError, match="not yet visible"):
        S.seal(str(parent_root), release_name="eric-tranche-020000",
              corpus_root=str(corpus_root), refresh_index=False,
              log=lambda m: None)


def test_seal_uses_l1_release_gz_store_field_names(tmp_path):
    """No transform needed: ERIC's own record already matches the
    `l1-release-gz` store's `url_fields`/`text_fields`
    (`data/persisted-input-stores.v1.json`) -- `register_l1_release.py`'s
    own field-shape guard proves this on the real row, unchanged."""
    parent_root = tmp_path / "materialized" / "tranche-020000.init"
    _write_markdown_shard(parent_root, [REAL_ERIC_ROW])
    shape = RL1.guard_field_shape(
        parent_root / S.MARKDOWN_SUBDIR / S.MARKDOWN_FILENAME)
    assert shape["url_field"] == "url"
    assert shape["text_field"] == "markdown"


def test_cli_seal_returns_nonzero_on_refusal(tmp_path, capsys):
    parent_root = tmp_path / "materialized" / "empty.init"
    (parent_root / S.MARKDOWN_SUBDIR).mkdir(parents=True)
    corpus_root = tmp_path / "corpus"

    rc = S.main(["--parent-root", str(parent_root),
                "--release-name", "eric-empty",
                "--corpus-root", str(corpus_root), "--no-refresh-index"])
    assert rc == 1
    assert "L1 registration FAILED" in capsys.readouterr().err


def test_cli_seal_returns_zero_on_success(tmp_path, monkeypatch, capsys):
    parent_root = tmp_path / "materialized" / "tranche-020000.init"
    _write_markdown_shard(parent_root, [REAL_ERIC_ROW])
    corpus_root = tmp_path / "corpus"
    monkeypatch.setattr(VLC.PI, "assert_pages_persisted",
                        lambda pages, **kw: {"verdict": "PASS"})

    rc = S.main(["--parent-root", str(parent_root),
                "--release-name", "eric-tranche-020000",
                "--corpus-root", str(corpus_root), "--no-refresh-index"])
    assert rc == 0
