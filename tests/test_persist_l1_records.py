#!/usr/bin/env python3
"""[test] ADR-105 / task 581 -- a lane files its inputs, then passes the guard.

Measured on the real cron inputs before this module existed (the 2026-09-11 and
2026-09-12 news stages, task 581's cron-dryrun artifact): every news
source FAILED the persisted-inputs guard — `news` 5,188 refused of 5,189 pages,
`chmedia` 3,585 of 3,586, `swissinfo` 1,603 of 1,603, `hn` 488 of 494 — while
the stage's OWN `news_documents.jsonl` held L1 markdown for 5,188 of the 5,676
refused pages. The text was never missing; it was never filed.

So the fix is filing, not a wider bar. These tests hold that line: the writer
refuses a record it cannot file honestly, the overlay is evidence rather than
an assertion, and a lane that persists first passes the guard in the same run —
before any index rebuild.

Nothing here fetches; the build-script test directory's no-network fixture would fail the
suite if it tried.
"""
from __future__ import annotations

import gzip
import json
import hashlib
import os
import pathlib
import sys

import pytest

from cytir.corpus import persist_l1_records as PL1  # noqa: E402
import cytir.corpus.persisted as PI  # noqa: E402
import cytir.core.common as AC  # noqa: E402

# One real news row, in the shape `news_documents.jsonl` carries.
NEWS_URL = "https://www.swissinfo.ch/eng/swiss-politics/example-article/12345"
NEWS_MD = "# Example article\n\nThe Federal Council said on Friday...\n"


def _rec(url=NEWS_URL, text=NEWS_MD, **kw):
    r = {"url": url, "markdown": text, "publisher": "swissinfo.ch"}
    r.update(kw)
    return r


# -- 1. the writer ---------------------------------------------------------

def test_records_land_in_a_path_the_tracked_registry_already_covers(tmp_path):
    written = PL1.write_l1_records([_rec()], release="freshness", origin="test",
                                   source="2026-09-12", root=str(tmp_path),
                                   log=lambda *a, **k: None)
    path = written[AC.canonical_url(NEWS_URL)]
    assert "/l1/release=freshness/source=2026-09-12/" in path
    # the registry is not edited to make this pass: an existing store's glob
    # already matches it.
    store = next(s for s in PI.stores() if s.store_id == "l1-release-gz")
    assert store.files(str(tmp_path)) == [path]


def test_the_written_record_is_the_declared_schema_with_a_verified_sha(tmp_path):
    PL1.write_l1_records([_rec()], release="r", source="s", origin="test",
                         root=str(tmp_path), log=lambda *a, **k: None)
    f = next(pathlib.Path(tmp_path).rglob("l1-markdown-*.jsonl.gz"))
    rec = json.loads(gzip.open(f, "rt", encoding="utf-8").readline())
    assert rec["schema"] == "aw.l1-markdown.v1"
    assert rec["url"] == NEWS_URL
    assert rec["canonical_url"] == AC.canonical_url(NEWS_URL)
    assert rec["markdown"] == NEWS_MD
    assert rec["markdown_sha256"] == hashlib.sha256(
        NEWS_MD.encode("utf-8")).hexdigest()


def test_a_receipt_says_what_was_written_from_what(tmp_path):
    PL1.write_l1_records([_rec(), _rec(url=NEWS_URL + "b")], release="r",
                         source="s", origin="test", root=str(tmp_path),
                         note="freshness_ingest", log=lambda *a, **k: None)
    r = json.loads((pathlib.Path(tmp_path) / "l1" / "release=r" / "source=s"
                    / "RECEIPT.json").read_text())
    assert r["records"] == 2 and r["distinct_pages"] == 2
    assert r["note"] == "freshness_ingest"
    assert r["files"][0]["rows"] == 2 and r["files"][0]["sha256"]


def test_a_record_with_no_text_is_refused_not_skipped(tmp_path):
    """A writer that silently drops rows is how a population stops being
    mentioned (corpus-doctrine 2)."""
    with pytest.raises(PL1.L1PersistError, match="no text"):
        PL1.write_l1_records([_rec(text="   ")], release="r", source="s",
                             origin="test", root=str(tmp_path),
                             log=lambda *a, **k: None)
    with pytest.raises(PL1.L1PersistError, match="no url"):
        PL1.write_l1_records([{"markdown": "x"}], release="r", source="s",
                             origin="test", root=str(tmp_path),
                             log=lambda *a, **k: None)


def test_a_declared_sha_that_does_not_match_the_text_is_refused(tmp_path):
    with pytest.raises(PL1.L1PersistError, match="does not"):
        PL1.write_l1_records([_rec(markdown_sha256="0" * 64)], release="r",
                             source="s", origin="test", root=str(tmp_path),
                             log=lambda *a, **k: None)


def test_a_missing_corpus_root_is_refused_rather_than_invented(tmp_path):
    with pytest.raises(PL1.L1PersistError, match="not a directory"):
        PL1.write_l1_records([_rec()], release="r", source="s", origin="t",
                             root=str(tmp_path / "absent"),
                             log=lambda *a, **k: None)


def test_a_crashed_run_leaves_no_half_file_a_later_index_would_read(tmp_path):
    PL1.write_l1_records([_rec()], release="r", source="s", origin="t",
                         root=str(tmp_path), log=lambda *a, **k: None)
    assert list(pathlib.Path(tmp_path).rglob("*.tmp")) == []


# -- 2. the overlay is evidence, not an assertion --------------------------

def test_overlay_refuses_a_path_that_does_not_exist(tmp_path):
    with pytest.raises(PI.PersistedInputsError, match="does not exist"):
        PI.overlay_from_paths({NEWS_URL: str(tmp_path / "nope.jsonl.gz")},
                              root=str(tmp_path))


def test_overlay_refuses_a_path_outside_every_registered_store(tmp_path):
    """Filing text somewhere no registry can see it is the defect the guard
    exists for -- an overlay must not be the way back in."""
    stray = tmp_path / "scratch" / "news_documents.jsonl"
    stray.parent.mkdir(parents=True)
    stray.write_text("{}\n")
    with pytest.raises(PI.PersistedInputsError, match="not inside any store"):
        PI.overlay_from_paths({NEWS_URL: str(stray)}, root=str(tmp_path))


def test_overlay_accepts_what_the_writer_actually_wrote(tmp_path):
    written = PL1.write_l1_records([_rec()], release="freshness", origin="test",
                                   source="2026-09-12", root=str(tmp_path),
                                   log=lambda *a, **k: None)
    ov = PI.overlay_from_paths(written, root=str(tmp_path))
    hit = ov[AC.canonical_url(NEWS_URL)]
    assert hit.layer == "l1" and hit.store_id == "l1-release-gz"


# -- 3. persist-then-check: the cron's own sequence ------------------------

def test_a_lane_that_files_first_passes_the_guard_in_the_same_run(tmp_path):
    """The index is a built artifact and cannot know about a page written
    seconds ago. Filing first + the overlay closes that gap without a bypass."""
    # an index built BEFORE the lane ran: it knows nothing
    empty = tmp_path / "corpus"
    (empty / "l1" / "release=old" / "source=none").mkdir(parents=True)
    with gzip.open(empty / "l1" / "release=old" / "source=none"
                   / "l1-markdown-00000.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps({"url": "https://other.test/x",
                            "markdown": "y"}) + "\n")
    idx_dir = tmp_path / "idx"
    PI.build_index(str(idx_dir), root=str(empty), log=lambda *a, **k: None)
    idx = PI.PersistedIndex.open(str(idx_dir))
    try:
        rows = [{"url": NEWS_URL}]
        with pytest.raises(PI.UnpersistedInputs):
            PI.assert_pages_persisted(rows, index=idx, label="before")

        written = PL1.write_l1_records([_rec()], release="freshness", origin="test",
                                       source="2026-09-12", root=str(empty),
                                       log=lambda *a, **k: None)
        v = PI.assert_pages_persisted(
            rows, index=idx, label="after",
            overlay=PI.overlay_from_paths(written, root=str(empty)))
        assert v["verdict"] == "PASS"
        assert v["persisted_in_this_run"] == 1

        # and it is not a one-run trick: the next index build sees it for real
        idx2_dir = tmp_path / "idx2"
        PI.build_index(str(idx2_dir), root=str(empty), log=lambda *a, **k: None)
        idx2 = PI.PersistedIndex.open(str(idx2_dir))
        try:
            assert PI.assert_pages_persisted(
                rows, index=idx2, label="rebuilt")["persisted"] == 1
        finally:
            idx2.close()
    finally:
        idx.close()


def test_the_overlay_does_not_rescue_a_page_the_lane_did_not_file(tmp_path):
    empty = tmp_path / "corpus"
    (empty / "l1" / "release=old" / "source=none").mkdir(parents=True)
    with gzip.open(empty / "l1" / "release=old" / "source=none"
                   / "l1-markdown-00000.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps({"url": "https://other.test/x",
                            "markdown": "y"}) + "\n")
    idx_dir = tmp_path / "idx"
    PI.build_index(str(idx_dir), root=str(empty), log=lambda *a, **k: None)
    idx = PI.PersistedIndex.open(str(idx_dir))
    try:
        written = PL1.write_l1_records([_rec()], release="freshness", origin="test",
                                       source="2026-09-12", root=str(empty),
                                       log=lambda *a, **k: None)
        with pytest.raises(PI.UnpersistedInputs) as exc:
            PI.assert_pages_persisted(
                [{"url": NEWS_URL}, {"url": "https://unfiled.test/a"}],
                index=idx, label="mixed",
                overlay=PI.overlay_from_paths(written, root=str(empty)))
        assert exc.value.verdict["missing"] == 1
        assert "https://unfiled.test/a" in str(exc.value)
    finally:
        idx.close()


# -- 4. the CLI files an existing news stage ------------------------------

def test_cli_files_a_news_documents_jsonl(tmp_path, monkeypatch, capsys):
    stage = tmp_path / "stage-2026-09-12"
    stage.mkdir()
    doc = stage / "news_documents.jsonl"
    doc.write_text("\n".join(json.dumps({
        "schema": "aw.news-document.v1", "url": NEWS_URL + str(i),
        "markdown": NEWS_MD, "headline": "h"}) for i in range(3)) + "\n")
    rc = PL1.main(["--from-jsonl", str(doc), "--release", "news-daily",
                   "--source", "2026-09-12", "--corpus-root", str(tmp_path),
                   "--origin", "news-stage:news_documents.jsonl"])
    assert rc == 0
    assert '"pages": 3' in capsys.readouterr().out
    files = list((tmp_path / "l1" / "release=news-daily"
                  / "source=2026-09-12").glob("l1-markdown-*.jsonl.gz"))
    assert len(files) == 1
    assert sum(1 for _ in gzip.open(files[0], "rt", encoding="utf-8")) == 3


def test_cli_refuses_a_textless_record_with_exit_2(tmp_path, capsys):
    doc = tmp_path / "bad.jsonl"
    doc.write_text(json.dumps({"url": NEWS_URL}) + "\n")
    assert PL1.main(["--from-jsonl", str(doc), "--release", "r",
                     "--source", "s", "--corpus-root", str(tmp_path),
                     "--origin", "test"]) == 2
