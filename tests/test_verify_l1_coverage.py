#!/usr/bin/env python3
"""Tests for cytir/corpus/verify_l1_coverage.py (task 664, 2026-09-26).

Real incident this guards against: CC tranche4's L1 registration ran
cleanly (identical manifest shape to tranche2/3, which worked), but the
shared persisted-input index never picked the new files up --
`persisted_index_is_stale()` reported "fresh" regardless. A drain hours
later refused 84,579 of 84,582 pages with `UnpersistedInputs` (ADR-105).
`verify_l1_coverage.verify()` runs the SAME real check right after
registration, so the gap is a same-cycle refusal, not a silent one.
"""
from __future__ import annotations

import json
from pathlib import Path

from cytir.corpus import verify_l1_coverage as V

# verify_l1_coverage.py imports `persisted_inputs` bare (sys.path-prepended
# pipeline/, same convention as cytir.index.compose's own PSP/PQ imports) --
# `V.PI` is THAT module object. A test-level `import pipeline.
# persisted_inputs as PI` would load a SECOND, distinct module object under
# Python's package-qualified name, whose `UnpersistedInputs` class is a
# different object from `V.PI.UnpersistedInputs` despite the identical
# name -- `except V.PI.UnpersistedInputs` would silently not catch it. Use
# `V.PI` everywhere here, never a separately-imported copy.
PI = V.PI


def _write_passages(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_sample_pages_dedupes_by_url_and_carries_the_digest(tmp_path):
    p = tmp_path / "passages_ok.jsonl"
    _write_passages(p, [
        {"url": "https://a.example/1", "content_digest": "sha-a"},
        {"url": "https://a.example/1", "content_digest": "sha-a-dup-line"},
        {"url": "https://b.example/2", "digest": "sha-b"},
    ])
    pages = V.sample_pages(str(p), sample=100)
    by_url = {pg["url"]: pg["sha"] for pg in pages}
    assert by_url == {"https://a.example/1": "sha-a",
                      "https://b.example/2": "sha-b"}


def test_sample_pages_caps_at_the_requested_size(tmp_path):
    p = tmp_path / "passages_ok.jsonl"
    _write_passages(p, [{"url": f"https://x.example/{i}", "content_digest": "d"}
                        for i in range(500)])
    pages = V.sample_pages(str(p), sample=50)
    assert len(pages) == 50
    assert len({pg["url"] for pg in pages}) == 50  # no duplicates in the sample


def test_sample_pages_reads_markdown_sha256_task_669(tmp_path):
    """ERIC's consolidated markdown shard (`finalize_eric_pdf_
    materialization_workers.py`'s `documents.jsonl.gz`) carries
    `markdown_sha256`, not `digest`/`content_digest` -- the same fallback
    `persisted_inputs._normalize` already accepts (task 669/C)."""
    p = tmp_path / "documents.jsonl"
    _write_passages(p, [
        {"url": "https://files.eric.ed.gov/fulltext/ED036150.pdf",
         "markdown_sha256": "sha-eric-1"},
    ])
    pages = V.sample_pages(str(p), sample=100)
    assert pages == [{"url": "https://files.eric.ed.gov/fulltext/ED036150.pdf",
                      "sha": "sha-eric-1"}]


def test_sample_pages_reads_a_gzipped_shard_task_669(tmp_path):
    import gzip
    p = tmp_path / "documents.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"url": "https://files.eric.ed.gov/fulltext/ED1.pdf",
                             "markdown_sha256": "sha-1"}) + "\n")
        fh.write(json.dumps({"url": "https://files.eric.ed.gov/fulltext/ED2.pdf",
                             "markdown_sha256": "sha-2"}) + "\n")
    pages = V.sample_pages(str(p), sample=100)
    assert {pg["url"]: pg["sha"] for pg in pages} == {
        "https://files.eric.ed.gov/fulltext/ED1.pdf": "sha-1",
        "https://files.eric.ed.gov/fulltext/ED2.pdf": "sha-2"}


def test_sample_pages_skips_rows_with_no_url(tmp_path):
    p = tmp_path / "passages_ok.jsonl"
    _write_passages(p, [{"content_digest": "d"}, {"url": "", "content_digest": "d"},
                        {"url": "https://ok.example/1", "content_digest": "d"}])
    pages = V.sample_pages(str(p), sample=100)
    assert [pg["url"] for pg in pages] == ["https://ok.example/1"]


def test_verify_passes_when_assert_pages_persisted_passes(tmp_path, monkeypatch):
    p = tmp_path / "passages_ok.jsonl"
    _write_passages(p, [{"url": "https://ok.example/1", "content_digest": "d"}])
    monkeypatch.setattr(V.PI, "assert_pages_persisted",
                        lambda pages, **kw: {"verdict": "PASS", "pages": len(pages)})

    result = V.verify(str(p), refresh_index=False)
    assert result == {"status": "PASS", "reason": None, "sampled": 1}


def test_verify_refuses_and_carries_the_real_reason(tmp_path, monkeypatch):
    """The real incident's exact shape: registration ran, but the index
    check finds the pages missing. The caller (a shell watchdog) needs the
    real UnpersistedInputs message, not a generic failure."""
    p = tmp_path / "passages_ok.jsonl"
    _write_passages(p, [{"url": "https://missing.example/1", "content_digest": "d"}])

    def fake_assert(pages, **kw):
        raise PI.UnpersistedInputs(
            "[persisted-inputs] REFUSED tranche4: 1 of 1 page(s) have no "
            "persisted L1 record", {"verdict": "REFUSED"})
    monkeypatch.setattr(V.PI, "assert_pages_persisted", fake_assert)

    result = V.verify(str(p), refresh_index=False)
    assert result["status"] == "REFUSED"
    assert "no persisted L1 record" in result["reason"]
    assert result["sampled"] == 1


def test_verify_refuses_cleanly_on_an_empty_passages_file(tmp_path):
    p = tmp_path / "passages_ok.jsonl"
    _write_passages(p, [])
    result = V.verify(str(p), refresh_index=False)
    assert result["status"] == "REFUSED"
    assert "no urls to sample" in result["reason"]


def test_verify_refreshes_the_index_first_by_default(tmp_path, monkeypatch):
    p = tmp_path / "passages_ok.jsonl"
    _write_passages(p, [{"url": "https://ok.example/1", "content_digest": "d"}])
    monkeypatch.setattr(V.PI, "assert_pages_persisted",
                        lambda pages, **kw: {"verdict": "PASS", "pages": len(pages)})

    calls = []
    from cytir.index import promote_queue as PQ
    monkeypatch.setattr(PQ, "refresh_persisted_index",
                        lambda **kw: calls.append(kw) or
                        {"rebuilt": False, "reason": "fresh"})
    monkeypatch.setattr(V, "sys", V.sys)  # noop, keeps import path stable
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "promote_queue", PQ)

    V.verify(str(p), refresh_index=True)
    assert calls, "refresh_persisted_index must be called when refresh_index=True"


def test_cli_exit_code_reflects_the_verdict(tmp_path, monkeypatch, capsys):
    p = tmp_path / "passages_ok.jsonl"
    _write_passages(p, [{"url": "https://ok.example/1", "content_digest": "d"}])
    monkeypatch.setattr(V.PI, "assert_pages_persisted",
                        lambda pages, **kw: {"verdict": "PASS", "pages": len(pages)})

    rc = V.main(["--passages", str(p), "--no-refresh-index"])
    assert rc == 0
    assert "PASS" in capsys.readouterr().out


def test_cli_exit_code_is_1_on_refusal(tmp_path, monkeypatch, capsys):
    p = tmp_path / "passages_ok.jsonl"
    _write_passages(p, [{"url": "https://missing.example/1", "content_digest": "d"}])

    def fake_assert(pages, **kw):
        raise PI.UnpersistedInputs("REFUSED for the test", {"verdict": "REFUSED"})
    monkeypatch.setattr(V.PI, "assert_pages_persisted", fake_assert)

    rc = V.main(["--passages", str(p), "--no-refresh-index"])
    assert rc == 1
    assert "REFUSED for the test" in capsys.readouterr().out


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
