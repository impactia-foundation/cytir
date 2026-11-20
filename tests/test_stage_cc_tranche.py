#!/usr/bin/env python3
"""Task 664, 2026-09-26: `stage_cc_tranche.py` promoted from a per-tranche
`tasks/` throwaway script into a real pipeline module (it now recurs, via
`cc_tranche_seal_watchdog.sh`, for every new CC tranche)."""
from __future__ import annotations

import gzip
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cytir.corpus import stage_cc_tranche as S  # noqa: E402


def _write_gated(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_shard(path: pathlib.Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _row(pid: str, url: str, domain: str = "example.com", text: str = "hello world") -> dict:
    return {"point_id": pid, "url": url, "domain": domain, "text": text,
           "title": "T", "lang": "en"}


def test_stages_a_clean_pack_into_single_all_shard(tmp_path):
    inp = tmp_path / "gated" / "passages_ok.jsonl"
    out = tmp_path / "t9_gpu"
    rows = [_row(f"id-{i}", f"https://example.com/{i}") for i in range(3)]
    _write_gated(inp, rows)

    rep = S.stage_tranche(str(inp), str(out), packs=1, shard_size=20_000)

    assert rep["input_rows"] == 3
    assert rep["staged"] == 3
    assert rep["packs"]["all"]["rows"] == 3
    shard = out / "all" / "chunks" / "shard-0000.texts.jsonl.gz"
    assert shard.is_file()
    staged_rows = _read_shard(shard)
    assert {r["point_id"] for r in staged_rows} == {"id-0", "id-1", "id-2"}
    # outlinks stripped, everything else kept
    assert "outlinks" not in staged_rows[0]
    assert staged_rows[0]["url"] == "https://example.com/0"

    stage_shard = out / "all" / "stage" / "shard-0000.texts.jsonl.gz"
    staged_dense = _read_shard(stage_shard)
    assert set(staged_dense[0]) == {"point_id", "text"}


def test_drops_empty_text_missing_point_id_and_duplicate_point_id_and_counts_them(tmp_path):
    inp = tmp_path / "gated" / "passages_ok.jsonl"
    out = tmp_path / "t9_gpu"
    rows = [
        _row("id-0", "https://example.com/0"),
        {**_row("id-1", "https://example.com/1"), "text": "   "},  # empty text
        {**_row("id-2", "https://example.com/2"), "point_id": None},  # missing id
        _row("id-0", "https://example.com/0-dup"),  # duplicate point_id (first wins)
    ]
    _write_gated(inp, rows)

    rep = S.stage_tranche(str(inp), str(out), packs=1)

    assert rep["input_rows"] == 4
    assert rep["staged"] == 1
    assert rep["empty_text"] == 1
    assert rep["missing_point_id"] == 1
    assert rep["duplicate_point_id"] == 1
    staged_rows = _read_shard(out / "all" / "chunks" / "shard-0000.texts.jsonl.gz")
    assert len(staged_rows) == 1
    assert staged_rows[0]["url"] == "https://example.com/0"  # the FIRST one, not the dup


def test_a_url_never_spans_two_packs(tmp_path):
    """The whole point of partitioning by domain (not row): every row for
    ONE domain must land in exactly one pack, however many packs there
    are -- a url split across packs would make promote_sealed_pack.py
    refuse the second pack (it already serves that url)."""
    inp = tmp_path / "gated" / "passages_ok.jsonl"
    out = tmp_path / "t9_gpu"
    rows = []
    for i in range(40):
        dom = f"site{i % 5}.example"  # 5 distinct domains, 8 rows each
        rows.append(_row(f"id-{i}", f"https://{dom}/page{i}", domain=dom))
    _write_gated(inp, rows)

    rep = S.stage_tranche(str(inp), str(out), packs=4, shard_size=20_000)

    assert rep["staged"] == 40
    assert set(rep["packs"]) == {"a", "b", "c", "d"}
    assert sum(p["rows"] for p in rep["packs"].values()) == 40
    domain_of = {}
    for pack_name in rep["packs"]:
        shard = out / pack_name / "chunks" / "shard-0000.texts.jsonl.gz"
        if not shard.is_file():
            continue
        for row in _read_shard(shard):
            dom = row["domain"]
            assert domain_of.setdefault(dom, pack_name) == pack_name, (
                f"{dom} appeared in more than one pack")


def test_shard_size_rotates_into_multiple_shards(tmp_path):
    inp = tmp_path / "gated" / "passages_ok.jsonl"
    out = tmp_path / "t9_gpu"
    rows = [_row(f"id-{i}", f"https://example.com/{i}") for i in range(5)]
    _write_gated(inp, rows)

    rep = S.stage_tranche(str(inp), str(out), packs=1, shard_size=2)

    assert rep["packs"]["all"]["shards"] == 3  # 2 + 2 + 1
    shards = sorted((out / "all" / "chunks").glob("shard-*.texts.jsonl.gz"))
    assert len(shards) == 3
    total = sum(len(_read_shard(s)) for s in shards)
    assert total == 5


def test_cli_writes_stage_report_json(tmp_path):
    inp = tmp_path / "gated" / "passages_ok.jsonl"
    out = tmp_path / "t9_gpu"
    _write_gated(inp, [_row("id-0", "https://example.com/0")])

    rc = S.main(["--in", str(inp), "--out", str(out)])

    assert rc == 0
    report = json.loads((out / "stage_report.json").read_text())
    assert report["staged"] == 1
