"""`cyt corpus facet-compact` and `cyt corpus card-content-dry-run` (task 1836)."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("pyarrow")

from cytir.corpus import facet_bundle as facets

APP = pathlib.Path(__file__).resolve().parents[1]


def _aw(*args, check=True):
    return subprocess.run([sys.executable, "-m", "cytir.cli", "corpus", *map(str, args)],
                          cwd=APP, check=check, text=True, capture_output=True)


def _event(value, source="homepage"):
    return facets.make_event(
        entity="CHE341859062", facet="prose", value=value, lang="en", source=source,
        evidence=value, source_url="https://source.example/", source_id=source,
        effective_at="2026-08-01", retrieved_at="2026-08-01",
        extractor_version="initial-v1", license="CC-BY-4.0", consent="search_ok",
        attribution=source, task="initial", selection=None)


@pytest.mark.parametrize("verb", ["facet-compact", "card-content-dry-run"])
def test_verb_is_routed_and_has_help(verb):
    out = _aw(verb, "--help").stdout
    assert f"cyt corpus {verb}" in out


def test_corpus_group_rejects_unknown_verb():
    assert _aw("nope", check=False).returncode == 2


def test_facet_compact_writes_current_state(tmp_path):
    root = tmp_path / "facet_events"
    ev = _event("A statement.")
    facets.publish_event_wave([ev], root / "000-initial.jsonl", task="initial")
    out = tmp_path / "current.parquet"
    report = json.loads(_aw("facet-compact", "--events-root", root, "--out", out).stdout)
    assert report["events"] == 1 and report["current"] == 1
    assert {r["event_id"] for r in facets.read_parquet(out)} == {ev["event_id"]}


def test_facet_compact_needs_a_source(tmp_path):
    assert _aw("facet-compact", "--out", tmp_path / "x.parquet", check=False).returncode == 2


def test_dry_run_refuses_to_overwrite(tmp_path):
    existing = tmp_path / "cards.jsonl"
    existing.write_text("", encoding="utf-8")
    src = tmp_path / "in.jsonl"
    src.write_text("", encoding="utf-8")
    r = _aw("card-content-dry-run", "--in", src, "--out", existing,
            "--events-out", tmp_path / "ev.jsonl", "--bootstrap-empty-current", check=False)
    assert r.returncode != 0 and "refusing to overwrite" in r.stderr


def test_dry_run_current_parquet_requires_events_root(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text("", encoding="utf-8")
    r = _aw("card-content-dry-run", "--in", src, "--out", tmp_path / "c.jsonl",
            "--events-out", tmp_path / "ev.jsonl",
            "--current-parquet", tmp_path / "cur.parquet", check=False)
    assert r.returncode != 0 and "requires its canonical --events-root" in r.stderr
