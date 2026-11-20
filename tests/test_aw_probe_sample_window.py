"""Slice A sampler: a seen query becomes eligible again after the re-eligibility window.

Fake ledger and fake run history only: no network, no real probes/ files.
"""
import json
import os
import random

from cytir.probe import sample as S


def _run(runs_dir, day, queries):
    d = os.path.join(runs_dir, day)
    os.makedirs(d)
    with open(os.path.join(d, "sample.jsonl"), "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps({"query": q}) + "\n")


def test_old_probe_is_eligible_recent_and_undated_stay_blocked(tmp_path):
    runs = str(tmp_path)
    _run(runs, "2026-08-01", ["Old Query"])
    _run(runs, "2026-10-20", ["Recent  query"])
    seen = {"old query", "recent query", "never dated"}
    blocked = S.blocking_seen(seen, "2026-10-31", 30, runs_dir=runs)
    assert blocked == {"recent query", "never dated"}


def test_window_zero_keeps_never_repeat(tmp_path):
    runs = str(tmp_path)
    _run(runs, "2026-01-01", ["a"])
    assert S.blocking_seen({"a"}, "2026-10-31", 0, runs_dir=runs) == {"a"}


def test_last_probe_wins_and_today_is_ignored(tmp_path):
    runs = str(tmp_path)
    _run(runs, "2026-08-01", ["a"])
    _run(runs, "2026-10-25", ["a"])
    _run(runs, "2026-10-31", ["b"])
    assert S.last_probed(runs, before="2026-10-31") == {"a": "2026-10-25"}


def test_exhausted_pool_draws_again_through_window(tmp_path):
    runs = str(tmp_path)
    qs = [f"query {i}" for i in range(8)]
    _run(runs, "2026-08-01", qs)
    ds = [{"name": "fake", "rows": [
        {"query": q, "lang": "en", "source_id": i, "stratum": "x"} for i, q in enumerate(qs)]}]
    seen = {S.norm(q) for q in qs}
    assert S.sample_verbatim(ds, 5, random.Random(1), seen) == []
    ds2 = [{"name": "fake", "rows": [dict(r) for r in ds[0]["rows"]]}]
    eligible = S.blocking_seen(seen, "2026-10-31", 30, runs_dir=runs)
    assert len(S.sample_verbatim(ds2, 5, random.Random(1), eligible)) == 5


def test_append_seen_never_duplicates(tmp_path, monkeypatch):
    path = tmp_path / "seen.txt"
    path.write_text("old query\n", encoding="utf-8")
    monkeypatch.setattr(S, "SEEN", str(path))
    S.append_seen(["old query", "new query"])
    assert path.read_text(encoding="utf-8").splitlines() == ["old query", "new query"]
