#!/usr/bin/env python3
"""RULING 43 (task 984): the drain's tombstone word-coverage check compares CONTENT words, not
snippet words, when the chunk policy changed and the composer declared them.

The numbers are the real refusal of the opencode docs re-cut (legacy 180-word chunks, epoch 1,
to the generic 768-word policy, epoch 4): `/docs/cli` was 14 old rows = 314 snippet words and
3 new rows = 60 snippet words, ratio 0.19, although the 3 new chunks hold the text.
"""
from __future__ import annotations

import pathlib
import sys


REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cytir.chunk import registry as CHREG  # noqa: E402
from cytir.index import promote_queue as Q  # noqa: E402
from cytir.news import compose as C  # noqa: E402

URL = "https://opencode.ai/docs/cli"
OLD_IDS = [f"old{i}" for i in range(14)]
TOMBS = [{"id": i, "url": URL, "reason": "superseded"} for i in OLD_IDS]
# 14 old rows, 314 snippet words in all (22 or 23 words each)
OLD_SNIPPETS = [" ".join(["w"] * (23 if i < 6 else 22)) for i in range(14)]
assert sum(len(s.split()) for s in OLD_SNIPPETS) == 314


def _old_payloads(epoch=1):
    return {i: {"snippet": s, "url": URL, **({"rule_epochs": {"chunk": epoch}} if epoch else {})}
            for i, s in zip(OLD_IDS, OLD_SNIPPETS)}


def _new_points(epoch=4, n=3, n_words=600):
    return [{"payload": {"url": URL, "snippet": " ".join(["n"] * 20), "chunk_n_words": n_words,
                         "rule_epochs": {"chunk": epoch}}} for _ in range(n)]


def _run(monkeypatch, points, *, declared, old=None):
    monkeypatch.setattr(Q.PD, "fetch_payload_fields",
                        lambda *a, **k: old if old is not None else _old_payloads())
    return Q.validate_tombstone_word_coverage(
        points, TOMBS, target_url="http://x", collection="c", content_words_by_url=declared)


def test_widths_come_from_the_policy_files():
    assert CHREG.epoch_word_width(1) == 180
    assert CHREG.epoch_word_width(4) == 768
    assert CHREG.epoch_word_width(6) is None      # two widths share the epoch: unknown, not guessed


def test_recut_passes_on_content_words_where_the_snippet_rule_gives_019(monkeypatch):
    rep = _run(monkeypatch, _new_points(), declared={URL: 1800})   # 1800 / (14 x 180) = 0.71
    assert rep["violations"] == []


def test_same_pack_without_the_declaration_is_refused_as_today(monkeypatch):
    rep = _run(monkeypatch, _new_points(), declared=None)
    [v] = rep["violations"]
    assert v["basis"] == "snippet" and "basis_fallback" not in v
    assert (v["kept_words"], v["old_words"], v["ratio"]) == (60, 314, 0.191)


def test_declared_pack_under_the_floor_is_still_refused(monkeypatch):
    rep = _run(monkeypatch, _new_points(), declared={URL: 1000})   # 1000 / 2520 = 0.397
    [v] = rep["violations"]
    assert v["basis"] == "content_words"
    assert (v["kept_words"], v["old_words"], v["ratio"]) == (1000, 2520, 0.397)


def test_no_epoch_change_ignores_the_declaration(monkeypatch):
    same = [{"payload": dict(p["payload"], rule_epochs={"chunk": 1})} for p in _new_points()]
    rep = _run(monkeypatch, same, declared={URL: 1800})
    [v] = rep["violations"]
    assert v["basis"] == "snippet" and v["basis_fallback"] == "chunk epoch is unchanged"


def test_unknown_old_epoch_falls_back_and_says_so(monkeypatch):
    rep = _run(monkeypatch, _new_points(), declared={URL: 1800}, old=_old_payloads(epoch=None))
    [v] = rep["violations"]
    assert v["basis"] == "snippet" and v["basis_fallback"] == "an old row has no rule_epochs.chunk"


def test_old_epoch_without_a_width_falls_back(monkeypatch):
    rep = _run(monkeypatch, _new_points(), declared={URL: 1800}, old=_old_payloads(epoch=6))
    [v] = rep["violations"]
    assert v["basis"] == "snippet" and "no known word width" in v["basis_fallback"]


def test_declared_key_is_normalised_like_the_check_keys_urls(monkeypatch):
    rep = _run(monkeypatch, _new_points(), declared={URL + "/": 1800})
    assert rep["violations"] == []


def test_composer_declares_chunk_words_only_for_superseded_urls_with_exact_counts():
    kept = _new_points() + [{"payload": {"url": "https://opencode.ai/docs/other", "chunk_n_words": 9}}]
    assert C.declared_content_words(kept, TOMBS) == {Q.document_url_key(URL): 1800}


def test_composer_leaves_out_a_url_with_a_row_missing_its_count():
    kept = _new_points()
    del kept[1]["payload"]["chunk_n_words"]
    assert C.declared_content_words(kept, TOMBS) == {}


def test_write_pack_puts_the_declaration_in_the_manifest(tmp_path):
    man = C.write_pack(str(tmp_path / "p"), kept=_new_points(), tombstones=TOMBS,
                       pack_id="p-1", family="docs", collection="c", days=["2026-10-19"])
    assert man["content_words_by_url"] == {Q.document_url_key(URL): 1800}
    bare = C.write_pack(str(tmp_path / "q"), kept=_new_points(), tombstones=[],
                        pack_id="p-2", family="docs", collection="c", days=["2026-10-19"])
    assert "content_words_by_url" not in bare
