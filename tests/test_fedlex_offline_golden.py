#!/usr/bin/env python3
"""Offline fedlex golden (task 677, follow-up of the S5 consent/source-run slice).

`cytir.sources.families.fedlex` fetches SPARQL and Akoma Ntoso XML live, so until now the only
fedlex reproduction check was a manual, networked run. This test replays SEALED HTTP
replies for one small federal act (`fixtures/fedlex_offline/cc-1948-195.tape.json`,
recorded live from fedlex.data.admin.ch on 2026-09-30: the consolidation-date SPARQL
reply, the de/fr/it title/short/SR metadata replies and the three XML files) through the
REAL `fedlex_dump.main()`: same resolver, same Akoma Ntoso parser, same windowing, same
title composition, same identity minting. It runs in CI with no network, and any request the
tape does not hold fails the test.

The pinned hash is over the exact output bytes with `last_seen` pinned to the recording date.

WHY `last_seen` IS PINNED (a correction to the 2026-09-30 note in ADDING-A-SOURCE.md)
-------------------------------------------------------------------------------------
Every fedlex row carries `last_seen` = the day the dump ran, so the bytes of ANY run, and so
the sha256 of the whole file, change every day even when nothing else does. The earlier
manual golden `98c4def6...` (recorded 2026-08-20) was read as "no longer reproduces from
either code path". It does reproduce: the same 732-passage BV output with `last_seen`
rewritten to 2026-08-20 hashes to exactly `98c4def6...`. What changed between the two hashes
is the date, not the extractor. `test_the_hash_is_date_dependent_and_the_content_is_not` proves
the mechanism on this fixture.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

httpx = pytest.importorskip("httpx")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
FIX = REPO / "tests" / "fixtures" / "fedlex_offline"
for p in (str(REPO),):
    if p not in sys.path:
        sys.path.insert(0, p)


def _tape() -> dict:
    return json.loads((FIX / "cc-1948-195.tape.json").read_text(encoding="utf-8"))


def _expected() -> dict:
    return json.loads((FIX / "cc-1948-195.expected.json").read_text(encoding="utf-8"))


def _run_offline(monkeypatch, tmp_path: pathlib.Path, *, last_seen: str) -> tuple[bytes, list[str]]:
    """Run the real fedlex_dump.main() against the sealed tape. Returns (output bytes,
    requests the tape did not hold)."""
    import cytir.sources.families.fedlex as fd

    tape = _tape()
    replies = {(t["method"], t["url"], t["form"]): t for t in tape["tape"]}
    unexpected: list[str] = []

    def handler(request: "httpx.Request") -> "httpx.Response":
        form = request.content.decode() if request.method == "POST" else None
        hit = replies.get((request.method, str(request.url), form))
        if hit is None:
            unexpected.append(f"{request.method} {request.url} {form!r}"[:200])
            return httpx.Response(599, text="not on the tape")
        headers = {"content-type": hit["content_type"]} if hit["content_type"] else {}
        return httpx.Response(hit["status"], content=hit["body"].encode("utf-8"), headers=headers)

    monkeypatch.setattr(fd, "H", httpx.Client(transport=httpx.MockTransport(handler),
                                              follow_redirects=True))
    monkeypatch.setattr(fd, "TODAY", last_seen)
    monkeypatch.setitem(fd.COMMON, "last_seen", last_seen)
    monkeypatch.setattr("time.sleep", lambda _s: None)        # the politeness floor, not a test of it
    out = tmp_path / "fedlex.jsonl"
    monkeypatch.setattr(sys, "argv", ["fedlex_dump.py", "--eli", tape["eli"], "--langs", *tape["langs"],
                                      "--out", str(out)])
    fd.main()
    return out.read_bytes(), unexpected


def test_offline_fedlex_dump_reproduces_the_sealed_golden(monkeypatch, tmp_path):
    want = _expected()
    data, unexpected = _run_offline(monkeypatch, tmp_path, last_seen=want["last_seen_pinned"])
    assert not unexpected, f"the run asked for something the tape does not hold: {unexpected}"
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    per_lang: dict[str, int] = {}
    for row in rows:
        per_lang[row["lang"]] = per_lang.get(row["lang"], 0) + 1
    assert len(rows) == want["rows"]
    assert per_lang == want["rows_per_lang"]
    assert rows[0]["title"] == want["first_title"]
    assert hashlib.sha256(data).hexdigest() == want["out_sha256"]


def test_the_hash_is_date_dependent_and_the_content_is_not(monkeypatch, tmp_path):
    """The same sealed inputs run on a different day: every byte that is not `last_seen`
    is identical, and the whole-file hash is not."""
    want = _expected()
    a, _ = _run_offline(monkeypatch, tmp_path, last_seen=want["last_seen_pinned"])
    other = tmp_path / "other"
    other.mkdir()
    b, _ = _run_offline(monkeypatch, other, last_seen="2026-08-20")
    assert hashlib.sha256(a).hexdigest() != hashlib.sha256(b).hexdigest()
    norm = lambda raw, day: raw.replace(f'"last_seen": "{day}"'.encode(), b'"last_seen": "X"')  # noqa: E731
    assert norm(a, want["last_seen_pinned"]) == norm(b, "2026-08-20")
    assert hashlib.sha256(norm(a, want["last_seen_pinned"])).hexdigest() == want["out_sha256_last_seen_X"]


def test_the_tape_is_what_it_claims_to_be():
    """Seven replies: the date query, three metadata queries, three XML files; all 200."""
    tape = _tape()["tape"]
    assert len(tape) == 7
    assert all(t["status"] == 200 for t in tape)
    assert sum(1 for t in tape if t["method"] == "POST") == 4
    assert sum(1 for t in tape if t["url"].endswith("-xml.xml")) == 3
