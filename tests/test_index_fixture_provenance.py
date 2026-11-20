#!/usr/bin/env python3
"""The sealed fixture stages of `cyt index build` / `cyt gate` (data/index-fixture/) are what the
stage door (`cytir.sources.stage_out.run_to_stage`) makes from the repo's own fixtures (slice 8 step 5,
specs/aw-index-serve-gate.md section 4; task 2561 remade the last two by hand stages through the door).

    python3 tests/test_index_fixture_provenance.py --write     # regenerate data/index-fixture/

The test below fails when a committed stage differs from what the door makes now (a changed
extractor must move the fixture on purpose). The stages ship in the public cut, and
`tests/test_aw_index_serve_gate.py` uses them.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import shutil
import sys
import tempfile

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent

FIXTURE_ROOT = REPO / "data" / "index-fixture"
OK = {"bucket": "ok", "status": "ok", "search_ok": True, "train_ok": True,
      "checked_at": "2026-09-30T00:00:00Z"}
#: family -> (pack id the door stamps, the day the family run is pinned to)
LODA_PACK_ID = "loda-exact-v1-local-fb45826c249c"
FEDLEX_PACK_ID = "fedlex-exact-v1-local-779c076b42c1"
FAMILIES = {
    "loda_exact": (LODA_PACK_ID, "2026-10-03"),
    "fedlex_exact": (FEDLEX_PACK_ID, "2026-09-30"),
}


def _no_consent(host, direct):          # a dump route contacts no host: asking is a failure
    raise AssertionError(f"a dump-route family asked for the consent of {host}")


def _stage_loda(dest: pathlib.Path, today: str) -> None:
    from unittest import mock
    from cytir.sources import stage_out
    from cytir.sources.families import loda
    # the sample articles are short and never ask for a token count; the family refuses a run
    # without the embed tokenizer (task 706), which CI does not have
    with mock.patch.object(loda, "embed_token_counter", lambda: (lambda text: len(text) // 3)):
        stage_out.run_to_stage(
            "loda_exact", out=dest, today=today, from_dir=REPO / "data" / "samples" / "legi-loda",
            options={"release": "aw-fixture-loda-v1"}, consent=_no_consent, log=lambda m: None)


def _stage_fedlex(dest: pathlib.Path, today: str) -> None:
    from cytir.acquire.tape import Tape
    from cytir.sources import stage_out
    tape = json.loads((HERE / "fixtures" / "fedlex_offline" / "cc-1948-195.tape.json")
                      .read_text(encoding="utf-8"))
    t = Tape.load(tape)
    t.unexpected = []
    import cytir.sources.families.fedlex as FX
    # the splitter guard (door windows vs the one law splitter) refuses real runs by design until
    # task 1038, so it is let through here only
    was = FX.DOOR_SPLITTER
    FX.DOOR_SPLITTER = FX.DECLARED_SPLITTER
    try:
        stage_out.run_to_stage(
            "fedlex_exact", out=dest, today=today, work_dir=dest.parent / (dest.name + "-work"),
            options={"eli": tape["eli"], "langs": tape["langs"]}, http=t.client(),
            consent=lambda h, d: {**OK, "host": h}, log=lambda m: None)
    finally:
        FX.DOOR_SPLITTER = was
    assert not t.unexpected, t.unexpected


STAGERS = {"loda_exact": _stage_loda, "fedlex_exact": _stage_fedlex}


def regenerate(root: pathlib.Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for family, (_pid, today) in FAMILIES.items():
            tmp_dest = pathlib.Path(tmp) / family
            STAGERS[family](tmp_dest, today)
            dest = root / family
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(tmp_dest, dest)


def _tree(root: pathlib.Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_the_committed_stages_are_what_the_family_doors_make_from_the_repo_fixtures(tmp_path):
    pytest.importorskip("httpx")
    regenerate(tmp_path)
    committed = {k: v for k, v in _tree(FIXTURE_ROOT).items() if k.split("/")[0] in FAMILIES}
    assert _tree(tmp_path) == committed, "data/index-fixture/ differs from the family doors' output: " \
        "regenerate with `python3 tests/test_index_fixture_provenance.py --write` and review the diff"


def test_every_stage_carries_the_doors_pack_json_shape_and_no_hand_marker():
    for family, (pack_id, _today) in FAMILIES.items():
        pj = json.loads((FIXTURE_ROOT / family / "PACK.json").read_text(encoding="utf-8"))
        assert "staged_by" not in pj, family
        assert pj["pack_id"] == pack_id and pj["family_id"] == family
        assert {"chunks_sha256", "source_records_sha256", "rows"} <= set(pj), (family, sorted(pj))


def test_every_stage_has_the_rows_the_slice_golds_point_into():
    rows = {}
    for family in FAMILIES:
        with gzip.open(FIXTURE_ROOT / family / "chunks" / "shard-0000.texts.jsonl.gz", "rt",
                       encoding="utf-8") as fh:
            rows[family] = [json.loads(x) for x in fh]
    assert {f: len(r) for f, r in rows.items()} == {"loda_exact": 3, "fedlex_exact": 63}


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit(__doc__)
    regenerate(FIXTURE_ROOT)
    print(f"wrote {FIXTURE_ROOT}")


def test_every_committed_stage_row_reproduces_its_id_under_the_scheme_its_family_declares():
    """Task 1041: the build refuses a stage whose declared `point_id` does not reproduce under the
    family's id scheme (`compose.verify_identity`, run on the stamped stage). Run the same check on the
    committed stage here, so the gate is not the first to notice (the "worse but present" control
    rewrote a fedlex row's text and kept its id, which a digest-addressed id forbids)."""
    from cytir.index import compose
    for family in FAMILIES:
        chunks = FIXTURE_ROOT / family / "chunks"
        res = compose.verify_identity(str(chunks), anchor_fn=compose.anchor_fn_for_family(family),
                                      scheme=compose.id_scheme_for_family(family))
        assert res["checked"] > 0 and res["matched"] == res["checked"], (family, res)
