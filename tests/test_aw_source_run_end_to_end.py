#!/usr/bin/env python3
"""`cyt source run <family>` ends at the sealed pack; `--embed` is RETIRED (slice 8 step 3).

One small real family (`loda_exact`, a dump route: no host, no network) over 8 real
articles in `data/samples/legi-loda`. Proves: acquire -> extract -> chunk -> pack still works;
the retired divergent build door (`--embed`, `cyt embed index`, the generic materializer)
refuses and names the one build verb, `cyt index build`, before anything is read or written.
Those doors built an index that differs from production's on four axes (point id, dense
text, BM25 weights, payload); the replacement is `cyt index build` (slice 8 step 2).
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest


@pytest.fixture(autouse=True)
def _loda_token_counter_stand_in(monkeypatch):
    """`loda_exact` refuses a run without the embed tokenizer (task 706 preflight). The sample
    articles here are all short and never ask for a count, so a stand-in is enough; CI has no
    `transformers`."""
    from cytir.sources.families import loda
    monkeypatch.setattr(loda, "embed_token_counter", lambda: (lambda text: len(text) // 3))

pytest.importorskip("httpx")
qc = pytest.importorskip("qdrant_client")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

FAMILY = "loda_exact"
TODAY = "2026-09-30"
SAMPLE = REPO / "data" / "samples" / "legi-loda"      # 8 real DILA LEGI articles (Etalab-2.0)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("AW_LANGGATE_OPTIONAL", "1")
    monkeypatch.setenv("AW_LID_MODEL", "/nonexistent/lid.176.bin")


def _compose(out, *, skip_consent=False):
    """A dump-route family contacts no host: `consent` failing the test proves it."""
    from cytir.sources import run_family
    return run_family.run_family(
        FAMILY, out_dir=out, today=TODAY, skip_consent=skip_consent,
        consent=lambda h, d: pytest.fail(f"consent for {h}"),
        options={"root": str(SAMPLE), "release": "aw-sample-loda-v1"}, log=lambda m: None)


def test_the_pack_still_composes_without_embed(tmp_path):
    manifest = _compose(tmp_path)
    assert manifest["records"]["count"] >= 1 and manifest["terminals"]["units_admitted"] >= 1
    assert (tmp_path / "pack" / "records.jsonl").is_file()
    assert not (tmp_path / "qdrant").exists()           # no embed step, no Qdrant


def test_source_run_embed_refuses_before_reading_or_writing(tmp_path, monkeypatch, capsys):
    from cytir.sources import run_family
    monkeypatch.setattr(run_family, "run_family",
                        lambda *a, **k: pytest.fail("the run started before the refusal"))
    out = tmp_path / "never"
    rc = run_family.main([FAMILY, "--out-dir", str(out), "--embed"])
    err = capsys.readouterr().err
    assert rc == 2 and "cyt index build" in err and "slice 8 step 2" not in err
    assert not out.exists()


def test_aw_embed_index_refuses_and_names_the_one_build_verb(tmp_path, monkeypatch):
    from cytir.embed import collection_stage as cs
    monkeypatch.setattr("cytir.core.qdrant.qdrant_client",
                        lambda *a, **k: pytest.fail("connected before the refusal"))
    recs = tmp_path / "r.jsonl"
    recs.write_text(json.dumps({"url": "https://e.org/a", "text": "t"}) + "\n")
    for extra in ([], ["--layout", "served"]):
        with pytest.raises(SystemExit, match="cyt index build"):
            cs.main(["--in", str(recs), "--collection", "c", *extra])


def test_the_generic_materializer_and_the_embed_bridge_are_gone():
    from cytir.embed import collection_stage as cs
    from cytir.sources import run_family
    assert not hasattr(cs, "generic_materializer")
    assert not hasattr(cs, "legacy_config_b_materializer")
    assert not hasattr(run_family, "embed_pack")
    assert not hasattr(run_family, "_embed_main")
    assert hasattr(run_family, "assert_embeddable")     # the shared consent precondition stays


def test_the_embed_flag_is_the_only_embed_option_left():
    from cytir.sources.run_family import build_parser
    a = build_parser().parse_args(["f", "--out-dir", "o", "--embed"])
    assert a.embed
    for gone in ("collection", "qdrant", "allow_remote", "layout", "dense_dim", "tei_url"):
        assert not hasattr(a, gone), gone


def test_nothing_names_the_retired_projection():
    """No shipped module and no pipeline script names the retired projection (removed, task 1378)."""
    import re
    bad = []
    for base in ("cytir", "pipeline", "server", "scripts"):
        for p in (REPO / base).rglob("*.py"):
            rel = p.relative_to(REPO).as_posix()
            if p.name.startswith("test_"):
                continue
            if re.search(r"\b(legacy_config_b_materializer|generic_materializer)\b",
                         p.read_text(encoding="utf-8", errors="replace")):
                bad.append(rel)
    assert not bad, bad
    assert "collection_stage" not in (REPO / "cytir/sources/run_family.py").read_text()
