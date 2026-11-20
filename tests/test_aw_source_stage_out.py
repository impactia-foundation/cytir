"""`cyt source run FAMILY --from DIR --out STAGE`: a family door that ends where `cyt index build`
starts (slice 7 step 1, task 696). No network, no Qdrant, no model.

Pins: the committed stage `data/index-fixture/dila_kali/` is byte for byte what the door makes from
the committed sample `data/samples/dila-kali/`; the run is deterministic and the stage names no
checkout or host; `--from` is refused for a family that needs a host and no consent is ever asked;
a stage is never overwritten; a pack whose declaration holds an `unknown` is refused (the reason the
sealed `loda_exact` and `fedlex_exact` stages are hand-made); a pack altered after it was sealed is
refused; the CLI refuses a `--from` without `--out`.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import pytest


@pytest.fixture(autouse=True)
def _loda_token_counter_stand_in(monkeypatch):
    """`loda_exact` refuses a run without the embed tokenizer (task 706 preflight). The sample
    articles here are all short and never ask for a count, so a stand-in is enough; CI has no
    `transformers`."""
    from cytir.sources.families import loda
    monkeypatch.setattr(loda, "embed_token_counter", lambda: (lambda text: len(text) // 3))

REPO = pathlib.Path(__file__).resolve().parent.parent
SAMPLE = REPO / "data" / "samples" / "dila-kali"
COMMITTED = REPO / "data" / "index-fixture" / "dila_kali"
TODAY = "2026-10-03"


def _tree(root: pathlib.Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _make(out: pathlib.Path, **kw):
    from cytir.sources import stage_out
    return stage_out.run_to_stage("dila_kali", out=out, today=TODAY, from_dir=SAMPLE,
                                  log=lambda m: None, **kw)


def test_the_committed_stage_is_what_the_family_door_makes_from_the_committed_sample(tmp_path):
    _make(tmp_path / "stage")
    assert _tree(tmp_path / "stage") == _tree(COMMITTED), \
        "regenerate: cyt source run dila_kali --from data/samples/dila-kali --out data/index-fixture/dila_kali --today 2026-10-03"


def test_two_runs_give_the_same_bytes_and_the_stage_names_no_host_or_path(tmp_path):
    _make(tmp_path / "a")
    _make(tmp_path / "b", work_dir=tmp_path / "work")
    assert _tree(tmp_path / "a") == _tree(tmp_path / "b")
    meta = (tmp_path / "a" / "PACK.json").read_text()
    assert str(tmp_path) not in meta and str(REPO) not in meta and "/tmp/" not in meta
    doc = json.loads(meta)
    assert doc["schema"] == "aw.family-stage.v1" and doc["family_id"] == "dila_kali"
    assert doc["pack_id"] == "dila-kali-v1-local-7105286712e7" and doc["rows"] == 1
    # the work dir keeps the sealed L0 (both XML files) and the pack, when it is asked for
    assert (tmp_path / "work" / "pack" / "manifest.json").is_file()
    assert len((tmp_path / "work" / "l0" / "index.jsonl").read_text().splitlines()) == 2


def test_the_stage_rows_carry_the_licence_the_url_and_a_point_id(tmp_path):
    _make(tmp_path / "stage")
    with gzip.open(tmp_path / "stage" / "chunks" / "shard-0000.texts.jsonl.gz", "rt",
                   encoding="utf-8") as fh:
        (row,) = [json.loads(x) for x in fh]
    assert row["license"] == "Etalab-2.0" and row["point_id"] and row["lang"] == "fr"
    assert row["url"] == "https://www.legifrance.gouv.fr/conv_coll/article/KALIARTI000005833790"


def test_from_is_refused_for_a_family_that_needs_a_host_and_asks_no_consent(tmp_path):
    from cytir.sources import stage_out
    for family in ("fedlex_exact", "govuk_content", "us_federal_register_ecfr"):
        with pytest.raises(stage_out.StageOutRefused, match="--from"):
            stage_out.run_to_stage(family, out=tmp_path / family, today=TODAY, from_dir=SAMPLE,
                                   log=lambda m: None)
        assert not (tmp_path / family).exists()


def test_a_stage_is_never_overwritten(tmp_path):
    from cytir.sources import stage_out
    _make(tmp_path / "stage")
    before = _tree(tmp_path / "stage")
    with pytest.raises(stage_out.StageOutRefused, match="exists"):
        _make(tmp_path / "stage")
    assert _tree(tmp_path / "stage") == before


def test_a_family_with_unknowns_in_its_declaration_makes_a_pack_but_no_stage(tmp_path):
    """A stage is made only from a clean declaration. `loda_exact` was the case until ruling 27
    (POLICY v3.3 declared `identity.minted_by: text-anchor-v0`): the same refusal is shown on a
    copy of the contracts where that one value is `unknown` again."""
    from cytir.sources import stage_out
    doc = json.loads((REPO / "data" / "source-family-contracts.v1.json").read_text(encoding="utf-8"))
    row = next(f for f in doc["families"] if f["family_id"] == "loda_exact")
    assert row["unknowns"] == [] and row["identity"]["minted_by"] == "text-anchor-v0"
    row["identity"]["minted_by"] = "unknown"
    row["unknowns"] = ["identity.minted_by"]
    contracts = tmp_path / "contracts.json"
    contracts.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(stage_out.StageOutRefused, match=r"unknowns .*identity\.minted_by"):
        stage_out.run_to_stage("loda_exact", out=tmp_path / "stage", today=TODAY,
                               from_dir=REPO / "data" / "samples" / "legi-loda",
                               options={"release": "aw-fixture-loda-v1"},
                               contracts_path=contracts, log=lambda m: None)
    assert not (tmp_path / "stage").exists()


def test_a_pack_altered_after_it_was_sealed_is_refused(tmp_path):
    from cytir.sources import stage_out
    _make(tmp_path / "stage", work_dir=tmp_path / "work")
    rec = tmp_path / "work" / "pack" / "records.jsonl"
    rec.write_bytes(rec.read_bytes().replace(b"FORTHAC", b"FORTHAX"))
    with pytest.raises(stage_out.StageOutRefused, match="sha256"):
        stage_out.stage_from_pack(tmp_path / "work" / "pack", "dila_kali", tmp_path / "again")
    assert not (tmp_path / "again").exists()


def test_a_run_that_admits_no_row_makes_no_stage(tmp_path):
    from cytir.sources import stage_out
    only_short = tmp_path / "in" / "global"
    only_short.mkdir(parents=True)
    src = SAMPLE / "global" / "KALI" / "KALIARTI000005833715.xml"       # not in force: excluded
    (only_short / src.name).write_bytes(src.read_bytes())
    with pytest.raises(Exception, match="0 records|nothing composed"):
        stage_out.run_to_stage("dila_kali", out=tmp_path / "stage", today=TODAY,
                               from_dir=tmp_path / "in", log=lambda m: None)
    assert not (tmp_path / "stage").exists()


def _cli(*args):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AW_", "CYTIR_"))}
    return subprocess.run([sys.executable, "-m", "cytir.cli", *args], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)


def test_the_cli_makes_the_stage_and_refuses_from_without_out(tmp_path):
    ok = _cli("source", "run", "dila_kali", "--from", str(SAMPLE), "--out", str(tmp_path / "s"),
              "--today", TODAY)
    assert ok.returncode == 0, ok.stderr
    assert _tree(tmp_path / "s") == _tree(COMMITTED)
    bad = _cli("source", "run", "dila_kali", "--from", str(SAMPLE), "--today", TODAY)
    assert bad.returncode == 2 and "--out" in bad.stderr
    none = _cli("source", "run", "dila_kali", "--today", TODAY)
    assert none.returncode == 2 and "--out-dir" in none.stderr
    net = _cli("source", "run", "fedlex_exact", "--from", str(SAMPLE), "--out", str(tmp_path / "n"))
    assert net.returncode == 1 and "REFUSED" in net.stderr and not (tmp_path / "n").exists()


def test_a_local_stage_never_carries_a_production_pack_id(tmp_path):
    from cytir.sources import family_stage as FS, stage_out
    meta = _make(tmp_path / "s")
    prod = {FS.pack_id("dila_kali", n) for n in range(64)}
    assert meta["pack_id"] not in prod and "-local-" in meta["pack_id"]
    assert meta["pack_id"] == stage_out.local_pack_id("dila_kali", meta["source_records_sha256"])
    assert meta["pack_id"].endswith(meta["source_records_sha256"][:12])


def test_the_work_dir_is_made_beside_the_stage_and_removed(tmp_path, monkeypatch):
    import tempfile
    seen = []
    real = tempfile.mkdtemp
    monkeypatch.setattr(tempfile, "mkdtemp", lambda **kw: seen.append(kw) or real(**kw))
    (tmp_path / "res").mkdir()
    _make(tmp_path / "res" / "stage")
    assert seen and pathlib.Path(seen[0]["dir"]) == tmp_path / "res"
    assert sorted(p.name for p in (tmp_path / "res").iterdir()) == ["stage"]


def test_out_without_from_needs_an_out_dir(tmp_path):
    from cytir.sources import stage_out
    with pytest.raises(stage_out.StageOutRefused, match="--out-dir"):
        stage_out.run_to_stage("fedlex_exact", out=tmp_path / "s", today=TODAY, log=lambda m: None)
    assert not (tmp_path / "s").exists()
    r = _cli("source", "run", "fedlex_exact", "--out", str(tmp_path / "s"))
    assert r.returncode == 2 and "--out-dir" in r.stderr and not (tmp_path / "s").exists()


def test_out_that_is_inside_equals_or_contains_a_live_queue_is_refused(tmp_path):
    """A stage written into (or above, or at) a queue with a LIVE-QUEUE marker would be read by
    the hourly drain as a pack: refused before the family runs, nothing created."""
    from cytir.sources import stage_out
    queue = tmp_path / "root" / "promotion-queue"
    queue.mkdir(parents=True)
    (queue / "LIVE-QUEUE").write_text("")
    for out in (queue / "stage", queue, tmp_path / "root"):
        with pytest.raises(stage_out.StageOutRefused, match="live promotion queue"):
            _make(out)
    assert sorted(p.name for p in queue.iterdir()) == ["LIVE-QUEUE"]
    # the same refusal at the writer
    with pytest.raises(stage_out.StageOutRefused, match="live promotion queue"):
        stage_out.stage_from_pack(tmp_path / "no-pack", "dila_kali", queue / "stage2")
    # a sibling of the queue is fine
    _make(tmp_path / "root" / "elsewhere")
    r = _cli("source", "run", "dila_kali", "--from", str(SAMPLE), "--out", str(queue / "cli"),
             "--today", TODAY)
    assert r.returncode == 1 and "live promotion queue" in r.stderr and not (queue / "cli").exists()
