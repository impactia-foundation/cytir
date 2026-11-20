"""`cytir.sources.family_stage` + the `--family-stage` discovery (task 696).

The route a `chunk.atomic` family takes from a `run_family` pack to the promotion queue:
stage -> L1 -> overlap report -> embed (faked here) -> compose. Real rows throughout: the four
real CASS/JADE decisions in `tests/fixtures/dila_juri/`, run through the family's own door.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cytir.index import compose  # noqa: E402
from cytir.sources import family_stage as FS  # noqa: E402

FIX = HERE / "fixtures" / "dila_juri"
FAMILY = "dila_juri"


@pytest.fixture
def pack(tmp_path):
    from cytir.sources import run_family
    root = tmp_path / "dump"
    d = root / "cass"
    d.mkdir(parents=True)
    for f in sorted(FIX.glob("*.xml")):
        shutil.copy(f, d / f.name)
    out = tmp_path / "run"
    run_family.run_family(FAMILY, out_dir=out, today="2026-10-03",
                          consent=lambda h, dd: pytest.fail("dump route: no host consent"),
                          options={"root": str(root)}, log=lambda m: None)
    return out / "pack"


def _stage(pack, tmp_path, **kw):
    return FS.stage_family_pack(pack, corpus_root=tmp_path / "corpus", family_id=FAMILY,
                                log=lambda m: None, **kw)


def _rows(stage):
    with gzip.open(stage / "chunks" / "shard-0000.texts.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_names_are_derived_from_the_family_id():
    assert FS.slug(FAMILY) == "dila-juri-v1"
    assert FS.pack_id(FAMILY, 3) == "dila-juri-v1-shard3"
    assert FS.l1_release(FAMILY, 3) == "dila-juri-v1-shard3"


def test_stage_cuts_shards_conserves_rows_and_files_l1(pack, tmp_path):
    receipts = _stage(pack, tmp_path, shard_rows=3)
    assert [r["rows"] for r in receipts] == [3, 1]          # 4 real decisions, 3 per stage
    corpus = tmp_path / "corpus"
    assert [n for n, _ in FS.list_stages(corpus, FAMILY)] == [0, 1]
    for n, d in FS.list_stages(corpus, FAMILY):
        meta = json.loads((d / "PACK.json").read_text())
        assert meta["pack_id"] == FS.pack_id(FAMILY, n) and meta["schema"] == FS.STAGE_SCHEMA
        assert FS.is_l1_filed(corpus, FAMILY, n)
        l1 = list((corpus / "l1" / f"release={FS.l1_release(FAMILY, n)}").glob("*.jsonl.gz"))
        assert len(l1) == 1


def test_a_second_pack_of_the_family_continues_the_numbering(pack, tmp_path):
    first = _stage(pack, tmp_path, shard_rows=3)               # stages 0 and 1
    assert [r["shard"] for r in first] == [0, 1]
    with pytest.raises(FS.StageError, match="never re-cut"):
        _stage(pack, tmp_path, shard_rows=2)                   # same pack, other slicing
    assert [r["shard"] for r in _stage(pack, tmp_path, shard_rows=3)] == [0, 1]   # idempotent
    # a different pack (the first two real decisions only) of the same family continues at 2
    rows = pack.joinpath("records.jsonl").read_text().splitlines()[:2]
    body = ("\n".join(rows) + "\n").encode()
    pack2 = tmp_path / "pack2"
    shutil.copytree(pack, pack2)
    (pack2 / "records.jsonl").write_bytes(body)
    import hashlib
    man = json.loads((pack2 / "manifest.json").read_text())
    man["records"].update(count=2, sha256=hashlib.sha256(body).hexdigest())
    man["terminals"].update(units_in=2, units_admitted=2, units_excluded=0)
    (pack2 / "manifest.json").write_text(json.dumps(man))
    second = _stage(pack2, tmp_path, shard_rows=3)
    assert [r["shard"] for r in second] == [2]
    assert [n for n, _ in FS.list_stages(tmp_path / "corpus", FAMILY)] == [0, 1, 2]


def test_a_binary_row_is_held_out_counted_and_the_rest_of_the_pack_stages(pack, tmp_path,
                                                                           monkeypatch):
    """task 777 (ADR-094): the guard quarantines the ROW, never the pack."""
    rows = [json.loads(line) for line in pack.joinpath("records.jsonl").read_text().splitlines()]
    bad = rows[1]
    monkeypatch.setattr(FS, "binary_content",
                        lambda t: "binary_content:test" if t == bad["text"] else None)
    (r,) = _stage(pack, tmp_path)
    assert r["rows"] == len(rows) - 1 and r["refused_binary"] == 1 and r["dropped_served"] == 0
    (_, d), = FS.list_stages(tmp_path / "corpus", FAMILY)
    assert bad["point_id"] not in {x["point_id"] for x in _rows(d)}
    (f,) = (tmp_path / "corpus" / "stages" / FS.slug(FAMILY)).glob("refused-binary-*.jsonl")
    assert [json.loads(x) for x in f.read_text().splitlines()] == [
        {"point_id": bad["point_id"], "url": bad["url"], "why": "binary_content:test"}]
    again = _stage(pack, tmp_path)                       # idempotent: same stage, same count
    assert again[0]["shard"] == r["shard"] and again[0]["refused_binary"] == 1


def test_drop_served_stages_only_net_new_rows_and_lists_every_drop(pack, tmp_path):
    rows = [json.loads(line) for line in pack.joinpath("records.jsonl").read_text().splitlines()]
    served_url = {("http://t", "aw_e8"): {rows[0]["url"]}}
    served_id = {("http://p", "aw_prod"): {rows[1]["point_id"]}}
    receipts = FS.stage_family_pack(
        pack, corpus_root=tmp_path / "corpus", family_id=FAMILY, log=lambda m: None,
        drop_served={"twin": ("http://t", "aw_e8"), "prod": ("http://p", "aw_prod")},
        retrieve=_fake_retrieve(served_id), served_urls=_fake_urls(served_url))
    (r,) = receipts
    assert r["rows"] == 2 and r["dropped_served"] == 2          # 4 = 2 staged + 2 dropped
    (_, d), = FS.list_stages(tmp_path / "corpus", FAMILY)
    assert {x["point_id"] for x in _rows(d)} == {rows[2]["point_id"], rows[3]["point_id"]}
    listed = {json.loads(line)["point_id"] for line in
              (d / "dropped-served-ids.jsonl").read_text().splitlines()}
    assert listed == {rows[0]["point_id"], rows[1]["point_id"]}
    meta = json.loads((d / "PACK.json").read_text())
    assert meta["source_rows"] == 4 and meta["rows"] == 2 and meta["dropped_served_sha256"]
    # a rerun after the pages landed must reuse the stage, not re-decide it
    again = FS.stage_family_pack(
        pack, corpus_root=tmp_path / "corpus", family_id=FAMILY, log=lambda m: None,
        drop_served={"twin": ("http://t", "aw_e8")}, retrieve=_fake_retrieve({}),
        served_urls=_no_urls)
    assert again[0]["shard"] == r["shard"] and again[0]["rows"] == 2


def test_drop_served_with_everything_served_stages_nothing(pack, tmp_path):
    urls = {r["url"] for r in map(json.loads, pack.joinpath("records.jsonl").read_text().splitlines())}
    receipts = FS.stage_family_pack(
        pack, corpus_root=tmp_path / "corpus", family_id=FAMILY, log=lambda m: None,
        drop_served={"twin": ("http://t", "aw_e8")}, retrieve=_fake_retrieve({}),
        served_urls=_fake_urls({("http://t", "aw_e8"): urls}))
    assert receipts == [] and FS.list_stages(tmp_path / "corpus", FAMILY) == []


def test_stage_rows_are_the_pack_rows_with_their_own_license(pack, tmp_path):
    _stage(pack, tmp_path)
    (_, d), = FS.list_stages(tmp_path / "corpus", FAMILY)
    staged = _rows(d)
    source = [json.loads(line) for line in pack.joinpath("records.jsonl").read_text().splitlines()]
    assert staged == source
    assert {r["license"] for r in staged} == {"Etalab-2.0"}


def test_stage_is_idempotent_and_never_overwrites(pack, tmp_path):
    _stage(pack, tmp_path)
    again = _stage(pack, tmp_path)
    assert again[0]["rows"] == 4
    (_, d), = FS.list_stages(tmp_path / "corpus", FAMILY)
    meta = json.loads((d / "PACK.json").read_text())
    meta["source_rows"] = 99
    (d / "PACK.json").write_text(json.dumps(meta))
    with pytest.raises(FS.StageError, match="differs from this pack"):
        _stage(pack, tmp_path)


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m["records"].update(count=0), "0 records"),
    (lambda m: m["terminals"].update(units_in=m["terminals"]["units_in"] + 1), "do not balance"),
    (lambda m: m.update(family_id="other"), "family_id"),
    (lambda m: m["consent"].update(checked=False), "consent was not checked"),
    (lambda m: m["contract"].update(unknowns=["route"]), "unknowns"),
])
def test_stage_refuses_a_bad_manifest(pack, tmp_path, mutate, match):
    mp = pack / "manifest.json"
    man = json.loads(mp.read_text())
    mutate(man)
    mp.write_text(json.dumps(man))
    with pytest.raises(FS.StageError, match=match):
        _stage(pack, tmp_path)
    assert not (tmp_path / "corpus" / "stages").exists() or not FS.list_stages(
        tmp_path / "corpus", FAMILY)


def test_stage_refuses_an_altered_records_file(pack, tmp_path):
    with open(pack / "records.jsonl", "ab") as fh:
        fh.write(b"\n")
    with pytest.raises(FS.StageError, match="sha256"):
        _stage(pack, tmp_path)


def test_stage_refuses_a_row_without_a_per_record_license_before_writing(pack, tmp_path):
    rows = [json.loads(line) for line in pack.joinpath("records.jsonl").read_text().splitlines()]
    rows[-1].pop("license")
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
    (pack / "records.jsonl").write_bytes(body)
    import hashlib
    man = json.loads((pack / "manifest.json").read_text())
    man["records"]["sha256"] = hashlib.sha256(body).hexdigest()
    (pack / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(FS.StageError, match="license"):
        _stage(pack, tmp_path, shard_rows=2)
    assert FS.list_stages(tmp_path / "corpus", FAMILY) == []


def test_stage_refuses_when_l1_filing_fails(pack, tmp_path):
    def failing(argv, env=None, cwd=None):
        return subprocess.CompletedProcess(argv, 3)
    with pytest.raises(FS.StageError, match="L1 filing"):
        FS.stage_family_pack(pack, corpus_root=tmp_path / "corpus", family_id=FAMILY,
                             runner=failing, log=lambda m: None)


def test_staged_ids_reproduce_under_the_family_anchor_scheme(pack, tmp_path):
    _stage(pack, tmp_path)
    (_, d), = FS.list_stages(tmp_path / "corpus", FAMILY)
    ident = compose.verify_identity(str(d / "chunks"),
                                    anchor_fn=compose.anchor_fn_for_family(FAMILY))
    assert ident["checked"] == ident["matched"] == 4
    assert compose.anchor_fn_for_family(FAMILY) is compose.raw_anchor_field


def _no_urls(url, collection, urls):
    return set()


def _fake_urls(served):
    def f(url, collection, urls):
        return {u for u in urls if u in served.get((url, collection), set())}
    return f


def _fake_retrieve(present):
    def retrieve(url, collection, ids):
        return {i for i in ids if i in present.get((url, collection), set())}
    return retrieve


def test_overlap_measures_by_id_writes_the_exclusion_and_pins_its_sha(pack, tmp_path):
    _stage(pack, tmp_path)
    (_, d), = FS.list_stages(tmp_path / "corpus", FAMILY)
    ids = [r["point_id"] for r in _rows(d)]
    assert FS.read_overlap(d) is None
    out = FS.check_overlap(
        d, targets={"twin": ("http://t", "aw_e8"), "prod": ("http://p", "aw_prod")},
        retrieve=_fake_retrieve({("http://t", "aw_e8"): {ids[0], ids[1]},
                                 ("http://p", "aw_prod"): {ids[0]}}),
        served_urls=_no_urls, log=lambda m: None)
    assert out["ids_served"] == 2 and out["rows_net_new"] == 2 and out["rows_excluded"] == 2
    assert out["targets"]["twin"]["ids_present"] == 2 and out["targets"]["prod"]["ids_present"] == 1
    got = [json.loads(line)["point_id"] for line in (d / "overlap-ids.jsonl").read_text().splitlines()]
    assert sorted(got) == sorted([ids[0], ids[1]])
    assert FS.read_overlap(d)["exclude_reason"] == FS.OVERLAP_REASON
    (d / "overlap-ids.jsonl").write_text('{"point_id": "x"}\n')
    with pytest.raises(FS.StageError, match="does not match"):
        FS.read_overlap(d)


def test_overlap_with_no_served_ids_writes_no_exclusion(pack, tmp_path):
    _stage(pack, tmp_path)
    (_, d), = FS.list_stages(tmp_path / "corpus", FAMILY)
    out = FS.check_overlap(d, targets={"twin": ("http://t", "aw_e8")},
                           retrieve=_fake_retrieve({}), served_urls=_no_urls, log=lambda m: None)
    assert out["ids_served"] == 0 and not (d / "overlap-ids.jsonl").exists()
    assert FS.read_overlap(d)["rows_net_new"] == 4 and out["needs_ruling"] is False


def _discover(tmp_path, logs):
    import cytir.ops.growth_pending_compose as GPC
    return GPC.discover_family_stage_pending(str(tmp_path / "corpus"), FAMILY,
                                             log=logs.append)


def test_discovery_waits_for_the_overlap_check_then_returns_a_job(pack, tmp_path):
    _stage(pack, tmp_path)
    logs: list[str] = []
    assert _discover(tmp_path, logs) == []
    assert any("AWAITING OVERLAP CHECK" in m for m in logs)
    (_, d), = FS.list_stages(tmp_path / "corpus", FAMILY)
    FS.check_overlap(d, targets={"twin": ("http://t", "aw_e8")}, retrieve=_fake_retrieve({}), served_urls=_no_urls,
                     log=lambda m: None)
    (job,) = _discover(tmp_path, [])
    assert job.pack_id == "dila-juri-v1-shard0" and job.family == FAMILY
    assert job.chunks_dir == str(d / "chunks") and job.exclude_ids_file is None


def test_discovery_passes_the_overlap_file_as_the_declared_exclusion(pack, tmp_path):
    _stage(pack, tmp_path)
    (_, d), = FS.list_stages(tmp_path / "corpus", FAMILY)
    first = _rows(d)[0]["point_id"]
    FS.check_overlap(d, targets={"twin": ("http://t", "aw_e8")},
                     retrieve=_fake_retrieve({("http://t", "aw_e8"): {first}}),
                     served_urls=_no_urls, log=lambda m: None)
    (job,) = _discover(tmp_path, [])
    assert job.exclude_ids_file == str(d / "overlap-ids.jsonl")
    assert job.exclude_ids_reason == FS.OVERLAP_REASON


def test_discovery_waits_for_l1(pack, tmp_path):
    _stage(pack, tmp_path)
    (_, d), = FS.list_stages(tmp_path / "corpus", FAMILY)
    shutil.rmtree(tmp_path / "corpus" / "l1")
    logs: list[str] = []
    assert _discover(tmp_path, logs) == []
    assert any("AWAITING L1" in m for m in logs)


def test_the_flag_reaches_pending_jobs(pack, tmp_path):
    import cytir.ops.growth_pending_compose as GPC
    _stage(pack, tmp_path)
    (_, d), = FS.list_stages(tmp_path / "corpus", FAMILY)
    FS.check_overlap(d, targets={"twin": ("http://t", "aw_e8")}, retrieve=_fake_retrieve({}), served_urls=_no_urls,
                     log=lambda m: None)
    args = GPC.build_parser().parse_args(["--family-stage", FAMILY])
    assert args.family_stage == [FAMILY]
    jobs = GPC.pending_jobs(cc_root=None, se_root=None, queue_dir=str(tmp_path / "q"),
                            corpus_root=str(tmp_path / "corpus"), family_stages=[FAMILY])
    assert [j.pack_id for j in jobs] == ["dila-juri-v1-shard0"]


def _write_vectors(vectors_dir, ids, dim=1024):
    import numpy as np
    vectors_dir.mkdir(parents=True, exist_ok=True)
    np.zeros((len(ids), dim), dtype="<f2").tofile(vectors_dir / "shard-0000.vectors.f16")
    with open(vectors_dir / "shard-0000.row_ids.jsonl", "w") as fh:
        for pid in ids:
            fh.write(json.dumps({"point_id": pid}) + "\n")
    (vectors_dir / "shard-0000.vectors.manifest.json").write_text(json.dumps(
        {"schema": "aw.413a.vectors.v1", "shard": "shard-0000", "n": len(ids), "dim": dim,
         "dtype": "float16"}))


def test_compose_stamps_the_dump_licence_consent_and_keeps_the_per_record_license(pack, tmp_path):
    """The whole bridge, vectors faked: stage -> overlap -> job -> compose. Consent is derived by
    the ADR-115 engine from the family's declared basis (r5, the dump licence), never defaulted."""
    _stage(pack, tmp_path)
    corpus = tmp_path / "corpus"
    (_, d), = FS.list_stages(corpus, FAMILY)
    FS.check_overlap(d, targets={"twin": ("http://t", "aw_e8")}, retrieve=_fake_retrieve({}), served_urls=_no_urls,
                     log=lambda m: None)
    (job,) = _discover(tmp_path, [])
    _write_vectors(pathlib.Path(job.vectors_dir), [r["point_id"] for r in _rows(d)])
    res = compose.compose_pack(
        pack_id=job.pack_id, family=job.family, chunks_dir=job.chunks_dir,
        vectors_dir=job.vectors_dir, collection="aw_e8", mrl=256,
        queue_dir=str(tmp_path / "queue"))
    manifest = json.loads((pathlib.Path(res["pack_dir"]) / "manifest.json").read_text())
    assert manifest["identity_check"] == "pack-declared" and "4/4" in manifest["identity_reason"]
    stamped = pathlib.Path(manifest["pack_chunks"])
    with gzip.open(stamped / "shard-0000.texts.jsonl.gz", "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    assert len(rows) == 4
    for r in rows:
        assert r["search_ok"] is True and r["train_ok"] is True
        assert r["consent_basis"] == "official-dump-licence-family"
        assert r["license"] == "Etalab-2.0"


# ---- the url overlap: the OLD representation of the same page, under another point id ----

def _stage_with_urls(pack, tmp_path):
    _stage(pack, tmp_path)
    (_, d), = FS.list_stages(tmp_path / "corpus", FAMILY)
    return d, [r["url"] for r in _rows(d)]


def test_url_overlap_over_the_drain_ceiling_needs_a_ruling_and_discovery_refuses(pack, tmp_path):
    d, urls = _stage_with_urls(pack, tmp_path)
    out = FS.check_overlap(
        d, targets={"twin": ("http://t", "aw_e8")}, retrieve=_fake_retrieve({}),
        served_urls=_fake_urls({("http://t", "aw_e8"): {urls[0]}}), log=lambda m: None)
    assert out["urls_served"] == 1 and out["url_share"] == 0.25
    assert out["needs_ruling"] is True and out["rows_excluded"] == 0     # nothing dropped silently
    assert not (d / "overlap-ids.jsonl").exists()
    logs: list[str] = []
    assert _discover(tmp_path, logs) == []
    assert any("AWAITING RULING" in m and "1 of 4 urls" in m for m in logs)


def test_exclude_served_urls_declares_the_drop_and_discovery_passes_it(pack, tmp_path):
    d, urls = _stage_with_urls(pack, tmp_path)
    rows = _rows(d)
    out = FS.check_overlap(
        d, targets={"twin": ("http://t", "aw_e8")}, retrieve=_fake_retrieve({}),
        served_urls=_fake_urls({("http://t", "aw_e8"): {urls[0], urls[1]}}),
        exclude_served_urls=True, log=lambda m: None)
    assert out["needs_ruling"] is False and out["exclude_served_urls"] is True
    assert out["rows_excluded"] == 2 and out["rows_net_new"] == 2 and out["ids_served"] == 0
    assert out["exclude_reason"] == FS.OVERLAP_REASON_URL
    got = {json.loads(line)["point_id"] for line in (d / "overlap-ids.jsonl").read_text().splitlines()}
    assert got == {rows[0]["point_id"], rows[1]["point_id"]}
    (job,) = _discover(tmp_path, [])
    assert job.exclude_ids_file == str(d / "overlap-ids.jsonl")
    assert job.exclude_ids_reason == FS.OVERLAP_REASON_URL


def test_url_overlap_under_the_ceiling_is_left_to_the_drain(pack, tmp_path, monkeypatch):
    d, urls = _stage_with_urls(pack, tmp_path)
    monkeypatch.setattr(FS, "URL_SHARE_MAX", 0.30)
    out = FS.check_overlap(
        d, targets={"twin": ("http://t", "aw_e8")}, retrieve=_fake_retrieve({}),
        served_urls=_fake_urls({("http://t", "aw_e8"): {urls[0]}}), log=lambda m: None)
    assert out["needs_ruling"] is False and out["rows_excluded"] == 0
    (job,) = _discover(tmp_path, [])
    assert job.exclude_ids_file is None


def test_only_pack_narrows_the_run_to_the_named_stage(pack, tmp_path, monkeypatch):
    import cytir.ops.growth_pending_compose as GPC
    _stage(pack, tmp_path, shard_rows=2)
    for _n, d in FS.list_stages(tmp_path / "corpus", FAMILY):
        FS.check_overlap(d, targets={"twin": ("http://t", "aw_e8")}, retrieve=_fake_retrieve({}),
                         served_urls=_no_urls, log=lambda m: None)
    seen = []
    monkeypatch.setattr(GPC.AIC, "embed_compose_enqueue",
                        lambda **kw: seen.append(kw["pack_id"]) or {"ready_sha": "x"})
    GPC.run(cc_root=None, se_root=None, queue_dir=str(tmp_path / "q"),
            corpus_root=str(tmp_path / "corpus"), family_stages=[FAMILY],
            only_packs=["dila-juri-v1-shard1"], log=lambda m: None)
    assert seen == ["dila-juri-v1-shard1"]
    assert GPC.build_parser().parse_args(["--only-pack", "a"]).only_pack == ["a"]


def test_overlap_cli_leaves_an_already_measured_stage_alone_unless_rechecked(
        pack, tmp_path, monkeypatch):
    """A composed pack pins the sha of its overlap-ids.jsonl; a re-measure after the pages landed
    would rewrite it and make the drain refuse the pack."""
    _stage(pack, tmp_path, shard_rows=2)                       # stages 0 and 1
    corpus = tmp_path / "corpus"
    (_, d0), (_, d1) = FS.list_stages(corpus, FAMILY)
    FS.check_overlap(d0, targets={"twin": ("http://t", "aw_e8")}, retrieve=_fake_retrieve({}),
                     served_urls=_no_urls, log=lambda m: None)
    measured = []
    monkeypatch.setattr(FS, "check_overlap",
                        lambda d, **kw: measured.append(pathlib.Path(d).name))
    argv = ["overlap", FAMILY, "--corpus-root", str(corpus), "--twin", "http://t",
            "--prod", "http://p"]
    FS.main(argv)
    assert measured == ["chunks-shard1"]                       # shard 0 was left alone
    measured.clear()
    FS.main(argv + ["--recheck"])
    assert measured == ["chunks-shard0", "chunks-shard1"]


def test_the_ceiling_is_the_drains_own_number():
    from cytir.index import promote_sealed_pack as PSP
    assert FS.URL_SHARE_MAX == PSP.ALREADY_SERVED_FAIL_SHARE_MAX == 0.05


def test_overlap_drops_a_url_a_sibling_stage_already_holds_and_counts_it(pack, tmp_path):
    """Task 703 (D1 c): a URL staged by another stage of the family (the other route, an earlier
    window) is as taken as a served one: its rows are excluded and the count is in OVERLAP.json."""
    _stage(pack, tmp_path)
    (_, d), = FS.list_stages(tmp_path / "corpus", FAMILY)
    rows = _rows(d)
    taken = rows[0]["url"]
    assert FS.stage_urls(d) == {r["url"] for r in rows}
    assert FS.other_stage_urls(tmp_path / "corpus", FAMILY, d) == set()     # the only stage
    out = FS.check_overlap(d, targets={"twin": ("http://t", "aw_e8")}, retrieve=_fake_retrieve({}),
                           served_urls=_no_urls, exclude_served_urls=True,
                           staged_elsewhere={taken, "https://elsewhere.example/x"},
                           log=lambda m: None)
    assert out["urls_staged_elsewhere"] == 1 and out["urls_served"] == 1
    want = {r["point_id"] for r in rows if r["url"] == taken}
    got = {json.loads(x)["point_id"] for x in (d / "overlap-ids.jsonl").read_text().splitlines()}
    assert got == want and out["rows_net_new"] == len(rows) - len(want)


def test_a_refused_url_is_counted_by_reason_and_never_listed_as_already_served(pack, tmp_path,
                                                                              monkeypatch):
    """task 2713: the stage door's URL admission refusal is its own receipt term, not `dropped_served`."""
    rows = [json.loads(line) for line in pack.joinpath("records.jsonl").read_text().splitlines()]
    bad = rows[1]
    monkeypatch.setattr(FS.UA, "refusal", lambda u: "search-result-page" if u == bad["url"] else None)
    logs = []
    (r,) = FS.stage_family_pack(pack, corpus_root=tmp_path / "corpus", family_id=FAMILY, log=logs.append)
    n_bad = sum(1 for x in rows if x["url"] == bad["url"])
    assert r["refused_urls"] == {"search-result-page": n_bad}
    assert r["dropped_served"] == 0 and r["rows"] == len(rows) - n_bad
    (_, d), = FS.list_stages(tmp_path / "corpus", FAMILY)
    assert not (d / "dropped-served-ids.jsonl").exists()
    got = [json.loads(x) for x in (d / "refused-urls.jsonl").read_text().splitlines()]
    assert {x["point_id"] for x in got} == {x["point_id"] for x in rows if x["url"] == bad["url"]}
    assert all(x["reason"] == "search-result-page" for x in got)
    assert json.loads((d / "PACK.json").read_text())["refused_urls_sha256"]
    assert any("refused" in m and "search-result-page=" in m for m in logs)
    assert not any("already served" in m and "search-result-page" in m for m in logs)
