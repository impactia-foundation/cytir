"""`de_openlegaldata_v1` on the family door, on REAL decisions (task 1373).

`cyt source run de_openlegaldata_v1 --from data/samples/openlegaldata --out STAGE`: no network, no Qdrant, no
model. The sample is one small parquet file of five real shard-0 rows (`data/samples/openlegaldata/SOURCE.md`
records the cut and every row's hash). Pins: the sample is the cut its provenance file names; the adapter
renders the same Markdown, byte for byte, as the extractor functions the shard chain has always called, and
chunks it as the declared chunker does; the real empty-body row is counted, not dropped; the committed fixture
stage is the door's output; the family names no host, reads no private tree and loads DuckDB lazily.
"""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

REPO = pathlib.Path(__file__).resolve().parent.parent
SAMPLES = REPO / "data" / "samples" / "openlegaldata"
REL = "original/parquet/dump-20260520/train-00000-of-00054.parquet"
SHARD = SAMPLES / REL
COMMITTED = REPO / "data" / "index-fixture" / "de_openlegaldata_v1"
TODAY = "2026-10-28"
FAMILY = "de_openlegaldata_v1"
SHARD_SHA256 = "a1561afa83a6e4139dbfec431e88015e80a4479d50cb68d9b79525c8528f98e8"
RENDERED_IDS = [58897, 58910, 58913, 58916]
EMPTY_ID = 59930
#: sha256 of the canonical-JSON lines of the Markdown rows `build_markdown_row` makes from the sample
#: (the extractor of the old shard chain)
RENDERED_SHA256 = "97ecf2d7801d71b31faedf464b494abe5de3d95830d1e7238342606ad7bb8e2c"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _make(out: pathlib.Path, **kw):
    from cytir.sources import stage_out
    return stage_out.run_to_stage(FAMILY, out=out, today=TODAY, from_dir=SAMPLES, log=lambda m: None, **kw)


def _tree(root: pathlib.Path) -> dict[str, str]:
    return {str(p.relative_to(root)): _sha(p.read_bytes()) for p in sorted(root.rglob("*")) if p.is_file()}


def _rows(stage: pathlib.Path) -> list[dict]:
    with gzip.open(stage / "chunks" / "shard-0000.texts.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(x) for x in fh]


def _old_rows() -> list[dict]:
    """What the shard chain's extractor makes from the shard: the old path."""
    from cytir.extract import openlegaldata as E

    parent = {"relative_path": REL, "sha256": SHARD_SHA256}
    out = []
    for raw in E.read_shard_rows(SHARD):
        if E.build_markdown(raw) is not None:
            out.append(E.build_markdown_row(raw, parent_archive=parent))
    return out


def _ctx(tmp_path, **options):
    from cytir.sources.family import RunContext

    return RunContext(out_dir=tmp_path, today=TODAY, options=options)


def test_the_sample_is_the_cut_its_provenance_file_names():
    assert _sha(SHARD.read_bytes()) == SHARD_SHA256
    assert sorted(str(p.relative_to(SAMPLES)) for p in SAMPLES.rglob("*") if p.is_file()) == sorted(
        [REL, "SOURCE.md"])
    note = (SAMPLES / "SOURCE.md").read_text(encoding="utf-8")
    assert SHARD_SHA256 in note and "UrhG" in note and "dump-20260520" in note
    from cytir.extract import openlegaldata as E
    rows = E.read_shard_rows(SHARD)
    assert [r["id"] for r in rows] == RENDERED_IDS + [EMPTY_ID]
    for r in rows:
        assert hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() in note
    assert [bool(r["markdown_content"]) for r in rows] == [True] * 4 + [False]    # a REAL empty body


def test_the_door_renders_what_the_old_chain_extractor_renders(tmp_path, word_only_chunking_allowed):
    from cytir.acquire.artifact_io import canonical_json
    old = _old_rows()
    assert [r["source_record_id"] for r in old] == RENDERED_IDS
    assert _sha("".join(canonical_json(r) + "\n" for r in old).encode()) == RENDERED_SHA256
    stage = _make(tmp_path / "stage")
    rows = _rows(tmp_path / "stage")
    assert stage["source_rows"] == len(rows) == 4
    by_url = {r["url"]: r for r in rows}
    for o in old:                                # one passage per decision on these short decisions
        r = by_url[o["url"]]
        assert r["title"] == o["title"] and r["lang"] == "de"
        assert r["license"] == "public_domain_official_work (UrhG §5)"
        assert r["anchor"] == f"document:{o['document_id']}:chunk-0"
        assert r["document_id"] == o["document_id"]


def test_the_chunks_are_the_declared_chunkers_over_the_old_markdown(tmp_path, word_only_chunking_allowed):
    from cytir.core import common as C
    from cytir.sources.families import openlegaldata as F
    from cytir.sources.family import RunContext

    contract = next(f for f in json.loads((REPO / "data" / "source-family-contracts.v1.json")
                                          .read_text())["families"] if f["family_id"] == FAMILY)
    selection = F._policy_selection(RunContext(out_dir=tmp_path, today=TODAY, contract=contract))
    _make(tmp_path / "stage")
    by_url = {r["url"]: r for r in _rows(tmp_path / "stage")}
    for o in _old_rows():
        direct = list(C.chunk_markdown_channels(o["markdown"], selection=selection))
        assert len(direct) == 1
        assert by_url[o["url"]]["text"] == direct[0].text
        assert by_url[o["url"]]["dense_text"] == direct[0].dense_text
        assert by_url[o["url"]]["sparse_text"] == direct[0].sparse_text


def test_the_real_empty_body_is_counted_and_not_dropped_silently(tmp_path):
    from cytir.sources.families.openlegaldata import OpenLegalDataFamily

    fam, ctx = OpenLegalDataFamily(), _ctx(tmp_path, root=str(SAMPLES))
    unit = fam.discover(ctx)[0]
    docs = fam.extract(fam.acquire(unit, ctx)[0], ctx)
    assert len(docs) == 4
    assert ctx.notes["per_shard"][REL] == {"rendered": 4, "empty_body": 1}


def test_the_pointer_seals_the_shard_not_its_rows(tmp_path):
    from cytir.sources.families.openlegaldata import OpenLegalDataFamily

    fam, ctx = OpenLegalDataFamily(), _ctx(tmp_path, root=str(SAMPLES))
    pointer = json.loads(fam.acquire(fam.discover(ctx)[0], ctx)[0].body)
    assert pointer["sha256"] == SHARD_SHA256 and pointer["relative_path"] == REL
    assert pointer["schema"] == "aw.openlegaldata-shard-pointer.v1"


def test_a_dump_route_family_contacts_no_host_and_is_registered():
    from cytir.sources import registry

    registry.load_modules()
    assert FAMILY in registry.registered()
    from cytir.sources.families.openlegaldata import OpenLegalDataFamily
    assert OpenLegalDataFamily.direct_crawl is False


def test_the_committed_fixture_stage_is_what_the_door_makes(tmp_path, word_only_chunking_allowed):
    _make(tmp_path / "stage")
    assert _tree(tmp_path / "stage") == _tree(COMMITTED), (
        "regenerate: cyt source run de_openlegaldata_v1 --from data/samples/openlegaldata "
        "--out data/index-fixture/de_openlegaldata_v1 --today 2026-10-28")


def test_a_missing_root_and_an_empty_walk_refuse(tmp_path):
    from cytir.sources.families.openlegaldata import OpenLegalDataFamily

    fam = OpenLegalDataFamily()
    with pytest.raises(ValueError, match="needs --opt root"):
        fam.discover(_ctx(tmp_path))
    with pytest.raises(RuntimeError, match="no .parquet shard"):
        fam.discover(_ctx(tmp_path, root=str(tmp_path)))


def _imports(path: pathlib.Path, top_only: bool = False) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in (tree.body if top_only else ast.walk(tree)):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def test_the_family_imports_no_private_tree_and_loads_duckdb_lazily():
    path = REPO / "cytir" / "sources" / "families" / "openlegaldata.py"
    assert not {"pipeline", "tasks"} & _imports(path)
    assert not {"duckdb", "pyarrow"} & _imports(path, top_only=True)
    code = ("import sys; import cytir.extract.openlegaldata, cytir.sources.families.openlegaldata;"
            "print(sorted(m for m in ('duckdb', 'pyarrow') if m in sys.modules))")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=120)
    assert res.returncode == 0, res.stderr[-1500:]
    assert res.stdout.strip().splitlines()[-1] == "[]"


def test_parse_shards_expands_ranges_dedupes_and_refuses_outside_the_dump():
    from cytir.sources.families import openlegaldata as F
    assert F.parse_shards("2-4,7,3") == [2, 3, 4, 7]
    with pytest.raises(ValueError, match="outside"):
        F.parse_shards("50-54")


def test_l1_writes_the_shard_release_under_out_dir_with_the_chain_release_name(tmp_path):
    from cytir.sources.families import openlegaldata as F

    out = tmp_path / "out"
    ctx = _ctx(out, root=str(SAMPLES), shards="0")
    m = F.OpenLegalDataFamily().l1(ctx)
    rel = out / "l1" / "release=openlegaldata-v1-shard0"
    assert m == {"shards": [{"shard": 0, "release": "openlegaldata-v1-shard0",
                             "status": "persisted", "documents": 4,
                             "skipped_empty": 1, "source_rows": 5}]}
    d = m["shards"][0]
    assert d["source_rows"] == d["documents"] + d["skipped_empty"]
    assert (rel / "_register_manifest.json").is_file()
    assert not (out / "_l1_src").exists()
    files = [p for p in rel.rglob("*.jsonl.gz")]
    assert len(files) == 1
    with gzip.open(files[0], "rt", encoding="utf-8") as fh:
        got = [json.loads(x) for x in fh]
    assert [r["source_record_id"] for r in got] == RENDERED_IDS
    assert F.l1_release_name(0) == "openlegaldata-v1-shard0"
    assert F.OpenLegalDataFamily().l1(ctx)["shards"][0]["status"] == "persisted"   # idempotent


def test_l1_skips_a_shard_the_corpus_already_files_and_never_writes_there(tmp_path):
    from cytir.sources.families import openlegaldata as F

    corpus = tmp_path / "corpus"
    filed = corpus / "l1" / "release=openlegaldata-v1-shard0"
    filed.mkdir(parents=True)
    (filed / "_register_manifest.json").write_text("{}")
    before = _tree(corpus)
    out = tmp_path / "out"
    m = F.OpenLegalDataFamily().l1(_ctx(out, root=str(SAMPLES), shards="0", corpus_root=str(corpus)))
    assert m == {"shards": [{"shard": 0, "release": "openlegaldata-v1-shard0", "status": "already_filed"}]}
    assert not (out / "l1").exists()
    assert _tree(corpus) == before


def test_helpers_name_the_shard_stage_and_the_l1_filing(tmp_path):
    from cytir.sources.families import openlegaldata as F
    mat, chunks = F.shard_dirs(tmp_path, 3)
    assert mat.name == "materialize-shard3" and chunks.name == "chunks-shard3"
    assert not F.is_chunked(chunks) and not F.is_l1_filed(tmp_path, 3)
    chunks.mkdir(parents=True)
    (chunks / "PACK.json").write_text("{}")
    assert F.is_chunked(chunks)
