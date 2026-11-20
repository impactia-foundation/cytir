"""`openalex_academic_dump` on the family door, on REAL works (slice 7 step 5, task 696).

`cyt source run openalex_academic_dump --from data/samples/openalex-academic --out STAGE`: no network, no
Qdrant, no model. The committed sample is one small parquet file: fifteen whole works cut from one part of
the public OpenAlex snapshot (CC0); `data/samples/openalex-academic/SOURCE.md` records the part's sha256,
the cut and every work's hash. The family's stage goes through `cyt index build` (consent derived by ADR-115
rule r1-openalex) and is part of the `index-serve-gate` chain (`tests/test_aw_index_serve_gate.py`).

Pins: the sample is the cut its provenance file names; the cut tool takes exactly the named works; the
extractor, moved out of the old materializer script, selects and renders what that script did (the sha256
of its output on the sample; a characterization test in the old tree runs the old script against the same
sha256 pins); the part-selection functions agree with the real frontier and plans of the old tranche
planner; the family imports no private tree and loads DuckDB and pyarrow only when it runs; the part streams;
authors are master-only; a run is byte-deterministic and the committed fixture stage is the door's output;
a part with no selectable work is excluded with its reason; a run that admits nothing refuses.
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
SAMPLES = REPO / "data" / "samples" / "openalex-academic"
REL = "updated_date=2026-02-09/part_0046.parquet"
PART = SAMPLES / REL
COMMITTED = REPO / "data" / "index-fixture" / "openalex_academic_dump"
TODAY = "2026-10-03"
FAMILY = "openalex_academic_dump"
PART_SHA256 = "cbed9d118651ddf639672c199e509f449eba5e7a5076579b4c95451a24735961"
SOURCE_PART_SHA256 = "220945f93be699cf538207a61e346a59fa1e0d5bc802f2aaa3c8636dba90a73a"
SOURCE_RUN_SHA256 = "b27163ed5bed84ce51289b690b21aeec655e77285449d5d4eab2b76fcad8b1cd"
WORKS = {
    "W1043719763": "31fac12f0da704a5cb76ec553c121a6bc4b552af91191d0763227d5df42de2b1",
    "W1060887079": "2ca964fbae28e66a2dba4c4515e79b57d037b2af810908f26df9f1b8af07c3bb",
    "W114559381": "dbf308355427451a1877bc8e5477b2ef1508d74c6758324ec94a13afdb232c87",
    "W127391412": "422e4ca88eeaa2efd4eb3abccd25a257c7fb678949f3ffe5ec2136238f613ef8",
    "W1433235132": "e3585dcf16665dcd63c3d4ae3d8f1fd381bc27e2e902971590974c831f71b93f",
    "W153375954": "5851ba951c83ecc52266f17bd43fbec57a6f6c05cc0be2106144daa5ef2123d9",
    "W1551380949": "42998013aa61a408b17ba0ee2e4ce741e602e65d481f45c420afe78c5d760f18",
    "W2019609318": "82e584070cbea8519f188be97715c940535a43f8798dd9e2d237b1a8075bb0b5",
    "W2062740319": "2b73b36313d04599322450ee898e96626fa27038adeb5f4e042d08ec669325d2",
    "W2266349331": "4d212421e1d66f9bd2fcff66611178a959b8451f25211f4dc964ea818eb239b8",
    "W2520232370": "eccc2abb0569462799c06f906b2c629c12c8a17db2d56fd7c6cca525e677a63e",
    "W2625258660": "841d41222f4fdce3a85cfb7ed0fb0359580114a3c6ccaf729f04e2a9b1c175ff",
    "W2905023637": "548c5239e6c786ac307d40523378626e09684a1c36fcf37a3e1c00f194048a42",
    "W3122717005": "d1369f6fcd62c92032aed035648c0c79e95bc86ffe946ffca146e944d8368de1",
    "W6959737561": "6a3b19bb0f360716ddc176a2796b6ec89ab6a475f37728176b12ec2314f6a776",
}
RECOVERY = "https://openalex.org/W3122717005"
SELECTED = [f"https://openalex.org/{w}" for w in (
    "W114559381", "W2019609318", "W2062740319", "W2266349331", "W2520232370", "W2905023637")]
#: Five works the old flat rule (`cited >= 1`) admitted and the ADR-102 ladder refuses (task 1250):
#: (publication year, citations, core venue) 2007/2, 2004/2, 2010/2, 2017/1, 2007/7.
LADDER_REFUSED = [f"https://openalex.org/{w}" for w in (
    "W1043719763", "W1433235132", "W1551380949", "W2625258660", "W3122717005")]
#: sha256 of the canonical-JSON lines of what the selection picks from the sample, and of the Markdown rows
#: the renderer makes from them: the same bytes the old script makes (checked with the script itself, run on
#: this sample, when the logic moved; the characterization test in the old tree re-checks)
SELECTED_RAW_SHA256 = "7a15cd4d3c4087b04d6578e4077a8451bb7e113f85cd193e5e7f5001bac88b42"
RENDERED_SHA256 = "61ce3a4726e4cf2c1a9498f691d0e4e5e5b431b81d345e88e41bdad0ee04ac78"
#: sha256 of the embedded surfaces of the sample's ten passages, one canonical-JSON line per chunk
#: `{"url", "ordinal", "text"}`: the dense input (`dense_text`) and the sparse input (`sparse_text`, equal
#: to the canonical text). Derived from the OLD stage script's own chunking call (`chunk_generic_records`
#: on the old renderer's Markdown, with the pinned policy); the family's stage meets them byte for byte.
DENSE_SURFACE_SHA256 = "009ea87d29048e66368a91a4e3c3b2237b92769730da4fec144bf4d138bbea1f"
SPARSE_SURFACE_SHA256 = "09cbc595c3e12c40ec21ffd8c00d357424229edd6940e876d322a281777ef8f9"
PLANS = {
    "min10000-excluding-pilot-v1": "3aa10a1ddd16328f229d7b5ca43469b1c71958e6a0e5fb416866c5a774573968",
    "min100000-excluding-prior-v1": "0d697cd6a39585a134af8b9707363294ee21dd38b290dc0550a735dc57bcdde7",
    "min2500000-excluding-prior253-v1": "7d6bd8b418e4b70d5943554a6085c066949e3e2134a00ca0689cadf95d51dfd9",
}
FRONTIER_SHA256 = "1f3a12f5628d80f8df87508af4ec3b23880266d70fd88874581340157a2cbff4"


def _tree(root: pathlib.Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _rows(stage: pathlib.Path) -> list[dict]:
    with gzip.open(stage / "chunks" / "shard-0000.texts.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(x) for x in fh]


def _make(out: pathlib.Path, from_dir=SAMPLES, **kw):
    from cytir.sources import stage_out
    return stage_out.run_to_stage(FAMILY, out=out, today=TODAY, from_dir=from_dir,
                                  log=lambda m: None, **kw)


def _canonical(rows) -> bytes:
    from cytir.acquire.artifact_io import canonical_json
    return "".join(canonical_json(r) + "\n" for r in rows).encode()


def _gz_json(path: pathlib.Path):
    return json.loads(gzip.decompress(path.read_bytes()))


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _tool():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cut_openalex", REPO / "tests" / "fixture_tools" / "cut_openalex.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def _work_hashes() -> dict[str, str]:
    return {i.rsplit("/", 1)[1]: h for i, h in _tool().work_hashes(PART)}


# ------------------------------------------------------------------ the sample and its cut
def test_the_sample_is_the_cut_its_provenance_file_names():
    assert _sha(PART.read_bytes()) == PART_SHA256
    assert sorted(str(p.relative_to(SAMPLES)) for p in SAMPLES.rglob("*") if p.is_file()) == sorted([
        REL, "SOURCE.md", "frontier.jsonl.gz",
        *(f"plans/{n}.PLAN.json.gz" for n in PLANS)])
    assert _work_hashes() == WORKS
    note = (SAMPLES / "SOURCE.md").read_text(encoding="utf-8")
    assert PART_SHA256 in note and SOURCE_PART_SHA256 in note and FRONTIER_SHA256 in note
    assert all(h in note for h in WORKS.values()) and "CC0" in note and "docs.openalex.org" in note
    assert PART.stat().st_size < 200_000


def test_the_sample_rows_are_whole_works_with_every_column():
    import pyarrow.parquet as pq
    schema = pq.ParquetFile(str(PART)).schema_arrow
    assert len(schema.names) == 49
    assert {"id", "doi", "title", "authorships", "abstract_inverted_index", "primary_location",
            "best_oa_location", "primary_topic", "topics", "keywords", "referenced_works",
            "is_retracted", "is_paratext", "cited_by_count", "language"} <= set(schema.names)


def test_the_cut_tool_takes_exactly_the_named_works_in_the_parts_order(tmp_path):
    """Decoys: the sample's own rows under other ids, plus the sample's rows; the tool keeps the named."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    tool = _tool()
    real = pq.read_table(str(PART))
    decoy = real.set_column(real.schema.get_field_index("id"), "id",
                            pa.array([f"https://openalex.org/W9{i}" for i in range(real.num_rows)]))
    source = tmp_path / "part.parquet"
    pq.write_table(pa.concat_tables([decoy.slice(0, 5), real, decoy.slice(5)]), str(source),
                   row_group_size=7)                                     # several row groups
    out = tmp_path / "out"
    hashes = tool.cut(source, out)
    assert [i.rsplit("/", 1)[1] for i, _ in hashes] == sorted(WORKS)
    assert dict((i.rsplit("/", 1)[1], h) for i, h in hashes) == WORKS
    got = pq.read_table(str(out / tool.RELATIVE_PATH))
    assert got.schema == real.schema and got.num_rows == real.num_rows
    assert got["id"].to_pylist() == real["id"].to_pylist()               # the part's own order
    assert set(tool.SELECTED) | {tool.RECOVERY} | set(tool.REJECTED) == tool.WANTED
    assert len(tool.WANTED) == 15 == len(WORKS)


# ------------------------------------------------------------------ the extractor (moved)
def test_the_selection_picks_what_the_pinned_policy_picks():
    from cytir.extract import openalex_academic as E
    raws = list(E.iter_selected(PART))
    assert [r["id"] for r in raws] == SELECTED                            # ascending id order
    assert _sha(_canonical(raws)) == SELECTED_RAW_SHA256
    # the four the selection refuses, and why each is refused
    refused = {"https://openalex.org/" + w for w in ("W153375954", "W127391412", "W6959737561", "W1060887079")}
    assert not (refused | set(LADDER_REFUSED)) & {r["id"] for r in raws}


def test_the_selection_keeps_what_the_n2_driver_kept(tmp_path):
    """Task 1250: the ADR-102 ladder moved here from the N2 driver (now deleted). The fixture holds what
    that driver's own query kept, run before it was deleted: on nineteen edge works (every band's
    boundary, core, the target-country carve-out) and on the sample part. The family selects the same."""
    import duckdb
    from cytir.extract import openalex_academic as E
    fx = json.loads((REPO / "tests" / "fixtures" / "openalex_ladder_vs_n2_driver.v1.json").read_text())
    con = duckdb.connect()
    con.execute("""CREATE TABLE w(id VARCHAR, doi VARCHAR, language VARCHAR, publication_year INT,
        cited_by_count INT, is_retracted BOOLEAN, is_paratext BOOLEAN, abstract_inverted_index VARCHAR,
        primary_topic STRUCT(field STRUCT(display_name VARCHAR), subfield STRUCT(display_name VARCHAR)),
        primary_location STRUCT(landing_page_url VARCHAR, source STRUCT(is_core BOOLEAN)),
        best_oa_location STRUCT(landing_page_url VARCHAR),
        authorships STRUCT(countries VARCHAR[])[])""")
    for c in fx["edge_cases"]:
        con.execute("INSERT INTO w VALUES (?,?,?,?,?,false,false,'{}',"
                    "{'field': {'display_name': 'Computer Science'}, 'subfield': {'display_name': 's'}},"
                    "{'landing_page_url': 'https://e.org/x', 'source': {'is_core': ?}},"
                    "{'landing_page_url': NULL}, [{'countries': ?}])",
                    [c["id"], "10.1/x", "en", c["publication_year"], c["cited_by_count"], c["is_core"],
                     c["countries"]])
    sql = E.selection_sql().replace("SELECT to_json(t) AS source_json", "SELECT t.id").replace(
        "read_parquet(?) AS t", "w AS t")
    assert sorted(r[0] for r in con.execute(sql).fetchall()) == fx["edge_driver_kept"]
    assert [r["id"] for r in E.iter_selected(PART)] == fx["sample_driver_kept_before_abstract_and_url_filters"]


def test_the_selection_does_not_depend_on_the_machines_time_zone():
    """`to_json` of a TIMESTAMPTZ follows the session zone, which DuckDB takes from the machine; CI's
    runner and a laptop must read the same bytes from the same part (found in the first CI run)."""
    import os
    code = ("import hashlib, sys; sys.path.insert(0, %r)\n"
            "from cytir.extract import openalex_academic as E\n"
            "from cytir.acquire.artifact_io import canonical_json\n"
            "print(hashlib.sha256(''.join(canonical_json(r) + '\\n' for r in E.iter_selected(%r)).encode())"
            ".hexdigest())" % (str(REPO), str(PART)))
    for zone in ("UTC", "Europe/Zurich", "America/New_York", "Asia/Kolkata"):
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                             env={**os.environ, "TZ": zone})
        assert res.returncode == 0, res.stderr[-1500:]
        assert res.stdout.strip() == SELECTED_RAW_SHA256, zone


def test_the_renderer_makes_the_old_scripts_markdown_and_recovers_the_one_it_refuses():
    from cytir.extract import openalex_academic as E
    rows, refused = [], []
    for raw in E.iter_selected(PART):
        try:
            rows.append(E.render_record(raw, relative_path=REL, parent_sha256=PART_SHA256))
        except ValueError as exc:
            refused.append((raw["id"], str(exc)))
    assert refused == []
    # the work the renderer refuses (no title once the heading is gone) is no longer selected: the
    # ladder drops it (2007, 7 citations); the renderer still refuses it when handed it directly
    import duckdb
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    (src,) = con.execute("SELECT to_json(t) FROM read_parquet(?) t WHERE id = ?", [str(PART), RECOVERY]).fetchone()
    with pytest.raises(ValueError, match="lacks title/full abstract"):
        E.render_record(json.loads(src), relative_path=REL, parent_sha256=PART_SHA256)
    assert [r["url"] for r in rows] == SELECTED
    assert _sha(_canonical(rows)) == RENDERED_SHA256
    first = rows[0]
    assert first["markdown"].startswith("# Improving the design and management of agile supply chains")
    assert first["source_metadata"]["metadata_license"] == "CC0-1.0"
    assert first["document_id"] == hashlib.sha256(
        b"openalex-academic-work\0https://openalex.org/W114559381").hexdigest()


def test_a_work_without_its_authors_is_the_same_text_minus_the_authors_section():
    from cytir.extract import openalex_academic as E
    for raw in E.iter_selected(PART):
        full = E.render_record(raw, relative_path=REL, parent_sha256=PART_SHA256)
        bare = E.render_record(raw, relative_path=REL, parent_sha256=PART_SHA256, with_authors=False)
        names = E.author_names(raw)
        assert names and "## Authors\n\n" + "; ".join(names) + "\n" in full["markdown"] + "\n"
        assert full["markdown"].replace("\n## Authors\n\n" + "; ".join(names) + "\n", "") == bare["markdown"]
        assert not any(n in bare["markdown"] for n in names)
        assert {k: v for k, v in full.items() if k != "markdown"} == {k: v for k, v in bare.items() if k != "markdown"}


def test_the_abstract_is_rebuilt_from_positions_and_a_broken_index_gives_nothing():
    from cytir.extract import openalex_academic as E
    assert E.reconstruct_abstract({"b": [1], "a": [0], "c": [2, 3]}) == "a b c c"
    assert E.reconstruct_abstract(json.dumps({"a": [0], "b": [1]})) == "a b"
    for broken in (None, [], {"a": "0"}, {"a": [0], "b": [0]}, {"a": [-1]}, {1: [0]}):
        assert E.reconstruct_abstract(broken) == ""


def test_the_part_streams_in_batches(monkeypatch):
    from cytir.extract import openalex_academic as E
    whole = [r["id"] for r in E.iter_selected(PART)]
    monkeypatch.setattr(E, "BATCH", 3)
    gen = E.iter_selected(PART)
    assert hasattr(gen, "__next__")                                    # a generator, not a list
    assert [r["id"] for r in gen] == whole


# ------------------------------------------------------------------ the part plan, on the old planner's real files
def test_the_whole_part_choice_on_the_real_frontier_is_the_pilot_part():
    from cytir.extract import openalex_academic as E
    raw = gzip.decompress((SAMPLES / "frontier.jsonl.gz").read_bytes())
    assert _sha(raw) == FRONTIER_SHA256
    rows = [json.loads(x) for x in raw.decode().splitlines()]
    assert len(rows) == 2216
    pilot = E.choose_whole_part(rows, 250)
    assert pilot["relative_path"] == REL and pilot["retained_legacy_records"] == 259
    assert pilot["expected_bytes"] == 152568007 and pilot["upstream_records"] == 159767
    assert E.choose_whole_part(rows, 259)["relative_path"] == REL
    with pytest.raises(ValueError, match="target must be positive"):
        E.choose_whole_part(rows, 0)
    with pytest.raises(ValueError, match="no source parts"):
        E.choose_whole_part([], 250)
    with pytest.raises(ValueError, match="invalid OpenAlex frontier row"):
        E.choose_whole_part([{"relative_path": "x.txt", "retained_legacy_records": 3}], 250)


def _plan(name):
    path = SAMPLES / "plans" / f"{name}.PLAN.json.gz"
    assert _sha(gzip.decompress(path.read_bytes())) == PLANS[name]
    return _gz_json(path)


def test_the_real_plans_give_their_parts_and_refuse_a_wrong_identity():
    from cytir.extract import openalex_academic as E
    one = _plan("min10000-excluding-pilot-v1")
    paths, n = E.planned_relative_paths(one, expected_source_run_sha256=SOURCE_RUN_SHA256, target_records=10000)
    assert paths == ["updated_date=2026-06-26/part_0036.parquet"] and n == 10132
    four = _plan("min100000-excluding-prior-v1")
    paths, n = E.planned_relative_paths(four, expected_source_run_sha256=SOURCE_RUN_SHA256, target_records=100000)
    assert len(paths) == 4 and n == 105882 and len(set(paths)) == 4
    for bad in ({"expected_source_run_sha256": "0" * 64, "target_records": 10000},
                {"expected_source_run_sha256": SOURCE_RUN_SHA256, "target_records": 9999}):
        with pytest.raises(ValueError, match="plan contract failed"):
            E.planned_relative_paths(one, **bad)
    drifted = json.loads(json.dumps(four))
    drifted["selected_parts"][0]["relative_path"] = drifted["selected_parts"][1]["relative_path"]
    with pytest.raises(ValueError, match="part identity failed"):
        E.planned_relative_paths(drifted, expected_source_run_sha256=SOURCE_RUN_SHA256, target_records=100000)


def test_the_partitioned_real_plan_splits_into_contiguous_slices_that_add_up():
    from cytir.extract import openalex_academic as E
    plan = _plan("min2500000-excluding-prior253-v1")
    kw = dict(expected_source_run_sha256=SOURCE_RUN_SHA256, target_records=2500000)
    all_paths, total, none = E.planned_partition_relative_paths(plan, partition_index=None, partition_count=None, **kw)
    assert len(all_paths) == 386 and total == 2504874 and none is None
    got, records = [], 0
    for i in range(4):
        paths, n, part = E.planned_partition_relative_paths(plan, partition_index=i, partition_count=4, **kw)
        assert part["partition_index"] == i and part["global_selected_records"] == total
        got += paths
        records += n
    assert got == all_paths and records == total
    for index, count, why in ((4, 4, "declared together"), (0, 3, "partition count drift"),
                              (None, 4, "declared together")):
        with pytest.raises(ValueError, match=why):
            E.planned_partition_relative_paths(plan, partition_index=index, partition_count=count, **kw)
    drift = json.loads(json.dumps(plan))
    drift["partitions"][1]["selected_records"] += 1
    with pytest.raises(ValueError, match="identity/conservation failed"):
        E.planned_partition_relative_paths(drift, partition_index=0, partition_count=4, **kw)


# ------------------------------------------------------------------ the package is self-contained
def _imports(path: pathlib.Path, top_only: bool = False) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in (tree.body if top_only else ast.walk(tree)):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def test_the_family_and_the_extractor_import_no_private_tree():
    for path in (REPO / "cytir" / "extract" / "openalex_academic.py",
                 REPO / "cytir" / "sources" / "families" / "openalex_academic_dump.py"):
        assert not {"pipeline", "tasks"} & _imports(path), path       # the cut ships neither
        assert not {"duckdb", "pyarrow"} & _imports(path, top_only=True), path   # lazy imports only


def test_importing_the_family_loads_neither_duckdb_nor_pyarrow():
    code = ("import sys; import cytir.extract.openalex_academic, cytir.sources.families.openalex_academic_dump;"
            "print(sorted(m for m in ('duckdb', 'pyarrow') if m in sys.modules))")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=120)
    assert res.returncode == 0, res.stderr[-1500:]
    assert res.stdout.strip().splitlines()[-1] == "[]"


# ------------------------------------------------------------------ the door
def test_two_runs_give_the_same_bytes_and_the_stage_names_no_host_or_path(tmp_path, word_only_chunking_allowed):
    _make(tmp_path / "a")
    _make(tmp_path / "b", work_dir=tmp_path / "work")
    assert _tree(tmp_path / "a") == _tree(tmp_path / "b")
    meta = (tmp_path / "a" / "PACK.json").read_text()
    assert str(tmp_path) not in meta and str(REPO) not in meta and "/tmp/" not in meta
    doc = json.loads(meta)
    assert doc["family_id"] == FAMILY and doc["rows"] == 6 and "-local-" in doc["pack_id"]
    for p in (tmp_path / "a").rglob("*.gz"):
        assert str(REPO).encode() not in gzip.decompress(p.read_bytes()), p     # no absolute path in a row
    manifest = json.loads((tmp_path / "work" / "pack" / "manifest.json").read_text())
    assert manifest["terminals"]["units_admitted"] == 1 and manifest["terminals"]["records"] == 6
    assert manifest["discover"]["per_part"] == {REL: {"selected": 6, "rendered": 6, "recovery_required": 0}}
    assert manifest["discover"]["chunker"]["policy_sha256"] == (
        "7dcb54c497e1855ad91ddafe80ff34b1da14f3d23b3af26dc7c7b53ab70ce6bf")
    index = [json.loads(x) for x in (tmp_path / "work" / "l0" / "index.jsonl").read_text().splitlines()]
    assert len(index) == 1 and index[0]["unit"] == REL
    assert index[0]["bytes"] < 1000                                    # L0 is the pointer, not the part


def test_l0_is_a_pointer_to_the_part_with_its_size_and_sha256(tmp_path, word_only_chunking_allowed):
    _make(tmp_path / "s", work_dir=tmp_path / "work")
    index = json.loads((tmp_path / "work" / "l0" / "index.jsonl").read_text().splitlines()[0])
    blobs = [p for p in (tmp_path / "work" / "l0").rglob("*") if p.is_file() and p.name != "index.jsonl"]
    sealed = next(p for p in blobs if _sha(p.read_bytes()) == index["sha256"])
    pointer = json.loads(sealed.read_bytes())
    assert pointer == {"schema": "aw.openalex-part-pointer.v1", "relative_path": REL,
                       "bytes": PART.stat().st_size, "sha256": PART_SHA256, "metadata_license": "CC0-1.0",
                       "snapshot": "https://docs.openalex.org"}


def test_the_committed_fixture_stage_is_what_the_door_makes(tmp_path, word_only_chunking_allowed):
    _make(tmp_path / "stage")
    assert _tree(tmp_path / "stage") == _tree(COMMITTED), (
        "regenerate: cyt source run openalex_academic_dump --from data/samples/openalex-academic "
        "--out data/index-fixture/openalex_academic_dump --today 2026-10-03")


def _surface(rows, key) -> str:
    return _sha(_canonical([{"url": r["url"], "ordinal": int(r["anchor"].rsplit("-", 1)[1]), "text": r[key]}
                            for r in rows]))


def test_the_embedded_surfaces_are_the_served_rows_surfaces_authors_included(tmp_path, word_only_chunking_allowed):
    """The embedding surface is a contract (founder ruling 7 allows names in embeddings): the dense and
    the sparse input of every chunk equal what the old stage script makes from the old renderer's
    Markdown. The pins come from the old script (`test_openalex_family_identity.py` in the old tree
    re-derives them there)."""
    from cytir.extract import openalex_academic as E
    _make(tmp_path / "s")
    rows = _rows(tmp_path / "s")
    assert _surface(rows, "dense_text") == DENSE_SURFACE_SHA256
    assert _surface(rows, "sparse_text") == SPARSE_SURFACE_SHA256 == _surface(rows, "text")
    authors = {w["id"]: E.author_names(w) for w in E.iter_selected(PART)}
    for r in rows:                                     # the old Markdown has an Authors section
        assert "Authors" in r["dense_text"] and all(n in r["dense_text"] for n in authors[r["url"]])


def test_the_rows_carry_licence_url_source_type_and_keep_the_authors_out_of_the_payload(tmp_path, word_only_chunking_allowed):
    from cytir.extract import openalex_academic as E
    _make(tmp_path / "s")
    rows = _rows(tmp_path / "s")
    assert [r["url"] for r in rows] == SELECTED
    assert len({r["point_id"] for r in rows}) == len(rows)
    authors = {w["id"]: E.author_names(w) for w in E.iter_selected(PART)}
    embedded = {"text", "dense_text", "sparse_text"}      # build input; stripped at index build
    for r in rows:
        assert r["url"].startswith("https://openalex.org/W") and r["domain"] == "openalex.org"
        assert r["license"] == "CC0-1.0" and r["lang"] in ("en", "fr", "de", "it")
        assert r["source_type"] == "reference" and r["tier"] == "C" and r["trust_tier"] == 0.70
        assert r["anchor"] == "chunk-0" and r["last_seen"] == TODAY and r["snippet"]
        assert r["text"].startswith(r["title"]) and "OpenAlex work: " + r["url"] in r["text"]
        m = r["_master"]
        names = authors[r["url"]]
        assert names and m["authors"] == names                         # the authors, master-only (ADR-039)
        assert m["source_part_sha256"] == PART_SHA256 and m["license_version"] == "1.0"
        shipped = json.dumps({k: v for k, v in r.items() if k not in embedded | {"_master"}},
                             ensure_ascii=False)
        assert not any(n in shipped for n in names), "an author name in the snippet or a payload field"
        assert not {"attribution", "owner", "authors", "license_version"} & set(r)
    assert rows[2]["_master"]["attribution"].startswith("arXiv")


def test_the_stage_is_one_the_index_build_reads_and_its_consent_derives(tmp_path, word_only_chunking_allowed):
    """`cyt index build --pack` checks the stage, its shards and vectors before it connects to anything; the
    consent stamp derives both flags by ADR-115 r1-openalex (the CC0 dedication is the instrument). The
    build itself runs in `tests/test_aw_index_serve_gate.py`."""
    sys.path.insert(0, str(REPO / "tests"))
    import standin_embedder as SE
    from cytir.consent import derive, stamp
    from cytir.index import build
    dst = tmp_path / "stage"
    _make(dst)
    SE.write_vectors(dst)
    pack = build.load_stage_pack(str(dst))
    build.check_vectors(pack)
    assert pack.family == FAMILY and pack.pack_id.startswith("openalex-academic-dump-v1-local-")
    cache = tmp_path / "consent-cache"
    cache.mkdir()
    ctx = stamp.load_context(as_of="2026-10-03T00:00:00Z", cache_dir=str(cache))
    got = {(d.klass, d.rule, d.search_ok, d.train_ok) for d in
           (derive.derive_consent_flags(r, ctx, family_id=FAMILY) for r in _rows(dst))}
    assert got == {("A", "r1-openalex", True, True)}


def test_the_family_never_asks_for_consent_and_runs_offline(tmp_path, word_only_chunking_allowed):
    from cytir.sources import run_family
    out = tmp_path / "o"
    run_family.run_family(
        FAMILY, out_dir=out, today=TODAY,
        consent=lambda h, d: pytest.fail(f"consent asked for {h}"),
        options={"root": str(SAMPLES)}, log=lambda m: None)
    assert json.loads((out / "pack" / "manifest.json").read_text())["terminals"]["units_admitted"] == 1


# ------------------------------------------------------------------ plan option, ledger and refusals
def _plan_file(tmp_path, paths):
    """A selection plan of the real schema naming `paths` (the one-part plan, re-pointed at the sample)."""
    plan = _plan("min10000-excluding-pilot-v1")
    plan["selected_part"]["relative_path"] = paths[0]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path


def test_a_plan_restricts_the_run_to_the_parts_it_names(tmp_path, word_only_chunking_allowed):
    root = tmp_path / "in"
    (root / "updated_date=2026-02-09").mkdir(parents=True)
    (root / REL).write_bytes(PART.read_bytes())
    other = root / "updated_date=2026-03-01" / "part_0001.parquet"
    other.parent.mkdir()
    other.write_bytes(PART.read_bytes())
    # without a plan both parts are units; the second holds the same works, so the door quarantines it
    # (a work in two parts would be two points with one id)
    _make(tmp_path / "s0", from_dir=root, work_dir=tmp_path / "w0")
    t = json.loads((tmp_path / "w0" / "pack" / "manifest.json").read_text())["terminals"]
    assert t["units_in"] == 2 and t["units_admitted"] == 1 and t["quarantined_reasons"] == {
        "duplicate_point_id": 1}
    plan = _plan_file(tmp_path, [REL])
    opts = {"plan": str(plan), "source_run_sha256": SOURCE_RUN_SHA256, "target_records": "10000"}
    from cytir.sources import run_family
    out = tmp_path / "o1"
    run_family.run_family(FAMILY, out_dir=out, today=TODAY, skip_consent=True,
                          options={"root": str(root), **opts}, log=lambda m: None)
    manifest = json.loads((out / "pack" / "manifest.json").read_text())
    assert manifest["terminals"]["units_in"] == 1 and manifest["discover"]["plan_parts"] == 1
    with pytest.raises(Exception, match="not under"):
        run_family.run_family(FAMILY, out_dir=tmp_path / "o2", today=TODAY, skip_consent=True,
                              options={"root": str(root), **opts,
                                       "plan": str(_plan_file(tmp_path, ["updated_date=2020-01-01/part_0000.parquet"]))},
                              log=lambda m: None)
    with pytest.raises(Exception, match="source_run_sha256"):
        run_family.run_family(FAMILY, out_dir=tmp_path / "o3", today=TODAY, skip_consent=True,
                              options={"root": str(root), "plan": str(plan)}, log=lambda m: None)


def _only(tmp_path, ids):
    """A snapshot directory whose part holds only the sample's rows with these ids."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    table = pq.read_table(str(PART))
    keep = table.filter(pc.is_in(table["id"], value_set=__import__("pyarrow").array(ids)))
    root = tmp_path / "root"
    (root / "updated_date=2026-02-09").mkdir(parents=True, exist_ok=True)
    pq.write_table(keep, str(root / REL))
    return root


def test_a_part_with_no_selectable_work_is_excluded_with_its_reason_and_the_ledger_balances(tmp_path, word_only_chunking_allowed):
    rejected = ["https://openalex.org/W153375954", "https://openalex.org/W127391412"]
    root = tmp_path / "root"
    (root / "updated_date=2026-02-09").mkdir(parents=True)
    import shutil
    shutil.copy(PART, root / REL)
    quiet = _only(tmp_path / "q", rejected) / REL
    dest = root / "updated_date=2026-03-01" / "part_0001.parquet"
    dest.parent.mkdir()
    shutil.copy(quiet, dest)
    _make(tmp_path / "s", from_dir=root, work_dir=tmp_path / "work")
    t = json.loads((tmp_path / "work" / "pack" / "manifest.json").read_text())["terminals"]
    assert t["units_in"] == 2 and t["units_admitted"] == 1 and t["units_excluded"] == 1
    assert t["excluded_reasons"] == {"no_work_selected_by_the_policy": 1}
    assert t["units_quarantined"] == 0 and t["records"] == 6


def test_a_run_that_admits_no_passage_refuses_and_writes_no_stage(tmp_path):
    """The smoke test's other half (task 440/634): a snapshot whose works the policy all refuses is a
    wrong input or a broken parser, never an empty source."""
    root = _only(tmp_path, ["https://openalex.org/W153375954", "https://openalex.org/W6959737561"])
    with pytest.raises(Exception, match="0 records admitted"):
        _make(tmp_path / "s", from_dir=root)
    assert not (tmp_path / "s").exists()


def test_an_empty_walk_and_a_missing_root_refuse(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(Exception, match="no .parquet part|empty walk|wrong root"):
        _make(tmp_path / "s1", from_dir=tmp_path / "empty")
    with pytest.raises(Exception, match="not a directory|does not exist|not a dir"):
        _make(tmp_path / "s2", from_dir=tmp_path / "nowhere")
    assert not (tmp_path / "s1").exists() and not (tmp_path / "s2").exists()
