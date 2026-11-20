"""`service_public_fiche` on the family door, on six REAL fiches (slice 7 step 3, task 696).

`cyt source run service_public_fiche --from data/samples/service-public-fiche --out STAGE`: no
network, no Qdrant, no model. The committed sample is six whole fiche files cut from the
publisher's archive (`data/samples/service-public-fiche/SOURCE.md` records the archive's sha256 and
the cut). The door's stage is complete for `cyt index build`, and under consent rule r1 of ADR-115
(dump licence, Etalab-2.0; ruling 26 item 22) the build accepts its rows. The committed stage
`data/index-fixture/service_public_fiche/` is the door's own output (a test below), and the family is
in the `index-serve-gate` chain (slice 7 step 7): its three golds are `ci-sp-1..3`.

Pins: the sample is the cut the provenance file names (sha256 of each file); a run is
byte-deterministic; the rows carry the licence, the canonical URL, the declared source type and no
bookkeeping of ours in their text (the declared, head-free surface); a file that is not a fiche is
excluded with its reason and the ledger still balances; an empty walk refuses; the renderer is the
restage tool's own (same body, the head optional).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import re
import shutil

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SAMPLE = REPO / "data" / "samples" / "service-public-fiche"
COMMITTED = REPO / "data" / "index-fixture" / "service_public_fiche"
TODAY = "2026-10-03"
FAMILY = "service_public_fiche"
CUT = {
    "F1613.xml": "edf1adb46f3e2afe60598518873c589052c7972f7ec55625da41be2e897475db",
    "F2725.xml": "9d6f8a7c568d7c9d15b5845e3b705d1e978debf9f0133955e9e6a8d5dfd68924",
    "F13476.xml": "35c89fd73300a65f2777465c72ac17b67383b60e4c374482f1f49b7b97c79cfb",
    "F24415.xml": "4a3ab0dfb9cb34852b444597f72c12e6ca3f4fa297b76e301d711ca3c2c7bd8c",
    "F31417.xml": "355564dccc834193529e0b46afd5f34f8f7b5533c4ce0bea57ce2c4720ba6844",
    "F36491.xml": "c8559967efa61df66311e37a5a4d4d37ec8fe65d45b5b48f3780e1329f292200",
}


def _tree(root: pathlib.Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _rows(stage: pathlib.Path) -> list[dict]:
    with gzip.open(stage / "chunks" / "shard-0000.texts.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(x) for x in fh]


def _make(out: pathlib.Path, from_dir=SAMPLE, **kw):
    from cytir.sources import stage_out
    return stage_out.run_to_stage(FAMILY, out=out, today=TODAY, from_dir=from_dir,
                                  log=lambda m: None, **kw)


def test_the_sample_is_the_cut_its_provenance_file_names():
    assert sorted(p.name for p in SAMPLE.glob("*.xml")) == sorted(CUT)
    for name, digest in CUT.items():
        assert hashlib.sha256((SAMPLE / name).read_bytes()).hexdigest() == digest, name
    note = (SAMPLE / "SOURCE.md").read_text(encoding="utf-8")
    assert all(d in note for d in CUT.values())
    assert "31fd718f4f4fa2fde1a67afce584a1da04d60d6d17705c26c33fb9cb1fe18afc" in note
    assert "Etalab-2.0" in note and "Licence Ouverte" in note
    assert sum(p.stat().st_size for p in SAMPLE.glob("*.xml")) < 150_000


def test_the_cut_tool_makes_the_same_files_from_an_archive(tmp_path):
    import importlib.util
    import zipfile
    spec = importlib.util.spec_from_file_location(
        "cut_service_public_fiche", REPO / "tests" / "fixture_tools" / "cut_service_public_fiche.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name in tool.MEMBERS:
            zf.writestr(name, (SAMPLE / name).read_bytes())
        zf.writestr("F999.xml", b"<other/>")          # a member the cut does not take
    rows = tool.cut(archive, tmp_path / "out")
    assert [r[0] for r in rows] == list(tool.MEMBERS) == list(CUT)
    assert {r[0]: r[2] for r in rows} == CUT
    assert not (tmp_path / "out" / "F999.xml").exists()


def test_two_runs_give_the_same_bytes_and_the_stage_names_no_host_or_path(tmp_path):
    _make(tmp_path / "a")
    _make(tmp_path / "b", work_dir=tmp_path / "work")
    assert _tree(tmp_path / "a") == _tree(tmp_path / "b")
    meta = (tmp_path / "a" / "PACK.json").read_text()
    assert str(tmp_path) not in meta and str(REPO) not in meta and "/tmp/" not in meta
    doc = json.loads(meta)
    assert doc["family_id"] == FAMILY and doc["rows"] == 6 and "-local-" in doc["pack_id"]
    # the work directory keeps the six sealed L0 files and a balanced pack
    manifest = json.loads((tmp_path / "work" / "pack" / "manifest.json").read_text())
    assert manifest["terminals"]["units_admitted"] == 6
    assert len((tmp_path / "work" / "l0" / "index.jsonl").read_text().splitlines()) == 6


def test_the_committed_fixture_stage_is_what_the_door_makes(tmp_path):
    _make(tmp_path / "stage")
    assert _tree(tmp_path / "stage") == _tree(COMMITTED), (
        "regenerate: cyt source run service_public_fiche --from data/samples/service-public-fiche "
        "--out data/index-fixture/service_public_fiche --today 2026-10-03")


def test_the_stage_does_not_depend_on_where_the_sample_or_the_checkout_lives(tmp_path):
    """The row stamp hashes the chunk policy's REPO-RELATIVE path. A stamp over the absolute path
    gave another prov_id, pack id and chunk hash in every checkout (the first CI run of this slice)."""
    elsewhere = tmp_path / "some" / "other" / "place"
    shutil.copytree(SAMPLE, elsewhere)
    _make(tmp_path / "a", from_dir=elsewhere)
    assert _tree(tmp_path / "a") == _tree(COMMITTED)
    for r in _rows(tmp_path / "a"):
        assert set(r["stamps"]) == {"chunk", "prov_id"}


def test_the_committed_stage_names_no_path_of_ours_and_no_person():
    for path in sorted(COMMITTED.rglob("*")):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        text = gzip.decompress(raw).decode("utf-8") if path.suffix == ".gz" else raw.decode("utf-8")
        assert "/home/" not in text and "/tmp/" not in text and "worktree" not in text, path.name
    rows = [json.loads(x) for x in gzip.open(
        COMMITTED / "chunks" / "shard-0000.texts.jsonl.gz", "rt", encoding="utf-8")]
    assert len(rows) == 6
    for r in rows:      # an institution is the only creator of a fiche: no author, no e-mail address
        assert "@" not in json.dumps(r) and not {"author", "authors", "owner"} & set(r["_master"])


def test_the_rows_carry_licence_url_source_type_and_a_head_free_text(tmp_path):
    _make(tmp_path / "s")
    rows = _rows(tmp_path / "s")
    assert sorted(r["url"].rsplit("/", 1)[1] for r in rows) == sorted(n[:-4] for n in CUT)
    assert len({r["point_id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["url"].startswith("https://www.service-public.gouv.fr/particuliers/vosdroits/F")
        assert r["license"] == "Etalab-2.0" and r["lang"] == "fr"
        assert r["source_type"] == "official" and r["trust_tier"] == 1.0 and r["tier"] == "A"
        assert r["domain"] == "www.service-public.gouv.fr"
        for field in ("text", "dense_text", "sparse_text"):
            assert not re.search(r"(?m)^aw_[a-z0-9_]+:|^---$", r[field]), (field, r["url"])
        assert r["text"].startswith(r["title"]) and r["snippet"] and r["snippet"] != r["title"]
        # the owner and the licence version are master-only: kept in the row, removed from the dump
        assert r["_master"]["license_version"] == "2.0" and "DILA" in r["_master"]["attribution"]
        assert r["_master"]["source_sha256"] == CUT[r["_master"]["fiche_id"] + ".xml"]


def test_the_stage_is_one_the_index_build_reads(tmp_path):
    """`cyt index build --pack` checks the stage dir, its shards and its vectors before it connects
    to anything. Stand-in vectors (as the CI job uses) make the stage complete; the checks pass."""
    import sys
    sys.path.insert(0, str(REPO / "tests"))
    import standin_embedder as SE
    from cytir.index import build
    dst = tmp_path / "stage"
    _make(dst)
    SE.write_vectors(dst)
    pack = build.load_stage_pack(str(dst))
    build.check_vectors(pack)
    assert pack.family == FAMILY and pack.pack_id.startswith("service-public-fiche-v1-local-")


def test_a_file_that_is_not_a_fiche_is_excluded_and_the_ledger_balances(tmp_path):
    root = tmp_path / "in"
    shutil.copytree(SAMPLE, root)
    (root / "F77.xml").write_bytes(b"<Publication ID='F77'/>")      # no title, URL or text
    (root / "F78.xml").write_bytes(b"<broken")                      # not even XML
    (root / "R1.xml").write_bytes(b"<x/>")                          # a member that is not a fiche file
    _make(tmp_path / "s", from_dir=root, work_dir=tmp_path / "work")
    manifest = json.loads((tmp_path / "work" / "pack" / "manifest.json").read_text())
    t = manifest["terminals"]
    assert t["units_in"] == 8 and t["units_admitted"] == 6 and t["units_excluded"] == 2
    assert t["excluded_reasons"] == {
        "not_a_fiche_or_missing_title_url_identifier_date_or_content": 2}
    assert t["units_quarantined"] == 0


def test_an_empty_walk_refuses_and_writes_nothing(tmp_path):
    from cytir.sources import stage_out
    (tmp_path / "empty").mkdir()
    with pytest.raises(Exception, match="no F<number>.xml|empty walk|wrong root"):
        _make(tmp_path / "s", from_dir=tmp_path / "empty")
    assert not (tmp_path / "s").exists()
    assert stage_out  # imported: the door under test


def test_the_family_never_asks_for_consent_and_runs_offline(tmp_path):
    from cytir.sources import run_family
    out = tmp_path / "o"
    run_family.run_family(
        FAMILY, out_dir=out, today=TODAY,
        consent=lambda h, d: pytest.fail(f"consent asked for {h}"),
        options={"root": str(SAMPLE)}, log=lambda m: None)
    manifest = json.loads((out / "pack" / "manifest.json").read_text())
    assert manifest["terminals"]["units_admitted"] == 6
    assert manifest["discover"]["chunker"]["policy_sha256"] == (
        "7dcb54c497e1855ad91ddafe80ff34b1da14f3d23b3af26dc7c7b53ab70ce6bf")


def test_the_head_is_optional_and_the_body_is_the_same():
    from cytir.extract import service_public_fiche as E
    xml = (SAMPLE / "F2725.xml").read_bytes()
    with_head = E.render(xml, "a" * 64, "F2725.xml")
    bare = E.render(xml, "", "F2725.xml", provenance_head=False)
    assert with_head["markdown"].startswith("---\naw_fiche_id: \"F2725\"")
    assert bare["markdown"].startswith("# Faut-il assurer un bateau de plaisance")
    assert with_head["markdown"].endswith(bare["markdown"])          # same title and body
    for key in ("document_id", "url", "title", "fiche_id", "modified_date", "source_sha256"):
        assert with_head[key] == bare[key]


def test_the_build_accepts_the_rows_under_rule_r1_the_dump_licence(tmp_path):
    """FLIPPED by POLICY v3.2 U1 (Tim 2026-10-05, ruling 26 item 22). The source itself publishes the
    dump under Licence Ouverte / Etalab-2.0, so the licence is the consent instrument: the family takes
    the dump rule r1 of ADR-115 (`cytir.consent.derive.INSTRUMENT_FAMILIES`), basis `dump-licence`, both
    flags true. A robots probe of the host is not the gate. Before the ruling the class was D and
    `cyt index build` stopped at the consent stamp with `ConsentStampRefused`."""
    import sys
    sys.path.insert(0, str(REPO / "tests"))
    import standin_embedder as SE
    from cytir.consent import derive, stamp
    from cytir.index import build
    dst = tmp_path / "s"
    _make(dst)
    SE.write_vectors(dst)
    pack = build.load_stage_pack(str(dst))      # `cyt index build --pack` reads the stage ...
    build.check_vectors(pack)                   # ... and its vectors before it connects to anything
    assert pack.family == FAMILY
    cache = tmp_path / "consent-cache"
    cache.mkdir()
    ctx = stamp.load_context(as_of="2026-10-03T00:00:00Z", cache_dir=str(cache))
    rows = _rows(dst)
    assert rows
    got = {(d.klass, d.rule, d.search_ok, d.train_ok, d.consent_basis) for d in
           (derive.derive_consent_flags(r, ctx, family_id=FAMILY) for r in rows)}
    assert got == {("A", "r1-service-public", True, True, "dump-licence")}


def test_a_row_without_the_etalab_licence_is_not_stamped_by_r1(tmp_path):
    """The licence the ROW carries is the instrument; a row that lost it falls to class D, never to a default."""
    from cytir.consent import derive, stamp
    _make(tmp_path / "s")
    cache = tmp_path / "consent-cache"
    cache.mkdir()
    ctx = stamp.load_context(as_of="2026-10-03T00:00:00Z", cache_dir=str(cache))
    row = dict(_rows(tmp_path / "s")[0], license="")
    d = derive.derive_consent_flags(row, ctx, family_id=FAMILY)
    assert d.klass == "D" and "instrument-licence-absent" in d.rule
