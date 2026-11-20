"""`dila_kali` + `dila_jorf` SourceFamilies (task 696): real KALI articles and JORF texts from the
DILA dump on the build box (`Freemium_kali_global_20250713`, JORF increments 2025), run through
`cyt source run`'s own door.

Pins: one record per admitted file; every file reaches one terminal with a reason; the record
carries the contract's licence and the canonical legifrance URL; the legacy CLIs share the parser;
the per-convention cap; a wrong root and an all-empty run refuse; staged ids reproduce.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

FIX = HERE / "fixtures"
TODAY = "2026-10-03"
KALI_OK = "KALIARTI000005833790.xml"
KALI_SHORT = "KALIARTI000005833715.xml"
JORF_FILES = ("JORFTEXT000051885835.xml", "JORFTEXT000051885941.xml", "JORFTEXT000051886873.xml")


def _dump(tmp_path, family, names):
    root = tmp_path / "dump"
    d = root / "global" / "TEXT" / "00"
    d.mkdir(parents=True)
    for n in names:
        shutil.copy(FIX / family / n, d / n)
    return root


def _run(family, dump, out, **opts):
    from cytir.sources import run_family
    return run_family.run_family(
        family, out_dir=out, today=TODAY, consent=lambda h, d: pytest.fail(f"consent for {h}"),
        options={"root": str(dump), **opts}, log=lambda m: None)


def _records(out):
    return [json.loads(line) for line in
            (out / "pack" / "records.jsonl").read_text(encoding="utf-8").splitlines()]


# ------------------------------------------------------------------------------------ KALI

def test_kali_admits_the_real_article_and_excludes_the_short_one_with_a_reason(tmp_path):
    dump = _dump(tmp_path, "dila_kali", [KALI_OK, KALI_SHORT])
    man = _run("dila_kali", dump, tmp_path / "o")
    t = man["terminals"]
    assert t["units_in"] == 2 and t["units_admitted"] == 1 and t["units_excluded"] == 1
    assert sum(t["excluded_reasons"].values()) == 1
    (row,) = _records(tmp_path / "o")
    assert row["provenance"] == "kali_article" and row["license"] == "Etalab-2.0"
    assert row["url"] == "https://www.legifrance.gouv.fr/conv_coll/article/KALIARTI000005833790"
    assert row["domain"] == "legifrance.gouv.fr" and row["lang"] == "fr"
    assert 260 <= len(row["text"]) <= 4100
    assert man["consent"]["verdicts"] == [] and man["route"] == "source_owned_bulk_dump"


def test_kali_per_convention_cap_is_a_terminal_not_a_silent_drop(tmp_path, monkeypatch):
    from cytir.sources.families import dila_kali
    monkeypatch.setattr(dila_kali, "MAX_ARTICLES_PER_CONV", 1)
    root = tmp_path / "dump" / "a"
    root.mkdir(parents=True)
    body = (FIX / "dila_kali" / KALI_OK).read_bytes()
    (root / "KALIARTI000000000001.xml").write_bytes(body)
    (root / "KALIARTI000000000002.xml").write_bytes(
        body.replace(b"KALIARTI000005833790", b"KALIARTI000000000002"))
    man = _run("dila_kali", tmp_path / "dump", tmp_path / "o")
    t = man["terminals"]
    assert t["units_admitted"] == 1 and t["units_excluded"] == 1
    assert list(t["excluded_reasons"]) == ["convention_cap_1"]


def test_kali_legacy_cli_parser_is_the_shared_one():
    from cytir.extract import dila_kali as legacy
    from cytir.extract.dila_kali import parse_article_bytes
    rec, key, why = parse_article_bytes((FIX / "dila_kali" / KALI_OK).read_bytes())
    assert why == "kept"
    assert legacy.parse_article(str(FIX / "dila_kali" / KALI_OK)) == (rec, key)
    assert legacy.parse_article(str(FIX / "dila_kali" / KALI_SHORT)) is None


def test_kali_legacy_cli_refuses_an_over_window_article(tmp_path):
    from cytir.extract import dila_kali as legacy
    long_xml = HERE / "fixtures" / "dila_kali_long" / "KALIARTI000052502103.xml"
    rec, _key = legacy.parse_article(str(long_xml))
    assert len(rec["text"]) > legacy.ROW_CHARS
    line = legacy.over_window_refusal(rec)
    assert line and "dila_kali" in line and "family door" in line
    assert legacy.over_window_refusal({"url": "u", "text": "x" * 100}) is None
    root = tmp_path / "d"
    root.mkdir()
    (root / long_xml.name).write_bytes(long_xml.read_bytes())
    sys.argv = ["dila_kali", "--root", str(root), "--out", str(tmp_path / "o.jsonl")]
    with pytest.raises(SystemExit) as ex:
        legacy.main()
    assert str(ex.value).startswith("REFUSED:")


# ------------------------------------------------------------------------------------ JORF

def test_jorf_floors_are_unchanged_and_every_file_has_a_terminal(tmp_path):
    from cytir.extract.dila_jorf import parse_text_bytes
    dump = _dump(tmp_path, "dila_jorf", JORF_FILES)
    man = _run("dila_jorf", dump, tmp_path / "o")
    t = man["terminals"]
    kept = [f for f in JORF_FILES
            if parse_text_bytes((FIX / "dila_jorf" / f).read_bytes())[0] is not None]
    assert 1 <= len(kept) < len(JORF_FILES)            # the floors really drop a real file
    assert t["units_in"] == 3 and t["units_admitted"] == len(kept)
    assert t["units_excluded"] == 3 - len(kept) and sum(t["excluded_reasons"].values()) == 3 - len(kept)
    rows = _records(tmp_path / "o")
    assert {r["provenance"] for r in rows} == {"jorf_recent"}
    assert {r["license"] for r in rows} == {"Etalab-2.0"}
    assert all(r["url"].startswith("https://www.legifrance.gouv.fr/jorf/id/JORFTEXT") for r in rows)
    assert man["discover"]["min_year"] == 2023


def test_jorf_a_run_that_admits_nothing_refuses(tmp_path):
    from cytir.sources import run_family
    dump = _dump(tmp_path, "dila_jorf", JORF_FILES)
    with pytest.raises(run_family.RunError, match="0 records admitted"):
        _run("dila_jorf", dump, tmp_path / "o", min_year="2999")


# ------------------------------------------------------------------------------------ both

@pytest.mark.parametrize("family", ["dila_kali", "dila_jorf"])
def test_wrong_root_is_an_error_not_an_empty_source(tmp_path, family):
    from cytir.sources import run_family
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="wrong root"):
        run_family.run_family(family, out_dir=tmp_path / "o", today=TODAY,
                              consent=lambda h, d: {}, options={"root": str(empty)},
                              log=lambda m: None)
    with pytest.raises(ValueError, match="needs --opt root"):
        run_family.run_family(family, out_dir=tmp_path / "o2", today=TODAY,
                              consent=lambda h, d: {}, options={}, log=lambda m: None)


@pytest.mark.parametrize("family,names", [("dila_kali", [KALI_OK]),
                                          ("dila_jorf", JORF_FILES[:2])])
def test_the_pack_stages_and_its_ids_reproduce_under_the_family_anchor(tmp_path, family, names):
    from cytir.index import compose
    from cytir.sources import family_stage as FS
    dump = _dump(tmp_path, family, list(names))
    try:
        _run(family, dump, tmp_path / "o")
    except Exception as exc:       # a JORF pair where both rows are floored out is a fixture error
        pytest.fail(f"run refused: {exc}")
    FS.stage_family_pack(tmp_path / "o" / "pack", corpus_root=tmp_path / "c", family_id=family,
                         log=lambda m: None)
    (_, d), = FS.list_stages(tmp_path / "c", family)
    ident = compose.verify_identity(str(d / "chunks"),
                                    anchor_fn=compose.anchor_fn_for_family(family))
    assert ident["checked"] == ident["matched"] >= 1


# ------------------------------------------------------------------ increments (versions)

def _tree(tmp_path, family, body, relpaths):
    root = tmp_path / "dump"
    for rel in relpaths:
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(body)
    return root


def test_the_newest_increment_wins_and_the_older_copies_are_excluded_with_a_reason(tmp_path):
    body = (FIX / "dila_kali" / KALI_OK).read_bytes()
    root = _tree(tmp_path, "dila_kali", body, [
        f"kali/global/x/{KALI_OK}", f"20250801-000000/kali/global/x/{KALI_OK}",
        f"20260101-000000/kali/global/x/{KALI_OK}"])
    man = _run("dila_kali", root, tmp_path / "o")
    t = man["terminals"]
    assert t["units_in"] == 3 and t["units_admitted"] == 1 and t["units_excluded"] == 2
    assert t["excluded_reasons"] == {"superseded_by_newer_increment": 2}
    (row,) = _records(tmp_path / "o")
    assert row["url"].endswith("KALIARTI000005833790")
    kept = [u["unit"] for u in man["units"] if u["terminal"] == "admitted"]
    assert kept == [f"20260101-000000/kali/global/x/{KALI_OK}"]
    assert man["discover"]["records"] == 1


def test_skip_ids_under_leaves_records_an_earlier_run_staged_to_that_run(tmp_path):
    body = (FIX / "dila_kali" / KALI_OK).read_bytes()
    prior = tmp_path / "prior"
    (prior / "kali/global").mkdir(parents=True)
    (prior / "kali/global" / KALI_OK).write_bytes(body)
    new = body.replace(b"KALIARTI000005833790", b"KALIARTI000005833791")
    root = tmp_path / "dump"
    (root / "20260101-000000/kali/global").mkdir(parents=True)
    (root / "20260101-000000/kali/global" / KALI_OK).write_bytes(body)
    (root / "20260101-000000/kali/global" / "KALIARTI000005833791.xml").write_bytes(new)
    man = _run("dila_kali", root, tmp_path / "o", skip_ids_under=str(prior))
    t = man["terminals"]
    assert t["units_admitted"] == 1 and t["excluded_reasons"] == {"already_in_prior_run": 1}
    assert man["discover"]["known_ids_in_prior_run"] == 1
    (row,) = _records(tmp_path / "o")
    assert row["url"].endswith("KALIARTI000005833791")


def test_juri_increments_only_run_adds_new_decisions_and_never_a_second_copy(tmp_path):
    cass = sorted((HERE / "fixtures" / "dila_juri").glob("JURITEXT*.xml"))
    prior = tmp_path / "prior" / "cass" / "global"
    prior.mkdir(parents=True)
    shutil.copy(cass[0], prior / cass[0].name)
    root = tmp_path / "dump" / "20260101-000000" / "cass" / "global"
    root.mkdir(parents=True)
    for f in cass:
        shutil.copy(f, root / f.name)
    man = _run("dila_juri", tmp_path / "dump", tmp_path / "o", skip_ids_under=str(tmp_path / "prior"))
    t = man["terminals"]
    assert t["units_in"] == len(cass) and t["units_admitted"] == len(cass) - 1
    assert t["excluded_reasons"] == {"already_in_prior_run": 1}


def test_jorf_l1_lists_the_members_the_limited_run_reads_bodies_never_retained(tmp_path, monkeypatch):
    import hashlib
    from cytir.sources import run_family
    from cytir.sources.families import dila_dump, dila_jorf
    dump = _dump(tmp_path, "dila_jorf", JORF_FILES)
    reads = []
    real = dila_dump.read_unit
    monkeypatch.setattr(dila_dump, "read_unit", lambda u: (reads.append(u.key), real(u))[1])
    seen = []
    real_l1 = dila_jorf.DilaJorfFamily.l1

    def spy(self, ctx):
        out = real_l1(self, ctx)
        seen.append(self)
        seen.append({k: v for k, v in self._l1_digests.items()})
        return out
    monkeypatch.setattr(dila_jorf.DilaJorfFamily, "l1", spy)
    man = run_family.run_family(
        "dila_jorf", out_dir=tmp_path / "o", today=TODAY, limit=2,
        consent=lambda h, d: pytest.fail(f"consent for {h}"),
        options={"root": str(dump)}, log=lambda m: None)
    rec = json.loads((tmp_path / "o" / "l1" / "dump-members.json").read_text(encoding="utf-8"))
    names = sorted(m["member"] for m in rec)
    assert [pathlib.Path(n).name for n in names] == sorted(JORF_FILES[:2])
    assert man["l1"]["members"] == 2
    # once by l1, once by acquire; the 3rd (cut) member is never read
    assert sorted(reads) == sorted(names + names)
    for m in rec:
        body = (dump / m["member"]).read_bytes()
        assert m["sha256"] == hashlib.sha256(body).hexdigest() and m["bytes"] == len(body)
    adapter, parked = seen
    assert len(parked) == 2
    assert not any(isinstance(x, (bytes, bytearray)) for v in parked.values() for x in v)
    assert adapter._l1_digests == {}                  # every selected member acquired -> drained
    assert man["terminals"]["units_in"] == 3          # the run's own ledger: 2 read, 1 cut


def test_jorf_acquire_refuses_a_member_that_changed_since_l1(tmp_path):
    from cytir.sources.family import RunContext
    from cytir.sources.families import dila_jorf
    dump = _dump(tmp_path, "dila_jorf", JORF_FILES[:1])
    ctx = RunContext(out_dir=tmp_path / "o", today=TODAY, options={"root": str(dump)})
    ad = dila_jorf.DilaJorfFamily()
    ad.l1(ctx)
    unit = ad.discover(ctx)[0]
    (dump / unit.key).write_bytes(b"<changed/>")
    with pytest.raises(RuntimeError, match="changed since l1"):
        ad.acquire(unit, ctx)
