"""`stackexchange_nonso_qa_p0` on the family door, on 13 REAL rows (slice 7 step 4, task 696).

`cyt source run stackexchange_nonso_qa_p0 --from data/samples/stackexchange-nonso --out STAGE`: no
network, no Qdrant, no model. The committed sample is thirteen whole `<row>` lines (six questions,
seven answers) of one site's `Posts.xml`, cut from the data-dump archive
(`data/samples/stackexchange-nonso/SOURCE.md` records the archive's sha256 and the cut). The family's
stage goes through `cyt index build` (consent derived by ADR-115 rule r2) and is part of the
`index-serve-gate` chain (`tests/test_aw_index_serve_gate.py`).

Pins: the sample is the cut its provenance file names (sha256 of the file and of each row) and holds
no owner id and no user profile; the cut tool takes exactly the named rows; the extractor (moved out
of the task-226 tree) makes the passages it always made (characterization: the sha256 of its output
on the sample, equal to the old script's); a run is byte-deterministic and the committed fixture
stage is the door's output; the rows carry the licence, the canonical answer URL and the declared
source type, and keep the author in the master-only part; a site with no question that clears the
bar is excluded with its reason; an empty or mis-named root refuses.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import pathlib
import re
import shutil

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SAMPLES = REPO / "data" / "samples" / "stackexchange-nonso"
SITE = "security.stackexchange.com"
POSTS = SAMPLES / SITE / "Posts.xml"
COMMITTED = REPO / "data" / "index-fixture" / "stackexchange_nonso_qa_p0"
TODAY = "2026-10-03"
FAMILY = "stackexchange_nonso_qa_p0"
POSTS_SHA256 = "70ca281d449331fac78e7cbfd28d354f88756f49532740b90eed8e26930f2565"
ARCHIVE_SHA256 = "cbd0cd8eeb565913c37328af7c0a729e77f48d283d9f917761ad7e76c84c3b62"
ROWS = {
    "16941": "c9fbff6685ca0a2f20246e22467e66f95d2e50fd93438e01b7d8b370ef2461a1",
    "16942": "4ddc9651cb86d1ea37711f65ad7fbf1f9e59d87750118c32d8d1fec809478e7e",
    "17138": "37e5cb5522b678c109f86c67a729a4b04d7e5bd9590dfffda1f12a01d8e79e4c",
    "24832": "d10fc11fab75fc25c56a5b7c78b4fc54fb0fdeee1652dfe6582aba92c974fee7",
    "24833": "8afa6b9bb57671121c29d31ba390c5d6c9382b11f02c3d59685d2a25d251460f",
    "63305": "c28a34210d3376d222a23999a1a14782f3f8c30f3317859bd6fce2cf63780dd6",
    "63309": "f4cc6c46960d003f414d32f12fe7b1182b46628f2a7ff447d7fce5027d65f198",
    "143964": "dd7b46abce2ed5005e844b7d1ab65b80b512dcf4b6602fbd566dfe2fc8939b95",
    "144186": "bdb1c03d5c132b8af4589f001049545e5eae3289327b5e084b988a4370da5fd7",
    "189582": "6a5624999073a33826ed7497c1f083b2998551dae80c4edd62700bf8358400ec",
    "189589": "7b1c4c71ecd3ec5f5f3b5e15a106660d48e3f2ee4efcdd5e9bc3ebe643809e97",
    "266446": "da5e35d321c1c1af84f4259e2f7987225dcb157fc2b7557dbdf62fcbec6b01d8",
    "266456": "ac147013589f4843456c253affac9c99c024855d07d15c1df12a39be561f3616",
}
#: sha256 of the extractor's JSON lines on the sample: the same bytes the task-226 script wrote
#: (checked with the script itself, run on this sample, when the logic moved)
EXTRACTOR_OUTPUT_SHA256 = "77cf356caf29f8713d582bbf044f608261a9055d08d064da1c1236d9f9966271"


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


def _extract(**kw):
    from cytir.extract import stackexchange_qa as E
    with POSTS.open("rb") as fh:
        return list(E.passages(E.iter_rows(fh), host=SITE, **kw))


# ------------------------------------------------------------------ the sample and its cut
def test_the_sample_is_the_cut_its_provenance_file_names():
    assert hashlib.sha256(POSTS.read_bytes()).hexdigest() == POSTS_SHA256
    assert sorted(p.name for p in SAMPLES.rglob("*") if p.is_file()) == ["Posts.xml", "SOURCE.md"]
    lines = POSTS.read_bytes().splitlines(keepends=True)
    got = {re.match(rb'\s*<row Id="(\d+)"', x).group(1).decode(): hashlib.sha256(x).hexdigest()
           for x in lines if b"<row " in x}
    assert got == ROWS
    note = (SAMPLES / "SOURCE.md").read_text(encoding="utf-8")
    assert POSTS_SHA256 in note and ARCHIVE_SHA256 in note and all(h in note for h in ROWS.values())
    assert "CC BY-SA" in note and "archive.org/details/stackexchange" in note
    assert POSTS.stat().st_size < 150_000


def test_the_sample_holds_no_owner_id_and_no_profile():
    """The attribution decision, as a test: every owner is a placeholder name the dump gives to a
    deleted account; no `OwnerUserId` and no `Users.xml`. (`LastEditorUserId` is a number, kept
    because the rows are whole.)"""
    text = POSTS.read_text(encoding="utf-8")
    assert "OwnerUserId" not in text
    names = re.findall(r'OwnerDisplayName="([^"]*)"', text)
    assert len(names) == 13 and all(re.fullmatch(r"user\d+", n) for n in names)
    assert not (SAMPLES / SITE / "Users.xml").exists()
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)       # no e-mail address


def test_the_cut_tool_takes_exactly_the_named_rows_byte_for_byte():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cut_stackexchange_nonso", REPO / "tests" / "fixture_tools" / "cut_stackexchange_nonso.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    real = POSTS.read_bytes().splitlines(keepends=True)
    decoys = [b'  <row Id="5" PostTypeId="1" Body="x" />\n', b'  <row Id="1694" PostTypeId="2" />\n']
    dump = real[:2] + decoys[:1] + real[2:6] + decoys[1:] + real[6:-1] + decoys[:1] + real[-1:]
    header, rows = tool.cut_lines(iter(dump))
    assert header == real[:2]
    assert [i for i, _ in rows] == list(tool.POST_IDS) == list(ROWS)
    assert b"".join(header + [r for _, r in rows] + [b"</posts>\n"]) == POSTS.read_bytes()


# ------------------------------------------------------------------ the extractor (moved)
def test_the_moved_extractor_makes_the_passages_the_task_226_script_made():
    recs = _extract()
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    assert hashlib.sha256(body.encode()).hexdigest() == EXTRACTOR_OUTPUT_SHA256
    assert [r["master_post_id"] for r in recs] == [
        "17138", "16942", "24833", "63309", "144186", "189589", "266456"]   # best score first
    by_id = {r["master_post_id"]: r for r in recs}
    first = by_id["16942"]
    assert first["url"] == f"https://{SITE}/a/16942" and first["license"] == "CC-BY-SA-3.0"
    assert first["text"].startswith("# Losing Private Keys\n\n")
    assert "\n\n## Top answer (score 3)\n\n" in first["text"]
    assert "## Accepted answer (score 5)" in by_id["24833"]["text"]
    assert first["master_author"] == "user10211" and first["master_author_url"] is None
    assert first["last_seen"] == "2024-04-07" and first["tags"] == []   # the 2024 dumps write |a|b|


def test_the_licence_follows_the_answers_date_and_the_boundaries_hold():
    from cytir.extract import stackexchange_qa as E
    assert E.lic_for("2011-04-08T23:59:59") == "CC-BY-SA-2.5"
    assert E.lic_for("2011-04-09T00:00:00") == "CC-BY-SA-3.0"
    assert E.lic_for("2018-05-02T12:00:00") == "CC-BY-SA-3.0"
    assert E.lic_for("2018-05-03T00:00:00") == "CC-BY-SA-4.0"
    got = {r["master_post_id"]: r["license"] for r in _extract()}
    assert got["189589"] == got["266456"] == "CC-BY-SA-4.0"
    # KNOWN, pinned as it is (reported to the founder): answer 144186 is dated 2016, so the date
    # rule says 3.0, while its own row says `ContentLicense="CC BY-SA 4.0"`
    assert got["144186"] == "CC-BY-SA-3.0" 
    row = next(x for x in POSTS.read_text().splitlines() if x.lstrip().startswith('<row Id="144186"'))
    assert 'ContentLicense="CC BY-SA 4.0"' in row


def test_a_users_map_gives_the_owner_a_display_name_and_a_profile_url():
    from cytir.extract import stackexchange_qa as E
    xml = (b'<?xml version="1.0"?><posts>'
           b'<row Id="1" PostTypeId="1" Score="9" Title="T" Body="&lt;p&gt;q&lt;/p&gt;" />'
           b'<row Id="2" PostTypeId="2" ParentId="1" Score="5" CreationDate="2020-01-01T00:00:00"'
           b' OwnerUserId="7" Body="&lt;p&gt;a&lt;/p&gt;" />'
           b'<row Id="3" PostTypeId="2" ParentId="1" Score="4" CreationDate="2020-01-01T00:00:00"'
           b' Body="&lt;p&gt;b&lt;/p&gt;" /></posts>')
    users = E.users_map(E.iter_rows(io.BytesIO(
        b'<users><row Id="7" DisplayName="Some Name" /></users>')))
    recs = list(E.passages(E.iter_rows(io.BytesIO(xml)), host="x.example.com", users=users))
    assert [(r["master_post_id"], r["master_author"], r["master_author_url"]) for r in recs] == [
        ("2", "Some Name", "https://x.example.com/users/7"), ("3", "user", None)]


# ------------------------------------------------------------------ the door
def test_two_runs_give_the_same_bytes_and_the_stage_names_no_host_or_path(tmp_path):
    _make(tmp_path / "a")
    _make(tmp_path / "b", work_dir=tmp_path / "work")
    assert _tree(tmp_path / "a") == _tree(tmp_path / "b")
    meta = (tmp_path / "a" / "PACK.json").read_text()
    assert str(tmp_path) not in meta and str(REPO) not in meta and "/tmp/" not in meta
    doc = json.loads(meta)
    assert doc["family_id"] == FAMILY and doc["rows"] == 7 and "-local-" in doc["pack_id"]
    manifest = json.loads((tmp_path / "work" / "pack" / "manifest.json").read_text())
    assert manifest["terminals"]["units_admitted"] == 1 and manifest["terminals"]["records"] == 7
    assert len((tmp_path / "work" / "l0" / "index.jsonl").read_text().splitlines()) == 1


def test_the_committed_fixture_stage_is_what_the_door_makes(tmp_path):
    _make(tmp_path / "stage")
    assert _tree(tmp_path / "stage") == _tree(COMMITTED), (
        "regenerate: cyt source run stackexchange_nonso_qa_p0 --from data/samples/stackexchange-nonso "
        "--out data/index-fixture/stackexchange_nonso_qa_p0 --today 2026-10-03")


def test_the_rows_carry_licence_url_source_type_and_keep_the_author_master_only(tmp_path):
    _make(tmp_path / "s")
    rows = _rows(tmp_path / "s")
    assert sorted(r["url"].rsplit("/", 1)[1] for r in rows) == sorted(
        i for i in ROWS if i in {"16942", "17138", "24833", "63309", "144186", "189589", "266456"})
    assert len({r["point_id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["url"].startswith(f"https://{SITE}/a/") and r["domain"] == SITE
        assert r["license"] in ("CC-BY-SA-3.0", "CC-BY-SA-4.0") and r["lang"] == "en"
        assert r["source_type"] == "community_qa" and r["tier"] == "C" and r["trust_tier"] == 0.55
        assert r["anchor"] == "chunk-0" and r["last_seen"] == TODAY and r["snippet"]
        assert r["text"].startswith(r["title"]) and "(score " in r["text"]
        m = r["_master"]
        # the author, the licence version and the post ids are master-only (ADR-039)
        assert re.fullmatch(r"user\d+", m["attribution"]) and m["author_url"] is None
        assert m["license_version"] == r["license"].rsplit("-", 1)[1]
        assert m["post_id"] == r["url"].rsplit("/", 1)[1] and m["question_id"]
        assert m["source_sha256"] == POSTS_SHA256
        assert not {"attribution", "owner", "author", "license_version"} & set(r)


def test_the_stage_is_one_the_index_build_reads_and_its_consent_derives(tmp_path):
    """`cyt index build --pack` checks the stage, its shards and vectors before it connects to
    anything; the consent stamp derives both flags by ADR-115 r2 (the dump's licence is the
    instrument). The build itself runs in `tests/test_aw_index_serve_gate.py`."""
    import sys
    sys.path.insert(0, str(REPO / "tests"))
    import standin_embedder as SE
    from cytir.consent import derive, stamp
    from cytir.index import build
    dst = tmp_path / "stage"
    _make(dst)
    SE.write_vectors(dst)
    pack = build.load_stage_pack(str(dst))
    build.check_vectors(pack)
    assert pack.family == FAMILY and pack.pack_id.startswith("stackexchange-nonso-qa-p0-v1-local-")
    cache = tmp_path / "consent-cache"
    cache.mkdir()
    ctx = stamp.load_context(as_of="2026-10-03T00:00:00Z", cache_dir=str(cache))
    got = {(d.klass, d.rule, d.search_ok, d.train_ok) for d in
           (derive.derive_consent_flags(r, ctx, family_id=FAMILY) for r in _rows(dst))}
    assert got == {("A", "r2-stackexchange", True, True)}


def test_the_family_never_asks_for_consent_and_runs_offline(tmp_path):
    from cytir.sources import run_family
    out = tmp_path / "o"
    run_family.run_family(
        FAMILY, out_dir=out, today=TODAY,
        consent=lambda h, d: pytest.fail(f"consent asked for {h}"),
        options={"root": str(SAMPLES)}, log=lambda m: None)
    manifest = json.loads((out / "pack" / "manifest.json").read_text())
    assert manifest["terminals"]["units_admitted"] == 1
    assert manifest["discover"]["chunker"]["policy_sha256"] == (
        "7dcb54c497e1855ad91ddafe80ff34b1da14f3d23b3af26dc7c7b53ab70ce6bf")


# ------------------------------------------------------------------ the ledger and the refusals
def test_a_site_with_no_question_that_clears_the_bar_is_excluded_and_the_ledger_balances(tmp_path):
    root = tmp_path / "in"
    shutil.copytree(SAMPLES, root)
    quiet = root / "quiet.stackexchange.com"
    quiet.mkdir()
    (quiet / "Posts.xml").write_bytes(
        b'<?xml version="1.0"?><posts><row Id="1" PostTypeId="1" Score="0" Title="t" Body="b" /></posts>')
    _make(tmp_path / "s", from_dir=root, work_dir=tmp_path / "work")
    t = json.loads((tmp_path / "work" / "pack" / "manifest.json").read_text())["terminals"]
    assert t["units_in"] == 2 and t["units_admitted"] == 1 and t["units_excluded"] == 1
    assert t["excluded_reasons"] == {"no_question_clears_the_score_bar": 1}
    assert t["units_quarantined"] == 0


def test_an_empty_walk_and_a_posts_file_outside_a_host_directory_refuse(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(Exception, match="no Posts.xml|empty walk|wrong root"):
        _make(tmp_path / "s1", from_dir=tmp_path / "empty")
    flat = tmp_path / "flat"
    flat.mkdir()
    shutil.copy(POSTS, flat / "Posts.xml")
    with pytest.raises(Exception, match="named by a site host"):
        _make(tmp_path / "s2", from_dir=flat)
    assert not (tmp_path / "s1").exists() and not (tmp_path / "s2").exists()
