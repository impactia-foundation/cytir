"""`stackoverflow_qa` on the family door, on REAL rows (slice 7 step 6, task 696).

`cyt source run stackoverflow_qa --from data/samples/stackoverflow-qa --out STAGE`: no network, no
Qdrant, no model. The committed sample is 24 whole `<row>` lines of the Stack Overflow data dump's
`Posts.xml` (ten questions with their accepted answers, four negative rows), cut from the posts archive
(`data/samples/stackoverflow-qa/SOURCE.md` records the archive's sha256 and the cut). The family's
stage goes through `cyt index build` (consent derived by ADR-115 rule r2) and is part of the
`index-serve-gate` chain (`tests/test_aw_index_serve_gate.py`).

Pins, in this order:
  * the sample is the cut its provenance file names (sha256 of the file and of each row), and holds no
    owner id on a selected post and no user profile;
  * the SELECTION is the production frontier's (the four negatives are refused, each for its reason);
  * the RENDERING is the live tranches' (the sha256 of the Markdown rows equals the one the old
    materializer makes (a characterization test in the old tree), and the one of the sealed
    tranche rows these ten posts came from);
  * IDENTITY and EMBEDDING SURFACE are the live ones: the point id, the document id and the dense,
    sparse and canonical texts of every row equal the rows the live pack holds for the same question
    (the `LIVE` table, read from the sealed tranche packs on the build box), and the dense and sparse
    surfaces equal the old stage script's own chunking call (pinned sha256);
  * the door is byte-deterministic and the committed fixture stage is the door's output.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SAMPLES = REPO / "data" / "samples" / "stackoverflow-qa"
POSTS = SAMPLES / "Posts.xml"
COMMITTED = REPO / "data" / "index-fixture" / "stackoverflow_qa"
TODAY = "2026-10-03"
FAMILY = "stackoverflow_qa"
SITE = "stackoverflow.com"
POSTS_SHA256 = "fa49210912c88cc2326762d9e5098f1296ae624ff49ca33949d5c75bb7e2a61a"
ARCHIVE_SHA256 = "1fcde86b9a0d701261a96698e78f65b7436d896eece5361ec7922d7c725c41cd"
ROWS: dict[str, str] = {
    "1311": "5759d1c59ecb8d81a48a58e90d0eff5786911eb33995e29603d2ff879e7f016c",
    "10595": "a3b8c264d7127af1f2b6e6d2ea4eb8112d6cc0136e56d13f0043bdaa6586e7ca",
    "38784": "016d059f009480d1a46f1691a636562b25bbcd621f3da16aefc44bb4f70c8224",
    "38834": "9d1025e207eb2edb83252d0ac6b6bcc63304269b22753ffe2c3eff6c9f9ed63e",
    "1130479": "60fad177d2da686e5610a7b0509e45cef87d0ec5d77ba2bdb12b30a0278b5eac",
    "1136143": "fe70b5673dcfa1fb4265e37d96cf9c99742a503ac90309f714db194f17647794",
    "3283306": "a327fd179afdc831d94c9b5f43538d01c5ce0f6a3b42bde3466d39dfd135ad96",
    "3283326": "118ab556785b80b3c844c97a3e5b2268874184ab84636f9efe6213965ae63a37",
    "9560723": "005462167c2662b06da1a7aa559e8461f6ceb1ef21dd5003a395da5d40f741cd",
    "9815605": "54cea17b8666de10603053c114630e75607391197dee724e2fce725c97223f15",
    "10269701": "0f009d490a42be9bbeb026a83ff7794d80fa3abf3d5335f0044adefc83b7e320",
    "10269828": "f22b5ea550cf357007d81f5c9220cad1930c870f53cdf1f4ebd9c4f74fc443cb",
    "30676720": "46baf89bf42289925d86ad79783cad4c896d202746f2d88564cd7cbf01815c84",
    "30803311": "a9f8f18e6f17168948ed709182c7006c6796f42be788b84a4ac4e74d6083acdf",
    "35666177": "942e322083e76f58edea083cde2d66b219b05f9b1ba36d06144f3dcebbec5651",
    "35666946": "6e81e45b648b0cfcc1b81fed9b8d0a37096c1c9c4305399aef01178c0f3b794e",
    "40897069": "90de156cfec7bcbaf1196b30f042f4c3ecca113e2485084d43b4376b64a1ab4c",
    "40917149": "df1912a822a3befb1f74b024f5a30cb1d975b9451155309e9f27be2a2b562d80",
    "42363405": "02dd2fb3d42cd65a991ed361a2ab95a1ed99109a9e4f249c1abb1f54d160abc8",
    "42363636": "a3e65510dae45692b04c4959aff6962796cf335536d149c1df01e4cf423c5fe1",
    "51081754": "1f180df2e15afa67f95c41d85532ce05f2cb9d34132214781ae1738089d10541",
    "51102564": "77687e21e217051037664a5d6d30fee786986e75a75fb46b9f6403dbd3835d53",
    "54220859": "ebc8d99276e0158292b69f6728cfad065986a84f29c5eb05dddee8587a229bcc",
    "54220863": "d27bfdc05bd135f086b68a631f146ece361d3e967e4773d8c6b156d57e4b1022",
}
#: question id -> accepted answer id (the ten pairs)
PAIRS = {
    "1130479": "1136143", "3283306": "3283326", "9560723": "9815605", "10269701": "10269828", "30676720": "30803311", "35666177": "35666946", "40897069": "40917149", "42363405": "42363636", "51081754": "51102564", "54220859": "54220863"
}
NEGATIVES = {
    "38784": "score_below_10", "10595": "score_below_10", "1311": "no_accepted_answer",
    "38834": "answer_not_accepted",
}
#: What the LIVE pack holds for each of the ten questions (sealed tranche packs, `aw.413a.stage.v1`
#: rows made by the old stage script): point id, document id and the
#: sha256 of the dense (`text`) and canonical (`canonical_text`) strings. `sparse_text` equals the
#: canonical string on every one of the ten. Each is ONE chunk.
LIVE = {
    "1130479": {"point_id": "fb417ddb-f57b-5dbb-9910-4f908dc8510a",
        "document_id": "e2618824e8e6a0ce14f670b12c5738de19345e06105a4130394f629d3a31d47f",
        "dense": "e7baf974317161ea6e19454337cc8ea5766b44724a4e935b5daeb6bba52d2a8e",
        "canonical": "7762b4414bf5e158436f1b7ad2f56d0f28dd26bbfbd33ff29a578e0ab63d32c1"},
    "3283306": {"point_id": "10681198-629e-5e95-be1d-d7ce23548caa",
        "document_id": "4c78ca60eadb1d13b334a3ca78d03e35745e5764a2ed46a7108e3dc5a0f5fe1e",
        "dense": "c1dceceeb652ca7fe926ed0be881d5fef46726fffca82c171b7429705adf447e",
        "canonical": "e44a6a54f9d667c4e53b00e7d57fd79f2cd9caf5ed3732520b8bb45c70cf66ee"},
    "9560723": {"point_id": "f617240b-c304-52bf-bd64-46c44c12564b",
        "document_id": "a166fdcb7b16b293824ca5c60b8228111c305921ba6419e3f1b01f4768a773b9",
        "dense": "398452a172614d22ffe47f93ba5ce2f3487912886214ad03b0323a9fde909f31",
        "canonical": "f4a38b5a5ba9f05991073dc5fc623479116a50bff1851aa423b8ddcf1605602f"},
    "10269701": {"point_id": "bc262d3e-ce21-5205-a29f-f975cf788030",
        "document_id": "6084d905094f500b314a3a10f5d5c469b8c3a360addc45b07398820287fd34c7",
        "dense": "9a93bf84591a52316e8abe20fc50744b63d11432b039c2804e311b1ee231042a",
        "canonical": "3356a3558fe8b123b3f8746c163fe05117525c8c06825f4b45c7e00ce06e4800"},
    "30676720": {"point_id": "49e5d10e-c0c0-562c-9920-d3ecef22439d",
        "document_id": "e7bfba9d1365ede2af6235db3866e4424d617ac389db505449bbff83cf66a76a",
        "dense": "afaa862acda451c871978f40523a029f8a548d2a40bfea83186389b64b000fe0",
        "canonical": "78543c576e201c16833e357cf2c38ef2f645c2d24c38c368d7b989a71e45c192"},
    "35666177": {"point_id": "9e481b51-d60f-5eb8-b07d-4c8ff0634cf7",
        "document_id": "b2cdae539f904001abf39b949faa33478d31ce22d21cd609c9af93085a0f9805",
        "dense": "db5317c483c738ed2d1e6fced8ce813f657cc6c7c685f6a4ffb32affabc209c8",
        "canonical": "6318169d6bb386a31484bbd1f6f8520c587ed07aea0090d4f93d9bfba243e10d"},
    "40897069": {"point_id": "e9deeba7-f972-595c-8bb6-91b5ac6d587a",
        "document_id": "916b6bb639cf6a796278badbdab11eadb97c7f13fa4e22f452e010909fc4e645",
        "dense": "c9cdd1121da993e1f09bf1e4aaa9bc8a82e1d16fd87084ce7ef7f9bb9ddc8c92",
        "canonical": "1fe494f19265ba8dd4b8a123a51c48e10977136bb435814ee1b0748deb2a8263"},
    "42363405": {"point_id": "66457f0b-4529-528f-b9aa-50612a0a3275",
        "document_id": "07d16b3f9f388b054881aab2ec0a0d41adc70dbd65cc296b6b1a7426424780b7",
        "dense": "e1fbb18f851f79b5b87c0dc1972e8ef56be12cae44dc9bb790045f879e444f08",
        "canonical": "b6868e7595e4ff130b3fce38cfa94fbd4c1dc3a57e4503a33508f68f5dceac71"},
    "51081754": {"point_id": "b12725ee-1e53-5f83-b605-1529af107c09",
        "document_id": "f9625faa2d885b39dfd4bae9c7e726fc11b2a0df2cebd1118e5294fbc24637f8",
        "dense": "b4a86a0b0de8388d8fad9a9ad6006dabe0c2230c3766683814dff6d1e9d9282c",
        "canonical": "6bb5901c5599b45f45e05eca9846cfc391f29ab654196fbcca97450c44d72127"},
    "54220859": {"point_id": "8c5c9a43-b108-526b-971d-9bb6e26c4b83",
        "document_id": "ac1bcffa079ad2f36af258b52959cd68b3e6d3c898db14f8d7e6a14ca68f11ee",
        "dense": "0d86653b56f076be6f036c57b380703cbf255d0e74863c1c85c3d7fb3588c58d",
        "canonical": "a0663810f730e75788b36171238b319badbb0fac044b4f933318e710b4c29a47"},
}
#: sha256 of the Markdown rows (canonical JSON lines, without the `extraction` field that names the
#: script) that the old production materializer makes on the sample; equal to the sealed rows'
MARKDOWN_SHA256 = "3f40c9e0536b825ceb6627e02d8c47a158f48287ad02e81ac609addafda7e3b9"
#: sha256 of the dense and the sparse input (canonical JSON lines of url, chunk ordinal, text) that
#: the old stage script's own chunking call makes from those Markdown rows
DENSE_SURFACE_SHA256 = "31a2611d6fbef483f85e17919a8077692993bfc756edb0c05c11eccc6a9dfaf9"
SPARSE_SURFACE_SHA256 = "7fd4ae9b48df422b5137d8685b123127aa87ae7a1502f126bdc18a4d143e6834"
POLICY_SHA256 = "7dcb54c497e1855ad91ddafe80ff34b1da14f3d23b3af26dc7c7b53ab70ce6bf"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _tree(root: pathlib.Path) -> dict[str, str]:
    return {str(p.relative_to(root)): _sha(p.read_bytes())
            for p in sorted(root.rglob("*")) if p.is_file()}


def _rows(stage: pathlib.Path) -> list[dict]:
    with gzip.open(stage / "chunks" / "shard-0000.texts.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(x) for x in fh]


def _make(out: pathlib.Path, from_dir=SAMPLES, **kw):
    from cytir.sources import stage_out
    return stage_out.run_to_stage(FAMILY, out=out, today=TODAY, from_dir=from_dir,
                                  log=lambda m: None, **kw)


def _lines(path=POSTS) -> list[bytes]:
    return pathlib.Path(path).read_bytes().splitlines(keepends=True)


def _row_id(line: bytes) -> str:
    return re.match(rb'\s*<row Id="(\d+)"', line).group(1).decode()


def _markdown_rows():
    """The family's own rendering of the sample: the Markdown row of every admitted pair."""
    from cytir.extract import stackoverflow_qa as E
    pairs, _ = E.select_pairs(POSTS)
    admitted, _ = E.admit(pairs, {})
    out = []
    for pair in admitted:
        row, quarantine = E.render(pair, {})
        assert quarantine is None
        out.append(row)
    return out


# ------------------------------------------------------------------ the sample and its cut
def test_the_sample_is_the_cut_its_provenance_file_names():
    assert _sha(POSTS.read_bytes()) == POSTS_SHA256
    assert sorted(p.name for p in SAMPLES.rglob("*") if p.is_file()) == ["Posts.xml", "SOURCE.md"]
    got = {_row_id(x): _sha(x) for x in _lines() if b"<row " in x}
    assert got == ROWS and len(got) == 24
    note = (SAMPLES / "SOURCE.md").read_text(encoding="utf-8")
    assert POSTS_SHA256 in note and ARCHIVE_SHA256 in note and all(h in note for h in ROWS.values())
    assert "CC BY-SA" in note and "archive.org/details/stackexchange" in note
    assert POSTS.stat().st_size < 60_000


def test_the_sample_names_no_natural_person():
    """The attribution decision, as a test: every selected post names its owner as the dump names a
    deleted account (`OwnerDisplayName="userNNN"`, no `OwnerUserId`), and the four negative rows
    carry a user NUMBER only (no name on the row, no `Users.xml` in the sample)."""
    by_id = {_row_id(x): x.decode() for x in _lines() if b"<row " in x}
    for q, a in PAIRS.items():
        for post in (by_id[q], by_id[a]):
            assert "OwnerUserId" not in post
            assert re.search(r'OwnerDisplayName="user\d+"', post)
    for i in NEGATIVES:
        assert "OwnerDisplayName" not in by_id[i] and "LastEditorDisplayName" not in by_id[i]
    text = POSTS.read_text(encoding="utf-8")
    assert not (SAMPLES / "Users.xml").exists()
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)       # no e-mail address
    assert "@" not in text                                       # and no handle (@name)
    assert not [n for n in re.findall(r'(?:Owner|LastEditor)DisplayName="([^"]*)"', text)
                if not re.fullmatch(r"user\d+", n)]       # no name but a deleted account's


def test_the_cut_tool_takes_exactly_the_named_rows_byte_for_byte():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cut_stackoverflow_qa", REPO / "tests" / "fixture_tools" / "cut_stackoverflow_qa.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    real = _lines()
    decoys = [b'  <row Id="5" PostTypeId="1" Body="x" />\r\n', b'  <row Id="1694" PostTypeId="2" />\r\n']
    dump = real[:2] + decoys[:1] + real[2:8] + decoys[1:] + real[8:-1] + decoys[:1] + real[-1:]
    data, found = tool.cut(iter(dump))
    assert data == POSTS.read_bytes()
    assert sorted(found) == sorted(tool.POST_IDS) == sorted(int(i) for i in ROWS)


# ------------------------------------------------------------------ the selection (the frontier's)
def test_the_selection_takes_the_ten_pairs_and_refuses_the_four_negatives():
    from cytir.extract import stackoverflow_qa as E
    pairs, counts = E.select_pairs(POSTS)
    assert [E._attrs(q)["Id"] for q, _ in pairs] == sorted(PAIRS, key=int)
    assert [E._attrs(a)["Id"] for _, a in pairs] == [PAIRS[q] for q in sorted(PAIRS, key=int)]
    # the ledger of the pass: 24 rows = 10 pairs (20) + 38784, 10595, 1311 (questions) + 38834 (answer)
    assert counts["rows"] == 24 and counts["questions"] == 13
    assert counts["questions_with_accepted_answer"] == 12      # 1311 has none
    assert counts["questions_selected"] == 10                  # 38784 and 10595 score below 10
    assert counts["pairs_with_answer"] == 10 and counts[E.REASON_NO_ANSWER] == 0
    admitted, why = E.admit(pairs, {})
    assert len(admitted) == 10 and why == {E.REASON_LICENCE: 0, E.REASON_OWNER: 0}


def test_an_answer_that_comes_before_its_question_in_the_file_is_still_paired(tmp_path):
    """The dump is not in id order (the first rows of the real file show it)."""
    from cytir.extract import stackoverflow_qa as E
    real = {_row_id(x): x for x in _lines() if b"<row " in x}
    q, a = real["54220859"], real["54220863"]
    p = tmp_path / "Posts.xml"
    p.write_bytes(b"<posts>\r\n" + a + q + b"</posts>")
    pairs, counts = E.select_pairs(p)
    assert pairs == [(q, a)] and counts["pairs_with_answer"] == 1


def test_the_rules_the_sample_cannot_show_hold_on_changed_real_rows(tmp_path):
    from cytir.extract import stackoverflow_qa as E
    real = {_row_id(x): x for x in _lines() if b"<row " in x}
    q, a = real["54220859"], real["54220863"]

    def run(q_, a_, **kw):
        p = tmp_path / "Posts.xml"
        p.write_bytes(b"<posts>\r\n" + q_ + a_ + b"</posts>")
        pairs, counts = E.select_pairs(p, **kw)
        return pairs, counts, E.admit(pairs, {})

    # the question score is the bar: 23 passes, a 9 does not
    assert run(q, a)[1]["questions_selected"] == 1
    assert run(q.replace(b'Score="23"', b'Score="9"'), a)[1]["questions_selected"] == 0
    assert run(q.replace(b'Score="23"', b'Score="10"'), a)[1]["questions_selected"] == 1
    # an answer that is not the question's child is not its accepted answer
    pairs, counts, _ = run(q, a.replace(b'ParentId="54220859"', b'ParentId="1"'))
    assert pairs == [] and counts[E.REASON_NO_ANSWER] == 1
    # a licence outside the three is refused, and counted
    _, _, (adm, why) = run(q.replace(b"CC BY-SA 4.0", b"CC BY-NC 4.0"), a)
    assert adm == [] and why[E.REASON_LICENCE] == 1
    # an owner with no name on the post and no user row is refused, and counted
    q_owner = q.replace(b'OwnerDisplayName="user6274128"', b'OwnerUserId="77"')
    _, _, (adm, why) = run(q_owner, a)
    assert adm == [] and why[E.REASON_OWNER] == 1
    # ids= takes only the named questions, and says how many it did not find
    pairs, counts, _ = run(q, a, ids={"1", "54220859"})
    assert len(pairs) == 1 and counts["ids_not_selected"] == 1
    pairs, counts, _ = run(q, a, ids={"1"})
    assert pairs == [] and counts["ids_not_selected"] == 1


def test_the_question_id_file_is_read_strictly():
    from cytir.extract import stackoverflow_qa as E
    assert E.read_ids(["# a tranche\n", "12\n", "\n", "13"]) == {"12", "13"}
    for bad in (["abc"], ["# only a comment"], []):
        with pytest.raises(ValueError):
            E.read_ids(bad)


# ------------------------------------------------------------------ the rendering (the live tranches')
def test_the_markdown_rows_are_the_ones_the_old_materializer_makes():
    from cytir.acquire.artifact_io import canonical_json
    rows = _markdown_rows()
    assert [r["question"]["post_id"] for r in rows] == sorted(PAIRS, key=int)
    body = "".join(canonical_json({k: v for k, v in r.items() if k != "extraction"}) + "\n"
                   for r in rows)
    assert _sha(body.encode()) == MARKDOWN_SHA256
    first = rows[0]
    assert first["url"] == f"https://{SITE}/questions/1130479" and first["lang"] == "en"
    assert first["markdown"].startswith("# How to build a DLL from the command line in Windows using MSVC\n\n- Question ID: 1130479\n")
    assert first["accepted_answer"]["license"] == "CC BY-SA 2.5"
    assert first["question"]["attribution"]["owner_profile_url"] is None


def test_every_row_has_the_live_point_id_document_id_and_texts_and_the_surfaces_equal_the_old_stage(word_only_chunking_allowed):
    """The point id is minted over `document:<id>:chunk-0` and the canonical text (the live scheme,
    `identity_check: pack-declared`); the dense, sparse and canonical texts equal the live pack's, and
    the surfaces equal the old stage script's own chunking call (pinned)."""
    from cytir.acquire.artifact_io import canonical_json
    from cytir.core import common as C
    contracts = json.loads((REPO / "data" / "source-family-contracts.v1.json").read_text())["families"]
    cp = next(f for f in contracts if f["family_id"] == FAMILY)["chunk_policy"]
    assert cp["sha256"] == POLICY_SHA256
    from cytir.chunk import registry as chunk_registry
    entry = chunk_registry.entry(cp["sha256"])
    sel = C.explicit_structured_chunker_selection(str(REPO / entry["file"]), cp["sha256"],
                                                  cp["tokenizer_revision"])
    dense, sparse = [], []
    for row in _markdown_rows():
        qid = row["question"]["post_id"]
        units = list(C.chunk_markdown_channels(row["markdown"], selection=sel))
        assert len(units) == 1, qid        # one chunk each; a longer document needs the tokenizer
        u = units[0]
        d, s = u.dense_text or u.text, u.sparse_text or u.text
        live = LIVE[qid]
        assert row["document_id"] == live["document_id"]
        assert C.mint_passage_point_id(row["url"], f"document:{row['document_id']}:chunk-0", u.text) == live["point_id"]
        assert _sha(d.encode()) == live["dense"] and _sha(u.text.encode()) == live["canonical"]
        assert s == u.text
        for acc, v in ((dense, d), (sparse, s)):
            acc.append(canonical_json({"url": row["url"], "ordinal": 0, "text": v}) + "\n")
    assert _sha("".join(dense).encode()) == DENSE_SURFACE_SHA256
    assert _sha("".join(sparse).encode()) == SPARSE_SURFACE_SHA256


# ------------------------------------------------------------------ the door
def test_two_runs_give_the_same_bytes_and_the_stage_names_no_host_or_path(tmp_path, word_only_chunking_allowed):
    _make(tmp_path / "a")
    _make(tmp_path / "b", work_dir=tmp_path / "work")
    assert _tree(tmp_path / "a") == _tree(tmp_path / "b")
    meta = (tmp_path / "a" / "PACK.json").read_text()
    assert str(tmp_path) not in meta and str(REPO) not in meta and "/tmp/" not in meta
    doc = json.loads(meta)
    assert doc["family_id"] == FAMILY and doc["rows"] == 10 and "-local-" in doc["pack_id"]
    manifest = json.loads((tmp_path / "work" / "pack" / "manifest.json").read_text())
    assert manifest["terminals"]["units_admitted"] == 1 and manifest["terminals"]["records"] == 10
    assert len((tmp_path / "work" / "l0" / "index.jsonl").read_text().splitlines()) == 1


def test_the_run_is_the_same_under_other_time_zones(tmp_path, word_only_chunking_allowed):
    """The extractor reads dates as the strings the dump holds and `last_seen` is `--today`; no
    machine zone may reach the bytes."""
    import os
    import subprocess
    import sys
    outs = {}
    for tz in ("UTC", "Pacific/Auckland", "America/Los_Angeles"):
        out = tmp_path / tz.replace("/", "_")
        r = subprocess.run(
            [sys.executable, "-c",
             "from cytir.chunk import chunker; chunker._ALLOW_WORD_ONLY_FOR_TESTS = True; "
             "from cytir.sources import stage_out; stage_out.run_to_stage("
             f"{FAMILY!r}, out={str(out)!r}, today={TODAY!r}, from_dir={str(SAMPLES)!r}, log=lambda m: None)"],
            capture_output=True, text=True, cwd=REPO, env={**os.environ, "TZ": tz})
        assert r.returncode == 0, r.stderr[-1500:]
        outs[tz] = _tree(out)
    assert outs["UTC"] == outs["Pacific/Auckland"] == outs["America/Los_Angeles"]


def test_the_committed_fixture_stage_is_what_the_door_makes(tmp_path, word_only_chunking_allowed):
    _make(tmp_path / "stage")
    assert _tree(tmp_path / "stage") == _tree(COMMITTED), (
        "regenerate: cyt source run stackoverflow_qa --from data/samples/stackoverflow-qa "
        "--out data/index-fixture/stackoverflow_qa --today 2026-10-03")


def test_the_rows_carry_the_live_identity_and_keep_names_master_only(tmp_path, word_only_chunking_allowed):
    _make(tmp_path / "s")
    rows = _rows(tmp_path / "s")
    assert [r["url"].rsplit("/", 1)[1] for r in rows] == sorted(PAIRS, key=int)
    for r in rows:
        qid = r["url"].rsplit("/", 1)[1]
        live = LIVE[qid]
        assert r["point_id"] == live["point_id"] and r["document_id"] == live["document_id"]
        assert r["chunk_ordinal"] == 0 and r["anchor"] == f"document:{live['document_id']}:chunk-0"
        assert _sha(r["dense_text"].encode()) == live["dense"]
        assert _sha(r["text"].encode()) == live["canonical"] and r["sparse_text"] == r["text"]
        assert r["url"].startswith(f"https://{SITE}/questions/") and r["domain"] == SITE
        assert r["source_type"] == "community" and r["tier"] == "B" and r["trust_tier"] == 0.55
        assert r["lang"] == "en" and r["last_seen"] == TODAY and r["title"] == r["title"].strip()
        # the licence of the ACCEPTED ANSWER (the live rule), as the SPDX-style name of ADR-039
        assert re.fullmatch(r"CC-BY-SA-(2\.5|3\.0|4\.0)", r["license"])
        m = r["_master"]
        assert m["answer_license"].replace(" ", "-") == r["license"]
        assert m["license_version"] == r["license"].rsplit("-", 1)[1]
        assert m["answer_id"] == PAIRS[qid] and m["question_id"] == qid
        assert m["source_sha256"] == POSTS_SHA256
        assert re.fullmatch(r"user\d+", m["attribution"]) and re.fullmatch(r"user\d+", m["question_author"])
        assert not {"attribution", "owner", "author", "license_version"} & set(r)
        # the snippet is the live one: the first 25 words of the embedded text
        from cytir.core import common as C
        assert r["snippet"] == C.extractive_snippet(r["dense_text"], 25)
        # names are in the embedded text (as in the live rows), never in the snippet or a compact field
        assert re.search(r"Question author: user\d+", r["text"])
        for key in ("title", "snippet", "license", "domain", "url", "source_type"):
            assert not re.search(r"\buser\d{3,}\b", str(r[key])), key
    # the compact dump has neither the author nor the licence version
    from cytir.core.common import compact_payload
    for r in rows:
        assert "_master" not in compact_payload(r)


def test_the_stage_is_one_the_index_build_reads_and_its_consent_derives(tmp_path, word_only_chunking_allowed):
    """`cyt index build --pack` checks the stage, its shards and vectors before it connects to
    anything; the consent stamp derives both flags by ADR-115 r2 (the dump's licence is the
    instrument). The build itself runs in `tests/test_aw_index_serve_gate.py`."""
    import sys
    sys.path.insert(0, str(REPO / "tests"))
    import standin_embedder as SE
    from cytir.consent import derive, stamp
    from cytir.index import build, compose
    dst = tmp_path / "stage"
    _make(dst)
    SE.write_vectors(dst)
    pack = build.load_stage_pack(str(dst))
    build.check_vectors(pack)
    assert pack.family == FAMILY and pack.pack_id.startswith("stackoverflow-qa-v1-local-")
    cache = tmp_path / "consent-cache"
    cache.mkdir()
    ctx = stamp.load_context(as_of="2026-10-03T00:00:00Z", cache_dir=str(cache))
    got = {(d.klass, d.rule, d.search_ok, d.train_ok) for d in
           (derive.derive_consent_flags(r, ctx, family_id=FAMILY) for r in _rows(dst))}
    assert got == {("A", "r2-stackexchange", True, True)}
    # the compose identity check (the live default anchor scheme, no per-family entry) reproduces
    # every row's point id from the row's own `document_id` and `chunk_ordinal`
    from cytir.core import common as C
    assert FAMILY not in compose.FAMILY_ANCHOR_FN
    for r in _rows(dst):
        anchor = compose.anchor_fn_for_family(FAMILY)(r)
        assert C.mint_passage_point_id(r["url"], anchor, r["text"]) == r["point_id"]


def test_the_family_never_asks_for_consent_and_runs_offline(tmp_path, word_only_chunking_allowed):
    from cytir.sources import run_family
    out = tmp_path / "o"
    run_family.run_family(
        FAMILY, out_dir=out, today=TODAY,
        consent=lambda h, d: pytest.fail(f"consent asked for {h}"),
        options={"root": str(SAMPLES)}, log=lambda m: None)
    manifest = json.loads((out / "pack" / "manifest.json").read_text())
    assert manifest["terminals"]["units_admitted"] == 1
    assert manifest["discover"]["chunker"]["policy_sha256"] == POLICY_SHA256


# ------------------------------------------------------------------ owners by id, the options, the refusals
def _synthetic_root(tmp_path, *, users: bytes | None, owner="77"):
    """Real rows with the deleted-account owner of the question replaced by an owner id (a changed
    copy of real rows: the only synthetic input of this module, named as such)."""
    real = {_row_id(x): x for x in _lines() if b"<row " in x}
    q = real["54220859"].replace(b'OwnerDisplayName="user6274128"', f'OwnerUserId="{owner}"'.encode(), 1)
    a = real["54220863"]
    root = tmp_path / "in"
    root.mkdir()
    (root / "Posts.xml").write_bytes(b"<posts>\r\n" + q + a + b"</posts>")
    if users is not None:
        (root / "Users.xml").write_bytes(users)
    return root


def test_an_owner_named_by_id_is_read_from_users_and_stays_master_only(tmp_path, word_only_chunking_allowed):
    users = (b'<users>\r\n  <row Id="76" DisplayName="Someone Else" />\r\n'
             b'  <row Id="77" Reputation="1" DisplayName="Jane Example" Location="Nowhere" />\r\n</users>')
    root = _synthetic_root(tmp_path, users=users)
    _make(tmp_path / "s", from_dir=root)
    (r,) = _rows(tmp_path / "s")
    assert "Question author: Jane Example" in r["text"]          # the embedded surface, as live
    assert r["_master"]["question_author"] == "Jane Example"     # and the master-only part
    assert "Jane" not in json.dumps({k: v for k, v in r.items()
                                     if k not in ("text", "dense_text", "sparse_text", "_master")})
    assert "Someone Else" not in json.dumps(r)
    # the user's profile URL is master-only; the answer's owner has none (deleted account)
    assert r["_master"]["author_url"] is None


def test_a_post_that_names_its_owner_by_id_only_refuses_without_users_xml(tmp_path):
    from cytir.sources import stage_out
    root = _synthetic_root(tmp_path, users=None)
    log: list[str] = []
    with pytest.raises(Exception, match="every one of 1 unit"):
        stage_out.run_to_stage(FAMILY, out=tmp_path / "s", today=TODAY, from_dir=root,
                               log=log.append, work_dir=tmp_path / "work")
    assert any("Users.xml" in m for m in log), log
    assert not (tmp_path / "s").exists()


def test_ids_takes_only_the_named_questions(tmp_path, word_only_chunking_allowed):
    ids = tmp_path / "ids.txt"
    ids.write_text("# a tranche\n1130479\n54220859\n999999\n")
    from cytir.sources import run_family
    out = tmp_path / "o"
    run_family.run_family(FAMILY, out_dir=out, today=TODAY, consent=lambda h, d: pytest.fail("consent"),
                          options={"root": str(SAMPLES), "ids": str(ids)}, log=lambda m: None)
    assert json.loads((out / "pack" / "manifest.json").read_text())["terminals"]["records"] == 2


def test_a_posts_file_with_no_selected_question_is_excluded_and_the_run_refuses(tmp_path):
    """Only the four negatives: the unit ends `excluded` with its reason, and a run that admits 0
    passages refuses (smoke-test-yield) and writes no stage."""
    from cytir.sources import stage_out
    root = tmp_path / "in"
    root.mkdir()
    real = [x for x in _lines() if b"<row " in x and _row_id(x) in NEGATIVES]
    (root / "Posts.xml").write_bytes(b"<posts>\r\n" + b"".join(real) + b"</posts>")
    log: list[str] = []
    with pytest.raises(Exception) as err:
        stage_out.run_to_stage(FAMILY, out=tmp_path / "s", today=TODAY, from_dir=root,
                               log=log.append, work_dir=tmp_path / "work")
    seen = str(err.value) + "\n".join(log)
    assert "excluded=1" in seen or "no_accepted_question_scores_10_or_more" in seen, seen
    assert not (tmp_path / "s").exists()


def test_an_empty_walk_and_a_missing_root_refuse(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(Exception, match="no Posts.xml|empty walk|wrong root"):
        _make(tmp_path / "s1", from_dir=tmp_path / "empty")
    assert not (tmp_path / "s1").exists()
