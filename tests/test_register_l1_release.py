"""Real-shape tests for example-producer/register_l1_release.py (task 641, 2026-09-23).

Field names/shapes below are the EXACT keys read off the real staged files
on the build host that this tool was built to fix:
  - <staged-root>/packs/tranche1/ch_health-pack-0001.jsonl
    (CC tranche 1, family cc_authority_frontier_w7)
  - <staged-root>/staged/apple.stackexchange.com.passages.jsonl
    (SE, family stackexchange_nonso_qa_p0)
Sample content is illustrative (the build host's own rows are not available from
this checkout), but every key and the general shape (a `url`, a `text`, no
`markdown` field) is taken verbatim from those two real files.

The tool's default mode COPIES with a sha256 verification (the coordinator's
2026-09-23 correction: a symlink into a scratch/work dir makes "persisted"
depend on that scratch dir surviving -- exactly what ADR-105/task 588 exists
to rule out). `--link`/`link=True` is kept for a fast local dry run only;
`test_persisted_inputs.py::test_a_symlinked_leaf_file_is_refused_not_indexed`
proves the real ADR-105 index refuses to index what it produces.
"""
import gzip
import json
import pathlib

import pytest

import cytir.corpus.register_l1_release as R


def _write_jsonl(path: pathlib.Path, rows, gz: bool = False):
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    else:
        with path.open("wt", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")


# Real record shape from task634/packs/tranche1/ch_health-pack-0001.jsonl
CC_TRANCHE1_ROW = {
    "point_id": "00019a57-0000-0000-0000-000000000001",
    "url": "https://courirensemble.ch/la-boite-a-musique-des-enfoires",
    "domain": "courirensemble.ch",
    "title": "La boite a musique des Enfoires",
    "text": "x" * 919,
    "snippet": "x" * 180,
    "lang": "fr",
    "tier": "C",
    "trust_tier": 0.35,
    "source_type": "community",
    "search_ok": True,
    "train_ok": True,
    "content_digest": "deadbeef" * 8,
    "last_seen": "2026-09-22",
    "consent_checked_at": "2026-09-20T00:00:00Z",
    "class_basis": "ADR-111-opt-out-minimum-record",
    "disc_class": "general_web",
    "n_endorser_regs": 0,
    "outlinks": [],
    "provenance": {"crawl": "CC-MAIN-2026-39"},
    "stamps": {},
}

# Real record shape from depth226/staged/apple.stackexchange.com.passages.jsonl
SE_ROW = {
    "url": "https://apple.stackexchange.com/a/9",
    "text": "y" * 1006,
    "snippet": "y" * 180,
    "title": "Some SE answer",
    "domain": "apple.stackexchange.com",
    "lang": "en",
    "tier": "B",
    "trust_tier": 0.55,
    "source_type": "community",
    "license": "cc-by-sa-4.0",
    "content_digest": "cafebabe" * 8,
    "last_seen": "2026-09-22",
    "tags": ["macos"],
    "provenance": {},
    "master_post_id": 9,
    "master_question_id": 3,
    "master_author": "someone",
    "master_author_url": "https://apple.stackexchange.com/users/1",
    "master_is_accepted": True,
    "master_answer_score": 5,
    "master_license_version": "4.0",
}


def test_cc_tranche1_shape_registers_a_verified_copy(tmp_path):
    source = tmp_path / "packs" / "tranche1"
    source.mkdir(parents=True)
    _write_jsonl(source / "ch_health-pack-0001.jsonl", [CC_TRANCHE1_ROW, CC_TRANCHE1_ROW])
    # A receipt sitting right next to it must NOT be mistaken for content.
    (source / "ch_health-pack-0001.receipt.json").write_text("{}")
    (source / "ch_health-pack-0001.vector_receipt.json").write_text("{}")

    corpus_root = tmp_path / "corpus"
    result = R.register_l1_release(str(source), "task641-cc-tranche1-w7",
                                    corpus_root=str(corpus_root))
    assert result["files"] == 1
    assert result["link"] is False
    copied = pathlib.Path(result["copied"][0])
    assert not copied.is_symlink()
    assert copied.read_bytes() == (source / "ch_health-pack-0001.jsonl").read_bytes()
    # Matches the l1-release-plain store glob verbatim.
    assert copied.match(str(corpus_root / "l1" / "release=*" / "*.jsonl"))
    assert result["checked"][0]["url_field"] == "url"
    assert result["checked"][0]["text_field"] == "text"

    manifest = json.loads((copied.parent / R.MANIFEST_NAME).read_text())
    assert manifest["files"][0]["file"] == "ch_health-pack-0001.jsonl"
    assert manifest["files"][0]["sha256"] == R.sha256_file(copied)
    assert manifest["files"][0]["source_path"] == str((source / "ch_health-pack-0001.jsonl").resolve())


def test_se_shape_registers_with_file_suffix_filter(tmp_path):
    source = tmp_path / "staged"
    source.mkdir(parents=True)
    _write_jsonl(source / "apple.stackexchange.com.passages.jsonl", [SE_ROW])
    _write_jsonl(source / "apple.stackexchange.com.passages.jsonl.manifest.json", [{"not": "content"}])

    corpus_root = tmp_path / "corpus"
    result = R.register_l1_release(str(source), "task641-se-nonso-p0",
                                    corpus_root=str(corpus_root),
                                    file_suffix=".passages.jsonl")
    assert result["files"] == 1
    assert result["copied"][0].endswith("apple.stackexchange.com.passages.jsonl")
    assert not pathlib.Path(result["copied"][0]).is_symlink()


def test_gz_variant_is_also_registered_as_a_copy(tmp_path):
    source = tmp_path / "gz-source"
    source.mkdir(parents=True)
    _write_jsonl(source / "shard-0000.jsonl.gz", [CC_TRANCHE1_ROW], gz=True)

    corpus_root = tmp_path / "corpus"
    result = R.register_l1_release(str(source), "task641-gz-release",
                                    corpus_root=str(corpus_root))
    assert result["files"] == 1
    out = pathlib.Path(result["copied"][0])
    assert out.name.endswith("shard-0000.jsonl.gz")
    assert not out.is_symlink()
    assert out.read_bytes() == (source / "shard-0000.jsonl.gz").read_bytes()


def test_link_mode_still_symlinks_for_a_fast_local_dry_run(tmp_path):
    source = tmp_path / "packs"
    source.mkdir(parents=True)
    _write_jsonl(source / "a.jsonl", [CC_TRANCHE1_ROW])

    corpus_root = tmp_path / "corpus"
    result = R.register_l1_release(str(source), "task641-linked", corpus_root=str(corpus_root),
                                    link=True)
    assert result["link"] is True
    linked = pathlib.Path(result["linked"][0])
    assert linked.is_symlink()
    assert linked.resolve() == (source / "a.jsonl").resolve()


def test_dry_run_checks_shape_but_does_not_write(tmp_path):
    source = tmp_path / "packs"
    source.mkdir(parents=True)
    _write_jsonl(source / "a.jsonl", [CC_TRANCHE1_ROW])

    corpus_root = tmp_path / "corpus"
    result = R.register_l1_release(str(source), "task641-dry", corpus_root=str(corpus_root),
                                    dry_run=True)
    assert result["dry_run"] is True
    assert result["files"] == 1
    assert not (corpus_root / "l1").exists()


def test_missing_url_field_refuses_before_copying(tmp_path):
    """The mechanical guard the ad hoc `ln -sfn` fix never had: a file that
    does not actually carry a url/canonical_url field must refuse, not
    silently file an unusable store into the registry."""
    source = tmp_path / "bad"
    source.mkdir(parents=True)
    bad_row = dict(CC_TRANCHE1_ROW)
    del bad_row["url"]
    _write_jsonl(source / "bad.jsonl", [bad_row])

    corpus_root = tmp_path / "corpus"
    with pytest.raises(R.RegisterL1Error, match="url_fields"):
        R.register_l1_release(str(source), "task641-bad", corpus_root=str(corpus_root))
    assert not (corpus_root / "l1").exists()


def test_missing_text_field_refuses_before_copying(tmp_path):
    source = tmp_path / "bad2"
    source.mkdir(parents=True)
    bad_row = dict(CC_TRANCHE1_ROW)
    del bad_row["text"]
    _write_jsonl(source / "bad2.jsonl", [bad_row])

    corpus_root = tmp_path / "corpus"
    with pytest.raises(R.RegisterL1Error, match="text_fields"):
        R.register_l1_release(str(source), "task641-bad2", corpus_root=str(corpus_root))


def test_no_matching_files_refuses(tmp_path):
    source = tmp_path / "empty"
    source.mkdir(parents=True)
    (source / "notes.txt").write_text("nothing here")

    corpus_root = tmp_path / "corpus"
    with pytest.raises(R.RegisterL1Error, match="nothing to register"):
        R.register_l1_release(str(source), "task641-empty", corpus_root=str(corpus_root))


def test_rerun_is_idempotent(tmp_path):
    source = tmp_path / "packs"
    source.mkdir(parents=True)
    _write_jsonl(source / "a.jsonl", [CC_TRANCHE1_ROW])

    corpus_root = tmp_path / "corpus"
    r1 = R.register_l1_release(str(source), "task641-idem", corpus_root=str(corpus_root))
    r2 = R.register_l1_release(str(source), "task641-idem", corpus_root=str(corpus_root))
    assert r1["copied"] == r2["copied"]
    out = pathlib.Path(r2["copied"][0])
    assert not out.is_symlink()
    assert out.read_bytes() == (source / "a.jsonl").read_bytes()


def test_copy_verification_failure_leaves_nothing_filed(tmp_path, monkeypatch):
    """If the post-copy sha256 ever disagreed with the source (a torn write,
    a bad disk), the tool must refuse rather than file corrupt content as if
    it were persisted -- the whole point of a verified copy over a bare
    `shutil.copyfile`."""
    source = tmp_path / "packs"
    source.mkdir(parents=True)
    _write_jsonl(source / "a.jsonl", [CC_TRANCHE1_ROW])
    corpus_root = tmp_path / "corpus"

    real_sha256_file = R.sha256_file
    calls = {"n": 0}

    def _lying_sha256_file(path):
        calls["n"] += 1
        # Second call is the post-copy check -- make it disagree.
        if calls["n"] == 2:
            return "0" * 64
        return real_sha256_file(path)

    monkeypatch.setattr(R, "sha256_file", _lying_sha256_file)
    with pytest.raises(R.RegisterL1Error, match="verification FAILED"):
        R.register_l1_release(str(source), "task641-torn", corpus_root=str(corpus_root))
    dest = corpus_root / "l1" / "release=task641-torn"
    assert not (dest / "a.jsonl").exists()


def test_insufficient_free_space_refuses_before_writing(tmp_path, monkeypatch):
    source = tmp_path / "packs"
    source.mkdir(parents=True)
    _write_jsonl(source / "a.jsonl", [CC_TRANCHE1_ROW])
    corpus_root = tmp_path / "corpus"

    class _TinyUsage:
        free = 1  # bytes -- smaller than any real file

    monkeypatch.setattr(R.shutil, "disk_usage", lambda *_a, **_k: _TinyUsage())
    with pytest.raises(R.RegisterL1Error, match="not enough free space"):
        R.register_l1_release(str(source), "task641-full", corpus_root=str(corpus_root))
    assert not (corpus_root / "l1").exists()


def test_recursive_files_nested_shards_and_keeps_relative_paths(tmp_path):
    """task 686: OpenAlex markdown sits at stages/markdown/data/<xx>/<sha>.jsonl.gz."""
    source = tmp_path / "markdown"
    (source / "data" / "bc").mkdir(parents=True)
    (source / "data" / "aa").mkdir(parents=True)
    row = {"url": "https://openalex.org/W1", "markdown": "# t\n\nbody"}
    _write_jsonl(source / "data" / "bc" / "x.jsonl.gz", [row], gz=True)
    _write_jsonl(source / "data" / "aa" / "x.jsonl.gz", [row], gz=True)
    (source / "receipt.json").write_text("{}")
    corpus_root = tmp_path / "corpus"

    with pytest.raises(R.RegisterL1Error):  # flat mode sees nothing at depth
        R.register_l1_release(str(source), "flat", corpus_root=str(corpus_root))
    result = R.register_l1_release(str(source), "oa", corpus_root=str(corpus_root),
                                    recursive=True)
    assert result["files"] == 2
    dest = corpus_root / "l1" / "release=oa"
    assert (dest / "data" / "bc" / "x.jsonl.gz").is_file()
    assert (dest / "data" / "aa" / "x.jsonl.gz").is_file()
    manifest = json.loads((dest / R.MANIFEST_NAME).read_text())
    assert sorted(e["file"] for e in manifest["files"]) == [
        "data/aa/x.jsonl.gz", "data/bc/x.jsonl.gz"]


def test_recursive_skips_symlinked_source_files(tmp_path):
    source = tmp_path / "markdown"
    (source / "d").mkdir(parents=True)
    real = tmp_path / "elsewhere.jsonl"
    _write_jsonl(real, [{"url": "https://openalex.org/W1", "markdown": "m"}])
    (source / "d" / "link.jsonl").symlink_to(real)
    with pytest.raises(R.RegisterL1Error):
        R.register_l1_release(str(source), "oa", corpus_root=str(tmp_path / "c"),
                               recursive=True)
