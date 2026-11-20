"""Task 657 — `cytir.index.compose`: identity verification + manifest compose
+ enqueue, on REAL rows.

`tests/fixtures/index_compose/task657_eric_real_rows.jsonl` is two REAL chunk records
from the task-656 ERIC smoke pack (the build box,
`education-eric-v1/packs/official-pdf-tranche-130000-first-000004-v1-
partitioned2/chunks/shard-0000.texts.jsonl.gz`, document ED239487, chunks
0-1) -- not synthesized. `test_verify_identity_matches_real_pack_ids` is the
exact check task 657's brief asked for: "check that it reproduces the ids of
an EXISTING ... family already in prod on real records" (ERIC is not in
prod yet, so this is the pack's own sealed ids, which is what
`identity_check: pack-declared` trusts). Confirmed against the FULL 111-row
pack directly on the build box during task 657 (111/111); this fixture is the
same two rows the module used there, kept small for CI.
"""
from __future__ import annotations

import gzip
import json
import pathlib

import pytest

import cytir.index.compose as compose
from cytir.index import promote_sealed_pack as PSP
from tests.helpers.sealed_evidence import require_sealed_probe

require_sealed_probe()  # real ERIC rows need the sealed consent probe, which the public cut lacks


HERE = pathlib.Path(__file__).resolve().parent
REAL_ROWS_PATH = HERE.parent / "tests" / "fixtures" / "index_compose" / "task657_eric_real_rows.jsonl"


def _real_rows() -> list[dict]:
    return [json.loads(line) for line in
           REAL_ROWS_PATH.read_text().splitlines() if line.strip()]


def _write_chunk_shard(chunks_dir: pathlib.Path, rows: list[dict]) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(chunks_dir / "shard-0000.texts.jsonl.gz", "wt",
                   encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _write_vector_manifest(vectors_dir: pathlib.Path, n: int,
                          dim: int = 1024) -> None:
    vectors_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "aw.413a.vectors.v1", "shard": "shard-0000", "n": n,
               "dim": dim, "dtype": "float16"}
    (vectors_dir / "shard-0000.vectors.manifest.json").write_text(
        json.dumps(manifest))


def _write_real_vector_shard(vectors_dir: pathlib.Path, ids: list[str],
                             dim: int = 1024) -> None:
    """A properly-SHAPED sealed vector shard (real row_ids.jsonl + a real
    .f16 data file, values are zeros -- only the shape/ids matter for
    these tests) -- unlike `_write_vector_manifest` above (a bare
    manifest, enough for `read_vector_native_dim`'s dim check), task 669's
    `backfill_lang` path reads the vectors dir through `promote_sealed_
    pack.read_vectors`/`normalize_sealed_pack_shape.target_ids`, which
    need the real id-bearing files."""
    import numpy as np
    vectors_dir.mkdir(parents=True, exist_ok=True)
    n = len(ids)
    np.zeros((n, dim), dtype="<f2").tofile(vectors_dir / "shard-0000.vectors.f16")
    with open(vectors_dir / "shard-0000.row_ids.jsonl", "w") as f:
        for pid in ids:
            f.write(json.dumps({"point_id": pid}) + "\n")
    manifest = {"schema": "aw.413a.vectors.v1", "shard": "shard-0000", "n": n,
               "dim": dim, "dtype": "float16"}
    (vectors_dir / "shard-0000.vectors.manifest.json").write_text(
        json.dumps(manifest))


# --------------------------------------------------------------- identity


def test_real_fixture_has_two_distinct_real_rows():
    rows = _real_rows()
    assert len(rows) == 2
    assert rows[0]["source_record_id"] == "ED239487"
    assert rows[0]["url"] == "https://files.eric.ed.gov/fulltext/ED239487.pdf"
    assert rows[0]["point_id"] == "879feb01-8dcf-5036-a831-118c056502cf"
    assert rows[1]["point_id"] == "2de486a9-5a63-5172-b151-97e08a96f6de"
    assert rows[0]["point_id"] != rows[1]["point_id"]


def test_verify_identity_matches_real_pack_ids(tmp_path):
    """aw_common.mint_passage_point_id reproduces the REAL pack's own ids --
    task 657's headline finding, on real records, not inferred."""
    rows = _real_rows()
    chunks_dir = tmp_path / "chunks"
    _write_chunk_shard(chunks_dir, rows)
    result = compose.verify_identity(str(chunks_dir))
    assert result == {"checked": 2, "matched": 2, "mismatches": []}


def test_verify_identity_refuses_on_any_mismatch(tmp_path):
    """The task 657 brief: 'refuse to compose on any id mismatch' --
    ONE corrupted row refuses the whole check, not a partial pass."""
    rows = _real_rows()
    rows[1] = {**rows[1], "point_id": "00000000-0000-0000-0000-000000000000"}
    chunks_dir = tmp_path / "chunks"
    _write_chunk_shard(chunks_dir, rows)
    with pytest.raises(compose.ComposeError, match="identity check FAILED"):
        compose.verify_identity(str(chunks_dir))


def test_verify_identity_refuses_the_b1_scheme_as_a_negative_control(tmp_path):
    """`cytir.core.ids.mint_point_id` (the b1/future scheme) must NOT reproduce
    these real rows -- proves the module is actually discriminating between
    the two schemes, not vacuously passing."""
    import cytir.core.ids as ids

    rows = _real_rows()
    for row in rows:
        anchor = compose.default_anchor(row)
        b1_id = ids.mint_point_id(row["url"], anchor)
        assert b1_id != row["point_id"], (
            "if this ever matches, cytir.core.ids has become the right scheme "
            "and compose.py's docstring/manifest reason need updating")


def test_verify_identity_refuses_empty_stage(tmp_path):
    chunks_dir = tmp_path / "chunks"
    _write_chunk_shard(chunks_dir, [])
    with pytest.raises(compose.ComposeError, match="0 rows"):
        compose.verify_identity(str(chunks_dir))


def test_verify_identity_refuses_row_without_point_id(tmp_path):
    rows = _real_rows()
    del rows[0]["point_id"]
    chunks_dir = tmp_path / "chunks"
    _write_chunk_shard(chunks_dir, rows)
    with pytest.raises(compose.ComposeError, match="carries no point_id"):
        compose.verify_identity(str(chunks_dir))


# --------------------------------------------------------- dims / widths


def test_read_vector_native_dim_refuses_when_absent(tmp_path):
    with pytest.raises(compose.ComposeError, match="embed must run"):
        compose.read_vector_native_dim(str(tmp_path / "novectors"))


def test_read_vector_native_dim_refuses_mixed_shards(tmp_path):
    vectors_dir = tmp_path / "vectors"
    _write_vector_manifest(vectors_dir, n=2, dim=1024)
    (vectors_dir / "shard-0001.vectors.manifest.json").write_text(
        json.dumps({"schema": "aw.413a.vectors.v1", "n": 3, "dim": 256}))
    with pytest.raises(compose.ComposeError, match="mixed vector dims"):
        compose.read_vector_native_dim(str(vectors_dir))


def test_read_vector_native_dim_reads_the_declared_width(tmp_path):
    vectors_dir = tmp_path / "vectors"
    _write_vector_manifest(vectors_dir, n=2, dim=1024)
    assert compose.read_vector_native_dim(str(vectors_dir)) == 1024


# ----------------------------------------------------------------- compose


def test_compose_pack_writes_manifest_and_ready(tmp_path):
    rows = _real_rows()
    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    queue_dir = tmp_path / "queue"
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=1024)

    result = compose.compose_pack(
        pack_id="test-eric-pack-1", family="eric_official_pdf",
        chunks_dir=str(chunks_dir), vectors_dir=str(vectors_dir),
        collection="aw_e8", mrl=256, queue_dir=str(queue_dir))

    pack_dir = pathlib.Path(result["pack_dir"])
    manifest_path = pack_dir / "manifest.json"
    ready_path = pack_dir / "READY"
    assert manifest_path.exists() and ready_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "aw.promotion-queue-pack.v1"
    assert manifest["pack_id"] == "test-eric-pack-1"
    assert manifest["family"] == "eric_official_pdf"
    assert manifest["pack_chunks"] == str(chunks_dir) + "-consent-stamped"
    assert manifest["vectors"] == str(vectors_dir)
    assert manifest["mrl"] == 256
    assert manifest["collection"] == "aw_e8"
    assert manifest["identity_check"] == "pack-declared"
    assert "2/2" in manifest["identity_reason"]
    assert "cytir.core.ids.mint_point_id" in manifest["identity_reason"]

    # READY names sha256(manifest.json) -- promote_queue.read_manifest's
    # own integrity check (an edit-after-ready refuses to drain).
    import hashlib
    expected_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert ready_path.read_text().strip() == expected_sha


def _trimmed_ids_file(tmp_path: pathlib.Path, ids: list[str]) -> pathlib.Path:
    f = tmp_path / "trimmed.jsonl.gz"
    with gzip.open(f, "wt", encoding="utf-8") as fh:
        for pid in ids:
            fh.write(json.dumps({"point_id": pid}) + "\n")
    return f


def test_compose_pack_pins_an_exclusion_file_in_the_manifest(tmp_path):
    """Task 680: the OpenAlex ladder trim travels as a sha-pinned id list the
    drain reads, never as a filtered copy of 12.8 GB of sealed vectors."""
    from cytir.index import exclude_file as XF
    rows = _real_rows()
    chunks_dir, vectors_dir = tmp_path / "chunks", tmp_path / "vectors"
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=1024)
    trimmed = _trimmed_ids_file(tmp_path, [rows[1]["point_id"]])

    result = compose.compose_pack(
        pack_id="p", family="eric_official_pdf", chunks_dir=str(chunks_dir),
        vectors_dir=str(vectors_dir), collection="aw_e8", mrl=256,
        queue_dir=str(tmp_path / "queue"), exclude_ids_file=str(trimmed),
        exclude_ids_reason="below_admission_floor")
    m = result["manifest"]
    assert m["exclude_ids_file"] == str(trimmed)
    assert m["exclude_ids_sha256"] == XF.sha256_file(trimmed)
    assert m["exclude_ids_reason"] == "below_admission_floor"
    assert m["rows"] == 2          # the pack stays whole; the rule is data
    # the two numbers, each under its own name: the alias `rows` is the stage count here
    assert (m["rows_staged"], m["rows_to_write_max"]) == (2, 1)


def test_compose_pack_refuses_an_exclusion_file_without_a_reason(tmp_path):
    rows = _real_rows()
    chunks_dir, vectors_dir = tmp_path / "chunks", tmp_path / "vectors"
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=1024)
    trimmed = _trimmed_ids_file(tmp_path, [rows[0]["point_id"]])
    queue = tmp_path / "queue"
    with pytest.raises(compose.ComposeError, match="exclude-ids-reason"):
        compose.compose_pack(
            pack_id="p", family="eric_official_pdf", chunks_dir=str(chunks_dir),
            vectors_dir=str(vectors_dir), collection="aw_e8", mrl=256,
            queue_dir=str(queue), exclude_ids_file=str(trimmed))
    assert not (queue / "p").exists()       # nothing half-written


def test_compose_pack_omits_the_exclusion_fields_when_not_given(tmp_path):
    rows = _real_rows()
    chunks_dir, vectors_dir = tmp_path / "chunks", tmp_path / "vectors"
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=1024)
    m = compose.compose_pack(
        pack_id="p", family="eric_official_pdf", chunks_dir=str(chunks_dir),
        vectors_dir=str(vectors_dir), collection="aw_e8", mrl=256,
        queue_dir=str(tmp_path / "queue"))["manifest"]
    assert not any(k.startswith("exclude_ids") for k in m)


def test_compose_pack_writes_license_from_and_family_license(tmp_path):
    """Task 664, 2026-09-26: `stackoverflow_qa`'s rows carry no top-level
    `license` and `stackoverflow.com` is not in the licence host table, so
    every tranche composed through this door refused at promotion with
    "unresolved license" -- this door had NO way to declare either a
    per-row lift path or a family-wide grant. Both must reach the
    manifest, the same fields `promote_queue.py`'s `plan_kwargs()` reads."""
    rows = _real_rows()
    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    queue_dir = tmp_path / "queue"
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=1024)

    result = compose.compose_pack(
        pack_id="test-so-pack-1", family="stackoverflow_qa",
        chunks_dir=str(chunks_dir), vectors_dir=str(vectors_dir),
        collection="aw_e8", mrl=256, queue_dir=str(queue_dir),
        license_from="source_metadata.accepted_answer.license",
        family_license="CC-BY-SA-4.0")

    manifest = json.loads(
        (pathlib.Path(result["pack_dir"]) / "manifest.json").read_text())
    assert manifest["license_from"] == "source_metadata.accepted_answer.license"
    assert manifest["family_license"] == "CC-BY-SA-4.0"


def test_compose_pack_omits_license_fields_when_not_given(tmp_path):
    """No regression: every EXISTING caller (CC, ERIC) that never passes
    either field must see the SAME manifest shape as before -- no new keys
    appear unasked."""
    rows = _real_rows()
    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=1024)

    result = compose.compose_pack(
        pack_id="test-eric-pack-no-license", family="eric_official_pdf",
        chunks_dir=str(chunks_dir), vectors_dir=str(vectors_dir),
        collection="aw_e8", mrl=256, queue_dir=str(tmp_path / "queue"))

    manifest = json.loads(
        (pathlib.Path(result["pack_dir"]) / "manifest.json").read_text())
    assert "license_from" not in manifest
    assert "family_license" not in manifest
    assert "language_scope_filter" not in manifest


def test_compose_pack_writes_language_scope_filter_when_given(tmp_path):
    """Tim's ruling, task 664, 2026-09-28: ERIC rows whose language cannot
    be detected confidently are DROPPED and RECORDED -- the same opt-in
    manifest field CC tranche 1 used. Wired through compose_pack() so
    every ERIC tranche composed through embed_compose_enqueue gets it
    automatically, without a per-tranche hand-edit of the manifest."""
    rows = _real_rows()
    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=1024)

    result = compose.compose_pack(
        pack_id="test-eric-pack-lang-scope", family="eric_official_pdf",
        chunks_dir=str(chunks_dir), vectors_dir=str(vectors_dir),
        collection="aw_e8", mrl=256, queue_dir=str(tmp_path / "queue"),
        language_scope_filter=True)

    manifest = json.loads(
        (pathlib.Path(result["pack_dir"]) / "manifest.json").read_text())
    assert manifest["language_scope_filter"] is True


def test_cli_language_scope_filter_flag_reaches_the_manifest(
        tmp_path, monkeypatch):
    rows = _real_rows()
    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    queue_dir = tmp_path / "queue"
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=1024)
    monkeypatch.setattr(compose, "embed_stage", lambda *a, **k: 0)

    argv = ["--pack-id", "test-eric-cli-lang-scope",
           "--family", "eric_official_pdf",
           "--chunks-dir", str(chunks_dir), "--vectors-dir", str(vectors_dir),
           "--collection", "aw_e8", "--mrl", "256",
           "--queue-dir", str(queue_dir), "--language-scope-filter"]
    assert compose.main(argv) == 0

    manifest = json.loads(
        (queue_dir / "test-eric-cli-lang-scope" / "manifest.json").read_text())
    assert manifest["language_scope_filter"] is True


# ------------------------------------------- task 669/C: backfill_lang


def test_compose_pack_backfill_lang_rewrites_chunks_and_fills_the_real_empty_rows(
        tmp_path, monkeypatch):
    """Both real ERIC fixture rows carry `lang: None` (the actual shape
    task 669/C found: 86.5%/77.9% lang payload-carry, nothing ever fills
    it). With `backfill_lang=True`, `compose_pack` physically rewrites
    `chunks_dir` to a sibling `-lang-backfilled` directory, fills `lang`
    there via the tracked detector, and points the manifest's
    `pack_chunks` at THAT directory -- identity still verifies 2/2
    (backfill only ever touches `lang`/`lang_source`, never url/anchor/
    text, so the minted id is unaffected). The ORIGINAL materialized
    chunks are left untouched on disk."""
    from cytir.acquire import fetch_process as FP  # noqa: PLC0415

    monkeypatch.setattr(FP, "detect_lang", lambda text, url, default: ("en", True))

    rows = _real_rows()
    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    queue_dir = tmp_path / "queue"
    _write_chunk_shard(chunks_dir, rows)
    _write_real_vector_shard(vectors_dir, [r["point_id"] for r in rows])

    result = compose.compose_pack(
        pack_id="test-eric-pack-lang-backfill", family="eric_official_pdf",
        chunks_dir=str(chunks_dir), vectors_dir=str(vectors_dir),
        collection="aw_e8", mrl=256, queue_dir=str(queue_dir),
        backfill_lang=True)

    manifest = json.loads(
        (pathlib.Path(result["pack_dir"]) / "manifest.json").read_text())
    backfilled_dir = str(chunks_dir) + "-lang-backfilled"
    assert manifest["pack_chunks"] == backfilled_dir + "-consent-stamped"
    assert manifest["lang_backfilled_from"] == str(chunks_dir)
    assert manifest["lang_backfill_report"]["filled"] == 2
    assert manifest["lang_backfill_report"]["already_declared"] == 0
    assert result["identity"] == {"checked": 2, "matched": 2, "mismatches": []}

    backfilled_rows = {r["point_id"]: r for r in PSP.read_records(backfilled_dir)}
    assert all(r["lang"] == "en" for r in backfilled_rows.values())
    assert all(r["lang_source"] for r in backfilled_rows.values())
    # the original materialization is untouched
    original_rows = {r["point_id"]: r for r in PSP.read_records(str(chunks_dir))}
    assert all(r["lang"] is None for r in original_rows.values())


def test_compose_pack_omits_lang_backfill_fields_when_not_given(tmp_path):
    """No regression: the default (`backfill_lang=False`) leaves the
    manifest's `pack_chunks` pointed at the ORIGINAL chunks_dir, exactly
    as every existing caller (CC, ERIC without the flag) already relies
    on."""
    rows = _real_rows()
    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=1024)

    result = compose.compose_pack(
        pack_id="test-eric-pack-no-backfill", family="eric_official_pdf",
        chunks_dir=str(chunks_dir), vectors_dir=str(vectors_dir),
        collection="aw_e8", mrl=256, queue_dir=str(tmp_path / "queue"))

    manifest = json.loads(
        (pathlib.Path(result["pack_dir"]) / "manifest.json").read_text())
    assert manifest["pack_chunks"] == str(chunks_dir) + "-consent-stamped"
    assert "lang_backfilled_from" not in manifest
    assert "lang_backfill_report" not in manifest


def test_cli_backfill_lang_flag_reaches_the_manifest(tmp_path, monkeypatch):
    from cytir.acquire import fetch_process as FP  # noqa: PLC0415

    monkeypatch.setattr(FP, "detect_lang", lambda text, url, default: ("en", True))
    monkeypatch.setattr(compose, "embed_stage", lambda *a, **k: 0)

    rows = _real_rows()
    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    queue_dir = tmp_path / "queue"
    _write_chunk_shard(chunks_dir, rows)
    _write_real_vector_shard(vectors_dir, [r["point_id"] for r in rows])

    argv = ["--pack-id", "test-eric-cli-lang-backfill",
           "--family", "eric_official_pdf",
           "--chunks-dir", str(chunks_dir), "--vectors-dir", str(vectors_dir),
           "--collection", "aw_e8", "--mrl", "256",
           "--queue-dir", str(queue_dir), "--backfill-lang"]
    assert compose.main(argv) == 0

    manifest = json.loads(
        (queue_dir / "test-eric-cli-lang-backfill" / "manifest.json").read_text())
    assert manifest["pack_chunks"] == str(chunks_dir) + "-lang-backfilled-consent-stamped"
    assert manifest["lang_backfill_report"]["filled"] == 2


def test_compose_pack_drop_undetected_lang_writes_the_waiver(tmp_path):
    """`drop_undetected_lang` mechanically implements Tim's own standing
    ruling for eric_official_pdf (plan.md rule 12, ADR-113) -- the SAME
    `lang_presence_waiver` manifest field CC tranche 1's Tim-issued waiver
    used (task 652), consumed unchanged by `promote_queue.py` at drain
    time. Not itself a waiver decision: the shape is Tim's own."""
    rows = _real_rows()
    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=1024)

    result = compose.compose_pack(
        pack_id="test-eric-pack-drop-lang", family="eric_official_pdf",
        chunks_dir=str(chunks_dir), vectors_dir=str(vectors_dir),
        collection="aw_e8", mrl=256, queue_dir=str(tmp_path / "queue"),
        drop_undetected_lang=True)

    manifest = json.loads(
        (pathlib.Path(result["pack_dir"]) / "manifest.json").read_text())
    waiver = manifest["lang_presence_waiver"]
    assert waiver["issuer"] == "Tim"
    assert waiver["scope"] == "eric_official_pdf"
    assert "ADR-113" in waiver["rule"] or "plan.md rule 12" in waiver["rule"]


def test_compose_pack_omits_the_waiver_when_not_given(tmp_path):
    rows = _real_rows()
    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=1024)

    result = compose.compose_pack(
        pack_id="test-eric-pack-no-waiver", family="eric_official_pdf",
        chunks_dir=str(chunks_dir), vectors_dir=str(vectors_dir),
        collection="aw_e8", mrl=256, queue_dir=str(tmp_path / "queue"))

    manifest = json.loads(
        (pathlib.Path(result["pack_dir"]) / "manifest.json").read_text())
    assert "lang_presence_waiver" not in manifest


def test_compose_pack_backfill_and_drop_together_land_the_full_eric_shape(
        tmp_path, monkeypatch):
    """The real ERIC call shape (task 669/C): `backfill_lang` fills what
    the detector can confidently place, `drop_undetected_lang` waives
    presence for the rest -- together, exactly what `embed_compose_
    enqueue.sh` now passes for every ERIC tranche."""
    from cytir.acquire import fetch_process as FP  # noqa: PLC0415

    monkeypatch.setattr(FP, "detect_lang", lambda text, url, default: ("en", True))

    rows = _real_rows()
    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    _write_chunk_shard(chunks_dir, rows)
    _write_real_vector_shard(vectors_dir, [r["point_id"] for r in rows])

    result = compose.compose_pack(
        pack_id="test-eric-pack-full-shape", family="eric_official_pdf",
        chunks_dir=str(chunks_dir), vectors_dir=str(vectors_dir),
        collection="aw_e8", mrl=256, queue_dir=str(tmp_path / "queue"),
        backfill_lang=True, drop_undetected_lang=True,
        language_scope_filter=True)

    manifest = json.loads(
        (pathlib.Path(result["pack_dir"]) / "manifest.json").read_text())
    assert manifest["pack_chunks"] == str(chunks_dir) + "-lang-backfilled-consent-stamped"
    assert manifest["lang_backfill_report"]["filled"] == 2
    assert manifest["lang_presence_waiver"]["issuer"] == "Tim"
    assert manifest["language_scope_filter"] is True


def test_compose_pack_refuses_wrong_sealed_dim(tmp_path):
    rows = _real_rows()
    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=256)  # not the sealed 1024

    with pytest.raises(compose.ComposeError, match="sealed 1024-dim"):
        compose.compose_pack(
            pack_id="test-eric-pack-2", family="eric_official_pdf",
            chunks_dir=str(chunks_dir), vectors_dir=str(vectors_dir),
            collection="aw_e8", mrl=256, queue_dir=str(tmp_path / "queue"))


def test_compose_pack_refuses_identity_mismatch_before_writing_anything(
        tmp_path):
    rows = _real_rows()
    rows[0] = {**rows[0], "point_id": "00000000-0000-0000-0000-000000000000"}
    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    queue_dir = tmp_path / "queue"
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=1024)

    with pytest.raises(compose.ComposeError, match="identity check FAILED"):
        compose.compose_pack(
            pack_id="test-eric-pack-3", family="eric_official_pdf",
            chunks_dir=str(chunks_dir), vectors_dir=str(vectors_dir),
            collection="aw_e8", mrl=256, queue_dir=str(queue_dir))
    assert not (queue_dir / "test-eric-pack-3").exists(), (
        "a refused identity check must not leave a partial pack dir behind")


def test_default_anchor_matches_the_producer_scheme():
    """`document:{document_id}:chunk-{chunk_ordinal}` -- the exact anchor
    `stage_structured_markdown_for_embedding.py` mints (task 657 NEXT.md
    point 3; confirmed against the real fixture rows here)."""
    rows = _real_rows()
    assert compose.default_anchor(rows[0]) == (
        "document:5412edb0c48cd1bc4fd9699993ee040f1cf7a8cadc069f1d9d0c81915"
        "b15512d:chunk-0")


# --------------------------------------------------------------- task 664
# CC (`cc_authority_frontier_w7`) real-record regression: growth_pending_
# compose.py's unattended CC discovery routed tranche4/5 through this
# module for the FIRST time (tranche2/3's original landing predates that
# discovery and used a bespoke script) and every attempt failed with
# `KeyError: 'document_id'` -- CC chunks (cytir.core.common.Passage) never
# carry `document_id`/`chunk_ordinal`; identity is per-URL via the row's own
# `anchor` field. Fixed by `raw_anchor_field` + `FAMILY_ANCHOR_FN`.

CC_REAL_ROWS_PATH = HERE.parent / "tests" / "fixtures" / "index_compose" / "task664_cc_real_rows.jsonl"


def _cc_real_rows() -> list[dict]:
    return [json.loads(line) for line in
           CC_REAL_ROWS_PATH.read_text().splitlines() if line.strip()]


def test_cc_real_fixture_has_two_distinct_rows_with_no_document_id():
    rows = _cc_real_rows()
    assert len(rows) == 2
    assert rows[0]["point_id"] == "67fdf3af-7a9a-59b0-b605-7934628646f9"
    assert rows[1]["point_id"] == "a29c34b6-829d-584c-a6cb-c54d419d3f11"
    for row in rows:
        assert "document_id" not in row
        assert "chunk_ordinal" not in row


def test_default_anchor_keyerrors_on_a_real_cc_row():
    """Proves the ORIGINAL bug is real, on a real record, not a synthesized
    one: the ERIC-shaped anchor function cannot even be called on a CC row."""
    rows = _cc_real_rows()
    with pytest.raises(KeyError, match="document_id"):
        compose.default_anchor(rows[0])


def test_anchor_fn_for_family_routes_cc_to_the_raw_anchor_field():
    assert (compose.anchor_fn_for_family("cc_authority_frontier_w7")
           is compose.raw_anchor_field)


def test_anchor_fn_for_family_keeps_default_anchor_for_every_other_family():
    """No regression: ERIC/SE and anything not explicitly listed keep the
    document/chunk-ordinal scheme."""
    for family in ("eric_official_pdf", "stackoverflow_qa",
                  "de_openlegaldata_v1", "some_future_family"):
        assert compose.anchor_fn_for_family(family) is compose.default_anchor


def test_verify_identity_matches_real_cc_rows_via_the_raw_anchor_field(
        tmp_path):
    rows = _cc_real_rows()
    chunks_dir = tmp_path / "chunks"
    _write_chunk_shard(chunks_dir, rows)

    identity = compose.verify_identity(
        str(chunks_dir), anchor_fn=compose.anchor_fn_for_family(
            "cc_authority_frontier_w7"))
    assert identity["checked"] == identity["matched"] == 2
    assert identity["mismatches"] == []


def test_compose_pack_composes_a_real_cc_pack_without_document_id(tmp_path):
    """The regression test: this call raised `KeyError: 'document_id'`
    before the fix -- growth_pending_compose.py's real CC discovery makes
    this exact call (family='cc_authority_frontier_w7', no anchor_fn
    override), so the test does the same."""
    rows = _cc_real_rows()
    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    queue_dir = tmp_path / "queue"
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=1024)

    result = compose.compose_pack(
        pack_id="test-cc-pack-1", family="cc_authority_frontier_w7",
        chunks_dir=str(chunks_dir), vectors_dir=str(vectors_dir),
        collection="aw_e8", mrl=256, queue_dir=str(queue_dir))

    manifest = json.loads(
        (pathlib.Path(result["pack_dir"]) / "manifest.json").read_text())
    assert manifest["identity_check"] == "pack-declared"
    assert "2/2" in manifest["identity_reason"]
    assert "anchor" in manifest["identity_reason"]


# ---------------------------------------------- task 663: ADR-113 ceiling

def _seed_queued_pack(queue_dir: pathlib.Path, pack_id: str, family: str,
                      rows: int, *, with_rows_field: bool = True) -> None:
    """A minimal already-queued pack: just a manifest.json, matching what
    `family_running_total` actually reads (it never scrolls a real chunks
    dir unless the manifest omits `rows`)."""
    pack_dir = queue_dir / pack_id
    pack_dir.mkdir(parents=True)
    manifest = {"schema": "aw.promotion-queue-pack.v1", "pack_id": pack_id,
               "family": family}
    if with_rows_field:
        manifest["rows"] = rows
    (pack_dir / "manifest.json").write_text(json.dumps(manifest))


def test_compose_pack_writes_rows_and_no_ceiling_key_by_default(tmp_path):
    rows = _real_rows()
    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    queue_dir = tmp_path / "queue"
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=1024)

    result = compose.compose_pack(
        pack_id="no-ceiling-pack", family="eric_official_pdf",
        chunks_dir=str(chunks_dir), vectors_dir=str(vectors_dir),
        collection="aw_e8", mrl=256, queue_dir=str(queue_dir))

    assert result["manifest"]["rows"] == 2
    assert "family_ceiling" not in result["manifest"]


def test_family_running_total_sums_only_the_matching_family(tmp_path):
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    _seed_queued_pack(queue_dir, "eric-a", "eric_official_pdf", 1000)
    _seed_queued_pack(queue_dir, "eric-b", "eric_official_pdf", 2500)
    _seed_queued_pack(queue_dir, "stackoverflow-a", "stackoverflow_qa", 999999)

    result = compose.family_running_total(str(queue_dir), "eric_official_pdf")
    assert result["total_rows"] == 3500
    assert {p["pack_id"] for p in result["packs"]} == {"eric-a", "eric-b"}


def test_family_running_total_excludes_the_named_pack_id(tmp_path):
    """A resume/re-seal of the SAME pack_id must not count itself twice
    against the ceiling it is about to be checked against."""
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    _seed_queued_pack(queue_dir, "eric-a", "eric_official_pdf", 1000)
    _seed_queued_pack(queue_dir, "eric-b", "eric_official_pdf", 2500)

    result = compose.family_running_total(
        str(queue_dir), "eric_official_pdf", exclude_pack_id="eric-b")
    assert result["total_rows"] == 1000
    assert [p["pack_id"] for p in result["packs"]] == ["eric-a"]


def test_family_running_total_counts_real_chunks_when_rows_field_absent(
        tmp_path):
    """A pack enqueued before this field existed still counts correctly --
    a real record count, not treated as zero rows."""
    queue_dir = tmp_path / "queue"
    chunks_dir = tmp_path / "legacy-chunks"
    _write_chunk_shard(chunks_dir, _real_rows())
    pack_dir = queue_dir / "legacy-pack"
    pack_dir.mkdir(parents=True)
    (pack_dir / "manifest.json").write_text(json.dumps({
        "schema": "aw.promotion-queue-pack.v1", "pack_id": "legacy-pack",
        "family": "eric_official_pdf", "pack_chunks": str(chunks_dir)}))

    result = compose.family_running_total(str(queue_dir), "eric_official_pdf")
    assert result["total_rows"] == 2


def test_compose_pack_refuses_past_the_ceiling_and_writes_nothing(tmp_path):
    rows = _real_rows()
    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=1024)
    _seed_queued_pack(queue_dir, "eric-already-queued", "eric_official_pdf",
                      4499999)

    with pytest.raises(compose.ComposeError, match="exceeds the ceiling"):
        compose.compose_pack(
            pack_id="eric-over-ceiling", family="eric_official_pdf",
            chunks_dir=str(chunks_dir), vectors_dir=str(vectors_dir),
            collection="aw_e8", mrl=256, queue_dir=str(queue_dir),
            ceiling=4_500_000)
    assert not (queue_dir / "eric-over-ceiling").exists(), (
        "a ceiling refusal must not leave a partial pack dir behind -- no "
        "manifest, no READY")


def test_compose_pack_allows_landing_exactly_at_the_ceiling(tmp_path):
    rows = _real_rows()
    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=1024)
    _seed_queued_pack(queue_dir, "eric-already-queued", "eric_official_pdf",
                      4_499_998)

    result = compose.compose_pack(
        pack_id="eric-lands-on-ceiling", family="eric_official_pdf",
        chunks_dir=str(chunks_dir), vectors_dir=str(vectors_dir),
        collection="aw_e8", mrl=256, queue_dir=str(queue_dir),
        ceiling=4_500_000)
    fc = result["manifest"]["family_ceiling"]
    assert fc == {"ceiling": 4_500_000, "prior_family_total": 4_499_998,
                 "this_pack_rows": 2, "projected_family_total": 4_500_000,
                 "prior_packs_counted": ["eric-already-queued"]}
    assert (queue_dir / "eric-lands-on-ceiling" / "READY").exists()


def test_compose_pack_resuming_the_same_pack_id_does_not_self_block(tmp_path):
    """Re-running compose for a pack that is ALREADY the thing sitting at
    the ceiling (a resume after some other failure) must not refuse on
    seeing its own prior manifest as if it were separate rows."""
    rows = _real_rows()
    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=1024)
    _seed_queued_pack(queue_dir, "eric-resumed", "eric_official_pdf", 4_499_998)

    result = compose.compose_pack(
        pack_id="eric-resumed", family="eric_official_pdf",
        chunks_dir=str(chunks_dir), vectors_dir=str(vectors_dir),
        collection="aw_e8", mrl=256, queue_dir=str(queue_dir),
        ceiling=4_500_000)
    assert result["manifest"]["family_ceiling"]["prior_family_total"] == 0
    assert result["manifest"]["rows"] == 2


def test_ceiling_is_scoped_to_the_named_family_only(tmp_path):
    """A different family's rows sitting in the same queue must never count
    against ERIC's ceiling."""
    rows = _real_rows()
    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=1024)
    _seed_queued_pack(queue_dir, "so-huge-pack", "stackoverflow_qa", 10_000_000)

    result = compose.compose_pack(
        pack_id="eric-unaffected", family="eric_official_pdf",
        chunks_dir=str(chunks_dir), vectors_dir=str(vectors_dir),
        collection="aw_e8", mrl=256, queue_dir=str(queue_dir),
        ceiling=4_500_000)
    assert result["manifest"]["family_ceiling"]["prior_family_total"] == 0


# --------------------------------------------------- CLI: --family-payload
# / --title-from (task 664, 2026-09-26). `compose_pack()`/
# `embed_compose_enqueue()` (the Python functions) always accepted
# `family_payload`/`title_from` -- the CLI simply never exposed either one,
# so every caller going through `python -m cytir.index.compose` (ERIC's
# embed_compose_enqueue.sh, growth_pending_compose.py) had no way to set
# them. Both ERIC and stackoverflow_qa rows carry no top-level
# source_type/trust_tier/title, so every pack composed through the CLI
# refused at promotion with PayloadCarryError (0% carry on all three).

def test_cli_parses_family_payload_json_and_forwards_it(monkeypatch):
    seen = {}

    def fake_embed_compose_enqueue(**kwargs):
        seen.update(kwargs)
        return {"ready_sha": "x"}
    monkeypatch.setattr(compose, "embed_compose_enqueue", fake_embed_compose_enqueue)

    compose.main([
        "--pack-id", "p1", "--family", "stackoverflow_qa",
        "--chunks-dir", "/tmp/c", "--vectors-dir", "/tmp/v",
        "--collection", "aw_e8", "--mrl", "256",
        "--family-payload", '{"source_type": "community_qa", "trust_tier": 0.6}',
        "--title-from", "source_metadata.title",
    ])
    assert seen["family_payload"] == {"source_type": "community_qa", "trust_tier": 0.6}
    assert seen["title_from"] == ["source_metadata.title"]


def test_cli_title_from_is_repeatable(monkeypatch):
    seen = {}

    def fake_embed_compose_enqueue(**kwargs):
        seen.update(kwargs)
        return {"ready_sha": "x"}
    monkeypatch.setattr(compose, "embed_compose_enqueue", fake_embed_compose_enqueue)

    compose.main([
        "--pack-id", "p1", "--family", "eric_official_pdf",
        "--chunks-dir", "/tmp/c", "--vectors-dir", "/tmp/v",
        "--collection", "aw_e8", "--mrl", "256",
        "--title-from", "source_metadata.title",
        "--title-from", "title",
    ])
    assert seen["title_from"] == ["source_metadata.title", "title"]


def test_cli_omitting_family_payload_forwards_none_not_an_error(monkeypatch):
    """No regression: every EXISTING caller (ERIC's embed_compose_enqueue.sh
    before this task, growth_pending_compose.py) never passes
    --family-payload at all."""
    seen = {}

    def fake_embed_compose_enqueue(**kwargs):
        seen.update(kwargs)
        return {"ready_sha": "x"}
    monkeypatch.setattr(compose, "embed_compose_enqueue", fake_embed_compose_enqueue)

    compose.main([
        "--pack-id", "p1", "--family", "eric_official_pdf",
        "--chunks-dir", "/tmp/c", "--vectors-dir", "/tmp/v",
        "--collection", "aw_e8", "--mrl", "256",
    ])
    assert seen["family_payload"] is None
    assert seen["title_from"] == []


def test_cli_family_payload_reaches_the_real_manifest(tmp_path):
    """End-to-end (no mock): the CLI's --family-payload/--title-from
    actually land in the written manifest.json, matching what
    aw_b1_projection.apply_family_payload/lift_published read at plan()
    time."""
    rows = _real_rows()
    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    queue_dir = tmp_path / "queue"
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=1024)

    compose.main([
        "--pack-id", "test-so-cli-pack", "--family", "stackoverflow_qa",
        "--chunks-dir", str(chunks_dir), "--vectors-dir", str(vectors_dir),
        "--collection", "aw_e8", "--mrl", "256",
        "--queue-dir", str(queue_dir), "--skip-embed",
        "--family-license", "CC-BY-SA-4.0", "--family-payload", '{"source_type": "community_qa", "trust_tier": 0.6}',
        "--title-from", "source_metadata.title",
    ])
    manifest = json.loads((queue_dir / "test-so-cli-pack" / "manifest.json").read_text())
    assert manifest["family_payload"] == {"source_type": "community_qa", "trust_tier": 0.6}
    assert manifest["title_from"] == ["source_metadata.title"]


# ------------------------------- task 680: an unavailable language gate refuses
# Real rows sampled from the build box on 2026-09-30: two from the raw chunks of
# eric-official-pdf-tranche-030000 (refused at the drain) and two from 020000
# (landed). Both tranches carry `lang: null` on every raw row, so the ONLY thing
# that separated them was whether the detector could run in the composing process.
ERIC_REAL_ROWS = {
    "030000": HERE.parent / "tests" / "fixtures" / "index_compose" / "task680_eric_030000_real_rows.jsonl",
    "020000": HERE.parent / "tests" / "fixtures" / "index_compose" / "task680_eric_020000_real_rows.jsonl",
}


def _tranche_rows(tranche: str) -> list[dict]:
    return [json.loads(line) for line in
            ERIC_REAL_ROWS[tranche].read_text().splitlines() if line.strip()]


def _compose_backfilled(tmp_path, tranche):
    rows = _tranche_rows(tranche)
    assert all(r["lang"] is None for r in rows)
    chunks_dir, vectors_dir = tmp_path / "chunks", tmp_path / "vectors"
    _write_chunk_shard(chunks_dir, rows)
    _write_real_vector_shard(vectors_dir, [r["point_id"] for r in rows])
    return dict(
        pack_id=f"test-eric-{tranche}", family="eric_official_pdf",
        chunks_dir=str(chunks_dir), vectors_dir=str(vectors_dir),
        collection="aw_e8", mrl=256, queue_dir=str(tmp_path / "queue"),
        backfill_lang=True, drop_undetected_lang=True,
        language_scope_filter=True), rows


@pytest.mark.parametrize("tranche", ["030000", "020000"])
def test_backfill_lang_refuses_and_writes_nothing_when_the_gate_cannot_run(
        tmp_path, monkeypatch, tranche):
    """030000/040000: the compose ran under the torch interpreter (numpy 2),
    fasttext could not predict, `left_gate_unavailable` == rows, and every row
    was then dropped as 'undetected' at the drain ('zero pages to check').
    An unavailable instrument is not an empty: refuse at the door."""
    from cytir.acquire import fetch_process as FP  # noqa: PLC0415

    def _gate_down(text, url, default):
        raise FP.LangGateInactive("fasttext predict failed on numpy 2.2.6")

    monkeypatch.setattr(FP, "detect_lang", _gate_down)
    kwargs, rows = _compose_backfilled(tmp_path, tranche)
    with pytest.raises(compose.ComposeError,
                       match=rf"unavailable for {len(rows)} of {len(rows)} rows"):
        compose.compose_pack(**kwargs)
    assert not list((tmp_path / "queue").rglob("READY"))
    assert not (tmp_path / "chunks-lang-backfilled").exists()


@pytest.mark.parametrize("tranche", ["030000", "020000"])
def test_backfill_lang_on_real_rows_with_a_working_gate_keeps_every_row(
        tmp_path, monkeypatch, tranche):
    from cytir.acquire import fetch_process as FP  # noqa: PLC0415

    monkeypatch.setattr(FP, "detect_lang", lambda text, url, default: ("en", True))
    kwargs, rows = _compose_backfilled(tmp_path, tranche)
    result = compose.compose_pack(**kwargs)
    report = result["manifest"]["lang_backfill_report"]
    assert report["filled"] == len(rows) and report["left_gate_unavailable"] == 0
    assert result["identity"]["matched"] == len(rows)


# ---- task 686: reseal
def _queued_pack(queue, pack_id, rows=10):
    import hashlib
    d = queue / pack_id
    d.mkdir(parents=True)
    b = json.dumps({"pack_id": pack_id, "rows": rows}, indent=1,
                   sort_keys=True).encode()
    (d / "manifest.json").write_bytes(b)
    (d / "READY").write_text(hashlib.sha256(b).hexdigest() + "\n")
    return d


def test_reseal_changes_sha_keeps_ready_consistent_and_records_why(tmp_path):
    import hashlib
    from cytir.index import reseal
    d = _queued_pack(tmp_path, "p1")
    res = reseal.reseal_pack(str(tmp_path), "p1", reason="L1 filed",
                             l1_releases=["rel-b", "rel-a"])
    new = (d / "manifest.json").read_bytes()
    assert res["new_sha"] == hashlib.sha256(new).hexdigest() != res["old_sha"]
    assert (d / "READY").read_text().strip() == res["new_sha"]
    m = json.loads(new)
    assert m["rows"] == 10
    assert m["resealed"]["l1_releases"] == ["rel-a", "rel-b"]
    assert m["resealed"]["previous_manifest_sha"] == res["old_sha"]


def test_reseal_refuses_terminal_and_inconsistent_packs(tmp_path):
    from cytir.index import reseal
    d = _queued_pack(tmp_path, "done")
    (d / "DONE").write_text("x")
    with pytest.raises(reseal.ResealError):
        reseal.reseal_pack(str(tmp_path), "done", reason="r")
    d2 = _queued_pack(tmp_path, "edited")
    (d2 / "manifest.json").write_text("{}")
    with pytest.raises(reseal.ResealError):
        reseal.reseal_pack(str(tmp_path), "edited", reason="r")


def test_family_running_total_counts_the_rows_a_pack_will_write(tmp_path):
    """The ceiling reads `rows_to_write_max`; a legacy pack (only `rows`) is read exactly as before."""
    queue_dir = tmp_path / "queue"
    _seed_queued_pack(queue_dir, "legacy", "eric_official_pdf", 1000)
    _seed_queued_pack(queue_dir, "new", "eric_official_pdf", 500)
    m = json.loads((queue_dir / "new" / "manifest.json").read_text())
    m.update(rows_staged=500, rows_to_write_max=120)           # `rows` is the alias: stage count
    (queue_dir / "new" / "manifest.json").write_text(json.dumps(m))
    got = compose.family_running_total(str(queue_dir), "eric_official_pdf")
    assert got["total_rows"] == 1000 + 120


def test_reseal_with_an_exclusion_keeps_rows_to_write_true_for_a_new_pack_only(tmp_path):
    from cytir.index import reseal
    rows = _real_rows()
    chunks_dir, vectors_dir = tmp_path / "chunks", tmp_path / "vectors"
    _write_chunk_shard(chunks_dir, rows)
    _write_vector_manifest(vectors_dir, n=2, dim=1024)
    queue = tmp_path / "q"
    compose.compose_pack(pack_id="new", family="eric_official_pdf", chunks_dir=str(chunks_dir),
                         vectors_dir=str(vectors_dir), collection="aw_e8", mrl=256,
                         queue_dir=str(queue))
    f = _trimmed_ids_file(tmp_path, [rows[0]["point_id"]])
    reseal.reseal_pack(str(queue), "new", reason="r", exclude_ids_file=str(f),
                       exclude_ids_reason="floor")
    m = json.loads((queue / "new" / "manifest.json").read_bytes())
    assert (m["rows"], m["rows_staged"], m["rows_to_write_max"]) == (2, 2, 1)
    # a pack sealed before the names existed is not given new ones
    d = _queued_pack(tmp_path / "q2", "old")
    reseal.reseal_pack(str(tmp_path / "q2"), "old", reason="r", exclude_ids_file=str(f),
                       exclude_ids_reason="floor")
    old = json.loads((d / "manifest.json").read_bytes())
    assert old["rows"] == 10 and "rows_to_write_max" not in old and "rows_staged" not in old


def test_reseal_pins_an_exclusion_file_the_drain_loader_accepts(tmp_path):
    """Task 690: excluding the ids an earlier pack already landed on both
    targets is a reseal with a sha-pinned task-680 exclusion file. What the
    reseal writes must load through the SAME loader the drain uses."""
    import hashlib
    from cytir.index import exclude_file as XF
    from cytir.index import reseal
    queue = tmp_path / "q"
    d = _queued_pack(queue, "p1")
    f = tmp_path / "overlap.jsonl"
    f.write_text(json.dumps({"point_id": "4f9e9175-9b2e-5f97-8115-23ba7ea9e4f0"})
                 + "\n")
    res = reseal.reseal_pack(str(queue), "p1", reason="smoke overlap",
                             exclude_ids_file=str(f),
                             exclude_ids_reason="already_landed_by_smoke")
    m = json.loads((d / "manifest.json").read_bytes())
    assert (d / "READY").read_text().strip() == res["new_sha"]
    assert m["exclude_ids_sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
    assert m["resealed"]["exclude_ids_added"] == res["exclude_ids_added"] == 1
    assert XF.load_exclude_ids(m["exclude_ids_file"],
                               sha256=m["exclude_ids_sha256"],
                               reason=m["exclude_ids_reason"]) == {
        "4f9e9175-9b2e-5f97-8115-23ba7ea9e4f0": "already_landed_by_smoke"}


def test_reseal_refuses_a_bad_or_second_exclusion_file(tmp_path):
    from cytir.index import reseal
    queue = tmp_path / "q"
    d = _queued_pack(queue, "p1")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    before = (d / "manifest.json").read_bytes()
    with pytest.raises(reseal.ResealError):   # excludes nothing
        reseal.reseal_pack(str(queue), "p1", reason="r",
                           exclude_ids_file=str(empty), exclude_ids_reason="x")
    f = tmp_path / "ids.jsonl"
    f.write_text(json.dumps({"point_id": "a"}) + "\n")
    with pytest.raises(reseal.ResealError):   # no reason
        reseal.reseal_pack(str(queue), "p1", reason="r",
                           exclude_ids_file=str(f))
    assert (d / "manifest.json").read_bytes() == before
    reseal.reseal_pack(str(queue), "p1", reason="r", exclude_ids_file=str(f),
                       exclude_ids_reason="x")
    with pytest.raises(reseal.ResealError):   # already names one
        reseal.reseal_pack(str(queue), "p1", reason="r2",
                           exclude_ids_file=str(f), exclude_ids_reason="y")
