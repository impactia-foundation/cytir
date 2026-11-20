import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

from cytir.acquire.deterministic_gzip import DeterministicGzipWriter
from cytir.acquire import stage_receipt as SS
from cytir.sources.jorf.recover_archives import (
    _artifact, _record_kind, _scan_archive, finalize,
)


def _tar(path: Path, rows: list[tuple[str, bytes]]) -> dict:
    with tarfile.open(path, "w:gz") as archive:
        for name, data in rows:
            info = tarfile.TarInfo(name); info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return {"uri": f"artifact://{path.name}", "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_record_kind_is_source_path_specific():
    assert _record_kind("x/texte/struct/JORF/TEXT/a/JORFTEXT1.xml") == "struct"
    assert _record_kind("x/texte/version/JORF/TEXT/a/JORFTEXT1.xml") == "version"
    assert _record_kind("x/article/JORFARTI1.xml") is None


def test_partition_scan_preserves_xml_and_explicit_suppression(tmp_path: Path):
    struct = b"<TEXTELR><META><ID>JORFTEXT1</ID></META></TEXTELR>"
    version = b"<TEXTE_VERSION><META><ID>JORFTEXT1</ID></META></TEXTE_VERSION>"
    archive = tmp_path / "JORF_20260816-000000.tar.gz"
    ref = _tar(archive, [
        ("x/liste_suppression_jorf.dat",
         b"jorf/global/texte/version/JORF/TEXT/a/JORFTEXT1\n"),
        ("x/jorf/global/texte/struct/JORF/TEXT/a/JORFTEXT1.xml", struct),
        ("x/jorf/global/texte/version/JORF/TEXT/a/JORFTEXT1.xml", version),
        ("x/jorf/global/texte/struct/JORF/TEXT/a/JORFTEXT2.xml", b"ignored"),
    ])
    events = tmp_path / "events.jsonl.gz"
    writer = DeterministicGzipWriter(events)
    result = _scan_archive(archive, ref, {"JORFTEXT1"}, 7, writer)
    writer.close()
    with gzip.open(events, "rt", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    assert result["target_suppressions"] == 1
    assert result["target_xml_upserts"] == 2
    assert [row["event"] for row in rows] == ["delete", "upsert", "upsert"]
    upsert = rows[1]
    assert base64.b64decode(upsert["xml_base64"]) == struct
    assert upsert["archive_ordinal"] == 7


def test_partition_scan_rejects_source_hash_drift(tmp_path: Path):
    archive = tmp_path / "JORF_20260816-000000.tar.gz"
    ref = _tar(archive, [("x/texte/struct/x/JORFTEXT1.xml", b"x")])
    ref["sha256"] = "0" * 64
    writer = DeterministicGzipWriter(tmp_path / "events.jsonl.gz")
    try:
        try:
            _scan_archive(archive, ref, {"JORFTEXT1"}, 0, writer)
        except ValueError as exc:
            assert "bytes/SHA drift" in str(exc)
        else:
            raise AssertionError("hash drift accepted")
    finally:
        writer.close()


def test_finalize_applies_suppression_before_same_increment_upsert(tmp_path: Path):
    frontier = tmp_path / "frontier.jsonl.gz"
    writer = DeterministicGzipWriter(frontier)
    writer.write({"jorftext_id": "JORFTEXT1"}); writer.close()
    plan_root = tmp_path / "plan"; plan_root.mkdir()
    plan = {
        "schema": "aw.jorf-exact-archive-recovery-plan.v1",
        "identity_frontier": _artifact(frontier, tmp_path),
        "partitions": [{"partition": 0, "archive_start": 0, "archive_end": 1}],
    }
    plan_path = plan_root / "PLAN.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n")
    part = tmp_path / "parts/partition-0000"; part.mkdir(parents=True)
    events_path = part / "events.jsonl.gz"
    event_writer = DeterministicGzipWriter(events_path)
    struct = b"<TEXTELR><META><ID>JORFTEXT1</ID></META><STRUCT><LIEN_ART id='JORFARTI1'/></STRUCT></TEXTELR>"
    version = b"<TEXTE_VERSION><META><ID>JORFTEXT1</ID></META></TEXTE_VERSION>"
    # Deliberately write the upsert before deletion.  phase=0 must still apply
    # the deletion first for this increment, leaving the upsert as final state.
    for kind, payload in (("version", version), ("struct", struct)):
        event_writer.write({"schema": "aw.jorf-archive-state-event.v1",
            "event": "upsert", "phase": 1, "archive_ordinal": 1,
            "archive_uri": "artifact://inc.tar.gz", "member_ordinal": 1,
            "member_name": f"x/texte/{kind}/JORFTEXT1.xml",
            "jorftext_id": "JORFTEXT1", "record_kind": kind,
            "xml_bytes": len(payload), "xml_sha256": hashlib.sha256(payload).hexdigest(),
            "xml_base64": base64.b64encode(payload).decode()})
    event_writer.write({"schema": "aw.jorf-archive-state-event.v1",
        "event": "delete", "phase": 0, "archive_ordinal": 1,
        "archive_uri": "artifact://inc.tar.gz", "member_ordinal": 99,
        "line_ordinal": 0, "member_name": "liste_suppression_jorf.dat",
        "suppressed_path": "texte/version/JORFTEXT1", "jorftext_id": "JORFTEXT1",
        "record_kind": "version"})
    event_writer.close()
    stage = SS.seal(artifact_root=tmp_path, artifact_paths=[events_path],
        receipt_path=part / "stages/events/receipt.json", records=3,
        fmt="test", logical_sha256=hashlib.sha256(b"events").hexdigest(),
        producer_id="test", producer_version="test")
    run = {"schema": "aw.jorf-exact-archive-recovery-partition-run.v1",
           "status": "complete", "partition": 0, "stages": {"events": stage}}
    (part / "RUN.json").write_text(json.dumps(run, sort_keys=True) + "\n")
    result = finalize(artifact_root=tmp_path, plan_path=plan_path,
                      partitions_root=tmp_path / "parts", output=tmp_path / "final")
    assert result["status"] == "source-records-complete-full-content-pending"
    assert result["counts"] == {"production_jorftext_ids": 1, "recovered": 1,
        "unresolved": 0, "relationships": 1, "partitions": 1}
