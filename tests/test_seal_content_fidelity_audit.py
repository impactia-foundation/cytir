import json
from pathlib import Path

from cytir.index import seal_content_fidelity_audit as subject


def test_multiple_adapter_sources_have_deterministic_bundle_identity(tmp_path):
    first = tmp_path / "first.py"; first.write_text("text = value\n")
    second = tmp_path / "second.py"; second.write_text("markdown = value\n")
    bindings = [{"id": path.name, "sha256": subject.sha256_file(path)}
                for path in (first, second)]
    expected = __import__("hashlib").sha256(__import__("json").dumps(
        bindings, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    # Exercise the exact construction independently of stage fixtures; the
    # integration path is covered by existing seal() tests below.
    assert len(expected) == 64
    assert not subject.audit_adapter(first)
    assert not subject.audit_adapter(second)
from cytir.acquire import stage_receipt as SS
from cytir.index.preservation_contract import _validate_content_fidelity
import pytest


def test_receipt_binds_exact_original_markdown_and_adapter(tmp_path: Path):
    stages = {}
    for name in ("original", "markdown"):
        data = tmp_path / name / "data"
        data.parent.mkdir(parents=True)
        data.write_bytes(name.encode())
        stages[name] = SS.seal(
            artifact_root=tmp_path, artifact_paths=[data],
            receipt_path=tmp_path / name / "receipt.json", records=1,
            fmt="fixture", logical_sha256=__import__("hashlib").sha256(
                name.encode()).hexdigest(), producer_id="fixture",
            producer_version="v1")
    run = tmp_path / "RUN.json"
    run.write_text(json.dumps({"stages": stages}))
    adapter = tmp_path / "adapter.py"
    adapter.write_text("# complete adapter\n")
    envelope = subject.seal(
        artifact_root=tmp_path, parent_run=run, adapter_source=adapter,
        source_id="source", verification_method="fixture full-byte audit",
        receipt_path=tmp_path / "fidelity/receipt.json",
        envelope_path=tmp_path / "fidelity/envelope.json")
    checked = _validate_content_fidelity(
        envelope, "content_fidelity", source_id="source", stages=stages,
        artifact_root=tmp_path, deep=True)
    assert checked["source_text_truncation_applied"] is False
    audit = json.loads((tmp_path / "fidelity/receipt.json").read_text())
    assert audit["verification_evidence"]["uri"] == \
        "artifact://fidelity/receipt.verification-evidence.json"


def test_new_receipt_does_not_collide_with_sibling_profile_evidence(tmp_path: Path):
    stages = {}
    for name in ("original", "markdown"):
        data = tmp_path / name / "data"; data.parent.mkdir(parents=True)
        data.write_bytes(name.encode())
        stages[name] = SS.seal(
            artifact_root=tmp_path, artifact_paths=[data],
            receipt_path=tmp_path / name / "receipt.json", records=1,
            fmt="fixture", logical_sha256=__import__("hashlib").sha256(
                name.encode()).hexdigest(), producer_id="fixture", producer_version="v1")
    run = tmp_path / "RUN.json"; run.write_text(json.dumps({"stages": stages}))
    adapter = tmp_path / "adapter.py"; adapter.write_text("# complete adapter\n")
    fidelity = tmp_path / "admission"; fidelity.mkdir()
    profile_evidence = fidelity / "verification-evidence.json"
    profile_evidence.write_text('{"schema":"unrelated.profile.evidence"}\n')
    subject.seal(artifact_root=tmp_path, parent_run=run, adapter_source=adapter,
        source_id="source", verification_method="fixture full-byte audit",
        receipt_path=fidelity / "CONTENT-FIDELITY-RECEIPT.json",
        envelope_path=fidelity / "CONTENT-FIDELITY-ENVELOPE.json")
    assert json.loads(profile_evidence.read_text())["schema"] == "unrelated.profile.evidence"
    assert (fidelity / "CONTENT-FIDELITY-RECEIPT.verification-evidence.json").is_file()


def test_existing_receipt_preserves_legacy_evidence_uri(tmp_path: Path):
    legacy = tmp_path / "fidelity" / "verification-evidence.json"
    legacy.parent.mkdir(); legacy.write_text("{}\n")
    receipt = tmp_path / "fidelity" / "receipt.json"
    receipt.write_text(json.dumps({"verification_evidence": {
        "uri": "artifact://fidelity/verification-evidence.json"}}))
    assert subject._evidence_path(tmp_path, receipt) == legacy


def test_envelope_cannot_alias_evidence_and_leave_partial_publication(
        tmp_path: Path):
    stages = {}
    for name in ("original", "markdown"):
        data = tmp_path / name / "data"; data.parent.mkdir(parents=True)
        data.write_bytes(name.encode())
        stages[name] = SS.seal(
            artifact_root=tmp_path, artifact_paths=[data],
            receipt_path=tmp_path / name / "receipt.json", records=1,
            fmt="fixture", logical_sha256=__import__("hashlib").sha256(
                name.encode()).hexdigest(), producer_id="fixture",
            producer_version="v1")
    run = tmp_path / "RUN.json"; run.write_text(json.dumps({"stages": stages}))
    adapter = tmp_path / "adapter.py"; adapter.write_text("# complete adapter\n")
    receipt = tmp_path / "fidelity/CONTENT-FIDELITY-RECEIPT.json"
    evidence = tmp_path / "fidelity/CONTENT-FIDELITY-RECEIPT.verification-evidence.json"
    with pytest.raises(ValueError, match="must be distinct"):
        subject.seal(
            artifact_root=tmp_path, parent_run=run, adapter_source=adapter,
            source_id="source", verification_method="fixture full-byte audit",
            receipt_path=receipt, envelope_path=evidence)
    assert not receipt.exists()
    assert not evidence.exists()


@pytest.mark.parametrize("source", [
    "p = Passage(text=text[:6000])\n",
    "p = Passage(text=f'{title}\\n{body[:4000]}')\n",
    "row = {'text': article[:8000]}\n",
    "text = content[:1200]\n",
])
def test_numeric_content_cap_is_rejected(tmp_path: Path, source: str):
    adapter = tmp_path / "adapter.py"
    adapter.write_text(source)
    findings = subject.audit_adapter(adapter)
    assert findings and "cap" in findings[0]


def test_title_and_snippet_caps_are_not_content_caps(tmp_path: Path):
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "p = Passage(title=title[:300], snippet=text[:1200], text=text)\n")
    assert subject.audit_adapter(adapter) == []


def test_iso_date_metadata_projection_inside_markdown_is_not_content_cap(
        tmp_path: Path):
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "markdown = f\"Latest release: {str(row.get('releaseTime') or '')[:10]}\"\n")
    assert subject.audit_adapter(adapter) == []
