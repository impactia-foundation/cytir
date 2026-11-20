import json
from pathlib import Path

import cytir.sources.rfc_editor.preserved as subject
from cytir.index.preservation_contract import _validate_stage


def test_rfc_rsync_tree_preserves_full_text_and_outlinks(tmp_path: Path):
    root = tmp_path / "run"
    raw = root / "input/rfcs"
    raw.mkdir(parents=True)
    (raw / "rfc1.txt").write_bytes(
        b"RFC one\n\nFull text https://example.org/reference.\n"
        b"Grammar example https://[not-an-ipv6-host/path.\n")
    before = root / "input/rsync-inventory.before.txt"
    after = root / "input/rsync-inventory.after.txt"
    before.write_text("rfc1.txt\n")
    after.write_text("rfc1.txt\n")
    run = subject.materialize(
        raw_dir=raw, before_list=before, after_list=after, root=root,
        module="rsync.rfc-editor.org::rfcs-text-only",
        started_at="2026-08-15T00:00:00Z",
        completed_at="2026-08-15T00:01:00Z",
        rsync_version="rsync version 3.2.7", fetch_vantage="fetch-box")
    assert run["counts"] == {"original_documents": 1,
                              "markdown_documents": 1,
                              "outlink_records": 1,
                              "quarantined_documents": 0}
    assert run == subject.materialize(
        raw_dir=raw, before_list=before, after_list=after, root=root,
        module="rsync.rfc-editor.org::rfcs-text-only",
        started_at="2026-08-15T00:00:00Z",
        completed_at="2026-08-15T00:01:00Z",
        rsync_version="rsync version 3.2.7", fetch_vantage="fetch-box")
    for name in ("original", "markdown", "outlinks", "quarantine"):
        verified = _validate_stage(run["stages"][name], name,
                                   artifact_root=root, deep=True)
        assert verified["records"] == run["stages"][name]["records"]
    metadata = json.loads((root / "input/rsync.transport.json").read_text())
    assert metadata["exit_status"] == 0
