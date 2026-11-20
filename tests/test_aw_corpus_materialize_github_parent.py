import json
from pathlib import Path

from cytir.corpus import materialize_github_parent as subject
from cytir.sources.github_repo_docs import acquire as GH
from cytir.index.preservation_contract import _validate_stage


def build_root(tmp_path: Path) -> None:
    api = GH.store_artifact(tmp_path, "original/api", "json", b"{}\n")
    original = GH.store_artifact(tmp_path, "original/blobs", "raw", b"# raw\n")
    markdown = GH.store_artifact(tmp_path, "markdown", "md", b"# Full markdown\n")
    outlink = GH.store_artifact(tmp_path, "outlinks", "json", (
        json.dumps({"links": [{"source_url": "https://github.com/o/r",
                               "target_url": "https://example.org/x",
                               "anchor_or_context": "X"}]}) + "\n").encode())
    complete = {
        "schema": GH.SCHEMA, "status": "complete", "repository": "o/r",
        "revision": "a" * 40,
        "tree": {"commit_response": api, "tree_response": api},
        "counts": {"original_documents": 1, "markdown_documents": 1,
                   "outlink_documents": 1, "outlinks": 1},
        "documents": [{"source_path": "README.md",
                       "source_url": "https://github.com/o/r/blob/a/README.md",
                       "original": original, "markdown": markdown,
                       "outlinks": {**outlink, "records": 1}}],
    }
    from cytir.extract import github_agent_resources as R
    live_clock = R.utc_now
    R.utc_now = lambda: "2026-10-30T00:00:00Z"   # the receipt stamps acquired_at; pin it
    try:
        quarantined = GH.quarantine_receipt(
            {"owner_repo": "gone/r", "branch": "main"}, "main",
            "repository_unavailable", [api], "404")
    finally:
        R.utc_now = live_clock
    receipts = tmp_path / "receipts"
    GH.atomic_json(receipts / GH.receipt_name("o/r"), complete)
    GH.atomic_json(receipts / GH.receipt_name("gone/r"), quarantined)
    GH.atomic_json(tmp_path / "RUN.json", {
        "schema": "aw.github-doc-acquisition-run.v1", "status": "complete",
        "counts": {"queue_repositories": 2, "terminal_repositories": 2,
                   "remaining_repositories": 0, "attempt_failures": 0,
                   "complete_repositories": 1, "quarantined_repositories": 1,
                   "markdown_documents": 1, "outlinks": 1},
    })


def test_terminal_acquisition_closes_markdown_outlinks_and_quarantine(tmp_path: Path):
    build_root(tmp_path)
    result = subject.materialize(root=tmp_path)
    assert result["counts"] == {
        "queue_repositories": 2, "complete_repositories": 1,
        "quarantined_repositories": 1, "markdown_documents": 1,
        "outlinks": 1,
    }
    for name, count in (("original", 2), ("markdown", 1),
                        ("outlinks", 1), ("quarantine", 1)):
        checked = _validate_stage(result["stages"][name], name,
                                  artifact_root=tmp_path, deep=True)
        assert checked["records"] == count


# Task 1594 parity pin. Frozen ONCE from the pre-move driver (the module as it
# stood before the move, run on this same fake root); the package must reproduce it.
_PRODUCER = "pipeline/materialize_github_acquisition_parent.py"  # frozen receipt label
_VERSION = "github-acquisition-parent.v1"
_EXPECTED_STAGES = {
    "original": (2, "terminal-github-api-source-evidence-tree",
                 "0eca4da27135b2611c11c1e3dcf6244858399e01a6399fb0a61108fd5305b340"),
    "markdown": (1, "canonical-jsonl+deterministic-gzip",
                 "a5275a7ba08fafcd819c6d1ab04128dfe191de70ef980720e77430db597f33d3"),
    "outlinks": (1, "canonical-jsonl+deterministic-gzip",
                 "5a9ce8c3749ac9864539a86d030b82746b094cbbc58c98e27126a73dc645e4bd"),
    "quarantine": (1, "canonical-jsonl+deterministic-gzip",
                   "ecfca9961de728073b4a538565cdf9acc6d36172007cb9e993581797f79997ba"),
}
_EXPECTED_RUN_SHA256 = "ae532b1233b70f4ca148d5726f725b13a276347682315e219b39f67579d69c17"


def test_parity_pin_stage_manifests_and_run_json_match_pre_move_driver(tmp_path: Path):
    import hashlib
    build_root(tmp_path)
    assert subject.PRODUCER == _PRODUCER and subject.VERSION == _VERSION
    assert subject.main(["--root", str(tmp_path)]) == 0
    run_path = tmp_path / "derived-parent-v1" / "RUN.json"
    run = json.loads(run_path.read_text())
    assert run["schema"] == "aw.github-derived-parent-run.v1"
    assert run["status"] == "ready_for_current_policy_chunking"
    assert run["counts"] == {"queue_repositories": 2, "complete_repositories": 1,
                             "quarantined_repositories": 1, "markdown_documents": 1,
                             "outlinks": 1}
    assert set(run["stages"]) == set(_EXPECTED_STAGES)
    for name, (records, fmt, logical) in _EXPECTED_STAGES.items():
        stage = run["stages"][name]
        assert stage["producer"] == {"id": _PRODUCER, "version": _VERSION}
        assert (stage["records"], stage["format"], stage["logical_sha256"]) == (
            records, fmt, logical)
    assert hashlib.sha256(run_path.read_bytes()).hexdigest() == _EXPECTED_RUN_SHA256
