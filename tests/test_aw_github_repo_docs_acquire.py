import gzip
import hashlib
import json
from pathlib import Path

from cytir.sources.github_repo_docs import acquire as subject
import httpx


def test_select_documents_pointer_and_deep():
    nodes = [
        {"type": "blob", "path": "README.md", "sha": "a" * 40},
        {"type": "blob", "path": "docs/guide.md", "sha": "b" * 40},
        {"type": "blob", "path": "CHANGELOG.md", "sha": "c" * 40},
        {"type": "blob", "path": "src/main.py", "sha": "d" * 40},
    ]
    assert [x["path"] for x in subject.select_documents(nodes, False, 6)] == ["README.md"]
    assert [x["path"] for x in subject.select_documents(nodes, True, 6)] == [
        "docs/guide.md", "README.md"
    ]


def test_empty_original_is_content_addressable_and_verifiable(tmp_path: Path):
    artifact = subject.store_artifact(tmp_path, "original/blobs", "raw", b"")
    assert artifact["body_bytes"] == 0
    assert artifact["body_sha256"] == subject.sha256_bytes(b"")
    subject.validate_artifact(tmp_path, artifact)


def test_git_blob_api_fallback_decodes_and_verifies_tree_identity():
    raw = b"# Git blob fallback\n"
    sha1 = hashlib.sha1(
        f"blob {len(raw)}\0".encode() + raw, usedforsecurity=False
    ).hexdigest()
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://api.github.com/repos/o/r/git/blobs/x"),
        json={"encoding": "base64", "content": __import__("base64").b64encode(raw).decode()},
    )
    assert subject.decode_git_blob(response, sha1) == raw
    try:
        subject.decode_git_blob(response, "0" * 40)
    except subject.AcquisitionError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("tree/blob identity drift must fail")


def test_markdown_normalizes_rst_and_carries_no_provenance():
    """Task 517: the four-line `aw_*` block used to be the document's head, and
    the chunker made it the head of chunk 0's embedding on 72,330 staged rows.
    The normalization is unchanged; the block is gone."""
    value = subject.source_to_markdown(
        b"Title\r\n=====\r\nSee `AW <https://websear.ch>`_.\r\n",
        source_path="docs/index.rst",
    ).decode()
    assert not value.startswith("---")
    assert "aw_parent_sha256" not in value
    assert value.startswith("# Title")
    assert "[AW](https://websear.ch)" in value
    assert "\r" not in value


def test_the_parent_binding_is_record_metadata_now():
    assert subject.source_provenance(
        owner_repo="impactia/aw", revision="a" * 40,
        source_path="docs/index.rst", parent_sha256="b" * 64) == {
            "repository": "impactia/aw", "revision": "a" * 40,
            "source_path": "docs/index.rst", "parent_sha256": "b" * 64}


def test_the_title_is_the_first_heading_else_the_basename():
    """Measured against 74,794 production documents: first ATX heading
    anywhere, else the path basename, agrees with the collection this build
    replaces on 90.26% and is better on 5,033 of the rest (task 517)."""
    body = subject.source_to_markdown(
        b"Title\n=====\nbody\n", source_path="docs/index.rst").decode()
    assert subject.compose_document_title(body, "docs/index.rst") == "Title"
    assert subject.compose_document_title("no heading\n", "docs/index.rst") \
        == "index.rst"


def test_outlinks_include_relative_and_receipt_can_be_empty():
    source = "https://github.com/o/r/blob/" + "a" * 40 + "/docs/readme.md"
    links = subject.extract_outlinks(
        "[Guide](guide.md) [Site](https://example.org/x) <mailto:a@example.org>", source)
    assert [x["target_url"] for x in links] == [
        "https://github.com/o/r/blob/" + "a" * 40 + "/docs/guide.md",
        "https://example.org/x",
        "mailto:a@example.org",
    ]
    assert subject.extract_outlinks("nothing linked", source) == []


def test_artifact_roundtrip_and_tamper_detection(tmp_path: Path):
    artifact = subject.store_artifact(tmp_path, "original/blobs", "raw", b"source")
    subject.validate_artifact(tmp_path, artifact)
    path = subject.artifact_path(tmp_path, artifact)
    assert gzip.decompress(path.read_bytes()) == b"source"
    path.write_bytes(b"tampered")
    try:
        subject.validate_artifact(tmp_path, artifact)
    except subject.AcquisitionError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("tamper must fail")


def test_terminal_receipt_resume_validation(tmp_path: Path):
    original = subject.store_artifact(tmp_path, "original/blobs", "raw", b"raw")
    markdown = subject.store_artifact(tmp_path, "markdown", "md", b"markdown")
    outlinks = subject.store_artifact(tmp_path, "outlinks", "json", b'{"links":[]}\n')
    api = subject.store_artifact(tmp_path, "original/api", "json", b"{}\n")
    receipt = {
        "schema": subject.SCHEMA,
        "status": "complete",
        "repository": "o/r",
        "tree": {"commit_response": api, "tree_response": api},
        "counts": {"original_documents": 1, "markdown_documents": 1,
                   "outlink_documents": 1, "outlinks": 0},
        "documents": [{"original": original, "markdown": markdown,
                       "outlinks": {**outlinks, "records": 0}}],
    }
    path = tmp_path / "receipts" / subject.receipt_name("o/r")
    subject.atomic_json(path, receipt)
    assert subject.load_terminal(tmp_path, "O/R") == receipt
    markdown_path = subject.artifact_path(tmp_path, markdown)
    markdown_path.unlink()
    assert subject.load_terminal(tmp_path, "o/r") is None


def test_quarantine_is_terminal_only_with_preserved_evidence(tmp_path: Path):
    api = subject.store_artifact(tmp_path, "original/api", "json", b"404\n")
    receipt = subject.quarantine_receipt(
        {"owner_repo": "gone/repo", "branch": "main"}, "main",
        "repository_unavailable", [api], "metadata=404")
    subject.validate_receipt(tmp_path, receipt, "gone/repo")
    path = tmp_path / "receipts" / subject.receipt_name("gone/repo")
    subject.atomic_json(path, receipt)
    assert subject.load_terminal(tmp_path, "gone/repo")["status"] == "quarantined"


def test_queue_rejects_casefold_duplicate(tmp_path: Path):
    path = tmp_path / "q.jsonl"
    path.write_text(
        json.dumps({"owner_repo": "Owner/Repo"}) + "\n" +
        json.dumps({"owner_repo": "owner/repo"}) + "\n")
    try:
        list(subject.iter_queue(path))
    except subject.AcquisitionError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate must fail")


def _tree_nodes():
    def blob(path, size=100):
        return {"type": "blob", "path": path, "sha": "a" * 40, "size": size}
    return [
        blob("README.md"), blob("CHANGELOG.md"), blob("NOTES.txt"), blob("LICENSE.md"),
        blob("Docs/Intro.MD"), blob("docs/guide/a.mdx"), blob("docs/guide/b.rst"),
        blob("doc/c.txt"), blob("website/docs/d.md"), blob("docs/CHANGELOG.md"),
        blob("docs/i18n/fr/a.md"), blob("docs/zh/a.md"), blob("docs/node_modules/x.md"),
        blob("docs/big.md", size=3_000_000), blob("docs/img.png"), blob("src/readme.md"),
        blob("docs/examples/e.md"),
        {"type": "tree", "path": "docs", "sha": "b" * 40},
    ]


def test_select_docs_tree_selection_sealed_fixture():
    got = [n["path"] for n in subject.select_docs_tree(_tree_nodes(), True, 400)]
    assert got == ["NOTES.txt", "README.md", "doc/c.txt", "Docs/Intro.MD", "docs/guide/a.mdx",
                   "docs/guide/b.rst", "website/docs/d.md"]
    with_log = [n["path"] for n in subject.select_docs_tree(_tree_nodes(), True, 400, include_changelog=True)]
    assert "CHANGELOG.md" in with_log and "docs/CHANGELOG.md" in with_log


def test_select_docs_tree_cap_cuts_deepest_first():
    got = [n["path"] for n in subject.select_docs_tree(_tree_nodes(), True, 3)]
    assert got == ["NOTES.txt", "README.md", "doc/c.txt"]
    many = [{"type": "blob", "path": f"docs/p{i:04d}.md", "sha": "a" * 40, "size": 10} for i in range(1000)]
    assert len(subject.select_docs_tree(many, True, subject.TREE_DEFAULT_MAX_FILES)) == 400


def test_select_docs_tree_link_only_yields_only_readme():
    assert [n["path"] for n in subject.select_docs_tree(_tree_nodes(), False, 400)] == ["README.md"]


def test_tree_flag_defaults():
    ns = subject.parser().parse_args(["run", "--queue", "q", "--root", "r", "--tree", "docs"])
    assert ns.tree == "docs" and ns.max_files is None
    assert subject.parser().parse_args(["run", "--queue", "q", "--root", "r"]).tree == "capped"


def _fake_repo(tmp_path: Path, files: dict[str, tuple[bytes, int | None]]):
    """Sealed GitHub fixture: path -> (served bytes, size the tree claims)."""
    nodes = []
    by_url = {}
    for path, (body, size) in files.items():
        sha = hashlib.sha1(f"blob {len(body)}\0".encode() + body,
                           usedforsecurity=False).hexdigest()
        node = {"path": path, "type": "blob", "sha": sha}
        if size is not None:
            node["size"] = size
        nodes.append(node)
        by_url[f"https://raw.githubusercontent.com/o/r/{'a' * 40}/{path}"] = body
    by_url["https://api.github.com/repos/o/r/commits/main"] = {"sha": "a" * 40}
    by_url[f"https://api.github.com/repos/o/r/git/trees/{'a' * 40}?recursive=1"] = {
        "tree": nodes, "truncated": False}

    class Client:
        def get(self, url):
            value = by_url[url]
            request = httpx.Request("GET", url)
            if isinstance(value, bytes):
                return httpx.Response(200, request=request, content=value)
            return httpx.Response(200, request=request, json=value)

    settings = subject.Settings(root=tmp_path, max_files=50, retries=0, timeout=1.0,
                                tree="docs")
    return Client(), settings


_REC = {"owner_repo": "o/r", "branch": "main", "license": "MIT"}


def test_empty_tree_blob_is_skipped_and_receipted(tmp_path: Path):
    client, settings = _fake_repo(tmp_path, {
        "README.md": (b"# Hi\n", 5), "docs/a.md": (b"# A\n", 4), "docs/empty.md": (b"", 0)})
    receipt = subject.acquire_repository(client, _REC, settings)
    assert receipt["status"] == "complete"
    assert receipt["skipped_empty"] == ["docs/empty.md"]
    assert receipt["counts"]["original_documents"] == 2
    subject.validate_receipt(tmp_path, receipt, "o/r")


def test_size_mismatch_still_quarantines_with_path_and_size(tmp_path: Path):
    client, settings = _fake_repo(tmp_path, {
        "README.md": (b"# Hi\n", 5), "docs/a.md": (b"", 40)})
    receipt = subject.acquire_repository(client, _REC, settings)
    assert receipt["status"] == "quarantined"
    assert receipt["quarantine"]["reason"] == "selected_document_empty"
    assert "docs/a.md" in receipt["quarantine"]["detail"] and "40" in receipt["quarantine"]["detail"]
    subject.validate_receipt(tmp_path, receipt, "o/r")


def test_all_empty_is_no_docs_content(tmp_path: Path):
    client, settings = _fake_repo(tmp_path, {
        "README.md": (b"", 0), "docs/a.md": (b"", 0)})
    receipt = subject.acquire_repository(client, _REC, settings)
    assert receipt["status"] == "quarantined"
    assert receipt["quarantine"]["reason"] == "no_docs_content"
    subject.validate_receipt(tmp_path, receipt, "o/r")


def test_unparsable_link_is_skipped_and_counted():
    stats: dict[str, int] = {}
    md = "[bad](http://[::1:50051/x) and [ok](https://example.org/a)\n"
    links = subject.extract_outlinks(md, "https://github.com/o/r/blob/x/a.md", stats)
    assert [x["target_url"] for x in links] == ["https://example.org/a"]
    assert stats == {"unparsable_links": 1}


def test_unparsable_link_repo_still_completes_and_receipts_the_count(tmp_path: Path):
    client, settings = _fake_repo(tmp_path, {
        "README.md": (b"# Hi\n[bad](http://[::]:50051/x\x5d[)\n[bad2](http://[x)\n", None)})
    receipt = subject.acquire_repository(client, _REC, settings)
    assert receipt["status"] == "complete"
    assert receipt["unparsable_links"] >= 1
    subject.validate_receipt(tmp_path, receipt, "o/r")


def test_file_exception_quarantines_the_repo_with_the_path(tmp_path: Path, monkeypatch):
    client, settings = _fake_repo(tmp_path, {"README.md": (b"# Hi\n", 5)})

    def boom(raw, *, source_path):
        raise ValueError("Invalid IPv6 URL")
    monkeypatch.setattr(subject, "source_to_markdown", boom)
    receipt = subject.acquire_repository(client, _REC, settings)
    assert receipt["status"] == "quarantined"
    assert receipt["quarantine"]["reason"] == "exception: ValueError: Invalid IPv6 URL"
    assert "README.md" in receipt["quarantine"]["detail"]
    subject.validate_receipt(tmp_path, receipt, "o/r")


def test_transport_error_still_propagates(tmp_path: Path, monkeypatch):
    client, settings = _fake_repo(tmp_path, {"README.md": (b"# Hi\n", 5)})

    def down(raw, *, source_path):
        raise httpx.ConnectError("down")
    monkeypatch.setattr(subject, "source_to_markdown", down)
    import pytest
    with pytest.raises(httpx.ConnectError):
        subject.acquire_repository(client, _REC, settings)


def test_run_continues_after_a_quarantined_file_exception(tmp_path: Path, monkeypatch):
    client, settings = _fake_repo(tmp_path, {"README.md": (b"# Hi\n", 5)})
    monkeypatch.setattr(subject, "source_to_markdown",
                        lambda raw, *, source_path: (_ for _ in ()).throw(ValueError("x")))
    receipt = subject.acquire_repository(client, _REC, settings)
    # a quarantined receipt is terminal: load_terminal accepts it, so run() counts it
    (tmp_path / "receipts").mkdir(exist_ok=True)
    subject.atomic_json(tmp_path / "receipts" / subject.receipt_name("o/r"), receipt)
    assert subject.load_terminal(tmp_path, "o/r")["status"] == "quarantined"
