"""`github_agent_resources` builders: the pure half of the GitHub docs acquirer (task 1115, S7).

Moved UNCHANGED out of the acquirer driver (`acquire_github_repo_docs.py`) (only the imports differ): document
selection, Markdown normalisation, title, outlinks, the L0 binding, and the receipt model
(`SCHEMA`, `quarantine_receipt`, `validate_receipt`, `load_terminal`, `receipt_name`). The driver keeps
the HTTP half (`GitHubClient`, `acquire_repository`, `run`) and imports every name below, so its
importers and the refresh lane keep working. Not on a live path by itself.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urljoin, urlparse

from cytir.acquire.github_docs_tree import ROOT_DOC
from cytir.acquire.preserved_stage_io import PreservedArtifactError, load_validated_artifact
from cytir.acquire.preserved_stage_io import validate_artifact as _validate_artifact

SCHEMA = "aw.github-doc-acquisition.v1"


DOC_EXTENSIONS = (".md", ".mdx", ".markdown", ".rst")
DOC_DIR = re.compile(
    r"(^|/)(docs?|documentation|guide|manual|handbook|website/(docs|content))(/|$)",
    re.I,
)
SKIP = re.compile(
    r"(CHANGELOG|CHANGES|HISTORY|CONTRIBUTING|CODE_OF_CONDUCT|LICENSE|SECURITY|"
    r"node_modules|(^|/)tests?(/|$)|(^|/)examples?(/|$))",
    re.I,
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
MD_LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)(?:\s+['\"][^)]*['\"])?\)")
AUTOLINK_RE = re.compile(r"<((?:https?|mailto):[^>\s]+)>", re.I)


class AcquisitionError(RuntimeError):
    """A repository could not be acquired without weakening the contract."""


def validate_artifact(root: Path, artifact: dict[str, Any]) -> None:
    """Retain the adapter's public error contract over shared artifact I/O."""
    try:
        _validate_artifact(root, artifact)
    except PreservedArtifactError as exc:
        raise AcquisitionError(str(exc)) from exc


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_owner_repo(value: str) -> str:
    parts = value.strip().strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        raise AcquisitionError(f"invalid owner/repo: {value!r}")
    return "/".join(parts)


def receipt_name(owner_repo: str) -> str:
    return hashlib.sha256(owner_repo.casefold().encode()).hexdigest() + ".json"


def select_documents(nodes: list[dict[str, Any]], permissive: bool,
                     max_files: int, exclude: str | None = None) -> list[dict[str, Any]]:
    """`exclude` (task 972): an optional regex on the repo path from the queue record's
    `exclude_regex`, e.g. a docs tree's translated copies, which would otherwise fill the file cap."""
    skip = re.compile(exclude) if exclude else None
    blobs = [
        node for node in nodes
        if node.get("type") == "blob"
        and str(node.get("path", "")).lower().endswith(DOC_EXTENSIONS)
        and not SKIP.search(str(node.get("path", "")))
        and not (skip and skip.search(str(node.get("path", ""))))
        and isinstance(node.get("sha"), str)
    ]
    docs = [node for node in blobs if DOC_DIR.search(str(node["path"]))]
    roots = [
        node for node in blobs
        if ROOT_DOC.search(PurePosixPath(str(node["path"])).name)
    ]
    if not permissive:
        return (roots or docs or blobs)[:1]
    ordered = docs + [node for node in roots if node not in docs]
    return (ordered or blobs)[:max_files]


def source_provenance(*, owner_repo: str, revision: str, source_path: str,
                      parent_sha256: str) -> dict:
    """The exact-L0 binding for a repository document, as RECORD METADATA.

    Until task 517 these four fields were written into the Markdown itself as a
    YAML front-matter block, and the chunker's section-adjacent policy then made
    that block the head of chunk 0 -- the snippet AND the head of the dense
    embedding -- on 72,330 of the 126,427 staged context7 rows, with
    `section_heading` literally "---". It is the same defect task 516 fixed for
    fedlex one family over (the ELI/SHA-256 preamble), and the same fix: the
    binding is metadata, not the document's first sentence.

    Nothing is lost. Every caller already stores these four values in fields of
    its own record (`repository`, `revision`, `source_path`,
    `parent_original_sha256`); this returns them as one block so a record can
    carry them under a single name and a reader can find them there.
    """
    return {"repository": owner_repo, "revision": revision,
            "source_path": source_path, "parent_sha256": parent_sha256}


def source_to_markdown(raw: bytes, *, source_path: str) -> bytes:
    """Identity-normalize repository text. NO provenance is prepended.

    See `source_provenance` for where the L0 binding went and why.
    """
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    if source_path.lower().endswith(".rst"):
        lines = text.splitlines()
        rendered: list[str] = []
        index = 0
        while index < len(lines):
            if index + 1 < len(lines) and lines[index].strip() and re.fullmatch(
                    r"[=\-~^\"'#*+`.]{3,}", lines[index + 1].strip()):
                marker = lines[index + 1].strip()[0]
                level = "#" if marker == "=" else "##"
                rendered.append(f"{level} {lines[index].strip()}")
                index += 2
                continue
            rendered.append(lines[index])
            index += 1
        text = "\n".join(rendered)
        text = re.sub(
            r"`([^`<]+)\s*<([^>]+)>`__?",
            lambda match: f"[{match.group(1).strip()}]({match.group(2)})",
            text,
        )
        text = re.sub(r":\w+(?::\w+)?:`~?([^`]+)`", r"`\1`", text)
    return (text.strip() + "\n").encode()


# An ATX heading: `#`..`######` followed by text, closing hashes optional.
_ATX_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")


def compose_document_title(markdown: str, source_path: str) -> str:
    """The title production gives a repository document, re-derived.

    NOT invented. The collection this rebuild replaces carries 252,626
    github.com context7 rows, 75,900 distinct (repository, source_path)
    documents, and every chunk of a document carries the SAME title (measured:
    0 of 75,900 keys disagree). Five candidate rules were scored against those
    74,794 matched documents in task 517 title oracle script:

        first ATX heading anywhere, else the path basename   90.26%  <- this
        first ATX heading anywhere (no fallback)             84.10%
        the path basename alone                              12.89%

    Its 7,288 disagreements run overwhelmingly the build's way: 5,033 are
    documents where PRODUCTION fell back to the filename and this rule found a
    real heading (chiefly reStructuredText, whose underline headings the
    epoch-4 normalizer converts and production's ingest did not), 2,103 are two
    different in-document titles, and 152 are the other way round. The
    alternative on the table -- composing "<owner>/<repo> - <path>" -- agrees
    with production on none of them.
    """
    for line in markdown.splitlines():
        m = _ATX_HEADING.match(line)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return source_path.rsplit("/", 1)[-1]


def extract_outlinks(markdown: str, source_url: str,
                     stats: dict[str, int] | None = None) -> list[dict[str, str]]:
    """Outlinks of one Markdown file. A target that urllib cannot parse (an
    unmatched "[" is an "Invalid IPv6 URL") is a file property: skip it, count it
    in stats["unparsable_links"], never raise (task 1012)."""
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in MD_LINK_RE.finditer(markdown):
        anchor, target = match.group(1).strip(), match.group(2).strip().strip("<>")
        if target.startswith("#"):
            continue
        try:
            url = urljoin(source_url, target)
            scheme = urlparse(url).scheme
        except ValueError:
            if stats is not None:
                stats["unparsable_links"] = stats.get("unparsable_links", 0) + 1
            continue
        if scheme not in {"http", "https", "mailto"}:
            continue
        key = (url, anchor)
        if key not in seen:
            seen.add(key)
            rows.append({"source_url": source_url, "target_url": url,
                         "anchor_or_context": anchor})
    for match in AUTOLINK_RE.finditer(markdown):
        url = match.group(1)
        key = (url, url)
        if key not in seen:
            seen.add(key)
            rows.append({"source_url": source_url, "target_url": url,
                         "anchor_or_context": url})
    return rows


def quarantine_receipt(record: dict[str, Any], branch: str, reason: str,
                       evidence: list[dict[str, Any]], detail: str, *,
                       revision: str | None = None) -> dict[str, Any]:
    """Terminal, evidence-backed source classification; never a false empty."""
    return {
        "schema": SCHEMA,
        "status": "quarantined",
        "acquired_at": utc_now(),
        "source_profile": "software_docs",
        "repository": record["owner_repo"],
        "requested_branch": str(record.get("branch") or "main"),
        "resolved_branch": branch,
        "revision": revision,
        "license": str(record.get("license") or "NOASSERTION").upper(),
        "route": "documented_source_api",
        "quarantine": {"reason": reason, "detail": detail, "evidence": evidence},
        "counts": {"original_documents": 0, "markdown_documents": 0,
                   "outlink_documents": 0, "outlinks": 0},
        "documents": [],
        "downstream": {"chunks": "not_applicable_quarantined",
                       "vectors": "not_applicable_quarantined"},
    }


def validate_receipt(root: Path, receipt: dict[str, Any], owner_repo: str) -> None:
    if receipt.get("schema") != SCHEMA or receipt.get("status") not in {
            "complete", "quarantined"}:
        raise AcquisitionError("not a terminal acquisition receipt")
    if str(receipt.get("repository", "")).casefold() != owner_repo.casefold():
        raise AcquisitionError("receipt repository does not match queue")
    documents = receipt.get("documents")
    if not isinstance(documents, list):
        raise AcquisitionError("receipt documents must be a list")
    if receipt["status"] == "quarantined":
        if documents:
            raise AcquisitionError("quarantined receipt cannot contain documents")
        quarantine = receipt.get("quarantine")
        if not isinstance(quarantine, dict) or not quarantine.get("reason"):
            raise AcquisitionError("quarantined receipt lacks a reason")
        evidence = quarantine.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise AcquisitionError("quarantined receipt lacks source evidence")
        for artifact in evidence:
            validate_artifact(root, artifact)
        if any(receipt.get("counts", {}).get(key) != 0 for key in (
                "original_documents", "markdown_documents", "outlink_documents", "outlinks")):
            raise AcquisitionError("quarantined receipt must have zero stage counts")
        return
    if not documents:
        raise AcquisitionError("complete receipt has no documents")
    validate_artifact(root, receipt["tree"]["commit_response"])
    validate_artifact(root, receipt["tree"]["tree_response"])
    for document in documents:
        for stage in ("original", "markdown", "outlinks"):
            validate_artifact(root, document[stage])
        if document.get("blob_api_response"):
            validate_artifact(root, document["blob_api_response"])
    counts = receipt.get("counts", {})
    if counts.get("original_documents") != len(documents):
        raise AcquisitionError("receipt document count disagrees")
    if counts.get("markdown_documents") != len(documents):
        raise AcquisitionError("receipt Markdown count disagrees")
    if counts.get("outlink_documents") != len(documents):
        raise AcquisitionError("receipt outlink-parent count disagrees")


def load_terminal(root: Path, owner_repo: str) -> dict[str, Any] | None:
    path = root / "receipts" / receipt_name(owner_repo)
    if not path.exists():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        validate_receipt(root, receipt, owner_repo)
        return receipt
    except (OSError, json.JSONDecodeError, KeyError, TypeError, AcquisitionError):
        return None


def candidates_from_receipt(root: Path, receipt: dict, repo: dict) -> list[dict]:
    if receipt.get("status") != "complete":
        raise RuntimeError(f"{repo['owner_repo']}: receipt status {receipt.get('status')!r}")
    got = str(receipt.get("license") or "").upper()
    if got not in ("", "NOASSERTION") and got != repo["license"].upper():
        raise RuntimeError(f"{repo['owner_repo']}: receipt licence {got} != pack licence {repo['license']}")
    out = []
    for d in receipt["documents"]:
        md = load_validated_artifact(root, d["markdown"]).decode("utf-8")
        out.append({"url": d["source_url"], "markdown": md, "title": d.get("title"),
                    "repository": repo["owner_repo"], "revision": receipt["revision"],
                    "source_path": d["source_path"], "license": repo["license"],
                    "trust_tier": repo["trust_tier"],
                    "acquired_at": receipt["acquired_at"], "route": "github_repo_docs"})
    return out
