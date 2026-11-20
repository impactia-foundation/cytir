#!/usr/bin/env python3
"""Seal an adapter audit binding complete L0/L1 content to their stage digests."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

from cytir.acquire.artifact_io import atomic_publish, sha256_file
from cytir.index.preservation_contract import (_validate_stage, CONTENT_FIDELITY_RECEIPT_SCHEMA,
                                   CONTENT_SCOPE)


SHA256 = re.compile(r"^[0-9a-f]{64}$")
CONTENT_KEYS = {"body", "content", "html", "markdown", "text"}

# --------------------------------------------------------------- which file holds the adapter's CODE
#: What an alias shim says about where the code moved (`pipeline/<name>` -> `<package>/<pkg>/<name>`). The text
#: is that of ANOTHER release: r20-r22 say `aw.`, a release cut after the rename says `cytir.`; both resolve.
_SHIM_TARGET = re.compile(r"Shim: moved to `((?:aw|cytir)(?:\.\w+)+)`")  # rename-codemod: keep
#: Anything that makes a SMALL `pipeline/<name>` look like a shim, whatever its wording.
_SHIM_MARKERS = re.compile(r"(?i)\bshim\b|sys\.modules\[__name__\]|import_module\(\s*[\"'](?:aw|cytir)\.")  # rename-codemod: keep
#: A shim is under 60 lines (the shim tests were retired at 0 shims, task 1745); real adapters are far larger.
_SHIM_MAX_BYTES = 4000


class AdapterResolutionError(ValueError):
    """The file that holds an adapter's code could not be named without guessing."""


def resolve_adapter(release_dir: Path, name: str) -> Path:
    """The file under `release_dir` that holds adapter `name`'s CODE.

    A release cut before the adapter moved into `cytir/` has the code in `pipeline/<name>`. One cut
    after has an alias shim there and the code in the package file the shim names. The content
    audit scans and hashes the code that runs, never a shim, so a shim resolves to its package
    file. REFUSES (never hashes silently) when `pipeline/<name>` is missing, when it names a
    package file that is not in the release, and when it LOOKS like a shim (small, shim-worded or
    aliasing an `aw` module) but names no package file the resolver can read."""
    path = release_dir / "pipeline" / name
    try:
        size = path.stat().st_size
        text = path.read_text(encoding="utf-8") if size <= _SHIM_MAX_BYTES else ""
    except OSError as exc:
        raise AdapterResolutionError(
            f"adapter {name}: pipeline/{name} is not readable in release {release_dir} ({exc})") from exc
    if size > _SHIM_MAX_BYTES:
        return path
    target = _SHIM_TARGET.search(text[:400])
    if target:
        package_file = release_dir / (target.group(1).replace(".", "/") + ".py")
        if not package_file.is_file():
            raise AdapterResolutionError(
                f"adapter {name}: pipeline/{name} is a shim for `{target.group(1)}`, which is not in "
                f"release {release_dir}")
        return package_file
    if _SHIM_MARKERS.search(text):
        raise AdapterResolutionError(
            f"adapter {name}: pipeline/{name} looks like a shim but does not say "
            f"'Shim: moved to `aw.<module>`', so the file that holds the code cannot be named; "
            f"refusing to hash the shim")
    return path


def adapter_id(root: Path | None, path: Path) -> str:
    """The identity a receipt binds an adapter to: its path RELATIVE to the release (two different
    files can share a base name, e.g. the shim and its package file), else the base name when no
    root is given (the command line, a single loose file)."""
    if root is None:
        return path.name
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise AdapterResolutionError(f"adapter {path} is outside the release {root}") from exc


def _numeric_cap(node: ast.AST) -> bool:
    return (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice)
            and node.slice.upper is not None
            and isinstance(node.slice.upper, ast.Constant)
            and isinstance(node.slice.upper.value, int))


def _iso_date_field_projection(node: ast.AST) -> bool:
    """Recognize a date-only projection, not a principal-content prefix cap."""
    if not _numeric_cap(node) or node.slice.upper.value != 10:
        return False
    value = node.value
    if (not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name)
            or value.func.id != "str" or len(value.args) != 1):
        return False
    source = value.args[0]
    # Permit the common ``str(record.get('releaseTime') or '')[:10]`` form.
    candidates = source.values if isinstance(source, ast.BoolOp) else [source]
    for candidate in candidates:
        if (isinstance(candidate, ast.Call)
                and isinstance(candidate.func, ast.Attribute)
                and candidate.func.attr == "get" and candidate.args
                and isinstance(candidate.args[0], ast.Constant)
                and isinstance(candidate.args[0].value, str)):
            key = candidate.args[0].value.casefold()
            if "date" in key or "time" in key:
                return True
    return False


def audit_adapter(path: Path) -> list[str]:
    """Return numeric prefix caps flowing directly to content-bearing fields."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    findings: list[str] = []
    for node in ast.walk(tree):
        expressions: list[tuple[str, ast.AST]] = []
        if isinstance(node, ast.Call):
            expressions.extend(
                (keyword.arg, keyword.value) for keyword in node.keywords
                if keyword.arg in CONTENT_KEYS)
        elif isinstance(node, ast.Dict):
            expressions.extend(
                (str(key.value), value)
                for key, value in zip(node.keys, node.values)
                if (isinstance(key, ast.Constant) and key.value in CONTENT_KEYS
                    and value is not None))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if node.value is not None:
                expressions.extend(
                    (target.id, node.value) for target in targets
                    if isinstance(target, ast.Name) and target.id in CONTENT_KEYS)
        for field, expression in expressions:
            for child in ast.walk(expression):
                if _numeric_cap(child) and not _iso_date_field_projection(child):
                    cap = child.slice.upper.value
                    findings.append(
                        f"{path}:{child.lineno}: {field} has [: {cap}] cap")
    return sorted(set(findings))


def _evidence_path(artifact_root: Path, receipt_path: Path) -> Path:
    """Choose a collision-free evidence path while preserving sealed receipts."""
    if receipt_path.exists():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            uri = str((receipt.get("verification_evidence") or {}).get("uri") or "")
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("existing content-fidelity receipt is unreadable") from exc
        if not uri.startswith("artifact://"):
            raise ValueError("existing content-fidelity receipt lacks portable evidence URI")
        relative = Path(uri.removeprefix("artifact://"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("existing content-fidelity evidence URI is unsafe")
        path = (artifact_root / relative).resolve()
        if not path.is_relative_to(artifact_root.resolve()):
            raise ValueError("existing content-fidelity evidence escapes artifact root")
        return path
    return receipt_path.with_name(f"{receipt_path.stem}.verification-evidence.json")


def seal(*, artifact_root: Path, parent_run: Path, adapter_source: Path,
         additional_adapter_sources: tuple[Path, ...] = (),
         adapter_root: Path | None = None,
         source_id: str, verification_method: str,
         receipt_path: Path, envelope_path: Path) -> dict:
    run = json.loads(parent_run.read_text(encoding="utf-8"))
    stages = run.get("stages") or {}
    original = _validate_stage(stages.get("original"), "original",
                               artifact_root=artifact_root, deep=True)
    markdown = _validate_stage(stages.get("markdown"), "markdown",
                               artifact_root=artifact_root, deep=True)
    adapter_sources = (adapter_source, *additional_adapter_sources)
    if len({path.resolve() for path in adapter_sources}) != len(adapter_sources):
        raise ValueError("content-fidelity adapter sources must be unique")
    adapter_bindings = [
        {"id": adapter_id(adapter_root, path), "sha256": sha256_file(path)}
        for path in adapter_sources]
    if len({row["id"] for row in adapter_bindings}) != len(adapter_bindings):
        raise ValueError("content-fidelity adapter source names must be unique")
    adapter_sha = (adapter_bindings[0]["sha256"] if len(adapter_bindings) == 1 else
                   hashlib.sha256(json.dumps(
                       adapter_bindings, sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest())
    if not source_id.strip() or not verification_method.strip() or not SHA256.fullmatch(adapter_sha):
        raise ValueError("content-fidelity identity is invalid")
    findings = sorted(finding for source in adapter_sources
                      for finding in audit_adapter(source))
    if findings:
        raise ValueError("pre-chunk content cap found: " + "; ".join(findings))
    evidence_path = _evidence_path(artifact_root, receipt_path)
    destinations = {
        receipt_path.resolve(), evidence_path.resolve(), envelope_path.resolve()}
    if len(destinations) != 3:
        raise ValueError(
            "content-fidelity receipt, evidence, and envelope paths must be distinct")
    evidence = {
        "schema": "aw.content-fidelity-verification-evidence.v1",
        "source_id": source_id,
        "adapter_source_sha256": adapter_sha,
        "adapter_sources": adapter_bindings,
        "verification_method": verification_method,
        "original_logical_sha256": original["logical_sha256"],
        "markdown_logical_sha256": markdown["logical_sha256"],
        "assertions": {
            "original_bytes_complete": True,
            "markdown_content_complete": True,
            "source_text_truncation_applied": False,
        },
    }
    evidence_encoded = (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode()
    if evidence_path.exists():
        if sha256_file(evidence_path) != hashlib.sha256(evidence_encoded).hexdigest():
            raise ValueError("immutable content-fidelity evidence collision")
    else:
        atomic_publish(evidence_path, evidence_encoded)
    evidence_relative = evidence_path.resolve().relative_to(artifact_root.resolve())
    evidence_artifact = {
        "uri": f"artifact://{evidence_relative.as_posix()}",
        "bytes": evidence_path.stat().st_size,
        "sha256": sha256_file(evidence_path),
    }
    audit = {"schema": CONTENT_FIDELITY_RECEIPT_SCHEMA,
             "source_id": source_id, "scope": CONTENT_SCOPE,
             "original_bytes_complete": True,
             "markdown_content_complete": True,
             "source_text_truncation_applied": False,
             "original_logical_sha256": original["logical_sha256"],
             "markdown_logical_sha256": markdown["logical_sha256"],
             "adapter_source_sha256": adapter_sha,
             "adapter_sources": adapter_bindings,
             "verification_method": verification_method,
             "verification_evidence": evidence_artifact}
    encoded = (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode()
    if receipt_path.exists():
        if sha256_file(receipt_path) != hashlib.sha256(encoded).hexdigest():
            raise ValueError("immutable content-fidelity receipt collision")
    else:
        atomic_publish(receipt_path, encoded)
    relative = receipt_path.resolve().relative_to(artifact_root.resolve())
    envelope = {"scope": CONTENT_SCOPE, "original_bytes_complete": True,
                "markdown_content_complete": True,
                "source_text_truncation_applied": False,
                "receipt": {"uri": f"artifact://{relative.as_posix()}",
                            "bytes": receipt_path.stat().st_size,
                            "sha256": sha256_file(receipt_path)}}
    envelope_encoded = (json.dumps(envelope, indent=2, sort_keys=True) + "\n").encode()
    if envelope_path.exists():
        if sha256_file(envelope_path) != hashlib.sha256(envelope_encoded).hexdigest():
            raise ValueError("immutable content-fidelity envelope collision")
    else:
        atomic_publish(envelope_path, envelope_encoded)
    return envelope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--adapter-source", type=Path, required=True)
    parser.add_argument("--additional-adapter-source", type=Path, action="append",
                        default=[])
    parser.add_argument("--adapter-root", type=Path, default=None,
                        help="bind each adapter to its path relative to this directory "
                             "(the release); default: its base name")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--verification-method", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--envelope", type=Path, required=True)
    args = parser.parse_args()
    result = seal(artifact_root=args.artifact_root.resolve(),
                  parent_run=args.parent_run.resolve(),
                  adapter_source=args.adapter_source.resolve(),
                  additional_adapter_sources=tuple(
                      path.resolve() for path in args.additional_adapter_source),
                  adapter_root=args.adapter_root,
                  source_id=args.source_id,
                  verification_method=args.verification_method,
                  receipt_path=args.receipt.resolve(),
                  envelope_path=args.envelope.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
