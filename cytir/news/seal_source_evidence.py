"""Turn one sealed news-lane arm into a tier-2 content-pack `sources[]` entry.

`seal_news_lane_stages.py` seals the arm's five `news_article` stages and the
lineage between them. The content-pack contract wants four more things bound to
that run before an apply may be authorized, and until now nothing produced any
of them for this lane (task 449 §6.4: "no producer exists for the rest"):

* **configuration** — the exact inventory, filters, versions and adapter
  parameters the run used, so a clone knows *which member* of the profile
  family to re-run rather than merely which profile;
* **content fidelity** — a receipt bound to the arm's own `original` and
  `markdown` logical digests, asserting no pre-chunk content cap;
* **profile compliance** — one PASS row per registry requirement, in registry
  order, produced by `audit_news_article_profile` re-deriving the semantics
  from the artifacts rather than restating them;
* **durability** — the arm's evidence exists off-box.

Nothing here re-states a count that another artifact already carries: the
stages and lineage are copied verbatim out of the arm's sealed `RUN.json`, and
every number in the configuration is read from the run, the spool or the
release.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

from cytir.acquire.artifact_io import atomic_publish, sha256_file
from cytir.news.source_queue import queue_path, queue_relpath
from cytir.ops.audits import news_article_profile as AUDIT
from cytir.index import preservation_contract as full
from cytir.index import seal_content_fidelity_audit as FIDELITY
from cytir.news import seal_lane_stages as LANE
from cytir.index import seal_source_profile_compliance as COMPLIANCE


CONFIGURATION_SCHEMA = "aw.news-lane-configuration.v1"
SOURCE_SCHEMA = "aw.content-pack-source-entry.v1"
PROFILE = "news_article"

# The modules whose source decides what text reaches the chunker for this lane.
#
# ADAPTER LAYOUT VERSIONS (the receipt's `adapter_layout_version`, in configuration.json):
#   1 - the adapters are named by BASE NAME under `pipeline/` (a shim there resolves to its package
#       file). Every receipt sealed before task 1310 (r20-r22 and earlier) is layout 1; the field
#       is absent from them, and `adapter_layout_version_of` reads an absent field as 1.
#   2 - the adapters are bound at their package `acquire/` paths, release-relative, with NO `pipeline/`
#       shim in the chain (task 1310, Tim 2026-10-25 D1 + D23). The release must hold the five files.
#       Layout 2 accepts BOTH package directories (rename task 2974; there is no layout 3):
#       `aw/acquire/` for a receipt sealed before the rename (every layout-2 receipt from task 1310 on)  # rename-codemod: keep
#       and `cytir/acquire/` for one sealed after it. A release holds one of the two (both: refused). The
#       receipt binds the release-relative path found, so a sealed receipt's ids and digests never change.
# Sealed layout-1 receipts stay valid records of what they hashed; they are never rewritten.
ADAPTER_LAYOUT_VERSION = 2
LEGACY_ADAPTER_SOURCES = ("sitemap_ingest.py", "discovery_ingest.py",
                          "preserved_page_fetch.py", "preserved_stage_io.py",
                          "fetch_process.py")
ADAPTER_SOURCES = tuple(f"cytir/acquire/{name}" for name in LEGACY_ADAPTER_SOURCES)
#: The package directory a layout-2 receipt sealed BEFORE the rename is bound at (the old name, kept on purpose).
PRE_RENAME_ADAPTER_SOURCES = tuple(f"aw/acquire/{name}" for name in LEGACY_ADAPTER_SOURCES)  # rename-codemod: keep
HTML_CAP_NAME = "_HTML_CAP"


def adapter_layout_version_of(configuration: dict) -> int:
    """The adapter layout a sealed configuration was cut under (absent == 1, the pre-1310 layout)."""
    return int(configuration.get("adapter_layout_version", 1))


def _layout_2_sources(release_dir: Path) -> tuple[str, ...]:
    """The five release-relative adapter paths of the package directory this release holds."""
    candidates = [ADAPTER_SOURCES]
    if PRE_RENAME_ADAPTER_SOURCES != ADAPTER_SOURCES:
        candidates.append(PRE_RENAME_ADAPTER_SOURCES)
    held = [c for c in candidates if any((release_dir / rel).exists() or (release_dir / rel).is_symlink() for rel in c)]
    if len(held) > 1:
        raise SourceEvidenceError(
            f"adapter layout 2: release {release_dir} holds adapters under both package directories "
            f"({held[0][0]} and {held[1][0]}); which one is the code that runs cannot be named")
    return held[0] if held else candidates[0]


def adapter_paths(release_dir: Path,
                  layout_version: int = ADAPTER_LAYOUT_VERSION) -> list[Path]:
    """The files that hold each adapter's CODE in this release.

    Layout 2 (default, new seals): the five `acquire/<name>` files of the release's package
    directory (`ADAPTER_SOURCES`, or `PRE_RENAME_ADAPTER_SOURCES` for a release cut before the
    rename; see ADAPTER LAYOUT VERSIONS), bound under their release-relative path. Refuses
    (`SourceEvidenceError`) when one is missing or is not a regular file, or when the release holds
    adapters under both package directories; a `pipeline/` shim is never consulted or hashed.

    Layout 1 (re-deriving a pre-1310 receipt): `pipeline/<name>`, a shim resolving to its package
    file (`FIDELITY.resolve_adapter`). For r20-r22 the `fetch_process.py` under that directory is a shim, and the
    previous code hashed the package file (`0fd17ce1...`), not the shim (`ee4d8b1c...`).
    """
    if layout_version == 2:
        sources = _layout_2_sources(release_dir)
        paths = [release_dir / rel for rel in sources]
        missing = [rel for rel, path in zip(sources, paths)
                   if not path.is_file() or path.is_symlink()]
        if missing:
            raise SourceEvidenceError(
                f"adapter layout 2 needs {missing} in release {release_dir}; a release cut "
                f"before task 1310 binds layout 1 (pipeline/ shims)")
        return paths
    if layout_version != 1:
        raise SourceEvidenceError(f"unknown adapter layout_version {layout_version}")
    try:
        return [FIDELITY.resolve_adapter(release_dir, name) for name in LEGACY_ADAPTER_SOURCES]
    except FIDELITY.AdapterResolutionError as exc:
        raise SourceEvidenceError(str(exc)) from exc


class SourceEvidenceError(RuntimeError):
    """The arm cannot be presented as a preserved content-pack source."""


def _artifact(root: Path, path: Path) -> dict:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise SourceEvidenceError(
            f"{path} is outside the portable artifact root {root}") from exc
    return {"uri": f"artifact://{relative.as_posix()}",
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _publish(path: Path, value: dict) -> dict:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise SourceEvidenceError(f"immutable collision: {path}")
    else:
        atomic_publish(path, encoded)
    return value


def _stage_rows(artifact_root: Path, stage: dict) -> list[dict]:
    return AUDIT._rows(artifact_root, stage)


def adapter_html_cap(sources: list[Path], environment: dict[str, str]) -> dict:
    """The pre-chunk HTML cap that was actually in effect, from the code.

    `_HTML_CAP` is not a literal: it is
    ``int(os.environ.get("AW_HTML_CAP", "<default>"))``, so the effective value
    is a function of the run's environment as well as the source. Both halves
    are resolved here — the constant is read out of whichever adapter module
    defines it, and the override is looked up in the run's own environment —
    so the measurement below cannot be made against a cap the run did not use.
    """
    for source in sources:
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                    isinstance(node.targets[0], ast.Name) and
                    node.targets[0].id == HTML_CAP_NAME):
                continue
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, int):
                return {"defined_in": source.name, "environment_variable": None,
                        "default": value.value, "override": None,
                        "effective": value.value}
            name, default = _environment_int(value)
            override = environment.get(name)
            return {"defined_in": source.name, "environment_variable": name,
                    "default": default, "override": override,
                    "effective": int(override) if override is not None else default}
    raise SourceEvidenceError(
        f"no {HTML_CAP_NAME} definition in {[s.name for s in sources]}")


def _environment_int(value: ast.AST) -> tuple[str, int]:
    """Match ``int(os.environ.get("NAME", "DEFAULT"))`` exactly, or refuse."""
    try:
        assert isinstance(value, ast.Call) and value.func.id == "int"
        inner = value.args[0]
        assert isinstance(inner, ast.Call) and inner.func.attr == "get"
        assert inner.func.value.attr == "environ"
        name, default = inner.args
        return name.value, int(default.value)
    except (AssertionError, AttributeError, IndexError, TypeError,
            ValueError) as exc:
        raise SourceEvidenceError(
            f"{HTML_CAP_NAME} is not a readable constant or environment "
            f"default: {ast.dump(value)[:120]}") from exc


def measure_truncation_headroom(artifact_root: Path, run: dict,
                                cap: int) -> dict:
    """How close this arm's originals came to the adapter's legacy HTML cap.

    The cap is a *named* constant applied only on the non-preserved extraction
    path, which `seal_content_fidelity_audit`'s AST scan cannot see (it detects
    literal slice bounds). So the claim "no pre-chunk truncation" is not left
    to the scanner: the sealed originals are measured, and an original that
    could not have been truncated because it never reached the cap is a fact
    about the data, not about the code.
    """
    sizes = [int(row["original_bytes"])
             for row in _stage_rows(artifact_root, run["stages"]["original"])]
    if not sizes:
        raise SourceEvidenceError("arm has no admitted originals to measure")
    effective = cap["effective"]
    return {"admitted_originals": len(sizes), "html_cap": cap,
            "html_cap_bytes": effective,
            "largest_original_bytes": max(sizes),
            "originals_at_or_over_cap": sum(1 for size in sizes
                                            if size >= effective)}


def default_extractor():
    """The deployed extraction function, imported only when actually used.

    `aw_common` pulls in the extraction stack; keeping the import inside the
    call lets every other producer and test in this module run in an
    environment that has no trafilatura.
    """
    from cytir.extract.html_extract import extract_content_markdown

    def extract(html: str, url: str) -> str:
        markdown, _mode = extract_content_markdown(html, url)
        return markdown or ""
    return extract


def reproduce_extraction(*, spool: Path, artifact_root: Path, run: dict,
                         cap: int, sample: int, extractor=None) -> dict:
    """Re-derive the sealed Markdown from the COMPLETE preserved bytes.

    The strongest available statement about "no pre-chunk truncation" is not
    that a scanner found no literal slice: it is that re-running the deployed
    extractor on the whole sealed response reproduces the exact Markdown the
    run sealed — and that doing it on the capped prefix does not (or, where it
    does, that a cap could not have changed the content). Both halves are
    recorded; the first is required, and a sample that fails to reproduce
    aborts the seal.
    """
    from cytir.acquire import preserved_stage_io as PS
    extract = extractor or default_extractor()
    originals = {row["document_id"]: row for row in
                 _stage_rows(artifact_root, run["stages"]["original"])}
    parents = _stage_rows(artifact_root, run["stages"]["markdown"])
    parents.sort(key=lambda row: row["document_id"])
    if not parents:
        raise SourceEvidenceError("arm has no markdown parents to reproduce")
    step = max(1, len(parents) // sample)
    chosen = parents[::step][:sample]
    reproduced = differs = 0
    pages = []
    for parent in chosen:
        original = originals[parent["document_id"]]
        body = PS.load_validated_artifact(
            spool / "pages", original["artifacts"]["body"])
        html = body.decode("utf-8", "replace") if isinstance(
            body, (bytes, bytearray)) else body
        full = hashlib.sha256(
            extract(html, parent["url"]).encode("utf-8")).hexdigest()
        if full != parent["body_markdown_sha256"]:
            raise SourceEvidenceError(
                f"re-extracting {parent['url']} from the complete preserved "
                "bytes does not reproduce the sealed Markdown "
                f"({full} != {parent['body_markdown_sha256']}); the sealed "
                "markdown stage is not the extraction of the whole response")
        reproduced += 1
        capped = hashlib.sha256(
            extract(html[:cap], parent["url"]).encode("utf-8")).hexdigest()
        differs += capped != full
        pages.append({"url": parent["url"],
                      "original_bytes": original["original_bytes"],
                      "reproduced_from_complete_bytes": True,
                      "capped_prefix_would_differ": capped != full})
    return {"sampled": len(chosen), "reproduced_from_complete_bytes": reproduced,
            "capped_prefix_would_differ": differs, "cap_bytes": cap,
            "pages": pages}


def fidelity_method(headroom: dict, reproduction: dict) -> str:
    cap = headroom["html_cap"]
    override = ("unset in this run" if cap["override"] is None
                else f"={cap['override']} in this run")
    return (
        "Adapter sources AST-audited for pre-chunk content caps; this lane "
        "always runs sitemap_ingest with --preserve-dir, on which the "
        "extraction input is the whole decoded response body and the legacy "
        "_HTML_CAP slice is not taken; the sealed original and markdown stage "
        "logical digests were re-derived from their artifacts; "
        f"{headroom['originals_at_or_over_cap']} of "
        f"{headroom['admitted_originals']} admitted originals reach the legacy "
        f"{headroom['html_cap_bytes']}-byte HTML cap (defined in "
        f"{cap['defined_in']}, {cap['environment_variable']} {override}; "
        f"largest original {headroom['largest_original_bytes']} bytes); "
        "re-running the deployed extractor on the COMPLETE preserved bytes "
        "reproduced the sealed Markdown digest on "
        f"{reproduction['reproduced_from_complete_bytes']}/"
        f"{reproduction['sampled']} sampled parents, and the same extraction "
        f"from the {reproduction['cap_bytes']}-byte prefix would have differed "
        f"on {reproduction['capped_prefix_would_differ']} of them.")


def build_configuration(*, run: dict, release_dir: Path, spool: Path,
                        staged_collection: str, staging_url: str,
                        stage_environment: dict[str, str],
                        release_manifest_sha256: str) -> dict:
    stage_script = release_dir / "deploy" / "scripts" / "news_refresh_stage.sh"
    queue = queue_path(release_dir)
    host_meta = release_dir / "data" / "news-host-meta.v1.json"
    registry = release_dir / "data" / "source-preservation-profiles.v1.json"
    return {
        "schema": CONFIGURATION_SCHEMA,
        "profile": PROFILE,
        "arm": run["arm"],
        "day": run["day"],
        "release": {
            "git_sha": (release_dir / "GIT_SHA").read_text(encoding="utf-8").strip(),
            "manifest_sha256": release_manifest_sha256,
        },
        "producer": run["producer"],
        "spool": spool.name,
        "inventory": {
            # The claim that the lane always preserves before it extracts is
            # a fact about this file; bind it so the claim is checkable.
            "stage_script": {"path": "pipeline/news_refresh_stage.sh",
                             "sha256": sha256_file(stage_script)},
            "queue": {"path": queue_relpath(),
                      "sha256": sha256_file(queue)},
            "host_meta": {"path": "data/news-host-meta.v1.json",
                          "sha256": sha256_file(host_meta)},
            "profile_registry": {
                "path": "data/source-preservation-profiles.v1.json",
                "sha256": sha256_file(registry)},
            "arm_routes": [{"host_suffix": suffix, "arm": arm}
                           for suffix, arm in LANE.ARM_ROUTES],
        },
        "filters": {
            "freshness_window_months": stage_environment["AW_NEWS_FRESH_MONTHS"],
            "max_per_domain": stage_environment["AW_RECENT_CAP"],
            "now": stage_environment["AW_NOW"],
            "resume": stage_environment["AW_NEWS_RESUME"],
            "consent": {"mode": "direct-crawl", "surface": "search-indexing",
                        "gate": "pipeline/consent_gate.py"},
        },
        "adapter_layout_version": ADAPTER_LAYOUT_VERSION,
        "adapter_parameters": {
            "extraction": "in-process (AW_INPROC_EXTRACT=1)",
            "chunk_policy": run["chunk_policy"],
            "language_gate_python": stage_environment["AW_NEWS_LANGGATE_PY"],
            "language_gate_requirements_sha256":
                stage_environment["AW_NEWS_LANGGATE_REQUIREMENTS_SHA256"],
        },
        "staging": {"url": staging_url, "collection": staged_collection},
        "counts": run["counts"],
    }


def seal(*, artifact_root: Path, run_path: Path, out_root: Path,
         release_dir: Path, spool: Path, source_id: str,
         staged_collection: str, staging_url: str,
         stage_environment: dict[str, str], release_manifest_sha256: str,
         durability_receipt: Path, extraction_sample: int = 24,
         extractor=None, expect_counts: dict[str, int] | None = None) -> dict:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("schema") != LANE.RUN_SCHEMA or run.get("profile") != PROFILE:
        raise SourceEvidenceError(f"{run_path} is not a sealed news-lane run")
    if run.get("status") != LANE.RUN_STATUS:
        raise SourceEvidenceError(
            f"run status is {run.get('status')!r}, not {LANE.RUN_STATUS!r}")
    registry_path = release_dir / "data" / "source-preservation-profiles.v1.json"
    profile = json.loads(registry_path.read_text())["profiles"][PROFILE]
    missing = [name for name in profile["required_stages"]
               if name not in run["stages"]]
    if missing:
        raise SourceEvidenceError(
            f"arm {run['arm']} seals no {', '.join(missing)} stage")
    out_root.mkdir(parents=True, exist_ok=True)

    configuration = _publish(out_root / "configuration.json", build_configuration(
        run=run, release_dir=release_dir, spool=spool,
        staged_collection=staged_collection, staging_url=staging_url,
        stage_environment=stage_environment,
        release_manifest_sha256=release_manifest_sha256))
    adapters = adapter_paths(release_dir)
    headroom = _publish(out_root / "truncation-headroom.json", {
        "schema": "aw.news-lane-truncation-headroom.v1", "arm": run["arm"],
        **measure_truncation_headroom(
            artifact_root, run,
            adapter_html_cap(adapters, stage_environment))})

    reproduction = _publish(out_root / "extraction-reproduction.json", {
        "schema": "aw.news-lane-extraction-reproduction.v1", "arm": run["arm"],
        **reproduce_extraction(
            spool=spool, artifact_root=artifact_root, run=run,
            cap=headroom["html_cap_bytes"], sample=extraction_sample,
            extractor=extractor)})

    fidelity = FIDELITY.seal(
        artifact_root=artifact_root, parent_run=run_path,
        adapter_source=adapters[0], additional_adapter_sources=adapters[1:],
        adapter_root=release_dir,
        source_id=source_id,
        verification_method=fidelity_method(headroom, reproduction),
        receipt_path=out_root / "content-fidelity.json",
        envelope_path=out_root / "content-fidelity.envelope.json")

    evidence_path, checks_path = AUDIT.audit(
        parent_root=artifact_root, parent_run=run_path,
        pack_root=run_path.parent, output_root=out_root / "profile-audit",
        registry_path=registry_path, source_id=source_id,
        expect_counts=expect_counts or {}, run_status=LANE.RUN_STATUS,
        pack_artifact_root=artifact_root)
    compliance = COMPLIANCE.seal(
        artifact_root=artifact_root, registry_path=registry_path,
        profile_id=PROFILE, source_id=source_id, checks_path=checks_path,
        evidence_path=evidence_path,
        receipt_path=out_root / "profile-compliance.json")

    source = {
        "source_id": source_id,
        "profile": PROFILE,
        "configuration": _artifact(artifact_root, out_root / "configuration.json"),
        "stages": {name: run["stages"][name]
                   for name in profile["required_stages"]},
        "content_fidelity": fidelity,
        "profile_compliance": compliance,
        "lineage": run["lineage"],
        "durability": {"status": "verified_offbox",
                       "receipt": _artifact(artifact_root, durability_receipt)},
    }
    _publish(out_root / "SOURCE.json", {"schema": SOURCE_SCHEMA,
                                        "source": source})
    # Fail closed here rather than at the apply: every requirement the shared
    # source validator makes of this entry is checked against the real
    # artifacts before the entry is offered to an intent.
    validated, vectors = _validate_one_source(
        source, registry_path=registry_path, artifact_root=artifact_root)
    return {"source": source, "source_id": validated,
            "vector_records": vectors,
            "configuration": configuration["schema"],
            "truncation_headroom": headroom,
            "extraction_reproduction": reproduction}


def _validate_one_source(source: dict, *, registry_path: Path,
                         artifact_root: Path) -> tuple[str, int]:
    import cytir.index.content_pack_preservation as CP
    registry = full.load_profiles(registry_path)
    validated, vectors = CP._validate_sources(
        [source], registry=registry,
        registry_sha256=sha256_file(registry_path), root=artifact_root,
        deep=True)
    return validated[0]["source_id"], vectors


def _counts(values: list[str]) -> dict[str, int]:
    out = {}
    for name, value in _pairs(values).items():
        if not value.strip().lstrip("-").isdigit():
            raise argparse.ArgumentTypeError(
                f"--expect-count wants NAME=INTEGER, got {name}={value!r}")
        out[name] = int(value)
    return out


def _pairs(values: list[str]) -> dict[str, str]:
    out = {}
    for item in values:
        name, _, value = item.partition("=")
        if not name.strip() or not _:
            raise argparse.ArgumentTypeError(f"--env wants NAME=VALUE, got {item!r}")
        out[name.strip()] = value
    return out


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--release-manifest-sha256", required=True)
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--staged-collection", required=True)
    parser.add_argument("--staging-url", required=True)
    parser.add_argument("--durability-receipt", type=Path, required=True)

    parser.add_argument("--extraction-sample", type=int, default=24)
    parser.add_argument("--expect-count", action="append", default=[],
                        metavar="NAME=INTEGER",
                        help="pin a run count known from upstream (repeatable)")
    parser.add_argument("--env", action="append", default=[],
                        metavar="NAME=VALUE",
                        help="one resolved stage environment value (repeatable)")
    args = parser.parse_args(argv)
    try:
        result = seal(
            artifact_root=args.artifact_root.resolve(), run_path=args.run.resolve(),
            out_root=args.out.resolve(), release_dir=args.release_dir.resolve(),
            spool=args.spool.resolve(), source_id=args.source_id,
            staged_collection=args.staged_collection,
            staging_url=args.staging_url, stage_environment=_pairs(args.env),
            release_manifest_sha256=args.release_manifest_sha256,
            durability_receipt=args.durability_receipt.resolve(),
            extraction_sample=args.extraction_sample,
            expect_counts=_counts(args.expect_count))
    except (SourceEvidenceError, full.ContractError, OSError, KeyError,
            ValueError) as exc:
        print(f"SOURCE EVIDENCE FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"source_id": result["source_id"],
                      "vector_records": result["vector_records"],
                      "truncation_headroom": result["truncation_headroom"],
                      "extraction_reproduction": {
                          key: value for key, value
                          in result["extraction_reproduction"].items()
                          if key != "pages"}},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
