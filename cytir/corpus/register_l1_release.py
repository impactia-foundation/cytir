#!/usr/bin/env python3
"""File a source's already-fetched text into the ADR-105 registered L1 tree.

WHY THIS EXISTS
----------------
Task 641 (2026-09-23): `promote_queue.py`'s persisted-inputs guard refused
CC tranche 1 (67,927/67,927 pages) and, on inspection, would refuse every
StackExchange pack too -- not because the text was never fetched, but
because it was fetched and staged OUTSIDE the registered `l1/` tree that
`data/persisted-input-stores.v1.json` globs (CC tranche 1's raw pages lived
in `<work>/task634/packs/tranche1/*.jsonl`; SE's in
`<work>/depth226/staged/*.passages.jsonl`). Both files already carry a
`url`/`canonical_url` field and a `text`/`markdown` field -- exactly the
`l1-release-plain` store's shape (`l1/**/*.jsonl`) -- they were simply never
filed where the registry looks.

The first fix was two `ln -sfn` loops typed by hand on the build box -- and the
coordinator caught the real defect in that: L1 is the PERSISTED layer
(ADR-105 exists so a collection can be rebuilt WITHOUT re-fetching), and a
symlink into a scratch work dir (`task634/packs/...`, `depth226/staged/...`)
makes "persisted" depend on a directory someone can legitimately clean up or
overwrite -- exactly the failure ADR-105 exists to prevent (task 588 filed
by digest-verified COPY for the identical reason). So the default here is a
verified COPY: read the source, write the destination, sha256 BOTH, refuse
if they don't match, and record every file's sha256 in a release manifest
that never gets globbed as content (`_register_manifest.json` -- `.json`,
not `.jsonl`). `--link` is kept for a fast local dry run only; the real
persisted-input index in `persisted_inputs.py` REFUSES to index a store file
that is a symlink (see `_refuse_symlink` there), so a `--link` release can
never actually pass the promotion path's ADR-105 check.

USAGE
-----
  python3 -m cytir.corpus.register_l1_release \\
      --source <work-root>/task634/packs/tranche1 \\
      --release task641-cc-tranche1-w7

  python3 -m cytir.corpus.register_l1_release \\
      --source <work-root>/depth226/staged \\
      --release task641-se-nonso-p0 --file-suffix .passages.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pathlib
import shutil
import time
from typing import Iterable

from cytir.core import paths as aw_paths

DEFAULT_URL_FIELDS = ("canonical_url", "url")
DEFAULT_TEXT_FIELDS = ("markdown", "text")
#: Files that are metadata ABOUT a pack, never the pack's own rows -- these
#: sit next to real `.jsonl` files in every source dir this tool has seen
#: (task634's `*.receipt.json`, `*.vector_receipt.json`) and must never be
#: mistaken for L1 content just because they end in `.json`-adjacent names.
RECEIPT_SUFFIXES = (".receipt.json", ".vector_receipt.json", ".manifest.json")
#: The per-release manifest this tool writes -- ends in `.json`, never
#: `.jsonl`/`.jsonl.gz`, so no store glob in
#: `data/persisted-input-stores.v1.json` ever mistakes it for content.
MANIFEST_NAME = "_register_manifest.json"
_CHUNK = 8 * 1024 * 1024


class RegisterL1Error(Exception):
    pass


def _candidate_files(source: pathlib.Path, file_suffix: str | None,
                     recursive: bool = False) -> list[pathlib.Path]:
    out = []
    for pattern in ("*.jsonl", "*.jsonl.gz"):
        found = source.rglob(pattern) if recursive else source.glob(pattern)
        for f in sorted(found):
            if recursive and (f.is_symlink() or not f.is_file()):
                continue
            name = f.name
            if any(name.endswith(suf) for suf in RECEIPT_SUFFIXES):
                continue
            if file_suffix and not name.endswith(file_suffix):
                continue
            out.append(f)
    return out


def _open_maybe_gz(path: pathlib.Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def _first_record(path: pathlib.Path) -> dict:
    with _open_maybe_gz(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            return json.loads(line)
    raise RegisterL1Error(f"{path}: no non-empty line to sample")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def guard_field_shape(path: pathlib.Path, *, url_fields: Iterable[str] = DEFAULT_URL_FIELDS,
                       text_fields: Iterable[str] = DEFAULT_TEXT_FIELDS) -> dict:
    """Refuse a file that does not actually carry what the registry's
    `l1-release-plain`/`l1-release-gz` stores require (`url_fields`,
    `text_fields` in `data/persisted-input-stores.v1.json`). Checked on the
    FIRST record only -- cheap, and the whole point is a fast mechanical
    gate before filing, not a full validation pass (persisted_inputs.py's
    own indexer reads every row anyway once filed)."""
    rec = _first_record(path)
    has_url = next((f for f in url_fields if rec.get(f)), None)
    has_text = next((f for f in text_fields if rec.get(f)), None)
    if not has_url:
        raise RegisterL1Error(
            f"{path}: first record has none of url_fields={list(url_fields)} "
            f"(keys present: {sorted(rec.keys())})")
    if not has_text:
        raise RegisterL1Error(
            f"{path}: first record has none of text_fields={list(text_fields)} "
            f"(keys present: {sorted(rec.keys())})")
    return {"url_field": has_url, "text_field": has_text, "sample_keys": sorted(rec.keys())}


def _require_space(src_files: list[pathlib.Path], dest_dir: pathlib.Path) -> int:
    """Refuse a copy that would not fit. Returns the total source bytes."""
    total = sum(f.stat().st_size for f in src_files)
    usage = shutil.disk_usage(dest_dir if dest_dir.exists() else dest_dir.parent.parent
                               if dest_dir.parent.exists() else dest_dir.anchor or "/")
    # A 10% margin, matching the caution the coordinator asked for --
    # "check free space first" is a refusal-before-writing guard, not a
    # best-effort try.
    if usage.free < total * 1.10:
        raise RegisterL1Error(
            f"not enough free space at {dest_dir}: need ~{total:,} bytes "
            f"(+10% margin), have {usage.free:,}")
    return total


def register_l1_release(source: str, release_name: str, *, corpus_root: str | None = None,
                         file_suffix: str | None = None,
                         url_fields: Iterable[str] = DEFAULT_URL_FIELDS,
                         text_fields: Iterable[str] = DEFAULT_TEXT_FIELDS,
                         dry_run: bool = False, link: bool = False,
                         recursive: bool = False) -> dict:
    """File `source`'s *.jsonl[.gz] files into `<corpus_root>/l1/release=<release_name>/`.

    Default (link=False): COPY each file, sha256 the source and the copy,
    refuse if they differ, and write a `_register_manifest.json` recording
    every file's sha256 -- the persisted record does not depend on the
    source directory surviving. `link=True` symlinks instead (fast, for a
    local dry run only); `persisted_inputs.build_index` refuses to index a
    symlinked store file, so a `--link` release can never satisfy the real
    ADR-105 gate on the promotion path.

    `recursive=True` (task 686, OpenAlex): walk `source` at any depth (the
    materializer's `stages/markdown/data/<xx>/<sha>.jsonl.gz` layout) and
    keep each file's path relative to `source` under the release dir, so two
    shards can never collide on a name. Symlinked source files are skipped
    in the recursive walk: a symlink is not a persisted copy.
    """
    src = pathlib.Path(source)
    if not src.is_dir():
        raise RegisterL1Error(f"source {src} is not a directory")
    croot = pathlib.Path(corpus_root or aw_paths.CORPUS_ROOT)
    dest = croot / "l1" / f"release={release_name}"
    files = _candidate_files(src, file_suffix, recursive)
    if not files:
        raise RegisterL1Error(
            f"no *.jsonl/*.jsonl.gz files under {src} "
            f"(file_suffix={file_suffix!r}) -- nothing to register")
    checked = []
    for f in files:
        checked.append({"file": f.name, **guard_field_shape(
            f, url_fields=url_fields, text_fields=text_fields)})
    if dry_run:
        return {"dry_run": True, "release": release_name, "dest": str(dest),
                "files": len(files), "link": link, "checked": checked}

    if link:
        dest.mkdir(parents=True, exist_ok=True)
        linked = []
        for f in files:
            out = dest / f.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.is_symlink() or out.exists():
                out.unlink()
            out.symlink_to(f.resolve())
            linked.append(str(out))
        return {"dry_run": False, "release": release_name, "dest": str(dest),
                "files": len(files), "link": True, "linked": linked, "checked": checked}

    total_bytes = _require_space(files, dest)
    dest.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    copied = []
    for f in files:
        src_sha = sha256_file(f)
        rel = f.relative_to(src)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")
        shutil.copyfile(f, tmp)
        dst_sha = sha256_file(tmp)
        if dst_sha != src_sha:
            tmp.unlink(missing_ok=True)
            raise RegisterL1Error(
                f"{f}: copy verification FAILED -- source sha256 {src_sha}, "
                f"copy sha256 {dst_sha}. Nothing was filed under {out}.")
        tmp.replace(out)
        copied.append(str(out))
        manifest_entries.append({
            "file": str(rel), "source_path": str(f.resolve()),
            "sha256": src_sha, "bytes": f.stat().st_size})
    manifest = {
        "schema": "aw.register-l1-release-manifest.v1",
        "release": release_name, "registered_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dir": str(src.resolve()), "total_bytes": total_bytes,
        "files": manifest_entries,
    }
    (dest / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1) + "\n",
                                       encoding="utf-8")
    return {"dry_run": False, "release": release_name, "dest": str(dest),
            "files": len(files), "link": False, "copied": copied,
            "total_bytes": total_bytes, "checked": checked}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="directory of staged *.jsonl[.gz] files")
    ap.add_argument("--release", required=True,
                     help="release name -> filed under l1/release=<release>/")
    ap.add_argument("--corpus-root", default=None)
    ap.add_argument("--file-suffix", default=None,
                     help="only file files whose name ends with this suffix")
    ap.add_argument("--recursive", action="store_true",
                     help="walk --source at any depth; relative paths are kept under the release")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--link", action="store_true",
                     help="symlink instead of copy -- local dry runs only; "
                          "the ADR-105 index refuses to index a symlinked store file")
    args = ap.parse_args(argv)
    result = register_l1_release(args.source, args.release, corpus_root=args.corpus_root,
                                  file_suffix=args.file_suffix, dry_run=args.dry_run,
                                  link=args.link, recursive=args.recursive)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
