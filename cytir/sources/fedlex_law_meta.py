"""Re-seal tool for data/fedlex-law-names.v1.json (task 2535, from task 2325).

The curated law-names file is byte-sealed: a sealed artifact (a maintainer-only file that never
ships; the task doc names its path) records the file's sha256 under `curated_names.sha256`, and
a pinned test fails CI when the live bytes drift from it. The retired seal producer left no
tool to re-seal a prose edit. This is that tool.

    cyt source fedlex-law-meta reseal --artifact FILE [--dry-run] [--write-artifact]

It recomputes the sha256 of the names file and prints what the artifact holds next to what was
computed. Exit 1 when they differ and --write-artifact is absent (a check CI can use); exit 0
when they match, or when no artifact is given or present (the expected sha is printed). The
artifact is rewritten ONLY with --write-artifact. Running a real re-seal is Tim's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

from cytir.core.paths import data_path

NAMES_FILE = "fedlex-law-names.v1.json"
TARGET_PATH = f"data/{NAMES_FILE}"


class ResealRefused(Exception):
    """A named reason the re-seal cannot be done; nothing was written."""


def sealed_entry(doc: dict) -> dict:
    """The one dict whose `path` is the names file (the test's own rule: exactly one)."""
    found: list[dict] = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("path") == TARGET_PATH:
                found.append(node)
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(doc)
    if len(found) != 1:
        raise ResealRefused(f"expected exactly one sealed entry for {TARGET_PATH}, found {len(found)}")
    if not isinstance(found[0].get("sha256"), str):
        raise ResealRefused(f"the sealed entry for {TARGET_PATH} carries no sha256 string")
    return found[0]


def reseal(names_path: pathlib.Path, artifact_path: pathlib.Path | None, *, write_artifact: bool,
           dry_run: bool) -> dict:
    """{"computed", "sealed" (None when the artifact is absent), "artifact_text" (new, or None),
    "changed"}. Writes the artifact only when write_artifact and not dry_run."""
    if not names_path.is_file():
        raise ResealRefused(f"names file {names_path} does not exist")
    computed = hashlib.sha256(names_path.read_bytes()).hexdigest()
    sealed = None
    new_text = None
    if artifact_path is not None and artifact_path.is_file():
        text = artifact_path.read_text(encoding="utf-8")
        sealed = sealed_entry(json.loads(text))["sha256"]
        if text.count(sealed) != 1:
            raise ResealRefused(f"the sealed sha {sealed} occurs {text.count(sealed)} times in "
                                f"{artifact_path.name}; refusing a blind text rewrite")
        new_text = text.replace(sealed, computed)
        sealed_entry(json.loads(new_text))   # re-read: still exactly one entry
    elif write_artifact:
        raise ResealRefused(f"--write-artifact given but the sealed artifact {artifact_path} is absent "
                            "(it never ships); run from the maintainer checkout")
    changed = sealed is not None and sealed != computed
    if write_artifact and not dry_run and changed:
        artifact_path.write_bytes(new_text.encode("utf-8"))
    return {"computed": computed, "sealed": sealed, "artifact_text": new_text, "changed": changed}


def main(argv: list[str], prog: str = "cyt source fedlex-law-meta") -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    ap.add_argument("verb", choices=["reseal"])
    ap.add_argument("--dry-run", action="store_true", help="compute and print; write nothing")
    ap.add_argument("--write-artifact", action="store_true",
                    help="rewrite curated_names.sha256 in --artifact (maintainer checkout only)")
    ap.add_argument("--names", default=None, help=f"default data/{NAMES_FILE}")
    ap.add_argument("--artifact", default=None, help="the sealed artifact; no default (see the task doc)")
    a = ap.parse_args(argv)
    names = pathlib.Path(a.names) if a.names else data_path(NAMES_FILE)
    artifact = pathlib.Path(a.artifact) if a.artifact else None
    try:
        r = reseal(names, artifact, write_artifact=a.write_artifact, dry_run=a.dry_run)
    except ResealRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"computed sha256 of {names.name}: {r['computed']}")
    if r["sealed"] is None:
        print(f"sealed artifact {artifact} is absent or not given: expected sha printed only")
    else:
        print(f"sealed in {artifact.name}: {r['sealed']}  ({'DIFFERS' if r['changed'] else 'matches'})")
    if a.write_artifact and r["changed"]:
        print("dry run: artifact NOT written" if a.dry_run else f"artifact rewritten: {artifact}")
    elif a.write_artifact:
        print("artifact already holds the computed sha; nothing to write")
    print("the pinned test holds no literal: it reads the artifact, so after --write-artifact it "
          f"passes as is. Expected value: {r['computed']}")
    if r["changed"] and not a.write_artifact:
        print("REFUSED-CHECK: sealed sha differs from the computed sha", file=sys.stderr)
        return 1
    return 0
