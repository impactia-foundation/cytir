"""repoint-policy: the one way to move a registered chunk policy to new bytes (task 2535).

The WRITE path lives here, not in `cytir.chunk.registry`: that module is imported by the drain
(promote_queue), whose import closure must not gain write sites (task 2574,
tests/test_promote_queue_dry_run.py). `cytir/cli.py` imports this module lazily.
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

from cytir.chunk.registry import (DEFAULT_PATH, REPO, ChunkPolicyRegistryError, data_path, load,
                               verify_against_files)

# ---------------------------------------------------------------------------
# repoint-policy: the one way to move a registered policy to new bytes (task 2535)
# ---------------------------------------------------------------------------
class RepointRefused(Exception):
    """A named reason the repoint would be wrong; nothing was written."""


def plan_repoint(old_sha: str, new_sha: str, *, registry_path: pathlib.Path,
                 contracts_path: pathlib.Path, repo: pathlib.Path, reason: str,
                 task: str = "2535") -> dict[str, Any]:
    """Compute every rewrite; write nothing. Raises RepointRefused.

    Returns {"registry_text", "contracts_text" (None when no family pins OLD), "counts",
    "families"}. Every occurrence of OLD in the registry (the `policies` key, the family's
    `current_policy_sha256`, a `successor_sha256`) is rewritten textually, so formatting and
    key order are untouched; the family-contract pin goes through
    `cytir.sources.contract.amend`, the repo's one way to edit sealed contracts (task 2496).
    """
    import hashlib
    for label, sha in (("--old-sha", old_sha), ("--new-sha", new_sha)):
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise RepointRefused(f"{label} is not a lowercase 64-hex sha256: {sha!r}")
    if old_sha == new_sha:
        raise RepointRefused("--old-sha equals --new-sha: nothing to repoint")
    if not reason.strip():
        raise RepointRefused("--reason is required")
    text = registry_path.read_text(encoding="utf-8")
    doc = load(registry_path)
    ent = doc["policies"].get(old_sha)
    if ent is None:
        raise RepointRefused(f"OLD sha {old_sha} is not a key of {registry_path.name} policies")
    if new_sha in doc["policies"]:
        raise RepointRefused(f"NEW sha {new_sha} is already registered; repoint moves a policy, "
                             "it never merges two")
    policy_file = repo / ent.get("file", "")
    if not policy_file.is_file():
        raise RepointRefused(f"registry entry names {ent.get('file')!r}, which does not exist under {repo}")
    have = hashlib.sha256(policy_file.read_bytes()).hexdigest()
    if have != new_sha:
        raise RepointRefused(f"NEW sha {new_sha} is not the sha256 of {ent.get('file')} (that file "
                             f"hashes to {have}); a re-seal means the file's bytes already changed")
    counts = text.count(old_sha)
    structural = _exact_string_hits(doc, old_sha)
    if counts != structural:
        raise RepointRefused(
            f"{counts} occurrence(s) of OLD in the registry text but only {structural} as a whole "
            "key or value: the rest sit inside prose, which a blind replace would rewrite. Edit "
            "the prose first; nothing was written")
    new_text = text.replace(old_sha, new_sha)
    problems = verify_against_files(json.loads(new_text), repo)
    if problems:
        raise RepointRefused("registry --verify would fail afterwards: " + "; ".join(problems[:5]))

    from cytir.sources import contract as C
    cdoc = json.loads(contracts_path.read_text(encoding="utf-8"))
    pinned = [f["family_id"] for f in cdoc.get("families", [])
              if (f.get("chunk_policy") or {}).get("sha256") == old_sha]
    if len(pinned) > 1:
        raise RepointRefused(f"{len(pinned)} families pin OLD ({pinned}); the amend mechanism "
                             "appends one row per family, repoint them one at a time by hand")
    new_ctext = None
    if pinned:
        before = _contract_problems(C, cdoc, repo)

        def check(edited: str) -> list[str]:
            # the edited contract is checked against the REWRITTEN registry, as --check will be
            dups = C.duplicate_keys(edited)
            if dups:
                return ["duplicate key: " + ", ".join(sorted(set(dups)))]
            rec, comp = C.pre_amendment_sha256(edited)
            if rec != comp:
                return [f"pre-amendment sha MISMATCH recorded={rec} computed={comp}"]
            return [p for p in _contract_problems(C, json.loads(edited), repo, registry_text=new_text)
                    if p not in before]

        _old, new_ctext = C.amend(contracts_path, pinned[0], [("chunk_policy.sha256", new_sha)],
                                  reason, task, check=check)
    return {"registry_text": new_text, "contracts_text": new_ctext,
            "counts": {"registry_occurrences": counts}, "families": pinned}


def _exact_string_hits(node: Any, needle: str) -> int:
    """How many dict keys / string values / list items equal `needle` exactly."""
    if isinstance(node, dict):
        return sum((k == needle) + _exact_string_hits(v, needle) for k, v in node.items())
    if isinstance(node, list):
        return sum(_exact_string_hits(v, needle) for v in node)
    return int(node == needle)


def _write_pair_atomically(first: tuple[pathlib.Path, bytes], second: tuple[pathlib.Path, bytes] | None) -> None:
    """Temp file + os.replace for each, in order. If the second replace fails, the first file is
    restored from its saved bytes and the error is re-raised."""
    import os
    saved = first[0].read_bytes()
    staged: list[tuple[pathlib.Path, pathlib.Path]] = []
    try:
        for path, data in (first, second) if second else (first,):
            tmp = path.with_name(path.name + ".tmp-repoint")
            tmp.write_bytes(data)
            staged.append((tmp, path))
        os.replace(*staged[0])
        try:
            if second:
                os.replace(*staged[1])
        except OSError:
            first[0].write_bytes(saved)
            raise
    finally:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)


def _contract_problems(C, cdoc: dict, repo: pathlib.Path, registry_text: str | None = None) -> list[str]:
    """validate_family problems of the contract's families against `repo`'s data/ (and, when
    given, a registry text that is not on disk yet)."""
    import tempfile
    probs: list[str] = []
    if registry_text is None:
        for fam in cdoc.get("families", []):
            probs.extend(C.validate_family(fam, repo=repo))
        return probs
    with tempfile.TemporaryDirectory() as td:       # a repo view whose registry is the new text
        root = pathlib.Path(td)
        (root / "data").mkdir()
        for f in repo.iterdir():      # code_home paths under cytir/ must still resolve
            if f.name not in ("data", ".git"):
                (root / f.name).symlink_to(f.resolve())
        for f in (repo / "data").iterdir():
            if f.is_file():
                (root / "data" / f.name).symlink_to(f.resolve())
        reg = root / "data" / pathlib.Path(DEFAULT_PATH).name
        reg.unlink(missing_ok=True)
        reg.write_text(registry_text, encoding="utf-8")
        for fam in cdoc.get("families", []):
            probs.extend(C.validate_family(fam, repo=root))
    return probs


def repoint_main(argv: list[str], prog: str = "cyt chunk registry repoint-policy") -> int:
    import argparse
    import difflib
    ap = argparse.ArgumentParser(prog=prog, description=(
        "Move one registered chunk policy to its new bytes: rewrite the registry's keys that carry "
        "OLD, amend the family-contract pin, refuse when the result would fail --verify/--check."))
    ap.add_argument("--old-sha", required=True)
    ap.add_argument("--new-sha", required=True)
    ap.add_argument("--reason", required=True)
    ap.add_argument("--task", default="2535", help="task number for the amendments row")
    ap.add_argument("--dry-run", action="store_true", help="print the planned rewrites, write nothing")
    ap.add_argument("--registry", default=None, help=f"default {DEFAULT_PATH}")
    ap.add_argument("--contracts", default=None, help="default data/source-family-contracts.v1.json")
    ap.add_argument("--repo", default=None, help="root holding data/chunk-policy-*.json (default: this checkout)")
    a = ap.parse_args(argv)
    from cytir.sources import contract as C
    reg = pathlib.Path(a.registry) if a.registry else data_path(pathlib.Path(DEFAULT_PATH).name)
    con = pathlib.Path(a.contracts) if a.contracts else data_path(pathlib.Path(C.DEFAULT_PATH).name)
    repo = pathlib.Path(a.repo) if a.repo else REPO
    try:
        dirty = C._vcs_dirty(con)
        if dirty:
            raise RepointRefused(f"{con.name} has uncommitted changes ({dirty}); old_file_sha256 "
                                 "must name a committed revision")
        plan = plan_repoint(a.old_sha, a.new_sha, registry_path=reg, contracts_path=con,
                            repo=repo, reason=a.reason, task=a.task.removeprefix("task").strip())
    except (RepointRefused, ChunkPolicyRegistryError, C.FamilyContractError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"registry {reg.name}: {plan['counts']['registry_occurrences']} occurrence(s) of OLD -> NEW")
    print(f"contract pin: families {plan['families'] or 'none'} (key chunk_policy.sha256, via amend)")
    if a.dry_run:
        for name, path, new in (("registry", reg, plan["registry_text"]),
                                ("contracts", con, plan["contracts_text"])):
            if new is not None:
                old = path.read_text(encoding="utf-8")
                sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                                           f"a/{name}", f"b/{name}", n=0))
        print("dry run: nothing written")
        return 0
    second = (con, plan["contracts_text"].encode("utf-8")) if plan["contracts_text"] is not None else None
    try:
        _write_pair_atomically((reg, plan["registry_text"].encode("utf-8")), second)
    except OSError as exc:
        print(f"REFUSED: write failed ({exc}); the registry was restored, nothing changed", file=sys.stderr)
        return 1
    print("written; now run `python3 -m cytir.chunk.registry --verify` and "
          "`python3 -m cytir.sources.contract --check`")
    return 0


