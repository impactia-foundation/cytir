#!/usr/bin/env python3
"""Prove a source's evidence exists off-box, and seal that proof.

The content-pack contract requires `durability.status == "verified_offbox"`
per source. Nothing produced that receipt for the news lane, and the one
artefact the contract cannot re-derive — the exact publisher response bytes —
is precisely the one whose off-box copy went missing for a while (task 450 §7).

"Verified" here is not a directory listing. The Hetzner Storage Box runs a
restricted shell with no `find`, and a file count would not catch a truncated
or stale copy anyway. Instead the lane's **own transport** is asked the only
question that matters: *is there anything left to send?* A dry-run `rsync`
of the local tree against its remote counterpart itemizes every file that
would still be transferred; zero is the pass condition, and any pending file
is named in the failure. That compares size and mtime per file, so a short,
missing or stale remote copy all fail closed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from cytir.acquire.artifact_io import atomic_publish, sha256_file 


RECEIPT_SCHEMA = "aw.offbox-durability-receipt.v1"
# rsync -i codes: the first column is the update type, the second the entry
# type. `>` means "would be sent to the remote"; `f` means a regular file.
PENDING_PREFIXES = (">f", "<f", "cf")


class DurabilityError(RuntimeError):
    """The evidence is not proven to exist off-box."""


def local_inventory(path: Path) -> dict:
    files = [item for item in path.rglob("*") if item.is_file()]
    if not files:
        raise DurabilityError(f"local evidence tree is empty: {path}")
    return {"files": len(files), "bytes": sum(item.stat().st_size
                                              for item in files)}


def rsync_pending(*, rsh: str, box: str, bwlimit: str = "8M"):
    """Files a dry-run rsync would still send for one tree, via the lane's RSH."""
    def pending(local: Path, remote: str) -> list[str]:
        # An empty `box` addresses a local path, which is how this comparison
        # is exercised against real rsync in the tests.
        transport = ["-e", rsh] if box else []
        target = f"{box}:{remote.rstrip('/')}/" if box else f"{remote.rstrip('/')}/"
        command = ["rsync", "-a", "--dry-run", "--itemize-changes",
                   f"--bwlimit={bwlimit}", "--timeout=600", *transport,
                   f"{local}/", target]
        run = subprocess.run(command, capture_output=True, text=True,
                             timeout=3600)
        if run.returncode:
            raise DurabilityError(
                f"off-box comparison failed for {remote}: "
                f"{run.stderr.strip()[:400]}")
        return [line for line in run.stdout.splitlines()
                if line[:2] in PENDING_PREFIXES]
    return pending


def seal(*, artifact_root: Path, receipt_path: Path, source_id: str,
         remote_host: str, remote_root: str, trees: list[tuple[str, str]],
         pending) -> dict:
    """`trees` = [(local path relative to artifact_root, remote suffix)]."""
    checked = []
    for relative, remote_suffix in trees:
        local_path = artifact_root / relative
        local = local_inventory(local_path)
        remote_path = f"{remote_root.rstrip('/')}/{remote_suffix.lstrip('/')}"
        outstanding = pending(local_path, remote_path)
        if outstanding:
            raise DurabilityError(
                f"off-box copy of {relative} is not current: "
                f"{len(outstanding)} file(s) would still be sent, first "
                f"{outstanding[0]!r}")
        checked.append({"path": relative, "remote_path": remote_path,
                        "local": local, "pending_transfers": 0})
    receipt = {
        "schema": RECEIPT_SCHEMA, "status": "verified_offbox",
        "source_id": source_id, "remote_host": remote_host,
        "remote_root": remote_root, "trees": checked,
        "method": ("a dry-run rsync of each local tree against its remote "
                   "counterpart, over the lane's own transport; the pass "
                   "condition is that nothing would still be sent, so a "
                   "missing, short or stale remote file all fail closed"),
    }
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    if receipt_path.exists():
        if receipt_path.read_bytes() != encoded:
            raise DurabilityError(f"immutable collision: {receipt_path}")
    else:
        atomic_publish(receipt_path, encoded)
    relative = receipt_path.resolve().relative_to(artifact_root.resolve())
    return {"uri": f"artifact://{relative.as_posix()}",
            "bytes": receipt_path.stat().st_size,
            "sha256": sha256_file(receipt_path), "trees": len(checked)}


def _tree(value: str) -> tuple[str, str]:
    local, _, remote = value.partition("=")
    if not local or not remote:
        raise argparse.ArgumentTypeError(
            f"--tree wants LOCAL_RELATIVE=REMOTE_SUFFIX, got {value!r}")
    return local, remote


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--box", required=True,
                        help="rsync remote, e.g. user@host")
    parser.add_argument("--rsh", required=True,
                        help="the -e transport the lane's own sync uses")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--bwlimit", default="8M")
    parser.add_argument("--tree", type=_tree, action="append", required=True)
    args = parser.parse_args()
    try:
        result = seal(
            artifact_root=args.artifact_root.resolve(),
            receipt_path=args.receipt.resolve(), source_id=args.source_id,
            remote_host=args.box, remote_root=args.remote_root,
            trees=args.tree,
            pending=rsync_pending(rsh=args.rsh, box=args.box,
                                  bwlimit=args.bwlimit))
    except (DurabilityError, OSError, ValueError,
            subprocess.SubprocessError) as exc:
        print(f"OFFBOX DURABILITY FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
