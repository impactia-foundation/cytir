#!/usr/bin/env python3
"""The committed plan must pin the committed inputs.

Cold-start reproducibility (Tim, 2026-08-19): a stranger with only a clean clone
must be able to run the flip gate and see what we see. Task 513 found they could
not: `assembly-plan.current.v1.json` pinned `claim-manifests.v1.json` at
c695f031… while the committed manifests hashed 758c2378… -- task 511 landed a
re-folded manifest index and committed a plan generated before it. Running the assembly guards script with
`--plan data/assembly/assembly-plan.current.v1.json` on a clean checkout returned 8 FAIL / `inputs-loadable: digest drift`, not the
6 PASS / 2 WAITING / 0 FAIL that tasks 505 and 511 both report -- because both
measured a plan generated during their own run and never landed it.

This test is the mechanism that makes that discoverable in CI rather than by a
stranger.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from tests.helpers.repo_files import repo_file_or_skip

REPO = pathlib.Path(__file__).resolve().parent.parent
PLAN = repo_file_or_skip("data/assembly/assembly-plan.current.v1.json")


def _sha(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


@pytest.mark.parametrize("section", ["inputs", "records"])
def test_every_pinned_artifact_matches_the_committed_file(section):
    plan = json.loads(PLAN.read_text())
    drift = []
    for key, binding in plan[section].items():
        uri = binding.get("uri")
        if not uri:
            continue
        path = REPO / uri
        if not path.exists():
            drift.append(f"{section}.{key}: {uri} does not exist")
            continue
        actual = _sha(uri)
        if actual != binding["sha256"]:
            drift.append(f"{section}.{key}: pinned {binding['sha256'][:12]}…, "
                         f"file is {actual[:12]}… ({uri})")
        if binding.get("bytes") and path.stat().st_size != binding["bytes"]:
            drift.append(f"{section}.{key}: byte count is stale")
    assert not drift, (
        "the committed plan pins something other than the committed file, so a "
        "clean clone cannot reproduce the guard result:\n  " + "\n  ".join(drift))
