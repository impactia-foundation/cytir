"""Task 771: `ChunkerSelection.stamps()` / `.provenance()` do not depend on the checkout path.

The stamp names the policy by its repo-relative path and its content sha256 (never an absolute
path); a change to the policy file's CONTENT still changes it. The four families that used a
per-family `_portable_stamps` (service_public_fiche, stackoverflow_qa, stackexchange_nonso_qa_p0,
openalex_academic_dump) got exactly this value, so their rows are unchanged."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import shutil

from cytir.core import common as C

REPO = pathlib.Path(__file__).resolve().parent.parent
POLICY = REPO / "data" / ("chunk-policy-struct-md-768w-o0-adjacent-sections-"
                          "tail-1152-tok2560.v1.json")
REV = "a" * 40


def _sel(path: pathlib.Path, sha: str | None = None):
    return C.explicit_structured_chunker_selection(
        str(path), sha or hashlib.sha256(path.read_bytes()).hexdigest(), REV)


def test_the_stamp_is_the_same_from_two_checkout_paths(tmp_path):
    other = tmp_path / "x" / "y" / "data" / POLICY.name
    other.parent.mkdir(parents=True)
    shutil.copy(POLICY, other)
    here, there = _sel(POLICY), _sel(other)
    assert here.policy_path != there.policy_path           # the selections do differ
    assert here.stamps() == there.stamps()
    assert here.provenance() == there.provenance()
    assert str(tmp_path) not in json.dumps(there.provenance())
    assert here.provenance()["chunker"]["policy_path"] == f"data/{POLICY.name}"


def test_the_stamp_is_what_the_per_family_portable_stamps_minted():
    sel = _sel(POLICY)
    old = dataclasses.replace(sel, policy_path=f"data/{POLICY.name}")
    assert sel.stamps() == old.stamps()


def test_the_stamp_changes_when_the_policy_content_changes(tmp_path):
    sel = _sel(POLICY)
    other = dataclasses.replace(sel, policy_sha256="0" * 64)
    assert other.stamps()["prov_id"] != sel.stamps()["prov_id"]
