"""Registry rows for files the public cut withholds (task 1934).

A row flagged `public: false` (with a `withheld_reason`) may have its file
ABSENT; every other check stays as strict as before.
"""
import copy
import fnmatch
import json

import pytest

from cytir.chunk import registry

REPO = registry.REPO
V4 = "data/chunk-reference-selector.v4.json"


def _v4_row():
    doc = registry.load()
    for sha, ent in doc["selectors"]["artifacts"].items():
        if ent["file"] == V4:
            return sha, ent, doc
    raise AssertionError("v4 row missing")


def _tmp_repo(tmp_path, *, flag, present, corrupt=False):
    sha, ent, real = _v4_row()
    ent = copy.deepcopy(ent)
    if flag:
        ent["public"] = False
        ent["withheld_reason"] = "test"
    (tmp_path / "data").mkdir()
    if present:
        body = (REPO / V4).read_bytes()
        if corrupt:
            body = body + b" "
        (tmp_path / V4).write_bytes(body)
    doc = {"policies": {}, "policy_families": real["policy_families"],
           "selectors": {"artifacts": {sha: ent}}}
    return doc, tmp_path


def test_withheld_row_absent_file_ok_and_listed(tmp_path):
    doc, repo = _tmp_repo(tmp_path, flag=True, present=False)
    assert registry.verify_against_files(doc, repo) == []
    assert registry.withheld(doc, repo) == [V4]


def test_withheld_row_present_file_still_strict(tmp_path):
    doc, repo = _tmp_repo(tmp_path, flag=True, present=True)
    assert registry.verify_against_files(doc, repo) == []
    other = tmp_path / "x"
    other.mkdir()
    doc, repo = _tmp_repo(other, flag=True, present=True, corrupt=True)
    assert any("sha256 on disk" in p
               for p in registry.verify_against_files(doc, repo))


def test_unflagged_missing_file_still_fails(tmp_path):
    doc, repo = _tmp_repo(tmp_path, flag=False, present=False)
    assert any("does not exist" in p
               for p in registry.verify_against_files(doc, repo))


def test_public_false_requires_reason(tmp_path):
    doc, repo = _tmp_repo(tmp_path, flag=True, present=False)
    ent = next(iter(doc["selectors"]["artifacts"].values()))
    del ent["withheld_reason"]
    problems = registry.verify_against_files(doc, repo)
    assert any("withheld_reason" in p for p in problems)
    assert any("does not exist" in p for p in problems)


def test_only_v1_to_v3_are_flagged():
    arts = registry.load()["selectors"]["artifacts"]
    flagged = sorted(e["file"] for e in arts.values() if e.get("public") is False)
    assert flagged == [f"data/chunk-reference-selector.v{n}.json"
                       for n in (1, 2, 3)]
    assert all(registry.is_withheld(e) for e in arts.values()
               if e.get("public") is False)


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)


def test_every_public_row_is_shipped_by_the_cut_manifest():
    manifest = REPO / "data" / "public-cut.v1.json"
    if not manifest.exists():
        pytest.skip("no cut manifest in this tree")
    pats = [s for s in _strings(json.loads(manifest.read_text()))
            if s.startswith("data/")]
    doc = registry.load()
    rows = list(doc["policies"].values()) + list(
        doc["selectors"]["artifacts"].values())
    for ent in rows:
        if ent.get("public", True) is False:
            continue
        assert any(fnmatch.fnmatch(ent["file"], p) for p in pats), ent["file"]
