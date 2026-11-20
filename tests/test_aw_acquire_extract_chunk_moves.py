#!/usr/bin/env python3
"""Task 677 (task 643 S5, second slice): the acquire / extract / chunk core moved
into the package.

    aw_common.py               -> cytir/core/common.py        (extract + chunk dispatch + Passage)
    pack_fetch.py              -> cytir/acquire/crawl.py
    fetch_process.py           -> cytir/acquire/fetch_process.py
    gz_shard_store.py          -> cytir/acquire/gz_shard_store.py
    chunk.py (deleted 1166)       -> cytir/chunk/chunker.py
    chunk_policy.py            -> cytir/chunk/policy.py
    cytir/chunk/registry.py   -> cytir/chunk/registry.py
    chunk_provenance.py        -> cytir/chunk/provenance.py   (shim removed, task 683 S9)
    dense_token_counter.py     -> cytir/chunk/token_counter.py
    model_assets.py            -> cytir/embed/model_assets.py

The old paths were alias shims; none remain, so the shim-identity, bare-import
and CLI-through-shim cases (zero cases since the shims went; removed task 2409; the arrow list is a literal, task 2427)
are gone. These tests prove (1) the moved code names no
sibling by its old flat name, (2) every `__file__`-relative data path still
lands in the repo's `data/` (the move is one directory deeper), and (3) a
golden: sealed pages in four languages
run through extract -> chunk -> Passage give the SAME bytes they gave BEFORE
the move (`fixtures/s5_acquire_extract_chunk/expected.json` was produced by the
pre-move tree, main@01bcf5cf).
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FIX = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "s5_acquire_extract_chunk"

#: shims removed over time: no caller names them, and no package module may either
REMOVED_SHIMS = {"chunk_provenance", "pack_fetch", "chunk_policy", "dense_token_counter",
                 "chunk", "chunk_policy_registry"}  # `chunk` task 1166; `chunk_policy_registry` task 1309
OLD_NAMES = set(REMOVED_SHIMS)
#: package modules held to the arrow (no old flat name, no sys.path edit): the chunker pair,
#: whose shims are gone but whose modules must stay clean. A literal, not derived from the tree.
ARROW = [("chunk", "cytir.chunk.chunker", False), ("chunk_policy_registry", "cytir.chunk.registry", False)]


#: Modules a caller may start BY PATH (`python3 cytir/core/common.py`): `sys.path[0]` is then their own
#: directory, so they put THIS checkout's repo root first before the first `aw` import (task 696
#: slice 3, test_aw_slice3_repoint.py). Only that one insert, of the repo root, is allowed.
ROOT_GUARD_MODULES = {"cytir.core.common", "cytir.acquire.sitemap_ingest", "cytir.sources.exact_url.cc_preserved",
                      "cytir.sources.exact_url.direct_preserved"}


def _is_repo_root_guard(module: str, call: ast.Call) -> bool:
    return (module in ROOT_GUARD_MODULES and len(call.args) == 2
            and ast.unparse(call.args[1]) == "_REPO_ROOT")


@pytest.mark.parametrize("old,new,_cli", ARROW)
def test_moved_code_names_no_sibling_by_its_old_flat_name(old, new, _cli):
    """The arrow: a package module imports the package, never a pipeline/ flat
    name (`import aw_common`, `from chunk_policy import ...`), and never edits
    sys.path to find one."""
    path = REPO / (new.replace(".", "/") + ".py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names if a.name.split(".")[0] in OLD_NAMES]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] in OLD_NAMES | {"aw_b1_identity"}:
                bad.append(node.module)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "insert"
              and "sys.path" in ast.unparse(node.func.value)
              and not _is_repo_root_guard(new, node)):
            bad.append("sys.path.insert")
    assert not bad, (new, bad)


def test_file_relative_data_paths_still_land_in_the_repo():
    """Every module here was one directory shallower. A wrong `parents[N]` would
    not raise; it would read a different (or no) file. Pin each to the repo."""
    from cytir.chunk import policy, registry
    from cytir.core import common
    from cytir.acquire import fetch_process
    from cytir.embed import model_assets

    assert policy.REPO == REPO
    assert policy.DEFAULT_POLICY_PATH.is_file()
    assert registry.REPO == REPO
    assert (registry.REPO / registry.DEFAULT_PATH).is_file()
    assert model_assets.REPO_ROOT == REPO
    # allowlist: the default path resolves to the tracked file
    assert common.allowlist_entry("admin.ch") is not None
    # build profile: no explicit selection reads data/build_config.json
    saved = os.environ.pop("AW_CHUNKER", None)
    try:
        assert common.chunker_selection().mode in common.CHUNKER_MODES
    finally:
        if saved is not None:
            os.environ["AW_CHUNKER"] = saved
    # wiki titles: the loader's first candidate is <repo>/data/...
    here = pathlib.Path(fetch_process.__file__).resolve().parent
    assert (here / "../../data").resolve() == REPO / "data"
    assert fetch_process.wiki_titles()


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, default=str).encode()).hexdigest()


def test_golden_extract_chunk_passage_bytes_equal_the_pre_move_tree(monkeypatch):
    """Sealed pages (DE/FR/IT/EN) -> markdown -> chunks -> Passage records.
    Expected values come from the tree BEFORE the move (legacy chunker, the
    live serving profile). One changed byte anywhere in extract, title,
    chunking, Passage construction or point-id minting changes a hash."""
    monkeypatch.setenv("AW_LANGGATE_OPTIONAL", "1")
    monkeypatch.setenv("AW_CHUNKER", "legacy")
    from cytir.core import common as C
    from cytir.acquire import fetch_process as FP

    pages = json.loads((FIX / "pages.json").read_text(encoding="utf-8"))
    expected = json.loads((FIX / "expected.json").read_text(encoding="utf-8"))
    assert set(pages) == set(expected) and len(pages) == 4
    for url, html in pages.items():
        want = expected[url]
        md, mode = C._extract_content_md(html, url)
        passages, links = C.process_html(url, html, {
            "content": True, "links": True,
            "passage_common": {"domain": "example.org", "tier": "A",
                               "source_type": "official", "trust_tier": 1.0}})
        units = C.chunk_markdown_channels(md)
        got = {
            "extract_mode": mode,
            "markdown_sha256": hashlib.sha256(md.encode()).hexdigest(),
            "n_units": len(units),
            "units_sha256": hashlib.sha256(json.dumps(
                [dataclasses.asdict(u) for u in units], sort_keys=True).encode()).hexdigest(),
            "n_passages": len(passages),
            "passages_sha256": _sha([dataclasses.asdict(p) for p in passages]),
            "links_sha256": hashlib.sha256(json.dumps(sorted(links)).encode()).hexdigest(),
            "legacy_window_sha256": hashlib.sha256(json.dumps(FP.chunk(md)).encode()).hexdigest(),
            "title": C.html_title(html),
        }
        assert got == want, url
