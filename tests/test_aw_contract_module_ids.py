"""Task 1016 (Q13): the pinned contract data names modules by dotted id next to the legacy text.

The original key (`minted_by`, `declared_by`, `instrument`, `*_source`) is untouched and is the
legacy string the readers compare and stamp; a sibling `<key>_modules` list carries the `aw.*` ids.
Fixtures spell private names in pieces, so this file is not itself a leak."""
from __future__ import annotations

import copy
import importlib
import json
import pathlib

import pytest

from cytir.ops import public_cut as PC
from cytir.sources import contract as C
from cytir.sources.families import fedlex, pdf_replay

APP = pathlib.Path(__file__).resolve().parents[1]
P = "pipe" + "line"
FILES = ["source-family-contracts", "flip-gate-contract", "pack-surface-contract"]


def _walk(x, path=()):
    if isinstance(x, dict):
        for k, v in x.items():
            yield path, k, v, x
            yield from _walk(v, path + (k,))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _walk(v, path + (i,))


def _present():
    """The contract files that exist in the tree this runs in (the public cut ships only some)."""
    return [n for n in FILES if (APP / "data" / f"{n}.v1.json").is_file()]


def _module_entries():
    for name in _present():
        doc = json.loads((APP / "data" / f"{name}.v1.json").read_text(encoding="utf-8"))
        for path, k, v, parent in _walk(doc):
            if k.endswith("_modules"):
                yield name, path, k, v, parent


def test_there_are_module_entries():
    present = _present()
    if not present:
        pytest.skip("no contract data file in this tree")
    n = sum(1 for _ in _module_entries())
    if len(present) == len(FILES):
        assert n >= 40
    else:                                   # the cut ships a subset: assert only what is there
        assert n >= 1 or "source-family-contracts" not in present


def test_every_module_id_imports_and_has_its_legacy_text_beside_it():
    for name, path, k, mods, parent in _module_entries():
        base = k[: -len("_modules")]
        assert base in parent, (name, path, k)          # the legacy text is still there
        assert isinstance(mods, list) and mods and all(isinstance(m, str) for m in mods)
        for m in mods:
            assert m.startswith("cytir."), (name, path, m)
            importlib.import_module(m)


def test_v1_schema_shim_loads(tmp_path):
    doc = json.loads((APP / "data" / "source-family-contracts.v1.json").read_text(encoding="utf-8"))
    assert doc["schema"] == C.SCHEMA
    doc["schema"] = C.SCHEMA_V1
    p = tmp_path / "c.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    assert C.load(p)["schema"] == C.SCHEMA_V1
    doc["schema"] = "aw.source-family-contract.v0"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(C.FamilyContractError):
        C.load(p)


def test_compared_readers_see_the_same_legacy_string_with_or_without_modules():
    fams = C.load()["families"]
    checked = 0
    for row in fams:
        ident = row.get("identity") or {}
        by = ident.get("minted_by")
        stripped = copy.deepcopy(row)
        for k in [k for k in stripped["identity"] if k.endswith("_modules")]:
            del stripped["identity"][k]
        assert stripped["identity"].get("minted_by") == by          # legacy text byte-identical
        assert fedlex.door_declaration_refusal(by) == fedlex.door_declaration_refusal(
            stripped["identity"].get("minted_by"))
        try:
            a = pdf_replay.id_rule(row)
        except pdf_replay.ReplayError:
            a = "refused"
        try:
            b = pdf_replay.id_rule(stripped)
        except pdf_replay.ReplayError:
            b = "refused"
        assert a == b
        checked += 1
    assert checked > 50


def test_known_legacy_strings_still_accepted():
    assert pdf_replay.id_rule({"identity": {"minted_by": "text-anchor-v0"}}) == pdf_replay.TEXT_ANCHOR_V0
    assert "id rule" not in (fedlex.door_declaration_refusal(fedlex.DOOR_ID_RULE) or "")
    # task 1038 (ruling 30): the contract row names `mint_point_id`; the generation-0 label is refused
    assert "id rule" in fedlex.door_declaration_refusal("text-anchor-v0")
    assert "id rule" not in (fedlex.door_declaration_refusal("mint_point_id") or "")


# --- the ratified prose exemption (RULING 46, option c) -----------------------------------------

def test_prose_exempt_names_only_existing_files_with_a_reason():
    doc = PC.load_allowlist(APP)
    entries = doc["prose_exempt"]
    assert entries
    for e in entries:
        assert (APP / e["path"]).is_file(), e
        assert "RULING 46" in e["why"]
        assert "*" not in e["path"]                       # an explicit list, never a glob


def test_prose_mention_is_not_counted_but_a_bare_path_value_is():
    prose = {"evidence": ["see pipeline/aw_b1_license.py for the licence rule"],
             "note": "pipeline/x.py resolves it"}
    assert PC.scan_bare_values(json.dumps(prose)) == {}
    bare = {"instrument": "pipeline/some_tool.py", "evidence": ["prose pipeline/y.py here"]}
    assert PC.scan_bare_values(json.dumps(bare)) == {"pipeline_ref": 1}
    assert PC.scan_text(json.dumps(prose))                 # the plain scanner would count it


def test_bare_value_with_surrounding_whitespace_and_a_path_key_count():
    padded = {"instrument": "  pipeline/some_tool.py\n"}
    assert PC.scan_bare_values(json.dumps(padded)) == {"pipeline_ref": 1}
    keyed = {"pipeline/some_tool.py": "a sentence"}
    assert PC.scan_bare_values(json.dumps(keyed)) == {"pipeline_ref": 1}
    assert PC.scan_bare_values(json.dumps({"a pipeline/x.py b": 1})) == {}   # a sentence key is prose
