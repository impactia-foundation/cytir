"""The contracts file never carries one key twice in an object (task 2227)."""
from __future__ import annotations

import json

import pytest

from cytir.sources import contract as contract_mod


def test_duplicate_key_is_found():
    text = '{"a": {"producer_ids": {"x": 1}, "producer_ids": {"y": 2}}, "b": 1}'
    assert contract_mod.duplicate_keys(text) == ["producer_ids"]


def test_same_key_in_sibling_objects_is_fine():
    text = '{"a": {"producer_ids": {}}, "b": {"producer_ids": {}}}'
    assert contract_mod.duplicate_keys(text) == []


def test_load_refuses_duplicate(tmp_path):
    good = contract_mod.data_path("source-family-contracts.v1.json").read_text()
    assert json.loads(good)
    p = tmp_path / "c.json"
    p.write_text(good.replace('"families"', '"zz": 1, "zz": 2, "families"', 1))
    with pytest.raises(contract_mod.FamilyContractError, match="duplicate key"):
        contract_mod.load(p)


def test_real_contracts_file_has_no_duplicate_key():
    p = contract_mod.data_path("source-family-contracts.v1.json")
    assert contract_mod.duplicate_keys(p.read_text()) == []
