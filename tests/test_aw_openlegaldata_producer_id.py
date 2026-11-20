"""The openlegaldata producer label is a declared, frozen rule (task 1316 / D6)."""
import json
import pathlib

from cytir.extract import openlegaldata

CONTRACTS = pathlib.Path(__file__).resolve().parent.parent / "data" / "source-family-contracts.v1.json"


def _row():
    doc = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    return next(f for f in doc["families"] if f["family_id"] == "de_openlegaldata_v1")


def test_producer_constant_is_the_declared_frozen_key():
    declared = _row()["producer_ids"]
    assert list(declared) == [openlegaldata.PRODUCER]
    entry = declared[openlegaldata.PRODUCER]
    assert entry["status"] == "frozen-label"
    assert entry["code_home"] == "cytir/extract/openlegaldata.py"
    assert pathlib.Path(__file__).resolve().parent.parent.joinpath(entry["code_home"]).is_file()
