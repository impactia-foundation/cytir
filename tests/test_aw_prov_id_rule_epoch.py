"""ADR-117 am.1 (task 1361): `rule_epochs.prov_id` names the portable prov_id rule."""
from cytir.core import ids as ID
from cytir.core import projection as PROJ

REC = {"url": "https://example.org/a", "text": "Some words in a passage.", "source_type": "web",
       "consent_search_ok": True}


def test_projected_row_carries_prov_id_epoch_2():
    _pid, _d, _s, payload = PROJ.project_record(REC)
    assert payload["rule_epochs"]["prov_id"] == 2
    assert ID.rule_epoch(payload, "prov_id") == 2


def test_preserved_old_row_without_entry_stays_absent_and_reads_1():
    old = {**REC, "rule_epochs": {"chunk": ID.EPOCHS["chunk"]}}
    _pid, _d, _s, payload = PROJ.project_record(old, preserve_rule_epochs=True)
    assert "prov_id" not in payload["rule_epochs"]
    assert ID.rule_epoch(payload, "prov_id") == 1


def test_preserved_row_with_entry_keeps_it():
    new = {**REC, "rule_epochs": {"chunk": ID.EPOCHS["chunk"], "prov_id": 2}}
    _pid, _d, _s, payload = PROJ.project_record(new, preserve_rule_epochs=True)
    assert payload["rule_epochs"]["prov_id"] == 2


def test_old_payload_reads_as_path_bound_1():
    assert ID.rule_epoch({"rule_epochs": {"chunk": 4}}, "prov_id") == 1
    assert ID.rule_epoch({}, "prov_id") == 1
    assert ID.rule_epoch(None, "prov_id") == 1
    assert ID.rule_epoch({"rule_epochs": {"prov_id": 1}}, "prov_id") == 1


def test_other_absent_keys_read_none():
    assert ID.rule_epoch({"rule_epochs": {}}, "chunk") is None


def test_every_projected_rule_epochs_value_has_a_code_in_the_committed_table():
    """`rule_epochs` is an enum key: a value with no code is refused by a live writer (task 675).
    The new prov_id=2 value of every base dict in the table must have its own appended code."""
    import json
    from cytir.index import payload_profile as PP
    with open("data/payload-enum-tables.v1.json") as f:
        table = set(json.load(f)["tables"]["rule_epochs"].values())
    base = [json.loads(v) for v in table if "prov_id" not in json.loads(v)]
    assert base
    for b in base:
        assert PP._canon({**b, "prov_id": 2}) in table
    _p, _d, _s, payload = PROJ.project_record(REC)
    assert PP._canon(payload["rule_epochs"]) in table
    for b in base:
        _p, _d, _s, payload = PROJ.project_record({**REC, "rule_epochs": dict(b)}, preserve_rule_epochs=True)
        assert PP._canon(payload["rule_epochs"]) in table
