"""Task 1391 / ADR-124: `family_id` is stamped by the family door, indexed, counted, never dumped."""
import json
import pathlib

from cytir.core import common
from cytir.core import projection as P
from cytir.core import serving_config
from cytir.distribute import rehearsal
from cytir.ops import family_census as FC
from cytir.sources import contract as contract_mod
from cytir.sources import run_family

REPO = pathlib.Path(__file__).resolve().parent.parent
SAMPLE = REPO / "data" / "samples" / "hf-model-cards"
OK = {"bucket": "ok", "search_ok": True, "train_ok": True, "status": "ok"}


def _declared_contracts(tmp_path):
    """The shipped row declares no consent basis, so the door refuses it (task 1406); this test
    exercises the stamp, so it runs a copy whose row declares the live per-origin basis."""
    doc = json.loads((REPO / contract_mod.DEFAULT_PATH).read_text(encoding="utf-8"))
    row = next(f for f in doc["families"] if f["family_id"] == "scratch_ai_hf_model_cards_exact")
    row["consent"].update(basis="declared: take the host's verdict", resolution="live-per-origin")
    row["unknowns"] = sorted(contract_mod.actual_unknowns(row))
    path = tmp_path / "contracts.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_dumped_row_never_carries_family_id_master_row_may():
    master = {"url": "https://x.ch/a", "title": "t", "license": "CC0-1.0", "family_id": "fedlex"}
    assert "family_id" in master
    assert "family_id" not in common.compact_payload(master)
    assert "family_id" in rehearsal.MASTER_ONLY
    assert "family_id" in contract_mod.ADR039_MASTER_ONLY


def test_projection_carries_family_id_and_index_is_keyword():
    assert "family_id" in P.BASE_PASSTHROUGH
    assert "family_id" in serving_config.FAMILY_ID_INDEXES
    assert "family_id" not in serving_config.PAYLOAD_INDEXES      # SERVED stays sealed
    assert {k: v for k, v in serving_config.FAMILY_ID_INDEXES.items()
            if k != "family_id"} == serving_config.PAYLOAD_INDEXES


def test_family_id_profile_is_served_plus_the_one_index():
    from cytir.core import serving_profile as SP
    a, b = SP.SERVED.manifest(), SP.SERVED_FAMILY_ID.manifest()
    assert b["payload_indexes"] == list(a["payload_indexes"]) + [["family_id", "keyword", False]]
    for k in a:
        if k not in ("payload_indexes", "name", "note"):
            assert a[k] == b[k], k
    cfg = serving_config.FAMILY_ID_CONFIG
    assert SP.profile_from_serving_config(cfg).payload_indexes[-1] == ("family_id", "keyword", False)
    assert "256-pq8-family" not in serving_config.CONFIGS


def test_the_door_stamps_every_row_with_the_registry_key(tmp_path):
    run_family.run_family(
        "scratch_ai_hf_model_cards_exact", out_dir=tmp_path / "o", today="2026-10-23",
        log=lambda m: None, consent=lambda h, d: {**OK, "host": h},
        options={"root": str(SAMPLE)}, contracts_path=_declared_contracts(tmp_path))
    _, recs = run_family.load_pack(tmp_path / "o" / "pack")
    assert recs and {r["family_id"] for r in recs} == {"scratch_ai_hf_model_cards_exact"}


class _Fake:
    def __init__(self, schema, counts):
        self.schema, self.counts, self.calls = schema, counts, []

    def __call__(self, method, url, body=None):
        self.calls.append((method, url, body))
        if method == "GET":
            return {"payload_schema": self.schema}
        assert url.endswith("/points/count") and body["exact"] is False
        return {"count": self.counts.get(body["filter"]["must"][0]["match"]["value"], 0)}


def test_census_counts_per_family_with_keyword_filter():
    fake = _Fake({"family_id": {"data_type": "keyword"}}, {"a": 5, "b": 2})
    slept = []
    out = FC.census("http://q/collections/c", ["a", "b"], http=fake, sleep=slept.append)
    assert out == {"a": 5, "b": 2} and slept == [0.2, 0.2]


def test_census_reports_absent_field_cleanly(capsys):
    fake = _Fake({"url": {}}, {})
    assert FC.census("http://q/collections/c", ["a"], http=fake) is None
    assert FC.main(["--collection-url", "http://q/collections/c"], http=fake,
                   sleep=lambda s: None) == 0
    assert "no family_id on this collection" in capsys.readouterr().out
    assert all(c[0] == "GET" for c in fake.calls)


def test_the_door_refuses_a_foreign_family_id(tmp_path, monkeypatch):
    import pytest
    from cytir.sources.families.hf_model_cards import HfModelCardsFamily
    real = HfModelCardsFamily.chunk

    def foreign(self, doc, ctx):
        for r in real(self, doc, ctx):
            yield {**r, "family_id": "some_other_family"}

    monkeypatch.setattr(HfModelCardsFamily, "chunk", foreign)
    with pytest.raises(run_family.RunError) as ei:
        run_family.run_family(
            "scratch_ai_hf_model_cards_exact", out_dir=tmp_path / "o", today="2026-10-23",
            log=lambda m: None, consent=lambda h, d: {**OK, "host": h},
            options={"root": str(SAMPLE)}, contracts_path=_declared_contracts(tmp_path))
    assert "scratch_ai_hf_model_cards_exact" in str(ei.value)
    assert "some_other_family" in str(ei.value)


def test_a_contract_that_lists_family_id_in_payload_minimum_fails_the_check():
    doc = contract_mod.load()
    row = next(f for f in doc["families"] if f.get("status") != "retired")
    fam = {**row, "payload_minimum": {
        **row["payload_minimum"],
        "fields": [*(row["payload_minimum"].get("fields") or []), "family_id"]}}
    problems = contract_mod.validate_family(fam)
    assert any("lists master-only field" in p and "family_id" in p for p in problems)


def test_served_profile_keeps_its_sealed_index_roster():
    from cytir.core import serving_profile as SP
    assert [i[0] for i in SP.SERVED.payload_indexes] == [
        "domain", "url", "source_type", "country", "tier", "lang", "trust_tier",
        "register_id", "lei", "published"]
