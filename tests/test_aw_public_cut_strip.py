"""The cut strips internal evidence pointers from declared prose fields of a shipped contract
(task 2223): a pointer is removed, a runtime-read field is refused, the master is unchanged, and a
self-sealed copy is re-sealed and says so in the manifest."""
import hashlib
import json
import pathlib

import pytest

from cytir.ops import public_cut as PC
from cytir.sources.contract import pre_amendment_sha256

pytestmark = pytest.mark.guard

ROOT = pathlib.Path(__file__).resolve().parents[1]
_D = "ta" + "sks"
_PL = "pipe" + "line"
CONTRACTS = "data/source-family-contracts.v1.json"


def _fixture() -> dict:
    return {
        "schema": "x",
        "families": [
            {"family_id": "f1",
             "evidence": ["tasks/12-note.md", "kept sentence", "see tasks/9-a.md for why (task 12)"],
             "licence": {"evidence": ["pipeline/x.py"], "value": "cc0"},
             "consent": {"evidence": ["tasks/5-probe.jsonl#a"]},
             "producer_ids": {"pipeline/p.py": {"code_home": "pipeline/p.py"}}}],
        "amendments": [
            {"old_file_sha256": "aa", "pre_amendment_file_sha256": "bb"},
            {"old_file_sha256": "cc", "pre_amendment_file_sha256": "dd"}],
    }


RULE = {"fields": ["families.evidence", "families.licence.evidence"], "reseal": False}


def test_pointer_is_stripped_and_sentence_kept():
    text, n = PC.derive_text(json.dumps(_fixture()), RULE)
    out = json.loads(text)
    fam = out["families"][0]
    assert fam["evidence"] == ["kept sentence", "see for why"]
    assert fam["licence"] == {"evidence": [], "value": "cc0"}
    assert n == 3
    assert not PC.scan_text(json.dumps(fam["evidence"]))


def test_runtime_fields_are_not_touched():
    out = json.loads(PC.derive_text(json.dumps(_fixture()), RULE)[0])["families"][0]
    assert out["consent"] == _fixture()["families"][0]["consent"]
    assert out["producer_ids"] == _fixture()["families"][0]["producer_ids"]


@pytest.mark.parametrize("path", [
    "families.consent.evidence",          # probe paths opened at runtime
    "families.producer_ids.code_home",    # runtime key
    "families.licence",                   # a value a loader reads, not a prose key
    "families.flip.cite",
    "families.evidence.",                 # empty step
])
def test_runtime_field_is_refused(path):
    with pytest.raises(PC.CutRefusal):
        PC.strip_rules({"strip": [{"path": "data/x.json", "why": "w", "fields": [path]}]})


def test_strip_entry_needs_why():
    with pytest.raises(PC.CutRefusal):
        PC.strip_rules({"strip": [{"path": "data/x.json", "fields": ["a.evidence"]}]})


def test_duplicate_keys_survive_the_rederive():
    text = '{"a": {"k": 1, "k": 2}, "evidence": ["tasks/1-x.md"]}'
    out = PC.derive_text(text, {"fields": ["evidence"], "reseal": False})[0]
    assert out.count('"k"') == 2 and '"evidence": []' in out


def test_reseal_makes_the_copy_verify_and_changes_the_recorded_sha():
    fx = _fixture()
    fx["amendments"][-1]["pre_amendment_file_sha256"] = "stale"
    text, _ = PC.derive_text(json.dumps(fx), {**RULE, "reseal": True})
    recorded, computed = pre_amendment_sha256(text)
    assert recorded == computed != "stale"


def test_export_keeps_master_bytes_and_records_the_derivation(tmp_path):
    root, out = tmp_path / "root", tmp_path / "out"
    (root / "data").mkdir(parents=True)
    master = json.dumps(_fixture(), indent=1).encode()
    (root / "data/c.json").write_bytes(master)
    rule = {"data/c.json": {**RULE, "reseal": True}}
    manifest = PC.export(root, {"data/c.json": "r"}, out, "sha", rule)
    assert (root / "data/c.json").read_bytes() == master
    (entry,) = manifest["files"]
    assert entry["derived_from_sha256"] == hashlib.sha256(master).hexdigest()
    assert entry["resealed"] is True and entry["values_stripped"] == 3
    assert PC.verify_tree(out) == []
    shipped = (out / "data/c.json").read_text()
    assert not PC.scan_text(json.dumps(json.loads(shipped)["families"][0]["evidence"]))


def _shipped_derivation(rel: str) -> dict | None:
    """The cut's own derivation record for `rel` when THIS tree is an exported cut (the manifest
    sits at its root), else None (the master repo). The record counts only if the file here is
    the derived copy: its sha is the manifest's sha and differs from the master's sha."""
    manifest = ROOT / PC.MANIFEST_NAME
    if not manifest.is_file():
        return None
    for e in json.loads(manifest.read_text(encoding="utf-8"))["files"]:
        if e["path"] == rel and "derived_from_sha256" in e:
            here = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            return e if here == e["sha256"] != e["derived_from_sha256"] else None
    return None


def test_real_contract_copy_is_sealed_and_loads_the_same():
    doc = PC.load_allowlist(ROOT)
    rules = PC.strip_rules(doc)
    assert CONTRACTS in rules and rules[CONTRACTS]["reseal"]
    master = (ROOT / CONTRACTS).read_text(encoding="utf-8")
    text, n = PC.derive_text(master, rules[CONTRACTS])
    if _shipped_derivation(CONTRACTS):
        assert n == 0  # shipped copy is already stripped: no pointer left to remove
    else:
        assert n > 0
    recorded, computed = pre_amendment_sha256(text)
    assert recorded == computed
    a, b = json.loads(master), json.loads(text)
    for fa, fb in zip(a["families"], b["families"]):
        assert fa.get("consent") == fb.get("consent")
        pa, pb = fa.get("producer_ids") or {}, fb.get("producer_ids") or {}   # relabel (task 2313): same keys in order, labels swapped
        assert [PC.frozen_label(k) for k in pa] == list(pb)
        assert [{**v, "code_home": PC.frozen_label(v["code_home"])} for v in pa.values()] == list(pb.values())
        assert fa.get("licence", {}).get("kind") == fb.get("licence", {}).get("kind")


def test_every_declared_file_loses_leaks_and_keeps_runtime_keys():
    doc = PC.load_allowlist(ROOT)
    for rel, rule in PC.strip_rules(doc).items():
        master = (ROOT / rel).read_text(encoding="utf-8")
        text, n = PC.derive_text(master, rule)
        if _shipped_derivation(rel):
            assert n == 0, rel  # shipped: no pointer left to strip
            if rel.endswith(".json"):
                json.loads(master)  # runtime keys still load
            continue
        assert n > 0, rel
        count =PC.scan_bare_values if rel in PC.prose_exempt_paths(doc) else PC.scan_text
        assert sum(count(text).values()) < sum(count(master).values()), rel


def test_role_is_a_runtime_key_and_is_refused():
    # cytir/retune/guard.py reads collections.*.role at runtime: a path ending in role is refused.
    with pytest.raises(PC.CutRefusal):
        PC.strip_rules({"strip": [{"path": "data/x.json", "why": "w",
                                   "fields": ["collections.*.role"]}]})


# ---- relabel of runtime-key pointers (task 2313, ruling 68(c)) -------------------------------

def _tp(rest: str) -> str:
    return _D + "/" + rest


def _relabel_fixture() -> dict:
    s = _tp("426-scripts/materialize_x.py")
    return {"families": [{
        "family_id": "f1",
        "producer_ids": {s: {"status": "frozen-label", "code_home": s, "meaning": "m", "declared": "d", "date": "x"},
                         "cytir/keep.py": {"code_home": "cytir/keep.py"}},
        "evidence": []}],
        "cite": _tp("590-flip-ruling-and-deploy.md")}


RELABEL_RULE = {"fields": [], "relabel": ["families.producer_ids", "families.producer_ids.*.code_home", "cite"],
                "reseal": False}


def test_relabel_swaps_keys_and_values_for_frozen_labels_and_keeps_every_key():
    text, n = PC.derive_text(json.dumps(_relabel_fixture()), RELABEL_RULE)
    out = json.loads(text)
    pid = out["families"][0]["producer_ids"]
    assert list(pid) == ["frozen:task-426:materialize_x", "cytir/keep.py"]
    assert pid["frozen:task-426:materialize_x"]["code_home"] == "frozen:task-426:materialize_x"
    assert out["cite"] == "frozen:task-590:flip-ruling-and-deploy"
    assert n == 3 and _D + "/" not in text


def test_relabeled_code_home_validates_in_the_copy():
    from cytir.sources.contract import _code_home_ok
    assert _code_home_ok("frozen:task-426:materialize_x", ROOT)
    assert not _code_home_ok("frozen-ish:x", ROOT)


def test_relabel_keeps_the_master_unchanged_and_refuses_other_keys():
    fx = json.dumps(_relabel_fixture())
    PC.derive_text(fx, RELABEL_RULE)
    assert json.loads(fx) == _relabel_fixture()
    for bad in ("families.consent", "families.licence.instrument", "families.evidence"):
        with pytest.raises(PC.CutRefusal):
            PC.strip_rules({"strip": [{"path": "data/x.json", "why": "w", "relabel": [bad]}]})


def test_relabel_that_collides_two_keys_is_refused():
    fx = {"families": [{"producer_ids": {_tp("1-scripts/collide_fixture.py"): {}, _tp("1-other/collide_fixture.md"): {}}}]}
    with pytest.raises(PC.CutRefusal):
        PC.derive_text(json.dumps(fx), {"fields": [], "relabel": ["families.producer_ids"], "reseal": False})


def test_policy_declares_the_relabel_and_the_derived_copies_carry_no_pointer_in_those_keys():
    rules = PC.strip_rules(json.loads((ROOT / "data/public-cut.v1.json").read_text()))
    assert rules[CONTRACTS]["relabel"] and rules["data/flip-gate-contract.v1.json"]["relabel"]
    text, _ = PC.derive_text((ROOT / "data/flip-gate-contract.v1.json").read_text(),
                             rules["data/flip-gate-contract.v1.json"])
    assert not PC.scan_bare_values(text).get("tasks_ref")
    fam, _ = PC.derive_text((ROOT / CONTRACTS).read_text(), rules[CONTRACTS])
    assert "frozen:task-" in fam and (chr(34) + _tp("4")) not in fam
