#!/usr/bin/env python3
"""The source-family interface itself (task 683, S7): Protocol, registry, framework.

The three exemplars (`test_aw_family_{fedlex,sitemap,cc}.py`) prove that real families run
through it. This file proves the rules that make it an interface rather than a convention:

  * a family is ADMITTED only by its declaration in `data/source-family-contracts.v1.json`
    (undeclared, retired, `--strict` with `unknown`, a chunk policy at odds with the declaration);
  * a stranger adds a family by writing ONE class (the toy family below is the whole change);
  * every unit reaches exactly one terminal; instrument failures are quarantined, never excluded
    as if the source had said no;
  * the embed door has the consent refusal S6 left open (`cyt embed index` too).

No network, no model, no server.
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cytir.sources import contract, registry, run_family  # noqa: E402
from cytir.sources.family import (CHUNK_NATIVE, Captured, Doc, Ledger, SourceFamily,  # noqa: E402
                               Unit, UnitExcluded)

OK = {"bucket": "ok", "status": "ok", "search_ok": True, "train_ok": True,
      "checked_at": "2026-09-30T00:00:00Z"}


def _consent_ok(host, direct):
    return {**OK, "host": host}


# ---------------------------------------------------------------------------------------
# a contracts file with one extra, fully declared row, cloned from a real declared family
# ---------------------------------------------------------------------------------------
def _contracts(tmp_path, family_id, *, mutate=None):
    doc = json.loads((REPO / contract.DEFAULT_PATH).read_text(encoding="utf-8"))
    row = copy.deepcopy(next(f for f in doc["families"]
                             if f["family_id"] == "de_openlegaldata_v1"))
    row["family_id"] = family_id
    row["status"] = "declared"
    if mutate:
        mutate(row)
    doc["families"].append(row)
    path = tmp_path / "contracts.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


class Toy:
    """The entire change a stranger makes: one class. Units are the lines of an option."""

    family_id = "toy_family_for_tests"
    route = "documented_source_api"
    direct_crawl = True
    chunk_policy_id = CHUNK_NATIVE

    def discover(self, ctx):
        return [Unit(key=f"u{i}", url=f"https://toy.example/{i}", meta={"text": t})
                for i, t in enumerate(ctx.options["texts"])]

    def acquire(self, unit, ctx):
        if unit.meta["text"] == "GONE":
            raise UnitExcluded("origin_gone:404")
        if unit.meta["text"] == "BOOM":
            raise TimeoutError("our egress")
        return [Captured(unit, unit.meta["text"].encode("utf-8"), {"status": 200})]

    def extract(self, cap, ctx):
        text = cap.body.decode("utf-8")
        return [] if text == "EMPTY" else [Doc(url=cap.unit.url, text=text)]

    def chunk(self, doc, ctx):
        from cytir.core.ids import record_point_id       # the one minter (core ratchet)
        if doc.text == "NOURL":
            return [{"text": "x", "point_id": record_point_id({"point_id": "nourl"})}]
        pid = record_point_id({"point_id": "same-id" if doc.text == "DUP" else doc.url})
        return [{"url": doc.url, "text": doc.text, "point_id": pid}]


@pytest.fixture
def toy(monkeypatch, tmp_path):
    monkeypatch.setitem(registry._FACTORIES, Toy.family_id, Toy)
    def unknown_policy(row):
        row["chunk_policy"].update(policy_id="unknown", sha256="unknown", epoch="unknown",
                                   policy_family=None)
        row["status"] = "backfilled"                 # a declared family holds no unknowns
        row["unknowns"] = sorted(contract.actual_unknowns(row))
    return _contracts(tmp_path, Toy.family_id, mutate=unknown_policy)


def _toy_run(tmp_path, toy, texts, **kw):
    return run_family.run_family(
        Toy.family_id, out_dir=tmp_path / "out", today="2026-09-30", consent=_consent_ok,
        options={"texts": texts}, contracts_path=toy, log=lambda m: None, **kw)


# ---------------------------------------------------------------------------------------
# admission
# ---------------------------------------------------------------------------------------
def test_the_three_exemplars_are_registered_and_declared():
    ids = registry.registered()
    for fid in ("fedlex_exact", "scratch_elcom_docs_exact", "cc_authority_frontier_w7"):
        assert fid in ids
        adapter, row, spec = registry.admit(fid)
        assert isinstance(adapter, SourceFamily) and row["family_id"] == fid
    assert registry.admit("cc_authority_frontier_w7")[2].registry_status == "current"


def test_an_undeclared_family_is_refused_not_defaulted(monkeypatch, tmp_path):
    class Ghost(Toy):
        family_id = "ghost_family_with_no_declaration"
    monkeypatch.setitem(registry._FACTORIES, Ghost.family_id, Ghost)
    with pytest.raises(registry.FamilyRefused, match="no declaration"):
        registry.admit(Ghost.family_id)
    with pytest.raises(registry.FamilyRefused, match="no family"):
        registry.admit("never_registered")


def test_a_retired_family_does_not_run(monkeypatch):
    class Old(Toy):
        family_id = "canada_official_news"               # retired in the contracts file
    monkeypatch.setitem(registry._FACTORIES, Old.family_id, Old)
    with pytest.raises(registry.FamilyRefused, match="retired"):
        registry.admit(Old.family_id)


def test_strict_refuses_a_family_that_declares_unknown_values():
    registry.admit("astra_html_exact")                    # grandfathered: runs
    with pytest.raises(registry.FamilyRefused, match="unknown"):
        registry.admit("astra_html_exact", strict=True)   # but `--strict` refuses it
    registry.admit("cc_authority_frontier_w7", strict=True)   # a complete declaration passes


def test_fedlex_and_loda_rows_have_no_unknowns_left():
    """The family door (`family_stage.check_pack`) refuses any pack whose manifest lists a contract
    unknown. This pins the two Protocol law families at ZERO unknowns after ruling 27 (POLICY v3.3:
    `identity.minted_by: text-anchor-v0`, `chunk_policy: native-chunked`), so a row cannot regain
    an unknown without a visible change here. Was one each (`identity.minted_by`) after ruling 26."""
    rows = {f["family_id"]: f for f in json.loads(
        (REPO / contract.DEFAULT_PATH).read_text(encoding="utf-8"))["families"]}
    fed, loda = rows["fedlex_exact"], rows["loda_exact"]
    assert fed["gate_classes"] == ["legal"]
    assert loda["title_shape"]["mode"] == "opted_out" and loda["title_shape"]["why"]
    assert loda["consent"]["terminal_schema_version"] == "adr092-direct-terminal.v8"
    # POLICY v3.2 (ruling 26, items 20a-c): fedlex declares route `bulk-direct` and the channel-split
    # representation. POLICY v3.3 (ruling 27, T6/T7): both law rows declare `chunk_policy:
    # native-chunked` (fedlex by Tim's later word the same day) and `identity.minted_by: text-anchor-v0`.
    assert fed["route"] == "bulk-direct"
    assert fed["representation"]["ruling"] == "RULED" and fed["representation"]["ruling_cite"]
    for row in (fed, loda):
        cp = row["chunk_policy"]
        assert (cp["policy_id"], cp["sha256"], cp["epoch"]) == ("native-chunked", None, None)
        # fedlex's label became `mint_point_id` in task 1038 (ruling 30); loda keeps generation 0
        assert ("mint_point_id" in row["identity"]["minted_by"]) == (row is fed)
        assert ("text-anchor-v0" in row["identity"]["minted_by"]) == (row is loda)
    assert "Dump route" not in fed["consent"]["basis"]
    assert fed["unknowns"] == [] and loda["unknowns"] == []
    for row in (fed, loda):
        assert contract.actual_unknowns(row) == row["unknowns"]


def test_the_adapter_and_its_declaration_must_agree_on_the_chunk_policy(monkeypatch, tmp_path):
    class Wrong(Toy):
        family_id = "family_declaring_a_policy"
        chunk_policy_id = "some-other-policy.v1"
    monkeypatch.setitem(registry._FACTORIES, Wrong.family_id, Wrong)
    path = _contracts(tmp_path, Wrong.family_id)          # clone keeps openlegaldata's policy
    with pytest.raises(registry.FamilyRefused, match="declares chunk policy"):
        registry.admit(Wrong.family_id, contracts_path=path)

    class Native(Toy):
        family_id = "family_declaring_a_policy_but_native"
    monkeypatch.setitem(registry._FACTORIES, Native.family_id, Native)
    sub = tmp_path / "two"
    sub.mkdir()
    path = _contracts(sub, Native.family_id)
    with pytest.raises(registry.FamilyRefused, match="declares chunk policy"):
        registry.admit(Native.family_id, contracts_path=path)   # a declared policy must be used


def test_an_adapter_claiming_a_policy_the_contract_does_not_declare_is_refused(monkeypatch,
                                                                              tmp_path):
    class Claims(Toy):
        family_id = "family_claiming_a_policy"
        chunk_policy_id = "chunk.generic.v9"
    monkeypatch.setitem(registry._FACTORIES, Claims.family_id, Claims)
    path = _contracts(tmp_path, Claims.family_id, mutate=lambda r: r["chunk_policy"].update(
        policy_id="unknown", sha256="unknown", epoch="unknown"))
    with pytest.raises(registry.FamilyRefused):
        registry.admit(Claims.family_id, contracts_path=path)


def test_one_family_has_one_adapter():
    registry.load_modules()   # the refusal needs fedlex_exact registered first (task 2389)
    class Other(Toy):
        family_id = "fedlex_exact"
    with pytest.raises(ValueError, match="already registered"):
        registry.register_family(Other)
    with pytest.raises(TypeError, match="family_id"):
        registry.register_family(type("NoId", (), {}))


def test_an_out_of_tree_adapter_is_found_by_module_name(monkeypatch, tmp_path):
    (tmp_path / "my_family_mod.py").write_text(
        "from cytir.sources.registry import register_family\n"
        "@register_family\nclass Mine:\n    family_id = 'mine_out_of_tree'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(registry._FACTORIES, "placeholder", object)
    registry.load_modules("my_family_mod")
    assert "mine_out_of_tree" in registry._FACTORIES
    registry._FACTORIES.pop("mine_out_of_tree")


# ---------------------------------------------------------------------------------------
# a stranger's family, and the terminals
# ---------------------------------------------------------------------------------------
def test_a_toy_family_runs_through_the_whole_framework(tmp_path, toy):
    m = _toy_run(tmp_path, toy, ["alpha", "beta"])
    assert m["terminals"]["units_admitted"] == 2 and m["records"]["count"] == 2
    rows = run_family.load_pack(tmp_path / "out" / "pack")[1]
    assert [r["text"] for r in rows] == ["alpha", "beta"]
    assert m["contract"]["row_sha256"] == run_family.contract_digest(
        contract.by_id(contract.load(toy))[Toy.family_id])
    assert (tmp_path / "out" / "l0" / "index.jsonl").read_text().count("\n") == 2


def test_every_unit_reaches_exactly_one_terminal_whatever_goes_wrong(tmp_path, toy):
    m = _toy_run(tmp_path, toy, ["ok", "GONE", "BOOM", "EMPTY", "NOURL", "DUP", "DUP"])
    t = m["terminals"]
    assert t["units_in"] == 7
    assert t["units_in"] == t["units_admitted"] + t["units_excluded"] + t["units_quarantined"]
    assert t["excluded_reasons"] == {"origin_gone:404": 1, "extract_empty": 1}
    assert t["quarantined_reasons"] == {
        "acquire_failed:TimeoutError": 1,       # our egress: held, not the source's verdict
        "record_shape:needs url,text,point_id": 1,
        "duplicate_point_id": 1}                # the 2nd DUP: first one wins, second is counted
    assert {u["unit"] for u in m["units"]} == {f"u{i}" for i in range(7)}


def test_the_limit_is_an_accounted_exclusion(tmp_path, toy):
    m = _toy_run(tmp_path, toy, ["a", "b", "c"], limit=1)
    assert m["terminals"]["units_in"] == 3 and m["terminals"]["units_admitted"] == 1
    assert m["terminals"]["excluded_reasons"] == {"run_limit": 2}


def test_a_run_never_overwrites_a_pack(tmp_path, toy):
    _toy_run(tmp_path, toy, ["a"])
    with pytest.raises(run_family.RunError, match="never overwrites"):
        _toy_run(tmp_path, toy, ["a"])


def test_the_ledger_refuses_a_second_terminal_and_a_reasonless_exclusion():
    led = Ledger()
    led.discover(2)
    led.close("a", "admitted", records=1)
    with pytest.raises(RuntimeError, match="exactly one"):
        led.close("a", "excluded", "x")
    with pytest.raises(ValueError, match="needs a reason"):
        led.close("b", "excluded")
    with pytest.raises(RuntimeError, match="conservation failed"):
        led.assert_balanced()                # b was discovered and never mentioned again


def test_consent_is_resolved_before_acquire_and_unresolved_is_held(tmp_path, toy):
    seen: list = []

    def consent(host, direct):
        seen.append((host, direct))
        raise TimeoutError("probe failed")
    m = run_family.run_family(Toy.family_id, out_dir=tmp_path / "o", today="2026-09-30",
                              consent=consent, options={"texts": ["a"]},
                              contracts_path=toy, log=lambda m: None)
    assert seen == [("toy.example", True)]
    assert m["terminals"]["quarantined_reasons"] == {"consent_unresolved:toy.example": 1}
    assert m["records"]["count"] == 0


def test_skip_consent_marks_the_pack_unchecked(tmp_path, toy):
    m = _toy_run(tmp_path, toy, ["a"], skip_consent=True)
    assert m["consent"]["checked"] is False and m["consent"]["verdicts"] == []


# ---------------------------------------------------------------------------------------
# the embed door
# ---------------------------------------------------------------------------------------
def test_aw_embed_index_passes_clean_hosts_and_counts_urlless_rows(tmp_path):
    from cytir.consent import embed_gate
    reg = tmp_path / "consent.json"
    reg.write_text(json.dumps({"schema": "x", "hosts": {}}))
    lines: list[str] = []
    got = embed_gate.assert_records_consent(
        [{"url": "https://a.example/x"}, {"text": "no url"}], registry=str(reg), log=lines.append)
    assert got == {"rows": 2, "hosts": 1, "rows_without_url": 1}
    assert "could NOT be consent-checked" in lines[0]


def test_the_cli_names_the_family_door(capsys):
    from cytir import cli
    assert cli.main(["source"]) == 2
    err = capsys.readouterr().err
    assert "FAMILY" in err and "families" in err


# ------------------------------------------------ `native` and `bulk-direct` (POLICY v3.2, ruling 26)
def _fedlex_row():
    doc = json.loads((REPO / contract.DEFAULT_PATH).read_text(encoding="utf-8"))
    return copy.deepcopy(next(f for f in doc["families"] if f["family_id"] == "fedlex_exact"))


@pytest.mark.parametrize("value", ["native", "native-chunked"])
def test_native_values_are_explicit_chunk_policies_not_a_file_lookup(value):
    """`native` = the chunk is the source's own article (one row per article); `native-chunked`
    (POLICY v3.3 §4.1, ruling 27 T6 a) = the same, and an over-window article may be split. There
    is no `data/<value>.json` and none is looked for; neither carries a sha256 or an epoch."""
    assert not (REPO / "data" / f"{value}.json").exists()
    row = _fedlex_row()
    row["chunk_policy"]["policy_id"] = value
    assert [p for p in contract.validate_family(row) if "chunk_policy" in p] == []
    row["chunk_policy"]["sha256"] = "0" * 64
    assert any(value in p and "sha256" in p for p in contract.validate_family(row))
    row["chunk_policy"]["sha256"] = None
    row["chunk_policy"]["epoch"] = 3
    assert any(value in p and "epoch" in p for p in contract.validate_family(row))


def test_fedlex_declares_native_chunked():
    assert _fedlex_row()["chunk_policy"]["policy_id"] == "native-chunked"


def test_a_native_declaration_needs_a_native_adapter():
    row = _fedlex_row()
    row["chunk_policy"]["policy_id"] = "native"
    spec = registry.resolve_chunk_spec(row, CHUNK_NATIVE)
    assert spec.declared == CHUNK_NATIVE and spec.sha256 is None and spec.registry_status is None
    with pytest.raises(registry.FamilyRefused, match="declares chunk policy 'native'"):
        registry.resolve_chunk_spec(row, "some-other-policy.v1")
    with pytest.raises(registry.FamilyRefused, match="declares chunk policy 'native'"):
        registry.resolve_chunk_spec(row, "native-chunked")      # the two are not interchangeable


def test_a_native_chunked_declaration_needs_a_native_chunked_adapter():
    row = _fedlex_row()
    spec = registry.resolve_chunk_spec(row, "native-chunked")
    assert spec.declared == "native-chunked" and spec.sha256 is None and spec.registry_status is None
    with pytest.raises(registry.FamilyRefused, match="declares chunk policy 'native-chunked'"):
        registry.resolve_chunk_spec(row, CHUNK_NATIVE)


def test_the_schema_file_names_bulk_direct_and_a_null_epoch():
    schema = json.loads((REPO / "data" / "source-family-contract.schema.v1.json").read_text())
    found = []

    def walk(n):
        if isinstance(n, dict):
            if "bulk-direct" in (n.get("enum") or []):
                found.append("route")
            ep = (n.get("properties") or {}).get("epoch")
            if ep and {"type": "null"} in ep.get("anyOf", []):
                found.append("epoch")
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(schema)
    assert sorted(found) == ["epoch", "route"]
    assert _fedlex_row()["route"] == "bulk-direct"
