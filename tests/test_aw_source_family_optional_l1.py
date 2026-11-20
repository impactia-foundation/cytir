"""ADR-105 am.1: `l1()` is an OPTIONAL stage on the SourceFamily Protocol (a `getattr` probe, the
`preflight` convention). A family without it is unchanged; a family with it has it run once, first."""
from __future__ import annotations

import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cytir.sources import registry, run_family  # noqa: E402
from cytir.sources.family import CHUNK_NATIVE, Captured, Doc, RunContext, SourceFamily, Unit  # noqa: E402
import test_aw_source_family_protocol as P  # noqa: E402

CALLS: list[str] = []


class Plain:
    family_id = "toy_plain_l1_for_tests"
    route = "documented_source_api"
    direct_crawl = True
    chunk_policy_id = CHUNK_NATIVE

    def discover(self, ctx):
        CALLS.append("discover")
        return [Unit(key="u0", url="https://toy.example/0")]

    def acquire(self, unit, ctx):
        return [Captured(unit, b"alpha", {"status": 200})]

    def extract(self, cap, ctx):
        return [Doc(url=cap.unit.url, text="alpha")]

    def chunk(self, doc, ctx):
        from cytir.core.ids import record_point_id
        return [{"url": doc.url, "text": doc.text, "point_id": record_point_id({"point_id": "a"})}]


class WithL1(Plain):
    family_id = "toy_with_l1_for_tests"

    def l1(self, ctx):
        CALLS.append("l1")
        (ctx.out_dir / "l1").mkdir(parents=True, exist_ok=True)
        (ctx.out_dir / "l1" / "inputs.txt").write_text("x")
        return {"path": "l1/inputs.txt", "files": 1}


def _contracts(tmp_path, cls):
    def mutate(row):
        row["chunk_policy"].update(policy_id="unknown", sha256="unknown", epoch="unknown",
                                   policy_family=None)
        row["status"] = "backfilled"
        from cytir.sources import contract
        row["unknowns"] = sorted(contract.actual_unknowns(row))
    return P._contracts(tmp_path, cls.family_id, mutate=mutate)


def _run(tmp_path, cls, monkeypatch, **kw):
    monkeypatch.setitem(registry._FACTORIES, cls.family_id, cls)
    CALLS.clear()
    return run_family.run_family(
        cls.family_id, out_dir=tmp_path / "out", today="2026-10-26", consent=P._consent_ok,
        contracts_path=_contracts(tmp_path, cls), log=lambda m: None, **kw)


def test_an_adapter_without_l1_still_conforms_and_its_manifest_has_no_l1_key(tmp_path, monkeypatch):
    assert isinstance(Plain(), SourceFamily) and isinstance(WithL1(), SourceFamily)
    m = _run(tmp_path, Plain, monkeypatch)
    assert "l1" not in m and CALLS == ["discover"]


def test_an_adapter_with_l1_has_it_called_once_and_before_discover(tmp_path, monkeypatch):
    m = _run(tmp_path, WithL1, monkeypatch)
    assert CALLS == ["l1", "discover"]
    assert m["l1"] == {"path": "l1/inputs.txt", "files": 1}
    assert (tmp_path / "out" / "l1" / "inputs.txt").exists()


def test_a_dry_run_does_not_call_l1(tmp_path, monkeypatch):
    _run(tmp_path, WithL1, monkeypatch, dry_run=True)
    assert "l1" not in CALLS and not (tmp_path / "out" / "l1").exists()


def test_l1_must_return_a_manifest_dict(tmp_path, monkeypatch):
    class Bad(WithL1):
        family_id = "toy_bad_l1_for_tests"

        def l1(self, ctx):
            return None
    with pytest.raises(run_family.RunError, match="l1"):
        _run(tmp_path, Bad, monkeypatch)


def test_dila_jorf_reference_l1_lists_dump_members_with_digests_and_is_idempotent(tmp_path):
    from cytir.sources.families.dila_jorf import DilaJorfFamily
    root = tmp_path / "dump"
    (root / "global").mkdir(parents=True)
    (root / "global" / "JORFTEXT000001.xml").write_bytes(b"<a/>")
    (root / "global" / "OTHER.xml").write_bytes(b"<b/>")
    ctx = RunContext(out_dir=tmp_path / "out", today="2026-10-26", options={"root": str(root)})
    first = DilaJorfFamily().l1(ctx)
    again = DilaJorfFamily().l1(ctx)
    assert first == again and first["members"] == 1
    listed = (tmp_path / "out" / "l1" / "dump-members.json").read_text()
    assert "JORFTEXT000001.xml" in listed and "OTHER" not in listed
