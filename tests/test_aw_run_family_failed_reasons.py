"""Task 1824: the "every unit failed" RunError names the quarantine reasons (class + message)."""
from __future__ import annotations

import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cytir.sources import contract, registry, run_family  # noqa: E402
import test_aw_run_family_collapse as C  # noqa: E402
import test_aw_source_family_protocol as P  # noqa: E402


class Boom(C.Rows):
    family_id = "toy_boom_for_tests"

    def acquire(self, unit, ctx):
        raise ConnectionError(f"origin down for {unit.key}")


@pytest.fixture
def boom_contracts(monkeypatch, tmp_path):
    monkeypatch.setitem(registry._FACTORIES, Boom.family_id, Boom)

    def mutate(row):
        row["chunk_policy"].update(policy_id="unknown", sha256="unknown", epoch="unknown",
                                   policy_family=None)
        row["status"] = "backfilled"
        row["unknowns"] = sorted(contract.actual_unknowns(row))
    return P._contracts(tmp_path, Boom.family_id, mutate=mutate)


def test_every_unit_failed_names_class_and_message(tmp_path, boom_contracts):
    with pytest.raises(run_family.RunError) as ei:
        run_family.run_family(
            Boom.family_id, out_dir=tmp_path / "o", today="2026-10-08", consent=P._consent_ok,
            options={"bodies": ["a:x", "b:y"]}, contracts_path=boom_contracts, log=lambda m: None)
    msg = str(ei.value)
    assert "every one of 2 unit(s) failed" in msg
    assert "1 distinct reason class(es)" in msg
    assert "acquire_failed:ConnectionError: origin down for u0" in msg
    assert "u1" in msg
