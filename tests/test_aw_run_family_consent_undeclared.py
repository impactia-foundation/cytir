"""Task 1406: ONE framework rule. A unit that contacts a host is refused when its family's contract
row has consent basis `unknown`. A declared row with `search_ok: null` is not refused here (it
takes the host's live verdict). A unit with no host proceeds. `plan` (dry run) never raises."""
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
from cytir.sources.family import CHUNK_NATIVE, Captured, Doc, Unit  # noqa: E402
import test_aw_source_family_protocol as P  # noqa: E402


ACQUIRED: list[str] = []


class Urls:
    """One unit per URL in `options["urls"]`; `file:` URLs contact no host."""

    family_id = "toy_undeclared_for_tests"
    route = "documented_source_api"
    direct_crawl = True
    chunk_policy_id = CHUNK_NATIVE

    def discover(self, ctx):
        return [Unit(key=f"u{i}", url=u) for i, u in enumerate(ctx.options["urls"])]

    def acquire(self, unit, ctx):
        ACQUIRED.append(unit.key)
        return [Captured(unit, b"some text", {"status": 200})]

    def extract(self, cap, ctx):
        return [Doc(url=cap.unit.url, text=cap.body.decode("utf-8"))]

    def chunk(self, doc, ctx):
        from cytir.core.ids import record_point_id
        return [{"url": doc.url, "text": doc.text,
                 "point_id": record_point_id({"point_id": doc.url})}]


def _contracts(monkeypatch, tmp_path, *, basis, search_ok):
    monkeypatch.setitem(registry._FACTORIES, Urls.family_id, Urls)

    def mutate(row):
        row["chunk_policy"].update(policy_id="unknown", sha256="unknown", epoch="unknown",
                                   policy_family=None)
        row["status"] = "backfilled"
        row["consent"].update(basis=basis, search_ok=search_ok,
                              resolution="live-per-origin" if basis != "unknown" else "unknown")
        from cytir.sources import contract
        row["unknowns"] = sorted(contract.actual_unknowns(row))
    return P._contracts(tmp_path, Urls.family_id, mutate=mutate)


def _run(tmp_path, contracts, urls, *, dry_run=False, called=None):
    ACQUIRED.clear()
    opts = {"urls": urls}

    def consent(host, direct):
        if called is not None:
            called.append(host)
        return P._consent_ok(host, direct)
    m = run_family.run_family(
        Urls.family_id, out_dir=tmp_path / "out", today="2026-10-30", consent=consent,
        options=opts, contracts_path=contracts, log=lambda m: None, dry_run=dry_run)
    return m, opts


def test_unknown_row_with_a_host_unit_is_quarantined_consent_undeclared(
        monkeypatch, tmp_path):
    c = _contracts(monkeypatch, tmp_path, basis="unknown", search_ok=None)
    called: list[str] = []
    m, opts = _run(tmp_path, c, ["https://toy.example/a", "file:///held/b.txt"], called=called)
    reasons = m["terminals"]["quarantined_reasons"]
    assert reasons == {f"consent_undeclared:{Urls.family_id}:toy.example": 1}
    assert ACQUIRED == ["u1"] and called == []     # the host was never probed


def test_unknown_row_refuses_the_run_when_every_unit_is_refused(monkeypatch, tmp_path):
    c = _contracts(monkeypatch, tmp_path, basis="unknown", search_ok=None)
    with pytest.raises(run_family.RunError, match="refused"):
        _run(tmp_path, c, ["https://toy.example/a", "https://toy.example/b"])


def test_declared_row_with_search_ok_none_is_not_refused_by_the_predicate(
        monkeypatch, tmp_path):
    c = _contracts(monkeypatch, tmp_path, basis="declared: take the host's verdict",
                   search_ok=None)
    called: list[str] = []
    m, opts = _run(tmp_path, c, ["https://toy.example/a"], called=called)
    assert called == ["toy.example"]                       # it reached the live gate
    assert ACQUIRED == ["u0"] and m["terminals"]["units_admitted"] == 1
    adapter, row, _ = registry.admit(Urls.family_id, contracts_path=c)
    assert run_family.undeclared_consent(Unit(key="k", url="https://toy.example/a"), row) is None


def test_unknown_row_with_only_file_units_proceeds(monkeypatch, tmp_path):
    c = _contracts(monkeypatch, tmp_path, basis="unknown", search_ok=None)
    m, opts = _run(tmp_path, c, ["file:///held/a.txt"])
    assert ACQUIRED == ["u0"] and m["terminals"]["units_admitted"] == 1


def test_plan_never_refuses_and_shows_the_refused_units_as_blocked(monkeypatch, tmp_path):
    c = _contracts(monkeypatch, tmp_path, basis="unknown", search_ok=None)
    plan, opts = _run(tmp_path, c, ["https://toy.example/a"], dry_run=True)
    assert plan["runnable"] == [] and ACQUIRED == []
    assert plan["blocked"] == {"u0": f"consent_undeclared:{Urls.family_id}:toy.example"}


def _run_skip(tmp_path, contracts, urls):
    ACQUIRED.clear()
    return run_family.run_family(
        Urls.family_id, out_dir=tmp_path / "out", today="2026-10-30", skip_consent=True,
        options={"urls": urls}, contracts_path=contracts, log=lambda m: None)


def test_skip_consent_still_quarantines_an_unknown_row_host_unit(monkeypatch, tmp_path):
    c = _contracts(monkeypatch, tmp_path, basis="unknown", search_ok=None)
    m = _run_skip(tmp_path, c, ["https://toy.example/a", "file:///held/b.txt"])
    assert m["terminals"]["quarantined_reasons"] == {
        f"consent_undeclared:{Urls.family_id}:toy.example": 1}
    assert ACQUIRED == ["u1"]                      # acquire never saw the host unit


def test_skip_consent_refuses_the_run_when_every_unit_is_refused(monkeypatch, tmp_path):
    c = _contracts(monkeypatch, tmp_path, basis="unknown", search_ok=None)
    with pytest.raises(run_family.RunError, match="refused"):
        _run_skip(tmp_path, c, ["https://toy.example/a"])
    assert ACQUIRED == []


def test_skip_consent_on_a_declared_row_still_proceeds(monkeypatch, tmp_path):
    c = _contracts(monkeypatch, tmp_path, basis="declared: take the host's verdict",
                   search_ok=None)
    m = _run_skip(tmp_path, c, ["https://toy.example/a"])
    assert ACQUIRED == ["u0"] and m["terminals"]["units_admitted"] == 1
