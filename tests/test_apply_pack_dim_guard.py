"""Task 563 -- the applier dim guard.

Neither `apply_pack_delta.py` (task 322) nor `cyt index surface-delta`
(task 546) checked that a pack's own dense width agreed with the target
collection's before writing a single vector. Task 560 found a 128-dim pack
(task 558's DOAB cards) aimed at the 256-dim `aw_e8` and caught it only by
hand -- Qdrant's own API happened to reject the mismatch, but nothing in the
repo would have refused it first, or named both sides of the disagreement.

These tests seal that gap at two levels, entirely against fakes -- no live
call, no network:

  1. The shared comparator, `serving_config.assert_dim_matches`/
     `assert_dim_matches_shape`, directly: a mismatch refuses naming both
     dims and both names; a match passes; the guard never refuses for a
     datatype/quantizer reason (it mirrors the target's own, so only
     dimension is a free variable); an unmapped quantizer kind (`turbo`)
     raises cleanly rather than guessing.

  2. Each applier's own preflight, via `main()`, against a fake Qdrant:
     `cyt index surface-delta` over its EXISTING fake-HTTP harness
     (`tests/test_aw_index_surface_delta.py`'s `install_fake_qdrant`,
     extended by this task to answer `GET /collections/<name>` -- reused
     here rather than forked); `apply_pack_delta.py`, which has no
     `qdrant_client.QdrantClient`-shaped fake yet, gets a minimal one below.

    python3 -m pytest tests/test_apply_pack_dim_guard.py -q
"""
from __future__ import annotations

import sys as _sys_1822
from pathlib import Path as _Path_1822

_sys_1822.path.insert(0, str(_Path_1822(__file__).resolve().parents[1] / "tests"))  # task 1822: the surface-delta fixtures live in tests/

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cytir.core import serving_config as sc

import cytir.index.pack_delta as APD
import cytir.index.surface_delta as APSD
import test_aw_index_surface_delta as T546  # reuse its fixture helpers


@pytest.fixture(autouse=True)
def _no_card_guard_read(monkeypatch):
    """task 820: `apply_pack_delta` now runs the drain's card check (an HTTP id retrieve) before
    any write; these fakes are not Qdrant. The guard's own tests: tests/test_aw_index_card_guard_writers.py."""
    monkeypatch.setattr(APD, "guard_card_overwrites", lambda *a, **k: {})
    monkeypatch.setattr(APD, "client_url", lambda client: "http://fake")


# ---------------------------------------------------------------------------
# 1. the shared comparator, directly
# ---------------------------------------------------------------------------
def _fake_info(dim, datatype=None, quantizer_kind=None, points=0, indexed=0,
              status="green"):
    """Shaped exactly like `shape_from_info` reads a real qdrant_client
    `CollectionInfo` -- attribute access all the way down, never a dict."""
    qc = None
    if quantizer_kind is not None:
        qc = SimpleNamespace(**{quantizer_kind: SimpleNamespace()})
    dense = SimpleNamespace(size=dim, datatype=datatype)
    params = SimpleNamespace(vectors={sc.DENSE: dense})
    config = SimpleNamespace(params=params, quantization_config=qc)
    return SimpleNamespace(config=config, points_count=points,
                           indexed_vectors_count=indexed, status=status)


class _FakeClient:
    """The one method `assert_dim_matches` needs."""

    def __init__(self, infos: dict):
        self.infos = infos

    def get_collection(self, name):
        return self.infos[name]


def test_assert_dim_matches_refuses_a_128_pack_against_a_256_collection():
    client = _FakeClient({"aw_e8": _fake_info(256, quantizer_kind="product")})
    with pytest.raises(RuntimeError) as exc:
        sc.assert_dim_matches(client, "aw_e8", 128,
                              source="pack aw_task560_doab_pack")
    msg = str(exc.value)
    assert "128" in msg and "256" in msg
    assert "aw_e8" in msg and "aw_task560_doab_pack" in msg


def test_assert_dim_matches_passes_when_dims_agree():
    client = _FakeClient({"aw_e8": _fake_info(256, quantizer_kind="product")})
    result = sc.assert_dim_matches(client, "aw_e8", 256, source="pack x")
    assert result["points"] == 0


@pytest.mark.parametrize("kind", [None, "scalar", "product", "binary"])
@pytest.mark.parametrize("dt", [None, "float16", "float32"])
def test_assert_dim_matches_never_refuses_for_datatype_or_quantizer(kind, dt):
    """The mirror copies the target's OWN datatype/quantizer, so a pack whose
    dim matches must pass no matter which quantizer/datatype the target
    happens to use -- this guard checks dimension, nothing else."""
    client = _FakeClient({"c": _fake_info(128, datatype=dt, quantizer_kind=kind)})
    sc.assert_dim_matches(client, "c", 128, source="pack")  # must not raise


def test_assert_dim_matches_raises_cleanly_on_an_unmapped_quantizer():
    client = _FakeClient({"c": _fake_info(128, quantizer_kind="turbo")})
    with pytest.raises(RuntimeError, match="turbo"):
        sc.assert_dim_matches(client, "c", 128, source="pack")


def test_assert_dim_matches_shape_works_from_shape_from_json():
    """The raw-HTTP flavour (`cyt index surface-delta`'s own way of
    reaching the target) goes through `shape_from_json`, not a client --
    the same comparator must refuse the same way."""
    raw = {"config": {"params": {"vectors": {"dense": {"size": 256}}},
                      "quantization_config": {"product": {}}}}
    shape = sc.shape_from_json(raw)
    with pytest.raises(RuntimeError, match="128.*256|256.*128"):
        sc.assert_dim_matches_shape(shape, "aw_e8", 128, source="--mrl 128")


# ---------------------------------------------------------------------------
# 2a. cyt index surface-delta, over its existing fake-HTTP harness
# ---------------------------------------------------------------------------
def test_surface_delta_refuses_a_128_pack_against_a_256_collection(
        tmp_path, monkeypatch):
    rows = T546.rows_for("mismatch", 3)
    vecs = T546.random_unit_matrix(3, seed=7)
    stage, vectors = T546.write_shard(tmp_path, "shard-0000", rows, vecs)
    # the collection's OWN declared width is 256; every point (if any
    # existed) would also carry 256-wide dense vectors -- but the guard must
    # refuse before a single one is even scrolled.
    coll = T546.FakeCollection({}, dim=256)
    T546.install_fake_qdrant(monkeypatch, coll)

    report = tmp_path / "report.json"
    with pytest.raises(SystemExit) as exc:
        T546.run_main(monkeypatch, [
            "--stage", stage, "--vectors", vectors, "--report", str(report),
            "--mrl", "128", "--no-sparse", "--apply"])
    msg = str(exc.value)
    assert "128" in msg and "256" in msg
    assert not report.exists()          # refused before writing anything
    assert coll.write_calls == 0        # refused before touching a vector


def test_surface_delta_matching_pack_report_unchanged(tmp_path, monkeypatch):
    """A dim-matching pack must produce the EXACT report this module wrote
    before task 563 -- the dim guard adds no field and changes no value on
    the happy path."""
    rows = T546.rows_for("clean", 6)
    vecs = T546.random_unit_matrix(6, seed=1)
    stage, vectors = T546.write_shard(tmp_path, "shard-0000", rows, vecs)
    # ADR-105 (task 581): a pack needs a store behind it. Built for real over
    # this fixture's own pages rather than stubbed out -- the guard under test
    # here is the dim guard, and neither guard may be faked to let the other
    # one pass.
    T546.install_persisted_index(monkeypatch, tmp_path, rows)
    coll = T546.FakeCollection({
        r["point_id"]: {"payload": {"url": r["url"]},
                        "vector": {"dense": [0.0] * 64}}
        for r in rows}, dim=64)
    T546.install_fake_qdrant(monkeypatch, coll)

    report = tmp_path / "report.json"
    rc = T546.run_main(monkeypatch, [
        "--stage", stage, "--vectors", vectors, "--report", str(report),
        "--mrl", "64", "--no-sparse", "--apply"])
    assert rc == 0
    doc = json.loads(report.read_text())
    # byte-identical to the pre-563 shape: same fields task 546 wrote, same
    # values, on a fixture identical to test_aw_index_surface_delta.py's
    # own `test_clean_delta_applies_unchanged`.
    assert doc["join_guard"]["id_join_rate"] == 1.0
    assert doc["join_guard"]["refused"] is False
    assert doc["two_surface_check"] == {
        "skipped": True, "reason": "--arm-a-vectors not given"}
    assert doc["conservation_ok"] is True
    assert doc["missing_in_collection"] == 0
    assert doc["rows"] == 6
    assert "dim_guard" not in doc        # the guard is a preflight, not a report field


# ---------------------------------------------------------------------------
# 2b. apply_pack_delta.py -- a minimal qdrant_client-shaped fake
# ---------------------------------------------------------------------------
class _FakeCount:
    def __init__(self, n):
        self.count = n


class _FakeQdrantClient:
    """Just enough of `qdrant_client.QdrantClient` for `apply_pack_delta.py`:
    `get_collection` (dim guard + Superseder's url-index precondition),
    `count`, and `scroll` for an EMPTY pack, which is as far as a fixture
    needs to go to prove the guard doesn't disturb the happy path -- the
    scroll/upsert/supersede body is task 322/348's own contract, out of
    scope here.
    """

    def __init__(self, registry: dict, *_, **__):
        self.registry = registry

    def get_collection(self, name):
        return self.registry[name]

    def count(self, name, exact=True):
        return _FakeCount(self.registry[name].points_count)

    def scroll(self, name, limit=1024, offset=None, with_payload=True,
              with_vectors=True):
        return [], None


def _registry_entry(dim, points=0, url_indexed=True):
    ent = _fake_info(dim, points=points, indexed=points)
    ent.payload_schema = ({"url": SimpleNamespace(data_type="keyword")}
                          if url_indexed else {})
    return ent


def test_pack_delta_refuses_a_128_pack_against_a_256_target(monkeypatch, capsys):
    registry = {
        "aw_task560_doab_pack": _registry_entry(128),
        "aw_e8": _registry_entry(256),
    }
    monkeypatch.setattr(APD, "qdrant_connect",
                        lambda *a, **k: _FakeQdrantClient(registry))
    monkeypatch.setattr(sys, "argv", [
        "apply_pack_delta.py", "--pack", "aw_task560_doab_pack",
        "--target", "aw_e8", "--dry-run"])
    with pytest.raises(SystemExit) as exc:
        APD.main()
    msg = str(exc.value)
    assert "128" in msg and "256" in msg
    assert "aw_e8" in msg and "aw_task560_doab_pack" in msg


def test_pack_delta_matching_pack_proceeds_past_the_guard(monkeypatch, capsys):
    """An empty, dim-matching pack must reach the normal dry-run report
    ('[delta] ... OK' path is unreachable in dry-run, but the join/verify
    print IS) -- proving the guard is a no-op on the happy path, not that
    the scroll/supersede body (task 322/348, untouched here) is re-verified."""
    registry = {
        "aw_task560_doab_pack": _registry_entry(256),
        "aw_e8": _registry_entry(256),
    }
    monkeypatch.setattr(APD, "qdrant_connect",
                        lambda *a, **k: _FakeQdrantClient(registry))
    monkeypatch.setattr(sys, "argv", [
        "apply_pack_delta.py", "--pack", "aw_task560_doab_pack",
        "--target", "aw_e8", "--dry-run"])
    APD.main()
    out = capsys.readouterr().out
    assert "pack aw_task560_doab_pack=0 -> target aw_e8=0" in out
    assert "scrolled 0: +0 new, ~0 overwrote-existing" in out
