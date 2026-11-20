"""A second `cyt index build` into the same `--out` can no longer succeed with an empty collection
(stranger QUICKSTART finding F6).  Real build path, local fixture Qdrant (the slice-8 fixture tests)."""
from __future__ import annotations

import pytest

import drain_transaction_harness as H  # noqa: E402
from cytir.index import build as B  # noqa: E402
from cytir.index import promote_queue as Q  # noqa: E402
from test_drain_build_identity import build_run, pack, stages, url  # noqa: E402,F401

@pytest.fixture(autouse=True, scope="module")
def _leave_no_collection_behind():
    """Task 2991b: the shared throwaway Qdrant keeps what a module builds. Left in it, these
    collections changed the drain golden receipts of the modules that run after this one in one
    process (the public CI's serial Qdrant step)."""
    yield
    from cytir.core.qdrant import qdrant_client
    client = qdrant_client(H.qdrant_url())
    for name in (TARGET, TARGET + "_link"):
        try:
            client.delete_collection(name)
        except Exception:  # noqa: BLE001 (it may never have been made)
            pass


TARGET = "aw_build_second_build"        # its own name: CI runs the build tests in parallel on one Qdrant


def test_a_second_build_with_recreate_into_the_same_out_rebuilds_the_collection(tmp_path, url, stages):
    code, _ = build_run(url, tmp_path, [stages["dila0"]], name="same", recreate=True, collection=TARGET)
    assert code == B.EXIT_OK
    first = H.dump_points(url, TARGET)
    assert first
    code, _ = build_run(url, tmp_path, [stages["dila0"]], name="same", recreate=True, collection=TARGET)
    assert code == B.EXIT_OK
    assert H.dump_points(url, TARGET) == first          # not an empty collection with rc 0


def test_a_build_that_lands_nothing_and_leaves_no_points_is_refused(tmp_path, url, stages, monkeypatch):
    monkeypatch.setattr(Q, "drain", lambda *a, **k: [])        # every pack passed over, no result
    with pytest.raises(B.BuildRefusal, match="0 points"):
        build_run(url, tmp_path, [stages["dila0"]], name="empty", recreate=True, collection=TARGET)


def test_a_state_dir_that_is_a_symlink_is_refused_and_its_foreign_done_survives(tmp_path, url, stages):
    """The clearing runs only inside --out: a `state/` that points at somebody else's directory
    (here one marked as a live queue, with a pack's DONE in it) is refused and nothing is deleted."""
    foreign = tmp_path / "foreign"
    pack_id = pack(stages["dila0"]).pack_id
    (foreign / pack_id).mkdir(parents=True)
    (foreign / "LIVE-QUEUE").write_text("live\n")
    done = foreign / pack_id / "DONE"
    done.write_text("done\n")
    out = tmp_path / "linked"
    out.mkdir()
    (out / "state").symlink_to(foreign, target_is_directory=True)
    with pytest.raises(B.BuildRefusal):
        build_run(url, tmp_path, [stages["dila0"]], name="linked", recreate=True,
                  collection="aw_build_second_build_link")
    assert done.exists()
