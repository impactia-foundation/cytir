"""Task 2404: `rehearse_upgrade` takes a client and a named source/target, so the server-side
rehearsal can run it on a scratch Qdrant. No throwaway server is assumed here: the
parameterisation is proven on local-mode clients with a renamed source and target."""
import json

import pytest

from cytir.core.qdrant import qdrant_client
from cytir.distribute import rehearsal as R
from cytir.distribute import server_rehearsal as sr
from cytir.distribute.sign import minisign_available

pytestmark = pytest.mark.skipif(not minisign_available(), reason="minisign not on PATH")


def test_explicit_client_and_names_give_the_same_report(tmp_path):
    fx_dir = R.DEFAULT_FIXTURE_DIR
    meta = json.loads((fx_dir / "fixture.json").read_text())
    src, fx = R._load_into_memory(fx_dir, R.SOURCE)
    sec, pub = R._ephemeral_keys(tmp_path)
    fresh = [R._top1(src, R.SOURCE, q["vector"]) for q in meta["queries"]]
    base = R.rehearse_upgrade(src, fx, meta, tmp_path / "a", sec, pub, fresh)
    # the same rows under another collection name, upgraded into a named target on a
    # client the caller owns (a scratch server's role)
    other, _ = R._load_into_memory(fx_dir, "slice")
    mine = qdrant_client(":memory:")
    got = R.rehearse_upgrade(other, fx, meta, tmp_path / "b", sec, pub, fresh,
                             client=mine, src_name="slice", target="upgrade")
    # local-mode renormalisation noise (`RENORM_MAX_ULP`) depends on the upsert batch, so the
    # count of rows that moved by an ULP may differ; everything else must be identical
    for rep in (base, got):
        rep["redump"].pop("dense_rows_ulp_drift")
    assert got == base
    assert mine.collection_exists("upgrade")
    assert not mine.collection_exists(R.TARGET)
    assert not other.collection_exists("upgrade")      # the source client is only read


def test_upgrade_stage_refuses_an_existing_upgrade_collection(tmp_path):
    class Scratch:
        def collection_exists(self, name):
            return name == "upgrade"
    with pytest.raises(sr.ScratchRefusal):
        sr.upgrade_stage(Scratch(), tmp_path, {"embedder": {}})


def test_upgrade_is_a_scratch_collection_check_scratch_refuses():
    assert "upgrade" in sr.SCRATCH_COLLECTIONS
