"""Round 4 of #559: what `cyt index build` and `cyt index promote` refuse, whatever the settings say.

B1 `--recreate` asks the endpoint again, under the target's lock, right before it deletes (the
reviewer moved `aw_prod` onto the target while packs composed, and the build deleted it).
B2 RUN_DIR may not overlap a queue the build did not make, even when the settings do not know it.
B3 the URL checked is the URL written (one normalised endpoint, the client's default port).
B4 `cyt index promote` is the drain's function, not the hourly unit: it refuses the live queue
   without the unit's persisted index, and the help says what it does not do.

Needs a throwaway Qdrant (`AW_TEST_QDRANT_URL`) for the build tests; skips without one. Its
fixtures (the sealed stages, the clean endpoint) are those of `test_drain_build_identity.py`.
"""
from __future__ import annotations

import json
import re

import pytest

import drain_transaction_harness as H  # noqa: E402
from cytir.index import promote_queue as Q                       # noqa: E402  (the alias of cytir.index.promote_queue)
from cytir.index import build as B  # noqa: E402
from cytir.index import compose as COMPOSE  # noqa: E402
from cytir.index import live_target as LT  # noqa: E402
from test_drain_build_refusals import _tree, unknown_live_queue  # noqa: E402,F401
from test_drain_build_identity import (  # noqa: E402,F401  (fixtures and helpers of the build tests)
    TARGET, _build_uses_the_content_check_override, _fresh_target, build_run, pack, stages, url)


def test_b1_recreate_rechecks_liveness_under_the_lock_right_before_it_deletes(
        tmp_path, url, stages, monkeypatch):
    """The reviewer's case: while the packs compose (minutes), `aw_prod` is moved onto the target.
    The plan-time check is stale by then; the build asks the endpoint again under the target's lock,
    right before the delete, and REFUSES. The collection is intact and nothing was recreated."""
    code, _ = build_run(url, tmp_path, [stages["dila0"]], name="first")
    assert code == B.EXIT_OK
    before = H.dump_points(url, TARGET)
    assert before
    real, moved = COMPOSE.compose_pack, []

    def compose_then_the_alias_moves(**kw):
        res = real(**kw)
        if not moved:
            H.set_alias(url, H.ALIAS, TARGET)          # prod now plays this collection
            moved.append(1)
        return res

    monkeypatch.setattr(B.COMPOSE, "compose_pack", compose_then_the_alias_moves)
    entered = []
    real_lock = Q.single_target_lock

    def watched_lock(*a, **k):
        entered.append(a[0].collection)
        return real_lock(*a, **k)

    monkeypatch.setattr(Q, "single_target_lock", watched_lock)
    with pytest.raises(B.BuildRefusal, match="LIVE endpoint"):
        build_run(url, tmp_path, [stages["dila0"]], name="second", recreate=True)
    assert moved == [1] and entered == [TARGET]            # the check ran under the target's lock
    assert H.api(url, "/aliases")["aliases"] == [{"alias_name": H.ALIAS, "collection_name": TARGET}]
    assert H.dump_points(url, TARGET) == before            # NOT deleted, NOT recreated


@pytest.mark.parametrize("where", ["queue", "inside-queue", "inside-pack-dir", "queue-parent",
                                   "queue-grandparent"])
def test_b2_a_run_dir_that_overlaps_a_queue_is_refused_whatever_the_settings(
        tmp_path, url, stages, unknown_live_queue, where):
    live = unknown_live_queue
    out = {"queue": live, "inside-queue": live / "run", "inside-pack-dir": live / "pack-a" / "run",
           "queue-parent": live.parent, "queue-grandparent": live.parent.parent}[where]
    before = _tree(live)
    with pytest.raises(B.BuildRefusal, match="queue"):
        B.run(mode="build", packs=[pack(stages["dila0"])], url=url, collection=TARGET,
              profile_name="served", out_dir=out, recreate=True, as_of=H.AS_OF,
              consent_cache_dir=str(H._empty_dir(tmp_path / "cc")), log=lambda m: None)
    assert _tree(live) == before, "the build wrote into the queue"
    assert not (out / "build-manifest.json").exists() and not (out / B.RUN_MARKER).exists()
    assert not (out / "queue").exists() or out == live.parent.parent or out == live.parent
    assert TARGET not in {c["name"] for c in H.api(url, "/collections")["collections"]}


def test_b2_a_run_dir_beside_a_queue_and_a_rerun_in_its_own_dir_are_allowed(
        tmp_path, url, stages, unknown_live_queue):
    out = unknown_live_queue.parent / "run-beside"
    for _ in range(2):          # the second run reuses RUN_DIR's own queue/ and state/ (--resume)
        code, _r = B.run(mode="build", packs=[pack(stages["dila0"])], url=url, collection=TARGET,
                         profile_name="served", out_dir=out, recreate=True, resume=True,
                         as_of=H.AS_OF, consent_cache_dir=str(H._empty_dir(tmp_path / "cc")),
                         log=lambda m: None)
        assert code == B.EXIT_OK
    assert (out / B.RUN_MARKER).is_file()


def test_b2_the_marker_records_the_absolute_paths_the_build_created_and_exempts_only_those(tmp_path):
    """Round 5, B2 residual: the marker used to exempt `queue/`, `state/`, `artifacts/` by NAME, so a
    marker in the PARENT of a live queue that happened to be called `queue` let a build through."""
    run = tmp_path / "run"
    B._mark_run_dir(run)
    rec = json.loads((run / B.RUN_MARKER).read_text())
    assert rec["schema"] == B.RUN_MARKER_SCHEMA and rec["run_dir"] == str(run.resolve())
    assert rec["created_dirs"] == [str(run.resolve() / n) for n in ("queue", "state", "artifacts")]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", rec["created_at"])
    # the live queue is a directory called `queue`, already holding a pack, in a dir with a marker
    parent = tmp_path / "parent-of-a-live-queue"
    (parent / "queue" / "pack-a").mkdir(parents=True)
    (parent / "queue" / "pack-a" / "READY").write_text("sha\n")
    B._mark_run_dir(parent)                                  # the build never created that queue
    assert str(parent.resolve() / "queue") not in json.loads((parent / B.RUN_MARKER).read_text())["created_dirs"]
    with pytest.raises(B.BuildRefusal, match="queue"):
        B.check_run_dir(parent)
    # a name-based marker of the old kind, or one copied from another RUN_DIR, exempts nothing
    for body in ({"schema": "aw.index-run.v1"},
                 {"schema": B.RUN_MARKER_SCHEMA, "run_dir": str(run.resolve()),
                  "created_dirs": [str(parent.resolve() / "queue")]}):
        (parent / B.RUN_MARKER).write_text(json.dumps(body))
        with pytest.raises(B.BuildRefusal, match="queue"):
            B.check_run_dir(parent)
    # the genuine marker still lets the build reuse its own trees
    (run / "queue" / "pack-a").mkdir(parents=True)
    (run / "queue" / "pack-a" / "READY").write_text("sha\n")
    B.check_run_dir(run)


def test_b3_the_url_checked_is_the_url_written_a_portless_url_gets_the_clients_port(
        tmp_path, url, stages, monkeypatch):
    """`QdrantClient("http://host")` talks to :6333 while urllib would list :80. One
    normalised endpoint goes to the liveness check and to the client."""
    host = url.rsplit(":", 1)[0]             # the throwaway's host, no port; nothing is sent (stubbed)
    checked, built = [], []

    class Stop(Exception):
        pass

    monkeypatch.setattr(Q.LT, "refuse_live_target", lambda u, c, **k: checked.append(u))

    def client(u, **kw):
        built.append(u)
        raise Stop

    monkeypatch.setattr(B, "qdrant_client", client)
    with pytest.raises(Stop):
        B.run(mode="build", packs=[pack(stages["dila0"])], url=host, collection=TARGET,
              profile_name="served", out_dir=tmp_path / "r", recreate=True, as_of=H.AS_OF,
              consent_cache_dir=str(H._empty_dir(tmp_path / "cc")), log=lambda m: None)
    assert checked == built == [f"{host}:{LT.QDRANT_DEFAULT_PORT}"]


def test_b3_a_portless_url_is_checked_where_the_client_will_write(tmp_path, url, stages, monkeypatch):
    """End to end, with NO request to a real default port (round 5, N1): the client's default port is
    pointed at the proven throwaway, so the port-less URL normalises to the throwaway and the build
    lands there. The unpatched constant is 6333, the default Qdrant port of every developer box and the
    legacy store on the CI runner: a test never hands it a request."""
    host, port = url.rsplit(":", 1)
    monkeypatch.setattr(LT, "QDRANT_DEFAULT_PORT", int(port))
    assert LT.normalize_endpoint(host) == url
    code, _ = build_run(host, tmp_path, [stages["dila0"]])
    assert code == B.EXIT_OK
    assert H.dump_points(url, TARGET)


def test_b1_an_unavailable_lock_refuses_before_any_delete(tmp_path, url, stages, monkeypatch):
    """The lock registry cannot be read (a stranger's work root that does not exist yet) or the lock
    is held: a refusal, with the collection untouched, never a traceback after a delete."""
    from cytir.ops import write_lock
    code, _ = build_run(url, tmp_path, [stages["dila0"]], name="first")
    assert code == B.EXIT_OK
    before = H.dump_points(url, TARGET)

    class RB:
        BatchError = write_lock.BatchError

        @staticmethod
        def acquire(name, task, purpose, *, pid=None):
            raise write_lock.LockRegistryUnreadable("the lock directory does not exist")

        @staticmethod
        def release(name, task):
            raise AssertionError("released a lock that was never taken")

    monkeypatch.setattr(Q, "_load_run_batch", lambda: RB)
    with pytest.raises(B.BuildRefusal, match="target lock could not be taken"):
        build_run(url, tmp_path, [stages["dila0"]], name="second", recreate=True)
    assert H.dump_points(url, TARGET) == before
