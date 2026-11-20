"""Make the package suite collectable, and offline, on a clean clone.

Lives in tests/ (task 696 slice 2); the private build tree's own conftest loads this file, so
there is one implementation.

A test that opens an operator-only file (deploy/, scripts/, the build trees) uses
tests/helpers/repo_files.py (repo_file_or_skip / repo_glob_or_skip), never a hand-written skip.

Task 529. Two problems this fixes, both measured on a fresh checkout:

1. **30 test modules could not even be COLLECTED.** `cytir.core.user_agent`
   refuses at import time when no crawler identity is configured -- correct
   behaviour for a fetcher, and the reason a stranger cloning this repo cannot
   accidentally crawl as us. But it means `pytest` died during
   collection on any machine without `operator.json`, so a third of the
   consent, materialization and ingest tests were unrunnable by exactly the
   person the cold-start rule exists for (memory: cold-start-reproducibility,
   Tim 2026-08-19).

2. **Nothing made "the tests never touch the network" true.** The escape hatch
   above is only safe if that holds, so this asserts it rather than assuming
   it: every outbound socket to a non-loopback address raises during a test
   session. A test that quietly reaches a real origin is a probe with no
   consent receipt, and it makes the suite's result depend on the weather.

Both are session-scoped and apply only under pytest.
"""
from __future__ import annotations

import os
import socket

import pytest

_ALLOW = "AW_ALLOW_UNIDENTIFIED_CRAWL"


def pytest_configure(config):
    # Only when the box has no identity of its own: on a configured machine the
    # suite must behave exactly as it does in production, not take a shortcut.
    if not os.environ.get("AW_OPERATOR_URL") and not os.environ.get(_ALLOW):
        os.environ[_ALLOW] = "1"
        config._aw_set_allow = True


def pytest_unconfigure(config):
    if getattr(config, "_aw_set_allow", False):
        os.environ.pop(_ALLOW, None)


class NetworkAccessDuringTests(RuntimeError):
    """A test tried to open a socket to something that is not this machine."""


def _is_local(address) -> bool:
    if not isinstance(address, tuple) or not address:
        return True                      # AF_UNIX and friends
    host = address[0]
    if host in ("", "localhost", "127.0.0.1", "::1", "0.0.0.0", "::"):
        return True
    return str(host).startswith("127.")


@pytest.fixture(autouse=True, scope="session")
def _no_network():
    """The suite is offline. Loopback stays open (subprocess/multiprocessing)."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def guard(self, address, *a, **kw):
        if not _is_local(address):
            raise NetworkAccessDuringTests(
                f"a test opened a socket to {address!r}. The test suite "
                "runs offline: a test that reaches a real origin is a fetch "
                "with no consent receipt, and it makes the result depend on "
                "the network. Use a recorded fixture under tests/testdata/.")
        return real_connect(self, address, *a, **kw)

    def guard_ex(self, address, *a, **kw):
        if not _is_local(address):
            raise NetworkAccessDuringTests(
                f"a test opened a socket to {address!r} (connect_ex)")
        return real_connect_ex(self, address, *a, **kw)

    socket.socket.connect = guard
    socket.socket.connect_ex = guard_ex
    try:
        yield
    finally:
        socket.socket.connect = real_connect
        socket.socket.connect_ex = real_connect_ex


@pytest.fixture(autouse=True)
def _no_operator_overlay(monkeypatch):
    """Every test in `tests/` runs as a stranger would: no operator overlay, no box file.

    The root `conftest.py` pins this for a session (task 696 slice 4, step B0). This fixture pins it
    per TEST, so a mixed session, where the operator's own suite turns an overlay on, cannot leak
    it into a stranger-side test.
    """
    from cytir.core import paths
    monkeypatch.delenv(paths.OVERLAY_ENV, raising=False)
    monkeypatch.setenv(paths.TEST_LOOKUP_ENV, "")
    paths.reset_settings_cache()
    yield
    paths.reset_settings_cache()


@pytest.fixture(autouse=True, scope="module")
def _source_family_registry_isolated_per_module():
    """Task 2389: the family registry is process-wide state. A module that registers a family
    (or whose test registers one by accident) must not leak it into the next module on the same
    worker: `register_family` refuses a second adapter for one family id, so a leak is a random
    red. Load every family module first, so the snapshot is complete (an import runs once per
    process; a family dropped here would never come back), then restore after the module."""
    from cytir.sources import registry
    registry.load_modules()
    before = dict(registry._FACTORIES)
    yield
    registry._FACTORIES.clear()
    registry._FACTORIES.update(before)


@pytest.fixture(autouse=True)
def _sparse_rule_readback_isolated():
    """`cytir.index.sparse_rule.LAST_READBACK` is a module-level dict the rule's readback fills per collection;
    `verify_written_form` copies it into the prod receipt. A test that ran the rule left its entry for the next
    test, whose golden receipt then gained a `sparse_rule_readback` key (task 2933: `test_promote_queue_dry_run`
    before `test_drain_content_pack_golden`). Empty before and after each test."""
    import sys

    def clear():
        sr = sys.modules.get("cytir.index.sparse_rule")
        if sr is not None:
            sr.LAST_READBACK.clear()
    clear()
    yield
    clear()


@pytest.fixture(autouse=True)
def _no_twin_write_hold(monkeypatch, tmp_path):
    """No test reads the box's aw_e10 twin-writer hold (`/var/lib/aw/twin-write-hold.json`, task
    756/864): a hold that exists on the box running the suite must not refuse a test's twin writer.
    A test of the hold sets its own file."""
    monkeypatch.setenv("AW_TWIN_WRITE_HOLD", str(tmp_path / "no-twin-write-hold.json"))


@pytest.fixture
def no_sparse_rule(monkeypatch):
    """Switch the aw_e9 BM25 prune rule off (task 717). For tests of OTHER contracts that write
    through the drain's door into `aw_e9` on recorded rows or a stand-in build: the payload form,
    or "a stranger's build equals what the drain puts on prod". With the rule on, prod's rows are
    pruned and stamped, which a stranger's build (no term set of its own) does not do; the rule
    and its write path are tested in tests/test_aw_sparse_rule.py."""
    from cytir.index import sparse_rule as SR
    monkeypatch.setattr(SR.SparseRule, "for_collection", classmethod(lambda cls, *a, **k: None))
    monkeypatch.setattr(SR, "declared_epoch", lambda *a, **k: None)
