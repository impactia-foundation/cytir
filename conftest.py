"""Root pytest configuration for the CI test run (task 530).

The ignore hook below excludes files that match pytest's `test_*.py` naming
convention but are NOT pytest test suites -- they are cattle: one-off
task-scoped checker/report scripts (module-level `print`/`sys.exit`, no
`def test_*`, meant to be run directly with `python3 <script>.py`, often
against build-box/production paths). Importing them under pytest either
does nothing useful (a `if __name__ == "__main__":` guard makes them
collect zero tests) or actively crashes collection.

Each file named below crashes the WHOLE pytest run if imported, so it cannot be
handled with a `pytest.mark` skip inside the file -- it never gets that far.
"""

#: Standalone checker scripts that are named `test_*.py` but are not pytest suites: they call
#: `sys.exit(...)` at module level with no `__main__` guard, so collecting them raises SystemExit
#: out of the import and takes down collection for the whole run (an INTERNALERROR; task 530
#: baseline). Matched by file name, so the rule holds wherever the script lives.
_NOT_PYTEST_SUITES = frozenset({"test_saturate_upsert.py"})


def pytest_ignore_collect(collection_path, config):
    if collection_path.name in _NOT_PYTEST_SUITES:
        return True
    return None


# --- a test session never reads the box's operator file ---------------------------
# CI and the cut rehearsal run ON the operator's build box. Once `/etc/aw/operator.json` exists
# there, `cytir.core.paths` would hand every test OUR values, and a stranger-side refusal defect
# (a setting that must refuse by name on a box with no overlay) would pass unseen. So the session
# pins the overlay lookup to "no box file" (`AW_TEST_OVERLAY_LOOKUP=""`, the mechanism the code has
# for this) and drops a stray `AW_OPERATOR_OVERLAY`. The operator-values tests (the operator suite)
# opt in explicitly: that suite's own conftest points `AW_OPERATOR_OVERLAY` at the repo's own
# `deploy/operator/impactia.json` (review of #567, round 2, item 3).
import os  # noqa: E402
import sys  # noqa: E402

import pytest  # noqa: E402

_LOOKUP, _NAMED = "AW_TEST_OVERLAY_LOOKUP", "AW_OPERATOR_OVERLAY"
# The crawler identity under BOTH spellings. A reader who
# exported CYTIR_OPERATOR_* (README, the CI workflow) must not make every test that sets AW_OPERATOR_*
# itself fail with EnvNameConflict (task 2991b). Tests that need an identity set their own.
_IDENTITY_ENV = tuple(f"{p}OPERATOR_{k}" for p in ("CYTIR_", "AW_") for k in ("URL", "CONTACT"))


@pytest.hookimpl(tryfirst=True)     # before the operator suite opts in with its own overlay
def pytest_configure(config):
    config._aw_saved_overlay_env = {k: os.environ.get(k) for k in (_LOOKUP, _NAMED) + _IDENTITY_ENV}
    for k in _IDENTITY_ENV:
        os.environ.pop(k, None)
    os.environ[_LOOKUP] = ""
    os.environ.pop(_NAMED, None)
    # The suite runs offline (the reach guard refuses sockets); a cached tokenizer must load without the hub's
    # revision check (task 2289). Absent cache stays absent: the chunker then refuses, loudly.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _install_live_port_guard()
    _install_reach_guard()


def pytest_unconfigure(config):
    _remove_reach_guard()
    _remove_live_port_guard()
    for k, v in getattr(config, "_aw_saved_overlay_env", {}).items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# --- the lock registry is never the real one under test ------------------------
# CI runs on the self-hosted CI runner, where `LOCK_DIR` is the live
# `TWIN-WRITE-LOCK` / `PROD-WRITE-LOCK` registry. A test that read it went red
# whenever a lane held a lock, which blocked merge-on-green for unrelated PRs.
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_write_lock_registry(tmp_path_factory, monkeypatch):
    from cytir.ops import write_lock
    registry = tmp_path_factory.mktemp("lock-registry")
    # setitem on the module dict: `LOCK_HOST` / `LOCK_DIR` resolve lazily (an unconfigured box
    # refuses by name), so `setattr` would read the setting first and refuse on a bare checkout.
    monkeypatch.setitem(write_lock.__dict__, "LOCK_HOST", write_lock.LOCAL_HOST)
    monkeypatch.setitem(write_lock.__dict__, "LOCK_DIR", str(registry))
    real_ssh = write_lock.ssh

    def guarded(host, command, timeout=120):
        if host != write_lock.LOCAL_HOST:
            raise AssertionError(
                f"a test reached host {host!r} through write_lock.ssh; tests use the "
                "local, isolated lock registry (conftest._isolated_write_lock_registry)")
        return real_ssh(host, command, timeout)
    monkeypatch.setattr(write_lock, "ssh", guarded)


@pytest.fixture(autouse=True)
def _prod_guard_env_is_clean(monkeypatch):
    """The prod gate (`cytir.index.prod_guard`) reads AW_REPLAY / AW_PROD_URL / AW_PROD_COLLECTIONS /
    AW_PROD_READ_BUDGET from the environment. A session run from a replay shell (AW_REPLAY=1) must
    not change what a test sees: clear them for every test; a test that needs one sets it itself.
    CI never sets AW_PROD_COLLECTIONS (the drain-golden harness names its throwaway prod `aw_e9`)."""
    for k in ("AW_REPLAY", "AW_PROD_URL", "AW_PROD_COLLECTIONS", "AW_PROD_READ_BUDGET"):
        monkeypatch.delenv(k, raising=False)
    from cytir.index import prod_guard as PG
    PG.reset()
    # no test asks an endpoint for its aliases (a connect from the guard would be a live-port
    # attempt); the tests of the alias probe patch `_fetch_aliases` themselves.
    monkeypatch.setattr(PG, "_fetch_aliases", lambda url: None)
    yield
    PG.reset()


@pytest.fixture(autouse=True)
def _consent_cache_is_an_empty_tmp_dir(tmp_path_factory, monkeypatch):
    """Task 684: `cytir.index.compose` stamps consent at compose time and refuses when the
    resolver's cache directory is missing. A test must never read the operator's real
    ~/aw-data/consent, so every test sees an empty temp cache unless it patches its own."""
    import cytir.consent.stamp as stamp
    d = tmp_path_factory.mktemp("consent-cache")
    monkeypatch.setattr(stamp, "default_cache_dir", lambda: str(d))


# --- no test connects to a live Qdrant port (round 5, N1 of #559; round 6) ----------------------
# The default Qdrant port is 6333: a stranger's own instance, and on a CI runner a legacy
# store. A connect to a live port by any spelling (a URL with its port stripped, a client default, an
# endpoint derived from a literal) is refused HERE, before the syscall, whatever the test says.
# The port list is `tests/throwaway_qdrant.LIVE_PORTS`: ONE definition (a test pins that).
#
# The patch is installed at `pytest_configure`, for the whole session, NOT per test: a module- or
# session-scoped fixture that connects runs before any function-scoped fixture, and a per-test patch
# let it through (round 5 review, measured). What it catches that a test body could swallow (a client
# that wraps the refusal as "cannot read"): every refused attempt is RECORDED, and the test fails at
# teardown (`live_port_probe` marks a test that proves the refusal itself). An attempt made OUTSIDE a
# test body (a broader-scoped fixture, collection) fails the next test's setup and the session.
#
# WHAT IT DOES NOT REACH, and so is not a safety claim: a SUBPROCESS and gRPC are reached only in part
# (see the next section and `tests/live_port_site/sitecustomize.py`, which lists the measured gaps:
# `python -I/-E/-S`, `os.posix_spawn`/`execve` with an explicit env, a launcher that strips the guard
# variables, C-level connects such as uvloop, non-Python children). For a subprocess build the stop is
# `cytir.index.live_target.refuse_live_target`, which asks the endpoint what it HOLDS and fails closed.
# A throwaway never listens on these ports (`tests/throwaway_qdrant.py` refuses them as well), so a
# test that targets only its own throwaway never trips any of this.
import importlib.util  # noqa: E402
import pathlib  # noqa: E402


def _load_throwaway_qdrant():
    """`tests/throwaway_qdrant.py`, registered as `throwaway_qdrant` so that every later
    `import throwaway_qdrant` (the harness, the tests) is THIS module, with one `_PROVEN` set."""
    mod = sys.modules.get("throwaway_qdrant")
    if mod is None:
        path = pathlib.Path(__file__).resolve().parent / "tests" / "throwaway_qdrant.py"
        spec = importlib.util.spec_from_file_location("throwaway_qdrant", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["throwaway_qdrant"] = mod
        spec.loader.exec_module(mod)
    return mod


_TQ = _load_throwaway_qdrant()
LIVE_PORTS = _TQ.LIVE_PORTS       # the one list; never copied here


class LiveQdrantPortTouched(AssertionError):
    """A test tried to connect to a port a live Qdrant listens on. Nothing was sent."""


class _GuardState:
    current: list | None = None     # the running test's attempts (None outside a test body)
    outside: list = []              # attempts made outside a test body (fixtures of wider scope)
    originals: tuple | None = None


def _refuse_live_port(address):
    # task 2949: the list AND the loopback ranges, minus the session's own throwaway (an allowlist)
    if (isinstance(address, tuple) and len(address) >= 2
            and _TQ.refused(address[0], address[1], _TQ.allowed_from_env())):
        (_GuardState.current if _GuardState.current is not None else _GuardState.outside).append(address)
        import threading
        import traceback
        where = "".join(traceback.format_stack(limit=12))
        sys.stderr.write(f"[live-port guard] thread {threading.current_thread().name} "
                         f"connect to {address!r} refused from:\n{where}\n")
        raise LiveQdrantPortTouched(f"a test connected to {address!r}, a live Qdrant port")


def _install_live_port_guard():
    import socket
    if _GuardState.originals is not None:
        return
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def connect(self, address, *a, **kw):
        _refuse_live_port(address)
        return real_connect(self, address, *a, **kw)

    def connect_ex(self, address, *a, **kw):
        _refuse_live_port(address)
        return real_connect_ex(self, address, *a, **kw)

    _GuardState.originals = (real_connect, real_connect_ex)
    socket.socket.connect, socket.socket.connect_ex = connect, connect_ex


def _remove_live_port_guard():
    import socket
    if _GuardState.originals is not None:
        socket.socket.connect, socket.socket.connect_ex = _GuardState.originals
        _GuardState.originals = None


# --- subprocess + gRPC reach (task 723, item 3 of the #559 follow-ups) ---------------------------
# A child Python gets `tests/live_port_site/sitecustomize.py` on its PYTHONPATH and the port list in
# `PYTEST_AW_LIVE_PORTS` (no `AW_` prefix: `cyt serve` strips those), so it refuses a connect to a live
# port itself; `Popen` is patched (in this process and in every child) so a caller-built `env=` keeps
# both; gRPC channel constructors are patched. EVERY refusal, here or in a child, leaves a file in
# `PYTEST_AW_LIVE_REFUSALS`: a test that left one fails at teardown unless it is marked
# `live_port_probe`, and one left outside a test fails the session. A swallowed exception hides nothing.
# Not reached: the list in the site file's docstring (`-I`/`-E`/`-S`, `posix_spawn`/`execve` with an
# explicit env, stripping launchers, C-level connects, non-Python children).
_SITE_DIR = str(pathlib.Path(__file__).resolve().parent / "tests" / "live_port_site")
_REACH_ENV = "PYTEST_AW_LIVE_PORTS"
_REFUSALS_ENV = "PYTEST_AW_LIVE_REFUSALS"
_RANGES_ENV = "PYTEST_AW_QDRANT_RANGES"       # task 2949: "a-b,c-d", written from throwaway_qdrant
_ALLOWED_ENV = "PYTEST_AW_QDRANT_ALLOWED"     # the session's throwaway, "host:port"


def _load_site_guard():
    mod = sys.modules.get("aw_live_port_site")
    if mod is None:
        spec = importlib.util.spec_from_file_location("aw_live_port_site", pathlib.Path(_SITE_DIR) / "sitecustomize.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["aw_live_port_site"] = mod
        spec.loader.exec_module(mod)
    return mod


class _ReachState:
    saved: dict | None = None
    popen_init = None
    refusal_root: str | None = None


def _install_reach_guard():
    if _ReachState.saved is not None:
        return
    import subprocess
    import tempfile
    _ReachState.saved = {k: os.environ.get(k) for k in (_REACH_ENV, _REFUSALS_ENV, "PYTHONPATH")}
    _ReachState.refusal_root = tempfile.mkdtemp(prefix="aw-live-refusals-")
    os.environ[_REFUSALS_ENV] = _ReachState.refusal_root
    ports_csv = ",".join(str(p) for p in sorted(LIVE_PORTS))
    os.environ[_REACH_ENV] = ports_csv
    for k in (_RANGES_ENV, _ALLOWED_ENV):
        _ReachState.saved[k] = os.environ.get(k)
    os.environ[_RANGES_ENV] = ",".join(f"{r.start}-{r.stop - 1}" for r in _TQ.QDRANT_RANGES)
    os.environ[_ALLOWED_ENV] = ",".join(f"{h}:{p}" for h, p in sorted(_TQ.allowed_from_env()))
    parts = [_SITE_DIR] + [x for x in os.environ.get("PYTHONPATH", "").split(os.pathsep) if x and x != _SITE_DIR]
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    guard = _load_site_guard()
    _ReachState.popen_init = subprocess.Popen.__init__
    guard.install_popen(ports_csv, _SITE_DIR)
    try:                                        # gRPC, in this process: only if the package is installed
        if importlib.util.find_spec("grpc") is not None:
            import grpc  # noqa: F401
            import grpc.aio  # noqa: F401
            guard.install_grpc(set(LIVE_PORTS))
    except Exception as exc:                    # a broken grpc install must not hide the other guards
        sys.stderr.write(f"[live-port guard] grpc not guarded: {exc!r}\n")


def _remove_reach_guard():
    if _ReachState.saved is None:
        return
    import shutil
    import subprocess
    if _ReachState.popen_init is not None:
        subprocess.Popen.__init__ = _ReachState.popen_init
        _ReachState.popen_init = None
    for k, v in _ReachState.saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    shutil.rmtree(_ReachState.refusal_root or "", ignore_errors=True)
    _ReachState.saved = None


def _refusal_files(d):
    try:
        return sorted(p for p in pathlib.Path(d).iterdir() if p.is_file())
    except OSError:
        return []


@pytest.fixture(autouse=True)
def _no_connect_to_a_live_qdrant_port(request):
    outside, _GuardState.outside = _GuardState.outside, []
    if outside:
        raise LiveQdrantPortTouched(
            f"a fixture of broader scope (or collection) tried to connect to a live Qdrant port: "
            f"{outside}")
    attempts: list = []
    _GuardState.current = attempts
    root = _ReachState.refusal_root
    per_test = None
    if root:
        per_test = tempfile_mkdtemp(root)
        os.environ[_REFUSALS_ENV] = per_test
    yield attempts
    _GuardState.current = None
    if per_test:
        os.environ[_REFUSALS_ENV] = root
        left = _refusal_files(per_test)
        if left and request.node.get_closest_marker("live_port_probe") is None:
            raise LiveQdrantPortTouched(
                "a process started by the test (or gRPC) was refused a live Qdrant port: "
                + "; ".join(p.read_text().strip() for p in left))
    if attempts and request.node.get_closest_marker("live_port_probe") is None:
        raise LiveQdrantPortTouched(f"a test tried to connect to a live Qdrant port: {attempts}")


def tempfile_mkdtemp(root):
    import tempfile
    return tempfile.mkdtemp(dir=root)


def pytest_sessionfinish(session, exitstatus):
    left = _refusal_files(_ReachState.refusal_root) if _ReachState.refusal_root else []
    if left:
        sys.stderr.write("[live-port guard] refusals outside any test: "
                         + "; ".join(p.read_text().strip() for p in left) + "\n")
        session.exitstatus = 1
    if _GuardState.outside:
        sys.stderr.write(f"[live-port guard] attempts outside any test: {_GuardState.outside}\n")
        session.exitstatus = 1


@pytest.fixture(autouse=True)
def _urlopen_fakes_reach_internal_http(monkeypatch):
    """Tests fake the Qdrant wire by replacing `urllib.request.urlopen`. The `cytir/index/` tools now go
    through `cytir.core.internal_http.send` (consent step 5, task 1648); while a test has replaced
    `urlopen`, `send` hands the request to that fake in the old shape, so those tests keep their seam."""
    import json as _json
    import urllib.error as _uerr
    import urllib.request as _ureq
    from cytir.core import internal_http as IH
    original = _ureq.urlopen
    real_send = IH.send

    def send(url, *, body=None, method=None, timeout=30.0, headers=None, data=None):
        fake = _ureq.urlopen
        if fake is original:
            return real_send(url, body=body, method=method, timeout=timeout, headers=headers, data=data)
        hdrs = dict(headers or {})
        data = data if body is None else None
        if body is not None:
            data = _json.dumps(body).encode()
            hdrs.setdefault("Content-Type", "application/json")
        req = _ureq.Request(url, data=data, method=method, headers=hdrs)
        try:
            with fake(req, timeout=timeout) as r:
                return int(getattr(r, "status", None) or getattr(r, "code", None) or 200), r.read()
        except _uerr.HTTPError as exc:
            return exc.code, exc.read()

    monkeypatch.setattr(IH, "send", send)


@pytest.fixture(autouse=True)
def _chunk_state_is_restored_after_each_test():
    """The chunker's selected policy and tokenizer revision are process-global; a test that changes them must not leak
    them to the next test in the same process (tasks 2281, 2317).

    `tests/` families left the epoch-4 policy active (2281). The delta-adapter stage tests in `pipeline/` run the
    real stage, which calls `select_policy` and `select_tokenizer_revision`, so a following `tests/test_chunk.py`
    chunked under the wrong epoch and tokenizer (2317). Lives in the ROOT conftest so it covers both trees. Only acts
    if the chunker is already imported."""
    import sys
    mod = sys.modules.get("cytir.chunk.chunker")
    previous = (mod.active_policy(), mod.TOKENIZER_REVISION) if mod is not None else None
    yield
    mod = sys.modules.get("cytir.chunk.chunker")
    if mod is None or previous is None:
        return
    policy, revision = previous
    if mod.active_policy() is not policy:
        mod._apply_policy(policy, allow_env_overrides=True)
    if mod.TOKENIZER_REVISION != revision:
        mod.select_tokenizer_revision(revision)


@pytest.fixture
def word_only_chunking_allowed(monkeypatch):
    """Opt ONE test in to word-only chunking on a box without the tokenizer (task 2289).

    Not autouse: each test that runs the chunker without `transformers` names itself, so the
    list of opted-in tests measures how much coverage runs word-only. Production never sets the flag."""
    from cytir.chunk import chunker  # noqa: PLC0415
    monkeypatch.setattr(chunker, "_ALLOW_WORD_ONLY_FOR_TESTS", True)
