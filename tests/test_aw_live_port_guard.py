"""The live-Qdrant-port guard of the root `conftest.py` (round 6 of #559: reviewer's follow-ups 2 and 3).

1. ONE port list. `conftest.LIVE_PORTS` IS `tests/throwaway_qdrant.LIVE_PORTS`, and no other
   file under the code trees spells that set as a literal (two identical copies had no test that
   pinned one to the other).
2. The guard is installed for the whole session (`pytest_configure`), so a module- or session-scoped
   fixture that connects to a live port is refused too. The reviewer's mutation, run in a child pytest
   with a stub listening on the "live" port: the stub accepts NOTHING, and a fixture that swallows
   the refusal still fails the run. A control on a port that is not live connects normally.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import site
import socket
import subprocess
import sys
import textwrap
import threading

import pytest

pytestmark = pytest.mark.guard

APP = pathlib.Path(__file__).resolve().parents[1]
ROOT_CONFTEST = APP / "conftest.py"
TQ_FILE = APP / "tests" / "throwaway_qdrant.py"

SITE_FILE = APP / "tests" / "live_port_site" / "sitecustomize.py"

pytestmark = pytest.mark.skipif(not (ROOT_CONFTEST.is_file() and TQ_FILE.is_file()),
                                reason="the root conftest and the throwaway gate are not in this tree")

#: Three ports of the list are enough to recognise it; the literal is never typed in this file.
_SIGNATURE = frozenset({int("63" + "33"), int("63" + "34"), int("63" + "98")})
_TREES = ("cytir", "tests", "pipeline", "server")


def _set_literals(path: pathlib.Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Set):
            consts = {e.value for e in node.elts if isinstance(e, ast.Constant)}
            if _SIGNATURE <= consts:
                yield node.lineno


def test_there_is_exactly_one_definition_of_the_live_port_list():
    files = [ROOT_CONFTEST] + [p for t in _TREES if (APP / t).is_dir() for p in (APP / t).rglob("*.py")]
    found = [f"{p.relative_to(APP)}:{n}" for p in files for n in _set_literals(p)]
    assert len(found) == 1 and found[0].startswith("tests/throwaway_qdrant.py:"), found


def test_the_conftest_list_is_the_gates_list_not_a_copy():
    spec = importlib.util.spec_from_file_location("aw_root_conftest_for_identity", ROOT_CONFTEST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)             # defines hooks; installs nothing (that is pytest_configure)
    assert mod.LIVE_PORTS is sys.modules["throwaway_qdrant"].LIVE_PORTS
    assert 6333 in mod.LIVE_PORTS and 6334 in mod.LIVE_PORTS and 6472 in mod.LIVE_PORTS
    assert 16333 in mod.LIVE_PORTS and 6486 in mod.LIVE_PORTS


def test_the_twin_port_is_refused_before_any_socket_connects():
    """Task 2940: 6486 (the twin, build box) is refused by the guard's own refusal function. That
    function raises before it reaches the real connect, so nothing is opened; no stub is bound on
    6486 and no socket is made to it."""
    spec = importlib.util.spec_from_file_location("aw_root_conftest_for_twin_refusal", ROOT_CONFTEST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(mod.LiveQdrantPortTouched):
        mod._refuse_live_port(("127.0.0.1", 6486))


# ------------------------------------------------------------------ the mutation, in a child pytest
class Stub:
    """A listener on a free loopback port that counts what it accepts."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.accepted = 0
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while True:
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            self.accepted += 1
            c.close()

    def close(self):
        self.sock.close()


CONFTEST = '''
import importlib.util, os
spec = importlib.util.spec_from_file_location("aw_root_conftest", {root!r})
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
if os.environ.get("STUB_IS_LIVE") == "1":
    m.LIVE_PORTS.add(int(os.environ["STUB_PORT"]))     # this child process only
pytest_configure = m.pytest_configure
pytest_unconfigure = m.pytest_unconfigure
pytest_sessionfinish = m.pytest_sessionfinish
_no_connect_to_a_live_qdrant_port = m._no_connect_to_a_live_qdrant_port
'''
INI = "[pytest]\nmarkers =\n    live_port_probe: the test proves the guard\n"

FIXTURES = {
    "session": '''
import os, socket, pytest
@pytest.fixture(scope="session")
def reach_out():
    s = socket.socket(); s.settimeout(2)
    s.connect(("127.0.0.1", int(os.environ["STUB_PORT"])))
def test_it(reach_out):
    pass
''',
    "module-swallowed": '''
import os, socket, pytest
@pytest.fixture(scope="module")
def reach_out():
    try:
        s = socket.socket(); s.settimeout(2)
        s.connect(("127.0.0.1", int(os.environ["STUB_PORT"])))
    except Exception:
        pass            # a client that wraps the refusal as "cannot read"
def test_it(reach_out):
    pass
''',
    "test-body-swallowed": '''
import os, socket
def test_it():
    try:
        s = socket.socket(); s.settimeout(2)
        s.connect(("127.0.0.1", int(os.environ["STUB_PORT"])))
    except Exception:
        pass
''',
}


def _child(tmp_path: pathlib.Path, body: str, stub: Stub, *, live: bool):
    (tmp_path / "conftest.py").write_text(CONFTEST.format(root=str(ROOT_CONFTEST)))
    (tmp_path / "pytest.ini").write_text(INI)
    (tmp_path / "test_child.py").write_text(textwrap.dedent(body))
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "STUB_PORT": str(stub.port),
           "STUB_IS_LIVE": "1" if live else "0", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONUSERBASE": site.getuserbase()}     # HOME is the temp dir: keep the user site (pytest)
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("kind", sorted(FIXTURES))
def test_a_connect_to_a_live_port_is_refused_wherever_it_is_made(tmp_path, kind):
    """The reviewer's mutation (a session-scoped fixture) and two ways a client can swallow the
    refusal. The child fails, names the guard, and the stub on the 'live' port accepted nothing."""
    stub = Stub()
    try:
        proc = _child(tmp_path, FIXTURES[kind], stub, live=True)
    finally:
        stub.close()
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "LiveQdrantPortTouched" in out or "live Qdrant port" in out, out
    assert stub.accepted == 0, f"the guard let a connect through ({kind})"


def test_the_control_a_port_that_is_not_live_connects_normally(tmp_path):
    stub = Stub()
    try:
        proc = _child(tmp_path, FIXTURES["session"], stub, live=False)
    finally:
        stub.close()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert stub.accepted == 1


# ------------------------------------------------------- subprocess + gRPC reach (task 723, item 3)
FIXTURES["subprocess"] = '''
import pytest
import os, subprocess, sys
@pytest.mark.live_port_probe
def test_it():
    code = ("import os, socket; s = socket.socket(); s.settimeout(2); "
            "s.connect(('127.0.0.1', int(os.environ['STUB_PORT'])))")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode != 0 and "LiveQdrantPortTouched" in p.stderr, p.stderr
'''
FIXTURES["subprocess-explicit-env"] = '''
import pytest
import os, subprocess, sys
@pytest.mark.live_port_probe
def test_it():
    code = ("import os, socket; s = socket.socket(); s.settimeout(2); "
            "s.connect(('127.0.0.1', int(os.environ['STUB_PORT'])))")
    env = {"PATH": "/usr/bin:/bin", "STUB_PORT": os.environ["STUB_PORT"]}      # a caller-built env
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert p.returncode != 0 and "LiveQdrantPortTouched" in p.stderr, p.stderr
'''
FIXTURES["grpc"] = '''
import pytest
import os, pytest
grpc = pytest.importorskip("grpc")
@pytest.mark.live_port_probe
def test_it():
    for make in (grpc.insecure_channel, grpc.aio.insecure_channel):
        with pytest.raises(AssertionError, match="live Qdrant port"):
            make("127.0.0.1:" + os.environ["STUB_PORT"])
        with pytest.raises(AssertionError, match="live Qdrant port"):
            make("dns:///localhost:" + os.environ["STUB_PORT"])
'''
FIXTURES["grpc-in-subprocess"] = '''
import pytest
import os, subprocess, sys, pytest
pytest.importorskip("grpc")
@pytest.mark.live_port_probe
def test_it():
    code = ("import grpc, os\\ngrpc.insecure_channel('127.0.0.1:' + os.environ['STUB_PORT'])")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode != 0 and "LiveQdrantPortTouched" in p.stderr, p.stderr
'''


@pytest.mark.parametrize("kind", ["subprocess", "subprocess-explicit-env", "grpc", "grpc-in-subprocess"])
def test_a_subprocess_and_a_grpc_channel_cannot_reach_a_live_port(tmp_path, kind):
    stub = Stub()
    try:
        proc = _child(tmp_path, FIXTURES[kind], stub, live=True)
    finally:
        stub.close()
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out                  # the child test asserts the refusal itself
    assert stub.accepted == 0, f"a connect reached the stub ({kind})"


def test_the_control_a_subprocess_to_a_port_that_is_not_live_connects(tmp_path):
    stub = Stub()
    try:
        proc = _child(tmp_path, FIXTURES["subprocess"].replace("p.returncode != 0 and \"LiveQdrantPortTouched\" in p.stderr",
                                                            "p.returncode == 0"), stub, live=False)
    finally:
        stub.close()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert stub.accepted == 1


def test_the_site_guard_holds_no_port_list_of_its_own():
    text = SITE_FILE.read_text()
    assert "LIVE_PORTS =" not in text and "import throwaway_qdrant" not in text
    assert not list(_set_literals(SITE_FILE))


def test_grpc_target_port_parser():
    spec = importlib.util.spec_from_file_location("site_guard_for_parse", SITE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = int("63" + "34")                    # no endpoint literal in a test file (test_throwaway_qdrant_guard)
    assert mod.target_port(f"h:{p}") == p
    assert mod.target_port(f"dns:///q.example:{p}") == p
    assert mod.target_port(f"ipv4:1.2.3.4:{p}") == p
    assert mod.target_port("no-port-here") is None


FIXTURES["swallowed-subprocess"] = '''
import os, subprocess, sys
def test_it():
    code = ("import os, socket\\ntry:\\n s = socket.socket(); s.settimeout(2)\\n"
            " s.connect(('127.0.0.1', int(os.environ['STUB_PORT'])))\\nexcept Exception:\\n pass")
    subprocess.run([sys.executable, "-c", code], check=True)       # the child swallows the refusal, exits 0
'''
FIXTURES["swallowed-grpc"] = '''
import os, pytest
grpc = pytest.importorskip("grpc")
def test_it():
    try:
        grpc.insecure_channel("127.0.0.1:" + os.environ["STUB_PORT"])
    except Exception:
        pass
'''
FIXTURES["grandchild-with-own-env"] = '''
import os, subprocess, sys
import pytest
@pytest.mark.live_port_probe
def test_it():
    inner = ("import os, socket; s = socket.socket(); s.settimeout(2); "
             "s.connect(('127.0.0.1', int(os.environ['STUB_PORT'])))")
    outer = ("import os, subprocess, sys; env = {'PATH': '/usr/bin:/bin', 'STUB_PORT': os.environ['STUB_PORT'], "
             "'PYTHONPATH': '/nonexistent'}; "
             "p = subprocess.run([sys.executable, '-c', %r], env=env, capture_output=True, text=True); "
             "sys.exit(0 if p.returncode != 0 and 'LiveQdrantPortTouched' in p.stderr else 7)" % inner)
    p = subprocess.run([sys.executable, "-c", outer], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
'''


@pytest.mark.parametrize("kind", ["swallowed-subprocess", "swallowed-grpc"])
def test_a_swallowed_refusal_still_fails_the_test(tmp_path, kind):
    """A child or a gRPC caller that catches the refusal leaves a record; the conftest fails the test."""
    stub = Stub()
    try:
        proc = _child(tmp_path, FIXTURES[kind], stub, live=True)
    finally:
        stub.close()
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "refused a live Qdrant port" in out, out
    assert stub.accepted == 0


def test_a_grandchild_with_its_own_env_and_an_overwritten_pythonpath_keeps_the_guard(tmp_path):
    stub = Stub()
    try:
        proc = _child(tmp_path, FIXTURES["grandchild-with-own-env"], stub, live=True)
    finally:
        stub.close()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert stub.accepted == 0


def test_the_guard_variables_survive_aw_serve_process_env():
    """`cyt serve` drops every ambient `AW_*` name before it execs uvicorn: the guard must not be one."""
    import os
    from cytir.ops import serve
    out, dropped = serve.process_env({}, ambient=dict(os.environ))
    assert os.environ.get("PYTEST_AW_LIVE_PORTS"), "the conftest did not set the guard"
    assert out["PYTEST_AW_LIVE_PORTS"] == os.environ["PYTEST_AW_LIVE_PORTS"]
    assert str(SITE_FILE.parent) in out["PYTHONPATH"].split(os.pathsep)
    assert not [d for d in dropped if "LIVE_PORTS" in d or "LIVE_REFUSALS" in d]


def test_a_child_without_a_port_list_says_so(tmp_path):
    """Started with `os.posix_spawn` (the Popen patch would add the list), so the list is missing."""
    import os
    err = tmp_path / "err.txt"
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(SITE_FILE.parent), "HOME": str(tmp_path)}
    actions = [(os.POSIX_SPAWN_OPEN, 2, str(err), os.O_WRONLY | os.O_CREAT, 0o600)]
    pid = os.posix_spawn(sys.executable, [sys.executable, "-c", "pass"], env, file_actions=actions)
    os.waitpid(pid, 0)
    assert "NOT active" in err.read_text(), err.read_text()


# ------------------------------------------------ task 2949: the guard is an ALLOWLIST on loopback
# Every proof below that sends a connect runs in a CHILD pytest whose `socket.connect` is a tripwire
# installed BEFORE the guard: a connect the guard lets through hits the tripwire, never a real port.
TRIP_CONFTEST = '''
import importlib.util, os, socket
def _trip(self, address, *a, **kw):
    raise OSError("TRIPWIRE: the guard let a connect through to %r" % (address,))
socket.socket.connect = socket.socket.connect_ex = _trip
spec = importlib.util.spec_from_file_location("aw_root_conftest", {root!r})
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
pytest_configure = m.pytest_configure
pytest_unconfigure = m.pytest_unconfigure
pytest_sessionfinish = m.pytest_sessionfinish
_no_connect_to_a_live_qdrant_port = m._no_connect_to_a_live_qdrant_port
'''
UNLISTED = [6390, 6392, 6399, 6421, 6444, 6456, 6475, 6483, 6484, 16457]


def _trip_child(tmp_path, body, extra_env=None, conftest=TRIP_CONFTEST):
    (tmp_path / "conftest.py").write_text(conftest.format(root=str(ROOT_CONFTEST)))
    (tmp_path / "pytest.ini").write_text(INI)
    (tmp_path / "test_child.py").write_text(textwrap.dedent(body))
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONUSERBASE": site.getuserbase(), **(extra_env or {})}
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)


def test_an_unlisted_loopback_qdrant_port_is_refused_before_the_syscall(tmp_path):
    """RED 1: ten ports of real services that no list names. Each connect must fail with the guard's
    exception (the tripwire below it would raise OSError instead)."""
    body = f'''
import socket, pytest
@pytest.mark.live_port_probe
@pytest.mark.parametrize("port", {UNLISTED!r})
def test_it(port):
    for host in ("127.0.0.1", "localhost", "::1"):
        s = socket.socket()
        with pytest.raises(AssertionError) as e:
            s.connect((host, port))
        assert type(e.value).__name__ == "LiveQdrantPortTouched", repr(e.value)
        with pytest.raises(AssertionError):
            s.connect_ex((host, port))
'''
    proc = _trip_child(tmp_path, body)
    assert proc.returncode == 0 and f"{len(UNLISTED)} passed" in proc.stdout, proc.stdout + proc.stderr


_STUB_CONFTEST = TRIP_CONFTEST.replace("spec.loader.exec_module(m)", '''spec.loader.exec_module(m)
_p = int(os.environ["STUB_PORT"])
m._TQ.QDRANT_RANGES = (range(_p, _p + 1),)      # this child's stub port stands for a Qdrant port''')
_STUB_CONFTEST = _STUB_CONFTEST.replace("socket.socket.connect = socket.socket.connect_ex = _trip", '''_real = socket._socket.socket.connect   # the C method, below any wrapper the outer session put on the child
def _only_stub(self, address, *a, **kw):
    if address[1] != int(os.environ["STUB_PORT"]):
        return _trip(self, address)
    return _real(self, address, *a, **kw)
socket.socket.connect = _only_stub          # only the stub is ever really reachable''')

_GRANDCHILD = '''
import os, socket, subprocess, sys, pytest
@pytest.mark.live_port_probe
def test_it():
    port = int(os.environ["STUB_PORT"])
    code = ("import os, socket; s = socket.socket(); s.settimeout(2); "
            "s.connect(('127.0.0.1', int(os.environ['STUB_PORT'])))")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    s = socket.socket(); s.settimeout(2)
    try:
        s.connect(("127.0.0.1", port)); here = "connected"
    except AssertionError as e:
        here = type(e).__name__
    open(os.environ["OUT"], "w").write(f"{here}|{p.returncode}")
'''


@pytest.mark.parametrize("own", [False, True])
def test_only_the_sessions_own_throwaway_may_be_reached_in_the_ranges(tmp_path, own):
    """RED 2 and 3: a stub on an ephemeral port stands for a port in the ranges. Without the
    throwaway URL the parent and a child Python are both refused (stub accepts nothing); with
    `AW_TEST_QDRANT_URL` set to the stub's host:port both connect."""
    stub = Stub()
    out = tmp_path / "out.txt"
    env = {"STUB_PORT": str(stub.port), "OUT": str(out)}
    if own:
        env["AW_TEST_QDRANT_URL"] = f"http://127.0.0.1:{stub.port}"
    try:
        proc = _trip_child(tmp_path, _GRANDCHILD, env, conftest=_STUB_CONFTEST)
    finally:
        stub.close()
    here, child_rc = out.read_text().split("|")
    if own:
        assert (here, child_rc) == ("connected", "0") and stub.accepted == 2, (here, child_rc, proc.stdout)
    else:
        assert here == "LiveQdrantPortTouched" and child_rc != "0" and stub.accepted == 0, \
            (here, child_rc, stub.accepted, proc.stdout + proc.stderr)


def test_a_throwaway_configured_inside_the_ranges_is_refused_by_name():
    """RED 4: not by luck of a socket error."""
    import throwaway_qdrant as TQ
    for url in ("http://127.0.0.1:" + str(6444), "http://localhost:" + str(6390), "http://127.0.0.1:" + str(16457)):
        with pytest.raises(AssertionError, match="never listens"):
            TQ.require_throwaway(url)


def test_the_range_edges():
    """RED 5: 6299, 6500, 16299, 16500 pass; 6300, 6499, 16300, 16499 refuse (loopback)."""
    import throwaway_qdrant as TQ
    for port in (6299, 6500, 16299, 16500, 26949):
        assert not TQ.refused("127.0.0.1", port), port
    for port in (6300, 6499, 16300, 16499):
        assert TQ.refused("127.0.0.1", port) and TQ.refused("::1", port) and TQ.refused("localhost", port), port
    # a non-loopback host is only held to the old list; the throwaway allows its own loopback spellings
    assert not TQ.refused("example.invalid", 6444)
    assert TQ.refused("example.invalid", 6333)
    assert not TQ.refused("localhost", 6444, {("127.0.0.1", 6444)})
    assert TQ.refused("127.0.0.1", 6445, {("127.0.0.1", 6444)})
    assert TQ.refused("127.0.0.1", 6333, {("127.0.0.1", 6333)})      # the old list is never allowed


def test_the_child_twin_agrees_with_the_gate(monkeypatch):
    import throwaway_qdrant as TQ
    spec = importlib.util.spec_from_file_location("aw_site_for_2949", SITE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setenv(mod.RANGES_ENV, ",".join(f"{r.start}-{r.stop - 1}" for r in TQ.QDRANT_RANGES))
    monkeypatch.setenv(mod.ALLOWED_ENV, "127.0.0.1:" + str(6444))
    ports = set(TQ.LIVE_PORTS)
    for host in ("127.0.0.1", "localhost", "::1", "example.invalid"):
        for port in (6299, 6300, 6390, 6444, 6499, 6500, 16299, 16300, 16457, 16499, 16500, 26949, *sorted(ports)):
            assert mod.refused(host, port, ports) == TQ.refused(host, port, {("127.0.0.1", 6444)}), (host, port)
    assert mod.target_host("dns:///127.0.0.1:6390") == "127.0.0.1" and mod.target_host("localhost:" + str(6390)) == "localhost"


def test_the_ranges_are_defined_once_and_never_copied():
    """RED 6: only tests/throwaway_qdrant.py holds the range bounds."""
    lo, hi = 6300, 6500
    hits = []
    for path in [ROOT_CONFTEST] + [p for t in _TREES if (APP / t).is_dir() for p in (APP / t).rglob("*.py")]:
        if path == pathlib.Path(__file__).resolve():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "range"
                    and [a.value for a in node.args if isinstance(a, ast.Constant)][:2] == [lo, hi]):
                hits.append(f"{path.relative_to(APP)}:{node.lineno}")
    assert len(hits) == 1 and hits[0].startswith("tests/throwaway_qdrant.py:"), hits
    assert "QDRANT_RANGES =" not in SITE_FILE.read_text()
