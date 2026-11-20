"""Live-Qdrant-port guard for Python CHILDREN and for gRPC (task 723, item 3; #559 follow-up 3).

The root `conftest.py` patches `socket.socket.connect` in the pytest process. A child Python does not
inherit that, and gRPC opens its sockets in its C core, below any Python patch. This file closes part
of both:

* `conftest.pytest_configure` puts this directory first on `PYTHONPATH` and the port list into
  `PYTEST_AW_LIVE_PORTS` (read from `tests/throwaway_qdrant.LIVE_PORTS`: this file holds NO copy).
  Task 2949: the loopback ranges and the session's throwaway come in `PYTEST_AW_QDRANT_RANGES` and
  `PYTEST_AW_QDRANT_ALLOWED`; `refused()` here is the child's twin of `throwaway_qdrant.refused`.
  The variable has no `AW_` prefix on purpose: `cyt serve` (`process_env`) drops every `AW_*` name.
  A child Python that starts normally imports this as `sitecustomize` and refuses a connect to a
  listed port (`socket.connect`, `connect_ex`). If it is imported WITHOUT a port list it says so on
  stderr (only a test session puts this file on a path).
* `subprocess.Popen` is patched here too (`install_popen`), in the pytest process and in every
  child, so a grandchild, or a child given a caller-built `env=`, gets the variables and the path.
* gRPC: `grpc.insecure_channel`, `grpc.secure_channel` and their `grpc.aio` twins refuse a target
  whose port is listed. In a child the patch is applied when `grpc` is first imported.
* Every refusal writes one stderr line and one file into the directory named by
  `PYTEST_AW_LIVE_REFUSALS`; the conftest fails the test (or the session) that left one unless the
  test is marked `live_port_probe`. A caller that swallows the exception does not hide it.

WHAT IT DOES NOT REACH (measured on review of #664; not a safety claim): `python -I`, `-E`, `-S`
(no sitecustomize); `os.posix_spawn`, `os.execve`, `os.spawnve` with an explicit env; a launcher
that strips the two variables; C-level connects in a child (uvloop, libcurl, any C extension; gRPC
is covered only at its four channel constructors); a `Popen` with `env` passed positionally; a child
that is not Python. `cytir.index.live_target.refuse_live_target` (asks the endpoint what it HOLDS)
stays the stop for a build. Nothing here is on any path on an operator box.
"""
import os
import re
import sys

ENV = "PYTEST_AW_LIVE_PORTS"
REFUSALS_ENV = "PYTEST_AW_LIVE_REFUSALS"
RANGES_ENV = "PYTEST_AW_QDRANT_RANGES"      # task 2949: "a-b,c-d" (written by the conftest from throwaway_qdrant)
ALLOWED_ENV = "PYTEST_AW_QDRANT_ALLOWED"    # the session's throwaway, "host:port[,host:port]"
_DONE = {"socket": False, "grpc": False}


class LiveQdrantPortTouched(AssertionError):
    """A process started by a test tried to reach a port a live Qdrant listens on."""


def _ports():
    try:
        return {int(p) for p in os.environ.get(ENV, "").split(",") if p.strip()}
    except ValueError:
        return set()


def _is_loopback(host):
    h = str(host).lower()
    return h in ("localhost", "::1", "0.0.0.0", "::", "0:0:0:0:0:0:0:1") or h.startswith(("127.", "::ffff:127."))


def refused(host, port, ports):
    """The child's twin of `throwaway_qdrant.refused`, from the env the conftest wrote: a listed port on
    any host, or a loopback port in the ranges that is not the session's throwaway."""
    if port in ports:
        return True
    ranges = []
    for part in os.environ.get(RANGES_ENV, "").split(","):
        lo, _, hi = part.partition("-")
        if lo.strip().isdigit() and hi.strip().isdigit():
            ranges.append((int(lo), int(hi)))
    if _is_loopback(host) and any(lo <= port <= hi for lo, hi in ranges):
        for item in os.environ.get(ALLOWED_ENV, "").split(","):
            h, _, p = item.strip().rpartition(":")
            if p.isdigit() and int(p) == port and _is_loopback(h.strip("[]")):
                return False
        return True
    return False


def target_host(target):
    """The host of a gRPC target (`host:port`, `dns:///host:port`, `ipv4:1.2.3.4:port`), else ''."""
    t = re.sub(r"^(dns:///|dns:|ipv4:|ipv6:|unix:)", "", str(target).strip())
    m = re.match(r"^\[?(.*?)\]?:\d+/?$", t)
    return m.group(1) if m else ""


def target_port(target):
    """The port of a gRPC target (`host:port`, `dns:///host:port`, `ipv4:1.2.3.4:port`), else None."""
    m = re.search(r":(\d+)/?$", str(target).strip())
    return int(m.group(1)) if m else None


def _refuse(what, ports):
    msg = f"[live-port guard] {what}, a live Qdrant port (pid {os.getpid()}, argv {sys.argv[:3]}): refused"
    try:
        sys.stderr.write(msg + "\n")
    except Exception:
        pass
    d = os.environ.get(REFUSALS_ENV)
    if d:
        try:
            with open(os.path.join(d, f"{os.getpid()}-{os.urandom(4).hex()}"), "w") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass
    raise LiveQdrantPortTouched(f"{what}, a live Qdrant port (pid {os.getpid()}): refused, nothing sent")


def install_socket(ports):
    import socket
    if _DONE["socket"]:
        return
    _DONE["socket"] = True
    real, real_ex = socket.socket.connect, socket.socket.connect_ex

    def check(address):
        if isinstance(address, tuple) and len(address) >= 2 and refused(address[0], address[1], ports):
            _refuse(f"a subprocess connected to {address!r}", ports)

    def connect(self, address, *a, **kw):
        check(address)
        return real(self, address, *a, **kw)

    def connect_ex(self, address, *a, **kw):
        check(address)
        return real_ex(self, address, *a, **kw)

    socket.socket.connect, socket.socket.connect_ex = connect, connect_ex


def install_popen(ports_csv, site_dir):
    """Make every `subprocess.Popen` carry the guard: with a caller-built `env=` the two variables and
    the path entry are put back. Returns the original `__init__` (the pytest process restores it)."""
    import subprocess
    real_init = subprocess.Popen.__init__
    if getattr(real_init, "_aw_live_port_guard", False):
        return real_init

    def init(self, *a, **kw):
        env = kw.get("env")
        if env is not None:
            env = dict(env)
            env[ENV] = os.environ.get(ENV) or ports_csv     # the live value: a session may extend it
            if os.environ.get(REFUSALS_ENV) and REFUSALS_ENV not in env:
                env[REFUSALS_ENV] = os.environ[REFUSALS_ENV]
            for name in (RANGES_ENV, ALLOWED_ENV):
                if name in os.environ:
                    env[name] = os.environ[name]
            cur = [x for x in env.get("PYTHONPATH", "").split(os.pathsep) if x and x != site_dir]
            env["PYTHONPATH"] = os.pathsep.join([site_dir] + cur)
            kw["env"] = env
        return real_init(self, *a, **kw)

    init._aw_live_port_guard = True
    subprocess.Popen.__init__ = init
    return real_init


def install_grpc(ports):
    """Patch the channel constructors of an already imported `grpc`."""
    grpc = sys.modules.get("grpc")
    if grpc is None or _DONE["grpc"] or not hasattr(grpc, "insecure_channel"):
        return
    _DONE["grpc"] = True

    def wrap(fn):
        def guarded(target, *a, **kw):
            tp = target_port(target)
            if tp is not None and refused(target_host(target), tp, ports):
                _refuse(f"a gRPC channel to {target!r}", ports)
            return fn(target, *a, **kw)
        guarded.__wrapped__ = fn
        return guarded

    for owner in (grpc, getattr(grpc, "aio", None)):
        for name in ("insecure_channel", "secure_channel"):
            if owner is not None and hasattr(owner, name):
                setattr(owner, name, wrap(getattr(owner, name)))


class _GrpcImportHook:
    """After `grpc` finishes importing in a child, patch it. Delegates the import itself."""

    def __init__(self, ports):
        self.ports = ports

    def find_spec(self, name, path=None, target=None):
        if name != "grpc":
            return None
        sys.meta_path.remove(self)
        try:
            import importlib.util
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader, ports = spec.loader, self.ports
        real_exec = loader.exec_module

        def exec_module(module):
            real_exec(module)
            sys.modules.setdefault("grpc", module)
            try:
                import grpc.aio  # noqa: F401  (so the aio twins are patched too)
            except Exception:
                pass
            install_grpc(ports)
            sys.meta_path[:] = [h for h in sys.meta_path if h is not self]
        loader.exec_module = exec_module
        return spec


def _chain_next_sitecustomize():
    """Run the next `sitecustomize` on the path (the distribution's), which ours shadows."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        from importlib.machinery import PathFinder
        spec = PathFinder.find_spec("sitecustomize", [p for p in sys.path
                                                      if os.path.abspath(p or ".") != here])
        if spec is not None and spec.loader is not None and spec.origin != __file__:
            import importlib.util
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
    except Exception as exc:
        sys.stderr.write(f"live-port site: chained sitecustomize failed: {exc!r}\n")


def _start():
    ports = _ports()
    if not ports:
        try:
            sys.stderr.write(f"[live-port guard] NOT active in pid {os.getpid()}: the port list "
                             f"({ENV}) is missing\n")
        except Exception:
            pass
        return
    install_socket(ports)
    install_popen(",".join(str(p) for p in sorted(ports)), os.path.dirname(os.path.abspath(__file__)))
    if "grpc" in sys.modules:
        install_grpc(ports)
    else:
        sys.meta_path.insert(0, _GrpcImportHook(ports))


if __name__ == "sitecustomize":
    _start()
    _chain_next_sitecustomize()
