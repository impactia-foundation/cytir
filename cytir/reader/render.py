"""Renderers: `render(url) -> html` of the page after JavaScript ran (task 2710).

`installed_browser` is the default (the user's own Chrome/Chromium/Edge/Brave; nothing bundled). `obscura`, `moli`
and `lightpanda` are used only when selected with AW_READER_RENDERER (or `--renderer`) and their binary is on PATH.

Limits by platform: the RSS watchdog reads /proc, so on macOS and Windows non-Chrome renderers have only the wall
timeout (and RLIMIT_CPU where `resource` exists; the child preexec is POSIX only). stderr is drained in a thread
(first 4 KB kept), so a renderer that floods it cannot stall until the timeout.

Redirects: the browser follows redirects itself and `--dump-dom` does not report the final URL, so a redirect hop
taken only inside the browser is NOT checked. The plain fetch of the same URL (read.py runs it first) already
validated that URL's hop chain; a page that redirects differently for a browser is a gap we accept and document.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
from dataclasses import dataclass

from . import config


class RenderError(RuntimeError):
    pass


def _is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def _group_rss(pgid: int) -> int:
    """Bytes of resident memory summed over the processes of group `pgid` (Linux /proc; 0 elsewhere)."""
    total, page = 0, os.sysconf("SC_PAGE_SIZE")
    try:
        pids = [d for d in os.listdir("/proc") if d.isnumeric()]
    except OSError:
        return 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat") as f:
                rest = f.read().rsplit(")", 1)[1].split()
            if int(rest[2]) == pgid:   # fields after the comm: state ppid pgrp ...
                total += int(rest[21]) * page
        except (OSError, IndexError, ValueError):
            continue
    return total


@dataclass
class Renderer:
    name: str
    binary: str

    limited = False   # True for non-Chrome renderers: RLIMIT_CPU in the child plus an RSS watchdog

    def argv(self, url: str) -> list[str]:
        raise NotImplementedError

    def _preexec(self) -> None:
        import resource
        # No RLIMIT_AS: V8 reserves a huge virtual range and obscura/moli abort at start even under 32 GB
        # (measured, task 2710). Memory is capped by the RSS watchdog in render() instead.
        resource.setrlimit(resource.RLIMIT_CPU, (config.ALT_RENDERER_RLIMIT_CPU_S,) * 2)

    def render(self, url: str) -> str:
        """Run in its own session; on timeout or oversize kill the WHOLE process group (Chrome's helpers outlive
        the parent); stdout is read bounded to MAX_BYTES."""
        p = subprocess.Popen(self.argv(url), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             start_new_session=True, preexec_fn=self._preexec if self.limited else None)
        timed_out: list[int] = []
        errbuf = bytearray()

        def drain() -> None:
            for chunk in iter(lambda: p.stderr.read(4096), b""):
                if len(errbuf) < 4096:
                    errbuf.extend(chunk[:4096 - len(errbuf)])
        drainer = threading.Thread(target=drain, daemon=True)
        drainer.start()
        over: list[int] = []

        def kill() -> None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

        timer = threading.Timer(config.RENDER_TIMEOUT_S, lambda: (timed_out.append(1), kill()))
        timer.start()
        stop = threading.Event()
        if self.limited:
            def watch() -> None:
                while not stop.wait(0.2):
                    if _group_rss(p.pid) > config.ALT_RENDERER_MAX_RSS:
                        over.append(1)
                        kill()
                        return
            threading.Thread(target=watch, daemon=True).start()
        try:
            out = p.stdout.read(config.MAX_BYTES + 1)
            if len(out) > config.MAX_BYTES:
                raise RenderError(f"{self.name} output over {config.MAX_BYTES} bytes")
            p.wait()
        finally:
            timer.cancel()
            stop.set()
            kill()   # the group, always: leaves no helper behind
            p.wait()
            p.stdout.close()
            drainer.join(2)
            err = bytes(errbuf)
        if over:
            raise RenderError(f"{self.name} killed: memory over {config.ALT_RENDERER_MAX_RSS >> 20} MB")
        if timed_out:
            raise RenderError(f"{self.name} timed out after {config.RENDER_TIMEOUT_S:.0f} s")
        text = out.decode("utf-8", errors="replace")
        if p.returncode != 0 or not text.strip():
            raise RenderError(f"{self.name} exit {p.returncode}: {err.decode('utf-8', 'replace')[-200:]}")
        return text


class InstalledBrowser(Renderer):
    def argv(self, url: str) -> list[str]:
        a = [self.binary, "--headless=new", "--disable-gpu", "--dump-dom",
             f"--virtual-time-budget={config.VIRTUAL_TIME_BUDGET_MS}"]
        if _is_root():
            a.append("--no-sandbox")   # Chrome refuses to start as root without it; never added otherwise
        return a + [url]


# The default settle of obscura/moli returns before fedlex's app has drawn the law (measured, task 2710).
ALT_WAIT_S = 10
MOLI_TIMEOUT_MS = 30000   # moli --wait networkidle needed ~30 s on fedlex; 10 s returned the shell


class Obscura(Renderer):
    limited = True

    def argv(self, url: str) -> list[str]:
        return [self.binary, "fetch", url, "--dump", "html", "--quiet", "--wait", str(ALT_WAIT_S)]
        # No --user-agent: obscura timed out at 6.5 GB RSS on fedlex when one was passed (measured, task 2710).


class Moli(Renderer):
    limited = True

    def argv(self, url: str) -> list[str]:
        return [self.binary, "fetch", "--dump", "html", "--wait", "networkidle", "-t", str(MOLI_TIMEOUT_MS), url]


class Lightpanda(Renderer):
    limited = True

    def argv(self, url: str) -> list[str]:
        return [self.binary, "fetch", "--dump", "html", url]


_ALT = {"obscura": Obscura, "moli": Moli, "lightpanda": Lightpanda}


def find_renderer(choice: str | None = None) -> Renderer | None:
    """The renderer for `choice` (default AW_READER_RENDERER; `auto` = installed browser), or None when absent."""
    choice = (choice or config.renderer_choice()).lower()
    if choice == "none":
        return None
    if choice in _ALT:
        b = shutil.which(choice)
        return _ALT[choice](choice, b) if b else None
    for name in config.BROWSER_BINARIES:   # auto and installed_browser
        b = shutil.which(name)
        if b:
            return InstalledBrowser(name, b)
    return None
