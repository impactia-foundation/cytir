"""Task 2254: the stand-in TEI server starts, answers like the job's handler, and stops on SIGTERM."""
from __future__ import annotations

import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import urllib.request

import standin_embedder as SE

SERVER = pathlib.Path(__file__).with_name("standin_tei_server.py")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_server_prints_ready_embeds_like_the_shared_handler_and_exits_on_sigterm():
    port = _free_port()
    p = subprocess.Popen([sys.executable, str(SERVER), "--port", str(port)], stdout=subprocess.PIPE, text=True,
                         env={**os.environ, "PYTHONPATH": str(SERVER.parent.parent)})  # aw importable as in CI (pip install -e .)
    try:
        assert p.stdout.readline().strip() == "ready"
        req = urllib.request.Request(f"http://127.0.0.1:{port}/embed", data=json.dumps({"inputs": ["hello world"]}).encode(),
                                     headers={"Content-Type": "application/json"})
        got = json.load(urllib.request.urlopen(req, timeout=10))
        assert got == [SE.embed("hello world").tolist()]
        info = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/info", timeout=10))
        assert info["stand_in"] == SE.KIND == "hashed-bow-v1"
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=15) == 0
    finally:
        if p.poll() is None:
            p.kill()
