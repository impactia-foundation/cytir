"""The stand-in TEI server of the `index-serve-gate` CI job (task 2254): NOT the pinned model.

    python tests/standin_tei_server.py --port N [--host 127.0.0.1]

Mounts `standin_embedder.tei_handler()` on `http.server`, prints `ready` once it listens, and exits
cleanly on SIGTERM. The CI job starts it in the background, points `cyt gate serve-parity prepare
--tei-url` at it and stops it at the end. The job's tests and this server share the SAME handler
(and the same KIND, `hashed-bow-v1`), so the fixture index's document vectors and the parity run's
query vectors agree. Nothing in `cytir/` imports this file or anything else under `tests/`.
"""
from __future__ import annotations

import argparse
import http.server
import signal
import sys
import threading

import standin_embedder as SE


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)
    srv = http.server.ThreadingHTTPServer((a.host, a.port), SE.tei_handler())
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=srv.shutdown, daemon=True).start())
    print("ready", flush=True)
    try:
        srv.serve_forever()
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
