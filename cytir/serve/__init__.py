"""`cytir.serve` -- the serving package: the JSON API, the MCP proxy and the retrieval stages.

`SERVER_REL` is the package's place inside any source tree (a checkout, a `git archive`, the
release tree the box renders from); `SERVER_DIR` is this checkout's copy. Every default path to a
serving artifact (`bm25_idf.json`, the fusion weights, the head and host tables) derives from one
of the two -- never from a hand-written `server/aw_server` string (S10 step 9, task 1130).
"""
import pathlib

from cytir.core.serve_layout import SERVER_REL  # noqa: F401 (re-export)

SERVER_DIR = pathlib.Path(__file__).resolve().parent
