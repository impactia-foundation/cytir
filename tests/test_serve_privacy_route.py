#!/usr/bin/env python3
"""Coverage for the /privacy notice route (task 390).

The privacy notice is a LAUNCH CONDITION of POLICY §4.9.1 (register-published
individuals): it must be on the serving surface before that facet is exposed
to public traffic. So the interesting assertions are not "does FastAPI route"
but "does the deployed tree actually serve the notice, and does it fail
visibly when the file is missing" — a stub-shaped fallback here would let an
operator believe the notice is live when it is not.

`retrieval`/`authority` are stubbed so this runs without qdrant, a model, or
any network — it tests app.py's surface, nothing behind it.
"""
import os
import pathlib
import sys
import types

HERE = pathlib.Path(os.path.abspath(__file__)).parent
WEBAPP = HERE.parent / "webapp"
sys.path.insert(0, str(HERE))

# Stub the heavy modules BEFORE importing app (they open clients at import).
_before = set(sys.modules)  # to drop every cytir.serve module imported against the stubs (task 2067)
installed = []  # only the stubs THIS file inserted; removed again below (no leak to later tests)
for name in ("cytir.serve.retrieval", "cytir.serve.authority"):
    mod = types.ModuleType(name)
    mod.COLLECTION = "test_privacy"          # retrieval
    mod.health_client = lambda: None         # retrieval
    mod.SIDECAR_FLOOR = 0.0                  # authority
    installed.append(name) if sys.modules.setdefault(name, mod) is mod else None

os.environ["AW_WEBAPP"] = str(WEBAPP / "index.html")
# The settings are read once, at the first import of `cytir.serve.settings` (S10 step 2): drop a
# copy an earlier test loaded so this env is the one `app` sees.
for name in ("cytir.serve.settings", "cytir.serve.app"):
    sys.modules.pop(name, None)

from fastapi.testclient import TestClient  # noqa: E402

from cytir.serve import app as app_module  # noqa: E402

# The app keeps its references to the stubs; drop them from sys.modules so a later
# `import cytir.serve.authority` (e.g. test_serve_s10_pure_modules) loads the real module.
for name in installed:
    sys.modules.pop(name, None)
# Task 2067: `app` (and any submodule first imported here) bound the stubs. Left cached, a later
# test in the same xdist worker that imports `cytir.serve.app` got a `retrieval` with no `search`
# (503 "search backend unavailable"). Drop them, and the package attribute `from cytir.serve import
# app` would otherwise still find.
_pkg = sys.modules.get("cytir.serve")
for name in [m for m in set(sys.modules) - _before if m.startswith("cytir.serve")]:
    sys.modules.pop(name, None)
    if _pkg is not None:
        _pkg.__dict__.pop(name.rpartition(".")[2], None)

client = TestClient(app_module.api)


def test_privacy_page_is_served_from_the_webapp_dir():
    r = client.get("/privacy")
    assert r.status_code == 200, r.status_code
    assert "text/html" in r.headers["content-type"]
    # Must revalidate: a corrected notice can never be served from cache.
    assert "no-cache" in r.headers.get("cache-control", "")


def _text():
    # Collapse whitespace: the source wraps at ~78 cols, so a phrase the notice
    # plainly makes can still span a newline. Match on prose, not on layout.
    return " ".join(client.get("/privacy").text.split())


def test_privacy_page_carries_the_load_bearing_commitments():
    body = _text()
    # The four things the notice exists to say. If a rewrite drops one of
    # these, the launch condition is no longer met and this fails loudly.
    for required in (
        "impactIA Foundation",          # controller identity
        "bender@impactia.org",          # contact + rights route
        "Art. 6(1)(f)",                 # lawful basis, stated not implied
        "TAKEDOWN PERSON",              # the person-name fast path
        # PRESENT TENSE, deliberately. The first cut of this notice said the
        # register-people facet was "not yet switched on" — inferred from a
        # top-k search that returned Wikipedia. Wrong: the names are live in
        # the bm25 sparse vector of the company cards (verified by point
        # retrieve, see task 390). Absence from top-k is not absence from the
        # index, and understating live processing is the one error a privacy
        # notice must never make. Lock the tense.
        "carry the people the register names",
    ):
        assert required in body, f"privacy notice no longer states: {required}"


def test_notice_does_not_claim_the_register_facet_is_dormant():
    body = _text()
    for forbidden in ("not yet switched on", "not yet live", "is not live"):
        assert forbidden not in body, f"privacy notice understates processing: {forbidden}"


def test_missing_notice_404s_instead_of_falling_back():
    original = app_module.PRIVACY
    app_module.PRIVACY = pathlib.Path("/nonexistent/privacy.html")
    try:
        r = client.get("/privacy")
        assert r.status_code == 404
        assert r.json()["status"] == "missing"
    finally:
        app_module.PRIVACY = original


def test_search_page_links_to_the_notice():
    # Discoverability is part of the requirement: an unlinked notice is not
    # "on the serving surface".
    assert 'href="/privacy"' in client.get("/").text
