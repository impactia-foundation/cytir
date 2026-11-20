"""The one guard that keeps the retune on the TWIN (review of #668, M1).

"Loopback" is not proof of the twin: on the build box a loopback port can be an ssh tunnel to
prod's qdrant (task 447: a tunnel stood in for the twin for 11 days), and a collection name can be
an alias of the serving collection. So the guard allows only what the box DECLARES to be the twin:

  a. the endpoint is the twin's (`cytir.distribute.dump.DEFAULT_TWIN_QDRANT_URL`, the home the dump
     tool already uses) and the collection is one the payload-profile declaration calls a twin and
     does not mark `serving` (`data/collection-payload-profiles.v1.json`). No new literal here;
  b. the collection is not `aw_prod`, not what any alias of `aw_prod` points at, and not an alias
     that resolves there (the endpoint's own `/aliases`; unreadable = refuse, we cannot tell);
     `cytir.index.prod_guard.is_prod` must also say no;
  c. the port's listener is not an `ssh` process (`ss -ltnp`; if `ss` cannot tell, say so loudly);
  d. every path the run writes (`--out-dir`, `--work-dir`, an arm's `--out`) is outside the serving
     paths.

Called from `emit.main`, `arm.main` and `arm.worker`: the arm is where queries and writes happen.
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import json
import os
import pathlib
import re
import subprocess
import sys
from urllib.parse import urlparse

EXIT_REFUSED = 2
LOOPBACK = {"localhost", "127.0.0.1", "::1"}
PROD_ALIAS = "aw_prod"
#: The role word of a throwaway fixture collection (task 2254): the CI job's `aw_gate_fixture`.
FIXTURE_ROLE = "twin-fixture"
OVERLAY_ENV = "AW_OPERATOR_OVERLAY"
#: A path under one of these is a path a server reads or a deploy writes.
SERVING_PATH_MARKERS = (
    "/aw/serve", "/opt/aw", "/app/aw/serve",  # rename-codemod: keep (the live image and the boxes keep the old name)
    "/cytir/serve", "/app/cytir/serve", "/deploy")


class Refused(SystemExit):
    def __init__(self, msg: str):
        super().__init__(f"REFUSED: {msg}")
        self.code = EXIT_REFUSED


def _hostport(url: str) -> tuple[str, int | None]:
    p = urlparse(url if "//" in url else "//" + url)
    host = (p.hostname or "").lower()
    return ("loopback" if host in LOOPBACK else host), p.port


def twin_endpoints() -> set[tuple[str, int | None]]:
    from cytir.core import twin       # read fresh: dump.DEFAULT_TWIN_QDRANT_URL is frozen at first import, which in
    return {_hostport(twin.configured_url())}   # a mixed session happens under the operator suite's overlay


def twin_collections() -> set[str]:
    """Collections the payload-profile declaration calls a twin and does not mark serving."""
    from cytir.index import payload_profile
    return {n for n, e in payload_profile.load_profiles().items()
            if str(e.get("role", "")).lower().startswith("twin") and e.get("serving") is not True
            and not _is_fixture(e)}


def _is_fixture(entry: dict) -> bool:
    return str(entry.get("role", "")).lower().startswith(FIXTURE_ROLE)


def fixture_collections() -> set[str]:
    """Collections declared `twin-fixture` and not `serving` (task 2254). They are NOT twins: they
    pass `refuse_non_twin` only when an operator overlay named the endpoint (see there)."""
    from cytir.index import payload_profile
    return {n for n, e in payload_profile.load_profiles().items()
            if _is_fixture(e) and e.get("serving") is not True}


def fetch_aliases(url: str) -> dict[str, str] | None:
    """{alias_name: collection_name} from the endpoint's own `/aliases`; None when unreadable."""
    try:
        with IH.urlopen(url.rstrip("/") + "/aliases", timeout=10) as r:
            return {e["alias_name"]: e["collection_name"] for e in json.loads(r.read())["result"]["aliases"]}
    except Exception:  # noqa: BLE001
        return None


def listener_is_ssh(port: int, ss_text: str | None = None) -> bool | None:
    """True when the process listening on loopback `port` is ssh; None when we cannot tell."""
    if ss_text is None:
        try:
            ss_text = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, timeout=10).stdout
        except Exception:  # noqa: BLE001
            return None
        if not ss_text:
            return None
    for line in ss_text.splitlines():
        if re.search(rf"[:\]]{port}\s", line):
            return 'users:(("ssh"' in line
    return False


def refuse_non_twin(url: str, collection: str, *, aliases: dict[str, str] | None = None,
                    ss_text: str | None = None, say=None) -> None:
    say = say or (lambda m: print(m, file=sys.stderr, flush=True))
    if _hostport(url) not in twin_endpoints():
        raise Refused(f"qdrant {url!r} is not a declared twin endpoint {sorted(twin_endpoints())} "
                      f"(loopback is not proof of the twin: a tunnel to prod is loopback too)")
    if collection in fixture_collections():
        # A throwaway fixture collection (CI): allowed only when an operator wrote an overlay that
        # NAMES this endpoint as the twin (`twin_url` comes from the overlay, not from the built-in
        # default: the endpoint check above would otherwise pass on the default twin port), so never
        # by default. It is also refused on any endpoint that holds the prod alias (below), and for
        # prod, its target or a serving collection (the checks below still run).
        from cytir.core import twin
        if not (os.environ.get(OVERLAY_ENV) and twin.url_is_named_by_overlay()):
            raise Refused(f"collection {collection!r} is a {FIXTURE_ROLE}: it needs an explicit "
                          f"operator overlay ({OVERLAY_ENV}) that names the twin endpoint (twin_url)")
    elif collection not in twin_collections():
        raise Refused(f"collection {collection!r} is not a declared twin collection {sorted(twin_collections())}")
    al = fetch_aliases(url) if aliases is None else aliases
    if al is None:
        raise Refused(f"cannot read {url}/aliases: cannot tell whether {collection!r} is behind {PROD_ALIAS}")
    prod_target = al.get(PROD_ALIAS)
    if collection in fixture_collections() and PROD_ALIAS in al:
        raise Refused(f"{url} holds {PROD_ALIAS}: a fixture never runs on a box that serves prod")
    resolved = al.get(collection, collection)
    if collection == PROD_ALIAS or (prod_target and resolved == prod_target):
        raise Refused(f"collection {collection!r} is {PROD_ALIAS} or its target ({prod_target!r}): that is prod")
    from cytir.index import prod_guard
    if prod_guard.is_prod(url, collection):
        raise Refused(f"cytir.index.prod_guard says ({url}, {collection}) is prod")
    port = _hostport(url)[1]
    if port is not None:
        ssh = listener_is_ssh(port, ss_text)
        if ssh:
            raise Refused(f"the listener on port {port} is an ssh process: that is a tunnel, not the twin")
        if ssh is None:
            say(f"WARNING: could not read the listener of port {port} (`ss -ltnp` unavailable): "
                f"the tunnel check is skipped; the declared-endpoint and alias checks still held")


def refuse_serving_path(path: str, what: str) -> None:
    flat = str(pathlib.Path(path).resolve()).replace(os.sep, "/")
    if any(m in flat for m in SERVING_PATH_MARKERS):
        raise Refused(f"{what} {flat} is a serving path; candidates go to data/retune/candidates/")
