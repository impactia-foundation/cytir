"""Where the TWIN is: the one accessor (task 832).

The twin is the build box's qdrant: every drain writes it first, every gate reads it. Its URL and
its collection used to be restated in about forty places (argparse defaults, module constants,
shell drivers, unit drop-ins), so moving it (the aw_e10 flip does) meant a hunt. Now two operator
settings name it, `twin_url` and `twin_collection` (`cytir.core.paths`), resolved in the one order
every setting uses: environment (`AW_TWIN_URL`, `AW_TWIN_COLLECTION`) -> operator overlay (the box
file `/etc/aw/operator.json`, where a box sets it) -> the default below. Read fresh on every call
(no cache), so a test or a long-lived process sees the current environment.

The twin is NOT prod: `cytir.index.prod_guard` classifies an endpoint by what it holds (the serving
alias), never by how its URL is spelled, so moving the twin cannot make it look like prod.
A ratchet test (the twin-endpoint ratchet) fails the build on a new raw literal outside its allowlist.
"""
from __future__ import annotations

import os

from cytir.core import envnames as _envnames

from cytir.core import paths


def url(env=None) -> str:
    """The twin qdrant URL, no trailing slash."""
    return paths.load_settings(env).twin_url.rstrip("/")


def collection(env=None) -> str:
    """The physical collection the twin holds."""
    return paths.load_settings(env).twin_collection


def configured_url() -> str:
    """The twin URL from the overlay / default ONLY: `AW_TWIN_URL` is ignored. For a guard that must
    decide what "the twin" is independently of a caller's environment (`cyt dump`)."""
    return url(_env_without_twin())


def _env_without_twin() -> dict:
    """`os.environ` without the twin variables under BOTH spellings and without the mirror's marker:
    a `CYTIR_TWIN_URL` left in place would be mirrored back to `AW_TWIN_URL` by the settings loader."""
    env = dict(os.environ)
    for name in ("AW_TWIN_URL", "AW_TWIN_COLLECTION"):
        _envnames.pop_knob(env, name)
    return env


def url_is_named_by_overlay() -> bool:
    """True only when the operator overlay itself sets `twin_url` (task 2254): not the environment,
    not the built-in default. For a guard that accepts a throwaway fixture only on an endpoint an
    operator wrote down as the twin."""
    return paths.load_settings(_env_without_twin()).source("twin_url").startswith("overlay")


if __name__ == "__main__":      # `python -m cytir.core.twin` prints "<url> <collection>"
    print(url(), collection())
