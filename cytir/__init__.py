"""aw — the authoritative-web package (task 646, migration plan in
task 643 §6).

This is the seed of the target layout (task 643 §6.1): an installable
package that the rest of the migration (S2-S10) moves real modules into,
each move leaving a shim at its old path so nothing else has to change on
the same day (task 643 §6.2). Nothing productive lives here yet.
"""
from __future__ import annotations

import os as _os

from cytir.core import envnames as _envnames  # stdlib only, no I/O

_envnames.normalise(_os.environ)  # CYTIR_X <-> AW_X, before any reader runs (task 2964)

__version__ = "0.1.1"

__all__ = ["__version__"]
