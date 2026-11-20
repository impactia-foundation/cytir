#!/usr/bin/env python3
"""Shim: the pack-schema validator lives in the package, `cytir.sources.schema` (task 735).

    python3 sources/_schema/validate.py [PACK.yaml ...]   ==   python3 -m cytir.sources.schema [PACK.yaml ...]

`pack.schema.json` beside this file stays the source of truth for the schema; the package ships a
byte-for-byte copy (`cytir/_data/pack.schema.json`) that a test holds equal to it.
"""
import pathlib
import sys

# run from a clone without installing: the repo root (two levels up) holds `cytir/`
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cytir.sources.schema import *  # noqa: F401,F403  (ROUTE_PARAMS, ValidationError, load_schema, ...)
from cytir.sources.schema import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
