"""cytir.distribute — distribution v0 (task 650, migration S8, task 643 §6.2).

What a stranger downloads and applies: a signed full dump plus daily delta
packs, built from the format `specs/dump-format.md` declares. Nothing in
here ever reads from prod (`cytir/distribute/dump.py` refuses on sight —
see `PROD_QDRANT_DENYLIST`); the source of a public dump is always the twin.

Modules:
    pack.py          the declared row/manifest schema shared by dump + apply
    license_check.py moved from the retired `assert_license` module (task 645
                      follow-up: it previously had no caller anywhere)
    sign.py           minisign wrapper (detached manifest signatures)
    dump.py           `cyt dump` — twin collection -> parquet shards + manifest
    apply.py          `cyt apply` — a self-hoster's verify + create + upsert +
                      delta tool
"""
from __future__ import annotations
