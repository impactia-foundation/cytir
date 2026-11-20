"""Shared walk for the DILA bulk-dump families (task 696). Not a family: nothing registers here.

A DILA `Freemium_<base>_global_*.tar.gz` extracts to a tree of one XML file per record. A family
names the file-name prefixes it takes; a unit is one such file, discovered in sorted order so a
run is deterministic (the per-convention cap of `dila_kali` depends on that order).
"""
from __future__ import annotations

import os
import pathlib
import re

from cytir.sources.family import Unit, UnitExcluded

_INCREMENT_DIR = re.compile(r"(?:^|/)(\d{8}-\d{6})/")
SUPERSEDED = "superseded_by_newer_increment"


def discover_units(family_id: str, options: dict, prefixes: tuple[str, ...], notes: dict
                   ) -> list[Unit]:
    root = options.get("root")
    if not root:
        raise ValueError(f"{family_id} needs --opt root=<directory containing the extracted DILA "
                         "dump tree>")
    if not os.path.isdir(root):
        raise ValueError(f"{family_id}: root {root!r} is not a directory")
    paths = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(".xml") and fn.startswith(prefixes):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    if not paths:
        raise RuntimeError(f"{family_id}: no {'/'.join(p + '*.xml' for p in prefixes)} under "
                           f"{root!r}; an empty walk is a wrong root, not an empty source")
    notes.update({"root": str(root), "files": len(paths)})
    return [Unit(key=os.path.relpath(p, root), url=pathlib.Path(p).resolve().as_uri(),
                 meta={"path": p}) for p in paths]


def record_id(unit_key: str) -> str:
    return os.path.basename(unit_key)[:-len(".xml")]


def newest_by_record(units: list[Unit]) -> dict[str, str]:
    """`record id -> key of the unit that carries its newest version`.

    A DILA increment (`<YYYYMMDD-HHMMSS>/<base>/global/...`) re-publishes a record whenever it
    changed, under the same id. The global dump is the oldest version (rank ""); increments rank by
    their own timestamp directory. The newest wins, so one record never becomes two points for one
    page; the older copies still reach a terminal (`excluded(superseded_by_newer_increment)`),
    nothing stops being mentioned."""
    best: dict[str, tuple[str, str]] = {}
    for u in units:
        m = _INCREMENT_DIR.search(u.key)
        rank = m.group(1) if m else ""
        rid = record_id(u.key)
        if rid not in best or (rank, u.key) > best[rid]:
            best[rid] = (rank, u.key)
    return {rid: key for rid, (_rank, key) in best.items()}


def ids_under(root: str | None, prefixes: tuple[str, ...]) -> set[str]:
    """Record ids already carried by an earlier run's tree (`--opt skip_ids_under=DIR`)."""
    if not root:
        return set()
    if not os.path.isdir(root):
        raise ValueError(f"skip_ids_under {root!r} is not a directory")
    return {fn[:-len(".xml")] for _d, _s, files in os.walk(root) for fn in files
            if fn.endswith(".xml") and fn.startswith(prefixes)}


ALREADY_IN_PRIOR_RUN = "already_in_prior_run"


class Versions:
    """Which version of each record a run takes, decided at discover time.

    * the newest increment wins (`superseded_by_newer_increment` for the others);
    * with `--opt skip_ids_under=DIR` a record already carried by that tree is not taken again
      (`already_in_prior_run`): an increments-only run then adds the NEW records and never a second
      representation of a page an earlier run already staged. An UPDATE to such a record is not
      taken either; whether updates replace the old row is a supersede ruling, not this walk's."""

    def __init__(self, units: list[Unit], options: dict, prefixes: tuple[str, ...], notes: dict):
        self.chosen = newest_by_record(units)
        self.known = ids_under(options.get("skip_ids_under"), prefixes)
        notes["records"] = len(self.chosen)
        notes["known_ids_in_prior_run"] = len(self.known)

    def check(self, unit: Unit) -> None:
        rid = record_id(unit.key)
        if self.chosen.get(rid) != unit.key:
            raise UnitExcluded(SUPERSEDED)
        if rid in self.known:
            raise UnitExcluded(ALREADY_IN_PRIOR_RUN)


def read_unit(unit: Unit) -> bytes:
    with open(unit.meta["path"], "rb") as fh:
        return fh.read()
